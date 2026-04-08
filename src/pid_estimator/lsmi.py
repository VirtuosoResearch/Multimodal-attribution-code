from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import MultimodalDataset


@dataclass(frozen=True)
class LSMIConfig:
    hidden_dim: int = 64
    batch_size: int = 32
    discriminator_epochs: int = 30
    entropy_epochs: int = 30
    lr: float = 1e-3
    num_workers: int = 0
    shuffle: bool = True
    verbose: bool = False
    seed: int | None = None


@dataclass
class LSMIEstimateResult:
    pointwise_components: torch.Tensor
    mean_components: torch.Tensor
    pointwise_info_1: torch.Tensor
    pointwise_info_2: torch.Tensor
    pointwise_info_joint: torch.Tensor
    pointwise_entropy_1: torch.Tensor
    pointwise_entropy_2: torch.Tensor
    discriminator_1: nn.Module
    discriminator_2: nn.Module
    discriminator_joint: nn.Module
    entropy_estimator_1: "MargKernel"
    entropy_estimator_2: "MargKernel"
    label_prior: torch.Tensor

    @property
    def redundancy(self) -> torch.Tensor:
        return self.mean_components[0]

    @property
    def unique_1(self) -> torch.Tensor:
        return self.mean_components[1]

    @property
    def unique_2(self) -> torch.Tensor:
        return self.mean_components[2]

    @property
    def synergy(self) -> torch.Tensor:
        return self.mean_components[3]


class ClassificationNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MargKernel(nn.Module):
    def __init__(self, dim: int, init_samples: torch.Tensor | None = None):
        super().__init__()
        self.K = 5
        self.d = dim
        self.use_tanh = True
        self.init_std = torch.tensor(1.0, dtype=torch.float32)
        self.logC = torch.tensor([-self.d / 2 * torch.log(torch.tensor(2.0 * torch.pi))], dtype=torch.float32)

        if init_samples is None:
            init_samples = self.init_std * torch.randn(self.K, self.d)
        self.means = nn.Parameter(init_samples, requires_grad=True)
        diag = self.init_std * torch.randn((1, self.K, self.d))
        self.logvar = nn.Parameter(diag, requires_grad=True)

        tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
        tri = tri.to(init_samples.dtype)
        self.tri = nn.Parameter(tri, requires_grad=True)

        weigh = torch.ones((1, self.K))
        self.weigh = nn.Parameter(weigh, requires_grad=True)

    def logpdf(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"x has to have shape [N, {self.d}] but got {tuple(x.shape)}")
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        logvar = self.logvar.tanh() if self.use_tanh else self.logvar
        var = logvar.exp()
        y = y * var
        y = y.to(self.tri.dtype)
        y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y**2, dim=2)
        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = -self.logpdf(x)
        if self.training:
            return torch.mean(values)
        return values


def RUS_adjustment(rus: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r_orig, u_1_orig, u_2_orig, s_orig = rus
    r_mean = r_orig.detach().mean()
    u1_mean = u_1_orig.detach().mean()
    u2_mean = u_2_orig.detach().mean()
    s_mean = s_orig.detach().mean()

    adjustment = torch.tensor(0.0, dtype=r_mean.dtype, device=r_mean.device)
    if r_mean < 0 or s_mean < 0:
        adjustment = -torch.min(r_mean, s_mean)
    elif u1_mean < 0 or u2_mean < 0:
        adjustment = torch.min(u1_mean, u2_mean)

    return (
        r_orig + adjustment,
        u_1_orig - adjustment,
        u_2_orig - adjustment,
        s_orig + adjustment,
    )


def estimate_lsmi(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    config: LSMIConfig | None = None,
    device: str | torch.device | None = None,
) -> LSMIEstimateResult:
    config = config or LSMIConfig()
    device = _resolve_device(device)
    _maybe_set_seed(config.seed)

    x1 = _ensure_feature_batch(x1).float()
    x2 = _ensure_feature_batch(x2).float()
    labels = _prepare_labels(labels)
    num_labels = int(num_labels if num_labels is not None else int(labels.max().item()) + 1)
    train_dataset = train_dataset or MultimodalDataset([x1, x2], labels)
    test_dataset = test_dataset or MultimodalDataset([x1, x2], labels)

    discrim_1 = ClassificationNetwork(x1.shape[1], config.hidden_dim, num_labels).to(device)
    discrim_2 = ClassificationNetwork(x2.shape[1], config.hidden_dim, num_labels).to(device)
    discrim_joint = ClassificationNetwork(x1.shape[1] + x2.shape[1], config.hidden_dim, num_labels).to(device)
    entropy_1 = MargKernel(dim=x1.shape[1]).to(device)
    entropy_2 = MargKernel(dim=x2.shape[1]).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    _train_discriminators(
        train_loader,
        [discrim_1, discrim_2, discrim_joint],
        lr=config.lr,
        device=device,
        epochs=config.discriminator_epochs,
        verbose=config.verbose,
    )
    _train_entropy_estimators(
        train_loader,
        [entropy_1, entropy_2],
        lr=config.lr,
        device=device,
        epochs=config.entropy_epochs,
        verbose=config.verbose,
    )

    label_prior = _estimate_label_prior(labels, num_labels).to(device)
    info_1 = _get_mutual_info(test_loader, discrim_1, modality="modality_1", device=device, label_prior=label_prior)
    info_2 = _get_mutual_info(test_loader, discrim_2, modality="modality_2", device=device, label_prior=label_prior)
    info_joint = _get_mutual_info(test_loader, discrim_joint, modality="modality_12", device=device, label_prior=label_prior)
    ent_1 = _get_entropy(test_loader, entropy_1, modality="modality_1", device=device)
    ent_2 = _get_entropy(test_loader, entropy_2, modality="modality_2", device=device)

    r_plus = torch.minimum(ent_1, ent_2)
    r_minus = torch.minimum(ent_1 - info_1, ent_2 - info_2)
    redundancy = r_plus - r_minus
    unique_1 = info_1 - redundancy
    unique_2 = info_2 - redundancy
    synergy = info_joint - redundancy - unique_1 - unique_2
    redundancy, unique_1, unique_2, synergy = RUS_adjustment([redundancy, unique_1, unique_2, synergy])

    pointwise_components = torch.stack([redundancy, unique_1, unique_2, synergy], dim=1).detach().cpu()
    mean_components = pointwise_components.mean(dim=0)
    return LSMIEstimateResult(
        pointwise_components=pointwise_components,
        mean_components=mean_components,
        pointwise_info_1=info_1.detach().cpu(),
        pointwise_info_2=info_2.detach().cpu(),
        pointwise_info_joint=info_joint.detach().cpu(),
        pointwise_entropy_1=ent_1.detach().cpu(),
        pointwise_entropy_2=ent_2.detach().cpu(),
        discriminator_1=discrim_1,
        discriminator_2=discrim_2,
        discriminator_joint=discrim_joint,
        entropy_estimator_1=entropy_1,
        entropy_estimator_2=entropy_2,
        label_prior=label_prior.detach().cpu(),
    )


def _train_discriminators(
    dataloader: DataLoader,
    models: list[nn.Module],
    *,
    lr: float,
    device: torch.device,
    epochs: int,
    verbose: bool,
) -> None:
    optimizer = torch.optim.Adam([parameter for model in models for parameter in model.parameters()], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for model in models:
            model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in dataloader:
            modal_1, modal_2, labels = _obtain_feature_input(batch, device=device)
            batch_size = modal_1.shape[0]
            out_1 = models[0](modal_1)
            out_2 = models[1](modal_2)
            out_joint = models[2](torch.cat([modal_1, modal_2], dim=1))
            optimizer.zero_grad()
            loss = criterion(out_1, labels) + criterion(out_2, labels) + criterion(out_joint, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        scheduler.step()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"[lsmi/disc] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")


def _train_entropy_estimators(
    dataloader: DataLoader,
    models: list[MargKernel],
    *,
    lr: float,
    device: torch.device,
    epochs: int,
    verbose: bool,
) -> None:
    optimizer = torch.optim.Adam([parameter for model in models for parameter in model.parameters()], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.1)
    for epoch in range(epochs):
        for model in models:
            model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in dataloader:
            modal_1, modal_2, _labels = _obtain_feature_input(batch, device=device)
            batch_size = modal_1.shape[0]
            optimizer.zero_grad()
            loss = models[0](modal_1) + models[1](modal_2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        scheduler.step()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"[lsmi/ent] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")


def _get_entropy(
    dataloader: DataLoader,
    model: MargKernel,
    *,
    modality: str,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in dataloader:
            modal_1, modal_2, _labels = _obtain_feature_input(batch, device=device)
            if modality == "modality_1":
                inputs = modal_1
            elif modality == "modality_2":
                inputs = modal_2
            else:
                raise ValueError(f"unsupported modality for entropy: {modality}")
            values.append(model(inputs))
    return torch.cat(values, dim=0).detach()


def _get_mutual_info(
    dataloader: DataLoader,
    model: nn.Module,
    *,
    modality: str,
    device: torch.device,
    label_prior: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in dataloader:
            modal_1, modal_2, labels = _obtain_feature_input(batch, device=device)
            if modality == "modality_1":
                inputs = modal_1
            elif modality == "modality_2":
                inputs = modal_2
            elif modality == "modality_12":
                inputs = torch.cat([modal_1, modal_2], dim=1)
            else:
                raise ValueError(f"unsupported modality for mutual information: {modality}")

            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=1)
            rows = torch.arange(inputs.shape[0], device=device)
            values.append(log_probs[rows, labels] - torch.log(label_prior[labels] + 1e-8))
    return torch.cat(values, dim=0).detach()


def _obtain_feature_input(batch: Any, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    modal_1 = _ensure_feature_batch(batch[0]).to(device)
    modal_2 = _ensure_feature_batch(batch[1]).to(device)
    labels = _prepare_labels(batch[2]).to(device)
    return modal_1, modal_2, labels


def _estimate_label_prior(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_labels).float()
    counts /= torch.clamp(counts.sum(), min=1.0)
    return counts


def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    if labels.ndim > 1:
        labels = labels.squeeze(-1)
    return labels


def _ensure_feature_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(-1)
    return value


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "ClassificationNetwork",
    "LSMIConfig",
    "LSMIEstimateResult",
    "MargKernel",
    "RUS_adjustment",
    "estimate_lsmi",
]
