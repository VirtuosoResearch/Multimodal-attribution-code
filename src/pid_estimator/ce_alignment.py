from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .data import MultimodalDataset


TensorLike = torch.Tensor


@dataclass(frozen=True)
class CEAlignmentConfig:
    hidden_dim: int = 32
    embed_dim: int = 10
    layers: int = 3
    activation: str = "relu"
    batch_size: int = 256
    discrim_epochs: int = 40
    ce_epochs: int = 10
    lr: float = 1e-3
    num_workers: int = 0
    shuffle: bool = True
    learned_discriminator: bool = True
    verbose: bool = False
    seed: int | None = None


@dataclass
class PIDEstimateResult:
    batch_components: torch.Tensor
    mean_components: torch.Tensor
    aligns: list[torch.Tensor]
    model: "CEAlignmentInformation"
    discriminator_1: Any
    discriminator_2: Any
    discriminator_12: Any
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


def sinkhorn_probs(matrix: torch.Tensor, x1_probs: torch.Tensor, x2_probs: torch.Tensor) -> tuple[torch.Tensor, bool]:
    matrix = matrix / (torch.sum(matrix, dim=0, keepdim=True) + 1e-8) * x2_probs[None]
    row_sum = torch.sum(matrix, dim=1)
    if torch.allclose(row_sum, x1_probs, rtol=0.0, atol=0.01):
        return matrix, True
    matrix = matrix / (torch.sum(matrix, dim=1, keepdim=True) + 1e-8) * x1_probs[:, None]
    col_sum = torch.sum(matrix, dim=0)
    if torch.allclose(col_sum, x2_probs, rtol=0.0, atol=0.01):
        return matrix, True
    return matrix, False


def mlp(dim: int, hidden_dim: int, output_dim: int, layers: int, activation: str) -> nn.Sequential:
    activation_cls = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }[activation]

    seq: list[nn.Module] = [nn.Linear(dim, hidden_dim), activation_cls()]
    for _ in range(layers):
        seq.extend([nn.Linear(hidden_dim, hidden_dim), activation_cls()])
    seq.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*seq)


def simple_discrim(xs: Sequence[torch.Tensor], y: torch.Tensor, num_labels: int) -> Callable[..., torch.Tensor]:
    xs = [_ensure_feature_batch(x) for x in xs]
    shape = [x.size(1) for x in xs] + [num_labels]
    table = torch.ones(*shape, dtype=torch.float32) * 1e-8
    y_indices = _prepare_labels(y)
    for index in range(len(y_indices)):
        entry = tuple([torch.argmax(x[index]).item() for x in xs] + [y_indices[index].item()])
        table[entry] += 1.0
    table /= torch.sum(table)

    def _lookup(*values: torch.Tensor) -> torch.Tensor:
        argmax_values = [torch.argmax(value, dim=1) for value in values]
        return torch.log(table.to(argmax_values[0].device)[tuple(argmax_values)])

    return _lookup


class Discrim(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, num_labels: int, layers: int, activation: str):
        super().__init__()
        self.mlp = mlp(x_dim, hidden_dim, num_labels, layers, activation)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat(x, dim=-1))


class CEAlignment(nn.Module):
    def __init__(self, x1_dim: int, x2_dim: int, hidden_dim: int, embed_dim: int, num_labels: int, layers: int, activation: str):
        super().__init__()
        self.num_labels = num_labels
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation)
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x1_probs: torch.Tensor, x2_probs: torch.Tensor) -> torch.Tensor:
        q_x1 = self.mlp1(x1).unflatten(1, (self.num_labels, -1))
        q_x2 = self.mlp2(x2).unflatten(1, (self.num_labels, -1))

        q_x1 = (q_x1 - torch.mean(q_x1, dim=2, keepdim=True)) / torch.sqrt(torch.var(q_x1, dim=2, keepdim=True) + 1e-8)
        q_x2 = (q_x2 - torch.mean(q_x2, dim=2, keepdim=True)) / torch.sqrt(torch.var(q_x2, dim=2, keepdim=True) + 1e-8)

        align = torch.einsum("ahx,bhx->abh", q_x1, q_x2) / math.sqrt(q_x1.size(-1))
        align = torch.exp(align)

        normalized = []
        for label_index in range(align.size(-1)):
            current = align[..., label_index]
            for _ in range(500):
                current, stop = sinkhorn_probs(current, x1_probs[:, label_index], x2_probs[:, label_index])
                if stop:
                    break
            normalized.append(current)
        normalized_tensor = torch.stack(normalized, dim=-1)
        if torch.isnan(normalized_tensor).any():
            raise RuntimeError("CE alignment produced NaN values")
        return normalized_tensor


class CEAlignmentInformation(nn.Module):
    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_labels: int,
        layers: int,
        activation: str,
        discrim_1: Any,
        discrim_2: Any,
        discrim_12: Any,
        p_y: torch.Tensor,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        self.discrim_1 = discrim_1
        self.discrim_2 = discrim_2
        self.discrim_12 = discrim_12
        if isinstance(self.discrim_1, nn.Module):
            self.discrim_1.eval()
        if isinstance(self.discrim_2, nn.Module):
            self.discrim_2.eval()
        if isinstance(self.discrim_12, nn.Module):
            self.discrim_12.eval()
        self.register_buffer("p_y", p_y)

    def align_parameters(self) -> list[nn.Parameter]:
        return list(self.align.parameters())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            p_y_x1 = torch.softmax(self.discrim_1(x1), dim=-1)
            p_y_x2 = torch.softmax(self.discrim_2(x2), dim=-1)
            p_y_x1x2 = torch.softmax(self.discrim_12(x1, x2), dim=-1)

        align = self.align(torch.flatten(x1, 1, -1), torch.flatten(x2, 1, -1), p_y_x1, p_y_x2)
        q_x2_x1y = align / (torch.sum(align, dim=1, keepdim=True) + 1e-8)
        normalizer = torch.einsum("aby,ay->ab", q_x2_x1y, p_y_x1)
        log_term = torch.log(q_x2_x1y + 1e-8) - torch.log(normalizer + 1e-8)[:, :, None]
        loss = torch.mean(torch.sum(torch.sum(p_y_x1[:, None, :] * q_x2_x1y * log_term, dim=-1), dim=-1))

        log_prior = torch.log(self.p_y + 1e-8)
        mi_y_x1 = torch.mean(torch.sum(p_y_x1 * (torch.log(p_y_x1 + 1e-8) - log_prior[None, :]), dim=-1))
        mi_y_x2 = torch.mean(torch.sum(p_y_x2 * (torch.log(p_y_x2 + 1e-8) - log_prior[None, :]), dim=-1))
        mi_y_x1x2 = torch.mean(torch.sum(p_y_x1x2 * (torch.log(p_y_x1x2 + 1e-8) - log_prior[None, :]), dim=-1))
        mi_q_y_x1x2 = p_y_x1[:, None, :] * q_x2_x1y * (
            log_term + torch.log(p_y_x1 + 1e-8)[:, None, :] - log_prior[None, None, :]
        )
        mi_q_y_x1x2 = torch.mean(torch.sum(torch.sum(mi_q_y_x1x2, dim=-1), dim=-1))

        redundancy = mi_y_x1 + mi_y_x2 - mi_q_y_x1x2
        unique1 = mi_q_y_x1x2 - mi_y_x2
        unique2 = mi_q_y_x1x2 - mi_y_x1
        synergy = mi_y_x1x2 - mi_q_y_x1x2

        return loss, torch.stack([redundancy, unique1, unique2, synergy], dim=0), align


class EncoderAndHead(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.head(self.encoder(*args, **kwargs))


class MM(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def estimate_pid_with_ce_alignment(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    discrim_1: Any = None,
    discrim_2: Any = None,
    discrim_12: Any = None,
    config: CEAlignmentConfig | None = None,
    device: str | torch.device | None = None,
) -> PIDEstimateResult:
    config = config or CEAlignmentConfig()
    device = _resolve_device(device)
    _maybe_set_seed(config.seed)

    x1 = x1.float()
    x2 = x2.float()
    labels = _prepare_labels(labels)
    num_labels = int(num_labels if num_labels is not None else int(labels.max().item()) + 1)

    train_dataset = train_dataset or MultimodalDataset([x1, x2], labels)
    test_dataset = test_dataset or MultimodalDataset([x1, x2], labels)

    model_discrim_1, model_discrim_2, model_discrim_12 = _build_discriminators(
        x1=x1,
        x2=x2,
        labels=labels,
        num_labels=num_labels,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        discrim_1=discrim_1,
        discrim_2=discrim_2,
        discrim_12=discrim_12,
        config=config,
        device=device,
    )

    p_y = torch.sum(torch.nn.functional.one_hot(labels, num_classes=num_labels).float(), dim=0) / len(labels)

    model = CEAlignmentInformation(
        x1_dim=_flattened_dim(x1),
        x2_dim=_flattened_dim(x2),
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        num_labels=num_labels,
        layers=config.layers,
        activation=config.activation,
        discrim_1=model_discrim_1,
        discrim_2=model_discrim_2,
        discrim_12=model_discrim_12,
        p_y=p_y.to(device),
    ).to(device)
    optimizer = optim.Adam(model.align_parameters(), lr=config.lr)

    train_loader = DataLoader(
        train_dataset,
        shuffle=config.shuffle,
        drop_last=_should_drop_last(train_dataset, config.batch_size),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model.train()
    _train_ce_alignment(model, train_loader, optimizer, device=device, num_epoch=config.ce_epochs, verbose=config.verbose)

    model.eval()
    batch_components, aligns = _eval_ce_alignment(model, test_loader, device=device)
    mean_components = batch_components.mean(dim=0)
    return PIDEstimateResult(
        batch_components=batch_components.detach().cpu(),
        mean_components=mean_components.detach().cpu(),
        aligns=aligns,
        model=model,
        discriminator_1=model_discrim_1,
        discriminator_2=model_discrim_2,
        discriminator_12=model_discrim_12,
        label_prior=p_y.detach().cpu(),
    )


def critic_ce_alignment(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    train_ds: Dataset,
    test_ds: Dataset,
    discrim_1: Any = None,
    discrim_2: Any = None,
    discrim_12: Any = None,
    learned_discrim: bool = True,
    shuffle: bool = True,
    discrim_epochs: int = 40,
    ce_epochs: int = 10,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], tuple[Any, Any, Any, Any, torch.Tensor]]:
    config = CEAlignmentConfig(
        learned_discriminator=learned_discrim,
        shuffle=shuffle,
        discrim_epochs=discrim_epochs,
        ce_epochs=ce_epochs,
    )
    result = estimate_pid_with_ce_alignment(
        x1,
        x2,
        labels,
        num_labels=num_labels,
        train_dataset=train_ds,
        test_dataset=test_ds,
        discrim_1=discrim_1,
        discrim_2=discrim_2,
        discrim_12=discrim_12,
        config=config,
        device=device,
    )
    models = (
        result.model,
        result.discriminator_1,
        result.discriminator_2,
        result.discriminator_12,
        result.label_prior,
    )
    return result.batch_components, result.aligns, models


def _build_discriminators(
    *,
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    train_dataset: Dataset,
    test_dataset: Dataset,
    discrim_1: Any,
    discrim_2: Any,
    discrim_12: Any,
    config: CEAlignmentConfig,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    if discrim_1 is not None:
        if discrim_2 is None or discrim_12 is None:
            raise ValueError("discrim_1/discrim_2/discrim_12 must be provided together")
        return discrim_1, discrim_2, discrim_12

    if not config.learned_discriminator:
        return (
            simple_discrim([x1], labels, num_labels),
            simple_discrim([x2], labels, num_labels),
            simple_discrim([x1, x2], labels, num_labels),
        )

    model_discrim_1 = Discrim(
        x_dim=_flattened_dim(x1),
        hidden_dim=config.hidden_dim,
        num_labels=num_labels,
        layers=config.layers,
        activation=config.activation,
    ).to(device)
    model_discrim_2 = Discrim(
        x_dim=_flattened_dim(x2),
        hidden_dim=config.hidden_dim,
        num_labels=num_labels,
        layers=config.layers,
        activation=config.activation,
    ).to(device)
    model_discrim_12 = Discrim(
        x_dim=_flattened_dim(x1) + _flattened_dim(x2),
        hidden_dim=config.hidden_dim,
        num_labels=num_labels,
        layers=config.layers,
        activation=config.activation,
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        shuffle=config.shuffle,
        drop_last=_should_drop_last(train_dataset, config.batch_size),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    for model, data_type in [
        (model_discrim_1, ([1], [0])),
        (model_discrim_2, ([2], [0])),
        (model_discrim_12, ([1], [2], [0])),
    ]:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        _train_discrim(model, train_loader, optimizer, data_type=data_type, device=device, num_epoch=config.discrim_epochs, verbose=config.verbose)
        model.eval()
        _eval_discrim(model, test_loader, data_type=data_type, device=device)

    return model_discrim_1, model_discrim_2, model_discrim_12


def _train_discrim(
    model: Discrim,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    *,
    data_type: Sequence[Sequence[int]],
    device: torch.device,
    num_epoch: int,
    verbose: bool,
) -> None:
    for epoch in range(num_epoch):
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, y = _prepare_discriminator_batch(batch, data_type, device)
            logits = model(*inputs)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 20 == 0 and batch_index % 128 == 0:
                print(f"[discrim] epoch={epoch + 1} batch={batch_index} loss={loss.item():.4f}")


def _eval_discrim(model: Discrim, test_loader: DataLoader, *, data_type: Sequence[Sequence[int]], device: torch.device) -> float:
    losses = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, y = _prepare_discriminator_batch(batch, data_type, device)
            logits = model(*inputs)
            losses.append(nn.CrossEntropyLoss()(logits, y).item())
    return float(sum(losses) / max(len(losses), 1))


def _train_ce_alignment(
    model: CEAlignmentInformation,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    *,
    device: torch.device,
    num_epoch: int,
    verbose: bool,
) -> None:
    for epoch in range(num_epoch):
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x1_batch, x2_batch, y_batch = _prepare_alignment_batch(batch, device)
            loss, _, _ = model(x1_batch, x2_batch, y_batch)
            loss.backward()
            optimizer.step()
            if verbose and batch_index % 32 == 0:
                print(f"[ce] epoch={epoch + 1} batch={batch_index} loss={loss.item():.4f}")


def _eval_ce_alignment(
    model: CEAlignmentInformation,
    test_loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    results = []
    aligns: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_loader:
            x1_batch, x2_batch, y_batch = _prepare_alignment_batch(batch, device)
            _, result, align = model(x1_batch, x2_batch, y_batch)
            results.append(result.detach().cpu())
            aligns.append(align.detach().cpu())
    if not results:
        raise ValueError("test_loader produced no batches")
    return torch.stack(results, dim=0), aligns


def _prepare_discriminator_batch(
    batch: Sequence[torch.Tensor],
    data_type: Sequence[Sequence[int]],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    inputs = []
    for group_index in range(len(data_type)):
        tensors = [_ensure_feature_batch(batch[index - 1]) for index in data_type[group_index]]
        combined = torch.cat(tensors, dim=1).to(device)
        if group_index != len(data_type) - 1:
            combined = combined.float()
        inputs.append(combined)
    labels = _prepare_labels(inputs[-1]).to(device)
    return inputs[:-1], labels


def _prepare_alignment_batch(batch: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1_batch = _ensure_feature_batch(batch[0]).float().to(device)
    x2_batch = _ensure_feature_batch(batch[1]).float().to(device)
    y_batch = _prepare_labels(batch[2]).to(device)
    return x1_batch, x2_batch, y_batch


def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    if labels.ndim > 1:
        labels = labels.squeeze(-1)
    return labels


def _ensure_feature_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(-1)
    return value


def _flattened_dim(tensor: torch.Tensor) -> int:
    tensor = _ensure_feature_batch(tensor)
    size = 1
    for dim in tensor.shape[1:]:
        size *= int(dim)
    return size


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _should_drop_last(dataset: Dataset, batch_size: int) -> bool:
    return len(dataset) >= batch_size


def _maybe_set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "CEAlignment",
    "CEAlignmentConfig",
    "CEAlignmentInformation",
    "Discrim",
    "EncoderAndHead",
    "MM",
    "MultimodalDataset",
    "PIDEstimateResult",
    "critic_ce_alignment",
    "estimate_pid_with_ce_alignment",
    "mlp",
    "simple_discrim",
    "sinkhorn_probs",
]
