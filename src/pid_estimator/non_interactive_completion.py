from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import MultimodalDataset
from .lsmi import ClassificationNetwork

_LOG2 = float(torch.log(torch.tensor(2.0)).item())


@dataclass(frozen=True)
class NonInteractiveCompletionConfig:
    hidden_dim: int = 64
    additive_hidden_dim: int = 64
    predictor_model: str = "mlp"
    additive_model: str = "mlp"
    batch_size: int = 64
    predictor_epochs: int = 10
    additive_epochs: int = 10
    predictor_lr: float = 1e-3
    additive_lr: float = 1e-3
    kernel_gamma: float | None = None
    kernel_ridge_lambda: float = 1e-2
    additive_feasibility_lambda: float = 0.0
    additive_rkhs_lambda: float = 0.0
    additive_anchor_lambda: float = 1.0
    num_workers: int = 0
    shuffle: bool = True
    calibration_blocks: int = 8
    seed: int | None = None
    verbose: bool = False


@dataclass
class NonInteractiveCompletionQueryEngine:
    block_component_sums: torch.Tensor
    block_sizes: torch.Tensor

    def query(self, subset_weights: torch.Tensor) -> torch.Tensor:
        weights = subset_weights.float().view(-1, 1)
        weighted_sum = torch.sum(self.block_component_sums * weights, dim=0)
        denominator = torch.sum(self.block_sizes * subset_weights.float()).clamp_min(1.0)
        return weighted_sum / denominator


@dataclass
class NonInteractiveCompletionResult:
    pointwise_components: torch.Tensor
    mean_components: torch.Tensor
    pointwise_values: torch.Tensor
    raw_completion: torch.Tensor
    projected_completion: torch.Tensor
    baseline_loss: torch.Tensor
    predictor_1: nn.Module
    predictor_2: nn.Module
    predictor_12: nn.Module
    additive_predictor: nn.Module
    label_prior: torch.Tensor
    block_indices: list[torch.Tensor]
    query_engine: NonInteractiveCompletionQueryEngine
    subset_mask: torch.Tensor | None = None
    available_counts: dict[str, int] | None = None

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

    @property
    def mean_components_bits(self) -> torch.Tensor:
        return self.mean_components / _LOG2

    @property
    def mean_values(self) -> torch.Tensor:
        return self.pointwise_values.mean(dim=0)

    @property
    def mean_values_bits(self) -> torch.Tensor:
        return self.mean_values / _LOG2

    @property
    def mean_raw_completion(self) -> torch.Tensor:
        return self.raw_completion.mean()

    @property
    def mean_raw_completion_bits(self) -> torch.Tensor:
        return self.mean_raw_completion / _LOG2

    @property
    def mean_projected_completion(self) -> torch.Tensor:
        return self.projected_completion.mean()

    @property
    def mean_projected_completion_bits(self) -> torch.Tensor:
        return self.mean_projected_completion / _LOG2

    @property
    def projection_delta(self) -> torch.Tensor:
        return self.projected_completion - self.raw_completion

    @property
    def mean_projection_delta(self) -> torch.Tensor:
        return self.projection_delta.mean()

    @property
    def mean_projection_delta_bits(self) -> torch.Tensor:
        return self.mean_projection_delta / _LOG2

    @property
    def mean_abs_projection_delta(self) -> torch.Tensor:
        return self.projection_delta.abs().mean()

    @property
    def mean_abs_projection_delta_bits(self) -> torch.Tensor:
        return self.mean_abs_projection_delta / _LOG2

    @property
    def clipped_fraction(self) -> torch.Tensor:
        return (self.projection_delta.abs() > 1e-8).float().mean()


class AdditiveLogitNetwork(nn.Module):
    def __init__(self, x1_dim: int, x2_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.modality_1 = nn.Sequential(
            nn.Linear(x1_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.modality_2 = nn.Sequential(
            nn.Linear(x2_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.modality_1(x1) + self.modality_2(x2) + self.bias


class AdditiveKernelRegressor(nn.Module):
    def __init__(
        self,
        train_x1: torch.Tensor,
        train_x2: torch.Tensor,
        dual_weights: torch.Tensor,
        bias: torch.Tensor,
        gamma_1: float,
        gamma_2: float,
        logit_scale: float,
    ):
        super().__init__()
        self.register_buffer("train_x1", train_x1.float())
        self.register_buffer("train_x2", train_x2.float())
        self.register_buffer("dual_weights", dual_weights.float())
        self.register_buffer("bias", bias.float())
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale), dtype=torch.float32))
        self.gamma_1 = float(gamma_1)
        self.gamma_2 = float(gamma_2)

    def modality_logits(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_1 = _rbf_kernel_torch(x1, self.train_x1, self.gamma_1)
        kernel_2 = _rbf_kernel_torch(x2, self.train_x2, self.gamma_2)
        return kernel_1 @ self.dual_weights, kernel_2 @ self.dual_weights

    def raw_logits(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        logits_1, logits_2 = self.modality_logits(x1.float(), x2.float())
        return logits_1 + logits_2 + self.bias

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.logit_scale * self.raw_logits(x1, x2)


class KernelLogitRegressor(nn.Module):
    def __init__(
        self,
        train_inputs: torch.Tensor,
        dual_weights: torch.Tensor,
        bias: torch.Tensor,
        gamma: float,
        logit_scale: float,
    ):
        super().__init__()
        self.register_buffer("train_inputs", train_inputs.float())
        self.register_buffer("dual_weights", dual_weights.float())
        self.register_buffer("bias", bias.float())
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale), dtype=torch.float32))
        self.gamma = float(gamma)

    def raw_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        kernel = _rbf_kernel_torch(inputs.float(), self.train_inputs, self.gamma)
        return kernel @ self.dual_weights + self.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logit_scale * self.raw_logits(inputs)


def estimate_non_interactive_completion_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    config: NonInteractiveCompletionConfig | None = None,
    device: str | torch.device | None = None,
) -> NonInteractiveCompletionResult:
    config = config or NonInteractiveCompletionConfig()
    device = _resolve_device(device)
    _maybe_set_seed(config.seed)

    x1 = _ensure_feature_batch(x1).float()
    x2 = _ensure_feature_batch(x2).float()
    labels = _prepare_labels(labels)
    num_labels = int(num_labels if num_labels is not None else int(labels.max().item()) + 1)
    train_dataset = train_dataset or MultimodalDataset([x1, x2], labels)
    test_dataset = test_dataset or MultimodalDataset([x1, x2], labels)
    label_prior = _estimate_label_prior(labels, num_labels).to(device)

    predictor_1, predictor_2, predictor_12 = _fit_predictors(
        train_dataset=train_dataset,
        x1_dim=x1.shape[1],
        x2_dim=x2.shape[1],
        num_labels=num_labels,
        config=config,
        device=device,
    )
    additive_predictor = _fit_additive_predictor(
        train_dataset=train_dataset,
        x1_dim=x1.shape[1],
        x2_dim=x2.shape[1],
        num_labels=num_labels,
        config=config,
        device=device,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        label_prior=label_prior,
    )
    baseline_loss, pointwise_values, raw_completion = _collect_value_functions(
        dataset=test_dataset,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        additive_predictor=additive_predictor,
        label_prior=label_prior,
        batch_size=config.batch_size,
        device=device,
    )
    projected_completion = _project_completion(pointwise_values, raw_completion)
    pointwise_components = _components_from_values(pointwise_values, projected_completion).detach().cpu()
    mean_components = pointwise_components.mean(dim=0)

    block_indices = _partition_indices(len(test_dataset), config.calibration_blocks, config.seed)
    query_engine = NonInteractiveCompletionQueryEngine(
        block_component_sums=_build_block_component_sums(pointwise_components, block_indices),
        block_sizes=torch.tensor([float(len(indices)) for indices in block_indices], dtype=torch.float32),
    )
    return NonInteractiveCompletionResult(
        pointwise_components=pointwise_components,
        mean_components=mean_components,
        pointwise_values=pointwise_values.detach().cpu(),
        raw_completion=raw_completion.detach().cpu(),
        projected_completion=projected_completion.detach().cpu(),
        baseline_loss=baseline_loss.detach().cpu(),
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        additive_predictor=additive_predictor,
        label_prior=label_prior.detach().cpu(),
        block_indices=block_indices,
        query_engine=query_engine,
        subset_mask=None,
        available_counts=None,
    )


def estimate_subset_mask_non_interactive_completion_pid(
    train_x1: torch.Tensor,
    train_x2: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    subset_mask: torch.Tensor,
    test_x1: torch.Tensor | None = None,
    test_x2: torch.Tensor | None = None,
    test_labels: torch.Tensor | None = None,
    num_labels: int | None = None,
    config: NonInteractiveCompletionConfig | None = None,
    device: str | torch.device | None = None,
) -> NonInteractiveCompletionResult:
    config = config or NonInteractiveCompletionConfig()
    device = _resolve_device(device)
    _maybe_set_seed(config.seed)

    train_x1 = _ensure_feature_batch(train_x1).float()
    train_x2 = _ensure_feature_batch(train_x2).float()
    train_labels = _prepare_labels(train_labels)
    subset_mask = torch.as_tensor(subset_mask).detach().cpu().bool()
    if subset_mask.ndim != 2 or subset_mask.shape != (train_x1.shape[0], 2):
        raise ValueError(
            f"subset_mask must have shape {(train_x1.shape[0], 2)} but got {tuple(subset_mask.shape)}"
        )

    if test_x1 is None:
        test_x1 = train_x1
    if test_x2 is None:
        test_x2 = train_x2
    if test_labels is None:
        test_labels = train_labels

    test_x1 = _ensure_feature_batch(test_x1).float()
    test_x2 = _ensure_feature_batch(test_x2).float()
    test_labels = _prepare_labels(test_labels)
    num_labels = int(num_labels if num_labels is not None else int(train_labels.max().item()) + 1)

    label_prior = _estimate_label_prior(train_labels, num_labels).to(device)
    modality_1_available = subset_mask[:, 0]
    modality_2_available = subset_mask[:, 1]
    any_available = modality_1_available | modality_2_available
    both_available = modality_1_available & modality_2_available

    predictor_1 = _fit_unimodal_predictor_or_none(
        inputs=train_x1[modality_1_available],
        labels=train_labels[modality_1_available],
        input_dim=train_x1.shape[1],
        num_labels=num_labels,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        epochs=config.predictor_epochs,
        lr=config.predictor_lr,
        predictor_model=config.predictor_model,
        kernel_gamma=config.kernel_gamma,
        kernel_ridge_lambda=config.kernel_ridge_lambda,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        device=device,
        verbose=config.verbose,
        tag="modality_1",
    )
    predictor_2 = _fit_unimodal_predictor_or_none(
        inputs=train_x2[modality_2_available],
        labels=train_labels[modality_2_available],
        input_dim=train_x2.shape[1],
        num_labels=num_labels,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        epochs=config.predictor_epochs,
        lr=config.predictor_lr,
        predictor_model=config.predictor_model,
        kernel_gamma=config.kernel_gamma,
        kernel_ridge_lambda=config.kernel_ridge_lambda,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        device=device,
        verbose=config.verbose,
        tag="modality_2",
    )

    masked_joint_dataset = _build_masked_joint_dataset(
        x1=train_x1,
        x2=train_x2,
        labels=train_labels,
        subset_mask=subset_mask,
    )
    predictor_12 = _fit_joint_predictor_or_none(
        dataset=masked_joint_dataset,
        input_dim=train_x1.shape[1] + train_x2.shape[1],
        num_labels=num_labels,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        epochs=config.predictor_epochs,
        lr=config.predictor_lr,
        predictor_model=config.predictor_model,
        kernel_gamma=config.kernel_gamma,
        kernel_ridge_lambda=config.kernel_ridge_lambda,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        device=device,
        verbose=config.verbose,
    )
    additive_predictor = _fit_additive_predictor_or_none(
        dataset=masked_joint_dataset,
        x1_dim=train_x1.shape[1],
        x2_dim=train_x2.shape[1],
        num_labels=num_labels,
        config=config,
        device=device,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        label_prior=label_prior,
    )

    test_dataset = MultimodalDataset([test_x1, test_x2], test_labels)
    baseline_loss, pointwise_values, raw_completion = _collect_value_functions_optional(
        dataset=test_dataset,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        additive_predictor=additive_predictor,
        label_prior=label_prior,
        batch_size=config.batch_size,
        device=device,
    )
    projected_completion = _project_completion(pointwise_values, raw_completion)
    pointwise_components = _components_from_values(pointwise_values, projected_completion).detach().cpu()
    mean_components = pointwise_components.mean(dim=0)
    block_indices = _partition_indices(len(test_dataset), config.calibration_blocks, config.seed)
    query_engine = NonInteractiveCompletionQueryEngine(
        block_component_sums=_build_block_component_sums(pointwise_components, block_indices),
        block_sizes=torch.tensor([float(len(indices)) for indices in block_indices], dtype=torch.float32),
    )
    return NonInteractiveCompletionResult(
        pointwise_components=pointwise_components,
        mean_components=mean_components,
        pointwise_values=pointwise_values.detach().cpu(),
        raw_completion=raw_completion.detach().cpu(),
        projected_completion=projected_completion.detach().cpu(),
        baseline_loss=baseline_loss.detach().cpu(),
        predictor_1=predictor_1 if predictor_1 is not None else _NullModule(),
        predictor_2=predictor_2 if predictor_2 is not None else _NullModule(),
        predictor_12=predictor_12 if predictor_12 is not None else _NullModule(),
        additive_predictor=additive_predictor if additive_predictor is not None else _NullModule(),
        label_prior=label_prior.detach().cpu(),
        block_indices=block_indices,
        query_engine=query_engine,
        subset_mask=subset_mask.to(torch.float32),
        available_counts={
            "modality_1": int(modality_1_available.sum().item()),
            "modality_2": int(modality_2_available.sum().item()),
            "both": int(both_available.sum().item()),
            "any": int(any_available.sum().item()),
            "total": int(train_x1.shape[0]),
        },
    )


def _fit_predictors(
    *,
    train_dataset: Dataset,
    x1_dim: int,
    x2_dim: int,
    num_labels: int,
    config: NonInteractiveCompletionConfig,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    predictor_model = config.predictor_model.lower()
    if predictor_model == "kernel":
        train_x1, train_x2, train_labels = _dataset_to_tensors(train_dataset)
        return (
            _fit_kernel_classifier(
                inputs=train_x1,
                labels=train_labels,
                num_labels=num_labels,
                gamma=config.kernel_gamma,
                ridge_lambda=config.kernel_ridge_lambda,
                device=device,
            ),
            _fit_kernel_classifier(
                inputs=train_x2,
                labels=train_labels,
                num_labels=num_labels,
                gamma=config.kernel_gamma,
                ridge_lambda=config.kernel_ridge_lambda,
                device=device,
            ),
            _fit_kernel_classifier(
                inputs=torch.cat([train_x1, train_x2], dim=1),
                labels=train_labels,
                num_labels=num_labels,
                gamma=config.kernel_gamma,
                ridge_lambda=config.kernel_ridge_lambda,
                device=device,
            ),
        )
    if predictor_model != "mlp":
        raise ValueError(f"unsupported predictor_model: {config.predictor_model}")
    models = (
        ClassificationNetwork(x1_dim, config.hidden_dim, num_labels).to(device),
        ClassificationNetwork(x2_dim, config.hidden_dim, num_labels).to(device),
        ClassificationNetwork(x1_dim + x2_dim, config.hidden_dim, num_labels).to(device),
    )
    optimizer = torch.optim.Adam([parameter for model in models for parameter in model.parameters()], lr=config.predictor_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(config.predictor_epochs // 2, 1), gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=False,
    )
    for epoch in range(config.predictor_epochs):
        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            x1_batch, x2_batch, y_batch = _dataset_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits_1 = models[0](x1_batch)
            logits_2 = models[1](x2_batch)
            logits_12 = models[2](torch.cat([x1_batch, x2_batch], dim=1))
            loss = criterion(logits_1, y_batch) + criterion(logits_2, y_batch) + criterion(logits_12, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x1_batch.shape[0]
            total_samples += x1_batch.shape[0]
        scheduler.step()
        if config.verbose and (epoch + 1) % 5 == 0:
            print(f"[non-interactive/predictors] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")
    for model in models:
        model.eval()
    return models


def _fit_unimodal_predictor_or_none(
    *,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    input_dim: int,
    num_labels: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    predictor_model: str,
    kernel_gamma: float | None,
    kernel_ridge_lambda: float,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    verbose: bool,
    tag: str,
) -> nn.Module | None:
    if inputs.shape[0] == 0:
        return None
    if predictor_model.lower() == "kernel":
        return _fit_kernel_classifier(
            inputs=inputs,
            labels=labels,
            num_labels=num_labels,
            gamma=kernel_gamma,
            ridge_lambda=kernel_ridge_lambda,
            device=device,
        )
    dataset = MultimodalDataset([inputs], labels)
    model = ClassificationNetwork(input_dim, hidden_dim, num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for batch in loader:
            x_batch = _ensure_feature_batch(batch[0]).float().to(device)
            y_batch = _prepare_labels(batch[1]).to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x_batch.shape[0]
            total_samples += x_batch.shape[0]
        scheduler.step()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"[non-interactive/{tag}] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")
    model.eval()
    return model


def _fit_joint_predictor_or_none(
    *,
    dataset: MultimodalDataset,
    input_dim: int,
    num_labels: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    predictor_model: str,
    kernel_gamma: float | None,
    kernel_ridge_lambda: float,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    verbose: bool,
) -> nn.Module | None:
    if len(dataset) == 0:
        return None
    if predictor_model.lower() == "kernel":
        x1, x2, labels = _dataset_to_tensors(dataset)
        return _fit_kernel_classifier(
            inputs=torch.cat([x1, x2], dim=1),
            labels=labels,
            num_labels=num_labels,
            gamma=kernel_gamma,
            ridge_lambda=kernel_ridge_lambda,
            device=device,
        )
    model = ClassificationNetwork(input_dim, hidden_dim, num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for batch in loader:
            x1_batch, x2_batch, y_batch = _dataset_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(torch.cat([x1_batch, x2_batch], dim=1))
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x1_batch.shape[0]
            total_samples += x1_batch.shape[0]
        scheduler.step()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"[non-interactive/joint-mask] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")
    model.eval()
    return model


def _fit_additive_predictor_or_none(
    *,
    dataset: MultimodalDataset,
    x1_dim: int,
    x2_dim: int,
    num_labels: int,
    config: NonInteractiveCompletionConfig,
    device: torch.device,
    predictor_1: nn.Module | None = None,
    predictor_2: nn.Module | None = None,
    predictor_12: nn.Module | None = None,
    label_prior: torch.Tensor | None = None,
) -> nn.Module | None:
    if len(dataset) == 0:
        return None
    return _fit_additive_predictor_from_dataset(
        dataset=dataset,
        x1_dim=x1_dim,
        x2_dim=x2_dim,
        num_labels=num_labels,
        config=config,
        device=device,
        tag="additive-mask",
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        label_prior=label_prior,
    )


def _fit_additive_predictor(
    *,
    train_dataset: Dataset,
    x1_dim: int,
    x2_dim: int,
    num_labels: int,
    config: NonInteractiveCompletionConfig,
    device: torch.device,
    predictor_1: nn.Module | None = None,
    predictor_2: nn.Module | None = None,
    predictor_12: nn.Module | None = None,
    label_prior: torch.Tensor | None = None,
) -> nn.Module:
    return _fit_additive_predictor_from_dataset(
        dataset=train_dataset,
        x1_dim=x1_dim,
        x2_dim=x2_dim,
        num_labels=num_labels,
        config=config,
        device=device,
        tag="additive",
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        label_prior=label_prior,
    )


def _fit_additive_predictor_from_dataset(
    *,
    dataset: Dataset,
    x1_dim: int,
    x2_dim: int,
    num_labels: int,
    config: NonInteractiveCompletionConfig,
    device: torch.device,
    tag: str,
    predictor_1: nn.Module | None,
    predictor_2: nn.Module | None,
    predictor_12: nn.Module | None,
    label_prior: torch.Tensor | None,
) -> nn.Module:
    additive_model = config.additive_model.lower()
    if additive_model == "mlp":
        model = AdditiveLogitNetwork(x1_dim, x2_dim, config.additive_hidden_dim, num_labels).to(device)
        return _fit_additive_mlp_predictor(model=model, dataset=dataset, config=config, device=device, tag=tag)
    if additive_model == "kernel":
        x1_train, x2_train, labels = _dataset_to_tensors(dataset)
        model = _fit_additive_kernel_predictor(
            x1=x1_train,
            x2=x2_train,
            labels=labels,
            num_labels=num_labels,
            config=config,
            device=device,
            predictor_1=predictor_1,
            predictor_2=predictor_2,
            predictor_12=predictor_12,
            label_prior=label_prior,
        )
        if config.verbose:
            print(
                f"[non-interactive/{tag}] model=kernel gamma_1={model.gamma_1:.4f} "
                f"gamma_2={model.gamma_2:.4f} ridge={config.kernel_ridge_lambda:.4f}"
            )
        return model
    raise ValueError(f"unsupported additive_model: {config.additive_model}")


def _fit_additive_mlp_predictor(
    *,
    model: AdditiveLogitNetwork,
    dataset: Dataset,
    config: NonInteractiveCompletionConfig,
    device: torch.device,
    tag: str,
) -> AdditiveLogitNetwork:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.additive_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(config.additive_epochs // 2, 1), gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=False,
    )
    for epoch in range(config.additive_epochs):
        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            x1_batch, x2_batch, y_batch = _dataset_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(x1_batch, x2_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x1_batch.shape[0]
            total_samples += x1_batch.shape[0]
        scheduler.step()
        if config.verbose and (epoch + 1) % 5 == 0:
            print(f"[non-interactive/{tag}] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")
    model.eval()
    return model


def _fit_additive_kernel_predictor(
    *,
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    config: NonInteractiveCompletionConfig,
    device: torch.device,
    predictor_1: nn.Module | None,
    predictor_2: nn.Module | None,
    predictor_12: nn.Module | None,
    label_prior: torch.Tensor | None,
) -> AdditiveKernelRegressor:
    x1 = _ensure_feature_batch(torch.as_tensor(x1).float()).detach().cpu()
    x2 = _ensure_feature_batch(torch.as_tensor(x2).float()).detach().cpu()
    labels = _prepare_labels(torch.as_tensor(labels)).detach().cpu()
    targets = torch.nn.functional.one_hot(labels, num_classes=num_labels).float()
    bias = targets.mean(dim=0)
    gamma_1 = _resolve_kernel_gamma(x1, config.kernel_gamma)
    gamma_2 = _resolve_kernel_gamma(x2, config.kernel_gamma)
    kernel_1 = _rbf_kernel_torch(x1, x1, gamma_1)
    kernel_2 = _rbf_kernel_torch(x2, x2, gamma_2)
    kernel_sum = kernel_1 + kernel_2
    regularized = kernel_sum + config.kernel_ridge_lambda * torch.eye(kernel_sum.shape[0], dtype=kernel_sum.dtype)
    rhs = targets - bias.unsqueeze(0)
    dual_weights = torch.linalg.solve(regularized, rhs)
    raw_train_logits = kernel_sum @ dual_weights + bias.unsqueeze(0)
    logit_scale = _fit_logit_scale(raw_train_logits, labels)
    if config.additive_epochs > 0 and config.additive_feasibility_lambda > 0.0:
        lower = upper = baseline = None
        if label_prior is not None:
            baseline, reference_values = _collect_reference_value_functions(
                x1=x1,
                x2=x2,
                labels=labels,
                predictor_1=predictor_1,
                predictor_2=predictor_2,
                predictor_12=predictor_12,
                label_prior=label_prior.detach().cpu(),
                batch_size=config.batch_size,
                device=device,
            )
            lower, upper = _completion_bounds_from_values(reference_values)
        dual_weights, logit_scale = _optimize_additive_kernel_parameters(
            kernel_sum=kernel_sum,
            bias=bias,
            labels=labels,
            initial_dual_weights=dual_weights,
            initial_logit_scale=logit_scale,
            baseline=baseline,
            lower=lower,
            upper=upper,
            lr=config.additive_lr,
            epochs=config.additive_epochs,
            feasibility_lambda=config.additive_feasibility_lambda,
            rkhs_lambda=config.additive_rkhs_lambda,
            anchor_lambda=config.additive_anchor_lambda,
        )
    model = AdditiveKernelRegressor(
        train_x1=x1,
        train_x2=x2,
        dual_weights=dual_weights,
        bias=bias,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        logit_scale=logit_scale,
    )
    model = model.to(device)
    model.eval()
    return model


def _collect_value_functions(
    *,
    dataset: Dataset,
    predictor_1: nn.Module,
    predictor_2: nn.Module,
    predictor_12: nn.Module,
    additive_predictor: nn.Module,
    label_prior: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    baseline_losses = []
    values = []
    raw_completion = []
    with torch.no_grad():
        for batch in loader:
            x1_batch, x2_batch, y_batch = _dataset_batch_to_device(batch, device)
            rows = torch.arange(y_batch.shape[0], device=device)
            baseline = -torch.log(label_prior[y_batch] + 1e-8)

            logits_1 = predictor_1(x1_batch)
            logits_2 = predictor_2(x2_batch)
            logits_12 = predictor_12(torch.cat([x1_batch, x2_batch], dim=1))
            logits_add = additive_predictor(x1_batch, x2_batch)

            log_probs_1 = torch.log_softmax(logits_1, dim=1)
            log_probs_2 = torch.log_softmax(logits_2, dim=1)
            log_probs_12 = torch.log_softmax(logits_12, dim=1)
            log_probs_add = torch.log_softmax(logits_add, dim=1)

            loss_1 = -log_probs_1[rows, y_batch]
            loss_2 = -log_probs_2[rows, y_batch]
            loss_12 = -log_probs_12[rows, y_batch]
            loss_add = -log_probs_add[rows, y_batch]

            v1 = baseline - loss_1
            v2 = baseline - loss_2
            v12 = baseline - loss_12
            c_raw = baseline - loss_add

            baseline_losses.append(baseline)
            values.append(torch.stack([v1, v2, v12], dim=1))
            raw_completion.append(c_raw)
    return (
        torch.cat(baseline_losses, dim=0).detach(),
        torch.cat(values, dim=0).detach(),
        torch.cat(raw_completion, dim=0).detach(),
    )


def _collect_value_functions_optional(
    *,
    dataset: Dataset,
    predictor_1: nn.Module | None,
    predictor_2: nn.Module | None,
    predictor_12: nn.Module | None,
    additive_predictor: nn.Module | None,
    label_prior: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    baseline_losses = []
    values = []
    raw_completion = []
    with torch.no_grad():
        for batch in loader:
            x1_batch, x2_batch, y_batch = _dataset_batch_to_device(batch, device)
            rows = torch.arange(y_batch.shape[0], device=device)
            baseline = -torch.log(label_prior[y_batch] + 1e-8)

            if predictor_1 is None:
                loss_1 = baseline
            else:
                log_probs_1 = torch.log_softmax(predictor_1(x1_batch), dim=1)
                loss_1 = -log_probs_1[rows, y_batch]
            if predictor_2 is None:
                loss_2 = baseline
            else:
                log_probs_2 = torch.log_softmax(predictor_2(x2_batch), dim=1)
                loss_2 = -log_probs_2[rows, y_batch]
            if predictor_12 is None:
                loss_12 = baseline
            else:
                log_probs_12 = torch.log_softmax(predictor_12(torch.cat([x1_batch, x2_batch], dim=1)), dim=1)
                loss_12 = -log_probs_12[rows, y_batch]
            if additive_predictor is None:
                loss_add = baseline
            else:
                log_probs_add = torch.log_softmax(additive_predictor(x1_batch, x2_batch), dim=1)
                loss_add = -log_probs_add[rows, y_batch]

            v1 = baseline - loss_1
            v2 = baseline - loss_2
            v12 = baseline - loss_12
            c_raw = baseline - loss_add

            baseline_losses.append(baseline)
            values.append(torch.stack([v1, v2, v12], dim=1))
            raw_completion.append(c_raw)
    return (
        torch.cat(baseline_losses, dim=0).detach(),
        torch.cat(values, dim=0).detach(),
        torch.cat(raw_completion, dim=0).detach(),
    )


def _project_completion(values: torch.Tensor, raw_completion: torch.Tensor) -> torch.Tensor:
    values = values.float()
    raw_completion = raw_completion.float()
    v1 = values[:, 0]
    v2 = values[:, 1]
    v12 = values[:, 2]
    lower = torch.maximum(v1, v2)
    upper = torch.minimum(torch.maximum(v12, lower), v1 + v2)
    upper = torch.maximum(upper, lower)
    return torch.minimum(torch.maximum(raw_completion, lower), upper)


def _components_from_values(values: torch.Tensor, completion: torch.Tensor) -> torch.Tensor:
    values = values.float()
    completion = completion.float()
    v1 = values[:, 0]
    v2 = values[:, 1]
    v12 = torch.maximum(values[:, 2], torch.maximum(v1, v2))
    redundancy = torch.clamp(v1 + v2 - completion, min=0.0)
    unique_1 = torch.clamp(completion - v2, min=0.0)
    unique_2 = torch.clamp(completion - v1, min=0.0)
    synergy = torch.clamp(v12 - completion, min=0.0)
    return torch.stack([redundancy, unique_1, unique_2, synergy], dim=1)


def _build_masked_joint_dataset(
    *,
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    subset_mask: torch.Tensor,
) -> MultimodalDataset:
    any_available = subset_mask[:, 0] | subset_mask[:, 1]
    x1_masked = x1[any_available].clone()
    x2_masked = x2[any_available].clone()
    mask_available = subset_mask[any_available]
    x1_masked[~mask_available[:, 0]] = 0.0
    x2_masked[~mask_available[:, 1]] = 0.0
    return MultimodalDataset([x1_masked, x2_masked], labels[any_available])


class _NullModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("Null module should not be called directly.")


def _dataset_to_tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, drop_last=False)
    batch = next(iter(loader))
    x1 = _ensure_feature_batch(batch[0]).float()
    x2 = _ensure_feature_batch(batch[1]).float()
    labels = _prepare_labels(batch[2])
    return x1, x2, labels


def _build_block_component_sums(pointwise_components: torch.Tensor, block_indices: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([pointwise_components[indices].sum(dim=0) for indices in block_indices], dim=0).float()


def _partition_indices(num_samples: int, num_blocks: int, seed: int | None) -> list[torch.Tensor]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    chunks = torch.chunk(permutation, min(num_blocks, num_samples))
    return [chunk for chunk in chunks if len(chunk) > 0]


def _dataset_batch_to_device(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = _ensure_feature_batch(batch[0]).float().to(device)
    x2 = _ensure_feature_batch(batch[1]).float().to(device)
    labels = _prepare_labels(batch[2]).to(device)
    return x1, x2, labels


def _estimate_label_prior(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_labels).float()
    counts /= counts.sum().clamp_min(1.0)
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


def _resolve_kernel_gamma(train_inputs: torch.Tensor, configured_gamma: float | None) -> float:
    if configured_gamma is not None:
        return float(configured_gamma)
    if train_inputs.shape[0] <= 1:
        return 1.0
    diffs = train_inputs[:, None, :] - train_inputs[None, :, :]
    sq_dists = (diffs * diffs).sum(dim=-1)
    positives = sq_dists[sq_dists > 0]
    if positives.numel() == 0:
        return 1.0
    return float(1.0 / positives.median().item())


def _rbf_kernel_torch(lhs: torch.Tensor, rhs: torch.Tensor, gamma: float) -> torch.Tensor:
    lhs = lhs.float()
    rhs = rhs.float()
    lhs_sq = (lhs * lhs).sum(dim=1, keepdim=True)
    rhs_sq = (rhs * rhs).sum(dim=1, keepdim=True).transpose(0, 1)
    sq_dist = lhs_sq + rhs_sq - 2.0 * lhs @ rhs.transpose(0, 1)
    return torch.exp(-gamma * torch.clamp(sq_dist, min=0.0))


def _fit_logit_scale(logits: torch.Tensor, labels: torch.Tensor, steps: int = 100) -> float:
    logits = logits.detach().float()
    labels = labels.detach().long()
    if logits.shape[0] == 0:
        return 1.0
    log_scale = nn.Parameter(torch.tensor(0.0, dtype=logits.dtype))
    optimizer = torch.optim.Adam([log_scale], lr=0.1)
    criterion = nn.CrossEntropyLoss()
    for _ in range(steps):
        optimizer.zero_grad()
        scaled_logits = torch.exp(log_scale) * logits
        loss = criterion(scaled_logits, labels)
        loss.backward()
        optimizer.step()
    return float(torch.exp(log_scale.detach()).item())


def _fit_kernel_classifier(
    *,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    gamma: float | None,
    ridge_lambda: float,
    device: torch.device,
) -> KernelLogitRegressor:
    inputs = _ensure_feature_batch(torch.as_tensor(inputs).float()).detach().cpu()
    labels = _prepare_labels(torch.as_tensor(labels)).detach().cpu()
    targets = torch.nn.functional.one_hot(labels, num_classes=num_labels).float()
    bias = targets.mean(dim=0)
    resolved_gamma = _resolve_kernel_gamma(inputs, gamma)
    kernel = _rbf_kernel_torch(inputs, inputs, resolved_gamma)
    regularized = kernel + ridge_lambda * torch.eye(kernel.shape[0], dtype=kernel.dtype)
    dual_weights = torch.linalg.solve(regularized, targets - bias.unsqueeze(0))
    raw_logits = kernel @ dual_weights + bias.unsqueeze(0)
    logit_scale = _fit_logit_scale(raw_logits, labels)
    model = KernelLogitRegressor(
        train_inputs=inputs,
        dual_weights=dual_weights,
        bias=bias,
        gamma=resolved_gamma,
        logit_scale=logit_scale,
    )
    model = model.to(device)
    model.eval()
    return model


def _collect_reference_value_functions(
    *,
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    predictor_1: nn.Module | None,
    predictor_2: nn.Module | None,
    predictor_12: nn.Module | None,
    label_prior: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MultimodalDataset([x1, x2], labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    baseline_losses = []
    values = []
    with torch.no_grad():
        for batch in loader:
            x1_batch, x2_batch, y_batch = _dataset_batch_to_device(batch, device)
            rows = torch.arange(y_batch.shape[0], device=device)
            baseline = -torch.log(label_prior[y_batch] + 1e-8)
            if predictor_1 is None:
                loss_1 = baseline
            else:
                loss_1 = -torch.log_softmax(predictor_1(x1_batch), dim=1)[rows, y_batch]
            if predictor_2 is None:
                loss_2 = baseline
            else:
                loss_2 = -torch.log_softmax(predictor_2(x2_batch), dim=1)[rows, y_batch]
            if predictor_12 is None:
                loss_12 = baseline
            else:
                joint_inputs = torch.cat([x1_batch, x2_batch], dim=1)
                loss_12 = -torch.log_softmax(predictor_12(joint_inputs), dim=1)[rows, y_batch]
            baseline_losses.append(baseline.detach().cpu())
            values.append(torch.stack([baseline - loss_1, baseline - loss_2, baseline - loss_12], dim=1).detach().cpu())
    return torch.cat(baseline_losses, dim=0), torch.cat(values, dim=0)


def _completion_bounds_from_values(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    v1 = values[:, 0]
    v2 = values[:, 1]
    v12 = values[:, 2]
    lower = torch.maximum(v1, v2)
    upper = torch.minimum(torch.maximum(v12, lower), v1 + v2)
    upper = torch.maximum(upper, lower)
    return lower, upper


def _optimize_additive_kernel_parameters(
    *,
    kernel_sum: torch.Tensor,
    bias: torch.Tensor,
    labels: torch.Tensor,
    initial_dual_weights: torch.Tensor,
    initial_logit_scale: float,
    baseline: torch.Tensor | None,
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
    lr: float,
    epochs: int,
    feasibility_lambda: float,
    rkhs_lambda: float,
    anchor_lambda: float,
) -> tuple[torch.Tensor, float]:
    if epochs <= 0:
        return initial_dual_weights.detach().cpu(), float(initial_logit_scale)
    labels = labels.detach().long()
    kernel_sum = kernel_sum.detach().float()
    bias = bias.detach().float()
    initial_dual_weights = initial_dual_weights.detach().float()
    initial_log_scale = torch.log(torch.tensor(float(initial_logit_scale), dtype=torch.float32).clamp_min(1e-6))
    dual_weights = nn.Parameter(initial_dual_weights.clone())
    log_scale = nn.Parameter(initial_log_scale.clone())
    optimizer = torch.optim.Adam([dual_weights, log_scale], lr=lr)
    criterion = nn.CrossEntropyLoss()
    baseline = baseline.float() if baseline is not None else None
    lower = lower.float() if lower is not None else None
    upper = upper.float() if upper is not None else None
    for _ in range(epochs):
        optimizer.zero_grad()
        raw_logits = kernel_sum @ dual_weights + bias.unsqueeze(0)
        scaled_logits = torch.exp(log_scale) * raw_logits
        ce_loss = criterion(scaled_logits, labels)
        feasibility_loss = ce_loss * 0.0
        if baseline is not None and lower is not None and upper is not None and feasibility_lambda > 0.0:
            rows = torch.arange(labels.shape[0])
            log_probs = torch.log_softmax(scaled_logits, dim=1)
            loss_add = -log_probs[rows, labels]
            raw_completion = baseline - loss_add
            feasibility_loss = (
                torch.relu(lower - raw_completion).pow(2).mean()
                + torch.relu(raw_completion - upper).pow(2).mean()
            )
        rkhs_loss = dual_weights.mul(kernel_sum @ dual_weights).sum() / max(kernel_sum.shape[0], 1)
        anchor_loss = (dual_weights - initial_dual_weights).pow(2).mean() + (log_scale - initial_log_scale).pow(2)
        loss = ce_loss + feasibility_lambda * feasibility_loss + rkhs_lambda * rkhs_loss + anchor_lambda * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([dual_weights, log_scale], max_norm=10.0)
        optimizer.step()
    return dual_weights.detach().cpu(), float(torch.exp(log_scale.detach()).item())


__all__ = [
    "AdditiveKernelRegressor",
    "AdditiveLogitNetwork",
    "KernelLogitRegressor",
    "NonInteractiveCompletionConfig",
    "NonInteractiveCompletionQueryEngine",
    "NonInteractiveCompletionResult",
    "estimate_non_interactive_completion_pid",
    "estimate_subset_mask_non_interactive_completion_pid",
]
