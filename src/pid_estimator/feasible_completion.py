from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .data import MultimodalDataset
from .non_interactive_completion import (
    AdditiveKernelRegressor,
    AdditiveLogitNetwork,
    KernelLogitRegressor,
    NonInteractiveCompletionConfig,
    NonInteractiveCompletionQueryEngine,
    _LOG2,
    _build_block_component_sums,
    _collect_value_functions,
    _components_from_values,
    _ensure_feature_batch,
    _estimate_label_prior,
    _fit_additive_predictor,
    _fit_predictors,
    _maybe_set_seed,
    _partition_indices,
    _prepare_labels,
    _project_completion,
    _rbf_kernel_torch,
    _resolve_device,
    _resolve_kernel_gamma,
)


@dataclass(frozen=True)
class FeasibleCompletionConfig:
    hidden_dim: int = 64
    additive_hidden_dim: int = 64
    predictor_model: str = "kernel"
    additive_model: str = "kernel"
    batch_size: int = 64
    predictor_epochs: int = 10
    additive_epochs: int = 10
    predictor_lr: float = 1e-3
    additive_lr: float = 1e-3
    kernel_gamma: float | None = None
    kernel_ridge_lambda: float = 1e-2
    completion_gamma: float | None = None
    completion_ridge_lambda: float = 1e-2
    calibration_blocks: int = 8
    seed: int | None = None
    verbose: bool = False


@dataclass
class FeasibleCompletionResult:
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
    completion_head: nn.Module
    label_prior: torch.Tensor
    pointwise_alpha: torch.Tensor
    block_indices: list[torch.Tensor]
    query_engine: NonInteractiveCompletionQueryEngine

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
    def mean_raw_completion(self) -> torch.Tensor:
        return self.raw_completion.mean()

    @property
    def mean_projected_completion(self) -> torch.Tensor:
        return self.projected_completion.mean()

    @property
    def projection_delta(self) -> torch.Tensor:
        return self.projected_completion - self.raw_completion

    @property
    def mean_abs_projection_delta(self) -> torch.Tensor:
        return self.projection_delta.abs().mean()

    @property
    def clipped_fraction(self) -> torch.Tensor:
        return (self.projection_delta.abs() > 1e-8).float().mean()


class KernelCompletionHead(nn.Module):
    def __init__(
        self,
        train_features: torch.Tensor,
        dual_weights: torch.Tensor,
        bias: torch.Tensor,
        gamma: float,
    ):
        super().__init__()
        self.register_buffer("train_features", train_features.float())
        self.register_buffer("dual_weights", dual_weights.float())
        self.register_buffer("bias", bias.float())
        self.gamma = float(gamma)

    def raw_scores(self, features: torch.Tensor) -> torch.Tensor:
        kernel = _rbf_kernel_torch(features.float(), self.train_features, self.gamma)
        return (kernel @ self.dual_weights).squeeze(-1) + self.bias.squeeze()

    def alpha(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.raw_scores(features))

    def forward(self, features: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.alpha(features)
        width = torch.clamp(upper - lower, min=0.0)
        completion = lower + alpha * width
        return completion, alpha


def estimate_feasible_completion_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    config: FeasibleCompletionConfig | None = None,
    device: str | torch.device | None = None,
) -> FeasibleCompletionResult:
    config = config or FeasibleCompletionConfig()
    device = _resolve_device(device)
    _maybe_set_seed(config.seed)

    x1 = _ensure_feature_batch(x1).float()
    x2 = _ensure_feature_batch(x2).float()
    labels = _prepare_labels(labels)
    num_labels = int(num_labels if num_labels is not None else int(labels.max().item()) + 1)
    train_dataset = train_dataset or MultimodalDataset([x1, x2], labels)
    test_dataset = test_dataset or MultimodalDataset([x1, x2], labels)
    label_prior = _estimate_label_prior(labels, num_labels).to(device)

    non_interactive_config = NonInteractiveCompletionConfig(
        hidden_dim=config.hidden_dim,
        additive_hidden_dim=config.additive_hidden_dim,
        predictor_model=config.predictor_model,
        additive_model=config.additive_model,
        batch_size=config.batch_size,
        predictor_epochs=config.predictor_epochs,
        additive_epochs=config.additive_epochs,
        predictor_lr=config.predictor_lr,
        additive_lr=config.additive_lr,
        kernel_gamma=config.kernel_gamma,
        kernel_ridge_lambda=config.kernel_ridge_lambda,
        calibration_blocks=config.calibration_blocks,
        seed=config.seed,
        verbose=config.verbose,
    )

    predictor_1, predictor_2, predictor_12 = _fit_predictors(
        train_dataset=train_dataset,
        x1_dim=x1.shape[1],
        x2_dim=x2.shape[1],
        num_labels=num_labels,
        config=non_interactive_config,
        device=device,
    )
    additive_predictor = _fit_additive_predictor(
        train_dataset=train_dataset,
        x1_dim=x1.shape[1],
        x2_dim=x2.shape[1],
        num_labels=num_labels,
        config=non_interactive_config,
        device=device,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        label_prior=label_prior,
    )

    train_baseline, train_values, train_raw_completion = _collect_value_functions(
        dataset=train_dataset,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        additive_predictor=additive_predictor,
        label_prior=label_prior,
        batch_size=config.batch_size,
        device=device,
    )
    train_values = train_values.detach().cpu()
    train_projected_completion = _project_completion(train_values, train_raw_completion.detach().cpu())
    train_lower, train_upper = _completion_bounds(train_values)
    train_features = _completion_feature_map(train_values)
    train_targets = _alpha_targets(train_projected_completion, train_lower, train_upper)
    completion_head = _fit_kernel_completion_head(
        features=train_features,
        alpha_targets=train_targets,
        gamma=config.completion_gamma,
        ridge_lambda=config.completion_ridge_lambda,
        device=device,
    )

    test_baseline, test_values, _unused_raw_completion = _collect_predictive_values_only(
        dataset=test_dataset,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        label_prior=label_prior,
        batch_size=config.batch_size,
        device=device,
    )
    test_values = test_values.detach().cpu()
    test_lower, test_upper = _completion_bounds(test_values)
    test_features = _completion_feature_map(test_values)
    with torch.no_grad():
        completion, alpha = completion_head(test_features.to(device), test_lower.to(device), test_upper.to(device))
    raw_completion = completion.detach().cpu()
    projected_completion = raw_completion.clone()
    pointwise_components = _components_from_values(test_values, projected_completion).detach().cpu()
    mean_components = pointwise_components.mean(dim=0)
    block_indices = _partition_indices(len(test_dataset), config.calibration_blocks, config.seed)
    query_engine = NonInteractiveCompletionQueryEngine(
        block_component_sums=_build_block_component_sums(pointwise_components, block_indices),
        block_sizes=torch.tensor([float(len(indices)) for indices in block_indices], dtype=torch.float32),
    )
    return FeasibleCompletionResult(
        pointwise_components=pointwise_components,
        mean_components=mean_components,
        pointwise_values=test_values,
        raw_completion=raw_completion,
        projected_completion=projected_completion,
        baseline_loss=test_baseline.detach().cpu(),
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        additive_predictor=additive_predictor,
        completion_head=completion_head,
        label_prior=label_prior.detach().cpu(),
        pointwise_alpha=alpha.detach().cpu(),
        block_indices=block_indices,
        query_engine=query_engine,
    )


def _completion_feature_map(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    lower, upper = _completion_bounds(values)
    v1 = values[:, 0:1]
    v2 = values[:, 1:2]
    v12 = values[:, 2:3]
    lower_col = lower.unsqueeze(1)
    upper_col = upper.unsqueeze(1)
    width = (upper - lower).unsqueeze(1)
    return torch.cat(
        [
            v1,
            v2,
            v12,
            lower_col,
            upper_col,
            width,
            torch.minimum(v1, v2),
            torch.maximum(v1, v2),
            v1 + v2,
            v12 - lower_col,
            v12 - (v1 + v2),
            v1 * v2,
        ],
        dim=1,
    ).float()


def _completion_bounds(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    v1 = values[:, 0]
    v2 = values[:, 1]
    v12 = values[:, 2]
    lower = torch.maximum(v1, v2)
    upper = torch.minimum(torch.maximum(v12, lower), v1 + v2)
    upper = torch.maximum(upper, lower)
    return lower, upper


def _alpha_targets(completion: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    width = torch.clamp(upper - lower, min=0.0)
    alpha = torch.where(width > 1e-8, (completion - lower) / width.clamp_min(1e-8), torch.zeros_like(completion))
    return alpha.clamp(min=eps, max=1.0 - eps)


def _fit_kernel_completion_head(
    *,
    features: torch.Tensor,
    alpha_targets: torch.Tensor,
    gamma: float | None,
    ridge_lambda: float,
    device: torch.device,
) -> KernelCompletionHead:
    features = features.detach().cpu().float()
    alpha_targets = alpha_targets.detach().cpu().float()
    logits_target = torch.logit(alpha_targets).unsqueeze(1)
    bias = logits_target.mean(dim=0)
    resolved_gamma = _resolve_kernel_gamma(features, gamma)
    kernel = _rbf_kernel_torch(features, features, resolved_gamma)
    regularized = kernel + ridge_lambda * torch.eye(kernel.shape[0], dtype=kernel.dtype)
    dual_weights = torch.linalg.solve(regularized, logits_target - bias.unsqueeze(0))
    head = KernelCompletionHead(
        train_features=features,
        dual_weights=dual_weights,
        bias=bias,
        gamma=resolved_gamma,
    )
    head = head.to(device)
    head.eval()
    return head


def _collect_predictive_values_only(
    *,
    dataset: Dataset,
    predictor_1: nn.Module,
    predictor_2: nn.Module,
    predictor_12: nn.Module,
    label_prior: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    baseline_losses = []
    values = []
    raw_placeholder = []
    with torch.no_grad():
        for batch in loader:
            x1_batch = _ensure_feature_batch(batch[0]).float().to(device)
            x2_batch = _ensure_feature_batch(batch[1]).float().to(device)
            y_batch = _prepare_labels(batch[2]).to(device)
            rows = torch.arange(y_batch.shape[0], device=device)
            baseline = -torch.log(label_prior[y_batch] + 1e-8)
            loss_1 = -torch.log_softmax(predictor_1(x1_batch), dim=1)[rows, y_batch]
            loss_2 = -torch.log_softmax(predictor_2(x2_batch), dim=1)[rows, y_batch]
            loss_12 = -torch.log_softmax(predictor_12(torch.cat([x1_batch, x2_batch], dim=1)), dim=1)[rows, y_batch]
            baseline_losses.append(baseline.detach())
            values.append(torch.stack([baseline - loss_1, baseline - loss_2, baseline - loss_12], dim=1).detach())
            raw_placeholder.append(torch.zeros_like(baseline))
    return torch.cat(baseline_losses, dim=0), torch.cat(values, dim=0), torch.cat(raw_placeholder, dim=0)


__all__ = [
    "FeasibleCompletionConfig",
    "FeasibleCompletionResult",
    "KernelCompletionHead",
    "estimate_feasible_completion_pid",
]
