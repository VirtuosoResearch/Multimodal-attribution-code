from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import MultimodalDataset
from .lsmi import ClassificationNetwork, LSMIConfig, estimate_lsmi


@dataclass(frozen=True)
class SurrogatePIDConfig:
    predictor_hidden_dim: int = 64
    predictor_epochs: int = 10
    predictor_lr: float = 1e-3
    batch_size: int = 64
    num_workers: int = 0
    shuffle: bool = True
    seed: int | None = None
    head_type: str = "mlp"
    head_hidden_dim: int = 32
    head_epochs: int = 200
    head_lr: float = 5e-3
    kernel_head_gamma: float | None = None
    kernel_head_ridge_lambda: float = 1e-2
    kernel_head_center_targets: bool = True
    calibration_blocks: int = 8
    calibration_subsets: int = 12
    lambda_semantic: float = 0.1
    lambda_nonnegative: float = 0.05
    lambda_mobius_prior: float = 0.1
    reference_estimator: str = "lsmi"
    reference_lsmi_discriminator_epochs: int = 5
    reference_lsmi_entropy_epochs: int = 5
    verbose: bool = False


@dataclass
class CalibrationRecord:
    subset_weights: torch.Tensor
    selected_indices: torch.Tensor
    target_components: torch.Tensor


@dataclass
class SurrogateQueryEngine:
    block_component_sums: torch.Tensor
    block_sizes: torch.Tensor

    def query(self, subset_weights: torch.Tensor) -> torch.Tensor:
        weights = subset_weights.float().view(-1, 1)
        weighted_sum = torch.sum(self.block_component_sums * weights, dim=0)
        denominator = torch.sum(self.block_sizes * subset_weights.float()).clamp_min(1.0)
        return weighted_sum / denominator


@dataclass
class KernelPIDHead:
    train_inputs: torch.Tensor
    alpha: torch.Tensor
    gamma: float
    full_components: torch.Tensor
    center_targets: bool

    def predict_components(self, summary_features: torch.Tensor) -> torch.Tensor:
        if summary_features.ndim == 1:
            summary_features = summary_features.unsqueeze(0)
        train_inputs = self.train_inputs.float().to(summary_features.device)
        alpha = self.alpha.float().to(summary_features.device)
        kernel = _rbf_kernel_torch(summary_features.float(), train_inputs, self.gamma)
        predictions = kernel @ alpha
        if self.center_targets:
            predictions = predictions + self.full_components.float().to(summary_features.device)
        return predictions


@dataclass
class KernelHeadQueryEngine:
    block_component_sums: torch.Tensor
    block_feature_sums: torch.Tensor
    block_sizes: torch.Tensor
    pid_head: KernelPIDHead

    def query(self, subset_weights: torch.Tensor) -> torch.Tensor:
        weights = subset_weights.float().view(-1, 1)
        component_sum = torch.sum(self.block_component_sums * weights, dim=0)
        denominator = torch.sum(self.block_sizes * subset_weights.float()).clamp_min(1.0)
        baseline = component_sum / denominator
        weighted_sum = torch.sum(self.block_feature_sums * weights, dim=0)
        summary_features = weighted_sum / denominator
        kernel_inputs = _kernel_residual_features(summary_features, baseline)
        residual = self.pid_head.predict_components(kernel_inputs).squeeze(0)
        return _project_pid_components(baseline + residual)


@dataclass
class SurrogatePIDResult:
    pointwise_components: torch.Tensor
    mean_components: torch.Tensor
    pointwise_predictive_scores: torch.Tensor
    calibration_records: list[CalibrationRecord]
    predictor_1: nn.Module
    predictor_2: nn.Module
    predictor_12: nn.Module
    redundancy_head: Any
    label_prior: torch.Tensor
    block_indices: list[torch.Tensor]
    query_engine: Any
    head_type: str

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


class RedundancyHead(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        min_info = F.relu(torch.minimum(scores[:, 0], scores[:, 1]))
        weight = torch.sigmoid(self.net(scores).squeeze(-1))
        return weight * min_info


class MobiusRedundancyHead(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(scores).squeeze(-1))


def estimate_surrogate_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    config: SurrogatePIDConfig | None = None,
    device: str | torch.device | None = None,
) -> SurrogatePIDResult:
    config = config or SurrogatePIDConfig()
    device = _resolve_device(device)
    _maybe_set_seed(config.seed)

    x1 = _ensure_feature_batch(x1).float()
    x2 = _ensure_feature_batch(x2).float()
    labels = _prepare_labels(labels)
    num_labels = int(num_labels if num_labels is not None else int(labels.max().item()) + 1)
    train_dataset = train_dataset or MultimodalDataset([x1, x2], labels)
    test_dataset = test_dataset or MultimodalDataset([x1, x2], labels)

    predictor_1, predictor_2, predictor_12 = _fit_predictors(
        train_dataset=train_dataset,
        x1_dim=x1.shape[1],
        x2_dim=x2.shape[1],
        num_labels=num_labels,
        config=config,
        device=device,
    )
    label_prior = _estimate_label_prior(labels, num_labels).to(device)

    train_scores = _collect_predictive_scores(train_dataset, predictor_1, predictor_2, predictor_12, label_prior, config.batch_size, device)
    calibration_records = _build_calibration_records(
        train_dataset=train_dataset,
        num_labels=num_labels,
        config=config,
        device=device,
    )

    test_scores = _collect_predictive_scores(test_dataset, predictor_1, predictor_2, predictor_12, label_prior, config.batch_size, device)
    block_indices = _partition_indices(len(test_dataset), config.calibration_blocks, config.seed)
    if config.head_type == "mlp":
        redundancy_head = RedundancyHead(input_dim=3, hidden_dim=config.head_hidden_dim).to(device)
        _fit_redundancy_head(
            redundancy_head=redundancy_head,
            predictive_scores=train_scores,
            calibration_records=calibration_records,
            config=config,
            device=device,
        )
        pointwise_components = _compute_components_from_scores(test_scores, redundancy_head).detach().cpu()
        mean_components = pointwise_components.mean(dim=0)
        query_engine = _build_query_engine(pointwise_components, block_indices)
    elif config.head_type == "mobius":
        redundancy_head = MobiusRedundancyHead(input_dim=3, hidden_dim=config.head_hidden_dim).to(device)
        _fit_mobius_head(
            redundancy_head=redundancy_head,
            predictive_scores=train_scores,
            calibration_records=calibration_records,
            config=config,
            device=device,
        )
        pointwise_components = _compute_mobius_components_from_scores(test_scores, redundancy_head).detach().cpu()
        mean_components = pointwise_components.mean(dim=0)
        query_engine = _build_query_engine(pointwise_components, block_indices)
    elif config.head_type == "uncertainty_mobius":
        redundancy_head = MobiusRedundancyHead(input_dim=11, hidden_dim=config.head_hidden_dim).to(device)
        _fit_uncertainty_mobius_head(
            redundancy_head=redundancy_head,
            predictive_scores=train_scores,
            calibration_records=calibration_records,
            config=config,
            device=device,
        )
        pointwise_components = _compute_uncertainty_mobius_components_from_scores(test_scores, redundancy_head).detach().cpu()
        mean_components = pointwise_components.mean(dim=0)
        query_engine = _build_query_engine(pointwise_components, block_indices)
    elif config.head_type == "kernel":
        baseline_head = RedundancyHead(input_dim=3, hidden_dim=config.head_hidden_dim).to(device)
        _fit_redundancy_head(
            redundancy_head=baseline_head,
            predictive_scores=train_scores,
            calibration_records=calibration_records,
            config=config,
            device=device,
        )
        train_pointwise = _compute_components_from_scores(train_scores, baseline_head).detach().cpu()
        test_pointwise = _compute_components_from_scores(test_scores, baseline_head).detach().cpu()
        redundancy_head = _fit_kernel_pid_residual_head(
            predictive_scores=train_scores.detach().cpu(),
            baseline_pointwise=train_pointwise,
            calibration_records=calibration_records,
            config=config,
        )
        block_feature_sums = _build_block_feature_sums(test_scores.detach().cpu(), block_indices)
        block_component_sums = _build_block_component_sums(test_pointwise, block_indices)
        query_engine = KernelHeadQueryEngine(
            block_component_sums=block_component_sums,
            block_feature_sums=block_feature_sums,
            block_sizes=torch.tensor([float(len(indices)) for indices in block_indices], dtype=torch.float32),
            pid_head=redundancy_head,
        )
        mean_components = query_engine.query(torch.ones(len(block_indices), dtype=torch.float32))
        pointwise_components = test_pointwise
    else:
        raise ValueError(f"unsupported head_type: {config.head_type}")

    return SurrogatePIDResult(
        pointwise_components=pointwise_components,
        mean_components=mean_components,
        pointwise_predictive_scores=test_scores.detach().cpu(),
        calibration_records=calibration_records,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        redundancy_head=redundancy_head,
        label_prior=label_prior.detach().cpu(),
        block_indices=block_indices,
        query_engine=query_engine,
        head_type=config.head_type,
    )


def _fit_predictors(
    *,
    train_dataset: Dataset,
    x1_dim: int,
    x2_dim: int,
    num_labels: int,
    config: SurrogatePIDConfig,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    models = (
        ClassificationNetwork(x1_dim, config.predictor_hidden_dim, num_labels).to(device),
        ClassificationNetwork(x2_dim, config.predictor_hidden_dim, num_labels).to(device),
        ClassificationNetwork(x1_dim + x2_dim, config.predictor_hidden_dim, num_labels).to(device),
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
            loss = (
                criterion(logits_1, y_batch)
                + criterion(logits_2, y_batch)
                + criterion(logits_12, y_batch)
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x1_batch.shape[0]
            total_samples += x1_batch.shape[0]
        scheduler.step()
        if config.verbose and (epoch + 1) % 5 == 0:
            print(f"[surrogate/predictor] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")
    for model in models:
        model.eval()
    return models


def _collect_predictive_scores(
    dataset: Dataset,
    predictor_1: nn.Module,
    predictor_2: nn.Module,
    predictor_12: nn.Module,
    label_prior: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    collected = []
    with torch.no_grad():
        for batch in loader:
            x1_batch, x2_batch, y_batch = _dataset_batch_to_device(batch, device)
            log_prior = torch.log(label_prior[y_batch] + 1e-8)
            rows = torch.arange(y_batch.shape[0], device=device)
            logits_1 = predictor_1(x1_batch)
            logits_2 = predictor_2(x2_batch)
            logits_12 = predictor_12(torch.cat([x1_batch, x2_batch], dim=1))
            log_probs_1 = torch.log_softmax(logits_1, dim=1)
            log_probs_2 = torch.log_softmax(logits_2, dim=1)
            log_probs_12 = torch.log_softmax(logits_12, dim=1)
            probs_1 = torch.softmax(logits_1, dim=1)
            probs_2 = torch.softmax(logits_2, dim=1)
            probs_12 = torch.softmax(logits_12, dim=1)
            m1 = log_probs_1[rows, y_batch] - log_prior
            m2 = log_probs_2[rows, y_batch] - log_prior
            m12 = log_probs_12[rows, y_batch] - log_prior
            h1 = -(probs_1 * log_probs_1).sum(dim=1)
            h2 = -(probs_2 * log_probs_2).sum(dim=1)
            h12 = -(probs_12 * log_probs_12).sum(dim=1)
            collected.append(torch.stack([m1, m2, m12, h1, h2, h12], dim=1))
    return torch.cat(collected, dim=0)


def _build_calibration_records(
    *,
    train_dataset: Dataset,
    num_labels: int,
    config: SurrogatePIDConfig,
    device: torch.device,
) -> list[CalibrationRecord]:
    block_indices = _partition_indices(len(train_dataset), config.calibration_blocks, config.seed)
    subset_weights_list = _generate_subset_weights(len(block_indices), config.calibration_subsets, config.seed)
    records = []
    for subset_weights in subset_weights_list:
        selected_indices = _indices_from_subset_weights(block_indices, subset_weights)
        subset_dataset = _subset_dataset(train_dataset, selected_indices)
        target_components = _reference_components(
            subset_dataset=subset_dataset,
            num_labels=num_labels,
            config=config,
            device=device,
        )
        records.append(
            CalibrationRecord(
                subset_weights=subset_weights.detach().cpu(),
                selected_indices=selected_indices.detach().cpu(),
                target_components=target_components.detach().cpu(),
            )
        )
    return records


def _fit_redundancy_head(
    *,
    redundancy_head: nn.Module,
    predictive_scores: torch.Tensor,
    calibration_records: list[CalibrationRecord],
    config: SurrogatePIDConfig,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(redundancy_head.parameters(), lr=config.head_lr)
    predictive_scores = predictive_scores.to(device)
    record_indices = [record.selected_indices.to(device) for record in calibration_records]
    record_targets = torch.stack([record.target_components for record in calibration_records], dim=0).to(device)

    for epoch in range(config.head_epochs):
        optimizer.zero_grad()
        pointwise = _compute_components_from_scores(predictive_scores, redundancy_head)
        calibration_predictions = torch.stack(
            [pointwise[indices].mean(dim=0) for indices in record_indices],
            dim=0,
        )
        calibration_loss = F.mse_loss(calibration_predictions, record_targets)
        semantic_loss = _semantic_loss(pointwise, predictive_scores)
        nonnegative_loss = _nonnegative_loss(pointwise)
        loss = calibration_loss + config.lambda_semantic * semantic_loss + config.lambda_nonnegative * nonnegative_loss
        loss.backward()
        optimizer.step()
        if config.verbose and (epoch + 1) % 50 == 0:
            print(
                "[surrogate/head] "
                f"epoch={epoch + 1} calib={calibration_loss.item():.4f} "
                f"sem={semantic_loss.item():.4f} nonneg={nonnegative_loss.item():.4f}"
            )
    redundancy_head.eval()


def _fit_mobius_head(
    *,
    redundancy_head: nn.Module,
    predictive_scores: torch.Tensor,
    calibration_records: list[CalibrationRecord],
    config: SurrogatePIDConfig,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(redundancy_head.parameters(), lr=config.head_lr)
    predictive_scores = predictive_scores.to(device)
    record_indices = [record.selected_indices.to(device) for record in calibration_records]
    record_targets = torch.stack([record.target_components for record in calibration_records], dim=0).to(device)

    for epoch in range(config.head_epochs):
        optimizer.zero_grad()
        pointwise = _compute_mobius_components_from_scores(predictive_scores, redundancy_head)
        calibration_predictions = torch.stack(
            [pointwise[indices].mean(dim=0) for indices in record_indices],
            dim=0,
        )
        calibration_loss = F.mse_loss(calibration_predictions, record_targets)
        mobius_prior_loss = _mobius_prior_loss(predictive_scores, redundancy_head)
        loss = calibration_loss + config.lambda_mobius_prior * mobius_prior_loss
        loss.backward()
        optimizer.step()
        if config.verbose and (epoch + 1) % 50 == 0:
            print(
                "[surrogate/mobius] "
                f"epoch={epoch + 1} calib={calibration_loss.item():.4f} "
                f"prior={mobius_prior_loss.item():.4f}"
            )
    redundancy_head.eval()


def _compute_components_from_scores(scores: torch.Tensor, redundancy_head: nn.Module) -> torch.Tensor:
    redundancy = redundancy_head(_core_scores(scores))
    return _compute_components_from_scores_with_redundancy(scores, redundancy)


def _compute_mobius_components_from_scores(scores: torch.Tensor, redundancy_head: nn.Module) -> torch.Tensor:
    scores = scores.float()
    m1 = torch.clamp(scores[:, 0], min=0.0)
    m2 = torch.clamp(scores[:, 1], min=0.0)
    m12 = torch.maximum(torch.clamp(scores[:, 2], min=0.0), torch.maximum(m1, m2))
    lower = torch.clamp(m1 + m2 - m12, min=0.0)
    upper = torch.minimum(m1, m2)
    alpha = redundancy_head(_core_scores(scores))
    redundancy = lower + alpha * torch.clamp(upper - lower, min=0.0)
    return _compute_components_from_scores_with_redundancy(scores, redundancy)


def _compute_uncertainty_mobius_components_from_scores(scores: torch.Tensor, redundancy_head: nn.Module) -> torch.Tensor:
    scores = scores.float()
    m1 = torch.clamp(scores[:, 0], min=0.0)
    m2 = torch.clamp(scores[:, 1], min=0.0)
    m12 = torch.maximum(torch.clamp(scores[:, 2], min=0.0), torch.maximum(m1, m2))
    lower = torch.clamp(m1 + m2 - m12, min=0.0)
    upper = torch.minimum(m1, m2)
    alpha = redundancy_head(_uncertainty_mobius_features(scores))
    redundancy = lower + alpha * torch.clamp(upper - lower, min=0.0)
    return _compute_components_from_scores_with_redundancy(scores, redundancy)


def _compute_components_from_scores_with_redundancy(scores: torch.Tensor, redundancy: torch.Tensor) -> torch.Tensor:
    m1 = torch.clamp(scores[:, 0], min=0.0)
    m2 = torch.clamp(scores[:, 1], min=0.0)
    m12 = torch.maximum(torch.clamp(scores[:, 2], min=0.0), torch.maximum(m1, m2))
    lower = torch.clamp(m1 + m2 - m12, min=0.0)
    upper = torch.minimum(m1, m2)
    redundancy = torch.minimum(torch.maximum(redundancy, lower), upper)
    unique_1 = torch.clamp(m1 - redundancy, min=0.0)
    unique_2 = torch.clamp(m2 - redundancy, min=0.0)
    synergy = torch.clamp(m12 - m1 - m2 + redundancy, min=0.0)
    return torch.stack([redundancy, unique_1, unique_2, synergy], dim=1)


def _semantic_loss(pointwise_components: torch.Tensor, predictive_scores: torch.Tensor) -> torch.Tensor:
    redundancy = pointwise_components[:, 0]
    m1 = predictive_scores[:, 0]
    m2 = predictive_scores[:, 1]
    return (F.relu(redundancy - m1).pow(2) + F.relu(redundancy - m2).pow(2)).mean()


def _nonnegative_loss(pointwise_components: torch.Tensor) -> torch.Tensor:
    return F.relu(-pointwise_components).pow(2).mean()


def _mobius_prior_loss(predictive_scores: torch.Tensor, redundancy_head: nn.Module) -> torch.Tensor:
    scores = predictive_scores.float()
    m1 = torch.clamp(scores[:, 0], min=0.0)
    m2 = torch.clamp(scores[:, 1], min=0.0)
    m12 = torch.maximum(torch.clamp(scores[:, 2], min=0.0), torch.maximum(m1, m2))
    additivity_gap = m12 - m1 - m2
    alpha = redundancy_head(_core_scores(scores))
    scale = additivity_gap.detach().abs().mean().clamp_min(1e-3)
    alpha_prior = torch.sigmoid(-additivity_gap / scale)
    return F.mse_loss(alpha, alpha_prior)


def _fit_uncertainty_mobius_head(
    *,
    redundancy_head: nn.Module,
    predictive_scores: torch.Tensor,
    calibration_records: list[CalibrationRecord],
    config: SurrogatePIDConfig,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(redundancy_head.parameters(), lr=config.head_lr)
    predictive_scores = predictive_scores.to(device)
    record_indices = [record.selected_indices.to(device) for record in calibration_records]
    record_targets = torch.stack([record.target_components for record in calibration_records], dim=0).to(device)

    for epoch in range(config.head_epochs):
        optimizer.zero_grad()
        pointwise = _compute_uncertainty_mobius_components_from_scores(predictive_scores, redundancy_head)
        calibration_predictions = torch.stack(
            [pointwise[indices].mean(dim=0) for indices in record_indices],
            dim=0,
        )
        calibration_loss = F.mse_loss(calibration_predictions, record_targets)
        uncertainty_prior_loss = _uncertainty_mobius_prior_loss(predictive_scores, redundancy_head)
        loss = calibration_loss + config.lambda_mobius_prior * uncertainty_prior_loss
        loss.backward()
        optimizer.step()
        if config.verbose and (epoch + 1) % 50 == 0:
            print(
                "[surrogate/uncertainty_mobius] "
                f"epoch={epoch + 1} calib={calibration_loss.item():.4f} "
                f"prior={uncertainty_prior_loss.item():.4f}"
            )
    redundancy_head.eval()


def _uncertainty_mobius_prior_loss(predictive_scores: torch.Tensor, redundancy_head: nn.Module) -> torch.Tensor:
    scores = predictive_scores.float()
    additivity_gap = scores[:, 2] - scores[:, 0] - scores[:, 1]
    h1 = scores[:, 3]
    h2 = scores[:, 4]
    h12 = scores[:, 5]
    uncertainty_gap = h12 - 0.5 * (h1 + h2)
    scale_add = additivity_gap.detach().abs().mean().clamp_min(1e-3)
    scale_unc = uncertainty_gap.detach().abs().mean().clamp_min(1e-3)
    alpha_prior = torch.sigmoid((-additivity_gap / scale_add) + (uncertainty_gap / scale_unc))
    alpha = redundancy_head(_uncertainty_mobius_features(scores))
    return F.mse_loss(alpha, alpha_prior)


def _core_scores(scores: torch.Tensor) -> torch.Tensor:
    return scores[:, :3].float()


def _uncertainty_mobius_features(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    core = _core_scores(scores)
    m1 = core[:, 0:1]
    m2 = core[:, 1:2]
    m12 = core[:, 2:3]
    h1 = scores[:, 3:4]
    h2 = scores[:, 4:5]
    h12 = scores[:, 5:6]
    additivity_gap = m12 - m1 - m2
    interaction_gain = m12 - torch.maximum(m1, m2)
    uncertainty_gap = h12 - 0.5 * (h1 + h2)
    entropy_diff = h1 - h2
    return torch.cat(
        [
            core,
            h1,
            h2,
            h12,
            additivity_gap,
            interaction_gain,
            uncertainty_gap,
            entropy_diff,
            torch.minimum(m1, m2),
        ],
        dim=1,
    )


def _fit_kernel_pid_head(
    *,
    predictive_scores: torch.Tensor,
    calibration_records: list[CalibrationRecord],
    config: SurrogatePIDConfig,
) -> KernelPIDHead:
    subset_inputs = []
    subset_targets = []
    full_components = None
    for record in calibration_records:
        subset_inputs.append(_summarize_scores(predictive_scores[record.selected_indices]))
        subset_targets.append(record.target_components)
        if torch.all(record.subset_weights > 0):
            full_components = record.target_components.float()

    train_inputs = torch.stack(subset_inputs, dim=0).float().cpu()
    targets = torch.stack(subset_targets, dim=0).float().cpu()
    if full_components is None:
        full_components = targets.mean(dim=0)
    if config.kernel_head_center_targets:
        targets = targets - full_components

    gamma = _resolve_kernel_gamma(train_inputs, config.kernel_head_gamma)
    kernel = _rbf_kernel_torch(train_inputs, train_inputs, gamma)
    regularized = kernel + config.kernel_head_ridge_lambda * torch.eye(kernel.shape[0], dtype=kernel.dtype)
    alpha = torch.linalg.solve(regularized, targets)
    return KernelPIDHead(
        train_inputs=train_inputs,
        alpha=alpha,
        gamma=gamma,
        full_components=full_components.float().cpu(),
        center_targets=config.kernel_head_center_targets,
    )


def _fit_kernel_pid_residual_head(
    *,
    predictive_scores: torch.Tensor,
    baseline_pointwise: torch.Tensor,
    calibration_records: list[CalibrationRecord],
    config: SurrogatePIDConfig,
) -> KernelPIDHead:
    subset_inputs = []
    subset_targets = []
    zero_components = torch.zeros(4, dtype=torch.float32)
    for record in calibration_records:
        indices = record.selected_indices.long()
        baseline_components = baseline_pointwise[indices].mean(dim=0)
        summary_features = _summarize_scores(predictive_scores[indices])
        subset_inputs.append(_kernel_residual_features(summary_features, baseline_components))
        subset_targets.append(record.target_components.float() - baseline_components.float())

    train_inputs = torch.stack(subset_inputs, dim=0).float().cpu()
    targets = torch.stack(subset_targets, dim=0).float().cpu()
    gamma = _resolve_kernel_gamma(train_inputs, config.kernel_head_gamma)
    kernel = _rbf_kernel_torch(train_inputs, train_inputs, gamma)
    regularized = kernel + config.kernel_head_ridge_lambda * torch.eye(kernel.shape[0], dtype=kernel.dtype)
    alpha = torch.linalg.solve(regularized, targets)
    return KernelPIDHead(
        train_inputs=train_inputs,
        alpha=alpha,
        gamma=gamma,
        full_components=zero_components,
        center_targets=False,
    )


def _build_block_component_sums(pointwise_components: torch.Tensor, block_indices: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([pointwise_components[indices].sum(dim=0) for indices in block_indices], dim=0).float()


def _build_block_feature_sums(predictive_scores: torch.Tensor, block_indices: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([_score_feature_map(predictive_scores[indices]).sum(dim=0) for indices in block_indices], dim=0).float()


def _summarize_scores(scores: torch.Tensor) -> torch.Tensor:
    return _score_feature_map(scores).mean(dim=0)


def _score_feature_map(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    m1 = scores[:, 0:1]
    m2 = scores[:, 1:2]
    m12 = scores[:, 2:3]
    min12 = torch.minimum(m1, m2)
    max12 = torch.maximum(m1, m2)
    interaction_gap = m12 - max12
    additivity_gap = m12 - m1 - m2
    prod12 = m1 * m2
    return torch.cat(
        [
            m1,
            m2,
            m12,
            m1.pow(2),
            m2.pow(2),
            m12.pow(2),
            min12,
            max12,
            interaction_gap,
            additivity_gap,
            prod12,
        ],
        dim=1,
    )


def _kernel_residual_features(summary_features: torch.Tensor, baseline_components: torch.Tensor) -> torch.Tensor:
    return_vector = summary_features.ndim == 1 and baseline_components.ndim == 1
    if summary_features.ndim == 1:
        summary_features = summary_features.unsqueeze(0)
    if baseline_components.ndim == 1:
        baseline_components = baseline_components.unsqueeze(0)
    redundancy = baseline_components[:, 0:1]
    unique_1 = baseline_components[:, 1:2]
    unique_2 = baseline_components[:, 2:3]
    synergy = baseline_components[:, 3:4]
    features = torch.cat(
        [
            summary_features.float(),
            baseline_components.float(),
            redundancy + synergy,
            unique_1 + unique_2,
            synergy - redundancy,
        ],
        dim=1,
    )
    if return_vector:
        return features.squeeze(0)
    return features


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


def _project_pid_components(components: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(components.float(), min=0.0)
    return values


def _reference_components(
    *,
    subset_dataset: Dataset,
    num_labels: int,
    config: SurrogatePIDConfig,
    device: torch.device,
) -> torch.Tensor:
    x1, x2, y = _dataset_to_tensors(subset_dataset)
    if config.reference_estimator != "lsmi":
        raise ValueError(f"unsupported reference_estimator: {config.reference_estimator}")
    result = estimate_lsmi(
        x1,
        x2,
        y,
        num_labels=num_labels,
        train_dataset=subset_dataset,
        test_dataset=subset_dataset,
        config=LSMIConfig(
            hidden_dim=config.predictor_hidden_dim,
            batch_size=min(config.batch_size, max(len(subset_dataset), 1)),
            discriminator_epochs=config.reference_lsmi_discriminator_epochs,
            entropy_epochs=config.reference_lsmi_entropy_epochs,
            num_workers=0,
            shuffle=True,
            verbose=False,
            seed=config.seed,
        ),
        device=device,
    )
    return result.mean_components.detach().cpu()


def _build_query_engine(pointwise_components: torch.Tensor, block_indices: list[torch.Tensor]) -> SurrogateQueryEngine:
    block_component_sums = []
    block_sizes = []
    for indices in block_indices:
        block_component_sums.append(pointwise_components[indices].sum(dim=0))
        block_sizes.append(float(len(indices)))
    return SurrogateQueryEngine(
        block_component_sums=torch.stack(block_component_sums, dim=0),
        block_sizes=torch.tensor(block_sizes, dtype=torch.float32),
    )


def _generate_subset_weights(num_blocks: int, num_subsets: int, seed: int | None) -> list[torch.Tensor]:
    generator = random.Random(seed)
    candidates: list[tuple[int, ...]] = [tuple(1 for _ in range(num_blocks))]
    for block in range(num_blocks):
        single = tuple(1 if idx == block else 0 for idx in range(num_blocks))
        candidates.append(single)
        leave_one_out = tuple(0 if idx == block else 1 for idx in range(num_blocks))
        candidates.append(leave_one_out)
    while len(candidates) < num_subsets:
        size = generator.randint(max(1, num_blocks // 2), num_blocks)
        selected = set(generator.sample(range(num_blocks), k=size))
        candidate = tuple(1 if idx in selected else 0 for idx in range(num_blocks))
        if sum(candidate) in {0, num_blocks}:
            continue
        candidates.append(candidate)
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(torch.tensor(candidate, dtype=torch.float32))
        if len(unique) >= num_subsets:
            break
    return unique


def _partition_indices(num_samples: int, num_blocks: int, seed: int | None) -> list[torch.Tensor]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    chunks = torch.chunk(permutation, min(num_blocks, num_samples))
    return [chunk for chunk in chunks if len(chunk) > 0]


def _indices_from_subset_weights(block_indices: list[torch.Tensor], subset_weights: torch.Tensor) -> torch.Tensor:
    selected = [block for block, weight in zip(block_indices, subset_weights.tolist()) if weight > 0]
    if not selected:
        return block_indices[0]
    return torch.cat(selected, dim=0)


def _subset_dataset(dataset: Dataset, indices: torch.Tensor) -> MultimodalDataset:
    x1, x2, y = _dataset_to_tensors(dataset)
    return MultimodalDataset([x1[indices], x2[indices]], y[indices])


def _dataset_to_tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(dataset, MultimodalDataset):
        x1 = _ensure_feature_batch(dataset.modalities[0]).float()
        x2 = _ensure_feature_batch(dataset.modalities[1]).float()
        y = _prepare_labels(dataset.labels)
        return x1, x2, y

    x1_list = []
    x2_list = []
    y_list = []
    for item in dataset:
        x1_list.append(torch.as_tensor(item[0]))
        x2_list.append(torch.as_tensor(item[1]))
        y_list.append(torch.as_tensor(item[2]))
    x1 = _ensure_feature_batch(torch.stack(x1_list)).float()
    x2 = _ensure_feature_batch(torch.stack(x2_list)).float()
    y = _prepare_labels(torch.stack(y_list))
    return x1, x2, y


def _dataset_batch_to_device(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = _ensure_feature_batch(batch[0]).float().to(device)
    x2 = _ensure_feature_batch(batch[1]).float().to(device)
    y = _prepare_labels(batch[2]).to(device)
    return x1, x2, y


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


__all__ = [
    "CalibrationRecord",
    "KernelHeadQueryEngine",
    "KernelPIDHead",
    "RedundancyHead",
    "SurrogatePIDConfig",
    "SurrogatePIDResult",
    "SurrogateQueryEngine",
    "estimate_surrogate_pid",
]
