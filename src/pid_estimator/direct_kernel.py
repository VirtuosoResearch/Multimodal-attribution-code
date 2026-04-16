from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import MultimodalDataset
from .lsmi import ClassificationNetwork


@dataclass(frozen=True)
class DirectKernelPIDConfig:
    predictor_hidden_dim: int = 64
    predictor_epochs: int = 10
    predictor_lr: float = 1e-3
    batch_size: int = 64
    num_workers: int = 0
    shuffle: bool = True
    kernel: str = "rbf"
    kernel_gamma: float | None = None
    ridge_lambda: float = 1e-2
    mobius_max_order: int = 2
    mobius_order_decay: float = 0.5
    hybrid_rbf_weight: float = 0.5
    calibration_blocks: int = 8
    refinement_epochs: int = 100
    refinement_lr: float = 5e-2
    lambda_anchor: float = 0.0
    lambda_info: float = 0.0
    lambda_redundancy: float = 2.5
    lambda_interval: float = 0.0
    lambda_rkhs: float = 0.0
    verbose: bool = False
    seed: int | None = None


@dataclass
class DirectKernelQueryEngine:
    block_component_sums: torch.Tensor
    block_sizes: torch.Tensor

    def query(self, subset_weights: torch.Tensor) -> torch.Tensor:
        weights = subset_weights.float().view(-1, 1)
        weighted_sum = torch.sum(self.block_component_sums * weights, dim=0)
        denominator = torch.sum(self.block_sizes * subset_weights.float()).clamp_min(1.0)
        return weighted_sum / denominator


@dataclass
class DirectKernelPIDResult:
    pointwise_components: torch.Tensor
    mean_components: torch.Tensor
    pointwise_predictive_scores: torch.Tensor
    alpha_raw: torch.Tensor
    alpha_calibrated: torch.Tensor
    dual_weights: torch.Tensor
    class_bias: torch.Tensor
    kernel: str
    kernel_gamma: float | None
    label_prior: torch.Tensor
    predictor_1: nn.Module
    predictor_2: nn.Module
    predictor_12: nn.Module
    block_indices: list[torch.Tensor]
    query_engine: DirectKernelQueryEngine

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


def estimate_direct_kernel_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    config: DirectKernelPIDConfig | None = None,
    device: str | torch.device | None = None,
) -> DirectKernelPIDResult:
    config = config or DirectKernelPIDConfig()
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
    test_scores, test_labels = _collect_predictive_scores(
        dataset=test_dataset,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        label_prior=label_prior,
        batch_size=config.batch_size,
        device=device,
    )

    m1, m2, m12, lower, width = _compute_pid_bounds(test_scores)
    continuous_features = _continuous_feature_map(test_scores)
    binary_features = _binary_feature_map(test_scores)
    alpha_prior = _build_alpha_prior(m1=m1, m2=m2, m12=m12)

    resolved_kernel, resolved_gamma = _resolve_kernel_family(
        continuous_features=continuous_features,
        binary_features=binary_features,
        alpha_prior=alpha_prior,
        config=config,
    )
    redundancy_prior = lower + alpha_prior * width
    prior_components = _components_from_redundancy(test_scores, redundancy_prior)
    kernel_matrix = _compute_kernel(
        continuous_lhs=continuous_features,
        continuous_rhs=continuous_features,
        binary_lhs=binary_features,
        binary_rhs=binary_features,
        kernel=resolved_kernel,
        kernel_gamma=resolved_gamma,
        config=config,
    )
    alpha_raw = alpha_prior
    pointwise_components, dual_weights, class_bias = _refine_joint_components(
        kernel_matrix=kernel_matrix,
        prior_components=prior_components,
        labels=test_labels,
        m1=m1,
        m2=m2,
        m12=m12,
        lower=lower,
        upper=lower + width,
        num_labels=num_labels,
        config=config,
    )
    alpha_calibrated = torch.where(
        width > 1e-8,
        torch.clamp((pointwise_components[:, 0] - lower) / width, min=0.0, max=1.0),
        torch.zeros_like(width),
    )
    pointwise_components = pointwise_components.detach().cpu()
    mean_components = pointwise_components.mean(dim=0)

    block_indices = _partition_indices(len(test_dataset), config.calibration_blocks, config.seed)
    query_engine = DirectKernelQueryEngine(
        block_component_sums=_build_block_component_sums(pointwise_components, block_indices),
        block_sizes=torch.tensor([float(len(indices)) for indices in block_indices], dtype=torch.float32),
    )
    return DirectKernelPIDResult(
        pointwise_components=pointwise_components,
        mean_components=mean_components,
        pointwise_predictive_scores=test_scores.detach().cpu(),
        alpha_raw=alpha_raw.detach().cpu(),
        alpha_calibrated=alpha_calibrated.detach().cpu(),
        dual_weights=dual_weights.detach().cpu(),
        class_bias=class_bias.detach().cpu(),
        kernel=resolved_kernel,
        kernel_gamma=resolved_gamma,
        label_prior=label_prior.detach().cpu(),
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        block_indices=block_indices,
        query_engine=query_engine,
    )


def _fit_predictors(
    *,
    train_dataset: Dataset,
    x1_dim: int,
    x2_dim: int,
    num_labels: int,
    config: DirectKernelPIDConfig,
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
            loss = criterion(logits_1, y_batch) + criterion(logits_2, y_batch) + criterion(logits_12, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x1_batch.shape[0]
            total_samples += x1_batch.shape[0]
        scheduler.step()
        if config.verbose and (epoch + 1) % 5 == 0:
            print(f"[direct-kernel/predictor] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")
    for model in models:
        model.eval()
    return models


def _collect_predictive_scores(
    *,
    dataset: Dataset,
    predictor_1: nn.Module,
    predictor_2: nn.Module,
    predictor_12: nn.Module,
    label_prior: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    collected_scores = []
    collected_labels = []
    with torch.no_grad():
        for batch in loader:
            x1_batch, x2_batch, y_batch = _dataset_batch_to_device(batch, device)
            rows = torch.arange(y_batch.shape[0], device=device)
            log_prior = torch.log(label_prior[y_batch] + 1e-8)

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
            collected_scores.append(torch.stack([m1, m2, m12, h1, h2, h12], dim=1))
            collected_labels.append(y_batch)
    return torch.cat(collected_scores, dim=0), torch.cat(collected_labels, dim=0)


def _compute_pid_bounds(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = scores.float()
    m1 = torch.clamp(scores[:, 0], min=0.0)
    m2 = torch.clamp(scores[:, 1], min=0.0)
    m12 = torch.maximum(torch.clamp(scores[:, 2], min=0.0), torch.maximum(m1, m2))
    lower = torch.clamp(m1 + m2 - m12, min=0.0)
    upper = torch.minimum(m1, m2)
    width = torch.clamp(upper - lower, min=0.0)
    return m1, m2, m12, lower, width


def _continuous_feature_map(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    return scores[:, :6]


def _binary_feature_map(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    m1 = scores[:, 0]
    m2 = scores[:, 1]
    m12 = scores[:, 2]
    h1 = scores[:, 3]
    h2 = scores[:, 4]
    h12 = scores[:, 5]
    features = torch.stack(
        [
            m1 > 0,
            m2 > 0,
            m12 > 0,
            h1 > h2,
            h1 > h12,
            h2 > h12,
            m1 >= m2,
            m1 >= m12,
            m2 >= m12,
        ],
        dim=1,
    )
    return features.float()


def _build_alpha_prior(*, m1: torch.Tensor, m2: torch.Tensor, m12: torch.Tensor) -> torch.Tensor:
    min_info = torch.minimum(m1, m2)
    agreement_gap = torch.abs(m1 - m2)
    interaction_gain = m12 - torch.maximum(m1, m2)
    scale_agreement = agreement_gap.detach().mean().clamp_min(1e-3)
    scale_interaction = interaction_gain.detach().abs().mean().clamp_min(1e-3)
    prior_logit = (min_info - agreement_gap) / scale_agreement - interaction_gain / scale_interaction
    return torch.sigmoid(prior_logit)


def _resolve_kernel_family(
    *,
    continuous_features: torch.Tensor,
    binary_features: torch.Tensor,
    alpha_prior: torch.Tensor,
    config: DirectKernelPIDConfig,
) -> tuple[str, float | None]:
    configured = config.kernel.strip().lower()
    if configured != "auto":
        gamma = _resolve_kernel_gamma(continuous_features, configured_gamma=config.kernel_gamma, kernel=config.kernel)
        return configured, gamma

    best_kernel = "rbf"
    best_gamma: float | None = None
    best_error = float("inf")
    for kernel in ("rbf", "mobius", "mobius_rbf"):
        gamma = _resolve_kernel_gamma(continuous_features, configured_gamma=config.kernel_gamma, kernel=kernel)
        error = _leave_one_out_prior_error(
            continuous_features=continuous_features,
            binary_features=binary_features,
            alpha_prior=alpha_prior,
            kernel=kernel,
            kernel_gamma=gamma,
            config=config,
        )
        if error < best_error:
            best_error = error
            best_kernel = kernel
            best_gamma = gamma
    return best_kernel, best_gamma


def _leave_one_out_prior_error(
    *,
    continuous_features: torch.Tensor,
    binary_features: torch.Tensor,
    alpha_prior: torch.Tensor,
    kernel: str,
    kernel_gamma: float | None,
    config: DirectKernelPIDConfig,
) -> float:
    num_samples = continuous_features.shape[0]
    if num_samples <= 2:
        return 0.0

    sq_errors = []
    for holdout in range(num_samples):
        mask = torch.ones(num_samples, dtype=torch.bool)
        mask[holdout] = False
        fit_kernel = _compute_kernel(
            continuous_lhs=continuous_features[mask],
            continuous_rhs=continuous_features[mask],
            binary_lhs=binary_features[mask],
            binary_rhs=binary_features[mask],
            kernel=kernel,
            kernel_gamma=kernel_gamma,
            config=config,
        )
        alpha = _fit_kernel_regressor(
            kernel_matrix=fit_kernel,
            targets=alpha_prior[mask],
            config=config,
        )
        kernel_row = _compute_kernel(
            continuous_lhs=continuous_features[holdout : holdout + 1],
            continuous_rhs=continuous_features[mask],
            binary_lhs=binary_features[holdout : holdout + 1],
            binary_rhs=binary_features[mask],
            kernel=kernel,
            kernel_gamma=kernel_gamma,
            config=config,
        )
        prediction = (kernel_row @ alpha).squeeze(0)
        sq_errors.append((prediction - alpha_prior[holdout]).pow(2))
    return float(torch.stack(sq_errors).mean().item())


def _fit_kernel_alpha(
    *,
    kernel_matrix: torch.Tensor,
    continuous_features: torch.Tensor,
    binary_features: torch.Tensor,
    alpha_prior: torch.Tensor,
    kernel: str,
    kernel_gamma: float | None,
    config: DirectKernelPIDConfig,
) -> torch.Tensor:
    dual = _fit_kernel_regressor(
        kernel_matrix=kernel_matrix,
        targets=alpha_prior,
        config=config,
    )
    return torch.clamp((kernel_matrix @ dual).float(), min=0.0, max=1.0)


def _refine_joint_components(
    *,
    kernel_matrix: torch.Tensor,
    prior_components: torch.Tensor,
    labels: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    m12: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_labels: int,
    config: DirectKernelPIDConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = prior_components.sum(dim=1, keepdim=True).clamp_min(1e-6)
    prior_weights = (prior_components / total).clamp_min(1e-6)
    prior_logits = torch.log(prior_weights)
    # Use the heuristic prior only as an initialization for the kernel logits.
    init_dual = _fit_kernel_regressor(
        kernel_matrix=kernel_matrix,
        targets=prior_logits,
        config=config,
    )
    dual_weights = nn.Parameter(init_dual.clone())
    class_bias = nn.Parameter(torch.zeros(num_labels, 4, dtype=kernel_matrix.dtype, device=kernel_matrix.device))
    optimizer = torch.optim.Adam([dual_weights, class_bias], lr=config.refinement_lr)

    target_i1, target_i2, target_i12, target_r, class_mask = _compute_joint_class_targets(
        m1=m1,
        m2=m2,
        m12=m12,
        labels=labels,
        num_labels=num_labels,
    )

    for epoch in range(config.refinement_epochs):
        optimizer.zero_grad()
        logits = kernel_matrix @ dual_weights + class_bias[labels]
        weights = torch.softmax(logits, dim=1)
        components = m12.unsqueeze(1) * weights

        info_1 = components[:, 0] + components[:, 1]
        info_2 = components[:, 0] + components[:, 2]
        info_12 = components.sum(dim=1)
        redundancy = components[:, 0]

        class_components = _classwise_means_matrix(components, labels, num_labels)
        class_i1 = _classwise_means_vector(info_1, labels, num_labels)
        class_i2 = _classwise_means_vector(info_2, labels, num_labels)
        class_i12 = _classwise_means_vector(info_12, labels, num_labels)

        point_loss = (
            F.mse_loss(info_1, m1)
            + F.mse_loss(info_2, m2)
        )
        info_loss = (
            F.mse_loss(class_i1[class_mask], target_i1[class_mask])
            + F.mse_loss(class_i2[class_mask], target_i2[class_mask])
            + F.mse_loss(class_i12[class_mask], target_i12[class_mask])
        ) if class_mask.any() else components.mean() * 0.0
        redundancy_loss = F.mse_loss(class_components[class_mask, 0], target_r[class_mask]) if class_mask.any() else components.mean() * 0.0
        interval_loss = (
            F.relu(lower - redundancy).pow(2)
            + F.relu(redundancy - upper).pow(2)
        ).mean()
        anchor_loss = F.mse_loss(components, prior_components)
        rkhs_loss = sum(
            torch.dot(dual_weights[:, index], kernel_matrix @ dual_weights[:, index])
            for index in range(4)
        ) / max(kernel_matrix.shape[0], 1)
        loss = (
            point_loss
            + config.lambda_info * info_loss
            + config.lambda_redundancy * redundancy_loss
            + config.lambda_interval * interval_loss
            + config.lambda_anchor * anchor_loss
            + config.lambda_rkhs * rkhs_loss
        )
        loss.backward()
        optimizer.step()
        if config.verbose and (epoch + 1) % 25 == 0:
            print(
                "[direct-kernel/joint] "
                f"epoch={epoch + 1} point={point_loss.item():.4f} info={info_loss.item():.4f} "
                f"red={redundancy_loss.item():.4f} interval={interval_loss.item():.4f} "
                f"anchor={anchor_loss.item():.4f} rkhs={rkhs_loss.item():.4f}"
            )

    with torch.no_grad():
        weights = torch.softmax(kernel_matrix @ dual_weights + class_bias[labels], dim=1)
        refined = m12.unsqueeze(1) * weights
    return refined, dual_weights.detach(), class_bias.detach()


def _fit_kernel_regressor(
    *,
    kernel_matrix: torch.Tensor,
    targets: torch.Tensor,
    config: DirectKernelPIDConfig,
) -> torch.Tensor:
    regularized = kernel_matrix + config.ridge_lambda * torch.eye(
        kernel_matrix.shape[0],
        dtype=kernel_matrix.dtype,
        device=kernel_matrix.device,
    )
    return torch.linalg.solve(regularized, targets.float())


def _compute_kernel(
    *,
    continuous_lhs: torch.Tensor,
    continuous_rhs: torch.Tensor,
    binary_lhs: torch.Tensor,
    binary_rhs: torch.Tensor,
    kernel: str,
    kernel_gamma: float | None,
    config: DirectKernelPIDConfig,
) -> torch.Tensor:
    kernel_name = kernel.strip().lower()
    if kernel_name == "rbf":
        if kernel_gamma is None:
            raise ValueError("kernel_gamma must be resolved for RBF direct-kernel estimators.")
        return _rbf_kernel_torch(continuous_lhs, continuous_rhs, kernel_gamma)
    if kernel_name == "mobius":
        return _mobius_kernel_torch(binary_lhs, binary_rhs, config.mobius_max_order, config.mobius_order_decay)
    if kernel_name in {"mobius_rbf", "hybrid"}:
        if kernel_gamma is None:
            raise ValueError("kernel_gamma must be resolved for hybrid direct-kernel estimators.")
        rbf = _rbf_kernel_torch(continuous_lhs, continuous_rhs, kernel_gamma)
        mobius = _mobius_kernel_torch(binary_lhs, binary_rhs, config.mobius_max_order, config.mobius_order_decay)
        mix = float(min(max(config.hybrid_rbf_weight, 0.0), 1.0))
        return mix * rbf + (1.0 - mix) * mobius
    raise ValueError(f"unsupported kernel={kernel!r}")


def _rbf_kernel_torch(lhs: torch.Tensor, rhs: torch.Tensor, gamma: float) -> torch.Tensor:
    lhs = lhs.float()
    rhs = rhs.float()
    lhs_sq = (lhs * lhs).sum(dim=1, keepdim=True)
    rhs_sq = (rhs * rhs).sum(dim=1, keepdim=True).T
    sq_dist = lhs_sq + rhs_sq - 2.0 * lhs @ rhs.T
    return torch.exp(-gamma * torch.clamp(sq_dist, min=0.0))


def _mobius_kernel_torch(lhs: torch.Tensor, rhs: torch.Tensor, max_order: int, order_decay: float) -> torch.Tensor:
    lhs_binary = (lhs > 0).float()
    rhs_binary = (rhs > 0).float()
    overlap = lhs_binary @ rhs_binary.T
    num_features = lhs_binary.shape[1]
    capped_order = max(1, min(int(max_order), num_features))

    kernel = torch.zeros_like(overlap)
    binomial = torch.ones_like(overlap)
    for order in range(1, capped_order + 1):
        binomial = binomial * torch.clamp(overlap - (order - 1), min=0.0) / float(order)
        scale = float(order_decay ** (order - 1))
        normalizer = max(math.comb(num_features, order), 1)
        kernel = kernel + (scale / normalizer) * binomial
    return kernel


def _resolve_kernel_gamma(features: torch.Tensor, *, configured_gamma: float | None, kernel: str) -> float | None:
    if kernel.strip().lower() == "mobius":
        return None
    if configured_gamma is not None:
        return float(configured_gamma)
    if features.shape[0] <= 1:
        return 1.0
    diffs = features[:, None, :] - features[None, :, :]
    sq_dists = (diffs * diffs).sum(dim=-1)
    positives = sq_dists[sq_dists > 0]
    if positives.numel() == 0:
        return 1.0
    return float(1.0 / positives.median().item())


def _compute_joint_class_targets(
    *,
    m1: torch.Tensor,
    m2: torch.Tensor,
    m12: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_i1 = torch.zeros(num_labels, dtype=m1.dtype, device=m1.device)
    target_i2 = torch.zeros(num_labels, dtype=m1.dtype, device=m1.device)
    target_i12 = torch.zeros(num_labels, dtype=m1.dtype, device=m1.device)
    target_r = torch.zeros(num_labels, dtype=m1.dtype, device=m1.device)
    class_mask = torch.zeros(num_labels, dtype=torch.bool, device=m1.device)
    for class_index in range(num_labels):
        members = labels == class_index
        if not members.any():
            continue
        target_i1[class_index] = m1[members].mean()
        target_i2[class_index] = m2[members].mean()
        target_i12[class_index] = m12[members].mean()
        target_r[class_index] = torch.minimum(target_i1[class_index], target_i2[class_index])
        class_mask[class_index] = True
    return target_i1, target_i2, target_i12, target_r, class_mask


def _classwise_means_vector(values: torch.Tensor, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    means = torch.zeros(num_labels, dtype=values.dtype, device=values.device)
    for class_index in range(num_labels):
        members = labels == class_index
        if members.any():
            means[class_index] = values[members].mean()
    return means


def _classwise_means_matrix(values: torch.Tensor, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    means = torch.zeros(num_labels, values.shape[1], dtype=values.dtype, device=values.device)
    for class_index in range(num_labels):
        members = labels == class_index
        if members.any():
            means[class_index] = values[members].mean(dim=0)
    return means


def _softplus_inverse(values: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(values.clamp_min(1e-6)))


def _refine_kernel_alpha(
    *,
    kernel_matrix: torch.Tensor,
    alpha_init: torch.Tensor,
    lower: torch.Tensor,
    width: torch.Tensor,
    labels: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    num_labels: int,
    config: DirectKernelPIDConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_targets, class_mask = _compute_class_redundancy_targets(
        m1=m1,
        m2=m2,
        labels=labels,
        num_labels=num_labels,
    )
    prior_logit = torch.logit(alpha_init.clamp(1e-4, 1.0 - 1e-4))
    dual_weights = nn.Parameter(torch.zeros(kernel_matrix.shape[0], dtype=kernel_matrix.dtype, device=kernel_matrix.device))
    class_bias = nn.Parameter(torch.zeros(num_labels, dtype=kernel_matrix.dtype, device=kernel_matrix.device))
    optimizer = torch.optim.Adam([dual_weights, class_bias], lr=config.refinement_lr)

    for epoch in range(config.refinement_epochs):
        optimizer.zero_grad()
        correction = kernel_matrix @ dual_weights + class_bias[labels]
        alpha = torch.sigmoid(prior_logit + correction)
        redundancy = lower + alpha * width
        class_means = _classwise_means(redundancy, labels, num_labels)
        class_loss = F.mse_loss(class_means[class_mask], class_targets[class_mask]) if class_mask.any() else redundancy.mean() * 0.0
        anchor_loss = F.mse_loss(alpha, alpha_init)
        rkhs_loss = torch.dot(dual_weights, kernel_matrix @ dual_weights) / max(kernel_matrix.shape[0], 1)
        loss = (
            config.lambda_anchor * anchor_loss
            + config.lambda_class * class_loss
            + config.lambda_rkhs * rkhs_loss
        )
        loss.backward()
        optimizer.step()
        if config.verbose and (epoch + 1) % 25 == 0:
            print(
                "[direct-kernel/refine] "
                f"epoch={epoch + 1} anchor={anchor_loss.item():.4f} "
                f"class={class_loss.item():.4f} rkhs={rkhs_loss.item():.4f}"
            )

    with torch.no_grad():
        alpha = torch.sigmoid(prior_logit + kernel_matrix @ dual_weights + class_bias[labels])
    return alpha, dual_weights.detach(), class_bias.detach()


def _compute_class_redundancy_targets(
    *,
    m1: torch.Tensor,
    m2: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(num_labels, dtype=m1.dtype, device=m1.device)
    mask = torch.zeros(num_labels, dtype=torch.bool, device=m1.device)
    for class_index in range(num_labels):
        class_members = labels == class_index
        if not class_members.any():
            continue
        targets[class_index] = torch.minimum(m1[class_members].mean(), m2[class_members].mean())
        mask[class_index] = True
    return targets, mask


def _classwise_means(values: torch.Tensor, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    means = torch.zeros(num_labels, dtype=values.dtype, device=values.device)
    for class_index in range(num_labels):
        class_members = labels == class_index
        if class_members.any():
            means[class_index] = values[class_members].mean()
    return means


def _components_from_redundancy(scores: torch.Tensor, redundancy: torch.Tensor) -> torch.Tensor:
    m1, m2, m12, _lower, _width = _compute_pid_bounds(scores)
    unique_1 = torch.clamp(m1 - redundancy, min=0.0)
    unique_2 = torch.clamp(m2 - redundancy, min=0.0)
    synergy = torch.clamp(m12 - m1 - m2 + redundancy, min=0.0)
    return torch.stack([redundancy, unique_1, unique_2, synergy], dim=1)


def _estimate_label_prior(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_labels).float()
    counts /= counts.sum().clamp_min(1.0)
    return counts


def _partition_indices(num_samples: int, num_blocks: int, seed: int | None) -> list[torch.Tensor]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    chunks = torch.chunk(permutation, min(num_blocks, num_samples))
    return [chunk for chunk in chunks if len(chunk) > 0]


def _build_block_component_sums(pointwise_components: torch.Tensor, block_indices: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([pointwise_components[indices].sum(dim=0) for indices in block_indices], dim=0)


def _dataset_batch_to_device(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = _ensure_feature_batch(batch[0]).to(device)
    x2 = _ensure_feature_batch(batch[1]).to(device)
    labels = _prepare_labels(batch[2]).to(device)
    return x1, x2, labels


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
    "DirectKernelPIDConfig",
    "DirectKernelPIDResult",
    "DirectKernelQueryEngine",
    "estimate_direct_kernel_pid",
]
