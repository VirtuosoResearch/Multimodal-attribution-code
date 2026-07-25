from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import Bounds, LinearConstraint, minimize
from torch.utils.data import DataLoader, Dataset

from .data import MultimodalDataset
from .lsmi import ClassificationNetwork, LSMIConfig, estimate_lsmi


@dataclass(frozen=True)
class KernelizedCompletionPIDConfig:
    predictor_hidden_dim: int = 64
    predictor_epochs: int = 10
    predictor_lr: float = 1e-3
    batch_size: int = 64
    num_workers: int = 0
    shuffle: bool = True
    seed: int | None = None
    calibration_blocks: int = 8
    calibration_subsets: int = 12
    reference_estimator: str = "auto"
    reference_lsmi_discriminator_epochs: int = 5
    reference_lsmi_entropy_epochs: int = 5
    kernel_gamma: float | None = None
    ridge_lambda: float = 1e-2
    completion_epochs: int = 250
    completion_lr: float = 5e-2
    lambda_completion_calibration: float = 1.0
    lambda_component_calibration: float = 1.0
    lambda_min_completion: float = 0.1
    lambda_rkhs: float = 1e-3
    min_alpha: float = 1e-4
    verbose: bool = False


@dataclass
class CompletionCalibrationRecord:
    subset_weights: torch.Tensor
    selected_indices: torch.Tensor
    target_components: torch.Tensor
    target_completion: torch.Tensor
    reference_estimator: str


@dataclass
class KernelizedCompletionQueryEngine:
    block_component_sums: torch.Tensor
    block_sizes: torch.Tensor

    def query(self, subset_weights: torch.Tensor) -> torch.Tensor:
        weights = subset_weights.float().view(-1, 1)
        weighted_sum = torch.sum(self.block_component_sums * weights, dim=0)
        denominator = torch.sum(self.block_sizes * subset_weights.float()).clamp_min(1.0)
        return weighted_sum / denominator


@dataclass
class KernelizedCompletionPIDResult:
    pointwise_components: torch.Tensor
    mean_components: torch.Tensor
    pointwise_predictive_scores: torch.Tensor
    pointwise_completion: torch.Tensor
    pointwise_alpha: torch.Tensor
    train_features: torch.Tensor
    dual_weights: torch.Tensor
    gamma: float
    calibration_records: list[CompletionCalibrationRecord]
    predictor_1: nn.Module
    predictor_2: nn.Module
    predictor_12: nn.Module
    label_prior: torch.Tensor
    block_indices: list[torch.Tensor]
    query_engine: KernelizedCompletionQueryEngine
    reference_estimator: str

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


def estimate_kernelized_completion_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int | None = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    config: KernelizedCompletionPIDConfig | None = None,
    device: str | torch.device | None = None,
) -> KernelizedCompletionPIDResult:
    config = config or KernelizedCompletionPIDConfig()
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
    test_scores = _collect_predictive_scores(test_dataset, predictor_1, predictor_2, predictor_12, label_prior, config.batch_size, device)

    train_features = _completion_feature_map(train_scores.detach().cpu())
    test_features = _completion_feature_map(test_scores.detach().cpu())
    resolved_reference = _resolve_reference_estimator(train_dataset, config.reference_estimator)
    calibration_records = _build_calibration_records(
        train_dataset=train_dataset,
        train_scores=train_scores.detach().cpu(),
        num_labels=num_labels,
        config=config,
        device=device,
        reference_estimator=resolved_reference,
    )

    dual_weights, gamma = _fit_completion_dual_weights(
        features=train_features,
        predictive_scores=train_scores.detach().cpu(),
        calibration_records=calibration_records,
        config=config,
    )

    train_kernel = _rbf_kernel_torch(train_features, train_features, gamma)
    test_kernel = _rbf_kernel_torch(test_features, train_features, gamma)
    train_completion, train_alpha = _predict_completion_from_dual_weights(
        predictive_scores=train_scores.detach().cpu(),
        kernel_matrix=train_kernel,
        dual_weights=dual_weights,
        min_alpha=config.min_alpha,
    )
    test_completion, test_alpha = _predict_completion_from_dual_weights(
        predictive_scores=test_scores.detach().cpu(),
        kernel_matrix=test_kernel,
        dual_weights=dual_weights,
        min_alpha=config.min_alpha,
    )

    _ = train_completion  # retained for debugging symmetry with the training path.
    pointwise_components = _components_from_completion(test_scores.detach().cpu(), test_completion)
    mean_components = pointwise_components.mean(dim=0)
    block_indices = _partition_indices(len(test_dataset), config.calibration_blocks, config.seed)
    query_engine = KernelizedCompletionQueryEngine(
        block_component_sums=_build_block_component_sums(pointwise_components, block_indices),
        block_sizes=torch.tensor([float(len(indices)) for indices in block_indices], dtype=torch.float32),
    )
    return KernelizedCompletionPIDResult(
        pointwise_components=pointwise_components,
        mean_components=mean_components,
        pointwise_predictive_scores=test_scores.detach().cpu(),
        pointwise_completion=test_completion,
        pointwise_alpha=test_alpha,
        train_features=train_features,
        dual_weights=dual_weights,
        gamma=gamma,
        calibration_records=calibration_records,
        predictor_1=predictor_1,
        predictor_2=predictor_2,
        predictor_12=predictor_12,
        label_prior=label_prior.detach().cpu(),
        block_indices=block_indices,
        query_engine=query_engine,
        reference_estimator=resolved_reference,
    )


def _fit_predictors(
    *,
    train_dataset: Dataset,
    x1_dim: int,
    x2_dim: int,
    num_labels: int,
    config: KernelizedCompletionPIDConfig,
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
            print(f"[kernelized-completion/predictor] epoch={epoch + 1} loss={total_loss / max(total_samples, 1):.4f}")
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
            v_a = torch.clamp(log_probs_1[rows, y_batch] - log_prior, min=0.0)
            v_b = torch.clamp(log_probs_2[rows, y_batch] - log_prior, min=0.0)
            v_ab = torch.clamp(log_probs_12[rows, y_batch] - log_prior, min=0.0)
            h_a = -(probs_1 * log_probs_1).sum(dim=1)
            h_b = -(probs_2 * log_probs_2).sum(dim=1)
            h_ab = -(probs_12 * log_probs_12).sum(dim=1)
            collected.append(torch.stack([v_a, v_b, v_ab, h_a, h_b, h_ab], dim=1))
    return torch.cat(collected, dim=0)


def _completion_feature_map(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    v_a = scores[:, 0:1]
    v_b = scores[:, 1:2]
    v_ab = scores[:, 2:3]
    h_a = scores[:, 3:4]
    h_b = scores[:, 4:5]
    h_ab = scores[:, 5:6]
    lower = torch.maximum(v_a, v_b)
    additive_upper = v_a + v_b
    upper = torch.minimum(torch.maximum(v_ab, lower), additive_upper)
    width = torch.clamp(upper - lower, min=0.0)
    interaction_gain = v_ab - lower
    additivity_gap = v_ab - additive_upper
    entropy_gap = h_ab - 0.5 * (h_a + h_b)
    confidence_diff = h_a - h_b
    return torch.cat(
        [
            v_a,
            v_b,
            v_ab,
            lower,
            upper,
            width,
            interaction_gain,
            additivity_gap,
            h_a,
            h_b,
            h_ab,
            entropy_gap,
            confidence_diff,
            torch.minimum(v_a, v_b),
            torch.maximum(v_a, v_b),
            v_a * v_b,
        ],
        dim=1,
    ).float()


def _predict_completion_from_dual_weights(
    *,
    predictive_scores: torch.Tensor,
    kernel_matrix: torch.Tensor,
    dual_weights: torch.Tensor,
    min_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = kernel_matrix @ dual_weights
    alpha = torch.sigmoid(raw)
    if min_alpha > 0.0:
        alpha = min_alpha + (1.0 - 2.0 * min_alpha) * alpha
    lower, upper = _completion_bounds(predictive_scores)
    completion = lower + alpha * torch.clamp(upper - lower, min=0.0)
    return completion.float(), alpha.float()


def _fit_completion_dual_weights(
    *,
    features: torch.Tensor,
    predictive_scores: torch.Tensor,
    calibration_records: list[CompletionCalibrationRecord],
    config: KernelizedCompletionPIDConfig,
) -> tuple[torch.Tensor, float]:
    gamma = _resolve_kernel_gamma(features, config.kernel_gamma)
    kernel_matrix = _rbf_kernel_torch(features, features, gamma)
    regularized = kernel_matrix + config.ridge_lambda * torch.eye(kernel_matrix.shape[0], dtype=kernel_matrix.dtype)
    # Start from the minimum-completion prior by matching zero logits in RKHS space.
    initial_dual = torch.linalg.solve(regularized, torch.zeros(features.shape[0], dtype=kernel_matrix.dtype))
    dual_weights = nn.Parameter(initial_dual)
    optimizer = torch.optim.Adam([dual_weights], lr=config.completion_lr)

    selected_indices = [record.selected_indices.long() for record in calibration_records]
    target_components = (
        torch.stack([record.target_components.float() for record in calibration_records], dim=0)
        if calibration_records
        else torch.zeros((0, 4), dtype=torch.float32)
    )
    target_completion = (
        torch.stack([record.target_completion.float() for record in calibration_records], dim=0).view(-1)
        if calibration_records
        else torch.zeros((0,), dtype=torch.float32)
    )

    for epoch in range(config.completion_epochs):
        optimizer.zero_grad()
        completion, _alpha = _predict_completion_from_dual_weights(
            predictive_scores=predictive_scores,
            kernel_matrix=kernel_matrix,
            dual_weights=dual_weights,
            min_alpha=config.min_alpha,
        )
        pointwise_components = _components_from_completion(predictive_scores, completion)

        if selected_indices:
            completion_predictions = torch.stack([completion[indices].mean() for indices in selected_indices], dim=0)
            component_predictions = torch.stack([pointwise_components[indices].mean(dim=0) for indices in selected_indices], dim=0)
            completion_loss = F.mse_loss(completion_predictions, target_completion)
            component_loss = F.mse_loss(component_predictions, target_components)
        else:
            completion_loss = completion.mean() * 0.0
            component_loss = pointwise_components.mean() * 0.0

        rkhs_loss = torch.dot(dual_weights, kernel_matrix @ dual_weights) / max(kernel_matrix.shape[0], 1)
        min_completion_loss = completion.mean()
        loss = (
            config.lambda_completion_calibration * completion_loss
            + config.lambda_component_calibration * component_loss
            + config.lambda_min_completion * min_completion_loss
            + config.lambda_rkhs * rkhs_loss
        )
        loss.backward()
        optimizer.step()
        if config.verbose and (epoch + 1) % 50 == 0:
            print(
                "[kernelized-completion/head] "
                f"epoch={epoch + 1} completion={completion_loss.item():.4f} "
                f"components={component_loss.item():.4f} "
                f"minn={min_completion_loss.item():.4f} rkhs={rkhs_loss.item():.4f}"
            )

    return dual_weights.detach().cpu(), gamma


def _completion_bounds(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = scores.float()
    v_a = scores[:, 0]
    v_b = scores[:, 1]
    v_ab = scores[:, 2]
    lower = torch.maximum(v_a, v_b)
    upper = torch.minimum(torch.maximum(v_ab, lower), v_a + v_b)
    upper = torch.maximum(upper, lower)
    return lower, upper


def _components_from_completion(scores: torch.Tensor, completion: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    v_a = scores[:, 0]
    v_b = scores[:, 1]
    v_ab = torch.maximum(scores[:, 2], torch.maximum(v_a, v_b))
    lower, upper = _completion_bounds(scores)
    completion = torch.minimum(torch.maximum(completion.float(), lower), upper)
    redundancy = torch.clamp(v_a + v_b - completion, min=0.0)
    unique_1 = torch.clamp(completion - v_b, min=0.0)
    unique_2 = torch.clamp(completion - v_a, min=0.0)
    synergy = torch.clamp(v_ab - completion, min=0.0)
    return torch.stack([redundancy, unique_1, unique_2, synergy], dim=1)


def _build_calibration_records(
    *,
    train_dataset: Dataset,
    train_scores: torch.Tensor,
    num_labels: int,
    config: KernelizedCompletionPIDConfig,
    device: torch.device,
    reference_estimator: str,
) -> list[CompletionCalibrationRecord]:
    if reference_estimator == "weak":
        return []
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
            reference_estimator=reference_estimator,
        )
        subset_completion = target_components[0] + target_components[1] + target_components[2]
        records.append(
            CompletionCalibrationRecord(
                subset_weights=subset_weights.detach().cpu(),
                selected_indices=selected_indices.detach().cpu(),
                target_components=target_components.detach().cpu(),
                target_completion=subset_completion.detach().cpu(),
                reference_estimator=reference_estimator,
            )
        )
    return records


def _reference_components(
    *,
    subset_dataset: Dataset,
    num_labels: int,
    config: KernelizedCompletionPIDConfig,
    device: torch.device,
    reference_estimator: str,
) -> torch.Tensor:
    x1, x2, y = _dataset_to_tensors(subset_dataset)
    if reference_estimator == "analytic":
        result = estimate_discrete_bertschinger_pid(x1, x2, y)
        return torch.tensor(
            [result["redundancy"], result["unique_1"], result["unique_2"], result["synergy"]],
            dtype=torch.float32,
        )
    if reference_estimator == "lsmi":
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
    raise ValueError(f"unsupported reference_estimator: {reference_estimator}")


def estimate_discrete_bertschinger_pid(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    x1_values = _tensor_rows_to_hashable(_ensure_feature_batch(x1))
    x2_values = _tensor_rows_to_hashable(_ensure_feature_batch(x2))
    y_values = _tensor_rows_to_hashable(_ensure_feature_batch(y))
    support = _build_discrete_support(x1_values, x2_values, y_values)
    p = support["joint"]
    p_at = p.sum(axis=1)
    p_bt = p.sum(axis=0)
    p_t = p.sum(axis=(0, 1))

    q0 = np.zeros_like(p)
    for a_idx in range(p.shape[0]):
        for b_idx in range(p.shape[1]):
            for t_idx in range(p.shape[2]):
                if p_t[t_idx] <= 0.0:
                    continue
                q0[a_idx, b_idx, t_idx] = p_at[a_idx, t_idx] * p_bt[b_idx, t_idx] / p_t[t_idx]
    q0 = q0 / q0.sum()

    num_states = q0.size
    constraint_rows = []
    constraint_lows = []
    constraint_highs = []

    for a_idx in range(p.shape[0]):
        for t_idx in range(p.shape[2]):
            row = np.zeros(num_states, dtype=np.float64)
            for b_idx in range(p.shape[1]):
                row[np.ravel_multi_index((a_idx, b_idx, t_idx), p.shape)] = 1.0
            constraint_rows.append(row)
            constraint_lows.append(p_at[a_idx, t_idx])
            constraint_highs.append(p_at[a_idx, t_idx])

    for b_idx in range(p.shape[1]):
        for t_idx in range(p.shape[2]):
            row = np.zeros(num_states, dtype=np.float64)
            for a_idx in range(p.shape[0]):
                row[np.ravel_multi_index((a_idx, b_idx, t_idx), p.shape)] = 1.0
            constraint_rows.append(row)
            constraint_lows.append(p_bt[b_idx, t_idx])
            constraint_highs.append(p_bt[b_idx, t_idx])

    linear_constraint = LinearConstraint(
        np.stack(constraint_rows, axis=0),
        lb=np.asarray(constraint_lows, dtype=np.float64),
        ub=np.asarray(constraint_highs, dtype=np.float64),
    )
    equality_rows = np.stack(constraint_rows, axis=0)
    equality_targets = np.asarray(constraint_lows, dtype=np.float64)
    bounds = Bounds(lb=np.zeros(num_states, dtype=np.float64), ub=np.ones(num_states, dtype=np.float64))

    def objective(flat_q: np.ndarray) -> tuple[float, np.ndarray]:
        q = flat_q.reshape(p.shape)
        q_ab = q.sum(axis=2, keepdims=True)
        safe_q = np.clip(q, 1e-12, None)
        safe_q_ab = np.clip(q_ab, 1e-12, None)
        safe_p_t = np.clip(p_t.reshape(1, 1, -1), 1e-12, None)
        info = np.sum(safe_q * (np.log(safe_q) - np.log(safe_q_ab) - np.log(safe_p_t)))
        gradient = np.log(safe_q) - np.log(safe_q_ab) - np.log(safe_p_t)
        return float(info / math.log(2.0)), gradient.reshape(-1) / math.log(2.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        optimum = minimize(
            fun=lambda q: objective(q)[0],
            x0=q0.reshape(-1),
            jac=lambda q: objective(q)[1],
            method="trust-constr",
            constraints=[linear_constraint],
            bounds=bounds,
            options={"verbose": 0, "maxiter": 500},
        )
    if not optimum.success:
        equality_constraints = [
            {
                "type": "eq",
                "fun": (lambda q, row=row, target=target: float(np.dot(row, q) - target)),
                "jac": (lambda q, row=row, target=target: row),
            }
            for row, target in zip(equality_rows, equality_targets)
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            optimum = minimize(
                fun=lambda q: objective(q)[0],
                x0=q0.reshape(-1),
                jac=lambda q: objective(q)[1],
                method="SLSQP",
                constraints=equality_constraints,
                bounds=[(0.0, 1.0)] * num_states,
                options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
            )
    if not optimum.success:
        raise RuntimeError(f"Bertschinger completion optimization failed: {optimum.message}")

    q_star = optimum.x.reshape(p.shape)
    n_star = _mutual_information_from_joint(q_star)
    v_a = _mutual_information_from_joint(p_at)
    v_b = _mutual_information_from_joint(p_bt)
    v_ab = _mutual_information_from_joint(p)
    redundancy = v_a + v_b - n_star
    unique_1 = n_star - v_b
    unique_2 = n_star - v_a
    synergy = v_ab - n_star
    return {
        "redundancy": float(max(redundancy, 0.0)),
        "unique_1": float(max(unique_1, 0.0)),
        "unique_2": float(max(unique_2, 0.0)),
        "synergy": float(max(synergy, 0.0)),
        "completion": float(n_star),
        "v_a": float(v_a),
        "v_b": float(v_b),
        "v_ab": float(v_ab),
    }


def _build_discrete_support(
    x1_values: list[Any],
    x2_values: list[Any],
    y_values: list[Any],
) -> dict[str, Any]:
    a_values = sorted(set(x1_values))
    b_values = sorted(set(x2_values))
    t_values = sorted(set(y_values))
    a_index = {value: idx for idx, value in enumerate(a_values)}
    b_index = {value: idx for idx, value in enumerate(b_values)}
    t_index = {value: idx for idx, value in enumerate(t_values)}
    joint = np.zeros((len(a_values), len(b_values), len(t_values)), dtype=np.float64)
    for a_value, b_value, t_value in zip(x1_values, x2_values, y_values):
        joint[a_index[a_value], b_index[b_value], t_index[t_value]] += 1.0
    joint /= max(len(y_values), 1)
    return {
        "joint": joint,
        "a_values": a_values,
        "b_values": b_values,
        "t_values": t_values,
    }


def _mutual_information_from_joint(joint: np.ndarray) -> float:
    if joint.ndim == 2:
        p_xt = joint
        p_x = p_xt.sum(axis=1, keepdims=True)
        p_t = p_xt.sum(axis=0, keepdims=True)
        safe_joint = np.clip(p_xt, 1e-12, None)
        safe_x = np.clip(p_x, 1e-12, None)
        safe_t = np.clip(p_t, 1e-12, None)
        return float(np.sum(safe_joint * (np.log2(safe_joint) - np.log2(safe_x) - np.log2(safe_t))))
    if joint.ndim == 3:
        p_abt = joint
        p_ab = p_abt.sum(axis=2, keepdims=True)
        p_t = p_abt.sum(axis=(0, 1), keepdims=True)
        safe_joint = np.clip(p_abt, 1e-12, None)
        safe_ab = np.clip(p_ab, 1e-12, None)
        safe_t = np.clip(p_t, 1e-12, None)
        return float(np.sum(safe_joint * (np.log2(safe_joint) - np.log2(safe_ab) - np.log2(safe_t))))
    raise ValueError(f"unsupported joint ndim: {joint.ndim}")


def _resolve_reference_estimator(dataset: Dataset, configured: str) -> str:
    if configured == "weak":
        return "weak"
    if configured == "analytic":
        return "analytic"
    if configured == "lsmi":
        return "lsmi"
    if configured != "auto":
        raise ValueError(f"unsupported reference_estimator: {configured}")
    x1, x2, y = _dataset_to_tensors(dataset)
    if _is_small_discrete_tensor(x1) and _is_small_discrete_tensor(x2) and _is_small_discrete_tensor(y):
        return "analytic"
    return "lsmi"


def _is_small_discrete_tensor(values: torch.Tensor, max_unique: int = 64) -> bool:
    values = torch.as_tensor(values).detach().cpu()
    if values.dtype.is_floating_point:
        rounded = values.round()
        if not torch.allclose(values, rounded, atol=1e-6):
            return False
        values = rounded.long()
    flattened = values.reshape(values.shape[0], -1)
    unique = torch.unique(flattened, dim=0)
    return unique.shape[0] <= max_unique


def _tensor_rows_to_hashable(values: torch.Tensor) -> list[Any]:
    values = torch.as_tensor(values).detach().cpu()
    flattened = values.reshape(values.shape[0], -1)
    rows = []
    for row in flattened:
        row_tuple = []
        for value in row.tolist():
            if isinstance(value, float):
                if abs(value - round(value)) < 1e-6:
                    row_tuple.append(int(round(value)))
                else:
                    row_tuple.append(float(value))
            else:
                row_tuple.append(int(value))
        rows.append(tuple(row_tuple) if len(row_tuple) > 1 else row_tuple[0])
    return rows


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


def _build_block_component_sums(pointwise_components: torch.Tensor, block_indices: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([pointwise_components[indices].sum(dim=0) for indices in block_indices], dim=0).float()


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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "CompletionCalibrationRecord",
    "KernelizedCompletionPIDConfig",
    "KernelizedCompletionPIDResult",
    "KernelizedCompletionQueryEngine",
    "estimate_discrete_bertschinger_pid",
    "estimate_kernelized_completion_pid",
]
