from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from kernel_surrogate.real_data_cmis import median_gamma


def fisher_corrected_score_grams(
    unimodal_gradients: np.ndarray,
    *,
    beta: float,
    fisher_lambda: float,
    normalize_features: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct Fisher-corrected sample and modality Gram matrices.

    unimodal_gradients has shape (num_groups, num_modalities, num_params). The
    inverse Fisher power is evaluated in the low-rank span of the score matrix.
    """
    gradients = np.nan_to_num(np.asarray(unimodal_gradients, dtype=np.float64), nan=0.0)
    if gradients.ndim != 3:
        raise ValueError(f"expected gradients with shape (G, M, P), got {gradients.shape}")
    num_groups, num_modalities, num_params = gradients.shape
    rows = gradients.reshape(num_groups * num_modalities, num_params)
    if rows.size == 0:
        raise ValueError("empty gradient matrix")

    u, singular_values, _ = np.linalg.svd(rows, full_matrices=False)
    trace = float(np.sum(singular_values**2) / max(num_groups * num_modalities, 1))
    trace_per_dim = trace / max(num_params, 1)
    if not math.isfinite(trace_per_dim) or trace_per_dim <= eps:
        corrected = np.zeros((num_groups * num_modalities, num_groups * num_modalities), dtype=np.float64)
    else:
        fisher_eigs = (singular_values**2 / max(num_groups * num_modalities, 1)) / trace_per_dim
        weights = np.power(np.maximum(fisher_eigs + float(fisher_lambda), eps), -float(beta))
        low_rank_features = u * (singular_values * np.sqrt(weights))[None, :]
        corrected = low_rank_features @ low_rank_features.T
    corrected = 0.5 * (corrected + corrected.T)

    if normalize_features:
        diag = np.maximum(np.diag(corrected), eps)
        denom = np.sqrt(diag[:, None] * diag[None, :])
        corrected = corrected / np.maximum(denom, eps)
        corrected = 0.5 * (corrected + corrected.T)
        np.fill_diagonal(corrected, np.where(diag > eps, 1.0, 0.0))

    block = corrected.reshape(num_groups, num_modalities, num_groups, num_modalities)
    sample_gram = np.zeros((num_groups, num_groups), dtype=np.float64)
    for modality in range(num_modalities):
        sample_gram += block[:, modality, :, modality]
    sample_gram /= max(num_modalities, 1)
    sample_gram = project_psd_normalized(sample_gram, normalize_diag=True, eps=eps)

    modality_gram = np.zeros((num_modalities, num_modalities), dtype=np.float64)
    for left in range(num_modalities):
        for right in range(num_modalities):
            modality_gram[left, right] = float(np.mean(block[:, left, :, right].diagonal()))
    modality_gram = project_psd_normalized(modality_gram, normalize_diag=True, eps=eps)

    stats = {
        "num_groups": int(num_groups),
        "num_modalities": int(num_modalities),
        "num_params": int(num_params),
        "beta": float(beta),
        "fisher_lambda": float(fisher_lambda),
        "normalize_features": bool(normalize_features),
        "score_norm_min": float(np.min(np.linalg.norm(rows, axis=1))),
        "score_norm_median": float(np.median(np.linalg.norm(rows, axis=1))),
        "score_norm_max": float(np.max(np.linalg.norm(rows, axis=1))),
        "trace_per_dim": float(trace_per_dim),
        "sample_gram_min_eig": float(np.min(np.linalg.eigvalsh(sample_gram))),
        "modality_gram_min_eig": float(np.min(np.linalg.eigvalsh(modality_gram))),
    }
    return sample_gram, modality_gram, stats


def project_psd_normalized(metric: np.ndarray, *, normalize_diag: bool, eps: float) -> np.ndarray:
    metric = np.nan_to_num(np.asarray(metric, dtype=np.float64), nan=0.0)
    metric = 0.5 * (metric + metric.T)
    vals, vecs = np.linalg.eigh(metric)
    vals = np.maximum(vals, 0.0)
    projected = (vecs * vals[None, :]) @ vecs.T
    projected = 0.5 * (projected + projected.T)
    if normalize_diag:
        diag = np.maximum(np.diag(projected), eps)
        projected = projected / np.maximum(np.sqrt(diag[:, None] * diag[None, :]), eps)
        projected = 0.5 * (projected + projected.T)
        np.fill_diagonal(projected, np.where(diag > eps, 1.0, 0.0))
    return projected


def hierarchical_availability(
    mask_blocks: dict[str, np.ndarray],
    indices: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    sample = 1.0 - np.asarray(mask_blocks["sample_deleted"], dtype=np.float64)[indices]
    modality_deleted = np.asarray(mask_blocks["modality_deleted"], dtype=np.float64)[indices]
    modality = (1.0 - modality_deleted).reshape(indices.size, sample.shape[1], 2)
    modality = modality * sample[:, :, None]
    return sample, modality


def fisher_structured_distance_parts(
    lhs_sample: np.ndarray,
    lhs_modality: np.ndarray,
    rhs_sample: np.ndarray,
    rhs_modality: np.ndarray,
    *,
    sample_metric: np.ndarray,
    modality_metric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_dist = metric_sqdist(lhs_sample, rhs_sample, sample_metric)
    lhs_sq = structured_modality_self(lhs_modality, sample_metric, modality_metric)
    rhs_sq = structured_modality_self(rhs_modality, sample_metric, modality_metric)
    cross = structured_modality_cross(lhs_modality, rhs_modality, sample_metric, modality_metric)
    modality_dist = np.maximum(lhs_sq[:, None] + rhs_sq[None, :] - 2.0 * cross, 0.0)
    return sample_dist, np.nan_to_num(modality_dist, nan=0.0, posinf=np.finfo(np.float64).max / 4.0)


def structured_modality_self(
    values: np.ndarray,
    sample_metric: np.ndarray,
    modality_metric: np.ndarray,
) -> np.ndarray:
    num_modalities = values.shape[2]
    out = np.zeros(values.shape[0], dtype=np.float64)
    for left in range(num_modalities):
        for right in range(num_modalities):
            weighted = values[:, :, left] @ sample_metric
            out += float(modality_metric[left, right]) * np.sum(weighted * values[:, :, right], axis=1)
    return out


def structured_modality_cross(
    lhs: np.ndarray,
    rhs: np.ndarray,
    sample_metric: np.ndarray,
    modality_metric: np.ndarray,
) -> np.ndarray:
    num_modalities = lhs.shape[2]
    out = np.zeros((lhs.shape[0], rhs.shape[0]), dtype=np.float64)
    for left in range(num_modalities):
        lhs_weighted = lhs[:, :, left] @ sample_metric
        for right in range(num_modalities):
            out += float(modality_metric[left, right]) * (lhs_weighted @ rhs[:, :, right].T)
    return out


def metric_sqdist(lhs: np.ndarray, rhs: np.ndarray, metric: np.ndarray) -> np.ndarray:
    lhs_metric = lhs @ metric
    lhs_sq = np.sum(lhs_metric * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum((rhs @ metric) * rhs, axis=1, keepdims=True).T
    return np.nan_to_num(
        np.maximum(lhs_sq + rhs_sq - 2.0 * lhs_metric @ rhs.T, 0.0),
        nan=0.0,
        posinf=np.finfo(np.float64).max / 4.0,
    )


def fit_fisher_enhanced_rbf(
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    sample_metric: np.ndarray,
    modality_metric: np.ndarray,
    *,
    ridge_lambda: float,
    gamma_a_scale: float,
    gamma_e_scale: float,
    train_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    if train_indices is None:
        train_indices = np.arange(len(targets), dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    sample, modality = hierarchical_availability(mask_blocks, train_indices)
    sample_dist, modality_dist = fisher_structured_distance_parts(
        sample,
        modality,
        sample,
        modality,
        sample_metric=sample_metric,
        modality_metric=modality_metric,
    )
    base_gamma_a = median_gamma(sample_dist)
    base_gamma_e = median_gamma(modality_dist)
    gamma_a = float(base_gamma_a * gamma_a_scale)
    gamma_e = float(base_gamma_e * gamma_e_scale)
    kernel = np.exp(-(gamma_a * sample_dist + gamma_e * modality_dist))
    target_mean = float(np.mean(targets[train_indices]))
    alpha = np.linalg.solve(
        kernel + float(ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64),
        targets[train_indices] - target_mean,
    )
    train_predictions = target_mean + kernel @ alpha
    return {
        "method": "fisher_enhanced_rbf",
        "alpha": alpha,
        "target_mean": target_mean,
        "gamma_a": gamma_a,
        "gamma_e": gamma_e,
        "base_gamma_a": base_gamma_a,
        "base_gamma_e": base_gamma_e,
        "train_predictions": train_predictions,
        "train_indices": train_indices.copy(),
        "train_sample": sample.copy(),
        "train_modality": modality.copy(),
        "sample_metric": np.asarray(sample_metric, dtype=np.float64).copy(),
        "modality_metric": np.asarray(modality_metric, dtype=np.float64).copy(),
    }


def predict_fisher_enhanced_rbf(
    mask_blocks: dict[str, np.ndarray],
    fit: dict[str, Any],
    *,
    query_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    sample, modality = hierarchical_availability(mask_blocks, query_indices)
    sample_dist, modality_dist = fisher_structured_distance_parts(
        sample,
        modality,
        fit["train_sample"],
        fit["train_modality"],
        sample_metric=fit["sample_metric"],
        modality_metric=fit["modality_metric"],
    )
    kernel = np.exp(-(float(fit["gamma_a"]) * sample_dist + float(fit["gamma_e"]) * modality_dist))
    return float(fit["target_mean"]) + kernel @ fit["alpha"]


def pearsonr(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denom = float(np.sqrt(np.sum(pred**2) * np.sum(truth**2)))
    if denom <= 1e-12 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(pred * truth) / denom)
