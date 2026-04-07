from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import auc, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from motivation.core import calibrated_mutual_knn, compute_knn_indices, write_json
from motivation.lsmi_estimator import estimate_samplewise_lsmi
BUCKETS = ("RED", "SYN", "UNQ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def _pairwise_average(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("Expected at least one pairwise value.")
    if len(values) == 1:
        return values[0]
    return np.mean(np.stack(values, axis=0), axis=0)


def _pairwise_average_scalar(values: list[float]) -> float:
    if not values:
        raise ValueError("Expected at least one pairwise value.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _cosine_rows(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.maximum(left_norm * right_norm, eps)
    return (np.sum(left * right, axis=1) / denom).astype(np.float32)


def _subspace_overlap(left: np.ndarray, right: np.ndarray, top_k: int) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Subspace overlap expects 2D matrices.")
    if left.shape[0] < 2 or right.shape[0] < 2:
        return float("nan")

    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    _, _, left_vh = np.linalg.svd(left_centered, full_matrices=False)
    _, _, right_vh = np.linalg.svd(right_centered, full_matrices=False)
    k_eff = min(top_k, left_vh.shape[0], right_vh.shape[0], left_vh.shape[1], right_vh.shape[1])
    if k_eff <= 0:
        return float("nan")
    basis_left = left_vh[:k_eff].T
    basis_right = right_vh[:k_eff].T
    singular = np.linalg.svd(basis_left.T @ basis_right, compute_uv=False)
    return float(np.clip(singular.mean(), 0.0, 1.0))


def _samplewise_rbf_cka(
    left: np.ndarray,
    right: np.ndarray,
    *,
    num_permutations: int,
    seed: int,
) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape[0] != right.shape[0]:
        raise ValueError("CKA expects aligned sample counts.")

    def _rbf_kernel(features: np.ndarray) -> np.ndarray:
        sq = np.sum(features**2, axis=1, keepdims=True)
        dist2 = np.maximum(sq + sq.T - 2.0 * (features @ features.T), 0.0)
        off_diag = dist2[np.triu_indices(dist2.shape[0], k=1)]
        sigma2 = float(np.median(off_diag[off_diag > 0])) if np.any(off_diag > 0) else 1.0
        sigma2 = max(sigma2, 1e-6)
        kernel = np.exp(-dist2 / (2.0 * sigma2)).astype(np.float32)
        return kernel

    def _center(kernel: np.ndarray) -> np.ndarray:
        row_mean = kernel.mean(axis=1, keepdims=True)
        col_mean = kernel.mean(axis=0, keepdims=True)
        grand_mean = kernel.mean()
        return (kernel - row_mean - col_mean + grand_mean).astype(np.float32)

    kernel_left = _center(_rbf_kernel(left))
    kernel_right = _center(_rbf_kernel(right))
    eps = 1e-8
    dataset_num = float(np.sum(kernel_left * kernel_right))
    dataset_den = float(np.linalg.norm(kernel_left) * np.linalg.norm(kernel_right))
    dataset_raw = dataset_num / max(dataset_den, eps)
    sample_raw = _cosine_rows(kernel_left, kernel_right, eps=eps)

    rng = np.random.default_rng(seed)
    null_scores = np.empty((num_permutations, left.shape[0]), dtype=np.float32)
    dataset_null = np.empty(num_permutations, dtype=np.float32)
    for index in range(num_permutations):
        perm = rng.permutation(left.shape[0])
        permuted_right = kernel_right[perm][:, perm]
        null_scores[index] = _cosine_rows(kernel_left, permuted_right, eps=eps)
        dataset_null[index] = float(np.sum(kernel_left * permuted_right)) / max(
            float(np.linalg.norm(kernel_left) * np.linalg.norm(permuted_right)),
            eps,
        )

    sample_mean = null_scores.mean(axis=0)
    sample_std = null_scores.std(axis=0, ddof=1 if num_permutations > 1 else 0)
    sample_std[sample_std < 1e-6] = 1e-6
    dataset_null_std = float(dataset_null.std(ddof=1 if num_permutations > 1 else 0))
    dataset_null_std = max(dataset_null_std, 1e-6)
    return {
        "sample_raw": sample_raw.astype(np.float32),
        "sample_z": ((sample_raw - sample_mean) / sample_std).astype(np.float32),
        "dataset_raw": float(dataset_raw),
        "dataset_z": float((dataset_raw - float(dataset_null.mean())) / dataset_null_std),
        "dataset_null_mean": float(dataset_null.mean()),
        "dataset_null_std": dataset_null_std,
    }


def _confidence_for_targets(task: str, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels)
    if task == "binary":
        targets = labels.astype(np.float32).reshape(-1)
        probs_1 = probs.reshape(-1)
        return np.where(targets > 0.5, probs_1, 1.0 - probs_1).astype(np.float32)

    labels = labels.astype(np.float32)
    denom = labels.sum(axis=1)
    denom = np.maximum(denom, 1.0)
    return ((probs * labels).sum(axis=1) / denom).astype(np.float32)


def _correctness_from_probs(task: str, probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels)
    if task == "binary":
        preds = (probs.reshape(-1) >= threshold).astype(np.float32)
        return (preds == labels.reshape(-1).astype(np.float32)).astype(np.int64)

    labels = labels.astype(np.float32)
    preds = (probs >= threshold).astype(np.float32)
    tp = (preds * labels).sum(axis=1)
    precision = tp / np.maximum(preds.sum(axis=1), 1.0)
    recall = tp / np.maximum(labels.sum(axis=1), 1.0)
    sample_f1 = np.where(
        precision + recall > 0,
        2.0 * precision * recall / np.maximum(precision + recall, 1e-8),
        0.0,
    )
    return (sample_f1 >= threshold).astype(np.int64)


def _sample_quality_from_probs(task: str, probs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    eps = 1e-6
    if task == "binary":
        probs_1 = np.clip(probs.reshape(-1), eps, 1.0 - eps)
        labels_1 = labels.reshape(-1)
        loss = -(labels_1 * np.log(probs_1) + (1.0 - labels_1) * np.log(1.0 - probs_1))
        confidence = np.where(labels_1 > 0.5, probs_1, 1.0 - probs_1)
        return {
            "confidence": confidence.astype(np.float32),
            "negative_loss": (-loss).astype(np.float32),
        }

    probs_clip = np.clip(probs, eps, 1.0 - eps)
    per_element_loss = -(labels * np.log(probs_clip) + (1.0 - labels) * np.log(1.0 - probs_clip))
    denom = np.maximum(labels.sum(axis=1), 1.0)
    pos_conf = ((probs_clip * labels).sum(axis=1) / denom).astype(np.float32)
    return {
        "confidence": pos_conf,
        "negative_loss": (-per_element_loss.mean(axis=1)).astype(np.float32),
    }


def _oof_auc(features: np.ndarray, targets: np.ndarray, *, seed: int, folds: int = 5) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, None]
    if not np.isfinite(features).all():
        col_means = np.nanmean(features, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        missing = ~np.isfinite(features)
        if np.any(missing):
            features = features.copy()
            features[missing] = np.take(col_means, np.where(missing)[1])
    if np.unique(targets).size < 2:
        return {"auc": float("nan"), "probs": np.full(targets.shape[0], np.nan, dtype=np.float32)}
    min_count = int(np.bincount(targets).min())
    if min_count < 2:
        return {"auc": float("nan"), "probs": np.full(targets.shape[0], np.nan, dtype=np.float32)}
    splitter = StratifiedKFold(n_splits=max(2, min(folds, min_count)), shuffle=True, random_state=seed)
    probs = np.zeros(targets.shape[0], dtype=np.float32)
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(features, targets)):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, solver="lbfgs", random_state=seed + fold_index)),
            ]
        )
        model.fit(features[train_idx], targets[train_idx])
        probs[test_idx] = model.predict_proba(features[test_idx])[:, 1].astype(np.float32)
    return {"auc": float(roc_auc_score(targets, probs)), "probs": probs}


def _oof_regression(features: np.ndarray, targets: np.ndarray, *, seed: int, folds: int = 5) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    if features.ndim == 1:
        features = features[:, None]
    if not np.isfinite(features).all():
        col_means = np.nanmean(features, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        missing = ~np.isfinite(features)
        if np.any(missing):
            features = features.copy()
            features[missing] = np.take(col_means, np.where(missing)[1])
    finite_targets = np.isfinite(targets)
    if not finite_targets.all():
        features = features[finite_targets]
        targets = targets[finite_targets]
    if targets.size < 3 or float(np.std(targets)) < 1e-8:
        return {"preds": np.full(targets.shape[0], np.nan, dtype=np.float32), "spearman": float("nan"), "pearson": float("nan"), "r2": float("nan")}

    splitter = KFold(n_splits=max(2, min(folds, targets.shape[0])), shuffle=True, random_state=seed)
    preds = np.zeros(targets.shape[0], dtype=np.float32)
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(features)):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=seed + fold_index)),
            ]
        )
        model.fit(features[train_idx], targets[train_idx])
        preds[test_idx] = model.predict(features[test_idx]).astype(np.float32)

    spearman = stats.spearmanr(targets, preds)
    pearson = stats.pearsonr(targets, preds)
    return {
        "preds": preds,
        "spearman": float(spearman.statistic),
        "pearson": float(pearson.statistic),
        "r2": float(r2_score(targets, preds)),
    }


def _bootstrap_auc_difference(
    targets: np.ndarray,
    left_probs: np.ndarray,
    right_probs: np.ndarray,
    *,
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    left_probs = np.asarray(left_probs, dtype=np.float32)
    right_probs = np.asarray(right_probs, dtype=np.float32)
    finite_mask = np.isfinite(left_probs) & np.isfinite(right_probs)
    if not np.all(finite_mask):
        targets = targets[finite_mask]
        left_probs = left_probs[finite_mask]
        right_probs = right_probs[finite_mask]
    if targets.size < 3 or np.unique(targets).size < 2:
        return {"mean_diff": float("nan"), "ci_95": None, "p_value_right_gt_left": None}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(num_bootstrap):
        indices = rng.integers(0, targets.shape[0], size=targets.shape[0])
        sample_targets = targets[indices]
        if np.unique(sample_targets).size < 2:
            continue
        diffs.append(
            float(roc_auc_score(sample_targets, right_probs[indices]) - roc_auc_score(sample_targets, left_probs[indices]))
        )
    if not diffs:
        return {"mean_diff": float("nan"), "ci_95": None, "p_value_right_gt_left": None}
    diffs_array = np.asarray(diffs, dtype=np.float32)
    return {
        "mean_diff": float(diffs_array.mean()),
        "ci_95": [float(np.quantile(diffs_array, 0.025)), float(np.quantile(diffs_array, 0.975))],
        "p_value_right_gt_left": float((1.0 + np.sum(diffs_array <= 0.0)) / (diffs_array.size + 1.0)),
    }


def _bootstrap_regression_difference(
    targets: np.ndarray,
    left_preds: np.ndarray,
    right_preds: np.ndarray,
    *,
    metric: str,
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    left_preds = np.asarray(left_preds, dtype=np.float32).reshape(-1)
    right_preds = np.asarray(right_preds, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(targets) & np.isfinite(left_preds) & np.isfinite(right_preds)
    if not np.all(finite_mask):
        targets = targets[finite_mask]
        left_preds = left_preds[finite_mask]
        right_preds = right_preds[finite_mask]
    if targets.size < 3:
        return {"mean_diff": float("nan"), "ci_95": None, "p_value_right_gt_left": None}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(num_bootstrap):
        indices = rng.integers(0, targets.shape[0], size=targets.shape[0])
        sample_targets = targets[indices]
        sample_left = left_preds[indices]
        sample_right = right_preds[indices]
        if metric == "spearman":
            left_value = float(stats.spearmanr(sample_targets, sample_left).statistic)
            right_value = float(stats.spearmanr(sample_targets, sample_right).statistic)
        elif metric == "r2":
            left_value = float(r2_score(sample_targets, sample_left))
            right_value = float(r2_score(sample_targets, sample_right))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        diffs.append(right_value - left_value)

    diffs_array = np.asarray(diffs, dtype=np.float32)
    return {
        "mean_diff": float(diffs_array.mean()),
        "ci_95": [float(np.quantile(diffs_array, 0.025)), float(np.quantile(diffs_array, 0.975))],
        "p_value_right_gt_left": float((1.0 + np.sum(diffs_array <= 0.0)) / (diffs_array.size + 1.0)),
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return ((values - mean) / std).astype(np.float32)


def _quantile_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, quantiles)
    bins = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    return np.clip(bins, 0, max(0, num_bins - 1))


def _cluster_pid_profiles(pid_features: np.ndarray, num_bins: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pid_features = np.asarray(pid_features, dtype=np.float32)
    if pid_features.ndim == 1:
        pid_features = pid_features[:, None]
    if pid_features.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, pid_features.shape[1]), dtype=np.float32)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(pid_features)
    if standardized.shape[0] < 2 or float(np.nanmax(np.abs(standardized))) < 1e-8:
        center = pid_features.mean(axis=0, keepdims=True).astype(np.float32)
        return np.zeros(pid_features.shape[0], dtype=np.int64), center

    n_clusters = max(1, min(num_bins, pid_features.shape[0]))
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    assignments = model.fit_predict(standardized).astype(np.int64)
    centers = scaler.inverse_transform(model.cluster_centers_).astype(np.float32)
    return assignments, centers


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_rate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values > 0.5))


def _safe_ttest(high: np.ndarray, low: np.ndarray) -> dict[str, float] | None:
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    if high.size < 2 or low.size < 2:
        return None
    test = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")
    return {"t_statistic": float(test.statistic), "p_value": float(test.pvalue)}


def compute_conditional_predictive_power(
    *,
    pid_features: np.ndarray,
    fisher_features: np.ndarray,
    correctness: np.ndarray,
    negative_loss: np.ndarray,
    negative_loss_advantage: np.ndarray,
    correctness_advantage: np.ndarray,
    beats_best_unimodal: np.ndarray,
    seed: int,
    num_bootstrap: int,
) -> dict[str, Any]:
    pid_features = np.asarray(pid_features, dtype=np.float32)
    fisher_features = np.asarray(fisher_features, dtype=np.float32)
    both_features = np.concatenate([pid_features, fisher_features], axis=1)
    continuous_targets = {
        "negative_loss": np.asarray(negative_loss, dtype=np.float32),
        "negative_loss_advantage": np.asarray(negative_loss_advantage, dtype=np.float32),
        "correctness_advantage": np.asarray(correctness_advantage, dtype=np.float32),
    }
    binary_targets = {
        "correctness": np.asarray(correctness, dtype=np.int64),
        "beats_best_unimodal": np.asarray(beats_best_unimodal, dtype=np.int64),
    }

    regression: dict[str, Any] = {}
    for offset, (name, target) in enumerate(continuous_targets.items(), start=1):
        pid_model = _oof_regression(pid_features, target, seed=seed + offset * 100 + 1)
        fisher_model = _oof_regression(fisher_features, target, seed=seed + offset * 100 + 2)
        both_model = _oof_regression(both_features, target, seed=seed + offset * 100 + 3)
        regression[name] = {
            "models": {
                "pid_only": {
                    "spearman": pid_model["spearman"],
                    "pearson": pid_model["pearson"],
                    "r2": pid_model["r2"],
                },
                "fisher_only": {
                    "spearman": fisher_model["spearman"],
                    "pearson": fisher_model["pearson"],
                    "r2": fisher_model["r2"],
                },
                "pid_plus_fisher": {
                    "spearman": both_model["spearman"],
                    "pearson": both_model["pearson"],
                    "r2": both_model["r2"],
                },
            },
            "bootstrap_tests": {
                "fisher_minus_pid_spearman": _bootstrap_regression_difference(
                    target,
                    pid_model["preds"],
                    fisher_model["preds"],
                    metric="spearman",
                    num_bootstrap=num_bootstrap,
                    seed=seed + offset * 100 + 11,
                ),
                "fisher_minus_pid_r2": _bootstrap_regression_difference(
                    target,
                    pid_model["preds"],
                    fisher_model["preds"],
                    metric="r2",
                    num_bootstrap=num_bootstrap,
                    seed=seed + offset * 100 + 21,
                ),
                "both_minus_pid_spearman": _bootstrap_regression_difference(
                    target,
                    pid_model["preds"],
                    both_model["preds"],
                    metric="spearman",
                    num_bootstrap=num_bootstrap,
                    seed=seed + offset * 100 + 31,
                ),
                "both_minus_pid_r2": _bootstrap_regression_difference(
                    target,
                    pid_model["preds"],
                    both_model["preds"],
                    metric="r2",
                    num_bootstrap=num_bootstrap,
                    seed=seed + offset * 100 + 41,
                ),
            },
        }

    classification: dict[str, Any] = {}
    for offset, (name, target) in enumerate(binary_targets.items(), start=1):
        if np.unique(target).size < 2:
            classification[name] = None
            continue
        pid_model = _oof_auc(pid_features, target, seed=seed + offset * 1000 + 1)
        fisher_model = _oof_auc(fisher_features, target, seed=seed + offset * 1000 + 2)
        both_model = _oof_auc(both_features, target, seed=seed + offset * 1000 + 3)
        classification[name] = {
            "models": {
                "pid_only": {"auc": pid_model["auc"]},
                "fisher_only": {"auc": fisher_model["auc"]},
                "pid_plus_fisher": {"auc": both_model["auc"]},
            },
            "bootstrap_tests": {
                "fisher_minus_pid_auc": _bootstrap_auc_difference(
                    target,
                    pid_model["probs"],
                    fisher_model["probs"],
                    num_bootstrap=num_bootstrap,
                    seed=seed + offset * 1000 + 11,
                ),
                "both_minus_pid_auc": _bootstrap_auc_difference(
                    target,
                    pid_model["probs"],
                    both_model["probs"],
                    num_bootstrap=num_bootstrap,
                    seed=seed + offset * 1000 + 21,
                ),
            },
        }

    return {"regression": regression, "classification": classification}


def compute_matched_bin_analysis(
    *,
    pid_features: np.ndarray,
    fisher_score: np.ndarray,
    negative_loss_advantage: np.ndarray,
    correctness: np.ndarray,
    beats_best_unimodal: np.ndarray,
    num_bins: int,
    seed: int,
) -> dict[str, Any]:
    pid_features = np.asarray(pid_features, dtype=np.float32)
    if pid_features.ndim == 1:
        pid_features = pid_features[:, None]
    fisher_score = np.asarray(fisher_score, dtype=np.float32).reshape(-1)
    negative_loss_advantage = np.asarray(negative_loss_advantage, dtype=np.float32).reshape(-1)
    correctness = np.asarray(correctness, dtype=np.int64).reshape(-1)
    beats_best_unimodal = np.asarray(beats_best_unimodal, dtype=np.int64).reshape(-1)
    bins, centers = _cluster_pid_profiles(pid_features, num_bins=num_bins, seed=seed)
    num_clusters = int(centers.shape[0])

    per_bin = []
    negative_loss_advantage_deltas = []
    correctness_deltas = []
    beats_best_unimodal_deltas = []
    for bin_index in range(num_clusters):
        mask = bins == bin_index
        if int(mask.sum()) < 4:
            continue
        bin_fisher = fisher_score[mask]
        threshold = float(np.median(bin_fisher))
        high_mask = mask.copy()
        low_mask = mask.copy()
        high_mask[mask] = bin_fisher >= threshold
        low_mask[mask] = bin_fisher < threshold
        if int(high_mask.sum()) < 2 or int(low_mask.sum()) < 2:
            continue

        high_negative_loss_advantage = negative_loss_advantage[high_mask]
        low_negative_loss_advantage = negative_loss_advantage[low_mask]
        high_correct = correctness[high_mask]
        low_correct = correctness[low_mask]
        high_beats_best = beats_best_unimodal[high_mask].astype(np.float32)
        low_beats_best = beats_best_unimodal[low_mask].astype(np.float32)

        negative_loss_advantage_delta = float(np.mean(high_negative_loss_advantage) - np.mean(low_negative_loss_advantage))
        correctness_delta = float(np.mean(high_correct) - np.mean(low_correct))
        beats_best_unimodal_delta = float(np.mean(high_beats_best) - np.mean(low_beats_best))
        negative_loss_advantage_deltas.append(negative_loss_advantage_delta)
        correctness_deltas.append(correctness_delta)
        beats_best_unimodal_deltas.append(beats_best_unimodal_delta)
        per_bin.append(
            {
                "cluster_index": bin_index,
                "count": int(mask.sum()),
                "pid_cluster_center": centers[bin_index].astype(float).tolist(),
                "fisher_threshold": threshold,
                "high_count": int(high_mask.sum()),
                "low_count": int(low_mask.sum()),
                "high_negative_loss_advantage": _safe_mean(high_negative_loss_advantage),
                "low_negative_loss_advantage": _safe_mean(low_negative_loss_advantage),
                "negative_loss_advantage_delta": negative_loss_advantage_delta,
                "high_correctness": _safe_mean(high_correct),
                "low_correctness": _safe_mean(low_correct),
                "correctness_delta": correctness_delta,
                "high_beats_best_unimodal_rate": _safe_rate(high_beats_best),
                "low_beats_best_unimodal_rate": _safe_rate(low_beats_best),
                "beats_best_unimodal_delta": beats_best_unimodal_delta,
                "negative_loss_advantage_ttest": _safe_ttest(high_negative_loss_advantage, low_negative_loss_advantage),
            }
        )

    valid_negative_loss_advantage = np.asarray(negative_loss_advantage_deltas, dtype=np.float32)
    valid_correctness = np.asarray(correctness_deltas, dtype=np.float32)
    valid_beats_best_unimodal = np.asarray(beats_best_unimodal_deltas, dtype=np.float32)
    return {
        "num_bins": num_bins,
        "num_clusters": num_clusters,
        "cluster_assignments": bins.astype(int).tolist(),
        "per_bin": per_bin,
        "summary": {
            "num_valid_bins": len(per_bin),
            "mean_negative_loss_advantage_delta": _safe_mean(valid_negative_loss_advantage),
            "mean_correctness_delta": _safe_mean(valid_correctness),
            "mean_beats_best_unimodal_delta": _safe_mean(valid_beats_best_unimodal),
            "positive_negative_loss_advantage_bins": int(np.sum(valid_negative_loss_advantage > 0.0)),
            "positive_correctness_bins": int(np.sum(valid_correctness > 0.0)),
            "positive_beats_best_unimodal_bins": int(np.sum(valid_beats_best_unimodal > 0.0)),
        },
    }


@dataclass
class ExperimentSplit:
    modalities: dict[str, np.ndarray]
    labels: np.ndarray
    mask: np.ndarray | None
    ids: list[str]


@dataclass
class DatasetSpec:
    name: str
    task: str
    modality_names: list[str]
    train: ExperimentSplit
    valid: ExperimentSplit
    test: ExperimentSplit


class MultiModalDataset(Dataset):
    def __init__(self, split: ExperimentSplit) -> None:
        self.modalities = {name: torch.from_numpy(value).float() for name, value in split.modalities.items()}
        self.labels = torch.from_numpy(split.labels).float()
        self.mask = None if split.mask is None else torch.from_numpy(split.mask).bool()
        self.names = list(split.modalities.keys())

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        modalities = {name: tensor[index] for name, tensor in self.modalities.items()}
        mask = None if self.mask is None else self.mask[index]
        return modalities, mask, self.labels[index]


def _collate_batch(items: list[tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor]]):
    names = list(items[0][0].keys())
    modalities = {name: torch.stack([row[0][name] for row in items], dim=0) for name in names}
    masks = None if items[0][1] is None else torch.stack([row[1] for row in items], dim=0)
    labels = torch.stack([row[2] for row in items], dim=0)
    return modalities, masks, labels


def _make_loader(split: ExperimentSplit, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(MultiModalDataset(split), batch_size=batch_size, shuffle=shuffle, drop_last=False, collate_fn=_collate_batch)


def load_mosi(path: str | Path, *, max_seq_len: int | None = 50) -> DatasetSpec:
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    def _build(split_name: str) -> ExperimentSplit:
        split = payload[split_name]
        text = np.asarray(split["text"], dtype=np.float32)
        audio = np.asarray(split["audio"], dtype=np.float32)
        video = np.asarray(split["vision"], dtype=np.float32)
        labels = np.asarray(split["labels"], dtype=np.float32).reshape(text.shape[0], -1)[:, 0]
        if max_seq_len is not None:
            text = text[:, :max_seq_len]
            audio = audio[:, :max_seq_len]
            video = video[:, :max_seq_len]
        text[~np.isfinite(text)] = 0.0
        audio[~np.isfinite(audio)] = 0.0
        video[~np.isfinite(video)] = 0.0
        mask = np.abs(text).sum(axis=-1) > 0
        keep = mask.any(axis=1)
        ids = [str(item) for item, flag in zip(split.get("id", [f"{split_name}_{i}" for i in range(text.shape[0])]), keep.tolist()) if flag]
        return ExperimentSplit(
            modalities={
                "text": text[keep].astype(np.float32),
                "audio": audio[keep].astype(np.float32),
                "video": video[keep].astype(np.float32),
            },
            labels=(labels[keep] > 0).astype(np.float32),
            mask=mask[keep].astype(bool),
            ids=ids,
        )

    return DatasetSpec(name="mosi", task="binary", modality_names=["text", "audio", "video"], train=_build("train"), valid=_build("valid"), test=_build("test"))


def load_affect(path: str | Path, *, dataset_name: str, max_seq_len: int | None = 50) -> DatasetSpec:
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    def _normalize_ids(raw_ids: Any, split_name: str, expected_size: int) -> list[str]:
        if raw_ids is None:
            return [f"{split_name}_{idx}" for idx in range(expected_size)]
        normalized: list[str] = []
        for item in raw_ids:
            if isinstance(item, np.ndarray):
                if item.size == 0:
                    normalized.append(f"{split_name}_{len(normalized)}")
                else:
                    normalized.append(str(item.reshape(-1)[0]))
            elif isinstance(item, (list, tuple)):
                normalized.append(str(item[0] if item else f"{split_name}_{len(normalized)}"))
            else:
                normalized.append(str(item))
        return normalized

    def _build(split_name: str) -> ExperimentSplit:
        split = payload[split_name]
        text = np.asarray(split["text"], dtype=np.float32)
        audio = np.asarray(split["audio"], dtype=np.float32)
        video = np.asarray(split["vision"], dtype=np.float32)
        labels = np.asarray(split["labels"], dtype=np.float32).reshape(text.shape[0], -1)[:, 0]
        if max_seq_len is not None:
            text = text[:, :max_seq_len]
            audio = audio[:, :max_seq_len]
            video = video[:, :max_seq_len]
        text[~np.isfinite(text)] = 0.0
        audio[~np.isfinite(audio)] = 0.0
        video[~np.isfinite(video)] = 0.0
        mask = np.abs(text).sum(axis=-1) > 0
        keep = mask.any(axis=1)
        ids = _normalize_ids(split.get("id"), split_name, text.shape[0])
        return ExperimentSplit(
            modalities={
                "text": text[keep].astype(np.float32),
                "audio": audio[keep].astype(np.float32),
                "video": video[keep].astype(np.float32),
            },
            labels=(labels[keep] > 0).astype(np.float32),
            mask=mask[keep].astype(bool),
            ids=[item for item, flag in zip(ids, keep.tolist()) if flag],
        )

    train = _build("train")
    valid = _build("valid")
    test = _build("test")
    return DatasetSpec(name=dataset_name, task="binary", modality_names=["text", "audio", "video"], train=train, valid=valid, test=test)


def load_imdb(
    path: str | Path,
    *,
    max_samples: int | None = None,
    train_max_samples: int | None = None,
    valid_max_samples: int | None = None,
    test_max_samples: int | None = None,
    text_dropout_prob: float = 0.0,
    text_dropout_seed: int = 0,
) -> DatasetSpec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MM-IMDb file not found at {path}")
    split_bounds = {"train": (0, 15552), "valid": (15552, 18160), "test": (18160, 25959)}

    split_max_samples = {
        "train": train_max_samples if train_max_samples is not None else max_samples,
        "valid": valid_max_samples if valid_max_samples is not None else max_samples,
        "test": test_max_samples if test_max_samples is not None else max_samples,
    }

    def _build(handle: h5py.File, split_name: str) -> ExperimentSplit:
        start, stop = split_bounds[split_name]
        local_max_samples = split_max_samples[split_name]
        if local_max_samples is not None:
            stop = min(stop, start + local_max_samples)
        text = np.asarray(handle["features"][start:stop], dtype=np.float32)
        image = np.asarray(handle["vgg_features"][start:stop], dtype=np.float32).reshape(stop - start, -1)
        labels = np.asarray(handle["genres"][start:stop], dtype=np.float32)
        imdb_ids = np.asarray(handle["imdb_ids"][start:stop], dtype=np.int64)
        if text_dropout_prob > 0.0:
            rng = np.random.default_rng(text_dropout_seed + {"train": 11, "valid": 23, "test": 37}[split_name])
            drop_mask = rng.random(text.shape[0]) < text_dropout_prob
            if np.any(drop_mask):
                text = text.copy()
                text[drop_mask] = 0.0
        ids = [str(int(item)) for item in imdb_ids]
        return ExperimentSplit(modalities={"text": text, "image": image}, labels=labels, mask=None, ids=ids)

    with h5py.File(path, "r") as handle:
        train = _build(handle, "train")
        valid = _build(handle, "valid")
        test = _build(handle, "test")
    return DatasetSpec(name="imdb", task="multilabel", modality_names=["text", "image"], train=train, valid=valid, test=test)


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, fusion_dim: int, dropout: float, rnn_type: str = "gru") -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.long().sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=inputs.shape[1])
        outputs = self.dropout(outputs)
        projected = self.proj(outputs)
        return projected, mask


class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, fusion_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.net(inputs).unsqueeze(1)
        seq_mask = torch.ones(inputs.shape[0], 1, device=inputs.device, dtype=torch.bool)
        return seq, seq_mask


class CrossModalBlock(nn.Module):
    def __init__(self, fusion_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, fusion_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        others: list[torch.Tensor],
        other_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        if not others:
            return query
        keys = torch.cat(others, dim=1)
        key_mask = torch.cat(other_masks, dim=1)
        if not bool(key_mask.any()):
            return query
        attended, _ = self.attn(query, keys, keys, key_padding_mask=~key_mask)
        hidden = self.norm1(query + self.dropout(attended))
        return self.norm2(hidden + self.dropout(self.ffn(hidden)))


class SharedResidualBlock(nn.Module):
    def __init__(self, fusion_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, fusion_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm1(hidden + self.dropout(self.ffn(hidden)))


class JointMultimodalModel(nn.Module):
    def __init__(
        self,
        *,
        modality_input_dims: dict[str, int],
        modality_kinds: dict[str, str],
        encoder_hidden_dim: int,
        fusion_dim: int,
        fusion_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        task: str,
        output_dim: int,
        cross_modal_enabled: bool = True,
        architecture: str = "shared_head",
    ) -> None:
        super().__init__()
        self.modality_names = list(modality_input_dims.keys())
        self.task = task
        self.cross_modal_enabled = cross_modal_enabled
        self.architecture = architecture
        if architecture == "cross_attention":
            self.fusion_layer_names = ["enc"] + [f"fusion_{index}" for index in range(1, fusion_layers + 1)]
        elif architecture == "shared_head":
            self.fusion_layer_names = ["enc"] + [f"shared_{index}" for index in range(1, fusion_layers + 1)]
        else:
            raise ValueError(f"Unsupported joint architecture: {architecture}")
        encoders: dict[str, nn.Module] = {}
        for name, dim in modality_input_dims.items():
            if modality_kinds[name] == "sequence":
                encoders[name] = SequenceEncoder(dim, encoder_hidden_dim, fusion_dim, dropout, rnn_type="gru")
            else:
                encoders[name] = VectorEncoder(dim, encoder_hidden_dim, fusion_dim, dropout)
        self.encoders = nn.ModuleDict(encoders)
        if architecture == "cross_attention":
            self.fusion_blocks = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            name: CrossModalBlock(fusion_dim=fusion_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
                            for name in self.modality_names
                        }
                    )
                    for _ in range(fusion_layers)
                ]
            )
            self.shared_blocks = None
            self.fusion_head = nn.Sequential(
                nn.LayerNorm(fusion_dim * len(self.modality_names)),
                nn.Linear(fusion_dim * len(self.modality_names), fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.fusion_blocks = None
            self.shared_blocks = nn.ModuleList(
                [SharedResidualBlock(fusion_dim=fusion_dim, ffn_dim=ffn_dim, dropout=dropout) for _ in range(fusion_layers)]
            )
            self.fusion_head = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.classifier = nn.Linear(fusion_dim, output_dim)

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        mask: torch.Tensor | None,
        *,
        active_modalities: set[str] | None = None,
        retain_layer_grads: bool = False,
        retain_fused_grad: bool = False,
    ) -> dict[str, Any]:
        active = set(self.modality_names) if active_modalities is None else set(active_modalities)
        sequences: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        pooled_layers: dict[str, dict[str, torch.Tensor]] = {layer_name: {} for layer_name in self.fusion_layer_names}
        layer_sequences: dict[str, dict[str, torch.Tensor]] = {layer_name: {} for layer_name in self.fusion_layer_names}

        for name in self.modality_names:
            if name in active:
                seq, seq_mask = self.encoders[name](modalities[name], mask)
            else:
                if modalities[name].ndim == 3:
                    batch, length = modalities[name].shape[:2]
                    seq = torch.zeros(batch, length, self.classifier.in_features, device=modalities[name].device)
                    seq_mask = torch.zeros(batch, length, dtype=torch.bool, device=modalities[name].device)
                else:
                    batch = modalities[name].shape[0]
                    seq = torch.zeros(batch, 1, self.classifier.in_features, device=modalities[name].device)
                    seq_mask = torch.zeros(batch, 1, dtype=torch.bool, device=modalities[name].device)
            if retain_layer_grads:
                seq.retain_grad()
            sequences[name] = seq
            masks[name] = seq_mask
            layer_sequences["enc"][name] = seq
            pooled = _masked_mean(seq, seq_mask)
            pooled_layers["enc"][name] = pooled

        current_seq = sequences
        if self.architecture == "cross_attention":
            assert self.fusion_blocks is not None
            for layer_index, block in enumerate(self.fusion_blocks, start=1):
                next_seq: dict[str, torch.Tensor] = {}
                for name in self.modality_names:
                    if self.cross_modal_enabled:
                        other_names = [item for item in self.modality_names if item != name and item in active]
                    else:
                        other_names = []
                    next_seq[name] = block[name](
                        current_seq[name],
                        masks[name],
                        [current_seq[item] for item in other_names],
                        [masks[item] for item in other_names],
                    )
                    if retain_layer_grads:
                        next_seq[name].retain_grad()
                    layer_sequences[f"fusion_{layer_index}"][name] = next_seq[name]
                    pooled = _masked_mean(next_seq[name], masks[name])
                    pooled_layers[f"fusion_{layer_index}"][name] = pooled
                current_seq = next_seq
            final_vectors = [pooled_layers[self.fusion_layer_names[-1]][name] for name in self.modality_names]
            fused_hidden = self.fusion_head(torch.cat(final_vectors, dim=1))
        else:
            assert self.shared_blocks is not None
            for layer_index, block in enumerate(self.shared_blocks, start=1):
                next_seq = {}
                for name in self.modality_names:
                    if name in active:
                        next_seq[name] = block(current_seq[name])
                    else:
                        next_seq[name] = current_seq[name]
                    if retain_layer_grads:
                        next_seq[name].retain_grad()
                    layer_sequences[f"shared_{layer_index}"][name] = next_seq[name]
                    pooled = _masked_mean(next_seq[name], masks[name])
                    pooled_layers[f"shared_{layer_index}"][name] = pooled
                current_seq = next_seq
            final_vectors = [pooled_layers[self.fusion_layer_names[-1]][name] for name in self.modality_names if name in active]
            if not final_vectors:
                batch_size = next(iter(modalities.values())).shape[0]
                hidden_dim = self.classifier.in_features
                fused_hidden = torch.zeros(batch_size, hidden_dim, device=self.classifier.weight.device)
            else:
                fused_hidden = torch.stack(final_vectors, dim=0).mean(dim=0)
            fused_hidden = self.fusion_head(fused_hidden)
        if retain_fused_grad:
            fused_hidden.retain_grad()
        logits = self.classifier(fused_hidden)
        return {
            "logits": logits,
            "layer_pooled": pooled_layers,
            "layer_sequences": layer_sequences,
            "layer_masks": masks,
            "fused_hidden": fused_hidden,
        }


class UnimodalModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        kind: str,
        encoder_hidden_dim: int,
        fusion_dim: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        if kind == "sequence":
            self.encoder = SequenceEncoder(input_dim, encoder_hidden_dim, fusion_dim, dropout, rnn_type="gru")
        else:
            self.encoder = VectorEncoder(input_dim, encoder_hidden_dim, fusion_dim, dropout)
        self.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, output_dim))

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
        seq, seq_mask = self.encoder(inputs, mask)
        pooled = _masked_mean(seq, seq_mask)
        logits = self.head(pooled)
        return {"logits": logits, "proj": pooled}


def _infer_modality_kinds(split: ExperimentSplit) -> dict[str, str]:
    return {name: ("sequence" if value.ndim == 3 else "vector") for name, value in split.modalities.items()}


def _move_modalities(modalities: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in modalities.items()}


def _compute_loss(task: str, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if task == "binary":
        logits_1 = logits.squeeze(-1)
        labels_1 = labels.view(-1)
        per_sample = F.binary_cross_entropy_with_logits(logits_1, labels_1, reduction="none")
        return per_sample.mean(), per_sample
    per_element = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    per_sample = per_element.mean(dim=1)
    return per_sample.mean(), per_sample


def build_joint_model(
    dataset: DatasetSpec,
    *,
    encoder_hidden_dim: int,
    fusion_dim: int,
    fusion_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    cross_modal_enabled: bool = True,
    architecture: str = "shared_head",
) -> JointMultimodalModel:
    kinds = _infer_modality_kinds(dataset.train)
    output_dim = 1 if dataset.task == "binary" else dataset.train.labels.shape[1]
    return JointMultimodalModel(
        modality_input_dims={name: dataset.train.modalities[name].shape[-1] for name in dataset.modality_names},
        modality_kinds=kinds,
        encoder_hidden_dim=encoder_hidden_dim,
        fusion_dim=fusion_dim,
        fusion_layers=fusion_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        task=dataset.task,
        output_dim=output_dim,
        cross_modal_enabled=cross_modal_enabled,
        architecture=architecture,
    )


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_joint_model(
    dataset: DatasetSpec,
    *,
    device: str,
    encoder_hidden_dim: int,
    fusion_dim: int,
    fusion_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
    cross_modal_enabled: bool = True,
    architecture: str = "shared_head",
) -> tuple[JointMultimodalModel, dict[str, float]]:
    set_seed(seed)
    model = build_joint_model(
        dataset,
        encoder_hidden_dim=encoder_hidden_dim,
        fusion_dim=fusion_dim,
        fusion_layers=fusion_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        cross_modal_enabled=cross_modal_enabled,
        architecture=architecture,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _make_loader(dataset.train, batch_size=batch_size, shuffle=True)
    valid_loader = _make_loader(dataset.valid, batch_size=batch_size, shuffle=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_valid = float("inf")
    stalled = 0
    metrics = {"best_valid_loss": float("inf")}
    for _ in range(epochs):
        model.train()
        for modalities, mask, labels in train_loader:
            modalities = _move_modalities(modalities, device)
            mask = None if mask is None else mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(modalities, mask)
            loss, _ = _compute_loss(dataset.task, outputs["logits"], labels)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_losses = []
        with torch.inference_mode():
            for modalities, mask, labels in valid_loader:
                modalities = _move_modalities(modalities, device)
                mask = None if mask is None else mask.to(device)
                labels = labels.to(device)
                outputs = model(modalities, mask)
                loss, _ = _compute_loss(dataset.task, outputs["logits"], labels)
                valid_losses.append(float(loss.item()))
        valid_loss = float(np.mean(valid_losses))
        if valid_loss + 1e-5 < best_valid:
            best_valid = valid_loss
            stalled = 0
            metrics["best_valid_loss"] = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stalled += 1
            if stalled >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, metrics


def train_joint_model_with_snapshots(
    dataset: DatasetSpec,
    *,
    device: str,
    encoder_hidden_dim: int,
    fusion_dim: int,
    fusion_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
    capture_epochs: list[int] | None = None,
    cross_modal_enabled: bool = True,
    architecture: str = "shared_head",
) -> tuple[JointMultimodalModel, dict[str, float], dict[str, dict[str, torch.Tensor]]]:
    set_seed(seed)
    model = build_joint_model(
        dataset,
        encoder_hidden_dim=encoder_hidden_dim,
        fusion_dim=fusion_dim,
        fusion_layers=fusion_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        cross_modal_enabled=cross_modal_enabled,
        architecture=architecture,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _make_loader(dataset.train, batch_size=batch_size, shuffle=True)
    valid_loader = _make_loader(dataset.valid, batch_size=batch_size, shuffle=False)

    capture_epochs_set = {int(epoch) for epoch in (capture_epochs or []) if int(epoch) > 0}
    checkpoints: dict[str, dict[str, torch.Tensor]] = {}
    best_state: dict[str, torch.Tensor] | None = None
    best_valid = float("inf")
    best_epoch = 0
    stalled = 0
    metrics: dict[str, float] = {"best_valid_loss": float("inf"), "best_epoch": 0.0, "stopped_epoch": 0.0}

    for epoch_index in range(1, epochs + 1):
        model.train()
        for modalities, mask, labels in train_loader:
            modalities = _move_modalities(modalities, device)
            mask = None if mask is None else mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(modalities, mask)
            loss, _ = _compute_loss(dataset.task, outputs["logits"], labels)
            loss.backward()
            optimizer.step()

        if epoch_index in capture_epochs_set:
            checkpoints[f"epoch_{epoch_index}"] = _clone_state_dict(model)

        model.eval()
        valid_losses = []
        with torch.inference_mode():
            for modalities, mask, labels in valid_loader:
                modalities = _move_modalities(modalities, device)
                mask = None if mask is None else mask.to(device)
                labels = labels.to(device)
                outputs = model(modalities, mask)
                loss, _ = _compute_loss(dataset.task, outputs["logits"], labels)
                valid_losses.append(float(loss.item()))
        valid_loss = float(np.mean(valid_losses))
        if valid_loss + 1e-5 < best_valid:
            best_valid = valid_loss
            best_epoch = epoch_index
            stalled = 0
            metrics["best_valid_loss"] = valid_loss
            metrics["best_epoch"] = float(epoch_index)
            best_state = _clone_state_dict(model)
        else:
            stalled += 1
            if stalled >= patience:
                metrics["stopped_epoch"] = float(epoch_index)
                break
    else:
        metrics["stopped_epoch"] = float(epochs)

    if best_state is None:
        best_state = _clone_state_dict(model)
        metrics["best_epoch"] = float(epochs)
        metrics["best_valid_loss"] = best_valid
    checkpoints["best"] = best_state
    model.load_state_dict(best_state)
    return model, metrics, checkpoints


def train_unimodal_models(
    dataset: DatasetSpec,
    *,
    device: str,
    encoder_hidden_dim: int,
    fusion_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
) -> tuple[dict[str, UnimodalModel], dict[str, dict[str, float]]]:
    kinds = _infer_modality_kinds(dataset.train)
    output_dim = 1 if dataset.task == "binary" else dataset.train.labels.shape[1]
    models: dict[str, UnimodalModel] = {}
    metrics: dict[str, dict[str, float]] = {}
    train_loader = _make_loader(dataset.train, batch_size=batch_size, shuffle=True)
    valid_loader = _make_loader(dataset.valid, batch_size=batch_size, shuffle=False)

    for modality_index, name in enumerate(dataset.modality_names):
        set_seed(seed + 97 * (modality_index + 1))
        model = UnimodalModel(
            input_dim=dataset.train.modalities[name].shape[-1],
            kind=kinds[name],
            encoder_hidden_dim=encoder_hidden_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            output_dim=output_dim,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_state: dict[str, torch.Tensor] | None = None
        best_valid = float("inf")
        stalled = 0
        metrics[name] = {"best_valid_loss": float("inf")}
        for _ in range(epochs):
            model.train()
            for modalities, mask, labels in train_loader:
                inputs = modalities[name].to(device)
                local_mask = None if mask is None else mask.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs, local_mask)
                loss, _ = _compute_loss(dataset.task, outputs["logits"], labels)
                loss.backward()
                optimizer.step()

            model.eval()
            valid_losses = []
            with torch.inference_mode():
                for modalities, mask, labels in valid_loader:
                    inputs = modalities[name].to(device)
                    local_mask = None if mask is None else mask.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs, local_mask)
                    loss, _ = _compute_loss(dataset.task, outputs["logits"], labels)
                    valid_losses.append(float(loss.item()))
            valid_loss = float(np.mean(valid_losses))
            if valid_loss + 1e-5 < best_valid:
                best_valid = valid_loss
                stalled = 0
                metrics[name]["best_valid_loss"] = valid_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stalled += 1
                if stalled >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        models[name] = model
    return models, metrics


def evaluate_joint_model(
    dataset: DatasetSpec,
    split: ExperimentSplit,
    joint_model: JointMultimodalModel,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    joint_logits: list[np.ndarray] = []
    layer_repr: dict[str, dict[str, list[np.ndarray]]] = {layer: {name: [] for name in dataset.modality_names} for layer in joint_model.fusion_layer_names}

    joint_model.eval()

    with torch.inference_mode():
        for modalities, mask, labels in loader:
            del labels
            modalities_device = _move_modalities(modalities, device)
            mask_device = None if mask is None else mask.to(device)
            outputs = joint_model(modalities_device, mask_device)
            joint_logits.append(outputs["logits"].detach().cpu().numpy().astype(np.float32))
            for layer_name in joint_model.fusion_layer_names:
                for modality_name in dataset.modality_names:
                    layer_repr[layer_name][modality_name].append(
                        outputs["layer_pooled"][layer_name][modality_name].detach().cpu().numpy().astype(np.float32)
                    )

    joint_logits_array = np.concatenate(joint_logits, axis=0)
    if dataset.task == "binary":
        joint_probs = torch.sigmoid(torch.from_numpy(joint_logits_array)).numpy().reshape(-1, 1).astype(np.float32)
    else:
        joint_probs = torch.sigmoid(torch.from_numpy(joint_logits_array)).numpy().astype(np.float32)

    return {
        "joint_probs": joint_probs,
        "layer_representations": {
            layer_name: {name: np.concatenate(chunks, axis=0) for name, chunks in modalities.items()}
            for layer_name, modalities in layer_repr.items()
        },
    }


def evaluate_models(
    dataset: DatasetSpec,
    split: ExperimentSplit,
    joint_model: JointMultimodalModel,
    unimodal_models: dict[str, UnimodalModel],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    outputs = evaluate_joint_model(dataset, split, joint_model, device=device, batch_size=batch_size)
    unimodal_logits: dict[str, list[np.ndarray]] = {name: [] for name in dataset.modality_names}
    unimodal_repr: dict[str, list[np.ndarray]] = {name: [] for name in dataset.modality_names}
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)

    for model in unimodal_models.values():
        model.eval()

    with torch.inference_mode():
        for modalities, mask, labels in loader:
            del labels
            modalities_device = _move_modalities(modalities, device)
            mask_device = None if mask is None else mask.to(device)
            for modality_name, model in unimodal_models.items():
                uni_out = model(modalities_device[modality_name], mask_device)
                unimodal_logits[modality_name].append(uni_out["logits"].detach().cpu().numpy().astype(np.float32))
                unimodal_repr[modality_name].append(uni_out["proj"].detach().cpu().numpy().astype(np.float32))

    unimodal_probs: dict[str, np.ndarray] = {}
    for modality_name, chunks in unimodal_logits.items():
        logits = np.concatenate(chunks, axis=0)
        if dataset.task == "binary":
            unimodal_probs[modality_name] = torch.sigmoid(torch.from_numpy(logits)).numpy().reshape(-1, 1).astype(np.float32)
        else:
            unimodal_probs[modality_name] = torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32)

    outputs["unimodal_probs"] = unimodal_probs
    outputs["unimodal_repr"] = {name: np.concatenate(chunks, axis=0) for name, chunks in unimodal_repr.items()}
    return outputs


def compute_interaction_buckets(
    dataset: DatasetSpec,
    split: ExperimentSplit,
    joint_probs: np.ndarray,
    unimodal_probs: dict[str, np.ndarray],
    layer_representations: dict[str, dict[str, np.ndarray]],
    *,
    device: str,
    bucket_margin: float,
    seed: int,
) -> dict[str, Any]:
    def _build_bucket_scores(
        *,
        redundancy_values: np.ndarray,
        synergy_values: np.ndarray,
        uniqueness_by_modality: dict[str, np.ndarray],
        normalize_scores: bool = False,
    ) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
        redundancy_values = np.asarray(redundancy_values, dtype=np.float32)
        synergy_values = np.asarray(synergy_values, dtype=np.float32)
        uniqueness_by_modality = {
            name: np.asarray(values, dtype=np.float32) for name, values in uniqueness_by_modality.items()
        }
        score_components = [redundancy_values, synergy_values, *[uniqueness_by_modality[name] for name in dataset.modality_names]]
        raw_score_matrix = np.stack(score_components, axis=1).astype(np.float32)
        if normalize_scores:
            score_matrix = np.stack([_zscore(raw_score_matrix[:, index]) for index in range(raw_score_matrix.shape[1])], axis=1)
        else:
            score_matrix = raw_score_matrix
        score_names = np.array(["RED", "SYN", *["UNQ"] * len(dataset.modality_names)], dtype=object)
        unique_names = np.array(["NONE", "NONE", *dataset.modality_names], dtype=object)
        sorted_scores = np.sort(score_matrix, axis=1)
        margin = sorted_scores[:, -1] - sorted_scores[:, -2]
        winner = score_matrix.argmax(axis=1)
        buckets_local = score_names[winner].astype(object)
        unique_modality = unique_names[winner].astype(object)
        unique_modality[buckets_local != "UNQ"] = "NONE"
        if bucket_margin > 0:
            ambiguous = margin < bucket_margin
            buckets_local[ambiguous] = "MIXED"
            unique_modality[ambiguous] = "NONE"
        summary_local = {bucket: int(np.sum(buckets_local == bucket)) for bucket in (*BUCKETS, "MIXED")}
        scores_local = {
            "redundancy": redundancy_values.astype(np.float32),
            "synergy": synergy_values.astype(np.float32),
            "uniqueness": {name: values.astype(np.float32) for name, values in uniqueness_by_modality.items()},
        }
        return {"buckets": buckets_local, "summary": summary_local, "unique_modality": unique_modality, "margin": margin.astype(np.float32)}, raw_score_matrix.astype(np.float32), scores_local

    labels = split.labels
    conf_joint = _confidence_for_targets(dataset.task, joint_probs, labels)
    conf_modalities = {name: _confidence_for_targets(dataset.task, probs, labels) for name, probs in unimodal_probs.items()}
    ordered_conf = np.stack([conf_modalities[name] for name in dataset.modality_names], axis=1)

    redundancy = ordered_conf.min(axis=1)
    synergy = conf_joint - ordered_conf.max(axis=1)
    uniqueness = []
    for index, name in enumerate(dataset.modality_names):
        others = np.delete(ordered_conf, index, axis=1)
        uniqueness.append(conf_modalities[name] - others.mean(axis=1))
    uniqueness_matrix = np.stack(uniqueness, axis=1)
    best_unique = uniqueness_matrix.max(axis=1)
    bucket_info, score_matrix, scores_payload = _build_bucket_scores(
        redundancy_values=redundancy,
        synergy_values=synergy,
        uniqueness_by_modality={name: uniqueness_matrix[:, idx] for idx, name in enumerate(dataset.modality_names)},
    )
    buckets = bucket_info["buckets"]
    summary = bucket_info["summary"]
    output: dict[str, Any] = {
        "buckets": buckets,
        "summary": summary,
        "scores": {
            **scores_payload,
            "conf_joint": conf_joint.astype(np.float32),
            "conf_modalities": {name: values.astype(np.float32) for name, values in conf_modalities.items()},
        },
    }

    if len(dataset.modality_names) >= 2:
        final_layer = joint_model_layer_name(layer_representations)
        pairwise_lsmi: dict[str, dict[str, np.ndarray]] = {}
        redundancy_terms: list[np.ndarray] = []
        synergy_terms: list[np.ndarray] = []
        unique_accumulator: dict[str, list[np.ndarray]] = {name: [] for name in dataset.modality_names}
        for pair_index, (left_name, right_name) in enumerate(combinations(dataset.modality_names, 2)):
            lsmi = estimate_samplewise_lsmi(
                modality_1_features=layer_representations[final_layer][left_name],
                modality_2_features=layer_representations[final_layer][right_name],
                probs_modality_1=unimodal_probs[left_name],
                probs_modality_2=unimodal_probs[right_name],
                probs_joint=joint_probs,
                labels=labels,
                device=device,
                entropy_epochs=15,
                entropy_batch_size=256,
                entropy_lr=1e-3,
                seed=seed + 1000 + pair_index,
            )
            pair_name = f"{left_name}__{right_name}"
            pairwise_lsmi[pair_name] = {
                "redundancy": lsmi.redundancy.astype(np.float32),
                f"unique_{left_name}": lsmi.unique_1.astype(np.float32),
                f"unique_{right_name}": lsmi.unique_2.astype(np.float32),
                "synergy": lsmi.synergy.astype(np.float32),
            }
            redundancy_terms.append(lsmi.redundancy.astype(np.float32))
            synergy_terms.append(lsmi.synergy.astype(np.float32))
            unique_accumulator[left_name].append(lsmi.unique_1.astype(np.float32))
            unique_accumulator[right_name].append(lsmi.unique_2.astype(np.float32))

        lsmi_redundancy = np.mean(np.stack(redundancy_terms, axis=0), axis=0).astype(np.float32)
        lsmi_synergy = np.mean(np.stack(synergy_terms, axis=0), axis=0).astype(np.float32)
        lsmi_unique = {
            f"unique_{name}": np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)
            for name, values in unique_accumulator.items()
        }
        lsmi_bucket_info, _, lsmi_scores = _build_bucket_scores(
            redundancy_values=lsmi_redundancy,
            synergy_values=lsmi_synergy,
            uniqueness_by_modality={name: lsmi_unique[f"unique_{name}"] for name in dataset.modality_names},
            normalize_scores=True,
        )
        output["lsmi"] = {
            "redundancy": lsmi_redundancy,
            **lsmi_unique,
            "synergy": lsmi_synergy,
            "pairwise": pairwise_lsmi,
            "bucket": lsmi_bucket_info["buckets"],
            "unique_modality": lsmi_bucket_info["unique_modality"],
            "bucket_gap": lsmi_bucket_info["margin"],
        }
        if len(dataset.modality_names) > 2:
            output["buckets"] = lsmi_bucket_info["buckets"]
            output["summary"] = lsmi_bucket_info["summary"]
            output["scores"] = lsmi_scores
    else:
        output["lsmi"] = None
    return output


def joint_model_layer_name(layer_representations: dict[str, dict[str, np.ndarray]]) -> str:
    keys = list(layer_representations.keys())
    return keys[-1]


def compute_representation_alignment(
    layer_representations: dict[str, dict[str, np.ndarray]],
    *,
    modality_names: list[str],
    k: int,
    num_permutations: int,
    knn_device: str,
    seed: int,
) -> dict[str, Any]:
    layer_names = list(layer_representations.keys())
    sample_mknn: dict[str, np.ndarray] = {}
    sample_cka: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for layer_index, layer_name in enumerate(layer_names):
        pair_mknn = []
        pair_cka = []
        pair_summary: dict[str, Any] = {}
        for pair_index, (left_name, right_name) in enumerate(combinations(modality_names, 2)):
            left = layer_representations[layer_name][left_name]
            right = layer_representations[layer_name][right_name]
            left_knn = compute_knn_indices(left, k=k, device=knn_device)
            right_knn = compute_knn_indices(right, k=k, device=knn_device)
            mknn = calibrated_mutual_knn(
                image_neighbors=left_knn,
                text_neighbors=right_knn,
                num_permutations=num_permutations,
                seed=seed + 101 * (layer_index + 1) + pair_index,
            )
            cka = _samplewise_rbf_cka(left, right, num_permutations=num_permutations, seed=seed + 401 * (layer_index + 1) + pair_index)
            pair_name = f"{left_name}__{right_name}"
            pair_summary[pair_name] = {
                "mknn_dataset_z": mknn["dataset_z"],
                "cka_dataset_z": cka["dataset_z"],
                "mknn_dataset_raw": mknn["dataset_raw"],
                "cka_dataset_raw": cka["dataset_raw"],
            }
            pair_mknn.append(mknn["sample_z"])
            pair_cka.append(cka["sample_z"])
        sample_mknn[layer_name] = _pairwise_average(pair_mknn)
        sample_cka[layer_name] = _pairwise_average(pair_cka)
        summaries[layer_name] = {
            "mknn_dataset_z": _pairwise_average_scalar([value["mknn_dataset_z"] for value in pair_summary.values()]),
            "cka_dataset_z": _pairwise_average_scalar([value["cka_dataset_z"] for value in pair_summary.values()]),
            "pairs": pair_summary,
        }
    return {"layer_names": layer_names, "sample_mknn": sample_mknn, "sample_cka": sample_cka, "summary": summaries}


def collect_gradient_signals(
    dataset: DatasetSpec,
    split: ExperimentSplit,
    model: JointMultimodalModel,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    layer_grads = {layer: {name: [] for name in dataset.modality_names} for layer in model.fusion_layer_names}
    partial_fused_grads = {name: [] for name in dataset.modality_names}
    model.eval()

    with torch.backends.cudnn.flags(enabled=False):
        for modalities, mask, labels in loader:
            modalities_device = _move_modalities(modalities, device)
            mask_device = None if mask is None else mask.to(device)
            labels = labels.to(device)

            model.zero_grad(set_to_none=True)
            outputs = model(modalities_device, mask_device, retain_layer_grads=True)
            _, per_sample_loss = _compute_loss(dataset.task, outputs["logits"], labels)
            per_sample_loss.sum().backward()
            for layer_name in model.fusion_layer_names:
                for modality_name in dataset.modality_names:
                    seq_grad = outputs["layer_sequences"][layer_name][modality_name].grad
                    seq_mask = outputs["layer_masks"][modality_name]
                    layer_grads[layer_name][modality_name].append(
                        _masked_mean(seq_grad, seq_mask).detach().cpu().numpy().astype(np.float32)
                    )

            for modality_name in dataset.modality_names:
                model.zero_grad(set_to_none=True)
                partial_outputs = model(
                    modalities_device,
                    mask_device,
                    active_modalities={modality_name},
                    retain_fused_grad=True,
                )
                _, partial_loss = _compute_loss(dataset.task, partial_outputs["logits"], labels)
                partial_loss.sum().backward()
                partial_fused_grads[modality_name].append(
                    partial_outputs["fused_hidden"].grad.detach().cpu().numpy().astype(np.float32)
                )

    layer_gradients = {
        layer_name: {name: np.concatenate(chunks, axis=0) for name, chunks in modalities.items()}
        for layer_name, modalities in layer_grads.items()
    }
    fused_partial = {name: np.concatenate(chunks, axis=0) for name, chunks in partial_fused_grads.items()}
    return {"layer_gradients": layer_gradients, "partial_fused_gradients": fused_partial}


def _parameter_scope_matches(name: str, scope: str, architecture: str) -> bool:
    if scope == "all":
        return True
    if scope == "representation":
        return name.startswith("fusion_head.")
    if scope == "classifier":
        return name.startswith("classifier.")
    if scope == "head":
        return name.startswith("fusion_head.") or name.startswith("classifier.")
    if scope == "shared_core":
        if architecture == "shared_head":
            return name.startswith("shared_blocks.") or name.startswith("fusion_head.")
        return name.startswith("fusion_blocks.") or name.startswith("fusion_head.")
    if scope == "shared":
        if architecture == "shared_head":
            return name.startswith("shared_blocks.") or name.startswith("fusion_head.") or name.startswith("classifier.")
        return name.startswith("fusion_blocks.") or name.startswith("fusion_head.") or name.startswith("classifier.")
    raise ValueError(f"Unsupported parameter scope: {scope}")


def _select_fisher_parameters(model: JointMultimodalModel, scope: str) -> tuple[list[str], list[nn.Parameter]]:
    names: list[str] = []
    params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _parameter_scope_matches(name, scope, model.architecture):
            names.append(name)
            params.append(param)
    if not params:
        raise ValueError(f"No trainable parameters matched scope={scope!r} for architecture={model.architecture!r}")
    return names, params


def _flatten_grad_list(grads: list[torch.Tensor | None], params: list[nn.Parameter]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for grad, param in zip(grads, params, strict=False):
        if grad is None:
            pieces.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            pieces.append(grad.detach().to(dtype=torch.float32).reshape(-1))
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def collect_empirical_fisher_signals(
    dataset: DatasetSpec,
    split: ExperimentSplit,
    model: JointMultimodalModel,
    *,
    device: str,
    batch_size: int,
    parameter_scope: str = "shared",
) -> dict[str, Any]:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    parameter_names, fisher_params = _select_fisher_parameters(model, parameter_scope)
    modality_names = list(dataset.modality_names)
    pair_names = list(combinations(modality_names, 2))
    sample_trace_by_modality: dict[str, list[float]] = {name: [] for name in modality_names}
    sample_inner_by_pair: dict[str, list[float]] = {f"{left}__{right}": [] for left, right in pair_names}
    sample_inner_pos_by_pair: dict[str, list[float]] = {f"{left}__{right}": [] for left, right in pair_names}
    sample_inner_neg_by_pair: dict[str, list[float]] = {f"{left}__{right}": [] for left, right in pair_names}
    sample_cos_by_pair: dict[str, list[float]] = {f"{left}__{right}": [] for left, right in pair_names}
    model.eval()

    with torch.backends.cudnn.flags(enabled=False):
        for modalities, mask, labels in loader:
            modalities_device = _move_modalities(modalities, device)
            mask_device = None if mask is None else mask.to(device)
            labels = labels.to(device)
            batch_size_local = labels.shape[0]
            batch_modality_grads: dict[str, list[torch.Tensor]] = {name: [] for name in modality_names}

            for sample_index in range(batch_size_local):
                modalities_single = {name: tensor[sample_index : sample_index + 1] for name, tensor in modalities_device.items()}
                mask_single = None if mask_device is None else mask_device[sample_index : sample_index + 1]
                label_single = labels[sample_index : sample_index + 1]
                for modality_name in modality_names:
                    model.zero_grad(set_to_none=True)
                    outputs = model(modalities_single, mask_single, active_modalities={modality_name})
                    loss, _ = _compute_loss(dataset.task, outputs["logits"], label_single)
                    grads = torch.autograd.grad(loss, fisher_params, allow_unused=True)
                    grad_vector = _flatten_grad_list(list(grads), fisher_params).cpu()
                    batch_modality_grads[modality_name].append(grad_vector)

            for modality_name, grad_vectors in batch_modality_grads.items():
                stacked = torch.stack(grad_vectors, dim=0)
                traces = torch.sum(stacked * stacked, dim=1)
                sample_trace_by_modality[modality_name].extend(traces.tolist())

            for left_name, right_name in pair_names:
                left_key = f"{left_name}__{right_name}"
                left_stacked = torch.stack(batch_modality_grads[left_name], dim=0)
                right_stacked = torch.stack(batch_modality_grads[right_name], dim=0)
                inner = torch.sum(left_stacked * right_stacked, dim=1)
                inner_pos = torch.clamp(inner, min=0.0)
                inner_neg = torch.clamp(-inner, min=0.0)
                left_norm = torch.linalg.norm(left_stacked, dim=1)
                right_norm = torch.linalg.norm(right_stacked, dim=1)
                denom = torch.clamp(left_norm * right_norm, min=1e-12)
                cosine = inner / denom
                sample_inner_by_pair[left_key].extend(inner.tolist())
                sample_inner_pos_by_pair[left_key].extend(inner_pos.tolist())
                sample_inner_neg_by_pair[left_key].extend(inner_neg.tolist())
                sample_cos_by_pair[left_key].extend(cosine.tolist())

    inner_arrays = {name: np.asarray(values, dtype=np.float32) for name, values in sample_inner_by_pair.items()}
    inner_pos_arrays = {name: np.asarray(values, dtype=np.float32) for name, values in sample_inner_pos_by_pair.items()}
    inner_neg_arrays = {name: np.asarray(values, dtype=np.float32) for name, values in sample_inner_neg_by_pair.items()}
    cos_arrays = {name: np.asarray(values, dtype=np.float32) for name, values in sample_cos_by_pair.items()}
    trace_arrays = {name: np.asarray(values, dtype=np.float32) for name, values in sample_trace_by_modality.items()}
    sample_pair_inner_mean = _pairwise_average([array for array in inner_arrays.values()]) if inner_arrays else np.array([], dtype=np.float32)
    sample_pair_inner_pos_mean = _pairwise_average([array for array in inner_pos_arrays.values()]) if inner_pos_arrays else np.array([], dtype=np.float32)
    sample_pair_inner_neg_mean = _pairwise_average([array for array in inner_neg_arrays.values()]) if inner_neg_arrays else np.array([], dtype=np.float32)
    sample_pair_cos_mean = _pairwise_average([array for array in cos_arrays.values()]) if cos_arrays else np.array([], dtype=np.float32)
    sample_trace_mean = _pairwise_average([array for array in trace_arrays.values()]) if trace_arrays else np.array([], dtype=np.float32)

    return {
        "parameter_scope": parameter_scope,
        "parameter_names": parameter_names,
        "num_parameters": int(sum(param.numel() for param in fisher_params)),
        "sample_trace_by_modality": trace_arrays,
        "sample_pair_inner": inner_arrays,
        "sample_pair_inner_pos": inner_pos_arrays,
        "sample_pair_inner_neg": inner_neg_arrays,
        "sample_pair_cos": cos_arrays,
        "sample_trace_mean": sample_trace_mean.astype(np.float32),
        "sample_pair_inner_mean": sample_pair_inner_mean.astype(np.float32),
        "sample_pair_inner_pos_mean": sample_pair_inner_pos_mean.astype(np.float32),
        "sample_pair_inner_neg_mean": sample_pair_inner_neg_mean.astype(np.float32),
        "sample_pair_cos_mean": sample_pair_cos_mean.astype(np.float32),
        "summary": {
            "trace_mean_by_modality": {name: float(np.mean(values)) for name, values in trace_arrays.items()},
            "pair_inner_mean": {name: float(np.mean(values)) for name, values in inner_arrays.items()},
            "pair_inner_pos_mean": {name: float(np.mean(values)) for name, values in inner_pos_arrays.items()},
            "pair_inner_neg_mean": {name: float(np.mean(values)) for name, values in inner_neg_arrays.items()},
            "pair_cos_mean": {name: float(np.mean(values)) for name, values in cos_arrays.items()},
            "sample_trace_mean": float(np.mean(sample_trace_mean)) if sample_trace_mean.size > 0 else float("nan"),
            "sample_pair_inner_mean": float(np.mean(sample_pair_inner_mean)) if sample_pair_inner_mean.size > 0 else float("nan"),
            "sample_pair_inner_pos_mean": float(np.mean(sample_pair_inner_pos_mean)) if sample_pair_inner_pos_mean.size > 0 else float("nan"),
            "sample_pair_inner_neg_mean": float(np.mean(sample_pair_inner_neg_mean)) if sample_pair_inner_neg_mean.size > 0 else float("nan"),
            "sample_pair_cos_mean": float(np.mean(sample_pair_cos_mean)) if sample_pair_cos_mean.size > 0 else float("nan"),
        },
    }


def sample_split(
    split: ExperimentSplit,
    *,
    max_samples: int | None,
    seed: int,
) -> ExperimentSplit:
    if max_samples is None or max_samples >= split.labels.shape[0]:
        return split
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(split.labels.shape[0], size=max_samples, replace=False))
    return ExperimentSplit(
        modalities={name: values[indices].copy() for name, values in split.modalities.items()},
        labels=split.labels[indices].copy(),
        mask=None if split.mask is None else split.mask[indices].copy(),
        ids=[split.ids[int(index)] for index in indices.tolist()],
    )


def cache_measurement_batches(
    split: ExperimentSplit,
    *,
    batch_size: int,
    max_batches: int,
) -> list[tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor]]:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    batches: list[tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batches.append(batch)
    if not batches:
        raise ValueError("No measurement batches were collected.")
    return batches


def _mean_loss_for_cached_batches(
    dataset: DatasetSpec,
    model: JointMultimodalModel,
    cached_batches: list[tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor]],
    *,
    device: str,
    active_modalities: set[str],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for modalities, mask, labels in cached_batches:
        modalities_device = _move_modalities(modalities, device)
        mask_device = None if mask is None else mask.to(device)
        labels_device = labels.to(device)
        outputs = model(modalities_device, mask_device, active_modalities=active_modalities)
        loss, _ = _compute_loss(dataset.task, outputs["logits"], labels_device)
        losses.append(loss)
    return torch.stack(losses, dim=0).mean()


def collect_modality_batch_gradients(
    dataset: DatasetSpec,
    model: JointMultimodalModel,
    cached_batches: list[tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor]],
    *,
    device: str,
    parameter_scope: str,
) -> dict[str, Any]:
    parameter_names, fisher_params = _select_fisher_parameters(model, parameter_scope)
    batch_gradients: dict[str, list[torch.Tensor]] = {name: [] for name in dataset.modality_names}
    model.eval()

    with torch.backends.cudnn.flags(enabled=False):
        for modality_name in dataset.modality_names:
            active_modalities = {modality_name}
            for batch in cached_batches:
                model.zero_grad(set_to_none=True)
                loss = _mean_loss_for_cached_batches(dataset, model, [batch], device=device, active_modalities=active_modalities)
                grads = torch.autograd.grad(loss, fisher_params, allow_unused=True)
                batch_gradients[modality_name].append(_flatten_grad_list(list(grads), fisher_params).cpu())

    gradient_matrices = {
        name: torch.stack(rows, dim=0).to(dtype=torch.float32) for name, rows in batch_gradients.items()
    }
    mean_gradients = {name: rows.mean(dim=0) for name, rows in gradient_matrices.items()}
    return {
        "parameter_names": parameter_names,
        "num_parameters": int(sum(param.numel() for param in fisher_params)),
        "parameters": fisher_params,
        "gradient_matrices": gradient_matrices,
        "mean_gradients": mean_gradients,
    }


def _as_numpy_matrix(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64)
    return np.asarray(values, dtype=np.float64)


def _cosine_similarity(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(left) * np.linalg.norm(right)), eps)
    return float(np.dot(left, right) / denom)


def _fisher_svd(gradient_matrix: torch.Tensor | np.ndarray) -> dict[str, Any]:
    grad_matrix = _as_numpy_matrix(gradient_matrix)
    if grad_matrix.ndim != 2:
        raise ValueError("Expected a 2D gradient matrix.")
    batch_count, param_dim = grad_matrix.shape
    if batch_count == 0:
        raise ValueError("Expected at least one gradient row.")
    scaled_rows = grad_matrix / math.sqrt(batch_count)
    _, singular_values, vh = np.linalg.svd(scaled_rows, full_matrices=False)
    eigenvalues = singular_values**2
    trace_value = float(np.mean(np.sum(grad_matrix * grad_matrix, axis=1)))
    return {
        "gradient_matrix": grad_matrix,
        "scaled_rows": scaled_rows,
        "basis": vh.T.astype(np.float64),
        "eigenvalues": eigenvalues.astype(np.float64),
        "trace": trace_value,
        "batch_count": int(batch_count),
        "param_dim": int(param_dim),
    }


def _effective_rank(eigenvalues: np.ndarray, *, energy_threshold: float, max_rank: int) -> int:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    positive = eigenvalues[eigenvalues > 1e-10]
    if positive.size == 0:
        return 0
    cumulative = np.cumsum(positive)
    cutoff = energy_threshold * float(cumulative[-1])
    rank = int(np.searchsorted(cumulative, cutoff, side="left") + 1)
    return int(max(1, min(rank, positive.size, max_rank)))


def _low_rank_logdet(scaled_rows: np.ndarray, *, param_dim: int, eps: float) -> float:
    if scaled_rows.size == 0:
        return float(param_dim * math.log(eps))
    gram = scaled_rows @ scaled_rows.T
    identity = np.eye(gram.shape[0], dtype=np.float64)
    sign, logabsdet = np.linalg.slogdet(identity + gram / eps)
    if sign <= 0:
        return float("nan")
    return float(param_dim * math.log(eps) + logabsdet)


def _fisher_quadratic_form(gradient_matrix: np.ndarray, vector: np.ndarray) -> float:
    projection = gradient_matrix @ vector.reshape(-1)
    return float(np.mean(projection * projection))


def _project_onto_basis(vector: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if basis.size == 0:
        zeros = np.zeros_like(vector)
        return zeros, vector.copy()
    coeff = basis.T @ vector
    parallel = basis @ coeff
    perpendicular = vector - parallel
    return parallel, perpendicular


def _project_rows_onto_complement(rows: np.ndarray, basis: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if basis.size == 0:
        return rows.copy()
    projection = (rows @ basis) @ basis.T
    return rows - projection


def _projected_lambda_max_low_rank(
    fisher_basis: np.ndarray,
    fisher_eigenvalues: np.ndarray,
    subspace_basis: np.ndarray,
) -> float:
    fisher_basis = np.asarray(fisher_basis, dtype=np.float64)
    fisher_eigenvalues = np.asarray(fisher_eigenvalues, dtype=np.float64)
    if fisher_basis.size == 0 or fisher_eigenvalues.size == 0:
        return 0.0
    r = min(fisher_basis.shape[1], fisher_eigenvalues.shape[0])
    basis = fisher_basis[:, :r]
    eigenvalues = np.clip(fisher_eigenvalues[:r], 0.0, None)
    if subspace_basis.size == 0:
        return float(eigenvalues[0]) if eigenvalues.size > 0 else 0.0
    overlap = subspace_basis.T @ basis
    gram = np.eye(r, dtype=np.float64) - overlap.T @ overlap
    gram = 0.5 * (gram + gram.T)
    gram = np.clip(gram, 0.0, None)
    sqrt_lambda = np.sqrt(eigenvalues)
    small = (sqrt_lambda[:, None] * gram) * sqrt_lambda[None, :]
    eigvals = np.linalg.eigvalsh(0.5 * (small + small.T))
    return float(max(float(eigvals[-1]), 0.0)) if eigvals.size > 0 else 0.0


def _random_subspace_complementarity_null(
    gradient_matrix: np.ndarray,
    *,
    trace_value: float,
    dim: int,
    rank: int,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    if num_samples <= 0:
        return {
            "num_samples": 0,
            "complementarity": [],
            "normalized_complementarity": [],
            "complementarity_mean": float("nan"),
            "complementarity_std": float("nan"),
            "complementarity_p95": float("nan"),
            "normalized_complementarity_mean": float("nan"),
            "normalized_complementarity_std": float("nan"),
            "normalized_complementarity_p95": float("nan"),
        }
    if rank <= 0:
        constant = float(trace_value)
        normalized = float(constant / max(trace_value, 1e-12))
        return {
            "num_samples": int(num_samples),
            "complementarity": [constant] * int(num_samples),
            "normalized_complementarity": [normalized] * int(num_samples),
            "complementarity_mean": constant,
            "complementarity_std": 0.0,
            "complementarity_p95": constant,
            "normalized_complementarity_mean": normalized,
            "normalized_complementarity_std": 0.0,
            "normalized_complementarity_p95": normalized,
        }

    rng = np.random.default_rng(seed)
    complementarity_values: list[float] = []
    normalized_values: list[float] = []
    for _ in range(num_samples):
        random_matrix = rng.standard_normal(size=(dim, rank))
        basis, _ = np.linalg.qr(random_matrix, mode="reduced")
        redundancy = float(np.sum((gradient_matrix @ basis) ** 2) / max(gradient_matrix.shape[0], 1))
        complementarity = float(max(trace_value - redundancy, 0.0))
        complementarity_values.append(complementarity)
        normalized_values.append(float(complementarity / max(trace_value, 1e-12)))
    complementarity_array = np.asarray(complementarity_values, dtype=np.float64)
    normalized_array = np.asarray(normalized_values, dtype=np.float64)
    return {
        "num_samples": int(num_samples),
        "complementarity": complementarity_array.astype(float).tolist(),
        "normalized_complementarity": normalized_array.astype(float).tolist(),
        "complementarity_mean": float(complementarity_array.mean()),
        "complementarity_std": float(complementarity_array.std(ddof=1)) if complementarity_array.size > 1 else 0.0,
        "complementarity_p95": float(np.quantile(complementarity_array, 0.95)),
        "normalized_complementarity_mean": float(normalized_array.mean()),
        "normalized_complementarity_std": float(normalized_array.std(ddof=1)) if normalized_array.size > 1 else 0.0,
        "normalized_complementarity_p95": float(np.quantile(normalized_array, 0.95)),
    }


def _random_subspace_orthogonal_lambda_null(
    fisher_aux: dict[str, Any],
    *,
    dim: int,
    rank: int,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    lambda_max_aux = float(fisher_aux["eigenvalues"][0]) if fisher_aux["eigenvalues"].size > 0 else 0.0
    if num_samples <= 0:
        return {
            "num_samples": 0,
            "orthogonal_lambda_max": [],
            "orthogonal_lambda_ratio": [],
            "orthogonal_lambda_max_mean": float("nan"),
            "orthogonal_lambda_max_std": float("nan"),
            "orthogonal_lambda_max_p95": float("nan"),
            "orthogonal_lambda_ratio_mean": float("nan"),
            "orthogonal_lambda_ratio_std": float("nan"),
            "orthogonal_lambda_ratio_p95": float("nan"),
        }
    if rank <= 0:
        return {
            "num_samples": int(num_samples),
            "orthogonal_lambda_max": [lambda_max_aux] * int(num_samples),
            "orthogonal_lambda_ratio": [1.0] * int(num_samples),
            "orthogonal_lambda_max_mean": lambda_max_aux,
            "orthogonal_lambda_max_std": 0.0,
            "orthogonal_lambda_max_p95": lambda_max_aux,
            "orthogonal_lambda_ratio_mean": 1.0,
            "orthogonal_lambda_ratio_std": 0.0,
            "orthogonal_lambda_ratio_p95": 1.0,
        }

    rng = np.random.default_rng(seed)
    orthogonal_lambda_max_values: list[float] = []
    orthogonal_ratio_values: list[float] = []
    fisher_basis = np.asarray(fisher_aux["basis"], dtype=np.float64)
    fisher_eigenvalues = np.asarray(fisher_aux["eigenvalues"], dtype=np.float64)
    for _ in range(num_samples):
        random_matrix = rng.standard_normal(size=(dim, rank))
        basis, _ = np.linalg.qr(random_matrix, mode="reduced")
        orthogonal_lambda_max = _projected_lambda_max_low_rank(fisher_basis, fisher_eigenvalues, basis)
        orthogonal_lambda_max_values.append(orthogonal_lambda_max)
        orthogonal_ratio_values.append(float(orthogonal_lambda_max / max(lambda_max_aux, 1e-12)))
    max_array = np.asarray(orthogonal_lambda_max_values, dtype=np.float64)
    ratio_array = np.asarray(orthogonal_ratio_values, dtype=np.float64)
    return {
        "num_samples": int(num_samples),
        "orthogonal_lambda_max": max_array.astype(float).tolist(),
        "orthogonal_lambda_ratio": ratio_array.astype(float).tolist(),
        "orthogonal_lambda_max_mean": float(max_array.mean()),
        "orthogonal_lambda_max_std": float(max_array.std(ddof=1)) if max_array.size > 1 else 0.0,
        "orthogonal_lambda_max_p95": float(np.quantile(max_array, 0.95)),
        "orthogonal_lambda_ratio_mean": float(ratio_array.mean()),
        "orthogonal_lambda_ratio_std": float(ratio_array.std(ddof=1)) if ratio_array.size > 1 else 0.0,
        "orthogonal_lambda_ratio_p95": float(np.quantile(ratio_array, 0.95)),
    }


def _hvp_for_cached_batches(
    dataset: DatasetSpec,
    model: JointMultimodalModel,
    cached_batches: list[tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor]],
    *,
    device: str,
    active_modalities: set[str],
    params: list[nn.Parameter],
    vector: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.backends.cudnn.flags(enabled=False):
        loss = _mean_loss_for_cached_batches(
            dataset,
            model,
            cached_batches,
            device=device,
            active_modalities=active_modalities,
        )
        first_grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        flat_first = torch.cat(
            [
                (grad if grad is not None else torch.zeros_like(param)).reshape(-1)
                for grad, param in zip(first_grads, params, strict=False)
            ],
            dim=0,
        ).to(dtype=torch.float32)
        dot = torch.dot(flat_first, vector)
        second_grads = torch.autograd.grad(dot, params, allow_unused=True)
    return _flatten_grad_list(list(second_grads), params).to(vector.device)


def _orthonormalize_columns(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    basis: list[torch.Tensor] = []
    for vector in vectors:
        work = vector
        for basis_vector in basis:
            work = work - torch.dot(work, basis_vector) * basis_vector
        norm = torch.linalg.norm(work)
        if torch.isfinite(norm) and float(norm.item()) > 1e-10:
            basis.append(work / norm)
    return basis


def _top_hessian_eigenspace(
    hvp_fn,
    *,
    dim: int,
    top_k: int,
    num_iters: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    if top_k <= 0:
        return {"eigenvalues": [], "basis": np.zeros((dim, 0), dtype=np.float64)}
    torch.manual_seed(seed)
    locked_basis: list[torch.Tensor] = []
    eigenvalues: list[float] = []
    for eigen_index in range(top_k):
        vector = torch.randn(dim, device=device, dtype=torch.float32)
        basis = _orthonormalize_columns([*locked_basis, vector])
        if len(basis) <= eigen_index:
            break
        vector = basis[eigen_index]
        for _ in range(num_iters):
            candidate = hvp_fn(vector)
            basis = _orthonormalize_columns([*locked_basis, candidate])
            if len(basis) <= eigen_index:
                break
            vector = basis[eigen_index]
        hv = hvp_fn(vector)
        eigenvalues.append(float(torch.dot(vector, hv).item()))
        locked_basis.append(vector.detach())
    if not locked_basis:
        return {"eigenvalues": [], "basis": np.zeros((dim, 0), dtype=np.float64)}
    basis_matrix = torch.stack([vector.cpu() for vector in locked_basis], dim=1).numpy().astype(np.float64)
    return {"eigenvalues": eigenvalues, "basis": basis_matrix}


def _hutchinson_trace(
    hvp_fn,
    *,
    dim: int,
    num_samples: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    estimates: list[float] = []
    for _ in range(num_samples):
        vector = torch.empty(dim, device=device, dtype=torch.float32).bernoulli_(0.5)
        vector = vector.mul_(2.0).sub_(1.0)
        hv = hvp_fn(vector)
        estimates.append(float(torch.dot(vector, hv).item()))
    values = np.asarray(estimates, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "samples": values.astype(float).tolist(),
    }


def measure_fisher_hessian_interactions(
    dataset: DatasetSpec,
    split: ExperimentSplit,
    model: JointMultimodalModel,
    *,
    device: str,
    train_lr: float,
    parameter_scope: str = "shared_core",
    measurement_batch_size: int = 32,
    max_measurement_samples: int | None = None,
    fisher_batches: int = 32,
    hessian_batches: int = 8,
    subspace_energy: float = 0.9,
    max_subspace_rank: int = 16,
    principal_top_k: int = 8,
    hessian_top_k: int = 3,
    hessian_power_iters: int = 12,
    hessian_trace_samples: int = 12,
    quadratic_form_samples: int = 16,
    random_subspace_samples: int = 128,
    beta: float = 1.0,
    fisher_eps: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    measurement_split = sample_split(split, max_samples=max_measurement_samples, seed=seed + 17)
    fisher_cached_batches = cache_measurement_batches(
        measurement_split,
        batch_size=measurement_batch_size,
        max_batches=fisher_batches,
    )
    hessian_cached_batches = cache_measurement_batches(
        measurement_split,
        batch_size=measurement_batch_size,
        max_batches=min(hessian_batches, len(fisher_cached_batches)),
    )
    gradient_payload = collect_modality_batch_gradients(
        dataset,
        model,
        fisher_cached_batches,
        device=device,
        parameter_scope=parameter_scope,
    )
    param_dim = int(gradient_payload["num_parameters"])
    modality_names = list(dataset.modality_names)
    fisher_cache = {
        name: _fisher_svd(matrix) for name, matrix in gradient_payload["gradient_matrices"].items()
    }
    mean_gradients = {
        name: _as_numpy_matrix(vector).reshape(-1) for name, vector in gradient_payload["mean_gradients"].items()
    }
    hessian_by_target: dict[str, Any] = {}
    directional_pairs: dict[str, Any] = {}

    for target_index, target_name in enumerate(modality_names):
        fisher_target = fisher_cache[target_name]
        target_rank = _effective_rank(
            fisher_target["eigenvalues"],
            energy_threshold=subspace_energy,
            max_rank=max_subspace_rank,
        )
        principal_rank = min(max(1, principal_top_k), fisher_target["basis"].shape[1])
        target_basis = fisher_target["basis"][:, :target_rank] if target_rank > 0 else np.zeros((param_dim, 0), dtype=np.float64)
        principal_basis = fisher_target["basis"][:, :principal_rank] if principal_rank > 0 else np.zeros((param_dim, 0), dtype=np.float64)
        weak_rank = min(principal_top_k, int(np.sum(fisher_target["eigenvalues"] > 1e-10)))
        weak_basis = (
            fisher_target["basis"][:, max(0, weak_rank - principal_top_k) : weak_rank]
            if weak_rank > 0
            else np.zeros((param_dim, 0), dtype=np.float64)
        )

        hvp_fn = lambda vector, target_name=target_name: _hvp_for_cached_batches(
            dataset,
            model,
            hessian_cached_batches,
            device=device,
            active_modalities={target_name},
            params=gradient_payload["parameters"],
            vector=vector,
        )
        target_dim = param_dim
        hessian_eigenspace = _top_hessian_eigenspace(
            hvp_fn,
            dim=target_dim,
            top_k=hessian_top_k,
            num_iters=hessian_power_iters,
            device=device,
            seed=seed + 1000 + target_index,
        )
        trace_estimate = _hutchinson_trace(
            hvp_fn,
            dim=target_dim,
            num_samples=hessian_trace_samples,
            device=device,
            seed=seed + 2000 + target_index,
        )
        fisher_overlap_rank = min(target_rank, hessian_eigenspace["basis"].shape[1])
        fisher_hessian_overlap = float("nan")
        if fisher_overlap_rank > 0:
            fisher_basis = target_basis[:, :fisher_overlap_rank]
            hessian_basis = hessian_eigenspace["basis"][:, :fisher_overlap_rank]
            overlap = fisher_basis.T @ hessian_basis
            fisher_hessian_overlap = float(np.sum(overlap * overlap) / fisher_overlap_rank)

        fisher_quadratics: list[float] = []
        hessian_quadratics: list[float] = []
        torch.manual_seed(seed + 3000 + target_index)
        for _ in range(quadratic_form_samples):
            probe = torch.empty(target_dim, device=device, dtype=torch.float32).bernoulli_(0.5)
            probe = probe.mul_(2.0).sub_(1.0)
            probe_cpu = probe.detach().cpu().numpy().astype(np.float64)
            fisher_quadratics.append(_fisher_quadratic_form(fisher_target["gradient_matrix"], probe_cpu))
            hessian_quadratics.append(float(torch.dot(probe, hvp_fn(probe)).item()))
        fisher_quadratics_array = np.asarray(fisher_quadratics, dtype=np.float64)
        hessian_quadratics_array = np.asarray(hessian_quadratics, dtype=np.float64)
        pearson = stats.pearsonr(fisher_quadratics_array, hessian_quadratics_array) if quadratic_form_samples >= 2 else None
        spearman = stats.spearmanr(fisher_quadratics_array, hessian_quadratics_array) if quadratic_form_samples >= 2 else None

        hessian_by_target[target_name] = {
            "fisher_target": {
                "trace": fisher_target["trace"],
                "lambda_max": float(fisher_target["eigenvalues"][0]) if fisher_target["eigenvalues"].size > 0 else 0.0,
                "effective_rank": int(target_rank),
                "principal_eigenvalues": fisher_target["eigenvalues"][:principal_top_k].astype(float).tolist(),
            },
            "hessian_target": {
                "lambda_max": float(hessian_eigenspace["eigenvalues"][0]) if hessian_eigenspace["eigenvalues"] else float("nan"),
                "top_eigenvalues": [float(value) for value in hessian_eigenspace["eigenvalues"]],
                "trace_hutchinson": trace_estimate,
                "fisher_hessian_subspace_overlap": fisher_hessian_overlap,
                "quadratic_form_correlation": {
                    "pearson": None if pearson is None else {"r": float(pearson.statistic), "p_value": float(pearson.pvalue)},
                    "spearman": None if spearman is None else {"rho": float(spearman.statistic), "p_value": float(spearman.pvalue)},
                },
            },
        }

        for aux_name in modality_names:
            if aux_name == target_name:
                continue
            fisher_aux = fisher_cache[aux_name]
            auxiliary_mean = mean_gradients[aux_name]
            target_mean = mean_gradients[target_name]
            g_dot = float(np.dot(target_mean, auxiliary_mean))
            g_cos = _cosine_similarity(target_mean, auxiliary_mean)
            batch_inner = np.sum(
                fisher_target["gradient_matrix"] * fisher_aux["gradient_matrix"],
                axis=1,
            )
            conflict_rate = float(np.mean(batch_inner < 0.0))
            redundancy = (
                float(np.sum((fisher_aux["gradient_matrix"] @ target_basis) ** 2) / fisher_aux["batch_count"])
                if target_basis.size > 0
                else 0.0
            )
            complementarity = float(max(fisher_aux["trace"] - redundancy, 0.0))
            orthogonal_lambda_max = _projected_lambda_max_low_rank(
                np.asarray(fisher_aux["basis"], dtype=np.float64),
                np.asarray(fisher_aux["eigenvalues"], dtype=np.float64),
                target_basis,
            )
            orthogonal_lambda_ratio = float(
                orthogonal_lambda_max / max(
                    float(fisher_aux["eigenvalues"][0]) if fisher_aux["eigenvalues"].size > 0 else 0.0,
                    1e-12,
                )
            )
            orthogonal_lambda_null = _random_subspace_orthogonal_lambda_null(
                fisher_aux,
                dim=param_dim,
                rank=int(target_rank),
                num_samples=random_subspace_samples,
                seed=seed + 5000 + 97 * target_index + 13 * modality_names.index(aux_name),
            )
            null_calibration = _random_subspace_complementarity_null(
                fisher_aux["gradient_matrix"],
                trace_value=float(fisher_aux["trace"]),
                dim=param_dim,
                rank=int(target_rank),
                num_samples=random_subspace_samples,
                seed=seed + 4000 + 97 * target_index + 13 * modality_names.index(aux_name),
            )
            normalized_complementarity = float(complementarity / max(fisher_aux["trace"], 1e-12))
            null_p95 = float(null_calibration["normalized_complementarity_p95"])
            null_support = bool(normalized_complementarity > null_p95) if math.isfinite(null_p95) else False
            null_p_value = float(
                (1.0 + np.sum(np.asarray(null_calibration["normalized_complementarity"], dtype=np.float64) >= normalized_complementarity))
                / (len(null_calibration["normalized_complementarity"]) + 1.0)
            ) if null_calibration["normalized_complementarity"] else float("nan")
            orth_lambda_p95 = float(orthogonal_lambda_null["orthogonal_lambda_ratio_p95"])
            orth_lambda_support = bool(orthogonal_lambda_ratio > orth_lambda_p95) if math.isfinite(orth_lambda_p95) else False
            orth_lambda_p_value = float(
                (1.0 + np.sum(np.asarray(orthogonal_lambda_null["orthogonal_lambda_ratio"], dtype=np.float64) >= orthogonal_lambda_ratio))
                / (len(orthogonal_lambda_null["orthogonal_lambda_ratio"]) + 1.0)
            ) if orthogonal_lambda_null["orthogonal_lambda_ratio"] else float("nan")
            principal_gains = (
                np.mean((fisher_aux["gradient_matrix"] @ principal_basis) ** 2, axis=0).astype(float).tolist()
                if principal_basis.size > 0
                else []
            )
            weak_gains = (
                np.mean((fisher_aux["gradient_matrix"] @ weak_basis) ** 2, axis=0).astype(float).tolist()
                if weak_basis.size > 0
                else []
            )
            auxiliary_parallel, auxiliary_perpendicular = _project_onto_basis(auxiliary_mean, target_basis)
            auxiliary_parallel_cos = _cosine_similarity(target_mean, auxiliary_parallel) if np.linalg.norm(auxiliary_parallel) > 1e-12 else float("nan")
            q_yy = float(torch.dot(
                torch.from_numpy(auxiliary_mean).to(device=device, dtype=torch.float32),
                hvp_fn(torch.from_numpy(auxiliary_mean).to(device=device, dtype=torch.float32)),
            ).item())
            q_yx = float(torch.dot(
                torch.from_numpy(auxiliary_mean).to(device=device, dtype=torch.float32),
                hvp_fn(torch.from_numpy(target_mean).to(device=device, dtype=torch.float32)),
            ).item())
            b_opt = (
                train_lr * beta * g_dot
                - (train_lr**2) * beta * q_yx
                - 0.5 * (train_lr**2) * (beta**2) * q_yy
            )
            directional_pairs[f"{aux_name}_to_{target_name}"] = {
                "target": target_name,
                "auxiliary": aux_name,
                "fisher": {
                    "trace_target": fisher_target["trace"],
                    "trace_auxiliary": fisher_aux["trace"],
                    "lambda_max_target": float(fisher_target["eigenvalues"][0]) if fisher_target["eigenvalues"].size > 0 else 0.0,
                    "lambda_max_auxiliary": float(fisher_aux["eigenvalues"][0]) if fisher_aux["eigenvalues"].size > 0 else 0.0,
                    "logdet_target": _low_rank_logdet(fisher_target["scaled_rows"], param_dim=param_dim, eps=fisher_eps),
                    "logdet_target_plus_auxiliary": _low_rank_logdet(
                        np.concatenate([fisher_target["scaled_rows"], fisher_aux["scaled_rows"]], axis=0),
                        param_dim=param_dim,
                        eps=fisher_eps,
                    ),
                    "logdet_gain_aux_to_target": _low_rank_logdet(
                        np.concatenate([fisher_target["scaled_rows"], fisher_aux["scaled_rows"]], axis=0),
                        param_dim=param_dim,
                        eps=fisher_eps,
                    )
                    - _low_rank_logdet(fisher_target["scaled_rows"], param_dim=param_dim, eps=fisher_eps),
                    "redundancy": redundancy,
                    "complementarity": complementarity,
                    "normalized_redundancy": float(redundancy / max(fisher_aux["trace"], 1e-12)),
                    "normalized_complementarity": normalized_complementarity,
                    "orthogonal_lambda_max": orthogonal_lambda_max,
                    "orthogonal_lambda_ratio": orthogonal_lambda_ratio,
                    "orthogonal_lambda_null": {
                        **orthogonal_lambda_null,
                        "supported_vs_random_subspace_95": orth_lambda_support,
                        "p_value_greater_equal_random": orth_lambda_p_value,
                    },
                    "null_calibration": {
                        **null_calibration,
                        "supported_vs_random_subspace_95": null_support,
                        "p_value_greater_equal_random": null_p_value,
                    },
                    "principal_direction_gains": principal_gains,
                    "weak_nonzero_direction_gains": weak_gains,
                },
                "gradient": {
                    "mean_dot": g_dot,
                    "mean_cosine": g_cos,
                    "batch_conflict_rate": conflict_rate,
                    "auxiliary_parallel_cosine": auxiliary_parallel_cos,
                    "auxiliary_parallel_norm": float(np.linalg.norm(auxiliary_parallel)),
                    "auxiliary_perpendicular_norm": float(np.linalg.norm(auxiliary_perpendicular)),
                },
                "hessian": {
                    "q_yy_given_target": q_yy,
                    "q_yx_given_target": q_yx,
                    "lambda_max_target_hessian": hessian_by_target[target_name]["hessian_target"]["lambda_max"],
                    "trace_target_hessian": hessian_by_target[target_name]["hessian_target"]["trace_hutchinson"]["mean"],
                    "fisher_hessian_subspace_overlap": hessian_by_target[target_name]["hessian_target"]["fisher_hessian_subspace_overlap"],
                    "quadratic_form_correlation": hessian_by_target[target_name]["hessian_target"]["quadratic_form_correlation"],
                },
                "local_transfer": {
                    "eta": float(train_lr),
                    "beta": float(beta),
                    "benefit": float(b_opt),
                    "first_order_term": float(train_lr * beta * g_dot),
                    "cross_curvature_term": float(-(train_lr**2) * beta * q_yx),
                    "self_curvature_term": float(-0.5 * (train_lr**2) * (beta**2) * q_yy),
                    "decision": "positive_transfer" if b_opt > 0.0 else "negative_transfer",
                },
            }

    return {
        "parameter_scope": parameter_scope,
        "num_parameters": param_dim,
        "measurement": {
            "num_samples": int(measurement_split.labels.shape[0]),
            "measurement_batch_size": int(measurement_batch_size),
            "fisher_batches": int(len(fisher_cached_batches)),
            "hessian_batches": int(len(hessian_cached_batches)),
            "sample_ids": measurement_split.ids,
        },
        "targets": hessian_by_target,
        "directional_pairs": directional_pairs,
    }


def compute_optimization_alignment(
    gradient_signals: dict[str, Any],
    layer_representations: dict[str, dict[str, np.ndarray]],
    *,
    modality_names: list[str],
    top_k: int,
    neighborhood_k: int,
) -> dict[str, Any]:
    pair_names = list(combinations(modality_names, 2))
    sample_layer_gradcos: dict[str, np.ndarray] = {}
    sample_layer_curvagree: dict[str, np.ndarray] = {}

    def _samplewise_local_subspace_overlap(
        left_grads: np.ndarray,
        right_grads: np.ndarray,
        anchor_features: np.ndarray,
    ) -> np.ndarray:
        left_grads = np.asarray(left_grads, dtype=np.float32)
        right_grads = np.asarray(right_grads, dtype=np.float32)
        anchor_features = np.asarray(anchor_features, dtype=np.float32)
        if anchor_features.shape[0] < 2:
            return np.full(anchor_features.shape[0], np.nan, dtype=np.float32)

        k_eff = min(max(neighborhood_k, top_k + 2), anchor_features.shape[0])
        model = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        model.fit(anchor_features)
        indices = model.kneighbors(anchor_features, return_distance=False)
        overlaps = np.empty(anchor_features.shape[0], dtype=np.float32)
        for sample_index, neighbor_idx in enumerate(indices):
            overlaps[sample_index] = _subspace_overlap(
                left_grads[neighbor_idx],
                right_grads[neighbor_idx],
                top_k=top_k,
            )
        return overlaps

    for layer_name, grads_by_modality in gradient_signals["layer_gradients"].items():
        pair_scores = []
        pair_curvagree = []
        for left_name, right_name in pair_names:
            pair_scores.append(_cosine_rows(grads_by_modality[left_name], grads_by_modality[right_name]))
            anchor_features = np.concatenate(
                [
                    np.asarray(layer_representations[layer_name][left_name], dtype=np.float32),
                    np.asarray(layer_representations[layer_name][right_name], dtype=np.float32),
                ],
                axis=1,
            )
            pair_curvagree.append(
                _samplewise_local_subspace_overlap(
                    grads_by_modality[left_name],
                    grads_by_modality[right_name],
                    anchor_features,
                )
            )
        sample_layer_gradcos[layer_name] = _pairwise_average(pair_scores)
        sample_layer_curvagree[layer_name] = _pairwise_average(pair_curvagree)

    partial_scores = []
    for left_name, right_name in pair_names:
        partial_scores.append(
            _cosine_rows(
                gradient_signals["partial_fused_gradients"][left_name],
                gradient_signals["partial_fused_gradients"][right_name],
            )
        )
    sample_gradcos = _pairwise_average(partial_scores)
    return {
        "sample_layer_gradcos": sample_layer_gradcos,
        "sample_layer_curvagree": sample_layer_curvagree,
        "sample_gradcos": sample_gradcos,
    }


def _bucket_trajectory(values_by_layer: dict[str, np.ndarray], buckets: np.ndarray) -> dict[str, dict[str, Any]]:
    layer_names = list(values_by_layer.keys())
    summary: dict[str, dict[str, Any]] = {}
    for bucket in BUCKETS:
        mask = buckets == bucket
        count = int(mask.sum())
        if count == 0:
            summary[bucket] = {"count": 0, "mean": [float("nan")] * len(layer_names), "se": [float("nan")] * len(layer_names)}
            continue
        matrix = np.stack([values_by_layer[layer][mask] for layer in layer_names], axis=1)
        mean = matrix.mean(axis=0)
        se = matrix.std(axis=0, ddof=1) / math.sqrt(count) if count > 1 else np.zeros(matrix.shape[1], dtype=np.float32)
        summary[bucket] = {"count": count, "mean": mean.astype(float).tolist(), "se": se.astype(float).tolist()}
    return summary


def _bootstrap_bucket_correlations(
    rep_scores: np.ndarray,
    opt_scores: np.ndarray,
    buckets: np.ndarray,
    *,
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    distributions: dict[str, np.ndarray] = {}
    for bucket in BUCKETS:
        indices = np.flatnonzero(buckets == bucket)
        if indices.size < 3:
            distributions[bucket] = np.array([], dtype=np.float32)
            continue
        draws = np.empty(num_bootstrap, dtype=np.float32)
        for draw_index in range(num_bootstrap):
            sample = rng.choice(indices, size=indices.size, replace=True)
            draws[draw_index] = float(stats.spearmanr(rep_scores[sample], opt_scores[sample]).statistic)
        distributions[bucket] = draws
    valid = {bucket: values for bucket, values in distributions.items() if values.size > 0}
    if len(valid) < 2:
        return {"bootstrap_rho": {bucket: values.astype(float).tolist() for bucket, values in distributions.items()}, "kruskal_wallis": None, "pairwise_mannwhitney": {}}
    ordered = [values for _, values in valid.items()]
    kruskal = stats.kruskal(*ordered)
    pairwise: dict[str, dict[str, float]] = {}
    keys = list(valid.keys())
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            test = stats.mannwhitneyu(valid[left], valid[right], alternative="two-sided")
            pairwise[f"{left}_vs_{right}"] = {"u_statistic": float(test.statistic), "p_value": float(test.pvalue)}
    return {
        "bootstrap_rho": {bucket: values.astype(float).tolist() for bucket, values in distributions.items()},
        "kruskal_wallis": {"h_statistic": float(kruskal.statistic), "p_value": float(kruskal.pvalue)},
        "pairwise_mannwhitney": pairwise,
    }


def analyze_decoupling(
    dataset: DatasetSpec,
    split: ExperimentSplit,
    interactions: dict[str, Any],
    rep_alignment: dict[str, Any],
    opt_alignment: dict[str, Any],
    joint_probs: np.ndarray,
    unimodal_probs: dict[str, np.ndarray],
    *,
    analysis_layer: str,
    pid_source: str,
    num_pid_bins: int,
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    buckets = interactions["buckets"]
    valid_mask = np.isin(buckets, list(BUCKETS))
    rep_cka = rep_alignment["sample_cka"][analysis_layer]
    rep_mknn = rep_alignment["sample_mknn"][analysis_layer]
    gradcos = opt_alignment["sample_gradcos"]
    curvagree = np.asarray(opt_alignment["sample_layer_curvagree"][analysis_layer], dtype=np.float32)
    correctness = _correctness_from_probs(dataset.task, joint_probs, split.labels)
    joint_quality = _sample_quality_from_probs(dataset.task, joint_probs, split.labels)
    unimodal_quality = {
        name: _sample_quality_from_probs(dataset.task, unimodal_probs[name], split.labels) for name in dataset.modality_names
    }
    unimodal_correctness = np.stack(
        [_correctness_from_probs(dataset.task, unimodal_probs[name], split.labels) for name in dataset.modality_names],
        axis=1,
    )
    unimodal_conf = np.stack(
        [_confidence_for_targets(dataset.task, unimodal_probs[name], split.labels) for name in dataset.modality_names],
        axis=1,
    )
    joint_gain = (joint_quality["confidence"] - unimodal_conf.max(axis=1)).astype(np.float32)
    unimodal_negative_loss = np.stack([unimodal_quality[name]["negative_loss"] for name in dataset.modality_names], axis=1)
    negative_loss_advantage = (joint_quality["negative_loss"] - unimodal_negative_loss.max(axis=1)).astype(np.float32)
    correctness_advantage = (correctness - unimodal_correctness.max(axis=1)).astype(np.float32)
    beats_best_unimodal = (correctness_advantage > 0.0).astype(np.int64)
    confidence_pid_redundancy = np.asarray(interactions["scores"]["redundancy"], dtype=np.float32)
    confidence_pid_synergy = np.asarray(interactions["scores"]["synergy"], dtype=np.float32)
    confidence_unique_matrix = np.stack(
        [np.asarray(values, dtype=np.float32) for values in interactions["scores"]["uniqueness"].values()],
        axis=1,
    )
    confidence_pid_best_unique = np.max(confidence_unique_matrix, axis=1).astype(np.float32)
    pid_source_used = "confidence"
    pid_feature_dict = {
        "redundancy": confidence_pid_redundancy,
        "synergy": confidence_pid_synergy,
        "best_unique": confidence_pid_best_unique,
    }
    pid_feature_matrix = np.stack(
        [confidence_pid_redundancy, confidence_pid_synergy, confidence_pid_best_unique],
        axis=1,
    ).astype(np.float32)
    pid_feature_names = ["redundancy", "synergy", "best_unique"]
    pid_synergy_score = confidence_pid_synergy
    if pid_source in {"auto", "lsmi"} and interactions["lsmi"] is not None:
        lsmi = interactions["lsmi"]
        lsmi_redundancy = np.asarray(lsmi["redundancy"], dtype=np.float32)
        lsmi_synergy = np.asarray(lsmi["synergy"], dtype=np.float32)
        if "unique_left" in lsmi and "unique_right" in lsmi:
            lsmi_unique_left = np.asarray(lsmi["unique_left"], dtype=np.float32)
            lsmi_unique_right = np.asarray(lsmi["unique_right"], dtype=np.float32)
            pid_feature_dict = {
                "redundancy": lsmi_redundancy,
                "unique_left": lsmi_unique_left,
                "unique_right": lsmi_unique_right,
                "synergy": lsmi_synergy,
            }
            pid_feature_names = ["redundancy", "unique_left", "unique_right", "synergy"]
            pid_feature_matrix = np.stack(
                [lsmi_redundancy, lsmi_unique_left, lsmi_unique_right, lsmi_synergy],
                axis=1,
            ).astype(np.float32)
            pid_source_used = "lsmi"
        else:
            unique_feature_names = sorted(
                [name for name in lsmi.keys() if name.startswith("unique_") and name != "unique_modality"]
            )
            pid_feature_dict = {"redundancy": lsmi_redundancy}
            for name in unique_feature_names:
                pid_feature_dict[name] = np.asarray(lsmi[name], dtype=np.float32)
            pid_feature_dict["synergy"] = lsmi_synergy
            pid_feature_names = ["redundancy", *unique_feature_names, "synergy"]
            pid_feature_matrix = np.stack([pid_feature_dict[name] for name in pid_feature_names], axis=1).astype(np.float32)
            pid_source_used = "lsmi_pairwise"
        pid_synergy_score = lsmi_synergy
    fisher_score = (0.5 * (_zscore(gradcos) + _zscore(curvagree))).astype(np.float32)
    valid_regression_mask = valid_mask & np.isfinite(rep_cka) & np.isfinite(rep_mknn) & np.isfinite(gradcos) & np.isfinite(curvagree)

    rep_threshold = float(np.nanmedian(rep_cka[valid_mask]))
    opt_threshold = float(np.nanmedian(gradcos[valid_mask]))
    rep_high = rep_cka >= rep_threshold
    opt_high = gradcos >= opt_threshold
    quadrants = np.full(buckets.shape[0], "NA", dtype=object)
    quadrants[rep_high & opt_high] = "Q1"
    quadrants[~rep_high & opt_high] = "Q2"
    quadrants[rep_high & ~opt_high] = "Q3"
    quadrants[~rep_high & ~opt_high] = "Q4"

    contingency: dict[str, Any] = {}
    for quadrant in ("Q1", "Q2", "Q3", "Q4"):
        mask = quadrants == quadrant
        contingency[quadrant] = {
            "count": int(mask.sum()),
            "accuracy": float(correctness[mask].mean()) if int(mask.sum()) > 0 else float("nan"),
        }

    q2_mask = quadrants == "Q2"
    q3_mask = quadrants == "Q3"
    fisher_test = None
    if int(q2_mask.sum()) > 0 and int(q3_mask.sum()) > 0:
        table = np.array(
            [
                [int(correctness[q2_mask].sum()), int(q2_mask.sum() - correctness[q2_mask].sum())],
                [int(correctness[q3_mask].sum()), int(q3_mask.sum() - correctness[q3_mask].sum())],
            ],
            dtype=np.int64,
        )
        odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")
        fisher_test = {"table": table.tolist(), "odds_ratio": float(odds_ratio), "p_value": float(p_value)}

    model_a = _oof_auc(np.stack([rep_cka[valid_mask], rep_mknn[valid_mask]], axis=1), correctness[valid_mask], seed=seed + 1)
    model_b = _oof_auc(np.stack([gradcos[valid_mask], curvagree[valid_mask]], axis=1), correctness[valid_mask], seed=seed + 2)
    model_c = _oof_auc(
        np.stack([rep_cka[valid_mask], rep_mknn[valid_mask], gradcos[valid_mask], curvagree[valid_mask]], axis=1),
        correctness[valid_mask],
        seed=seed + 3,
    )
    auc_tests = {
        "B_minus_A": _bootstrap_auc_difference(
            correctness[valid_mask], model_a["probs"], model_b["probs"], num_bootstrap=num_bootstrap, seed=seed + 11
        ),
        "C_minus_B": _bootstrap_auc_difference(
            correctness[valid_mask], model_b["probs"], model_c["probs"], num_bootstrap=num_bootstrap, seed=seed + 17
        ),
    }

    overall_spearman = stats.spearmanr(rep_cka[valid_mask], gradcos[valid_mask])
    per_bucket = {}
    for bucket in BUCKETS:
        mask = buckets == bucket
        if int(mask.sum()) < 3:
            per_bucket[bucket] = None
            continue
        corr = stats.spearmanr(rep_cka[mask], gradcos[mask])
        per_bucket[bucket] = {"rho": float(corr.statistic), "p_value": float(corr.pvalue), "count": int(mask.sum())}

    bootstrap_corr = _bootstrap_bucket_correlations(rep_cka[valid_mask], gradcos[valid_mask], buckets[valid_mask], num_bootstrap=max(100, num_bootstrap // 2), seed=seed + 29)
    continuous_targets = {
        "confidence": joint_quality["confidence"][valid_regression_mask],
        "negative_loss": joint_quality["negative_loss"][valid_regression_mask],
        "joint_gain": joint_gain[valid_regression_mask],
    }
    rep_features = np.stack([rep_cka[valid_regression_mask], rep_mknn[valid_regression_mask]], axis=1)
    opt_features = np.stack([gradcos[valid_regression_mask], curvagree[valid_regression_mask]], axis=1)
    both_features = np.concatenate([rep_features, opt_features], axis=1)
    continuous_analysis: dict[str, Any] = {}
    for offset, (target_name, target_values) in enumerate(continuous_targets.items(), start=1):
        model_a_reg = _oof_regression(rep_features, target_values, seed=seed + 100 * offset + 1)
        model_b_reg = _oof_regression(opt_features, target_values, seed=seed + 100 * offset + 2)
        model_c_reg = _oof_regression(both_features, target_values, seed=seed + 100 * offset + 3)
        continuous_analysis[target_name] = {
            "models": {
                "A_rep_only": {
                    "spearman": model_a_reg["spearman"],
                    "pearson": model_a_reg["pearson"],
                    "r2": model_a_reg["r2"],
                },
                "B_opt_only": {
                    "spearman": model_b_reg["spearman"],
                    "pearson": model_b_reg["pearson"],
                    "r2": model_b_reg["r2"],
                },
                "C_both": {
                    "spearman": model_c_reg["spearman"],
                    "pearson": model_c_reg["pearson"],
                    "r2": model_c_reg["r2"],
                },
            },
            "bootstrap_tests": {
                "B_minus_A_spearman": _bootstrap_regression_difference(
                    target_values,
                    model_a_reg["preds"],
                    model_b_reg["preds"],
                    metric="spearman",
                    num_bootstrap=num_bootstrap,
                    seed=seed + 100 * offset + 11,
                ),
                "B_minus_A_r2": _bootstrap_regression_difference(
                    target_values,
                    model_a_reg["preds"],
                    model_b_reg["preds"],
                    metric="r2",
                    num_bootstrap=num_bootstrap,
                    seed=seed + 100 * offset + 21,
                ),
                "C_minus_B_spearman": _bootstrap_regression_difference(
                    target_values,
                    model_b_reg["preds"],
                    model_c_reg["preds"],
                    metric="spearman",
                    num_bootstrap=num_bootstrap,
                    seed=seed + 100 * offset + 31,
                ),
                "C_minus_B_r2": _bootstrap_regression_difference(
                    target_values,
                    model_b_reg["preds"],
                    model_c_reg["preds"],
                    metric="r2",
                    num_bootstrap=num_bootstrap,
                    seed=seed + 100 * offset + 41,
                ),
            },
        }
    pid_features = pid_feature_matrix[valid_regression_mask]
    fisher_features = np.stack([gradcos[valid_regression_mask], curvagree[valid_regression_mask]], axis=1)
    conditional_predictive_power = compute_conditional_predictive_power(
        pid_features=pid_features,
        fisher_features=fisher_features,
        correctness=correctness[valid_regression_mask],
        negative_loss=joint_quality["negative_loss"][valid_regression_mask],
        negative_loss_advantage=negative_loss_advantage[valid_regression_mask],
        correctness_advantage=correctness_advantage[valid_regression_mask],
        beats_best_unimodal=beats_best_unimodal[valid_regression_mask],
        seed=seed + 5000,
        num_bootstrap=num_bootstrap,
    )
    matched_bin_analysis = compute_matched_bin_analysis(
        pid_features=pid_feature_matrix[valid_regression_mask],
        fisher_score=fisher_score[valid_regression_mask],
        negative_loss_advantage=negative_loss_advantage[valid_regression_mask],
        correctness=correctness[valid_regression_mask],
        beats_best_unimodal=beats_best_unimodal[valid_regression_mask],
        num_bins=num_pid_bins,
        seed=seed + 6000,
    )
    return {
        "analysis_layer": analysis_layer,
        "sample_ids": [str(item) for item in split.ids],
        "correctness": correctness.astype(int).tolist(),
        "rep_threshold": rep_threshold,
        "opt_threshold": opt_threshold,
        "quadrants": quadrants.tolist(),
        "contingency": contingency,
        "q2_vs_q3_fisher_exact": fisher_test,
        "overall_spearman": {"rho": float(overall_spearman.statistic), "p_value": float(overall_spearman.pvalue)},
        "per_bucket_spearman": per_bucket,
        "bootstrap_bucket_correlation_tests": bootstrap_corr,
        "auc_models": {
            "A_rep_only": {"auc": model_a["auc"]},
            "B_opt_only": {"auc": model_b["auc"]},
            "C_both": {"auc": model_c["auc"]},
        },
        "auc_bootstrap_tests": auc_tests,
        "features": {
            "confidence_pid_redundancy": confidence_pid_redundancy.astype(float).tolist(),
            "confidence_pid_synergy": confidence_pid_synergy.astype(float).tolist(),
            "confidence_pid_best_unique": confidence_pid_best_unique.astype(float).tolist(),
            "pid_feature_names": pid_feature_names,
            "pid_source_used": pid_source_used,
            "pid_feature_values": {name: values.astype(float).tolist() for name, values in pid_feature_dict.items()},
            "rep_cka": rep_cka.astype(float).tolist(),
            "rep_mknn": rep_mknn.astype(float).tolist(),
            "gradcos": gradcos.astype(float).tolist(),
            "curvagree": curvagree.astype(float).tolist(),
            "fisher_score": fisher_score.astype(float).tolist(),
            "confidence": joint_quality["confidence"].astype(float).tolist(),
            "negative_loss": joint_quality["negative_loss"].astype(float).tolist(),
            "joint_gain": joint_gain.astype(float).tolist(),
            "negative_loss_advantage": negative_loss_advantage.astype(float).tolist(),
            "correctness_advantage": correctness_advantage.astype(float).tolist(),
            "beats_best_unimodal": beats_best_unimodal.astype(int).tolist(),
        },
        "pid_source_used": pid_source_used,
        "valid_mask": valid_mask.astype(bool).tolist(),
        "valid_regression_mask": valid_regression_mask.astype(bool).tolist(),
        "oof_probs": {
            "A_rep_only": model_a["probs"].astype(float).tolist(),
            "B_opt_only": model_b["probs"].astype(float).tolist(),
            "C_both": model_c["probs"].astype(float).tolist(),
        },
        "continuous_outcome_analysis": continuous_analysis,
        "conditional_predictive_power": conditional_predictive_power,
        "matched_bin_analysis": matched_bin_analysis,
    }


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _serialize_nested_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return value.astype(float).tolist()
        return value.tolist()
    if isinstance(value, dict):
        return {key: _serialize_nested_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_nested_arrays(item) for item in value]
    return value


def plot_scatter(rep_scores: np.ndarray, opt_scores: np.ndarray, buckets: np.ndarray, output_path: str | Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = {"RED": "#2b8cbe", "SYN": "#d95f0e", "UNQ": "#238b45", "MIXED": "#8c8c8c"}
    for bucket in ("RED", "SYN", "UNQ", "MIXED"):
        mask = buckets == bucket
        if not np.any(mask):
            continue
        ax.scatter(rep_scores[mask], opt_scores[mask], s=12, alpha=0.7, color=colors[bucket], label=f"{bucket} (n={int(mask.sum())})")
    ax.set_xlabel("Representational alignment (CKA z)")
    ax.set_ylabel("Optimization alignment (GradCos)")
    ax.set_title("Rep. alignment vs opt. alignment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_trajectories(
    layer_names: list[str],
    rep_summary: dict[str, dict[str, Any]],
    grad_summary: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    x = np.arange(len(layer_names))
    colors = {"RED": "#2b8cbe", "SYN": "#d95f0e", "UNQ": "#238b45"}
    for axis, summary, title, ylabel in (
        (axes[0], rep_summary, "CKA Trajectory", "CKA z"),
        (axes[1], grad_summary, "GradCos Trajectory", "GradCos"),
    ):
        for bucket in BUCKETS:
            if summary[bucket]["count"] == 0:
                continue
            mean = np.array(summary[bucket]["mean"], dtype=np.float32)
            se = np.array(summary[bucket]["se"], dtype=np.float32)
            axis.plot(x, mean, marker="o", color=colors[bucket], label=f"{bucket} (n={summary[bucket]['count']})")
            axis.fill_between(x, mean - se, mean + se, color=colors[bucket], alpha=0.15)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(layer_names)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_quadrant_heatmap(contingency: dict[str, Any], output_path: str | Path) -> None:
    values = np.array(
        [
            [contingency["Q1"]["accuracy"], contingency["Q2"]["accuracy"]],
            [contingency["Q3"]["accuracy"], contingency["Q4"]["accuracy"]],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [f"Q1\nn={contingency['Q1']['count']}", f"Q2\nn={contingency['Q2']['count']}"],
            [f"Q3\nn={contingency['Q3']['count']}", f"Q4\nn={contingency['Q4']['count']}"],
        ]
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    image = ax.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{labels[row, col]}\nacc={values[row, col]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_xticks([0, 1], ["High Rep", "Low Rep"])
    ax.set_yticks([0, 1], ["High Opt", "Low Opt"])
    ax.set_title("Quadrant accuracy")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(targets: np.ndarray, model_probs: dict[str, np.ndarray], output_path: str | Path) -> None:
    if np.unique(targets).size < 2:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    colors = {"A_rep_only": "#4daf4a", "B_opt_only": "#e41a1c", "C_both": "#377eb8"}
    for name, probs in model_probs.items():
        if np.isnan(probs).all():
            continue
        fpr, tpr, _ = roc_curve(targets, probs)
        ax.plot(fpr, tpr, linewidth=2.0, color=colors[name], label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Correctness prediction ROC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_curvagree(layer_names: list[str], curvagree_summary: dict[str, dict[str, Any]], output_path: str | Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = {"RED": "#2b8cbe", "SYN": "#d95f0e", "UNQ": "#238b45"}
    x = np.arange(len(layer_names))
    for bucket in BUCKETS:
        if curvagree_summary[bucket]["count"] == 0:
            continue
        values = np.array(curvagree_summary[bucket]["mean"], dtype=np.float32)
        se = np.array(curvagree_summary[bucket]["se"], dtype=np.float32)
        ax.plot(x, values, marker="o", linewidth=2.0, color=colors[bucket], label=bucket)
        ax.fill_between(x, values - se, values + se, color=colors[bucket], alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("CurvAgree proxy")
    ax.set_title("Gradient-covariance subspace overlap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_matched_bin_effects(matched_analysis: dict[str, Any], output_path: str | Path) -> None:
    per_bin = matched_analysis["per_bin"]
    if not per_bin:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.arange(len(per_bin))
    negative_loss_advantage_delta = np.array([item["negative_loss_advantage_delta"] for item in per_bin], dtype=np.float32)
    correctness_delta = np.array([item["correctness_delta"] for item in per_bin], dtype=np.float32)
    beats_best_unimodal_delta = np.array([item["beats_best_unimodal_delta"] for item in per_bin], dtype=np.float32)

    axes[0].bar(x, negative_loss_advantage_delta, color="#d95f0e", alpha=0.85)
    axes[0].axhline(0.0, color="#4d4d4d", linewidth=1.0)
    axes[0].set_title("High-Fisher minus Low-Fisher")
    axes[0].set_ylabel("Negative-loss advantage delta")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(item["cluster_index"]) for item in per_bin])
    axes[0].set_xlabel("PID-profile cluster")

    axes[1].bar(x - 0.18, correctness_delta, width=0.36, color="#2b8cbe", label="Correctness delta")
    axes[1].bar(x + 0.18, beats_best_unimodal_delta, width=0.36, color="#238b45", label="Beats-best-unimodal delta")
    axes[1].axhline(0.0, color="#4d4d4d", linewidth=1.0)
    axes[1].set_title("Matched-bin outcome deltas")
    axes[1].set_ylabel("Rate delta")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(item["cluster_index"]) for item in per_bin])
    axes[1].set_xlabel("PID-profile cluster")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(_ensure_parent(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    *,
    dataset_name: str,
    mosi_path: str | Path,
    imdb_path: str | Path,
    humor_path: str | Path,
    sarcasm_path: str | Path,
    analysis_split: str,
    max_seq_len: int,
    max_samples: int | None,
    train_max_samples: int | None,
    valid_max_samples: int | None,
    test_max_samples: int | None,
    imdb_text_dropout_prob: float,
    device: str,
    knn_device: str,
    encoder_hidden_dim: int,
    fusion_dim: int,
    fusion_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    k: int,
    num_alignment_permutations: int,
    num_bootstrap: int,
    top_k_subspace: int,
    pid_source: str,
    num_pid_bins: int,
    bucket_margin: float,
    output_json: str | Path,
    figure_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    if dataset_name == "mosi":
        dataset = load_mosi(mosi_path, max_seq_len=max_seq_len)
    elif dataset_name == "imdb":
        dataset = load_imdb(
            imdb_path,
            max_samples=max_samples,
            train_max_samples=train_max_samples,
            valid_max_samples=valid_max_samples,
            test_max_samples=test_max_samples,
            text_dropout_prob=imdb_text_dropout_prob,
            text_dropout_seed=seed + 17,
        )
    elif dataset_name == "humor":
        dataset = load_affect(humor_path, dataset_name="humor", max_seq_len=max_seq_len)
    elif dataset_name == "sarcasm":
        dataset = load_affect(sarcasm_path, dataset_name="sarcasm", max_seq_len=max_seq_len)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    split = {"train": dataset.train, "valid": dataset.valid, "test": dataset.test}[analysis_split]
    joint_model, joint_train = train_joint_model(
        dataset,
        device=device,
        encoder_hidden_dim=encoder_hidden_dim,
        fusion_dim=fusion_dim,
        fusion_layers=fusion_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        seed=seed,
    )
    unimodal_models, unimodal_train = train_unimodal_models(
        dataset,
        device=device,
        encoder_hidden_dim=encoder_hidden_dim,
        fusion_dim=fusion_dim,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        seed=seed + 1000,
    )
    evaluation = evaluate_models(dataset, split, joint_model, unimodal_models, device=device, batch_size=eval_batch_size)
    interactions = compute_interaction_buckets(
        dataset,
        split,
        evaluation["joint_probs"],
        evaluation["unimodal_probs"],
        evaluation["layer_representations"],
        device=device,
        bucket_margin=bucket_margin,
        seed=seed + 2000,
    )
    rep_alignment = compute_representation_alignment(
        evaluation["layer_representations"],
        modality_names=dataset.modality_names,
        k=k,
        num_permutations=num_alignment_permutations,
        knn_device=knn_device,
        seed=seed + 3000,
    )
    gradients = collect_gradient_signals(dataset, split, joint_model, device=device, batch_size=eval_batch_size)
    opt_alignment = compute_optimization_alignment(
        gradients,
        evaluation["layer_representations"],
        modality_names=dataset.modality_names,
        top_k=top_k_subspace,
        neighborhood_k=max(2 * top_k_subspace, 25),
    )

    analysis_layer = rep_alignment["layer_names"][-1]
    analysis = analyze_decoupling(
        dataset,
        split,
        interactions,
        rep_alignment,
        opt_alignment,
        evaluation["joint_probs"],
        evaluation["unimodal_probs"],
        analysis_layer=analysis_layer,
        pid_source=pid_source,
        num_pid_bins=num_pid_bins,
        num_bootstrap=num_bootstrap,
        seed=seed + 4000,
    )

    rep_summary = _bucket_trajectory(rep_alignment["sample_cka"], interactions["buckets"])
    grad_summary = _bucket_trajectory(opt_alignment["sample_layer_gradcos"], interactions["buckets"])
    curvagree_summary = _bucket_trajectory(opt_alignment["sample_layer_curvagree"], interactions["buckets"])
    figure_dir = Path(figure_dir)
    figures = {
        "scatter": str(figure_dir / f"{dataset_name}_figure1_scatter.png"),
        "trajectories": str(figure_dir / f"{dataset_name}_figure2_trajectories.png"),
        "quadrants": str(figure_dir / f"{dataset_name}_figure3_quadrants.png"),
        "roc": str(figure_dir / f"{dataset_name}_figure4_roc.png"),
        "curvagree": str(figure_dir / f"{dataset_name}_figure5_curvagree.png"),
        "matched_bins": str(figure_dir / f"{dataset_name}_figure6_matched_bins.png"),
    }
    plot_scatter(
        rep_alignment["sample_cka"][analysis_layer],
        opt_alignment["sample_gradcos"],
        interactions["buckets"],
        figures["scatter"],
    )
    plot_alignment_trajectories(rep_alignment["layer_names"], rep_summary, grad_summary, figures["trajectories"])
    plot_quadrant_heatmap(analysis["contingency"], figures["quadrants"])
    valid_mask = np.asarray(analysis["valid_mask"], dtype=bool)
    plot_roc_curves(
        np.asarray(analysis["correctness"], dtype=np.int64)[valid_mask],
        {
            name: np.asarray(values, dtype=np.float32)
            for name, values in analysis["oof_probs"].items()
        },
        figures["roc"],
    )
    plot_curvagree(rep_alignment["layer_names"], curvagree_summary, figures["curvagree"])
    plot_matched_bin_effects(analysis["matched_bin_analysis"], figures["matched_bins"])

    results = {
        "dataset": dataset_name,
        "analysis_split": analysis_split,
        "config": {
            "mosi_path": str(mosi_path),
            "imdb_path": str(imdb_path),
            "humor_path": str(humor_path),
            "sarcasm_path": str(sarcasm_path),
            "max_seq_len": max_seq_len,
            "max_samples": max_samples,
            "train_max_samples": train_max_samples,
            "valid_max_samples": valid_max_samples,
            "test_max_samples": test_max_samples,
            "imdb_text_dropout_prob": imdb_text_dropout_prob,
            "device": device,
            "knn_device": knn_device,
            "encoder_hidden_dim": encoder_hidden_dim,
            "fusion_dim": fusion_dim,
            "fusion_layers": fusion_layers,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "dropout": dropout,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "k": k,
            "num_alignment_permutations": num_alignment_permutations,
            "num_bootstrap": num_bootstrap,
            "top_k_subspace": top_k_subspace,
            "pid_source": pid_source,
            "num_pid_bins": num_pid_bins,
            "bucket_margin": bucket_margin,
            "seed": seed,
        },
        "training": {"joint": joint_train, "unimodal": unimodal_train},
        "bucket_summary": interactions["summary"],
        "representation_alignment": {
            "layer_names": rep_alignment["layer_names"],
            "summary": rep_alignment["summary"],
            "bucket_cka_trajectory": rep_summary,
        },
        "optimization_alignment": {
            "sample_gradcos_summary": {
                "mean": float(np.mean(opt_alignment["sample_gradcos"])),
                "std": float(np.std(opt_alignment["sample_gradcos"])),
            },
            "sample_curvagree_summary": {
                "mean": float(np.nanmean(opt_alignment["sample_layer_curvagree"][analysis_layer])),
                "std": float(np.nanstd(opt_alignment["sample_layer_curvagree"][analysis_layer])),
            },
            "bucket_gradcos_trajectory": grad_summary,
            "bucket_curvagree_trajectory": curvagree_summary,
        },
        "analysis": analysis,
        "figures": figures,
    }
    if interactions["lsmi"] is not None:
        results["lsmi"] = _serialize_nested_arrays(interactions["lsmi"])
    write_json(results, output_json)
    return results
