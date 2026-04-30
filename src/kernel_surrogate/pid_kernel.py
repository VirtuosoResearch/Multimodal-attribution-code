from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from fisher_geometry.fisher_metrics import compute_loss
from kernel_surrogate.real_data_cmis import (
    median_gamma,
    method_masks_and_metric,
    metric_sqdist,
    select_gradient_parameters,
)


def compute_group_pid_gradients(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: str,
    gradient_params: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unimodal gradients (G, 2, P) and full multimodal gradients (G, P)."""
    params = select_gradient_parameters(model, gradient_params=gradient_params)
    num_params = sum(int(param.numel()) for param in params)
    num_groups = len(group_indices)
    unimodal = np.empty((num_groups, 2, num_params), dtype=np.float32)
    full = np.empty((num_groups, num_params), dtype=np.float32)

    train_x = data["train_x"]
    train_y = data["train_y"]
    train_labels = data["train_labels"]
    for group_id, indices in enumerate(group_indices):
        index = torch.as_tensor(indices, dtype=torch.long)
        bx = train_x[index].to(device)
        by = train_y[index].to(device)
        bl = train_labels[index].to(device)
        for mode, use_x, use_y in ((0, True, False), (1, False, True), (2, True, True)):
            model.zero_grad(set_to_none=True)
            logits = model(bx, by, use_x=use_x, use_y=use_y)
            loss = compute_loss(task, logits, bl)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            row = full[group_id] if mode == 2 else unimodal[group_id, mode]
            offset = 0
            for param, grad in zip(params, grads):
                width = int(param.numel())
                if grad is None:
                    row[offset : offset + width] = 0.0
                else:
                    row[offset : offset + width] = grad.detach().reshape(-1).float().cpu().numpy()
                offset += width
        if (group_id + 1) % 10 == 0 or group_id + 1 == num_groups:
            print(f"  PID gradients {group_id + 1}/{num_groups} groups", flush=True)
    return unimodal, full


def fic_interaction_gram(
    unimodal_gradients: np.ndarray,
    full_gradients: np.ndarray,
    *,
    damping: float = 1e-3,
    eps_floor: float = 1e-12,
    project_psd: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the signed Fisher-normalized non-additive interaction Gram.

    The additive Fisher inverse is applied with a low-rank Woodbury formula:
    A = Z^T Z / G + eps I, where Z stacks the two unimodal group gradients.
    """
    unimodal = np.nan_to_num(np.asarray(unimodal_gradients, dtype=np.float64), nan=0.0)
    full = np.nan_to_num(np.asarray(full_gradients, dtype=np.float64), nan=0.0)
    if unimodal.ndim != 3 or unimodal.shape[1] != 2:
        raise ValueError(f"expected unimodal gradients with shape (G, 2, P), got {unimodal.shape}")
    if full.shape != (unimodal.shape[0], unimodal.shape[2]):
        raise ValueError(f"full gradient shape {full.shape} is incompatible with {unimodal.shape}")

    num_groups = int(unimodal.shape[0])
    z = unimodal.reshape(num_groups * 2, unimodal.shape[2])
    residual = full - unimodal[:, 0, :] - unimodal[:, 1, :]
    unimodal_norm_sq = np.sum(z * z, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    scale = float(np.mean(unimodal_norm_sq) / max(num_groups, 1))
    eps_abs = max(float(damping) * scale, float(eps_floor))

    zz = z @ z.T
    bmat = zz + (num_groups * eps_abs) * np.eye(zz.shape[0], dtype=np.float64)
    cct = residual @ residual.T
    czt = residual @ z.T
    try:
        correction = czt @ np.linalg.solve(bmat, czt.T)
    except np.linalg.LinAlgError:
        correction = czt @ np.linalg.pinv(bmat, hermitian=True) @ czt.T
    gram = (cct - correction) / eps_abs
    gram = 0.5 * (gram + gram.T)

    eig_before = np.linalg.eigvalsh(gram)
    if project_psd:
        vals, vecs = np.linalg.eigh(gram)
        vals = np.maximum(vals, 0.0)
        gram = (vecs * vals[None, :]) @ vecs.T
        gram = 0.5 * (gram + gram.T)

    pre_normalize_diag = np.diag(gram).copy()
    if normalize:
        denom = np.sqrt(np.maximum(pre_normalize_diag, float(eps_floor)))
        gram = gram / np.maximum(denom[:, None] * denom[None, :], float(eps_floor))
        gram = 0.5 * (gram + gram.T)
        np.fill_diagonal(gram, np.where(pre_normalize_diag > eps_floor, 1.0, 0.0))

    eig_after = np.linalg.eigvalsh(gram)
    stats = {
        "num_groups": num_groups,
        "num_params": int(unimodal.shape[2]),
        "damping": float(damping),
        "eps_abs": float(eps_abs),
        "unimodal_norm_min": float(np.min(np.sqrt(np.maximum(unimodal_norm_sq, 0.0)))),
        "unimodal_norm_median": float(np.median(np.sqrt(np.maximum(unimodal_norm_sq, 0.0)))),
        "unimodal_norm_max": float(np.max(np.sqrt(np.maximum(unimodal_norm_sq, 0.0)))),
        "residual_norm_min": float(np.min(residual_norm)),
        "residual_norm_median": float(np.median(residual_norm)),
        "residual_norm_max": float(np.max(residual_norm)),
        "gram_diag_mean_before_normalize": float(np.mean(pre_normalize_diag)),
        "gram_min_eig_before_psd": float(np.min(eig_before)),
        "gram_min_eig_final": float(np.min(eig_after)),
        "gram_rank_final": int(np.sum(eig_after > 1e-10)),
        "project_psd": bool(project_psd),
        "normalize": bool(normalize),
    }
    return np.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0), stats


def interaction_activation(mask_blocks: dict[str, np.ndarray], indices: Sequence[int] | np.ndarray) -> np.ndarray:
    sample_selected = 1.0 - np.asarray(mask_blocks["sample_deleted"], dtype=np.float64)[indices]
    modality_deleted = np.asarray(mask_blocks["modality_deleted"], dtype=np.float64)[indices]
    modality_available = (1.0 - modality_deleted).reshape(len(sample_selected), sample_selected.shape[1], 2)
    modality_available = modality_available * sample_selected[:, :, None]
    return modality_available[:, :, 0] * modality_available[:, :, 1]


def fit_pid_fic_surrogate(
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    syn_metric: np.ndarray,
    *,
    ridge_lambda: float,
    gamma_scale: float,
    syn_gamma_scale: float,
    lambda_syn: float,
    structured_shared_weight: float,
    train_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    if train_indices is None:
        train_indices = np.arange(len(targets), dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if syn_metric.shape != (int(mask_blocks["num_groups"][0]), int(mask_blocks["num_groups"][0])):
        raise ValueError(f"syn_metric shape {syn_metric.shape} does not match num_groups={mask_blocks['num_groups'][0]}")

    marginal_masks, marginal_metric = method_masks_and_metric(
        "structured_rbf",
        mask_blocks,
        np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64),
        train_indices=train_indices,
        structured_shared_weight=structured_shared_weight,
    )
    syn_masks = interaction_activation(mask_blocks, train_indices)
    marginal_dist = metric_sqdist(marginal_masks, marginal_masks, marginal_metric)
    syn_dist = metric_sqdist(syn_masks, syn_masks, syn_metric)
    base_gamma = median_gamma(marginal_dist)
    base_syn_gamma = median_gamma(syn_dist)
    gamma = float(base_gamma * gamma_scale)
    syn_gamma = float(base_syn_gamma * syn_gamma_scale)
    kernel_marginal = np.exp(-gamma * marginal_dist)
    kernel_syn = np.exp(-syn_gamma * syn_dist)
    kernel = kernel_marginal + float(lambda_syn) * kernel_syn

    target_mean = float(np.mean(targets[train_indices]))
    alpha = np.linalg.solve(
        kernel + float(ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64),
        targets[train_indices] - target_mean,
    )
    train_predictions = target_mean + kernel @ alpha
    return {
        "method": "pid_fic",
        "alpha": alpha,
        "target_mean": target_mean,
        "gamma": gamma,
        "base_gamma": base_gamma,
        "syn_gamma": syn_gamma,
        "base_syn_gamma": base_syn_gamma,
        "lambda_syn": float(lambda_syn),
        "train_predictions": train_predictions,
        "train_indices": train_indices.copy(),
        "marginal_train_masks": marginal_masks.copy(),
        "marginal_metric": marginal_metric.copy(),
        "syn_train_masks": syn_masks.copy(),
        "syn_metric": np.asarray(syn_metric, dtype=np.float64).copy(),
    }


def predict_pid_fic_surrogate(
    mask_blocks: dict[str, np.ndarray],
    fit: dict[str, Any],
    *,
    query_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    query_indices = np.asarray(query_indices, dtype=np.int64)
    query_marginal_masks, _ = method_masks_and_metric(
        "structured_rbf",
        mask_blocks,
        np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64),
        train_indices=query_indices,
        structured_shared_weight=1.0,
    )
    query_syn_masks = interaction_activation(mask_blocks, query_indices)
    marginal_dist = metric_sqdist(query_marginal_masks, fit["marginal_train_masks"], fit["marginal_metric"])
    syn_dist = metric_sqdist(query_syn_masks, fit["syn_train_masks"], fit["syn_metric"])
    kernel = np.exp(-float(fit["gamma"]) * marginal_dist) + float(fit["lambda_syn"]) * np.exp(
        -float(fit["syn_gamma"]) * syn_dist
    )
    return float(fit["target_mean"]) + kernel @ fit["alpha"]


def pearsonr(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denom = float(math.sqrt(float(np.sum(pred**2) * np.sum(truth**2))))
    if denom <= 1e-12 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(pred * truth) / denom)
