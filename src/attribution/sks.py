from __future__ import annotations

import numpy as np


def sks_masks_and_metric(
    mask_blocks: dict[str, np.ndarray],
    *,
    train_indices: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample = mask_blocks["sample_deleted"][train_indices]
    modality = mask_blocks["modality_deleted"][train_indices]
    flat = np.concatenate([sample, modality], axis=1)

    num_groups = sample.shape[1]
    struct_metric = np.zeros(
        (num_groups + modality.shape[1], num_groups + modality.shape[1]),
        dtype=np.float64,
    )
    struct_metric[:num_groups, :num_groups] = np.eye(num_groups, dtype=np.float64)
    struct_metric[num_groups:, num_groups:] = sks_metric(
        num_groups,
        rho=rho,
    )
    return flat, struct_metric


def sks_metric(num_groups: int, *, rho: float) -> np.ndarray:
    rho_value = float(rho)
    if not -1.0 <= rho_value <= 1.0:
        raise ValueError(
            f"rho must be in [-1, 1] for a PSD sks metric, got {rho_value}"
        )
    block = np.asarray([[1.0, rho_value], [rho_value, 1.0]], dtype=np.float64)
    metric = np.zeros((num_groups * 2, num_groups * 2), dtype=np.float64)
    for group_id in range(num_groups):
        sl = slice(group_id * 2, group_id * 2 + 2)
        metric[sl, sl] = block
    return metric


__all__ = ["sks_masks_and_metric", "sks_metric"]
