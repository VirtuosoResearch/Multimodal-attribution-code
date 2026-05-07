from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_METHODS = (
    "kernel_surrogate_1_rbf",
    "kernel_surrogate_1_structured_rbf",
    "kernel_surrogate_2_rbf",
    "kernel_surrogate_1_linear",
    "kernel_surrogate_2_linear",
    "kernel_surrogate_hier_linear",
    "kernel_surrogate_hier_centered_linear",
    "influence_function_linear",
    "raw_flat_rbf",
)


@dataclass(frozen=True)
class KernelAttributionConfig:
    ridge_lambda: float = 1e-3
    train_fraction: float = 0.8
    validation_fraction: float = 0.2
    split_seed: int = 0
    target_metric: str = "test_loss"
    input_unit: str = "sample"
    methods: tuple[str, ...] = DEFAULT_METHODS
    search_hyperparams: bool = False
    search_strategy: str = "loo"
    refit_on_train_validation: bool = True
    ridge_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1, 10)
    gamma_scale_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, )
    modality_shared_grid: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class MaskBlocks:
    sample_deleted: np.ndarray
    modality_deleted: np.ndarray
    flat_deleted: np.ndarray
    targets: np.ndarray
    run_ids: tuple[int, ...]
    dataset_size: int
    num_modalities: int


@dataclass(frozen=True)
class KernelFit:
    method: str
    alpha: np.ndarray
    target_mean: float
    train_indices: np.ndarray
    ridge_lambda: float
    loo_lds: float | None = None
    loo_rmse: float | None = None
    validation_lds: float | None = None
    validation_rmse: float | None = None
    test_search_lds: float | None = None
    test_search_rmse: float | None = None
    gamma_sample: float | None = None
    gamma_modality: float | None = None
    gamma_flat: float | None = None
    modality_shared_weight: float | None = None
    linear_modality_weight: float | None = None


@dataclass(frozen=True)
class MethodEvaluation:
    method: str
    rmse: float
    lds: float
    num_train: int
    num_validation: int
    num_test: int
    predictions: np.ndarray
    targets: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    fit: KernelFit


def available_dataset_seeds(result_dir: str | Path) -> list[tuple[str, int]]:
    """Return dataset/seed result files produced by lds_real_train.py."""
    result_path = Path(result_dir)
    pairs: list[tuple[str, int]] = []
    for path in sorted(result_path.glob("*_seed*.json")):
        if path.name == "all_results.json":
            continue
        match = re.fullmatch(r"(.+)_seed(\d+)\.json", path.name)
        if match is None:
            continue
        pairs.append((match.group(1), int(match.group(2))))
    return pairs


def load_result_payload(result_dir: str | Path, dataset: str, seed: int) -> dict[str, Any]:
    path = Path(result_dir) / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing LDS result file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "runs" not in payload or "group_partition" not in payload:
        raise ValueError(f"invalid LDS result payload: {path}")
    return payload


def build_mask_blocks(
    payload: dict[str, Any],
    *,
    target_metric: str = "test_acc",
    input_unit: str = "sample",
) -> MaskBlocks:
    """Build exact sample-level hierarchical and flat masks from LDS subset runs.

    For kernel_surrogate_1 we use d_j=1 for deleted samples and E_jm=1 for
    modality deletions among present samples. For kernel_surrogate_2, the
    coordinates a=1-s and B=s(1-G) are numerically the same blocks.
    raw_flat_rbf uses one atom-level deletion vector with no separate sample
    block.
    """
    runs = list(payload.get("runs", []))
    if len(runs) < 2:
        raise ValueError("need at least two subset runs for LDS evaluation")

    resolved_input_unit = input_unit.strip().lower()
    if resolved_input_unit not in {"sample", "group"}:
        raise ValueError(f"input_unit must be 'sample' or 'group', got {input_unit!r}")

    group_partition = payload["group_partition"]
    group_indices = group_partition.get("group_indices", [])
    num_modalities = int(group_partition.get("num_modalities", 0))
    if num_modalities <= 0:
        raise ValueError("group_partition.num_modalities must be positive")
    dataset_size = (
        int(group_partition.get("num_groups", 0))
        if resolved_input_unit == "group"
        else _infer_dataset_size(group_indices, runs)
    )
    if dataset_size <= 0:
        raise ValueError(f"could not infer positive {resolved_input_unit} count")

    num_runs = len(runs)
    sample_deleted = np.ones((num_runs, dataset_size), dtype=np.float32)
    modality_deleted = np.zeros((num_runs, dataset_size * num_modalities), dtype=np.float32)
    flat_deleted = np.ones((num_runs, dataset_size * num_modalities), dtype=np.float32)
    targets = np.empty(num_runs, dtype=np.float64)
    run_ids: list[int] = []

    modality_offsets = np.arange(num_modalities, dtype=np.int64)
    index_to_group = (
        _build_index_to_group(group_indices)
        if resolved_input_unit == "group"
        else {}
    )
    for row, run in enumerate(runs):
        targets[row] = _resolve_run_target(run, target_metric=target_metric, row=row)
        run_ids.append(int(run.get("sample_index", row)))

        data_indices = np.asarray(run.get("data_indices", []), dtype=np.int64)
        keep_mask = np.asarray(run.get("sample_modality_mask", []), dtype=bool)
        if data_indices.ndim != 1:
            raise ValueError(f"run {row}: data_indices must be 1D")
        if keep_mask.shape != (data_indices.shape[0], num_modalities):
            raise ValueError(
                f"run {row}: sample_modality_mask shape {keep_mask.shape} does not match "
                f"({data_indices.shape[0]}, {num_modalities})"
            )
        if (
            resolved_input_unit == "sample"
            and data_indices.size
            and (data_indices.min() < 0 or data_indices.max() >= dataset_size)
        ):
            raise ValueError(f"run {row}: data index out of inferred range 0..{dataset_size - 1}")

        deletion_mask = np.logical_not(keep_mask)
        if resolved_input_unit == "sample":
            sample_deleted[row, data_indices] = 0.0
            flat_positions = data_indices[:, None] * num_modalities + modality_offsets[None, :]
            modality_deleted[row, flat_positions.reshape(-1)] = deletion_mask.reshape(-1).astype(np.float32)
            flat_deleted[row, flat_positions.reshape(-1)] = deletion_mask.reshape(-1).astype(np.float32)
        else:
            group_ids = np.asarray(run.get("group_ids", []), dtype=np.int64)
            if group_ids.ndim != 1:
                raise ValueError(f"run {row}: group_ids must be 1D")
            if group_ids.size and (group_ids.min() < 0 or group_ids.max() >= dataset_size):
                raise ValueError(f"run {row}: group id out of inferred range 0..{dataset_size - 1}")
            sample_deleted[row, group_ids] = 0.0

            local_groups = np.asarray(
                [index_to_group.get(int(index), -1) for index in data_indices.tolist()],
                dtype=np.int64,
            )
            if np.any(local_groups < 0):
                raise ValueError(f"run {row}: some data_indices are missing from group_partition")
            for group_id in group_ids.tolist():
                group_mask = local_groups == int(group_id)
                if not np.any(group_mask):
                    continue
                group_deletion_rate = deletion_mask[group_mask].mean(axis=0).astype(np.float32)
                positions = int(group_id) * num_modalities + modality_offsets
                modality_deleted[row, positions] = group_deletion_rate
                flat_deleted[row, positions] = group_deletion_rate

    return MaskBlocks(
        sample_deleted=sample_deleted,
        modality_deleted=modality_deleted,
        flat_deleted=flat_deleted,
        targets=targets,
        run_ids=tuple(run_ids),
        dataset_size=dataset_size,
        num_modalities=num_modalities,
    )


def _resolve_run_target(run: dict[str, Any], *, target_metric: str, row: int) -> float:
    if target_metric in run:
        value = run[target_metric]
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"run {row}: target metric {target_metric!r} is not scalar")

    if target_metric == "test_sample_loss_mean":
        sample_losses = np.asarray(run.get("test_sample_losses", []), dtype=np.float64)
        if sample_losses.size == 0:
            raise KeyError(f"run {row}: missing non-empty test_sample_losses")
        return float(sample_losses.mean())

    match = re.fullmatch(r"test_sample_loss(?::|_)(\d+)", target_metric)
    if match is not None:
        sample_index = int(match.group(1))
        sample_losses = run.get("test_sample_losses", [])
        if sample_index < 0 or sample_index >= len(sample_losses):
            raise IndexError(
                f"run {row}: test sample index {sample_index} is outside saved "
                f"test_sample_losses length {len(sample_losses)}"
            )
        return float(sample_losses[sample_index])

    raise KeyError(f"run {row} does not contain target metric {target_metric!r}")


def evaluate_methods(
    blocks: MaskBlocks,
    *,
    config: KernelAttributionConfig | None = None,
) -> dict[str, MethodEvaluation]:
    config = config or KernelAttributionConfig()
    search_strategy = config.search_strategy.strip().lower()
    if search_strategy not in {"loo", "validation", "test"}:
        raise ValueError(f"search_strategy must be 'loo', 'validation', or 'test', got {config.search_strategy!r}")
    validation_indices = np.asarray([], dtype=np.int64)
    if config.search_hyperparams and search_strategy == "validation":
        train_indices, validation_indices, test_indices = split_train_validation_test_indices(
            len(blocks.targets),
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            seed=config.split_seed,
        )
    else:
        train_indices, test_indices = split_indices(
            len(blocks.targets),
            train_fraction=config.train_fraction,
            seed=config.split_seed,
        )
    evaluations: dict[str, MethodEvaluation] = {}
    for method in config.methods:
        if config.search_hyperparams and search_strategy == "validation":
            fit = fit_best_validation_kernel_surrogate(
                blocks,
                train_indices=train_indices,
                validation_indices=validation_indices,
                method=method,
                ridge_grid=config.ridge_grid,
                gamma_scale_grid=config.gamma_scale_grid,
                modality_shared_grid=config.modality_shared_grid,
                refit_on_train_validation=config.refit_on_train_validation,
            )
        elif config.search_hyperparams and search_strategy == "test":
            fit = fit_best_test_kernel_surrogate(
                blocks,
                train_indices=train_indices,
                test_indices=test_indices,
                method=method,
                ridge_grid=config.ridge_grid,
                gamma_scale_grid=config.gamma_scale_grid,
                modality_shared_grid=config.modality_shared_grid,
            )
        elif config.search_hyperparams:
            fit = fit_best_kernel_surrogate(
                blocks,
                train_indices=train_indices,
                method=method,
                ridge_grid=config.ridge_grid,
                gamma_scale_grid=config.gamma_scale_grid,
                modality_shared_grid=config.modality_shared_grid,
            )
        else:
            fit = fit_kernel_surrogate(
                blocks,
                train_indices=train_indices,
                method=method,
                ridge_lambda=config.ridge_lambda,
            )
        predictions = predict_kernel_surrogate(blocks, fit, query_indices=test_indices)
        targets = blocks.targets[test_indices]
        evaluations[method] = MethodEvaluation(
            method=method,
            rmse=rmse(predictions, targets),
            lds=spearmanr(predictions, targets),
            num_train=int(train_indices.size),
            num_validation=int(validation_indices.size),
            num_test=int(test_indices.size),
            predictions=predictions,
            targets=targets,
            train_indices=train_indices.copy(),
            validation_indices=validation_indices.copy(),
            test_indices=test_indices.copy(),
            fit=fit,
        )
    return evaluations


def split_indices(num_items: int, *, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_items < 2:
        raise ValueError("need at least two items to split train/test")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_items)
    num_train = int(round(num_items * train_fraction))
    num_train = min(max(num_train, 1), num_items - 1)
    train_indices = np.sort(order[:num_train]).astype(np.int64, copy=False)
    test_indices = np.sort(order[num_train:]).astype(np.int64, copy=False)
    return train_indices, test_indices


def split_train_validation_test_indices(
    num_items: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_items < 3:
        raise ValueError("need at least three items to split train/validation/test")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1.0, "
            f"got {train_fraction + validation_fraction}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_items)
    num_train = int(round(num_items * train_fraction))
    num_validation = int(round(num_items * validation_fraction))
    num_train = min(max(num_train, 1), num_items - 2)
    num_validation = min(max(num_validation, 1), num_items - num_train - 1)
    train_indices = np.sort(order[:num_train]).astype(np.int64, copy=False)
    validation_indices = np.sort(order[num_train : num_train + num_validation]).astype(np.int64, copy=False)
    test_indices = np.sort(order[num_train + num_validation :]).astype(np.int64, copy=False)
    return train_indices, validation_indices, test_indices


def fit_kernel_surrogate(
    blocks: MaskBlocks,
    *,
    train_indices: Sequence[int] | np.ndarray,
    method: str,
    ridge_lambda: float = 1e-3,
    gamma_sample: float | None = None,
    gamma_modality: float | None = None,
    gamma_flat: float | None = None,
    modality_shared_weight: float | None = None,
    linear_modality_weight: float | None = None,
    loo_lds: float | None = None,
    loo_rmse: float | None = None,
    validation_lds: float | None = None,
    validation_rmse: float | None = None,
    test_search_lds: float | None = None,
    test_search_rmse: float | None = None,
) -> KernelFit:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.ndim != 1 or train_indices.size == 0:
        raise ValueError("train_indices must be a non-empty 1D array")
    method_name = normalize_method_name(method)
    target_mean = float(blocks.targets[train_indices].mean())
    centered_targets = blocks.targets[train_indices] - target_mean

    train_kernel, gammas = compute_kernel(
        blocks,
        lhs_indices=train_indices,
        rhs_indices=train_indices,
        method=method_name,
        gamma_sample=gamma_sample,
        gamma_modality=gamma_modality,
        gamma_flat=gamma_flat,
        modality_shared_weight=modality_shared_weight,
        linear_modality_weight=linear_modality_weight,
    )
    regularized = train_kernel + float(ridge_lambda) * np.eye(train_kernel.shape[0], dtype=np.float64)
    alpha = np.linalg.solve(regularized, centered_targets)
    return KernelFit(
        method=method_name,
        alpha=alpha.astype(np.float64, copy=False),
        target_mean=target_mean,
        train_indices=train_indices.copy(),
        ridge_lambda=float(ridge_lambda),
        loo_lds=loo_lds,
        loo_rmse=loo_rmse,
        validation_lds=validation_lds,
        validation_rmse=validation_rmse,
        test_search_lds=test_search_lds,
        test_search_rmse=test_search_rmse,
        gamma_sample=gammas.get("gamma_sample"),
        gamma_modality=gammas.get("gamma_modality"),
        gamma_flat=gammas.get("gamma_flat"),
        modality_shared_weight=gammas.get("modality_shared_weight"),
        linear_modality_weight=gammas.get("linear_modality_weight"),
    )


def fit_best_kernel_surrogate(
    blocks: MaskBlocks,
    *,
    train_indices: Sequence[int] | np.ndarray,
    method: str,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> KernelFit:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    method_name = normalize_method_name(method)
    search = search_kernel_hyperparams(
        blocks,
        train_indices=train_indices,
        method=method_name,
        ridge_grid=ridge_grid,
        gamma_scale_grid=gamma_scale_grid,
        modality_shared_grid=modality_shared_grid,
    )
    return fit_kernel_surrogate(
        blocks,
        train_indices=train_indices,
        method=method_name,
        ridge_lambda=float(search["ridge_lambda"]),
        gamma_sample=search.get("gamma_sample"),
        gamma_modality=search.get("gamma_modality"),
        gamma_flat=search.get("gamma_flat"),
        modality_shared_weight=search.get("modality_shared_weight"),
        linear_modality_weight=search.get("linear_modality_weight"),
        loo_lds=float(search["loo_lds"]),
        loo_rmse=float(search["loo_rmse"]),
    )


def fit_best_validation_kernel_surrogate(
    blocks: MaskBlocks,
    *,
    train_indices: Sequence[int] | np.ndarray,
    validation_indices: Sequence[int] | np.ndarray,
    method: str,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
    refit_on_train_validation: bool = True,
) -> KernelFit:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    method_name = normalize_method_name(method)
    search = search_kernel_hyperparams_validation(
        blocks,
        train_indices=train_indices,
        validation_indices=validation_indices,
        method=method_name,
        ridge_grid=ridge_grid,
        gamma_scale_grid=gamma_scale_grid,
        modality_shared_grid=modality_shared_grid,
    )
    fit_indices = (
        np.sort(np.concatenate([train_indices, validation_indices])).astype(np.int64, copy=False)
        if refit_on_train_validation
        else train_indices
    )
    return fit_kernel_surrogate(
        blocks,
        train_indices=fit_indices,
        method=method_name,
        ridge_lambda=float(search["ridge_lambda"]),
        gamma_sample=search.get("gamma_sample"),
        gamma_modality=search.get("gamma_modality"),
        gamma_flat=search.get("gamma_flat"),
        modality_shared_weight=search.get("modality_shared_weight"),
        linear_modality_weight=search.get("linear_modality_weight"),
        validation_lds=float(search["validation_lds"]),
        validation_rmse=float(search["validation_rmse"]),
    )


def fit_best_test_kernel_surrogate(
    blocks: MaskBlocks,
    *,
    train_indices: Sequence[int] | np.ndarray,
    test_indices: Sequence[int] | np.ndarray,
    method: str,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> KernelFit:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    method_name = normalize_method_name(method)
    search = search_kernel_hyperparams_validation(
        blocks,
        train_indices=train_indices,
        validation_indices=test_indices,
        method=method_name,
        ridge_grid=ridge_grid,
        gamma_scale_grid=gamma_scale_grid,
        modality_shared_grid=modality_shared_grid,
    )
    return fit_kernel_surrogate(
        blocks,
        train_indices=train_indices,
        method=method_name,
        ridge_lambda=float(search["ridge_lambda"]),
        gamma_sample=search.get("gamma_sample"),
        gamma_modality=search.get("gamma_modality"),
        gamma_flat=search.get("gamma_flat"),
        modality_shared_weight=search.get("modality_shared_weight"),
        linear_modality_weight=search.get("linear_modality_weight"),
        test_search_lds=float(search["validation_lds"]),
        test_search_rmse=float(search["validation_rmse"]),
    )


def search_kernel_hyperparams(
    blocks: MaskBlocks,
    *,
    train_indices: Sequence[int] | np.ndarray,
    method: str,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> dict[str, float | None]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    method_name = normalize_method_name(method)
    targets = blocks.targets[train_indices]
    if train_indices.size < 3:
        raise ValueError("need at least three train subsets for leave-one-out hyperparameter search")

    candidates: list[dict[str, float | None]] = []
    if method_name in {"kernel_surrogate_1_rbf", "kernel_surrogate_2_rbf"}:
        sample_dist = pairwise_sqeuclidean(blocks.sample_deleted[train_indices], blocks.sample_deleted[train_indices])
        modality_dist = pairwise_sqeuclidean(blocks.modality_deleted[train_indices], blocks.modality_deleted[train_indices])
        base_gamma_sample = median_heuristic_gamma(sample_dist)
        base_gamma_modality = median_heuristic_gamma(modality_dist)
        for sample_scale in gamma_scale_grid:
            for modality_scale in gamma_scale_grid:
                gamma_sample = float(base_gamma_sample * float(sample_scale))
                gamma_modality = float(base_gamma_modality * float(modality_scale))
                kernel = np.exp(-gamma_sample * sample_dist - gamma_modality * modality_dist)
                for ridge in ridge_grid:
                    loo_predictions = _loo_predictions_from_kernel(kernel, targets, ridge_lambda=float(ridge))
                    candidates.append(
                        {
                            "ridge_lambda": float(ridge),
                            "gamma_sample": gamma_sample,
                            "gamma_modality": gamma_modality,
                            "gamma_flat": None,
                            "modality_shared_weight": None,
                            "linear_modality_weight": None,
                            "loo_lds": spearmanr(loo_predictions, targets),
                            "loo_rmse": rmse(loo_predictions, targets),
                        }
                    )
    elif method_name == "kernel_surrogate_1_structured_rbf":
        sample_dist = pairwise_sqeuclidean(blocks.sample_deleted[train_indices], blocks.sample_deleted[train_indices])
        base_modality_dist = pairwise_sqeuclidean(
            blocks.modality_deleted[train_indices],
            blocks.modality_deleted[train_indices],
        )
        shared_modality_dist = structured_modality_shared_sqdistance(
            blocks,
            lhs_indices=train_indices,
            rhs_indices=train_indices,
        )
        base_gamma_sample = median_heuristic_gamma(sample_dist)
        for shared_weight in modality_shared_grid:
            modality_dist = base_modality_dist + float(shared_weight) * shared_modality_dist
            base_gamma_modality = median_heuristic_gamma(modality_dist)
            for sample_scale in gamma_scale_grid:
                for modality_scale in gamma_scale_grid:
                    gamma_sample = float(base_gamma_sample * float(sample_scale))
                    gamma_modality = float(base_gamma_modality * float(modality_scale))
                    kernel = np.exp(-gamma_sample * sample_dist - gamma_modality * modality_dist)
                    for ridge in ridge_grid:
                        loo_predictions = _loo_predictions_from_kernel(kernel, targets, ridge_lambda=float(ridge))
                        candidates.append(
                            {
                                "ridge_lambda": float(ridge),
                                "gamma_sample": gamma_sample,
                                "gamma_modality": gamma_modality,
                                "gamma_flat": None,
                                "modality_shared_weight": float(shared_weight),
                                "linear_modality_weight": None,
                                "loo_lds": spearmanr(loo_predictions, targets),
                                "loo_rmse": rmse(loo_predictions, targets),
                            }
                        )
    elif method_name == "raw_flat_rbf":
        flat_dist = pairwise_sqeuclidean(blocks.flat_deleted[train_indices], blocks.flat_deleted[train_indices])
        base_gamma_flat = median_heuristic_gamma(flat_dist)
        for scale in gamma_scale_grid:
            gamma_flat = float(base_gamma_flat * float(scale))
            kernel = np.exp(-gamma_flat * flat_dist)
            for ridge in ridge_grid:
                loo_predictions = _loo_predictions_from_kernel(kernel, targets, ridge_lambda=float(ridge))
                candidates.append(
                    {
                        "ridge_lambda": float(ridge),
                        "gamma_sample": None,
                        "gamma_modality": None,
                        "gamma_flat": gamma_flat,
                        "modality_shared_weight": None,
                        "linear_modality_weight": None,
                        "loo_lds": spearmanr(loo_predictions, targets),
                        "loo_rmse": rmse(loo_predictions, targets),
                    }
                )
    elif method_name in {"kernel_surrogate_1_linear", "kernel_surrogate_2_linear"}:
        sample = blocks.sample_deleted[train_indices]
        modality = blocks.modality_deleted[train_indices]
        kernel = sample @ sample.T + modality @ modality.T
        for ridge in ridge_grid:
            loo_predictions = _loo_predictions_from_kernel(kernel, targets, ridge_lambda=float(ridge))
            candidates.append(
                {
                    "ridge_lambda": float(ridge),
                    "gamma_sample": None,
                    "gamma_modality": None,
                    "gamma_flat": None,
                    "modality_shared_weight": None,
                    "linear_modality_weight": None,
                    "loo_lds": spearmanr(loo_predictions, targets),
                    "loo_rmse": rmse(loo_predictions, targets),
                }
            )
    elif method_name == "influence_function_linear":
        flat = blocks.flat_deleted[train_indices]
        kernel = flat @ flat.T
        for ridge in ridge_grid:
            loo_predictions = _loo_predictions_from_kernel(kernel, targets, ridge_lambda=float(ridge))
            candidates.append(
                {
                    "ridge_lambda": float(ridge),
                    "gamma_sample": None,
                    "gamma_modality": None,
                    "gamma_flat": None,
                    "modality_shared_weight": None,
                    "linear_modality_weight": None,
                    "loo_lds": spearmanr(loo_predictions, targets),
                    "loo_rmse": rmse(loo_predictions, targets),
                }
            )
    elif method_name in {"kernel_surrogate_hier_linear", "kernel_surrogate_hier_centered_linear"}:
        sample_kernel, modality_kernel = _normalized_hierarchical_linear_kernel(
            blocks,
            lhs_indices=train_indices,
            rhs_indices=train_indices,
            centered=method_name == "kernel_surrogate_hier_centered_linear",
        )
        modality_weights = sorted({0.0, *[float(value) for value in gamma_scale_grid], 8.0, 16.0})
        for modality_weight in modality_weights:
            kernel = sample_kernel + float(modality_weight) * modality_kernel
            for ridge in ridge_grid:
                loo_predictions = _loo_predictions_from_kernel(kernel, targets, ridge_lambda=float(ridge))
                candidates.append(
                    {
                        "ridge_lambda": float(ridge),
                        "gamma_sample": None,
                        "gamma_modality": None,
                        "gamma_flat": None,
                        "modality_shared_weight": None,
                        "linear_modality_weight": float(modality_weight),
                        "loo_lds": spearmanr(loo_predictions, targets),
                        "loo_rmse": rmse(loo_predictions, targets),
                    }
                )
    else:
        raise ValueError(f"unsupported method {method!r}")

    finite_candidates = [
        candidate
        for candidate in candidates
        if math.isfinite(float(candidate["loo_lds"])) and math.isfinite(float(candidate["loo_rmse"]))
    ]
    if not finite_candidates:
        raise RuntimeError(f"no finite hyperparameter candidates for method={method_name}")
    return max(
        finite_candidates,
        key=lambda candidate: (
            float(candidate["loo_lds"]),
            -float(candidate["loo_rmse"]),
            -float(candidate["ridge_lambda"]),
        ),
    )


def search_kernel_hyperparams_validation(
    blocks: MaskBlocks,
    *,
    train_indices: Sequence[int] | np.ndarray,
    validation_indices: Sequence[int] | np.ndarray,
    method: str,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> dict[str, float | None]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    method_name = normalize_method_name(method)
    train_targets = blocks.targets[train_indices]
    validation_targets = blocks.targets[validation_indices]
    if train_indices.size < 2:
        raise ValueError("need at least two train subsets for validation hyperparameter search")
    if validation_indices.size < 2:
        raise ValueError("need at least two validation subsets for validation hyperparameter search")

    candidates: list[dict[str, float | None]] = []
    if method_name in {"kernel_surrogate_1_rbf", "kernel_surrogate_2_rbf"}:
        train_sample_dist = pairwise_sqeuclidean(blocks.sample_deleted[train_indices], blocks.sample_deleted[train_indices])
        train_modality_dist = pairwise_sqeuclidean(blocks.modality_deleted[train_indices], blocks.modality_deleted[train_indices])
        validation_sample_dist = pairwise_sqeuclidean(blocks.sample_deleted[validation_indices], blocks.sample_deleted[train_indices])
        validation_modality_dist = pairwise_sqeuclidean(
            blocks.modality_deleted[validation_indices],
            blocks.modality_deleted[train_indices],
        )
        base_gamma_sample = median_heuristic_gamma(train_sample_dist)
        base_gamma_modality = median_heuristic_gamma(train_modality_dist)
        for sample_scale in gamma_scale_grid:
            for modality_scale in gamma_scale_grid:
                gamma_sample = float(base_gamma_sample * float(sample_scale))
                gamma_modality = float(base_gamma_modality * float(modality_scale))
                train_kernel = np.exp(-gamma_sample * train_sample_dist - gamma_modality * train_modality_dist)
                validation_kernel = np.exp(
                    -gamma_sample * validation_sample_dist - gamma_modality * validation_modality_dist
                )
                for ridge in ridge_grid:
                    validation_predictions = _query_predictions_from_kernel(
                        train_kernel,
                        validation_kernel,
                        train_targets,
                        ridge_lambda=float(ridge),
                    )
                    candidates.append(
                        {
                            "ridge_lambda": float(ridge),
                            "gamma_sample": gamma_sample,
                            "gamma_modality": gamma_modality,
                            "gamma_flat": None,
                            "modality_shared_weight": None,
                            "linear_modality_weight": None,
                            "validation_lds": spearmanr(validation_predictions, validation_targets),
                            "validation_rmse": rmse(validation_predictions, validation_targets),
                        }
                    )
    elif method_name == "kernel_surrogate_1_structured_rbf":
        train_sample_dist = pairwise_sqeuclidean(blocks.sample_deleted[train_indices], blocks.sample_deleted[train_indices])
        validation_sample_dist = pairwise_sqeuclidean(blocks.sample_deleted[validation_indices], blocks.sample_deleted[train_indices])
        train_base_modality_dist = pairwise_sqeuclidean(
            blocks.modality_deleted[train_indices],
            blocks.modality_deleted[train_indices],
        )
        validation_base_modality_dist = pairwise_sqeuclidean(
            blocks.modality_deleted[validation_indices],
            blocks.modality_deleted[train_indices],
        )
        train_shared_modality_dist = structured_modality_shared_sqdistance(
            blocks,
            lhs_indices=train_indices,
            rhs_indices=train_indices,
        )
        validation_shared_modality_dist = structured_modality_shared_sqdistance(
            blocks,
            lhs_indices=validation_indices,
            rhs_indices=train_indices,
        )
        base_gamma_sample = median_heuristic_gamma(train_sample_dist)
        for shared_weight in modality_shared_grid:
            train_modality_dist = train_base_modality_dist + float(shared_weight) * train_shared_modality_dist
            validation_modality_dist = (
                validation_base_modality_dist + float(shared_weight) * validation_shared_modality_dist
            )
            base_gamma_modality = median_heuristic_gamma(train_modality_dist)
            for sample_scale in gamma_scale_grid:
                for modality_scale in gamma_scale_grid:
                    gamma_sample = float(base_gamma_sample * float(sample_scale))
                    gamma_modality = float(base_gamma_modality * float(modality_scale))
                    train_kernel = np.exp(-gamma_sample * train_sample_dist - gamma_modality * train_modality_dist)
                    validation_kernel = np.exp(
                        -gamma_sample * validation_sample_dist - gamma_modality * validation_modality_dist
                    )
                    for ridge in ridge_grid:
                        validation_predictions = _query_predictions_from_kernel(
                            train_kernel,
                            validation_kernel,
                            train_targets,
                            ridge_lambda=float(ridge),
                        )
                        candidates.append(
                            {
                                "ridge_lambda": float(ridge),
                                "gamma_sample": gamma_sample,
                                "gamma_modality": gamma_modality,
                                "gamma_flat": None,
                                "modality_shared_weight": float(shared_weight),
                                "linear_modality_weight": None,
                                "validation_lds": spearmanr(validation_predictions, validation_targets),
                                "validation_rmse": rmse(validation_predictions, validation_targets),
                            }
                        )
    elif method_name == "raw_flat_rbf":
        train_flat_dist = pairwise_sqeuclidean(blocks.flat_deleted[train_indices], blocks.flat_deleted[train_indices])
        validation_flat_dist = pairwise_sqeuclidean(blocks.flat_deleted[validation_indices], blocks.flat_deleted[train_indices])
        base_gamma_flat = median_heuristic_gamma(train_flat_dist)
        for scale in gamma_scale_grid:
            gamma_flat = float(base_gamma_flat * float(scale))
            train_kernel = np.exp(-gamma_flat * train_flat_dist)
            validation_kernel = np.exp(-gamma_flat * validation_flat_dist)
            for ridge in ridge_grid:
                validation_predictions = _query_predictions_from_kernel(
                    train_kernel,
                    validation_kernel,
                    train_targets,
                    ridge_lambda=float(ridge),
                )
                candidates.append(
                    {
                        "ridge_lambda": float(ridge),
                        "gamma_sample": None,
                        "gamma_modality": None,
                        "gamma_flat": gamma_flat,
                        "modality_shared_weight": None,
                        "linear_modality_weight": None,
                        "validation_lds": spearmanr(validation_predictions, validation_targets),
                        "validation_rmse": rmse(validation_predictions, validation_targets),
                    }
                )
    elif method_name in {"kernel_surrogate_1_linear", "kernel_surrogate_2_linear"}:
        train_sample = blocks.sample_deleted[train_indices]
        train_modality = blocks.modality_deleted[train_indices]
        validation_sample = blocks.sample_deleted[validation_indices]
        validation_modality = blocks.modality_deleted[validation_indices]
        train_kernel = train_sample @ train_sample.T + train_modality @ train_modality.T
        validation_kernel = validation_sample @ train_sample.T + validation_modality @ train_modality.T
        for ridge in ridge_grid:
            validation_predictions = _query_predictions_from_kernel(
                train_kernel,
                validation_kernel,
                train_targets,
                ridge_lambda=float(ridge),
            )
            candidates.append(
                {
                    "ridge_lambda": float(ridge),
                    "gamma_sample": None,
                    "gamma_modality": None,
                    "gamma_flat": None,
                    "modality_shared_weight": None,
                    "linear_modality_weight": None,
                    "validation_lds": spearmanr(validation_predictions, validation_targets),
                    "validation_rmse": rmse(validation_predictions, validation_targets),
                }
            )
    elif method_name == "influence_function_linear":
        train_flat = blocks.flat_deleted[train_indices]
        validation_flat = blocks.flat_deleted[validation_indices]
        train_kernel = train_flat @ train_flat.T
        validation_kernel = validation_flat @ train_flat.T
        for ridge in ridge_grid:
            validation_predictions = _query_predictions_from_kernel(
                train_kernel,
                validation_kernel,
                train_targets,
                ridge_lambda=float(ridge),
            )
            candidates.append(
                {
                    "ridge_lambda": float(ridge),
                    "gamma_sample": None,
                    "gamma_modality": None,
                    "gamma_flat": None,
                    "modality_shared_weight": None,
                    "linear_modality_weight": None,
                    "validation_lds": spearmanr(validation_predictions, validation_targets),
                    "validation_rmse": rmse(validation_predictions, validation_targets),
                }
            )
    elif method_name in {"kernel_surrogate_hier_linear", "kernel_surrogate_hier_centered_linear"}:
        sample_kernel, modality_kernel = _normalized_hierarchical_linear_kernel(
            blocks,
            lhs_indices=train_indices,
            rhs_indices=train_indices,
            centered=method_name == "kernel_surrogate_hier_centered_linear",
        )
        validation_sample_kernel, validation_modality_kernel = _normalized_hierarchical_linear_kernel(
            blocks,
            lhs_indices=validation_indices,
            rhs_indices=train_indices,
            centered=method_name == "kernel_surrogate_hier_centered_linear",
        )
        modality_weights = sorted({0.0, *[float(value) for value in gamma_scale_grid], 8.0, 16.0})
        for modality_weight in modality_weights:
            train_kernel = sample_kernel + float(modality_weight) * modality_kernel
            validation_kernel = validation_sample_kernel + float(modality_weight) * validation_modality_kernel
            for ridge in ridge_grid:
                validation_predictions = _query_predictions_from_kernel(
                    train_kernel,
                    validation_kernel,
                    train_targets,
                    ridge_lambda=float(ridge),
                )
                candidates.append(
                    {
                        "ridge_lambda": float(ridge),
                        "gamma_sample": None,
                        "gamma_modality": None,
                        "gamma_flat": None,
                        "modality_shared_weight": None,
                        "linear_modality_weight": float(modality_weight),
                        "validation_lds": spearmanr(validation_predictions, validation_targets),
                        "validation_rmse": rmse(validation_predictions, validation_targets),
                    }
                )
    else:
        raise ValueError(f"unsupported method {method!r}")

    finite_candidates = [
        candidate
        for candidate in candidates
        if math.isfinite(float(candidate["validation_lds"])) and math.isfinite(float(candidate["validation_rmse"]))
    ]
    if not finite_candidates:
        raise RuntimeError(f"no finite validation hyperparameter candidates for method={method_name}")
    return max(
        finite_candidates,
        key=lambda candidate: (
            float(candidate["validation_lds"]),
            -float(candidate["validation_rmse"]),
            -float(candidate["ridge_lambda"]),
        ),
    )


def predict_kernel_surrogate(
    blocks: MaskBlocks,
    fit: KernelFit,
    *,
    query_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    query_indices = np.asarray(query_indices, dtype=np.int64)
    kernel, _ = compute_kernel(
        blocks,
        lhs_indices=query_indices,
        rhs_indices=fit.train_indices,
        method=fit.method,
        gamma_sample=fit.gamma_sample,
        gamma_modality=fit.gamma_modality,
        gamma_flat=fit.gamma_flat,
        modality_shared_weight=fit.modality_shared_weight,
        linear_modality_weight=fit.linear_modality_weight,
    )
    return fit.target_mean + kernel @ fit.alpha


def source_modality_effects(
    blocks: MaskBlocks,
    fit: KernelFit,
    *,
    source_indices: Sequence[int],
) -> np.ndarray:
    """Estimate dataset-level leave-one-source-modality attribution effects.

    The returned array has shape [len(source_indices), num_modalities]. Entry
    (j, m) is g(anchor) - g(anchor with source sample j modality m deleted).
    This is the RBF analogue of the local deletion-difference score described
    in kernel_surrogate_2.md. It is not used for LDS, which evaluates subset
    predictions directly.
    """
    method = normalize_method_name(fit.method)
    if method == "raw_flat_rbf":
        anchor = np.zeros((1, blocks.dataset_size * blocks.num_modalities), dtype=np.float32)
        queries = np.repeat(anchor, len(source_indices) * blocks.num_modalities, axis=0)
        for source_offset, source_index in enumerate(source_indices):
            for modality_index in range(blocks.num_modalities):
                row = source_offset * blocks.num_modalities + modality_index
                queries[row, int(source_index) * blocks.num_modalities + modality_index] = 1.0
        train = blocks.flat_deleted[fit.train_indices]
        anchor_train_kernel = _rbf_kernel_from_arrays(
            anchor,
            train,
            gamma=_require_gamma(fit.gamma_flat, "gamma_flat"),
        )
        query_train_kernel = _rbf_kernel_from_arrays(
            queries,
            train,
            gamma=_require_gamma(fit.gamma_flat, "gamma_flat"),
        )
    else:
        anchor_sample = np.zeros((1, blocks.dataset_size), dtype=np.float32)
        anchor_modality = np.zeros((1, blocks.dataset_size * blocks.num_modalities), dtype=np.float32)
        query_sample = np.repeat(anchor_sample, len(source_indices) * blocks.num_modalities, axis=0)
        query_modality = np.repeat(anchor_modality, len(source_indices) * blocks.num_modalities, axis=0)
        for source_offset, source_index in enumerate(source_indices):
            for modality_index in range(blocks.num_modalities):
                row = source_offset * blocks.num_modalities + modality_index
                query_modality[row, int(source_index) * blocks.num_modalities + modality_index] = 1.0
        train_sample = blocks.sample_deleted[fit.train_indices]
        train_modality = blocks.modality_deleted[fit.train_indices]
        anchor_train_kernel = _hierarchical_rbf_kernel_from_arrays(
            anchor_sample,
            anchor_modality,
            train_sample,
            train_modality,
            gamma_sample=_require_gamma(fit.gamma_sample, "gamma_sample"),
            gamma_modality=_require_gamma(fit.gamma_modality, "gamma_modality"),
        )
        query_train_kernel = _hierarchical_rbf_kernel_from_arrays(
            query_sample,
            query_modality,
            train_sample,
            train_modality,
            gamma_sample=_require_gamma(fit.gamma_sample, "gamma_sample"),
            gamma_modality=_require_gamma(fit.gamma_modality, "gamma_modality"),
        )

    anchor_prediction = float(fit.target_mean + anchor_train_kernel @ fit.alpha)
    query_predictions = fit.target_mean + query_train_kernel @ fit.alpha
    effects = anchor_prediction - query_predictions.reshape(len(source_indices), blocks.num_modalities)
    return effects


def compute_kernel(
    blocks: MaskBlocks,
    *,
    lhs_indices: Sequence[int] | np.ndarray,
    rhs_indices: Sequence[int] | np.ndarray,
    method: str,
    gamma_sample: float | None = None,
    gamma_modality: float | None = None,
    gamma_flat: float | None = None,
    modality_shared_weight: float | None = None,
    linear_modality_weight: float | None = None,
) -> tuple[np.ndarray, dict[str, float | None]]:
    method_name = normalize_method_name(method)
    lhs_indices = np.asarray(lhs_indices, dtype=np.int64)
    rhs_indices = np.asarray(rhs_indices, dtype=np.int64)

    if method_name in {"kernel_surrogate_1_rbf", "kernel_surrogate_2_rbf"}:
        lhs_sample = blocks.sample_deleted[lhs_indices]
        rhs_sample = blocks.sample_deleted[rhs_indices]
        lhs_modality = blocks.modality_deleted[lhs_indices]
        rhs_modality = blocks.modality_deleted[rhs_indices]
        sample_dist = pairwise_sqeuclidean(lhs_sample, rhs_sample)
        modality_dist = pairwise_sqeuclidean(lhs_modality, rhs_modality)
        if gamma_sample is None:
            gamma_sample = median_heuristic_gamma(
                pairwise_sqeuclidean(blocks.sample_deleted[rhs_indices], blocks.sample_deleted[rhs_indices])
            )
        if gamma_modality is None:
            gamma_modality = median_heuristic_gamma(
                pairwise_sqeuclidean(blocks.modality_deleted[rhs_indices], blocks.modality_deleted[rhs_indices])
            )
        kernel = np.exp(-float(gamma_sample) * sample_dist - float(gamma_modality) * modality_dist)
        return kernel, {"gamma_sample": float(gamma_sample), "gamma_modality": float(gamma_modality), "gamma_flat": None}

    if method_name == "kernel_surrogate_1_structured_rbf":
        resolved_shared_weight = 1.0 if modality_shared_weight is None else float(modality_shared_weight)
        lhs_sample = blocks.sample_deleted[lhs_indices]
        rhs_sample = blocks.sample_deleted[rhs_indices]
        sample_dist = pairwise_sqeuclidean(lhs_sample, rhs_sample)
        modality_dist = structured_modality_sqdistance(
            blocks,
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
            modality_shared_weight=resolved_shared_weight,
        )
        if gamma_sample is None:
            gamma_sample = median_heuristic_gamma(
                pairwise_sqeuclidean(blocks.sample_deleted[rhs_indices], blocks.sample_deleted[rhs_indices])
            )
        if gamma_modality is None:
            gamma_modality = median_heuristic_gamma(
                structured_modality_sqdistance(
                    blocks,
                    lhs_indices=rhs_indices,
                    rhs_indices=rhs_indices,
                    modality_shared_weight=resolved_shared_weight,
                )
            )
        kernel = np.exp(-float(gamma_sample) * sample_dist - float(gamma_modality) * modality_dist)
        return kernel, {
            "gamma_sample": float(gamma_sample),
            "gamma_modality": float(gamma_modality),
            "gamma_flat": None,
            "modality_shared_weight": resolved_shared_weight,
        }

    if method_name == "raw_flat_rbf":
        lhs = blocks.flat_deleted[lhs_indices]
        rhs = blocks.flat_deleted[rhs_indices]
        dist = pairwise_sqeuclidean(lhs, rhs)
        if gamma_flat is None:
            gamma_flat = median_heuristic_gamma(pairwise_sqeuclidean(blocks.flat_deleted[rhs_indices], blocks.flat_deleted[rhs_indices]))
        kernel = np.exp(-float(gamma_flat) * dist)
        return kernel, {"gamma_sample": None, "gamma_modality": None, "gamma_flat": float(gamma_flat)}

    if method_name in {"kernel_surrogate_1_linear", "kernel_surrogate_2_linear"}:
        lhs_sample = blocks.sample_deleted[lhs_indices]
        rhs_sample = blocks.sample_deleted[rhs_indices]
        lhs_modality = blocks.modality_deleted[lhs_indices]
        rhs_modality = blocks.modality_deleted[rhs_indices]
        kernel = lhs_sample @ rhs_sample.T + lhs_modality @ rhs_modality.T
        return kernel.astype(np.float64, copy=False), {"gamma_sample": None, "gamma_modality": None, "gamma_flat": None}

    if method_name == "influence_function_linear":
        # First-order influence baseline: additive effects over flattened
        # sample-modality deletion atoms with ridge regularization.
        lhs_flat = blocks.flat_deleted[lhs_indices]
        rhs_flat = blocks.flat_deleted[rhs_indices]
        kernel = lhs_flat @ rhs_flat.T
        return kernel.astype(np.float64, copy=False), {"gamma_sample": None, "gamma_modality": None, "gamma_flat": None}

    if method_name in {"kernel_surrogate_hier_linear", "kernel_surrogate_hier_centered_linear"}:
        sample_kernel, modality_kernel = _normalized_hierarchical_linear_kernel(
            blocks,
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
            centered=method_name == "kernel_surrogate_hier_centered_linear",
        )
        resolved_weight = 1.0 if linear_modality_weight is None else float(linear_modality_weight)
        kernel = sample_kernel + resolved_weight * modality_kernel
        return kernel.astype(np.float64, copy=False), {
            "gamma_sample": None,
            "gamma_modality": None,
            "gamma_flat": None,
            "linear_modality_weight": resolved_weight,
        }

    raise ValueError(f"unsupported method {method!r}")


def normalize_method_name(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_")
    aliases = {
        "ks1": "kernel_surrogate_1_rbf",
        "ks1_rbf": "kernel_surrogate_1_rbf",
        "hierarchical_rbf_v1": "kernel_surrogate_1_rbf",
        "ks1_structured": "kernel_surrogate_1_structured_rbf",
        "ks1_structured_rbf": "kernel_surrogate_1_structured_rbf",
        "structured_rbf_v1": "kernel_surrogate_1_structured_rbf",
        "hierarchical_structured_rbf_v1": "kernel_surrogate_1_structured_rbf",
        "ks2": "kernel_surrogate_2_rbf",
        "ks2_rbf": "kernel_surrogate_2_rbf",
        "hierarchical_rbf_v2": "kernel_surrogate_2_rbf",
        "flat_rbf": "raw_flat_rbf",
        "raw_rbf": "raw_flat_rbf",
        "ks1_linear": "kernel_surrogate_1_linear",
        "ks2_linear": "kernel_surrogate_2_linear",
        "if": "influence_function_linear",
        "if_linear": "influence_function_linear",
        "influence": "influence_function_linear",
        "influence_function": "influence_function_linear",
        "hlinear": "kernel_surrogate_hier_linear",
        "hier_linear": "kernel_surrogate_hier_linear",
        "hierarchical_linear": "kernel_surrogate_hier_linear",
        "balanced_linear": "kernel_surrogate_hier_linear",
        "hcenter_linear": "kernel_surrogate_hier_centered_linear",
        "hier_centered_linear": "kernel_surrogate_hier_centered_linear",
        "centered_hier_linear": "kernel_surrogate_hier_centered_linear",
    }
    return aliases.get(normalized, normalized)


def pairwise_sqeuclidean(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs64 = np.asarray(lhs, dtype=np.float64)
    rhs64 = np.asarray(rhs, dtype=np.float64)
    lhs_sq = np.sum(lhs64 * lhs64, axis=1, keepdims=True)
    rhs_sq = np.sum(rhs64 * rhs64, axis=1, keepdims=True).T
    dist = lhs_sq + rhs_sq - 2.0 * lhs64 @ rhs64.T
    return np.maximum(dist, 0.0)


def structured_modality_sqdistance(
    blocks: MaskBlocks,
    *,
    lhs_indices: Sequence[int] | np.ndarray,
    rhs_indices: Sequence[int] | np.ndarray,
    modality_shared_weight: float,
) -> np.ndarray:
    base_dist = pairwise_sqeuclidean(
        blocks.modality_deleted[np.asarray(lhs_indices, dtype=np.int64)],
        blocks.modality_deleted[np.asarray(rhs_indices, dtype=np.int64)],
    )
    if float(modality_shared_weight) == 0.0:
        return base_dist
    shared_dist = structured_modality_shared_sqdistance(
        blocks,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
    )
    return base_dist + float(modality_shared_weight) * shared_dist


def structured_modality_shared_sqdistance(
    blocks: MaskBlocks,
    *,
    lhs_indices: Sequence[int] | np.ndarray,
    rhs_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Pairwise distance term induced by the 11^T component of A_m."""
    lhs_indices = np.asarray(lhs_indices, dtype=np.int64)
    rhs_indices = np.asarray(rhs_indices, dtype=np.int64)
    lhs = blocks.modality_deleted[lhs_indices].reshape(lhs_indices.size, blocks.dataset_size, blocks.num_modalities)
    rhs = blocks.modality_deleted[rhs_indices].reshape(rhs_indices.size, blocks.dataset_size, blocks.num_modalities)
    lhs_shared = lhs.sum(axis=2)
    rhs_shared = rhs.sum(axis=2)
    return pairwise_sqeuclidean(lhs_shared, rhs_shared)


def median_heuristic_gamma(pairwise_distances: np.ndarray) -> float:
    distances = np.asarray(pairwise_distances, dtype=np.float64)
    positive = distances[distances > 1e-12]
    if positive.size == 0:
        return 1.0
    median = float(np.median(positive))
    if not math.isfinite(median) or median <= 0.0:
        return 1.0
    return 1.0 / median


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def spearmanr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_rank = rankdata(np.asarray(lhs, dtype=np.float64))
    rhs_rank = rankdata(np.asarray(rhs, dtype=np.float64))
    lhs_centered = lhs_rank - lhs_rank.mean()
    rhs_centered = rhs_rank - rhs_rank.mean()
    denom = float(np.sqrt(np.sum(lhs_centered**2) * np.sum(rhs_centered**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(lhs_centered * rhs_centered) / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def evaluations_to_records(
    *,
    dataset: str,
    seed: int,
    evaluations: dict[str, MethodEvaluation],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method, evaluation in evaluations.items():
        records.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "rmse": float(evaluation.rmse),
                "lds": float(evaluation.lds),
                "num_train": int(evaluation.num_train),
                "num_validation": int(evaluation.num_validation),
                "num_test": int(evaluation.num_test),
                "train_indices": [int(index) for index in evaluation.train_indices.tolist()],
                "validation_indices": [int(index) for index in evaluation.validation_indices.tolist()],
                "test_indices": [int(index) for index in evaluation.test_indices.tolist()],
                "ridge_lambda": float(evaluation.fit.ridge_lambda),
                "loo_lds": evaluation.fit.loo_lds,
                "loo_rmse": evaluation.fit.loo_rmse,
                "validation_lds": evaluation.fit.validation_lds,
                "validation_rmse": evaluation.fit.validation_rmse,
                "test_search_lds": evaluation.fit.test_search_lds,
                "test_search_rmse": evaluation.fit.test_search_rmse,
                "gamma_sample": evaluation.fit.gamma_sample,
                "gamma_modality": evaluation.fit.gamma_modality,
                "gamma_flat": evaluation.fit.gamma_flat,
                "modality_shared_weight": evaluation.fit.modality_shared_weight,
                "linear_modality_weight": evaluation.fit.linear_modality_weight,
            }
        )
    return records


def summarize_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(str(record["method"]), []).append(record)

    summary: list[dict[str, Any]] = []
    for method, method_records in sorted(by_method.items()):
        rmses = np.asarray([float(record["rmse"]) for record in method_records], dtype=np.float64)
        lds_values = np.asarray([float(record["lds"]) for record in method_records], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "num_dataset_runs": int(len(method_records)),
                "mean_rmse": float(np.nanmean(rmses)),
                "std_rmse": float(np.nanstd(rmses)),
                "mean_lds": float(np.nanmean(lds_values)),
                "std_lds": float(np.nanstd(lds_values)),
            }
        )
    return summary


def _infer_dataset_size(group_indices: Iterable[Sequence[int]], runs: Sequence[dict[str, Any]]) -> int:
    max_index = -1
    total_from_groups = 0
    for indices in group_indices:
        if not indices:
            continue
        total_from_groups += len(indices)
        local_max = max(int(index) for index in indices)
        max_index = max(max_index, local_max)
    for run in runs:
        data_indices = run.get("data_indices", [])
        if data_indices:
            max_index = max(max_index, max(int(index) for index in data_indices))
    return max(max_index + 1, total_from_groups)


def _build_index_to_group(group_indices: Iterable[Sequence[int]]) -> dict[int, int]:
    index_to_group: dict[int, int] = {}
    for group_id, indices in enumerate(group_indices):
        for index in indices:
            index_to_group[int(index)] = int(group_id)
    return index_to_group


def _loo_predictions_from_kernel(kernel: np.ndarray, targets: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.empty(targets.shape[0], dtype=np.float64)
    for holdout in range(targets.shape[0]):
        keep = np.ones(targets.shape[0], dtype=bool)
        keep[holdout] = False
        fit_kernel = kernel[np.ix_(keep, keep)]
        query_kernel = kernel[holdout, keep]
        fit_targets = targets[keep]
        target_mean = float(fit_targets.mean())
        centered_targets = fit_targets - target_mean
        regularized = fit_kernel + float(ridge_lambda) * np.eye(fit_kernel.shape[0], dtype=np.float64)
        alpha = np.linalg.solve(regularized, centered_targets)
        predictions[holdout] = target_mean + query_kernel @ alpha
    return predictions


def _query_predictions_from_kernel(
    train_kernel: np.ndarray,
    query_train_kernel: np.ndarray,
    train_targets: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    train_kernel = np.asarray(train_kernel, dtype=np.float64)
    query_train_kernel = np.asarray(query_train_kernel, dtype=np.float64)
    train_targets = np.asarray(train_targets, dtype=np.float64)
    target_mean = float(train_targets.mean())
    centered_targets = train_targets - target_mean
    regularized = train_kernel + float(ridge_lambda) * np.eye(train_kernel.shape[0], dtype=np.float64)
    alpha = np.linalg.solve(regularized, centered_targets)
    return target_mean + query_train_kernel @ alpha


def _normalized_hierarchical_linear_kernel(
    blocks: MaskBlocks,
    *,
    lhs_indices: Sequence[int] | np.ndarray,
    rhs_indices: Sequence[int] | np.ndarray,
    centered: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    lhs_indices = np.asarray(lhs_indices, dtype=np.int64)
    rhs_indices = np.asarray(rhs_indices, dtype=np.int64)
    lhs_sample = blocks.sample_deleted[lhs_indices]
    rhs_sample = blocks.sample_deleted[rhs_indices]
    lhs_modality = blocks.modality_deleted[lhs_indices]
    rhs_modality = blocks.modality_deleted[rhs_indices]
    if centered:
        sample_center = rhs_sample.mean(axis=0, keepdims=True)
        modality_center = rhs_modality.mean(axis=0, keepdims=True)
        lhs_sample = lhs_sample - sample_center
        rhs_sample = rhs_sample - sample_center
        lhs_modality = lhs_modality - modality_center
        rhs_modality = rhs_modality - modality_center
    sample_kernel = (lhs_sample @ rhs_sample.T) / float(max(blocks.dataset_size, 1))
    modality_kernel = (lhs_modality @ rhs_modality.T) / float(max(blocks.dataset_size * blocks.num_modalities, 1))
    return sample_kernel.astype(np.float64, copy=False), modality_kernel.astype(np.float64, copy=False)


def _rbf_kernel_from_arrays(lhs: np.ndarray, rhs: np.ndarray, *, gamma: float) -> np.ndarray:
    return np.exp(-float(gamma) * pairwise_sqeuclidean(lhs, rhs))


def _hierarchical_rbf_kernel_from_arrays(
    lhs_sample: np.ndarray,
    lhs_modality: np.ndarray,
    rhs_sample: np.ndarray,
    rhs_modality: np.ndarray,
    *,
    gamma_sample: float,
    gamma_modality: float,
) -> np.ndarray:
    sample_dist = pairwise_sqeuclidean(lhs_sample, rhs_sample)
    modality_dist = pairwise_sqeuclidean(lhs_modality, rhs_modality)
    return np.exp(-float(gamma_sample) * sample_dist - float(gamma_modality) * modality_dist)


def _require_gamma(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required for this attribution query")
    return float(value)
