"""Generate Gentle Push group indices for downstream data selection.

This follows the repository's existing group-selection protocol: fit a kernel
surrogate to LDS subset losses, sample additional LDS-like masks, estimate each
group's conditional loss reduction, and export the top-k whole groups.  Both
the hierarchical structured RBF and raw-flat RBF are exported under the same
query distribution and budget, together with a random-group baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_surrogate.kernel_attribution import (  # noqa: E402
    MaskBlocks,
    build_mask_blocks,
    fit_best_kernel_surrogate,
    fit_kernel_surrogate,
    load_result_payload,
    predict_kernel_surrogate,
)


DATASET_NAME = "gentle_push_lf"
DEFAULT_RUN_NAME = (
    "gentle_push_lf_group_unit_kmeans_auxacq40_mod50_group_"
    "main-control_g25_trainavg3"
)
DEFAULT_RESULT_ROOT = (
    REPO_ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "src"
    / "kernel_surrogate"
    / "results"
    / "gentle_push_auxacq_data_selection_indices"
)
METHODS = {
    "structured_rbf": "kernel_surrogate_1_structured_rbf",
    "raw_flat_rbf": "raw_flat_rbf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--surrogate-seed", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--selection-budget",
        type=float,
        default=0.2,
        help=(
            "Exact fraction of training samples receiving auxiliary modalities. "
            "Samples are drawn from the highest-ranked groups. Set to 0 to keep "
            "whole --top-k groups instead."
        ),
    )
    parser.add_argument("--num-score-samples", type=int, default=20000)
    parser.add_argument(
        "--score-group-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of groups receiving all auxiliary modalities in surrogate "
            "scoring queries. Set independently from the richer attribution-training "
            "mask distribution; 0.2 matches the downstream selection budget."
        ),
    )
    parser.add_argument("--prediction-batch-size", type=int, default=5000)
    parser.add_argument("--sampling-seed", type=int, default=12345)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--search-hyperparams",
        action="store_true",
        help="Choose each surrogate's kernel parameters by leave-one-run-out LDS.",
    )
    parser.add_argument(
        "--ridge-grid",
        nargs="+",
        type=float,
        default=[1e-5, 1e-4, 1e-3, 1e-2],
    )
    parser.add_argument(
        "--gamma-scale-grid",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--modality-shared-grid",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--structured-shared-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the symmetric cross-modality term. One uses the structured "
            "modality geometry; zero collapses to raw-flat geometry when every "
            "sample remains present."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _sample_query_blocks(
    *,
    num_samples: int,
    num_groups: int,
    num_modalities: int,
    group_sizes: list[int],
    groups_per_subset: int,
    modality_keep_rate: float,
    seed: int,
    intervention: str = "subset_retrain",
    main_modality_index: int = 0,
) -> tuple[MaskBlocks, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    selected = np.zeros((num_samples, num_groups), dtype=bool)
    if intervention == "auxiliary_acquisition":
        sample_deleted = np.zeros((num_samples, num_groups), dtype=np.float32)
        modality_deleted = np.zeros(
            (num_samples, num_groups, num_modalities), dtype=np.float32
        )
        aux_indices = [
            index for index in range(num_modalities) if index != int(main_modality_index)
        ]
        modality_deleted[:, :, aux_indices] = 1.0
        for row in range(num_samples):
            group_ids = rng.choice(
                num_groups, size=int(groups_per_subset), replace=False
            )
            selected[row, group_ids] = True
            modality_deleted[row, group_ids[:, None], aux_indices] = 0.0
        flat_deleted = modality_deleted.reshape(
            num_samples, num_groups * num_modalities
        ).copy()
        return (
            MaskBlocks(
                sample_deleted=sample_deleted,
                modality_deleted=modality_deleted.reshape(
                    num_samples, num_groups * num_modalities
                ),
                flat_deleted=flat_deleted,
                targets=np.full(num_samples, np.nan, dtype=np.float64),
                run_ids=tuple(range(num_samples)),
                dataset_size=num_groups,
                num_modalities=num_modalities,
            ),
            selected,
        )

    sample_deleted = np.ones((num_samples, num_groups), dtype=np.float32)
    modality_deleted = np.zeros(
        (num_samples, num_groups * num_modalities), dtype=np.float32
    )
    flat_deleted = np.ones(
        (num_samples, num_groups * num_modalities), dtype=np.float32
    )

    for row in range(num_samples):
        group_ids = rng.choice(
            num_groups, size=int(groups_per_subset), replace=False
        )
        selected[row, group_ids] = True
        sample_deleted[row, group_ids] = 0.0
        for group_id in group_ids.tolist():
            group_size = int(group_sizes[group_id])
            kept_counts = rng.binomial(
                group_size,
                float(modality_keep_rate),
                size=num_modalities,
            )
            deletion_rates = 1.0 - kept_counts.astype(np.float32) / max(group_size, 1)
            sl = slice(
                int(group_id) * num_modalities,
                (int(group_id) + 1) * num_modalities,
            )
            modality_deleted[row, sl] = deletion_rates
            flat_deleted[row, sl] = deletion_rates

    return (
        MaskBlocks(
            sample_deleted=sample_deleted,
            modality_deleted=modality_deleted,
            flat_deleted=flat_deleted,
            targets=np.full(num_samples, np.nan, dtype=np.float64),
            run_ids=tuple(range(num_samples)),
            dataset_size=num_groups,
            num_modalities=num_modalities,
        ),
        selected,
    )


def _combine_blocks(train: MaskBlocks, query: MaskBlocks) -> MaskBlocks:
    if train.dataset_size != query.dataset_size:
        raise ValueError("train/query dataset_size mismatch")
    if train.num_modalities != query.num_modalities:
        raise ValueError("train/query num_modalities mismatch")
    num_train = int(train.targets.shape[0])
    num_query = int(query.targets.shape[0])
    return MaskBlocks(
        sample_deleted=np.concatenate(
            [train.sample_deleted, query.sample_deleted], axis=0
        ),
        modality_deleted=np.concatenate(
            [train.modality_deleted, query.modality_deleted], axis=0
        ),
        flat_deleted=np.concatenate([train.flat_deleted, query.flat_deleted], axis=0),
        targets=np.concatenate([train.targets, query.targets], axis=0),
        run_ids=tuple(range(num_train + num_query)),
        dataset_size=train.dataset_size,
        num_modalities=train.num_modalities,
    )


def _predict_queries(
    blocks: MaskBlocks,
    *,
    method: str,
    num_train: int,
    batch_size: int,
    ridge_lambda: float,
    shared_weight: float,
    search_hyperparams: bool,
    ridge_grid: list[float],
    gamma_scale_grid: list[float],
    modality_shared_grid: list[float],
) -> tuple[np.ndarray, Any]:
    train_indices = np.arange(num_train, dtype=np.int64)
    if search_hyperparams:
        fit = fit_best_kernel_surrogate(
            blocks,
            train_indices=train_indices,
            method=method,
            ridge_grid=tuple(float(value) for value in ridge_grid),
            gamma_scale_grid=tuple(float(value) for value in gamma_scale_grid),
            modality_shared_grid=tuple(
                float(value) for value in modality_shared_grid
            ),
        )
    else:
        fit = fit_kernel_surrogate(
            blocks,
            train_indices=train_indices,
            method=method,
            ridge_lambda=float(ridge_lambda),
            modality_shared_weight=(
                float(shared_weight)
                if method == "kernel_surrogate_1_structured_rbf"
                else None
            ),
        )
    predictions = np.empty(blocks.targets.shape[0] - num_train, dtype=np.float64)
    for start in range(num_train, blocks.targets.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), blocks.targets.shape[0])
        query_indices = np.arange(start, stop, dtype=np.int64)
        predictions[start - num_train : stop - num_train] = predict_kernel_surrogate(
            blocks,
            fit,
            query_indices=query_indices,
        )
    return predictions, fit


def _loss_reduction_scores(
    predictions: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    present_counts = selected.sum(axis=0).astype(np.int64)
    absent_counts = (~selected).sum(axis=0).astype(np.int64)
    if np.any(present_counts == 0) or np.any(absent_counts == 0):
        raise ValueError("every group must occur both present and absent in score samples")
    present_loss = selected.astype(np.float64).T @ predictions / present_counts
    absent_loss = (~selected).astype(np.float64).T @ predictions / absent_counts
    return absent_loss - present_loss, present_loss, absent_loss


def _indices_for_groups(
    group_indices: list[list[int]], group_ids: list[int]
) -> list[int]:
    selected: list[int] = []
    for group_id in group_ids:
        selected.extend(int(index) for index in group_indices[int(group_id)])
    return sorted(set(selected))


def _budgeted_indices_from_ranked_groups(
    *,
    group_indices: list[list[int]],
    ranked_group_ids: list[int],
    target_count: int,
    seed: int,
    min_group_count: int = 0,
) -> tuple[list[int], list[int], int]:
    candidate_groups: list[int] = []
    candidate_indices: list[int] = []
    for group_id in ranked_group_ids:
        candidate_groups.append(int(group_id))
        candidate_indices.extend(int(index) for index in group_indices[int(group_id)])
        if (
            len(candidate_indices) >= int(target_count)
            and len(candidate_groups) >= int(min_group_count)
        ):
            break
    if len(candidate_indices) < int(target_count):
        raise ValueError(
            f"ranked groups contain {len(candidate_indices)} samples, "
            f"less than target_count={target_count}"
        )
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(
        np.asarray(candidate_indices, dtype=np.int64),
        size=int(target_count),
        replace=False,
    )
    return (
        sorted(int(value) for value in chosen.tolist()),
        candidate_groups,
        len(candidate_indices),
    )


def _write_index(
    *,
    output_root: Path,
    method: str,
    group_ids: list[int],
    group_scores: list[float],
    group_indices: list[list[int]],
    args: argparse.Namespace,
    selected_indices: list[int] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> Path:
    indices = (
        _indices_for_groups(group_indices, group_ids)
        if selected_indices is None
        else [int(value) for value in selected_indices]
    )
    path = output_root / method / DATASET_NAME / f"seed{int(args.selection_seed)}.json"
    payload = {
        "selected_indices": indices,
        "selected_group_ids": [int(value) for value in group_ids],
        "selected_group_scores": [float(value) for value in group_scores],
        "selected_fraction": float(
            len(indices) / max(sum(len(values) for values in group_indices), 1)
        ),
        "metadata": {
            "dataset": DATASET_NAME,
            "surrogate_seed": int(args.surrogate_seed),
            "data_selection_seed": int(args.selection_seed),
            "surrogate_method": method,
            "score": (
                "mean predicted test_loss(group absent) - "
                "mean predicted test_loss(group present)"
            ),
            "direction": "highest_loss_reduction",
            "selection_unit": "whole_groups",
            "top_k": int(args.top_k),
            "selection_budget": float(args.selection_budget),
            "run_name": str(args.run_name),
            "ridge_lambda": float(args.ridge_lambda),
            "structured_shared_weight": float(args.structured_shared_weight),
            **(metadata_extra or {}),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    result_dir = args.result_root / args.run_name
    payload = load_result_payload(result_dir, DATASET_NAME, int(args.surrogate_seed))
    train_blocks = build_mask_blocks(
        payload,
        target_metric="test_loss",
        input_unit="group",
    )
    partition = payload["group_partition"]
    num_groups = int(partition["num_groups"])
    num_modalities = int(partition["num_modalities"])
    group_sizes = [int(value) for value in partition["group_sizes"]]
    group_indices = [
        [int(index) for index in values]
        for values in partition["group_indices"]
    ]
    if not 0 < int(args.top_k) <= num_groups:
        raise ValueError(f"top_k must be in [1, {num_groups}], got {args.top_k}")
    if not 0.0 <= float(args.selection_budget) <= 1.0:
        raise ValueError(
            f"selection_budget must be in [0, 1], got {args.selection_budget}"
        )
    dataset_size = int(sum(group_sizes))
    target_count = (
        max(1, int(round(dataset_size * float(args.selection_budget))))
        if float(args.selection_budget) > 0.0
        else None
    )

    config = payload["config"]
    intervention = str(config.get("intervention", "subset_retrain"))
    if intervention == "auxiliary_acquisition":
        score_group_fraction = float(args.score_group_fraction)
        if not 0.0 < score_group_fraction <= 1.0:
            raise ValueError(
                "score_group_fraction must be in (0, 1] for "
                f"auxiliary_acquisition, got {score_group_fraction}"
            )
        groups_per_subset = int(np.ceil(num_groups * score_group_fraction))
    else:
        score_group_fraction = float(config["data_keep_rate"])
        groups_per_subset = int(np.ceil(num_groups * score_group_fraction))
    main_modality = str(config.get("main_modality", "control"))
    modality_names = [str(value) for value in partition["modality_names"]]
    if main_modality not in modality_names:
        raise ValueError(
            f"main modality {main_modality!r} is not in payload modalities {modality_names}"
        )
    query_blocks, selected = _sample_query_blocks(
        num_samples=int(args.num_score_samples),
        num_groups=num_groups,
        num_modalities=num_modalities,
        group_sizes=group_sizes,
        groups_per_subset=groups_per_subset,
        modality_keep_rate=float(config["modality_keep_rate"]),
        seed=int(args.sampling_seed),
        intervention=intervention,
        main_modality_index=modality_names.index(main_modality),
    )
    blocks = _combine_blocks(train_blocks, query_blocks)
    num_train = int(train_blocks.targets.shape[0])

    summary: dict[str, Any] = {
        "config": {
            "dataset": DATASET_NAME,
            "run_name": str(args.run_name),
            "surrogate_seed": int(args.surrogate_seed),
            "selection_seed": int(args.selection_seed),
            "top_k": int(args.top_k),
            "num_score_samples": int(args.num_score_samples),
            "groups_per_subset": int(groups_per_subset),
            "score_group_fraction": float(score_group_fraction),
            "intervention": intervention,
            "main_modality": main_modality,
            "num_groups": int(num_groups),
            "num_modalities": int(num_modalities),
            "selection_budget": float(args.selection_budget),
            "target_selected_count": target_count,
        },
        "methods": {},
    }
    for output_name, method in METHODS.items():
        print(f"[{output_name}] predicting {args.num_score_samples} query masks...", flush=True)
        predictions, fit = _predict_queries(
            blocks,
            method=method,
            num_train=num_train,
            batch_size=int(args.prediction_batch_size),
            ridge_lambda=float(args.ridge_lambda),
            shared_weight=float(args.structured_shared_weight),
            search_hyperparams=bool(args.search_hyperparams),
            ridge_grid=[float(value) for value in args.ridge_grid],
            gamma_scale_grid=[float(value) for value in args.gamma_scale_grid],
            modality_shared_grid=[float(value) for value in args.modality_shared_grid],
        )
        scores, present_loss, absent_loss = _loss_reduction_scores(
            predictions, selected
        )
        order = np.argsort(-scores, kind="stable")
        ranked_group_ids = [int(value) for value in order.tolist()]
        if target_count is None:
            group_ids = ranked_group_ids[: int(args.top_k)]
            selected_indices = _indices_for_groups(group_indices, group_ids)
            candidate_pool_size = len(selected_indices)
        else:
            selected_indices, group_ids, candidate_pool_size = (
                _budgeted_indices_from_ranked_groups(
                    group_indices=group_indices,
                    ranked_group_ids=ranked_group_ids,
                    target_count=int(target_count),
                    seed=int(args.selection_seed) * 1009 + 17,
                    min_group_count=int(args.top_k),
                )
            )
        path = _write_index(
            output_root=args.output_root,
            method=output_name,
            group_ids=group_ids,
            group_scores=[float(scores[group_id]) for group_id in group_ids],
            group_indices=group_indices,
            args=args,
            selected_indices=selected_indices,
            metadata_extra={
                "mean_query_prediction": float(predictions.mean()),
                "candidate_pool_size": int(candidate_pool_size),
                "target_selected_count": target_count,
                "fit_ridge_lambda": float(fit.ridge_lambda),
                "fit_gamma_sample": fit.gamma_sample,
                "fit_gamma_modality": fit.gamma_modality,
                "fit_gamma_flat": fit.gamma_flat,
                "fit_modality_shared_weight": fit.modality_shared_weight,
                "fit_loo_lds": fit.loo_lds,
                "selection_unit": (
                    "whole_groups"
                    if target_count is None
                    else "random_indices_within_ranked_groups"
                ),
            },
        )
        score_rows = [
            {
                "group_id": int(group_id),
                "loss_reduction": float(scores[group_id]),
                "mean_loss_present": float(present_loss[group_id]),
                "mean_loss_absent": float(absent_loss[group_id]),
                "group_size": int(group_sizes[group_id]),
            }
            for group_id in range(num_groups)
        ]
        scores_path = args.output_root / "scores" / f"{output_name}_seed{args.surrogate_seed}.json"
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(json.dumps(score_rows, indent=2), encoding="utf-8")
        summary["methods"][output_name] = {
            "selected_group_ids": group_ids,
            "selected_group_scores": [float(scores[group_id]) for group_id in group_ids],
            "index_path": str(path),
            "scores_path": str(scores_path),
            "fit": {
                "ridge_lambda": float(fit.ridge_lambda),
                "gamma_sample": fit.gamma_sample,
                "gamma_modality": fit.gamma_modality,
                "gamma_flat": fit.gamma_flat,
                "modality_shared_weight": fit.modality_shared_weight,
                "loo_lds": fit.loo_lds,
                "loo_rmse": fit.loo_rmse,
            },
        }
        print(f"  groups={group_ids} -> {path}", flush=True)

    rng = np.random.default_rng(int(args.selection_seed) + 3000)
    random_ranking = [
        int(value)
        for value in rng.permutation(num_groups).tolist()
    ]
    if target_count is None:
        random_groups = sorted(random_ranking[: int(args.top_k)])
        random_indices = _indices_for_groups(group_indices, random_groups)
        random_pool_size = len(random_indices)
    else:
        random_indices, random_groups, random_pool_size = (
            _budgeted_indices_from_ranked_groups(
                group_indices=group_indices,
                ranked_group_ids=random_ranking,
                target_count=int(target_count),
                seed=int(args.selection_seed) + 4000,
                min_group_count=int(args.top_k),
            )
        )
    random_path = _write_index(
        output_root=args.output_root,
        method="random_group",
        group_ids=random_groups,
        group_scores=[],
        group_indices=group_indices,
        args=args,
        selected_indices=random_indices,
        metadata_extra={
            "selection_seed": int(args.selection_seed) + 3000,
            "candidate_pool_size": int(random_pool_size),
            "target_selected_count": target_count,
            "selection_unit": (
                "whole_groups"
                if target_count is None
                else "random_indices_within_ranked_groups"
            ),
        },
    )
    summary["methods"]["random_group"] = {
        "selected_group_ids": random_groups,
        "selected_group_scores": [],
        "index_path": str(random_path),
    }

    summary_path = args.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[random_group] groups={random_groups} -> {random_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
