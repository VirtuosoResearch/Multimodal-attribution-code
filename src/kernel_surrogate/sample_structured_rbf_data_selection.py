from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_surrogate.real_data_cmis import (
    build_group_mask_blocks,
    fit_surrogate_method,
    predict_surrogate_method,
)
from kernel_surrogate.structured_rbf_best_hparams import get_best_structured_rbf_hparams


DEFAULT_DATASETS = ("humor", "imdb", "mosi", "mosei", "sarcasm")
DEFAULT_RESULT_ROOT = REPO_ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src" / "kernel_surrogate" / "results" / "structured_rbf_data_selection_indices"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the best structured_rbf kernel surrogate and sample it to rank "
            "groups for data_selection index-file inputs."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--surrogate-seed", type=int, default=0)
    parser.add_argument(
        "--data-selection-seeds",
        nargs="+",
        type=int,
        default=[1],
        help="Seed filenames to write for src.data_selection index-template usage.",
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-score-samples", type=int, default=20000)
    parser.add_argument("--prediction-batch-size", type=int, default=5000)
    parser.add_argument("--sampling-seed", type=int, default=12345)
    parser.add_argument(
        "--selection-query-mode",
        choices=["group_score", "budget_allocation"],
        default="group_score",
        help=(
            "group_score ranks individual groups from LDS-like sampled masks. "
            "budget_allocation samples complete data-selection allocations and "
            "scores those masks directly with the surrogate."
        ),
    )
    parser.add_argument(
        "--main-modality",
        choices=["x", "y"],
        default="y",
        help="Main modality kept for every training sample in budget_allocation mode.",
    )
    parser.add_argument(
        "--selection-budget",
        type=float,
        default=None,
        help=(
            "Optional data-selection budget used only to derive how many groups "
            "to export when --select-group-count is not set."
        ),
    )
    parser.add_argument(
        "--select-group-count",
        type=int,
        default=None,
        help="Number of highest/lowest scoring groups to export. Defaults to 1.",
    )
    parser.add_argument(
        "--score-conditioning",
        choices=["present", "absent"],
        default="present",
        help=(
            "Score a group by the mean surrogate output when the group is present "
            "in sampled subsets, or when it is absent. Used only when "
            "--group-score-mode=conditional_mean_loss."
        ),
    )
    parser.add_argument(
        "--group-score-mode",
        choices=["conditional_mean_loss", "loss_reduction"],
        default="conditional_mean_loss",
        help=(
            "conditional_mean_loss ranks groups by mean predicted loss under "
            "--score-conditioning. loss_reduction ranks by "
            "mean_loss(group absent) - mean_loss(group present), so larger is "
            "better for selecting useful groups."
        ),
    )
    return parser.parse_args()


def _load_payload(result_root: Path, run_name: str, dataset: str, seed: int) -> dict[str, Any]:
    path = result_root / run_name / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing LDS payload: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _groups_per_subset(num_groups: int, data_keep_rate: float) -> int:
    keep_rate = float(data_keep_rate)
    if not 0.0 < keep_rate <= 1.0:
        raise ValueError(f"data_keep_rate must be in (0, 1], got {data_keep_rate}")
    return min(num_groups, max(1, int(math.ceil(num_groups * keep_rate))))


def _selection_group_count(
    *,
    num_groups: int,
    select_group_count: int | None,
    selection_budget: float | None,
) -> int:
    if select_group_count is not None:
        count = int(select_group_count)
    elif selection_budget is not None:
        if not 0.0 < float(selection_budget) <= 1.0:
            raise ValueError(f"selection_budget must be in (0, 1], got {selection_budget}")
        count = int(math.ceil(num_groups * float(selection_budget)))
    else:
        count = 1
    return min(num_groups, max(1, count))


def _target_index_count(dataset_size: int, selection_budget: float | None) -> int | None:
    if selection_budget is None:
        return None
    budget = float(selection_budget)
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"selection_budget must be in (0, 1], got {selection_budget}")
    if budget == 1.0:
        return int(dataset_size)
    return min(int(dataset_size), max(1, int(round(int(dataset_size) * budget))))


def _sample_query_blocks(
    *,
    num_groups: int,
    group_sizes: Sequence[int],
    num_samples: int,
    groups_per_subset: int,
    modality_keep_rate: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = np.zeros((num_samples, num_groups), dtype=bool)
    sample_deleted = np.ones((num_samples, num_groups), dtype=np.float64)
    modality_deleted = np.zeros((num_samples, num_groups * 2), dtype=np.float64)
    flat_deleted = np.ones((num_samples, num_groups * 2), dtype=np.float64)
    group_sizes_array = np.asarray(group_sizes, dtype=np.int64)
    if group_sizes_array.shape != (num_groups,):
        raise ValueError(
            f"group_sizes length {group_sizes_array.shape[0]} does not match num_groups={num_groups}"
        )
    keep_prob = float(modality_keep_rate)
    if not 0.0 < keep_prob <= 1.0:
        raise ValueError(f"modality_keep_rate must be in (0, 1], got {modality_keep_rate}")

    for row in range(num_samples):
        groups = rng.choice(num_groups, size=groups_per_subset, replace=False)
        selected[row, groups] = True
        sample_deleted[row, groups] = 0.0
        for group_id in groups.tolist():
            size = int(group_sizes_array[group_id])
            kept = rng.binomial(size, keep_prob, size=2).astype(np.float64) / max(size, 1)
            deleted = 1.0 - kept
            sl = slice(group_id * 2, group_id * 2 + 2)
            modality_deleted[row, sl] = deleted
            flat_deleted[row, sl] = deleted

    return (
        {
            "sample_deleted": sample_deleted,
            "modality_deleted": modality_deleted,
            "flat_deleted": flat_deleted,
            "num_groups": np.asarray([num_groups], dtype=np.int64),
        },
        selected,
    )


def _predict_in_batches(
    query_blocks: dict[str, np.ndarray],
    fit: dict[str, Any],
    *,
    batch_size: int,
) -> np.ndarray:
    num_rows = int(query_blocks["sample_deleted"].shape[0])
    identity_metric = np.eye(query_blocks["modality_deleted"].shape[1], dtype=np.float64)
    predictions = np.empty(num_rows, dtype=np.float64)
    for start in range(0, num_rows, batch_size):
        stop = min(start + int(batch_size), num_rows)
        predictions[start:stop] = predict_surrogate_method(
            "structured_rbf",
            query_blocks,
            fit,
            metric=identity_metric,
            query_indices=np.arange(start, stop, dtype=np.int64),
        )
    return predictions


def _index_to_group(group_indices: Sequence[Sequence[int]], dataset_size: int) -> np.ndarray:
    mapping = np.full(int(dataset_size), -1, dtype=np.int64)
    for group_id, indices in enumerate(group_indices):
        mapping[np.asarray(indices, dtype=np.int64)] = int(group_id)
    if np.any(mapping < 0):
        missing = np.flatnonzero(mapping < 0)[:10].tolist()
        raise ValueError(f"group partition does not cover dataset indices, e.g. {missing}")
    return mapping


def _sample_budget_allocation_counts(
    *,
    group_indices: Sequence[Sequence[int]],
    group_sizes: Sequence[int],
    dataset_size: int,
    target_count: int,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    index_to_group = _index_to_group(group_indices, dataset_size)
    counts = np.zeros((int(num_samples), len(group_sizes)), dtype=np.int64)
    for row in range(int(num_samples)):
        chosen = rng.choice(int(dataset_size), size=int(target_count), replace=False)
        counts[row] = np.bincount(index_to_group[chosen], minlength=len(group_sizes))

    # Include single-group candidates. They are cheap and important when the
    # surrogate strongly prefers concentrating the auxiliary budget.
    for group_id, group_size in enumerate(group_sizes):
        if group_id >= counts.shape[0]:
            break
        counts[group_id] = 0
        remaining = int(target_count)
        for candidate_group in [group_id, *range(len(group_sizes))]:
            if remaining <= 0:
                break
            add = min(int(group_sizes[candidate_group]), remaining)
            counts[group_id, candidate_group] += add
            remaining -= add
    return counts


def _allocation_query_blocks(
    *,
    counts: np.ndarray,
    group_sizes: Sequence[int],
    main_modality: str,
) -> dict[str, np.ndarray]:
    counts = np.asarray(counts, dtype=np.float64)
    group_sizes_array = np.asarray(group_sizes, dtype=np.float64)
    num_samples, num_groups = counts.shape
    if group_sizes_array.shape != (num_groups,):
        raise ValueError("allocation count width must match group_sizes")

    aux_keep_rate = counts / np.maximum(group_sizes_array[None, :], 1.0)
    sample_deleted = np.zeros((num_samples, num_groups), dtype=np.float64)
    modality_deleted = np.zeros((num_samples, num_groups * 2), dtype=np.float64)
    flat_deleted = np.zeros((num_samples, num_groups * 2), dtype=np.float64)
    aux_modality = 0 if main_modality == "y" else 1
    main_modality_index = 1 - aux_modality
    for group_id in range(num_groups):
        aux_deleted = 1.0 - aux_keep_rate[:, group_id]
        modality_deleted[:, group_id * 2 + aux_modality] = aux_deleted
        flat_deleted[:, group_id * 2 + aux_modality] = aux_deleted
        modality_deleted[:, group_id * 2 + main_modality_index] = 0.0
        flat_deleted[:, group_id * 2 + main_modality_index] = 0.0
    return {
        "sample_deleted": sample_deleted,
        "modality_deleted": modality_deleted,
        "flat_deleted": flat_deleted,
        "num_groups": np.asarray([num_groups], dtype=np.int64),
    }


def _indices_from_group_counts(
    *,
    group_indices: Sequence[Sequence[int]],
    counts: Sequence[int],
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for group_id, count in enumerate(counts):
        resolved_count = int(count)
        if resolved_count <= 0:
            continue
        candidates = np.asarray(group_indices[group_id], dtype=np.int64)
        if resolved_count > candidates.size:
            raise ValueError(
                f"requested {resolved_count} samples from group {group_id}, "
                f"but group has {candidates.size}"
            )
        chosen = rng.choice(candidates, size=resolved_count, replace=False)
        selected.extend(int(index) for index in chosen.tolist())
    return sorted(selected)


def _group_scores(
    predictions: np.ndarray,
    selected: np.ndarray,
    *,
    conditioning: str,
) -> tuple[np.ndarray, np.ndarray]:
    condition = selected if conditioning == "present" else ~selected
    counts = condition.sum(axis=0).astype(np.int64)
    if np.any(counts == 0):
        bad = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"some groups were never observed under conditioning={conditioning}: {bad}")
    scores = (condition.astype(np.float64).T @ predictions.astype(np.float64)) / counts
    return scores, counts


def _compute_group_scores(
    predictions: np.ndarray,
    selected: np.ndarray,
    *,
    score_mode: str,
    conditioning: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if score_mode == "conditional_mean_loss":
        scores, counts = _group_scores(predictions, selected, conditioning=conditioning)
        return scores, counts, f"mean predicted test_loss when group is {conditioning}"
    if score_mode == "loss_reduction":
        present_scores, present_counts = _group_scores(predictions, selected, conditioning="present")
        absent_scores, absent_counts = _group_scores(predictions, selected, conditioning="absent")
        return (
            absent_scores - present_scores,
            np.minimum(present_counts, absent_counts),
            "mean predicted test_loss(group absent) - mean predicted test_loss(group present)",
        )
    raise ValueError(f"unsupported group score mode: {score_mode}")


def _direction_names(score_mode: str) -> tuple[str, str]:
    if score_mode == "loss_reduction":
        return "lowest_loss_reduction", "highest_loss_reduction"
    return "lowest_predicted_loss", "highest_predicted_loss"


def _indices_for_groups(group_indices: Sequence[Sequence[int]], group_ids: Sequence[int]) -> list[int]:
    indices: list[int] = []
    for group_id in group_ids:
        indices.extend(int(index) for index in group_indices[int(group_id)])
    return sorted(set(indices))


def _budgeted_indices_from_ranked_groups(
    *,
    group_indices: Sequence[Sequence[int]],
    ranked_group_ids: Sequence[int],
    target_count: int,
    seed: int,
) -> tuple[list[int], list[int], int]:
    candidate_group_ids: list[int] = []
    candidates: list[int] = []
    seen: set[int] = set()
    for group_id in ranked_group_ids:
        candidate_group_ids.append(int(group_id))
        for index in group_indices[int(group_id)]:
            value = int(index)
            if value not in seen:
                candidates.append(value)
                seen.add(value)
        if len(candidates) >= int(target_count):
            break
    if len(candidates) < int(target_count):
        raise ValueError(
            f"ranked groups only provide {len(candidates)} indices, "
            f"but target_count={target_count}"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(candidates, dtype=np.int64), size=int(target_count), replace=False)
    return sorted(int(index) for index in chosen.tolist()), candidate_group_ids, len(candidates)


def _write_index_files(
    *,
    output_root: Path,
    direction: str,
    dataset: str,
    data_selection_seeds: Sequence[int],
    payloads_by_seed: dict[int, dict[str, Any]],
) -> list[str]:
    written: list[str] = []
    for seed in data_selection_seeds:
        path = output_root / direction / dataset / f"seed{int(seed)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payloads_by_seed[int(seed)], indent=2), encoding="utf-8")
        written.append(str(path))
    return written


def _process_budget_allocation(
    args: argparse.Namespace,
    *,
    dataset: str,
    hparams: dict[str, Any],
    payload: dict[str, Any],
    fit: dict[str, Any],
    group_sizes: Sequence[int],
    group_indices: Sequence[Sequence[int]],
) -> dict[str, Any]:
    dataset_size = int(sum(int(size) for size in group_sizes))
    target_count = _target_index_count(dataset_size, args.selection_budget)
    if target_count is None:
        raise ValueError("budget_allocation mode requires --selection-budget")

    counts = _sample_budget_allocation_counts(
        group_indices=group_indices,
        group_sizes=group_sizes,
        dataset_size=dataset_size,
        target_count=int(target_count),
        num_samples=int(args.num_score_samples),
        seed=int(args.sampling_seed) + stable_dataset_offset(dataset),
    )
    query_blocks = _allocation_query_blocks(
        counts=counts,
        group_sizes=group_sizes,
        main_modality=str(args.main_modality),
    )
    predictions = _predict_in_batches(
        query_blocks,
        fit,
        batch_size=int(args.prediction_batch_size),
    )
    low_order = np.argsort(predictions, kind="mergesort")
    best_row = int(low_order[0])
    worst_row = int(low_order[-1])
    low_direction = "lowest_allocation_loss"
    high_direction = "highest_allocation_loss"

    def _payloads_for(row: int, direction: str, seed_offset: int) -> dict[int, dict[str, Any]]:
        row_counts = counts[int(row)].astype(np.int64, copy=False)
        selected_groups = [int(group_id) for group_id, count in enumerate(row_counts.tolist()) if int(count) > 0]
        return {
            int(seed): {
                "selected_indices": _indices_from_group_counts(
                    group_indices=group_indices,
                    counts=row_counts,
                    seed=int(args.sampling_seed) + stable_dataset_offset(dataset) + int(seed) * 1009 + seed_offset,
                ),
                "selected_group_ids": selected_groups,
                "selected_group_counts": [int(row_counts[group_id]) for group_id in selected_groups],
                "predicted_loss": float(predictions[int(row)]),
                "allocation_row": int(row),
                "metadata": {
                    "dataset": dataset,
                    "surrogate_seed": int(args.surrogate_seed),
                    "surrogate_method": "structured_rbf",
                    "surrogate_target": "test_loss",
                    "selection_query_mode": "budget_allocation",
                    "main_modality": str(args.main_modality),
                    "budgeted_modality": "x" if str(args.main_modality) == "y" else "y",
                    "target_selected_count": int(target_count),
                    "num_train_samples": int(dataset_size),
                    "direction": direction,
                    "data_selection_seed": int(seed),
                    "hparams": {
                        "run_name": str(hparams["run_name"]),
                        "setting": str(hparams["setting"]),
                        "ridge_lambda": float(hparams["ridge_lambda"]),
                        "gamma_scale": float(hparams["gamma_scale"]),
                        "structured_shared_weight": float(hparams["structured_shared_weight"]),
                    },
                    "sampling": {
                        "num_score_samples": int(args.num_score_samples),
                        "sampling_seed": int(args.sampling_seed) + stable_dataset_offset(dataset),
                    },
                },
            }
            for seed in args.data_selection_seeds
        }

    low_payloads = _payloads_for(best_row, low_direction, 41)
    high_payloads = _payloads_for(worst_row, high_direction, 53)
    for payloads in (low_payloads, high_payloads):
        for seed_payload in payloads.values():
            seed_payload["selected_fraction"] = float(
                len(seed_payload["selected_indices"]) / max(dataset_size, 1)
            )

    low_paths = _write_index_files(
        output_root=args.output_root,
        direction=low_direction,
        dataset=dataset,
        data_selection_seeds=args.data_selection_seeds,
        payloads_by_seed=low_payloads,
    )
    high_paths = _write_index_files(
        output_root=args.output_root,
        direction=high_direction,
        dataset=dataset,
        data_selection_seeds=args.data_selection_seeds,
        payloads_by_seed=high_payloads,
    )

    scores_path = (
        args.output_root
        / "scores"
        / f"{dataset}_budget_allocation_surrogate_seed{int(args.surrogate_seed)}.json"
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "dataset": dataset,
                    "selection_query_mode": "budget_allocation",
                    "main_modality": str(args.main_modality),
                    "target_selected_count": int(target_count),
                    "num_score_samples": int(args.num_score_samples),
                },
                "prediction_summary": {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions)),
                    "best_row": int(best_row),
                    "worst_row": int(worst_row),
                },
                "best_allocation": {
                    "predicted_loss": float(predictions[best_row]),
                    "group_counts": counts[best_row].astype(int).tolist(),
                },
                "worst_allocation": {
                    "predicted_loss": float(predictions[worst_row]),
                    "group_counts": counts[worst_row].astype(int).tolist(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    first_seed = int(args.data_selection_seeds[0])
    return {
        "dataset": dataset,
        low_direction: {
            "group_ids": low_payloads[first_seed]["selected_group_ids"],
            "predicted_loss": float(predictions[best_row]),
            "num_indices": int(len(low_payloads[first_seed]["selected_indices"])),
            "paths": low_paths,
        },
        high_direction: {
            "group_ids": high_payloads[first_seed]["selected_group_ids"],
            "predicted_loss": float(predictions[worst_row]),
            "num_indices": int(len(high_payloads[first_seed]["selected_indices"])),
            "paths": high_paths,
        },
        "low_direction": low_direction,
        "high_direction": high_direction,
        "scores_path": str(scores_path),
    }


def process_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    hparams = get_best_structured_rbf_hparams(dataset)
    payload = _load_payload(
        args.result_root,
        str(hparams["run_name"]),
        dataset,
        int(args.surrogate_seed),
    )
    mask_blocks, targets = build_group_mask_blocks(payload)
    identity_metric = np.eye(mask_blocks["modality_deleted"].shape[1], dtype=np.float64)
    fit = fit_surrogate_method(
        "structured_rbf",
        mask_blocks,
        targets,
        identity_metric,
        ridge_lambda=float(hparams["ridge_lambda"]),
        gamma_scale=float(hparams["gamma_scale"]),
        structured_shared_weight=float(hparams["structured_shared_weight"]),
    )

    num_groups = int(payload["group_partition"]["num_groups"])
    group_sizes = [int(size) for size in payload["group_partition"]["group_sizes"]]
    group_indices = payload["group_partition"]["group_indices"]
    if str(args.selection_query_mode) == "budget_allocation":
        return _process_budget_allocation(
            args,
            dataset=dataset,
            hparams=hparams,
            payload=payload,
            fit=fit,
            group_sizes=group_sizes,
            group_indices=group_indices,
        )
    group_count = _selection_group_count(
        num_groups=num_groups,
        select_group_count=args.select_group_count,
        selection_budget=args.selection_budget,
    )
    query_blocks, selected = _sample_query_blocks(
        num_groups=num_groups,
        group_sizes=group_sizes,
        num_samples=int(args.num_score_samples),
        groups_per_subset=_groups_per_subset(
            num_groups,
            float(payload["config"].get("data_keep_rate", 1.0)),
        ),
        modality_keep_rate=float(payload["config"].get("modality_keep_rate", 1.0)),
        seed=int(args.sampling_seed) + stable_dataset_offset(dataset),
    )
    predictions = _predict_in_batches(
        query_blocks,
        fit,
        batch_size=int(args.prediction_batch_size),
    )
    scores, counts, score_description = _compute_group_scores(
        predictions,
        selected,
        score_mode=str(args.group_score_mode),
        conditioning=str(args.score_conditioning),
    )
    low_direction, high_direction = _direction_names(str(args.group_score_mode))
    order_low = np.argsort(scores, kind="mergesort")
    low_ranked_groups = [int(group_id) for group_id in order_low.tolist()]
    high_ranked_groups = [int(group_id) for group_id in order_low[::-1].tolist()]
    dataset_size = int(sum(group_sizes))
    target_count = _target_index_count(dataset_size, args.selection_budget)

    if target_count is None:
        low_groups = low_ranked_groups[:group_count]
        high_groups = high_ranked_groups[:group_count]
        low_seed_payloads = {
            int(seed): {
                "selected_indices": _indices_for_groups(group_indices, low_groups),
                "selected_group_ids": low_groups,
                "selected_group_scores": [float(scores[group_id]) for group_id in low_groups],
                "candidate_group_ids": low_groups,
                "candidate_pool_size": len(_indices_for_groups(group_indices, low_groups)),
            }
            for seed in args.data_selection_seeds
        }
        high_seed_payloads = {
            int(seed): {
                "selected_indices": _indices_for_groups(group_indices, high_groups),
                "selected_group_ids": high_groups,
                "selected_group_scores": [float(scores[group_id]) for group_id in high_groups],
                "candidate_group_ids": high_groups,
                "candidate_pool_size": len(_indices_for_groups(group_indices, high_groups)),
            }
            for seed in args.data_selection_seeds
        }
    else:
        low_seed_payloads = {}
        high_seed_payloads = {}
        for seed in args.data_selection_seeds:
            low_indices, low_groups, low_pool_size = _budgeted_indices_from_ranked_groups(
                group_indices=group_indices,
                ranked_group_ids=low_ranked_groups,
                target_count=target_count,
                seed=int(args.sampling_seed) + stable_dataset_offset(dataset) + int(seed) * 1009 + 17,
            )
            high_indices, high_groups, high_pool_size = _budgeted_indices_from_ranked_groups(
                group_indices=group_indices,
                ranked_group_ids=high_ranked_groups,
                target_count=target_count,
                seed=int(args.sampling_seed) + stable_dataset_offset(dataset) + int(seed) * 1009 + 31,
            )
            low_seed_payloads[int(seed)] = {
                "selected_indices": low_indices,
                "selected_group_ids": low_groups,
                "selected_group_scores": [float(scores[group_id]) for group_id in low_groups],
                "candidate_group_ids": low_groups,
                "candidate_pool_size": int(low_pool_size),
            }
            high_seed_payloads[int(seed)] = {
                "selected_indices": high_indices,
                "selected_group_ids": high_groups,
                "selected_group_scores": [float(scores[group_id]) for group_id in high_groups],
                "candidate_group_ids": high_groups,
                "candidate_pool_size": int(high_pool_size),
            }

    common_metadata = {
        "dataset": dataset,
        "surrogate_seed": int(args.surrogate_seed),
        "surrogate_method": "structured_rbf",
        "surrogate_target": "test_loss",
        "score": score_description,
        "hparams": {
            "run_name": str(hparams["run_name"]),
            "setting": str(hparams["setting"]),
            "ridge_lambda": float(hparams["ridge_lambda"]),
            "gamma_scale": float(hparams["gamma_scale"]),
            "structured_shared_weight": float(hparams["structured_shared_weight"]),
        },
        "sampling": {
            "num_score_samples": int(args.num_score_samples),
            "sampling_seed": int(args.sampling_seed) + stable_dataset_offset(dataset),
            "data_keep_rate": float(payload["config"].get("data_keep_rate", 1.0)),
            "modality_keep_rate": float(payload["config"].get("modality_keep_rate", 1.0)),
            "group_score_mode": str(args.group_score_mode),
            "score_conditioning": str(args.score_conditioning),
        },
        "selection_unit": "random_indices_within_ranked_groups" if target_count is not None else "whole_groups",
        "target_selected_count": None if target_count is None else int(target_count),
        "fallback_group_count": int(group_count),
        "num_train_samples": int(dataset_size),
    }
    for seed, seed_payload in low_seed_payloads.items():
        seed_payload["selected_fraction"] = float(len(seed_payload["selected_indices"]) / max(dataset_size, 1))
        seed_payload["metadata"] = {
            **common_metadata,
            "direction": low_direction,
            "data_selection_seed": int(seed),
        }
    for seed, seed_payload in high_seed_payloads.items():
        seed_payload["selected_fraction"] = float(len(seed_payload["selected_indices"]) / max(dataset_size, 1))
        seed_payload["metadata"] = {
            **common_metadata,
            "direction": high_direction,
            "data_selection_seed": int(seed),
        }
    low_paths = _write_index_files(
        output_root=args.output_root,
        direction=low_direction,
        dataset=dataset,
        data_selection_seeds=args.data_selection_seeds,
        payloads_by_seed=low_seed_payloads,
    )
    high_paths = _write_index_files(
        output_root=args.output_root,
        direction=high_direction,
        dataset=dataset,
        data_selection_seeds=args.data_selection_seeds,
        payloads_by_seed=high_seed_payloads,
    )

    scores_path = (
        args.output_root
        / "scores"
        / f"{dataset}_{args.group_score_mode}_surrogate_seed{int(args.surrogate_seed)}.json"
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    score_rows = [
        {
            "group_id": int(group_id),
            "score": float(scores[group_id]),
            "condition_count": int(counts[group_id]),
            "group_size": int(group_sizes[group_id]),
        }
        for group_id in range(num_groups)
    ]
    scores_path.write_text(
        json.dumps(
            {
                "metadata": common_metadata,
                "prediction_summary": {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions)),
                },
                "groups": score_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "dataset": dataset,
        low_direction: {
            "group_ids": low_seed_payloads[int(args.data_selection_seeds[0])]["selected_group_ids"],
            "scores": low_seed_payloads[int(args.data_selection_seeds[0])]["selected_group_scores"],
            "num_indices": int(len(low_seed_payloads[int(args.data_selection_seeds[0])]["selected_indices"])),
            "paths": low_paths,
        },
        high_direction: {
            "group_ids": high_seed_payloads[int(args.data_selection_seeds[0])]["selected_group_ids"],
            "scores": high_seed_payloads[int(args.data_selection_seeds[0])]["selected_group_scores"],
            "num_indices": int(len(high_seed_payloads[int(args.data_selection_seeds[0])]["selected_indices"])),
            "paths": high_paths,
        },
        "low_direction": low_direction,
        "high_direction": high_direction,
        "scores_path": str(scores_path),
    }


def stable_dataset_offset(dataset: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(str(dataset)))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        print(f"[{dataset}] fitting structured_rbf and sampling group scores...", flush=True)
        row = process_dataset(args, str(dataset))
        rows.append(row)
        low_direction = str(row["low_direction"])
        high_direction = str(row["high_direction"])
        low = row[low_direction]
        high = row[high_direction]
        low_value = low.get("scores", low.get("predicted_loss"))
        high_value = high.get("scores", high.get("predicted_loss"))
        print(
            f"[{dataset}] {low_direction} groups={low['group_ids']} value={low_value} "
            f"indices={low['num_indices']}; {high_direction} groups={high['group_ids']} "
            f"value={high_value} indices={high['num_indices']}",
            flush=True,
        )

    if rows:
        low_direction = str(rows[0]["low_direction"])
        high_direction = str(rows[0]["high_direction"])
    else:
        low_direction, high_direction = _direction_names(str(args.group_score_mode))
    summary = {
        "config": {
            "datasets": [str(dataset) for dataset in args.datasets],
            "surrogate_seed": int(args.surrogate_seed),
            "data_selection_seeds": [int(seed) for seed in args.data_selection_seeds],
            "num_score_samples": int(args.num_score_samples),
            "prediction_batch_size": int(args.prediction_batch_size),
            "selection_query_mode": str(args.selection_query_mode),
            "main_modality": str(args.main_modality),
            "selection_budget": args.selection_budget,
            "select_group_count": args.select_group_count,
            "group_score_mode": str(args.group_score_mode),
            "score_conditioning": str(args.score_conditioning),
            "result_root": str(args.result_root),
            "output_root": str(args.output_root),
        },
        "results": rows,
        "data_selection_index_templates": {
            low_direction: str(args.output_root / low_direction / "{dataset}" / "seed{seed}.json"),
            high_direction: str(args.output_root / high_direction / "{dataset}" / "seed{seed}.json"),
        },
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}", flush=True)
    print("Index templates:", flush=True)
    print(f"  {low_direction}:  {summary['data_selection_index_templates'][low_direction]}", flush=True)
    print(f"  {high_direction}: {summary['data_selection_index_templates'][high_direction]}", flush=True)


if __name__ == "__main__":
    main()
