from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .sks import sks_masks_and_metric
except ImportError:
    from sks import sks_masks_and_metric


def build_group_mask_blocks(payload: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    partition = payload["group_partition"]
    num_groups = int(partition["num_groups"])
    num_modalities = int(partition.get("num_modalities", 2))
    if num_modalities != 2:
        raise ValueError(f"expected two modalities, got {num_modalities}")
    runs = payload["runs"]
    sample_deleted = np.ones((len(runs), num_groups), dtype=np.float64)
    modality_deleted = np.zeros((len(runs), num_groups * num_modalities), dtype=np.float64)
    flat_deleted = np.ones((len(runs), num_groups * num_modalities), dtype=np.float64)
    targets = np.empty(len(runs), dtype=np.float64)
    group_to_indices = {
        group_id: np.asarray(indices, dtype=np.int64)
        for group_id, indices in enumerate(partition["group_indices"])
    }
    for row, run in enumerate(runs):
        targets[row] = float(run.get("test_loss", run.get("loss")))
        data_indices = np.asarray(run["data_indices"], dtype=np.int64)
        keep_mask = np.asarray(run["sample_modality_mask"], dtype=bool)
        index_to_local = {int(index): local for local, index in enumerate(data_indices.tolist())}
        for group_id in [int(value) for value in run["group_ids"]]:
            sample_deleted[row, group_id] = 0.0
            local_rows = [
                index_to_local[int(index)]
                for index in group_to_indices[group_id].tolist()
                if int(index) in index_to_local
            ]
            if not local_rows:
                continue
            group_keep = keep_mask[np.asarray(local_rows, dtype=np.int64)].mean(axis=0)
            deleted = 1.0 - group_keep
            modality_deleted[row, group_id * 2 : group_id * 2 + 2] = deleted
            flat_deleted[row, group_id * 2 : group_id * 2 + 2] = deleted
    return {
        "sample_deleted": sample_deleted,
        "modality_deleted": modality_deleted,
        "flat_deleted": flat_deleted,
        "num_groups": np.asarray([num_groups], dtype=np.int64),
    }, targets


def metric_sqdist(lhs: np.ndarray, rhs: np.ndarray, metric: np.ndarray) -> np.ndarray:
    lhs_metric = lhs @ metric
    lhs_sq = np.sum(lhs_metric * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum((rhs @ metric) * rhs, axis=1, keepdims=True).T
    dist = np.maximum(lhs_sq + rhs_sq - 2.0 * lhs_metric @ rhs.T, 0.0)
    return np.nan_to_num(dist, nan=0.0, posinf=np.finfo(np.float64).max / 4.0, neginf=0.0)


def median_gamma(distances: np.ndarray) -> float:
    values = distances[np.triu_indices(distances.shape[0], k=1)]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0
    return 1.0 / float(np.median(values))


def split_indices(num_items: int, *, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_items < 2:
        raise ValueError("need at least two LDS runs")
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_items)
    num_train = min(max(1, int(round(num_items * train_fraction))), num_items - 1)
    return (
        np.sort(order[:num_train]).astype(np.int64, copy=False),
        np.sort(order[num_train:]).astype(np.int64, copy=False),
    )


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearmanr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if lhs.size < 2 or rhs.size != lhs.size:
        return float("nan")
    lhs_rank = _rankdata(lhs)
    rhs_rank = _rankdata(rhs)
    lhs_centered = lhs_rank - lhs_rank.mean()
    rhs_centered = rhs_rank - rhs_rank.mean()
    denom = float(np.linalg.norm(lhs_centered) * np.linalg.norm(rhs_centered))
    if denom <= 0.0:
        return float("nan")
    return float(lhs_centered @ rhs_centered / denom)


def fit_sks_surrogate(
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    *,
    rho: float,
    ridge_lambda: float,
    gamma_scale: float,
    train_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    if train_indices is None:
        train_indices = np.arange(len(targets), dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    train_masks, metric = sks_masks_and_metric(mask_blocks, train_indices=train_indices, rho=float(rho))
    distances = metric_sqdist(train_masks, train_masks, metric)
    base_gamma = median_gamma(distances)
    gamma = float(base_gamma * float(gamma_scale))
    kernel = np.exp(-gamma * distances)
    target_mean = float(np.mean(targets[train_indices]))
    alpha = np.linalg.solve(
        kernel + float(ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64),
        targets[train_indices] - target_mean,
    )
    train_predictions = target_mean + kernel @ alpha
    return {
        "method": "sks",
        "alpha": alpha,
        "target_mean": target_mean,
        "gamma": gamma,
        "base_gamma": float(base_gamma),
        "train_predictions": train_predictions,
        "train_indices": train_indices.copy(),
        "train_masks": train_masks.copy(),
        "metric": metric.copy(),
        "atom_offset": int(mask_blocks["sample_deleted"].shape[1]),
    }


def predict_sks(
    mask_blocks: dict[str, np.ndarray],
    fit: dict[str, Any],
    *,
    rho: float,
    query_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    query_indices = np.asarray(query_indices, dtype=np.int64)
    query_masks, _ = sks_masks_and_metric(mask_blocks, train_indices=query_indices, rho=float(rho))
    distances = metric_sqdist(query_masks, fit["train_masks"], fit["metric"])
    kernel = np.exp(-float(fit["gamma"]) * distances)
    return float(fit["target_mean"]) + kernel @ fit["alpha"]


def second_derivative_scores(
    *,
    train_masks: np.ndarray,
    metric: np.ndarray,
    alpha: np.ndarray,
    gamma: float,
    num_groups: int,
    atom_offset: int,
) -> np.ndarray:
    anchor = np.zeros(train_masks.shape[1], dtype=np.float64)
    diffs = anchor[None, :] - train_masks
    q = diffs @ metric
    kernel = np.exp(-float(gamma) * np.einsum("mi,ij,mj->m", diffs, metric, diffs))
    weighted = alpha * kernel
    scores = np.empty((num_groups, num_groups), dtype=np.float64)
    for u in range(num_groups):
        atom_u = int(atom_offset) + 2 * u
        for v in range(num_groups):
            atom_v = int(atom_offset) + 2 * v + 1
            scores[u, v] = float(
                np.sum(
                    weighted
                    * (
                        4.0 * float(gamma) ** 2 * q[:, atom_u] * q[:, atom_v]
                        - 2.0 * float(gamma) * metric[atom_u, atom_v]
                    )
                )
            )
    return scores


def normalize_pair_scores(scores: np.ndarray, *, eps: float) -> np.ndarray:
    diag = np.diag(scores).astype(np.float64, copy=False)
    denom = np.sqrt(np.maximum(np.abs(diag[:, None] * diag[None, :]), float(eps)))
    return np.asarray(scores, dtype=np.float64) / denom


def group_score_payload(pair_scores: np.ndarray) -> dict[str, list[float]]:
    finite = np.nan_to_num(pair_scores, nan=0.0, posinf=0.0, neginf=0.0)
    abs_scores = np.abs(finite)
    return {
        "x": abs_scores.sum(axis=1).astype(float).tolist(),
        "y": abs_scores.sum(axis=0).astype(float).tolist(),
        "combined": (abs_scores.sum(axis=1) + abs_scores.sum(axis=0)).astype(float).tolist(),
        "x_signed": finite.sum(axis=1).astype(float).tolist(),
        "y_signed": finite.sum(axis=0).astype(float).tolist(),
        "combined_signed": (finite.sum(axis=1) + finite.sum(axis=0)).astype(float).tolist(),
    }


def top_pairs(scores: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
    num_groups = scores.shape[0]
    rows, cols = np.where(~np.eye(num_groups, dtype=bool))
    values = np.nan_to_num(scores[rows, cols], nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(-np.abs(values), kind="mergesort")
    pairs: list[dict[str, Any]] = []
    for index in order[: min(int(top_k), order.size)]:
        pairs.append(
            {
                "pair": [int(rows[index]), int(cols[index])],
                "score": float(values[index]),
                "abs_score": float(abs(values[index])),
            }
        )
    return pairs


def finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=np.float64)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def fit_and_score(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    mask_blocks, targets = build_group_mask_blocks(payload)
    if args.train_fraction < 1.0:
        train_indices, test_indices = split_indices(
            len(targets),
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
        )
    else:
        train_indices = np.arange(len(targets), dtype=np.int64)
        test_indices = train_indices
    fit = fit_sks_surrogate(
        mask_blocks,
        targets,
        rho=float(args.rho),
        ridge_lambda=float(args.ridge_lambda),
        gamma_scale=float(args.gamma_scale),
        train_indices=train_indices,
    )
    predictions = predict_sks(mask_blocks, fit, rho=float(args.rho), query_indices=test_indices)
    raw_scores = second_derivative_scores(
        train_masks=fit["train_masks"],
        metric=fit["metric"],
        alpha=fit["alpha"],
        gamma=float(fit["gamma"]),
        num_groups=int(mask_blocks["num_groups"][0]),
        atom_offset=int(fit["atom_offset"]),
    )
    pair_scores = normalize_pair_scores(raw_scores, eps=float(args.normalization_eps))
    partition = payload["group_partition"]
    return {
        "metadata": {
            "method": "sks",
            "input": str(args.input),
            "dataset": partition.get("dataset_name"),
            "num_groups": int(partition["num_groups"]),
            "num_runs": int(len(targets)),
            "rho": float(args.rho),
            "ridge_lambda": float(args.ridge_lambda),
            "gamma_scale": float(args.gamma_scale),
            "train_fraction": float(args.train_fraction),
            "seed": int(args.seed),
            "target": "test_loss",
        },
        "fit": {
            "target_mean": float(fit["target_mean"]),
            "gamma": float(fit["gamma"]),
            "base_gamma": float(fit["base_gamma"]),
            "train_rmse": rmse(fit["train_predictions"], targets[train_indices]),
            "train_spearman": spearmanr(fit["train_predictions"], targets[train_indices]),
            "eval_rmse": rmse(predictions, targets[test_indices]),
            "eval_spearman": spearmanr(predictions, targets[test_indices]),
            "num_train": int(train_indices.size),
            "num_eval": int(test_indices.size),
        },
        "pair_score_summary": finite_summary(pair_scores),
        "top_pairs_by_abs_score": top_pairs(pair_scores, top_k=int(args.top_k)),
        "pair_scores": pair_scores.astype(float).tolist(),
        "raw_pair_scores": raw_scores.astype(float).tolist(),
        "modality_group_scores": group_score_payload(pair_scores),
        "group_partition": {
            "group_sizes": [int(value) for value in partition.get("group_sizes", [])],
            "group_method": partition.get("group_method"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an SKS kernel surrogate and export group attribution scores."
    )
    parser.add_argument("--input", type=Path, required=True, help="LDS group payload JSON.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--gamma-scale", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalization-eps", type=float, default=1e-12)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.train_fraction) <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1]")
    if not math.isfinite(float(args.rho)):
        raise ValueError("--rho must be finite")
    output = fit_and_score(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    print(f"saved SKS attribution scores to {args.output}", flush=True)


if __name__ == "__main__":
    main()

