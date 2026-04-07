from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from motivation.core import write_json
from opt_motivation import default_output_dir
from opt_motivation.analyze_imdb_sample_level import json_load


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="IMDb Fisher negative controls across multiseed runs.")
    parser.add_argument("--runs-dir", type=str, default=str(output_dir / "multiseed_runs"))
    parser.add_argument("--output-json", type=str, default=str(output_dir / "imdb_fisher_controls.json"))
    parser.add_argument("--neighborhood-k", type=int, default=25)
    parser.add_argument("--num-repeats", type=int, default=10)
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(finite) == 1:
        return {"mean": finite[0], "std": 0.0, "n": 1}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return {"mean": mean, "std": math.sqrt(max(var, 0.0)), "n": len(finite)}


def _extract_seed_features(payload: dict[str, Any]) -> dict[str, Any]:
    analysis = payload["analysis"]
    features = analysis["features"]
    valid_mask = np.asarray(analysis["valid_regression_mask"], dtype=bool)
    pid_names = list(features["pid_feature_names"])
    pid_feature_values = features["pid_feature_values"]
    pid = np.stack([np.asarray(pid_feature_values[name], dtype=np.float32) for name in pid_names], axis=1)[valid_mask]
    fisher = np.asarray(features["fisher_score"], dtype=np.float32)[valid_mask]
    correctness = np.asarray(analysis["correctness"], dtype=np.int64)[valid_mask].astype(np.float32)
    sample_ids = [str(item) for item in analysis.get("sample_ids", [])]
    if sample_ids:
        sample_ids = [item for item, keep in zip(sample_ids, valid_mask.tolist()) if keep]
    else:
        sample_ids = [str(index) for index in range(pid.shape[0])]
    return {
        "seed": int(payload["config"]["seed"]),
        "pid_feature_names": pid_names,
        "sample_ids": sample_ids,
        "pid": pid,
        "fisher": fisher,
        "correctness": correctness,
    }


def _zscore_columns(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std


def _build_pid_cache(pid_features: np.ndarray, neighborhood_k: int) -> dict[str, np.ndarray]:
    pid_z = _zscore_columns(np.asarray(pid_features, dtype=np.float32))
    model = NearestNeighbors(n_neighbors=min(neighborhood_k + 1, pid_z.shape[0]), metric="euclidean")
    model.fit(pid_z)
    distances, indices = model.kneighbors(pid_z)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    pair_left: list[int] = []
    pair_right: list[int] = []
    pair_pid_distance: list[float] = []
    seen: set[tuple[int, int]] = set()
    for left in range(indices.shape[0]):
        for right in indices[left]:
            pair = (left, int(right)) if left < int(right) else (int(right), left)
            if pair[0] == pair[1] or pair in seen:
                continue
            seen.add(pair)
            pair_left.append(pair[0])
            pair_right.append(pair[1])
            pair_pid_distance.append(float(np.linalg.norm(pid_z[pair[0]] - pid_z[pair[1]])))

    return {
        "indices": indices,
        "distances": distances,
        "pair_left": np.asarray(pair_left, dtype=np.int64),
        "pair_right": np.asarray(pair_right, dtype=np.int64),
        "pair_pid_distance": np.asarray(pair_pid_distance, dtype=np.float32),
    }


def _quartile_delta(values: np.ndarray, targets: np.ndarray) -> float:
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    low = targets[values <= q1]
    high = targets[values >= q3]
    if low.size == 0 or high.size == 0:
        return float("nan")
    return float(high.mean() - low.mean())


def _positive_fraction(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1e-8]
    if nonzero.size == 0:
        return float("nan")
    return float((nonzero > 0).mean())


def _concordance_rate(fisher_deltas: np.ndarray, outcome_deltas: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    concordant = np.sign(fisher_deltas[mask]) == np.sign(outcome_deltas[mask])
    return float(concordant.mean())


def _snapshot_from_cache(
    *,
    cache: dict[str, np.ndarray],
    fisher: np.ndarray,
    correctness: np.ndarray,
) -> dict[str, float]:
    indices = cache["indices"]
    local_correctness = correctness[indices].mean(axis=1)
    correctness_residual = correctness - local_correctness
    fisher_rank = (fisher[:, None] > fisher[indices]).mean(axis=1).astype(np.float32)
    rho = stats.spearmanr(fisher_rank, correctness_residual).statistic

    correctness_deltas: list[float] = []
    for anchor in range(indices.shape[0]):
        neighbor_idx = indices[anchor]
        neighbor_fisher = fisher[neighbor_idx]
        q1 = np.quantile(neighbor_fisher, 0.25)
        q3 = np.quantile(neighbor_fisher, 0.75)
        low_mask = neighbor_fisher <= q1
        high_mask = neighbor_fisher >= q3
        if int(low_mask.sum()) < 2 or int(high_mask.sum()) < 2:
            continue
        high_correctness = float(correctness[neighbor_idx[high_mask]].mean())
        low_correctness = float(correctness[neighbor_idx[low_mask]].mean())
        correctness_deltas.append(high_correctness - low_correctness)

    correctness_deltas_arr = np.asarray(correctness_deltas, dtype=np.float32)
    pair_left = cache["pair_left"]
    pair_right = cache["pair_right"]
    pair_pid_distance = cache["pair_pid_distance"]
    fisher_deltas = fisher[pair_left] - fisher[pair_right]
    outcome_deltas = correctness[pair_left] - correctness[pair_right]
    valid = (np.abs(fisher_deltas) > 1e-6) & (np.abs(outcome_deltas) > 1e-6)
    q25 = float(np.quantile(pair_pid_distance, 0.25)) if pair_pid_distance.size > 0 else float("nan")
    q10 = float(np.quantile(pair_pid_distance, 0.10)) if pair_pid_distance.size > 0 else float("nan")

    return {
        "residual_rho": float(rho),
        "residual_quartile_delta": _quartile_delta(fisher_rank, correctness_residual),
        "local_mean_delta": float(correctness_deltas_arr.mean()) if correctness_deltas_arr.size > 0 else float("nan"),
        "local_nonzero_positive_fraction": _positive_fraction(correctness_deltas_arr),
        "pair_all_concordance": _concordance_rate(fisher_deltas, outcome_deltas, valid),
        "pair_q25_concordance": _concordance_rate(
            fisher_deltas,
            outcome_deltas,
            valid & (pair_pid_distance <= q25),
        ),
        "pair_q10_concordance": _concordance_rate(
            fisher_deltas,
            outcome_deltas,
            valid & (pair_pid_distance <= q10),
        ),
    }


def _aggregate_metric_rows(metric_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    metric_names = list(metric_rows[0].keys())
    return {name: _mean_std([row[name] for row in metric_rows]) for name in metric_names}


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    payloads = [json_load(path) for path in sorted(runs_dir.glob("imdb_lsmi_fisher_seed*.json"))]
    if not payloads:
        raise FileNotFoundError(f"No IMDb runs found in {runs_dir}")

    seed_records = [_extract_seed_features(payload) for payload in payloads]
    reference_ids = seed_records[0]["sample_ids"]
    for record in seed_records[1:]:
        if record["sample_ids"] != reference_ids:
            raise ValueError("Seed sample IDs do not align for control analysis.")

    per_seed: list[dict[str, Any]] = []
    real_rows: list[dict[str, float]] = []
    fisher_shuffle_rows: list[dict[str, float]] = []
    pid_assignment_shuffle_rows: list[dict[str, float]] = []

    for record in seed_records:
        rng = np.random.default_rng(record["seed"])
        real_cache = _build_pid_cache(record["pid"], neighborhood_k=args.neighborhood_k)
        real_metrics = _snapshot_from_cache(
            cache=real_cache,
            fisher=record["fisher"],
            correctness=record["correctness"],
        )
        real_rows.append(real_metrics)

        fisher_shuffle_metrics: list[dict[str, float]] = []
        pid_assignment_shuffle_metrics: list[dict[str, float]] = []
        for _ in range(args.num_repeats):
            fisher_perm = rng.permutation(record["fisher"].shape[0])
            fisher_shuffle_metrics.append(
                _snapshot_from_cache(
                    cache=real_cache,
                    fisher=record["fisher"][fisher_perm],
                    correctness=record["correctness"],
                )
            )

            pid_perm = rng.permutation(record["pid"].shape[0])
            shuffled_cache = _build_pid_cache(record["pid"][pid_perm], neighborhood_k=args.neighborhood_k)
            pid_assignment_shuffle_metrics.append(
                _snapshot_from_cache(
                    cache=shuffled_cache,
                    fisher=record["fisher"],
                    correctness=record["correctness"],
                )
            )

        fisher_summary = _aggregate_metric_rows(fisher_shuffle_metrics)
        pid_assignment_summary = _aggregate_metric_rows(pid_assignment_shuffle_metrics)
        fisher_shuffle_rows.extend(fisher_shuffle_metrics)
        pid_assignment_shuffle_rows.extend(pid_assignment_shuffle_metrics)
        per_seed.append(
            {
                "seed": record["seed"],
                "real": real_metrics,
                "fisher_shuffle": fisher_summary,
                "pid_assignment_shuffle": pid_assignment_summary,
            }
        )

    write_json(
        {
            "dataset": "imdb",
            "seeds": [record["seed"] for record in seed_records],
            "neighborhood_k": args.neighborhood_k,
            "num_repeats": args.num_repeats,
            "per_seed": per_seed,
            "aggregate_across_seeds": {
                "real": _aggregate_metric_rows(real_rows),
                "fisher_shuffle": _aggregate_metric_rows(fisher_shuffle_rows),
                "pid_assignment_shuffle": _aggregate_metric_rows(pid_assignment_shuffle_rows),
            },
        },
        Path(args.output_json),
    )


if __name__ == "__main__":
    main()
