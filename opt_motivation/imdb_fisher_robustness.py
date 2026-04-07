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

from motivation.core import write_json
from opt_motivation import default_output_dir
from opt_motivation.analyze_imdb_sample_level import (
    _compute_local_fisher_stratification_analysis,
    _compute_neighborhood_residual_analysis,
    _compute_pair_concordance_analysis,
    json_load,
)


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="IMDb Fisher robustness analysis across neighborhood sizes.")
    parser.add_argument("--runs-dir", type=str, default=str(output_dir / "multiseed_runs"))
    parser.add_argument("--output-json", type=str, default=str(output_dir / "imdb_fisher_robustness.json"))
    parser.add_argument("--neighborhood-k", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--top-k", type=int, default=25)
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
    neg_adv = np.asarray(features["negative_loss_advantage"], dtype=np.float32)[valid_mask]
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
        "negative_loss_advantage": neg_adv,
    }


def _metric_snapshot(
    residual: dict[str, Any],
    local: dict[str, Any],
    pair: dict[str, Any],
) -> dict[str, float]:
    return {
        "residual_rho": float(residual["fisher_rank_vs_residual"]["correctness"]["rho"]),
        "residual_quartile_delta": float(residual["quartile_contrast"]["correctness_residual_by_fisher_rank"]["delta"]),
        "local_mean_delta": float(local["correctness_delta"]["mean"]),
        "local_nonzero_positive_fraction": float(local["correctness_delta"]["signed_fraction_test"]["positive_fraction"]),
        "pair_all_concordance": float(pair["filtered_correctness_concordance"]["all_nonzero_pairs"]["concordance_rate"]),
        "pair_q25_concordance": float(pair["filtered_correctness_concordance"]["tight_pid_nonzero_pairs"]["concordance_rate"]),
        "pair_q10_concordance": float(pair["filtered_correctness_concordance"]["very_tight_pid_nonzero_pairs"]["concordance_rate"]),
    }


def _run_analysis(record: dict[str, Any], neighborhood_k: int, top_k: int) -> dict[str, Any]:
    imdb_ids = np.asarray([int(item) for item in record["sample_ids"]], dtype=np.int64)
    residual = _compute_neighborhood_residual_analysis(
        pid_features=record["pid"],
        fisher=record["fisher"],
        correctness=record["correctness"],
        neg_adv=record["negative_loss_advantage"],
        neighborhood_k=neighborhood_k,
    )
    local = _compute_local_fisher_stratification_analysis(
        pid_features=record["pid"],
        fisher=record["fisher"],
        correctness=record["correctness"],
        neg_adv=record["negative_loss_advantage"],
        imdb_ids=imdb_ids,
        neighborhood_k=neighborhood_k,
        top_k=top_k,
    )
    pair = _compute_pair_concordance_analysis(
        pid_features=record["pid"],
        fisher=record["fisher"],
        correctness=record["correctness"],
        neg_adv=record["negative_loss_advantage"],
        imdb_ids=imdb_ids,
        neighborhood_k=neighborhood_k,
        top_k=top_k,
    )
    return {
        "residual": residual,
        "local": local,
        "pair": pair,
        "metrics": _metric_snapshot(residual, local, pair),
    }


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
            raise ValueError("Seed sample IDs do not align for robustness analysis.")

    aggregate_record = {
        "seed": -1,
        "sample_ids": reference_ids,
        "pid_feature_names": seed_records[0]["pid_feature_names"],
        "pid": np.mean(np.stack([record["pid"] for record in seed_records], axis=0), axis=0),
        "fisher": np.mean(np.stack([record["fisher"] for record in seed_records], axis=0), axis=0),
        "correctness": np.mean(np.stack([record["correctness"] for record in seed_records], axis=0), axis=0),
        "negative_loss_advantage": np.mean(
            np.stack([record["negative_loss_advantage"] for record in seed_records], axis=0),
            axis=0,
        ),
    }

    per_k: dict[str, Any] = {}
    for neighborhood_k in args.neighborhood_k:
        seed_runs = []
        metric_rows = []
        for record in seed_records:
            result = _run_analysis(record, neighborhood_k=neighborhood_k, top_k=args.top_k)
            seed_runs.append({"seed": record["seed"], **result})
            metric_rows.append(result["metrics"])
        aggregate_result = _run_analysis(aggregate_record, neighborhood_k=neighborhood_k, top_k=args.top_k)

        metric_names = list(metric_rows[0].keys())
        per_k[str(neighborhood_k)] = {
            "per_seed": seed_runs,
            "aggregate_across_seeds": {
                name: _mean_std([row[name] for row in metric_rows]) for name in metric_names
            },
            "mean_feature_analysis": aggregate_result,
        }

    write_json(
        {
            "dataset": "imdb",
            "seeds": [record["seed"] for record in seed_records],
            "pid_feature_names": seed_records[0]["pid_feature_names"],
            "neighborhood_k_values": args.neighborhood_k,
            "results_by_k": per_k,
        },
        Path(args.output_json),
    )


if __name__ == "__main__":
    main()
