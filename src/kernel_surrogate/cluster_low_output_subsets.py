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
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_surrogate.real_data_cmis import (
    build_group_mask_blocks,
    fit_surrogate_method,
    predict_surrogate_method,
)
from kernel_surrogate.structured_rbf_best_hparams import (
    BEST_STRUCTURED_RBF_HPARAMS_80_20,
    get_best_structured_rbf_hparams,
)


DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_ROOT = ROOT / "src" / "kernel_surrogate" / "results" / "low_output_subset_clusters"
DEFAULT_CLUSTER_COUNTS = (10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the fixed best structured_rbf surrogate on existing LDS subset data, "
            "rank existing sampled subsets by RBF output, and save the sample-level "
            "indices for the top-k lower-output subsets."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=sorted(BEST_STRUCTURED_RBF_HPARAMS_80_20))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cluster-counts", nargs="+", type=int, default=list(DEFAULT_CLUSTER_COUNTS))
    return parser.parse_args()


def load_payload(result_root: Path, run_name: str, dataset: str, seed: int) -> dict[str, Any]:
    path = result_root / run_name / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing LDS payload: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fit_best_structured_rbf(payload: dict[str, Any], hparams: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    mask_blocks, targets = build_group_mask_blocks(payload)
    num_atoms = int(mask_blocks["modality_deleted"].shape[1])
    identity_metric = np.eye(num_atoms, dtype=np.float64)
    all_indices = np.arange(len(targets), dtype=np.int64)
    fit = fit_surrogate_method(
        "structured_rbf",
        mask_blocks,
        targets,
        identity_metric,
        ridge_lambda=float(hparams["ridge_lambda"]),
        gamma_scale=float(hparams["gamma_scale"]),
        structured_shared_weight=float(hparams["structured_shared_weight"]),
        train_indices=all_indices,
    )
    return fit, mask_blocks, targets


def predict_existing_subsets(fit: dict[str, Any], mask_blocks: dict[str, np.ndarray]) -> np.ndarray:
    num_atoms = int(mask_blocks["modality_deleted"].shape[1])
    identity_metric = np.eye(num_atoms, dtype=np.float64)
    indices = np.arange(mask_blocks["sample_deleted"].shape[0], dtype=np.int64)
    return np.asarray(
        predict_surrogate_method(
            "structured_rbf",
            mask_blocks,
            fit,
            metric=identity_metric,
            query_indices=indices,
        ),
        dtype=np.float64,
    )


def effective_sample_indices(run: dict[str, Any]) -> list[int]:
    data_indices = np.asarray(run["data_indices"], dtype=np.int64)
    keep_mask = np.asarray(run["sample_modality_mask"], dtype=bool)
    effective_local = np.flatnonzero(keep_mask.any(axis=1))
    return [int(index) for index in data_indices[effective_local].tolist()]


def subset_score_rows(payload: dict[str, Any], targets: np.ndarray, predictions: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(predictions, kind="mergesort")
    ranks = np.empty(predictions.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, predictions.shape[0] + 1, dtype=np.int64)
    rows = []
    for run_index, run in enumerate(payload["runs"]):
        rows.append(
            {
                "subset_index": int(run_index),
                "sample_index": int(run.get("sample_index", run_index)),
                "subset_seed": int(run["subset_seed"]),
                "rbf_predicted_loss": float(predictions[run_index]),
                "observed_test_loss": float(targets[run_index]),
                "prediction_minus_observed": float(predictions[run_index] - targets[run_index]),
                "low_output_rank": int(ranks[run_index]),
                "group_ids": [int(group_id) for group_id in run["group_ids"]],
                "num_data_indices": int(len(run["data_indices"])),
                "num_effective_sample_indices": int(len(effective_sample_indices(run))),
            }
        )
    rows.sort(key=lambda row: (float(row["rbf_predicted_loss"]), int(row["subset_index"])))
    return rows


def selected_subset_clusters(
    *,
    payload: dict[str, Any],
    score_rows: list[dict[str, Any]],
    num_clusters: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs_by_index = {int(index): run for index, run in enumerate(payload["runs"])}
    selected_rows = score_rows[: min(int(num_clusters), len(score_rows))]
    clusters: list[dict[str, Any]] = []
    sample_membership: dict[int, dict[str, Any]] = {}

    for cluster_id, row in enumerate(selected_rows):
        subset_index = int(row["subset_index"])
        run = runs_by_index[subset_index]
        sample_indices = effective_sample_indices(run)
        cluster = {
            "cluster_id": int(cluster_id),
            "subset_index": subset_index,
            "sample_index": int(row["sample_index"]),
            "subset_seed": int(row["subset_seed"]),
            "rbf_predicted_loss": float(row["rbf_predicted_loss"]),
            "observed_test_loss": float(row["observed_test_loss"]),
            "low_output_rank": int(row["low_output_rank"]),
            "group_ids": [int(group_id) for group_id in row["group_ids"]],
            "num_samples": int(len(sample_indices)),
            "sample_indices": sample_indices,
        }
        clusters.append(cluster)
        for sample_index in sample_indices:
            sample = int(sample_index)
            item = sample_membership.setdefault(
                sample,
                {
                    "sample_index": sample,
                    "cluster_ids": [],
                    "subset_indices": [],
                    "rbf_predicted_losses": [],
                    "observed_test_losses": [],
                },
            )
            item["cluster_ids"].append(int(cluster_id))
            item["subset_indices"].append(subset_index)
            item["rbf_predicted_losses"].append(float(row["rbf_predicted_loss"]))
            item["observed_test_losses"].append(float(row["observed_test_loss"]))

    sample_assignments = []
    for item in sample_membership.values():
        pred = np.asarray(item["rbf_predicted_losses"], dtype=np.float64)
        obs = np.asarray(item["observed_test_losses"], dtype=np.float64)
        sample_assignments.append(
            {
                "sample_index": int(item["sample_index"]),
                "cluster_ids": [int(value) for value in item["cluster_ids"]],
                "subset_indices": [int(value) for value in item["subset_indices"]],
                "num_selected_subsets": int(pred.size),
                "mean_rbf_predicted_loss": float(pred.mean()),
                "min_rbf_predicted_loss": float(pred.min()),
                "mean_observed_test_loss": float(obs.mean()),
                "min_observed_test_loss": float(obs.min()),
            }
        )
    sample_assignments.sort(key=lambda row: int(row["sample_index"]))
    return clusters, sample_assignments


def write_dataset_outputs(
    *,
    dataset: str,
    seed: int,
    payload: dict[str, Any],
    hparams: dict[str, Any],
    targets: np.ndarray,
    predictions: np.ndarray,
    score_rows: list[dict[str, Any]],
    cluster_counts: list[int],
    output_root: Path,
) -> list[dict[str, Any]]:
    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "dataset": dataset,
        "seed": int(seed),
        "method": "structured_rbf",
        "run_name": str(hparams["run_name"]),
        "hparams": hparams,
        "group_method": payload["group_partition"]["group_method"],
        "num_groups": int(payload["group_partition"]["num_groups"]),
        "num_subset_runs": int(len(payload["runs"])),
        "observed_loss_mean": float(np.mean(targets)),
        "observed_loss_std": float(np.std(targets)),
        "rbf_prediction_mean": float(np.mean(predictions)),
        "rbf_prediction_std": float(np.std(predictions)),
        "selection": "top-k existing sampled subsets with lowest structured_rbf predicted loss",
    }
    score_payload = dict(common)
    score_payload["subset_scores"] = score_rows
    (dataset_dir / "low_output_subset_scores.json").write_text(
        json.dumps(score_payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    summary_rows = []
    for requested_k in cluster_counts:
        clusters, sample_assignments = selected_subset_clusters(
            payload=payload,
            score_rows=score_rows,
            num_clusters=int(requested_k),
        )
        output = dict(common)
        output["requested_num_clusters"] = int(requested_k)
        output["num_clusters"] = int(len(clusters))
        output["clusters"] = clusters
        output["sample_assignments"] = sample_assignments
        output["num_unique_samples"] = int(len(sample_assignments))
        out_path = dataset_dir / f"structured_rbf_low_output_subset_clusters_k{int(requested_k)}.json"
        out_path.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
        summary_rows.append(
            {
                "dataset": dataset,
                "k": int(requested_k),
                "num_clusters": int(len(clusters)),
                "num_unique_samples": int(len(sample_assignments)),
                "lowest_subset_rbf_output": float(score_rows[0]["rbf_predicted_loss"]),
                "highest_selected_subset_rbf_output": float(clusters[-1]["rbf_predicted_loss"]),
                "path": str(out_path),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        hparams = get_best_structured_rbf_hparams(dataset)
        payload = load_payload(args.result_root, str(hparams["run_name"]), dataset, int(args.seed))
        print(f"[{dataset}] fitting structured_rbf and ranking existing sampled subsets...", flush=True)
        fit, mask_blocks, targets = fit_best_structured_rbf(payload, hparams)
        predictions = predict_existing_subsets(fit, mask_blocks)
        rows = subset_score_rows(payload, targets, predictions)
        dataset_summary = write_dataset_outputs(
            dataset=dataset,
            seed=int(args.seed),
            payload=payload,
            hparams=hparams,
            targets=targets,
            predictions=predictions,
            score_rows=rows,
            cluster_counts=[int(k) for k in args.cluster_counts],
            output_root=args.output_root,
        )
        summary_rows.extend(dataset_summary)
        print(
            f"  saved k={','.join(str(row['k']) for row in dataset_summary)}; "
            f"lowest RBF output={rows[0]['rbf_predicted_loss']:.6g}",
            flush=True,
        )

    summary = {
        "config": {
            "datasets": [str(dataset) for dataset in args.datasets],
            "cluster_counts": [int(k) for k in args.cluster_counts],
            "selection": "top-k existing sampled subsets with lowest structured_rbf predicted loss",
            "result_root": str(args.result_root),
            "output_root": str(args.output_root),
        },
        "rows": summary_rows,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    (args.output_root / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary), flush=True)
    print(f"Saved summary to {args.output_root / 'summary.json'}", flush=True)


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Low-Output Subset Clusters",
        "",
        f"- selection: `{summary['config']['selection']}`",
        "",
        "| dataset | k | clusters | unique samples | lowest RBF output | highest selected RBF output |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['k']} | {row['num_clusters']} | "
            f"{row['num_unique_samples']} | {row['lowest_subset_rbf_output']:.6g} | "
            f"{row['highest_selected_subset_rbf_output']:.6g} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
