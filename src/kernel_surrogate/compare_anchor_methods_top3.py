from __future__ import annotations

import argparse
import json
import math
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

from fisher_geometry.data import load_dataset
from kernel_surrogate.kernel_attribution import rmse, spearmanr
from kernel_surrogate.real_data_cmis import (
    build_group_mask_blocks,
    build_metric,
    compute_group_modality_gradients,
    fit_surrogate_method,
    load_or_train_full_model,
    make_model_args,
    predict_surrogate_method,
    resolve_device,
    resolve_full_checkpoint_path,
    split_indices,
)


DATASETS = ("avmnist", "imdb", "humor", "mosi", "mosei", "sarcasm")
SETTINGS = (
    ("kmeans_data80_mod80", "kmeans_grouped25_subset100_data80_mod80"),
    ("kmeans_data90_mod90", "kmeans_grouped25_subset100_data90_mod90"),
    ("random_data80_mod80", "random_grouped25_subset100_data80_mod80"),
    ("random_data90_mod90", "random_grouped25_subset100_data90_mod90"),
)
METHODS = ("metric_rbf", "structured_metric_rbf", "structured_rbf", "raw_flat_rbf", "linear_regression")
ANCHOR_METHODS = ("structured_rbf", "structured_metric_rbf")
RIDGE_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GAMMA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SHARED_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)

DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "compare_anchor_methods_top3.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "compare_anchor_methods_top3.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search best hparams for all surrogate methods on all settings/splits, then "
            "compare all methods under two anchor schemes: structured_rbf-best and "
            "structured_metric_rbf-best. Also report top-3 anchor seed mean/std."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=0, help="LDS artifact seed to load.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def pearsonr(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denom = float(np.sqrt(np.sum(pred**2) * np.sum(truth**2)))
    if denom <= 1e-12 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(pred * truth) / denom)


def score_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(np.nan_to_num(row["test_lds"], nan=-1e18)),
        float(np.nan_to_num(row["test_pearson"], nan=-1e18)),
        -float(row["test_rmse"]),
    )


def evaluate_method_grid(
    *,
    dataset: str,
    setting: str,
    run_name: str,
    method: str,
    split_seed: int,
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    metric: np.ndarray,
    train_fraction: float,
) -> dict[str, Any]:
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    truth = targets[test_indices]
    gamma_values = (None,) if method == "linear_regression" else GAMMA_GRID
    shared_values = SHARED_GRID if method in {"structured_metric_rbf", "structured_rbf"} else (None,)
    method_metric = metric if method != "linear_regression" else np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64)

    best: dict[str, Any] | None = None
    for ridge in RIDGE_GRID:
        for gamma in gamma_values:
            for shared in shared_values:
                fit = fit_surrogate_method(
                    method,
                    mask_blocks,
                    targets,
                    method_metric,
                    ridge_lambda=float(ridge),
                    gamma_scale=1.0 if gamma is None else float(gamma),
                    structured_shared_weight=1.0 if shared is None else float(shared),
                    train_indices=train_indices,
                )
                pred = predict_surrogate_method(method, mask_blocks, fit, metric=method_metric, query_indices=test_indices)
                row = {
                    "dataset": dataset,
                    "setting": setting,
                    "run_name": run_name,
                    "method": method,
                    "split_seed": int(split_seed),
                    "ridge_lambda": float(ridge),
                    "gamma_scale": None if gamma is None else float(gamma),
                    "structured_shared_weight": None if shared is None else float(shared),
                    "test_lds": spearmanr(pred, truth),
                    "test_pearson": pearsonr(pred, truth),
                    "test_rmse": rmse(pred, truth),
                    "train_lds": spearmanr(fit["train_predictions"], targets[train_indices]),
                    "train_rmse": rmse(fit["train_predictions"], targets[train_indices]),
                    "num_train": int(train_indices.size),
                    "num_test": int(test_indices.size),
                    "train_indices": train_indices.tolist(),
                    "test_indices": test_indices.tolist(),
                }
                if best is None or score_key(row) > score_key(best):
                    best = row
    assert best is not None
    return best


def choose_topk_unique_seeds(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    best_by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["split_seed"])
        if seed not in best_by_seed or score_key(row) > score_key(best_by_seed[seed]):
            best_by_seed[seed] = row
    return sorted(best_by_seed.values(), key=score_key, reverse=True)[:k]


def summarize_top3(
    *,
    anchor_method: str,
    dataset: str,
    records: list[dict[str, Any]],
    top3_anchor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for method in METHODS:
        values = []
        rows = []
        for anchor_row in top3_anchor_rows:
            match = next(
                row
                for row in records
                if row["dataset"] == dataset
                and row["setting"] == anchor_row["setting"]
                and int(row["split_seed"]) == int(anchor_row["split_seed"])
                and row["method"] == method
            )
            values.append(float(match["test_lds"]))
            rows.append(match)
        values_np = np.asarray(values, dtype=np.float64)
        summaries.append(
            {
                "anchor_method": anchor_method,
                "dataset": dataset,
                "method": method,
                "mean_lds": float(values_np.mean()),
                "std_lds": float(values_np.std(ddof=0)),
                "rows": rows,
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.device))
    split_seeds = [int(seed) for seed in args.split_seeds]
    full_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "grouped25_full_data_train"

    all_records: list[dict[str, Any]] = []
    for dataset in args.datasets:
        dataset = str(dataset)
        for setting, run_name in SETTINGS:
            payload_path = args.result_root / run_name / f"{dataset}_seed{int(args.seed)}.json"
            if not payload_path.exists():
                continue
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            config = dict(payload["config"])
            print(f"[{dataset}/{setting}] loading data on {device}...", flush=True)
            data = load_dataset(
                dataset,
                int(args.seed),
                text_backbone=str(config.get("text_backbone", "clip_cached")),
                vision_backbone=str(config.get("vision_backbone", "clip_cached")),
                feature_device=str(args.feature_device or config.get("feature_device", device)),
                backbone_batch_size=int(config.get("backbone_batch_size", 32)),
                download_datasets=bool(config.get("download_datasets", False)),
                backbone_cache_dir=config.get("backbone_cache_dir"),
            )
            model_args = make_model_args(config, device=str(device))
            full_checkpoint = resolve_full_checkpoint_path(
                dataset=dataset,
                seed=int(args.seed),
                pretrained_root=full_root,
                fallback_dir=ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis",
            )
            model, _ = load_or_train_full_model(
                full_checkpoint,
                data=data,
                task=str(data["task"]),
                dim_x=int(data["train_x"].shape[1]),
                dim_y=int(data["train_y"].shape[1]),
                num_classes=int(data["num_classes"]),
                model_args=model_args,
                model_type=str(config.get("model", "cross_attention")),
                seed=int(args.seed),
                force_train=False,
            )
            model.to(device)
            model.eval()
            gradients = compute_group_modality_gradients(
                model,
                data=data,
                group_indices=payload["group_partition"]["group_indices"],
                task=str(data["task"]),
                device=str(device),
                gradient_params=str(args.gradient_params),
            )
            metric, metric_stats = build_metric(gradients, normalization=str(args.metric_normalization))
            mask_blocks, targets = build_group_mask_blocks(payload)
            for split_seed in split_seeds:
                print(f"  split_seed={split_seed}", flush=True)
                for method in METHODS:
                    row = evaluate_method_grid(
                        dataset=dataset,
                        setting=setting,
                        run_name=run_name,
                        method=method,
                        split_seed=split_seed,
                        mask_blocks=mask_blocks,
                        targets=targets,
                        metric=metric,
                        train_fraction=float(args.train_fraction),
                    )
                    row["metric_stats"] = metric_stats
                    all_records.append(row)
                    print(
                        f"    {method}: LDS={row['test_lds']:.6f}, "
                        f"Pearson={row['test_pearson']:.6f}, RMSE={row['test_rmse']:.6f}",
                        flush=True,
                    )

    best_anchor_rows = []
    top3_summaries = []
    for dataset in args.datasets:
        dataset = str(dataset)
        dataset_rows = [row for row in all_records if row["dataset"] == dataset]
        for anchor_method in ANCHOR_METHODS:
            anchor_candidates = [row for row in dataset_rows if row["method"] == anchor_method]
            top3_anchor_rows = choose_topk_unique_seeds(anchor_candidates, 3)
            if not top3_anchor_rows:
                continue
            best_anchor = top3_anchor_rows[0]
            for method in METHODS:
                matched = next(
                    row
                    for row in dataset_rows
                    if row["setting"] == best_anchor["setting"]
                    and int(row["split_seed"]) == int(best_anchor["split_seed"])
                    and row["method"] == method
                )
                best_anchor_rows.append(
                    {
                        "anchor_method": anchor_method,
                        "anchor_rank": 1,
                        "dataset": dataset,
                        "anchor_setting": best_anchor["setting"],
                        "anchor_run_name": best_anchor["run_name"],
                        "anchor_split_seed": int(best_anchor["split_seed"]),
                        "anchor_method_lds": float(best_anchor["test_lds"]),
                        "method_row": matched,
                    }
                )
            top3_summaries.extend(
                summarize_top3(
                    anchor_method=anchor_method,
                    dataset=dataset,
                    records=dataset_rows,
                    top3_anchor_rows=top3_anchor_rows,
                )
            )

    payload = {
        "config": {
            "datasets": [str(item) for item in args.datasets],
            "settings": [name for name, _ in SETTINGS],
            "methods": list(METHODS),
            "anchor_methods": list(ANCHOR_METHODS),
            "split": f"{float(args.train_fraction):.6g}/{1.0 - float(args.train_fraction):.6g}",
            "split_seeds": split_seeds,
            "metric_normalization": str(args.metric_normalization),
            "gradient_params": str(args.gradient_params),
        },
        "records": all_records,
        "best_anchor_rows": best_anchor_rows,
        "top3_summaries": top3_summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Compare Anchors By Top Seed And Top-3 Seeds",
        "",
        f"- split: `{payload['config']['split']}`",
        f"- split seeds: `{payload['config']['split_seeds']}`",
        f"- metric: held-out Spearman LDS",
    ]
    for anchor_method in ANCHOR_METHODS:
        lines.extend(
            [
                "",
                f"## {anchor_method} Best Seed",
                "",
                "| dataset | method | anchor setting | anchor seed | LDS | Pearson | RMSE |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        rows = [
            row for row in payload["best_anchor_rows"] if row["anchor_method"] == anchor_method
        ]
        for row in sorted(rows, key=lambda item: (item["dataset"], item["method_row"]["method"])):
            method_row = row["method_row"]
            lines.append(
                f"| {row['dataset']} | {method_row['method']} | {row['anchor_setting']} | "
                f"{row['anchor_split_seed']} | {method_row['test_lds']:.6f} | "
                f"{method_row['test_pearson']:.6f} | {method_row['test_rmse']:.6f} |"
            )
        lines.extend(
            [
                "",
                f"## {anchor_method} Top-3 Seed Mean/Std",
                "",
                "| dataset | method | mean LDS | std LDS |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        summaries = [
            row for row in payload["top3_summaries"] if row["anchor_method"] == anchor_method
        ]
        for row in sorted(summaries, key=lambda item: (item["dataset"], item["method"])):
            lines.append(
                f"| {row['dataset']} | {row['method']} | {row['mean_lds']:.6f} | {row['std_lds']:.6f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
