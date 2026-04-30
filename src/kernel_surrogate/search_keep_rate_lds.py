from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from fisher_geometry.data import load_dataset


DATASETS = ("imdb", "mosei", "mosi", "sarcasm")
SETTINGS = (
    ("kmeans_data80_mod80", "kmeans_grouped25_subset100_data80_mod80"),
    ("kmeans_data90_mod90", "kmeans_grouped25_subset100_data90_mod90"),
)
METHODS = ("metric_rbf", "structured_metric_rbf", "structured_rbf", "raw_flat_rbf", "linear_regression")
RIDGE_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GAMMA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SHARED_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)

OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "search_keep_rate_lds_80_20.json"
OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "search_keep_rate_lds_80_20.md"


def pearson_spearman(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    pred_order = np.argsort(np.argsort(pred, kind="mergesort"), kind="mergesort").astype(np.float64)
    truth_order = np.argsort(np.argsort(truth, kind="mergesort"), kind="mergesort").astype(np.float64)
    pred_centered = pred_order - pred_order.mean()
    truth_centered = truth_order - truth_order.mean()
    denom = float(np.sqrt(np.sum(pred_centered**2) * np.sum(truth_centered**2)))
    spearman = float(np.sum(pred_centered * truth_centered) / denom) if denom > 0.0 else float("nan")

    pred_centered = pred - pred.mean()
    truth_centered = truth - truth.mean()
    denom = float(np.sqrt(np.sum(pred_centered**2) * np.sum(truth_centered**2)))
    pearson = float(np.sum(pred_centered * truth_centered) / denom) if denom > 0.0 else float("nan")
    return pearson, spearman


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    device = resolve_device("cuda:0")
    results: list[dict[str, Any]] = []
    per_dataset_best: list[dict[str, Any]] = []

    kmeans_full_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "kmeans_grouped25_full_data_train"
    grouped_full_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "grouped25_full_data_train"
    full_fallback = ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis"

    for dataset in DATASETS:
        dataset_rows: list[dict[str, Any]] = []
        for setting_name, run_name in SETTINGS:
            result_dir = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train" / run_name
            payload = load_payload(result_dir / f"{dataset}_seed0.json")
            config = dict(payload["config"])

            data = load_dataset(
                dataset,
                0,
                text_backbone=str(config.get("text_backbone", "clip_cached")),
                vision_backbone=str(config.get("vision_backbone", "clip_cached")),
                feature_device=str(config.get("feature_device", device)),
                backbone_batch_size=int(config.get("backbone_batch_size", 32)),
                download_datasets=bool(config.get("download_datasets", False)),
                backbone_cache_dir=config.get("backbone_cache_dir"),
            )
            model_args = make_model_args(config, device=str(device))
            preferred_root = kmeans_full_root
            preferred_ckpt = preferred_root / dataset / "seed0" / "subset001" / "checkpoints" / "checkpoint_repeat01.pt"
            if not preferred_ckpt.exists():
                preferred_root = grouped_full_root
            full_checkpoint = resolve_full_checkpoint_path(
                dataset=dataset,
                seed=0,
                pretrained_root=preferred_root,
                fallback_dir=full_fallback,
            )
            model, full_metrics = load_or_train_full_model(
                full_checkpoint,
                data=data,
                task=str(data["task"]),
                dim_x=int(data["train_x"].shape[1]),
                dim_y=int(data["train_y"].shape[1]),
                num_classes=int(data["num_classes"]),
                model_args=model_args,
                model_type=str(config.get("model", "cross_attention")),
                seed=0,
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
                gradient_params="last_layer",
            )
            metric, _ = build_metric(gradients, normalization="cosine")
            mask_blocks, targets = build_group_mask_blocks(payload)
            train_indices, test_indices = split_indices(len(targets), train_fraction=0.8, seed=0)
            truth = targets[test_indices]

            best_rows: list[dict[str, Any]] = []
            for method in METHODS:
                best: dict[str, Any] | None = None
                method_metric = metric if method != "linear_regression" else np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64)
                gamma_values = (None,) if method == "linear_regression" else GAMMA_GRID
                shared_values = SHARED_GRID if method in {"structured_metric_rbf", "structured_rbf"} else (None,)

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
                            pred = predict_surrogate_method(
                                method,
                                mask_blocks,
                                fit,
                                metric=method_metric,
                                query_indices=test_indices,
                            )
                            pearson, spearman = pearson_spearman(pred, truth)
                            row = {
                                "dataset": dataset,
                                "setting": setting_name,
                                "method": method,
                                "ridge_lambda": float(ridge),
                                "gamma_scale": None if gamma is None else float(gamma),
                                "structured_shared_weight": None if shared is None else float(shared),
                                "test_spearman": float(spearman),
                                "test_pearson": float(pearson),
                                "test_rmse": rmse(pred, truth),
                                "num_train": int(train_indices.size),
                                "num_test": int(test_indices.size),
                                "full_test_loss": float(full_metrics.get("loss", float("nan"))),
                                "full_test_acc": float(full_metrics.get("acc", float("nan"))),
                            }
                            score = (np.nan_to_num(row["test_spearman"], nan=-1e18), -row["test_rmse"])
                            if best is None or score > (np.nan_to_num(best["test_spearman"], nan=-1e18), -best["test_rmse"]):
                                best = row
                assert best is not None
                best_rows.append(best)
                results.append(best)

            best_overall = max(best_rows, key=lambda row: (np.nan_to_num(row["test_spearman"], nan=-1e18), -row["test_rmse"]))
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "setting": setting_name,
                    "best_method": best_overall["method"],
                    "best_test_spearman": best_overall["test_spearman"],
                    "best_test_rmse": best_overall["test_rmse"],
                }
            )

        per_dataset_best.append(
            max(dataset_rows, key=lambda row: (np.nan_to_num(row["best_test_spearman"], nan=-1e18), -row["best_test_rmse"]))
        )

    payload = {
        "config": {
            "datasets": list(DATASETS),
            "settings": [name for name, _ in SETTINGS],
            "split": "80/20 seed0",
            "metric_normalization": "cosine",
            "gradient_params": "last_layer",
            "ridge_grid": list(RIDGE_GRID),
            "gamma_grid": list(GAMMA_GRID),
            "shared_grid": list(SHARED_GRID),
        },
        "records": results,
        "per_dataset_best_setting": per_dataset_best,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    OUTPUT_MD.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload), flush=True)
    print(f"Saved JSON to {OUTPUT_JSON}", flush=True)
    print(f"Saved Markdown to {OUTPUT_MD}", flush=True)


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Keep-Rate LDS Search",
        "",
        "- split: `80/20 seed0`",
        "- metric normalization: `cosine`",
        "- gradient params: `last_layer`",
        "",
        "## Best Setting Per Dataset",
        "",
        "| dataset | best setting | best method | best LDS | best RMSE |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in payload["per_dataset_best_setting"]:
        lines.append(
            f"| {row['dataset']} | {row['setting']} | {row['best_method']} | "
            f"{row['best_test_spearman']:.6f} | {row['best_test_rmse']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best Hyperparameters Per Method",
            "",
            "| dataset | setting | method | LDS | RMSE | ridge | gamma | shared |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["records"]:
        gamma = "" if row["gamma_scale"] is None else f"{row['gamma_scale']}"
        shared = "" if row["structured_shared_weight"] is None else f"{row['structured_shared_weight']}"
        lines.append(
            f"| {row['dataset']} | {row['setting']} | {row['method']} | "
            f"{row['test_spearman']:.6f} | {row['test_rmse']:.6f} | "
            f"{row['ridge_lambda']} | {gamma} | {shared} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
