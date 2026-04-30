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
from kernel_surrogate.real_data_cmis import (
    DEFAULT_PRETRAINED_FULL_CHECKPOINT_ROOT,
    analytical_second_derivative_at_zero,
    build_group_mask_blocks,
    build_metric,
    compute_group_modality_gradients,
    fit_surrogate_method,
    load_or_train_full_model,
    make_model_args,
    normalize_cmis_scores,
    resolve_device,
    resolve_full_checkpoint_path,
)


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "grouped50_subset100_data50_mod50"
)
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "real_subset_irev.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "real_subset_irev.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Interaction Residual Explained Variance on real subsets with cross-fitting."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--datasets", nargs="+", default=["avmnist", "humor", "imdb", "mosei", "mosi", "sarcasm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--pretrained-full-checkpoint-root", type=Path, default=DEFAULT_PRETRAINED_FULL_CHECKPOINT_ROOT)
    parser.add_argument("--full-checkpoint-dir", type=Path, default=ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis")
    parser.add_argument("--kernel-methods", nargs="+", default=["metric_rbf", "structured_rbf", "raw_flat_rbf"])
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
    parser.add_argument("--ridge-lambda", type=float, default=1e-5)
    parser.add_argument("--gamma-scale", type=float, default=1e-3)
    parser.add_argument("--structured-shared-weight", type=float, default=1.0)
    parser.add_argument("--cmis-normalization-eps", type=float, default=1e-12)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.device))
    dataset_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []

    for dataset in args.datasets:
        payload_path = args.result_dir / f"{dataset}_seed{args.seed}.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        config = dict(payload["config"])
        print(f"[{dataset}] loading real subset payload...", flush=True)
        mask_blocks, targets = build_group_mask_blocks(payload)

        metric = np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64)
        metric_stats: dict[str, Any] = {}
        if "metric_rbf" in args.kernel_methods:
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
            full_checkpoint = resolve_full_checkpoint_path(
                dataset=dataset,
                seed=int(args.seed),
                pretrained_root=args.pretrained_full_checkpoint_root,
                fallback_dir=args.full_checkpoint_dir,
            )
            model_args = make_model_args(config, device=str(device))
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
            print(f"[{dataset}] computing {args.gradient_params} metric on {device}...", flush=True)
            gradients = compute_group_modality_gradients(
                model,
                data=data,
                group_indices=payload["group_partition"]["group_indices"],
                task=str(data["task"]),
                device=str(device),
                gradient_params=str(args.gradient_params),
            )
            metric, metric_stats = build_metric(gradients, normalization=str(args.metric_normalization))

        method_rows: list[dict[str, Any]] = []
        for method in args.kernel_methods:
            print(f"[{dataset}] cross-fitting IREV for {method}...", flush=True)
            irev_result = crossfit_irev(
                mask_blocks,
                targets,
                metric=metric,
                method=str(method),
                ridge_lambda=float(args.ridge_lambda),
                gamma_scale=float(args.gamma_scale),
                structured_shared_weight=float(args.structured_shared_weight),
                cmis_eps=float(args.cmis_normalization_eps),
                num_folds=int(args.num_folds),
                seed=int(args.random_seed),
            )
            row = {"dataset": dataset, "method": str(method), **irev_result}
            method_rows.append(row)
            all_records.append(row)
            print(
                f"  {method}: IREV={row['irev']:.6f}, residual var={row['residual_variance']:.6g}, "
                f"activation-residual corr={row['activation_residual_corr']:.6f}",
                flush=True,
            )

        dataset_rows.append(
            {
                "dataset": dataset,
                "seed": int(args.seed),
                "num_subsets": int(targets.shape[0]),
                "metric_stats": metric_stats,
                "methods": method_rows,
            }
        )

    summary = summarize_records(all_records)
    output = {
        "config": {
            "result_dir": str(args.result_dir),
            "datasets": list(args.datasets),
            "seed": int(args.seed),
            "unit": "group",
            "gradient_params": str(args.gradient_params),
            "metric_normalization": str(args.metric_normalization),
            "kernel_methods": [str(method) for method in args.kernel_methods],
            "ridge_lambda": float(args.ridge_lambda),
            "gamma_scale": float(args.gamma_scale),
            "structured_shared_weight": float(args.structured_shared_weight),
            "cmis_normalization_eps": float(args.cmis_normalization_eps),
            "num_folds": int(args.num_folds),
            "random_seed": int(args.random_seed),
        },
        "summary": summary,
        "dataset_results": dataset_rows,
        "records": all_records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def crossfit_irev(
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    *,
    metric: np.ndarray,
    method: str,
    ridge_lambda: float,
    gamma_scale: float,
    structured_shared_weight: float,
    cmis_eps: float,
    num_folds: int,
    seed: int,
) -> dict[str, Any]:
    z1 = np.asarray(mask_blocks["modality_deleted"][:, 0::2], dtype=np.float64)
    z2 = np.asarray(mask_blocks["modality_deleted"][:, 1::2], dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    num_items = int(targets.shape[0])
    num_folds = max(2, min(int(num_folds), num_items))
    order = np.random.default_rng(seed).permutation(num_items)
    folds = [np.sort(chunk.astype(np.int64, copy=False)) for chunk in np.array_split(order, num_folds) if chunk.size]

    heldout_residuals: list[np.ndarray] = []
    heldout_predictions: list[np.ndarray] = []
    heldout_activations: list[np.ndarray] = []

    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(num_items, dtype=np.int64), test_idx, assume_unique=True)
        if train_idx.size < 2 or test_idx.size == 0:
            continue
        fit = fit_surrogate_method(
            method,
            mask_blocks,
            targets,
            metric,
            ridge_lambda=ridge_lambda,
            gamma_scale=gamma_scale,
            structured_shared_weight=structured_shared_weight,
            train_indices=train_idx,
        )
        raw_scores = analytical_second_derivative_at_zero(
            fit["train_masks"],
            fit["metric"],
            alpha=fit["alpha"],
            gamma=float(fit["gamma"]),
            num_groups=int(mask_blocks["num_groups"][0]),
            atom_offset=int(fit.get("atom_offset", 0)),
        )
        hhat = normalize_cmis_scores(raw_scores, eps=cmis_eps)
        np.fill_diagonal(hhat, 0.0)

        x_add_train = additive_design(z1[train_idx], z2[train_idx])
        x_add_test = additive_design(z1[test_idx], z2[test_idx])
        beta, *_ = np.linalg.lstsq(x_add_train, targets[train_idx], rcond=None)
        residual_train = targets[train_idx] - x_add_train @ beta
        residual_test = targets[test_idx] - x_add_test @ beta

        activation_train = interaction_activation_score(z1[train_idx], z2[train_idx], hhat)
        activation_test = interaction_activation_score(z1[test_idx], z2[test_idx], hhat)
        denom = float(np.dot(activation_train, activation_train))
        eta = 0.0 if denom <= 1e-12 else float(np.dot(activation_train, residual_train) / denom)
        heldout_residuals.append(residual_test)
        heldout_predictions.append(eta * activation_test)
        heldout_activations.append(activation_test)

    residuals = np.concatenate(heldout_residuals, axis=0) if heldout_residuals else np.asarray([], dtype=np.float64)
    predictions = np.concatenate(heldout_predictions, axis=0) if heldout_predictions else np.asarray([], dtype=np.float64)
    activations = np.concatenate(heldout_activations, axis=0) if heldout_activations else np.asarray([], dtype=np.float64)
    if residuals.size == 0:
        return {
            "irev": float("nan"),
            "residual_variance": float("nan"),
            "predicted_residual_variance": float("nan"),
            "activation_residual_corr": float("nan"),
            "num_subsets": int(num_items),
        }
    denom = float(np.sum(residuals**2))
    irev = float("nan") if denom <= 1e-12 else float(1.0 - np.sum((residuals - predictions) ** 2) / denom)
    return {
        "irev": irev,
        "residual_variance": float(np.var(residuals)),
        "predicted_residual_variance": float(np.var(predictions)),
        "activation_residual_corr": pearsonr(activations, residuals),
        "num_subsets": int(num_items),
    }


def additive_design(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    intercept = np.ones((z1.shape[0], 1), dtype=np.float64)
    return np.concatenate([intercept, z1, z2], axis=1)


def interaction_activation_score(z1: np.ndarray, z2: np.ndarray, hhat: np.ndarray) -> np.ndarray:
    return np.sum((z1 @ hhat) * z2, axis=1)


def pearsonr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denom <= 1e-12 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(left * right) / denom)


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for method, rows in by_method.items():
        summary.append(
            {
                "method": method,
                "num_dataset_runs": len(rows),
                "mean_irev": float(np.nanmean([row["irev"] for row in rows])),
                "std_irev": float(np.nanstd([row["irev"] for row in rows])),
                "mean_activation_residual_corr": float(np.nanmean([row["activation_residual_corr"] for row in rows])),
                "mean_residual_variance": float(np.nanmean([row["residual_variance"] for row in rows])),
                "mean_predicted_residual_variance": float(np.nanmean([row["predicted_residual_variance"] for row in rows])),
            }
        )
    return summary


def to_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# Real-Subset IREV Evaluation",
        "",
        f"- Result dir: `{config['result_dir']}`",
        f"- Unit: `{config['unit']}`",
        f"- Metric RBF gradients: `{config['gradient_params']}`",
        f"- Metric normalization: `{config['metric_normalization']}`",
        f"- Folds: `{config['num_folds']}`",
        "",
        "## Summary",
        "",
        "| method | mean IREV | std IREV | mean activation-residual corr | mean residual variance | mean predicted residual variance |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['method']} | {row['mean_irev']:.6g} | {row['std_irev']:.6g} | "
            f"{row['mean_activation_residual_corr']:.6g} | {row['mean_residual_variance']:.6g} | "
            f"{row['mean_predicted_residual_variance']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Per Dataset",
            "",
            "| dataset | method | IREV | activation-residual corr | residual variance | predicted residual variance | subsets |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset_result in payload["dataset_results"]:
        dataset = dataset_result["dataset"]
        for row in dataset_result["methods"]:
            lines.append(
                f"| {dataset} | {row['method']} | {row['irev']:.6g} | {row['activation_residual_corr']:.6g} | "
                f"{row['residual_variance']:.6g} | {row['predicted_residual_variance']:.6g} | {row['num_subsets']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
