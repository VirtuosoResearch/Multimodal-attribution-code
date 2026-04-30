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
from kernel_surrogate.structured_rbf_best_hparams import get_best_structured_rbf_hparams


METHODS = ("metric_rbf", "structured_metric_rbf", "structured_rbf", "raw_flat_rbf", "linear_regression")
RIDGE_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GAMMA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SHARED_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)

DEFAULT_STRUCTURED_SHUFFLES_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "structured_rbf_best_hparams_shuffles_80_20.json"
DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "compare_methods_at_structured_best_seed.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "compare_methods_at_structured_best_seed.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Anchor on the best structured_rbf split for each dataset, then compare "
            "other methods on the same setting and split_seed."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["avmnist", "imdb"])
    parser.add_argument("--structured-shuffles-json", type=Path, default=DEFAULT_STRUCTURED_SHUFFLES_JSON)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seed", type=int, default=0, help="LDS artifact seed to load.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
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


def load_structured_anchor_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["dataset"]): dict(row) for row in payload["best_per_dataset"]}


def evaluate_method_grid(
    *,
    dataset: str,
    method: str,
    setting: str,
    run_name: str,
    split_seed: int,
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    metric: np.ndarray,
    train_fraction: float,
) -> dict[str, Any]:
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    truth = targets[test_indices]
    best: dict[str, Any] | None = None
    gamma_values = (None,) if method == "linear_regression" else GAMMA_GRID
    shared_values = SHARED_GRID if method in {"structured_metric_rbf", "structured_rbf"} else (None,)
    method_metric = metric if method != "linear_regression" else np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64)

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


def evaluate_method_fixed(
    *,
    dataset: str,
    method: str,
    setting: str,
    run_name: str,
    split_seed: int,
    ridge_lambda: float,
    gamma_scale: float | None,
    structured_shared_weight: float | None,
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    metric: np.ndarray,
    train_fraction: float,
) -> dict[str, Any]:
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    truth = targets[test_indices]
    method_metric = metric if method != "linear_regression" else np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64)
    fit = fit_surrogate_method(
        method,
        mask_blocks,
        targets,
        method_metric,
        ridge_lambda=float(ridge_lambda),
        gamma_scale=1.0 if gamma_scale is None else float(gamma_scale),
        structured_shared_weight=1.0 if structured_shared_weight is None else float(structured_shared_weight),
        train_indices=train_indices,
    )
    pred = predict_surrogate_method(method, mask_blocks, fit, metric=method_metric, query_indices=test_indices)
    return {
        "dataset": dataset,
        "setting": setting,
        "run_name": run_name,
        "method": method,
        "split_seed": int(split_seed),
        "ridge_lambda": float(ridge_lambda),
        "gamma_scale": gamma_scale,
        "structured_shared_weight": structured_shared_weight,
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


def main() -> None:
    args = parse_args()
    anchors = load_structured_anchor_rows(args.structured_shuffles_json)
    device = resolve_device(str(args.device))
    full_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "grouped25_full_data_train"

    results = []
    for dataset in args.datasets:
        dataset = str(dataset)
        if dataset not in anchors:
            raise KeyError(f"no structured_rbf anchor found for dataset {dataset!r}")
        anchor = anchors[dataset]
        structured_hparams = get_best_structured_rbf_hparams(dataset)
        run_name = str(anchor["run_name"])
        setting = str(anchor["setting"])
        split_seed = int(anchor["split_seed"])
        payload_path = args.result_root / run_name / f"{dataset}_seed{int(args.seed)}.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        config = dict(payload["config"])

        print(f"[{dataset}] anchor structured_rbf at {setting}, split_seed={split_seed}", flush=True)
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

        dataset_rows = []
        for method in METHODS:
            if method == "structured_rbf":
                row = evaluate_method_fixed(
                    dataset=dataset,
                    method=method,
                    setting=setting,
                    run_name=run_name,
                    split_seed=split_seed,
                    ridge_lambda=float(structured_hparams["ridge_lambda"]),
                    gamma_scale=float(structured_hparams["gamma_scale"]),
                    structured_shared_weight=float(structured_hparams["structured_shared_weight"]),
                    mask_blocks=mask_blocks,
                    targets=targets,
                    metric=metric,
                    train_fraction=float(args.train_fraction),
                )
            else:
                row = evaluate_method_grid(
                    dataset=dataset,
                    method=method,
                    setting=setting,
                    run_name=run_name,
                    split_seed=split_seed,
                    mask_blocks=mask_blocks,
                    targets=targets,
                    metric=metric,
                    train_fraction=float(args.train_fraction),
                )
            row["anchor"] = {
                "structured_setting": setting,
                "structured_run_name": run_name,
                "structured_split_seed": split_seed,
                "structured_best_lds": float(anchor["test_lds"]),
            }
            row["metric_stats"] = metric_stats
            dataset_rows.append(row)
            print(
                f"  {method}: LDS={row['test_lds']:.6f}, Pearson={row['test_pearson']:.6f}, RMSE={row['test_rmse']:.6f}",
                flush=True,
            )
        results.extend(dataset_rows)

    payload = {
        "config": {
            "datasets": [str(item) for item in args.datasets],
            "methods": list(METHODS),
            "anchor_source": str(args.structured_shuffles_json),
            "split": f"{float(args.train_fraction):.6g}/{1.0 - float(args.train_fraction):.6g}",
            "metric_normalization": str(args.metric_normalization),
            "gradient_params": str(args.gradient_params),
        },
        "records": results,
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
        "# Compare Methods At Structured-RBF Best Seed",
        "",
        f"- anchor source: `{payload['config']['anchor_source']}`",
        f"- split: `{payload['config']['split']}`",
        f"- metric: held-out Spearman LDS",
        "",
        "## Results",
        "",
        "| dataset | method | anchor setting | split_seed | LDS | Pearson | RMSE | ridge | gamma | shared |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(payload["records"], key=lambda item: (item["dataset"], item["method"])):
        gamma = "" if row["gamma_scale"] is None else f"{row['gamma_scale']}"
        shared = "" if row["structured_shared_weight"] is None else f"{row['structured_shared_weight']}"
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['anchor']['structured_setting']} | "
            f"{row['split_seed']} | {row['test_lds']:.6f} | {row['test_pearson']:.6f} | "
            f"{row['test_rmse']:.6f} | {row['ridge_lambda']} | {gamma} | {shared} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
