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


METHODS = (
    "metric_rbf",
    "structured_metric_rbf",
    "structured_rbf",
    "sks",
    "raw_flat_rbf",
    "linear_regression",
)
DEFAULT_HPARAMS_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "search_keep_rate_lds_80_20.json"
DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "metric_vs_structured_best_hparams_shuffles_80_20.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "metric_vs_structured_best_hparams_shuffles_80_20.md"
DEFAULT_RIDGE_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_GAMMA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_SHARED_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
DEFAULT_RHO_GRID = (-0.95, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.95)
DEFAULT_SETTINGS = ("kmeans_data80_mod80", "kmeans_data90_mod90")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate recorded best metric_rbf, structured_metric_rbf, structured_rbf, sks, raw_flat_rbf, and linear_regression hyperparameters "
            "on aligned 80/20 shuffles and export raw held-out LDS rows."
        )
    )
    parser.add_argument("--hparams-json", type=Path, default=DEFAULT_HPARAMS_JSON)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument(
        "--search-mode",
        choices=["per_split", "fixed_hparams"],
        default="per_split",
        help="per_split grid-searches hparams independently for every split seed; fixed_hparams keeps the old recorded-hparams protocol.",
    )
    parser.add_argument("--seed", type=int, default=0, help="LDS artifact seed to load.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=list(DEFAULT_RIDGE_GRID))
    parser.add_argument("--gamma-grid", nargs="+", type=float, default=list(DEFAULT_GAMMA_GRID))
    parser.add_argument("--shared-grid", nargs="+", type=float, default=list(DEFAULT_SHARED_GRID))
    parser.add_argument("--rho-grid", nargs="+", type=float, default=list(DEFAULT_RHO_GRID))
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


def summarize_topk_fixed_seed(
    records: list[dict[str, Any]],
    *,
    top_k: int,
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in records}):
        for method in methods:
            candidates = [row for row in records if row["dataset"] == dataset and row["method"] == method]
            if not candidates:
                continue
            best_by_seed: dict[int, dict[str, Any]] = {}
            for row in candidates:
                seed = int(row["split_seed"])
                if seed not in best_by_seed or score_key(row) > score_key(best_by_seed[seed]):
                    best_by_seed[seed] = row
            top_rows = sorted(best_by_seed.values(), key=score_key, reverse=True)[:top_k]
            lds = np.asarray([float(row["test_lds"]) for row in top_rows], dtype=np.float64)
            summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "top_k": int(len(top_rows)),
                    "mean_lds": float(lds.mean()),
                    "std_lds": float(lds.std(ddof=0)),
                    "rows": top_rows,
                }
            )
    return summaries


def run_name_for_setting(setting: str) -> str:
    return f"kmeans_grouped25_subset100_{setting.removeprefix('kmeans_')}"


def load_best_hparams(path: Path, *, methods: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload["records"]:
        if row["method"] in methods:
            rows.append(dict(row))
    return rows


def iter_grid_hparams(
    method: str,
    *,
    ridge_grid: Sequence[float],
    gamma_grid: Sequence[float],
    shared_grid: Sequence[float],
    rho_grid: Sequence[float],
) -> Sequence[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if method == "linear_regression":
        for ridge in ridge_grid:
            rows.append(
                {
                    "method": method,
                    "ridge_lambda": float(ridge),
                    "gamma_scale": None,
                    "structured_shared_weight": None,
                }
            )
        return rows
    if method in {"structured_metric_rbf", "structured_rbf"}:
        shared_values: Sequence[float | None] = tuple(float(value) for value in shared_grid)
    elif method == "sks":
        shared_values = tuple(float(value) for value in rho_grid)
    else:
        shared_values = (None,)
    for ridge in ridge_grid:
        for gamma in gamma_grid:
            for shared in shared_values:
                rows.append(
                    {
                        "method": method,
                        "ridge_lambda": float(ridge),
                        "gamma_scale": float(gamma),
                        "structured_shared_weight": None if shared is None else float(shared),
                    }
                )
    return rows


def evaluate_split(
    *,
    hparams: dict[str, Any],
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    metric: np.ndarray,
    split_seed: int,
    train_fraction: float,
) -> dict[str, Any]:
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    method = str(hparams["method"])
    gamma_scale = 1.0 if hparams.get("gamma_scale") is None else float(hparams["gamma_scale"])
    fit = fit_surrogate_method(
        method,
        mask_blocks,
        targets,
        metric,
        ridge_lambda=float(hparams["ridge_lambda"]),
        gamma_scale=gamma_scale,
        structured_shared_weight=1.0
        if hparams.get("structured_shared_weight") is None
        else float(hparams["structured_shared_weight"]),
        train_indices=train_indices,
    )
    pred = predict_surrogate_method(method, mask_blocks, fit, metric=metric, query_indices=test_indices)
    truth = targets[test_indices]
    return {
        "dataset": str(hparams["dataset"]),
        "setting": str(hparams["setting"]),
        "run_name": run_name_for_setting(str(hparams["setting"])),
        "method": method,
        "ridge_lambda": float(hparams["ridge_lambda"]),
        "gamma_scale": hparams.get("gamma_scale"),
        "structured_shared_weight": hparams.get("structured_shared_weight"),
        "split_seed": int(split_seed),
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


def evaluate_grid_for_split(
    *,
    dataset: str,
    setting: str,
    method: str,
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    metric: np.ndarray,
    split_seed: int,
    train_fraction: float,
    ridge_grid: Sequence[float],
    gamma_grid: Sequence[float],
    shared_grid: Sequence[float],
    rho_grid: Sequence[float],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for hparams in iter_grid_hparams(
        method,
        ridge_grid=ridge_grid,
        gamma_grid=gamma_grid,
        shared_grid=shared_grid,
        rho_grid=rho_grid,
    ):
        row = evaluate_split(
            hparams={
                **hparams,
                "dataset": dataset,
                "setting": setting,
            },
            mask_blocks=mask_blocks,
            targets=targets,
            metric=metric,
            split_seed=split_seed,
            train_fraction=train_fraction,
        )
        if best is None or score_key(row) > score_key(best):
            best = row
    if best is None:
        raise ValueError(f"empty grid for method={method}")
    return best


def main() -> None:
    args = parse_args()
    selected_methods = tuple(str(item) for item in args.methods)
    source_hparams_rows = load_best_hparams(args.hparams_json, methods=set(selected_methods))
    if args.search_mode == "fixed_hparams":
        hparams_rows = source_hparams_rows
        if args.datasets is not None:
            datasets = {str(item) for item in args.datasets}
            hparams_rows = [row for row in hparams_rows if str(row["dataset"]) in datasets]
        if args.settings is not None:
            settings = {str(item) for item in args.settings}
            hparams_rows = [row for row in hparams_rows if str(row["setting"]) in settings]
        if not hparams_rows:
            raise ValueError("no matching kernel surrogate hparams found")
        dataset_setting_pairs = sorted({(str(row["dataset"]), str(row["setting"])) for row in hparams_rows})
    else:
        if args.datasets is not None:
            datasets = [str(item) for item in args.datasets]
        else:
            datasets = sorted({str(row["dataset"]) for row in source_hparams_rows})
        if args.settings is not None:
            settings = [str(item) for item in args.settings]
        else:
            settings = list(DEFAULT_SETTINGS)
        if not datasets:
            raise ValueError("no datasets provided and no datasets could be inferred from hparams-json")
        dataset_setting_pairs = [(dataset, setting) for dataset in datasets for setting in settings]
        hparams_rows = []

    split_seeds = [int(seed) for seed in args.split_seeds]
    device = resolve_device(str(args.device))
    kmeans_full_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "kmeans_grouped25_full_data_train"
    grouped_full_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "grouped25_full_data_train"
    full_fallback = ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis"

    records: list[dict[str, Any]] = []
    rows_by_dataset_setting: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if args.search_mode == "fixed_hparams":
        for row in hparams_rows:
            rows_by_dataset_setting.setdefault((str(row["dataset"]), str(row["setting"])), []).append(row)
    else:
        for dataset, setting in dataset_setting_pairs:
            rows_by_dataset_setting[(dataset, setting)] = []

    for (dataset, setting), setting_hparams in sorted(rows_by_dataset_setting.items()):
        run_name = run_name_for_setting(setting)
        payload_path = args.result_root / run_name / f"{dataset}_seed{int(args.seed)}.json"
        if not payload_path.exists():
            print(f"[{dataset}/{setting}] skipping missing payload: {payload_path}", flush=True)
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        config = dict(payload["config"])
        needs_metric = any(method in {"metric_rbf", "structured_metric_rbf"} for method in selected_methods)
        mask_blocks, targets = build_group_mask_blocks(payload)
        if needs_metric:
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
            preferred_root = kmeans_full_root
            preferred_ckpt = preferred_root / dataset / "seed0" / "subset001" / "checkpoints" / "checkpoint_repeat01.pt"
            if not preferred_ckpt.exists():
                preferred_root = grouped_full_root
            full_checkpoint = resolve_full_checkpoint_path(
                dataset=dataset,
                seed=int(args.seed),
                pretrained_root=preferred_root,
                fallback_dir=full_fallback,
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
            print(f"[{dataset}/{setting}] computing metric_rbf gradient metric...", flush=True)
            gradients = compute_group_modality_gradients(
                model,
                data=data,
                group_indices=payload["group_partition"]["group_indices"],
                task=str(data["task"]),
                device=str(device),
                gradient_params=str(args.gradient_params),
            )
            metric, metric_stats = build_metric(gradients, normalization=str(args.metric_normalization))
        else:
            metric = np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64)
            metric_stats = {"metric_source": "identity_unused"}
        if args.search_mode == "fixed_hparams":
            for hparams in sorted(setting_hparams, key=lambda item: item["method"]):
                method_metric = metric
                if str(hparams["method"]) == "linear_regression":
                    method_metric = np.eye(mask_blocks["flat_deleted"].shape[1], dtype=np.float64)
                for split_seed in split_seeds:
                    result = evaluate_split(
                        hparams=hparams,
                        mask_blocks=mask_blocks,
                        targets=targets,
                        metric=method_metric,
                        split_seed=split_seed,
                        train_fraction=float(args.train_fraction),
                    )
                    result["metric_stats"] = metric_stats
                    records.append(result)
                    print(
                        f"  {result['method']} split_seed={split_seed}: "
                        f"LDS={result['test_lds']:.6f}, Pearson={result['test_pearson']:.6f}, "
                        f"RMSE={result['test_rmse']:.6f}",
                        flush=True,
                    )
        else:
            for split_seed in split_seeds:
                print(f"[{dataset}/{setting}] grid-search split_seed={split_seed}", flush=True)
                for method in selected_methods:
                    method_metric = metric if method in {"metric_rbf", "structured_metric_rbf"} else np.eye(
                        mask_blocks["flat_deleted"].shape[1], dtype=np.float64
                    )
                    result = evaluate_grid_for_split(
                        dataset=dataset,
                        setting=setting,
                        method=method,
                        mask_blocks=mask_blocks,
                        targets=targets,
                        metric=method_metric,
                        split_seed=split_seed,
                        train_fraction=float(args.train_fraction),
                        ridge_grid=[float(value) for value in args.ridge_grid],
                        gamma_grid=[float(value) for value in args.gamma_grid],
                        shared_grid=[float(value) for value in args.shared_grid],
                        rho_grid=[float(value) for value in args.rho_grid],
                    )
                    result["metric_stats"] = metric_stats
                    records.append(result)
                    print(
                        f"  {result['method']}: LDS={result['test_lds']:.6f}, "
                        f"Pearson={result['test_pearson']:.6f}, RMSE={result['test_rmse']:.6f}, "
                        f"ridge={result['ridge_lambda']}, gamma={result['gamma_scale']}, "
                        f"shared={result['structured_shared_weight']}",
                        flush=True,
                    )

    best_by_dataset_method = []
    comparison = []
    for dataset in sorted({str(row["dataset"]) for row in records}):
        method_bests = {}
        for method in selected_methods:
            candidates = [row for row in records if row["dataset"] == dataset and row["method"] == method]
            if candidates:
                best = max(candidates, key=score_key)
                best_by_dataset_method.append(best)
                method_bests[method] = best
        if all(method in method_bests for method in selected_methods):
            winner = max(selected_methods, key=lambda method: score_key(method_bests[method]))
            comparison.append(
                {
                    "dataset": dataset,
                    "winner": winner,
                    "best_by_method": method_bests,
                }
            )

    payload = {
        "config": {
            "source_hparams": str(args.hparams_json),
            "search_mode": str(args.search_mode),
            "methods": list(selected_methods),
            "split": f"{float(args.train_fraction):.6g}/{1.0 - float(args.train_fraction):.6g}",
            "split_seeds": split_seeds,
            "artifact_seed": int(args.seed),
            "metric_normalization": str(args.metric_normalization),
            "gradient_params": str(args.gradient_params),
            "result_root": str(args.result_root),
            "ridge_grid": [float(value) for value in args.ridge_grid],
            "gamma_grid": [float(value) for value in args.gamma_grid],
            "shared_grid": [float(value) for value in args.shared_grid],
            "rho_grid": [float(value) for value in args.rho_grid],
        },
        "comparison": comparison,
        "best_by_dataset_method": best_by_dataset_method,
        "top3_fixed_seed_summaries": summarize_topk_fixed_seed(records, top_k=3, methods=selected_methods),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def to_markdown(payload: dict[str, Any]) -> str:
    methods = [str(method) for method in payload["config"]["methods"]]
    wins = {method: 0 for method in methods}
    for row in payload["comparison"]:
        wins[str(row["winner"])] = wins.get(str(row["winner"]), 0) + 1
    lines = [
        "# Metric RBF vs Structured RBF vs Linear Shuffle Evaluation",
        "",
        f"- search mode: `{payload['config'].get('search_mode', 'fixed_hparams')}`",
        f"- source hparams: `{payload['config']['source_hparams']}`",
        f"- split: `{payload['config']['split']}`",
        f"- split seeds: `{payload['config']['split_seeds']}`",
        f"- metric: held-out Spearman LDS",
        f"- wins: `{wins}`",
        "",
            "## Winner Summary",
            "",
        "| dataset | winner | " + " | ".join(f"{method} best LDS" for method in methods) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in methods) + " |",
    ]
    for row in payload["comparison"]:
        best_by_method = row["best_by_method"]
        lines.append(
            f"| {row['dataset']} | {row['winner']} | "
            + " | ".join(f"{best_by_method[method]['test_lds']:.6f}" for method in methods)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best By Dataset And Method",
            "",
            "| dataset | method | setting | split_seed | LDS | Pearson | RMSE | ridge | gamma | shared |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(payload["best_by_dataset_method"], key=lambda item: (item["dataset"], item["method"])):
        shared = "" if row["structured_shared_weight"] is None else f"{row['structured_shared_weight']}"
        gamma = "" if row["gamma_scale"] is None else f"{row['gamma_scale']}"
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['setting']} | {row['split_seed']} | "
            f"{row['test_lds']:.6f} | {row['test_pearson']:.6f} | {row['test_rmse']:.6f} | "
            f"{row['ridge_lambda']} | {gamma} | {shared} |"
        )
    lines.extend(
        [
            "",
            "## Top-3 Seed Mean/Std",
            "",
            "| dataset | method | top-k | mean LDS | std LDS | seeds |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(payload["top3_fixed_seed_summaries"], key=lambda item: (item["dataset"], item["method"])):
        seeds = ", ".join(str(int(item["split_seed"])) for item in row["rows"])
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['top_k']} | "
            f"{row['mean_lds']:.6f} | {row['std_lds']:.6f} | {seeds} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
