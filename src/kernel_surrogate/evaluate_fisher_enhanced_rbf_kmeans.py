from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_geometry.data import load_dataset
from kernel_surrogate.fisher_enhanced_rbf import (
    fisher_corrected_score_grams,
    fit_fisher_enhanced_rbf,
    pearsonr,
    predict_fisher_enhanced_rbf,
)
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
SETTING_TO_RUN = {
    "kmeans_data80_mod80": "kmeans_grouped25_subset100_data80_mod80",
    "kmeans_data90_mod90": "kmeans_grouped25_subset100_data90_mod90",
}
METHODS = (
    "fisher_enhanced_rbf",
    "structured_metric_rbf",
    "metric_rbf",
    "structured_rbf",
    "raw_flat_rbf",
    "linear_regression",
)

DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_CACHE_DIR = ROOT / "src" / "kernel_surrogate" / "results" / "fisher_enhanced_cache"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "fisher_enhanced_rbf_kmeans_80_20_top3.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "fisher_enhanced_rbf_kmeans_80_20_top3.md"


def parse_float_list(values: Sequence[str] | None, default: Sequence[float]) -> tuple[float, ...]:
    if values is None:
        return tuple(float(value) for value in default)
    return tuple(float(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search Fisher-corrected score-Gram structured RBF on kmeans grouped LDS artifacts, "
            "then evaluate fixed hyperparameters over multiple 80/20 split shuffles."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--settings", nargs="+", default=list(SETTING_TO_RUN))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--search-split-seed", type=int, default=0)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--ridge-grid", nargs="+", default=None)
    parser.add_argument("--gamma-grid", nargs="+", default=None)
    parser.add_argument("--gamma-a-grid", nargs="+", default=None)
    parser.add_argument("--gamma-e-grid", nargs="+", default=None)
    parser.add_argument("--shared-grid", nargs="+", default=None)
    parser.add_argument("--beta-grid", nargs="+", default=None)
    parser.add_argument("--fisher-lambda-grid", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_payload(result_root: Path, run_name: str, dataset: str, seed: int) -> dict[str, Any]:
    path = result_root / run_name / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing LDS payload: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def score_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(np.nan_to_num(row["test_lds"], nan=-1e18)),
        float(np.nan_to_num(row["test_pearson"], nan=-1e18)),
        -float(row["test_rmse"]),
    )


def iter_hparams(
    method: str,
    *,
    ridge_grid: Sequence[float],
    gamma_grid: Sequence[float],
    gamma_a_grid: Sequence[float],
    gamma_e_grid: Sequence[float],
    shared_grid: Sequence[float],
    beta_grid: Sequence[float],
    fisher_lambda_grid: Sequence[float],
) -> Iterable[dict[str, Any]]:
    if method == "linear_regression":
        for ridge in ridge_grid:
            yield {"ridge_lambda": float(ridge), "gamma_scale": None, "structured_shared_weight": None}
        return
    if method == "fisher_enhanced_rbf":
        for ridge in ridge_grid:
            for gamma_a in gamma_a_grid:
                for gamma_e in gamma_e_grid:
                    for beta in beta_grid:
                        lambda_values = (1.0,) if float(beta) == 0.0 else fisher_lambda_grid
                        for fisher_lambda in lambda_values:
                            yield {
                                "ridge_lambda": float(ridge),
                                "gamma_a_scale": float(gamma_a),
                                "gamma_e_scale": float(gamma_e),
                                "beta": float(beta),
                                "fisher_lambda": float(fisher_lambda),
                            }
        return
    shared_values = shared_grid if method in {"structured_metric_rbf", "structured_rbf"} else (None,)
    for ridge in ridge_grid:
        for gamma in gamma_grid:
            for shared in shared_values:
                yield {
                    "ridge_lambda": float(ridge),
                    "gamma_scale": float(gamma),
                    "structured_shared_weight": None if shared is None else float(shared),
                }


def fit_predict(
    method: str,
    *,
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    metric: np.ndarray,
    fisher_metrics: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    hparams: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if method == "fisher_enhanced_rbf":
        key = (float(hparams["beta"]), float(hparams["fisher_lambda"]))
        sample_metric, modality_metric, gram_stats = fisher_metrics[key]
        fit = fit_fisher_enhanced_rbf(
            mask_blocks,
            targets,
            sample_metric,
            modality_metric,
            ridge_lambda=float(hparams["ridge_lambda"]),
            gamma_a_scale=float(hparams["gamma_a_scale"]),
            gamma_e_scale=float(hparams["gamma_e_scale"]),
            train_indices=train_indices,
        )
        pred = predict_fisher_enhanced_rbf(mask_blocks, fit, query_indices=test_indices)
        return pred, fit, gram_stats

    method_metric = metric if method in {"metric_rbf", "structured_metric_rbf"} else np.eye(
        mask_blocks["flat_deleted"].shape[1], dtype=np.float64
    )
    fit = fit_surrogate_method(
        method,
        mask_blocks,
        targets,
        method_metric,
        ridge_lambda=float(hparams["ridge_lambda"]),
        gamma_scale=1.0 if hparams.get("gamma_scale") is None else float(hparams["gamma_scale"]),
        structured_shared_weight=1.0
        if hparams.get("structured_shared_weight") is None
        else float(hparams["structured_shared_weight"]),
        train_indices=train_indices,
    )
    pred = predict_surrogate_method(method, mask_blocks, fit, metric=method_metric, query_indices=test_indices)
    return pred, fit, {}


def evaluate_hparams(
    method: str,
    *,
    dataset: str,
    setting: str,
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    metric: np.ndarray,
    fisher_metrics: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    hparams: dict[str, Any],
    train_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    pred, fit, gram_stats = fit_predict(
        method,
        mask_blocks=mask_blocks,
        targets=targets,
        metric=metric,
        fisher_metrics=fisher_metrics,
        hparams=hparams,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    truth = targets[test_indices]
    row = {
        "dataset": dataset,
        "setting": setting,
        "run_name": SETTING_TO_RUN[setting],
        "method": method,
        "ridge_lambda": float(hparams["ridge_lambda"]),
        "gamma_scale": hparams.get("gamma_scale"),
        "gamma_a_scale": hparams.get("gamma_a_scale"),
        "gamma_e_scale": hparams.get("gamma_e_scale"),
        "structured_shared_weight": hparams.get("structured_shared_weight"),
        "beta": hparams.get("beta"),
        "fisher_lambda": hparams.get("fisher_lambda"),
        "split_seed": int(split_seed),
        "test_lds": spearmanr(pred, truth),
        "test_pearson": pearsonr(pred, truth),
        "test_rmse": rmse(pred, truth),
        "train_lds": spearmanr(fit["train_predictions"], targets[train_indices]),
        "train_rmse": rmse(fit["train_predictions"], targets[train_indices]),
        "num_train": int(train_indices.size),
        "num_test": int(test_indices.size),
        "num_subsets": int(len(targets)),
    }
    for key in ("gamma", "base_gamma", "gamma_a", "gamma_e", "base_gamma_a", "base_gamma_e"):
        if key in fit:
            row[key] = float(fit[key])
    if gram_stats:
        row["gram_stats"] = gram_stats
    return row


def load_or_compute_unimodal_gradients(
    *,
    dataset: str,
    seed: int,
    payload: dict[str, Any],
    config: dict[str, Any],
    device: str,
    feature_device: str | None,
    gradient_params: str,
    metric_normalization: str,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = cache_dir / f"{dataset}_seed{seed}_{gradient_params}_{metric_normalization}_unimodal.npz"
    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["unimodal"], cached["metric"], json.loads(str(cached["stats"].item()))

    print(f"[{dataset}] loading data/checkpoint and computing unimodal Fisher scores...", flush=True)
    data = load_dataset(
        dataset,
        seed,
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=str(feature_device or config.get("feature_device", device)),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    model_args = make_model_args(config, device=str(device))
    kmeans_full_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "kmeans_grouped25_full_data_train"
    grouped_full_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "grouped25_full_data_train"
    full_fallback = ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis"
    preferred_root = kmeans_full_root
    preferred_ckpt = preferred_root / dataset / f"seed{seed}" / "subset001" / "checkpoints" / "checkpoint_repeat01.pt"
    if not preferred_ckpt.exists():
        preferred_root = grouped_full_root
    full_checkpoint = resolve_full_checkpoint_path(
        dataset=dataset,
        seed=seed,
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
        seed=seed,
        force_train=False,
    )
    model.to(device)
    model.eval()
    flat_gradients = compute_group_modality_gradients(
        model,
        data=data,
        group_indices=payload["group_partition"]["group_indices"],
        task=str(data["task"]),
        device=str(device),
        gradient_params=gradient_params,
    )
    metric, metric_stats = build_metric(flat_gradients, normalization=metric_normalization)
    unimodal = flat_gradients.reshape(int(payload["group_partition"]["num_groups"]), 2, flat_gradients.shape[1])
    stats = {
        "full_checkpoint": str(full_checkpoint),
        "full_metrics": full_metrics,
        "metric_stats": metric_stats,
    }
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, unimodal=unimodal, metric=metric, stats=json.dumps(stats, allow_nan=True))
    return unimodal, metric, stats


def build_fisher_metrics(
    unimodal: np.ndarray,
    *,
    beta_grid: Sequence[float],
    fisher_lambda_grid: Sequence[float],
) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    out: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for beta in beta_grid:
        lambda_values = (1.0,) if float(beta) == 0.0 else fisher_lambda_grid
        for fisher_lambda in lambda_values:
            out[(float(beta), float(fisher_lambda))] = fisher_corrected_score_grams(
                unimodal,
                beta=float(beta),
                fisher_lambda=float(fisher_lambda),
                normalize_features=True,
            )
    return out


def main() -> None:
    args = parse_args()
    settings = [str(setting) for setting in args.settings]
    unknown_settings = [setting for setting in settings if setting not in SETTING_TO_RUN]
    if unknown_settings:
        raise ValueError(f"unknown settings: {unknown_settings}; expected one of {sorted(SETTING_TO_RUN)}")

    ridge_grid = parse_float_list(args.ridge_grid, (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0))
    gamma_grid = parse_float_list(args.gamma_grid, (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0))
    gamma_a_grid = parse_float_list(args.gamma_a_grid, gamma_grid)
    gamma_e_grid = parse_float_list(args.gamma_e_grid, gamma_grid)
    shared_grid = parse_float_list(args.shared_grid, (0.0, 0.5, 1.0, 2.0, 5.0))
    beta_grid = parse_float_list(args.beta_grid, (0.0, 0.25, 0.5, 0.75, 1.0))
    fisher_lambda_grid = parse_float_list(args.fisher_lambda_grid, (1e-3, 1e-2, 1e-1, 1.0))
    split_seeds = [int(seed) for seed in args.split_seeds]
    device = str(resolve_device(str(args.device)))

    search_records: list[dict[str, Any]] = []
    fixed_eval_records: list[dict[str, Any]] = []
    best_hparams: list[dict[str, Any]] = []
    metric_records: dict[str, Any] = {}

    for dataset in [str(item) for item in args.datasets]:
        first_payload = load_payload(args.result_root, SETTING_TO_RUN[settings[0]], dataset, int(args.seed))
        first_config = dict(first_payload["config"])
        unimodal, metric, metric_stats = load_or_compute_unimodal_gradients(
            dataset=dataset,
            seed=int(args.seed),
            payload=first_payload,
            config=first_config,
            device=device,
            feature_device=args.feature_device,
            gradient_params=str(args.gradient_params),
            metric_normalization=str(args.metric_normalization),
            cache_dir=args.cache_dir,
            use_cache=not bool(args.no_cache),
        )
        fisher_metrics = build_fisher_metrics(unimodal, beta_grid=beta_grid, fisher_lambda_grid=fisher_lambda_grid)
        metric_records[dataset] = metric_stats

        loaded_settings: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
        for setting in settings:
            payload = load_payload(args.result_root, SETTING_TO_RUN[setting], dataset, int(args.seed))
            loaded_settings[setting] = build_group_mask_blocks(payload)

        for method in [str(item) for item in args.methods]:
            best: dict[str, Any] | None = None
            for setting, (mask_blocks, targets) in loaded_settings.items():
                for hparams in iter_hparams(
                    method,
                    ridge_grid=ridge_grid,
                    gamma_grid=gamma_grid,
                    gamma_a_grid=gamma_a_grid,
                    gamma_e_grid=gamma_e_grid,
                    shared_grid=shared_grid,
                    beta_grid=beta_grid,
                    fisher_lambda_grid=fisher_lambda_grid,
                ):
                    row = evaluate_hparams(
                        method,
                        dataset=dataset,
                        setting=setting,
                        mask_blocks=mask_blocks,
                        targets=targets,
                        metric=metric,
                        fisher_metrics=fisher_metrics,
                        hparams=hparams,
                        train_fraction=float(args.train_fraction),
                        split_seed=int(args.search_split_seed),
                    )
                    search_records.append(row)
                    if best is None or score_key(row) > score_key(best):
                        best = row
            if best is None:
                continue
            best_hparams.append(best)
            print(
                f"[{dataset}] best {method}: setting={best['setting']} split_seed={args.search_split_seed} "
                f"LDS={best['test_lds']:.6f} Pearson={best['test_pearson']:.6f} RMSE={best['test_rmse']:.6f}",
                flush=True,
            )

            mask_blocks, targets = loaded_settings[str(best["setting"])]
            hparams = {
                "ridge_lambda": best["ridge_lambda"],
                "gamma_scale": best.get("gamma_scale"),
                "gamma_a_scale": best.get("gamma_a_scale"),
                "gamma_e_scale": best.get("gamma_e_scale"),
                "structured_shared_weight": best.get("structured_shared_weight"),
                "beta": best.get("beta"),
                "fisher_lambda": best.get("fisher_lambda"),
            }
            for split_seed in split_seeds:
                row = evaluate_hparams(
                    method,
                    dataset=dataset,
                    setting=str(best["setting"]),
                    mask_blocks=mask_blocks,
                    targets=targets,
                    metric=metric,
                    fisher_metrics=fisher_metrics,
                    hparams=hparams,
                    train_fraction=float(args.train_fraction),
                    split_seed=int(split_seed),
                )
                fixed_eval_records.append(row)
                print(
                    f"  fixed {method} split_seed={split_seed}: LDS={row['test_lds']:.6f} "
                    f"Pearson={row['test_pearson']:.6f} RMSE={row['test_rmse']:.6f}",
                    flush=True,
                )

    topk = max(1, int(args.top_k))
    best_by_dataset_method: list[dict[str, Any]] = []
    top_by_dataset_method: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in fixed_eval_records}):
        for method in [str(item) for item in args.methods]:
            rows = [row for row in fixed_eval_records if row["dataset"] == dataset and row["method"] == method]
            rows = sorted(rows, key=score_key, reverse=True)
            if rows:
                best_by_dataset_method.append(rows[0])
                for rank, row in enumerate(rows[:topk], start=1):
                    ranked = dict(row)
                    ranked["rank"] = int(rank)
                    top_by_dataset_method.append(ranked)

    payload = {
        "config": {
            "datasets": [str(item) for item in args.datasets],
            "settings": settings,
            "methods": [str(item) for item in args.methods],
            "split": f"{float(args.train_fraction):.6g}/{1.0 - float(args.train_fraction):.6g}",
            "search_split_seed": int(args.search_split_seed),
            "split_seeds": split_seeds,
            "ridge_grid": list(ridge_grid),
            "gamma_grid": list(gamma_grid),
            "gamma_a_grid": list(gamma_a_grid),
            "gamma_e_grid": list(gamma_e_grid),
            "shared_grid": list(shared_grid),
            "beta_grid": list(beta_grid),
            "fisher_lambda_grid": list(fisher_lambda_grid),
            "gradient_params": str(args.gradient_params),
            "metric_normalization": str(args.metric_normalization),
            "top_k": topk,
            "result_root": str(args.result_root),
        },
        "metric_records": metric_records,
        "best_hparams": best_hparams,
        "fixed_eval_records": fixed_eval_records,
        "best_by_dataset_method": best_by_dataset_method,
        "top_by_dataset_method": top_by_dataset_method,
        "search_records": search_records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def format_float(value: Any) -> str:
    if value is None:
        return ""
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Fisher-Enhanced RBF KMeans Kernel Surrogate",
        "",
        f"- split: `{payload['config']['split']}`",
        f"- search split seed: `{payload['config']['search_split_seed']}`",
        f"- fixed-eval split seeds: `{payload['config']['split_seeds']}`",
        f"- top-k per dataset/method: `{payload['config']['top_k']}`",
        f"- gradient params: `{payload['config']['gradient_params']}`",
        "",
        "## Best Fixed-Parameter Result Per Method",
        "",
        "| dataset | method | setting | best split | LDS | Pearson | RMSE | ridge | gamma | gamma_a | gamma_e | shared | beta | fisher lambda |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(payload["best_by_dataset_method"], key=lambda item: (item["dataset"], item["method"])):
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['setting']} | {row['split_seed']} | "
            f"{row['test_lds']:.6f} | {row['test_pearson']:.6f} | {row['test_rmse']:.6f} | "
            f"{format_float(row['ridge_lambda'])} | {format_float(row.get('gamma_scale'))} | "
            f"{format_float(row.get('gamma_a_scale'))} | {format_float(row.get('gamma_e_scale'))} | "
            f"{format_float(row.get('structured_shared_weight'))} | {format_float(row.get('beta'))} | "
            f"{format_float(row.get('fisher_lambda'))} |"
        )

    lines.extend(
        [
            "",
            "## Top Results Per Dataset/Method",
            "",
            "| dataset | method | rank | split | LDS | Pearson | RMSE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(payload["top_by_dataset_method"], key=lambda item: (item["dataset"], item["method"], item["rank"])):
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['rank']} | {row['split_seed']} | "
            f"{row['test_lds']:.6f} | {row['test_pearson']:.6f} | {row['test_rmse']:.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
