from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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

from kernel_surrogate.evaluate_kernel_attribution import _resolve_dataset_seed_pairs
from kernel_surrogate.evaluate_sample_level_lds import (
    DEFAULT_CACHE_DIR,
    EXISTING_LDS_DATASETS,
    MultiTargetFit,
    extract_per_sample_losses_from_runs,
    fit_multi_target_surrogate,
    predict_multi_target,
    samplewise_spearman,
    summarize_sample_level_records,
)
from kernel_surrogate.kernel_attribution import (
    DEFAULT_METHODS,
    KernelAttributionConfig,
    MaskBlocks,
    build_mask_blocks,
    normalize_method_name,
    rmse,
)


DEFAULT_TRAIN_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "gradest_fullparam_all6_g50_s50_iter5_loss1000"
)
DEFAULT_EVAL_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "cross_attention_group_unit_data_keep25_mod_keep90_g50_trainavg3"
)
DEFAULT_OUTPUT_JSON = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "results"
    / "gradest_train_real_eval_sample_lds.json"
)
DEFAULT_OUTPUT_MD = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "results"
    / "gradest_train_real_eval_sample_lds.md"
)


def main() -> None:
    default_config = KernelAttributionConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Fit sample-level kernel surrogates on gradient-estimation subset losses "
            "and evaluate the predicted subset ranking against real-train subset losses."
        )
    )
    parser.add_argument("--train-result-dir", type=Path, default=DEFAULT_TRAIN_RESULT_DIR)
    parser.add_argument("--eval-result-dir", type=Path, default=DEFAULT_EVAL_RESULT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(EXISTING_LDS_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--input-unit", choices=["sample", "group"], default="group")
    parser.add_argument("--search-hyperparams", action="store_true")
    parser.add_argument("--ridge-lambda", type=float, default=default_config.ridge_lambda)
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=list(default_config.ridge_grid))
    parser.add_argument("--gamma-scale-grid", nargs="+", type=float, default=list(default_config.gamma_scale_grid))
    parser.add_argument("--modality-shared-grid", nargs="+", type=float, default=list(default_config.modality_shared_grid))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    train_pairs = set(_resolve_dataset_seed_pairs(args.train_result_dir, args.datasets, args.seeds))
    eval_pairs = set(_resolve_dataset_seed_pairs(args.eval_result_dir, args.datasets, args.seeds))
    dataset_seed_pairs = sorted(train_pairs & eval_pairs)
    if not dataset_seed_pairs:
        raise FileNotFoundError(
            "no overlapping dataset/seed result files between "
            f"train_result_dir={args.train_result_dir} and eval_result_dir={args.eval_result_dir}"
        )

    records: list[dict[str, Any]] = []
    dataset_results: list[dict[str, Any]] = []

    for dataset, seed in dataset_seed_pairs:
        train_payload, train_runs, train_blocks, train_losses = _load_blocks_and_losses(
            args.train_result_dir,
            dataset=dataset,
            seed=seed,
            input_unit=args.input_unit,
        )
        eval_payload, eval_runs, eval_blocks, eval_losses = _load_blocks_and_losses(
            args.eval_result_dir,
            dataset=dataset,
            seed=seed,
            input_unit=args.input_unit,
        )
        train_losses, eval_losses = _align_loss_columns(train_losses, eval_losses)
        combined_blocks = _concat_blocks(train_blocks, eval_blocks)
        train_indices = np.arange(train_losses.shape[0], dtype=np.int64)
        eval_indices = np.arange(
            train_losses.shape[0],
            train_losses.shape[0] + eval_losses.shape[0],
            dtype=np.int64,
        )

        print(
            f"[{dataset}] seed={seed}: fitting on {train_losses.shape[0]} gradient-estimation subsets, "
            f"evaluating on {eval_losses.shape[0]} real subsets, "
            f"{eval_losses.shape[1]} test samples...",
            flush=True,
        )

        method_rows: list[dict[str, Any]] = []
        for method in args.methods:
            method_name = normalize_method_name(method)
            fit = fit_multi_target_surrogate(
                combined_blocks,
                targets=train_losses,
                train_indices=train_indices,
                test_indices=eval_indices,
                method=method_name,
                ridge_lambda=args.ridge_lambda,
                search_hyperparams=args.search_hyperparams,
                search_strategy="loo",
                ridge_grid=args.ridge_grid,
                gamma_scale_grid=args.gamma_scale_grid,
                modality_shared_grid=args.modality_shared_grid,
            )
            predictions = predict_multi_target(combined_blocks, fit, query_indices=eval_indices)
            sample_lds = samplewise_spearman(predictions, eval_losses)
            row = {
                "dataset": dataset,
                "seed": int(seed),
                "method": method_name,
                "mean_sample_lds": float(np.nanmean(sample_lds)),
                "std_sample_lds": float(np.nanstd(sample_lds)),
                "median_sample_lds": float(np.nanmedian(sample_lds)),
                "rmse": rmse(predictions.reshape(-1), eval_losses.reshape(-1)),
                "num_train_subsets": int(train_losses.shape[0]),
                "num_test_subsets": int(eval_losses.shape[0]),
                "num_test_samples": int(eval_losses.shape[1]),
                "ridge_lambda": float(fit.ridge_lambda),
                "loo_mean_sample_lds": fit.loo_mean_sample_lds,
                "loo_rmse": fit.loo_rmse,
                "gamma_sample": fit.gamma_sample,
                "gamma_modality": fit.gamma_modality,
                "gamma_flat": fit.gamma_flat,
                "modality_shared_weight": fit.modality_shared_weight,
                "linear_modality_weight": fit.linear_modality_weight,
                "train_result_file": str(args.train_result_dir / f"{dataset}_seed{seed}.json"),
                "eval_result_file": str(args.eval_result_dir / f"{dataset}_seed{seed}.json"),
            }
            records.append(row)
            method_rows.append(row)
            print(
                f"  {method_name}: real-eval mean sample LDS={row['mean_sample_lds']:.6f}, "
                f"RMSE={row['rmse']:.6f}"
                + (
                    f", grad-train LOO={row['loo_mean_sample_lds']:.6f}"
                    if row["loo_mean_sample_lds"] is not None
                    else ""
                ),
                flush=True,
            )

        dataset_results.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "train_source": _source_summary(train_payload, train_runs),
                "eval_source": _source_summary(eval_payload, eval_runs),
                "methods": method_rows,
            }
        )

    output = {
        "config": {
            "train_result_dir": str(args.train_result_dir),
            "eval_result_dir": str(args.eval_result_dir),
            "datasets": [dataset for dataset, _ in dataset_seed_pairs],
            "seeds": sorted({int(seed) for _, seed in dataset_seed_pairs}),
            "target_metric": "per_sample_test_loss",
            "input_unit": args.input_unit,
            "fit_target_source": "gradient_estimation",
            "eval_target_source": "real_subset_training",
            "search_hyperparams": bool(args.search_hyperparams),
            "ridge_lambda": float(args.ridge_lambda),
            "ridge_grid": [float(value) for value in args.ridge_grid],
            "gamma_scale_grid": [float(value) for value in args.gamma_scale_grid],
            "modality_shared_grid": [float(value) for value in args.modality_shared_grid],
            "methods": [normalize_method_name(method) for method in args.methods],
        },
        "summary": summarize_sample_level_records(records),
        "dataset_results": dataset_results,
        "records": records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def _load_blocks_and_losses(
    result_dir: Path,
    *,
    dataset: str,
    seed: int,
    input_unit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], MaskBlocks, np.ndarray]:
    path = result_dir / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing result file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = sorted(payload["runs"], key=lambda run: int(run.get("sample_index", 0)))
    blocks = build_mask_blocks(payload, target_metric="test_loss", input_unit=input_unit)
    losses = extract_per_sample_losses_from_runs(runs)
    if losses is None:
        raise ValueError(f"{path}: missing aligned non-empty test_sample_losses")
    if losses.shape[0] != blocks.targets.shape[0]:
        raise ValueError(
            f"{path}: loss rows {losses.shape[0]} != mask rows {blocks.targets.shape[0]}"
        )
    return payload, runs, blocks, losses


def _align_loss_columns(lhs: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = min(int(lhs.shape[1]), int(rhs.shape[1]))
    if width <= 0:
        raise ValueError("cannot evaluate with zero per-sample loss columns")
    return lhs[:, :width].astype(np.float64, copy=False), rhs[:, :width].astype(np.float64, copy=False)


def _concat_blocks(train_blocks: MaskBlocks, eval_blocks: MaskBlocks) -> MaskBlocks:
    if train_blocks.dataset_size != eval_blocks.dataset_size:
        raise ValueError(
            "train/eval block dataset_size mismatch: "
            f"{train_blocks.dataset_size} vs {eval_blocks.dataset_size}"
        )
    if train_blocks.num_modalities != eval_blocks.num_modalities:
        raise ValueError(
            "train/eval block num_modalities mismatch: "
            f"{train_blocks.num_modalities} vs {eval_blocks.num_modalities}"
        )
    return replace(
        train_blocks,
        sample_deleted=np.concatenate([train_blocks.sample_deleted, eval_blocks.sample_deleted], axis=0),
        modality_deleted=np.concatenate([train_blocks.modality_deleted, eval_blocks.modality_deleted], axis=0),
        flat_deleted=np.concatenate([train_blocks.flat_deleted, eval_blocks.flat_deleted], axis=0),
        targets=np.concatenate([train_blocks.targets, eval_blocks.targets], axis=0),
        run_ids=tuple(range(train_blocks.targets.shape[0] + eval_blocks.targets.shape[0])),
    )


def _source_summary(payload: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    config = payload.get("config", {})
    return {
        "dataset": payload.get("dataset"),
        "seed": payload.get("seed"),
        "num_runs": int(len(runs)),
        "run_name": config.get("run_name"),
        "sampling_backend": config.get("sampling_backend", "real_train"),
        "gradient_space": config.get("gradient_space"),
        "gradient_max_iter": config.get("gradient_max_iter"),
    }


def to_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# Gradient-Estimation Train / Real-Subset Eval LDS",
        "",
        f"- Fit target source: `{config['fit_target_source']}`",
        f"- Eval target source: `{config['eval_target_source']}`",
        f"- Train result dir: `{config['train_result_dir']}`",
        f"- Eval result dir: `{config['eval_result_dir']}`",
        f"- Input unit: `{config['input_unit']}`",
        f"- Hyperparameter search on gradient-estimation train targets: `{config['search_hyperparams']}`",
        "",
        "## Summary",
        "",
        "| Method | Dataset runs | Real-eval mean LDS | Std LDS | Real-eval mean RMSE | Std RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {method} | {num_dataset_runs} | {mean_sample_lds:.6f} | {std_sample_lds:.6f} | {mean_rmse:.6f} | {std_rmse:.6f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Per Dataset",
            "",
            "| Dataset | Method | Real-eval mean LDS | Median LDS | RMSE | Grad-train LOO LDS | Train subsets | Eval subsets | Test samples |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_result in payload["dataset_results"]:
        dataset = dataset_result["dataset"]
        for row in dataset_result["methods"]:
            loo = "" if row.get("loo_mean_sample_lds") is None else f"{float(row['loo_mean_sample_lds']):.6f}"
            lines.append(
                f"| {dataset} | {row['method']} | {row['mean_sample_lds']:.6f} | "
                f"{row['median_sample_lds']:.6f} | {row['rmse']:.6f} | {loo} | "
                f"{row['num_train_subsets']} | {row['num_test_subsets']} | {row['num_test_samples']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
