from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fisher_geometry.data import load_dataset
from src.kernel_surrogate.kernel_attribution import rmse, spearmanr
from src.kernel_surrogate.real_data_cmis import (
    DEFAULT_PRETRAINED_FULL_CHECKPOINT_ROOT,
    analytical_second_derivative_at_zero,
    build_group_mask_blocks,
    build_metric,
    compute_group_modality_gradients,
    fit_surrogate_method,
    load_or_train_full_model,
    make_model_args,
    normalize_cmis_scores,
    predict_surrogate_method,
    rank_group_pairs,
    resolve_device,
    resolve_full_checkpoint_path,
    run_checkpoint_protocol_a_for_kernel_methods,
    summarize_pair_dominance,
)


DEFAULT_TRAIN_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "result_old"
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
    / "grouped50_subset100_data50_mod50"
)
DEFAULT_FULL_RESULT_DIR = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train" / "grouped50_full_data_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "gradest_train_real_eval_group_lds.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "gradest_train_real_eval_group_lds.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit group-level kernel surrogates on gradient-estimation subset losses "
            "and evaluate them on real LDS subset losses."
        )
    )
    parser.add_argument("--train-result-dir", type=Path, default=DEFAULT_TRAIN_RESULT_DIR)
    parser.add_argument("--eval-result-dir", type=Path, default=DEFAULT_EVAL_RESULT_DIR)
    parser.add_argument("--full-result-dir", type=Path, default=DEFAULT_FULL_RESULT_DIR)
    parser.add_argument("--datasets", nargs="+", default=["avmnist", "humor", "imdb", "mosei", "mosi", "sarcasm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--pretrained-full-checkpoint-root", type=Path, default=DEFAULT_PRETRAINED_FULL_CHECKPOINT_ROOT)
    parser.add_argument("--full-checkpoint-dir", type=Path, default=ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis")
    parser.add_argument(
        "--kernel-methods",
        nargs="+",
        default=["metric_rbf", "structured_metric_rbf", "structured_rbf", "raw_flat_rbf"],
    )
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
    parser.add_argument("--ridge-lambda", type=float, default=1e-5)
    parser.add_argument("--gamma-scale", type=float, default=1e-3)
    parser.add_argument("--structured-shared-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--cmis-normalization-eps", type=float, default=1e-12)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    device = resolve_device(str(args.device))
    records: list[dict[str, Any]] = []
    dataset_results: list[dict[str, Any]] = []

    for dataset in args.datasets:
        print(f"[{dataset}] loading train/eval LDS payloads...", flush=True)
        train_payload = load_payload(args.train_result_dir, dataset=dataset, seed=int(args.seed))
        eval_payload = load_payload(args.eval_result_dir, dataset=dataset, seed=int(args.seed))
        full_payload = load_payload(args.full_result_dir, dataset=dataset, seed=int(args.seed))
        full_loss = float(full_payload["runs"][0]["test_loss"])
        train_blocks, train_targets = build_group_mask_blocks(train_payload)
        eval_blocks, eval_targets = build_group_mask_blocks(eval_payload)
        combined_blocks = concat_group_mask_blocks(train_blocks, eval_blocks)
        train_indices = np.arange(train_targets.shape[0], dtype=np.int64)
        eval_indices = np.arange(train_targets.shape[0], train_targets.shape[0] + eval_targets.shape[0], dtype=np.int64)
        targets = np.concatenate([train_targets, eval_targets], axis=0)

        metric = None
        metric_stats = None
        if "metric_rbf" in args.kernel_methods:
            metric, metric_stats = compute_metric_for_dataset(
                dataset,
                train_payload=train_payload,
                seed=int(args.seed),
                device=device,
                feature_device=args.feature_device,
                pretrained_root=args.pretrained_full_checkpoint_root,
                fallback_dir=args.full_checkpoint_dir,
                gradient_params=str(args.gradient_params),
                metric_normalization=str(args.metric_normalization),
            )
        else:
            num_atoms = int(combined_blocks["flat_deleted"].shape[1])
            metric = np.eye(num_atoms, dtype=np.float64)
            metric_stats = {}

        method_rows: list[dict[str, Any]] = []
        for method in args.kernel_methods:
            print(
                f"[{dataset}] fitting {method} on {train_targets.shape[0]} grad-est subsets; "
                f"testing on {eval_targets.shape[0]} real subsets...",
                flush=True,
            )
            fit = fit_surrogate_method(
                str(method),
                combined_blocks,
                targets,
                metric,
                ridge_lambda=float(args.ridge_lambda),
                gamma_scale=float(args.gamma_scale),
                structured_shared_weight=float(args.structured_shared_weight),
                train_indices=train_indices,
            )
            raw_scores = analytical_second_derivative_at_zero(
                fit["train_masks"],
                fit["metric"],
                alpha=fit["alpha"],
                gamma=float(fit["gamma"]),
                num_groups=int(combined_blocks["num_groups"][0]),
                atom_offset=int(fit.get("atom_offset", 0)),
            )
            cmis_scores = normalize_cmis_scores(raw_scores, eps=float(args.cmis_normalization_eps))
            cmis_ranking = rank_group_pairs(cmis_scores, top_k=int(args.top_k))
            cmis_dominance = summarize_pair_dominance(cmis_ranking["top_pairs"])
            train_predictions = np.asarray(fit["train_predictions"], dtype=np.float64)
            eval_predictions = predict_surrogate_method(
                str(method),
                combined_blocks,
                fit,
                metric=metric,
                query_indices=eval_indices,
            )
            row = {
                "dataset": dataset,
                "seed": int(args.seed),
                "method": str(method),
                "heldout_lds": spearmanr(eval_predictions, eval_targets),
                "heldout_pearson": pearsonr(eval_predictions, eval_targets),
                "heldout_rmse": rmse(eval_predictions, eval_targets),
                "train_lds": spearmanr(train_predictions, train_targets),
                "train_pearson": pearsonr(train_predictions, train_targets),
                "train_rmse": rmse(train_predictions, train_targets),
                "num_train_subsets": int(train_targets.shape[0]),
                "num_eval_subsets": int(eval_targets.shape[0]),
                "train_target_mean": float(np.mean(train_targets)),
                "eval_target_mean": float(np.mean(eval_targets)),
                "train_target_std": float(np.std(train_targets)),
                "eval_target_std": float(np.std(eval_targets)),
                "ridge_lambda": float(args.ridge_lambda),
                "gamma": float(fit["gamma"]) if "gamma" in fit else None,
                "base_gamma": float(fit["base_gamma"]) if "base_gamma" in fit else None,
                "cmis_score_summary": {
                    "score_mean": cmis_ranking["score_mean"],
                    "score_std": cmis_ranking["score_std"],
                    "score_min": cmis_ranking["score_min"],
                    "score_max": cmis_ranking["score_max"],
                    "positive_fraction": cmis_ranking["positive_fraction"],
                },
                "top_pairs": cmis_ranking["top_pairs"],
                "top_pair_dominance": cmis_dominance,
            }
            records.append(row)
            method_rows.append(row)
            print(
                f"  {method}: real LDS={row['heldout_lds']:.6f}, "
                f"Pearson={row['heldout_pearson']:.6f}, RMSE={row['heldout_rmse']:.6f}",
                flush=True,
            )

        cmis_protocol_a = run_checkpoint_protocol_a_for_kernel_methods(
            method_rows,
            payload=eval_payload,
            full_loss=full_loss,
            top_k=int(args.top_k),
            random_repeats=int(args.random_repeats),
            random_seed=int(args.random_seed),
        )
        cmis_by_method = {str(row["method"]): row for row in cmis_protocol_a}
        for row in method_rows:
            protocol_row = cmis_by_method.get(str(row["method"]))
            if protocol_row is not None:
                row["cmis_protocol_a"] = protocol_row
        for row in cmis_protocol_a:
            if row.get("skipped"):
                continue
            print(
                f"  CMIS Protocol A {row['method']}: high increase="
                f"{row['high_loss_increase_vs_full']:.6f}, random increase="
                f"{row['random_loss_increase_mean']:.6f}, effect="
                f"{row['effect_size_vs_random_loss_increases']:.6f}",
                flush=True,
            )

        dataset_results.append(
            {
                "dataset": dataset,
                "seed": int(args.seed),
                "train_source": source_summary(train_payload),
                "eval_source": source_summary(eval_payload),
                "full_loss": full_loss,
                "metric_stats": metric_stats,
                "methods": method_rows,
                "cmis_protocol_a": cmis_protocol_a,
            }
        )

    output = {
        "config": {
            "train_result_dir": str(args.train_result_dir),
            "eval_result_dir": str(args.eval_result_dir),
            "full_result_dir": str(args.full_result_dir),
            "datasets": list(args.datasets),
            "seed": int(args.seed),
            "fit_target_source": "gradient_estimation_subset_test_loss",
            "eval_target_source": "real_train_subset_test_loss",
            "unit": "group",
            "gradient_params": str(args.gradient_params),
            "metric_normalization": str(args.metric_normalization),
            "kernel_methods": [str(method) for method in args.kernel_methods],
            "ridge_lambda": float(args.ridge_lambda),
            "gamma_scale": float(args.gamma_scale),
            "structured_shared_weight": float(args.structured_shared_weight),
            "top_k": int(args.top_k),
            "random_repeats": int(args.random_repeats),
            "random_seed": int(args.random_seed),
        },
        "summary": summarize_records(records),
        "cmis_protocol_a_summary": summarize_cmis_protocol_a(dataset_results),
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


def load_payload(result_dir: Path, *, dataset: str, seed: int) -> dict[str, Any]:
    path = result_dir / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing LDS result payload: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def concat_group_mask_blocks(lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if int(lhs["num_groups"][0]) != int(rhs["num_groups"][0]):
        raise ValueError(f"num_groups mismatch: {lhs['num_groups'][0]} vs {rhs['num_groups'][0]}")
    return {
        "flat_deleted": np.concatenate([lhs["flat_deleted"], rhs["flat_deleted"]], axis=0),
        "sample_deleted": np.concatenate([lhs["sample_deleted"], rhs["sample_deleted"]], axis=0),
        "modality_deleted": np.concatenate([lhs["modality_deleted"], rhs["modality_deleted"]], axis=0),
        "num_groups": lhs["num_groups"].copy(),
    }


def compute_metric_for_dataset(
    dataset: str,
    *,
    train_payload: dict[str, Any],
    seed: int,
    device: torch.device,
    feature_device: str | None,
    pretrained_root: Path,
    fallback_dir: Path,
    gradient_params: str,
    metric_normalization: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = dict(train_payload["config"])
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
    full_checkpoint = resolve_full_checkpoint_path(
        dataset=dataset,
        seed=seed,
        pretrained_root=pretrained_root,
        fallback_dir=fallback_dir,
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
        seed=seed,
        force_train=False,
    )
    model.to(device)
    model.eval()
    print(f"[{dataset}] computing {gradient_params} group-modality gradients on {device}...", flush=True)
    gradients = compute_group_modality_gradients(
        model,
        data=data,
        group_indices=train_payload["group_partition"]["group_indices"],
        task=str(data["task"]),
        device=str(device),
        gradient_params=gradient_params,
    )
    return build_metric(gradients, normalization=metric_normalization)


def make_model_args(config: dict[str, Any], *, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dim=int(config.get("hidden_dim", 256)),
        proj_dim=int(config.get("proj_dim", 256)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        batch_size=int(config.get("batch_size", 4196)),
        eval_batch_size=int(config.get("eval_batch_size") or config.get("batch_size", 4196)),
        epochs=int(config.get("epochs", 15)),
        lr=float(config.get("lr", 5e-4)),
        device=device,
    )


def pearsonr(lhs: Sequence[float] | np.ndarray, rhs: Sequence[float] | np.ndarray) -> float:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(left * right) / denom)


def summarize_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(str(record["method"]), []).append(record)
    summary: list[dict[str, Any]] = []
    for method, rows in by_method.items():
        summary.append(
            {
                "method": method,
                "num_dataset_runs": len(rows),
                "mean_heldout_lds": float(np.nanmean([row["heldout_lds"] for row in rows])),
                "std_heldout_lds": float(np.nanstd([row["heldout_lds"] for row in rows])),
                "mean_heldout_pearson": float(np.nanmean([row["heldout_pearson"] for row in rows])),
                "mean_heldout_rmse": float(np.nanmean([row["heldout_rmse"] for row in rows])),
                "mean_train_lds": float(np.nanmean([row["train_lds"] for row in rows])),
                "mean_train_rmse": float(np.nanmean([row["train_rmse"] for row in rows])),
            }
        )
    return summary


def summarize_cmis_protocol_a(dataset_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for dataset_result in dataset_results:
        for row in dataset_result.get("cmis_protocol_a", []):
            if row.get("skipped"):
                continue
            by_method.setdefault(str(row["method"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for method, rows in by_method.items():
        summary.append(
            {
                "method": method,
                "num_dataset_runs": len(rows),
                "mean_effect_size": float(np.nanmean([row["effect_size_vs_random_loss_increases"] for row in rows])),
                "std_effect_size": float(np.nanstd([row["effect_size_vs_random_loss_increases"] for row in rows])),
                "mean_high_loss_increase": float(np.nanmean([row["high_loss_increase_vs_full"] for row in rows])),
                "mean_random_loss_increase": float(np.nanmean([row["random_loss_increase_mean"] for row in rows])),
                "mean_observations": float(np.nanmean([row["num_observations"] for row in rows])),
            }
        )
    return summary


def source_summary(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    return {
        "dataset": payload.get("dataset"),
        "seed": payload.get("seed"),
        "num_runs": len(payload.get("runs", [])),
        "sampling_backend": config.get("sampling_backend", "real_train"),
        "run_name": config.get("run_name"),
        "num_groups": config.get("num_groups"),
        "data_keep_rate": config.get("data_keep_rate"),
        "modality_keep_rate": config.get("modality_keep_rate"),
        "gradient_space": config.get("gradient_space"),
        "gradient_max_iter": config.get("gradient_max_iter"),
    }


def to_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# Gradient-Estimation Train / Real-Subset Eval Group LDS",
        "",
        f"- Fit target source: `{config['fit_target_source']}`",
        f"- Eval target source: `{config['eval_target_source']}`",
        f"- Train result dir: `{config['train_result_dir']}`",
        f"- Eval result dir: `{config['eval_result_dir']}`",
        f"- Full result dir: `{config['full_result_dir']}`",
        f"- Unit: `{config['unit']}`",
        f"- Metric RBF gradients: `{config['gradient_params']}`",
        f"- Metric normalization: `{config['metric_normalization']}`",
        "",
        "## Summary",
        "",
        "| method | mean real LDS | std real LDS | mean real Pearson | mean real RMSE | mean grad-train LDS | mean grad-train RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['method']} | {row['mean_heldout_lds']:.6g} | {row['std_heldout_lds']:.6g} | "
            f"{row['mean_heldout_pearson']:.6g} | {row['mean_heldout_rmse']:.6g} | "
            f"{row['mean_train_lds']:.6g} | {row['mean_train_rmse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## CMIS Protocol A Summary",
            "",
            "| method | mean effect size | std effect size | mean high increase | mean random increase | mean observations |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["cmis_protocol_a_summary"]:
        lines.append(
            f"| {row['method']} | {row['mean_effect_size']:.6g} | {row['std_effect_size']:.6g} | "
            f"{row['mean_high_loss_increase']:.6g} | {row['mean_random_loss_increase']:.6g} | "
            f"{row['mean_observations']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Per Dataset",
            "",
            "| dataset | method | real LDS | real Pearson | real RMSE | grad-train LDS | train subsets | real subsets |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset_result in payload["dataset_results"]:
        dataset = dataset_result["dataset"]
        for row in dataset_result["methods"]:
            lines.append(
                f"| {dataset} | {row['method']} | {row['heldout_lds']:.6g} | "
                f"{row['heldout_pearson']:.6g} | {row['heldout_rmse']:.6g} | "
                f"{row['train_lds']:.6g} | {row['num_train_subsets']} | {row['num_eval_subsets']} |"
            )
    lines.extend(
        [
            "",
            "## CMIS Protocol A Per Dataset",
            "",
            "| dataset | method | high increase | random increase | effect size | observations | max row count | max col count |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset_result in payload["dataset_results"]:
        dataset = dataset_result["dataset"]
        dominance_by_method = {
            str(row["method"]): row.get("top_pair_dominance", {})
            for row in dataset_result.get("methods", [])
        }
        for row in dataset_result.get("cmis_protocol_a", []):
            if row.get("skipped"):
                continue
            dominance = dominance_by_method.get(str(row["method"]), {})
            lines.append(
                f"| {dataset} | {row['method']} | {row['high_loss_increase_vs_full']:.6g} | "
                f"{row['random_loss_increase_mean']:.6g} | "
                f"{row['effect_size_vs_random_loss_increases']:.6g} | {row['num_observations']} | "
                f"{dominance.get('max_row_count', '')} | {dominance.get('max_col_count', '')} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
