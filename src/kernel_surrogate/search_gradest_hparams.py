from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
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

from kernel_surrogate import lds_real_train
from kernel_surrogate.evaluate_gradest_to_real_group_lds import (
    compute_embedding_cosine_baseline,
    compute_metric_for_dataset,
    concat_group_mask_blocks,
    interaction_residual_explained_variance,
    load_payload,
    pearsonr,
)
from kernel_surrogate.kernel_attribution import rmse, spearmanr
from kernel_surrogate.real_data_cmis import (
    analytical_second_derivative_at_zero,
    build_group_mask_blocks,
    fit_surrogate_method,
    normalize_cmis_scores,
    predict_surrogate_method,
    rank_group_pairs,
    resolve_device,
    run_checkpoint_protocol_a_for_kernel_methods,
    summarize_pair_dominance,
)


def make_args(
    *,
    dataset: str,
    device: str,
    feature_device: str,
    data_keep_rate: float,
    modality_keep_rate: float,
    group_method: str,
    num_groups: int,
    num_subset_samples: int,
    partition_seed: int,
    gradient_ridge_lambda: float,
    gradient_max_iter: int,
    gradient_lr: float,
    run_name: str,
) -> argparse.Namespace:
    args = SimpleNamespace(
        model="cross_attention",
        datasets=[dataset],
        seeds=[0],
        device=device,
        feature_device=feature_device,
        text_backbone="clip_cached",
        vision_backbone="clip_cached",
        backbone_batch_size=32,
        download_datasets=False,
        backbone_cache_dir=None,
        batch_size=4196,
        eval_batch_size=4196,
        epochs=15,
        lr=5e-4,
        dropout=0.1,
        hidden_dim=256,
        proj_dim=256,
        num_heads=4,
        num_layers=2,
        num_groups=num_groups,
        group_method=group_method,
        data_keep_rate=data_keep_rate,
        modality_keep_rate=modality_keep_rate,
        data_missing_rate=None,
        modality_missing_rate=None,
        sampling_backend="gradient_estimation",
        gradient_space="parameter",
        gradient_proj_dim=64,
        gradient_ridge_lambda=gradient_ridge_lambda,
        gradient_max_iter=gradient_max_iter,
        gradient_lr=gradient_lr,
        gradient_batch_size=256,
        gradient_projection_mode="finite_diff",
        gradient_finite_diff_epsilon=1e-3,
        gradient_projection_seed_base=91_000,
        gradient_init_seed_offset=0,
        num_subset_samples=num_subset_samples,
        train_repeat_seeds=[0],
        max_test_loss_samples=1000,
        partition_seed=partition_seed,
        subset_sampling_seed_base=17_000,
        group_cache_dir=None,
        run_name=run_name,
        checkpoint_root=ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train",
        result_root=ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train",
        checkpoint_dir=None,
        result_dir=None,
    )
    lds_real_train._validate_args(args)
    lds_real_train._resolve_output_dirs(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    return args


def evaluate_combo(
    *,
    train_result_dir: Path,
    eval_result_dir: Path,
    full_result_dir: Path,
    pretrained_full_checkpoint_root: Path,
    dataset: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    train_payload = load_payload(train_result_dir, dataset=dataset, seed=seed)
    eval_payload = load_payload(eval_result_dir, dataset=dataset, seed=seed)
    full_payload = load_payload(full_result_dir, dataset=dataset, seed=seed)
    full_loss = float(full_payload["runs"][0]["test_loss"])
    train_blocks, train_targets = build_group_mask_blocks(train_payload)
    eval_blocks, eval_targets = build_group_mask_blocks(eval_payload)
    combined_blocks = concat_group_mask_blocks(train_blocks, eval_blocks)
    train_indices = np.arange(train_targets.shape[0], dtype=np.int64)
    eval_indices = np.arange(train_targets.shape[0], train_targets.shape[0] + eval_targets.shape[0], dtype=np.int64)
    targets = np.concatenate([train_targets, eval_targets], axis=0)

    metric, _ = compute_metric_for_dataset(
        dataset,
        train_payload=train_payload,
        seed=seed,
        device=device,
        feature_device=str(device),
        pretrained_root=pretrained_full_checkpoint_root,
        fallback_dir=ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis",
        gradient_params="last_layer",
        metric_normalization="cosine",
    )
    embedding_baseline = compute_embedding_cosine_baseline(
        dataset,
        train_payload=train_payload,
        seed=seed,
        device=device,
        feature_device=str(device),
        pretrained_root=pretrained_full_checkpoint_root,
        fallback_dir=ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis",
        top_k=20,
    )

    method_rows: list[dict[str, Any]] = []
    for method in ("metric_rbf", "structured_metric_rbf", "structured_rbf", "raw_flat_rbf"):
        fit = fit_surrogate_method(
            str(method),
            combined_blocks,
            targets,
            metric,
            ridge_lambda=1e-5,
            gamma_scale=1e-3,
            structured_shared_weight=1.0,
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
        cmis_scores = normalize_cmis_scores(raw_scores, eps=1e-12)
        cmis_ranking = rank_group_pairs(cmis_scores, top_k=20)
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
            "method": str(method),
            "heldout_lds": spearmanr(eval_predictions, eval_targets),
            "heldout_pearson": pearsonr(eval_predictions, eval_targets),
            "heldout_rmse": rmse(eval_predictions, eval_targets),
            "train_lds": spearmanr(train_predictions, train_targets),
            "train_rmse": rmse(train_predictions, train_targets),
            "irev": interaction_residual_explained_variance(
                eval_blocks,
                eval_targets,
                cmis_scores,
                num_folds=5,
                seed=123,
            ),
            "top_pairs": cmis_ranking["top_pairs"],
            "top_pair_dominance": summarize_pair_dominance(cmis_ranking["top_pairs"]),
        }
        method_rows.append(row)

    protocol_rows = run_checkpoint_protocol_a_for_kernel_methods(
        method_rows + [embedding_baseline],
        payload=eval_payload,
        full_loss=full_loss,
        top_k=20,
        random_repeats=200,
        random_seed=123,
    )
    protocol_by_method = {row["method"]: row for row in protocol_rows}
    return {
        "method_rows": method_rows,
        "protocol_rows": protocol_rows,
        "best_real_lds_method": max(method_rows, key=lambda row: float(row["heldout_lds"])),
        "embedding_protocol": protocol_by_method.get("embedding_cosine"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--group-method", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--data-keep-rate", type=float, default=0.9)
    parser.add_argument("--modality-keep-rate", type=float, default=0.9)
    parser.add_argument("--num-groups", type=int, default=25)
    parser.add_argument("--num-subset-samples", type=int, default=100)
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=ROOT / "src" / "kernel_surrogate" / "results" / "search_gradest_hparams_humor_kmeans_data90.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "src" / "kernel_surrogate" / "results" / "search_gradest_hparams_humor_kmeans_data90.md")
    args = parser.parse_args()

    device = resolve_device(args.device)
    token = f"{int(args.data_keep_rate * 100)}_{int(args.modality_keep_rate * 100)}"
    eval_run_name = f"{args.group_method}_grouped25_subset100_data{int(args.data_keep_rate*100)}_mod{int(args.modality_keep_rate*100)}"
    eval_result_dir = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train" / eval_run_name
    full_result_dir = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train" / (
        f"{args.group_method}_grouped25_full_data_train" if args.group_method == "kmeans" else "grouped25_full_data_train"
    )
    pretrained_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / (
        f"{args.group_method}_grouped25_full_data_train" if args.group_method == "kmeans" else "grouped25_full_data_train"
    )
    if not pretrained_root.exists():
        pretrained_root = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "grouped25_full_data_train"

    configs = []
    for ridge, max_iter, lr in product((1e-4, 1e-2, 1.0), (3, 5, 10), (0.1, 0.3, 1.0)):
        run_name = (
            f"gradest_search_{args.group_method}_g25_d{int(args.data_keep_rate*100)}_m{int(args.modality_keep_rate*100)}_"
            f"r{ridge:g}_i{max_iter}_lr{lr:g}"
        )
        configs.append((ridge, max_iter, lr, run_name))

    records: list[dict[str, Any]] = []
    for ridge, max_iter, lr, run_name in configs:
        lds_args = make_args(
            dataset=args.dataset,
            device=str(device),
            feature_device=args.feature_device,
            data_keep_rate=args.data_keep_rate,
            modality_keep_rate=args.modality_keep_rate,
            group_method=args.group_method,
            num_groups=args.num_groups,
            num_subset_samples=args.num_subset_samples,
            partition_seed=args.partition_seed,
            gradient_ridge_lambda=float(ridge),
            gradient_max_iter=int(max_iter),
            gradient_lr=float(lr),
            run_name=run_name,
        )
        result = lds_real_train.run_single(args.dataset, int(args.seed), lds_args)
        train_result_dir = lds_args.result_dir
        eval_result = evaluate_combo(
            train_result_dir=train_result_dir,
            eval_result_dir=eval_result_dir,
            full_result_dir=full_result_dir,
            pretrained_full_checkpoint_root=pretrained_root,
            dataset=args.dataset,
            seed=int(args.seed),
            device=device,
        )
        best_method = dict(eval_result["best_real_lds_method"])
        record = {
            "run_name": run_name,
            "gradient_ridge_lambda": float(ridge),
            "gradient_max_iter": int(max_iter),
            "gradient_lr": float(lr),
            "num_completed_subsets": int(result["summary"]["num_completed_subsets"]),
            "mean_gradest_test_loss": float(result["summary"]["mean_test_loss"]),
            "mean_gradest_test_acc": float(result["summary"]["mean_test_acc"]),
            "best_real_lds_method": best_method["method"],
            "best_real_lds": float(best_method["heldout_lds"]),
            "best_real_rmse": float(best_method["heldout_rmse"]),
            "best_real_pearson": float(best_method["heldout_pearson"]),
            "all_methods": eval_result["method_rows"],
            "protocol_a": eval_result["protocol_rows"],
            "embedding_protocol": eval_result["embedding_protocol"],
        }
        records.append(record)

    best = max(records, key=lambda row: (row["best_real_lds"], -row["best_real_rmse"]))
    payload = {
        "config": {
            "dataset": args.dataset,
            "group_method": args.group_method,
            "data_keep_rate": args.data_keep_rate,
            "modality_keep_rate": args.modality_keep_rate,
            "seed": int(args.seed),
            "num_subset_samples": int(args.num_subset_samples),
            "search_grid": {
                "gradient_ridge_lambda": [1e-4, 1e-2, 1.0],
                "gradient_max_iter": [3, 5, 10],
                "gradient_lr": [0.1, 0.3, 1.0],
            },
        },
        "best": best,
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
    best = payload["best"]
    lines = [
        "# Gradest Hyperparameter Search",
        "",
        f"- dataset: `{payload['config']['dataset']}`",
        f"- setting: `{payload['config']['group_method']}` / `data{int(payload['config']['data_keep_rate']*100)} mod{int(payload['config']['modality_keep_rate']*100)}`",
        f"- num subsets: `{payload['config']['num_subset_samples']}`",
        "",
        "## Best",
        "",
        f"- ridge: `{best['gradient_ridge_lambda']}`",
        f"- max_iter: `{best['gradient_max_iter']}`",
        f"- lr: `{best['gradient_lr']}`",
        f"- best real LDS method: `{best['best_real_lds_method']}`",
        f"- best real LDS: `{best['best_real_lds']:.6f}`",
        "",
        "## All Configs",
        "",
        "| ridge | max_iter | lr | best method | best real LDS | best real RMSE |",
        "| ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in sorted(payload["records"], key=lambda item: (item["gradient_ridge_lambda"], item["gradient_max_iter"], item["gradient_lr"])):
        lines.append(
            f"| {row['gradient_ridge_lambda']} | {row['gradient_max_iter']} | {row['gradient_lr']} | "
            f"{row['best_real_lds_method']} | {row['best_real_lds']:.6f} | {row['best_real_rmse']:.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
