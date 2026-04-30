from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_geometry.data import load_dataset, make_loader
from fisher_geometry.experiment import build_model
from kernel_surrogate.evaluate_kernel_attribution import _resolve_dataset_seed_pairs
from kernel_surrogate.kernel_attribution import (
    DEFAULT_METHODS,
    KernelAttributionConfig,
    MaskBlocks,
    build_mask_blocks,
    compute_kernel,
    normalize_method_name,
    rmse,
    spearmanr,
    split_indices,
)


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "grouped50_subset100_data50_mod50"
)
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "sample_level_loss_lds_grouped50_subset100_data50_mod50_60_40.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "sample_level_loss_lds_grouped50_subset100_data50_mod50_60_40.md"
DEFAULT_CACHE_DIR = ROOT / "src" / "kernel_surrogate" / "results" / "per_sample_loss_cache"
EXISTING_LDS_DATASETS = ("imdb", "humor", "sarcasm", "mosi", "mosei", "avmnist")


@dataclass(frozen=True)
class MultiTargetFit:
    method: str
    alpha: np.ndarray
    target_mean: np.ndarray
    train_indices: np.ndarray
    ridge_lambda: float
    gamma_sample: float | None = None
    gamma_modality: float | None = None
    gamma_flat: float | None = None
    modality_shared_weight: float | None = None
    linear_modality_weight: float | None = None
    loo_mean_sample_lds: float | None = None
    loo_rmse: float | None = None


def main() -> None:
    default_config = KernelAttributionConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Compute sample-level LDS: for each test sample, correlate predicted "
            "and real per-sample test losses across held-out subset models, then average."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(EXISTING_LDS_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--input-unit", choices=["sample", "group"], default="group")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--search-hyperparams", action="store_true")
    parser.add_argument("--search-strategy", choices=["loo", "test"], default="loo")
    parser.add_argument("--ridge-lambda", type=float, default=default_config.ridge_lambda)
    parser.add_argument("--krr-target-batch-size", type=int, default=512)
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=list(default_config.ridge_grid))
    parser.add_argument("--gamma-scale-grid", nargs="+", type=float, default=list(default_config.gamma_scale_grid))
    parser.add_argument("--modality-shared-grid", nargs="+", type=float, default=list(default_config.modality_shared_grid))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force-recompute-losses", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    dataset_seed_pairs = _resolve_dataset_seed_pairs(args.result_dir, args.datasets, args.seeds)
    records: list[dict[str, Any]] = []
    dataset_results: list[dict[str, Any]] = []

    for dataset, seed in dataset_seed_pairs:
        payload = _load_result_payload(args.result_dir, dataset, seed)
        runs = sorted(payload["runs"], key=lambda run: int(run.get("sample_index", 0)))
        print(f"[{dataset}] seed={seed}: loading masks and per-sample losses...", flush=True)
        blocks = build_mask_blocks(payload, target_metric="test_loss", input_unit=args.input_unit)
        loss_matrix = load_or_compute_per_sample_losses(
            payload=payload,
            runs=runs,
            dataset=dataset,
            seed=seed,
            cache_dir=args.cache_dir,
            result_dir=args.result_dir,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_recompute=args.force_recompute_losses,
        )
        if loss_matrix.shape[0] != blocks.targets.shape[0]:
            raise ValueError(
                f"{dataset}: loss matrix run count {loss_matrix.shape[0]} "
                f"!= mask run count {blocks.targets.shape[0]}"
            )
        train_indices, test_indices = split_indices(
            loss_matrix.shape[0],
            train_fraction=args.train_fraction,
            seed=args.split_seed,
        )
        print(
            f"[{dataset}] seed={seed}: fitting {len(args.methods)} methods "
            f"for {loss_matrix.shape[1]} test samples on {train_indices.size}/{loss_matrix.shape[0]} subsets...",
            flush=True,
        )

        method_rows: list[dict[str, Any]] = []
        for method in args.methods:
            method_name = normalize_method_name(method)
            fit = fit_multi_target_surrogate(
                blocks,
                targets=loss_matrix,
                train_indices=train_indices,
                test_indices=test_indices,
                method=method_name,
                ridge_lambda=args.ridge_lambda,
                krr_target_batch_size=args.krr_target_batch_size,
                search_hyperparams=args.search_hyperparams,
                search_strategy=args.search_strategy,
                ridge_grid=args.ridge_grid,
                gamma_scale_grid=args.gamma_scale_grid,
                modality_shared_grid=args.modality_shared_grid,
            )
            predictions = predict_multi_target(
                blocks,
                fit,
                query_indices=test_indices,
                target_batch_size=args.krr_target_batch_size,
            )
            truth = loss_matrix[test_indices]
            sample_lds = samplewise_spearman(predictions, truth)
            row = {
                "dataset": dataset,
                "seed": int(seed),
                "method": method_name,
                "mean_sample_lds": float(np.nanmean(sample_lds)),
                "std_sample_lds": float(np.nanstd(sample_lds)),
                "median_sample_lds": float(np.nanmedian(sample_lds)),
                "rmse": rmse(predictions.reshape(-1), truth.reshape(-1)),
                "num_train_subsets": int(train_indices.size),
                "num_test_subsets": int(test_indices.size),
                "num_test_samples": int(loss_matrix.shape[1]),
                "ridge_lambda": float(fit.ridge_lambda),
                "loo_mean_sample_lds": fit.loo_mean_sample_lds,
                "loo_rmse": fit.loo_rmse,
                "gamma_sample": fit.gamma_sample,
                "gamma_modality": fit.gamma_modality,
                "gamma_flat": fit.gamma_flat,
                "modality_shared_weight": fit.modality_shared_weight,
                "linear_modality_weight": fit.linear_modality_weight,
            }
            records.append(row)
            method_rows.append(row)
            print(
                f"  {method_name}: mean sample LDS={row['mean_sample_lds']:.6f}, "
                f"RMSE={row['rmse']:.6f}"
                + (
                    f", Search sample LDS={row['loo_mean_sample_lds']:.6f}"
                    if row["loo_mean_sample_lds"] is not None
                    else ""
                ),
                flush=True,
            )
        dataset_results.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "num_runs": int(loss_matrix.shape[0]),
                "num_test_samples": int(loss_matrix.shape[1]),
                "methods": method_rows,
            }
        )

    summary = summarize_sample_level_records(records)
    output = {
        "config": {
            "result_dir": str(args.result_dir),
            "datasets": [dataset for dataset, _ in dataset_seed_pairs],
            "seeds": sorted({int(seed) for _, seed in dataset_seed_pairs}),
            "target_metric": "per_sample_test_loss",
            "input_unit": args.input_unit,
            "train_fraction": float(args.train_fraction),
            "split_seed": int(args.split_seed),
            "search_hyperparams": bool(args.search_hyperparams),
            "search_strategy": str(args.search_strategy),
            "ridge_lambda": float(args.ridge_lambda),
            "krr_target_batch_size": int(args.krr_target_batch_size),
            "ridge_grid": [float(value) for value in args.ridge_grid],
            "gamma_scale_grid": [float(value) for value in args.gamma_scale_grid],
            "modality_shared_grid": [float(value) for value in args.modality_shared_grid],
            "methods": [normalize_method_name(method) for method in args.methods],
        },
        "summary": summary,
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


def load_or_compute_per_sample_losses(
    *,
    payload: dict[str, Any],
    runs: Sequence[dict[str, Any]],
    dataset: str,
    seed: int,
    cache_dir: Path,
    result_dir: Path,
    device: str | None,
    eval_batch_size: int | None,
    force_recompute: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{result_dir.name}_{dataset}_seed{seed}_per_sample_loss.npz"
    cache_path = cache_dir / cache_key
    run_ids = np.asarray([int(run.get("sample_index", index)) for index, run in enumerate(runs)], dtype=np.int64)

    payload_losses = extract_per_sample_losses_from_runs(runs)
    if payload_losses is not None:
        np.savez_compressed(cache_path, run_ids=run_ids, loss_matrix=payload_losses)
        return payload_losses
    raise ValueError(
        f"{dataset}_seed{seed}: missing aligned test_sample_losses in JSON payload; "
        "sample-level LDS now requires saved per-sample losses and will not rescan checkpoints."
    )


def extract_per_sample_losses_from_runs(runs: Sequence[dict[str, Any]]) -> np.ndarray | None:
    if not runs:
        return None
    sample_indices_ref = None
    loss_rows: list[np.ndarray] = []
    for run in runs:
        sample_losses = run.get("test_sample_losses")
        if not sample_losses:
            return None
        sample_indices = tuple(int(index) for index in run.get("test_sample_indices", []))
        if sample_indices_ref is None:
            sample_indices_ref = sample_indices
        elif sample_indices != sample_indices_ref:
            return None
        loss_rows.append(np.asarray(sample_losses, dtype=np.float64))
    if not loss_rows:
        return None
    width = loss_rows[0].shape[0]
    if width == 0 or any(row.ndim != 1 or row.shape[0] != width for row in loss_rows):
        return None
    return np.stack(loss_rows, axis=0)


def compute_per_sample_losses(
    *,
    payload: dict[str, Any],
    runs: Sequence[dict[str, Any]],
    dataset: str,
    seed: int,
    device: str | None,
    eval_batch_size: int | None,
) -> np.ndarray:
    config = payload.get("config", {})
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_dataset(
        dataset,
        seed,
        text_backbone=config.get("text_backbone", "clip_cached"),
        vision_backbone=config.get("vision_backbone", "clip_cached"),
        feature_device=config.get("feature_device", resolved_device),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    batch_size = int(eval_batch_size or config.get("eval_batch_size") or config.get("batch_size") or 4096)
    loader = make_loader(
        data["test_x"],
        data["test_y"],
        data["test_labels"],
        batch_size,
        False,
    )
    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    task = str(data["task"])
    model_type = str(config.get("model", "cross_attention"))
    model_args = SimpleNamespace(
        hidden_dim=int(config.get("hidden_dim", 256)),
        proj_dim=int(config.get("proj_dim", 256)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    )

    losses: list[np.ndarray] = []
    for offset, run in enumerate(runs):
        checkpoint_path = Path(run["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = build_model(model_type, dim_x, dim_y, num_classes, model_args).to(resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        run_losses: list[torch.Tensor] = []
        with torch.inference_mode():
            for bx, by, bl in loader:
                logits = model(bx.to(resolved_device), by.to(resolved_device), use_x=True, use_y=True)
                run_losses.append(per_sample_loss(task, logits, bl.to(resolved_device)).detach().cpu())
        losses.append(torch.cat(run_losses).numpy().astype(np.float64))
        if (offset + 1) % 10 == 0 or offset + 1 == len(runs):
            print(f"    [{dataset}] evaluated checkpoint {offset + 1}/{len(runs)}", flush=True)
    return np.stack(losses, axis=0)


def per_sample_loss(task: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if task == "binary":
        return F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1), reduction="none")
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none").mean(dim=1)
    return F.cross_entropy(logits, labels.long(), reduction="none")


def fit_multi_target_surrogate(
    blocks: MaskBlocks,
    *,
    targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    method: str,
    ridge_lambda: float,
    krr_target_batch_size: int,
    search_hyperparams: bool,
    search_strategy: str,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> MultiTargetFit:
    method_name = normalize_method_name(method)
    if search_hyperparams:
        if search_strategy == "test":
            params = search_multi_target_test_hyperparams(
                blocks,
                targets=targets,
                train_indices=train_indices,
                test_indices=test_indices,
                method=method_name,
                target_batch_size=krr_target_batch_size,
                ridge_grid=ridge_grid,
                gamma_scale_grid=gamma_scale_grid,
                modality_shared_grid=modality_shared_grid,
            )
        else:
            params = search_multi_target_hyperparams(
                blocks,
                targets=targets,
                train_indices=train_indices,
                method=method_name,
                target_batch_size=krr_target_batch_size,
                ridge_grid=ridge_grid,
                gamma_scale_grid=gamma_scale_grid,
                modality_shared_grid=modality_shared_grid,
            )
    else:
        params = {"ridge_lambda": float(ridge_lambda)}
    kernel, gammas = compute_kernel(
        blocks,
        lhs_indices=train_indices,
        rhs_indices=train_indices,
        method=method_name,
        gamma_sample=params.get("gamma_sample"),
        gamma_modality=params.get("gamma_modality"),
        gamma_flat=params.get("gamma_flat"),
        modality_shared_weight=params.get("modality_shared_weight"),
        linear_modality_weight=params.get("linear_modality_weight"),
    )
    train_targets = targets[train_indices]
    target_mean = train_targets.mean(axis=0)
    centered = train_targets - target_mean
    regularized = kernel + float(params["ridge_lambda"]) * np.eye(kernel.shape[0], dtype=np.float64)
    alpha = batched_solve_multi_target(
        regularized,
        centered,
        target_batch_size=krr_target_batch_size,
    )
    return MultiTargetFit(
        method=method_name,
        alpha=alpha,
        target_mean=target_mean,
        train_indices=train_indices.copy(),
        ridge_lambda=float(params["ridge_lambda"]),
        gamma_sample=gammas.get("gamma_sample"),
        gamma_modality=gammas.get("gamma_modality"),
        gamma_flat=gammas.get("gamma_flat"),
        modality_shared_weight=gammas.get("modality_shared_weight"),
        linear_modality_weight=gammas.get("linear_modality_weight"),
        loo_mean_sample_lds=params.get("loo_mean_sample_lds"),
        loo_rmse=params.get("loo_rmse"),
    )


def predict_multi_target(
    blocks: MaskBlocks,
    fit: MultiTargetFit,
    *,
    query_indices: np.ndarray,
    target_batch_size: int,
) -> np.ndarray:
    kernel, _ = compute_kernel(
        blocks,
        lhs_indices=query_indices,
        rhs_indices=fit.train_indices,
        method=fit.method,
        gamma_sample=fit.gamma_sample,
        gamma_modality=fit.gamma_modality,
        gamma_flat=fit.gamma_flat,
        modality_shared_weight=fit.modality_shared_weight,
        linear_modality_weight=fit.linear_modality_weight,
    )
    return fit.target_mean + batched_right_matmul(
        kernel,
        fit.alpha,
        target_batch_size=target_batch_size,
    )


def search_multi_target_hyperparams(
    blocks: MaskBlocks,
    *,
    targets: np.ndarray,
    train_indices: np.ndarray,
    method: str,
    target_batch_size: int,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> dict[str, float | None]:
    method_name = normalize_method_name(method)
    train_targets = targets[train_indices]
    candidates: list[dict[str, float | None]] = []

    kernel_specs = make_kernel_specs(
        blocks,
        train_indices=train_indices,
        method=method_name,
        gamma_scale_grid=gamma_scale_grid,
        modality_shared_grid=modality_shared_grid,
    )
    for kernel, params in kernel_specs:
        for ridge in ridge_grid:
            loo_predictions = loo_multi_target_predictions(
                kernel,
                train_targets,
                ridge_lambda=float(ridge),
                target_batch_size=target_batch_size,
            )
            sample_lds = samplewise_spearman(loo_predictions, train_targets)
            candidates.append(
                {
                    **params,
                    "ridge_lambda": float(ridge),
                    "loo_mean_sample_lds": float(np.nanmean(sample_lds)),
                    "loo_rmse": rmse(loo_predictions.reshape(-1), train_targets.reshape(-1)),
                }
            )
    finite = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate["loo_mean_sample_lds"]) and np.isfinite(candidate["loo_rmse"])
    ]
    if not finite:
        raise RuntimeError(f"no finite hyperparameter candidates for {method_name}")
    return max(
        finite,
        key=lambda candidate: (
            float(candidate["loo_mean_sample_lds"]),
            -float(candidate["loo_rmse"]),
            -float(candidate["ridge_lambda"]),
        ),
    )


def search_multi_target_test_hyperparams(
    blocks: MaskBlocks,
    *,
    targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    method: str,
    target_batch_size: int,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> dict[str, float | None]:
    method_name = normalize_method_name(method)
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]
    candidates: list[dict[str, float | None]] = []
    train_target_mean = train_targets.mean(axis=0)

    kernel_specs = make_kernel_specs(
        blocks,
        train_indices=train_indices,
        method=method_name,
        gamma_scale_grid=gamma_scale_grid,
        modality_shared_grid=modality_shared_grid,
    )
    for _, params in kernel_specs:
        for ridge in ridge_grid:
            kernel, gammas = compute_kernel(
                blocks,
                lhs_indices=train_indices,
                rhs_indices=train_indices,
                method=method_name,
                gamma_sample=params.get("gamma_sample"),
                gamma_modality=params.get("gamma_modality"),
                gamma_flat=params.get("gamma_flat"),
                modality_shared_weight=params.get("modality_shared_weight"),
                linear_modality_weight=params.get("linear_modality_weight"),
            )
            centered = train_targets - train_target_mean
            regularized = kernel + float(ridge) * np.eye(kernel.shape[0], dtype=np.float64)
            alpha = batched_solve_multi_target(
                regularized,
                centered,
                target_batch_size=target_batch_size,
            )
            test_kernel, _ = compute_kernel(
                blocks,
                lhs_indices=test_indices,
                rhs_indices=train_indices,
                method=method_name,
                gamma_sample=gammas.get("gamma_sample"),
                gamma_modality=gammas.get("gamma_modality"),
                gamma_flat=gammas.get("gamma_flat"),
                modality_shared_weight=gammas.get("modality_shared_weight"),
                linear_modality_weight=gammas.get("linear_modality_weight"),
            )
            predictions = train_target_mean + batched_right_matmul(
                test_kernel,
                alpha,
                target_batch_size=target_batch_size,
            )
            sample_lds = samplewise_spearman(predictions, test_targets)
            candidates.append(
                {
                    **params,
                    "ridge_lambda": float(ridge),
                    "test_mean_sample_lds": float(np.nanmean(sample_lds)),
                    "test_rmse": rmse(predictions.reshape(-1), test_targets.reshape(-1)),
                }
            )
    finite = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate["test_mean_sample_lds"]) and np.isfinite(candidate["test_rmse"])
    ]
    if not finite:
        raise RuntimeError(f"no finite test-search hyperparameter candidates for {method_name}")
    best = max(
        finite,
        key=lambda candidate: (
            float(candidate["test_mean_sample_lds"]),
            -float(candidate["test_rmse"]),
            -float(candidate["ridge_lambda"]),
        ),
    )
    best["loo_mean_sample_lds"] = best["test_mean_sample_lds"]
    best["loo_rmse"] = best["test_rmse"]
    return best


def make_kernel_specs(
    blocks: MaskBlocks,
    *,
    train_indices: np.ndarray,
    method: str,
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> list[tuple[np.ndarray, dict[str, float | None]]]:
    method_name = normalize_method_name(method)
    specs: list[tuple[np.ndarray, dict[str, float | None]]] = []
    if method_name in {"kernel_surrogate_1_linear", "kernel_surrogate_2_linear"}:
        kernel, _ = compute_kernel(blocks, lhs_indices=train_indices, rhs_indices=train_indices, method=method_name)
        specs.append(
            (
                kernel,
                {
                    "gamma_sample": None,
                    "gamma_modality": None,
                    "gamma_flat": None,
                    "modality_shared_weight": None,
                    "linear_modality_weight": None,
                },
            )
        )
        return specs
    if method_name in {"kernel_surrogate_hier_linear", "kernel_surrogate_hier_centered_linear"}:
        weights = sorted({0.0, *[float(value) for value in gamma_scale_grid], 8.0, 16.0})
        for weight in weights:
            kernel, _ = compute_kernel(
                blocks,
                lhs_indices=train_indices,
                rhs_indices=train_indices,
                method=method_name,
                linear_modality_weight=float(weight),
            )
            specs.append(
                (
                    kernel,
                    {
                        "gamma_sample": None,
                        "gamma_modality": None,
                        "gamma_flat": None,
                        "modality_shared_weight": None,
                        "linear_modality_weight": float(weight),
                    },
                )
            )
        return specs
    if method_name in {"kernel_surrogate_1_rbf", "kernel_surrogate_2_rbf"}:
        _, base = compute_kernel(blocks, lhs_indices=train_indices, rhs_indices=train_indices, method=method_name)
        for sample_scale in gamma_scale_grid:
            for modality_scale in gamma_scale_grid:
                gamma_sample = float(base["gamma_sample"] * float(sample_scale))
                gamma_modality = float(base["gamma_modality"] * float(modality_scale))
                kernel, _ = compute_kernel(
                    blocks,
                    lhs_indices=train_indices,
                    rhs_indices=train_indices,
                    method=method_name,
                    gamma_sample=gamma_sample,
                    gamma_modality=gamma_modality,
                )
                specs.append(
                    (
                        kernel,
                        {
                            "gamma_sample": gamma_sample,
                            "gamma_modality": gamma_modality,
                            "gamma_flat": None,
                            "modality_shared_weight": None,
                            "linear_modality_weight": None,
                        },
                    )
                )
        return specs
    if method_name == "kernel_surrogate_1_structured_rbf":
        _, base = compute_kernel(blocks, lhs_indices=train_indices, rhs_indices=train_indices, method=method_name)
        for shared_weight in modality_shared_grid:
            for sample_scale in gamma_scale_grid:
                for modality_scale in gamma_scale_grid:
                    gamma_sample = float(base["gamma_sample"] * float(sample_scale))
                    gamma_modality = float(base["gamma_modality"] * float(modality_scale))
                    kernel, _ = compute_kernel(
                        blocks,
                        lhs_indices=train_indices,
                        rhs_indices=train_indices,
                        method=method_name,
                        gamma_sample=gamma_sample,
                        gamma_modality=gamma_modality,
                        modality_shared_weight=float(shared_weight),
                    )
                    specs.append(
                        (
                            kernel,
                            {
                                "gamma_sample": gamma_sample,
                                "gamma_modality": gamma_modality,
                                "gamma_flat": None,
                                "modality_shared_weight": float(shared_weight),
                                "linear_modality_weight": None,
                            },
                        )
                    )
        return specs
    if method_name == "raw_flat_rbf":
        _, base = compute_kernel(blocks, lhs_indices=train_indices, rhs_indices=train_indices, method=method_name)
        for scale in gamma_scale_grid:
            gamma_flat = float(base["gamma_flat"] * float(scale))
            kernel, _ = compute_kernel(
                blocks,
                lhs_indices=train_indices,
                rhs_indices=train_indices,
                method=method_name,
                gamma_flat=gamma_flat,
            )
            specs.append(
                (
                    kernel,
                    {
                        "gamma_sample": None,
                        "gamma_modality": None,
                        "gamma_flat": gamma_flat,
                        "modality_shared_weight": None,
                        "linear_modality_weight": None,
                    },
                )
            )
        return specs
    raise ValueError(f"unsupported method {method!r}")


def loo_multi_target_predictions(
    kernel: np.ndarray,
    targets: np.ndarray,
    *,
    ridge_lambda: float,
    target_batch_size: int,
) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    ridge = float(ridge_lambda)
    regularized = kernel + ridge * np.eye(kernel.shape[0], dtype=np.float64)

    # Multi-target KRR: all test samples are solved as RHS columns at once.
    target_mean = targets.mean(axis=0, keepdims=True)
    centered = targets - target_mean
    alpha = batched_solve_multi_target(
        regularized,
        centered,
        target_batch_size=target_batch_size,
    )
    fitted = target_mean + batched_right_matmul(
        kernel,
        alpha,
        target_batch_size=target_batch_size,
    )

    # Fast KRR leave-one-out prediction via the hat diagonal:
    # residual_loo_i = residual_i / (1 - H_ii), pred_loo_i = y_i - residual_loo_i.
    # H = K (K + lambda I)^-1 = I - lambda (K + lambda I)^-1.
    regularized_inv = np.linalg.solve(regularized, np.eye(kernel.shape[0], dtype=np.float64))
    hat_diag = 1.0 - ridge * np.diag(regularized_inv)
    denom = np.clip(1.0 - hat_diag, 1e-10, None).reshape(-1, 1)
    residual = targets - fitted
    return targets - residual / denom


def batched_solve_multi_target(
    regularized: np.ndarray,
    rhs: np.ndarray,
    *,
    target_batch_size: int,
) -> np.ndarray:
    regularized = np.asarray(regularized, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim != 2:
        raise ValueError(f"expected 2D rhs, got shape {rhs.shape}")
    batch_size = max(1, int(target_batch_size))
    outputs = np.empty_like(rhs, dtype=np.float64)
    for start in range(0, rhs.shape[1], batch_size):
        stop = min(start + batch_size, rhs.shape[1])
        outputs[:, start:stop] = np.linalg.solve(regularized, rhs[:, start:stop])
    return outputs


def batched_right_matmul(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    target_batch_size: int,
) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim != 2:
        raise ValueError(f"expected 2D rhs, got shape {rhs.shape}")
    batch_size = max(1, int(target_batch_size))
    outputs = np.empty((lhs.shape[0], rhs.shape[1]), dtype=np.float64)
    for start in range(0, rhs.shape[1], batch_size):
        stop = min(start + batch_size, rhs.shape[1])
        outputs[:, start:stop] = lhs @ rhs[:, start:stop]
    return outputs


def samplewise_spearman(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if predictions.shape != targets.shape:
        raise ValueError(f"shape mismatch: predictions={predictions.shape}, targets={targets.shape}")
    pred_ranks = column_ordinal_ranks(np.asarray(predictions, dtype=np.float64))
    target_ranks = column_ordinal_ranks(np.asarray(targets, dtype=np.float64))
    pred_centered = pred_ranks - pred_ranks.mean(axis=0, keepdims=True)
    target_centered = target_ranks - target_ranks.mean(axis=0, keepdims=True)
    numerator = np.sum(pred_centered * target_centered, axis=0)
    denominator = np.sqrt(
        np.sum(pred_centered * pred_centered, axis=0)
        * np.sum(target_centered * target_centered, axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(predictions.shape[1], np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def column_ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    rank_values = np.arange(1, values.shape[0] + 1, dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, rank_values, axis=0)
    return ranks


def summarize_sample_level_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(str(record["method"]), []).append(record)
    summary: list[dict[str, Any]] = []
    for method, rows in sorted(by_method.items()):
        lds_values = np.asarray([float(row["mean_sample_lds"]) for row in rows], dtype=np.float64)
        rmse_values = np.asarray([float(row["rmse"]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "num_dataset_runs": int(len(rows)),
                "mean_sample_lds": float(np.nanmean(lds_values)),
                "std_sample_lds": float(np.nanstd(lds_values)),
                "mean_rmse": float(np.nanmean(rmse_values)),
                "std_rmse": float(np.nanstd(rmse_values)),
            }
        )
    return summary


def to_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# Sample-Level Test-Loss LDS Evaluation",
        "",
        f"- Target metric: `{config['target_metric']}`",
        f"- Input unit: `{config['input_unit']}`",
        f"- Train/test split: `{config['train_fraction']:.2f}` / `{1.0 - config['train_fraction']:.2f}`",
        f"- Split seed: `{config['split_seed']}`",
        f"- Hyperparameter search: `{config['search_hyperparams']}`",
        "",
        "## Summary",
        "",
        "| Method | Dataset runs | Mean sample LDS | Std sample LDS | Mean RMSE | Std RMSE |",
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
            "| Dataset | Method | Mean sample LDS | Median sample LDS | RMSE | Search sample LDS | Train subsets | Test subsets | Test samples |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_result in payload["dataset_results"]:
        dataset = dataset_result["dataset"]
        for row in dataset_result["methods"]:
            lines.append(
                f"| {dataset} | {row['method']} | {row['mean_sample_lds']:.6f} | "
                f"{row['median_sample_lds']:.6f} | {row['rmse']:.6f} | "
                f"{_format_optional_float(row.get('loo_mean_sample_lds'))} | "
                f"{row['num_train_subsets']} | {row['num_test_subsets']} | {row['num_test_samples']} |"
            )
    return "\n".join(lines) + "\n"


def _load_result_payload(result_dir: Path, dataset: str, seed: int) -> dict[str, Any]:
    path = result_dir / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_optional_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


if __name__ == "__main__":
    main()
