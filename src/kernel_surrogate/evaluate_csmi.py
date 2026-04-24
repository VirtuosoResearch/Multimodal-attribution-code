from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fisher_geometry.data import load_dataset, make_loader
from src.fisher_geometry.experiment import build_model
from src.kernel_surrogate.kernel_attribution import (
    DEFAULT_METHODS,
    KernelAttributionConfig,
    MaskBlocks,
    compute_kernel,
    evaluate_methods,
    fit_best_kernel_surrogate,
    normalize_method_name,
    rankdata,
    rmse,
    spearmanr,
)
from src.kernel_surrogate.lds_real_train import _per_sample_loss, _train_model

try:
    from src.pid_estimator.subset_loader import build_grouped_subset_loader
except ModuleNotFoundError:
    from pid_estimator.subset_loader import build_grouped_subset_loader


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "cross_attention_group_unit_data_keep25_mod_keep90_g50"
)
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "csmi_exact_validation.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "csmi_exact_validation.md"
DEFAULT_METHODS_CSMI = (
    "kernel_surrogate_1_rbf",
    "kernel_surrogate_1_structured_rbf",
    "kernel_surrogate_hier_linear",
    "raw_flat_rbf",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute target-modality usefulness F_{i,v} from saved LDS subset checkpoints, "
            "fit CSMI surrogates, and validate surrogate CSMI against exact "
            "leave-one-source-modality retraining."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset", default="sarcasm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS_CSMI))
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--search-strategy", choices=["loo", "validation", "test"], default="validation")
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--gamma-scale-grid", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument("--modality-shared-grid", nargs="+", type=float, default=[0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--target-selection", choices=["first", "random", "variance"], default="variance")
    parser.add_argument("--num-target-samples", type=int, default=8)
    parser.add_argument("--target-sample-seed", type=int, default=0)
    parser.add_argument("--target-indices", nargs="+", type=int, default=None)
    parser.add_argument("--num-source-groups", type=int, default=4)
    parser.add_argument("--source-group-seed", type=int, default=0)
    parser.add_argument("--source-groups", nargs="+", type=int, default=None)
    parser.add_argument("--exact-train-seed-base", type=int, default=910_000)
    parser.add_argument("--exact-epochs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--progress-log", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    payload = _load_result_payload(args.result_dir, args.dataset, args.seed)
    config = dict(payload.get("config", {}))
    device = str(args.device or config.get("device", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable; falling back from {device} to cpu.", flush=True)
        device = "cpu"
    eval_batch_size = int(config.get("eval_batch_size") or config.get("batch_size") or 1024)
    batch_size = int(config.get("batch_size") or 1024)

    progress_log = args.progress_log
    _progress(progress_log, f"[{args.dataset}] seed={args.seed}: loading data...")
    data = load_dataset(
        args.dataset,
        args.seed,
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=str("cpu" if device == "cpu" else config.get("feature_device", device)),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    runs = sorted(payload["runs"], key=lambda run: int(run.get("sample_index", 0)))
    if not runs:
        raise ValueError(f"no runs found in {args.result_dir / f'{args.dataset}_seed{args.seed}.json'}")
    num_modalities = int(payload["group_partition"]["num_modalities"])
    if num_modalities != 2:
        raise ValueError(f"current CSMI evaluator expects exactly 2 modalities, got {num_modalities}")

    all_test_indices = np.arange(int(data["test_labels"].shape[0]), dtype=np.int64)
    full_subset_f_all: np.ndarray | None = None
    if args.target_indices:
        target_indices = np.asarray(sorted({int(index) for index in args.target_indices}), dtype=np.int64)
    elif args.target_selection == "variance":
        print(
            f"[{args.dataset}] seed={args.seed}: scoring all {all_test_indices.size} test samples "
            "to select high-variance target samples...",
            flush=True,
        )
        _progress(
            progress_log,
            f"[{args.dataset}] seed={args.seed}: scoring all {all_test_indices.size} test samples for target selection...",
        )
        full_subset_f_all = evaluate_checkpoint_runs(
            runs=runs,
            data=data,
            target_indices=all_test_indices,
            task=str(payload["task"]),
            device=device,
            eval_batch_size=eval_batch_size,
            progress_log=progress_log,
        )
        target_indices = select_target_indices_by_variance(
            full_subset_f_all,
            num_target_samples=args.num_target_samples,
        )
    elif args.target_selection == "random":
        rng = np.random.default_rng(args.target_sample_seed)
        count = min(int(args.num_target_samples), all_test_indices.size)
        target_indices = np.sort(rng.choice(all_test_indices, size=count, replace=False)).astype(np.int64, copy=False)
    else:
        target_indices = all_test_indices[: min(int(args.num_target_samples), all_test_indices.size)].astype(np.int64, copy=False)
    if target_indices.size == 0:
        raise ValueError("no target samples selected")
    if target_indices.min() < 0 or target_indices.max() >= all_test_indices.size:
        raise IndexError("target_indices must be valid test-sample indices")

    _progress(
        progress_log,
        f"[{args.dataset}] seed={args.seed}: selected target test indices {target_indices.tolist()}",
    )
    if full_subset_f_all is None:
        full_subset_f = evaluate_checkpoint_runs(
            runs=runs,
            data=data,
            target_indices=target_indices,
            task=str(payload["task"]),
            device=device,
            eval_batch_size=eval_batch_size,
            progress_log=progress_log,
        )
    else:
        full_subset_f = full_subset_f_all[:, target_indices, :]

    blocks = build_mask_blocks_from_payload(payload)
    pair_blocks = build_pair_blocks(blocks, full_subset_f)
    experiment1 = evaluate_target_modality_surrogates(
        pair_blocks,
        methods=args.methods,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        search_strategy=args.search_strategy,
        ridge_grid=args.ridge_grid,
        gamma_scale_grid=args.gamma_scale_grid,
        modality_shared_grid=args.modality_shared_grid,
    )

    if args.source_groups is not None:
        source_groups = np.asarray(sorted({int(group_id) for group_id in args.source_groups}), dtype=np.int64)
    else:
        rng = np.random.default_rng(args.source_group_seed)
        num_groups = int(payload["group_partition"]["num_groups"])
        count = min(int(args.num_source_groups), num_groups)
        source_groups = np.sort(rng.choice(np.arange(num_groups, dtype=np.int64), size=count, replace=False)).astype(np.int64, copy=False)
    if source_groups.size == 0:
        raise ValueError("no source groups selected")
    _progress(
        progress_log,
        f"[{args.dataset}] seed={args.seed}: exact-validation source groups {source_groups.tolist()}",
    )

    exact_result = run_exact_csmi_validation(
        payload=payload,
        data=data,
        blocks=blocks,
        pair_blocks=pair_blocks,
        methods=args.methods,
        source_groups=source_groups,
        target_indices=target_indices,
        task=str(payload["task"]),
        device=device,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        exact_train_seed_base=int(args.exact_train_seed_base),
        ridge_grid=args.ridge_grid,
        gamma_scale_grid=args.gamma_scale_grid,
        modality_shared_grid=args.modality_shared_grid,
        top_k=int(args.top_k),
        progress_log=progress_log,
        exact_epochs=args.exact_epochs,
    )

    output = {
        "config": {
            "result_dir": str(args.result_dir),
            "dataset": str(args.dataset),
            "seed": int(args.seed),
            "device": device,
            "methods": [normalize_method_name(method) for method in args.methods],
            "train_fraction": float(args.train_fraction),
            "validation_fraction": float(args.validation_fraction),
            "split_seed": int(args.split_seed),
            "search_strategy": str(args.search_strategy),
            "ridge_grid": [float(value) for value in args.ridge_grid],
            "gamma_scale_grid": [float(value) for value in args.gamma_scale_grid],
            "modality_shared_grid": [float(value) for value in args.modality_shared_grid],
            "target_selection": str(args.target_selection),
            "target_indices": [int(index) for index in target_indices.tolist()],
            "source_groups": [int(index) for index in source_groups.tolist()],
            "top_k": int(args.top_k),
            "exact_train_seed_base": int(args.exact_train_seed_base),
            "exact_epochs": None if args.exact_epochs is None else int(args.exact_epochs),
        },
        "subset_result_summary": payload.get("summary", {}),
        "experiment_1_surrogate_validation": experiment1,
        "experiment_2_exact_csmi_validation": exact_result,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"\nSaved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)
    _progress(progress_log, f"[{args.dataset}] seed={args.seed}: completed")


def _load_result_payload(result_dir: Path, dataset: str, seed: int) -> dict[str, Any]:
    path = result_dir / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing result file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "runs" not in payload or "group_partition" not in payload:
        raise ValueError(f"invalid result payload: {path}")
    return payload


def build_mask_blocks_from_payload(payload: dict[str, Any]) -> MaskBlocks:
    runs = list(payload["runs"])
    group_partition = payload["group_partition"]
    group_indices = group_partition["group_indices"]
    num_groups = int(group_partition["num_groups"])
    num_modalities = int(group_partition["num_modalities"])
    sample_deleted = np.ones((len(runs), num_groups), dtype=np.float32)
    modality_deleted = np.zeros((len(runs), num_groups * num_modalities), dtype=np.float32)
    flat_deleted = np.ones((len(runs), num_groups * num_modalities), dtype=np.float32)
    targets = np.zeros(len(runs), dtype=np.float64)
    run_ids: list[int] = []
    index_to_group: dict[int, int] = {}
    for group_id, indices in enumerate(group_indices):
        for index in indices:
            index_to_group[int(index)] = int(group_id)

    for row, run in enumerate(runs):
        run_ids.append(int(run.get("sample_index", row)))
        group_ids = np.asarray(run.get("group_ids", []), dtype=np.int64)
        if group_ids.size:
            sample_deleted[row, group_ids] = 0.0

        data_indices = np.asarray(run.get("data_indices", []), dtype=np.int64)
        keep_mask = np.asarray(run.get("sample_modality_mask", []), dtype=bool)
        if keep_mask.shape != (data_indices.size, num_modalities):
            raise ValueError(
                "sample_modality_mask shape mismatch: "
                f"expected {(data_indices.size, num_modalities)}, got {tuple(keep_mask.shape)}"
            )
        if data_indices.size == 0:
            continue
        deletion_mask = np.logical_not(keep_mask)
        local_groups = np.asarray([index_to_group[int(index)] for index in data_indices.tolist()], dtype=np.int64)
        for group_id in group_ids.tolist():
            group_mask = local_groups == int(group_id)
            if not np.any(group_mask):
                continue
            deletion_rate = deletion_mask[group_mask].mean(axis=0).astype(np.float32)
            start = int(group_id) * num_modalities
            end = start + num_modalities
            modality_deleted[row, start:end] = deletion_rate
            flat_deleted[row, start:end] = deletion_rate
    return MaskBlocks(
        sample_deleted=sample_deleted,
        modality_deleted=modality_deleted,
        flat_deleted=flat_deleted,
        targets=targets,
        run_ids=tuple(run_ids),
        dataset_size=num_groups,
        num_modalities=num_modalities,
    )


def load_model_from_checkpoint(checkpoint_path: str | Path, *, device: str) -> torch.nn.Module:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    build_args = SimpleNamespace(
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        proj_dim=int(model_config.get("proj_dim", 256)),
        num_heads=int(model_config.get("num_heads", 4)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    model = build_model(
        str(model_config["model"]),
        int(model_config["dim_x"]),
        int(model_config["dim_y"]),
        int(model_config["num_classes"]),
        build_args,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_target_modality_usefulness(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    device: str,
    eval_batch_size: int,
) -> np.ndarray:
    if x.shape[0] != y.shape[0] or x.shape[0] != labels.shape[0]:
        raise ValueError("x, y, labels must agree in length")
    model.eval()
    full_loss_chunks: list[np.ndarray] = []
    masked_loss_chunks = [[], []]
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), int(eval_batch_size)):
            end = min(start + int(eval_batch_size), int(x.shape[0]))
            bx = x[start:end].to(device)
            by = y[start:end].to(device)
            bl = labels[start:end].to(device)
            full_logits = model(bx, by, use_x=True, use_y=True)
            full_loss = _per_sample_loss(task, full_logits, bl).detach().cpu().numpy()
            full_loss_chunks.append(full_loss)
            x_masked_logits = model(bx, by, use_x=False, use_y=True)
            y_masked_logits = model(bx, by, use_x=True, use_y=False)
            masked_loss_chunks[0].append(_per_sample_loss(task, x_masked_logits, bl).detach().cpu().numpy())
            masked_loss_chunks[1].append(_per_sample_loss(task, y_masked_logits, bl).detach().cpu().numpy())
    full_loss = np.concatenate(full_loss_chunks, axis=0)
    masked_x_loss = np.concatenate(masked_loss_chunks[0], axis=0)
    masked_y_loss = np.concatenate(masked_loss_chunks[1], axis=0)
    usefulness = np.stack([masked_x_loss - full_loss, masked_y_loss - full_loss], axis=1)
    return usefulness.astype(np.float64, copy=False)


def evaluate_checkpoint_runs(
    *,
    runs: Sequence[dict[str, Any]],
    data: dict[str, Any],
    target_indices: np.ndarray,
    task: str,
    device: str,
    eval_batch_size: int,
    progress_log: Path | None = None,
) -> np.ndarray:
    target_x = data["test_x"][target_indices]
    target_y = data["test_y"][target_indices]
    target_labels = data["test_labels"][target_indices]
    values: list[np.ndarray] = []
    for row, run in enumerate(runs):
        checkpoint_path = str(run.get("checkpoint_path") or "")
        if not checkpoint_path:
            checkpoint_paths = run.get("checkpoint_paths", [])
            if not checkpoint_paths:
                raise ValueError(f"run {row} is missing checkpoint_path/checkpoint_paths")
            checkpoint_path = str(checkpoint_paths[0])
        _progress(
            progress_log,
            f"  evaluating checkpoint run {row + 1}/{len(runs)}: {Path(checkpoint_path).name}",
        )
        model = load_model_from_checkpoint(checkpoint_path, device=device)
        usefulness = evaluate_target_modality_usefulness(
            model,
            x=target_x,
            y=target_y,
            labels=target_labels,
            task=task,
            device=device,
            eval_batch_size=eval_batch_size,
        )
        values.append(usefulness)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return np.stack(values, axis=0)


def select_target_indices_by_variance(full_subset_f: np.ndarray, *, num_target_samples: int) -> np.ndarray:
    variance_score = np.nanvar(full_subset_f, axis=0).mean(axis=1)
    order = np.argsort(-variance_score)
    count = min(int(num_target_samples), int(order.shape[0]))
    return np.sort(order[:count].astype(np.int64, copy=False))


def build_pair_blocks(blocks: MaskBlocks, full_subset_f: np.ndarray) -> list[dict[str, Any]]:
    pair_blocks: list[dict[str, Any]] = []
    num_targets = int(full_subset_f.shape[1])
    num_modalities = int(full_subset_f.shape[2])
    for target_offset in range(num_targets):
        for modality_index in range(num_modalities):
            pair_blocks.append(
                {
                    "target_offset": int(target_offset),
                    "modality_index": int(modality_index),
                    "blocks": MaskBlocks(
                        sample_deleted=blocks.sample_deleted,
                        modality_deleted=blocks.modality_deleted,
                        flat_deleted=blocks.flat_deleted,
                        targets=full_subset_f[:, target_offset, modality_index].astype(np.float64, copy=True),
                        run_ids=blocks.run_ids,
                        dataset_size=blocks.dataset_size,
                        num_modalities=blocks.num_modalities,
                    ),
                }
            )
    return pair_blocks


def evaluate_target_modality_surrogates(
    pair_blocks: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str],
    train_fraction: float,
    validation_fraction: float,
    split_seed: int,
    search_strategy: str,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
) -> dict[str, Any]:
    config = KernelAttributionConfig(
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
        split_seed=int(split_seed),
        methods=tuple(normalize_method_name(method) for method in methods),
        search_hyperparams=True,
        search_strategy=str(search_strategy),
        ridge_grid=tuple(float(value) for value in ridge_grid),
        gamma_scale_grid=tuple(float(value) for value in gamma_scale_grid),
        modality_shared_grid=tuple(float(value) for value in modality_shared_grid),
    )
    pair_rows: list[dict[str, Any]] = []
    for pair in pair_blocks:
        evaluations = evaluate_methods(pair["blocks"], config=config)
        for method_name, evaluation in evaluations.items():
            pair_rows.append(
                {
                    "method": method_name,
                    "target_offset": int(pair["target_offset"]),
                    "modality_index": int(pair["modality_index"]),
                    "lds": float(evaluation.lds),
                    "rmse": float(evaluation.rmse),
                    "num_train": int(evaluation.num_train),
                    "num_validation": int(evaluation.num_validation),
                    "num_test": int(evaluation.num_test),
                    "ridge_lambda": float(evaluation.fit.ridge_lambda),
                    "validation_lds": evaluation.fit.validation_lds,
                    "validation_rmse": evaluation.fit.validation_rmse,
                    "test_search_lds": evaluation.fit.test_search_lds,
                    "test_search_rmse": evaluation.fit.test_search_rmse,
                    "loo_lds": evaluation.fit.loo_lds,
                    "loo_rmse": evaluation.fit.loo_rmse,
                    "gamma_sample": evaluation.fit.gamma_sample,
                    "gamma_modality": evaluation.fit.gamma_modality,
                    "gamma_flat": evaluation.fit.gamma_flat,
                    "modality_shared_weight": evaluation.fit.modality_shared_weight,
                    "linear_modality_weight": evaluation.fit.linear_modality_weight,
                }
            )

    summary: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    for method, rows in sorted(by_method.items()):
        lds_values = np.asarray([float(row["lds"]) for row in rows], dtype=np.float64)
        rmse_values = np.asarray([float(row["rmse"]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "num_target_pairs": int(len(rows)),
                "mean_lds": float(np.nanmean(lds_values)),
                "std_lds": float(np.nanstd(lds_values)),
                "mean_rmse": float(np.nanmean(rmse_values)),
                "std_rmse": float(np.nanstd(rmse_values)),
                "positive_lds_fraction": float(np.mean(lds_values > 0.0)),
            }
        )
    return {
        "pair_results": pair_rows,
        "summary": summary,
    }


def run_exact_csmi_validation(
    *,
    payload: dict[str, Any],
    data: dict[str, Any],
    blocks: MaskBlocks,
    pair_blocks: Sequence[dict[str, Any]],
    methods: Sequence[str],
    source_groups: np.ndarray,
    target_indices: np.ndarray,
    task: str,
    device: str,
    batch_size: int,
    eval_batch_size: int,
    exact_train_seed_base: int,
    ridge_grid: Sequence[float],
    gamma_scale_grid: Sequence[float],
    modality_shared_grid: Sequence[float],
    top_k: int,
    progress_log: Path | None = None,
    exact_epochs: int | None = None,
) -> dict[str, Any]:
    grouped_loader = build_full_grouped_loader(
        payload=payload,
        data=data,
        batch_size=batch_size,
    )
    model_args = training_args_from_payload(payload, device=device, exact_epochs=exact_epochs)
    all_group_ids = list(range(grouped_loader.num_groups))

    _progress(progress_log, "[exact] training anchor full-data model...")
    anchor_loader = grouped_loader.build_subset_loader(
        all_group_ids,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    anchor_model, _ = _train_model(
        anchor_loader,
        dim_x=int(data["train_x"].shape[1]),
        dim_y=int(data["train_y"].shape[1]),
        num_classes=int(data["num_classes"]),
        task=task,
        model_type=str(payload["config"]["model"]),
        args=model_args,
        seed=int(exact_train_seed_base),
    )
    anchor_f = evaluate_target_modality_usefulness(
        anchor_model,
        x=data["test_x"][target_indices],
        y=data["test_y"][target_indices],
        labels=data["test_labels"][target_indices],
        task=task,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    del anchor_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    exact_deleted_f = np.empty((source_groups.size, blocks.num_modalities, target_indices.size, blocks.num_modalities), dtype=np.float64)
    for source_offset, source_group in enumerate(source_groups.tolist()):
        for source_modality in range(blocks.num_modalities):
            _progress(
                progress_log,
                f"[exact] training deleted model for source group {source_group}, modality {source_modality}...",
            )
            group_modality_mask = np.ones((grouped_loader.num_groups, grouped_loader.num_modalities), dtype=bool)
            group_modality_mask[int(source_group), int(source_modality)] = False
            deleted_loader = grouped_loader.build_subset_loader(
                all_group_ids,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                modality_mask=group_modality_mask,
            )
            train_seed = int(exact_train_seed_base + 1 + source_offset * blocks.num_modalities + source_modality)
            model, _ = _train_model(
                deleted_loader,
                dim_x=int(data["train_x"].shape[1]),
                dim_y=int(data["train_y"].shape[1]),
                num_classes=int(data["num_classes"]),
                task=task,
                model_type=str(payload["config"]["model"]),
                args=model_args,
                seed=train_seed,
            )
            deleted_f = evaluate_target_modality_usefulness(
                model,
                x=data["test_x"][target_indices],
                y=data["test_y"][target_indices],
                labels=data["test_labels"][target_indices],
                task=task,
                device=device,
                eval_batch_size=eval_batch_size,
            )
            exact_deleted_f[source_offset, source_modality] = deleted_f
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    exact_csmi = anchor_f[None, None, :, :] - exact_deleted_f

    method_results: list[dict[str, Any]] = []
    for method in methods:
        method_name = normalize_method_name(method)
        _progress(progress_log, f"[surrogate] fitting pairwise surrogates for {method_name}...")
        surrogate_scores = np.empty_like(exact_csmi)
        pair_fit_rows: list[dict[str, Any]] = []
        for pair in pair_blocks:
            fit = fit_best_kernel_surrogate(
                pair["blocks"],
                train_indices=np.arange(pair["blocks"].targets.shape[0], dtype=np.int64),
                method=method_name,
                ridge_grid=ridge_grid,
                gamma_scale_grid=gamma_scale_grid,
                modality_shared_grid=modality_shared_grid,
            )
            predicted_anchor = predict_custom_query(
                pair["blocks"],
                fit,
                sample_deleted=np.zeros((1, pair["blocks"].dataset_size), dtype=np.float32),
                modality_deleted=np.zeros((1, pair["blocks"].dataset_size * pair["blocks"].num_modalities), dtype=np.float32),
                flat_deleted=np.zeros((1, pair["blocks"].dataset_size * pair["blocks"].num_modalities), dtype=np.float32),
            )[0]
            query_modality = np.zeros(
                (source_groups.size * pair["blocks"].num_modalities, pair["blocks"].dataset_size * pair["blocks"].num_modalities),
                dtype=np.float32,
            )
            query_sample = np.zeros((source_groups.size * pair["blocks"].num_modalities, pair["blocks"].dataset_size), dtype=np.float32)
            query_flat = np.zeros_like(query_modality)
            for source_offset, source_group in enumerate(source_groups.tolist()):
                for source_modality in range(pair["blocks"].num_modalities):
                    row = source_offset * pair["blocks"].num_modalities + source_modality
                    column = int(source_group) * pair["blocks"].num_modalities + source_modality
                    query_modality[row, column] = 1.0
                    query_flat[row, column] = 1.0
            predicted_deleted = predict_custom_query(
                pair["blocks"],
                fit,
                sample_deleted=query_sample,
                modality_deleted=query_modality,
                flat_deleted=query_flat,
            ).reshape(source_groups.size, pair["blocks"].num_modalities)
            surrogate_effect = predicted_anchor - predicted_deleted
            surrogate_scores[:, :, int(pair["target_offset"]), int(pair["modality_index"])] = surrogate_effect
            pair_fit_rows.append(
                {
                    "target_offset": int(pair["target_offset"]),
                    "modality_index": int(pair["modality_index"]),
                    "ridge_lambda": float(fit.ridge_lambda),
                    "loo_lds": fit.loo_lds,
                    "loo_rmse": fit.loo_rmse,
                    "gamma_sample": fit.gamma_sample,
                    "gamma_modality": fit.gamma_modality,
                    "gamma_flat": fit.gamma_flat,
                    "modality_shared_weight": fit.modality_shared_weight,
                    "linear_modality_weight": fit.linear_modality_weight,
                }
            )

        exact_flat = exact_csmi.reshape(-1)
        surrogate_flat = surrogate_scores.reshape(-1)
        method_results.append(
            {
                "method": method_name,
                "metrics": compare_csmi_scores(
                    surrogate_flat,
                    exact_flat,
                    top_k=int(top_k),
                ),
                "pair_fits": pair_fit_rows,
            }
        )

    return {
        "anchor_f": anchor_f.tolist(),
        "exact_csmi_shape": list(exact_csmi.shape),
        "method_results": method_results,
    }


def build_full_grouped_loader(
    *,
    payload: dict[str, Any],
    data: dict[str, Any],
    batch_size: int,
) -> Any:
    config = payload["config"]
    base_train_loader = make_loader(
        data["train_x"],
        data["train_y"],
        data["train_labels"],
        batch_size,
        True,
    )
    partition_seed = config.get("partition_seed")
    if partition_seed is None:
        partition_seed = int(payload["seed"])
    return build_grouped_subset_loader(
        base_train_loader,
        dataset_name=str(payload["dataset"]),
        num_groups=int(config["num_groups"]),
        seed=int(partition_seed),
        cache_dir=config.get("group_cache_dir"),
    )


def training_args_from_payload(
    payload: dict[str, Any],
    *,
    device: str,
    exact_epochs: int | None = None,
) -> SimpleNamespace:
    config = payload["config"]
    return SimpleNamespace(
        device=device,
        lr=float(config.get("lr", 5e-4)),
        epochs=int(exact_epochs if exact_epochs is not None else config.get("epochs", 15)),
        dropout=float(config.get("dropout", 0.1)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        proj_dim=int(config.get("proj_dim", 256)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
    )


def predict_custom_query(
    blocks: MaskBlocks,
    fit: Any,
    *,
    sample_deleted: np.ndarray,
    modality_deleted: np.ndarray,
    flat_deleted: np.ndarray,
) -> np.ndarray:
    sample_deleted = np.asarray(sample_deleted, dtype=np.float32)
    modality_deleted = np.asarray(modality_deleted, dtype=np.float32)
    flat_deleted = np.asarray(flat_deleted, dtype=np.float32)
    if sample_deleted.shape[0] != modality_deleted.shape[0] or sample_deleted.shape[0] != flat_deleted.shape[0]:
        raise ValueError("query mask arrays must have matching row count")
    augmented_blocks = MaskBlocks(
        sample_deleted=np.concatenate([blocks.sample_deleted, sample_deleted], axis=0),
        modality_deleted=np.concatenate([blocks.modality_deleted, modality_deleted], axis=0),
        flat_deleted=np.concatenate([blocks.flat_deleted, flat_deleted], axis=0),
        targets=np.concatenate([blocks.targets, np.zeros(sample_deleted.shape[0], dtype=np.float64)], axis=0),
        run_ids=tuple(list(blocks.run_ids) + list(range(sample_deleted.shape[0]))),
        dataset_size=blocks.dataset_size,
        num_modalities=blocks.num_modalities,
    )
    query_indices = np.arange(blocks.targets.shape[0], blocks.targets.shape[0] + sample_deleted.shape[0], dtype=np.int64)
    kernel, _ = compute_kernel(
        augmented_blocks,
        lhs_indices=query_indices,
        rhs_indices=fit.train_indices,
        method=fit.method,
        gamma_sample=fit.gamma_sample,
        gamma_modality=fit.gamma_modality,
        gamma_flat=fit.gamma_flat,
        modality_shared_weight=fit.modality_shared_weight,
        linear_modality_weight=fit.linear_modality_weight,
    )
    return fit.target_mean + kernel @ fit.alpha


def compare_csmi_scores(surrogate: np.ndarray, exact: np.ndarray, *, top_k: int) -> dict[str, Any]:
    surrogate = np.asarray(surrogate, dtype=np.float64).reshape(-1)
    exact = np.asarray(exact, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(surrogate) & np.isfinite(exact)
    surrogate = surrogate[finite_mask]
    exact = exact[finite_mask]
    if surrogate.size == 0:
        raise ValueError("no finite CSMI scores to compare")
    sign_accuracy = float(np.mean(np.sign(surrogate) == np.sign(exact)))
    k = min(int(top_k), int(surrogate.size))
    exact_order_pos = np.argsort(-exact)
    surrogate_order_pos = np.argsort(-surrogate)
    exact_order_neg = np.argsort(exact)
    surrogate_order_neg = np.argsort(surrogate)
    topk_pos_overlap = float(len(set(exact_order_pos[:k].tolist()) & set(surrogate_order_pos[:k].tolist())) / max(k, 1))
    topk_neg_overlap = float(len(set(exact_order_neg[:k].tolist()) & set(surrogate_order_neg[:k].tolist())) / max(k, 1))
    return {
        "num_scores": int(surrogate.size),
        "spearman": float(spearmanr(surrogate, exact)),
        "rmse": float(rmse(surrogate, exact)),
        "sign_accuracy": sign_accuracy,
        "topk_positive_overlap": topk_pos_overlap,
        "topk_negative_overlap": topk_neg_overlap,
        "exact_mean": float(np.mean(exact)),
        "surrogate_mean": float(np.mean(surrogate)),
    }


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CSMI Validation",
        "",
        f"- Dataset: `{payload['config']['dataset']}`",
        f"- Seed: `{payload['config']['seed']}`",
        f"- Result dir: `{payload['config']['result_dir']}`",
        f"- Target indices: `{payload['config']['target_indices']}`",
        f"- Source groups: `{payload['config']['source_groups']}`",
        "",
        "## Experiment 1: Surrogate Validation for F_{i,v}",
        "",
        "| Method | Target pairs | Mean LDS | Std LDS | Mean RMSE | Std RMSE | Positive LDS fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["experiment_1_surrogate_validation"]["summary"]:
        lines.append(
            f"| {row['method']} | {row['num_target_pairs']} | {row['mean_lds']:.6f} | "
            f"{row['std_lds']:.6f} | {row['mean_rmse']:.6f} | {row['std_rmse']:.6f} | "
            f"{row['positive_lds_fraction']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Experiment 2: Exact CSMI Validation",
            "",
            "| Method | Scores | Spearman | RMSE | SignAcc | TopK+ | TopK- |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["experiment_2_exact_csmi_validation"]["method_results"]:
        metric = row["metrics"]
        lines.append(
            f"| {row['method']} | {metric['num_scores']} | {metric['spearman']:.6f} | "
            f"{metric['rmse']:.6f} | {metric['sign_accuracy']:.6f} | "
            f"{metric['topk_positive_overlap']:.6f} | {metric['topk_negative_overlap']:.6f} |"
        )
    return "\n".join(lines)


def _progress(path: Path | None, message: str) -> None:
    print(message, flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


if __name__ == "__main__":
    main()
