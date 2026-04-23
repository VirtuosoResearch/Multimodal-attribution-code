from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.fisher_geometry.data import load_dataset, make_loader
from src.fisher_geometry.experiment import build_model, set_seed
from src.fisher_geometry.fisher_metrics import compute_loss
try:
    from src.pid_estimator.subset_loader import build_grouped_subset_loader
except ModuleNotFoundError:
    from pid_estimator.subset_loader import build_grouped_subset_loader


DEFAULT_CHECKPOINT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train"
DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
SUPPORTED_REAL_DATASETS = (
    "food101",
    "oxford_pets",
    "imdb",
    "humor",
    "sarcasm",
    "mosi",
    "mosei",
    "avmnist",
)


def _result_path(result_dir: Path, *, dataset_name: str, seed: int) -> Path:
    return result_dir / f"{dataset_name}_seed{seed}.json"


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    dataset_name: str,
    seed: int,
    sample_index: int,
    repeat_index: int | None = None,
) -> Path:
    base = f"{dataset_name}_seed{seed}_subset{sample_index + 1:03d}"
    if repeat_index is not None:
        base = f"{base}_repeat{repeat_index + 1:02d}"
    return checkpoint_dir / f"{base}.pt"


def _subset_metric_path(
    result_dir: Path,
    *,
    dataset_name: str,
    seed: int,
    sample_index: int,
) -> Path:
    return result_dir / dataset_name / f"seed{seed}_subset{sample_index + 1:03d}.json"


def _rate_token(rate: float) -> str:
    percent = float(rate) * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return str(int(round(percent)))
    return f"{percent:.4g}".replace(".", "p")


def _default_run_name(args: argparse.Namespace) -> str:
    num_repeats = len(args.train_repeat_seeds)
    return (
        f"{args.model}_"
        "group_unit_"
        f"data_keep{_rate_token(args.data_keep_rate)}_"
        f"mod_keep{_rate_token(args.modality_keep_rate)}_"
        f"g{args.num_groups}_"
        f"trainavg{num_repeats}"
    )


def _resolve_output_dirs(args: argparse.Namespace) -> None:
    run_name = args.run_name or _default_run_name(args)
    args.run_name = run_name
    if args.result_dir is None:
        args.result_dir = args.result_root / run_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.checkpoint_root / run_name


def _serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    for key in ("checkpoint_dir", "checkpoint_root", "result_dir", "result_root", "group_cache_dir"):
        value = payload.get(key)
        if value is not None:
            payload[key] = str(value)
    return payload


def _metric_name_for_task(task: str) -> str:
    if task == "multilabel":
        return "sample_f1"
    return "accuracy"


def _per_sample_loss(
    task: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if task == "binary":
        return F.binary_cross_entropy_with_logits(
            logits.view(-1),
            labels.float().view(-1),
            reduction="none",
        )
    if task == "multilabel":
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels.float(),
            reduction="none",
        )
        return loss.reshape(loss.shape[0], -1).mean(dim=1)
    return F.cross_entropy(logits, labels, reduction="none")


def _evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    task: str,
    device: str,
    max_sample_losses: int,
) -> dict[str, float | str | list[float] | int]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_total = 0.0
    sample_losses: list[float] = []

    with torch.inference_mode():
        for bx, by, bl in loader:
            bx = bx.to(device)
            by = by.to(device)
            bl = bl.to(device)

            logits = model(bx, by, use_x=True, use_y=True)
            loss = compute_loss(task, logits, bl)
            per_sample_loss = _per_sample_loss(task, logits, bl)
            batch_size = int(bl.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            if len(sample_losses) < max_sample_losses:
                remaining = max_sample_losses - len(sample_losses)
                sample_losses.extend(
                    float(value)
                    for value in per_sample_loss[:remaining].detach().cpu().tolist()
                )

            if task == "multiclass":
                preds = logits.argmax(dim=1)
                metric_total += float((preds == bl).sum().item())
            elif task == "binary":
                preds = (logits.view(-1) > 0).long()
                metric_total += float((preds == bl.view(-1).long()).sum().item())
            else:
                labels = bl.float()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                true_positive = (preds * labels).sum(dim=1)
                precision = true_positive / preds.sum(dim=1).clamp(min=1.0)
                recall = true_positive / labels.sum(dim=1).clamp(min=1.0)
                denom = (precision + recall).clamp(min=1e-8)
                sample_f1 = torch.where(
                    precision + recall > 0.0,
                    2.0 * precision * recall / denom,
                    torch.zeros_like(precision),
                )
                metric_total += float(sample_f1.sum().item())

    if total_samples == 0:
        raise ValueError("evaluation loader produced no batches")

    return {
        "loss": total_loss / total_samples,
        "acc": metric_total / total_samples,
        "metric_name": _metric_name_for_task(task),
        "sample_losses": sample_losses,
        "num_sample_losses": int(len(sample_losses)),
    }


def _train_model(
    train_loader: DataLoader,
    *,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    task: str,
    model_type: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.nn.Module, list[float]]:
    set_seed(seed)
    model = build_model(model_type, dim_x, dim_y, num_classes, args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epoch_losses: list[float] = []

    for _ in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for bx, by, bl in train_loader:
            bx = bx.to(args.device)
            by = by.to(args.device)
            bl = bl.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, by, use_x=True, use_y=True)
            loss = compute_loss(task, logits, bl)
            loss.backward()
            optimizer.step()

            batch_size = int(bl.shape[0])
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        epoch_losses.append(running_loss / max(total_samples, 1))

    model.eval()
    return model, epoch_losses


def _resolved_groups_per_subset(num_groups: int, data_keep_rate: float) -> int:
    keep_rate = float(data_keep_rate)
    if keep_rate <= 0.0:
        raise ValueError(f"data_keep_rate={data_keep_rate} keeps no data")
    return min(num_groups, max(1, int(math.ceil(num_groups * keep_rate))))


def _sample_subset_specs(
    *,
    grouped_loader,
    num_subset_samples: int,
    groups_per_subset: int,
    modality_keep_prob: float,
    subset_sampling_seed_base: int,
    existing_signatures: set[tuple[tuple[int, ...], bytes]],
) -> list[dict[str, Any]]:
    if groups_per_subset > grouped_loader.num_groups:
        raise ValueError(
            f"groups_per_subset={groups_per_subset} exceeds available groups={grouped_loader.num_groups}"
        )

    sampled: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(num_subset_samples * 500, 500)
    while len(sampled) < num_subset_samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                "failed to sample enough unique LDS subsets; "
                f"requested={num_subset_samples}, collected={len(sampled)}"
            )

        subset_seed = int(subset_sampling_seed_base + attempts)
        rng = np.random.default_rng(subset_seed)
        group_ids = sorted(rng.choice(grouped_loader.num_groups, size=groups_per_subset, replace=False).tolist())
        subset_indices = grouped_loader.get_subset_indices(group_ids).clone().cpu()
        modality_mask = rng.random((int(subset_indices.numel()), grouped_loader.num_modalities)) < modality_keep_prob
        signature = (
            tuple(int(group_id) for group_id in group_ids),
            np.packbits(modality_mask.reshape(-1).astype(np.uint8), bitorder="little").tobytes(),
        )
        attempts += 1
        if signature in existing_signatures:
            continue
        existing_signatures.add(signature)
        sampled.append(
            {
                "subset_seed": subset_seed,
                "group_ids": [int(group_id) for group_id in group_ids],
                "subset_indices": subset_indices.clone().cpu(),
                "modality_mask": modality_mask.astype(bool, copy=True),
            }
        )
    return sampled


def _group_partition_payload(grouped_loader) -> dict[str, Any]:
    return {
        "cache_path": str(grouped_loader.cache_path),
        "num_groups": int(grouped_loader.num_groups),
        "num_modalities": int(grouped_loader.num_modalities),
        "modality_names": [str(name) for name in grouped_loader.modality_names],
        "group_sizes": [int(size) for size in grouped_loader.group_sizes],
        "group_indices": [
            [int(index) for index in indices.tolist()]
            for indices in grouped_loader.group_indices
        ],
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    if not runs:
        return {
            "num_completed_subsets": 0,
            "mean_test_loss": float("nan"),
            "std_test_loss": float("nan"),
            "mean_test_acc": float("nan"),
            "std_test_acc": float("nan"),
            "mean_effective_sample_keep_ratio": float("nan"),
            "mean_modality_keep_ratio": float("nan"),
        }

    losses = np.asarray([float(run["test_loss"]) for run in runs], dtype=np.float64)
    accs = np.asarray([float(run["test_acc"]) for run in runs], dtype=np.float64)
    sample_keep = np.asarray([float(run["effective_sample_keep_ratio"]) for run in runs], dtype=np.float64)
    modality_keep = np.asarray([float(run["modality_keep_ratio_over_selected"]) for run in runs], dtype=np.float64)
    return {
        "num_completed_subsets": int(len(runs)),
        "mean_test_loss": float(losses.mean()),
        "std_test_loss": float(losses.std()),
        "mean_test_acc": float(accs.mean()),
        "std_test_acc": float(accs.std()),
        "mean_effective_sample_keep_ratio": float(sample_keep.mean()),
        "mean_modality_keep_ratio": float(modality_keep.mean()),
    }


def _initial_result_payload(
    *,
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
    grouped_loader,
    task: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "seed": int(seed),
        "task": task,
        "metric_name": _metric_name_for_task(task),
        "config": _serialize_args(args),
        "group_partition": _group_partition_payload(grouped_loader),
        "runs": [],
        "summary": _summarize_runs([]),
    }


def _config_matches(existing: dict[str, Any], args: argparse.Namespace) -> bool:
    current_config = _serialize_args(args)
    existing_config = existing.get("config")
    if not isinstance(existing_config, dict):
        return False
    existing_config = dict(existing_config)

    # Let users increase the target number of subset samples and resume instead
    # of overwriting already trained subsets.
    ignored_resume_keys = {"num_subset_samples"}
    for key in ignored_resume_keys:
        current_config.pop(key, None)
        existing_config.pop(key, None)
    return existing_config == current_config


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _existing_subset_signatures(runs: list[dict[str, Any]]) -> set[tuple[tuple[int, ...], bytes]]:
    signatures: set[tuple[tuple[int, ...], bytes]] = set()
    for run in runs:
        group_ids = tuple(int(group_id) for group_id in run.get("group_ids", []))
        modality_mask = np.asarray(run.get("sample_modality_mask", []), dtype=bool)
        mask_bytes = np.packbits(modality_mask.reshape(-1).astype(np.uint8), bitorder="little").tobytes()
        signatures.add((group_ids, mask_bytes))
    return signatures


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.data_keep_rate <= 1.0:
        raise ValueError(f"data_keep_rate must be in (0, 1], got {args.data_keep_rate}")
    if not 0.0 < args.modality_keep_rate <= 1.0:
        raise ValueError(f"modality_keep_rate must be in (0, 1], got {args.modality_keep_rate}")
    if not args.train_repeat_seeds:
        raise ValueError("train_repeat_seeds must contain at least one seed")
    if args.max_test_loss_samples < 0:
        raise ValueError(f"max_test_loss_samples must be >= 0, got {args.max_test_loss_samples}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device} but CUDA is unavailable")


def run_single(dataset_name: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    result_path = _result_path(args.result_dir, dataset_name=dataset_name, seed=seed)

    print(f"\n[{dataset_name}] seed={seed}: loading data...", flush=True)
    data = load_dataset(
        dataset_name,
        seed,
        text_backbone=args.text_backbone,
        vision_backbone=args.vision_backbone,
        feature_device=args.feature_device,
        backbone_batch_size=args.backbone_batch_size,
        download_datasets=args.download_datasets,
        backbone_cache_dir=args.backbone_cache_dir,
    )

    base_train_loader = make_loader(
        data["train_x"],
        data["train_y"],
        data["train_labels"],
        args.batch_size,
        True,
    )
    partition_seed = args.partition_seed if args.partition_seed is not None else seed
    grouped_loader = build_grouped_subset_loader(
        base_train_loader,
        dataset_name=dataset_name,
        num_groups=args.num_groups,
        seed=partition_seed,
        cache_dir=args.group_cache_dir,
    )
    groups_per_subset = _resolved_groups_per_subset(grouped_loader.num_groups, args.data_keep_rate)
    modality_keep_prob = float(args.modality_keep_rate)

    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not _config_matches(payload, args):
            print(
                f"[{dataset_name}] seed={seed}: existing result uses different config, overwriting",
                flush=True,
            )
            payload = _initial_result_payload(
                dataset_name=dataset_name,
                seed=seed,
                args=args,
                grouped_loader=grouped_loader,
                task=data["task"],
            )
        else:
            payload["config"] = _serialize_args(args)
            print(
                f"[{dataset_name}] seed={seed}: resuming with {len(payload.get('runs', []))} finished subsets",
                flush=True,
            )
    else:
        payload = _initial_result_payload(
            dataset_name=dataset_name,
            seed=seed,
            args=args,
            grouped_loader=grouped_loader,
            task=data["task"],
        )

    completed_runs = payload.get("runs", [])
    remaining = args.num_subset_samples - len(completed_runs)
    if remaining <= 0:
        payload["summary"] = _summarize_runs(completed_runs)
        _save_json(result_path, payload)
        return payload

    subset_sampling_seed_base = args.subset_sampling_seed_base + seed * 100_000
    subset_specs = _sample_subset_specs(
        grouped_loader=grouped_loader,
        num_subset_samples=remaining,
        groups_per_subset=groups_per_subset,
        modality_keep_prob=modality_keep_prob,
        subset_sampling_seed_base=subset_sampling_seed_base,
        existing_signatures=_existing_subset_signatures(completed_runs),
    )

    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    task = str(data["task"])
    test_loader = make_loader(
        data["test_x"],
        data["test_y"],
        data["test_labels"],
        args.eval_batch_size,
        False,
    )

    for offset, subset_spec in enumerate(subset_specs, start=len(completed_runs)):
        subset_indices = subset_spec["subset_indices"]
        group_ids = subset_spec["group_ids"]
        modality_mask = subset_spec["modality_mask"]
        subset_train_loader = grouped_loader.build_subset_loader(
            group_ids,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            modality_mask=modality_mask,
        )

        sample_modality_mask = np.asarray(modality_mask, dtype=bool)
        kept_atom_indices = []
        for row_index, dataset_index in enumerate(subset_indices.tolist()):
            for modality_index, keep in enumerate(sample_modality_mask[row_index].tolist()):
                if keep:
                    kept_atom_indices.append(
                        int(dataset_index) * grouped_loader.num_modalities + int(modality_index)
                    )
        effective_sample_count = int(sample_modality_mask.any(axis=1).sum())

        print(
            f"[{dataset_name}] seed={seed}: subset {offset + 1}/{args.num_subset_samples} "
            f"training {len(args.train_repeat_seeds)} repeat seeds on {len(group_ids)} groups / "
            f"{int(subset_indices.numel())} selected samples...",
            flush=True,
        )

        repeat_payloads: list[dict[str, Any]] = []
        checkpoint_paths: list[str] = []
        repeat_test_losses: list[float] = []
        repeat_test_accs: list[float] = []
        repeat_final_train_losses: list[float] = []
        repeat_sample_losses: list[list[float]] = []
        for repeat_index, repeat_seed in enumerate(args.train_repeat_seeds):
            train_seed = seed * 1_000_000 + offset * 10_000 + int(repeat_seed)
            print(
                f"[{dataset_name}] seed={seed}: subset {offset + 1}/{args.num_subset_samples} "
                f"repeat {repeat_index + 1}/{len(args.train_repeat_seeds)} train_seed={train_seed}",
                flush=True,
            )
            model, epoch_losses = _train_model(
                subset_train_loader,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                task=task,
                model_type=args.model,
                args=args,
                seed=train_seed,
            )
            test_metrics = _evaluate_classifier(
                model,
                test_loader,
                task=task,
                device=args.device,
                max_sample_losses=args.max_test_loss_samples,
            )
            checkpoint_path = _checkpoint_path(
                args.checkpoint_dir,
                dataset_name=dataset_name,
                seed=seed,
                sample_index=offset,
                repeat_index=repeat_index,
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            scalar_test_metrics = {
                "loss": float(test_metrics["loss"]),
                "acc": float(test_metrics["acc"]),
                "metric_name": str(test_metrics["metric_name"]),
                "num_sample_losses": int(test_metrics["num_sample_losses"]),
            }
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model_config": {
                        "model": args.model,
                        "dim_x": dim_x,
                        "dim_y": dim_y,
                        "num_classes": num_classes,
                        "dropout": args.dropout,
                        "hidden_dim": args.hidden_dim,
                        "proj_dim": args.proj_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                    },
                    "dataset": dataset_name,
                    "task": task,
                    "seed": int(seed),
                    "partition_seed": int(partition_seed),
                    "subset_seed": int(subset_spec["subset_seed"]),
                    "train_seed": int(train_seed),
                    "train_repeat_seed": int(repeat_seed),
                    "train_repeat_index": int(repeat_index),
                    "sample_index": int(offset),
                    "num_groups": int(grouped_loader.num_groups),
                    "groups_per_subset": int(groups_per_subset),
                    "selected_group_count": int(len(group_ids)),
                    "group_ids": [int(group_id) for group_id in group_ids],
                    "data_indices": [int(index) for index in subset_indices.tolist()],
                    "sample_modality_mask": sample_modality_mask.astype(int).tolist(),
                    "kept_atom_indices": kept_atom_indices,
                    "effective_sample_count": effective_sample_count,
                    "test_metrics": scalar_test_metrics,
                    "epoch_losses": [float(loss) for loss in epoch_losses],
                    "group_cache_path": str(grouped_loader.cache_path),
                },
                checkpoint_path,
            )
            checkpoint_paths.append(str(checkpoint_path))
            repeat_test_losses.append(float(test_metrics["loss"]))
            repeat_test_accs.append(float(test_metrics["acc"]))
            repeat_final_train_losses.append(float(epoch_losses[-1]))
            repeat_sample_losses.append([float(loss) for loss in test_metrics["sample_losses"]])
            repeat_payloads.append(
                {
                    "train_repeat_index": int(repeat_index),
                    "train_repeat_seed": int(repeat_seed),
                    "train_seed": int(train_seed),
                    "checkpoint_path": str(checkpoint_path),
                    "final_train_loss": float(epoch_losses[-1]),
                    "test_loss": float(test_metrics["loss"]),
                    "test_acc": float(test_metrics["acc"]),
                    "num_test_sample_losses": int(test_metrics["num_sample_losses"]),
                }
            )

        min_sample_loss_count = min((len(losses) for losses in repeat_sample_losses), default=0)
        if min_sample_loss_count:
            sample_loss_matrix = np.asarray(
                [losses[:min_sample_loss_count] for losses in repeat_sample_losses],
                dtype=np.float64,
            )
            test_sample_losses = sample_loss_matrix.mean(axis=0).tolist()
            test_sample_loss_std = sample_loss_matrix.std(axis=0).tolist()
        else:
            test_sample_losses = []
            test_sample_loss_std = []

        test_loss_array = np.asarray(repeat_test_losses, dtype=np.float64)
        test_acc_array = np.asarray(repeat_test_accs, dtype=np.float64)
        final_train_loss_array = np.asarray(repeat_final_train_losses, dtype=np.float64)

        run_payload = {
            "sample_index": int(offset),
            "subset_seed": int(subset_spec["subset_seed"]),
            "train_repeat_seeds": [int(repeat_seed) for repeat_seed in args.train_repeat_seeds],
            "train_repeats": repeat_payloads,
            "group_ids": [int(group_id) for group_id in group_ids],
            "data_indices": [int(index) for index in subset_indices.tolist()],
            "sample_modality_mask": sample_modality_mask.astype(int).tolist(),
            "kept_atom_indices": kept_atom_indices,
            "groups_per_subset": int(groups_per_subset),
            "selected_group_count": int(len(group_ids)),
            "selected_sample_count": int(subset_indices.numel()),
            "effective_sample_count": effective_sample_count,
            "subset_size": int(subset_indices.numel()),
            "effective_sample_keep_ratio": float(effective_sample_count / max(len(data["train_labels"]), 1)),
            "selected_sample_keep_ratio": float(subset_indices.numel() / max(len(data["train_labels"]), 1)),
            "modality_keep_ratio_over_selected": float(sample_modality_mask.mean()),
            "modality_missing_ratio_over_selected": float(1.0 - sample_modality_mask.mean()),
            "checkpoint_path": checkpoint_paths[0] if checkpoint_paths else "",
            "checkpoint_paths": checkpoint_paths,
            "final_train_loss": float(final_train_loss_array.mean()),
            "final_train_loss_std": float(final_train_loss_array.std()),
            "test_loss": float(test_loss_array.mean()),
            "test_loss_std": float(test_loss_array.std()),
            "test_acc": float(test_acc_array.mean()),
            "test_acc_std": float(test_acc_array.std()),
            "test_metric_name": _metric_name_for_task(task),
            "max_test_loss_samples": int(args.max_test_loss_samples),
            "num_test_sample_losses": int(len(test_sample_losses)),
            "test_sample_indices": [int(index) for index in range(len(test_sample_losses))],
            "test_sample_losses": [float(loss) for loss in test_sample_losses],
            "test_sample_loss_std": [float(loss) for loss in test_sample_loss_std],
        }
        metric_path = _subset_metric_path(
            args.result_dir,
            dataset_name=dataset_name,
            seed=seed,
            sample_index=offset,
        )
        run_payload["metric_json_path"] = str(metric_path)
        _save_json(metric_path, run_payload)

        completed_runs.append(run_payload)
        payload["runs"] = completed_runs
        payload["summary"] = _summarize_runs(completed_runs)
        _save_json(result_path, payload)

    return payload


def _write_combined_results(result_dir: Path, all_results: list[dict[str, Any]]) -> None:
    combined_path = result_dir / "all_results.json"
    _save_json(
        combined_path,
        {
            "num_dataset_runs": int(len(all_results)),
            "results": all_results,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LDS-style real-data baseline by sampling train subsets with "
            "sample-level and modality-level missingness, training repeated models per subset, "
            "and saving subset metadata, checkpoints, and test metrics."
        )
    )
    parser.add_argument("--model", choices=["concat_mlp", "shared_mlp", "cross_attention"], default="cross_attention")
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_REAL_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default="cuda:0")
    parser.add_argument("--text-backbone", default="clip_cached")
    parser.add_argument("--vision-backbone", default="clip_cached")
    parser.add_argument("--backbone-batch-size", type=int, default=32)
    parser.add_argument("--download-datasets", action="store_true")
    parser.add_argument("--backbone-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4196)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-groups", type=int, default=50)
    parser.add_argument("--data-keep-rate", type=float, default=0.25)
    parser.add_argument("--modality-keep-rate", type=float, default=0.9)
    parser.add_argument("--data-missing-rate", type=float, default=None)
    parser.add_argument("--modality-missing-rate", type=float, default=None)
    parser.add_argument("--num-subset-samples", type=int, default=50)
    parser.add_argument("--train-repeat-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-test-loss-samples", type=int, default=1000)
    parser.add_argument("--partition-seed", type=int, default=None)
    parser.add_argument("--subset-sampling-seed-base", type=int, default=17_000)
    parser.add_argument("--group-cache-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.data_missing_rate is not None:
        args.data_keep_rate = 1.0 - float(args.data_missing_rate)
    if args.modality_missing_rate is not None:
        args.modality_keep_rate = 1.0 - float(args.modality_missing_rate)
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    _validate_args(args)
    _resolve_output_dirs(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        for seed in args.seeds:
            result = run_single(dataset_name, seed, args)
            all_results.append(result)
            print(
                f"[{dataset_name}] seed={seed}: completed {result['summary']['num_completed_subsets']} subsets, "
                f"mean test acc={result['summary']['mean_test_acc']:.4f}",
                flush=True,
            )

    _write_combined_results(args.result_dir, all_results)


if __name__ == "__main__":
    main()
