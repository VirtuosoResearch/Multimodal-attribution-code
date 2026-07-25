from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fisher_geometry.data import load_dataset, make_loader
from src.fisher_geometry.experiment import build_model, set_seed
from src.fisher_geometry.fisher_metrics import compute_loss
from src.pid_estimator.subset_loader import build_grouped_subset_loader


DEFAULT_CHECKPOINT_DIR = ROOT / "src" / "pid_estimator" / "subset_checkpoints"
DEFAULT_RESULT_DIR = ROOT / "src" / "pid_estimator" / "subset_results"
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
) -> Path:
    return checkpoint_dir / f"{dataset_name}_seed{seed}_subset{sample_index + 1:03d}.pt"


def _serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    for key in ("checkpoint_dir", "result_dir", "group_cache_dir"):
        value = payload.get(key)
        if value is not None:
            payload[key] = str(value)
    return payload


def _metric_name_for_task(task: str) -> str:
    if task == "multilabel":
        return "sample_f1"
    return "accuracy"


def _evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    task: str,
    device: str,
) -> dict[str, float | str]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_total = 0.0

    with torch.inference_mode():
        for bx, by, bl in loader:
            bx = bx.to(device)
            by = by.to(device)
            bl = bl.to(device)

            logits = model(bx, by, use_x=True, use_y=True)
            loss = compute_loss(task, logits, bl)
            batch_size = int(bl.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

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


def _sample_group_ids(
    *,
    num_groups: int,
    groups_per_subset: int,
    num_samples: int,
    sampling_seed_base: int,
    existing_group_ids: list[list[int]],
) -> list[tuple[int, list[int]]]:
    if groups_per_subset > num_groups:
        raise ValueError(
            f"groups_per_subset={groups_per_subset} exceeds available groups={num_groups}"
        )

    seen = {tuple(int(group_id) for group_id in group_ids) for group_ids in existing_group_ids}
    sampled: list[tuple[int, list[int]]] = []
    candidate_offset = 0
    max_attempts = max(num_samples * 1000, 1000)

    while len(sampled) < num_samples:
        if candidate_offset >= max_attempts:
            raise RuntimeError(
                "failed to sample enough unique group subsets; "
                f"requested={num_samples}, collected={len(sampled)}"
            )
        sample_seed = int(sampling_seed_base + candidate_offset)
        rng = np.random.default_rng(sample_seed)
        group_ids = sorted(rng.choice(num_groups, size=groups_per_subset, replace=False).tolist())
        key = tuple(int(group_id) for group_id in group_ids)
        candidate_offset += 1
        if key in seen:
            continue
        seen.add(key)
        sampled.append((sample_seed, [int(group_id) for group_id in group_ids]))
    return sampled


def _group_partition_payload(grouped_loader) -> dict[str, Any]:
    return {
        "cache_path": str(grouped_loader.cache_path),
        "num_groups": int(grouped_loader.num_groups),
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
        }

    losses = np.asarray([float(run["test_loss"]) for run in runs], dtype=np.float64)
    accs = np.asarray([float(run["test_acc"]) for run in runs], dtype=np.float64)
    return {
        "num_completed_subsets": int(len(runs)),
        "mean_test_loss": float(losses.mean()),
        "std_test_loss": float(losses.std()),
        "mean_test_acc": float(accs.mean()),
        "std_test_acc": float(accs.std()),
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
    return existing.get("config") == _serialize_args(args)


def _save_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


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
    if args.groups_per_subset > grouped_loader.num_groups:
        raise ValueError(
            f"{dataset_name}: groups_per_subset={args.groups_per_subset} exceeds "
            f"resolved num_groups={grouped_loader.num_groups}"
        )

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
    existing_group_ids = [run["group_ids"] for run in completed_runs]
    remaining = args.num_subset_samples - len(completed_runs)
    if remaining <= 0:
        payload["summary"] = _summarize_runs(completed_runs)
        _save_result(result_path, payload)
        return payload

    sampling_seed_base = args.subset_sampling_seed_base + seed * 10_000
    sampled_group_sets = _sample_group_ids(
        num_groups=grouped_loader.num_groups,
        groups_per_subset=args.groups_per_subset,
        num_samples=remaining,
        sampling_seed_base=sampling_seed_base,
        existing_group_ids=existing_group_ids,
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

    for offset, (subset_seed, group_ids) in enumerate(sampled_group_sets, start=len(completed_runs)):
        subset_indices = grouped_loader.get_subset_indices(group_ids)
        subset_train_loader = grouped_loader.build_subset_loader(
            group_ids,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        print(
            f"[{dataset_name}] seed={seed}: subset {offset + 1}/{args.num_subset_samples} "
            f"training on {len(group_ids)} groups / {int(subset_indices.numel())} samples...",
            flush=True,
        )
        train_seed = seed * 100_000 + offset
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
        test_metrics = _evaluate_classifier(model, test_loader, task=task, device=args.device)

        checkpoint_path = _checkpoint_path(
            args.checkpoint_dir,
            dataset_name=dataset_name,
            seed=seed,
            sample_index=offset,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
                "subset_seed": int(subset_seed),
                "train_seed": int(train_seed),
                "sample_index": int(offset),
                "num_groups": int(grouped_loader.num_groups),
                "groups_per_subset": int(args.groups_per_subset),
                "group_ids": [int(group_id) for group_id in group_ids],
                "data_indices": [int(index) for index in subset_indices.tolist()],
                "test_metrics": test_metrics,
                "epoch_losses": [float(loss) for loss in epoch_losses],
                "group_cache_path": str(grouped_loader.cache_path),
            },
            checkpoint_path,
        )

        run_payload = {
            "sample_index": int(offset),
            "subset_seed": int(subset_seed),
            "train_seed": int(train_seed),
            "group_ids": [int(group_id) for group_id in group_ids],
            "data_indices": [int(index) for index in subset_indices.tolist()],
            "subset_size": int(subset_indices.numel()),
            "checkpoint_path": str(checkpoint_path),
            "final_train_loss": float(epoch_losses[-1]),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "test_metric_name": str(test_metrics["metric_name"]),
        }
        completed_runs.append(run_payload)
        payload["runs"] = completed_runs
        payload["summary"] = _summarize_runs(completed_runs)
        _save_result(result_path, payload)

    return payload


def _write_combined_results(result_dir: Path, all_results: list[dict[str, Any]]) -> None:
    combined_path = result_dir / "all_results.json"
    combined_payload = {
        "num_dataset_runs": int(len(all_results)),
        "results": all_results,
    }
    _save_result(combined_path, combined_payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train multimodal models on random train subsets of each real dataset "
            "using cached group partitions from pid_estimator.subset_loader."
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-groups", type=int, default=50)
    parser.add_argument("--groups-per-subset", type=int, default=40)
    parser.add_argument("--num-subset-samples", type=int, default=50)
    parser.add_argument("--partition-seed", type=int, default=None)
    parser.add_argument("--subset-sampling-seed-base", type=int, default=17_000)
    parser.add_argument("--group-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
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
