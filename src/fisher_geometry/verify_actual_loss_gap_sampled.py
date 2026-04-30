from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

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

from fisher_geometry.data import load_dataset, make_loader
from fisher_geometry.experiment import build_model, set_seed
from fisher_geometry.fisher_metrics import compute_loss
from fisher_geometry.verify_humor_fisher_gpt_metrics import (
    DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT,
    DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT,
    load_model,
    make_model_args,
    resolve_full_checkpoint,
    resolve_requested_device,
)


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "kmeans_grouped25_subset100_data80_mod80"
)
DEFAULT_LOSS_OSFI_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "fisher_loss_osfi"
DEFAULT_OUTPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "actual_val_loss_gap_sampled"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sampled actual target loss gap validation for fisher_test_loss.md. "
            "For selected target groups, train a self-only model from scratch on that group "
            "and compare validation loss against the full-data checkpoint."
        )
    )
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--loss-osfi-dir", type=Path, default=DEFAULT_LOSS_OSFI_DIR)
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--full-checkpoint-roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT, DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--target-groups", nargs="*", type=int, default=None)
    parser.add_argument(
        "--selection",
        choices=["loss_osfi_quantiles", "first"],
        default="loss_osfi_quantiles",
    )
    parser.add_argument("--num-targets", type=int, default=3)
    parser.add_argument("--min-group-size", type=int, default=100)
    parser.add_argument("--fallback-min-group-size", type=int, default=50)
    parser.add_argument("--min-eval-group-size", type=int, default=5)
    parser.add_argument("--eval-split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--val-assignment", choices=["nearest_train_group_centroid"], default="nearest_train_group_centroid")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train-seed-base", type=int, default=88000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = resolve_requested_device(str(args.device))
    payload_path = args.result_dir / f"{args.dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
    data = load_dataset(
        args.dataset,
        int(args.seed),
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=str(args.feature_device or config.get("feature_device", device)),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    model_args = make_model_args(config, device=str(device))
    if args.epochs is not None:
        model_args.epochs = int(args.epochs)
    if args.batch_size is not None:
        model_args.batch_size = int(args.batch_size)
        model_args.eval_batch_size = int(args.batch_size)
    if args.lr is not None:
        model_args.lr = float(args.lr)

    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    task = str(data["task"])
    model_type = str(config.get("model", "cross_attention"))
    group_indices = [[int(index) for index in group] for group in payload["group_partition"]["group_indices"]]
    eval_group_indices = resolve_eval_group_indices(
        data=data,
        group_indices=group_indices,
        eval_split=str(args.eval_split),
    )
    target_groups, effective_min_group_size = resolve_target_groups(
        explicit=args.target_groups,
        dataset=str(args.dataset),
        seed=int(args.seed),
        group_indices=group_indices,
        eval_group_indices=eval_group_indices,
        loss_osfi_dir=args.loss_osfi_dir,
        selection=str(args.selection),
        num_targets=int(args.num_targets),
        min_group_size=int(args.min_group_size),
        fallback_min_group_size=int(args.fallback_min_group_size),
        min_eval_group_size=int(args.min_eval_group_size),
    )
    full_checkpoint = resolve_full_checkpoint(
        args.full_checkpoint,
        roots=args.full_checkpoint_roots,
        dataset=str(args.dataset),
        seed=int(args.seed),
    )
    loss_osfi_lookup = load_loss_osfi_lookup(args.loss_osfi_dir, dataset=str(args.dataset), seed=int(args.seed))

    print(f"Loading full checkpoint: {full_checkpoint}", flush=True)
    full_model = load_model(
        full_checkpoint,
        model_type=model_type,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        device=str(device),
    )

    output_json = args.output_json or (args.output_dir / f"{args.dataset}_seed{args.seed}_actual_loss_gap_sampled.json")
    checkpoint_dir = output_json.parent / "self_only_checkpoints" / str(args.dataset)
    records = []
    for target_group in target_groups:
        print(f"[{args.dataset}] target group {target_group}: training self-only model", flush=True)
        train_indices = group_indices[target_group]
        eval_indices = eval_group_indices[target_group]
        if len(eval_indices) < int(args.min_eval_group_size):
            print(
                f"  skipped: eval group has {len(eval_indices)} samples "
                f"(< {int(args.min_eval_group_size)})",
                flush=True,
            )
            continue
        full_loss = group_loss_value(
            full_model,
            data=data,
            indices=eval_indices,
            split=str(args.eval_split),
            task=task,
            device=str(device),
        )
        self_ckpt = checkpoint_dir / f"target_g{target_group:02d}_seed{args.seed}_epochs{int(model_args.epochs)}.pt"
        if self_ckpt.exists() and not args.force:
            self_model = load_model(
                self_ckpt,
                model_type=model_type,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                device=str(device),
            )
            train_losses = []
        else:
            self_model, train_losses = train_self_only_model(
                data=data,
                indices=train_indices,
                task=task,
                model_type=model_type,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                seed=int(args.train_seed_base) + int(args.seed) * 1000 + int(target_group),
            )
            save_checkpoint(self_ckpt, self_model, model_type, dim_x, dim_y, num_classes, model_args, task, int(args.seed))
            self_model.to(device)
        self_loss = group_loss_value(
            self_model,
            data=data,
            indices=eval_indices,
            split=str(args.eval_split),
            task=task,
            device=str(device),
        )
        actual_gap = float(self_loss - full_loss)
        pred = loss_osfi_lookup.get(int(target_group), {})
        pred_score = pred.get("loss_osfi")
        records.append(
            {
                "target_group": int(target_group),
                "group_size": int(len(train_indices)),
                "train_group_size": int(len(train_indices)),
                "eval_group_size": int(len(eval_indices)),
                "eval_split": str(args.eval_split),
                "self_loss": float(self_loss),
                "full_loss": float(full_loss),
                "actual_loss_gap": actual_gap,
                "self_train_epoch_losses": [float(value) for value in train_losses],
                "pred_loss_gap": pred.get("pred_loss_gap"),
                "loss_osfi": pred.get("loss_osfi"),
                "pred_score": pred_score,
                "pred_score_name": "loss_osfi",
                "osfi": pred.get("osfi"),
                "pred_positive": None if pred_score is None else bool(float(pred_score) > 0.0),
                "actual_positive": bool(actual_gap > 0.0),
                "sign_agree_pred_score": (
                    None if pred_score is None else bool((float(pred_score) > 0.0) == (actual_gap > 0.0))
                ),
            }
        )
        print(
            f"  full_loss={full_loss:.6f}, self_loss={self_loss:.6f}, "
            f"actual_gap={actual_gap:.6f}",
            flush=True,
        )

    summary = summarize(records)
    output = {
        "config": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "payload_path": str(payload_path),
            "full_checkpoint": str(full_checkpoint),
            "device": str(device),
            "feature_device": str(args.feature_device or config.get("feature_device", device)),
            "model": model_type,
            "target_groups": [int(value) for value in target_groups],
            "selection": str(args.selection),
            "min_group_size": int(args.min_group_size),
            "fallback_min_group_size": int(args.fallback_min_group_size),
            "effective_min_group_size": int(effective_min_group_size),
            "min_eval_group_size": int(args.min_eval_group_size),
            "eval_split": str(args.eval_split),
            "val_assignment": str(args.val_assignment),
            "epochs": int(model_args.epochs),
            "batch_size": int(model_args.batch_size),
            "lr": float(model_args.lr),
            "train_seed_base": int(args.train_seed_base),
            "comparison": "self_only_from_scratch_validation_loss_minus_full_checkpoint_validation_loss",
            "pred_score": "loss_osfi",
        },
        "summary": summary,
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md = args.output_md or output_json.with_suffix(".md")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


def resolve_target_groups(
    *,
    explicit: Sequence[int] | None,
    dataset: str,
    seed: int,
    group_indices: Sequence[Sequence[int]],
    eval_group_indices: Sequence[Sequence[int]],
    loss_osfi_dir: Path,
    selection: str,
    num_targets: int,
    min_group_size: int,
    fallback_min_group_size: int,
    min_eval_group_size: int,
) -> tuple[list[int], int]:
    num_groups = len(group_indices)
    if explicit:
        groups = sorted({int(value) for value in explicit})
        bad = [value for value in groups if value < 0 or value >= num_groups]
        if bad:
            raise ValueError(f"target group(s) out of range 0..{num_groups - 1}: {bad}")
        filtered = [
            value
            for value in groups
            if len(group_indices[value]) >= int(fallback_min_group_size)
            and len(eval_group_indices[value]) >= int(min_eval_group_size)
        ]
        dropped = sorted(set(groups) - set(filtered))
        if dropped:
            print(
                f"Dropped explicit target groups below train min size {fallback_min_group_size} "
                f"or eval min size {min_eval_group_size}: {dropped}",
                flush=True,
            )
        return filtered, int(fallback_min_group_size)

    preferred_candidates = [
        group
        for group, indices in enumerate(group_indices)
        if len(indices) >= int(min_group_size) and len(eval_group_indices[group]) >= int(min_eval_group_size)
    ]
    fallback_candidates = [
        group
        for group, indices in enumerate(group_indices)
        if len(indices) >= int(fallback_min_group_size) and len(eval_group_indices[group]) >= int(min_eval_group_size)
    ]
    if len(preferred_candidates) >= int(num_targets):
        candidates = preferred_candidates
        effective_min_group_size = int(min_group_size)
    else:
        candidates = fallback_candidates
        effective_min_group_size = int(fallback_min_group_size)
    if not candidates:
        print(
            f"No target groups satisfy min size {min_group_size} or fallback {fallback_min_group_size}; "
            f"also requiring eval size {min_eval_group_size}. Writing an empty result.",
            flush=True,
        )
        return [], effective_min_group_size

    if selection == "first":
        return candidates[: min(int(num_targets), len(candidates))], effective_min_group_size
    lookup = load_loss_osfi_lookup(loss_osfi_dir, dataset=dataset, seed=seed)
    if not lookup:
        return candidates[: min(int(num_targets), len(candidates))], effective_min_group_size
    ordered = sorted(
        [group for group in candidates if group in lookup],
        key=lambda group: float(lookup[group].get("loss_osfi", 0.0)),
    )
    if not ordered:
        return candidates[: min(int(num_targets), len(candidates))], effective_min_group_size
    if int(num_targets) <= 1:
        return [int(ordered[len(ordered) // 2])], effective_min_group_size
    positions = np.linspace(0, len(ordered) - 1, int(num_targets))
    selected = []
    for pos in positions:
        selected.append(int(ordered[int(round(float(pos)))]))
    return sorted(dict.fromkeys(selected)), effective_min_group_size


def load_loss_osfi_lookup(loss_osfi_dir: Path, *, dataset: str, seed: int) -> dict[int, dict[str, Any]]:
    path = loss_osfi_dir / f"{dataset}_seed{seed}_loss_osfi_cuda1.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["target_group"]): dict(row) for row in payload.get("kmeans", {}).get("target_records", [])}


def resolve_eval_group_indices(
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    eval_split: str,
) -> list[list[int]]:
    if eval_split == "train":
        return [[int(index) for index in indices] for indices in group_indices]

    train_features = normalized_joint_features(data["train_x"], data["train_y"])
    eval_features = normalized_joint_features(data[f"{eval_split}_x"], data[f"{eval_split}_y"])
    centroids = []
    for indices in group_indices:
        index = torch.as_tensor(indices, dtype=torch.long)
        centroids.append(train_features[index].mean(dim=0))
    centroid_tensor = torch.stack(centroids, dim=0)
    assignments = torch.cdist(eval_features.float(), centroid_tensor.float()).argmin(dim=1).cpu().numpy()
    assigned: list[list[int]] = [[] for _ in group_indices]
    for eval_index, group_id in enumerate(assignments.tolist()):
        assigned[int(group_id)].append(int(eval_index))
    return assigned


def normalized_joint_features(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = torch.nn.functional.normalize(x.float(), dim=1)
    y_norm = torch.nn.functional.normalize(y.float(), dim=1)
    return torch.cat([x_norm, y_norm], dim=1).cpu()


def train_self_only_model(
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    seed: int,
) -> tuple[torch.nn.Module, list[float]]:
    set_seed(int(seed))
    index = torch.as_tensor(indices, dtype=torch.long)
    loader = make_loader(
        data["train_x"][index],
        data["train_y"][index],
        data["train_labels"][index],
        int(model_args.batch_size),
        True,
    )
    model = build_model(model_type, dim_x, dim_y, num_classes, model_args).to(model_args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_args.lr), weight_decay=1e-4)
    epoch_losses: list[float] = []
    for epoch in range(int(model_args.epochs)):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for bx, by, bl in loader:
            bx = bx.to(model_args.device)
            by = by.to(model_args.device)
            bl = bl.to(model_args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
            loss.backward()
            optimizer.step()
            batch_size = int(bl.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        epoch_loss = total_loss / max(total_samples, 1)
        epoch_losses.append(epoch_loss)
        print(f"    epoch {epoch + 1}/{int(model_args.epochs)}: loss={epoch_loss:.6f}", flush=True)
    model.eval()
    return model, epoch_losses


def group_loss_value(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    split: str,
    task: str,
    device: str,
) -> float:
    index = torch.as_tensor(indices, dtype=torch.long)
    bx = data[f"{split}_x"][index].to(device)
    by = data[f"{split}_y"][index].to(device)
    bl = data[f"{split}_labels"][index].to(device)
    model.eval()
    with torch.inference_mode():
        loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
    return float(loss.detach().cpu().item())


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    task: str,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_type": model_type,
            "model_config": {
                "dim_x": int(dim_x),
                "dim_y": int(dim_y),
                "num_classes": int(num_classes),
                "hidden_dim": int(model_args.hidden_dim),
                "proj_dim": int(model_args.proj_dim),
                "num_heads": int(model_args.num_heads),
                "num_layers": int(model_args.num_layers),
                "dropout": float(model_args.dropout),
            },
            "task": task,
            "seed": int(seed),
        },
        path,
    )


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    actual = np.asarray([float(row["actual_loss_gap"]) for row in records], dtype=np.float64)
    pred = np.asarray(
        [float(row["pred_score"]) if row.get("pred_score") is not None else np.nan for row in records],
        dtype=np.float64,
    )
    mask = np.isfinite(pred)
    return {
        "num_targets": int(len(records)),
        "mean_actual_loss_gap": float(np.mean(actual)) if actual.size else float("nan"),
        "actual_positive_fraction": float(np.mean(actual > 0.0)) if actual.size else float("nan"),
        "mean_loss_osfi": float(np.mean(pred[mask])) if np.any(mask) else float("nan"),
        "loss_osfi_actual_sign_agreement": float(np.mean((pred[mask] > 0.0) == (actual[mask] > 0.0))) if np.any(mask) else float("nan"),
        "loss_osfi_actual_spearman": spearmanr(pred[mask], actual[mask]) if np.sum(mask) >= 2 else float("nan"),
        "loss_osfi_actual_pearson": pearsonr(pred[mask], actual[mask]) if np.sum(mask) >= 2 else float("nan"),
    }


def spearmanr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    from kernel_surrogate.kernel_attribution import spearmanr as _spearmanr

    return float(_spearmanr(np.asarray(lhs, dtype=np.float64), np.asarray(rhs, dtype=np.float64)))


def pearsonr(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    x = np.asarray(list(lhs), dtype=np.float64)
    y = np.asarray(list(rhs), dtype=np.float64)
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def to_markdown(output: dict[str, Any]) -> str:
    cfg = output["config"]
    summary = output["summary"]
    lines = [
        "# Sampled Actual Loss Gap",
        "",
        f"- dataset: `{cfg['dataset']}`, seed: `{cfg['seed']}`, device: `{cfg['device']}`",
        f"- target groups: `{cfg['target_groups']}`, selection: `{cfg['selection']}`",
        f"- min group size: `{cfg['effective_min_group_size']}` (requested `{cfg['min_group_size']}`, fallback `{cfg['fallback_min_group_size']}`)",
        f"- eval split: `{cfg['eval_split']}`, val assignment: `{cfg['val_assignment']}`",
        f"- epochs: `{cfg['epochs']}`, batch size: `{cfg['batch_size']}`, lr: `{cfg['lr']}`",
        f"- comparison: `{cfg['comparison']}`",
        f"- pred score: `{cfg['pred_score']}`",
        "",
        "## Summary",
        "",
        f"- mean actual loss gap: `{summary['mean_actual_loss_gap']:.6g}`",
        f"- actual positive fraction: `{summary['actual_positive_fraction']:.3f}`",
        f"- L-OSFI/actual sign agreement: `{summary['loss_osfi_actual_sign_agreement']:.3f}`",
        f"- L-OSFI/actual Spearman: `{summary['loss_osfi_actual_spearman']:.3f}`",
        "",
        "## Records",
        "",
        "| group | train size | eval size | pred gap | L-OSFI | full val loss | self val loss | actual val gap | sign agree |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output["records"]:
        pred_gap = row.get("pred_loss_gap")
        loss_osfi = row.get("loss_osfi")
        lines.append(
            f"| {row['target_group']} | {row['train_group_size']} | {row['eval_group_size']} | "
            f"{float(pred_gap):.6g} | {float(loss_osfi):.6g} | "
            f"{row['full_loss']:.6g} | {row['self_loss']:.6g} | "
            f"{row['actual_loss_gap']:.6g} | {row['sign_agree_pred_score']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
