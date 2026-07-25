"""PID-matched heldout-group fine-tuning test for real_motivating_example.md.

Protocol:
1. Load an existing kmeans group partition from kernel_surrogate LDS artifacts.
2. Train a base checkpoint on a 50% subset of kmeans groups.
3. Compute PID profiles for the heldout groups with pid_estimator.lsmi.
4. Match heldout source/target groups by small PID-profile distance.
5. Fine-tune each matched source group from the same base checkpoint and report
   FIC(source, target), target-group loss/tau change, and full test loss.

Example:
    conda run -n vlm python -m fisher_geometry.pid_matched_group_finetune_test \
        --dataset humor --seed 0 --device cuda:0 --top-pairs 20
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fisher_geometry.data import load_dataset, make_loader
from fisher_geometry.experiment import build_model, set_seed
from fisher_geometry.fisher_metrics import compute_loss
from fisher_geometry.real_motivating_example_experiment import (
    clone_model_from_base,
    compute_group_fic,
    compute_group_gradients,
    estimate_pid_profiles,
    evaluate_groups,
    fine_tune_source,
    pearsonr,
    spearmanr,
)
from fisher_geometry.verify_actual_loss_gap_sampled import resolve_eval_group_indices
from fisher_geometry.verify_humor_fisher_gpt_metrics import (
    load_model,
    resolve_requested_device,
    select_named_parameters,
)
from kernel_surrogate.real_data_cmis import make_model_args


ROOT = REPO_ROOT
DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "kmeans_grouped25_subset100_data50_mod50"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "src"
    / "fisher_geometry"
    / "verification_results"
    / "real_motivating_example"
    / "pid_matched_group_finetune"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 50%-group base ckpt, PID-match heldout groups, then fine-tune and report FIC/test loss."
    )
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--base-groups", nargs="*", type=int, default=None)
    parser.add_argument(
        "--base-run-index",
        type=int,
        default=0,
        help="Use this LDS run's group_ids as the 50%% base subset when --base-groups is omitted.",
    )
    parser.add_argument("--base-group-fraction", type=float, default=0.5)
    parser.add_argument("--base-group-seed", type=int, default=31000)
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--ft-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=71000)
    parser.add_argument("--ft-seed-base", type=int, default=72000)
    parser.add_argument("--force-base-train", action="store_true")
    parser.add_argument("--force-finetune", action="store_true")
    parser.add_argument(
        "--finetune-train-data",
        choices=["base_plus_source", "source_only"],
        default="base_plus_source",
        help="Training data used for each source-group fine-tune from the base checkpoint.",
    )
    parser.add_argument("--gradient-params", choices=["last_layer", "fusion", "all"], default="last_layer")
    parser.add_argument("--fic-damping", type=float, default=1e-2)
    parser.add_argument("--pid-method", choices=["lsmi"], default="lsmi")
    parser.add_argument("--pid-profile-dim", type=int, default=32)
    parser.add_argument("--pid-hidden-dim", type=int, default=32)
    parser.add_argument("--pid-batch-size", type=int, default=64)
    parser.add_argument("--pid-discriminator-epochs", type=int, default=3)
    parser.add_argument("--pid-entropy-epochs", type=int, default=3)
    parser.add_argument("--pid-seed", type=int, default=44000)
    parser.add_argument("--min-heldout-group-size", type=int, default=8)
    parser.add_argument("--neighbors-per-target", type=int, default=1)
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument("--target-eval-split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    device = resolve_requested_device(str(args.device))
    payload_path = args.result_dir / f"{args.dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
    group_partition = dict(payload["group_partition"])
    if str(group_partition.get("group_method", "")) != "kmeans":
        raise ValueError(f"expected kmeans group payload, got {group_partition.get('group_method')!r}")

    data = load_dataset(
        str(args.dataset),
        int(args.seed),
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=str(args.feature_device or config.get("feature_device", device)),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    model_args = make_model_args(config, device=str(device))
    if args.base_epochs is not None:
        model_args.epochs = int(args.base_epochs)
    if args.lr is not None:
        model_args.lr = float(args.lr)
    if args.batch_size is not None:
        model_args.batch_size = int(args.batch_size)
    if args.eval_batch_size is not None:
        model_args.eval_batch_size = int(args.eval_batch_size)
    elif args.batch_size is not None:
        model_args.eval_batch_size = int(args.batch_size)

    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    task = str(data["task"])
    model_type = str(config.get("model", "cross_attention"))
    full_eval_split = "test" if "test_x" in data else "val"
    group_indices = [[int(index) for index in group] for group in group_partition["group_indices"]]
    num_groups = len(group_indices)
    base_groups = resolve_base_groups(
        explicit=args.base_groups,
        payload=payload,
        run_index=int(args.base_run_index),
        num_groups=num_groups,
        fraction=float(args.base_group_fraction),
        seed=int(args.base_group_seed),
    )
    heldout_groups = [
        group
        for group in range(num_groups)
        if group not in set(base_groups) and len(group_indices[group]) >= int(args.min_heldout_group_size)
    ]
    if len(heldout_groups) < 2:
        raise ValueError(f"need at least 2 eligible heldout groups, got {heldout_groups}")

    output_json = args.output_json or (
        args.output_dir / f"{args.dataset}_seed{args.seed}_pid_matched_group_finetune.json"
    )
    output_md = args.output_md or output_json.with_suffix(".md")
    checkpoint_dir = output_json.parent / "checkpoints" / str(args.dataset)
    base_checkpoint = checkpoint_dir / base_checkpoint_name(args, base_groups)

    print(
        f"[{args.dataset}] base groups={base_groups} heldout={heldout_groups} "
        f"epochs={int(model_args.epochs)} device={device}",
        flush=True,
    )
    base_model, base_epoch_losses = load_or_train_base_model(
        checkpoint_path=base_checkpoint,
        force=bool(args.force_base_train),
        data=data,
        group_indices=group_indices,
        base_groups=base_groups,
        task=task,
        model_type=model_type,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        seed=int(args.train_seed),
    )
    base_model.to(device)
    base_model.eval()

    base_test_metrics = evaluate_classifier(
        base_model,
        make_eval_loader(data, split=full_eval_split, batch_size=int(model_args.eval_batch_size)),
        task=task,
        device=str(device),
    )

    pid_profile_groups = sorted({int(group) for group in list(base_groups) + list(heldout_groups)})
    print(f"[{args.dataset}] computing PID profiles on base + heldout groups...", flush=True)
    pid_profiles = estimate_pid_profiles(
        data=data,
        group_indices=group_indices,
        groups=pid_profile_groups,
        method=str(args.pid_method),
        task=task,
        device=str(device),
        profile_dim=int(args.pid_profile_dim),
        hidden_dim=int(args.pid_hidden_dim),
        batch_size=int(args.pid_batch_size),
        discriminator_epochs=int(args.pid_discriminator_epochs),
        entropy_epochs=int(args.pid_entropy_epochs),
        seed=int(args.pid_seed),
    )
    profiled_base_groups = [group for group in base_groups if group in pid_profiles]
    profiled_source_groups = [group for group in heldout_groups if group in pid_profiles]
    if not profiled_base_groups or not profiled_source_groups:
        raise ValueError(
            "PID profiling left no usable base/source groups: "
            f"base={profiled_base_groups}, source={profiled_source_groups}"
        )

    print(f"[{args.dataset}] computing base-checkpoint FIC over kmeans groups...", flush=True)
    named_params = select_named_parameters(base_model, mode=str(args.gradient_params))
    param_names = [name for name, _ in named_params]
    gradients = compute_group_gradients(
        base_model,
        data=data,
        group_indices=group_indices,
        task=task,
        device=str(device),
        param_names=param_names,
    )
    fic_payload = compute_group_fic(
        gradients,
        group_indices=group_indices,
        damping=float(args.fic_damping),
    )

    pair_specs = select_pid_matched_base_source_pairs(
        base_groups=profiled_base_groups,
        source_groups=profiled_source_groups,
        profiles=pid_profiles,
        neighbors_per_source=int(args.neighbors_per_target),
        top_pairs=int(args.top_pairs),
    )
    finetune_sources = sorted({int(row["source_group"]) for row in pair_specs})
    print(
        f"[{args.dataset}] selected {len(pair_specs)} PID-nearest base-source pairs; "
        f"fine-tuning {len(finetune_sources)} unique source groups",
        flush=True,
    )
    source_results = finetune_sources_from_base(
        base_model,
        sources=finetune_sources,
        checkpoint_dir=checkpoint_dir,
        force=bool(args.force_finetune),
        data=data,
        group_indices=group_indices,
        task=task,
        model_type=model_type,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        base_checkpoint=base_checkpoint,
        base_test_loss=float(base_test_metrics["loss"]),
        base_groups=base_groups,
        full_eval_split=full_eval_split,
        target_eval_groups=[],
        target_groups=[],
        target_eval_split=str(args.target_eval_split),
        finetune_train_data=str(args.finetune_train_data),
        ft_epochs=int(args.ft_epochs),
        ft_seed_base=int(args.ft_seed_base),
        device=str(device),
    )

    records = []
    for spec in pair_specs:
        base_group = int(spec["base_group"])
        source = int(spec["source_group"])
        ft = source_results[source]
        k_value = float(fic_payload["K"][base_group, source])
        delta_eval_loss = float(base_test_metrics["loss"] - float(ft["test_metrics"]["loss"]))
        records.append(
            {
                "base_group": base_group,
                "source_group": source,
                "base_group_size": int(len(group_indices[base_group])),
                "source_size": int(len(group_indices[source])),
                "pid_distance": float(spec["pid_distance"]),
                "pid_similarity": float(1.0 / (1.0 + float(spec["pid_distance"]))),
                "base_pid_profile": pid_profiles[base_group],
                "source_pid_profile": pid_profiles[source],
                "K_base_source": k_value,
                "K_plus_base_source": float(max(k_value, 0.0)),
                "FIC_base_source": float(fic_payload["FIC"][base_group, source]),
                "base_test_loss": float(base_test_metrics["loss"]),
                "finetuned_test_loss": float(ft["test_metrics"]["loss"]),
                "eval_loss_ratio_ft_over_base": safe_ratio(float(ft["test_metrics"]["loss"]), float(base_test_metrics["loss"])),
                "delta_test_loss_base_minus_ft": delta_eval_loss,
                "delta_eval_loss_ratio_base_minus_ft": safe_ratio(delta_eval_loss, float(base_test_metrics["loss"])),
                "finetuned_test_acc": float(ft["test_metrics"]["acc"]),
            }
        )

    output = {
        "config": {
            "dataset": str(args.dataset),
            "seed": int(args.seed),
            "payload_path": str(payload_path),
            "result_dir": str(args.result_dir),
            "group_method": "kmeans",
            "num_groups": int(num_groups),
            "base_groups": [int(group) for group in base_groups],
            "heldout_groups": [int(group) for group in heldout_groups],
            "profiled_base_groups": [int(group) for group in profiled_base_groups],
            "profiled_source_groups": [int(group) for group in profiled_source_groups],
            "base_checkpoint": str(base_checkpoint),
            "device": str(device),
            "feature_device": str(args.feature_device or config.get("feature_device", device)),
            "model": model_type,
            "base_epochs": int(model_args.epochs),
            "ft_epochs": int(args.ft_epochs),
            "lr": float(model_args.lr),
            "batch_size": int(model_args.batch_size),
            "eval_batch_size": int(model_args.eval_batch_size),
            "gradient_params": str(args.gradient_params),
            "num_gradient_parameters": int(sum(param.numel() for _, param in named_params)),
            "fic_damping": float(args.fic_damping),
            "pid_method": str(args.pid_method),
            "pid_profile_dim": int(args.pid_profile_dim),
            "pid_hidden_dim": int(args.pid_hidden_dim),
            "target_eval_split": str(args.target_eval_split),
            "full_eval_split": full_eval_split,
            "finetune_train_data": str(args.finetune_train_data),
            "pair_unit": "base_group_to_source_group",
            "fic_pair_definition": "FIC(base_group, source_group), where base_group was used to train the base checkpoint",
            "neighbors_per_target": int(args.neighbors_per_target),
            "top_pairs": int(args.top_pairs),
        },
        "base_training": {
            "epoch_losses": [float(loss) for loss in base_epoch_losses],
            "test_metrics": base_test_metrics,
        },
        "summary": summarize_records(records),
        "source_finetunes": source_results,
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


def resolve_base_groups(
    *,
    explicit: Sequence[int] | None,
    payload: dict[str, Any],
    run_index: int,
    num_groups: int,
    fraction: float,
    seed: int,
) -> list[int]:
    if explicit:
        groups = sorted({int(group) for group in explicit})
    else:
        runs = payload.get("runs", [])
        if runs and 0 <= int(run_index) < len(runs) and runs[int(run_index)].get("group_ids"):
            groups = sorted({int(group) for group in runs[int(run_index)]["group_ids"]})
        else:
            rng = np.random.default_rng(int(seed))
            count = max(1, min(num_groups - 1, int(math.ceil(num_groups * float(fraction)))))
            groups = sorted(rng.choice(num_groups, size=count, replace=False).astype(int).tolist())
    bad = [group for group in groups if group < 0 or group >= num_groups]
    if bad:
        raise ValueError(f"base group id(s) out of range 0..{num_groups - 1}: {bad}")
    if len(groups) >= num_groups:
        raise ValueError("base groups must leave at least one heldout group")
    return groups


def base_checkpoint_name(args: argparse.Namespace, base_groups: Sequence[int]) -> str:
    token = "-".join(str(int(group)) for group in base_groups)
    if len(token) > 120:
        import hashlib

        token = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    epoch_token = args.base_epochs if args.base_epochs is not None else "cfg"
    return (
        f"{args.dataset}_seed{int(args.seed)}_kmeans_basegroups_{token}_"
        f"epochs{epoch_token}_trainseed{int(args.train_seed)}.pt"
    )


def load_or_train_base_model(
    *,
    checkpoint_path: Path,
    force: bool,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    base_groups: Sequence[int],
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: Any,
    seed: int,
) -> tuple[torch.nn.Module, list[float]]:
    if checkpoint_path.exists() and not force:
        print(f"Loading base checkpoint: {checkpoint_path}", flush=True)
        model = load_model(
            checkpoint_path,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=str(model_args.device),
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return model, [float(loss) for loss in payload.get("epoch_losses", [])]

    print(f"Training base checkpoint on {len(base_groups)} kmeans groups: {checkpoint_path}", flush=True)
    set_seed(int(seed))
    indices = torch.as_tensor(flatten_group_indices(group_indices, base_groups), dtype=torch.long)
    loader = make_loader(
        data["train_x"][indices],
        data["train_y"][indices],
        data["train_labels"][indices],
        int(model_args.batch_size),
        True,
    )
    model = build_model(model_type, dim_x, dim_y, num_classes, model_args).to(model_args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_args.lr), weight_decay=1e-4)
    epoch_losses: list[float] = []
    for epoch in range(int(model_args.epochs)):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for bx, by, bl in loader:
            bx = bx.to(model_args.device)
            by = by.to(model_args.device)
            bl = bl.to(model_args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * int(bl.shape[0])
            total_samples += int(bl.shape[0])
        epoch_loss = total_loss / max(total_samples, 1)
        epoch_losses.append(float(epoch_loss))
        print(f"  base epoch {epoch + 1}/{int(model_args.epochs)}: loss={epoch_loss:.6f}", flush=True)
    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_type": model_type,
            "model_config": {
                "dim_x": dim_x,
                "dim_y": dim_y,
                "num_classes": num_classes,
                "hidden_dim": model_args.hidden_dim,
                "proj_dim": model_args.proj_dim,
                "num_heads": model_args.num_heads,
                "num_layers": model_args.num_layers,
                "dropout": model_args.dropout,
            },
            "task": task,
            "base_groups": [int(group) for group in base_groups],
            "data_indices": [int(index) for index in indices.tolist()],
            "epoch_losses": [float(loss) for loss in epoch_losses],
            "seed": int(seed),
        },
        checkpoint_path,
    )
    return model, epoch_losses


def flatten_group_indices(group_indices: Sequence[Sequence[int]], groups: Sequence[int]) -> list[int]:
    indices: list[int] = []
    for group in groups:
        indices.extend(int(index) for index in group_indices[int(group)])
    return indices


def make_eval_loader(data: dict[str, Any], *, split: str, batch_size: int) -> DataLoader:
    if f"{split}_x" not in data:
        if split == "test" and "val_x" in data:
            split = "val"
        else:
            raise KeyError(f"missing split {split!r}; available x splits={[key[:-2] for key in data if key.endswith('_x')]}")
    return make_loader(
        data[f"{split}_x"],
        data[f"{split}_y"],
        data[f"{split}_labels"],
        int(batch_size),
        False,
    )


def evaluate_classifier(
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
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_samples += batch_size
            if task == "multiclass":
                metric_total += float((logits.argmax(dim=1) == bl).sum().item())
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
    return {
        "loss": float(total_loss / max(total_samples, 1)),
        "acc": float(metric_total / max(total_samples, 1)),
        "metric_name": "sample_f1" if task == "multilabel" else "accuracy",
    }


def pid_vector(profile: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(profile.get("norm_R", 0.0)),
            float(profile.get("norm_U1", 0.0)),
            float(profile.get("norm_U2", 0.0)),
            float(profile.get("norm_S", 0.0)),
        ],
        dtype=np.float64,
    )


def select_pid_matched_pairs(
    *,
    groups: Sequence[int],
    profiles: dict[int, dict[str, Any]],
    neighbors_per_target: int,
    top_pairs: int,
) -> list[dict[str, float | int]]:
    rows = []
    vectors = {int(group): pid_vector(profiles[int(group)]) for group in groups}
    for target in groups:
        distances = []
        for source in groups:
            if int(source) == int(target):
                continue
            dist = float(np.linalg.norm(vectors[int(source)] - vectors[int(target)]))
            distances.append((dist, int(source), int(target)))
        for rank, (dist, source, target) in enumerate(sorted(distances)[: max(1, int(neighbors_per_target))]):
            rows.append(
                {
                    "source_group": int(source),
                    "target_group": int(target),
                    "pid_distance": float(dist),
                    "neighbor_rank": int(rank + 1),
                }
            )
    rows.sort(key=lambda row: (float(row["pid_distance"]), int(row["target_group"]), int(row["source_group"])))
    return rows[: max(1, int(top_pairs))]


def select_pid_matched_base_source_pairs(
    *,
    base_groups: Sequence[int],
    source_groups: Sequence[int],
    profiles: dict[int, dict[str, Any]],
    neighbors_per_source: int,
    top_pairs: int,
) -> list[dict[str, float | int]]:
    rows = []
    vectors = {int(group): pid_vector(profiles[int(group)]) for group in list(base_groups) + list(source_groups)}
    for source in source_groups:
        distances = []
        for base_group in base_groups:
            dist = float(np.linalg.norm(vectors[int(base_group)] - vectors[int(source)]))
            distances.append((dist, int(base_group), int(source)))
        for rank, (dist, base_group, source_group) in enumerate(
            sorted(distances)[: max(1, int(neighbors_per_source))]
        ):
            rows.append(
                {
                    "base_group": int(base_group),
                    "source_group": int(source_group),
                    "pid_distance": float(dist),
                    "neighbor_rank": int(rank + 1),
                }
            )
    rows.sort(key=lambda row: (float(row["pid_distance"]), int(row["source_group"]), int(row["base_group"])))
    return rows[: max(1, int(top_pairs))]


def finetune_sources_from_base(
    base_model: torch.nn.Module,
    *,
    sources: Sequence[int],
    checkpoint_dir: Path,
    force: bool,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: Any,
    base_checkpoint: Path,
    base_test_loss: float,
    base_groups: Sequence[int],
    full_eval_split: str,
    target_eval_groups: Sequence[Sequence[int]],
    target_groups: Sequence[int],
    target_eval_split: str,
    finetune_train_data: str,
    ft_epochs: int,
    ft_seed_base: int,
    device: str,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    test_loader = make_eval_loader(data, split=full_eval_split, batch_size=int(model_args.eval_batch_size))
    for source in sources:
        ckpt = checkpoint_dir / (
            f"finetune_{finetune_train_data}_from_{base_checkpoint.stem}_"
            f"source_g{int(source):02d}_epochs{int(ft_epochs)}.pt"
        )
        if ckpt.exists() and not force:
            print(f"  loading source finetune checkpoint g{source}: {ckpt}", flush=True)
            model = load_model(
                ckpt,
                model_type=model_type,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                device=device,
            )
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            train_losses = [float(loss) for loss in payload.get("train_losses", [])]
        else:
            print(f"  fine-tuning source group {source}", flush=True)
            model = clone_model_from_base(
                base_model,
                model_type=model_type,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                device=device,
            )
            if finetune_train_data == "base_plus_source":
                train_groups = sorted({int(group) for group in list(base_groups) + [int(source)]})
                train_indices = flatten_group_indices(group_indices, train_groups)
            elif finetune_train_data == "source_only":
                train_groups = [int(source)]
                train_indices = [int(index) for index in group_indices[int(source)]]
            else:
                raise ValueError(f"unsupported finetune_train_data={finetune_train_data!r}")
            train_losses = fine_tune_source(
                model,
                data=data,
                indices=train_indices,
                task=task,
                device=device,
                lr=float(model_args.lr),
                batch_size=int(model_args.batch_size),
                epochs=int(ft_epochs),
                seed=int(ft_seed_base) + int(source),
                shuffle_modality2=False,
                shuffle_seed=0,
            )
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "source_group": int(source),
                    "finetune_train_data": str(finetune_train_data),
                    "train_groups": [int(group) for group in train_groups],
                    "train_indices": [int(index) for index in train_indices],
                    "base_checkpoint": str(base_checkpoint),
                    "train_losses": [float(loss) for loss in train_losses],
                    "ft_epochs": int(ft_epochs),
                },
                ckpt,
            )
        test_metrics = evaluate_classifier(model, test_loader, task=task, device=device)
        tau, loss = evaluate_groups(
            model,
            data=data,
            groups=target_eval_groups,
            target_groups=target_groups,
            split=target_eval_split,
            task=task,
            device=device,
        )
        results[int(source)] = {
            "checkpoint": str(ckpt),
            "source_size": int(len(group_indices[int(source)])),
            "finetune_train_data": str(finetune_train_data),
            "train_losses": [float(loss_value) for loss_value in train_losses],
            "test_metrics": test_metrics,
            "delta_test_loss_base_minus_ft": float(base_test_loss - float(test_metrics["loss"])),
            "target_tau": {str(group): float(value) for group, value in tau.items()},
            "target_loss": {str(group): float(value) for group, value in loss.items()},
        }
    return results


def finite_values(records: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records], dtype=np.float64)


def safe_ratio(numer: float, denom: float, eps: float = 1e-12) -> float:
    if abs(float(denom)) <= float(eps):
        return float("nan")
    return float(numer) / float(denom)


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_pairs": 0}
    pid_similarity = finite_values(records, "pid_similarity")
    pid_distance = finite_values(records, "pid_distance")
    fic_key = "FIC_base_source" if "FIC_base_source" in records[0] else "FIC_source_target"
    k_plus_key = "K_plus_base_source" if "K_plus_base_source" in records[0] else "K_plus_source_target"
    fic = finite_values(records, fic_key)
    k_plus = finite_values(records, k_plus_key)
    delta_test_loss = finite_values(records, "delta_test_loss_base_minus_ft")
    eval_loss_ratio = ensure_ratio_values(
        records,
        numer_key="delta_test_loss_base_minus_ft",
        denom_key="base_test_loss",
        existing_key="delta_eval_loss_ratio_base_minus_ft",
    )
    summary = {
        "num_pairs": int(len(records)),
        "num_unique_sources": int(len({int(row["source_group"]) for row in records})),
        "pair_unit": "base_group_to_source_group" if fic_key == "FIC_base_source" else "source_group_to_target_group",
        "pid_distance_mean": float(np.mean(pid_distance)),
        "pid_distance_median": float(np.median(pid_distance)),
        "pid_distance_max": float(np.max(pid_distance)),
        "mean_pid_distance": float(np.mean(finite_values(records, "pid_distance"))),
        "mean_fic": float(np.mean(fic)),
        "mean_delta_test_loss_base_minus_ft": float(np.mean(delta_test_loss)),
        "mean_delta_eval_loss_ratio_base_minus_ft": float(np.nanmean(eval_loss_ratio)),
        "fic_split_eval_loss_ratio": fic_split_summary(
            records,
            fic_key=fic_key,
            value_key="delta_eval_loss_ratio_base_minus_ft",
            fallback_numer_key="delta_test_loss_base_minus_ft",
            fallback_denom_key="base_test_loss",
        ),
        "pid_similarity_vs_fic_spearman": spearmanr(pid_similarity, fic),
        "pid_similarity_vs_kplus_spearman": spearmanr(pid_similarity, k_plus),
        "fic_vs_delta_test_loss_spearman": spearmanr(fic, delta_test_loss),
        "fic_vs_delta_eval_loss_ratio_spearman": spearmanr(fic, eval_loss_ratio),
    }
    if "delta_target_loss_base_minus_ft" in records[0]:
        delta_target_loss = finite_values(records, "delta_target_loss_base_minus_ft")
        delta_tau = finite_values(records, "delta_target_tau")
        target_loss_ratio = ensure_ratio_values(
            records,
            numer_key="delta_target_loss_base_minus_ft",
            denom_key="base_target_loss",
            existing_key="delta_target_loss_ratio_base_minus_ft",
        )
        summary.update(
            {
                "mean_delta_target_loss_base_minus_ft": float(np.mean(delta_target_loss)),
                "mean_delta_target_loss_ratio_base_minus_ft": float(np.nanmean(target_loss_ratio)),
                "fic_split_target_loss_ratio": fic_split_summary(
                    records,
                    fic_key=fic_key,
                    value_key="delta_target_loss_ratio_base_minus_ft",
                    fallback_numer_key="delta_target_loss_base_minus_ft",
                    fallback_denom_key="base_target_loss",
                ),
                "fic_vs_delta_target_loss_spearman": spearmanr(fic, delta_target_loss),
                "fic_vs_delta_target_loss_pearson": pearsonr(fic, delta_target_loss),
                "fic_vs_delta_tau_spearman": spearmanr(fic, delta_tau),
                "fic_vs_delta_target_loss_ratio_spearman": spearmanr(fic, target_loss_ratio),
            }
        )
    return summary


def ensure_ratio_values(
    records: Sequence[dict[str, Any]],
    *,
    numer_key: str,
    denom_key: str,
    existing_key: str,
) -> np.ndarray:
    values = []
    for row in records:
        if existing_key in row:
            values.append(float(row[existing_key]))
        else:
            values.append(safe_ratio(float(row[numer_key]), float(row[denom_key])))
    return np.asarray(values, dtype=np.float64)


def fic_split_summary(
    records: Sequence[dict[str, Any]],
    *,
    fic_key: str,
    value_key: str,
    fallback_numer_key: str,
    fallback_denom_key: str,
    fraction: float = 1.0 / 3.0,
) -> dict[str, Any]:
    rows = []
    for row in records:
        if value_key in row:
            value = float(row[value_key])
        else:
            value = safe_ratio(float(row[fallback_numer_key]), float(row[fallback_denom_key]))
        if not np.isfinite(value):
            continue
        rows.append(
            {
                "fic": float(row[fic_key]),
                "pid_distance": float(row["pid_distance"]),
                "value": value,
            }
        )
    if not rows:
        return {"num_pairs": 0}
    rows = sorted(rows, key=lambda row: row["fic"])
    k = max(1, min(len(rows), int(math.ceil(len(rows) * float(fraction)))))
    low = rows[:k]
    high = rows[-k:]
    low_mean = float(np.mean([row["value"] for row in low]))
    high_mean = float(np.mean([row["value"] for row in high]))
    return {
        "num_pairs": int(len(rows)),
        "split_count": int(k),
        "low_fic_mean": float(np.mean([row["fic"] for row in low])),
        "high_fic_mean": float(np.mean([row["fic"] for row in high])),
        "low_pid_distance_mean": float(np.mean([row["pid_distance"] for row in low])),
        "high_pid_distance_mean": float(np.mean([row["pid_distance"] for row in high])),
        "low_mean": low_mean,
        "high_mean": high_mean,
        "high_minus_low": float(high_mean - low_mean),
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def to_markdown(output: dict[str, Any]) -> str:
    cfg = output["config"]
    summary = output["summary"]
    lines = [
        "# PID-Matched Group Fine-Tuning Test",
        "",
        f"- dataset: `{cfg['dataset']}`, seed: `{cfg['seed']}`, device: `{cfg['device']}`",
        f"- kmeans payload: `{cfg['payload_path']}`",
        f"- base checkpoint: `{cfg['base_checkpoint']}`",
        f"- base groups ({len(cfg['base_groups'])}/{cfg['num_groups']}): `{cfg['base_groups']}`",
        f"- profiled base groups: `{cfg.get('profiled_base_groups', cfg.get('base_groups', []))}`",
        f"- profiled source groups: `{cfg.get('profiled_source_groups', cfg.get('profiled_groups', []))}`",
        f"- finetune train data: `{cfg.get('finetune_train_data', 'base_plus_source')}`",
        f"- pair unit: `{cfg.get('pair_unit', summary.get('pair_unit', 'source_group_to_target_group'))}`",
        f"- PID: `{cfg['pid_method']}`, FIC params: `{cfg['gradient_params']}` ({cfg['num_gradient_parameters']} params)",
        "",
        "## Summary",
        "",
        f"- pairs: `{summary.get('num_pairs', 0)}`, unique finetuned sources: `{summary.get('num_unique_sources', 0)}`",
        f"- base full test loss: `{fmt(output['base_training']['test_metrics']['loss'], 6)}`",
        f"- PID distance median / max: `{fmt(summary.get('pid_distance_median'))}` / `{fmt(summary.get('pid_distance_max'))}`",
        f"- mean FIC: `{fmt(summary.get('mean_fic'))}`",
        f"- mean Delta test loss(base-ft): `{fmt(summary.get('mean_delta_test_loss_base_minus_ft'))}`",
        f"- mean Delta eval loss ratio: `{fmt(summary.get('mean_delta_eval_loss_ratio_base_minus_ft'))}`",
        f"- high-FIC minus low-FIC eval-loss ratio: `{fmt(summary.get('fic_split_eval_loss_ratio', {}).get('high_minus_low'))}`",
        "",
        "## PID-Matched Base-Source Records",
        "",
        "| base group | source group | PID dist | FIC(base,source) | K+ | base eval loss | ft eval loss | Delta eval ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if "mean_delta_target_loss_ratio_base_minus_ft" in summary:
        lines[lines.index(f"- mean Delta eval loss ratio: `{fmt(summary.get('mean_delta_eval_loss_ratio_base_minus_ft'))}`")] += (
            f"\n- mean Delta target loss ratio: `{fmt(summary.get('mean_delta_target_loss_ratio_base_minus_ft'))}`"
        )
    for row in output["records"]:
        eval_ratio = row.get(
            "delta_eval_loss_ratio_base_minus_ft",
            safe_ratio(float(row["delta_test_loss_base_minus_ft"]), float(row["base_test_loss"])),
        )
        if "base_group" in row:
            lines.append(
                f"| {row['base_group']} | {row['source_group']} | "
                f"{fmt(row['pid_distance'])} | {fmt(row['FIC_base_source'])} | {fmt(row['K_plus_base_source'])} | "
                f"{fmt(row['base_test_loss'])} | {fmt(row['finetuned_test_loss'])} | {fmt(eval_ratio)} |"
            )
            continue
        target_ratio = row.get(
            "delta_target_loss_ratio_base_minus_ft",
            safe_ratio(float(row["delta_target_loss_base_minus_ft"]), float(row["base_target_loss"])),
        )
        lines.append(
            f"| {row['source_group']} | {row['target_group']} | "
            f"{fmt(row['pid_distance'])} | {fmt(row['FIC_source_target'])} | {fmt(row['K_plus_source_target'])} | "
            f"{fmt(row['base_target_loss'])} | {fmt(row['finetuned_target_loss'])} | "
            f"{fmt(target_ratio)} | {fmt(row['finetuned_test_loss'])} | "
            f"{fmt(eval_ratio)} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
