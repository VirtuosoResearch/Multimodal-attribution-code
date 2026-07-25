"""Real-dataset validation for Fisher cross-sample modal interaction.

Implements Experiment 1 from real_motivating_example.md on existing
group/checkpoint artifacts under src/kernel_surrogate/subset_checkpoints.

For each selected source group, the script fine-tunes from the same full-data
checkpoint on either the real source group or a label-preserving modality-2
shuffle of that source group. It then checks whether group-level FIC predicts
the target groups whose multimodal contrast changes the most.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fisher_geometry.data import load_dataset, make_loader
from fisher_geometry.experiment import build_model, set_seed
from fisher_geometry.fisher_metrics import compute_loss
from fisher_geometry.verify_actual_loss_gap_sampled import resolve_eval_group_indices
from fisher_geometry.verify_humor_fisher_gpt_metrics import (
    DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT,
    DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT,
    load_model,
    make_model_args,
    resolve_full_checkpoint,
    resolve_requested_device,
    select_named_parameters,
)
from pid_estimator.lsmi import LSMIConfig, estimate_lsmi


ROOT = REPO_ROOT
DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "kmeans_grouped25_subset100_data80_mod80"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "src"
    / "fisher_geometry"
    / "verification_results"
    / "real_motivating_example"
)
DEFAULT_FULL_CHECKPOINT_DIR = (
    ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-dataset source-group interaction-ablation validation for FIC."
    )
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument("--fallback-full-checkpoint-dir", type=Path, default=DEFAULT_FULL_CHECKPOINT_DIR)
    parser.add_argument(
        "--full-checkpoint-roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT, DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--gradient-params", choices=["last_layer", "fusion", "all"], default="last_layer")
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--source-groups", nargs="*", type=int, default=None)
    parser.add_argument("--target-groups", nargs="*", type=int, default=None)
    parser.add_argument("--num-source-groups", type=int, default=5)
    parser.add_argument("--num-target-groups", type=int, default=5)
    parser.add_argument(
        "--source-selection",
        choices=["largest", "high_content", "all"],
        default="largest",
        help="How to choose source groups when --source-groups is not provided.",
    )
    parser.add_argument("--content-buckets", type=int, default=4)
    parser.add_argument("--pid-profile-method", choices=["none", "lsmi"], default="none")
    parser.add_argument("--pid-profile-dim", type=int, default=32)
    parser.add_argument("--pid-hidden-dim", type=int, default=32)
    parser.add_argument("--pid-batch-size", type=int, default=64)
    parser.add_argument("--pid-discriminator-epochs", type=int, default=3)
    parser.add_argument("--pid-entropy-epochs", type=int, default=3)
    parser.add_argument("--pid-seed", type=int, default=44000)
    parser.add_argument("--min-group-size", type=int, default=50)
    parser.add_argument("--target-split", choices=["train", "val"], default="train")
    parser.add_argument("--ft-epochs", type=int, default=3)
    parser.add_argument("--ft-lr", type=float, default=None)
    parser.add_argument("--ft-batch-size", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=2028)
    parser.add_argument("--train-seed-base", type=int, default=93000)
    parser.add_argument("--lift-k", type=int, default=2)
    args = parser.parse_args()

    device = resolve_requested_device(str(args.device))
    payload_path = args.result_dir / f"{args.dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
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
    if args.ft_lr is not None:
        model_args.lr = float(args.ft_lr)
    if args.ft_batch_size is not None:
        model_args.batch_size = int(args.ft_batch_size)

    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    task = str(data["task"])
    model_type = str(config.get("model", "cross_attention"))
    group_indices = [[int(index) for index in group] for group in payload["group_partition"]["group_indices"]]
    target_eval_indices = (
        resolve_eval_group_indices(data=data, group_indices=group_indices, eval_split="val")
        if str(args.target_split) == "val"
        else group_indices
    )
    target_groups = select_groups(
        explicit=args.target_groups,
        group_indices=target_eval_indices,
        num_groups=int(args.num_target_groups),
        min_group_size=1 if str(args.target_split) == "val" else int(args.min_group_size),
    )
    full_checkpoint = resolve_full_checkpoint(
        args.full_checkpoint,
        roots=args.full_checkpoint_roots,
        dataset=str(args.dataset),
        seed=int(args.seed),
    )

    base_model, full_checkpoint = load_compatible_model(
        full_checkpoint,
        explicit=args.full_checkpoint is not None,
        fallback_dir=args.fallback_full_checkpoint_dir,
        dataset=str(args.dataset),
        seed=int(args.seed),
        model_type=model_type,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        device=str(device),
    )
    all_group_ids = list(range(len(group_indices)))
    train_tau_all, _ = evaluate_groups(
        base_model,
        data=data,
        groups=group_indices,
        target_groups=all_group_ids,
        split="train",
        task=task,
        device=str(device),
    )
    source_content = {int(group): float(train_tau_all[int(group)]) for group in all_group_ids}
    tau_bucket_lookup = make_content_buckets(
        source_content,
        num_buckets=int(args.content_buckets),
        eligible_groups=[
            group
            for group, indices in enumerate(group_indices)
            if len(indices) >= int(args.min_group_size)
        ],
    )
    source_groups = select_groups(
        explicit=args.source_groups,
        group_indices=group_indices,
        num_groups=int(args.num_source_groups),
        min_group_size=int(args.min_group_size),
        mode=str(args.source_selection),
        scores=source_content,
    )
    pid_profiles = estimate_pid_profiles(
        data=data,
        group_indices=group_indices,
        groups=source_groups,
        method=str(args.pid_profile_method),
        task=task,
        device=str(device),
        profile_dim=int(args.pid_profile_dim),
        hidden_dim=int(args.pid_hidden_dim),
        batch_size=int(args.pid_batch_size),
        discriminator_epochs=int(args.pid_discriminator_epochs),
        entropy_epochs=int(args.pid_entropy_epochs),
        seed=int(args.pid_seed),
    )
    pid_bucket_lookup = (
        make_pid_profile_buckets(
            pid_profiles,
            num_buckets=int(args.content_buckets),
            seed=int(args.pid_seed),
        )
        if pid_profiles
        else {}
    )
    conditional_bucket_key = "source_pid_bucket" if pid_bucket_lookup else "source_content_bucket"
    named_params = select_named_parameters(base_model, mode=str(args.gradient_params))
    param_names = [name for name, _ in named_params]
    print(
        f"[{args.dataset}] computing group FIC for {len(group_indices)} groups "
        f"with {sum(param.numel() for _, param in named_params)} parameters",
        flush=True,
    )
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
        damping=float(args.damping),
    )
    source_delta = compute_source_intervention_gradients(
        base_model,
        data=data,
        group_indices=group_indices,
        source_groups=source_groups,
        task=task,
        device=str(device),
        param_names=param_names,
        shuffle_seed=int(args.shuffle_seed),
    )
    intervention_predictors = compute_intervention_predictors(
        gradients,
        source_delta,
        fic_payload=fic_payload,
        source_groups=source_groups,
        target_groups=target_groups,
    )
    base_tau, base_loss = evaluate_groups(
        base_model,
        data=data,
        groups=target_eval_indices,
        target_groups=target_groups,
        split=str(args.target_split),
        task=task,
        device=str(device),
    )

    records = []
    source_results = {}
    for source_group in source_groups:
        print(f"[{args.dataset}] source group {source_group}: real vs shuffled fine-tune", flush=True)
        real_model = clone_model_from_base(
            base_model,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=str(device),
        )
        abl_model = clone_model_from_base(
            base_model,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=str(device),
        )
        real_losses = fine_tune_source(
            real_model,
            data=data,
            indices=group_indices[source_group],
            task=task,
            device=str(device),
            lr=float(model_args.lr),
            batch_size=int(model_args.batch_size),
            epochs=int(args.ft_epochs),
            seed=int(args.train_seed_base) + int(source_group),
            shuffle_modality2=False,
            shuffle_seed=int(args.shuffle_seed) + int(source_group),
        )
        abl_losses = fine_tune_source(
            abl_model,
            data=data,
            indices=group_indices[source_group],
            task=task,
            device=str(device),
            lr=float(model_args.lr),
            batch_size=int(model_args.batch_size),
            epochs=int(args.ft_epochs),
            seed=int(args.train_seed_base) + 1000 + int(source_group),
            shuffle_modality2=True,
            shuffle_seed=int(args.shuffle_seed) + int(source_group),
        )
        real_tau, real_loss = evaluate_groups(
            real_model,
            data=data,
            groups=target_eval_indices,
            target_groups=target_groups,
            split=str(args.target_split),
            task=task,
            device=str(device),
        )
        abl_tau, abl_loss = evaluate_groups(
            abl_model,
            data=data,
            groups=target_eval_indices,
            target_groups=target_groups,
            split=str(args.target_split),
            task=task,
            device=str(device),
        )
        source_results[int(source_group)] = {
            "real_train_losses": real_losses,
            "ablated_train_losses": abl_losses,
        }
        for target_group in target_groups:
            key = (int(source_group), int(target_group))
            k_value = float(fic_payload["K"][source_group, target_group])
            records.append(
                {
                    "source_group": int(source_group),
                    "target_group": int(target_group),
                    "source_size": int(len(group_indices[source_group])),
                    "target_size": int(len(target_eval_indices[target_group])),
                    "source_interaction_content_tau": float(source_content[int(source_group)]),
                    "source_content_bucket": int(tau_bucket_lookup.get(int(source_group), -1)),
                    "source_pid_bucket": int(pid_bucket_lookup.get(int(source_group), -1)),
                    "source_pid_profile": pid_profiles.get(int(source_group), {}),
                    "K_baseline_c_source_c_target": k_value,
                    "K_plus_baseline_c_source_c_target": float(max(k_value, 0.0)),
                    "FIC_baseline_c_source_c_target": float(fic_payload["FIC"][source_group, target_group]),
                    "P_tau_h": float(intervention_predictors["P_tau_h"][key]),
                    "P_tau_h_plus": float(max(intervention_predictors["P_tau_h"][key], 0.0)),
                    "P_tau_c": float(intervention_predictors["P_tau_c"][key]),
                    "P_tau_c_plus": float(max(intervention_predictors["P_tau_c"][key], 0.0)),
                    "P_loss_h": float(intervention_predictors["P_loss_h"][key]),
                    "P_loss_h_plus": float(max(intervention_predictors["P_loss_h"][key], 0.0)),
                    "source_energy": float(fic_payload["energy"][source_group]),
                    "target_energy": float(fic_payload["energy"][target_group]),
                    "base_tau": float(base_tau[target_group]),
                    "real_tau": float(real_tau[target_group]),
                    "ablated_tau": float(abl_tau[target_group]),
                    "delta_tau_real_minus_ablated": float(real_tau[target_group] - abl_tau[target_group]),
                    "base_fused_loss": float(base_loss[target_group]),
                    "real_fused_loss": float(real_loss[target_group]),
                    "ablated_fused_loss": float(abl_loss[target_group]),
                    "delta_loss_ablated_minus_real": float(abl_loss[target_group] - real_loss[target_group]),
                }
            )

    summary = summarize(
        records,
        target_groups=target_groups,
        lift_k=int(args.lift_k),
        content_bucket_key=conditional_bucket_key,
    )
    output = {
        "config": {
            "dataset": str(args.dataset),
            "seed": int(args.seed),
            "payload_path": str(payload_path),
            "result_dir": str(args.result_dir),
            "full_checkpoint": str(full_checkpoint),
            "device": str(device),
            "feature_device": str(args.feature_device or config.get("feature_device", device)),
            "model": model_type,
            "gradient_params": str(args.gradient_params),
            "num_gradient_parameters": int(sum(param.numel() for _, param in named_params)),
            "damping": float(args.damping),
            "source_groups": [int(value) for value in source_groups],
            "source_selection": str(args.source_selection),
            "target_groups": [int(value) for value in target_groups],
            "source_interaction_content": {
                str(group): float(source_content[int(group)]) for group in source_groups
            },
            "tau_content_buckets": {
                str(group): int(tau_bucket_lookup.get(int(group), -1))
                for group in source_groups
            },
            "pid_profile_method": str(args.pid_profile_method),
            "pid_multilabel_label_mode": (
                "argmax_label_for_lsmi"
                if task == "multilabel" and str(args.pid_profile_method) == "lsmi"
                else None
            ),
            "pid_profiles": {str(group): pid_profiles.get(int(group), {}) for group in source_groups},
            "pid_profile_buckets": {str(group): int(pid_bucket_lookup.get(int(group), -1)) for group in source_groups},
            "conditional_bucket_key": conditional_bucket_key,
            "content_bucket_count": int(args.content_buckets),
            "target_split": str(args.target_split),
            "ft_epochs": int(args.ft_epochs),
            "ft_lr": float(model_args.lr),
            "ft_batch_size": int(model_args.batch_size),
            "ablation": "label_preserving_modality2_shuffle_within_source_group",
            "predictor": (
                "intervention-aligned: P_tau_h=c_target^T A_add^-1 delta_h_source; "
                "P_tau_c=c_target^T A_add^-1 delta_c_source; "
                "P_loss_h=h_target^T A_add^-1 delta_h_source"
            ),
            "baseline_predictor": "K_baseline=c_source^T A_add^-1 c_target",
        },
        "gradient_stats": gradient_stats(gradients),
        "summary": summary,
        "source_training": source_results,
        "records": records,
    }

    output_json = args.output_json or (
        args.output_dir / f"{args.dataset}_seed{args.seed}_real_motivating_example.json"
    )
    output_md = args.output_md or output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


def select_groups(
    *,
    explicit: Sequence[int] | None,
    group_indices: Sequence[Sequence[int]],
    num_groups: int,
    min_group_size: int,
    mode: str = "largest",
    scores: dict[int, float] | None = None,
) -> list[int]:
    if explicit:
        groups = sorted({int(value) for value in explicit})
        bad = [value for value in groups if value < 0 or value >= len(group_indices)]
        if bad:
            raise ValueError(f"group id(s) out of range 0..{len(group_indices) - 1}: {bad}")
        return groups
    candidates = [
        group_id
        for group_id, indices in enumerate(group_indices)
        if len(indices) >= int(min_group_size)
    ]
    if not candidates:
        candidates = list(range(len(group_indices)))
    if mode == "all":
        ordered = sorted(candidates)
        return [int(value) for value in ordered]
    if mode == "high_content":
        if scores is None:
            raise ValueError("mode='high_content' requires scores")
        ordered = sorted(
            candidates,
            key=lambda group_id: (float(scores.get(int(group_id), float("-inf"))), len(group_indices[group_id])),
            reverse=True,
        )
    else:
        ordered = sorted(candidates, key=lambda group_id: len(group_indices[group_id]), reverse=True)
    return [int(value) for value in ordered[: min(int(num_groups), len(ordered))]]


def make_content_buckets(
    scores: dict[int, float],
    *,
    num_buckets: int,
    eligible_groups: Sequence[int],
) -> dict[int, int]:
    groups = [int(group) for group in eligible_groups if np.isfinite(float(scores.get(int(group), np.nan)))]
    if not groups:
        return {}
    ordered = sorted(groups, key=lambda group: float(scores[group]))
    bucket_count = max(1, min(int(num_buckets), len(ordered)))
    buckets: dict[int, int] = {}
    for rank, group in enumerate(ordered):
        bucket = min(bucket_count - 1, int(rank * bucket_count / len(ordered)))
        buckets[int(group)] = int(bucket)
    return buckets


def estimate_pid_profiles(
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    groups: Sequence[int],
    method: str,
    task: str,
    device: str,
    profile_dim: int,
    hidden_dim: int,
    batch_size: int,
    discriminator_epochs: int,
    entropy_epochs: int,
    seed: int,
) -> dict[int, dict[str, Any]]:
    if method == "none":
        return {}
    if task not in {"binary", "multiclass", "multilabel"}:
        print(f"Skipping LSMI PID profiles for unsupported task={task!r}", flush=True)
        return {}
    labels_for_pid = data["train_labels"]
    if task == "binary":
        num_labels = 2
    elif task == "multiclass":
        num_labels = int(data["num_classes"])
    else:
        labels_for_pid = torch.argmax(labels_for_pid.float(), dim=1).long()
        num_labels = int(data["num_classes"])
    projected_x, projected_y = pid_projection(
        data["train_x"],
        data["train_y"],
        dim=int(profile_dim),
        seed=int(seed),
    )
    profiles: dict[int, dict[str, Any]] = {}
    for group in groups:
        indices = group_indices[int(group)]
        if len(indices) < 8:
            continue
        index = torch.as_tensor(indices, dtype=torch.long)
        config = LSMIConfig(
            hidden_dim=int(hidden_dim),
            batch_size=max(1, min(int(batch_size), len(indices))),
            discriminator_epochs=int(discriminator_epochs),
            entropy_epochs=int(entropy_epochs),
            lr=1e-3,
            verbose=False,
            seed=int(seed) + int(group),
        )
        result = estimate_lsmi(
            projected_x[index],
            projected_y[index],
            labels_for_pid[index],
            num_labels=num_labels,
            config=config,
            device=device,
        )
        raw = result.mean_components.detach().cpu().numpy().astype(np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw_clipped = np.maximum(raw, 0.0)
        denom = float(np.sum(raw_clipped)) + 1e-12
        norm = raw_clipped / denom
        profiles[int(group)] = {
            "R": float(raw[0]),
            "U1": float(raw[1]),
            "U2": float(raw[2]),
            "S": float(raw[3]),
            "norm_R": float(norm[0]),
            "norm_U1": float(norm[1]),
            "norm_U2": float(norm[2]),
            "norm_S": float(norm[3]),
            "mass": float(np.sum(raw_clipped)),
            "method": method,
        }
        print(
            f"  PID profile group {group}: "
            f"R={raw[0]:.4g} U1={raw[1]:.4g} U2={raw[2]:.4g} S={raw[3]:.4g}",
            flush=True,
        )
    return profiles


def pid_projection(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _project(values: torch.Tensor, offset: int) -> torch.Tensor:
        values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        values = torch.nn.functional.normalize(values, dim=1)
        if dim <= 0 or values.shape[1] <= dim:
            return values.cpu()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(offset))
        proj = torch.randn(values.shape[1], int(dim), generator=gen, dtype=torch.float32)
        proj = proj / np.sqrt(float(dim))
        return (values.cpu() @ proj).float()

    return _project(x, 101), _project(y, 202)


def make_pid_profile_buckets(
    profiles: dict[int, dict[str, Any]],
    *,
    num_buckets: int,
    seed: int,
) -> dict[int, int]:
    groups = sorted(profiles)
    if not groups:
        return {}
    matrix = np.asarray(
        [
            [
                float(profiles[group]["norm_R"]),
                float(profiles[group]["norm_U1"]),
                float(profiles[group]["norm_U2"]),
                float(profiles[group]["norm_S"]),
            ]
            for group in groups
        ],
        dtype=np.float64,
    )
    k = max(1, min(int(num_buckets), len(groups)))
    labels = simple_kmeans(matrix, k=k, seed=int(seed))
    return {int(group): int(label) for group, label in zip(groups, labels.tolist())}


def simple_kmeans(values: np.ndarray, *, k: int, seed: int, iterations: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] <= int(k):
        return np.arange(values.shape[0], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, values.shape[0]))
    centers = [values[first]]
    while len(centers) < int(k):
        dist = np.min(
            np.stack([np.sum((values - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        next_idx = int(np.argmax(dist))
        centers.append(values[next_idx])
    centers_arr = np.stack(centers, axis=0)
    labels = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(int(iterations)):
        dist = np.stack([np.sum((values - center) ** 2, axis=1) for center in centers_arr], axis=1)
        new_labels = np.argmin(dist, axis=1).astype(np.int64)
        new_centers = centers_arr.copy()
        for cluster in range(int(k)):
            members = values[new_labels == cluster]
            if members.size:
                new_centers[cluster] = members.mean(axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers_arr = new_centers
    return labels


def load_compatible_model(
    checkpoint: Path,
    *,
    explicit: bool,
    fallback_dir: Path,
    dataset: str,
    seed: int,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: Any,
    device: str,
) -> tuple[torch.nn.Module, Path]:
    print(f"[{dataset}] loading full checkpoint: {checkpoint}", flush=True)
    try:
        model = load_model(
            checkpoint,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=device,
        )
        return model, checkpoint
    except RuntimeError as exc:
        if explicit or "size mismatch" not in str(exc):
            raise
        fallback = Path(fallback_dir) / f"{dataset}_seed{seed}_full.pt"
        if not fallback.exists() or fallback == checkpoint:
            raise
        print(
            f"[{dataset}] checkpoint shape mismatch; retrying fallback checkpoint: {fallback}",
            flush=True,
        )
        model = load_model(
            fallback,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=device,
        )
        return model, fallback


def parameters_by_name(model: torch.nn.Module, param_names: Sequence[str]) -> list[torch.nn.Parameter]:
    lookup = dict(model.named_parameters())
    return [lookup[name] for name in param_names]


def flatten_grads(
    grads: Sequence[torch.Tensor | None],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    chunks = []
    for grad_value, param in zip(grads, params):
        if grad_value is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float32, device=param.device))
        else:
            chunks.append(grad_value.detach().reshape(-1).float())
    return torch.cat(chunks, dim=0).detach().cpu()


def objective_gradient(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    task: str,
    device: str,
    param_names: Sequence[str],
    use_x: bool,
    use_y: bool,
) -> tuple[torch.Tensor, float]:
    params = parameters_by_name(model, param_names)
    index = torch.as_tensor(indices, dtype=torch.long)
    bx = data["train_x"][index].to(device)
    by = data["train_y"][index].to(device)
    bl = data["train_labels"][index].to(device)
    model.zero_grad(set_to_none=True)
    loss = compute_loss(task, model(bx, by, use_x=use_x, use_y=use_y), bl)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return flatten_grads(grads, params), float(loss.detach().cpu().item())


def compute_group_gradients(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: str,
    param_names: Sequence[str],
) -> dict[str, Any]:
    full = []
    single0 = []
    single1 = []
    losses = {"full": [], "single_0": [], "single_1": []}
    for group_id, indices in enumerate(group_indices):
        h, full_loss = objective_gradient(
            model,
            data=data,
            indices=indices,
            task=task,
            device=device,
            param_names=param_names,
            use_x=True,
            use_y=True,
        )
        g0, loss0 = objective_gradient(
            model,
            data=data,
            indices=indices,
            task=task,
            device=device,
            param_names=param_names,
            use_x=True,
            use_y=False,
        )
        g1, loss1 = objective_gradient(
            model,
            data=data,
            indices=indices,
            task=task,
            device=device,
            param_names=param_names,
            use_x=False,
            use_y=True,
        )
        full.append(h)
        single0.append(g0)
        single1.append(g1)
        losses["full"].append(full_loss)
        losses["single_0"].append(loss0)
        losses["single_1"].append(loss1)
        if (group_id + 1) % 5 == 0 or group_id + 1 == len(group_indices):
            print(f"  group gradients {group_id + 1}/{len(group_indices)}", flush=True)
    return {
        "full": torch.stack(full).numpy().astype(np.float32),
        "single_0": torch.stack(single0).numpy().astype(np.float32),
        "single_1": torch.stack(single1).numpy().astype(np.float32),
        "losses": losses,
    }


def compute_group_fic(
    gradients: dict[str, Any],
    *,
    group_indices: Sequence[Sequence[int]],
    damping: float,
) -> dict[str, np.ndarray]:
    h = np.asarray(gradients["full"], dtype=np.float64)
    g0 = np.asarray(gradients["single_0"], dtype=np.float64)
    g1 = np.asarray(gradients["single_1"], dtype=np.float64)
    c = h - g0 - g1
    sizes = np.asarray([len(indices) for indices in group_indices], dtype=np.float64)
    weights = sizes / max(float(sizes.sum()), 1.0)
    a_add = (g0.T * weights) @ g0 + (g1.T * weights) @ g1
    a_add += float(damping) * np.eye(a_add.shape[0], dtype=np.float64)
    a_inv = np.linalg.inv(a_add)
    a_inv_c_t = a_inv @ c.T
    k = c @ a_inv_c_t
    energy = np.clip(np.diag(k), 0.0, None)
    denom = energy[:, None] * energy[None, :]
    fic = np.zeros_like(k)
    valid = denom > 1e-20
    fic[valid] = np.clip((k[valid] * k[valid]) / denom[valid], 0.0, 1.0)
    return {
        "K": k.astype(np.float32),
        "K_plus": np.maximum(k, 0.0).astype(np.float32),
        "FIC": fic.astype(np.float32),
        "energy": energy.astype(np.float32),
        "A_inv": a_inv.astype(np.float32),
    }


def objective_gradient_from_tensors(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    device: str,
    param_names: Sequence[str],
    use_x: bool,
    use_y: bool,
) -> torch.Tensor:
    params = parameters_by_name(model, param_names)
    model.zero_grad(set_to_none=True)
    bx = x.to(device)
    by = y.to(device)
    bl = labels.to(device)
    loss = compute_loss(task, model(bx, by, use_x=use_x, use_y=use_y), bl)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return flatten_grads(grads, params)


def source_intervention_gradients_for_group(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    task: str,
    device: str,
    param_names: Sequence[str],
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    index = torch.as_tensor(indices, dtype=torch.long)
    x_real = data["train_x"][index]
    y_real = data["train_y"][index]
    labels = data["train_labels"][index]
    y_abl = shuffled_modality2_within_labels(y_real, labels, seed=int(shuffle_seed))

    h_real = objective_gradient_from_tensors(
        model,
        x=x_real,
        y=y_real,
        labels=labels,
        task=task,
        device=device,
        param_names=param_names,
        use_x=True,
        use_y=True,
    )
    g0_real = objective_gradient_from_tensors(
        model,
        x=x_real,
        y=y_real,
        labels=labels,
        task=task,
        device=device,
        param_names=param_names,
        use_x=True,
        use_y=False,
    )
    g1_real = objective_gradient_from_tensors(
        model,
        x=x_real,
        y=y_real,
        labels=labels,
        task=task,
        device=device,
        param_names=param_names,
        use_x=False,
        use_y=True,
    )
    h_abl = objective_gradient_from_tensors(
        model,
        x=x_real,
        y=y_abl,
        labels=labels,
        task=task,
        device=device,
        param_names=param_names,
        use_x=True,
        use_y=True,
    )
    g0_abl = objective_gradient_from_tensors(
        model,
        x=x_real,
        y=y_abl,
        labels=labels,
        task=task,
        device=device,
        param_names=param_names,
        use_x=True,
        use_y=False,
    )
    g1_abl = objective_gradient_from_tensors(
        model,
        x=x_real,
        y=y_abl,
        labels=labels,
        task=task,
        device=device,
        param_names=param_names,
        use_x=False,
        use_y=True,
    )
    c_real = h_real - g0_real - g1_real
    c_abl = h_abl - g0_abl - g1_abl
    return {
        "h_real": h_real.numpy().astype(np.float32),
        "h_abl": h_abl.numpy().astype(np.float32),
        "c_real": c_real.numpy().astype(np.float32),
        "c_abl": c_abl.numpy().astype(np.float32),
        "delta_h": (h_real - h_abl).numpy().astype(np.float32),
        "delta_c": (c_real - c_abl).numpy().astype(np.float32),
    }


def compute_source_intervention_gradients(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    source_groups: Sequence[int],
    task: str,
    device: str,
    param_names: Sequence[str],
    shuffle_seed: int,
) -> dict[int, dict[str, np.ndarray]]:
    output = {}
    for source_group in source_groups:
        output[int(source_group)] = source_intervention_gradients_for_group(
            model,
            data=data,
            indices=group_indices[source_group],
            task=task,
            device=device,
            param_names=param_names,
            shuffle_seed=int(shuffle_seed) + int(source_group),
        )
    return output


def compute_intervention_predictors(
    gradients: dict[str, Any],
    source_delta: dict[int, dict[str, np.ndarray]],
    *,
    fic_payload: dict[str, np.ndarray],
    source_groups: Sequence[int],
    target_groups: Sequence[int],
) -> dict[str, dict[tuple[int, int], float]]:
    h_target = np.asarray(gradients["full"], dtype=np.float64)
    c_target = (
        np.asarray(gradients["full"], dtype=np.float64)
        - np.asarray(gradients["single_0"], dtype=np.float64)
        - np.asarray(gradients["single_1"], dtype=np.float64)
    )
    a_inv = np.asarray(fic_payload["A_inv"], dtype=np.float64)
    out = {"P_tau_h": {}, "P_tau_c": {}, "P_loss_h": {}}
    for source in source_groups:
        delta_h = np.asarray(source_delta[int(source)]["delta_h"], dtype=np.float64)
        delta_c = np.asarray(source_delta[int(source)]["delta_c"], dtype=np.float64)
        solved_delta_h = a_inv @ delta_h
        solved_delta_c = a_inv @ delta_c
        for target in target_groups:
            key = (int(source), int(target))
            out["P_tau_h"][key] = float(c_target[int(target)] @ solved_delta_h)
            out["P_tau_c"][key] = float(c_target[int(target)] @ solved_delta_c)
            out["P_loss_h"][key] = float(h_target[int(target)] @ solved_delta_h)
    return out


def clone_model_from_base(
    base_model: torch.nn.Module,
    *,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: Any,
    device: str,
) -> torch.nn.Module:
    model = build_model(model_type, dim_x, dim_y, num_classes, model_args)
    model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    model.to(device)
    model.eval()
    return model


def shuffled_modality2_within_labels(
    train_y: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    shuffled = train_y.clone()
    keys = label_keys(labels)
    by_key: dict[tuple[Any, ...], list[int]] = {}
    for index, key in enumerate(keys):
        by_key.setdefault(key, []).append(index)
    for indices in by_key.values():
        if len(indices) < 2:
            continue
        permuted = rng.permutation(np.asarray(indices, dtype=np.int64))
        shuffled[torch.as_tensor(indices, dtype=torch.long)] = train_y[
            torch.as_tensor(permuted, dtype=torch.long)
        ]
    return shuffled


def label_keys(labels: torch.Tensor) -> list[tuple[Any, ...]]:
    if labels.ndim == 1:
        return [(int(value),) for value in labels.detach().cpu().tolist()]
    arr = labels.detach().cpu().numpy()
    return [tuple(np.asarray(row).astype(int).tolist()) for row in arr]


def fine_tune_source(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    task: str,
    device: str,
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
    shuffle_modality2: bool,
    shuffle_seed: int,
) -> list[float]:
    set_seed(int(seed))
    index = torch.as_tensor(indices, dtype=torch.long)
    train_x = data["train_x"][index]
    train_y = data["train_y"][index]
    train_labels = data["train_labels"][index]
    if shuffle_modality2:
        train_y = shuffled_modality2_within_labels(train_y, train_labels, seed=int(shuffle_seed))
    loader = make_loader(train_x, train_y, train_labels, int(batch_size), True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    epoch_losses = []
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for bx, by, bl in loader:
            bx = bx.to(device)
            by = by.to(device)
            bl = bl.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * int(bl.shape[0])
            total_samples += int(bl.shape[0])
        epoch_loss = total_loss / max(total_samples, 1)
        epoch_losses.append(float(epoch_loss))
        print(f"    {'abl' if shuffle_modality2 else 'real'} epoch {epoch + 1}/{epochs}: loss={epoch_loss:.6f}", flush=True)
    model.eval()
    return epoch_losses


def per_sample_loss(
    task: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if task == "binary":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1), labels.float().view(-1), reduction="none"
        )
    if task == "multilabel":
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        return loss.mean(dim=1)
    return torch.nn.functional.cross_entropy(logits, labels.long(), reduction="none")


def log_prior(labels: torch.Tensor, task: str) -> torch.Tensor:
    if task == "binary":
        p = labels.float().mean().clamp(1e-6, 1.0 - 1e-6)
        return torch.where(labels.float() > 0.5, torch.log(p), torch.log1p(-p))
    if task == "multilabel":
        p = labels.float().mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
        logp = torch.where(labels.float() > 0.5, torch.log(p), torch.log1p(-p))
        return logp.mean(dim=1)
    counts = torch.bincount(labels.long(), minlength=int(labels.max().item()) + 1).float()
    p = (counts / counts.sum().clamp(min=1.0)).clamp(1e-6, 1.0)
    return torch.log(p[labels.long()])


def evaluate_group_tau_loss(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    device: str,
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        bx = x.to(device)
        by = y.to(device)
        bl = labels.to(device)
        full_loss = per_sample_loss(task, model(bx, by, use_x=True, use_y=True), bl)
        x_loss = per_sample_loss(task, model(bx, by, use_x=True, use_y=False), bl)
        y_loss = per_sample_loss(task, model(bx, by, use_x=False, use_y=True), bl)
        tau = -full_loss + x_loss + y_loss + log_prior(bl.detach().cpu(), task).to(device)
    return float(tau.mean().detach().cpu().item()), float(full_loss.mean().detach().cpu().item())


def evaluate_groups(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    groups: Sequence[Sequence[int]],
    target_groups: Sequence[int],
    split: str,
    task: str,
    device: str,
) -> tuple[dict[int, float], dict[int, float]]:
    tau_values = {}
    loss_values = {}
    x_all = data[f"{split}_x"]
    y_all = data[f"{split}_y"]
    labels_all = data[f"{split}_labels"]
    for group in target_groups:
        indices = groups[group]
        if len(indices) == 0:
            tau_values[int(group)] = float("nan")
            loss_values[int(group)] = float("nan")
            continue
        index = torch.as_tensor(indices, dtype=torch.long)
        tau, loss = evaluate_group_tau_loss(
            model,
            x=x_all[index],
            y=y_all[index],
            labels=labels_all[index],
            task=task,
            device=device,
        )
        tau_values[int(group)] = tau
        loss_values[int(group)] = loss
    return tau_values, loss_values


def gradient_stats(gradients: dict[str, Any]) -> dict[str, float]:
    h = np.asarray(gradients["full"], dtype=np.float64)
    g0 = np.asarray(gradients["single_0"], dtype=np.float64)
    g1 = np.asarray(gradients["single_1"], dtype=np.float64)
    c = h - g0 - g1
    return {
        "full_grad_norm_mean": float(np.mean(np.linalg.norm(h, axis=1))),
        "single0_grad_norm_mean": float(np.mean(np.linalg.norm(g0, axis=1))),
        "single1_grad_norm_mean": float(np.mean(np.linalg.norm(g1, axis=1))),
        "residual_grad_norm_mean": float(np.mean(np.linalg.norm(c, axis=1))),
        "residual_grad_norm_max": float(np.max(np.linalg.norm(c, axis=1))),
    }


def summarize(
    records: Sequence[dict[str, Any]],
    *,
    target_groups: Sequence[int],
    lift_k: int,
    content_bucket_key: str | None = None,
) -> dict[str, Any]:
    k_plus_baseline = np.asarray([float(row["K_plus_baseline_c_source_c_target"]) for row in records], dtype=np.float64)
    fic_baseline = np.asarray([float(row["FIC_baseline_c_source_c_target"]) for row in records], dtype=np.float64)
    p_tau_h = np.asarray([float(row["P_tau_h"]) for row in records], dtype=np.float64)
    p_tau_h_plus = np.asarray([float(row["P_tau_h_plus"]) for row in records], dtype=np.float64)
    p_tau_c = np.asarray([float(row["P_tau_c"]) for row in records], dtype=np.float64)
    p_tau_c_plus = np.asarray([float(row["P_tau_c_plus"]) for row in records], dtype=np.float64)
    p_loss_h = np.asarray([float(row["P_loss_h"]) for row in records], dtype=np.float64)
    p_loss_h_plus = np.asarray([float(row["P_loss_h_plus"]) for row in records], dtype=np.float64)
    delta_tau = np.asarray([float(row["delta_tau_real_minus_ablated"]) for row in records], dtype=np.float64)
    delta_loss = np.asarray([float(row["delta_loss_ablated_minus_real"]) for row in records], dtype=np.float64)
    offdiag_mask = np.asarray(
        [int(row["source_group"]) != int(row["target_group"]) for row in records],
        dtype=bool,
    )
    summary = {
        "num_pairs": int(len(records)),
        "delta_tau_mean": safe_mean(delta_tau),
        "delta_loss_mean": safe_mean(delta_loss),
        "P_tau_h_delta_tau_spearman": spearmanr(p_tau_h, delta_tau),
        "P_tau_h_delta_tau_pearson": pearsonr(p_tau_h, delta_tau),
        "P_tau_h_plus_delta_tau_spearman": spearmanr(p_tau_h_plus, delta_tau),
        "P_tau_c_delta_tau_spearman": spearmanr(p_tau_c, delta_tau),
        "P_tau_c_delta_tau_pearson": pearsonr(p_tau_c, delta_tau),
        "P_tau_c_plus_delta_tau_spearman": spearmanr(p_tau_c_plus, delta_tau),
        "P_loss_h_delta_loss_spearman": spearmanr(p_loss_h, delta_loss),
        "P_loss_h_delta_loss_pearson": pearsonr(p_loss_h, delta_loss),
        "P_loss_h_plus_delta_loss_spearman": spearmanr(p_loss_h_plus, delta_loss),
        "baseline_K_plus_delta_tau_spearman": spearmanr(k_plus_baseline, delta_tau),
        "baseline_K_plus_delta_loss_spearman": spearmanr(k_plus_baseline, delta_loss),
        "baseline_FIC_delta_tau_spearman": spearmanr(fic_baseline, delta_tau),
        "baseline_FIC_delta_loss_spearman": spearmanr(fic_baseline, delta_loss),
        "offdiag_P_tau_h_delta_tau_spearman": spearmanr(p_tau_h[offdiag_mask], delta_tau[offdiag_mask]),
        "offdiag_P_tau_c_delta_tau_spearman": spearmanr(p_tau_c[offdiag_mask], delta_tau[offdiag_mask]),
        "offdiag_P_loss_h_delta_loss_spearman": spearmanr(p_loss_h[offdiag_mask], delta_loss[offdiag_mask]),
        "offdiag_baseline_K_plus_delta_tau_spearman": spearmanr(k_plus_baseline[offdiag_mask], delta_tau[offdiag_mask]),
        "offdiag_baseline_K_plus_delta_loss_spearman": spearmanr(k_plus_baseline[offdiag_mask], delta_loss[offdiag_mask]),
        "lift_at_k_tau_P_tau_h": lift_at_k(records, target_groups=target_groups, score_key="P_tau_h", value_key="delta_tau_real_minus_ablated", k=int(lift_k)),
        "lift_at_k_tau_P_tau_c": lift_at_k(records, target_groups=target_groups, score_key="P_tau_c", value_key="delta_tau_real_minus_ablated", k=int(lift_k)),
        "lift_at_k_loss_P_loss_h": lift_at_k(records, target_groups=target_groups, score_key="P_loss_h", value_key="delta_loss_ablated_minus_real", k=int(lift_k)),
        "baseline_lift_at_k_tau_K_plus": lift_at_k(records, target_groups=target_groups, score_key="K_plus_baseline_c_source_c_target", value_key="delta_tau_real_minus_ablated", k=int(lift_k)),
        "baseline_lift_at_k_loss_K_plus": lift_at_k(records, target_groups=target_groups, score_key="K_plus_baseline_c_source_c_target", value_key="delta_loss_ablated_minus_real", k=int(lift_k)),
    }
    if content_bucket_key is not None:
        summary["conditional_content"] = {
            "bucket_key": content_bucket_key,
            "lift_k": int(lift_k),
            "P_tau_h_tau": conditional_lift(
                records,
                target_groups=target_groups,
                score_key="P_tau_h",
                value_key="delta_tau_real_minus_ablated",
                bucket_key=content_bucket_key,
                k=int(lift_k),
            ),
            "P_tau_c_tau": conditional_lift(
                records,
                target_groups=target_groups,
                score_key="P_tau_c",
                value_key="delta_tau_real_minus_ablated",
                bucket_key=content_bucket_key,
                k=int(lift_k),
            ),
            "P_loss_h_loss": conditional_lift(
                records,
                target_groups=target_groups,
                score_key="P_loss_h",
                value_key="delta_loss_ablated_minus_real",
                bucket_key=content_bucket_key,
                k=int(lift_k),
            ),
            "baseline_K_plus_tau": conditional_lift(
                records,
                target_groups=target_groups,
                score_key="K_plus_baseline_c_source_c_target",
                value_key="delta_tau_real_minus_ablated",
                bucket_key=content_bucket_key,
                k=int(lift_k),
            ),
        }
    return summary


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def pearsonr(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    x = np.asarray(list(lhs), dtype=np.float64)
    y = np.asarray(list(rhs), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearmanr(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    x = np.asarray(list(lhs), dtype=np.float64)
    y = np.asarray(list(rhs), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    return pearsonr(rank_average(x), rank_average(y))


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def lift_at_k(
    records: Sequence[dict[str, Any]],
    *,
    target_groups: Sequence[int],
    score_key: str,
    value_key: str,
    k: int,
) -> float:
    lifts = []
    for target in target_groups:
        rows = [
            row
            for row in records
            if int(row["target_group"]) == int(target)
            and np.isfinite(float(row[score_key]))
            and np.isfinite(float(row[value_key]))
        ]
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: float(row[score_key]), reverse=True)
        top = np.asarray([float(row[value_key]) for row in ordered[: max(1, min(int(k), len(ordered)))]], dtype=np.float64)
        all_values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
        denom = float(np.mean(all_values))
        if abs(denom) <= 1e-12:
            continue
        lifts.append(float(np.mean(top) / denom))
    return float(np.mean(lifts)) if lifts else float("nan")


def conditional_lift(
    records: Sequence[dict[str, Any]],
    *,
    target_groups: Sequence[int],
    score_key: str,
    value_key: str,
    bucket_key: str,
    k: int,
) -> dict[str, Any]:
    diffs = []
    cells = []
    buckets = sorted({
        int(row[bucket_key])
        for row in records
        if row.get(bucket_key) is not None and int(row[bucket_key]) >= 0
    })
    for target in target_groups:
        for bucket in buckets:
            rows = [
                row
                for row in records
                if int(row["target_group"]) == int(target)
                and int(row.get(bucket_key, -1)) == int(bucket)
                and np.isfinite(float(row[score_key]))
                and np.isfinite(float(row[value_key]))
            ]
            if len(rows) < 2:
                continue
            ordered = sorted(rows, key=lambda row: float(row[score_key]), reverse=True)
            kk = max(1, min(int(k), len(ordered) // 2))
            top = np.asarray([float(row[value_key]) for row in ordered[:kk]], dtype=np.float64)
            bottom = np.asarray([float(row[value_key]) for row in ordered[-kk:]], dtype=np.float64)
            diff = float(np.mean(top) - np.mean(bottom))
            diffs.append(diff)
            cells.append(
                {
                    "target_group": int(target),
                    "bucket": int(bucket),
                    "num_sources": int(len(rows)),
                    "top_mean": float(np.mean(top)),
                    "bottom_mean": float(np.mean(bottom)),
                    "difference": diff,
                }
            )
    return {
        "num_cells": int(len(cells)),
        "mean_top_minus_bottom": float(np.mean(diffs)) if diffs else float("nan"),
        "positive_fraction": float(np.mean(np.asarray(diffs) > 0.0)) if diffs else float("nan"),
        "cells": cells,
    }


def to_markdown(output: dict[str, Any]) -> str:
    cfg = output["config"]
    summary = output["summary"]
    lines = [
        "# Real Motivating Example",
        "",
        f"- dataset: `{cfg['dataset']}`, seed: `{cfg['seed']}`, device: `{cfg['device']}`",
        f"- checkpoint: `{cfg['full_checkpoint']}`",
        f"- source groups: `{cfg['source_groups']}`",
        f"- source selection: `{cfg.get('source_selection', 'largest')}`, content buckets: `{cfg.get('content_bucket_count', 'n/a')}`",
        f"- PID profile method: `{cfg.get('pid_profile_method', 'none')}`, conditional bucket: `{cfg.get('conditional_bucket_key', 'source_content_bucket')}`",
        f"- target groups: `{cfg['target_groups']}` on `{cfg['target_split']}`",
        f"- FIC params: `{cfg['gradient_params']}` ({cfg['num_gradient_parameters']} parameters), damping `{cfg['damping']}`",
        f"- intervention: `{cfg['ablation']}`, ft epochs `{cfg['ft_epochs']}`, lr `{cfg['ft_lr']}`",
        "",
        "## Summary",
        "",
        f"- pairs: `{summary['num_pairs']}`",
        f"- mean Delta tau(real-abl): `{summary['delta_tau_mean']:.6g}`",
        f"- mean Delta loss(abl-real): `{summary['delta_loss_mean']:.6g}`",
        f"- P_tau_h vs Delta tau Spearman: `{summary['P_tau_h_delta_tau_spearman']:.3f}`",
        f"- P_tau_c vs Delta tau Spearman: `{summary['P_tau_c_delta_tau_spearman']:.3f}`",
        f"- P_loss_h vs Delta loss Spearman: `{summary['P_loss_h_delta_loss_spearman']:.3f}`",
        f"- baseline K+ vs Delta tau Spearman: `{summary['baseline_K_plus_delta_tau_spearman']:.3f}`",
        f"- baseline K+ vs Delta loss Spearman: `{summary['baseline_K_plus_delta_loss_spearman']:.3f}`",
        f"- offdiag P_tau_h vs Delta tau Spearman: `{summary['offdiag_P_tau_h_delta_tau_spearman']:.3f}`",
        f"- offdiag P_tau_c vs Delta tau Spearman: `{summary['offdiag_P_tau_c_delta_tau_spearman']:.3f}`",
        f"- offdiag P_loss_h vs Delta loss Spearman: `{summary['offdiag_P_loss_h_delta_loss_spearman']:.3f}`",
        f"- Lift@k tau by P_tau_h: `{summary['lift_at_k_tau_P_tau_h']:.3f}`",
        f"- Lift@k tau by P_tau_c: `{summary['lift_at_k_tau_P_tau_c']:.3f}`",
        f"- Lift@k loss by P_loss_h: `{summary['lift_at_k_loss_P_loss_h']:.3f}`",
        f"- baseline Lift@k tau by K+: `{summary['baseline_lift_at_k_tau_K_plus']:.3f}`",
        f"- baseline Lift@k loss by K+: `{summary['baseline_lift_at_k_loss_K_plus']:.3f}`",
        "",
    ]
    conditional = summary.get("conditional_content")
    if conditional:
        lines.extend(
            [
                "## Conditional Content Lift",
                "",
                f"- bucket key: `{conditional['bucket_key']}`, lift k: `{conditional['lift_k']}`",
                f"- P_tau_h tau Top-Bottom: `{conditional['P_tau_h_tau']['mean_top_minus_bottom']:.6g}` "
                f"(positive cells `{conditional['P_tau_h_tau']['positive_fraction']:.3f}`, n `{conditional['P_tau_h_tau']['num_cells']}`)",
                f"- P_tau_c tau Top-Bottom: `{conditional['P_tau_c_tau']['mean_top_minus_bottom']:.6g}` "
                f"(positive cells `{conditional['P_tau_c_tau']['positive_fraction']:.3f}`, n `{conditional['P_tau_c_tau']['num_cells']}`)",
                f"- P_loss_h loss Top-Bottom: `{conditional['P_loss_h_loss']['mean_top_minus_bottom']:.6g}` "
                f"(positive cells `{conditional['P_loss_h_loss']['positive_fraction']:.3f}`, n `{conditional['P_loss_h_loss']['num_cells']}`)",
                f"- baseline K+ tau Top-Bottom: `{conditional['baseline_K_plus_tau']['mean_top_minus_bottom']:.6g}` "
                f"(positive cells `{conditional['baseline_K_plus_tau']['positive_fraction']:.3f}`, n `{conditional['baseline_K_plus_tau']['num_cells']}`)",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Records By P_tau_h",
            "",
        "| source | tau bucket | PID bucket | tau content | PID S | target | P_tau_h | P_tau_c | P_loss_h | baseline K+ | Delta tau | Delta loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    top = sorted(output["records"], key=lambda row: float(row["P_tau_h"]), reverse=True)[:20]
    for row in top:
        lines.append(
            f"| {row['source_group']} | {row.get('source_content_bucket', -1)} | "
            f"{row.get('source_pid_bucket', -1)} | "
            f"{row.get('source_interaction_content_tau', float('nan')):.6g} | "
            f"{row.get('source_pid_profile', {}).get('S', float('nan')):.6g} | "
            f"{row['target_group']} | "
            f"{row['P_tau_h']:.6g} | {row['P_tau_c']:.6g} | "
            f"{row['P_loss_h']:.6g} | {row['K_plus_baseline_c_source_c_target']:.6g} | "
            f"{row['delta_tau_real_minus_ablated']:.6g} | "
            f"{row['delta_loss_ablated_minus_real']:.6g} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
