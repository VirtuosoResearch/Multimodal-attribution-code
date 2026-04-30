from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

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
from kernel_surrogate.kernel_attribution import spearmanr
from kernel_surrogate.real_data_cmis import make_model_args, resolve_device


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "kmeans_grouped25_subset100_data80_mod80"
)
DEFAULT_FULL_CHECKPOINT = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_checkpoints"
    / "lds_real_train"
    / "kmeans_grouped25_full_data_train"
    / "humor"
    / "seed0"
    / "subset001"
    / "checkpoints"
    / "checkpoint_repeat01.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "humor_group_cgmi"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group-level empirical validation for verification.md on the humor dataset. "
            "The unit is the existing k-means group, and each group gradient is the "
            "mean gradient over the exact samples assigned to that group."
        )
    )
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--full-checkpoint", type=Path, default=DEFAULT_FULL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--gradient-params", choices=["all", "last_layer", "fusion_head"], default="fusion_head")
    parser.add_argument("--source-groups", nargs="*", type=int, default=None)
    parser.add_argument("--target-groups", nargs="*", type=int, default=None)
    parser.add_argument("--modalities", nargs="*", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--fisher-damping", type=float, default=1e-3)
    parser.add_argument("--fisher-basis", choices=["full_group", "group_modality"], default="full_group")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-tracin", action="store_true")
    parser.add_argument("--trajectory-checkpoints", type=int, default=5)
    parser.add_argument("--force-train-trajectory", action="store_true")
    parser.add_argument("--allow-final-as-trajectory", action="store_true")
    parser.add_argument("--compute-ntk", action="store_true")
    parser.add_argument("--ntk-top-k", type=int, default=5)
    parser.add_argument("--compute-gt", action="store_true")
    parser.add_argument("--force-train-gt", action="store_true")
    parser.add_argument("--max-gt-source-interventions", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
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
    task = str(data["task"])
    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    model_type = str(config.get("model", "cross_attention"))
    model_args = make_model_args(config, device=str(device))
    group_indices = [[int(index) for index in group] for group in payload["group_partition"]["group_indices"]]
    num_groups = len(group_indices)
    source_groups = resolve_group_selection(args.source_groups, num_groups)
    target_groups = resolve_group_selection(args.target_groups, num_groups)
    modalities = [int(value) for value in args.modalities]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / f"{args.dataset}_seed{args.seed}_group_cgmi.json")

    print(f"Loading full checkpoint: {args.full_checkpoint}", flush=True)
    base_model = load_model(
        args.full_checkpoint,
        model_type=model_type,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        device=str(device),
    )
    named_params = selected_named_parameters(base_model, str(args.gradient_params))
    param_names = [name for name, _ in named_params]
    print(
        f"Selected {len(named_params)} parameter tensors "
        f"({sum(p.numel() for _, p in named_params):,} scalars).",
        flush=True,
    )

    print("Computing final-checkpoint group gradients...", flush=True)
    final_cache = build_group_gradient_cache(
        base_model,
        data=data,
        group_indices=group_indices,
        task=task,
        device=str(device),
        param_names=param_names,
    )
    fisher_basis = build_fisher_basis(final_cache, basis=str(args.fisher_basis))

    records = initialize_records(source_groups, target_groups, modalities)
    if not args.skip_static:
        print("Computing static Fisher-preconditioned scores...", flush=True)
        add_static_scores(
            records,
            final_cache=final_cache,
            fisher_basis=fisher_basis,
            damping=float(args.fisher_damping),
        )

    trajectory_manifest: dict[str, Any] | None = None
    if (not args.skip_tracin) or args.compute_ntk:
        trajectory_manifest = ensure_trajectory_checkpoints(
            args.output_dir / "trajectory",
            data=data,
            task=task,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            seed=int(args.seed),
            count=int(args.trajectory_checkpoints),
            force=bool(args.force_train_trajectory),
            fallback_checkpoint=args.full_checkpoint if args.allow_final_as_trajectory else None,
        )

    if trajectory_manifest is not None and not args.skip_tracin:
        print(f"Computing TracIn scores from {len(trajectory_manifest['checkpoints'])} checkpoint(s)...", flush=True)
        add_tracin_scores(
            records,
            trajectory_manifest=trajectory_manifest,
            data=data,
            group_indices=group_indices,
            task=task,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=str(device),
            param_names=param_names,
        )

    if trajectory_manifest is not None and args.compute_ntk:
        print(
            f"Computing group-level NTK projection shifts with top-k={int(args.ntk_top_k)}...",
            flush=True,
        )
        add_ntk_projection_scores(
            records,
            trajectory_manifest=trajectory_manifest,
            data=data,
            group_indices=group_indices,
            task=task,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=str(device),
            param_names=param_names,
            top_k=int(args.ntk_top_k),
        )

    gt_manifest: dict[str, Any] | None = None
    if args.compute_gt:
        source_interventions = unique_source_interventions(records)
        if args.max_gt_source_interventions > 0:
            source_interventions = source_interventions[: int(args.max_gt_source_interventions)]
        print(f"Computing LOGO GT for {len(source_interventions)} source-modality intervention(s)...", flush=True)
        gt_manifest = add_ground_truth_scores(
            records,
            source_interventions=source_interventions,
            output_dir=args.output_dir / "gt_interventions",
            data=data,
            group_indices=group_indices,
            task=task,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            seed=int(args.seed),
            base_model=base_model,
            device=str(device),
            force=bool(args.force_train_gt),
        )

    summary = summarize_records(records)
    output = {
        "config": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "result_dir": str(args.result_dir),
            "payload_path": str(payload_path),
            "full_checkpoint": str(args.full_checkpoint),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "model": model_type,
            "gradient_params": str(args.gradient_params),
            "parameter_names": param_names,
            "num_groups": int(num_groups),
            "source_groups": source_groups,
            "target_groups": target_groups,
            "modalities": modalities,
            "fisher_damping": float(args.fisher_damping),
            "fisher_basis": str(args.fisher_basis),
            "compute_ntk": bool(args.compute_ntk),
            "ntk_top_k": int(args.ntk_top_k),
            "compute_gt": bool(args.compute_gt),
        },
        "gradient_stats": gradient_stats(final_cache),
        "trajectory": trajectory_manifest,
        "gt": gt_manifest,
        "summary": summary,
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_json}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def resolve_group_selection(values: Sequence[int] | None, num_groups: int) -> list[int]:
    if values is None or len(values) == 0:
        return list(range(num_groups))
    groups = sorted({int(value) for value in values})
    bad = [value for value in groups if value < 0 or value >= num_groups]
    if bad:
        raise ValueError(f"group id(s) out of range 0..{num_groups - 1}: {bad}")
    return groups


def resolve_requested_device(requested: str) -> torch.device:
    if requested.startswith("cuda"):
        try:
            device = torch.device(requested)
            torch.cuda.get_device_properties(device)
            return device
        except Exception as exc:
            print(
                f"Direct CUDA probe failed for {requested}: {exc}. "
                "Falling back to shared resolver.",
                flush=True,
            )
    return resolve_device(requested)


def load_model(
    checkpoint_path: Path,
    *,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    device: str,
) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    model = build_model(model_type, dim_x, dim_y, num_classes, model_args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def selected_named_parameters(model: torch.nn.Module, mode: str) -> list[tuple[str, torch.nn.Parameter]]:
    if mode in {"all", "fusion_head"}:
        named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    elif mode == "last_layer":
        named = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and name.startswith("classifier.")
        ]
        if not named:
            named = [
                (name, param)
                for name, param in model.named_parameters()
                if param.requires_grad and ("classifier" in name or "head" in name)
            ]
    else:
        raise ValueError(f"unsupported gradient parameter mode: {mode}")
    if not named:
        raise ValueError(f"no parameters selected for gradient parameter mode: {mode}")
    return named


def parameters_by_name(model: torch.nn.Module, param_names: Sequence[str]) -> list[torch.nn.Parameter]:
    lookup = dict(model.named_parameters())
    return [lookup[name] for name in param_names]


def flatten_grads(
    grads: Sequence[torch.Tensor | None],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    chunks = []
    for grad, param in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float32, device=param.device))
        else:
            chunks.append(grad.detach().reshape(-1).float())
    return torch.cat(chunks, dim=0).detach().cpu()


def group_loss_gradient(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    task: str,
    device: str,
    param_names: Sequence[str],
    mode: str,
    modality: int | None = None,
) -> torch.Tensor:
    params = parameters_by_name(model, param_names)
    index = torch.as_tensor(indices, dtype=torch.long)
    bx = data["train_x"][index].to(device)
    by = data["train_y"][index].to(device)
    bl = data["train_labels"][index].to(device)

    if mode == "full":
        use_x, use_y = True, True
    elif mode == "modality":
        if modality is None:
            raise ValueError("mode='modality' requires modality")
        use_x, use_y = modality == 0, modality == 1
    elif mode == "masked":
        if modality is None:
            raise ValueError("mode='masked' requires modality")
        bx = bx.clone()
        by = by.clone()
        if modality == 0:
            bx.zero_()
        elif modality == 1:
            by.zero_()
        else:
            raise ValueError(f"unsupported modality: {modality}")
        use_x, use_y = True, True
    else:
        raise ValueError(f"unsupported gradient mode: {mode}")

    model.zero_grad(set_to_none=True)
    logits = model(bx, by, use_x=use_x, use_y=use_y)
    loss = compute_loss(task, logits, bl)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return flatten_grads(grads, params)


def build_group_gradient_cache(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: str,
    param_names: Sequence[str],
) -> dict[str, Any]:
    full: list[torch.Tensor] = []
    modality: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    masked_delta: dict[int, list[torch.Tensor]] = {0: [], 1: []}

    for group_id, indices in enumerate(group_indices):
        g_full = group_loss_gradient(
            model,
            data=data,
            indices=indices,
            task=task,
            device=device,
            param_names=param_names,
            mode="full",
        )
        full.append(g_full)
        for modality_id in (0, 1):
            g_mod = group_loss_gradient(
                model,
                data=data,
                indices=indices,
                task=task,
                device=device,
                param_names=param_names,
                mode="modality",
                modality=modality_id,
            )
            g_masked = group_loss_gradient(
                model,
                data=data,
                indices=indices,
                task=task,
                device=device,
                param_names=param_names,
                mode="masked",
                modality=modality_id,
            )
            modality[modality_id].append(g_mod)
            masked_delta[modality_id].append(g_masked - g_full)
        if (group_id + 1) % 5 == 0 or group_id + 1 == len(group_indices):
            print(f"  gradients {group_id + 1}/{len(group_indices)} groups", flush=True)

    return {
        "full": torch.stack(full).numpy().astype(np.float32),
        "modality": {str(key): torch.stack(value).numpy().astype(np.float32) for key, value in modality.items()},
        "masked_delta": {str(key): torch.stack(value).numpy().astype(np.float32) for key, value in masked_delta.items()},
    }


def build_fisher_basis(cache: dict[str, Any], *, basis: str) -> np.ndarray:
    if basis == "full_group":
        return np.asarray(cache["full"], dtype=np.float64)
    if basis == "group_modality":
        return np.concatenate(
            [
                np.asarray(cache["modality"]["0"], dtype=np.float64),
                np.asarray(cache["modality"]["1"], dtype=np.float64),
            ],
            axis=0,
        )
    raise ValueError(f"unsupported fisher basis: {basis}")


def initialize_records(
    source_groups: Sequence[int],
    target_groups: Sequence[int],
    modalities: Sequence[int],
) -> list[dict[str, Any]]:
    records = []
    for source_group in source_groups:
        for source_modality in modalities:
            for target_group in target_groups:
                for target_modality in modalities:
                    records.append(
                        {
                            "source_group": int(source_group),
                            "source_modality": int(source_modality),
                            "target_group": int(target_group),
                            "target_modality": int(target_modality),
                        }
                    )
    return records


def add_static_scores(
    records: list[dict[str, Any]],
    *,
    final_cache: dict[str, Any],
    fisher_basis: np.ndarray,
    damping: float,
) -> None:
    basis = np.asarray(fisher_basis, dtype=np.float64)
    for record in records:
        target = np.asarray(final_cache["modality"][str(record["target_modality"])][record["target_group"]], dtype=np.float64)
        source = np.asarray(final_cache["masked_delta"][str(record["source_modality"])][record["source_group"]], dtype=np.float64)
        record["static_fisher"] = float(preconditioned_inner(target, source, basis, damping=float(damping)))
        record["static_dot"] = float(np.dot(target, source))


def preconditioned_inner(
    left: np.ndarray,
    right: np.ndarray,
    basis: np.ndarray,
    *,
    damping: float,
) -> float:
    damping = float(damping)
    if damping <= 0.0:
        raise ValueError("fisher damping must be positive")
    rows = np.asarray(basis, dtype=np.float64)
    if rows.size == 0:
        return float(np.dot(left, right) / damping)
    num_rows = max(int(rows.shape[0]), 1)
    u = rows / math.sqrt(num_rows)
    gram = u @ u.T
    middle = np.eye(gram.shape[0], dtype=np.float64) + gram / damping
    u_left = u @ left
    u_right = u @ right
    correction = u_left @ np.linalg.solve(middle, u_right)
    return float(np.dot(left, right) / damping - correction / (damping * damping))


def ensure_trajectory_checkpoints(
    trajectory_dir: Path,
    *,
    data: dict[str, Any],
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    seed: int,
    count: int,
    force: bool,
    fallback_checkpoint: Path | None,
) -> dict[str, Any]:
    manifest_path = trajectory_dir / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(Path(item["path"]).exists() for item in manifest.get("checkpoints", [])):
            return manifest
    if fallback_checkpoint is not None and not force:
        return {
            "mode": "final_checkpoint_fallback",
            "note": "This is not a true trajectory; pass --force-train-trajectory for TracIn.",
            "checkpoints": [
                {
                    "path": str(fallback_checkpoint),
                    "step": None,
                    "fraction": 1.0,
                    "lr": float(model_args.lr),
                }
            ],
        }
    return train_trajectory_checkpoints(
        trajectory_dir,
        data=data,
        task=task,
        model_type=model_type,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        seed=seed,
        count=count,
    )


def train_trajectory_checkpoints(
    trajectory_dir: Path,
    *,
    data: dict[str, Any],
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    seed: int,
    count: int,
) -> dict[str, Any]:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))
    model = build_model(model_type, dim_x, dim_y, num_classes, model_args).to(model_args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_args.lr), weight_decay=1e-4)
    loader = make_loader(
        data["train_x"],
        data["train_y"],
        data["train_labels"],
        int(model_args.batch_size),
        True,
    )
    total_steps = max(1, int(model_args.epochs) * len(loader))
    fractions = [float(i) / float(max(int(count), 1)) for i in range(1, max(int(count), 1) + 1)]
    checkpoints: list[dict[str, Any]] = []
    next_checkpoint = 0
    global_step = 0
    epoch_losses: list[float] = []

    print(
        f"Training full-data trajectory for {model_args.epochs} epochs "
        f"({total_steps} optimizer steps)...",
        flush=True,
    )
    for epoch in range(int(model_args.epochs)):
        running_loss = 0.0
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
            global_step += 1
            batch_size = int(bl.shape[0])
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            frac_now = global_step / total_steps
            while next_checkpoint < len(fractions) and frac_now >= fractions[next_checkpoint]:
                path = trajectory_dir / f"checkpoint_{next_checkpoint + 1:02d}.pt"
                save_checkpoint(path, model, model_type, dim_x, dim_y, num_classes, model_args, task, seed)
                checkpoints.append(
                    {
                        "path": str(path),
                        "step": int(global_step),
                        "fraction": float(fractions[next_checkpoint]),
                        "lr": float(model_args.lr),
                    }
                )
                next_checkpoint += 1
        epoch_losses.append(running_loss / max(total_samples, 1))
        print(f"  epoch {epoch + 1}/{model_args.epochs}: loss={epoch_losses[-1]:.6f}", flush=True)

    if not checkpoints or checkpoints[-1]["fraction"] < 1.0:
        path = trajectory_dir / f"checkpoint_{len(checkpoints) + 1:02d}.pt"
        save_checkpoint(path, model, model_type, dim_x, dim_y, num_classes, model_args, task, seed)
        checkpoints.append({"path": str(path), "step": int(global_step), "fraction": 1.0, "lr": float(model_args.lr)})
    manifest = {
        "mode": "trained_full_trajectory",
        "seed": int(seed),
        "epochs": int(model_args.epochs),
        "batch_size": int(model_args.batch_size),
        "lr": float(model_args.lr),
        "epoch_losses": [float(value) for value in epoch_losses],
        "checkpoints": checkpoints,
    }
    (trajectory_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


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
                "model": model_type,
                "dim_x": int(dim_x),
                "dim_y": int(dim_y),
                "num_classes": int(num_classes),
                "dropout": float(model_args.dropout),
                "hidden_dim": int(model_args.hidden_dim),
                "proj_dim": int(model_args.proj_dim),
                "num_heads": int(model_args.num_heads),
                "num_layers": int(model_args.num_layers),
            },
            "task": task,
            "seed": int(seed),
        },
        path,
    )


def add_tracin_scores(
    records: list[dict[str, Any]],
    *,
    trajectory_manifest: dict[str, Any],
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    device: str,
    param_names: Sequence[str],
) -> None:
    for record in records:
        record["tracin"] = 0.0

    for item in trajectory_manifest["checkpoints"]:
        model = load_model(
            Path(item["path"]),
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=device,
        )
        cache = build_group_gradient_cache(
            model,
            data=data,
            group_indices=group_indices,
            task=task,
            device=device,
            param_names=param_names,
        )
        lr = float(item.get("lr", model_args.lr))
        for record in records:
            target = np.asarray(cache["modality"][str(record["target_modality"])][record["target_group"]], dtype=np.float64)
            source = np.asarray(cache["masked_delta"][str(record["source_modality"])][record["source_group"]], dtype=np.float64)
            record["tracin"] = float(record["tracin"] + lr * np.dot(target, source))


def add_ntk_projection_scores(
    records: list[dict[str, Any]],
    *,
    trajectory_manifest: dict[str, Any],
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    device: str,
    param_names: Sequence[str],
    top_k: int,
) -> None:
    for record in records:
        record["ntk_projection_path"] = []

    source_interventions = unique_source_interventions(records)
    for item in trajectory_manifest["checkpoints"]:
        model = load_model(
            Path(item["path"]),
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=device,
        )
        cache = build_group_gradient_cache(
            model,
            data=data,
            group_indices=group_indices,
            task=task,
            device=device,
            param_names=param_names,
        )
        full_rows = np.asarray(cache["full"], dtype=np.float64)
        baseline_basis = top_right_singular_basis(full_rows, top_k=top_k)
        masked_basis_by_source: dict[tuple[int, int], np.ndarray] = {}
        for source_group, source_modality in source_interventions:
            masked_rows = full_rows.copy()
            masked_rows[source_group] = (
                full_rows[source_group]
                + np.asarray(cache["masked_delta"][str(source_modality)][source_group], dtype=np.float64)
            )
            masked_basis_by_source[(source_group, source_modality)] = top_right_singular_basis(
                masked_rows,
                top_k=top_k,
            )

        for record in records:
            target = np.asarray(cache["modality"][str(record["target_modality"])][record["target_group"]], dtype=np.float64)
            source_key = (int(record["source_group"]), int(record["source_modality"]))
            base_energy = projection_energy(target, baseline_basis, normalize=True)
            masked_energy = projection_energy(target, masked_basis_by_source[source_key], normalize=True)
            record["ntk_projection_path"].append(
                {
                    "fraction": float(item.get("fraction", 1.0)),
                    "step": item.get("step"),
                    "base_energy": float(base_energy),
                    "masked_source_energy": float(masked_energy),
                    "delta_energy": float(masked_energy - base_energy),
                }
            )

    for record in records:
        path = record.get("ntk_projection_path", [])
        deltas = np.asarray([point["delta_energy"] for point in path], dtype=np.float64)
        if deltas.size == 0:
            continue
        record["ntk_projection_delta_final"] = float(deltas[-1])
        record["ntk_projection_delta_abs_max"] = float(np.max(np.abs(deltas)))


def top_right_singular_basis(rows: np.ndarray, *, top_k: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {rows.shape}")
    rank = min(max(int(top_k), 1), rows.shape[0], rows.shape[1])
    _, _, vh = np.linalg.svd(rows, full_matrices=False)
    return vh[:rank].T.copy()


def projection_energy(vector: np.ndarray, basis: np.ndarray, *, normalize: bool) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.size == 0:
        return 0.0
    coeff = basis.T @ vector
    energy = float(np.dot(coeff, coeff))
    if normalize:
        denom = float(np.dot(vector, vector))
        if denom <= 1e-24:
            return 0.0
        return energy / denom
    return energy


def unique_source_interventions(records: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for record in records:
        key = (int(record["source_group"]), int(record["source_modality"]))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def add_ground_truth_scores(
    records: list[dict[str, Any]],
    *,
    source_interventions: Sequence[tuple[int, int]],
    output_dir: Path,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    seed: int,
    base_model: torch.nn.Module,
    device: str,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_reliance = compute_target_reliance(
        base_model,
        data=data,
        group_indices=group_indices,
        task=task,
        device=device,
    )
    trained: list[dict[str, Any]] = []
    for source_group, source_modality in source_interventions:
        ckpt_path = output_dir / f"source_g{source_group:02d}_m{source_modality}.pt"
        if not ckpt_path.exists() or force:
            print(f"  training GT intervention source group={source_group}, modality={source_modality}", flush=True)
            model = train_source_masked_model(
                data=data,
                group_indices=group_indices,
                source_group=source_group,
                source_modality=source_modality,
                task=task,
                model_type=model_type,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                seed=seed * 1_000_000 + source_group * 10 + source_modality,
            )
            save_checkpoint(ckpt_path, model, model_type, dim_x, dim_y, num_classes, model_args, task, seed)
        model = load_model(
            ckpt_path,
            model_type=model_type,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            device=device,
        )
        intervention_reliance = compute_target_reliance(
            model,
            data=data,
            group_indices=group_indices,
            task=task,
            device=device,
        )
        for record in records:
            if int(record["source_group"]) != source_group or int(record["source_modality"]) != source_modality:
                continue
            target_group = int(record["target_group"])
            target_modality = int(record["target_modality"])
            record["gt_cgmi"] = float(
                intervention_reliance[target_modality, target_group]
                - base_reliance[target_modality, target_group]
            )
        trained.append(
            {
                "source_group": int(source_group),
                "source_modality": int(source_modality),
                "checkpoint": str(ckpt_path),
            }
        )
    return {"mode": "logo_source_modality_retraining", "interventions": trained}


def compute_target_reliance(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: str,
) -> np.ndarray:
    reliance = np.empty((2, len(group_indices)), dtype=np.float64)
    for group_id, indices in enumerate(group_indices):
        full = group_loss_value(
            model,
            data=data,
            indices=indices,
            task=task,
            device=device,
            masked_modality=None,
        )
        for modality in (0, 1):
            masked = group_loss_value(
                model,
                data=data,
                indices=indices,
                task=task,
                device=device,
                masked_modality=modality,
            )
            reliance[modality, group_id] = masked - full
    return reliance


def group_loss_value(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    task: str,
    device: str,
    masked_modality: int | None,
) -> float:
    index = torch.as_tensor(indices, dtype=torch.long)
    bx = data["train_x"][index].to(device)
    by = data["train_y"][index].to(device)
    bl = data["train_labels"][index].to(device)
    if masked_modality is not None:
        bx = bx.clone()
        by = by.clone()
        if masked_modality == 0:
            bx.zero_()
        elif masked_modality == 1:
            by.zero_()
        else:
            raise ValueError(f"unsupported modality: {masked_modality}")
    model.eval()
    with torch.inference_mode():
        loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
    return float(loss.detach().cpu().item())


def train_source_masked_model(
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    source_group: int,
    source_modality: int,
    task: str,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    seed: int,
) -> torch.nn.Module:
    set_seed(int(seed))
    train_x = data["train_x"].clone()
    train_y = data["train_y"].clone()
    index = torch.as_tensor(group_indices[source_group], dtype=torch.long)
    if source_modality == 0:
        train_x[index] = 0.0
    elif source_modality == 1:
        train_y[index] = 0.0
    else:
        raise ValueError(f"unsupported modality: {source_modality}")
    loader = make_loader(train_x, train_y, data["train_labels"], int(model_args.batch_size), True)
    model = build_model(model_type, dim_x, dim_y, num_classes, model_args).to(model_args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_args.lr), weight_decay=1e-4)
    train_epochs(model, loader, optimizer, task=task, device=str(model_args.device), epochs=int(model_args.epochs))
    model.eval()
    return model


def train_epochs(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    task: str,
    device: str,
    epochs: int,
) -> list[float]:
    epoch_losses: list[float] = []
    for epoch in range(int(epochs)):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for bx, by, bl in loader:
            bx = bx.to(device)
            by = by.to(device)
            bl = bl.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
            loss.backward()
            optimizer.step()
            batch_size = int(bl.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        epoch_losses.append(total_loss / max(total_samples, 1))
        print(f"    epoch {epoch + 1}/{epochs}: loss={epoch_losses[-1]:.6f}", flush=True)
    return epoch_losses


def gradient_stats(cache: dict[str, Any]) -> dict[str, Any]:
    full = np.asarray(cache["full"], dtype=np.float64)
    modality = np.concatenate(
        [np.asarray(cache["modality"]["0"], dtype=np.float64), np.asarray(cache["modality"]["1"], dtype=np.float64)],
        axis=0,
    )
    delta = np.concatenate(
        [
            np.asarray(cache["masked_delta"]["0"], dtype=np.float64),
            np.asarray(cache["masked_delta"]["1"], dtype=np.float64),
        ],
        axis=0,
    )
    return {
        "full_group_norm_mean": float(np.linalg.norm(full, axis=1).mean()),
        "full_group_norm_max": float(np.linalg.norm(full, axis=1).max()),
        "modality_norm_mean": float(np.linalg.norm(modality, axis=1).mean()),
        "masked_delta_norm_mean": float(np.linalg.norm(delta, axis=1).mean()),
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_records": int(len(records))}
    gt = np.asarray([record.get("gt_cgmi", np.nan) for record in records], dtype=np.float64)
    valid_gt = np.isfinite(gt)
    for key in ("static_fisher", "static_dot", "tracin"):
        values = np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64)
        valid = np.isfinite(values)
        summary[f"{key}_available"] = int(valid.sum())
        if valid.any():
            summary[f"{key}_mean"] = float(values[valid].mean())
            summary[f"{key}_std"] = float(values[valid].std())
        if valid_gt.any() and np.logical_and(valid_gt, valid).sum() >= 2:
            mask = np.logical_and(valid_gt, valid)
            summary[f"{key}_spearman_to_gt"] = spearmanr(values[mask], gt[mask])
            summary[f"{key}_pearson_to_gt"] = pearsonr(values[mask], gt[mask])
    for key in ("ntk_projection_delta_final", "ntk_projection_delta_abs_max"):
        values = np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64)
        valid = np.isfinite(values)
        summary[f"{key}_available"] = int(valid.sum())
        if valid.any():
            summary[f"{key}_mean"] = float(values[valid].mean())
            summary[f"{key}_std"] = float(values[valid].std())
        if valid_gt.any() and np.logical_and(valid_gt, valid).sum() >= 2:
            mask = np.logical_and(valid_gt, valid)
            summary[f"{key}_spearman_to_gt"] = spearmanr(values[mask], gt[mask])
            summary[f"{key}_pearson_to_gt"] = pearsonr(values[mask], gt[mask])
    summary["gt_available"] = int(valid_gt.sum())
    return summary


def pearsonr(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    x = np.asarray(list(lhs), dtype=np.float64)
    y = np.asarray(list(rhs), dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


if __name__ == "__main__":
    main()
