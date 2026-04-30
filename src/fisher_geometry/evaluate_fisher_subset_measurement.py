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
from torch.func import functional_call, grad, vmap

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_geometry.data import load_dataset
from fisher_geometry.experiment import build_model
from fisher_geometry.fisher_metrics import compute_loss, fisher_eigen
from src.slq import SLQ


DEFAULT_SUBSET_RUN = "kmeans_grouped25_subset100_data80_mod80"
DEFAULT_FULL_RUN = "kmeans_grouped25_full_data_train"
DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "results"
    / "fisher_subset_measurement_humor_kmeans_data80_mod80.json"
)
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_JSON.with_suffix(".md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate LDS subset checkpoints with inverse-free Fisher measurements "
            "defined in src/fisher_geometry/fisher_test.md."
        )
    )
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset-run", default=DEFAULT_SUBSET_RUN)
    parser.add_argument("--full-run", default=DEFAULT_FULL_RUN)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--max-subsets", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=0,
        help="Training samples per group for Fisher construction; 0 uses every train sample.",
    )
    parser.add_argument(
        "--test-gradient-samples",
        type=int,
        default=0,
        help="Test samples for the Metric-1 projection gradient; 0 uses the full test split.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--grad-batch-size", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--gradient-unit", choices=["group", "sample"], default="group")
    parser.add_argument("--eigensolver", choices=["dual", "slq"], default="slq")
    parser.add_argument("--slq-lanczos-steps", type=int, default=32)
    parser.add_argument("--slq-reorthogonalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--parameter-scope",
        choices=["all", "classifier", "fusion"],
        default="all",
        help="Model parameters used as theta for Fisher gradients.",
    )
    parser.add_argument(
        "--source-group-mode",
        choices=["omitted", "all"],
        default="omitted",
        help="For subset rows, analyze omitted source groups or all groups.",
    )
    parser.add_argument("--source-group-limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    args.device = _resolve_torch_device(str(args.device))
    np.random.seed(int(args.sample_seed))
    torch.manual_seed(int(args.sample_seed))
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(args.sample_seed))

    subset_payload = _load_payload(args.result_root / args.subset_run, args.dataset, args.seed)
    full_payload = _load_payload(args.result_root / args.full_run, args.dataset, args.seed)
    config = dict(subset_payload.get("config", {}))
    feature_device = str(args.feature_device or ("cpu" if args.device == "cpu" else config.get("feature_device", args.device)))

    print(f"[{args.dataset}] seed={args.seed}: loading data...", flush=True)
    data = load_dataset(
        args.dataset,
        args.seed,
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=feature_device,
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    group_indices = [
        [int(index) for index in indices]
        for indices in subset_payload["group_partition"]["group_indices"]
    ]
    num_groups = int(subset_payload["group_partition"]["num_groups"])
    if num_groups != len(group_indices):
        raise ValueError("group_partition num_groups does not match group_indices")

    sample_indices, sample_group_ids = _balanced_group_sample(
        group_indices,
        samples_per_group=int(args.samples_per_group),
        seed=int(args.sample_seed),
    )
    test_gradient_indices = _sample_split_indices(
        int(data["test_labels"].shape[0]),
        max_samples=int(args.test_gradient_samples),
        seed=int(args.sample_seed) + 13,
    )

    subset_runs = sorted(
        subset_payload["runs"],
        key=lambda run: int(run.get("sample_index", 0)),
    )[: int(args.max_subsets)]
    if not subset_runs:
        raise ValueError(f"no subset runs found in {args.result_root / args.subset_run}")
    full_runs = sorted(full_payload["runs"], key=lambda run: int(run.get("sample_index", 0)))
    if not full_runs:
        raise ValueError(f"no full-data runs found in {args.result_root / args.full_run}")

    all_group_set = set(range(num_groups))
    subset_missing_groups = sorted(
        {
            group_id
            for run in subset_runs
            for group_id in all_group_set.difference({int(g) for g in run.get("group_ids", [])})
        }
    )
    rows: list[dict[str, Any]] = []
    eval_runs: list[tuple[str, dict[str, Any], list[int]]] = []
    for run in subset_runs:
        kept_groups = sorted({int(g) for g in run.get("group_ids", [])})
        omitted_groups = sorted(all_group_set.difference(kept_groups))
        source_groups = omitted_groups if args.source_group_mode == "omitted" else list(range(num_groups))
        eval_runs.append(("subset", run, source_groups))

    full_source_groups = subset_missing_groups if args.source_group_mode == "omitted" else list(range(num_groups))
    eval_runs.append(("full", full_runs[0], full_source_groups))

    for row_index, (checkpoint_kind, run, source_groups) in enumerate(eval_runs, start=1):
        if args.source_group_limit is not None:
            source_groups = source_groups[: int(args.source_group_limit)]
        checkpoint_path = _checkpoint_path(run)
        print(
            f"[{checkpoint_kind}] {row_index}/{len(eval_runs)}: loading {checkpoint_path}",
            flush=True,
        )
        model, checkpoint_payload = _load_model_from_checkpoint(checkpoint_path, device=str(args.device))
        selected_param_names = _selected_param_names(model, args.parameter_scope)
        if not selected_param_names:
            raise ValueError(f"no parameters selected for scope={args.parameter_scope}")

        row = evaluate_checkpoint(
            model,
            data=data,
            task=str(data["task"]),
            sample_indices=sample_indices,
            sample_group_ids=sample_group_ids,
            test_gradient_indices=test_gradient_indices,
            source_groups=source_groups,
            selected_param_names=selected_param_names,
            top_k=int(args.top_k),
            gradient_unit=str(args.gradient_unit),
            eigensolver=str(args.eigensolver),
            slq_lanczos_steps=int(args.slq_lanczos_steps),
            slq_reorthogonalize=bool(args.slq_reorthogonalize),
            device=str(args.device),
            grad_batch_size=int(args.grad_batch_size),
        )
        row.update(
            {
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_dataset": checkpoint_payload.get("dataset"),
                "checkpoint_sample_index": int(run.get("sample_index", 0)),
                "checkpoint_test_loss": _optional_float(run.get("test_loss")),
                "checkpoint_test_acc": _optional_float(run.get("test_acc")),
                "selected_group_count": int(run.get("selected_group_count", len(run.get("group_ids", [])))),
                "selected_sample_count": int(run.get("selected_sample_count", len(run.get("data_indices", [])))),
                "effective_sample_count": int(run.get("effective_sample_count", run.get("selected_sample_count", 0))),
                "kept_group_ids": [int(group_id) for group_id in run.get("group_ids", [])],
                "source_group_ids": [int(group_id) for group_id in source_groups],
                "parameter_scope": str(args.parameter_scope),
                "num_selected_parameters": int(
                    sum(model.state_dict()[name].numel() for name in selected_param_names)
                ),
            }
        )
        rows.append(row)
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary = summarize_rows(rows)
    output = {
        "config": {
            "dataset": str(args.dataset),
            "seed": int(args.seed),
            "subset_run": str(args.subset_run),
            "full_run": str(args.full_run),
            "result_root": str(args.result_root),
            "max_subsets": int(args.max_subsets),
            "device": str(args.device),
            "samples_per_group": int(args.samples_per_group),
            "test_gradient_samples": int(args.test_gradient_samples),
            "fisher_gradient_source": "train",
            "projection_gradient_source": "test",
            "top_k": int(args.top_k),
            "grad_batch_size": int(args.grad_batch_size),
            "gradient_unit": str(args.gradient_unit),
            "eigensolver": str(args.eigensolver),
            "slq_lanczos_steps": int(args.slq_lanczos_steps),
            "slq_reorthogonalize": bool(args.slq_reorthogonalize),
            "sample_seed": int(args.sample_seed),
            "parameter_scope": str(args.parameter_scope),
            "source_group_mode": str(args.source_group_mode),
            "source_group_limit": args.source_group_limit,
            "measurement_definition": "src/fisher_geometry/fisher_test.md",
        },
        "summary": summary,
        "conclusions": build_conclusions(summary),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"\nSaved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def _load_payload(result_dir: Path, dataset: str, seed: int) -> dict[str, Any]:
    path = result_dir / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing result payload: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "runs" not in payload or "group_partition" not in payload:
        raise ValueError(f"invalid LDS result payload: {path}")
    return payload


def _resolve_torch_device(requested: str) -> str:
    if not requested.startswith("cuda"):
        return requested
    try:
        _ = torch.empty(1, device=requested)
        return requested
    except Exception as exc:
        print(f"CUDA device {requested} unavailable ({exc}); falling back to cpu.", flush=True)
        return "cpu"


def _checkpoint_path(run: dict[str, Any]) -> Path:
    checkpoint_path = str(run.get("checkpoint_path") or "")
    if not checkpoint_path:
        checkpoint_paths = run.get("checkpoint_paths", [])
        if not checkpoint_paths:
            raise ValueError("run is missing checkpoint_path/checkpoint_paths")
        checkpoint_path = str(checkpoint_paths[0])
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return path


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(payload["model_config"])
    model_args = SimpleNamespace(
        dropout=float(model_config.get("dropout", 0.1)),
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        proj_dim=int(model_config.get("proj_dim", 256)),
        num_heads=int(model_config.get("num_heads", 4)),
        num_layers=int(model_config.get("num_layers", 2)),
    )
    model = build_model(
        str(model_config["model"]),
        int(model_config["dim_x"]),
        int(model_config["dim_y"]),
        int(model_config["num_classes"]),
        model_args,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _selected_param_names(model: torch.nn.Module, scope: str) -> tuple[str, ...]:
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if scope == "all":
        return tuple(names)
    if scope == "classifier":
        return tuple(name for name in names if name.startswith("classifier."))
    if scope == "fusion":
        prefixes = (
            "fusion_",
            "proj_",
            "shared_mlp.",
            "layers.",
            "pool_norm.",
            "classifier.",
            "cls_",
            "null_embed",
        )
        return tuple(name for name in names if name.startswith(prefixes))
    raise ValueError(f"unknown parameter scope: {scope}")


def _balanced_group_sample(
    group_indices: Sequence[Sequence[int]],
    *,
    samples_per_group: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    sampled_groups: list[int] = []
    for group_id, indices in enumerate(group_indices):
        values = np.asarray(list(indices), dtype=np.int64)
        if values.size == 0:
            continue
        count = int(values.size) if int(samples_per_group) <= 0 else min(int(samples_per_group), int(values.size))
        chosen = values if count == int(values.size) else rng.choice(values, size=count, replace=False)
        sampled_indices.extend(int(index) for index in chosen.tolist())
        sampled_groups.extend([int(group_id)] * count)
    if not sampled_indices:
        raise ValueError("balanced group sampling produced no samples")
    order = np.argsort(np.asarray(sampled_indices, dtype=np.int64), kind="mergesort")
    indices = torch.tensor([sampled_indices[int(i)] for i in order], dtype=torch.long)
    groups = torch.tensor([sampled_groups[int(i)] for i in order], dtype=torch.long)
    return indices, groups


def _sample_split_indices(num_samples: int, *, max_samples: int, seed: int) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("cannot sample from an empty split")
    if int(max_samples) <= 0 or int(max_samples) >= int(num_samples):
        return torch.arange(int(num_samples), dtype=torch.long)
    rng = np.random.default_rng(seed)
    values = rng.choice(np.arange(int(num_samples), dtype=np.int64), size=int(max_samples), replace=False)
    values.sort()
    return torch.from_numpy(values).long()


def evaluate_checkpoint(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    task: str,
    sample_indices: torch.Tensor,
    sample_group_ids: torch.Tensor,
    test_gradient_indices: torch.Tensor,
    source_groups: Sequence[int],
    selected_param_names: Sequence[str],
    top_k: int,
    gradient_unit: str,
    eigensolver: str,
    slq_lanczos_steps: int,
    slq_reorthogonalize: bool,
    device: str,
    grad_batch_size: int,
) -> dict[str, Any]:
    x = data["train_x"][sample_indices].clone()
    y = data["train_y"][sample_indices].clone()
    labels = data["train_labels"][sample_indices].clone()
    if gradient_unit == "group":
        base_grads, base_grad_group_ids = per_group_average_selected_param_gradients(
            model,
            x,
            y,
            labels,
            sample_group_ids,
            task=task,
            selected_param_names=selected_param_names,
            device=device,
        )
    else:
        base_grads = per_example_selected_param_gradients(
            model,
            x,
            y,
            labels,
            task=task,
            selected_param_names=selected_param_names,
            device=device,
            grad_batch_size=grad_batch_size,
        )
        base_grad_group_ids = sample_group_ids
    full_fisher = fisher_top_eigen(
        base_grads,
        top_k=top_k,
        eigensolver=eigensolver,
        device=device,
        slq_lanczos_steps=slq_lanczos_steps,
        slq_reorthogonalize=slq_reorthogonalize,
    )
    full_vecs = full_fisher["top_k_vecs"].float()
    full_top_mass_abs = _topk_abs_mass(full_fisher["eigenvalues"], top_k)

    test_x = data["test_x"][test_gradient_indices].clone()
    test_y = data["test_y"][test_gradient_indices].clone()
    test_labels = data["test_labels"][test_gradient_indices].clone()
    test_grad = average_selected_param_gradient(
        model,
        test_x,
        test_y,
        test_labels,
        task=task,
        selected_param_names=selected_param_names,
        device=device,
        batch_size=grad_batch_size,
    )

    pair_rows: list[dict[str, Any]] = []
    modality_names = ("x", "y")
    for source_group in source_groups:
        for modality_index, modality_name in enumerate(modality_names):
            px = x.clone()
            py = y.clone()
            source_mask = sample_group_ids == int(source_group)
            if modality_index == 0:
                px[source_mask] = 0.0
            else:
                py[source_mask] = 0.0
            if gradient_unit == "group":
                perturbed_grads, _ = per_group_average_selected_param_gradients(
                    model,
                    px,
                    py,
                    labels,
                    sample_group_ids,
                    task=task,
                    selected_param_names=selected_param_names,
                    device=device,
                )
            else:
                perturbed_grads = per_example_selected_param_gradients(
                    model,
                    px,
                    py,
                    labels,
                    task=task,
                    selected_param_names=selected_param_names,
                    device=device,
                    grad_batch_size=grad_batch_size,
                )
            perturbed_fisher = fisher_top_eigen(
                perturbed_grads,
                top_k=top_k,
                eigensolver=eigensolver,
                device=device,
                slq_lanczos_steps=slq_lanczos_steps,
                slq_reorthogonalize=slq_reorthogonalize,
            )
            perturbed_vecs = perturbed_fisher["top_k_vecs"].float()
            alignment_distance = subspace_alignment_distance(full_vecs, perturbed_vecs, top_k=top_k)
            topk_mass_decay = full_top_mass_abs - _topk_abs_mass(perturbed_fisher["eigenvalues"], top_k)
            projection_loss = spectral_projection_loss(full_vecs, perturbed_vecs, test_grad, top_k=top_k)
            pair_rows.append(
                {
                    "source_group": int(source_group),
                    "source_modality": modality_name,
                    "projection_gradient_source": "test",
                    "projection_gradient_samples": int(test_gradient_indices.numel()),
                    "projection_loss": float(projection_loss),
                    "alignment_distance": float(alignment_distance),
                    "topk_spectral_mass_decay": float(topk_mass_decay),
                    "full_topk_spectral_mass": float(full_top_mass_abs),
                    "perturbed_topk_spectral_mass": float(_topk_abs_mass(perturbed_fisher["eigenvalues"], top_k)),
                    "full_trace": float(full_fisher["trace"]),
                    "perturbed_trace": float(perturbed_fisher["trace"]),
                    "full_rank": int(full_fisher["rank"]),
                    "perturbed_rank": int(perturbed_fisher["rank"]),
                }
            )

    return {
        "fisher_samples_used": int(sample_indices.numel()),
        "projection_gradient_samples_used": int(test_gradient_indices.numel()),
        "fisher_gradient_rows": int(base_grads.shape[0]),
        "gradient_unit": str(gradient_unit),
        "fisher_gradient_source": "train",
        "projection_gradient_source": "test",
        "base_gradient_group_ids": [int(group_id) for group_id in base_grad_group_ids.tolist()],
        "top_k": int(top_k),
        "eigensolver": str(eigensolver),
        "num_source_group_modalities": int(len(pair_rows)),
        "mean_projection_loss": _safe_mean([row["projection_loss"] for row in pair_rows]),
        "mean_alignment_distance": _safe_mean([row["alignment_distance"] for row in pair_rows]),
        "mean_topk_spectral_mass_decay": _safe_mean([row["topk_spectral_mass_decay"] for row in pair_rows]),
        "std_projection_loss": _safe_std([row["projection_loss"] for row in pair_rows]),
        "std_alignment_distance": _safe_std([row["alignment_distance"] for row in pair_rows]),
        "std_topk_spectral_mass_decay": _safe_std([row["topk_spectral_mass_decay"] for row in pair_rows]),
        "pair_measurements": pair_rows,
    }


def per_example_selected_param_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    *,
    task: str,
    selected_param_names: Sequence[str],
    device: str,
    grad_batch_size: int,
) -> torch.Tensor:
    model.eval()
    all_params = {name: param.detach() for name, param in model.named_parameters()}
    selected_param_names = tuple(selected_param_names)
    selected_params = tuple(all_params[name] for name in selected_param_names)

    def loss_single(selected_values: tuple[torch.Tensor, ...], bx: torch.Tensor, by: torch.Tensor, bl: torch.Tensor) -> torch.Tensor:
        params = dict(all_params)
        params.update(dict(zip(selected_param_names, selected_values)))
        logits = functional_call(
            model,
            params,
            (bx.unsqueeze(0), by.unsqueeze(0)),
            kwargs={"use_x": True, "use_y": True},
        )
        return compute_loss(task, logits, bl.unsqueeze(0))

    grad_fn = grad(loss_single)
    vmapped_grad = vmap(grad_fn, in_dims=(None, 0, 0, 0))
    all_grads: list[torch.Tensor] = []
    count = int(x.shape[0])
    for start in range(0, count, int(grad_batch_size)):
        end = min(start + int(grad_batch_size), count)
        bx = x[start:end].to(device)
        by = y[start:end].to(device)
        bl = labels[start:end].to(device)
        per_sample = vmapped_grad(selected_params, bx, by, bl)
        flat = torch.cat([grad_value.reshape(end - start, -1) for grad_value in per_sample], dim=1)
        all_grads.append(flat.detach().cpu())
    return torch.cat(all_grads, dim=0)


def per_group_average_selected_param_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    task: str,
    selected_param_names: Sequence[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] != group_ids.shape[0]:
        raise ValueError("x and group_ids must have the same length")
    model.eval()
    params_by_name = dict(model.named_parameters())
    selected_params = [params_by_name[name] for name in selected_param_names]
    unique_groups = torch.unique(group_ids.detach().cpu(), sorted=True)
    group_grads: list[torch.Tensor] = []
    for group_id in unique_groups.tolist():
        mask = group_ids == int(group_id)
        if not bool(mask.any()):
            continue
        bx = x[mask].to(device)
        by = y[mask].to(device)
        bl = labels[mask].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(bx, by, use_x=True, use_y=True)
        loss = compute_loss(task, logits, bl)
        grads = torch.autograd.grad(
            loss,
            selected_params,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        flat_parts = [
            torch.zeros_like(param).reshape(-1) if grad_value is None else grad_value.reshape(-1)
            for grad_value, param in zip(grads, selected_params)
        ]
        group_grads.append(torch.cat(flat_parts).detach().cpu())
    if not group_grads:
        raise ValueError("no group-average gradients were computed")
    return torch.stack(group_grads, dim=0), unique_groups.to(dtype=torch.long)


def average_selected_param_gradient(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    *,
    task: str,
    selected_param_names: Sequence[str],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    if x.shape[0] == 0:
        raise ValueError("cannot compute an average gradient from zero samples")
    model.eval()
    params_by_name = dict(model.named_parameters())
    selected_params = [params_by_name[name] for name in selected_param_names]
    total = int(x.shape[0])
    accum: torch.Tensor | None = None
    for start in range(0, total, max(int(batch_size), 1)):
        end = min(start + max(int(batch_size), 1), total)
        bx = x[start:end].to(device)
        by = y[start:end].to(device)
        bl = labels[start:end].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(bx, by, use_x=True, use_y=True)
        loss = compute_loss(task, logits, bl)
        grads = torch.autograd.grad(
            loss,
            selected_params,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        flat = torch.cat(
            [
                torch.zeros_like(param).reshape(-1) if grad_value is None else grad_value.reshape(-1)
                for grad_value, param in zip(grads, selected_params)
            ]
        ).detach().cpu()
        weight = float(end - start) / float(total)
        accum = flat * weight if accum is None else accum + flat * weight
    assert accum is not None
    return accum


def fisher_top_eigen(
    grad_matrix: torch.Tensor,
    *,
    top_k: int,
    eigensolver: str,
    device: str,
    slq_lanczos_steps: int,
    slq_reorthogonalize: bool,
) -> dict[str, Any]:
    if eigensolver == "dual":
        return fisher_eigen(grad_matrix, top_k=top_k)
    if eigensolver != "slq":
        raise ValueError(f"unknown eigensolver: {eigensolver}")

    g = grad_matrix.float()
    n, d = g.shape
    trace = float(g.pow(2).sum().item() / max(n, 1))
    if n == 0 or d == 0 or trace <= 0.0:
        return {
            "eigenvalues": torch.zeros(0, dtype=torch.float32),
            "top_k_vecs": torch.zeros((d, 0), dtype=torch.float32),
            "trace": trace,
            "lambda_max": 0.0,
            "rank": 0,
            "topk_mass": 0.0,
        }

    # Keep explicit gradient rows, but avoid dense EFIM construction.
    # The operator is Fv = G^T(Gv)/N, matching fisher_eigen's EFIM scaling.
    g_device = g.to(device)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return g_device.T @ (g_device @ v) / float(max(n, 1))

    slq = SLQ(matvec=matvec, dim=d, device=torch.device(device))
    lanczos_steps = max(int(top_k) + 2, int(slq_lanczos_steps))
    top_evals, top_vecs = slq.top_k_eigen(
        min(int(top_k), d),
        num_lanczos=min(lanczos_steps, d),
        reorthogonalize=bool(slq_reorthogonalize),
    )
    top_evals = torch.clamp(top_evals.detach().cpu().float(), min=0.0)
    top_vecs = top_vecs.detach().cpu().float()
    if not torch.isfinite(top_vecs).all():
        top_vecs = torch.zeros((d, 0), dtype=torch.float32)
        top_evals = torch.zeros(0, dtype=torch.float32)
    elif top_vecs.numel() > 0:
        top_vecs = torch.linalg.qr(top_vecs, mode="reduced").Q.float()
    return {
        "eigenvalues": top_evals,
        "top_k_vecs": top_vecs,
        "trace": trace,
        "lambda_max": float(top_evals[0].item()) if top_evals.numel() else 0.0,
        "rank": min(n, d),
        "topk_mass": float(top_evals.sum().item() / max(trace, 1e-12)),
    }


def spectral_projection_loss(
    full_vecs: torch.Tensor,
    perturbed_vecs: torch.Tensor,
    target_grad: torch.Tensor,
    *,
    top_k: int,
) -> float:
    k = min(int(top_k), full_vecs.shape[1], perturbed_vecs.shape[1])
    if k == 0:
        return 0.0
    g = target_grad.float().view(-1)
    full_energy = float((full_vecs[:, :k].T @ g).pow(2).sum().item())
    perturbed_energy = float((perturbed_vecs[:, :k].T @ g).pow(2).sum().item())
    return full_energy - perturbed_energy


def subspace_alignment_distance(
    full_vecs: torch.Tensor,
    perturbed_vecs: torch.Tensor,
    *,
    top_k: int,
) -> float:
    k = min(int(top_k), full_vecs.shape[1], perturbed_vecs.shape[1])
    if k == 0:
        return 0.0
    overlap = full_vecs[:, :k].float().T @ perturbed_vecs[:, :k].float()
    trace_pp = float(overlap.pow(2).sum().item())
    distance = 1.0 - trace_pp / max(k, 1)
    return float(min(max(distance, 0.0), 1.0))


def _topk_abs_mass(eigenvalues: torch.Tensor, top_k: int) -> float:
    if eigenvalues.numel() == 0 or top_k <= 0:
        return 0.0
    return float(eigenvalues[: min(int(top_k), int(eigenvalues.numel()))].sum().item())


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    subset_rows = [row for row in rows if row["checkpoint_kind"] == "subset"]
    full_rows = [row for row in rows if row["checkpoint_kind"] == "full"]
    return {
        "num_rows": int(len(rows)),
        "num_subset_rows": int(len(subset_rows)),
        "num_full_rows": int(len(full_rows)),
        "subset_mean_test_loss": _safe_mean([row["checkpoint_test_loss"] for row in subset_rows]),
        "subset_std_test_loss": _safe_std([row["checkpoint_test_loss"] for row in subset_rows]),
        "subset_mean_test_acc": _safe_mean([row["checkpoint_test_acc"] for row in subset_rows]),
        "subset_std_test_acc": _safe_std([row["checkpoint_test_acc"] for row in subset_rows]),
        "subset_mean_projection_loss": _safe_mean([row["mean_projection_loss"] for row in subset_rows]),
        "subset_std_projection_loss": _safe_std([row["mean_projection_loss"] for row in subset_rows]),
        "subset_mean_alignment_distance": _safe_mean([row["mean_alignment_distance"] for row in subset_rows]),
        "subset_std_alignment_distance": _safe_std([row["mean_alignment_distance"] for row in subset_rows]),
        "subset_mean_topk_spectral_mass_decay": _safe_mean([row["mean_topk_spectral_mass_decay"] for row in subset_rows]),
        "subset_std_topk_spectral_mass_decay": _safe_std([row["mean_topk_spectral_mass_decay"] for row in subset_rows]),
        "full_test_loss": full_rows[0]["checkpoint_test_loss"] if full_rows else float("nan"),
        "full_test_acc": full_rows[0]["checkpoint_test_acc"] if full_rows else float("nan"),
        "full_mean_projection_loss": full_rows[0]["mean_projection_loss"] if full_rows else float("nan"),
        "full_mean_alignment_distance": full_rows[0]["mean_alignment_distance"] if full_rows else float("nan"),
        "full_mean_topk_spectral_mass_decay": full_rows[0]["mean_topk_spectral_mass_decay"] if full_rows else float("nan"),
    }


def build_conclusions(summary: dict[str, Any]) -> list[str]:
    conclusions: list[str] = []
    mass_decay = float(summary["subset_mean_topk_spectral_mass_decay"])
    alignment = float(summary["subset_mean_alignment_distance"])
    projection = float(summary["subset_mean_projection_loss"])
    if mass_decay < 0.0:
        conclusions.append(
            "Negative top-k spectral mass decay is present on average, consistent with modality competition or suppression in the training Fisher geometry."
        )
    else:
        conclusions.append(
            "Top-k spectral mass decay is positive on average, so this run does not support the negative-decay competition signal."
        )
    conclusions.append(
        f"Average subspace alignment distance is {alignment:.6g}, measuring how much train-Fisher principal subspaces move after source-group modality masking."
    )
    conclusions.append(
        f"Average test-gradient projection loss is {projection:.6g}; this is the Metric-1 generalization signal because the probe gradient is computed on test data."
    )
    return conclusions


def to_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    summary = payload["summary"]
    conclusions = payload.get("conclusions", [])
    lines = [
        "# Fisher Subset Measurement Validation",
        "",
        f"- dataset: `{config['dataset']}` seed `{config['seed']}`",
        f"- subset run: `{config['subset_run']}`",
        f"- full run: `{config['full_run']}`",
        f"- measurement: `{config['measurement_definition']}`",
        f"- subset aggregation: first `{config['max_subsets']}` subset checkpoints, averaged",
        f"- Fisher gradient source: `{config['fisher_gradient_source']}`; projection gradient source: `{config['projection_gradient_source']}`",
        f"- top-k: `{config['top_k']}`, train samples/group: `{_format_samples_per_group(config['samples_per_group'])}`, test gradient samples: `{_format_samples_per_group(config['test_gradient_samples'])}`, parameter scope: `{config['parameter_scope']}`",
        f"- eigensolver: `{config['eigensolver']}`",
        "",
        "## Aggregate Result",
        "",
        "| metric | subset mean | subset std | full reference |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = [
        (
            "test_loss",
            summary["subset_mean_test_loss"],
            summary["subset_std_test_loss"],
            summary["full_test_loss"],
        ),
        (
            "test_acc",
            summary["subset_mean_test_acc"],
            summary["subset_std_test_acc"],
            summary["full_test_acc"],
        ),
        (
            "projection_loss",
            summary["subset_mean_projection_loss"],
            summary["subset_std_projection_loss"],
            summary["full_mean_projection_loss"],
        ),
        (
            "alignment_distance",
            summary["subset_mean_alignment_distance"],
            summary["subset_std_alignment_distance"],
            summary["full_mean_alignment_distance"],
        ),
        (
            "topk_spectral_mass_decay",
            summary["subset_mean_topk_spectral_mass_decay"],
            summary["subset_std_topk_spectral_mass_decay"],
            summary["full_mean_topk_spectral_mass_decay"],
        ),
    ]
    for metric_name, subset_mean, subset_std, full_value in metric_rows:
        lines.append(
            f"| {metric_name} | {_format_value(subset_mean)} | "
            f"{_format_value(subset_std)} | {_format_value(full_value)} |"
        )
    lines.extend(["", "## Conclusions", ""])
    for conclusion in conclusions:
        lines.append(f"- {conclusion}")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            "| field | value |",
            "|---|---:|",
            f"| subset checkpoints averaged | {summary['num_subset_rows']} |",
            f"| full checkpoints | {summary['num_full_rows']} |",
        ]
    )
    return "\n".join(lines) + "\n"


def _optional_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _safe_std(values: Sequence[float]) -> float:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr))


def _safe_min(values: Sequence[float]) -> float:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmin(arr))


def _safe_max(values: Sequence[float]) -> float:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr))


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _format_samples_per_group(value: Any) -> str:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return str(value)
    return "all" if ivalue <= 0 else str(ivalue)


if __name__ == "__main__":
    main()
