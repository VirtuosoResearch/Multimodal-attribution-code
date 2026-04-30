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
DEFAULT_OUTPUT_JSON = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "verification_results"
    / "humor_fisher_eigenspace"
    / "humor_seed0_fisher_eigenspace.json"
)
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_JSON.with_suffix(".md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate fisher_test.md on humor with existing kmeans25 groups. "
            "The empirical Fisher is built from group-average gradients at the "
            "fixed full-data checkpoint; perturbations mask one source group's "
            "one modality and replace that group's gradient row."
        )
    )
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--full-checkpoint", type=Path, default=DEFAULT_FULL_CHECKPOINT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--gradient-params", choices=["all", "last_layer", "fusion"], default="all")
    parser.add_argument("--group-weighting", choices=["uniform", "size"], default="uniform")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--top-records", type=int, default=12)
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
    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    task = str(data["task"])
    model_type = str(config.get("model", "cross_attention"))
    group_indices = [[int(index) for index in group] for group in payload["group_partition"]["group_indices"]]
    group_sizes = np.asarray([len(group) for group in group_indices], dtype=np.float64)
    group_weights = make_group_weights(group_sizes, mode=str(args.group_weighting))

    print(f"Loading checkpoint: {args.full_checkpoint}", flush=True)
    model = load_model(
        args.full_checkpoint,
        model_type=model_type,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        device=str(device),
    )
    named_params = select_named_parameters(model, mode=str(args.gradient_params))
    param_names = [name for name, _ in named_params]
    print(
        f"Selected {len(named_params)} tensors / "
        f"{sum(param.numel() for _, param in named_params):,} parameters.",
        flush=True,
    )

    print("Computing full and source-masked group-average gradients...", flush=True)
    full_rows, masked_rows_by_modality = compute_group_rows(
        model,
        data=data,
        group_indices=group_indices,
        task=task,
        device=str(device),
        param_names=param_names,
    )
    print("Computing test-set mean gradient...", flush=True)
    eval_batch_size = int(args.eval_batch_size or config.get("eval_batch_size") or config.get("batch_size", 4096))
    test_gradient = split_mean_gradient(
        model,
        x=data["test_x"],
        y=data["test_y"],
        labels=data["test_labels"],
        task=task,
        device=str(device),
        param_names=param_names,
        batch_size=eval_batch_size,
    )

    k = min(max(int(args.top_k), 1), len(group_indices), full_rows.shape[1])
    full_spectrum = top_eigensystem_from_rows(full_rows, top_k=k, row_weights=group_weights)
    records: list[dict[str, Any]] = []
    coupling_matrix = np.empty((len(group_indices) * 2, len(group_indices)), dtype=np.float64)

    for group_id in range(len(group_indices)):
        for modality in range(2):
            perturbed = full_rows.copy()
            perturbed[group_id] = masked_rows_by_modality[modality][group_id]
            masked_spectrum = top_eigensystem_from_rows(perturbed, top_k=k, row_weights=group_weights)
            alignment = subspace_alignment_distance(full_spectrum["vectors"], masked_spectrum["vectors"])
            mass_decay = float(np.sum(full_spectrum["values"]) - np.sum(masked_spectrum["values"]))
            test_projection_loss = spectral_projection_loss(
                test_gradient,
                full_spectrum["vectors"],
                masked_spectrum["vectors"],
                normalize=False,
            )
            test_projection_loss_norm = spectral_projection_loss(
                test_gradient,
                full_spectrum["vectors"],
                masked_spectrum["vectors"],
                normalize=True,
            )
            source_index = group_id * 2 + modality
            train_losses = []
            for target_group in range(len(group_indices)):
                train_loss = spectral_projection_loss(
                    full_rows[target_group],
                    full_spectrum["vectors"],
                    masked_spectrum["vectors"],
                    normalize=True,
                )
                coupling_matrix[source_index, target_group] = train_loss
                if target_group != group_id:
                    train_losses.append(float(train_loss))

            records.append(
                {
                    "source_group": int(group_id),
                    "source_modality": int(modality),
                    "group_size": int(group_sizes[group_id]),
                    "mass_decay": mass_decay,
                    "mass_growth": float(-mass_decay),
                    "alignment_distance": float(alignment),
                    "test_projection_loss": float(test_projection_loss),
                    "test_projection_loss_normalized": float(test_projection_loss_norm),
                    "train_cross_projection_loss_mean": float(np.mean(train_losses)),
                    "train_cross_projection_loss_abs_mean": float(np.mean(np.abs(train_losses))),
                    "train_cross_projection_loss_abs_max": float(np.max(np.abs(train_losses))),
                    "masked_top_eigenvalues": [float(value) for value in masked_spectrum["values"]],
                }
            )

    lds_effects = estimate_lds_modality_effects(payload, ridge_lambda=float(args.ridge_lambda))
    attach_lds_effects(records, lds_effects)
    summary = summarize(records, full_spectrum, coupling_matrix, top_records=int(args.top_records))
    output = {
        "config": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "payload_path": str(payload_path),
            "result_dir": str(args.result_dir),
            "full_checkpoint": str(args.full_checkpoint),
            "device": str(device),
            "feature_device": str(args.feature_device or config.get("feature_device", device)),
            "model": model_type,
            "gradient_params": str(args.gradient_params),
            "parameter_names": param_names,
            "num_parameters": int(full_rows.shape[1]),
            "num_groups": int(len(group_indices)),
            "group_weighting": str(args.group_weighting),
            "top_k": int(k),
            "ridge_lambda": float(args.ridge_lambda),
            "unit": "kmeans_group",
        },
        "full_spectrum": {
            "top_eigenvalues": [float(value) for value in full_spectrum["values"]],
            "topk_mass": float(np.sum(full_spectrum["values"])),
            "trace_group_fisher": float(full_spectrum["trace"]),
        },
        "gradient_stats": gradient_stats(full_rows, masked_rows_by_modality, test_gradient),
        "lds_effect_model": lds_effects["summary"],
        "summary": summary,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {args.output_json}", flush=True)
    if args.output_md is not None:
        print(f"Wrote {args.output_md}", flush=True)


def resolve_requested_device(requested: str) -> torch.device:
    if requested.startswith("cuda"):
        try:
            device = torch.device(requested)
            torch.cuda.get_device_properties(device)
            return device
        except Exception as exc:
            print(f"CUDA probe failed for {requested}: {exc}; falling back.", flush=True)
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


def select_named_parameters(model: torch.nn.Module, *, mode: str) -> list[tuple[str, torch.nn.Parameter]]:
    if mode == "all":
        named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    elif mode == "last_layer":
        named = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and name.startswith("classifier.")
        ]
    elif mode == "fusion":
        prefixes = ("layers.", "pool_norm.", "classifier.", "cls_x", "cls_y", "null_embed")
        named = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and (name.startswith(prefixes) or "fusion" in name or "shared_mlp" in name)
        ]
    else:
        raise ValueError(f"unsupported parameter mode: {mode}")
    if not named:
        raise ValueError(f"no trainable parameters selected for mode={mode!r}")
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


def group_mean_gradient(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    task: str,
    device: str,
    param_names: Sequence[str],
    masked_modality: int | None,
) -> torch.Tensor:
    params = parameters_by_name(model, param_names)
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
    model.zero_grad(set_to_none=True)
    logits = model(bx, by, use_x=True, use_y=True)
    loss = compute_loss(task, logits, bl)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return flatten_grads(grads, params)


def compute_group_rows(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: str,
    param_names: Sequence[str],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    full_rows = []
    masked_by_modality: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for group_id, indices in enumerate(group_indices):
        full_rows.append(
            group_mean_gradient(
                model,
                data=data,
                indices=indices,
                task=task,
                device=device,
                param_names=param_names,
                masked_modality=None,
            )
        )
        for modality in (0, 1):
            masked_by_modality[modality].append(
                group_mean_gradient(
                    model,
                    data=data,
                    indices=indices,
                    task=task,
                    device=device,
                    param_names=param_names,
                    masked_modality=modality,
                )
            )
        if (group_id + 1) % 5 == 0 or group_id + 1 == len(group_indices):
            print(f"  group gradients {group_id + 1}/{len(group_indices)}", flush=True)
    return (
        torch.stack(full_rows).numpy().astype(np.float32),
        {key: torch.stack(value).numpy().astype(np.float32) for key, value in masked_by_modality.items()},
    )


def split_mean_gradient(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    device: str,
    param_names: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    params = parameters_by_name(model, param_names)
    total = int(labels.shape[0])
    accumulator: torch.Tensor | None = None
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bx = x[start:end].to(device)
        by = y[start:end].to(device)
        bl = labels[start:end].to(device)
        model.zero_grad(set_to_none=True)
        loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        flat = flatten_grads(grads, params) * float(end - start)
        accumulator = flat if accumulator is None else accumulator + flat
    if accumulator is None:
        raise ValueError("cannot compute a mean gradient for an empty split")
    return (accumulator / float(total)).numpy().astype(np.float32)


def make_group_weights(group_sizes: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones_like(group_sizes, dtype=np.float64) / float(group_sizes.size)
    if mode == "size":
        return group_sizes.astype(np.float64) / float(np.sum(group_sizes))
    raise ValueError(f"unsupported group weighting: {mode}")


def top_eigensystem_from_rows(
    rows: np.ndarray,
    *,
    top_k: int,
    row_weights: np.ndarray,
) -> dict[str, Any]:
    rows64 = np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.asarray(row_weights, dtype=np.float64).reshape(-1)
    if rows64.shape[0] != weights.shape[0]:
        raise ValueError(f"row count {rows64.shape[0]} does not match weights {weights.shape[0]}")
    weighted = rows64 * np.sqrt(np.maximum(weights, 0.0))[:, None]
    gram = weighted @ weighted.T
    trace = float(np.trace(gram))
    evals, u = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    u = u[:, order]
    keep = min(int(top_k), evals.shape[0])
    values = evals[:keep]
    vectors = []
    for idx, value in enumerate(values):
        if value <= 1e-24:
            vectors.append(np.zeros(rows64.shape[1], dtype=np.float64))
            continue
        vec = weighted.T @ u[:, idx] / math.sqrt(float(value))
        norm = float(np.linalg.norm(vec))
        vectors.append(vec / max(norm, 1e-24))
    basis = np.stack(vectors, axis=1) if vectors else np.zeros((rows64.shape[1], 0), dtype=np.float64)
    return {"values": values.astype(np.float64), "vectors": basis, "trace": trace}


def projection_energy(vector: np.ndarray, basis: np.ndarray) -> float:
    vector64 = np.asarray(vector, dtype=np.float64)
    basis64 = np.asarray(basis, dtype=np.float64)
    if basis64.size == 0:
        return 0.0
    coeff = basis64.T @ vector64
    return float(np.dot(coeff, coeff))


def spectral_projection_loss(
    vector: np.ndarray,
    full_basis: np.ndarray,
    masked_basis: np.ndarray,
    *,
    normalize: bool,
) -> float:
    value = projection_energy(vector, full_basis) - projection_energy(vector, masked_basis)
    if normalize:
        denom = float(np.dot(np.asarray(vector, dtype=np.float64), np.asarray(vector, dtype=np.float64)))
        if denom <= 1e-24:
            return 0.0
        value /= denom
    return float(value)


def subspace_alignment_distance(full_basis: np.ndarray, masked_basis: np.ndarray) -> float:
    k = min(full_basis.shape[1], masked_basis.shape[1])
    if k == 0:
        return 0.0
    overlap = float(np.linalg.norm(full_basis[:, :k].T @ masked_basis[:, :k], ord="fro") ** 2)
    return float(max(0.0, min(1.0, 1.0 - overlap / float(k))))


def estimate_lds_modality_effects(payload: dict[str, Any], *, ridge_lambda: float) -> dict[str, Any]:
    num_groups = int(payload["group_partition"]["num_groups"])
    num_modalities = int(payload["group_partition"]["num_modalities"])
    targets = np.asarray([float(run["test_loss"]) for run in payload["runs"]], dtype=np.float64)
    features = np.zeros((len(payload["runs"]), num_groups * num_modalities), dtype=np.float64)
    group_to_indices = {
        group_id: np.asarray(indices, dtype=np.int64)
        for group_id, indices in enumerate(payload["group_partition"]["group_indices"])
    }
    for row, run in enumerate(payload["runs"]):
        data_indices = np.asarray(run["data_indices"], dtype=np.int64)
        keep_mask = np.asarray(run["sample_modality_mask"], dtype=bool)
        index_to_local = {int(index): local for local, index in enumerate(data_indices.tolist())}
        for group_id in range(num_groups):
            locals_in_group = [
                index_to_local[int(index)]
                for index in group_to_indices[group_id].tolist()
                if int(index) in index_to_local
            ]
            if not locals_in_group:
                features[row, group_id * num_modalities : (group_id + 1) * num_modalities] = 1.0
                continue
            keep_rate = keep_mask[np.asarray(locals_in_group, dtype=np.int64)].mean(axis=0)
            features[row, group_id * num_modalities : (group_id + 1) * num_modalities] = 1.0 - keep_rate
    mean_x = features.mean(axis=0, keepdims=True)
    std_x = features.std(axis=0, keepdims=True)
    standardized = (features - mean_x) / np.maximum(std_x, 1e-8)
    centered_y = targets - float(np.mean(targets))
    lhs = standardized.T @ standardized + float(ridge_lambda) * np.eye(standardized.shape[1], dtype=np.float64)
    coef_standardized = np.linalg.solve(lhs, standardized.T @ centered_y)
    predictions = float(np.mean(targets)) + standardized @ coef_standardized
    coef = coef_standardized / np.maximum(std_x.reshape(-1), 1e-8)
    return {
        "effects": coef.reshape(num_groups, num_modalities),
        "summary": {
            "num_runs": int(len(payload["runs"])),
            "target_loss_mean": float(np.mean(targets)),
            "target_loss_std": float(np.std(targets)),
            "ridge_lambda": float(ridge_lambda),
            "fit_rmse": rmse(predictions, targets),
            "fit_pearson": pearsonr(predictions, targets),
            "fit_spearman": spearmanr(predictions, targets),
            "effect_min": float(np.min(coef)),
            "effect_max": float(np.max(coef)),
        },
    }


def attach_lds_effects(records: list[dict[str, Any]], lds_effects: dict[str, Any]) -> None:
    effects = np.asarray(lds_effects["effects"], dtype=np.float64)
    for record in records:
        group_id = int(record["source_group"])
        modality = int(record["source_modality"])
        record["lds_modality_loss_effect"] = float(effects[group_id, modality])


def gradient_stats(
    full_rows: np.ndarray,
    masked_rows_by_modality: dict[int, np.ndarray],
    test_gradient: np.ndarray,
) -> dict[str, Any]:
    norms = np.linalg.norm(np.asarray(full_rows, dtype=np.float64), axis=1)
    masked_delta = np.concatenate(
        [np.asarray(masked_rows_by_modality[m], dtype=np.float64) - np.asarray(full_rows, dtype=np.float64) for m in (0, 1)],
        axis=0,
    )
    delta_norms = np.linalg.norm(masked_delta, axis=1)
    return {
        "group_gradient_norm_min": float(np.min(norms)),
        "group_gradient_norm_median": float(np.median(norms)),
        "group_gradient_norm_max": float(np.max(norms)),
        "masked_delta_norm_min": float(np.min(delta_norms)),
        "masked_delta_norm_median": float(np.median(delta_norms)),
        "masked_delta_norm_max": float(np.max(delta_norms)),
        "test_gradient_norm": float(np.linalg.norm(test_gradient)),
    }


def summarize(
    records: Sequence[dict[str, Any]],
    full_spectrum: dict[str, Any],
    coupling_matrix: np.ndarray,
    *,
    top_records: int,
) -> dict[str, Any]:
    mass_decay = values(records, "mass_decay")
    mass_growth = values(records, "mass_growth")
    align = values(records, "alignment_distance")
    test_proj = values(records, "test_projection_loss_normalized")
    train_abs = values(records, "train_cross_projection_loss_abs_mean")
    lds = values(records, "lds_modality_loss_effect")
    offdiag = []
    num_groups = coupling_matrix.shape[1]
    for source_index in range(coupling_matrix.shape[0]):
        source_group = source_index // 2
        for target_group in range(num_groups):
            if target_group != source_group:
                offdiag.append(float(coupling_matrix[source_index, target_group]))
    offdiag_np = np.asarray(offdiag, dtype=np.float64)
    top_by_alignment = sorted(records, key=lambda row: float(row["alignment_distance"]), reverse=True)[:top_records]
    top_by_mass_growth = sorted(records, key=lambda row: float(row["mass_growth"]), reverse=True)[:top_records]
    top_by_test_projection = sorted(records, key=lambda row: float(row["test_projection_loss_normalized"]), reverse=True)[
        :top_records
    ]
    summary = {
        "hypothesis_competition_negative_mass_decay": {
            "holds": bool(np.mean(mass_decay < 0.0) >= 0.5 and float(np.median(mass_decay)) < 0.0),
            "negative_fraction": float(np.mean(mass_decay < 0.0)),
            "mass_decay_mean": float(np.mean(mass_decay)),
            "mass_decay_median": float(np.median(mass_decay)),
            "mass_growth_max": float(np.max(mass_growth)),
            "mass_growth_positive_count": int(np.sum(mass_growth > 0.0)),
        },
        "hypothesis_geometric_hubs_alignment_variance": {
            "holds": bool(coefficient_of_variation(align) >= 0.5),
            "alignment_mean": float(np.mean(align)),
            "alignment_std": float(np.std(align)),
            "alignment_cv": coefficient_of_variation(align),
            "alignment_min": float(np.min(align)),
            "alignment_max": float(np.max(align)),
            "top_to_median_ratio": float(np.max(align) / (np.median(align) + 1e-12)),
        },
        "hypothesis_train_cross_group_coupling": {
            "holds": bool(float(np.mean(np.abs(offdiag_np))) > 1e-6),
            "offdiag_projection_loss_mean": float(np.mean(offdiag_np)),
            "offdiag_projection_loss_abs_mean": float(np.mean(np.abs(offdiag_np))),
            "offdiag_projection_loss_abs_median": float(np.median(np.abs(offdiag_np))),
            "offdiag_projection_loss_abs_max": float(np.max(np.abs(offdiag_np))),
            "nonzero_abs_gt_1e_6_fraction": float(np.mean(np.abs(offdiag_np) > 1e-6)),
        },
        "lds_correlations": {
            "alignment_vs_lds_spearman": spearmanr(align, lds),
            "alignment_vs_lds_pearson": pearsonr(align, lds),
            "mass_growth_vs_lds_spearman": spearmanr(mass_growth, lds),
            "mass_growth_vs_lds_pearson": pearsonr(mass_growth, lds),
            "test_projection_vs_lds_spearman": spearmanr(test_proj, lds),
            "test_projection_vs_lds_pearson": pearsonr(test_proj, lds),
            "train_cross_abs_vs_lds_spearman": spearmanr(train_abs, lds),
            "train_cross_abs_vs_lds_pearson": pearsonr(train_abs, lds),
        },
        "full_topk_mass": float(np.sum(full_spectrum["values"])),
        "full_trace": float(full_spectrum["trace"]),
        "top_by_alignment": compact_records(top_by_alignment),
        "top_by_mass_growth": compact_records(top_by_mass_growth),
        "top_by_test_projection": compact_records(top_by_test_projection),
    }
    summary["overall_holds"] = bool(
        summary["hypothesis_competition_negative_mass_decay"]["holds"]
        and summary["hypothesis_geometric_hubs_alignment_variance"]["holds"]
        and summary["hypothesis_train_cross_group_coupling"]["holds"]
    )
    return summary


def values(records: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(record[key]) for record in records], dtype=np.float64)


def compact_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "source_group",
        "source_modality",
        "group_size",
        "mass_decay",
        "mass_growth",
        "alignment_distance",
        "test_projection_loss_normalized",
        "train_cross_projection_loss_abs_mean",
        "lds_modality_loss_effect",
    )
    return [{key: record[key] for key in keys} for record in records]


def coefficient_of_variation(values_np: np.ndarray) -> float:
    arr = np.asarray(values_np, dtype=np.float64)
    return float(np.std(arr) / (abs(float(np.mean(arr))) + 1e-12))


def pearsonr(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    x = np.asarray(list(lhs), dtype=np.float64)
    y = np.asarray(list(rhs), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)) ** 2)))


def to_markdown(output: dict[str, Any]) -> str:
    cfg = output["config"]
    summary = output["summary"]
    comp = summary["hypothesis_competition_negative_mass_decay"]
    hubs = summary["hypothesis_geometric_hubs_alignment_variance"]
    coupling = summary["hypothesis_train_cross_group_coupling"]
    lds = summary["lds_correlations"]
    lines = [
        "# Humor Fisher Eigenspace Verification",
        "",
        f"- dataset: `{cfg['dataset']}`, seed: `{cfg['seed']}`, groups: `{cfg['num_groups']}`, top-k: `{cfg['top_k']}`",
        f"- checkpoint: `{cfg['full_checkpoint']}`",
        f"- gradient params: `{cfg['gradient_params']}` ({cfg['num_parameters']:,} scalars), group weighting: `{cfg['group_weighting']}`",
        "",
        "## Verdict",
        "",
        f"- overall_holds: **{summary['overall_holds']}**",
        f"- competition / negative mass decay: **{comp['holds']}**; negative_fraction={comp['negative_fraction']:.3f}, median_decay={comp['mass_decay_median']:.6g}",
        f"- geometric hubs / alignment variance: **{hubs['holds']}**; cv={hubs['alignment_cv']:.3f}, max/median={hubs['top_to_median_ratio']:.3f}",
        f"- train cross-group coupling: **{coupling['holds']}**; offdiag_abs_mean={coupling['offdiag_projection_loss_abs_mean']:.6g}, nonzero_fraction={coupling['nonzero_abs_gt_1e_6_fraction']:.3f}",
        "",
        "## LDS Check",
        "",
        f"- ridge fit on existing subset losses: RMSE={output['lds_effect_model']['fit_rmse']:.6g}, Spearman={output['lds_effect_model']['fit_spearman']:.3f}",
        f"- alignment vs LDS modality effect: Spearman={lds['alignment_vs_lds_spearman']:.3f}, Pearson={lds['alignment_vs_lds_pearson']:.3f}",
        f"- mass growth vs LDS modality effect: Spearman={lds['mass_growth_vs_lds_spearman']:.3f}, Pearson={lds['mass_growth_vs_lds_pearson']:.3f}",
        f"- test projection loss vs LDS modality effect: Spearman={lds['test_projection_vs_lds_spearman']:.3f}, Pearson={lds['test_projection_vs_lds_pearson']:.3f}",
        "",
        "## Top Alignment Hubs",
        "",
        "| group | modality | size | align | mass_growth | test_proj_norm | LDS_effect |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["top_by_alignment"]:
        lines.append(
            f"| {row['source_group']} | {row['source_modality']} | {row['group_size']} | "
            f"{row['alignment_distance']:.6g} | {row['mass_growth']:.6g} | "
            f"{row['test_projection_loss_normalized']:.6g} | {row['lds_modality_loss_effect']:.6g} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
