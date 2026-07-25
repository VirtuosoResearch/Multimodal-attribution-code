from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT

from fisher_geometry.data import load_dataset
from fisher_geometry.data import make_loader
from fisher_geometry.experiment import build_model
from fisher_geometry.fisher_metrics import compute_loss
from kernel_surrogate.kernel_attribution import rmse, spearmanr
from kernel_surrogate.lds_real_train import _evaluate_classifier, _train_model
from kernel_surrogate.real_data_cmis import (
    build_group_mask_blocks,
    build_metric,
    compute_group_modality_gradients,
    make_model_args,
    resolve_device,
    select_gradient_parameters,
    split_indices,
)


DATASETS = ("avmnist", "imdb", "humor", "mosi", "mosei", "sarcasm")
KMEANS_SETTINGS = (
    ("kmeans_data80_mod80", "kmeans_grouped25_subset100_data80_mod80"),
    ("kmeans_data90_mod90", "kmeans_grouped25_subset100_data90_mod90"),
)
KERNEL_METHODS = (
    "metric_rbf",
    "structured_metric_rbf",
    "structured_rbf",
    "raw_flat_rbf",
    "linear_regression",
)

DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT = (
    ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "kmeans_grouped25_full_data_train"
)
DEFAULT_GROUPED25_FULL_CHECKPOINT_ROOT = (
    ROOT / "src" / "kernel_surrogate" / "subset_checkpoints" / "lds_real_train" / "grouped25_full_data_train"
)
DEFAULT_FULL_CHECKPOINT_DIR = ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "influence_function_lds.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "influence_function_lds.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a group-level influence-function LDS predictor on kmeans group subset runs."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=[name for name, _ in KMEANS_SETTINGS],
        default=[name for name, _ in KMEANS_SETTINGS],
        help="Kmeans group-level settings to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=0, help="LDS artifact/model seed.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--pretrained-full-checkpoint-root", type=Path, default=DEFAULT_GROUPED25_FULL_CHECKPOINT_ROOT)
    parser.add_argument("--alternate-full-checkpoint-root", type=Path, default=DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT)
    parser.add_argument("--full-checkpoint-dir", type=Path, default=DEFAULT_FULL_CHECKPOINT_DIR)
    parser.add_argument("--force-train-full", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--include-kernel", action="store_true", help="Also run kernel surrogate LDS comparison.")
    parser.add_argument("--kernel-methods", nargs="+", default=list(KERNEL_METHODS), choices=list(KERNEL_METHODS))
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
    parser.add_argument("--influence-damping", type=float, default=1e-3)
    parser.add_argument("--influence-cg-iters", type=int, default=100)
    parser.add_argument("--influence-cg-tol", type=float, default=1e-6)
    parser.add_argument("--influence-batch-size", type=int, default=512)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def _flatten_tensors(tensors: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    flat = []
    for tensor, param in zip(tensors, params):
        if tensor is None:
            flat.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            flat.append(tensor.reshape(-1))
    if not flat:
        raise ValueError("cannot flatten an empty parameter list")
    return torch.cat(flat)


def _unflatten_like(vector: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    offset = 0
    for param in params:
        width = int(param.numel())
        chunks.append(vector[offset : offset + width].view_as(param))
        offset += width
    if offset != int(vector.numel()):
        raise ValueError(f"vector has {vector.numel()} elements, consumed {offset}")
    return chunks


def _batch_slices(num_items: int, batch_size: int) -> Sequence[slice]:
    resolved = max(1, int(batch_size))
    return [slice(start, min(start + resolved, num_items)) for start in range(0, num_items, resolved)]


def pearsonr(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denom = float(np.sqrt(np.sum(pred**2) * np.sum(truth**2)))
    if denom <= 1e-12 or not np.isfinite(denom):
        return float("nan")
    return float(np.sum(pred * truth) / denom)


def loss_and_grad(
    model: torch.nn.Module,
    params: Sequence[torch.nn.Parameter],
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    device: torch.device,
    batch_size: int,
) -> tuple[float, torch.Tensor]:
    if int(labels.shape[0]) == 0:
        return float("nan"), torch.zeros(sum(int(param.numel()) for param in params), device=device)
    model.eval()
    total_loss = 0.0
    total_items = int(labels.shape[0])
    grad_accum = torch.zeros(sum(int(param.numel()) for param in params), device=device)
    for sl in _batch_slices(total_items, batch_size):
        bx = x[sl].to(device)
        by = y[sl].to(device)
        bl = labels[sl].to(device)
        weight = float(int(bl.shape[0])) / float(total_items)
        logits = model(bx, by, use_x=True, use_y=True)
        loss = compute_loss(task, logits, bl)
        grads = torch.autograd.grad(loss * weight, params, allow_unused=True)
        grad_accum += _flatten_tensors(grads, params).detach()
        total_loss += float(loss.detach().item()) * int(bl.shape[0])
    return total_loss / float(total_items), grad_accum


def hvp_full_train(
    model: torch.nn.Module,
    params: Sequence[torch.nn.Parameter],
    vector: torch.Tensor,
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_labels: torch.Tensor,
    task: str,
    device: torch.device,
    batch_size: int,
    damping: float,
) -> torch.Tensor:
    model.eval()
    total_items = int(train_labels.shape[0])
    vector_chunks = _unflatten_like(vector, params)
    hvp_accum = torch.zeros_like(vector)
    for sl in _batch_slices(total_items, batch_size):
        bx = train_x[sl].to(device)
        by = train_y[sl].to(device)
        bl = train_labels[sl].to(device)
        weight = float(int(bl.shape[0])) / float(total_items)
        logits = model(bx, by, use_x=True, use_y=True)
        loss = compute_loss(task, logits, bl) * weight
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grad_dot = torch.zeros((), device=device)
        for grad, chunk in zip(grads, vector_chunks):
            if grad is not None:
                grad_dot = grad_dot + torch.sum(grad * chunk)
        hvps = torch.autograd.grad(grad_dot, params, allow_unused=True)
        hvp_accum += _flatten_tensors(hvps, params).detach()
    return hvp_accum + float(damping) * vector


def conjugate_gradient(
    matvec,
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    initial_norm = float(torch.sqrt(residual_sq).detach().cpu().item())
    if initial_norm <= float(tol):
        return solution, {"num_iter": 0, "initial_residual_norm": initial_norm, "final_residual_norm": initial_norm}

    final_norm = initial_norm
    stopped_reason = "max_iter"
    for iteration in range(1, int(max_iter) + 1):
        mat_direction = matvec(direction)
        denom = torch.dot(direction, mat_direction)
        if float(torch.abs(denom).detach().cpu().item()) <= 1e-30:
            stopped_reason = "small_denominator"
            break
        step = residual_sq / denom
        solution = solution + step * direction
        residual = residual - step * mat_direction
        next_residual_sq = torch.dot(residual, residual)
        final_norm = float(torch.sqrt(torch.clamp(next_residual_sq, min=0.0)).detach().cpu().item())
        if final_norm <= float(tol) * max(initial_norm, 1.0):
            stopped_reason = "tol"
            residual_sq = next_residual_sq
            break
        beta = next_residual_sq / torch.clamp(residual_sq, min=1e-30)
        direction = residual + beta * direction
        residual_sq = next_residual_sq
    else:
        iteration = int(max_iter)

    return solution.detach(), {
        "num_iter": int(iteration),
        "initial_residual_norm": initial_norm,
        "final_residual_norm": final_norm,
        "stopped_reason": stopped_reason,
    }


def ensure_test_split_aliases(data: dict[str, Any]) -> None:
    if "test_x" in data and "test_y" in data and "test_labels" in data:
        return
    if "val_x" not in data or "val_y" not in data or "val_labels" not in data:
        raise KeyError("dataset must provide either test_* or val_* tensors")
    data["test_x"] = data["val_x"]
    data["test_y"] = data["val_y"]
    data["test_labels"] = data["val_labels"]


def group_tensors(
    data: dict[str, Any],
    indices: Sequence[int],
    *,
    masked_modality: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    x = data["train_x"][index].clone()
    y = data["train_y"][index].clone()
    labels = data["train_labels"][index].clone()
    if masked_modality == 0:
        x.zero_()
    elif masked_modality == 1:
        y.zero_()
    elif masked_modality is not None:
        raise ValueError(f"unsupported modality index: {masked_modality}")
    return x, y, labels


def compute_group_influence_predictions(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    payload: dict[str, Any],
    mask_blocks: dict[str, np.ndarray],
    task: str,
    device: torch.device,
    gradient_params: str,
    damping: float,
    cg_iters: int,
    cg_tol: float,
    batch_size: int,
) -> dict[str, Any]:
    params = select_gradient_parameters(model, gradient_params=gradient_params)
    num_params = sum(int(param.numel()) for param in params)
    print(f"  group IF params={num_params}, damping={damping:g}", flush=True)

    full_train_loss, full_train_grad = loss_and_grad(
        model,
        params,
        x=data["train_x"],
        y=data["train_y"],
        labels=data["train_labels"],
        task=task,
        device=device,
        batch_size=batch_size,
    )
    full_test_loss, test_grad = loss_and_grad(
        model,
        params,
        x=data["test_x"],
        y=data["test_y"],
        labels=data["test_labels"],
        task=task,
        device=device,
        batch_size=batch_size,
    )

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return hvp_full_train(
            model,
            params,
            vector,
            train_x=data["train_x"],
            train_y=data["train_y"],
            train_labels=data["train_labels"],
            task=task,
            device=device,
            batch_size=batch_size,
            damping=damping,
        )

    sensitivity, cg_info = conjugate_gradient(matvec, test_grad, max_iter=cg_iters, tol=cg_tol)
    del full_train_grad

    group_indices = payload["group_partition"]["group_indices"]
    num_groups = int(payload["group_partition"]["num_groups"])
    num_modalities = int(payload["group_partition"]["num_modalities"])
    if num_modalities != 2:
        raise ValueError(f"expected two modalities, got {num_modalities}")
    total_train = float(int(data["train_labels"].shape[0]))
    group_scores = np.zeros(num_groups, dtype=np.float64)
    modality_scores = np.zeros((num_groups, num_modalities), dtype=np.float64)
    group_sizes = np.zeros(num_groups, dtype=np.int64)

    for group_id, indices in enumerate(group_indices):
        group_sizes[group_id] = int(len(indices))
        weight = float(group_sizes[group_id]) / max(total_train, 1.0)
        gx, gy, gl = group_tensors(data, indices)
        _, full_group_grad = loss_and_grad(
            model,
            params,
            x=gx,
            y=gy,
            labels=gl,
            task=task,
            device=device,
            batch_size=batch_size,
        )
        group_scores[group_id] = float(torch.dot(sensitivity, full_group_grad).detach().cpu().item()) * weight
        for modality in range(num_modalities):
            mx, my, ml = group_tensors(data, indices, masked_modality=modality)
            _, masked_grad = loss_and_grad(
                model,
                params,
                x=mx,
                y=my,
                labels=ml,
                task=task,
                device=device,
                batch_size=batch_size,
            )
            modality_scores[group_id, modality] = (
                -float(torch.dot(sensitivity, masked_grad - full_group_grad).detach().cpu().item()) * weight
            )
        if (group_id + 1) % 5 == 0 or group_id + 1 == num_groups:
            print(f"    group IF scores {group_id + 1}/{num_groups}", flush=True)

    sample_deleted = np.asarray(mask_blocks["sample_deleted"], dtype=np.float64)
    modality_deleted = np.asarray(mask_blocks["modality_deleted"], dtype=np.float64)
    group_delta = sample_deleted @ group_scores
    modality_delta = modality_deleted @ modality_scores.reshape(-1)
    pred_array = np.asarray(full_test_loss + group_delta + modality_delta, dtype=np.float64)
    delta_array = np.asarray(group_delta + modality_delta, dtype=np.float64)
    deletion_only_pred = np.asarray(full_test_loss + group_delta, dtype=np.float64)

    return {
        "predictions": pred_array,
        "deltas": delta_array,
        "deletion_only_predictions": deletion_only_pred,
        "group_scores": group_scores,
        "modality_scores": modality_scores,
        "group_sizes": group_sizes,
        "stats": {
            "gradient_params": gradient_params,
            "num_params": int(num_params),
            "damping": float(damping),
            "cg": cg_info,
            "full_train_loss_at_checkpoint": float(full_train_loss),
            "full_test_loss_at_checkpoint": float(full_test_loss),
            "prediction_mean": float(np.mean(pred_array)),
            "prediction_std": float(np.std(pred_array)),
            "delta_mean": float(np.mean(delta_array)),
            "delta_std": float(np.std(delta_array)),
            "score_level": "kmeans_group",
            "prediction_formula": (
                "full_test_loss + sample_deleted @ group_deletion_scores + "
                "modality_deleted @ group_modality_mask_scores"
            ),
        },
    }

def influence_row(
    *,
    dataset: str,
    setting: str,
    run_name: str,
    split_seed: int,
    predictions: np.ndarray,
    targets: np.ndarray,
    train_fraction: float,
    stats: dict[str, Any],
) -> dict[str, Any]:
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    truth = targets[test_indices]
    pred = predictions[test_indices]
    return {
        "dataset": dataset,
        "setting": setting,
        "run_name": run_name,
        "method": "influence_function",
        "split_seed": int(split_seed),
        "test_lds": spearmanr(pred, truth),
        "test_pearson": pearsonr(pred, truth),
        "test_rmse": rmse(pred, truth),
        "all_lds": spearmanr(predictions, targets),
        "all_pearson": pearsonr(predictions, targets),
        "all_rmse": rmse(predictions, targets),
        "num_train": int(train_indices.size),
        "num_test": int(test_indices.size),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "influence_stats": stats,
    }


def choose_full_checkpoint_path(
    *,
    dataset: str,
    seed: int,
    preferred_root: Path,
    alternate_root: Path,
    fallback_dir: Path,
    dim_x: int,
    dim_y: int,
    num_classes: int,
) -> Path:
    for root in (preferred_root, alternate_root):
        candidate = _full_checkpoint_candidate(root, dataset=dataset, seed=seed)
        if checkpoint_is_compatible(candidate, dim_x=dim_x, dim_y=dim_y, num_classes=num_classes):
            return candidate
    return fallback_dir / f"{dataset}_seed{int(seed)}_full.pt"


def _full_checkpoint_candidate(root: Path, *, dataset: str, seed: int) -> Path:
    return root / dataset / f"seed{int(seed)}" / "subset001" / "checkpoints" / "checkpoint_repeat01.pt"


def checkpoint_is_compatible(path: Path, *, dim_x: int, dim_y: int, num_classes: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    config = dict(payload.get("model_config", {}))
    if config.get("dim_x") is not None and int(config["dim_x"]) != int(dim_x):
        return False
    if config.get("dim_y") is not None and int(config["dim_y"]) != int(dim_y):
        return False
    if config.get("num_classes") is not None and int(config["num_classes"]) != int(num_classes):
        return False
    state = dict(payload.get("model_state_dict", {}))
    proj_x = state.get("proj_x.weight")
    proj_y = state.get("proj_y.weight")
    classifier = state.get("classifier.weight")
    if proj_x is not None and int(proj_x.shape[1]) != int(dim_x):
        return False
    if proj_y is not None and int(proj_y.shape[1]) != int(dim_y):
        return False
    if classifier is not None and int(classifier.shape[0]) != int(num_classes):
        return False
    return True


def load_or_train_full_model_current(
    checkpoint_path: Path,
    *,
    data: dict[str, Any],
    task: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: Any,
    model_type: str,
    seed: int,
    force_train: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if checkpoint_path.exists() and not force_train:
        print(f"Loading full-data checkpoint: {checkpoint_path}", flush=True)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(model_type, dim_x, dim_y, num_classes, model_args)
        model.load_state_dict(payload["model_state_dict"])
        return model, dict(payload.get("test_metrics", {}))

    print(f"Training full-data checkpoint: {checkpoint_path}", flush=True)
    train_loader = make_loader(
        data["train_x"],
        data["train_y"],
        data["train_labels"],
        batch_size=int(model_args.batch_size),
        shuffle=True,
    )
    test_loader = make_loader(
        data["test_x"],
        data["test_y"],
        data["test_labels"],
        batch_size=int(model_args.eval_batch_size),
        shuffle=False,
    )
    model, epoch_losses = _train_model(
        [train_loader],
        loader_epochs=[int(model_args.epochs)],
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        task=task,
        model_type=model_type,
        args=model_args,
        seed=int(seed),
    )
    test_metrics = _evaluate_classifier(
        model,
        test_loader,
        task=task,
        device=str(model_args.device),
        max_sample_losses=1000,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_type": model_type,
            "model_config": {
                **vars(model_args),
                "dim_x": int(dim_x),
                "dim_y": int(dim_y),
                "num_classes": int(num_classes),
            },
            "dataset": data.get("name", ""),
            "task": task,
            "seed": int(seed),
            "epoch_losses": [float(value) for value in epoch_losses],
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    return model.cpu(), dict(test_metrics)


def summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in records:
        key = (str(row["dataset"]), str(row["setting"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    rows = []
    for (dataset, setting, method), items in sorted(grouped.items()):
        lds = np.asarray([float(item["test_lds"]) for item in items], dtype=np.float64)
        pearson = np.asarray([float(item["test_pearson"]) for item in items], dtype=np.float64)
        rmses = np.asarray([float(item["test_rmse"]) for item in items], dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "method": method,
                "mean_lds": float(np.nanmean(lds)),
                "std_lds": float(np.nanstd(lds)),
                "mean_pearson": float(np.nanmean(pearson)),
                "mean_rmse": float(np.nanmean(rmses)),
                "num_splits": int(len(items)),
            }
        )
    return rows


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Influence Function LDS",
        "",
        f"- unit: `{payload['config']['unit']}`",
        f"- settings: `{payload['config']['settings']}`",
        f"- split: `{payload['config']['split']}`",
        f"- split seeds: `{payload['config']['split_seeds']}`",
        f"- influence damping: `{payload['config']['influence_damping']}`",
        f"- include kernel: `{payload['config']['include_kernel']}`",
        "",
        "## Summary",
        "",
        "| dataset | setting | method | mean LDS | std LDS | mean Pearson | mean RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['dataset']} | {row['setting']} | {row['method']} | "
            f"{row['mean_lds']:.6f} | {row['std_lds']:.6f} | "
            f"{row['mean_pearson']:.6f} | {row['mean_rmse']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.device))
    split_seeds = [int(seed) for seed in args.split_seeds]
    setting_lookup = dict(KMEANS_SETTINGS)
    selected_settings = [(name, setting_lookup[name]) for name in args.settings]

    all_records: list[dict[str, Any]] = []
    dataset_setting_details: list[dict[str, Any]] = []

    for dataset in [str(item) for item in args.datasets]:
        for setting, run_name in selected_settings:
            payload_path = args.result_root / run_name / f"{dataset}_seed{int(args.seed)}.json"
            if not payload_path.exists():
                print(f"[{dataset}/{setting}] missing {payload_path}, skipping", flush=True)
                continue
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            group_method = str(payload.get("group_partition", {}).get("group_method", ""))
            if group_method != "kmeans":
                raise ValueError(f"{payload_path} is not a kmeans group payload: group_method={group_method!r}")

            config = dict(payload["config"])
            print(f"[{dataset}/{setting}] loading data/checkpoint on {device}...", flush=True)
            data = load_dataset(
                dataset,
                int(args.seed),
                text_backbone=str(config.get("text_backbone", "clip_cached")),
                vision_backbone=str(config.get("vision_backbone", "clip_cached")),
                feature_device=str(args.feature_device or config.get("feature_device", device)),
                backbone_batch_size=int(config.get("backbone_batch_size", 32)),
                download_datasets=bool(config.get("download_datasets", False)),
                backbone_cache_dir=config.get("backbone_cache_dir"),
            )
            ensure_test_split_aliases(data)
            model_args = make_model_args(config, device=str(device))
            dim_x = int(data["train_x"].shape[1])
            dim_y = int(data["train_y"].shape[1])
            num_classes = int(data["num_classes"])
            full_checkpoint = choose_full_checkpoint_path(
                dataset=dataset,
                seed=int(args.seed),
                preferred_root=args.pretrained_full_checkpoint_root,
                alternate_root=args.alternate_full_checkpoint_root,
                fallback_dir=args.full_checkpoint_dir,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
            )
            model, full_metrics = load_or_train_full_model_current(
                full_checkpoint,
                data=data,
                task=str(data["task"]),
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                model_type=str(config.get("model", "cross_attention")),
                seed=int(args.seed),
                force_train=bool(args.force_train_full),
            )
            model.to(device)
            model.eval()

            mask_blocks, targets = build_group_mask_blocks(payload)
            metric = None
            metric_stats = None
            if bool(args.include_kernel):
                print(f"[{dataset}/{setting}] computing kernel metric...", flush=True)
                gradients = compute_group_modality_gradients(
                    model,
                    data=data,
                    group_indices=payload["group_partition"]["group_indices"],
                    task=str(data["task"]),
                    device=str(device),
                    gradient_params=str(args.gradient_params),
                )
                metric, metric_stats = build_metric(gradients, normalization=str(args.metric_normalization))

            print(f"[{dataset}/{setting}] computing influence-function predictions...", flush=True)
            influence = compute_group_influence_predictions(
                model,
                data=data,
                payload=payload,
                mask_blocks=mask_blocks,
                task=str(data["task"]),
                device=device,
                gradient_params=str(args.gradient_params),
                damping=float(args.influence_damping),
                cg_iters=int(args.influence_cg_iters),
                cg_tol=float(args.influence_cg_tol),
                batch_size=int(args.influence_batch_size),
            )

            setting_records: list[dict[str, Any]] = []
            for split_seed in split_seeds:
                row = influence_row(
                    dataset=dataset,
                    setting=setting,
                    run_name=run_name,
                    split_seed=split_seed,
                    predictions=influence["predictions"],
                    targets=targets,
                    train_fraction=float(args.train_fraction),
                    stats=influence["stats"],
                )
                all_records.append(row)
                setting_records.append(row)
                print(
                    f"  split_seed={split_seed} influence_function: "
                    f"LDS={row['test_lds']:.6f}, Pearson={row['test_pearson']:.6f}, RMSE={row['test_rmse']:.6f}",
                    flush=True,
                )
                if not bool(args.include_kernel):
                    continue
                assert metric is not None
                assert metric_stats is not None
                from kernel_surrogate.compare_anchor_methods_top3 import evaluate_method_grid

                for method in [str(item) for item in args.kernel_methods]:
                    kernel_row = evaluate_method_grid(
                        dataset=dataset,
                        setting=setting,
                        run_name=run_name,
                        method=method,
                        split_seed=int(split_seed),
                        mask_blocks=mask_blocks,
                        targets=targets,
                        metric=metric,
                        train_fraction=float(args.train_fraction),
                    )
                    kernel_row["metric_stats"] = metric_stats
                    all_records.append(kernel_row)
                    setting_records.append(kernel_row)
                    print(
                        f"  split_seed={split_seed} {method}: "
                        f"LDS={kernel_row['test_lds']:.6f}, "
                        f"Pearson={kernel_row['test_pearson']:.6f}, RMSE={kernel_row['test_rmse']:.6f}",
                        flush=True,
                    )

            dataset_setting_details.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "run_name": run_name,
                    "payload_path": str(payload_path),
                    "full_checkpoint": str(full_checkpoint),
                    "full_checkpoint_metrics": full_metrics,
                    "metric_stats": metric_stats,
                    "influence_stats": influence["stats"],
                    "influence_predictions": influence["predictions"].tolist(),
                    "influence_deltas": influence["deltas"].tolist(),
                    "group_if_scores": influence["group_scores"].tolist(),
                    "group_modality_if_scores": influence["modality_scores"].tolist(),
                    "group_sizes": influence["group_sizes"].tolist(),
                    "targets": targets.tolist(),
                    "records": setting_records,
                }
            )

    output = {
        "config": {
            "datasets": [str(item) for item in args.datasets],
            "settings": [name for name, _ in selected_settings],
            "run_names": [run_name for _, run_name in selected_settings],
            "seed": int(args.seed),
            "unit": "kmeans_group",
            "split": f"{float(args.train_fraction):.6g}/{1.0 - float(args.train_fraction):.6g}",
            "split_seeds": split_seeds,
            "include_kernel": bool(args.include_kernel),
            "kernel_methods": [str(item) for item in args.kernel_methods],
            "gradient_params": str(args.gradient_params),
            "metric_normalization": str(args.metric_normalization),
            "influence_damping": float(args.influence_damping),
            "influence_cg_iters": int(args.influence_cg_iters),
            "influence_cg_tol": float(args.influence_cg_tol),
            "influence_batch_size": int(args.influence_batch_size),
            "pretrained_full_checkpoint_root": str(args.pretrained_full_checkpoint_root),
            "alternate_full_checkpoint_root": str(args.alternate_full_checkpoint_root),
        },
        "summary": summarize(all_records),
        "records": all_records,
        "dataset_setting_details": dataset_setting_details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
