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

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_geometry.data import load_dataset, make_loader
from fisher_geometry.experiment import build_model
from fisher_geometry.fisher_metrics import compute_loss
from kernel_surrogate.kernel_attribution import rmse, spearmanr
from kernel_surrogate.lds_real_train import _evaluate_classifier, _train_model


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "grouped50_subset100_data50_mod50"
)
DEFAULT_FULL_CHECKPOINT_DIR = ROOT / "src" / "kernel_surrogate" / "full_checkpoints" / "real_data_cmis"
DEFAULT_PRETRAINED_FULL_CHECKPOINT_ROOT = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_checkpoints"
    / "lds_real_train"
    / "grouped50_full_data_train"
)
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "real_data_cmis_humor.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "real_data_cmis_humor.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group-level real-data CMIS evaluation using existing LDS subset runs. "
            "This matches the current codebase artifacts, where subsets are sampled "
            "over groups rather than individual samples."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--full-checkpoint-dir", type=Path, default=DEFAULT_FULL_CHECKPOINT_DIR)
    parser.add_argument("--pretrained-full-checkpoint-root", type=Path, default=DEFAULT_PRETRAINED_FULL_CHECKPOINT_ROOT)
    parser.add_argument("--force-train-full", action="store_true")
    parser.add_argument("--ridge-lambda", type=float, default=1e-5)
    parser.add_argument("--gamma-scale", type=float, default=1e-3)
    parser.add_argument("--metric-normalization", choices=["cosine", "diag_mean", "none"], default="cosine")
    parser.add_argument("--cmis-normalization-eps", type=float, default=1e-12)
    parser.add_argument(
        "--kernel-methods",
        nargs="+",
        default=["metric_rbf", "structured_metric_rbf", "structured_rbf", "raw_flat_rbf"],
        choices=["metric_rbf", "structured_metric_rbf", "structured_rbf", "raw_flat_rbf", "linear_regression"],
    )
    parser.add_argument("--gradient-params", choices=["all", "last_layer"], default="last_layer")
    parser.add_argument("--structured-shared-weight", type=float, default=1.0)
    parser.add_argument("--comparison-train-fraction", type=float, default=0.8)
    parser.add_argument("--comparison-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--real-random-repeats", type=int, default=200)
    parser.add_argument("--protocol-a-mode", choices=["checkpoint_group_id", "retrain"], default="checkpoint_group_id")
    parser.add_argument("--skip-real-intervention", action="store_true")
    parser.add_argument("--intervention-train-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    payload_path = args.result_dir / f"{args.dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
    device = resolve_device(args.device)
    data = load_dataset(
        args.dataset,
        args.seed,
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

    full_checkpoint = resolve_full_checkpoint_path(
        dataset=args.dataset,
        seed=int(args.seed),
        pretrained_root=args.pretrained_full_checkpoint_root,
        fallback_dir=args.full_checkpoint_dir,
    )
    model, full_metrics = load_or_train_full_model(
        full_checkpoint,
        data=data,
        task=task,
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

    print(f"Computing group-modality gradients for {args.dataset} on {device}...", flush=True)
    gradients = compute_group_modality_gradients(
        model,
        data=data,
        group_indices=payload["group_partition"]["group_indices"],
        task=task,
        device=str(device),
        gradient_params=str(args.gradient_params),
    )
    metric, metric_stats = build_metric(gradients, normalization=str(args.metric_normalization))
    mask_blocks, targets = build_group_mask_blocks(payload)
    deletion_masks = mask_blocks["flat_deleted"]
    comparison = compare_surrogate_methods(
        mask_blocks,
        targets,
        metric=metric,
        methods=args.kernel_methods,
        ridge_lambda=float(args.ridge_lambda),
        gamma_scale=float(args.gamma_scale),
        structured_shared_weight=float(args.structured_shared_weight),
        train_fraction=float(args.comparison_train_fraction),
        seed=int(args.comparison_seed),
        top_k=int(args.top_k),
        cmis_eps=float(args.cmis_normalization_eps),
    )
    fit = fit_surrogate_method(
        "metric_rbf",
        mask_blocks,
        targets,
        metric,
        ridge_lambda=float(args.ridge_lambda),
        gamma_scale=float(args.gamma_scale),
        structured_shared_weight=float(args.structured_shared_weight),
    )
    raw_scores = analytical_second_derivative_at_zero(
        deletion_masks,
        metric,
        alpha=fit["alpha"],
        gamma=float(fit["gamma"]),
        num_groups=int(payload["group_partition"]["num_groups"]),
        atom_offset=int(fit.get("atom_offset", 0)),
    )
    scores = normalize_cmis_scores(raw_scores, eps=float(args.cmis_normalization_eps))
    raw_ranking = rank_group_pairs(raw_scores, top_k=int(args.top_k))
    ranking = rank_group_pairs(scores, top_k=int(args.top_k))
    dominance = summarize_pair_dominance(ranking["top_pairs"])
    surrogate_intervention = surrogate_pair_intervention(
        deletion_masks,
        metric,
        fit,
        scores,
        top_k=int(args.top_k),
        random_repeats=int(args.random_repeats),
        seed=int(args.random_seed),
    )
    if args.skip_real_intervention:
        real_intervention = {"skipped": True}
        protocol_a_kernel_results = []
    else:
        if args.protocol_a_mode == "checkpoint_group_id":
            protocol_a_kernel_results = run_checkpoint_protocol_a_for_kernel_methods(
                comparison,
                payload=payload,
                full_loss=float(full_metrics.get("loss", float("nan"))),
                top_k=int(args.top_k),
                random_repeats=int(args.real_random_repeats),
                random_seed=int(args.random_seed),
            )
            real_intervention = next(
                (dict(row) for row in protocol_a_kernel_results if row.get("method") == "metric_rbf"),
                {"skipped": True},
            )
        else:
            real_intervention = real_retraining_intervention(
                data=data,
                group_indices=payload["group_partition"]["group_indices"],
                ranked_pairs=[tuple(item["pair"]) for item in ranking["top_pairs"]],
                task=task,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                model_type=str(config.get("model", "cross_attention")),
                full_loss=float(full_metrics.get("loss", float("nan"))),
                top_k=int(args.top_k),
                random_repeats=int(args.real_random_repeats),
                random_seed=int(args.random_seed),
                train_seed=int(args.intervention_train_seed),
            )
            protocol_a_kernel_results = run_protocol_a_for_kernel_methods(
                comparison,
                data=data,
                group_indices=payload["group_partition"]["group_indices"],
                task=task,
                dim_x=dim_x,
                dim_y=dim_y,
                num_classes=num_classes,
                model_args=model_args,
                model_type=str(config.get("model", "cross_attention")),
                full_loss=float(full_metrics.get("loss", float("nan"))),
                top_k=int(args.top_k),
                random_repeats=int(args.real_random_repeats),
                random_seed=int(args.random_seed),
                train_seed=int(args.intervention_train_seed),
                cached_metric_result=real_intervention,
            )

    output = {
        "config": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "result_dir": str(args.result_dir),
            "payload_path": str(payload_path),
            "device": str(device),
            "full_checkpoint": str(full_checkpoint),
            "pretrained_full_checkpoint_root": str(args.pretrained_full_checkpoint_root),
            "ridge_lambda": float(args.ridge_lambda),
            "gamma_scale": float(args.gamma_scale),
            "metric_normalization": str(args.metric_normalization),
            "cmis_normalization_eps": float(args.cmis_normalization_eps),
            "kernel_methods": [str(method) for method in args.kernel_methods],
            "gradient_params": str(args.gradient_params),
            "structured_shared_weight": float(args.structured_shared_weight),
            "comparison_train_fraction": float(args.comparison_train_fraction),
            "comparison_seed": int(args.comparison_seed),
            "top_k": int(args.top_k),
            "random_repeats": int(args.random_repeats),
            "real_random_repeats": int(args.real_random_repeats),
            "protocol_a_mode": str(args.protocol_a_mode),
            "unit": "group",
            "num_groups": int(payload["group_partition"]["num_groups"]),
            "num_subset_runs": int(len(payload["runs"])),
        },
        "full_checkpoint_metrics": full_metrics,
        "metric_stats": metric_stats,
        "kernel_comparison": comparison,
        "surrogate_fit": {
            "gamma": float(fit["gamma"]),
            "base_gamma": float(fit["base_gamma"]),
            "train_rmse": rmse(fit["train_predictions"], targets),
            "train_spearman": spearmanr(fit["train_predictions"], targets),
            "target_loss_mean": float(np.mean(targets)),
            "target_loss_std": float(np.std(targets)),
        },
        "raw_cmis_ranking": raw_ranking,
        "cmis_ranking": ranking,
        "top_pair_dominance": dominance,
        "surrogate_self_check": surrogate_intervention,
        "real_retraining_intervention": real_intervention,
        "protocol_a_kernel_results": protocol_a_kernel_results,
        "limitations": [
            "This implementation is group-level because the available LDS artifacts are group-subset masks.",
            "Protocol A uses existing subset checkpoints selected by checkpoint group_ids; it is a group-level checkpoint intervention, not a per-sample attention-matrix edit.",
            "The humor cached features do not include raw text/image payloads in this loader, so Protocol C visualization is not emitted by this script.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print("\n" + to_markdown(output), flush=True)
    print(f"\nSaved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable; falling back from {requested} to cpu.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def make_model_args(config: dict[str, Any], *, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dim=int(config.get("hidden_dim", 256)),
        proj_dim=int(config.get("proj_dim", 256)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        batch_size=int(config.get("batch_size", 4196)),
        eval_batch_size=int(config.get("eval_batch_size") or config.get("batch_size", 4196)),
        epochs=int(config.get("epochs", 15)),
        lr=float(config.get("lr", 5e-4)),
        device=device,
    )


def resolve_full_checkpoint_path(
    *,
    dataset: str,
    seed: int,
    pretrained_root: Path,
    fallback_dir: Path,
) -> Path:
    pretrained = (
        pretrained_root
        / dataset
        / f"seed{seed}"
        / "subset001"
        / "checkpoints"
        / "checkpoint_repeat01.pt"
    )
    if pretrained.exists():
        return pretrained
    return fallback_dir / f"{dataset}_seed{seed}_full.pt"


def load_or_train_full_model(
    checkpoint_path: Path,
    *,
    data: dict[str, Any],
    task: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
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
        train_loader,
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
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "model_config": vars(model_args),
            "dataset": data.get("name", ""),
            "task": task,
            "seed": int(seed),
            "epoch_losses": epoch_losses,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    return model.cpu(), dict(test_metrics)


def compute_group_modality_gradients(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: str,
    gradient_params: str,
) -> np.ndarray:
    params = select_gradient_parameters(model, gradient_params=gradient_params)
    num_params = sum(int(param.numel()) for param in params)
    num_groups = len(group_indices)
    gradients = np.empty((num_groups * 2, num_params), dtype=np.float32)
    train_x = data["train_x"]
    train_y = data["train_y"]
    train_labels = data["train_labels"]

    for group_id, indices in enumerate(group_indices):
        index = torch.as_tensor(indices, dtype=torch.long)
        bx = train_x[index].to(device)
        by = train_y[index].to(device)
        bl = train_labels[index].to(device)
        for modality in range(2):
            model.zero_grad(set_to_none=True)
            logits = model(bx, by, use_x=modality == 0, use_y=modality == 1)
            loss = compute_loss(task, logits, bl)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            offset = 0
            row = group_id * 2 + modality
            for param, grad in zip(params, grads):
                width = int(param.numel())
                if grad is None:
                    gradients[row, offset : offset + width] = 0.0
                else:
                    gradients[row, offset : offset + width] = grad.detach().reshape(-1).float().cpu().numpy()
                offset += width
        if (group_id + 1) % 10 == 0 or group_id + 1 == num_groups:
            print(f"  gradients {group_id + 1}/{num_groups} groups", flush=True)
    return gradients


def select_gradient_parameters(model: torch.nn.Module, *, gradient_params: str) -> list[torch.nn.Parameter]:
    if gradient_params == "all":
        params = [param for param in model.parameters() if param.requires_grad]
    elif gradient_params == "last_layer":
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
        params = [param for _, param in named]
    else:
        raise ValueError(f"unsupported gradient_params={gradient_params!r}")
    if not params:
        raise ValueError(f"no parameters selected for gradient_params={gradient_params!r}")
    return params


def build_group_mask_blocks(payload: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    num_groups = int(payload["group_partition"]["num_groups"])
    num_modalities = int(payload["group_partition"]["num_modalities"])
    if num_modalities != 2:
        raise ValueError(f"expected two modalities, got {num_modalities}")
    sample_deleted = np.ones((len(payload["runs"]), num_groups), dtype=np.float64)
    modality_deleted = np.zeros((len(payload["runs"]), num_groups * num_modalities), dtype=np.float64)
    flat_deleted = np.ones((len(payload["runs"]), num_groups * num_modalities), dtype=np.float64)
    targets = np.empty(len(payload["runs"]), dtype=np.float64)
    group_to_indices = {
        int(group_id): np.asarray(indices, dtype=np.int64)
        for group_id, indices in enumerate(payload["group_partition"]["group_indices"])
    }
    for row, run in enumerate(payload["runs"]):
        targets[row] = float(run["test_loss"])
        data_indices = np.asarray(run["data_indices"], dtype=np.int64)
        keep_mask = np.asarray(run["sample_modality_mask"], dtype=bool)
        index_to_local = {int(index): local for local, index in enumerate(data_indices.tolist())}
        for group_id in [int(value) for value in run["group_ids"]]:
            sample_deleted[row, group_id] = 0.0
            locals_in_group = [
                index_to_local[int(index)]
                for index in group_to_indices[group_id].tolist()
                if int(index) in index_to_local
            ]
            if not locals_in_group:
                continue
            group_keep = keep_mask[np.asarray(locals_in_group, dtype=np.int64)].mean(axis=0)
            group_deleted = 1.0 - group_keep
            modality_deleted[row, group_id * 2 : group_id * 2 + 2] = group_deleted.astype(np.float64)
            flat_deleted[row, group_id * 2 : group_id * 2 + 2] = group_deleted.astype(np.float64)
    return {
        "sample_deleted": sample_deleted,
        "modality_deleted": modality_deleted,
        "flat_deleted": flat_deleted,
        "num_groups": np.asarray([num_groups], dtype=np.int64),
    }, targets


def build_metric(gradients: np.ndarray, *, normalization: str) -> tuple[np.ndarray, dict[str, Any]]:
    grad64 = np.nan_to_num(np.asarray(gradients, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(grad64, axis=1)
    stats = {
        "normalization": normalization,
        "gradient_norm_min": float(np.min(norms)),
        "gradient_norm_median": float(np.median(norms)),
        "gradient_norm_max": float(np.max(norms)),
        "gradient_norm_max_over_median": float(np.max(norms) / (np.median(norms) + 1e-12)),
    }
    if normalization == "cosine":
        normalized = grad64 / np.maximum(norms[:, None], 1e-12)
        metric = normalized @ normalized.T
        stats["metric_diag_mean"] = float(np.mean(np.diag(metric)))
        return metric.astype(np.float64, copy=False), stats
    metric = grad64 @ grad64.T
    if normalization == "diag_mean":
        metric = normalize_psd_metric(metric)
    elif normalization != "none":
        raise ValueError(f"unsupported metric normalization: {normalization}")
    stats["metric_diag_mean"] = float(np.mean(np.diag(metric)))
    return metric.astype(np.float64, copy=False), stats


def normalize_psd_metric(metric: np.ndarray) -> np.ndarray:
    metric = np.asarray(metric, dtype=np.float64)
    diag_mean = float(np.mean(np.diag(metric)))
    if not math.isfinite(diag_mean) or diag_mean <= 0.0:
        return metric
    return metric / diag_mean


def normalize_cmis_scores(scores: np.ndarray, *, eps: float) -> np.ndarray:
    diag = np.diag(scores).astype(np.float64, copy=False)
    denom = np.sqrt(np.maximum(np.abs(diag[:, None] * diag[None, :]), float(eps)))
    return np.asarray(scores, dtype=np.float64) / denom


def compare_surrogate_methods(
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    *,
    metric: np.ndarray,
    methods: Sequence[str],
    ridge_lambda: float,
    gamma_scale: float,
    structured_shared_weight: float,
    train_fraction: float,
    seed: int,
    top_k: int,
    cmis_eps: float,
) -> list[dict[str, Any]]:
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=seed)
    rows: list[dict[str, Any]] = []
    num_groups = int(mask_blocks["num_groups"][0])
    for method in methods:
        fit = fit_surrogate_method(
            method,
            mask_blocks,
            targets,
            metric,
            ridge_lambda=ridge_lambda,
            gamma_scale=gamma_scale,
            structured_shared_weight=structured_shared_weight,
            train_indices=train_indices,
        )
        predictions = predict_surrogate_method(method, mask_blocks, fit, metric=metric, query_indices=test_indices)
        truth = targets[test_indices]
        row: dict[str, Any] = {
            "method": str(method),
            "num_train": int(train_indices.size),
            "num_test": int(test_indices.size),
            "test_rmse": rmse(predictions, truth),
            "test_spearman": spearmanr(predictions, truth),
            "train_rmse": rmse(fit["train_predictions"], targets[train_indices]),
            "train_spearman": spearmanr(fit["train_predictions"], targets[train_indices]),
            "ridge_lambda": float(ridge_lambda),
        }
        if "gamma" in fit:
            row["gamma"] = float(fit["gamma"])
            row["base_gamma"] = float(fit["base_gamma"])
        if method != "linear_regression":
            raw_scores = analytical_second_derivative_at_zero(
                fit["train_masks"],
                fit["metric"],
                alpha=fit["alpha"],
                gamma=float(fit["gamma"]),
                num_groups=num_groups,
                atom_offset=int(fit.get("atom_offset", 0)),
            )
            scores = normalize_cmis_scores(raw_scores, eps=cmis_eps)
            ranking = rank_group_pairs(scores, top_k=top_k)
            row["cmis_score_summary"] = {
                "score_mean": ranking["score_mean"],
                "score_std": ranking["score_std"],
                "score_min": ranking["score_min"],
                "score_max": ranking["score_max"],
                "positive_fraction": ranking["positive_fraction"],
            }
            row["top_pairs"] = ranking["top_pairs"]
            row["top_pair_dominance"] = summarize_pair_dominance(ranking["top_pairs"])
        else:
            row["cmis_note"] = "First-order linear regression has zero second derivative, so no CMIS ranking is defined."
        rows.append(row)
    return rows


def split_indices(num_items: int, *, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_items < 2:
        raise ValueError("need at least two items")
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_items)
    num_train = min(max(1, int(round(num_items * train_fraction))), num_items - 1)
    train = np.sort(order[:num_train]).astype(np.int64, copy=False)
    test = np.sort(order[num_train:]).astype(np.int64, copy=False)
    return train, test


def fit_surrogate_method(
    method: str,
    mask_blocks: dict[str, np.ndarray],
    targets: np.ndarray,
    metric: np.ndarray,
    *,
    ridge_lambda: float,
    gamma_scale: float,
    structured_shared_weight: float,
    train_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    if train_indices is None:
        train_indices = np.arange(len(targets), dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if method == "linear_regression":
        x_train = mask_blocks["flat_deleted"][train_indices].astype(np.float64)
        mean_x = x_train.mean(axis=0, keepdims=True)
        centered_x = x_train - mean_x
        target_mean = float(targets[train_indices].mean())
        centered_y = targets[train_indices] - target_mean
        gram = centered_x.T @ centered_x + float(ridge_lambda) * np.eye(centered_x.shape[1], dtype=np.float64)
        coef = np.linalg.solve(gram, centered_x.T @ centered_y)
        train_predictions = target_mean + centered_x @ coef
        return {
            "method": method,
            "coef": coef,
            "target_mean": target_mean,
            "feature_mean": mean_x.reshape(-1),
            "train_indices": train_indices.copy(),
            "train_predictions": train_predictions,
        }

    train_masks, train_metric = method_masks_and_metric(
        method,
        mask_blocks,
        metric,
        train_indices=train_indices,
        structured_shared_weight=structured_shared_weight,
    )
    distances = metric_sqdist(train_masks, train_masks, train_metric)
    base_gamma = median_gamma(distances)
    gamma = float(base_gamma * gamma_scale)
    kernel = np.exp(-gamma * distances)
    target_mean = float(np.mean(targets[train_indices]))
    alpha = np.linalg.solve(
        kernel + float(ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64),
        targets[train_indices] - target_mean,
    )
    train_predictions = target_mean + kernel @ alpha
    return {
        "method": method,
        "alpha": alpha,
        "target_mean": target_mean,
        "gamma": gamma,
        "base_gamma": base_gamma,
        "train_predictions": train_predictions,
        "train_indices": train_indices.copy(),
        "train_masks": train_masks.copy(),
        "metric": train_metric.copy(),
        "atom_offset": int(mask_blocks["sample_deleted"].shape[1])
        if method in {"structured_rbf", "structured_metric_rbf"}
        else 0,
    }


def predict_surrogate_method(
    method: str,
    mask_blocks: dict[str, np.ndarray],
    fit: dict[str, Any],
    *,
    metric: np.ndarray,
    query_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    query_indices = np.asarray(query_indices, dtype=np.int64)
    if method == "linear_regression":
        x_query = mask_blocks["flat_deleted"][query_indices].astype(np.float64)
        return float(fit["target_mean"]) + (x_query - fit["feature_mean"]) @ fit["coef"]
    query_masks, _ = method_masks_and_metric(
        method,
        mask_blocks,
        metric,
        train_indices=query_indices,
        structured_shared_weight=1.0,
    )
    distances = metric_sqdist(query_masks, fit["train_masks"], fit["metric"])
    kernel = np.exp(-float(fit["gamma"]) * distances)
    return float(fit["target_mean"]) + kernel @ fit["alpha"]


def method_masks_and_metric(
    method: str,
    mask_blocks: dict[str, np.ndarray],
    metric: np.ndarray,
    *,
    train_indices: np.ndarray,
    structured_shared_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "metric_rbf":
        return mask_blocks["flat_deleted"][train_indices], metric
    if method == "raw_flat_rbf":
        dim = mask_blocks["flat_deleted"].shape[1]
        return mask_blocks["flat_deleted"][train_indices], np.eye(dim, dtype=np.float64)
    if method == "structured_rbf":
        sample = mask_blocks["sample_deleted"][train_indices]
        modality = mask_blocks["modality_deleted"][train_indices]
        flat = np.concatenate([sample, modality], axis=1)
        num_groups = sample.shape[1]
        num_atoms = modality.shape[1]
        struct_metric = np.zeros((num_groups + num_atoms, num_groups + num_atoms), dtype=np.float64)
        struct_metric[:num_groups, :num_groups] = np.eye(num_groups, dtype=np.float64)
        struct_metric[num_groups:, num_groups:] = structured_modality_metric(
            num_groups,
            shared_weight=structured_shared_weight,
        )
        return flat, struct_metric
    if method == "structured_metric_rbf":
        sample = mask_blocks["sample_deleted"][train_indices]
        modality = mask_blocks["modality_deleted"][train_indices]
        flat = np.concatenate([sample, modality], axis=1)
        num_groups = sample.shape[1]
        num_atoms = modality.shape[1]
        if metric.shape != (num_atoms, num_atoms):
            raise ValueError(
                f"metric shape {metric.shape} does not match modality atom count {num_atoms}"
            )
        struct_metric = np.zeros((num_groups + num_atoms, num_groups + num_atoms), dtype=np.float64)
        struct_metric[:num_groups, :num_groups] = np.eye(num_groups, dtype=np.float64)
        struct_metric[num_groups:, num_groups:] = metric + structured_modality_metric(
            num_groups,
            shared_weight=structured_shared_weight,
        )
        return flat, struct_metric
    raise ValueError(f"unsupported surrogate method: {method}")


def structured_modality_metric(num_groups: int, *, shared_weight: float) -> np.ndarray:
    metric = np.eye(num_groups * 2, dtype=np.float64)
    if float(shared_weight) == 0.0:
        return metric
    for group_id in range(num_groups):
        sl = slice(group_id * 2, group_id * 2 + 2)
        metric[sl, sl] += float(shared_weight)
    return metric


def fit_metric_rbf_surrogate(
    masks: np.ndarray,
    targets: np.ndarray,
    metric: np.ndarray,
    *,
    ridge_lambda: float,
    gamma_scale: float,
) -> dict[str, Any]:
    distances = metric_sqdist(masks, masks, metric)
    base_gamma = median_gamma(distances)
    gamma = float(base_gamma * gamma_scale)
    kernel = np.exp(-gamma * distances)
    target_mean = float(np.mean(targets))
    alpha = np.linalg.solve(
        kernel + float(ridge_lambda) * np.eye(kernel.shape[0], dtype=np.float64),
        targets - target_mean,
    )
    predictions = target_mean + kernel @ alpha
    return {
        "alpha": alpha,
        "target_mean": target_mean,
        "gamma": gamma,
        "base_gamma": base_gamma,
        "predictions": predictions,
        "train_masks": masks.copy(),
        "metric": metric.copy(),
    }


def analytical_second_derivative_at_zero(
    train_masks: np.ndarray,
    metric: np.ndarray,
    *,
    alpha: np.ndarray,
    gamma: float,
    num_groups: int,
    atom_offset: int = 0,
) -> np.ndarray:
    anchor = np.zeros(train_masks.shape[1], dtype=np.float64)
    diffs = anchor[None, :] - train_masks
    q = diffs @ metric
    kernel = np.exp(-float(gamma) * np.einsum("mi,ij,mj->m", diffs, metric, diffs))
    weighted = alpha * kernel
    scores = np.empty((num_groups, num_groups), dtype=np.float64)
    for u in range(num_groups):
        atom_u = int(atom_offset) + 2 * u
        for v in range(num_groups):
            atom_v = int(atom_offset) + 2 * v + 1
            scores[u, v] = float(
                np.sum(
                    weighted
                    * (
                        4.0 * float(gamma) ** 2 * q[:, atom_u] * q[:, atom_v]
                        - 2.0 * float(gamma) * metric[atom_u, atom_v]
                    )
                )
            )
    return scores


def metric_sqdist(lhs: np.ndarray, rhs: np.ndarray, metric: np.ndarray) -> np.ndarray:
    lhs_metric = lhs @ metric
    lhs_sq = np.sum(lhs_metric * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum((rhs @ metric) * rhs, axis=1, keepdims=True).T
    dist = np.maximum(lhs_sq + rhs_sq - 2.0 * lhs_metric @ rhs.T, 0.0)
    return np.nan_to_num(dist, nan=0.0, posinf=np.finfo(np.float64).max / 4.0, neginf=0.0)


def median_gamma(distances: np.ndarray) -> float:
    values = distances[np.triu_indices(distances.shape[0], k=1)]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0
    return 1.0 / float(np.median(values))


def rank_group_pairs(scores: np.ndarray, *, top_k: int) -> dict[str, Any]:
    num_groups = scores.shape[0]
    rows, cols = np.where(~np.eye(num_groups, dtype=bool))
    values = np.nan_to_num(scores[rows, cols], nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(-values, kind="mergesort")
    top = []
    for index in order[: min(top_k, order.size)]:
        u = int(rows[index])
        v = int(cols[index])
        top.append({"pair": [u, v], "score": float(values[index])})
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        finite_values = np.asarray([0.0], dtype=np.float64)
    return {
        "top_pairs": top,
        "score_mean": float(np.mean(finite_values)),
        "score_std": float(np.std(finite_values)),
        "score_min": float(np.min(finite_values)),
        "score_max": float(np.max(finite_values)),
        "positive_fraction": float(np.mean(finite_values > 0.0)),
    }


def summarize_pair_dominance(top_pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for item in top_pairs:
        u, v = [int(value) for value in item["pair"]]
        row_counts[u] = row_counts.get(u, 0) + 1
        col_counts[v] = col_counts.get(v, 0) + 1
    return {
        "row_group_counts": {str(key): int(value) for key, value in sorted(row_counts.items())},
        "col_group_counts": {str(key): int(value) for key, value in sorted(col_counts.items())},
        "max_row_count": int(max(row_counts.values()) if row_counts else 0),
        "max_col_count": int(max(col_counts.values()) if col_counts else 0),
    }


def real_retraining_intervention(
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    ranked_pairs: Sequence[tuple[int, int]],
    task: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    model_type: str,
    full_loss: float,
    top_k: int,
    random_repeats: int,
    random_seed: int,
    train_seed: int,
) -> dict[str, Any]:
    num_groups = len(group_indices)
    selected = [(int(u), int(v)) for u, v in ranked_pairs[: min(top_k, len(ranked_pairs))]]
    all_pairs = [(u, v) for u in range(num_groups) for v in range(num_groups) if u != v]
    rng = np.random.default_rng(random_seed)

    print(f"Running real retraining intervention for top-{len(selected)} CMIS pairs...", flush=True)
    high_metrics = train_masked_real_model(
        data=data,
        group_indices=group_indices,
        pairs=selected,
        task=task,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        model_type=model_type,
        seed=train_seed,
    )

    random_rows = []
    for repeat in range(random_repeats):
        sample_indices = rng.choice(len(all_pairs), size=len(selected), replace=False)
        random_pairs = [all_pairs[int(index)] for index in sample_indices.tolist()]
        print(f"Running real random intervention {repeat + 1}/{random_repeats}...", flush=True)
        metrics = train_masked_real_model(
            data=data,
            group_indices=group_indices,
            pairs=random_pairs,
            task=task,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            model_type=model_type,
            seed=train_seed,
        )
        random_rows.append(
            {
                "repeat": int(repeat),
                "loss": float(metrics["loss"]),
                "acc": float(metrics["acc"]),
                "loss_increase_vs_full": float(metrics["loss"] - full_loss),
            }
        )
    random_losses = np.asarray([row["loss"] for row in random_rows], dtype=np.float64)
    random_increases = random_losses - float(full_loss)
    high_increase = float(high_metrics["loss"] - full_loss)
    return {
        "skipped": False,
        "top_k": int(len(selected)),
        "selected_pairs": [[int(u), int(v)] for u, v in selected],
        "full_loss": float(full_loss),
        "high_loss": float(high_metrics["loss"]),
        "high_acc": float(high_metrics["acc"]),
        "high_loss_increase_vs_full": high_increase,
        "random_repeats": int(random_repeats),
        "random_loss_mean": float(np.mean(random_losses)) if random_losses.size else float("nan"),
        "random_loss_std": float(np.std(random_losses)) if random_losses.size else float("nan"),
        "random_loss_increase_mean": float(np.mean(random_increases)) if random_increases.size else float("nan"),
        "random_loss_increase_std": float(np.std(random_increases)) if random_increases.size else float("nan"),
        "effect_size_vs_random_loss_increases": float(
            (high_increase - np.mean(random_increases)) / (np.std(random_increases) + 1e-12)
        )
        if random_increases.size
        else float("nan"),
        "random_runs": random_rows,
    }


def run_protocol_a_for_kernel_methods(
    comparison: Sequence[dict[str, Any]],
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    model_type: str,
    full_loss: float,
    top_k: int,
    random_repeats: int,
    random_seed: int,
    train_seed: int,
    cached_metric_result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_row in comparison:
        method = str(method_row["method"])
        if "top_pairs" not in method_row:
            rows.append(
                {
                    "method": method,
                    "skipped": True,
                    "reason": method_row.get("cmis_note", "method has no CMIS ranking"),
                }
            )
            continue
        if method == "metric_rbf" and not cached_metric_result.get("skipped", False):
            result = dict(cached_metric_result)
            result["method"] = method
            rows.append(result)
            continue
        print(f"Running Protocol A real retraining for kernel={method}...", flush=True)
        result = real_retraining_intervention(
            data=data,
            group_indices=group_indices,
            ranked_pairs=[tuple(item["pair"]) for item in method_row["top_pairs"]],
            task=task,
            dim_x=dim_x,
            dim_y=dim_y,
            num_classes=num_classes,
            model_args=model_args,
            model_type=model_type,
            full_loss=full_loss,
            top_k=top_k,
            random_repeats=random_repeats,
            random_seed=random_seed,
            train_seed=train_seed,
        )
        result["method"] = method
        rows.append(result)
    return rows


def run_checkpoint_protocol_a_for_kernel_methods(
    comparison: Sequence[dict[str, Any]],
    *,
    payload: dict[str, Any],
    full_loss: float,
    top_k: int,
    random_repeats: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_row in comparison:
        method = str(method_row["method"])
        if "top_pairs" not in method_row:
            rows.append(
                {
                    "method": method,
                    "skipped": True,
                    "reason": method_row.get("cmis_note", "method has no CMIS ranking"),
                }
            )
            continue
        result = checkpoint_match_protocol_a(
            payload=payload,
            ranked_pairs=[tuple(item["pair"]) for item in method_row["top_pairs"]],
            full_loss=full_loss,
            top_k=top_k,
            random_repeats=random_repeats,
            random_seed=random_seed,
        )
        result["method"] = method
        rows.append(result)
    return rows


def checkpoint_match_protocol_a(
    *,
    payload: dict[str, Any],
    ranked_pairs: Sequence[tuple[int, int]],
    full_loss: float,
    top_k: int,
    random_repeats: int,
    random_seed: int,
) -> dict[str, Any]:
    selected = [(int(u), int(v)) for u, v in ranked_pairs[: min(top_k, len(ranked_pairs))]]
    num_groups = int(payload["group_partition"]["num_groups"])
    all_pairs = [(u, v) for u in range(num_groups) for v in range(num_groups) if u != v]
    subset_losses = np.asarray([float(run["test_loss"]) for run in payload["runs"]], dtype=np.float64)
    group_sets = [set(int(group_id) for group_id in run.get("group_ids", [])) for run in payload["runs"]]
    high_indices_by_pair, high_pair_losses = group_id_exclusion_losses(
        group_sets,
        subset_losses,
        selected,
    )

    rng = np.random.default_rng(random_seed)
    random_means = []
    random_rows = []
    for repeat in range(random_repeats):
        sample_indices = rng.choice(len(all_pairs), size=len(selected), replace=False)
        random_pairs = [all_pairs[int(index)] for index in sample_indices.tolist()]
        indices_by_pair, losses = group_id_exclusion_losses(group_sets, subset_losses, random_pairs)
        mean_loss = float(np.mean(losses)) if losses.size else float("nan")
        random_means.append(mean_loss)
        random_rows.append(
            {
                "repeat": int(repeat),
                "num_observations": int(losses.size),
                "mean_loss": mean_loss,
                "mean_loss_increase_vs_full": float(mean_loss - full_loss),
                "mean_qualifying_checkpoints_per_pair": float(np.mean([len(indices) for indices in indices_by_pair])),
            }
        )
    random_means_arr = np.asarray([value for value in random_means if np.isfinite(value)], dtype=np.float64)
    high_loss = float(np.mean(high_pair_losses)) if high_pair_losses.size else float("nan")
    random_loss = float(np.mean(random_means_arr)) if random_means_arr.size else float("nan")
    return {
        "skipped": False,
        "mode": "checkpoint_group_id",
        "top_k": int(len(selected)),
        "selected_pairs": [[int(u), int(v)] for u, v in selected],
        "num_observations": int(high_pair_losses.size),
        "mean_qualifying_checkpoints_per_pair": float(np.mean([len(indices) for indices in high_indices_by_pair])),
        "qualifying_subset_indices_by_pair": [
            [int(index) for index in indices.tolist()]
            for indices in high_indices_by_pair
        ],
        "full_loss": float(full_loss),
        "high_loss": high_loss,
        "high_loss_increase_vs_full": float(high_loss - full_loss),
        "random_repeats": int(random_repeats),
        "random_loss_mean": random_loss,
        "random_loss_std": float(np.std(random_means_arr)) if random_means_arr.size else float("nan"),
        "random_loss_increase_mean": float(random_loss - full_loss),
        "random_loss_increase_std": float(np.std(random_means_arr)) if random_means_arr.size else float("nan"),
        "effect_size_vs_random_loss_increases": float(
            (high_loss - random_loss) / (np.std(random_means_arr) + 1e-12)
        )
        if random_means_arr.size
        else float("nan"),
        "random_runs": random_rows,
    }


def group_id_exclusion_losses(
    group_sets: Sequence[set[int]],
    subset_losses: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> tuple[list[np.ndarray], np.ndarray]:
    indices_by_pair: list[np.ndarray] = []
    losses: list[float] = []
    for u, v in pairs:
        indices = np.asarray(
            [
                run_index
                for run_index, group_ids in enumerate(group_sets)
                if int(u) not in group_ids and int(v) not in group_ids
            ],
            dtype=np.int64,
        )
        indices_by_pair.append(indices)
        losses.extend(float(value) for value in subset_losses[indices].tolist())
    return indices_by_pair, np.asarray(losses, dtype=np.float64)


def train_masked_real_model(
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    pairs: Sequence[tuple[int, int]],
    task: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    model_type: str,
    seed: int,
) -> dict[str, float]:
    masked = dict(data)
    train_x = data["train_x"].clone()
    train_y = data["train_y"].clone()
    x_groups = {int(u) for u, _ in pairs}
    y_groups = {int(v) for _, v in pairs}
    for group_id in x_groups:
        train_x[torch.as_tensor(group_indices[group_id], dtype=torch.long)] = 0.0
    for group_id in y_groups:
        train_y[torch.as_tensor(group_indices[group_id], dtype=torch.long)] = 0.0
    masked["train_x"] = train_x
    masked["train_y"] = train_y
    train_loader = make_loader(
        masked["train_x"],
        masked["train_y"],
        masked["train_labels"],
        batch_size=int(model_args.batch_size),
        shuffle=True,
    )
    test_loader = make_loader(
        masked["test_x"],
        masked["test_y"],
        masked["test_labels"],
        batch_size=int(model_args.eval_batch_size),
        shuffle=False,
    )
    model, _ = _train_model(
        train_loader,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        task=task,
        model_type=model_type,
        args=model_args,
        seed=int(seed),
    )
    metrics = _evaluate_classifier(
        model,
        test_loader,
        task=task,
        device=str(model_args.device),
        max_sample_losses=0,
    )
    return {"loss": float(metrics["loss"]), "acc": float(metrics["acc"])}


def surrogate_pair_intervention(
    train_masks: np.ndarray,
    metric: np.ndarray,
    fit: dict[str, Any],
    scores: np.ndarray,
    *,
    top_k: int,
    random_repeats: int,
    seed: int,
) -> dict[str, Any]:
    num_groups = scores.shape[0]
    rows, cols = np.where(~np.eye(num_groups, dtype=bool))
    values = scores[rows, cols]
    order = np.argsort(-values, kind="mergesort")
    pairs = [(int(rows[index]), int(cols[index])) for index in order]
    selected = pairs[: min(top_k, len(pairs))]
    anchor = np.zeros((1, train_masks.shape[1]), dtype=np.float64)
    base = float(predict_metric_rbf(anchor, fit, metric)[0])
    high_masks = pair_deletion_queries(selected, num_groups)
    high_increase = predict_metric_rbf(high_masks, fit, metric) - base

    rng = np.random.default_rng(seed)
    random_means = []
    random_all = []
    for _ in range(random_repeats):
        sample_indices = rng.choice(len(pairs), size=len(selected), replace=False)
        random_pairs = [pairs[int(index)] for index in sample_indices.tolist()]
        random_increase = predict_metric_rbf(pair_deletion_queries(random_pairs, num_groups), fit, metric) - base
        random_means.append(float(np.mean(random_increase)))
        random_all.extend(float(value) for value in random_increase.tolist())
    random_means_arr = np.asarray(random_means, dtype=np.float64)
    return {
        "base_surrogate_loss": base,
        "top_k": int(len(selected)),
        "high_mean_predicted_loss_increase": float(np.mean(high_increase)),
        "high_median_predicted_loss_increase": float(np.median(high_increase)),
        "random_mean_of_means": float(np.mean(random_means_arr)),
        "random_std_of_means": float(np.std(random_means_arr)),
        "random_pair_mean_predicted_loss_increase": float(np.mean(random_all)),
        "effect_size_vs_random_means": float(
            (np.mean(high_increase) - np.mean(random_means_arr)) / (np.std(random_means_arr) + 1e-12)
        ),
    }


def pair_deletion_queries(pairs: Sequence[tuple[int, int]], num_groups: int) -> np.ndarray:
    queries = np.zeros((len(pairs), num_groups * 2), dtype=np.float64)
    for row, (u, v) in enumerate(pairs):
        queries[row, int(u) * 2] = 1.0
        queries[row, int(v) * 2 + 1] = 1.0
    return queries


def predict_metric_rbf(queries: np.ndarray, fit: dict[str, Any], metric: np.ndarray) -> np.ndarray:
    distances = metric_sqdist(queries, fit["train_masks"], metric)
    kernel = np.exp(-float(fit["gamma"]) * distances)
    return float(fit["target_mean"]) + kernel @ fit["alpha"]


def to_markdown(output: dict[str, Any]) -> str:
    fit = output["surrogate_fit"]
    self_check = output["surrogate_self_check"]
    real = output["real_retraining_intervention"]
    full = output["full_checkpoint_metrics"]
    dominance = output["top_pair_dominance"]
    lines = [
        "# Real-Data CMIS Evaluation",
        "",
        "## Configuration",
        "",
        f"- dataset: {output['config']['dataset']}",
        f"- unit: {output['config']['unit']}",
        f"- groups: {output['config']['num_groups']}",
        f"- subset runs: {output['config']['num_subset_runs']}",
        f"- device: {output['config']['device']}",
        f"- metric normalization: {output['config']['metric_normalization']}",
        f"- gradient params: {output['config'].get('gradient_params', 'all')}",
        f"- Protocol A mode: {output['config'].get('protocol_a_mode', 'unknown')}",
        "",
        "## Full Checkpoint",
        "",
        f"- test loss: {float(full.get('loss', float('nan'))):.6g}",
        f"- test acc: {float(full.get('acc', float('nan'))):.6g}",
        "",
        "## Surrogate Fit",
        "",
        f"- train RMSE: {fit['train_rmse']:.6g}",
        f"- train Spearman: {fit['train_spearman']:.6g}",
        f"- gamma: {fit['gamma']:.6g}",
        "",
        "## Protocol A Kernel Comparison",
        "",
        "| method | high loss increase | random loss increase | effect size | mode |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in output.get("protocol_a_kernel_results", []):
        if row.get("skipped"):
            lines.append(f"| {row['method']} | skipped | skipped | skipped |")
        else:
            lines.append(
                f"| {row['method']} | {row['high_loss_increase_vs_full']:.6g} | "
                f"{row['random_loss_increase_mean']:.6g} | "
                f"{row['effect_size_vs_random_loss_increases']:.6g} | "
                f"{row.get('mode', output['config'].get('protocol_a_mode', ''))} |"
            )
    lines.extend(
        [
            "",
        "## Surrogate Fit Diagnostics",
        "",
        "| method | test RMSE | test Spearman | train RMSE | train Spearman |",
        "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output.get("kernel_comparison", []):
        lines.append(
            f"| {row['method']} | {row['test_rmse']:.6g} | {row['test_spearman']:.6g} | "
            f"{row['train_rmse']:.6g} | {row['train_spearman']:.6g} |"
        )
    lines.extend(
        [
            "",
        "## Protocol A Checkpoint Group-ID Intervention",
        "",
        ]
    )
    if real.get("skipped"):
        lines.append("- skipped: true")
    else:
        lines.extend(
            [
                f"- top-K: {real['top_k']}",
                f"- high-CMIS real test loss: {real['high_loss']:.6g}",
                f"- high-CMIS real loss increase vs full: {real['high_loss_increase_vs_full']:.6g}",
                f"- random checkpoint loss increase mean: {real['random_loss_increase_mean']:.6g}",
                f"- random checkpoint loss increase std: {real['random_loss_increase_std']:.6g}",
                f"- effect size vs random checkpoint interventions: {real['effect_size_vs_random_loss_increases']:.6g}",
            ]
        )
    lines.extend(
        [
            "",
            "## Surrogate Self-Check",
            "",
            f"- top-K: {self_check['top_k']}",
            f"- high-CMIS mean predicted loss increase: {self_check['high_mean_predicted_loss_increase']:.6g}",
            f"- random mean predicted loss increase: {self_check['random_mean_of_means']:.6g}",
            f"- effect size vs random means: {self_check['effect_size_vs_random_means']:.6g}",
            "",
            "## Dominance Check",
            "",
            f"- max row group count in top-K: {dominance['max_row_count']}",
            f"- max column group count in top-K: {dominance['max_col_count']}",
            f"- row counts: {dominance['row_group_counts']}",
            f"- column counts: {dominance['col_group_counts']}",
        ]
    )
    lines.extend(
        [
        "",
        "## Top CMIS Group Pairs",
        "",
        ]
    )
    for item in output["cmis_ranking"]["top_pairs"][:20]:
        lines.append(f"- {item['pair']}: {item['score']:.6g}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in output["limitations"])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
