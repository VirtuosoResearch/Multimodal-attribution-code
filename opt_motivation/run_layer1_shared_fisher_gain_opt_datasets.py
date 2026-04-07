from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opt_motivation.core import (  # noqa: E402
    _compute_loss,
    _flatten_grad_list,
    _make_loader,
    _move_modalities,
    _select_fisher_parameters,
    build_joint_model,
    load_affect,
    load_imdb,
    set_seed,
)


@dataclass(frozen=True)
class ArmCheckpoint:
    step: int
    fraction: float
    state_dict: dict[str, torch.Tensor]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-1 shared Fisher gain validation on IMDb/Humor/Sarcasm.")
    parser.add_argument("--dataset", type=str, required=True, choices=["imdb", "humor", "sarcasm"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--imdb-path", type=str, default="/data/zhenshuo/datasets/multimodal_imdb.hdf5")
    parser.add_argument("--humor-path", type=str, default="/data/zhenshuo/datasets/humor/humor.pkl")
    parser.add_argument("--sarcasm-path", type=str, default="/data/zhenshuo/datasets/sarcasm/sarcasm.pkl")
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-bc", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=128)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--parameter-scope", type=str, default="shared", choices=["shared", "shared_core", "head", "classifier"])
    parser.add_argument("--fisher-samples", type=int, default=256)
    parser.add_argument("--k-weak", type=int, default=8)
    parser.add_argument("--k-target", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--checkpoint-count", type=int, default=8)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--valid-max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    return parser


def _default_epochs(dataset: str) -> int:
    return 10 if dataset == "imdb" else 30


def _load_dataset(args: argparse.Namespace):
    if args.dataset == "imdb":
        return load_imdb(
            args.imdb_path,
            train_max_samples=args.train_max_samples,
            valid_max_samples=args.valid_max_samples,
            test_max_samples=args.test_max_samples,
        )
    if args.dataset == "humor":
        return load_affect(args.humor_path, dataset_name="humor", max_seq_len=args.max_seq_len)
    return load_affect(args.sarcasm_path, dataset_name="sarcasm", max_seq_len=args.max_seq_len)


def _target_and_aux_modalities(dataset_name: str) -> tuple[list[str], list[str]]:
    if dataset_name == "imdb":
        return ["text"], ["image"]
    return ["text"], ["audio", "video"]


def _permute_aux_split(split, aux_modalities: list[str], *, seed: int):
    rng = np.random.default_rng(seed)
    count = split.labels.shape[0]
    perm = rng.permutation(count)
    modalities = {name: values.copy() for name, values in split.modalities.items()}
    for name in aux_modalities:
        modalities[name] = split.modalities[name][perm].copy()
    return type(split)(
        modalities=modalities,
        labels=split.labels.copy(),
        mask=None if split.mask is None else split.mask.copy(),
        ids=list(split.ids),
    )


def _capture_checkpoints(
    model: torch.nn.Module,
    checkpoints: list[ArmCheckpoint],
    step: int,
    total_steps: int,
    fractions: tuple[float, ...],
) -> list[ArmCheckpoint]:
    updated = list(checkpoints)
    current_fraction = step / max(total_steps, 1)
    for frac in fractions:
        if any(abs(item.fraction - frac) < 1e-9 for item in updated):
            continue
        if current_fraction >= frac:
            updated.append(
                ArmCheckpoint(
                    step=step,
                    fraction=frac,
                    state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
            )
    return updated


def _checkpoint_fractions(count: int) -> tuple[float, ...]:
    count = max(int(count), 1)
    return tuple(index / count for index in range(1, count + 1))


def _train_arm(
    dataset,
    *,
    arm_name: str,
    target_modalities: list[str],
    aux_modalities: list[str],
    device: str,
    batch_size: int,
    epochs: int,
    lr: float,
    encoder_hidden_dim: int,
    fusion_dim: int,
    fusion_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    seed: int,
    checkpoint_fractions: tuple[float, ...],
) -> dict[str, Any]:
    set_seed(seed)
    model = build_joint_model(
        dataset,
        encoder_hidden_dim=encoder_hidden_dim,
        fusion_dim=fusion_dim,
        fusion_layers=fusion_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        architecture="shared_head",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _make_loader(dataset.train, batch_size=batch_size, shuffle=True)
    base_steps = len(train_loader)
    total_steps = epochs * base_steps * 2
    global_step = 0
    checkpoints: list[ArmCheckpoint] = []
    target_set = set(target_modalities)
    aux_set = set(aux_modalities)

    for _ in range(epochs):
        model.train()
        if arm_name == "B":
            for loader_pass in (train_loader, train_loader):
                for modalities, mask, labels in loader_pass:
                    modalities = _move_modalities(modalities, device)
                    mask = None if mask is None else mask.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(modalities, mask, active_modalities=target_set)
                    loss, _ = _compute_loss(dataset.task, outputs["logits"], labels)
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    checkpoints = _capture_checkpoints(model, checkpoints, global_step, total_steps, checkpoint_fractions)
        else:
            for modalities, mask, labels in train_loader:
                modalities = _move_modalities(modalities, device)
                mask = None if mask is None else mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs_x = model(modalities, mask, active_modalities=target_set)
                loss_x, _ = _compute_loss(dataset.task, outputs_x["logits"], labels)
                loss_x.backward()
                optimizer.step()
                global_step += 1
                checkpoints = _capture_checkpoints(model, checkpoints, global_step, total_steps, checkpoint_fractions)

                optimizer.zero_grad(set_to_none=True)
                outputs_y = model(modalities, mask, active_modalities=aux_set)
                loss_y, _ = _compute_loss(dataset.task, outputs_y["logits"], labels)
                loss_y.backward()
                optimizer.step()
                global_step += 1
                checkpoints = _capture_checkpoints(model, checkpoints, global_step, total_steps, checkpoint_fractions)

    if not any(abs(item.fraction - 1.0) < 1e-9 for item in checkpoints):
        checkpoints.append(
            ArmCheckpoint(
                step=global_step,
                fraction=1.0,
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        )
    checkpoints = sorted(checkpoints, key=lambda item: item.fraction)
    return {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "checkpoints": checkpoints,
        "total_steps": total_steps,
    }


def _sample_f1_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(np.float32)
    labels = labels.astype(np.float32)
    tp = (preds * labels).sum(axis=1)
    precision = tp / np.maximum(preds.sum(axis=1), 1.0)
    recall = tp / np.maximum(labels.sum(axis=1), 1.0)
    f1 = np.where(precision + recall > 0.0, 2.0 * precision * recall / np.maximum(precision + recall, 1e-8), 0.0)
    return float(np.mean(f1))


def _evaluate_target_metric(model, split, *, device: str, batch_size: int, active_modalities: set[str], task: str) -> float:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for modalities, mask, labels in loader:
            modalities = _move_modalities(modalities, device)
            mask = None if mask is None else mask.to(device)
            outputs = model(modalities, mask, active_modalities=active_modalities)
            logits_list.append(outputs["logits"].detach().cpu().numpy().astype(np.float32))
            labels_list.append(labels.numpy().astype(np.float32))
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    if task == "binary":
        preds = (probs.reshape(-1) >= 0.5).astype(np.float32)
        return float(np.mean(preds == labels.reshape(-1).astype(np.float32)))
    return _sample_f1_from_probs(probs, labels)


def _per_example_gradients(
    model,
    split,
    *,
    device: str,
    max_samples: int,
    active_modalities: set[str],
    parameter_scope: str,
    task: str,
) -> torch.Tensor:
    _, params = _select_fisher_parameters(model, parameter_scope)
    loader = _make_loader(split, batch_size=1, shuffle=False)
    grads: list[torch.Tensor] = []
    model.eval()
    count = 0
    for modalities, mask, labels in loader:
        if count >= max_samples:
            break
        modalities = _move_modalities(modalities, device)
        mask = None if mask is None else mask.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(modalities, mask, active_modalities=active_modalities)
        loss, _ = _compute_loss(task, outputs["logits"], labels)
        grad_tuple = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False, create_graph=False)
        grads.append(_flatten_grad_list(list(grad_tuple), params).detach().cpu())
        count += 1
    return torch.stack(grads, dim=0)


def _fisher_eigendecomp(grad_matrix: torch.Tensor) -> dict[str, Any]:
    g = grad_matrix.float()
    num_samples = g.shape[0]
    scaled = g / math.sqrt(max(num_samples, 1))
    gram = scaled @ scaled.T
    evals_small, evecs_small = torch.linalg.eigh(gram)
    evals_small = torch.clamp(evals_small, min=0.0)
    positive_mask = evals_small > 1e-10
    positive_evals = evals_small[positive_mask]
    positive_small_vecs = evecs_small[:, positive_mask]
    if positive_evals.numel() > 0:
        param_vecs = (scaled.T @ positive_small_vecs) / torch.sqrt(positive_evals).unsqueeze(0)
        param_vecs = torch.nn.functional.normalize(param_vecs, dim=0)
        lambda_max = float(positive_evals.max().item())
    else:
        param_vecs = torch.zeros((g.shape[1], 0), dtype=torch.float32)
        lambda_max = 0.0
    trace_value = float((g.pow(2).sum(dim=1).mean()).item())
    rank = int(positive_evals.numel())
    logdet_eps = 1e-6
    total_dim = int(g.shape[1])
    logdet_value = float(torch.log(positive_evals + logdet_eps).sum().item() + (total_dim - rank) * math.log(logdet_eps))
    effective_rank = float(trace_value / max(lambda_max, 1e-12))
    return {
        "grad_matrix": g,
        "eigenvalues": positive_evals.cpu(),
        "eigenvectors": param_vecs.cpu(),
        "trace": trace_value,
        "lambda_max": lambda_max,
        "logdet": logdet_value,
        "effective_rank": effective_rank,
    }


def _combined_fisher_stats(gx: torch.Tensor, gy: torch.Tensor, alpha: float) -> dict[str, float]:
    hx = gx.float() / math.sqrt(max(gx.shape[0], 1))
    hy = gy.float() * math.sqrt(alpha / max(gy.shape[0], 1))
    h = torch.cat([hx, hy], dim=0)
    gram = h @ h.T
    evals = torch.linalg.eigvalsh(gram)
    evals = torch.clamp(evals, min=0.0)
    lambda_max = float(evals.max().item()) if evals.numel() > 0 else 0.0
    trace_value = float(evals.sum().item())
    rank = int((evals > 1e-10).sum().item())
    eps = 1e-6
    total_dim = int(gx.shape[1])
    logdet_value = float(torch.log(evals[evals > 1e-10] + eps).sum().item() + (total_dim - rank) * math.log(eps))
    effective_rank = float(trace_value / max(lambda_max, 1e-12))
    return {"trace": trace_value, "lambda_max": lambda_max, "logdet": logdet_value, "effective_rank": effective_rank}


def _weak_direction_gain(fx_stats: dict[str, Any], gy: torch.Tensor, k_weak: int) -> float:
    eigenvalues = fx_stats["eigenvalues"]
    eigenvectors = fx_stats["eigenvectors"]
    if eigenvalues.numel() == 0 or eigenvectors.shape[1] == 0:
        return 0.0
    k = min(k_weak, int(eigenvalues.numel()))
    weak_vecs = eigenvectors[:, :k]
    proj = gy.float() @ weak_vecs.float()
    return float((proj.pow(2).sum() / max(gy.shape[0], 1)).item())


def _target_subspace_gain(fx_stats: dict[str, Any], gy: torch.Tensor, k_target: int) -> float:
    eigenvalues = fx_stats["eigenvalues"]
    eigenvectors = fx_stats["eigenvectors"]
    if eigenvalues.numel() == 0 or eigenvectors.shape[1] == 0:
        return 0.0
    k = min(k_target, int(eigenvalues.numel()))
    target_vecs = eigenvectors[:, -k:]
    proj = gy.float() @ target_vecs.float()
    return float((proj.pow(2).sum() / max(gy.shape[0], 1)).item())


def _performance_weighted_gain(gy: torch.Tensor, target_val_grad: torch.Tensor) -> tuple[float, float]:
    direction = target_val_grad.float()
    norm_sq = float(direction.dot(direction).item())
    if norm_sq <= 1e-20:
        return 0.0, 0.0
    projections = gy.float() @ direction
    raw_gain = float((projections.pow(2).mean()).item())
    normalized_gain = float(raw_gain / norm_sq)
    return raw_gain, normalized_gain


def _first_order_alignment(gx_mean: torch.Tensor, gy_mean: torch.Tensor) -> tuple[float, float]:
    gx_mean = gx_mean.float()
    gy_mean = gy_mean.float()
    dot = float(torch.dot(gx_mean, gy_mean).item())
    denom = float(torch.linalg.norm(gx_mean).item() * torch.linalg.norm(gy_mean).item())
    return dot, float(dot / max(denom, 1e-12))


def _quadratic_penalty(gx: torch.Tensor, gy_mean: torch.Tensor) -> float:
    projections = gx.float() @ gy_mean.float()
    return float(projections.pow(2).mean().item())


def _quadratic_score(
    gx_mean: torch.Tensor,
    gy_mean: torch.Tensor,
    gx: torch.Tensor,
    *,
    eta: float,
) -> tuple[float, float, float]:
    s_grad, _ = _first_order_alignment(gx_mean, gy_mean)
    quad_penalty = _quadratic_penalty(gx, gy_mean)
    s_quad = float(s_grad - 0.5 * eta * quad_penalty)
    denom = float(torch.linalg.norm(gx_mean).item() * torch.linalg.norm(gy_mean).item())
    return quad_penalty, s_quad, float(s_quad / max(denom, 1e-12))


def _reference_checkpoint_stats(
    checkpoint: ArmCheckpoint,
    *,
    dataset,
    model_kwargs: dict[str, Any],
    device: str,
    fisher_samples: int,
    target_modalities: list[str],
    parameter_scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = build_joint_model(dataset, architecture="shared_head", **model_kwargs).to(device)
    model.load_state_dict(checkpoint.state_dict)
    target_metric = _evaluate_target_metric(model, dataset.valid, device=device, batch_size=64, active_modalities=set(target_modalities), task=dataset.task)
    gx = _per_example_gradients(
        model,
        dataset.valid,
        device=device,
        max_samples=fisher_samples,
        active_modalities=set(target_modalities),
        parameter_scope=parameter_scope,
        task=dataset.task,
    )
    fx_stats = _fisher_eigendecomp(gx)
    reference = {"gx": gx, "fx_stats": fx_stats, "target_val_grad": gx.mean(dim=0).cpu()}
    serializable = {
        "fraction": checkpoint.fraction,
        "step": checkpoint.step,
        "target_val_metric": target_metric,
        "F_X": {
            "trace": fx_stats["trace"],
            "lambda_max": fx_stats["lambda_max"],
            "logdet": fx_stats["logdet"],
            "effective_rank": fx_stats["effective_rank"],
        },
    }
    return serializable, reference


def _checkpoint_metrics(
    checkpoint: ArmCheckpoint,
    *,
    dataset,
    model_kwargs: dict[str, Any],
    device: str,
    fisher_samples: int,
    alpha: float,
    eta: float,
    k_weak: int,
    k_target: int,
    target_modalities: list[str],
    aux_modalities: list[str],
    parameter_scope: str,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = build_joint_model(dataset, architecture="shared_head", **model_kwargs).to(device)
    model.load_state_dict(checkpoint.state_dict)
    target_metric = _evaluate_target_metric(model, dataset.valid, device=device, batch_size=64, active_modalities=set(target_modalities), task=dataset.task)
    gx = _per_example_gradients(
        model,
        dataset.valid,
        device=device,
        max_samples=fisher_samples,
        active_modalities=set(target_modalities),
        parameter_scope=parameter_scope,
        task=dataset.task,
    )
    fx_stats = _fisher_eigendecomp(gx)
    gy = _per_example_gradients(
        model,
        dataset.valid,
        device=device,
        max_samples=fisher_samples,
        active_modalities=set(aux_modalities),
        parameter_scope=parameter_scope,
        task=dataset.task,
    )
    fy_stats = _fisher_eigendecomp(gy)
    shared_stats = _combined_fisher_stats(gx, gy, alpha)
    weak_gain = _weak_direction_gain(fx_stats, gy, k_weak)
    gx_mean = gx.mean(dim=0).cpu()
    gy_mean = gy.mean(dim=0).cpu()
    s_grad, s_grad_norm = _first_order_alignment(gx_mean, gy_mean)
    quad_penalty, s_quad_f, s_quad_f_norm = _quadratic_score(gx_mean, gy_mean, gx, eta=eta)
    if reference is None:
        target_gain = _target_subspace_gain(fx_stats, gy, k_target)
        perf_gain, perf_gain_norm = _performance_weighted_gain(gy, gx_mean)
    else:
        target_gain = _target_subspace_gain(reference["fx_stats"], gy, k_target)
        perf_gain, perf_gain_norm = _performance_weighted_gain(gy, reference["target_val_grad"])
    target_gain_norm = float(target_gain / max(fy_stats["trace"], 1e-12))
    return {
        "fraction": checkpoint.fraction,
        "step": checkpoint.step,
        "target_val_metric": target_metric,
        "F_X": {
            "trace": fx_stats["trace"],
            "lambda_max": fx_stats["lambda_max"],
            "logdet": fx_stats["logdet"],
            "effective_rank": fx_stats["effective_rank"],
        },
        "F_Y": {
            "trace": fy_stats["trace"],
            "lambda_max": fy_stats["lambda_max"],
            "logdet": fy_stats["logdet"],
            "effective_rank": fy_stats["effective_rank"],
        },
        "F_shared": shared_stats,
        "delta_trace": float(shared_stats["trace"] - fx_stats["trace"]),
        "delta_logdet": float(shared_stats["logdet"] - fx_stats["logdet"]),
        "delta_erank": float(shared_stats["effective_rank"] - fx_stats["effective_rank"]),
        "G_weak": weak_gain,
        "G_target": target_gain,
        "G_target_norm": target_gain_norm,
        "G_perf": perf_gain,
        "G_perf_norm": perf_gain_norm,
        "gbar_x_norm": float(torch.linalg.norm(gx_mean).item()),
        "gbar_y_norm": float(torch.linalg.norm(gy_mean).item()),
        "S_grad": s_grad,
        "S_grad_norm": s_grad_norm,
        "Q_Fx": quad_penalty,
        "S_quad_F": s_quad_f,
        "S_quad_F_norm": s_quad_f_norm,
    }


def _align_checkpoint_lookup(metrics: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    return {float(item["fraction"]): item for item in metrics}


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    set_seed(seed)
    dataset_related = _load_dataset(args)
    target_modalities, aux_modalities = _target_and_aux_modalities(args.dataset)
    dataset_unrelated = type(dataset_related)(
        name=dataset_related.name,
        task=dataset_related.task,
        modality_names=list(dataset_related.modality_names),
        train=_permute_aux_split(dataset_related.train, aux_modalities, seed=seed + 1000),
        valid=_permute_aux_split(dataset_related.valid, aux_modalities, seed=seed + 2000),
        test=_permute_aux_split(dataset_related.test, aux_modalities, seed=seed + 3000),
    )
    epochs_bc = int(args.epochs_bc) if args.epochs_bc is not None else _default_epochs(args.dataset)
    checkpoint_fractions = _checkpoint_fractions(args.checkpoint_count)

    train_kwargs = {
        "target_modalities": target_modalities,
        "aux_modalities": aux_modalities,
        "device": args.device,
        "batch_size": args.batch_size,
        "epochs": epochs_bc,
        "lr": args.lr,
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "fusion_dim": args.fusion_dim,
        "fusion_layers": args.fusion_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
        "checkpoint_fractions": checkpoint_fractions,
    }
    model_kwargs = {
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "fusion_dim": args.fusion_dim,
        "fusion_layers": args.fusion_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
    }

    arms = {
        "B": _train_arm(dataset_related, arm_name="B", seed=seed + 100, **train_kwargs),
        "C_related": _train_arm(dataset_related, arm_name="C_related", seed=seed + 200, **train_kwargs),
        "C_unrelated": _train_arm(dataset_unrelated, arm_name="C_unrelated", seed=seed + 300, **train_kwargs),
    }

    metrics_by_arm: dict[str, list[dict[str, Any]]] = {}
    reference_by_fraction: dict[float, dict[str, Any]] = {}
    metrics_b: list[dict[str, Any]] = []
    for checkpoint in arms["B"]["checkpoints"]:
        metric_b, reference_b = _reference_checkpoint_stats(
            checkpoint,
            dataset=dataset_related,
            model_kwargs=model_kwargs,
            device=args.device,
            fisher_samples=args.fisher_samples,
            target_modalities=target_modalities,
            parameter_scope=args.parameter_scope,
        )
        metrics_b.append(metric_b)
        reference_by_fraction[float(checkpoint.fraction)] = reference_b
    metrics_by_arm["B"] = metrics_b

    for arm_name, arm_dataset in (("C_related", dataset_related), ("C_unrelated", dataset_unrelated)):
        arm_metrics: list[dict[str, Any]] = []
        for checkpoint in arms[arm_name]["checkpoints"]:
            arm_metrics.append(
                _checkpoint_metrics(
                    checkpoint,
                    dataset=arm_dataset,
                    model_kwargs=model_kwargs,
                    device=args.device,
                    fisher_samples=args.fisher_samples,
                    alpha=args.alpha,
                    eta=args.lr,
                    k_weak=args.k_weak,
                    k_target=args.k_target,
                    target_modalities=target_modalities,
                    aux_modalities=aux_modalities,
                    parameter_scope=args.parameter_scope,
                    reference=reference_by_fraction[float(checkpoint.fraction)],
                )
            )
        metrics_by_arm[arm_name] = arm_metrics

    b_lookup = _align_checkpoint_lookup(metrics_by_arm["B"])
    for arm_name in ("C_related", "C_unrelated"):
        for item in metrics_by_arm[arm_name]:
            baseline = b_lookup[item["fraction"]]
            item["delta_acc_vs_B"] = float(item["target_val_metric"] - baseline["target_val_metric"])

    corr_payload: dict[str, Any] = {}
    for arm_name in ("C_related", "C_unrelated"):
        xs_trace = [item["delta_trace"] for item in metrics_by_arm[arm_name]]
        xs_logdet = [item["delta_logdet"] for item in metrics_by_arm[arm_name]]
        xs_target = [item["G_target_norm"] for item in metrics_by_arm[arm_name]]
        xs_perf = [item["G_perf_norm"] for item in metrics_by_arm[arm_name]]
        xs_s_grad = [item["S_grad"] for item in metrics_by_arm[arm_name]]
        xs_s_quad = [item["S_quad_F"] for item in metrics_by_arm[arm_name]]
        xs_s_quad_norm = [item["S_quad_F_norm"] for item in metrics_by_arm[arm_name]]
        ys = [item["delta_acc_vs_B"] for item in metrics_by_arm[arm_name]]
        corr_payload[arm_name] = {
            "corr_delta_trace_vs_delta_acc": _corr(xs_trace, ys),
            "corr_delta_logdet_vs_delta_acc": _corr(xs_logdet, ys),
            "corr_G_target_norm_vs_delta_acc": _corr(xs_target, ys),
            "corr_G_perf_norm_vs_delta_acc": _corr(xs_perf, ys),
            "corr_S_grad_vs_delta_acc": _corr(xs_s_grad, ys),
            "corr_S_quad_F_vs_delta_acc": _corr(xs_s_quad, ys),
            "corr_S_quad_F_norm_vs_delta_acc": _corr(xs_s_quad_norm, ys),
        }

    metric_name = "accuracy" if dataset_related.task == "binary" else "sample_f1"
    return {
        "dataset": args.dataset,
        "seed": seed,
        "setting": {
            "model": "shared_head",
            "shared_block": args.parameter_scope,
            "target_modalities": target_modalities,
            "aux_modalities": aux_modalities,
            "negative_control": "global_permuted_auxiliary",
            "target_metric": metric_name,
            "alpha": args.alpha,
        },
        "training": {
            arm_name: {
                "total_steps": arm_payload["total_steps"],
                "checkpoint_fractions": [float(item.fraction) for item in arm_payload["checkpoints"]],
            }
            for arm_name, arm_payload in arms.items()
        },
        "metrics": metrics_by_arm,
        "correlations": corr_payload,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = run(args.seed, args)
    payload["config"] = {
        "device": args.device,
        "batch_size": args.batch_size,
        "epochs_bc": int(args.epochs_bc) if args.epochs_bc is not None else _default_epochs(args.dataset),
        "lr": args.lr,
        "dropout": args.dropout,
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "fusion_dim": args.fusion_dim,
        "fusion_layers": args.fusion_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "parameter_scope": args.parameter_scope,
        "fisher_samples": args.fisher_samples,
        "k_weak": args.k_weak,
        "k_target": args.k_target,
        "alpha": args.alpha,
        "checkpoint_count": args.checkpoint_count,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
