from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/data/zhenshuo/multimodal")
UML_ROOT = ROOT / "Unpaired-Multimodal-Learning" / "vision_language"


@dataclass(frozen=True)
class ArmCheckpoint:
    step: int
    fraction: float
    state_dict: dict[str, torch.Tensor]


class SharedClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-1 shared-parameter Fisher gain experiment on Food101.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--feature-dir", type=str, default=str(UML_ROOT / "features"))
    parser.add_argument("--clip-encoder", type=str, default="ViT-B/16")
    parser.add_argument("--text-type", type=str, default="template_mining")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs-a", type=int, default=6)
    parser.add_argument("--epochs-bc", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--fisher-samples", type=int, default=256)
    parser.add_argument("--k-weak", type=int, default=8)
    parser.add_argument("--k-target", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clip_name_for_path(clip_encoder: str) -> str:
    return clip_encoder.replace("/", "-")


def _load_food101_payloads(feature_dir: Path, clip_encoder: str, text_type: str) -> dict[str, Any]:
    image_path = feature_dir / "image" / clip_encoder / "food101" / "crop" / "shot_-1-seed_1.pth"
    test_path = feature_dir / "image" / clip_encoder / "food101" / "test.pth"
    text_path = feature_dir / "text" / clip_encoder / "food101" / f"{text_type}.pth"
    image_trainval = torch.load(image_path, map_location="cpu")
    image_test = torch.load(test_path, map_location="cpu")
    text_payload = torch.load(text_path, map_location="cpu")
    return {"trainval": image_trainval, "test": image_test, "text": text_payload}


def _build_text_by_class(text_payload: dict[str, Any]) -> torch.Tensor:
    features = text_payload["features"].float()
    labels = text_payload["labels"].long()
    num_classes = int(labels.max().item()) + 1
    text_by_class = torch.zeros(num_classes, features.shape[1], dtype=torch.float32)
    for label in range(num_classes):
        mask = labels == label
        text_by_class[label] = features[mask].mean(dim=0)
    return text_by_class


def _derangement(num_classes: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(num_classes)
        if np.all(perm != np.arange(num_classes)):
            return torch.from_numpy(perm.astype(np.int64))


def _tensor_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(features, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _evaluate_accuracy(model: SharedClassifier, features: torch.Tensor, labels: torch.Tensor, *, device: str, batch_size: int) -> float:
    loader = _tensor_loader(features, labels, batch_size, False)
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            logits = model(batch_features.to(device))
            preds = torch.argmax(logits, dim=1)
            batch_labels = batch_labels.to(device)
            correct += int((preds == batch_labels).sum().item())
            total += int(batch_labels.numel())
    return float(correct / max(total, 1))


def _capture_checkpoints(
    model: SharedClassifier,
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


def _train_arm(
    *,
    arm_name: str,
    train_image_features: torch.Tensor,
    train_related_text: torch.Tensor,
    train_unrelated_text: torch.Tensor,
    train_labels: torch.Tensor,
    val_image_features: torch.Tensor,
    val_labels: torch.Tensor,
    device: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    dim = int(train_image_features.shape[1])
    num_classes = int(train_labels.max().item()) + 1
    set_seed(seed)
    model = SharedClassifier(dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    image_loader = _tensor_loader(train_image_features, train_labels, batch_size, True)
    related_loader = _tensor_loader(train_related_text, train_labels, batch_size, True)
    unrelated_loader = _tensor_loader(train_unrelated_text, train_labels, batch_size, True)
    base_steps = len(image_loader)
    total_steps = epochs * base_steps
    if arm_name in {"B", "C_related", "C_unrelated"}:
        total_steps *= 2
    checkpoints: list[ArmCheckpoint] = []
    global_step = 0
    fractions = (0.25, 0.50, 1.0)

    for _ in range(epochs):
        model.train()
        if arm_name == "A":
            for batch_features, batch_labels in image_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features.to(device))
                loss = F.cross_entropy(logits, batch_labels.to(device))
                loss.backward()
                optimizer.step()
                global_step += 1
                checkpoints = _capture_checkpoints(model, checkpoints, global_step, total_steps, fractions)
        elif arm_name == "B":
            for loader_pass in (image_loader, image_loader):
                for batch_features, batch_labels in loader_pass:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch_features.to(device))
                    loss = F.cross_entropy(logits, batch_labels.to(device))
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    checkpoints = _capture_checkpoints(model, checkpoints, global_step, total_steps, fractions)
        else:
            text_loader = related_loader if arm_name == "C_related" else unrelated_loader
            for (image_features, image_labels), (text_features, text_labels) in zip(image_loader, text_loader):
                optimizer.zero_grad(set_to_none=True)
                logits_x = model(image_features.to(device))
                loss_x = F.cross_entropy(logits_x, image_labels.to(device))
                loss_x.backward()
                optimizer.step()
                global_step += 1
                checkpoints = _capture_checkpoints(model, checkpoints, global_step, total_steps, fractions)

                optimizer.zero_grad(set_to_none=True)
                logits_y = model(text_features.to(device))
                loss_y = F.cross_entropy(logits_y, text_labels.to(device))
                loss_y.backward()
                optimizer.step()
                global_step += 1
                checkpoints = _capture_checkpoints(model, checkpoints, global_step, total_steps, fractions)

    if not any(abs(item.fraction - 1.0) < 1e-9 for item in checkpoints):
        checkpoints.append(
            ArmCheckpoint(
                step=global_step,
                fraction=1.0,
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        )
    checkpoints = sorted(checkpoints, key=lambda item: item.fraction)
    final_acc = _evaluate_accuracy(model, val_image_features, val_labels, device=device, batch_size=batch_size)
    return {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "checkpoints": checkpoints,
        "final_val_acc": final_acc,
        "total_steps": total_steps,
    }


def _per_example_gradients(
    model: SharedClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
    max_samples: int,
) -> torch.Tensor:
    count = min(max_samples, features.shape[0])
    grads = []
    model.eval()
    for index in range(count):
        feat = features[index : index + 1].to(device)
        label = labels[index : index + 1].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(feat)
        loss = F.cross_entropy(logits, label)
        grad_tuple = torch.autograd.grad(loss, list(model.head.parameters()), retain_graph=False, create_graph=False)
        flat = torch.cat([grad.reshape(-1) for grad in grad_tuple], dim=0).detach().cpu()
        grads.append(flat)
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
    logdet_value = float(
        torch.log(positive_evals + logdet_eps).sum().item()
        + (total_dim - rank) * math.log(logdet_eps)
    )
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
    return {
        "trace": trace_value,
        "lambda_max": lambda_max,
        "logdet": logdet_value,
        "effective_rank": effective_rank,
    }


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


def _reference_checkpoint_stats(
    checkpoint: ArmCheckpoint,
    *,
    model_dim: int,
    num_classes: int,
    val_image_features: torch.Tensor,
    val_labels: torch.Tensor,
    device: str,
    fisher_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = SharedClassifier(model_dim, num_classes).to(device)
    model.load_state_dict(checkpoint.state_dict)
    target_acc = _evaluate_accuracy(model, val_image_features, val_labels, device=device, batch_size=512)
    gx = _per_example_gradients(model, val_image_features, val_labels, device=device, max_samples=fisher_samples)
    fx_stats = _fisher_eigendecomp(gx)
    reference = {
        "gx": gx,
        "fx_stats": fx_stats,
        "target_val_grad": gx.mean(dim=0).cpu(),
    }
    serializable = {
        "fraction": checkpoint.fraction,
        "step": checkpoint.step,
        "target_val_acc": target_acc,
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
    model_dim: int,
    num_classes: int,
    val_image_features: torch.Tensor,
    val_related_text: torch.Tensor,
    val_unrelated_text: torch.Tensor,
    val_labels: torch.Tensor,
    device: str,
    fisher_samples: int,
    alpha: float,
    k_weak: int,
    k_target: int,
    aux_mode: str | None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = SharedClassifier(model_dim, num_classes).to(device)
    model.load_state_dict(checkpoint.state_dict)
    target_acc = _evaluate_accuracy(model, val_image_features, val_labels, device=device, batch_size=512)
    gx = _per_example_gradients(model, val_image_features, val_labels, device=device, max_samples=fisher_samples)
    fx_stats = _fisher_eigendecomp(gx)
    result: dict[str, Any] = {
        "fraction": checkpoint.fraction,
        "step": checkpoint.step,
        "target_val_acc": target_acc,
        "F_X": {
            "trace": fx_stats["trace"],
            "lambda_max": fx_stats["lambda_max"],
            "logdet": fx_stats["logdet"],
            "effective_rank": fx_stats["effective_rank"],
        },
    }
    if aux_mode is None:
        return result
    aux_features = val_related_text if aux_mode == "related" else val_unrelated_text
    gy = _per_example_gradients(model, aux_features, val_labels, device=device, max_samples=fisher_samples)
    fy_stats = _fisher_eigendecomp(gy)
    shared_stats = _combined_fisher_stats(gx, gy, alpha)
    weak_gain = _weak_direction_gain(fx_stats, gy, k_weak)
    if reference is None:
        target_gain = _target_subspace_gain(fx_stats, gy, k_target)
        perf_gain, perf_gain_norm = _performance_weighted_gain(gy, gx.mean(dim=0).cpu())
    else:
        target_gain = _target_subspace_gain(reference["fx_stats"], gy, k_target)
        perf_gain, perf_gain_norm = _performance_weighted_gain(gy, reference["target_val_grad"])
    target_gain_norm = float(target_gain / max(fy_stats["trace"], 1e-12))
    result["F_Y"] = {
        "trace": fy_stats["trace"],
        "lambda_max": fy_stats["lambda_max"],
        "logdet": fy_stats["logdet"],
        "effective_rank": fy_stats["effective_rank"],
    }
    result["F_shared"] = shared_stats
    result["delta_trace"] = float(shared_stats["trace"] - fx_stats["trace"])
    result["delta_logdet"] = float(shared_stats["logdet"] - fx_stats["logdet"])
    result["delta_erank"] = float(shared_stats["effective_rank"] - fx_stats["effective_rank"])
    result["G_weak"] = weak_gain
    result["G_target"] = target_gain
    result["G_target_norm"] = target_gain_norm
    result["G_perf"] = perf_gain
    result["G_perf_norm"] = perf_gain_norm
    return result


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
    feature_dir = Path(args.feature_dir)
    clip_encoder = _clip_name_for_path(args.clip_encoder)
    payloads = _load_food101_payloads(feature_dir, clip_encoder, args.text_type)
    text_by_class = _build_text_by_class(payloads["text"])
    num_classes = int(text_by_class.shape[0])
    unrelated_perm = _derangement(num_classes, seed + 1000)
    unrelated_text_by_class = text_by_class[unrelated_perm]

    train = payloads["trainval"]["train"]
    val = payloads["trainval"]["val"]
    train_image = train["features"].float()
    train_labels = train["labels"].long()
    val_image = val["features"].float()
    val_labels = val["labels"].long()

    train_related_text = text_by_class[train_labels]
    train_unrelated_text = unrelated_text_by_class[train_labels]
    val_related_text = text_by_class[val_labels]
    val_unrelated_text = unrelated_text_by_class[val_labels]

    arms = {
        "A": _train_arm(
            arm_name="A",
            train_image_features=train_image,
            train_related_text=train_related_text,
            train_unrelated_text=train_unrelated_text,
            train_labels=train_labels,
            val_image_features=val_image,
            val_labels=val_labels,
            device=args.device,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs_a,
            seed=seed,
        ),
        "B": _train_arm(
            arm_name="B",
            train_image_features=train_image,
            train_related_text=train_related_text,
            train_unrelated_text=train_unrelated_text,
            train_labels=train_labels,
            val_image_features=val_image,
            val_labels=val_labels,
            device=args.device,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs_bc,
            seed=seed + 100,
        ),
        "C_related": _train_arm(
            arm_name="C_related",
            train_image_features=train_image,
            train_related_text=train_related_text,
            train_unrelated_text=train_unrelated_text,
            train_labels=train_labels,
            val_image_features=val_image,
            val_labels=val_labels,
            device=args.device,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs_bc,
            seed=seed + 200,
        ),
        "C_unrelated": _train_arm(
            arm_name="C_unrelated",
            train_image_features=train_image,
            train_related_text=train_related_text,
            train_unrelated_text=train_unrelated_text,
            train_labels=train_labels,
            val_image_features=val_image,
            val_labels=val_labels,
            device=args.device,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs_bc,
            seed=seed + 300,
        ),
    }

    model_dim = int(train_image.shape[1])
    metrics_by_arm: dict[str, list[dict[str, Any]]] = {}
    reference_by_fraction: dict[float, dict[str, Any]] = {}

    metrics_by_arm["A"] = [
        _checkpoint_metrics(
            checkpoint,
            model_dim=model_dim,
            num_classes=num_classes,
            val_image_features=val_image,
            val_related_text=val_related_text,
            val_unrelated_text=val_unrelated_text,
            val_labels=val_labels,
            device=args.device,
            fisher_samples=args.fisher_samples,
            alpha=args.alpha,
            k_weak=args.k_weak,
            k_target=args.k_target,
            aux_mode=None,
        )
        for checkpoint in arms["A"]["checkpoints"]
    ]

    metrics_b: list[dict[str, Any]] = []
    for checkpoint in arms["B"]["checkpoints"]:
        metric_b, reference_b = _reference_checkpoint_stats(
            checkpoint,
            model_dim=model_dim,
            num_classes=num_classes,
            val_image_features=val_image,
            val_labels=val_labels,
            device=args.device,
            fisher_samples=args.fisher_samples,
        )
        metrics_b.append(metric_b)
        reference_by_fraction[float(checkpoint.fraction)] = reference_b
    metrics_by_arm["B"] = metrics_b

    for arm_name, aux_mode in (("C_related", "related"), ("C_unrelated", "unrelated")):
        arm_metrics: list[dict[str, Any]] = []
        for checkpoint in arms[arm_name]["checkpoints"]:
            arm_metrics.append(
                _checkpoint_metrics(
                    checkpoint,
                    model_dim=model_dim,
                    num_classes=num_classes,
                    val_image_features=val_image,
                    val_related_text=val_related_text,
                    val_unrelated_text=val_unrelated_text,
                    val_labels=val_labels,
                    device=args.device,
                    fisher_samples=args.fisher_samples,
                    alpha=args.alpha,
                    k_weak=args.k_weak,
                    k_target=args.k_target,
                    aux_mode=aux_mode,
                    reference=reference_by_fraction[float(checkpoint.fraction)],
                )
            )
        metrics_by_arm[arm_name] = arm_metrics

    b_lookup = _align_checkpoint_lookup(metrics_by_arm["B"])
    for arm_name in ("C_related", "C_unrelated"):
        for item in metrics_by_arm[arm_name]:
            baseline = b_lookup[item["fraction"]]
            item["delta_acc_vs_B"] = float(item["target_val_acc"] - baseline["target_val_acc"])

    corr_payload: dict[str, float] = {}
    for arm_name in ("C_related", "C_unrelated"):
        xs_trace = [item["delta_trace"] for item in metrics_by_arm[arm_name]]
        xs_logdet = [item["delta_logdet"] for item in metrics_by_arm[arm_name]]
        xs_target = [item["G_target_norm"] for item in metrics_by_arm[arm_name]]
        xs_perf = [item["G_perf_norm"] for item in metrics_by_arm[arm_name]]
        ys = [item["delta_acc_vs_B"] for item in metrics_by_arm[arm_name]]
        corr_payload[arm_name] = {
            "corr_delta_trace_vs_delta_acc": _corr(xs_trace, ys),
            "corr_delta_logdet_vs_delta_acc": _corr(xs_logdet, ys),
            "corr_G_target_norm_vs_delta_acc": _corr(xs_target, ys),
            "corr_G_perf_norm_vs_delta_acc": _corr(xs_perf, ys),
        }

    return {
        "dataset": "food101",
        "seed": seed,
        "setting": {
            "benchmark_note": "Food101 used because local machine currently has Food101 CLIP image/text features but not SUN397 or Stanford Cars features.",
            "negative_control_note": "Unrelated auxiliary condition uses a fixed class-permuted Food101 text control, keeping the same label space but breaking semantic alignment.",
            "model": "frozen_clip_features_shared_linear_head",
            "shared_block": "whole classifier head",
            "clip_encoder": args.clip_encoder,
            "text_type": args.text_type,
            "alpha": args.alpha,
        },
        "training": {
            arm_name: {
                "total_steps": arm_payload["total_steps"],
                "final_val_acc": arm_payload["final_val_acc"],
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
        "epochs_a": args.epochs_a,
        "epochs_bc": args.epochs_bc,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "fisher_samples": args.fisher_samples,
        "k_weak": args.k_weak,
        "k_target": args.k_target,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
