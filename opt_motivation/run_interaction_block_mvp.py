from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/data/zhenshuo/multimodal")
OPT_ROOT = ROOT / "Multimodal-representation-and-attribution"
OPT_MOTIVATION_ROOT = OPT_ROOT / "opt_motivation"
UML_ROOT = ROOT / "Unpaired-Multimodal-Learning" / "vision_language"

for path in (OPT_ROOT, UML_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from opt_motivation.core import (  # noqa: E402
    _compute_loss,
    _make_loader,
    _move_modalities,
    auto_device,
    load_imdb,
    set_seed,
    train_joint_model,
)
from features import text_outdir  # noqa: E402


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    tensor_name: str
    row_start: int
    row_end: int
    tensor_shape: tuple[int, ...]

    @property
    def num_rows(self) -> int:
        return int(self.row_end - self.row_start)


class Food101FusionModel(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.fusion_in = nn.Linear(image_dim + text_dim, hidden_dim)
        self.fusion_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        *,
        image_active: bool = True,
        text_active: bool = True,
    ) -> torch.Tensor:
        if not image_active:
            image_features = torch.zeros_like(image_features)
        if not text_active:
            text_features = torch.zeros_like(text_features)
        hidden = torch.cat([image_features, text_features], dim=1)
        hidden = F.gelu(self.fusion_in(hidden))
        hidden = self.dropout(hidden)
        hidden = F.gelu(self.fusion_hidden(hidden))
        hidden = self.dropout(hidden)
        return self.classifier(hidden)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run interaction-residual Fisher MVP on IMDb or Food101.")
    parser.add_argument("--dataset", choices=["imdb", "food101"], required=True)
    parser.add_argument("--device", type=str, default=auto_device())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--measurement-batch-size", type=int, default=16)
    parser.add_argument("--measurement-batches", type=int, default=16)
    parser.add_argument("--shuffle-trials", type=int, default=8)
    parser.add_argument("--max-random-block-samples", type=int, default=16)
    parser.add_argument("--min-block-rows", type=int, default=8)
    parser.add_argument("--max-blocks-per-tensor", type=int, default=4)
    parser.add_argument("--intervention-alpha", type=float, default=0.0)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--imdb-path", type=str, default="/data/zhenshuo/datasets/multimodal_imdb.hdf5")
    parser.add_argument("--imdb-encoder-hidden-dim", type=int, default=256)
    parser.add_argument("--imdb-fusion-dim", type=int, default=128)
    parser.add_argument("--imdb-fusion-layers", type=int, default=2)
    parser.add_argument("--imdb-num-heads", type=int, default=4)
    parser.add_argument("--imdb-ffn-dim", type=int, default=256)
    parser.add_argument("--imdb-dropout", type=float, default=0.1)
    parser.add_argument("--imdb-batch-size", type=int, default=32)
    parser.add_argument("--imdb-epochs", type=int, default=10)
    parser.add_argument("--imdb-lr", type=float, default=1e-4)
    parser.add_argument("--imdb-patience", type=int, default=5)
    parser.add_argument("--food101-feature-dir", type=str, default=str(UML_ROOT / "features"))
    parser.add_argument("--food101-result-dir", type=str, default=str(UML_ROOT / "experiments"))
    parser.add_argument("--food101-clip-encoder", type=str, default="ViT-B/16")
    parser.add_argument("--food101-text-type", type=str, default="template_mining")
    parser.add_argument("--food101-hidden-dim", type=int, default=512)
    parser.add_argument("--food101-dropout", type=float, default=0.1)
    parser.add_argument("--food101-epochs", type=int, default=20)
    parser.add_argument("--food101-lr", type=float, default=1e-3)
    parser.add_argument("--food101-batch-size", type=int, default=256)
    parser.add_argument("--food101-patience", type=int, default=5)
    return parser


def _chunk_weight_rows(
    tensor_name: str,
    tensor: torch.Tensor,
    *,
    min_block_rows: int,
    max_blocks_per_tensor: int,
) -> list[BlockSpec]:
    if tensor.ndim != 2:
        return []
    rows = int(tensor.shape[0])
    if rows < 2 * min_block_rows:
        return []
    num_blocks = min(max_blocks_per_tensor, rows // min_block_rows)
    if num_blocks < 2:
        return []
    block_size = math.ceil(rows / num_blocks)
    blocks: list[BlockSpec] = []
    for index, row_start in enumerate(range(0, rows, block_size)):
        row_end = min(rows, row_start + block_size)
        if row_end - row_start <= 0:
            continue
        blocks.append(
            BlockSpec(
                block_id=f"{tensor_name}[rows={row_start}:{row_end}]",
                tensor_name=tensor_name,
                row_start=row_start,
                row_end=row_end,
                tensor_shape=tuple(int(x) for x in tensor.shape),
            )
        )
    return blocks if len(blocks) >= 2 else []


def _candidate_blocks_for_imdb(model: nn.Module, *, min_block_rows: int, max_blocks_per_tensor: int) -> list[BlockSpec]:
    patterns = [
        re.compile(r"shared_blocks\.\d+\.ffn\.0\.weight"),
        re.compile(r"shared_blocks\.\d+\.ffn\.3\.weight"),
        re.compile(r"fusion_head\.1\.weight"),
        re.compile(r"classifier\.weight"),
    ]
    blocks: list[BlockSpec] = []
    for name, param in model.named_parameters():
        if not any(pattern.fullmatch(name) for pattern in patterns):
            continue
        blocks.extend(
            _chunk_weight_rows(
                name,
                param.detach(),
                min_block_rows=min_block_rows,
                max_blocks_per_tensor=max_blocks_per_tensor,
            )
        )
    return blocks


def _candidate_blocks_for_food101(model: nn.Module, *, min_block_rows: int, max_blocks_per_tensor: int) -> list[BlockSpec]:
    blocks: list[BlockSpec] = []
    for name, param in model.named_parameters():
        if name not in {"fusion_in.weight", "fusion_hidden.weight", "classifier.weight"}:
            continue
        blocks.extend(
            _chunk_weight_rows(
                name,
                param.detach(),
                min_block_rows=min_block_rows,
                max_blocks_per_tensor=max_blocks_per_tensor,
            )
        )
    return blocks


def _fisher_stats(rows: list[np.ndarray]) -> dict[str, float]:
    if not rows:
        return {"trace": float("nan"), "lambda_max": float("nan"), "effective_rank": float("nan")}
    matrix = np.stack(rows, axis=0).astype(np.float64)
    trace_value = float(np.mean(np.sum(matrix * matrix, axis=1)))
    scaled = matrix / math.sqrt(matrix.shape[0])
    singular_values = np.linalg.svd(scaled, compute_uv=False, full_matrices=False)
    lambda_max = float(singular_values[0] ** 2) if singular_values.size > 0 else 0.0
    effective_rank = float(trace_value / max(lambda_max, 1e-12))
    return {"trace": trace_value, "lambda_max": lambda_max, "effective_rank": effective_rank}


def _vector_for_block(grad: torch.Tensor, block: BlockSpec) -> np.ndarray:
    sliced = grad[block.row_start : block.row_end].reshape(-1)
    return sliced.detach().cpu().numpy().astype(np.float64)


def _make_exact_label_keys(labels: torch.Tensor) -> list[tuple[int, ...] | int]:
    labels_cpu = labels.detach().cpu()
    if labels_cpu.ndim == 1:
        return [int(item) for item in labels_cpu.tolist()]
    return [tuple(int(x) for x in row.tolist()) for row in labels_cpu]


def _label_conditional_permutation(labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    keys = _make_exact_label_keys(labels)
    grouped: dict[Any, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        grouped[key].append(index)
    permutation = np.arange(len(keys))
    for indices in grouped.values():
        if len(indices) <= 1:
            continue
        shuffled = np.asarray(indices)[rng.permutation(len(indices))]
        permutation[np.asarray(indices)] = shuffled
    return torch.from_numpy(permutation.astype(np.int64))


def _global_permutation(size: int, *, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.permutation(size).astype(np.int64))


def _imdb_condition_loss(
    model: nn.Module,
    dataset_task: str,
    modalities: dict[str, torch.Tensor],
    mask: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    active_modalities: set[str],
) -> torch.Tensor:
    outputs = model(modalities, mask, active_modalities=active_modalities)
    loss, _ = _compute_loss(dataset_task, outputs["logits"], labels)
    return loss


def _grad_dict_from_loss(loss: torch.Tensor, params: dict[str, nn.Parameter]) -> dict[str, torch.Tensor]:
    names = list(params.keys())
    param_values = [params[name] for name in names]
    grads = torch.autograd.grad(
        loss,
        param_values,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    output: dict[str, torch.Tensor] = {}
    for name, param, grad in zip(names, param_values, grads):
        output[name] = torch.zeros_like(param) if grad is None else grad.detach()
    return output


def _gather_rows_from_grad_dict(grad_dict: dict[str, torch.Tensor], blocks: list[BlockSpec]) -> dict[str, np.ndarray]:
    return {block.block_id: _vector_for_block(grad_dict[block.tensor_name], block) for block in blocks}


def _compute_random_block_thresholds(
    score_by_block: dict[str, dict[str, float]],
    blocks: list[BlockSpec],
    *,
    max_random_block_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    by_tensor: dict[str, list[BlockSpec]] = defaultdict(list)
    for block in blocks:
        by_tensor[block.tensor_name].append(block)
    rng = np.random.default_rng(seed)
    thresholds: dict[str, dict[str, float]] = {}
    for block in blocks:
        peers = [peer for peer in by_tensor[block.tensor_name] if peer.block_id != block.block_id]
        if not peers:
            thresholds[block.block_id] = {
                "trace_p95": float("nan"),
                "lambda_max_p95": float("nan"),
            }
            continue
        if len(peers) > max_random_block_samples:
            peer_indices = rng.choice(len(peers), size=max_random_block_samples, replace=False)
            peers = [peers[int(index)] for index in peer_indices]
        trace_values = np.asarray([score_by_block[peer.block_id]["trace"] for peer in peers], dtype=np.float64)
        lambda_values = np.asarray([score_by_block[peer.block_id]["lambda_max"] for peer in peers], dtype=np.float64)
        thresholds[block.block_id] = {
            "trace_p95": float(np.quantile(trace_values, 0.95)),
            "lambda_max_p95": float(np.quantile(lambda_values, 0.95)),
        }
    return thresholds


def _measurement_batch_indices(size: int, *, batch_size: int, max_batches: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(size)
    total = min(size, batch_size * max_batches)
    perm = perm[:total]
    batches: list[np.ndarray] = []
    for start in range(0, total, batch_size):
        batch = perm[start : start + batch_size]
        if len(batch) == batch_size:
            batches.append(batch)
    return batches


def _evaluate_imdb_loss(
    model: nn.Module,
    dataset_task: str,
    split,
    *,
    device: str,
    batch_size: int,
    active_modalities: set[str],
    ablate_block: BlockSpec | None = None,
    alpha: float = 0.0,
) -> float:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    param = None
    backup = None
    if ablate_block is not None:
        named = dict(model.named_parameters())
        param = named[ablate_block.tensor_name]
        backup = param.data[ablate_block.row_start : ablate_block.row_end].clone()
        param.data[ablate_block.row_start : ablate_block.row_end].mul_(alpha)
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for modalities, mask, labels in loader:
            modalities_device = _move_modalities(modalities, device)
            mask_device = None if mask is None else mask.to(device)
            labels = labels.to(device)
            loss = _imdb_condition_loss(
                model,
                dataset_task,
                modalities_device,
                mask_device,
                labels,
                active_modalities=active_modalities,
            )
            losses.append(float(loss.item()))
    if param is not None and backup is not None:
        param.data[ablate_block.row_start : ablate_block.row_end].copy_(backup)
    return float(np.mean(losses))


def _food101_image_paths(feature_dir: Path, encoder_name: str, seed: int) -> tuple[Path, Path]:
    trainval = feature_dir / "image" / encoder_name / "food101" / "crop" / f"shot_-1-seed_{seed}.pth"
    if not trainval.exists():
        fallback = feature_dir / "image" / encoder_name / "food101" / "crop" / "shot_-1-seed_1.pth"
        if fallback.exists():
            trainval = fallback
    test = feature_dir / "image" / encoder_name / "food101" / "test.pth"
    return trainval, test


def _average_text_by_class(feature_dir: Path, encoder_name: str, text_type: str) -> torch.Tensor:
    payload = torch.load(Path(text_outdir(str(feature_dir), encoder_name.replace("-", "/"), "food101", text_type)), map_location="cpu")
    features = payload["features"].float()
    labels = payload["labels"].long()
    num_classes = int(labels.max().item()) + 1
    output = torch.zeros(num_classes, features.shape[1], dtype=torch.float32)
    for label in range(num_classes):
        mask = labels == label
        output[label] = features[mask].mean(dim=0)
    return output


def _load_food101_splits(args: argparse.Namespace) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    encoder_name = args.food101_clip_encoder.replace("/", "-")
    feature_dir = Path(args.food101_feature_dir)
    trainval_path, test_path = _food101_image_paths(feature_dir, encoder_name, args.seed)
    trainval = torch.load(trainval_path, map_location="cpu")
    test = torch.load(test_path, map_location="cpu")
    text_by_class = _average_text_by_class(feature_dir, encoder_name, args.food101_text_type)
    splits: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for split_name in ("train", "val"):
        image_features = trainval[split_name]["features"].float()
        labels = trainval[split_name]["labels"].long()
        text_features = text_by_class[labels]
        splits[split_name] = (image_features, text_features, labels)
    test_features = test["features"].float()
    test_labels = test["labels"].long()
    splits["test"] = (test_features, text_by_class[test_labels], test_labels)
    return splits


def _train_food101_fusion(args: argparse.Namespace) -> tuple[Food101FusionModel, dict[str, float], dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    splits = _load_food101_splits(args)
    train_ds = TensorDataset(*splits["train"])
    val_split = splits["val"]
    device = args.device
    image_dim = int(splits["train"][0].shape[1])
    text_dim = int(splits["train"][1].shape[1])
    num_classes = int(splits["train"][2].max().item()) + 1
    model = Food101FusionModel(image_dim, text_dim, args.food101_hidden_dim, num_classes, args.food101_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.food101_lr)
    loader = DataLoader(train_ds, batch_size=args.food101_batch_size, shuffle=True)
    best_state = None
    best_val_loss = float("inf")
    best_val_acc = 0.0
    stalled = 0
    for _ in range(args.food101_epochs):
        model.train()
        for image_features, text_features, labels in loader:
            image_features = image_features.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image_features, text_features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = _evaluate_food101(model, val_split, device=device, batch_size=args.food101_batch_size, image_active=True, text_active=True)
        if val_loss + 1e-5 < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= args.food101_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = _evaluate_food101(model, splits["test"], device=device, batch_size=args.food101_batch_size, image_active=True, text_active=True)
    return model, {
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }, splits


def _evaluate_food101(
    model: Food101FusionModel,
    split: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: str,
    batch_size: int,
    image_active: bool,
    text_active: bool,
    ablate_block: BlockSpec | None = None,
    alpha: float = 0.0,
) -> tuple[float, float]:
    dataset = TensorDataset(*split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    param = None
    backup = None
    if ablate_block is not None:
        named = dict(model.named_parameters())
        param = named[ablate_block.tensor_name]
        backup = param.data[ablate_block.row_start : ablate_block.row_end].clone()
        param.data[ablate_block.row_start : ablate_block.row_end].mul_(alpha)
    losses: list[float] = []
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for image_features, text_features, labels in loader:
            image_features = image_features.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            logits = model(image_features, text_features, image_active=image_active, text_active=text_active)
            losses.append(float(F.cross_entropy(logits, labels).item()))
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    if param is not None and backup is not None:
        param.data[ablate_block.row_start : ablate_block.row_end].copy_(backup)
    return float(np.mean(losses)), float(correct / max(total, 1))


def _pair_blocks_by_tensor(blocks: list[BlockSpec]) -> dict[str, list[BlockSpec]]:
    grouped: dict[str, list[BlockSpec]] = defaultdict(list)
    for block in blocks:
        grouped[block.tensor_name].append(block)
    return grouped


def _measure_imdb_blocks(
    model: nn.Module,
    dataset,
    split,
    blocks: list[BlockSpec],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    loader = _make_loader(split, batch_size=args.measurement_batch_size, shuffle=False)
    params = {name: param for name, param in model.named_parameters() if any(block.tensor_name == name for block in blocks)}
    measurement_batches = _measurement_batch_indices(len(split.labels), batch_size=args.measurement_batch_size, max_batches=args.measurement_batches, seed=args.seed + 17)
    tensor_blocks = _pair_blocks_by_tensor(blocks)
    block_rows_real: dict[str, list[np.ndarray]] = {block.block_id: [] for block in blocks}
    block_rows_full: dict[str, list[np.ndarray]] = {block.block_id: [] for block in blocks}
    block_rows_shuffle: dict[str, list[list[np.ndarray]]] = {block.block_id: [[] for _ in range(args.shuffle_trials)] for block in blocks}
    device = args.device
    all_modalities = set(dataset.modality_names)
    target_active = {"text"}
    aux_active = {"image"}

    for batch_index, batch_indices in enumerate(measurement_batches):
        modalities = {
            name: torch.as_tensor(split.modalities[name][batch_indices], dtype=torch.float32, device=device)
            for name in dataset.modality_names
        }
        mask = None if split.mask is None else torch.as_tensor(split.mask[batch_indices], device=device)
        labels = torch.as_tensor(split.labels[batch_indices], device=device)

        loss_xy = _imdb_condition_loss(model, dataset.task, modalities, mask, labels, active_modalities=all_modalities)
        loss_x0 = _imdb_condition_loss(model, dataset.task, modalities, mask, labels, active_modalities=target_active)
        loss_0y = _imdb_condition_loss(model, dataset.task, modalities, mask, labels, active_modalities=aux_active)
        loss_00 = _imdb_condition_loss(model, dataset.task, modalities, mask, labels, active_modalities=set())

        grad_xy = _grad_dict_from_loss(loss_xy, params)
        grad_x0 = _grad_dict_from_loss(loss_x0, params)
        grad_0y = _grad_dict_from_loss(loss_0y, params)
        grad_00 = _grad_dict_from_loss(loss_00, params)

        for block in blocks:
            g_xy = _vector_for_block(grad_xy[block.tensor_name], block)
            g_x0 = _vector_for_block(grad_x0[block.tensor_name], block)
            g_0y = _vector_for_block(grad_0y[block.tensor_name], block)
            g_00 = _vector_for_block(grad_00[block.tensor_name], block)
            block_rows_real[block.block_id].append(g_xy - g_x0 - g_0y + g_00)
            block_rows_full[block.block_id].append(g_xy)

        for trial in range(args.shuffle_trials):
            perm = _label_conditional_permutation(labels, seed=args.seed + 1000 + trial * 100 + batch_index).to(device)
            shuffled_modalities = {
                "text": modalities["text"],
                "image": modalities["image"][perm],
            }
            loss_xy_s = _imdb_condition_loss(model, dataset.task, shuffled_modalities, mask, labels, active_modalities=all_modalities)
            loss_0y_s = _imdb_condition_loss(model, dataset.task, shuffled_modalities, mask, labels, active_modalities=aux_active)
            grad_xy_s = _grad_dict_from_loss(loss_xy_s, params)
            grad_0y_s = _grad_dict_from_loss(loss_0y_s, params)
            for block in blocks:
                g_xy_s = _vector_for_block(grad_xy_s[block.tensor_name], block)
                g_0y_s = _vector_for_block(grad_0y_s[block.tensor_name], block)
                g_x0 = _vector_for_block(grad_x0[block.tensor_name], block)
                g_00 = _vector_for_block(grad_00[block.tensor_name], block)
                block_rows_shuffle[block.block_id][trial].append(g_xy_s - g_x0 - g_0y_s + g_00)

    block_scores: dict[str, Any] = {}
    real_scores = {block.block_id: _fisher_stats(block_rows_real[block.block_id]) for block in blocks}
    full_scores = {block.block_id: _fisher_stats(block_rows_full[block.block_id]) for block in blocks}
    random_thresholds = _compute_random_block_thresholds(real_scores, blocks, max_random_block_samples=args.max_random_block_samples, seed=args.seed + 77)
    for block in blocks:
        shuffle_trace = []
        shuffle_lmax = []
        for trial in range(args.shuffle_trials):
            stats = _fisher_stats(block_rows_shuffle[block.block_id][trial])
            shuffle_trace.append(stats["trace"])
            shuffle_lmax.append(stats["lambda_max"])
        trace_p95_shuffle = float(np.quantile(np.asarray(shuffle_trace, dtype=np.float64), 0.95))
        lmax_p95_shuffle = float(np.quantile(np.asarray(shuffle_lmax, dtype=np.float64), 0.95))
        score = real_scores[block.block_id]
        full_trace = full_scores[block.block_id]["trace"]
        random_thresh = random_thresholds[block.block_id]
        supported = (
            score["trace"] > trace_p95_shuffle
            and score["lambda_max"] > lmax_p95_shuffle
            and score["trace"] > random_thresh["trace_p95"]
            and score["lambda_max"] > random_thresh["lambda_max_p95"]
        )
        block_scores[block.block_id] = {
            "block_id": block.block_id,
            "tensor_name": block.tensor_name,
            "row_start": block.row_start,
            "row_end": block.row_end,
            "real": {
                **score,
                "ratio": float(score["trace"] / max(full_trace, 1e-12)),
            },
            "shuffle_null": {
                "trace_p95": trace_p95_shuffle,
                "lambda_max_p95": lmax_p95_shuffle,
            },
            "random_block_null": random_thresh,
            "supported": bool(supported),
        }
    return {
        "pair": "text_image",
        "candidate_blocks": block_scores,
    }


def _measure_food101_blocks(
    model: Food101FusionModel,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    blocks: list[BlockSpec],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_features, text_features, labels = splits["test"]
    batch_indices_list = _measurement_batch_indices(len(labels), batch_size=args.measurement_batch_size, max_batches=args.measurement_batches, seed=args.seed + 17)
    params = {name: param for name, param in model.named_parameters() if any(block.tensor_name == name for block in blocks)}
    block_rows_real: dict[str, list[np.ndarray]] = {block.block_id: [] for block in blocks}
    block_rows_full: dict[str, list[np.ndarray]] = {block.block_id: [] for block in blocks}
    block_rows_shuffle: dict[str, list[list[np.ndarray]]] = {block.block_id: [[] for _ in range(args.shuffle_trials)] for block in blocks}
    device = args.device
    for batch_index, batch_indices in enumerate(batch_indices_list):
        image_batch = image_features[batch_indices].to(device)
        text_batch = text_features[batch_indices].to(device)
        label_batch = labels[batch_indices].to(device)
        loss_xy = F.cross_entropy(model(image_batch, text_batch, image_active=True, text_active=True), label_batch)
        loss_x0 = F.cross_entropy(model(image_batch, text_batch, image_active=True, text_active=False), label_batch)
        loss_0y = F.cross_entropy(model(image_batch, text_batch, image_active=False, text_active=True), label_batch)
        loss_00 = F.cross_entropy(model(image_batch, text_batch, image_active=False, text_active=False), label_batch)
        grad_xy = _grad_dict_from_loss(loss_xy, params)
        grad_x0 = _grad_dict_from_loss(loss_x0, params)
        grad_0y = _grad_dict_from_loss(loss_0y, params)
        grad_00 = _grad_dict_from_loss(loss_00, params)
        for block in blocks:
            g_xy = _vector_for_block(grad_xy[block.tensor_name], block)
            g_x0 = _vector_for_block(grad_x0[block.tensor_name], block)
            g_0y = _vector_for_block(grad_0y[block.tensor_name], block)
            g_00 = _vector_for_block(grad_00[block.tensor_name], block)
            block_rows_real[block.block_id].append(g_xy - g_x0 - g_0y + g_00)
            block_rows_full[block.block_id].append(g_xy)
        for trial in range(args.shuffle_trials):
            perm = _global_permutation(label_batch.shape[0], seed=args.seed + 2000 + trial * 100 + batch_index).to(device)
            shuffled_text = text_batch[perm]
            loss_xy_s = F.cross_entropy(model(image_batch, shuffled_text, image_active=True, text_active=True), label_batch)
            loss_0y_s = F.cross_entropy(model(image_batch, shuffled_text, image_active=False, text_active=True), label_batch)
            grad_xy_s = _grad_dict_from_loss(loss_xy_s, params)
            grad_0y_s = _grad_dict_from_loss(loss_0y_s, params)
            for block in blocks:
                g_xy_s = _vector_for_block(grad_xy_s[block.tensor_name], block)
                g_0y_s = _vector_for_block(grad_0y_s[block.tensor_name], block)
                g_x0 = _vector_for_block(grad_x0[block.tensor_name], block)
                g_00 = _vector_for_block(grad_00[block.tensor_name], block)
                block_rows_shuffle[block.block_id][trial].append(g_xy_s - g_x0 - g_0y_s + g_00)

    block_scores: dict[str, Any] = {}
    real_scores = {block.block_id: _fisher_stats(block_rows_real[block.block_id]) for block in blocks}
    full_scores = {block.block_id: _fisher_stats(block_rows_full[block.block_id]) for block in blocks}
    random_thresholds = _compute_random_block_thresholds(real_scores, blocks, max_random_block_samples=args.max_random_block_samples, seed=args.seed + 77)
    for block in blocks:
        shuffle_trace = []
        shuffle_lmax = []
        for trial in range(args.shuffle_trials):
            stats = _fisher_stats(block_rows_shuffle[block.block_id][trial])
            shuffle_trace.append(stats["trace"])
            shuffle_lmax.append(stats["lambda_max"])
        trace_p95_shuffle = float(np.quantile(np.asarray(shuffle_trace, dtype=np.float64), 0.95))
        lmax_p95_shuffle = float(np.quantile(np.asarray(shuffle_lmax, dtype=np.float64), 0.95))
        score = real_scores[block.block_id]
        full_trace = full_scores[block.block_id]["trace"]
        random_thresh = random_thresholds[block.block_id]
        supported = (
            score["trace"] > trace_p95_shuffle
            and score["lambda_max"] > lmax_p95_shuffle
            and score["trace"] > random_thresh["trace_p95"]
            and score["lambda_max"] > random_thresh["lambda_max_p95"]
        )
        block_scores[block.block_id] = {
            "block_id": block.block_id,
            "tensor_name": block.tensor_name,
            "row_start": block.row_start,
            "row_end": block.row_end,
            "real": {
                **score,
                "ratio": float(score["trace"] / max(full_trace, 1e-12)),
            },
            "shuffle_null": {
                "trace_p95": trace_p95_shuffle,
                "lambda_max_p95": lmax_p95_shuffle,
            },
            "random_block_null": random_thresh,
            "supported": bool(supported),
        }
    return {
        "pair": "image_text",
        "candidate_blocks": block_scores,
    }


def _pick_top_and_random_block(block_scores: dict[str, Any], blocks: list[BlockSpec], *, seed: int) -> tuple[BlockSpec | None, BlockSpec | None]:
    if not block_scores:
        return None, None
    supported_ids = [name for name, payload in block_scores.items() if payload.get("supported", False)]
    if supported_ids:
        top_block_id = max(supported_ids, key=lambda name: block_scores[name]["real"]["trace"])
    else:
        top_block_id = max(block_scores, key=lambda name: block_scores[name]["real"]["trace"])
    block_by_id = {block.block_id: block for block in blocks}
    top_block = block_by_id[top_block_id]
    same_tensor = [block for block in blocks if block.tensor_name == top_block.tensor_name and block.block_id != top_block.block_id]
    if not same_tensor:
        return top_block, None
    rng = np.random.default_rng(seed)
    random_block = same_tensor[int(rng.integers(0, len(same_tensor)))]
    return top_block, random_block


def _intervention_imdb(model: nn.Module, dataset_task: str, split, blocks: list[BlockSpec], block_scores: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    top_block, random_block = _pick_top_and_random_block(block_scores, blocks, seed=args.seed + 909)
    if top_block is None:
        return {}
    baseline_joint = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"text", "image"})
    baseline_text = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"text"})
    baseline_image = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"image"})
    top_joint = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"text", "image"}, ablate_block=top_block, alpha=args.intervention_alpha)
    top_text = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"text"}, ablate_block=top_block, alpha=args.intervention_alpha)
    top_image = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"image"}, ablate_block=top_block, alpha=args.intervention_alpha)
    result = {
        "baseline": {"joint_loss": baseline_joint, "text_only_loss": baseline_text, "image_only_loss": baseline_image},
        "top_block": {
            "block_id": top_block.block_id,
            "joint_loss": top_joint,
            "text_only_loss": top_text,
            "image_only_loss": top_image,
            "joint_delta": float(top_joint - baseline_joint),
            "text_only_delta": float(top_text - baseline_text),
            "image_only_delta": float(top_image - baseline_image),
        },
    }
    if random_block is not None:
        rand_joint = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"text", "image"}, ablate_block=random_block, alpha=args.intervention_alpha)
        rand_text = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"text"}, ablate_block=random_block, alpha=args.intervention_alpha)
        rand_image = _evaluate_imdb_loss(model, dataset_task, split, device=args.device, batch_size=args.measurement_batch_size, active_modalities={"image"}, ablate_block=random_block, alpha=args.intervention_alpha)
        result["random_block"] = {
            "block_id": random_block.block_id,
            "joint_loss": rand_joint,
            "text_only_loss": rand_text,
            "image_only_loss": rand_image,
            "joint_delta": float(rand_joint - baseline_joint),
            "text_only_delta": float(rand_text - baseline_text),
            "image_only_delta": float(rand_image - baseline_image),
        }
    return result


def _intervention_food101(model: Food101FusionModel, splits: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], blocks: list[BlockSpec], block_scores: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    top_block, random_block = _pick_top_and_random_block(block_scores, blocks, seed=args.seed + 909)
    if top_block is None:
        return {}
    baseline_joint = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=True, text_active=True)
    baseline_image = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=True, text_active=False)
    baseline_text = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=False, text_active=True)
    top_joint = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=True, text_active=True, ablate_block=top_block, alpha=args.intervention_alpha)
    top_image = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=True, text_active=False, ablate_block=top_block, alpha=args.intervention_alpha)
    top_text = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=False, text_active=True, ablate_block=top_block, alpha=args.intervention_alpha)
    result = {
        "baseline": {
            "joint_loss": baseline_joint[0],
            "joint_acc": baseline_joint[1],
            "image_only_loss": baseline_image[0],
            "image_only_acc": baseline_image[1],
            "text_only_loss": baseline_text[0],
            "text_only_acc": baseline_text[1],
        },
        "top_block": {
            "block_id": top_block.block_id,
            "joint_loss": top_joint[0],
            "joint_acc": top_joint[1],
            "image_only_loss": top_image[0],
            "image_only_acc": top_image[1],
            "text_only_loss": top_text[0],
            "text_only_acc": top_text[1],
            "joint_loss_delta": float(top_joint[0] - baseline_joint[0]),
            "image_only_loss_delta": float(top_image[0] - baseline_image[0]),
            "text_only_loss_delta": float(top_text[0] - baseline_text[0]),
            "joint_acc_delta": float(top_joint[1] - baseline_joint[1]),
            "image_only_acc_delta": float(top_image[1] - baseline_image[1]),
            "text_only_acc_delta": float(top_text[1] - baseline_text[1]),
        },
    }
    if random_block is not None:
        rand_joint = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=True, text_active=True, ablate_block=random_block, alpha=args.intervention_alpha)
        rand_image = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=True, text_active=False, ablate_block=random_block, alpha=args.intervention_alpha)
        rand_text = _evaluate_food101(model, splits["test"], device=args.device, batch_size=args.food101_batch_size, image_active=False, text_active=True, ablate_block=random_block, alpha=args.intervention_alpha)
        result["random_block"] = {
            "block_id": random_block.block_id,
            "joint_loss": rand_joint[0],
            "joint_acc": rand_joint[1],
            "image_only_loss": rand_image[0],
            "image_only_acc": rand_image[1],
            "text_only_loss": rand_text[0],
            "text_only_acc": rand_text[1],
            "joint_loss_delta": float(rand_joint[0] - baseline_joint[0]),
            "image_only_loss_delta": float(rand_image[0] - baseline_image[0]),
            "text_only_loss_delta": float(rand_text[0] - baseline_text[0]),
            "joint_acc_delta": float(rand_joint[1] - baseline_joint[1]),
            "image_only_acc_delta": float(rand_image[1] - baseline_image[1]),
            "text_only_acc_delta": float(rand_text[1] - baseline_text[1]),
        }
    return result


def run_imdb(args: argparse.Namespace) -> dict[str, Any]:
    print(f"[imdb][seed={args.seed}] loading dataset", flush=True)
    dataset = load_imdb(args.imdb_path)
    print(f"[imdb][seed={args.seed}] training model", flush=True)
    model, train_metrics = train_joint_model(
        dataset,
        device=args.device,
        encoder_hidden_dim=args.imdb_encoder_hidden_dim,
        fusion_dim=args.imdb_fusion_dim,
        fusion_layers=args.imdb_fusion_layers,
        num_heads=args.imdb_num_heads,
        ffn_dim=args.imdb_ffn_dim,
        dropout=args.imdb_dropout,
        batch_size=args.imdb_batch_size,
        epochs=args.imdb_epochs,
        lr=args.imdb_lr,
        patience=args.imdb_patience,
        seed=args.seed,
        architecture="shared_head",
    )
    print(f"[imdb][seed={args.seed}] collecting candidate blocks", flush=True)
    blocks = _candidate_blocks_for_imdb(model, min_block_rows=args.min_block_rows, max_blocks_per_tensor=args.max_blocks_per_tensor)
    print(f"[imdb][seed={args.seed}] measuring interaction Fisher on {len(blocks)} blocks", flush=True)
    measurement = _measure_imdb_blocks(model, dataset, dataset.test, blocks, args=args)
    print(f"[imdb][seed={args.seed}] running intervention", flush=True)
    intervention = _intervention_imdb(model, dataset.task, dataset.test, blocks, measurement["candidate_blocks"], args=args)
    return {
        "dataset": "imdb",
        "seed": args.seed,
        "setting": {
            "model": "shared_head",
            "target_modality": "text",
            "auxiliary_modality": "image",
            "training_hparams": {
                "encoder_hidden_dim": args.imdb_encoder_hidden_dim,
                "fusion_dim": args.imdb_fusion_dim,
                "fusion_layers": args.imdb_fusion_layers,
                "num_heads": args.imdb_num_heads,
                "ffn_dim": args.imdb_ffn_dim,
                "dropout": args.imdb_dropout,
                "batch_size": args.imdb_batch_size,
                "epochs": args.imdb_epochs,
                "lr": args.imdb_lr,
                "patience": args.imdb_patience,
            },
            "candidate_tensors": sorted({block.tensor_name for block in blocks}),
        },
        "training": train_metrics,
        "measurement": measurement,
        "intervention": intervention,
    }


def run_food101(args: argparse.Namespace) -> dict[str, Any]:
    print(f"[food101][seed={args.seed}] training fusion model", flush=True)
    model, train_metrics, splits = _train_food101_fusion(args)
    print(f"[food101][seed={args.seed}] collecting candidate blocks", flush=True)
    blocks = _candidate_blocks_for_food101(model, min_block_rows=args.min_block_rows, max_blocks_per_tensor=args.max_blocks_per_tensor)
    print(f"[food101][seed={args.seed}] measuring interaction Fisher on {len(blocks)} blocks", flush=True)
    measurement = _measure_food101_blocks(model, splits, blocks, args=args)
    print(f"[food101][seed={args.seed}] running intervention", flush=True)
    intervention = _intervention_food101(model, splits, blocks, measurement["candidate_blocks"], args=args)
    return {
        "dataset": "food101",
        "seed": args.seed,
        "setting": {
            "model": "concat_mlp_fusion",
            "target_modality": "image",
            "auxiliary_modality": "text",
            "training_hparams": {
                "hidden_dim": args.food101_hidden_dim,
                "dropout": args.food101_dropout,
                "epochs": args.food101_epochs,
                "lr": args.food101_lr,
                "batch_size": args.food101_batch_size,
                "patience": args.food101_patience,
                "clip_encoder": args.food101_clip_encoder,
                "text_type": args.food101_text_type,
            },
            "candidate_tensors": sorted({block.tensor_name for block in blocks}),
        },
        "training": train_metrics,
        "measurement": measurement,
        "intervention": intervention,
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "imdb":
        payload = run_imdb(args)
    else:
        payload = run_food101(args)
    payload["config"] = {
        "device": args.device,
        "measurement_batch_size": args.measurement_batch_size,
        "measurement_batches": args.measurement_batches,
        "shuffle_trials": args.shuffle_trials,
        "max_random_block_samples": args.max_random_block_samples,
        "min_block_rows": args.min_block_rows,
        "max_blocks_per_tensor": args.max_blocks_per_tensor,
        "intervention_alpha": args.intervention_alpha,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
