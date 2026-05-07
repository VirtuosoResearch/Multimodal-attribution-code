"""Budgeted multimodal data selection experiment.

Usage:
    conda activate vlm
    python -m src.data_selection.experiment \
        --datasets avmnist mosi --model concat_mlp \
        --selection-method random --budget 0.2

The experiment keeps one main modality for every training sample, keeps the
auxiliary modality only for a selected training subset, and evaluates on full
multimodal validation/test data. New data attribution methods can plug in by
registering a selector in ``data_selection.selection``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_selection.datasets import DatasetTensors, SplitTensors, load_dataset
from data_selection.models import ConcatSelectionMLP, TokenTransformerSelectionModel
from data_selection.selection import (
    SelectionContext,
    available_selectors,
    build_selector,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_dataset(
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
) -> DatasetTensors:
    return load_dataset(
        dataset_name,
        seed=seed,
        full_modality=args.main_modality,
        eval_split=args.eval_split,
        standardize=args.standardize,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_seq_len=args.max_seq_len,
    )


def _make_loader(
    split: SplitTensors,
    aux_available: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(split.main, split.aux, split.labels, aux_available.float()),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def _num_outputs(task: str, num_classes: int) -> int:
    if task == "binary":
        return 1
    return num_classes


def build_model(
    args: argparse.Namespace,
    *,
    main_shape: tuple[int, ...],
    aux_shape: tuple[int, ...],
    task: str,
    num_classes: int,
) -> nn.Module:
    out_dim = _num_outputs(task, num_classes)
    if args.model == "concat_mlp":
        return ConcatSelectionMLP(
            main_shape,
            aux_shape,
            args.hidden_dim,
            out_dim,
            depth=args.mlp_depth,
            dropout=args.dropout,
            include_mask_feature=not args.no_mask_feature,
            encoder_layers=args.encoder_layers,
            num_heads=args.num_heads,
        )
    if args.model == "transformer":
        return TokenTransformerSelectionModel(
            main_shape,
            aux_shape,
            out_dim,
            proj_dim=args.proj_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_mult=args.ffn_mult,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {args.model}")


def compute_loss(task: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if task == "multiclass":
        return nn.functional.cross_entropy(logits, labels.long().view(-1))
    if task == "binary":
        return nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1), labels.float().view(-1))
    if task == "multilabel":
        return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
    raise ValueError(f"Unknown task: {task}")


def _score_predictions(
    task: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, int]:
    if task == "multiclass":
        correct = int((logits.argmax(dim=1).cpu() == labels.long().view(-1)).sum())
        return float(correct), int(labels.shape[0])
    if task == "binary":
        preds = (logits.view(-1).cpu() > 0).long()
        correct = int((preds == labels.long().view(-1)).sum())
        return float(correct), int(labels.shape[0])

    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds = (probs >= 0.5).astype(np.float32)
    tp = (preds * labels_np).sum(axis=1)
    precision = tp / np.maximum(preds.sum(axis=1), 1.0)
    recall = tp / np.maximum(labels_np.sum(axis=1), 1.0)
    f1 = np.where(
        precision + recall > 0,
        2 * precision * recall / np.maximum(precision + recall, 1e-8),
        0.0,
    )
    return float(f1.sum()), int(labels_np.shape[0])


def evaluate(
    model: nn.Module,
    split: SplitTensors,
    *,
    task: str,
    batch_size: int,
    device: str,
    use_full_aux: bool,
) -> dict[str, float]:
    aux_available = torch.ones(split.main.shape[0])
    if not use_full_aux:
        aux_available.zero_()
    loader = _make_loader(split, aux_available, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_score = 0.0
    total_count = 0
    with torch.inference_mode():
        for main, aux, labels, mask in loader:
            logits = model(main.to(device), aux.to(device), mask.to(device))
            loss = compute_loss(task, logits, labels.to(device))
            score, count = _score_predictions(task, logits, labels)
            total_loss += float(loss.item()) * count
            total_score += score
            total_count += count
    metric_name = "sample_f1" if task == "multilabel" else "accuracy"
    return {
        metric_name: total_score / max(total_count, 1),
        "loss": total_loss / max(total_count, 1),
    }


def train_one_arm(
    data: DatasetTensors,
    args: argparse.Namespace,
    *,
    arm_name: str,
    aux_mask: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    model = build_model(
        args,
        main_shape=data.main_shape,
        aux_shape=data.aux_shape,
        task=data.task,
        num_classes=data.num_classes,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = _make_loader(
        data.train,
        aux_mask,
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_state = None
    best_eval_loss = float("inf")
    best_epoch = -1
    patience_left = args.patience
    completed_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for main, aux, labels, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(main.to(args.device), aux.to(args.device), mask.to(args.device))
            loss = compute_loss(data.task, logits, labels.to(args.device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.shape[0]
            total_count += labels.shape[0]
        completed_epochs = epoch + 1

        valid_metrics = None
        if args.patience > 0 or args.verbose:
            valid_metrics = evaluate(
                model,
                data.valid,
                task=data.task,
                batch_size=args.eval_batch_size,
                device=args.device,
                use_full_aux=(arm_name != "main_only"),
            )
        if args.patience > 0 and valid_metrics is not None:
            if valid_metrics["loss"] < best_eval_loss:
                best_eval_loss = valid_metrics["loss"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_left = args.patience
            else:
                patience_left -= 1

        metric_key = "sample_f1" if data.task == "multilabel" else "accuracy"
        train_loss = total_loss / max(total_count, 1)
        message = (
            f"    {arm_name} epoch={epoch + 1:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f}"
        )
        if valid_metrics is not None:
            message += (
                f" valid_loss={valid_metrics['loss']:.4f} "
                f"valid_{metric_key}={valid_metrics[metric_key]:.4f}"
            )
        print(message, flush=True)
        if args.patience > 0 and patience_left <= 0:
            break

    if args.patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
    final_valid = evaluate(
        model,
        data.valid,
        task=data.task,
        batch_size=args.eval_batch_size,
        device=args.device,
        use_full_aux=(arm_name != "main_only"),
    )
    final_eval = evaluate(
        model,
        data.eval,
        task=data.task,
        batch_size=args.eval_batch_size,
        device=args.device,
        use_full_aux=(arm_name != "main_only"),
    )
    return {
        "arm": arm_name,
        "train_seed": seed,
        "selected_aux_count": int(aux_mask.sum().item()),
        "selected_aux_fraction": float(aux_mask.float().mean().item()),
        "best_epoch": best_epoch + 1 if args.patience > 0 else completed_epochs,
        "valid": final_valid,
        "eval": final_eval,
    }


def run_dataset(dataset_name: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    print(f"[{dataset_name}] seed={seed}: loading data")
    data = _prepare_dataset(dataset_name, seed, args)
    n_train = data.train.main.shape[0]
    selector = build_selector(args)
    selection = selector.select(
        SelectionContext(
            dataset_name=dataset_name,
            seed=seed,
            budget=args.budget,
            labels=data.train.labels,
            main_features=data.train.main,
            aux_features=data.train.aux,
            output_dir=Path(args.output_dir),
        )
    )
    selected_fraction = float(selection.mask.float().mean().item())
    if args.strict_budget_tolerance >= 0:
        diff = abs(selected_fraction - args.budget)
        if diff > args.strict_budget_tolerance:
            raise ValueError(
                f"Selection method {selection.name} selected fraction "
                f"{selected_fraction:.6f}, expected budget {args.budget:.6f} "
                f"within tolerance {args.strict_budget_tolerance:.6f}")

    arms = []
    arm_specs = []
    requested_arms = ["selected" if arm == "random" else arm for arm in args.arms]
    if "full" in requested_arms:
        arm_specs.append(("full", torch.ones(n_train, dtype=torch.bool)))
    if "main_only" in requested_arms:
        arm_specs.append(("main_only", torch.zeros(n_train, dtype=torch.bool)))
    if "selected" in requested_arms:
        arm_specs.append((selection.name, selection.mask))

    for arm_name, mask in arm_specs:
        print(
            f"[{dataset_name}] seed={seed}: training {arm_name} "
            f"({int(mask.sum())}/{n_train} aux samples)"
        )
        arm_seed_offsets = {"full": 11, "main_only": 23, "random": 37}
        train_seed = seed + args.model_seed_offset
        if not args.same_init_across_arms:
            train_seed = seed + arm_seed_offsets.get(arm_name, 101)
        arms.append(train_one_arm(
            data,
            args,
            arm_name=arm_name,
            aux_mask=mask,
            seed=train_seed,
        ))

    metric_key = "sample_f1" if data.task == "multilabel" else "accuracy"
    summary = {arm["arm"]: arm["eval"][metric_key] for arm in arms}
    if "full" in summary and "main_only" in summary:
        summary["full_minus_main_only"] = summary["full"] - summary["main_only"]
    if "full" in summary and selection.name in summary:
        summary[f"{selection.name}_minus_full"] = summary[selection.name] - summary["full"]
    if "main_only" in summary and selection.name in summary:
        summary[f"{selection.name}_minus_main_only"] = (
            summary[selection.name] - summary["main_only"])

    return {
        "dataset": dataset_name,
        "seed": seed,
        "model": args.model,
        "selection_method": selection.name,
        "selection_metadata": selection.metadata,
        "task": data.task,
        "metric": metric_key,
        "main_modality": args.main_modality,
        "full_modality": data.full_modality,
        "budgeted_modality": data.budgeted_modality,
        "budget": args.budget,
        "eval_split": data.eval_split,
        "shapes": {"main": data.main_shape, "aux": data.aux_shape},
        "selected_indices": selection.indices,
        "arms": arms,
        "summary": summary,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _budget_slug(budget: float) -> str:
    return f"budget_{budget:.6g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Budgeted multimodal data selection experiment")
    parser.add_argument("--datasets", nargs="+", default=["avmnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--model", choices=["concat_mlp", "transformer"],
                        default="concat_mlp")
    parser.add_argument("--main-modality", "--full-modality",
                        choices=["x", "y"], default="x",
                        help=("Modality kept for every training sample. "
                              "The other modality is budgeted by the selector."))
    parser.add_argument("--selection-method", choices=available_selectors(),
                        default="random")
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--arms", nargs="+",
                        choices=["full", "main_only", "selected", "random"],
                        default=["full", "main_only", "selected"])
    parser.add_argument("--stratified-random", action="store_true")
    parser.add_argument("--selection-seed-offset", type=int, default=1000)
    parser.add_argument(
        "--num-select-groups",
        type=int,
        default=1,
        help="Number of LDS groups to keep for selection_method=random_group.",
    )
    parser.add_argument("--model-seed-offset", type=int, default=2000)
    parser.add_argument("--same-init-across-arms",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-file", default=None)
    parser.add_argument("--index-template", default=None)
    parser.add_argument(
        "--group-result-root",
        default="src/attribution/results/lds_real_train",
        help="Root containing LDS result runs for selection_method=group_index.",
    )
    parser.add_argument(
        "--group-run-name",
        default=None,
        help=(
            "LDS run name used by selection_method=group_index. If omitted, "
            "the selector reads metadata.hparams.run_name from the index JSON."
        ),
    )
    parser.add_argument(
        "--group-seed",
        type=int,
        default=None,
        help=(
            "LDS result seed used by selection_method=group_index. If omitted, "
            "the selector reads metadata.surrogate_seed from the index JSON, "
            "falling back to 0."
        ),
    )
    parser.add_argument("--strict-budget-tolerance", type=float, default=-1.0,
                        help=("Fail if selected fraction differs from budget "
                              "by more than this amount; negative disables."))
    parser.add_argument("--eval-split", choices=["auto", "val", "test"],
                        default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--no-mask-feature", action="store_true")
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--output-dir",
                        default="src/data_selection/results/random_budget")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / _budget_slug(args.budget)
    all_results = []
    for dataset_name in args.datasets:
        for seed in args.seeds:
            result = run_dataset(dataset_name, seed, args)
            all_results.append(result)
            result_path = (
                out_dir / args.model / args.selection_method
                / dataset_name / f"seed{seed}.json")
            _write_json(result_path, result)
            print(f"[{dataset_name}] seed={seed}: saved {result_path}")

    combined_path = out_dir / args.model / args.selection_method / "all_results.json"
    _write_json(combined_path, all_results)

    print("\nSummary")
    for result in all_results:
        summary = result["summary"]
        metric = result["metric"]
        pieces = [
            f"{name}={value:.4f}"
            for name, value in summary.items()
            if isinstance(value, (int, float))
        ]
        print(
            f"  {result['dataset']} seed={result['seed']} "
            f"metric={metric}: " + ", ".join(pieces)
        )
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
