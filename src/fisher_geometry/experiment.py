"""Fisher Geometry experiment runner.

Trains a multimodal fusion model (Concat-MLP or Cross-Attention Transformer)
under three arms (B=unimodal, C_related, C_unrelated) and measures Fisher
geometry metrics at training checkpoints.

Usage:
    conda activate vlm

    # Cross-Attention Transformer (modern)
    python -m fisher_geometry.experiment \
        --model cross_attention \
        --datasets food101 oxford_pets imdb humor sarcasm \
        --seeds 1 2 3 --device cuda:0

    # Concat-MLP (baseline)
    python -m fisher_geometry.experiment \
        --model concat_mlp \
        --datasets food101 oxford_pets imdb humor sarcasm \
        --seeds 1 2 3 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_geometry.models import (
    ConcatFusionModel,
    CrossAttentionFusionModel,
    SharedMLPFusionModel,
    TrafficPairGNNMLPFusionModel,
)
from fisher_geometry.fisher_metrics import (
    compute_loss,
    per_example_gradients,
    fisher_eigen,
    subspace_overlap,
    residual_energy,
    mean_sample_cosine,
)
from fisher_geometry.data import load_dataset, permute_aux, make_loader


# ---------------------------------------------------------------------------
# Seed and checkpointing utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class ArmCheckpoint:
    step: int
    fraction: float
    state_dict: dict[str, torch.Tensor]


def _checkpoint_fractions(count: int) -> tuple[float, ...]:
    count = max(int(count), 1)
    return tuple(i / count for i in range(1, count + 1))


def _capture_checkpoints(
    model: nn.Module,
    checkpoints: list[ArmCheckpoint],
    step: int,
    total_steps: int,
    fractions: tuple[float, ...],
) -> list[ArmCheckpoint]:
    updated = list(checkpoints)
    frac_now = step / max(total_steps, 1)
    for frac in fractions:
        if any(abs(c.fraction - frac) < 1e-9 for c in updated):
            continue
        if frac_now >= frac:
            updated.append(ArmCheckpoint(
                step=step, fraction=frac,
                state_dict={k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}))
    return updated


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             labels: torch.Tensor, *, device: str, batch_size: int,
             task: str, use_x: bool = True, use_y: bool = True) -> float:
    loader = make_loader(x, y, labels, batch_size, False)
    model.eval()
    if task == "multiclass":
        correct = total = 0
        with torch.inference_mode():
            for bx, by, bl in loader:
                logits = model(bx.to(device), by.to(device),
                               use_x=use_x, use_y=use_y)
                correct += int((logits.argmax(1) == bl.to(device)).sum())
                total += bl.shape[0]
        return correct / max(total, 1)
    elif task == "binary":
        correct = total = 0
        with torch.inference_mode():
            for bx, by, bl in loader:
                logits = model(bx.to(device), by.to(device),
                               use_x=use_x, use_y=use_y)
                preds = (logits.view(-1) > 0).long()
                correct += int((preds == bl.to(device).view(-1)).sum())
                total += bl.shape[0]
        return correct / max(total, 1)
    else:
        all_probs, all_labels = [], []
        with torch.inference_mode():
            for bx, by, bl in loader:
                logits = model(bx.to(device), by.to(device),
                               use_x=use_x, use_y=use_y)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(bl.numpy())
        probs = np.concatenate(all_probs)
        labels_np = np.concatenate(all_labels)
        preds = (probs >= 0.5).astype(np.float32)
        tp = (preds * labels_np).sum(axis=1)
        prec = tp / np.maximum(preds.sum(axis=1), 1.0)
        rec = tp / np.maximum(labels_np.sum(axis=1), 1.0)
        f1 = np.where(prec + rec > 0,
                       2 * prec * rec / np.maximum(prec + rec, 1e-8), 0.0)
        return float(np.mean(f1))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    args: argparse.Namespace,
    *,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    model_kwargs = model_kwargs or {}
    if model_type == "concat_mlp":
        return ConcatFusionModel(
            dim_x, dim_y, hidden_dim=args.hidden_dim,
            num_classes=num_classes, dropout=args.dropout)
    elif model_type == "shared_mlp":
        return SharedMLPFusionModel(
            dim_x, dim_y, hidden_dim=args.hidden_dim,
            num_classes=num_classes, dropout=args.dropout)
    elif model_type == "cross_attention":
        return CrossAttentionFusionModel(
            dim_x, dim_y, proj_dim=args.proj_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            num_classes=num_classes, dropout=args.dropout)
    elif model_type == "traffic_pair_gnn_mlp":
        if (
            "x_node_dim" not in model_kwargs
            or "x_edge_dim" not in model_kwargs
            or "y_node_dim" not in model_kwargs
        ):
            raise ValueError(
                "traffic_pair_gnn_mlp requires model_kwargs with "
                "x_node_dim, x_edge_dim, and y_node_dim"
            )
        return TrafficPairGNNMLPFusionModel(
            dim_y=dim_y,
            y_node_dim=int(model_kwargs["y_node_dim"]),
            x_node_dim=int(model_kwargs["x_node_dim"]),
            x_edge_dim=int(model_kwargs["x_edge_dim"]),
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            message_passing_layers=int(getattr(args, "traffic_gnn_layers", 2)),
            classifier_layers=int(getattr(args, "traffic_classifier_layers", 2)),
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Train + measure Fisher geometry at checkpoints
# ---------------------------------------------------------------------------

def _train_and_measure(
    data: dict[str, Any],
    *,
    arm_name: str,
    model_type: str,
    device: str,
    args: argparse.Namespace,
    fisher_samples: int,
    checkpoint_count: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    set_seed(seed)
    dim_x = data["train_x"].shape[1]
    dim_y = data["train_y"].shape[1]
    num_classes = data["num_classes"]
    task = data["task"]

    model = build_model(model_type, dim_x, dim_y, num_classes, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    loader = make_loader(data["train_x"], data["train_y"],
                         data["train_labels"], args.batch_size, True)
    base_steps = len(loader)
    use_y_train = (arm_name != "B")
    total_steps = args.epochs * base_steps * (2 if arm_name == "B" else 1)

    fractions = _checkpoint_fractions(checkpoint_count)
    checkpoints: list[ArmCheckpoint] = []
    global_step = 0

    for _ in range(args.epochs):
        model.train()
        iters = 2 if arm_name == "B" else 1
        for _ in range(iters):
            for bx, by, bl in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(bx.to(device), by.to(device),
                               use_x=True, use_y=use_y_train)
                loss = compute_loss(task, logits, bl.to(device))
                loss.backward()
                optimizer.step()
                global_step += 1
                checkpoints = _capture_checkpoints(
                    model, checkpoints, global_step, total_steps, fractions)

    if not any(abs(c.fraction - 1.0) < 1e-9 for c in checkpoints):
        checkpoints.append(ArmCheckpoint(
            step=global_step, fraction=1.0,
            state_dict={k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}))
    checkpoints = sorted(checkpoints, key=lambda c: c.fraction)

    # --- Measure Fisher geometry at each checkpoint ---
    metrics = []
    for ckpt in checkpoints:
        m = build_model(model_type, dim_x, dim_y, num_classes, args).to(device)
        m.load_state_dict(ckpt.state_dict)

        acc = evaluate(m, data["val_x"], data["val_y"], data["val_labels"],
                       device=device, batch_size=512, task=task,
                       use_x=True, use_y=use_y_train)
        # Y-only and X-only baselines for label leakage check
        acc_x_only = evaluate(m, data["val_x"], data["val_y"],
                              data["val_labels"], device=device,
                              batch_size=512, task=task,
                              use_x=True, use_y=False)
        acc_y_only = evaluate(m, data["val_x"], data["val_y"],
                              data["val_labels"], device=device,
                              batch_size=512, task=task,
                              use_x=False, use_y=True)

        # Cap fisher_samples to actual validation size
        actual_fisher = min(fisher_samples, data["val_x"].shape[0])

        gx = per_example_gradients(
            m, data["val_x"], data["val_y"], data["val_labels"],
            device=device, max_samples=actual_fisher, task=task,
            use_x=True, use_y=False)
        gy = per_example_gradients(
            m, data["val_x"], data["val_y"], data["val_labels"],
            device=device, max_samples=actual_fisher, task=task,
            use_x=False, use_y=use_y_train)
        gxy = per_example_gradients(
            m, data["val_x"], data["val_y"], data["val_labels"],
            device=device, max_samples=actual_fisher, task=task,
            use_x=True, use_y=use_y_train)

        fx = fisher_eigen(gx, top_k=top_k)
        fy = fisher_eigen(gy, top_k=top_k)
        fxy = fisher_eigen(gxy, top_k=top_k)

        metrics.append({
            "fraction": ckpt.fraction,
            "acc_xy": acc,
            "acc_x_only": acc_x_only,
            "acc_y_only": acc_y_only,
            "fisher_samples_used": actual_fisher,
            "F_X_trace": fx["trace"],
            "F_Y_trace": fy["trace"],
            "F_XY_trace": fxy["trace"],
            "delta_trace": fxy["trace"] - fx["trace"],
            "F_X_logdet": fx["logdet"],
            "F_XY_logdet": fxy["logdet"],
            "logdet_gain": fxy["logdet"] - fx["logdet"],
            "F_X_topk_mass": fx["topk_mass"],
            "F_Y_topk_mass": fy["topk_mass"],
            "F_XY_topk_mass": fxy["topk_mass"],
            "subspace_overlap": subspace_overlap(
                fx["top_k_vecs"], fy["top_k_vecs"]),
            "residual_energy": residual_energy(gy, fx["top_k_vecs"]),
            "mean_sample_cos": mean_sample_cosine(gx, gy),
            "F_X_rank": fx["rank"],
            "F_Y_rank": fy["rank"],
            "F_XY_rank": fxy["rank"],
            "top_k": top_k,
        })

    return {"arm": arm_name, "checkpoints": metrics}


# ---------------------------------------------------------------------------
# Average multiple unrelated arm runs (different permutation seeds)
# ---------------------------------------------------------------------------

def _average_arm_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average checkpoint metrics across multiple unrelated arm runs."""
    if len(runs) == 1:
        return runs[0]
    # All runs should have the same number of checkpoints
    n_ckpts = len(runs[0]["checkpoints"])
    averaged_ckpts = []
    for ci in range(n_ckpts):
        merged = {}
        for key in runs[0]["checkpoints"][ci]:
            vals = [r["checkpoints"][ci][key] for r in runs]
            if isinstance(vals[0], (int, float)):
                merged[key] = float(np.mean(vals))
            else:
                merged[key] = vals[0]  # non-numeric: take first
        averaged_ckpts.append(merged)
    return {"arm": "C_unrelated", "checkpoints": averaged_ckpts}


# ---------------------------------------------------------------------------
# Single-dataset driver
# ---------------------------------------------------------------------------

def run_single(dataset_name: str, seed: int,
               args: argparse.Namespace) -> dict[str, Any]:
    print(f"\n  [{dataset_name}] seed={seed}: loading data...")
    data_related = load_dataset(
        dataset_name,
        seed,
        text_backbone=args.text_backbone,
        vision_backbone=args.vision_backbone,
        feature_device=args.feature_device,
        backbone_batch_size=args.backbone_batch_size,
        download_datasets=args.download_datasets,
        backbone_cache_dir=args.backbone_cache_dir,
    )

    kw = dict(model_type=args.model, device=args.device, args=args,
              fisher_samples=args.fisher_samples,
              checkpoint_count=args.checkpoint_count, top_k=args.top_k)

    print(f"  [{dataset_name}] seed={seed}: arm B (x-only)...")
    arm_b = _train_and_measure(data_related, arm_name="B",
                               seed=seed + 100, **kw)
    print(f"  [{dataset_name}] seed={seed}: arm C_related...")
    arm_rel = _train_and_measure(data_related, arm_name="C_related",
                                 seed=seed + 200, **kw)

    # Multiple permutation seeds for unrelated arm (Concern 7)
    perm_seeds = [seed + 1000, seed + 2000, seed + 3000]
    unrel_runs = []
    for pi, ps in enumerate(perm_seeds):
        data_unrelated = permute_aux(data_related, ps)
        print(f"  [{dataset_name}] seed={seed}: arm C_unrelated "
              f"(perm {pi+1}/{len(perm_seeds)})...")
        run = _train_and_measure(data_unrelated, arm_name="C_unrelated",
                                 seed=seed + 300 + pi * 10, **kw)
        unrel_runs.append(run)
    # Average the unrelated runs' checkpoint metrics
    arm_unrel = _average_arm_runs(unrel_runs)

    def _final(arm):
        return arm["checkpoints"][-1] if arm["checkpoints"] else {}

    rf, uf, bf = _final(arm_rel), _final(arm_unrel), _final(arm_b)
    summary = {
        "C_related_logdet_gain": rf.get("logdet_gain", 0),
        "C_unrelated_logdet_gain": uf.get("logdet_gain", 0),
        "C_related_topk_mass_XY": rf.get("F_XY_topk_mass", 0),
        "C_unrelated_topk_mass_XY": uf.get("F_XY_topk_mass", 0),
        "C_related_subspace_overlap": rf.get("subspace_overlap", 0),
        "C_unrelated_subspace_overlap": uf.get("subspace_overlap", 0),
        "C_related_residual_energy": rf.get("residual_energy", 0),
        "C_unrelated_residual_energy": uf.get("residual_energy", 0),
        "C_related_mean_sample_cos": rf.get("mean_sample_cos", 0),
        "C_unrelated_mean_sample_cos": uf.get("mean_sample_cos", 0),
        "C_related_acc": rf.get("acc_xy", 0),
        "C_unrelated_acc": uf.get("acc_xy", 0),
        "B_acc": bf.get("acc_xy", 0),
        # Label leakage check (Concern 1)
        "C_related_acc_x_only": rf.get("acc_x_only", 0),
        "C_related_acc_y_only": rf.get("acc_y_only", 0),
        "fisher_samples_used": rf.get("fisher_samples_used", 0),
        # Metric type for clarity (Concern 8)
        "metric_type": "micro-F1" if data_related["task"] == "multilabel"
                       else "accuracy",
        "num_perm_seeds": len(perm_seeds),
    }
    return {
        "dataset": dataset_name, "seed": seed, "model": args.model,
        "text_backbone": args.text_backbone,
        "vision_backbone": args.vision_backbone,
        "top_k": args.top_k,
        "arms": {"B": arm_b, "C_related": arm_rel, "C_unrelated": arm_unrel},
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fisher Geometry: compare Concat-MLP vs Cross-Attention")
    parser.add_argument("--model", choices=["concat_mlp", "shared_mlp", "cross_attention"],
                        default="cross_attention")
    parser.add_argument("--datasets", nargs="+",
                        default=["food101", "oxford_pets", "imdb",
                                 "humor", "sarcasm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default="cuda:0")
    parser.add_argument("--text-backbone", default="clip_cached")
    parser.add_argument("--vision-backbone", default="clip_cached")
    parser.add_argument("--backbone-batch-size", type=int, default=32)
    parser.add_argument("--download-datasets", action="store_true")
    parser.add_argument("--backbone-cache-dir", default=None)
    # Shared
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fisher-samples", type=int, default=200)
    parser.add_argument("--checkpoint-count", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    # Concat-MLP specific
    parser.add_argument("--hidden-dim", type=int, default=256)
    # Cross-Attention specific
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            f"src/fisher_geometry/results/fisher_geometry_{args.model}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds in args.datasets:
        for seed in args.seeds:
            result = run_single(ds, seed, args)
            all_results.append(result)
            fname = out_dir / f"{ds}_seed{seed}.json"
            fname.write_text(json.dumps(result, indent=2, default=str))
            print(f"  Saved: {fname}")

    combined = out_dir / "all_results.json"
    combined.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nAll results saved to {combined}")

    # Summary table
    print(f"\n{'='*130}")
    print(f"FISHER GEOMETRY — {args.model} (final checkpoint, "
          f"averaged over seeds)")
    print(f"{'='*130}")
    hdr = (f"{'Dataset':<15} {'acc_XY':>8} {'acc_U':>8} {'acc_B':>8} "
           f"{'Xonly':>8} {'Yonly':>8} | "
           f"{'cos_R':>8} {'cos_U':>8} | "
           f"{'overlap_R':>10} {'overlap_U':>10} | "
           f"{'resid_R':>8} {'resid_U':>8} | "
           f"{'topk_R':>8} {'topk_U':>8}")
    print(hdr)
    print("-" * 145)

    for ds in args.datasets:
        rs = [r for r in all_results if r["dataset"] == ds]
        if not rs:
            continue
        s = {}
        for k in rs[0]["summary"]:
            vals = [r["summary"][k] for r in rs]
            if isinstance(vals[0], (int, float)):
                s[k] = float(np.mean(vals))
            else:
                s[k] = vals[0]
        metric_label = s.get("metric_type", "acc")
        print(
            f"{ds:<15} "
            f"{s['C_related_acc']:8.4f} {s['C_unrelated_acc']:8.4f} "
            f"{s['B_acc']:8.4f} "
            f"{s.get('C_related_acc_x_only',0):8.4f} "
            f"{s.get('C_related_acc_y_only',0):8.4f} | "
            f"{s['C_related_mean_sample_cos']:8.4f} "
            f"{s['C_unrelated_mean_sample_cos']:8.4f} | "
            f"{s['C_related_subspace_overlap']:10.4f} "
            f"{s['C_unrelated_subspace_overlap']:10.4f} | "
            f"{s['C_related_residual_energy']:8.4f} "
            f"{s['C_unrelated_residual_energy']:8.4f} | "
            f"{s['C_related_topk_mass_XY']:8.4f} "
            f"{s['C_unrelated_topk_mass_XY']:8.4f} "
            f" [{metric_label}]")


if __name__ == "__main__":
    main()
