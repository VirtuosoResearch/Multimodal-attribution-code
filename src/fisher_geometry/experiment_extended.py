"""Extended Fisher Geometry experiments.

Runs four new analysis directions on trained models:
  1. Interaction Fisher (F_XY - F_X - F_Y)
  2. Spectral shape metrics (effective rank, participation ratio, spectral decay)
  3. Per-layer / shared-parameter Fisher decomposition
  4. Gradient alignment dynamics during training

Usage:
    conda activate vlm
    python -m src.fisher_geometry.experiment_extended \
        --model cross_attention \
        --datasets imdb humor sarcasm mosi mosei avmnist oxford_pets \
        --seeds 1 2 3 --device cuda:0
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fisher_geometry.models import (
    ConcatFusionModel,
    CrossAttentionFusionModel,
    SharedMLPFusionModel,
)
from src.fisher_geometry.fisher_metrics import (
    compute_loss,
    per_example_gradients,
    per_example_gradients_grouped,
    fisher_eigen,
    spectral_metrics,
    interaction_fisher,
    subspace_overlap,
    residual_energy,
    mean_sample_cosine,
)
from src.fisher_geometry.data import load_dataset, permute_aux, perturb_aux, make_loader
from src.fisher_geometry.experiment import (
    set_seed,
    evaluate,
    build_model,
)


# ---------------------------------------------------------------------------
# Parameter group definitions per model type
# ---------------------------------------------------------------------------

def _get_param_groups(model_type: str) -> dict[str, list[str]]:
    if model_type == "cross_attention":
        return {
            "proj_x": ["proj_x"],
            "proj_y": ["proj_y"],
            "attn_x": ["cross_attn_x", "norm_x", "ffn_x"],
            "attn_y": ["cross_attn_y", "norm_y", "ffn_y"],
            "shared": ["cls_x", "cls_y", "classifier", "pool_norm", "null_embed"],
        }
    elif model_type == "concat_mlp":
        return {
            "fusion": ["fusion_in", "fusion_hidden"],
            "classifier": ["classifier"],
        }
    elif model_type == "shared_mlp":
        return {
            "proj_x": ["proj_x"],
            "proj_y": ["proj_y"],
            "shared_mlp": ["shared_mlp"],
            "classifier": ["classifier"],
        }
    return {}


# ---------------------------------------------------------------------------
# Train and collect gradient alignment at every step
# ---------------------------------------------------------------------------

def _train_with_alignment_tracking(
    data: dict[str, Any],
    *,
    model_type: str,
    device: str,
    args: argparse.Namespace,
    seed: int,
    alignment_samples: int = 200,
) -> dict[str, Any]:
    """Train with both modalities and track cos(grad_x, grad_y) at each step."""
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

    alignment_history = []
    step = 0
    sample_every = max(1, len(loader) // 5)  # ~5 measurements per epoch

    for epoch in range(args.epochs):
        model.train()
        for bx, by, bl in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx.to(device), by.to(device),
                           use_x=True, use_y=True)
            loss = compute_loss(task, logits, bl.to(device))
            loss.backward()
            optimizer.step()

            if step % sample_every == 0:
                # Quick alignment measurement on a subset
                n_al = min(alignment_samples, data["train_x"].shape[0])
                model.eval()
                gbs = 64 if model_type == "cross_attention" else 512
                gx = per_example_gradients(
                    model, data["train_x"][:n_al], data["train_y"][:n_al],
                    data["train_labels"][:n_al],
                    device=device, max_samples=n_al, task=task,
                    use_x=True, use_y=False, grad_batch_size=gbs)
                gy = per_example_gradients(
                    model, data["train_x"][:n_al], data["train_y"][:n_al],
                    data["train_labels"][:n_al],
                    device=device, max_samples=n_al, task=task,
                    use_x=False, use_y=True, grad_batch_size=gbs)
                cos_val = mean_sample_cosine(gx, gy)
                # Also track gradient norms
                gx_norm = float(gx.norm(dim=1).mean().item())
                gy_norm = float(gy.norm(dim=1).mean().item())
                alignment_history.append({
                    "step": step, "epoch": epoch,
                    "cosine": cos_val,
                    "gx_norm": gx_norm, "gy_norm": gy_norm,
                    "loss": float(loss.item()),
                })
                model.train()
            step += 1

    # Final eval
    model.eval()
    acc = evaluate(model, data["val_x"], data["val_y"], data["val_labels"],
                   device=device, batch_size=512, task=task,
                   use_x=True, use_y=True)
    return {
        "alignment_history": alignment_history,
        "final_acc": acc,
        "final_state_dict": {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()},
    }


# ---------------------------------------------------------------------------
# Full extended analysis on a trained model
# ---------------------------------------------------------------------------

def _analyze_model(
    model: nn.Module,
    data: dict[str, Any],
    *,
    model_type: str,
    device: str,
    fisher_samples: int,
    top_k: int,
    task: str,
    use_y_train: bool,
) -> dict[str, Any]:
    """Run all extended analyses on a single model checkpoint."""
    actual_fisher = min(fisher_samples, data["val_x"].shape[0])
    gbs = 64 if model_type == "cross_attention" else 512

    # Standard gradients
    gx = per_example_gradients(
        model, data["val_x"], data["val_y"], data["val_labels"],
        device=device, max_samples=actual_fisher, task=task,
        use_x=True, use_y=False, grad_batch_size=gbs)
    gy = per_example_gradients(
        model, data["val_x"], data["val_y"], data["val_labels"],
        device=device, max_samples=actual_fisher, task=task,
        use_x=False, use_y=use_y_train, grad_batch_size=gbs)
    gxy = per_example_gradients(
        model, data["val_x"], data["val_y"], data["val_labels"],
        device=device, max_samples=actual_fisher, task=task,
        use_x=True, use_y=use_y_train, grad_batch_size=gbs)

    # 1. Standard Fisher eigen + new spectral metrics
    fx = fisher_eigen(gx, top_k=top_k)
    fy = fisher_eigen(gy, top_k=top_k)
    fxy = fisher_eigen(gxy, top_k=top_k)

    spec_x = spectral_metrics(fx["eigenvalues"])
    spec_y = spectral_metrics(fy["eigenvalues"])
    spec_xy = spectral_metrics(fxy["eigenvalues"])

    # 2. Interaction Fisher
    interaction = interaction_fisher(gx, gy, gxy, top_k=top_k)

    # 3. Per-layer Fisher
    param_groups = _get_param_groups(model_type)
    group_metrics = {}
    if param_groups:
        for modality, (ux, uy) in [("x_only", (True, False)),
                                    ("y_only", (False, use_y_train)),
                                    ("xy", (True, use_y_train))]:
            grads_grouped = per_example_gradients_grouped(
                model, data["val_x"], data["val_y"], data["val_labels"],
                device=device, max_samples=actual_fisher, task=task,
                use_x=ux, use_y=uy, param_groups=param_groups,
                grad_batch_size=gbs)
            for gname, gmat in grads_grouped.items():
                fe = fisher_eigen(gmat, top_k=min(top_k, 5))
                sm = spectral_metrics(fe["eigenvalues"])
                group_metrics[f"{modality}/{gname}"] = {
                    "trace": fe["trace"],
                    "logdet": fe["logdet"],
                    "topk_mass": fe["topk_mass"],
                    "rank": fe["rank"],
                    **sm,
                }

    # 4. Standard metrics for comparison
    result = {
        # Standard
        "F_X_trace": fx["trace"],
        "F_Y_trace": fy["trace"],
        "F_XY_trace": fxy["trace"],
        "F_X_topk_mass": fx["topk_mass"],
        "F_Y_topk_mass": fy["topk_mass"],
        "F_XY_topk_mass": fxy["topk_mass"],
        "subspace_overlap": subspace_overlap(fx["top_k_vecs"], fy["top_k_vecs"]),
        "residual_energy": residual_energy(gy, fx["top_k_vecs"]),
        "mean_sample_cos": mean_sample_cosine(gx, gy),
        # Spectral shape
        "spec_X": spec_x,
        "spec_Y": spec_y,
        "spec_XY": spec_xy,
        # Interaction Fisher
        "interaction_trace": interaction["trace"],
        "interaction_pos_trace": interaction["pos_trace"],
        "interaction_neg_trace": interaction["neg_trace"],
        "interaction_pos_rank": interaction["pos_rank"],
        "interaction_neg_rank": interaction["neg_rank"],
        "interaction_top_evals": interaction["top_eigenvalues"],
        # Per-layer
        "group_metrics": group_metrics,
    }
    return result


# ---------------------------------------------------------------------------
# Single-dataset driver
# ---------------------------------------------------------------------------

def run_single(dataset_name: str, seed: int,
               args: argparse.Namespace,
               out_dir: Path) -> dict[str, Any]:
    save_path = out_dir / f"{dataset_name}_seed{seed}.json"
    results_by_alpha = {}
    done_alphas = set()
    if save_path.exists():
        prev = json.loads(save_path.read_text())
        results_by_alpha = prev.get("results", {})
        done_alphas = set(results_by_alpha.keys())
        if len(done_alphas) >= len(args.noise_levels):
            print(f"  [{dataset_name}] seed={seed}: already done, skipping")
            return prev
        print(f"  [{dataset_name}] seed={seed}: resuming, "
              f"{len(done_alphas)} noise levels done")

    print(f"\n  [{dataset_name}] seed={seed}: loading data...")
    data_related = load_dataset(
        dataset_name, seed,
        text_backbone=args.text_backbone,
        vision_backbone=args.vision_backbone,
        feature_device=args.feature_device,
        backbone_batch_size=args.backbone_batch_size,
        download_datasets=args.download_datasets,
        backbone_cache_dir=args.backbone_cache_dir,
    )

    dim_x = data_related["train_x"].shape[1]
    dim_y = data_related["train_y"].shape[1]
    num_classes = data_related["num_classes"]
    task = data_related["task"]

    noise_levels = args.noise_levels

    for alpha in noise_levels:
        if str(alpha) in done_alphas:
            print(f"  [{dataset_name}] seed={seed}: alpha={alpha:.2f} already done, skipping")
            continue
        data_noisy = perturb_aux(data_related, alpha, seed + 1000,
                                 mode="gaussian")
        label = f"alpha={alpha:.1f}"
        print(f"  [{dataset_name}] seed={seed}: training {label} + alignment...")
        train_result = _train_with_alignment_tracking(
            data_noisy, model_type=args.model, device=args.device,
            args=args, seed=seed + 200 + int(alpha * 100),
            alignment_samples=args.fisher_samples)

        print(f"  [{dataset_name}] seed={seed}: analyzing {label}...")
        model_a = build_model(args.model, dim_x, dim_y, num_classes, args).to(args.device)
        model_a.load_state_dict(train_result["final_state_dict"])
        model_a.eval()
        analysis_a = _analyze_model(
            model_a, data_noisy, model_type=args.model,
            device=args.device, fisher_samples=args.fisher_samples,
        top_k=args.top_k, task=task, use_y_train=True)

        results_by_alpha[str(alpha)] = {
            "acc": train_result["final_acc"],
            "alignment_history": train_result["alignment_history"],
            "analysis": analysis_a,
        }

        # Incremental save after each noise level
        partial = {
            "dataset": dataset_name, "seed": seed, "model": args.model,
            "noise_levels": noise_levels, "noise_mode": "replace",
            "results": results_by_alpha,
        }
        save_path.write_text(json.dumps(partial, indent=2, default=str))
        print(f"  [{dataset_name}] seed={seed} {label}: saved")

    result = {
        "dataset": dataset_name,
        "seed": seed,
        "model": args.model,
        "noise_levels": noise_levels,
        "noise_mode": "replace",
        "results": results_by_alpha,
    }

    save_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {save_path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extended Fisher Geometry experiments")
    parser.add_argument("--model", choices=["concat_mlp", "shared_mlp", "cross_attention"],
                        default="cross_attention")
    parser.add_argument("--datasets", nargs="+",
                        default=["imdb", "humor", "sarcasm", "mosi", "mosei",
                                 "avmnist", "oxford_pets"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default="cuda:0")
    parser.add_argument("--text-backbone", default="clip_cached")
    parser.add_argument("--vision-backbone", default="clip_cached")
    parser.add_argument("--backbone-batch-size", type=int, default=32)
    parser.add_argument("--download-datasets", action="store_true")
    parser.add_argument("--backbone-cache-dir", default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fisher-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--noise-levels", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"src/fisher_geometry/results/extended_{args.model}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds in args.datasets:
        for seed in args.seeds:
            result = run_single(ds, seed, args, out_dir)
            all_results.append(result)

    combined = out_dir / "all_results.json"
    combined.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nAll results saved to {combined}")


if __name__ == "__main__":
    main()
