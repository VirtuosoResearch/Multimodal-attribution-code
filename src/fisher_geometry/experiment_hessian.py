"""Hessian spectral analysis for multimodal fusion models via SLQ.

Computes Hessian trace, log-determinant, and spectral density for three
modality configurations (X-only, Y-only, XY) using Stochastic Lanczos
Quadrature. Also computes the interaction Hessian H_XY - H_X - H_Y.

Usage:
    conda activate vlm
    python -m fisher_geometry.experiment_hessian \
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
)
from fisher_geometry.fisher_metrics import compute_loss
from fisher_geometry.data import load_dataset, permute_aux, make_loader
from fisher_geometry.experiment import (
    set_seed, evaluate, build_model,
)
from src.slq.slq import SLQ


# ---------------------------------------------------------------------------
# Multimodal Hessian-vector product
# ---------------------------------------------------------------------------

def multimodal_hessian_matvec(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
    task: str,
    use_x: bool,
    use_y: bool,
    max_samples: int = 500,
    batch_size: int = 256,
) -> tuple[callable, int]:
    """Build Hessian-vector product operator for multimodal model.

    Returns (matvec_fn, num_params).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    n = min(max_samples, x.shape[0])
    loader = make_loader(x[:n], y[:n], labels[:n], batch_size, False)
    batches = [(bx.to(device), by.to(device), bl.to(device))
               for bx, by, bl in loader]
    total_samples = sum(bx.shape[0] for bx, _, _ in batches)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(v)
        for bx, by, bl in batches:
            bs = bx.shape[0]
            weight = bs / total_samples
            model.zero_grad(set_to_none=True)
            # Disable efficient attention to allow second-order gradients
            with torch.nn.attention.sdpa_kernel(
                    [torch.nn.attention.SDPBackend.MATH]):
                logits = model(bx, by, use_x=use_x, use_y=use_y)
            loss = compute_loss(task, logits, bl)
            grads = torch.autograd.grad(loss, params, create_graph=True,
                                        allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, params)]
            g_flat = torch.cat([g.reshape(-1) for g in grads])
            gv = torch.dot(g_flat, v)
            hvp = torch.autograd.grad(gv, params, retain_graph=False,
                                      allow_unused=True)
            hvp = [h if h is not None else torch.zeros_like(p)
                   for h, p in zip(hvp, params)]
            hvp_flat = torch.cat([h.reshape(-1) for h in hvp])
            result += hvp_flat.detach() * weight
        return result

    return matvec, num_params


# ---------------------------------------------------------------------------
# Hessian analysis for one model
# ---------------------------------------------------------------------------

def _hessian_analysis(
    model: nn.Module,
    data: dict[str, Any],
    *,
    device: str,
    task: str,
    use_x: bool,
    use_y: bool,
    max_samples: int,
    num_lanczos: int,
    num_probes: int,
) -> dict[str, Any]:
    """Compute Hessian spectral properties via SLQ."""
    model.eval()
    matvec, num_params = multimodal_hessian_matvec(
        model, data["val_x"], data["val_y"], data["val_labels"],
        device=device, task=task, use_x=use_x, use_y=use_y,
        max_samples=max_samples)

    slq = SLQ(matvec=matvec, dim=num_params, device=torch.device(device))
    trace = slq.trace(num_lanczos=num_lanczos, num_probes=num_probes,
                      reorthogonalize=True)
    logdet = slq.logdet(num_lanczos=num_lanczos, num_probes=num_probes,
                        reorthogonalize=True)
    nodes, weights = slq.spectral_density(
        num_lanczos=num_lanczos, num_probes=num_probes,
        reorthogonalize=True)

    # Compute spectral stats from density
    pos_mask = nodes > 1e-10
    neg_mask = nodes < -1e-10
    pos_mass = float(weights[pos_mask].sum().item()) if pos_mask.any() else 0.0
    neg_mass = float(weights[neg_mask].sum().item()) if neg_mask.any() else 0.0

    return {
        "trace": trace,
        "logdet": logdet,
        "num_params": num_params,
        "pos_spectral_mass": pos_mass,
        "neg_spectral_mass": neg_mass,
        "spectral_nodes": nodes.cpu().tolist()[:100],  # top 100 for storage
        "spectral_weights": weights.cpu().tolist()[:100],
    }


# ---------------------------------------------------------------------------
# Single dataset driver
# ---------------------------------------------------------------------------

def run_single(dataset_name: str, seed: int,
               args: argparse.Namespace,
               out_dir: Path) -> dict[str, Any]:
    save_path = out_dir / f"{dataset_name}_seed{seed}.json"
    if save_path.exists():
        print(f"  [{dataset_name}] seed={seed}: already done, skipping")
        return json.loads(save_path.read_text())

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
    data_unrelated = permute_aux(data_related, seed + 1000)

    dim_x = data_related["train_x"].shape[1]
    dim_y = data_related["train_y"].shape[1]
    num_classes = data_related["num_classes"]
    task = data_related["task"]

    results = {}
    for arm_name, data in [("C_related", data_related),
                           ("C_unrelated", data_unrelated)]:
        print(f"  [{dataset_name}] seed={seed}: training {arm_name}...")
        set_seed(seed + (200 if arm_name == "C_related" else 300))
        model = build_model(args.model, dim_x, dim_y, num_classes, args).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=1e-4)
        loader = make_loader(data["train_x"], data["train_y"],
                             data["train_labels"], args.batch_size, True)
        for _ in range(args.epochs):
            model.train()
            for bx, by, bl in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(bx.to(args.device), by.to(args.device),
                               use_x=True, use_y=True)
                loss = compute_loss(task, logits, bl.to(args.device))
                loss.backward()
                optimizer.step()

        model.eval()
        acc = evaluate(model, data["val_x"], data["val_y"], data["val_labels"],
                       device=args.device, batch_size=512, task=task,
                       use_x=True, use_y=True)

        print(f"  [{dataset_name}] seed={seed}: Hessian analysis {arm_name}...")
        h_x = _hessian_analysis(
            model, data, device=args.device, task=task,
            use_x=True, use_y=False,
            max_samples=args.hessian_samples,
            num_lanczos=args.num_lanczos, num_probes=args.num_probes)
        h_y = _hessian_analysis(
            model, data, device=args.device, task=task,
            use_x=False, use_y=True,
            max_samples=args.hessian_samples,
            num_lanczos=args.num_lanczos, num_probes=args.num_probes)
        h_xy = _hessian_analysis(
            model, data, device=args.device, task=task,
            use_x=True, use_y=True,
            max_samples=args.hessian_samples,
            num_lanczos=args.num_lanczos, num_probes=args.num_probes)

        results[arm_name] = {
            "acc": acc,
            "H_X": h_x,
            "H_Y": h_y,
            "H_XY": h_xy,
            "interaction_trace": h_xy["trace"] - h_x["trace"] - h_y["trace"],
            "interaction_logdet": h_xy["logdet"] - h_x["logdet"] - h_y["logdet"],
        }

    result = {
        "dataset": dataset_name, "seed": seed, "model": args.model,
        **results,
    }
    save_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {save_path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hessian spectral analysis via SLQ")
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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hessian-samples", type=int, default=300,
                        help="Max samples for Hessian-vector products")
    parser.add_argument("--num-lanczos", type=int, default=50,
                        help="Lanczos iterations for SLQ")
    parser.add_argument("--num-probes", type=int, default=10,
                        help="Number of random probes for SLQ")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"src/fisher_geometry/results/hessian_{args.model}"
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
