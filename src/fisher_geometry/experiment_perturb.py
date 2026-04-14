"""Fisher Geometry perturbation sweep experiment.

Instead of binary related/unrelated, trains models at multiple noise levels
alpha in {0, 0.25, 0.5, 0.75, 1.0} where:

    Y_perturbed = (1 - alpha) * Y_original + alpha * Y_permuted

If top-k spectrum mass (or any Fisher metric) changes monotonically with
alpha, that is strong evidence for a genuine geometric signature.

Usage:
    conda activate vlm
    python -m src.fisher_geometry.experiment_perturb \
        --model cross_attention \
        --datasets food101 oxford_pets imdb humor sarcasm \
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
    fisher_eigen,
    subspace_overlap,
    residual_energy,
    mean_sample_cosine,
)
from src.fisher_geometry.data import load_dataset, perturb_aux, make_loader
from src.fisher_geometry.experiment import (
    set_seed,
    ArmCheckpoint,
    _checkpoint_fractions,
    _capture_checkpoints,
    evaluate,
    build_model,
)


def _train_and_measure_final(
    data: dict[str, Any],
    *,
    model_type: str,
    device: str,
    args: argparse.Namespace,
    fisher_samples: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    """Train with both modalities and measure Fisher geometry at final checkpoint only."""
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

    for _ in range(args.epochs):
        model.train()
        for bx, by, bl in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx.to(device), by.to(device),
                           use_x=True, use_y=True)
            loss = compute_loss(task, logits, bl.to(device))
            loss.backward()
            optimizer.step()

    # --- Measure at final checkpoint ---
    model.eval()
    acc_xy = evaluate(model, data["val_x"], data["val_y"], data["val_labels"],
                      device=device, batch_size=512, task=task,
                      use_x=True, use_y=True)
    acc_x = evaluate(model, data["val_x"], data["val_y"], data["val_labels"],
                     device=device, batch_size=512, task=task,
                     use_x=True, use_y=False)
    acc_y = evaluate(model, data["val_x"], data["val_y"], data["val_labels"],
                     device=device, batch_size=512, task=task,
                     use_x=False, use_y=True)

    actual_fisher = min(fisher_samples, data["val_x"].shape[0])

    gx = per_example_gradients(
        model, data["val_x"], data["val_y"], data["val_labels"],
        device=device, max_samples=actual_fisher, task=task,
        use_x=True, use_y=False)
    gy = per_example_gradients(
        model, data["val_x"], data["val_y"], data["val_labels"],
        device=device, max_samples=actual_fisher, task=task,
        use_x=False, use_y=True)
    gxy = per_example_gradients(
        model, data["val_x"], data["val_y"], data["val_labels"],
        device=device, max_samples=actual_fisher, task=task,
        use_x=True, use_y=True)

    fx = fisher_eigen(gx, top_k=top_k)
    fy = fisher_eigen(gy, top_k=top_k)
    fxy = fisher_eigen(gxy, top_k=top_k)

    return {
        "acc_xy": acc_xy,
        "acc_x_only": acc_x,
        "acc_y_only": acc_y,
        "fisher_samples_used": actual_fisher,
        "F_X_trace": fx["trace"],
        "F_Y_trace": fy["trace"],
        "F_XY_trace": fxy["trace"],
        "F_X_topk_mass": fx["topk_mass"],
        "F_Y_topk_mass": fy["topk_mass"],
        "F_XY_topk_mass": fxy["topk_mass"],
        "F_X_logdet": fx["logdet"],
        "F_XY_logdet": fxy["logdet"],
        "logdet_gain": fxy["logdet"] - fx["logdet"],
        "subspace_overlap": subspace_overlap(
            fx["top_k_vecs"], fy["top_k_vecs"]),
        "residual_energy": residual_energy(gy, fx["top_k_vecs"]),
        "mean_sample_cos": mean_sample_cosine(gx, gy),
        "F_X_rank": fx["rank"],
        "F_Y_rank": fy["rank"],
        "F_XY_rank": fxy["rank"],
    }


def _make_result_dict(dataset_name, seed, args, noise_levels, results_by_noise):
    return {
        "dataset": dataset_name,
        "seed": seed,
        "model": args.model,
        "text_backbone": args.text_backbone,
        "vision_backbone": args.vision_backbone,
        "noise_levels": noise_levels,
        "noise_mode": args.noise_mode,
        "num_perm": args.num_perm,
        "results": results_by_noise,
    }


def run_single(dataset_name: str, seed: int,
               args: argparse.Namespace,
               out_dir: Path | None = None) -> dict[str, Any]:
    # Resume from partial results if available
    save_path = out_dir / f"{dataset_name}_seed{seed}.json" if out_dir else None
    results_by_noise: dict[str, Any] = {}
    if save_path and save_path.exists():
        prev = json.loads(save_path.read_text())
        results_by_noise = prev.get("results", {})
        done_alphas = set(results_by_noise.keys())
        print(f"\n  [{dataset_name}] seed={seed}: resuming, "
              f"{len(done_alphas)} noise levels already done")
    else:
        done_alphas = set()

    print(f"\n  [{dataset_name}] seed={seed}: loading data...")
    data_clean = load_dataset(
        dataset_name,
        seed,
        text_backbone=args.text_backbone,
        vision_backbone=args.vision_backbone,
        feature_device=args.feature_device,
        backbone_batch_size=args.backbone_batch_size,
        download_datasets=args.download_datasets,
        backbone_cache_dir=args.backbone_cache_dir,
    )

    noise_levels = args.noise_levels

    for alpha in noise_levels:
        key = str(alpha)
        if key in done_alphas:
            print(f"  [{dataset_name}] seed={seed} alpha={alpha:.2f}: "
                  f"already done, skipping")
            continue

        # Average over multiple permutation seeds for stability
        perm_seeds = [seed + 1000 + i * 1000 for i in range(args.num_perm)]
        perm_results = []
        for pi, ps in enumerate(perm_seeds):
            data_noisy = perturb_aux(data_clean, alpha, ps,
                                        mode=args.noise_mode)
            print(f"  [{dataset_name}] seed={seed} alpha={alpha:.2f} "
                  f"perm {pi+1}/{len(perm_seeds)}...")
            r = _train_and_measure_final(
                data_noisy, model_type=args.model, device=args.device,
                args=args, fisher_samples=args.fisher_samples,
                seed=seed + 200 + int(alpha * 100) + pi * 10,
                top_k=args.top_k)
            perm_results.append(r)

        # Average across permutation seeds
        avg = {}
        for k in perm_results[0]:
            vals = [r[k] for r in perm_results]
            if isinstance(vals[0], (int, float)):
                avg[k] = float(np.nanmean(vals))
                avg[k + "_std"] = float(np.nanstd(vals))
            else:
                avg[k] = vals[0]
        results_by_noise[key] = avg

        # Save after each noise level
        if save_path:
            result = _make_result_dict(dataset_name, seed, args,
                                       noise_levels, results_by_noise)
            save_path.write_text(json.dumps(result, indent=2, default=str))
            print(f"  [{dataset_name}] seed={seed} alpha={alpha:.2f}: "
                  f"saved checkpoint to {save_path}")

    return _make_result_dict(dataset_name, seed, args,
                             noise_levels, results_by_noise)


def main():
    parser = argparse.ArgumentParser(
        description="Fisher Geometry perturbation sweep")
    parser.add_argument("--model", choices=["concat_mlp", "shared_mlp", "cross_attention"],
                        default="cross_attention")
    parser.add_argument("--datasets", nargs="+",
                        default=["food101", "oxford_pets", "imdb",
                                 "humor", "sarcasm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--noise-levels", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--noise-mode", choices=["gaussian", "permute", "additive"],
                        default="gaussian",
                        help="Noise type: gaussian (interpolate signal/noise, "
                             "alpha=1 is pure Gaussian noise), permute "
                             "(interpolate with shuffled), additive (Y + alpha*noise)")
    parser.add_argument("--num-perm", type=int, default=3,
                        help="Number of noise seeds per noise level")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default="cuda:0")
    parser.add_argument("--text-backbone", default="clip_cached")
    parser.add_argument("--vision-backbone", default="clip_cached")
    parser.add_argument("--backbone-batch-size", type=int, default=32)
    parser.add_argument("--download-datasets", action="store_true")
    parser.add_argument("--backbone-cache-dir", default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fisher-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            f"src/fisher_geometry/results/perturb_{args.model}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds in args.datasets:
        for seed in args.seeds:
            result = run_single(ds, seed, args, out_dir=out_dir)
            all_results.append(result)
            fname = out_dir / f"{ds}_seed{seed}.json"
            fname.write_text(json.dumps(result, indent=2, default=str))
            print(f"  Saved: {fname}")

    combined = out_dir / "all_results.json"
    combined.write_text(json.dumps(all_results, indent=2, default=str))

    # Summary table
    print(f"\n{'='*120}")
    print(f"PERTURBATION SWEEP — {args.model} (averaged over seeds)")
    print(f"{'='*120}")

    for ds in args.datasets:
        rs = [r for r in all_results if r["dataset"] == ds]
        if not rs:
            continue
        print(f"\n  {ds}:")
        print(f"  {'alpha':>8} {'acc_xy':>10} {'topk_XY':>10} "
              f"{'overlap':>10} {'resid':>10} {'cos':>10} "
              f"{'logdet_g':>12}")
        print(f"  {'-'*72}")
        for alpha in args.noise_levels:
            key = str(alpha)
            vals = {}
            for k in ["acc_xy", "F_XY_topk_mass", "subspace_overlap",
                       "residual_energy", "mean_sample_cos", "logdet_gain"]:
                v = [r["results"][key][k] for r in rs if key in r["results"]]
                vals[k] = np.nanmean(v) if v else float("nan")
            print(f"  {alpha:8.2f} {vals['acc_xy']:10.4f} "
                  f"{vals['F_XY_topk_mass']:10.4f} "
                  f"{vals['subspace_overlap']:10.4f} "
                  f"{vals['residual_energy']:10.4f} "
                  f"{vals['mean_sample_cos']:10.4f} "
                  f"{vals['logdet_gain']:12.2f}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
