"""Full-network true-Hessian SLQ diagnostics for multimodal interaction.

This script loads the existing full-data checkpoints used by the kernel
surrogate experiments and estimates Hessian spectral statistics with stochastic
Lanczos quadrature (SLQ).  It compares the curvature of the full multimodal
objective with the two unimodal objectives:

    H_xy, H_x, H_y, and trace_delta = tr(H_xy) - tr(H_x) - tr(H_y).

The Hessian is the true parameter Hessian over all trainable parameters by
default, not the empirical Fisher/GGN.  Since true Hessians can be indefinite,
the script reports signed spectral summaries instead of log-determinants.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

from fisher_geometry.data import load_dataset
from fisher_geometry.experiment_hessian import multimodal_hessian_matvec
from kernel_surrogate.real_data_cmis import make_model_args
from fisher_geometry.verify_humor_fisher_gpt_metrics import (
    DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT,
    DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT,
    DEFAULT_RESULT_DIR,
    load_model,
    resolve_full_checkpoint,
    resolve_requested_device,
)
from src.slq.slq import SLQ


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "verification_results"
    / "full_hessian_slq"
)


def spectral_stats(nodes: torch.Tensor, weights: torch.Tensor, *, dim: int) -> dict[str, Any]:
    nodes = nodes.detach().float().cpu()
    weights = weights.detach().float().cpu()
    weighted = weights * nodes
    pos = nodes > 1e-10
    neg = nodes < -1e-10
    trace = float(dim * weighted.sum().item())
    pos_trace = float(dim * weighted[pos].sum().item()) if bool(pos.any()) else 0.0
    neg_trace = float(dim * weighted[neg].sum().item()) if bool(neg.any()) else 0.0
    neg_trace_abs = -neg_trace
    abs_trace = pos_trace + neg_trace_abs
    return {
        "dim": int(dim),
        "trace": trace,
        "mean_eigenvalue": float((weights * nodes).sum().item()),
        "positive_spectral_mass": float(weights[pos].sum().item()) if bool(pos.any()) else 0.0,
        "negative_spectral_mass": float(weights[neg].sum().item()) if bool(neg.any()) else 0.0,
        "positive_trace": pos_trace,
        "negative_trace": neg_trace,
        "negative_trace_abs": neg_trace_abs,
        "absolute_trace": abs_trace,
        "negative_to_positive_trace_ratio": float(neg_trace_abs / max(abs(pos_trace), 1e-12)),
        "min_ritz": float(nodes.min().item()) if nodes.numel() else 0.0,
        "max_ritz": float(nodes.max().item()) if nodes.numel() else 0.0,
        "ritz_nodes": [float(v) for v in nodes.tolist()],
        "ritz_weights": [float(v) for v in weights.tolist()],
    }


def analyze_objective(
    model: torch.nn.Module,
    data: dict[str, Any],
    *,
    device: torch.device,
    task: str,
    use_x: bool,
    use_y: bool,
    hessian_samples: int,
    hessian_batch_size: int,
    num_lanczos: int,
    num_probes: int,
) -> dict[str, Any]:
    matvec, dim = multimodal_hessian_matvec(
        model,
        data["val_x"],
        data["val_y"],
        data["val_labels"],
        device=str(device),
        task=task,
        use_x=use_x,
        use_y=use_y,
        max_samples=int(hessian_samples),
        batch_size=int(hessian_batch_size),
    )
    slq = SLQ(matvec=matvec, dim=int(dim), device=device)
    nodes, weights = slq.spectral_density(
        num_lanczos=min(int(num_lanczos), int(dim)),
        num_probes=int(num_probes),
        reorthogonalize=True,
    )
    return spectral_stats(nodes, weights, dim=int(dim))


def run_dataset(dataset: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    save_path = args.output_dir / f"{dataset}_seed{args.seed}_full_hessian_slq.json"
    if save_path.exists() and not args.overwrite:
        print(f"[{dataset}] existing result, skipping: {save_path}", flush=True)
        return json.loads(save_path.read_text(encoding="utf-8"))

    payload_path = args.result_dir / f"{dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
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
    model_args = make_model_args(config, device=str(device))
    checkpoint = resolve_full_checkpoint(
        args.full_checkpoint,
        roots=args.full_checkpoint_roots,
        dataset=dataset,
        seed=int(args.seed),
    )
    model = load_model(
        checkpoint,
        model_type=str(config.get("model", "cross_attention")),
        dim_x=int(data["train_x"].shape[1]),
        dim_y=int(data["train_y"].shape[1]),
        num_classes=int(data["num_classes"]),
        model_args=model_args,
        device=str(device),
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{dataset}] checkpoint={checkpoint}", flush=True)
    print(f"[{dataset}] full-network parameters={num_params:,}", flush=True)

    task = str(data["task"])
    objectives = {
        "H_x": (True, False),
        "H_y": (False, True),
        "H_xy": (True, True),
    }
    hessian: dict[str, Any] = {}
    for name, (use_x, use_y) in objectives.items():
        print(f"[{dataset}] SLQ {name}...", flush=True)
        hessian[name] = analyze_objective(
            model,
            data,
            device=device,
            task=task,
            use_x=use_x,
            use_y=use_y,
            hessian_samples=int(args.hessian_samples),
            hessian_batch_size=int(args.hessian_batch_size),
            num_lanczos=int(args.num_lanczos),
            num_probes=int(args.num_probes),
        )

    hx, hy, hxy = hessian["H_x"], hessian["H_y"], hessian["H_xy"]
    delta = {
        "trace_xy_minus_x_minus_y": hxy["trace"] - hx["trace"] - hy["trace"],
        "positive_trace_xy_minus_x_minus_y": (
            hxy["positive_trace"] - hx["positive_trace"] - hy["positive_trace"]
        ),
        "negative_abs_trace_xy_minus_x_minus_y": (
            hxy["negative_trace_abs"] - hx["negative_trace_abs"] - hy["negative_trace_abs"]
        ),
        "negative_mass_xy_minus_x_minus_y": (
            hxy["negative_spectral_mass"]
            - hx["negative_spectral_mass"]
            - hy["negative_spectral_mass"]
        ),
    }
    result = {
        "dataset": dataset,
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "config": {
            "device": str(device),
            "hessian_samples": int(args.hessian_samples),
            "hessian_batch_size": int(args.hessian_batch_size),
            "num_lanczos": int(args.num_lanczos),
            "num_probes": int(args.num_probes),
            "parameter_scope": "all_trainable_parameters",
            "hessian_type": "true_loss_hessian",
            "slq_reorthogonalize": True,
        },
        "hessian": hessian,
        "interaction_delta": delta,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{dataset}] wrote {save_path}", flush=True)
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"datasets": [], "summary": {}}

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for result in results:
            obj: Any = result
            for key in path:
                obj = obj[key]
            values.append(float(obj))
        return float(np.mean(values))

    summary = {
        "num_datasets": len(results),
        "mean_trace_delta_xy_minus_x_minus_y": mean(("interaction_delta", "trace_xy_minus_x_minus_y")),
        "mean_positive_trace_delta_xy_minus_x_minus_y": mean(
            ("interaction_delta", "positive_trace_xy_minus_x_minus_y")
        ),
        "mean_negative_abs_trace_delta_xy_minus_x_minus_y": mean(
            ("interaction_delta", "negative_abs_trace_xy_minus_x_minus_y")
        ),
        "mean_Hxy_negative_spectral_mass": mean(("hessian", "H_xy", "negative_spectral_mass")),
        "mean_Hx_negative_spectral_mass": mean(("hessian", "H_x", "negative_spectral_mass")),
        "mean_Hy_negative_spectral_mass": mean(("hessian", "H_y", "negative_spectral_mass")),
        "datasets_positive_trace_delta": [
            result["dataset"]
            for result in results
            if float(result["interaction_delta"]["trace_xy_minus_x_minus_y"]) > 0.0
        ],
        "datasets_more_negative_abs_curvature_in_xy": [
            result["dataset"]
            for result in results
            if float(result["interaction_delta"]["negative_abs_trace_xy_minus_x_minus_y"]) > 0.0
        ],
    }
    return {
        "datasets": [result["dataset"] for result in results],
        "summary": summary,
        "results": results,
    }


def to_markdown(bundle: dict[str, Any]) -> str:
    lines = ["# Full-Network Hessian SLQ Diagnostic", ""]
    summary = bundle["summary"]
    lines.extend(
        [
            "## Setup",
            "",
            "- Hessian type: true held-out loss Hessian, not Fisher/GGN.",
            "- Parameter scope: all trainable network parameters.",
            "- SLQ target: spectral density of `H_x`, `H_y`, and `H_xy`.",
            "- Interaction scalar: `tr(H_xy) - tr(H_x) - tr(H_y)`.",
            "- Because the true Hessian can be indefinite, log-determinants are not reported.",
            "",
            "## Formulation",
            "",
            "For a trained model parameter vector \\(\\hat\\theta\\), define held-out objectives",
            "",
            "\\[",
            "\\ell_x(\\theta)=\\frac{1}{n}\\sum_{i=1}^n \\mathcal L(f_\\theta(x_i,\\varnothing),y_i),\\quad",
            "\\ell_y(\\theta)=\\frac{1}{n}\\sum_{i=1}^n \\mathcal L(f_\\theta(\\varnothing,z_i),y_i),",
            "\\]",
            "",
            "\\[",
            "\\ell_{xy}(\\theta)=\\frac{1}{n}\\sum_{i=1}^n \\mathcal L(f_\\theta(x_i,z_i),y_i).",
            "\\]",
            "",
            "The full-network true Hessians are",
            "",
            "\\[",
            "H_x=\\nabla_\\theta^2\\ell_x(\\hat\\theta),\\quad",
            "H_y=\\nabla_\\theta^2\\ell_y(\\hat\\theta),\\quad",
            "H_{xy}=\\nabla_\\theta^2\\ell_{xy}(\\hat\\theta).",
            "\\]",
            "",
            "The reported interaction trace is",
            "",
            "\\[",
            "\\Delta_{\\operatorname{tr}}=\\operatorname{tr}(H_{xy})-\\operatorname{tr}(H_x)-\\operatorname{tr}(H_y).",
            "\\]",
            "",
            "SLQ estimates the spectral measure of each Hessian through Lanczos tridiagonalizations. For random probe \\(q_r\\), Lanczos gives \\(T_r=Q_r^\\top H Q_r\\). If \\((\\lambda_{rk},s_{rk})\\) are eigenpairs of \\(T_r\\), then",
            "",
            "\\[",
            "\\widehat{\\operatorname{tr}}(H)",
            "=p\\cdot\\frac{1}{R}\\sum_{r=1}^R\\sum_k (s_{rk,1})^2\\lambda_{rk},",
            "\\]",
            "",
            "where \\(p\\) is the number of trainable parameters. The same quadrature nodes are used to estimate positive and negative spectral mass.",
            "",
            "## Aggregate",
            "",
            f"- datasets: {', '.join(bundle['datasets'])}",
            f"- mean trace delta: {summary.get('mean_trace_delta_xy_minus_x_minus_y', 0.0):.6g}",
            f"- mean positive-trace delta: {summary.get('mean_positive_trace_delta_xy_minus_x_minus_y', 0.0):.6g}",
            f"- mean negative-absolute-trace delta: {summary.get('mean_negative_abs_trace_delta_xy_minus_x_minus_y', 0.0):.6g}",
            f"- mean Hxy negative spectral mass: {summary.get('mean_Hxy_negative_spectral_mass', 0.0):.6g}",
            f"- datasets with positive trace delta: {', '.join(summary.get('datasets_positive_trace_delta', [])) or 'none'}",
            f"- datasets with extra negative absolute curvature in Hxy: {', '.join(summary.get('datasets_more_negative_abs_curvature_in_xy', [])) or 'none'}",
            "",
            "## Per-Dataset Results",
            "",
            "| Dataset | tr(Hx) | tr(Hy) | tr(Hxy) | tr Delta | Hxy NegMass | NegAbs Delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in bundle["results"]:
        hx = result["hessian"]["H_x"]
        hy = result["hessian"]["H_y"]
        hxy = result["hessian"]["H_xy"]
        delta = result["interaction_delta"]
        lines.append(
            "| {ds} | {hx:.6g} | {hy:.6g} | {hxy:.6g} | {dt:.6g} | {nm:.6g} | {nd:.6g} |".format(
                ds=result["dataset"],
                hx=hx["trace"],
                hy=hy["trace"],
                hxy=hxy["trace"],
                dt=delta["trace_xy_minus_x_minus_y"],
                nm=hxy["negative_spectral_mass"],
                nd=delta["negative_abs_trace_xy_minus_x_minus_y"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A nonzero trace delta means the full multimodal loss surface is not explained by adding the two unimodal Hessian traces.",
            "- Positive negative-absolute-trace delta means the multimodal objective has more negative curvature magnitude than the two unimodal objectives combined under this SLQ approximation.",
            "- This is a full-network curvature diagnostic. It supports cross-modal interaction in optimization geometry, but it is not itself the structured kernel surrogate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["avmnist", "humor", "imdb", "mosei", "mosi", "sarcasm"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--full-checkpoint-roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT, DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--hessian-samples", type=int, default=64)
    parser.add_argument("--hessian-batch-size", type=int, default=64)
    parser.add_argument("--num-lanczos", type=int, default=20)
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = resolve_requested_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [run_dataset(dataset, args, device) for dataset in args.datasets]
    bundle = aggregate(results)
    json_path = args.output_dir / "full_hessian_slq_all_datasets.json"
    md_path = args.output_dir / "full_hessian_slq_all_datasets.md"
    json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(bundle), encoding="utf-8")
    print(to_markdown(bundle), flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
