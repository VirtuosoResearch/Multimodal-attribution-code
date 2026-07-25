"""Numerical checks for OSFI-style Lanczos inverse quadrature.

This script constructs Fisher-like matrices of the form

    A = damping * I + G.T @ diag(w) @ G

and compares the OSFI implementation of Lanczos inverse quadrature against a
dense solve for z.T @ A^{-1} @ z.  The construction matches the low-rank
group-as-sample curvature described in
src/kernel_surrogate/osfi_loss_osfi_experiment_summary.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kernel_surrogate.verify_humor_fisher_gpt_metrics import inverse_quadrature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Lanczos inverse quadrature on low-rank Fisher matrices."
    )
    parser.add_argument("--dim", type=int, default=513, help="Parameter dimension d.")
    parser.add_argument("--blocks", type=int, default=50, help="Number of Fisher blocks.")
    parser.add_argument("--damping", type=float, default=1e-3, help="Damping lambda.")
    parser.add_argument("--directions", type=int, default=32, help="Random test directions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help=(
            "Matrix construction dtype. The imported OSFI quadrature casts "
            "the Lanczos start vector to float32, so float32 is the faithful mode."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 51, 64],
        help="Lanczos step counts to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("src/slq/inverse_lanczos_accuracy_results.json"),
    )
    return parser.parse_args()


def make_low_rank_fisher(
    *,
    dim: int,
    blocks: int,
    damping: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    # Mildly decaying row scales make the spectrum less artificial while
    # preserving the exact low-rank-plus-damping OSFI structure.
    rows = torch.randn(blocks, dim, generator=gen, dtype=dtype).to(device)
    row_scales = torch.linspace(1.0, 0.15, blocks, device=device, dtype=dtype).reshape(-1, 1)
    rows = rows * row_scales / math.sqrt(dim)

    raw_weights = torch.rand(blocks, generator=gen, dtype=dtype).to(device) + 0.2
    weights = raw_weights / raw_weights.sum()

    dense = damping * torch.eye(dim, device=device, dtype=dtype)
    dense = dense + rows.T @ (weights.reshape(-1, 1) * rows)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(device=device).reshape(-1)
        row_view = rows.to(dtype=vector.dtype)
        weight_view = weights.to(dtype=vector.dtype)
        return damping * vector + row_view.T @ (weight_view * (row_view @ vector))

    return rows, weights, dense, matvec


def dense_inverse_quadrature(dense: torch.Tensor, vector: torch.Tensor) -> float:
    vector = vector.to(device=dense.device, dtype=dense.dtype).reshape(-1)
    solution = torch.linalg.solve(dense, vector)
    return float(torch.dot(vector, solution).detach().cpu().item())


def summarize_errors(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "max": float(tensor.max().item()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    rows, weights, dense, matvec = make_low_rank_fisher(
        dim=args.dim,
        blocks=args.blocks,
        damping=args.damping,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 17)
    directions = torch.randn(args.directions, args.dim, generator=gen, dtype=dtype).to(device)
    directions = torch.nn.functional.normalize(directions, dim=1)

    # Include directions that matter for OSFI: row-span directions and a random
    # low-rank interaction/loss-like vector.
    row_span_direction = rows.T @ weights
    row_span_direction = row_span_direction / torch.linalg.norm(row_span_direction)
    loss_like_direction = rows.T @ torch.randn(args.blocks, generator=gen, dtype=dtype).to(device)
    loss_like_direction = loss_like_direction / torch.linalg.norm(loss_like_direction)
    test_vectors = torch.cat(
        [directions, row_span_direction.reshape(1, -1), loss_like_direction.reshape(1, -1)],
        dim=0,
    )

    exact_values = [dense_inverse_quadrature(dense, vector) for vector in test_vectors]
    records = []
    for steps in args.steps:
        clipped_steps = max(1, min(int(steps), args.dim))
        rel_errors = []
        abs_errors = []
        for vector, exact in zip(test_vectors, exact_values):
            estimate = inverse_quadrature(
                vector,
                matvec,
                num_lanczos=clipped_steps,
                device=device,
            )
            abs_error = abs(estimate - exact)
            rel_error = abs_error / max(abs(exact), 1e-30)
            abs_errors.append(abs_error)
            rel_errors.append(rel_error)
        records.append(
            {
                "steps": clipped_steps,
                "relative_error": summarize_errors(rel_errors),
                "absolute_error": summarize_errors(abs_errors),
            }
        )

    exact_trace = float(torch.trace(torch.linalg.inv(dense)).detach().cpu().item())
    trace_records = []
    for steps in args.steps:
        clipped_steps = max(1, min(int(steps), args.dim))
        estimates = [
            inverse_quadrature(vector, matvec, num_lanczos=clipped_steps, device=device)
            for vector in test_vectors
        ]
        # For unit random probes, d * E[z.T A^{-1} z] estimates trace(A^{-1}).
        random_probe_trace = args.dim * sum(estimates[: args.directions]) / args.directions
        trace_records.append(
            {
                "steps": clipped_steps,
                "random_probe_trace_estimate": float(random_probe_trace),
                "relative_error": float(abs(random_probe_trace - exact_trace) / abs(exact_trace)),
            }
        )

    eigenvalues = torch.linalg.eigvalsh(dense)
    result = {
        "config": {
            "dim": args.dim,
            "blocks": args.blocks,
            "damping": args.damping,
            "directions": args.directions,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": str(device),
        },
        "spectrum": {
            "lambda_min": float(eigenvalues.min().detach().cpu().item()),
            "lambda_max": float(eigenvalues.max().detach().cpu().item()),
            "condition_number": float((eigenvalues.max() / eigenvalues.min()).detach().cpu().item()),
        },
        "quadrature_records": records,
        "trace_exact": exact_trace,
        "trace_records": trace_records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("OSFI-style inverse quadrature accuracy")
    print(
        f"dim={args.dim}, blocks={args.blocks}, damping={args.damping:g}, "
        f"condition={result['spectrum']['condition_number']:.4g}"
    )
    print("")
    print("steps | mean rel err | median rel err | max rel err")
    print("----- | ------------ | -------------- | -----------")
    for row in records:
        rel = row["relative_error"]
        print(
            f"{row['steps']:>5d} | {rel['mean']:.3e} | "
            f"{rel['median']:.3e} | {rel['max']:.3e}"
        )
    print("")
    print(f"Exact trace(A^-1): {exact_trace:.8e}")
    print(f"Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
