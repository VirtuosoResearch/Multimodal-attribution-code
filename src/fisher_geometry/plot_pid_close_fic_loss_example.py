"""Plot PID-close source-pair examples comparing FIC gap and loss gap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULT_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "fisher_geometry"
    / "verification_results"
    / "real_motivating_example"
    / "pid_matched_group_finetune_base1"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create example plots for PID-close source pairs: FIC gap vs fine-tuned test loss gap."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", default="pid_close_fic_loss_example")
    args = parser.parse_args()

    rows = load_rows(args.result_dir, seed=int(args.seed))
    if not rows:
        raise FileNotFoundError(f"no source-pair loss rows found in {args.result_dir}")

    output_png = args.result_dir / f"{args.output_prefix}.png"
    output_pdf = args.result_dir / f"{args.output_prefix}.pdf"
    make_plot(rows, output_png, output_pdf)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


def load_rows(result_dir: Path, *, seed: int) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_dir.glob(f"*_seed{seed}_pid_matched_source_pair_loss.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["config"]["dataset"]
        summary = payload["summary"]
        for record in payload["records"]:
            if record.get("test_loss_gap_norm") is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "source_pair": f"{record['source_group_a']}-{record['source_group_b']}",
                    "pid_close_percentile": float(record["PID_close_percentile"]),
                    "pid_distance": float(record["source_pair_pid_distance"]),
                    "fic_gap_norm": float(record["FIC_gap_norm"]),
                    "test_loss_gap_norm": float(record["test_loss_gap_norm"]),
                    "fic_gap": float(record["FIC_gap"]),
                    "test_loss_gap": float(record["test_loss_gap"]),
                    "summary": summary,
                }
            )
    return rows


def make_plot(rows: list[dict[str, Any]], output_png: Path, output_pdf: Path) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    colors = dict(zip(datasets, plt.cm.tab10(np.linspace(0.0, 1.0, len(datasets)))))

    fig = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[0, 1])

    for dataset in datasets:
        ds_rows = [row for row in rows if row["dataset"] == dataset]
        x = np.asarray([row["fic_gap_norm"] for row in ds_rows], dtype=float)
        y = np.asarray([row["test_loss_gap_norm"] for row in ds_rows], dtype=float)
        ax_scatter.scatter(
            x,
            y,
            s=42,
            alpha=0.78,
            color=colors[dataset],
            edgecolor="white",
            linewidth=0.6,
            label=dataset,
        )

    ax_scatter.set_title("PID-close source pairs")
    ax_scatter.set_xlabel("FIC gap (normalized within selected PID-close pairs)")
    ax_scatter.set_ylabel("Fine-tuned test loss gap (normalized)")
    ax_scatter.set_xlim(-0.04, 1.04)
    ax_scatter.set_ylim(-0.04, 1.04)
    ax_scatter.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax_scatter.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")

    y_positions = np.arange(len(datasets))
    for y_pos, dataset in zip(y_positions, datasets):
        ds_rows = sorted([row for row in rows if row["dataset"] == dataset], key=lambda row: row["fic_gap_norm"])
        third = max(1, len(ds_rows) // 3)
        low = ds_rows[:third]
        high = ds_rows[-third:]
        low_x = float(np.mean([row["fic_gap_norm"] for row in low]))
        high_x = float(np.mean([row["fic_gap_norm"] for row in high]))
        low_loss = float(np.mean([row["test_loss_gap_norm"] for row in low]))
        high_loss = float(np.mean([row["test_loss_gap_norm"] for row in high]))
        separation = abs(high_loss - low_loss)
        color = colors[dataset]
        ax_summary.plot([low_loss, high_loss], [y_pos, y_pos], color=color, linewidth=3, alpha=0.85)
        ax_summary.scatter([low_loss], [y_pos], s=70, color="white", edgecolor=color, linewidth=2.0, zorder=3)
        ax_summary.scatter([high_loss], [y_pos], s=70, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax_summary.text(
            min(1.02, max(low_loss, high_loss) + 0.035),
            y_pos,
            f"|Delta|={separation:.2f}",
            va="center",
            fontsize=9,
            color="#333333",
        )
        ax_summary.text(
            -0.02,
            y_pos + 0.26,
            f"FIC: {low_x:.2f}->{high_x:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#555555",
        )

    ax_summary.set_title("Loss-gap separation from low- to high-FIC bins")
    ax_summary.set_xlabel("Mean test loss gap (normalized)")
    ax_summary.set_yticks(y_positions)
    ax_summary.set_yticklabels(datasets)
    ax_summary.set_xlim(-0.04, 1.12)
    ax_summary.set_ylim(-0.7, len(datasets) - 0.3)
    ax_summary.grid(True, axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax_summary.scatter([], [], s=70, color="white", edgecolor="#333333", linewidth=2.0, label="low-FIC bin")
    ax_summary.scatter([], [], s=70, color="#333333", edgecolor="white", linewidth=0.8, label="high-FIC bin")
    ax_summary.legend(frameon=False, fontsize=9, loc="lower right")

    max_close = max(row["pid_close_percentile"] for row in rows)
    fig.suptitle(
        "Example: PID-close groups can have different FIC and different fine-tuning behavior",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        f"All points are selected from PID-close source pairs; maximum PID-close percentile = {max_close:.3f}. "
        "PID distance uses signed raw LSMI components after dataset-level z-scoring.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
