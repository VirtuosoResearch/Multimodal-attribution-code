from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize layer-1 shared Fisher gain results on Food101.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    return parser


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt(pair: tuple[float, float], precision: int = 4) -> str:
    if math.isnan(pair[0]):
        return "nan"
    return f"{pair[0]:.{precision}f} ± {pair[1]:.{precision}f}"


def _load_runs(input_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(input_dir.glob("food101_seed*.json")):
        runs.append(json.loads(path.read_text()))
    return sorted(runs, key=lambda item: item["seed"])


def _per_fraction_summary(runs: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    fractions = sorted({float(item["fraction"]) for run in runs for item in run["metrics"]["C_related"]})
    summary: dict[float, dict[str, Any]] = {}
    for frac in fractions:
        related = [item for run in runs for item in run["metrics"]["C_related"] if float(item["fraction"]) == frac]
        unrelated = [item for run in runs for item in run["metrics"]["C_unrelated"] if float(item["fraction"]) == frac]
        support_h1 = sum(1 for item in related if item["delta_trace"] > 0 and item["delta_logdet"] > 0)
        support_h2 = sum(
            1
            for rel, unrel in zip(related, unrelated)
            if rel["G_target_norm"] > unrel["G_target_norm"] and rel["G_perf_norm"] > unrel["G_perf_norm"]
        )
        summary[frac] = {
            "related": {
                "delta_trace": _mean_std([item["delta_trace"] for item in related]),
                "delta_logdet": _mean_std([item["delta_logdet"] for item in related]),
                "delta_erank": _mean_std([item["delta_erank"] for item in related]),
                "G_target": _mean_std([item["G_target"] for item in related]),
                "G_target_norm": _mean_std([item["G_target_norm"] for item in related]),
                "G_perf": _mean_std([item["G_perf"] for item in related]),
                "G_perf_norm": _mean_std([item["G_perf_norm"] for item in related]),
                "delta_acc_vs_B": _mean_std([item["delta_acc_vs_B"] for item in related]),
                "target_val_acc": _mean_std([item["target_val_acc"] for item in related]),
            },
            "unrelated": {
                "delta_trace": _mean_std([item["delta_trace"] for item in unrelated]),
                "delta_logdet": _mean_std([item["delta_logdet"] for item in unrelated]),
                "delta_erank": _mean_std([item["delta_erank"] for item in unrelated]),
                "G_target": _mean_std([item["G_target"] for item in unrelated]),
                "G_target_norm": _mean_std([item["G_target_norm"] for item in unrelated]),
                "G_perf": _mean_std([item["G_perf"] for item in unrelated]),
                "G_perf_norm": _mean_std([item["G_perf_norm"] for item in unrelated]),
                "delta_acc_vs_B": _mean_std([item["delta_acc_vs_B"] for item in unrelated]),
                "target_val_acc": _mean_std([item["target_val_acc"] for item in unrelated]),
            },
            "support_counts": {
                "H1_related_positive_gain": int(support_h1),
                "H2_related_beats_unrelated_on_target_relevant_gain": int(support_h2),
            },
        }
    return summary


def _global_correlations(runs: list[dict[str, Any]]) -> dict[str, Any]:
    related = defaultdict(list)
    unrelated = defaultdict(list)
    for run in runs:
        corr_rel = run["correlations"]["C_related"]
        corr_unrel = run["correlations"]["C_unrelated"]
        for key, value in corr_rel.items():
            related[key].append(value)
        for key, value in corr_unrel.items():
            unrelated[key].append(value)
    return {
        "related": {key: _mean_std(values) for key, values in related.items()},
        "unrelated": {key: _mean_std(values) for key, values in unrelated.items()},
    }


def _decision(summary: dict[float, dict[str, Any]], correlations: dict[str, Any], num_seeds: int) -> dict[str, Any]:
    h1_ckpts = []
    h2_alignment_ckpts = []
    for frac, item in summary.items():
        h1_ckpts.append(item["support_counts"]["H1_related_positive_gain"] >= max(2, num_seeds - 1))
        h2_alignment_ckpts.append(item["support_counts"]["H2_related_beats_unrelated_on_target_relevant_gain"] >= max(2, num_seeds - 1))
    related_corr = correlations["related"]
    unrelated_corr = correlations["unrelated"]
    h2_coupling = (
        related_corr["corr_G_target_norm_vs_delta_acc"][0] > 0
        and related_corr["corr_G_target_norm_vs_delta_acc"][0] > unrelated_corr["corr_G_target_norm_vs_delta_acc"][0]
    )
    h2 = bool(all(h2_alignment_ckpts) and h2_coupling)
    h3 = (
        related_corr["corr_G_perf_norm_vs_delta_acc"][0] > 0
        and related_corr["corr_G_perf_norm_vs_delta_acc"][0] > unrelated_corr["corr_G_perf_norm_vs_delta_acc"][0]
    )
    return {
        "H1_L1_supported": bool(all(h1_ckpts)),
        "H2_L1_supported": h2,
        "H3_L1_supported": bool(h3),
        "H2_components": {
            "target_aligned_gain_checkpoints_supported": bool(all(h2_alignment_ckpts)),
            "curvature_performance_coupling_supported": bool(h2_coupling),
        },
    }


def _write_md(runs: list[dict[str, Any]], summary: dict[float, dict[str, Any]], correlations: dict[str, Any], decision: dict[str, Any], output_md: Path) -> None:
    first = runs[0]
    lines = []
    lines.append("# Layer-1 Shared-Parameter Fisher Gain on Food101")
    lines.append("")
    lines.append("This report follows the revised layer-1 experiment: measure whether the auxiliary modality adds useful Fisher curvature on the whole shared classifier head, rather than searching for localized interaction blocks.")
    lines.append("")
    lines.append("## Setting")
    lines.append("")
    lines.append(f"- Dataset: `Food101`")
    lines.append(f"- Positive condition: related Food101 text prompts")
    lines.append(f"- Negative control: class-permuted Food101 text prompts with the same label space")
    lines.append(f"- Model: `{first['setting']['model']}`")
    lines.append(f"- Shared block: `{first['setting']['shared_block']}`")
    lines.append(f"- Encoder: `{first['setting']['clip_encoder']}`")
    lines.append(f"- Seeds: {', '.join(str(run['seed']) for run in runs)}")
    lines.append("")
    lines.append("For each checkpoint, we compute")
    lines.append("")
    lines.append("```math")
    lines.append("\\widehat F_X = \\frac{1}{N_X}\\sum_i g_X^{(i)} g_X^{(i)\\top}, \\qquad")
    lines.append("\\widehat F_Y = \\frac{1}{N_Y}\\sum_j g_Y^{(j)} g_Y^{(j)\\top}, \\qquad")
    lines.append("\\widehat F_{\\mathrm{shared}} = \\widehat F_X + \\alpha \\widehat F_Y.")
    lines.append("```")
    lines.append("")
    lines.append("We report `Δtrace`, `Δlogdet`, target-subspace-weighted Fisher gain (`G_target`, `G_target_norm`), performance-weighted Fisher gain (`G_perf`, `G_perf_norm`), and target-only validation accuracy gain over the compute-matched target-only control (Arm B).")
    lines.append("")
    lines.append("## Checkpoint Results")
    lines.append("")
    lines.append("| Ckpt | Related Δtrace | Unrelated Δtrace | Related G_target_norm | Unrelated G_target_norm | Related G_perf_norm | Unrelated G_perf_norm | Related Δacc vs B | Unrelated Δacc vs B | H1 count | H2 count |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for frac in sorted(summary):
        item = summary[frac]
        lines.append(
            f"| {int(frac * 100)}% | "
            f"{_fmt(item['related']['delta_trace'])} | "
            f"{_fmt(item['unrelated']['delta_trace'])} | "
            f"{_fmt(item['related']['G_target_norm'])} | "
            f"{_fmt(item['unrelated']['G_target_norm'])} | "
            f"{_fmt(item['related']['G_perf_norm'])} | "
            f"{_fmt(item['unrelated']['G_perf_norm'])} | "
            f"{_fmt(item['related']['delta_acc_vs_B'])} | "
            f"{_fmt(item['unrelated']['delta_acc_vs_B'])} | "
            f"{item['support_counts']['H1_related_positive_gain']}/{len(runs)} | "
            f"{item['support_counts']['H2_related_beats_unrelated_on_target_relevant_gain']}/{len(runs)} |"
        )
    lines.append("")
    lines.append("## Correlation")
    lines.append("")
    lines.append(f"- Related `corr(Δtrace, Δacc)`: {_fmt(correlations['related']['corr_delta_trace_vs_delta_acc'])}")
    lines.append(f"- Related `corr(Δlogdet, Δacc)`: {_fmt(correlations['related']['corr_delta_logdet_vs_delta_acc'])}")
    lines.append(f"- Related `corr(G_target_norm, Δacc)`: {_fmt(correlations['related']['corr_G_target_norm_vs_delta_acc'])}")
    lines.append(f"- Related `corr(G_perf_norm, Δacc)`: {_fmt(correlations['related']['corr_G_perf_norm_vs_delta_acc'])}")
    lines.append(f"- Unrelated `corr(Δtrace, Δacc)`: {_fmt(correlations['unrelated']['corr_delta_trace_vs_delta_acc'])}")
    lines.append(f"- Unrelated `corr(Δlogdet, Δacc)`: {_fmt(correlations['unrelated']['corr_delta_logdet_vs_delta_acc'])}")
    lines.append(f"- Unrelated `corr(G_target_norm, Δacc)`: {_fmt(correlations['unrelated']['corr_G_target_norm_vs_delta_acc'])}")
    lines.append(f"- Unrelated `corr(G_perf_norm, Δacc)`: {_fmt(correlations['unrelated']['corr_G_perf_norm_vs_delta_acc'])}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- H1-L1 supported: `{decision['H1_L1_supported']}`")
    lines.append(f"- H2-L1 supported: `{decision['H2_L1_supported']}`")
    lines.append(f"- H3-L1 supported: `{decision['H3_L1_supported']}`")
    lines.append(f"- H2 target-aligned gain component: `{decision['H2_components']['target_aligned_gain_checkpoints_supported']}`")
    lines.append(f"- H2 coupling component: `{decision['H2_components']['curvature_performance_coupling_supported']}`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Raw shared Fisher gain remains a diagnostic, not the main claim.")
    lines.append("- The revised H2 asks whether related auxiliary data yields stronger target-aligned Fisher gain and better curvature-performance coupling than the permuted control.")
    output_md.write_text("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    runs = _load_runs(Path(args.input_dir))
    if not runs:
        raise SystemExit("no food101_seed*.json files found")
    summary = _per_fraction_summary(runs)
    correlations = _global_correlations(runs)
    decision = _decision(summary, correlations, len(runs))
    payload = {
        "setting": runs[0]["setting"],
        "config": runs[0]["config"],
        "seeds": [run["seed"] for run in runs],
        "per_checkpoint": summary,
        "correlations": correlations,
        "decision": decision,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(payload, indent=2))
    _write_md(runs, summary, correlations, decision, output_md)


if __name__ == "__main__":
    main()
