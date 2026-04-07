from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize interaction-block MVP results across seeds.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    return parser


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _format_mean_std(values: list[float], *, precision: int = 4) -> str:
    mean_value, std_value = _mean_std(values)
    if math.isnan(mean_value):
        return "nan"
    return f"{mean_value:.{precision}f} ± {std_value:.{precision}f}"


def _pair_to_str(pair: tuple[float, float], *, precision: int = 4) -> str:
    mean_value, std_value = pair
    if math.isnan(mean_value):
        return "nan"
    return f"{mean_value:.{precision}f} ± {std_value:.{precision}f}"


def _load_runs(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        grouped[payload["dataset"]].append(payload)
    return grouped


def _summarize_blocks(runs: list[dict[str, Any]]) -> dict[str, Any]:
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for block_id, payload in run["measurement"]["candidate_blocks"].items():
            blocks[block_id].append(payload)
    summary: dict[str, Any] = {}
    for block_id, entries in blocks.items():
        summary[block_id] = {
            "tensor_name": entries[0]["tensor_name"],
            "row_start": entries[0]["row_start"],
            "row_end": entries[0]["row_end"],
            "support_count": int(sum(1 for entry in entries if entry["supported"])),
            "trace": _mean_std([entry["real"]["trace"] for entry in entries]),
            "lambda_max": _mean_std([entry["real"]["lambda_max"] for entry in entries]),
            "effective_rank": _mean_std([entry["real"]["effective_rank"] for entry in entries]),
            "ratio": _mean_std([entry["real"]["ratio"] for entry in entries]),
            "shuffle_trace_p95": _mean_std([entry["shuffle_null"]["trace_p95"] for entry in entries]),
            "shuffle_lambda_p95": _mean_std([entry["shuffle_null"]["lambda_max_p95"] for entry in entries]),
            "random_trace_p95": _mean_std([entry["random_block_null"]["trace_p95"] for entry in entries]),
            "random_lambda_p95": _mean_std([entry["random_block_null"]["lambda_max_p95"] for entry in entries]),
        }
    return summary


def _top_blocks(block_summary: dict[str, Any], *, top_k: int = 6) -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        block_summary.items(),
        key=lambda item: (item[1]["support_count"], item[1]["trace"][0]),
        reverse=True,
    )[:top_k]


def _summarize_intervention(runs: list[dict[str, Any]]) -> dict[str, Any]:
    top_joint = []
    top_target = []
    top_aux = []
    rand_joint = []
    rand_target = []
    rand_aux = []
    top_ids = []
    rand_ids = []
    for run in runs:
        intervention = run.get("intervention", {})
        top_block = intervention.get("top_block")
        if top_block is None:
            continue
        top_ids.append(top_block["block_id"])
        if run["dataset"] == "imdb":
            top_joint.append(top_block["joint_delta"])
            top_target.append(top_block["text_only_delta"])
            top_aux.append(top_block["image_only_delta"])
            random_block = intervention.get("random_block")
            if random_block is not None:
                rand_ids.append(random_block["block_id"])
                rand_joint.append(random_block["joint_delta"])
                rand_target.append(random_block["text_only_delta"])
                rand_aux.append(random_block["image_only_delta"])
        else:
            top_joint.append(-top_block["joint_acc_delta"])
            top_target.append(-top_block["image_only_acc_delta"])
            top_aux.append(-top_block["text_only_acc_delta"])
            random_block = intervention.get("random_block")
            if random_block is not None:
                rand_ids.append(random_block["block_id"])
                rand_joint.append(-random_block["joint_acc_delta"])
                rand_target.append(-random_block["image_only_acc_delta"])
                rand_aux.append(-random_block["text_only_acc_delta"])
    return {
        "top_block_ids": top_ids,
        "random_block_ids": rand_ids,
        "top_joint_effect": _mean_std(top_joint),
        "top_target_effect": _mean_std(top_target),
        "top_aux_effect": _mean_std(top_aux),
        "random_joint_effect": _mean_std(rand_joint),
        "random_target_effect": _mean_std(rand_target),
        "random_aux_effect": _mean_std(rand_aux),
    }


def _dataset_caveat(dataset_name: str) -> str | None:
    if dataset_name == "food101":
        return (
            "Food101 text features are class-template derived, so label-conditional shuffle is degenerate. "
            "The interaction null here uses global text shuffle, which is harsher than the IMDb null."
        )
    return None


def _write_markdown(summary: dict[str, Any], output_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Interaction Fisher Block MVP Results")
    lines.append("")
    lines.append("This report follows the revised blockwise interaction-Fisher design based on interaction residual gradients.")
    lines.append("")
    lines.append("For each candidate block `B`, we compute")
    lines.append("")
    lines.append("```math")
    lines.append("r_B = g_B^{xy} - g_B^{x0} - g_B^{0y} + g_B^{00}")
    lines.append("```")
    lines.append("")
    lines.append("and define the interaction Fisher block")
    lines.append("")
    lines.append("```math")
    lines.append("\\widehat F_{\\mathrm{int}}^{(B)} = \\frac{1}{N}\\sum_i r_B^{(i)} r_B^{(i)\\top}.")
    lines.append("```")
    lines.append("")
    lines.append("Each block is scored by interaction trace, top eigenvalue, effective rank, and interaction-to-full Fisher ratio. A block is counted as supported in a seed only if both interaction trace and top eigenvalue exceed both nulls:")
    lines.append("")
    lines.append("- Null A: modality shuffle that breaks cross-modal coupling")
    lines.append("- Null B: same-tensor random block baseline")
    lines.append("")
    lines.append("A block is treated as a stable interaction block if it is supported in at least `2/3` seeds.")
    lines.append("")
    for dataset_name, dataset_summary in summary["datasets"].items():
        seed_count = len(dataset_summary["seeds"])
        lines.append(f"## {dataset_name.capitalize()}")
        lines.append("")
        lines.append(f"Seeds: {', '.join(str(seed) for seed in dataset_summary['seeds'])}")
        lines.append("")
        lines.append(f"Model: `{dataset_summary['setting']['model']}`")
        lines.append("")
        lines.append(
            f"Target modality: `{dataset_summary['setting']['target_modality']}`. "
            f"Auxiliary modality: `{dataset_summary['setting']['auxiliary_modality']}`."
        )
        lines.append("")
        caveat = _dataset_caveat(dataset_name)
        if caveat:
            lines.append(f"Caveat: {caveat}")
            lines.append("")
        lines.append(
            f"Stable supported blocks: `{dataset_summary['supported_block_count']}` / `{dataset_summary['num_blocks']}`."
        )
        lines.append("")
        lines.append("| Block | Support | Trace | Lambda_max | E-rank | Ratio | Shuffle trace p95 | Random-block trace p95 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for block_id, block in dataset_summary["top_blocks"]:
            lines.append(
                f"| {block_id} | {block['support_count']}/{seed_count} | "
                f"{_pair_to_str(block['trace'])} | "
                f"{_pair_to_str(block['lambda_max'])} | "
                f"{_pair_to_str(block['effective_rank'])} | "
                f"{_pair_to_str(block['ratio'])} | "
                f"{_pair_to_str(block['shuffle_trace_p95'])} | "
                f"{_pair_to_str(block['random_trace_p95'])} |"
            )
        lines.append("")
        intervention = dataset_summary["intervention"]
        lines.append("Intervention summary:")
        lines.append(
            f"- Top-block effect on joint metric: {intervention['top_joint_effect'][0]:.4f} ± {intervention['top_joint_effect'][1]:.4f}"
        )
        lines.append(
            f"- Top-block effect on target-only metric: {intervention['top_target_effect'][0]:.4f} ± {intervention['top_target_effect'][1]:.4f}"
        )
        lines.append(
            f"- Top-block effect on auxiliary-only metric: {intervention['top_aux_effect'][0]:.4f} ± {intervention['top_aux_effect'][1]:.4f}"
        )
        if not math.isnan(intervention["random_joint_effect"][0]):
            lines.append(
                f"- Random-block effect on joint metric: {intervention['random_joint_effect'][0]:.4f} ± {intervention['random_joint_effect'][1]:.4f}"
            )
            lines.append(
                f"- Random-block effect on target-only metric: {intervention['random_target_effect'][0]:.4f} ± {intervention['random_target_effect'][1]:.4f}"
            )
            lines.append(
                f"- Random-block effect on auxiliary-only metric: {intervention['random_aux_effect'][0]:.4f} ± {intervention['random_aux_effect'][1]:.4f}"
            )
        lines.append("")
        lines.append(
            f"Model-level verdict: {'interaction parameter subgraph detected' if dataset_summary['supported_block_count'] > 0 else 'no stable interaction block detected under the current nulls'}."
        )
        lines.append("")
    output_md.write_text("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    grouped = _load_runs(input_dir)
    summary: dict[str, Any] = {"datasets": {}}
    for dataset_name, runs in grouped.items():
        runs = sorted(runs, key=lambda item: item["seed"])
        block_summary = _summarize_blocks(runs)
        top_blocks = _top_blocks(block_summary)
        supported_block_count = int(sum(1 for block in block_summary.values() if block["support_count"] >= 2))
        summary["datasets"][dataset_name] = {
            "seeds": [run["seed"] for run in runs],
            "setting": runs[0]["setting"],
            "config": runs[0]["config"],
            "num_blocks": len(block_summary),
            "supported_block_count": supported_block_count,
            "blocks": block_summary,
            "top_blocks": top_blocks,
            "intervention": _summarize_intervention(runs),
        }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    _write_markdown(summary, output_md)


if __name__ == "__main__":
    main()
