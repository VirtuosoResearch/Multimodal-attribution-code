from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize opt_motivation hypothesis-test results across datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["imdb", "humor", "sarcasm"],
        choices=["imdb", "humor", "sarcasm"],
        help="Datasets to summarize.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing experiment result JSON files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "hypothesis_summary_3datasets.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=RESULTS_DIR / "hypothesis_summary_3datasets.md",
        help="Output Markdown path.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if value != value:
        return "NaN"
    return f"{value:.{digits}f}"


def _summarize_dataset(dataset: str, results_dir: Path) -> dict[str, Any]:
    dec_path = results_dir / f"{dataset}_decoupling_results.json"
    fisher_path = results_dir / f"{dataset}_empirical_fisher_interaction_signaware.json"
    decoupling = _load_json(dec_path)
    fisher = _load_json(fisher_path)

    analysis = decoupling["analysis"]
    auc_models = analysis["auc_models"]
    auc_bootstrap = analysis["auc_bootstrap_tests"]
    per_bucket = analysis["per_bucket_spearman"]
    q2_q3 = analysis["q2_vs_q3_fisher_exact"]

    fisher_analysis = fisher["interaction_analysis"]
    fisher_alignment = fisher_analysis["spearman"]["alignment_score"]
    fisher_conflict = fisher_analysis["spearman"]["conflict_score"]
    fisher_power = fisher_analysis["single_feature_conditional_power"]["alignment_score"]["classification"]["correctness"]

    b_minus_a = auc_bootstrap["B_minus_A"]
    c_minus_b = auc_bootstrap["C_minus_B"]
    both_minus_pid = fisher_power["bootstrap_tests"]["both_minus_pid_auc"]

    return {
        "dataset": dataset,
        "source_files": {
            "decoupling": str(dec_path),
            "fisher": str(fisher_path),
        },
        "config": {
            "decoupling_device": _safe_get(decoupling, "config", "device"),
            "decoupling_epochs": _safe_get(decoupling, "config", "epochs"),
            "fisher_seed": _safe_get(fisher, "seed"),
        },
        "decoupling": {
            "overall_spearman": analysis["overall_spearman"],
            "per_bucket_spearman": per_bucket,
            "auc_models": {
                "rep_only": auc_models["A_rep_only"]["auc"],
                "opt_only": auc_models["B_opt_only"]["auc"],
                "both": auc_models["C_both"]["auc"],
            },
            "auc_bootstrap": {
                "opt_minus_rep": b_minus_a,
                "both_minus_opt": c_minus_b,
            },
            "q2_vs_q3_fisher_exact": q2_q3,
            "contingency": analysis["contingency"],
            "verdict": {
                "decoupling_present": abs(float(analysis["overall_spearman"]["rho"])) < 0.7,
                "opt_beats_rep": float(b_minus_a["mean_diff"]) > 0.0 and float(b_minus_a["p_value_right_gt_left"]) < 0.05,
                "both_adds_over_opt": float(c_minus_b["mean_diff"]) > 0.0 and float(c_minus_b["p_value_right_gt_left"]) < 0.05,
                "q3_worse_than_q2_supported": float(q2_q3["odds_ratio"]) > 1.0 and float(q2_q3["p_value"]) < 0.05,
            },
        },
        "fisher": {
            "alignment_vs_pid_synergy": fisher_alignment["vs_pid_synergy"],
            "alignment_vs_correctness": fisher_alignment["vs_correctness"],
            "conflict_vs_correctness": fisher_conflict["vs_correctness"],
            "alignment_correctness_auc": fisher_power["models"],
            "alignment_correctness_bootstrap": fisher_power["bootstrap_tests"],
            "verdict": {
                "alignment_adds_over_pid": float(both_minus_pid["mean_diff"]) > 0.0
                and float(both_minus_pid["p_value_right_gt_left"]) < 0.05,
            },
        },
        "hessian": {
            "implemented": False,
            "note": "hessian_hypo_test.md defines Hessian/HVP metrics, but the current opt_motivation codebase does not provide a runnable Hessian measurement script.",
        },
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hypothesis Test Summary",
        "",
        "Datasets: " + ", ".join(summary["datasets"]),
        "",
        "Note: `scarcam` is mapped to the implemented dataset name `sarcasm`.",
        "Note: Hessian metrics in `hessian_hypo_test.md` are not implemented in the current codebase; the summary below covers the decoupling and Fisher analyses that are runnable today.",
        "",
    ]

    for dataset in summary["datasets"]:
        item = summary["results"][dataset]
        dec = item["decoupling"]
        fisher = item["fisher"]
        lines.extend(
            [
                f"## {dataset}",
                "",
                "### Decoupling",
                f"- Overall Spearman rho: {_fmt_float(dec['overall_spearman']['rho'])} (p={_fmt_float(dec['overall_spearman']['p_value'])})",
                f"- Rep-only / Opt-only / Both AUC: {_fmt_float(dec['auc_models']['rep_only'])} / {_fmt_float(dec['auc_models']['opt_only'])} / {_fmt_float(dec['auc_models']['both'])}",
                f"- Opt minus Rep AUC: {_fmt_float(dec['auc_bootstrap']['opt_minus_rep']['mean_diff'])} "
                f"[{_fmt_float(dec['auc_bootstrap']['opt_minus_rep']['ci_95'][0])}, {_fmt_float(dec['auc_bootstrap']['opt_minus_rep']['ci_95'][1])}], "
                f"p={_fmt_float(dec['auc_bootstrap']['opt_minus_rep']['p_value_right_gt_left'])}",
                f"- Both minus Opt AUC: {_fmt_float(dec['auc_bootstrap']['both_minus_opt']['mean_diff'])} "
                f"[{_fmt_float(dec['auc_bootstrap']['both_minus_opt']['ci_95'][0])}, {_fmt_float(dec['auc_bootstrap']['both_minus_opt']['ci_95'][1])}], "
                f"p={_fmt_float(dec['auc_bootstrap']['both_minus_opt']['p_value_right_gt_left'])}",
                f"- Q2 vs Q3 Fisher exact: odds ratio={_fmt_float(dec['q2_vs_q3_fisher_exact']['odds_ratio'])}, "
                f"p={_fmt_float(dec['q2_vs_q3_fisher_exact']['p_value'])}",
                f"- Verdict: decoupling_present={dec['verdict']['decoupling_present']}, "
                f"opt_beats_rep={dec['verdict']['opt_beats_rep']}, "
                f"both_adds_over_opt={dec['verdict']['both_adds_over_opt']}, "
                f"q3_worse_than_q2_supported={dec['verdict']['q3_worse_than_q2_supported']}",
                "",
                "### Fisher",
                f"- Alignment score vs PID synergy: rho={_fmt_float(fisher['alignment_vs_pid_synergy']['rho'])}, "
                f"p={_fmt_float(fisher['alignment_vs_pid_synergy']['p_value'])}",
                f"- Alignment score vs correctness: rho={_fmt_float(fisher['alignment_vs_correctness']['rho'])}, "
                f"p={_fmt_float(fisher['alignment_vs_correctness']['p_value'])}",
                f"- Conflict score vs correctness: rho={_fmt_float(fisher['conflict_vs_correctness']['rho'])}, "
                f"p={_fmt_float(fisher['conflict_vs_correctness']['p_value'])}",
                f"- PID-only / Fisher-only / PID+Fisher AUC: "
                f"{_fmt_float(fisher['alignment_correctness_auc']['pid_only']['auc'])} / "
                f"{_fmt_float(fisher['alignment_correctness_auc']['fisher_only']['auc'])} / "
                f"{_fmt_float(fisher['alignment_correctness_auc']['pid_plus_fisher']['auc'])}",
                f"- PID+Fisher minus PID AUC: {_fmt_float(fisher['alignment_correctness_bootstrap']['both_minus_pid_auc']['mean_diff'])} "
                f"[{_fmt_float(fisher['alignment_correctness_bootstrap']['both_minus_pid_auc']['ci_95'][0])}, "
                f"{_fmt_float(fisher['alignment_correctness_bootstrap']['both_minus_pid_auc']['ci_95'][1])}], "
                f"p={_fmt_float(fisher['alignment_correctness_bootstrap']['both_minus_pid_auc']['p_value_right_gt_left'])}",
                f"- Verdict: alignment_adds_over_pid={fisher['verdict']['alignment_adds_over_pid']}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = {dataset: _summarize_dataset(dataset, args.results_dir) for dataset in args.datasets}
    summary = {
        "datasets": args.datasets,
        "results_dir": str(args.results_dir),
        "results": results,
    }

    args.output_json.write_text(json.dumps(summary, indent=2))
    args.output_md.write_text(_markdown_report(summary))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
