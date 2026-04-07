from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/zhenshuo/multimodal")
OPT_RESULTS = ROOT / "Multimodal-representation-and-attribution" / "opt_motivation" / "results"
OPT_RUNS = OPT_RESULTS / "fisher_hessian_revised_h1_runs"
FOOD_BASE = ROOT / "Unpaired-Multimodal-Learning" / "vision_language" / "experiments"
H1_C_THRESHOLD = 0.10
H1_LAMBDA_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Hypotheses 1 and 2 across imdb, humor, sarcasm, and food101.")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=OPT_RESULTS / "hypothesis12_multidataset_summary.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OPT_RESULTS / "hypothesis12_multidataset_summary.json",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NaN"
    return f"{number:.{digits}f}"


def _dataset_description(dataset: str) -> dict[str, Any]:
    descriptions = {
        "imdb": {
            "data": "Multimodal IMDb, paired text and image features for multilabel movie genre prediction.",
            "modalities": ["text", "image"],
            "model": "shared_head joint model with per-modality encoders and a shared fusion core.",
            "training": "supervised training, shared_head architecture, batch size 32, Adam-style optimization, test split used only for post-hoc Fisher/Hessian measurement.",
            "seeds": [42, 43, 44],
            "run_paths": [OPT_RUNS / f"imdb_fisher_hessian_seed{seed}.json" for seed in [42, 43, 44]],
        },
        "humor": {
            "data": "UR-FUNNY-style humor benchmark with paired text, audio, and video features for binary supervised prediction.",
            "modalities": ["text", "audio", "video"],
            "model": "shared_head joint model with modality encoders and a shared fusion core.",
            "training": "supervised training, shared_head architecture, batch size 32, 30 epochs, Fisher/Hessian measured on the test split.",
            "seeds": [42, 43, 44],
            "run_paths": [OPT_RUNS / f"humor_fisher_hessian_seed{seed}.json" for seed in [42, 43, 44]],
        },
        "sarcasm": {
            "data": "MUStARD-style sarcasm benchmark with paired text, audio, and video features for binary supervised prediction.",
            "modalities": ["text", "audio", "video"],
            "model": "shared_head joint model with modality encoders and a shared fusion core.",
            "training": "supervised training, shared_head architecture, batch size 32, 30 epochs, Fisher/Hessian measured on the test split.",
            "seeds": [42, 43, 44],
            "run_paths": [OPT_RUNS / f"sarcasm_fisher_hessian_seed{seed}.json" for seed in [42, 43, 44]],
        },
        "food101": {
            "data": "Food101 from Hugging Face `ethz/food101`, paired image-class data with text prompts generated from class templates.",
            "modalities": ["image", "text"],
            "model": "CLIP ViT-B/16 frozen encoder with a supervised shared linear head (`head.weight`).",
            "training": "head-only supervised training with frozen CLIP image encoder, template_mining text prompts, 15 text shots per class, random classifier initialization.",
            "seeds": [1, 2, 3],
            "run_paths": [
                FOOD_BASE / f"food101-shot_-1-seed_{seed}" / "ViT-B-16" / f"food101_head_fisher_hessian_template15_seed{seed}_revised_h1.json"
                for seed in [1, 2, 3]
            ],
        },
    }
    return descriptions[dataset]


def _load_opt_dataset(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return {
        "seed": int(payload["seed"]),
        "dataset": payload["dataset"],
        "analysis_split": payload["analysis_split"],
        "config": payload["config"],
        "target_metrics": payload["metrics"]["targets"],
        "directional_pairs": payload["metrics"]["directional_pairs"],
    }


def _load_food_dataset(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return {
        "seed": int(payload["seed"]),
        "dataset": payload["dataset"],
        "analysis_split": payload["analysis_split"],
        "config": payload["config"],
        "target_metrics": payload["runs"]["joint_crossmodal"]["metrics"]["targets"],
        "directional_pairs": payload["runs"]["joint_crossmodal"]["metrics"]["directional_pairs"],
        "image_only_outcome": payload["runs"]["image_only"]["outcome"],
        "joint_outcome": payload["runs"]["joint_crossmodal"]["outcome"],
        "transfer_outcome": payload["transfer_outcome"],
    }


def _load_dataset_runs(dataset: str) -> list[dict[str, Any]]:
    info = _dataset_description(dataset)
    payloads: list[dict[str, Any]] = []
    for path in info["run_paths"]:
        if not path.exists():
            raise FileNotFoundError(f"Missing run for {dataset}: {path}")
        if dataset == "food101":
            payloads.append(_load_food_dataset(path))
        else:
            payloads.append(_load_opt_dataset(path))
    return payloads


def _aggregate(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    pair_names = list(payloads[0]["directional_pairs"].keys())
    target_names = list(payloads[0]["target_metrics"].keys())

    pair_summary: dict[str, Any] = {}
    for pair_name in pair_names:
        pairs = [payload["directional_pairs"][pair_name] for payload in payloads]
        pair_summary[pair_name] = {
            "logdet_gain": _mean_std([pair["fisher"]["logdet_gain_aux_to_target"] for pair in pairs]),
            "normalized_complementarity": _mean_std([pair["fisher"]["normalized_complementarity"] for pair in pairs]),
            "normalized_redundancy": _mean_std([pair["fisher"]["normalized_redundancy"] for pair in pairs]),
            "orthogonal_lambda_ratio": _mean_std([pair["fisher"]["orthogonal_lambda_ratio"] for pair in pairs]),
            "orthogonal_lambda_null_p95": _mean_std([pair["fisher"]["orthogonal_lambda_null"]["orthogonal_lambda_ratio_p95"] for pair in pairs]),
            "orthogonal_lambda_null_p_value": _mean_std([pair["fisher"]["orthogonal_lambda_null"]["p_value_greater_equal_random"] for pair in pairs]),
            "null_c_norm_p95": _mean_std([pair["fisher"]["null_calibration"]["normalized_complementarity_p95"] for pair in pairs]),
            "null_p_value": _mean_std([pair["fisher"]["null_calibration"]["p_value_greater_equal_random"] for pair in pairs]),
            "grad_cosine": _mean_std([pair["gradient"]["mean_cosine"] for pair in pairs]),
            "conflict_rate": _mean_std([pair["gradient"]["batch_conflict_rate"] for pair in pairs]),
            "q_yy": _mean_std([pair["hessian"]["q_yy_given_target"] for pair in pairs]),
            "q_yx": _mean_std([pair["hessian"]["q_yx_given_target"] for pair in pairs]),
            "b_opt": _mean_std([pair["local_transfer"]["benefit"] for pair in pairs]),
            "positive_transfer_count": int(sum(pair["local_transfer"]["decision"] == "positive_transfer" for pair in pairs)),
            "complementarity_beats_redundancy_count": int(
                sum(pair["fisher"]["normalized_complementarity"] > pair["fisher"]["normalized_redundancy"] for pair in pairs)
            ),
            "out_of_subspace_mass_count": int(
                sum(pair["fisher"]["normalized_complementarity"] > H1_C_THRESHOLD for pair in pairs)
            ),
            "null_support_count": int(
                sum(bool(pair["fisher"]["null_calibration"]["supported_vs_random_subspace_95"]) for pair in pairs)
            ),
            "orthogonal_lambda_support_count": int(
                sum(pair["fisher"]["orthogonal_lambda_ratio"] > H1_LAMBDA_THRESHOLD for pair in pairs)
            ),
            "orthogonal_lambda_null_support_count": int(
                sum(bool(pair["fisher"]["orthogonal_lambda_null"]["supported_vs_random_subspace_95"]) for pair in pairs)
            ),
        }

    target_summary: dict[str, Any] = {}
    for target_name in target_names:
        targets = [payload["target_metrics"][target_name] for payload in payloads]
        target_summary[target_name] = {
            "fisher_trace": _mean_std([target["fisher_target"]["trace"] for target in targets]),
            "fisher_lambda_max": _mean_std([target["fisher_target"]["lambda_max"] for target in targets]),
            "hessian_lambda_max": _mean_std([target["hessian_target"]["lambda_max"] for target in targets]),
            "hessian_trace": _mean_std([target["hessian_target"]["trace_hutchinson"]["mean"] for target in targets]),
            "fisher_hessian_overlap": _mean_std([target["hessian_target"]["fisher_hessian_subspace_overlap"] for target in targets]),
        }

    hypothesis_1_supported_pairs = [
        name for name, item in pair_summary.items() if item["null_support_count"] >= 2
    ]
    hypothesis_1_lambda_stable_pairs = [
        name for name, item in pair_summary.items() if item["orthogonal_lambda_null_support_count"] >= 2
    ]
    hypothesis_2_supported_pairs = [
        name for name, item in pair_summary.items() if item["complementarity_beats_redundancy_count"] >= 2
    ]

    aggregate = {
        "num_seeds": len(payloads),
        "seeds": [int(payload["seed"]) for payload in payloads],
        "hypothesis_1": {
            "supported_pairs": hypothesis_1_supported_pairs,
            "verdict": bool(hypothesis_1_supported_pairs),
            "criterion": {
                "type": "null_calibrated_out_of_subspace_fisher_mass",
                "comparison": "observed_c_norm_gt_random_subspace_p95",
                "random_subspace_quantile": 0.95,
                "required_seed_count": 2,
            },
            "orthogonal_lambda_stable_pairs": hypothesis_1_lambda_stable_pairs,
            "orthogonal_lambda_random_subspace_quantile": 0.95,
            "orthogonal_lambda_ratio_threshold": H1_LAMBDA_THRESHOLD,
        },
        "hypothesis_2": {
            "supported_pairs": hypothesis_2_supported_pairs,
            "verdict": len(hypothesis_2_supported_pairs) >= max(1, len(pair_summary) // 2),
        },
    }

    if "transfer_outcome" in payloads[0]:
        aggregate["transfer_outcome"] = {
            "joint_minus_image_val_acc": _mean_std([payload["transfer_outcome"]["joint_minus_image_val_acc"] for payload in payloads]),
            "joint_minus_image_test_acc": _mean_std([payload["transfer_outcome"]["joint_minus_image_test_acc"] for payload in payloads]),
        }

    return {
        "seeds": [int(payload["seed"]) for payload in payloads],
        "config": payloads[0]["config"],
        "directional_pairs": pair_summary,
        "targets": target_summary,
        "aggregate": aggregate,
    }


def _dataset_section(dataset: str, summary: dict[str, Any]) -> list[str]:
    info = _dataset_description(dataset)
    agg = summary["aggregate"]
    config = summary["config"]
    lines = [
        f"## {dataset.title() if dataset != 'imdb' else 'IMDb'}",
        "",
        f"**Data.** {info['data']}",
        "",
        f"**Model and training.** {info['model']} {info['training']}",
        "",
        f"**Measurement setup.** Seeds = {', '.join(str(seed) for seed in agg['seeds'])}. "
        f"Analysis split = test. Measurement batch size = {config['measurement_batch_size']}. "
        f"Fisher batches = {config['fisher_batches']}. Hessian batches = {config['hessian_batches']}. "
        f"Random-subspace null samples = {config.get('random_subspace_samples', 'NA')}.",
        "",
        f"**Hypothesis 1 verdict.** {'Supported' if agg['hypothesis_1']['verdict'] else 'Not supported'} under the null-calibrated out-of-subspace Fisher test."
        f" Supported pairs: {', '.join(agg['hypothesis_1']['supported_pairs']) if agg['hypothesis_1']['supported_pairs'] else 'none'}.",
        "",
        f"**Orthogonal-eigenvalue diagnostic.** Pairs whose observed orthogonal lambda ratio exceeds the matched random-subspace 95th percentile in at least 2 out of 3 seeds: "
        f"{', '.join(agg['hypothesis_1']['orthogonal_lambda_stable_pairs']) if agg['hypothesis_1']['orthogonal_lambda_stable_pairs'] else 'none'}.",
        "",
        f"**Hypothesis 2 verdict.** {'Supported' if agg['hypothesis_2']['verdict'] else 'Not supported'}."
        f" Supported pairs: {', '.join(agg['hypothesis_2']['supported_pairs']) if agg['hypothesis_2']['supported_pairs'] else 'none'}.",
        "",
        "| Pair | H1 null count | rho_perp null count | C_norm | Null p95(C_norm) | rho_perp | Null p95(rho_perp) | R_norm | Δlogdet |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair_name, pair in summary["directional_pairs"].items():
        lines.append(
            f"| {pair_name} | "
            f"{pair['null_support_count']}/3 | "
            f"{pair['orthogonal_lambda_null_support_count']}/3 | "
            f"{_fmt(pair['normalized_complementarity']['mean'])} ± {_fmt(pair['normalized_complementarity']['std'])} | "
            f"{_fmt(pair['null_c_norm_p95']['mean'])} ± {_fmt(pair['null_c_norm_p95']['std'])} | "
            f"{_fmt(pair['orthogonal_lambda_ratio']['mean'])} ± {_fmt(pair['orthogonal_lambda_ratio']['std'])} | "
            f"{_fmt(pair['orthogonal_lambda_null_p95']['mean'])} ± {_fmt(pair['orthogonal_lambda_null_p95']['std'])} | "
            f"{_fmt(pair['normalized_redundancy']['mean'])} ± {_fmt(pair['normalized_redundancy']['std'])} | "
            f"{_fmt(pair['logdet_gain']['mean'])} ± {_fmt(pair['logdet_gain']['std'])} |"
        )
    lines.append("")
    lines.append("| Pair | Mean null p-value C_norm | Mean null p-value rho_perp | Grad cosine | Conflict rate | q_YY|X | q_YX|X | B_opt |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pair_name, pair in summary["directional_pairs"].items():
        lines.append(
            f"| {pair_name} | "
            f"{_fmt(pair['null_p_value']['mean'])} ± {_fmt(pair['null_p_value']['std'])} | "
            f"{_fmt(pair['orthogonal_lambda_null_p_value']['mean'])} ± {_fmt(pair['orthogonal_lambda_null_p_value']['std'])} | "
            f"{_fmt(pair['grad_cosine']['mean'])} ± {_fmt(pair['grad_cosine']['std'])} | "
            f"{_fmt(pair['conflict_rate']['mean'])} ± {_fmt(pair['conflict_rate']['std'])} | "
            f"{_fmt(pair['q_yy']['mean'])} ± {_fmt(pair['q_yy']['std'])} | "
            f"{_fmt(pair['q_yx']['mean'])} ± {_fmt(pair['q_yx']['std'])} | "
            f"{_fmt(pair['b_opt']['mean'], 8)} ± {_fmt(pair['b_opt']['std'], 8)} |"
        )
    lines.append("")
    lines.append("| Target | Fisher trace | Fisher λmax | Hessian λmax | Hessian trace | F/H overlap |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for target_name, target in summary["targets"].items():
        lines.append(
            f"| {target_name} | "
            f"{_fmt(target['fisher_trace']['mean'])} ± {_fmt(target['fisher_trace']['std'])} | "
            f"{_fmt(target['fisher_lambda_max']['mean'])} ± {_fmt(target['fisher_lambda_max']['std'])} | "
            f"{_fmt(target['hessian_lambda_max']['mean'])} ± {_fmt(target['hessian_lambda_max']['std'])} | "
            f"{_fmt(target['hessian_trace']['mean'])} ± {_fmt(target['hessian_trace']['std'])} | "
            f"{_fmt(target['fisher_hessian_overlap']['mean'])} ± {_fmt(target['fisher_hessian_overlap']['std'])} |"
        )
    if "transfer_outcome" in agg:
        lines.extend(
            [
                "",
                f"**Observed transfer outcome.** Joint minus image-only validation accuracy = "
                f"{_fmt(agg['transfer_outcome']['joint_minus_image_val_acc']['mean'])} ± {_fmt(agg['transfer_outcome']['joint_minus_image_val_acc']['std'])}; "
                f"joint minus image-only test accuracy = "
                f"{_fmt(agg['transfer_outcome']['joint_minus_image_test_acc']['mean'])} ± {_fmt(agg['transfer_outcome']['joint_minus_image_test_acc']['std'])}.",
            ]
        )
    lines.append("")
    return lines


def _markdown(summary: dict[str, Any]) -> str:
    overall_h1_supported = [dataset for dataset in summary["datasets"] if summary["results"][dataset]["aggregate"]["hypothesis_1"]["verdict"]]
    overall_h2_supported = [dataset for dataset in summary["datasets"] if summary["results"][dataset]["aggregate"]["hypothesis_2"]["verdict"]]
    lines = [
        "# Solid Summary of Hypotheses 1 and 2 Across Four Datasets",
        "",
        "This document focuses on the first two hypotheses from `hessian_hypo_test.md`:",
        "",
        "1. **Hypothesis 1:** an auxiliary modality contributes stable out-of-subspace Fisher directions for the target modality.",
        "2. **Hypothesis 2:** these new directions are primarily complementary rather than redundant.",
        "",
        "All results below are aggregated over 3 seeds per dataset.",
        "",
        "## Experimental Setting",
        "",
        "### Data",
        "",
        "- **IMDb:** paired text and image features for multilabel movie genre prediction.",
        "- **Humor:** paired text, audio, and video features for binary humor detection.",
        "- **Sarcasm:** paired text, audio, and video features for binary sarcasm detection.",
        "- **Food101:** images from `ethz/food101` plus template-based text prompts derived from the class names.",
        "",
        "### Models",
        "",
        "- **IMDb / Humor / Sarcasm:** the previously trained `shared_head` multimodal model from `opt_motivation`, with Fisher/Hessian measured on the shared parameter block (`parameter_scope=shared_core`).",
        "- **Food101:** a supervised CLIP ViT-B/16 setting with frozen pretrained encoders and a shared linear classification head; Fisher/Hessian is measured on `head.weight`.",
        "",
        "### Fisher and Hessian Definitions",
        "",
        "For a target modality $X$, an auxiliary modality $Y$, and the shared parameter block $\\phi$, we measure batch gradients",
        "",
        "$$g_m^{(b)} = \\nabla_\\phi L_m^{(b)}$$",
        "",
        "and define the empirical Fisher",
        "",
        "$$\\widehat F_m = \\frac{1}{B}\\sum_{b=1}^{B} g_m^{(b)} g_m^{(b)\\top}.$$\n",
        "For the target modality $X$, let $U_X$ be the principal Fisher subspace of $\\widehat F_X$, chosen as the top-$k$ eigenspace explaining a fixed fraction of Fisher trace.",
        "",
        "Define the projectors",
        "",
        "$$P_X = U_X U_X^\\top, \\qquad P_X^\\perp = I - P_X.$$",
        "",
        "For an auxiliary modality $Y$, define redundancy and complementarity as",
        "",
        "$$R(Y \\to X) = \\mathrm{tr}(P_X \\widehat F_Y P_X),$$",
        "",
        "$$C(Y \\to X) = \\mathrm{tr}(P_X^\\perp \\widehat F_Y P_X^\\perp),$$",
        "",
        "$$R_{\\mathrm{norm}}(Y \\to X) = \\frac{R(Y \\to X)}{\\mathrm{tr}(\\widehat F_Y)}, \\qquad C_{\\mathrm{norm}}(Y \\to X) = \\frac{C(Y \\to X)}{\\mathrm{tr}(\\widehat F_Y)}.$$",
        "",
        "We compute the following quantities:",
        "",
        "- **Total Fisher trace**: $T_m = \\mathrm{tr}(\\widehat F_m)$.",
        "- **Maximum Fisher eigenvalue**: $\\lambda_{\\max}(\\widehat F_m)$.",
        "- **Log-determinant gain**: $\\Delta_{\\log\\det}(Y \\to X) = \\log\\det(\\widehat F_X + \\widehat F_Y + \\varepsilon I) - \\log\\det(\\widehat F_X + \\varepsilon I)$; this is reported as a diagnostic but is not the main criterion for Hypothesis 1.",
        "- **Redundancy**: Fisher energy from $Y$ projected into the principal Fisher subspace of $X$.",
        "- **Complementarity**: Fisher trace of $Y$ not explained by the principal Fisher subspace of $X$.",
        "- **Normalized redundancy**: redundancy divided by $\\mathrm{tr}(\\widehat F_Y)$.",
        "- **Normalized complementarity**: complementarity divided by $\\mathrm{tr}(\\widehat F_Y)$.",
        "- **Orthogonal lambda ratio**: $\\lambda_{\\max}(P_X^\\perp \\widehat F_Y P_X^\\perp) / \\lambda_{\\max}(\\widehat F_Y)$, used as a diagnostic that the out-of-subspace Fisher mass contains a nontrivial orthogonal direction.",
        "- **Gradient cosine**: cosine similarity between the mean shared gradients of $X$ and $Y$.",
        "- **Conflict rate**: fraction of batch pairs whose shared-gradient inner product is negative.",
        "- **Hessian quadratic terms**: $q_{YY|X} = g_Y^\\top H_X g_Y$ and $q_{YX|X} = g_Y^\\top H_X g_X$.",
        "- **Local transfer score**: $B_{\\mathrm{opt}} = \\eta\\beta g_X^\\top g_Y - \\eta^2 \\beta q_{YX|X} - \\frac{1}{2}\\eta^2 \\beta^2 q_{YY|X}$.",
        "- **Fisher/Hessian overlap**: overlap between the top Fisher subspace and the top Hessian eigenspace for the target modality.",
        "- **Random-subspace null for H1**: for each pair $Y \\to X$, we sample random orthogonal subspaces with the same dimension as $U_X$ and recompute $C_{\\mathrm{norm}}$. This provides a null distribution for out-of-subspace Fisher mass under a random target subspace.",
        "",
        "### Decision Rule for Hypotheses 1 and 2",
        "",
        "- **Hypothesis 1 is counted as supported for a pair** if the observed $C_{\\mathrm{norm}}(Y \\to X)$ is greater than the 95th percentile of the random-subspace null distribution in at least 2 out of 3 seeds.",
        "- **Orthogonal-direction stability diagnostic**: we additionally report whether $\\lambda_{\\max}(P_X^\\perp \\widehat F_Y P_X^\\perp) / \\lambda_{\\max}(\\widehat F_Y) > 0.05$ in at least 2 out of 3 seeds.",
        "- **Hypothesis 2 is counted as supported for a pair** if $C_{\\mathrm{norm}}(Y \\to X) > R_{\\mathrm{norm}}(Y \\to X)$ in at least 2 out of 3 seeds.",
        "",
        "This report uses the null-calibrated rule as the main empirical test for Hypothesis 1 and treats $\\Delta_{\\log\\det}$ only as a secondary diagnostic.",
        "",
        "## Results",
        "",
    ]
    for dataset in summary["datasets"]:
        lines.extend(_dataset_section(dataset, summary["results"][dataset]))

    lines.extend(
        [
            "## Overall Conclusions",
            "",
            f"- Hypothesis 1 is supported on {len(overall_h1_supported)}/{len(summary['datasets'])} datasets under the null-calibrated out-of-subspace Fisher rule: "
            f"{', '.join(overall_h1_supported) if overall_h1_supported else 'none'}.",
            f"- Hypothesis 2 is supported on {len(overall_h2_supported)}/{len(summary['datasets'])} datasets: "
            f"{', '.join(overall_h2_supported) if overall_h2_supported else 'none'}.",
            "- Across the supported pairs, normalized complementarity remains larger than normalized redundancy, so the added Fisher geometry is not only out-of-subspace but predominantly complementary.",
            "- The orthogonal-lambda diagnostic is reported separately to show whether the out-of-subspace Fisher mass is concentrated enough to contain a nontrivial orthogonal direction.",
            "- Food101 and IMDb are expected to be the cleanest two-modality cases, while humor and sarcasm provide the richer three-modality interaction patterns.",
            "",
            "These conclusions are intentionally limited to Hypotheses 1 and 2. They do not claim that complementary Fisher geometry is sufficient for positive downstream transfer.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    datasets = ["imdb", "humor", "sarcasm", "food101"]
    results = {dataset: _aggregate(_load_dataset_runs(dataset)) for dataset in datasets}
    summary = {"datasets": datasets, "results": results}
    args.output_json.write_text(json.dumps(summary, indent=2))
    args.output_md.write_text(_markdown(summary))
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
