from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pid_estimator.non_interactive_completion import (
    NonInteractiveCompletionConfig,
    estimate_subset_mask_non_interactive_completion_pid,
)
from pid_estimator.subset_pid_kernel_experiment import enumerate_masks, select_score_target
from pid_estimator.synthetic_data import make_binary_logic_dataset


@dataclass(frozen=True)
class SearchConfig:
    logic_cases: tuple[str, ...] = ("redundant", "unique0", "unique1", "xor")
    targets: tuple[str, ...] = (
        "v12",
        "completion",
        "joint_minus_completion",
        "joint_minus_best",
        "completion_minus_best",
        "min_single",
        "max_single",
        "mean_single",
        "sum_single",
        "joint_minus_mean_single",
        "completion_minus_mean_single",
        "joint_minus_sum_single",
        "completion_minus_sum_single",
        "joint_accuracy",
        "joint_disagreement_accuracy",
        "joint_rescue_rate",
        "synergy",
        "synergy_minus_redundancy",
    )
    blend_alphas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    predictor_model: str = "kernel"
    additive_model: str = "kernel"
    predictor_epochs: int = 20
    additive_epochs: int = 20
    kernel_ridge_lambda: float = 1e-3
    batch_size: int = 4
    seed: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Search scalar-target and same-game PID rule combinations.")
    parser.add_argument("--logic-cases", nargs="+", default=["redundant", "unique0", "unique1", "xor"])
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["v12", "completion", "joint_minus_completion", "synergy", "synergy_minus_redundancy"],
    )
    parser.add_argument("--blend-alphas", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--predictor-model", type=str, default="kernel")
    parser.add_argument("--additive-model", type=str, default="kernel")
    parser.add_argument("--predictor-epochs", type=int, default=20)
    parser.add_argument("--additive-epochs", type=int, default=20)
    parser.add_argument("--kernel-ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=str,
        default="src/pid_estimator/results/subset_pid_combo_search.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/pid_estimator/results/subset_pid_combo_search.md",
    )
    args = parser.parse_args()

    config = SearchConfig(
        logic_cases=tuple(args.logic_cases),
        targets=tuple(args.targets),
        blend_alphas=tuple(args.blend_alphas),
        predictor_model=args.predictor_model,
        additive_model=args.additive_model,
        predictor_epochs=args.predictor_epochs,
        additive_epochs=args.additive_epochs,
        kernel_ridge_lambda=args.kernel_ridge_lambda,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    payload = run_search(config)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload))


def run_search(config: SearchConfig) -> dict[str, object]:
    oracle_scores_by_case = {
        logic_case: collect_target_scores(logic_case=logic_case, config=config)
        for logic_case in config.logic_cases
    }
    rule_specs = build_rule_specs()
    expected = {
        "redundant": "R",
        "unique0": "U1",
        "unique1": "U2",
        "xor": "S",
    }
    combos = []
    all_target_names = list(config.targets) + [f"blend_v12_completion_{alpha:.2f}" for alpha in config.blend_alphas]
    for target_name in all_target_names:
        for rule_spec in rule_specs:
            case_results = {}
            num_correct = 0
            for logic_case in config.logic_cases:
                scores = oracle_scores_by_case[logic_case][target_name]
                components = estimate_same_game_components(scores, rule_spec=rule_spec)
                means = {name: float(np.mean(values)) for name, values in components.items()}
                dominant = positive_dominant_component(means)
                expected_component = expected.get(logic_case)
                is_correct = dominant == expected_component
                num_correct += int(is_correct)
                case_results[logic_case] = {
                    "means": means,
                    "dominant_positive": dominant,
                    "expected": expected_component,
                    "is_correct": is_correct,
                }
            combos.append(
                {
                    "target": target_name,
                    "rule": rule_spec["name"],
                    "num_correct": num_correct,
                    "case_results": case_results,
                }
            )
    combos.sort(
        key=lambda item: (
            -int(item["num_correct"]),
            0 if item["target"] == "completion" else 1,
            item["target"],
            item["rule"],
        )
    )
    return {
        "experiment": "subset_pid_combo_search",
        "config": asdict(config),
        "combos": combos,
        "best_combos": combos[:10],
    }


def collect_target_scores(*, logic_case: str, config: SearchConfig) -> dict[str, np.ndarray]:
    train_x1, train_x2, train_labels, _dataset = make_binary_logic_dataset(logic_case, repeat=1)
    masks = enumerate_masks(num_rows=int(train_x1.shape[0]), num_modalities=2)
    eval_config = NonInteractiveCompletionConfig(
        predictor_model=config.predictor_model,
        additive_model=config.additive_model,
        predictor_epochs=config.predictor_epochs,
        additive_epochs=config.additive_epochs,
        kernel_ridge_lambda=config.kernel_ridge_lambda,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    base_target_names = set(config.targets) | {"v12", "completion"}
    scores = {target_name: [] for target_name in base_target_names}
    for subset_mask in masks:
        result = estimate_subset_mask_non_interactive_completion_pid(
            train_x1,
            train_x2,
            train_labels,
            subset_mask=subset_mask,
            num_labels=2,
            config=eval_config,
            device="cpu",
        )
        for target_name in config.targets:
            scores[target_name].append(
                select_score_target(
                    result,
                    target_name=target_name,
                    eval_x1=train_x1,
                    eval_x2=train_x2,
                    eval_labels=train_labels,
                )
            )
        if "v12" not in config.targets:
            scores["v12"].append(
                select_score_target(
                    result,
                    target_name="v12",
                    eval_x1=train_x1,
                    eval_x2=train_x2,
                    eval_labels=train_labels,
                )
            )
        if "completion" not in config.targets:
            scores["completion"].append(
                select_score_target(
                    result,
                    target_name="completion",
                    eval_x1=train_x1,
                    eval_x2=train_x2,
                    eval_labels=train_labels,
                )
            )
    arrays = {target_name: np.asarray(values, dtype=np.float64) for target_name, values in scores.items()}
    for alpha in config.blend_alphas:
        arrays[f"blend_v12_completion_{alpha:.2f}"] = (
            (1.0 - float(alpha)) * arrays["v12"] + float(alpha) * arrays["completion"]
        )
    return arrays


def build_rule_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"name": "signed_min", "kind": "signed_min"},
        {"name": "positive_min", "kind": "positive_min"},
        {"name": "raw_min", "kind": "raw_min"},
        {"name": "zero_redundancy", "kind": "zero"},
        {"name": "coinfo_half", "kind": "coinfo_half"},
    ]
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        specs.append({"name": f"completion_alpha_{alpha:.2f}", "kind": "completion_alpha", "alpha": alpha})
    return specs


def estimate_same_game_components(
    scores: np.ndarray,
    *,
    rule_spec: dict[str, object],
) -> dict[str, np.ndarray]:
    num_atoms = int(round(np.log2(scores.shape[0])))
    if 2**num_atoms != scores.shape[0]:
        raise ValueError("scores must enumerate all masks")
    rows = num_atoms // 2
    modality1_atoms = [2 * index for index in range(rows)]
    modality2_atoms = [2 * index + 1 for index in range(rows)]
    outputs = {"R": [], "U1": [], "U2": [], "S": []}
    for atom_left in modality1_atoms:
        for atom_right in modality2_atoms:
            if atom_right == atom_left + 1:
                continue
            left_mask = 1 << atom_left
            right_mask = 1 << atom_right
            for index in range(2**num_atoms):
                if (index & left_mask) or (index & right_mask):
                    continue
                g00 = scores[index]
                g10 = scores[index | left_mask]
                g01 = scores[index | right_mask]
                g11 = scores[index | left_mask | right_mask]
                i1 = float(g10 - g00)
                i2 = float(g01 - g00)
                i12 = float(g11 - g00)
                redundancy = redundancy_from_rule(i1, i2, i12, rule_spec)
                unique_1 = i1 - redundancy
                unique_2 = i2 - redundancy
                synergy = i12 - i1 - i2 + redundancy
                outputs["R"].append(redundancy)
                outputs["U1"].append(unique_1)
                outputs["U2"].append(unique_2)
                outputs["S"].append(synergy)
    return {name: np.asarray(values, dtype=np.float64) for name, values in outputs.items()}


def redundancy_from_rule(i1: float, i2: float, i12: float, rule_spec: dict[str, object]) -> float:
    kind = str(rule_spec["kind"])
    if kind == "signed_min":
        positive_shared = min(max(i1, 0.0), max(i2, 0.0))
        negative_shared = min(max(-i1, 0.0), max(-i2, 0.0))
        return float(positive_shared - negative_shared)
    if kind == "positive_min":
        return float(min(max(i1, 0.0), max(i2, 0.0)))
    if kind == "raw_min":
        return float(min(i1, i2))
    if kind == "zero":
        return 0.0
    if kind == "coinfo_half":
        return float(0.5 * (i1 + i2 - i12))
    if kind == "completion_alpha":
        alpha = float(rule_spec["alpha"])
        lower = max(i1, i2)
        upper = min(max(i12, lower), i1 + i2)
        upper = max(upper, lower)
        completion = lower + alpha * (upper - lower)
        return float(i1 + i2 - completion)
    raise ValueError(f"unknown rule kind: {kind}")


def positive_dominant_component(means: dict[str, float]) -> str | None:
    positive_items = [(name, value) for name, value in means.items() if value > 1e-8]
    if not positive_items:
        return None
    positive_items.sort(key=lambda item: item[1], reverse=True)
    return positive_items[0][0]


def to_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Subset PID Combo Search",
        "",
        "Search over `scalar target + same-game decomposition rule` combinations.",
        "",
        "Selection criterion: positive dominant PID component should match",
        "- `redundant -> R`",
        "- `unique0 -> U1`",
        "- `unique1 -> U2`",
        "- `xor -> S`",
        "",
        "## Best Combos",
        "",
        "| target | rule | correct cases | redundant | unique0 | unique1 | xor |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for combo in payload["best_combos"]:
        case_results = combo["case_results"]
        lines.append(
            "| {target} | {rule} | {num_correct} | {r} | {u0} | {u1} | {x} |".format(
                target=combo["target"],
                rule=combo["rule"],
                num_correct=combo["num_correct"],
                r=case_results["redundant"]["dominant_positive"],
                u0=case_results["unique0"]["dominant_positive"],
                u1=case_results["unique1"]["dominant_positive"],
                x=case_results["xor"]["dominant_positive"],
            )
        )
    lines.extend(["", "## Details", ""])
    for combo in payload["best_combos"][:5]:
        lines.append(f"### {combo['target']} + {combo['rule']}")
        lines.append("")
        lines.append(f"- correct cases: `{combo['num_correct']}/4`")
        for logic_case, result in combo["case_results"].items():
            means = result["means"]
            lines.append(
                f"- {logic_case}: dominant=`{result['dominant_positive']}`, expected=`{result['expected']}`, "
                f"means=`R={means['R']:.4f}, U1={means['U1']:.4f}, U2={means['U2']:.4f}, S={means['S']:.4f}`"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
