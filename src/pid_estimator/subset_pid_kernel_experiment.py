from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pid_estimator.non_interactive_completion import (
    NonInteractiveCompletionConfig,
    estimate_subset_mask_non_interactive_completion_pid,
)
from pid_estimator.subset_attribution import (
    RBFSubsetSurrogateConfig,
    aggregate_cross_modal_pid_masses,
    build_cross_modal_candidate_edges,
    estimate_cross_modal_pid_from_scores,
    estimate_cross_modal_pid_from_surrogate,
    fit_rbf_subset_surrogate,
)
from pid_estimator.subset_attribution_experiment import pearson_corr, r2_score, rmse, spearman_corr
from pid_estimator.synthetic_data import make_binary_logic_dataset


@dataclass(frozen=True)
class SubsetPIDKernelExperimentConfig:
    logic_cases: tuple[str, ...] = ("redundant", "unique0", "unique1", "xor")
    train_fraction: float = 0.5
    split_seed: int = 0
    kernel_gamma: float | None = None
    ridge_lambda: float = 1e-3
    score_target: str = "v12"
    blend_alpha: float = 0.4
    repeat: int = 1
    predictor_model: str = "kernel"
    additive_model: str = "kernel"
    predictor_epochs: int = 20
    additive_epochs: int = 20
    additive_feasibility_lambda: float = 0.0
    additive_rkhs_lambda: float = 0.0
    additive_anchor_lambda: float = 1.0
    batch_size: int = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a single Hamming-RBF subset-score surrogate g(s) and evaluate PID-compatible coalition decomposition."
    )
    parser.add_argument("--logic-cases", nargs="+", default=["redundant", "unique0", "unique1", "xor"])
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--kernel-gamma", type=float, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--score-target",
        type=str,
        default="v12",
        choices=[
            "v1",
            "v2",
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
            "redundancy",
            "unique1",
            "unique2",
            "synergy",
            "synergy_minus_redundancy",
            "blend_v12_completion",
        ],
    )
    parser.add_argument("--blend-alpha", type=float, default=0.4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--predictor-model", type=str, default="kernel")
    parser.add_argument("--additive-model", type=str, default="kernel")
    parser.add_argument("--predictor-epochs", type=int, default=20)
    parser.add_argument("--additive-epochs", type=int, default=20)
    parser.add_argument("--additive-feasibility-lambda", type=float, default=0.0)
    parser.add_argument("--additive-rkhs-lambda", type=float, default=0.0)
    parser.add_argument("--additive-anchor-lambda", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output-json",
        type=str,
        default="src/pid_estimator/results/subset_pid_kernel_experiment.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/pid_estimator/results/subset_pid_kernel_experiment.md",
    )
    args = parser.parse_args()

    config = SubsetPIDKernelExperimentConfig(
        logic_cases=tuple(args.logic_cases),
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        kernel_gamma=args.kernel_gamma,
        ridge_lambda=args.ridge_lambda,
        score_target=args.score_target,
        blend_alpha=args.blend_alpha,
        repeat=args.repeat,
        predictor_model=args.predictor_model,
        additive_model=args.additive_model,
        predictor_epochs=args.predictor_epochs,
        additive_epochs=args.additive_epochs,
        additive_feasibility_lambda=args.additive_feasibility_lambda,
        additive_rkhs_lambda=args.additive_rkhs_lambda,
        additive_anchor_lambda=args.additive_anchor_lambda,
        batch_size=args.batch_size,
    )

    payload = {
        "experiment": "subset_pid_kernel_same_game",
        "config": asdict(config),
        "cases": [run_logic_case_experiment(config, logic_case=logic_case) for logic_case in config.logic_cases],
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload))


def run_logic_case_experiment(
    config: SubsetPIDKernelExperimentConfig,
    *,
    logic_case: str,
) -> dict[str, object]:
    train_x1, train_x2, train_labels, _dataset = make_binary_logic_dataset(logic_case, repeat=config.repeat)
    num_rows = int(train_x1.shape[0])
    atom_samples = np.repeat(np.arange(num_rows, dtype=np.int64), 2)
    atom_modalities = np.tile(np.arange(2, dtype=np.int64), num_rows)
    candidate_edges = build_cross_modal_candidate_edges(atom_modalities, atom_samples)

    subset_masks = enumerate_masks(num_rows=num_rows, num_modalities=2)
    oracle_scores = []
    eval_config = NonInteractiveCompletionConfig(
        predictor_model=config.predictor_model,
        additive_model=config.additive_model,
        predictor_epochs=config.predictor_epochs,
        additive_epochs=config.additive_epochs,
        kernel_gamma=config.kernel_gamma,
        kernel_ridge_lambda=config.ridge_lambda,
        additive_feasibility_lambda=config.additive_feasibility_lambda,
        additive_rkhs_lambda=config.additive_rkhs_lambda,
        additive_anchor_lambda=config.additive_anchor_lambda,
        batch_size=config.batch_size,
        seed=config.split_seed,
    )
    for subset_mask in subset_masks:
        result = estimate_subset_mask_non_interactive_completion_pid(
            train_x1,
            train_x2,
            train_labels,
            subset_mask=subset_mask,
            num_labels=2,
            config=eval_config,
            device="cpu",
        )
        oracle_scores.append(
            float(
                select_score_target(
                    result,
                    target_name=config.score_target,
                    blend_alpha=config.blend_alpha,
                    eval_x1=train_x1,
                    eval_x2=train_x2,
                    eval_labels=train_labels,
                )
            )
        )

    subset_mask_array = np.stack([mask.reshape(-1).cpu().numpy().astype(bool) for mask in subset_masks], axis=0)
    oracle_score_array = np.asarray(oracle_scores, dtype=np.float64)
    train_indices, test_indices = train_test_split(subset_mask_array.shape[0], config.train_fraction, config.split_seed)

    model = fit_rbf_subset_surrogate(
        subset_mask_array[train_indices],
        oracle_score_array[train_indices],
        num_atoms=subset_mask_array.shape[1],
        candidate_edges=candidate_edges,
        config=RBFSubsetSurrogateConfig(
            ridge_lambda=config.ridge_lambda,
            gamma=config.kernel_gamma,
        ),
    )

    predicted_test = np.asarray(model.predict_batch(subset_mask_array[test_indices]), dtype=np.float64)
    predicted_all = np.asarray(model.predict_batch(subset_mask_array), dtype=np.float64)

    true_pid_by_pair = estimate_cross_modal_pid_from_scores(
        oracle_score_array,
        num_atoms=subset_mask_array.shape[1],
        atom_samples=atom_samples,
        atom_modalities=atom_modalities,
        modality_pair=(0, 1),
    )
    estimated_pid_by_pair = estimate_cross_modal_pid_from_surrogate(
        model,
        num_atoms=subset_mask_array.shape[1],
        atom_samples=atom_samples,
        atom_modalities=atom_modalities,
        modality_pair=(0, 1),
        exhaustive=True,
    )

    true_pair_array, estimated_pair_array = pair_component_arrays(true_pid_by_pair, estimated_pid_by_pair)
    masses_true = aggregate_cross_modal_pid_masses(true_pid_by_pair)
    masses_estimated = aggregate_cross_modal_pid_masses(estimated_pid_by_pair)
    true_component_means = {
        key: float(np.mean(true_pair_array[key])) for key in ["R", "U1", "U2", "S"]
    }
    estimated_component_means = {
        key: float(np.mean(estimated_pair_array[key])) for key in ["R", "U1", "U2", "S"]
    }

    return {
        "logic_case": logic_case,
        "score_target": config.score_target,
        "num_atoms": int(subset_mask_array.shape[1]),
        "num_masks_total": int(subset_mask_array.shape[0]),
        "num_masks_train": int(train_indices.shape[0]),
        "num_masks_test": int(test_indices.shape[0]),
        "candidate_edges": [list(edge) for edge in candidate_edges],
        "subset_metrics": {
            "lds_spearman_test": spearman_corr(oracle_score_array[test_indices], predicted_test),
            "lds_spearman_all": spearman_corr(oracle_score_array, predicted_all),
            "subset_r2_test": r2_score(oracle_score_array[test_indices], predicted_test),
            "subset_rmse_test": rmse(oracle_score_array[test_indices], predicted_test),
        },
        "pid_component_metrics": {
            name: {
                "rmse": rmse(true_pair_array[name], estimated_pair_array[name]),
                "pearson": pearson_corr(true_pair_array[name], estimated_pair_array[name]),
                "spearman": spearman_corr(true_pair_array[name], estimated_pair_array[name]),
            }
            for name in ["R", "U1", "U2", "S"]
        },
        "component_means": {
            "true": true_component_means,
            "estimated": estimated_component_means,
            "true_dominant": dominant_component(true_component_means),
            "estimated_dominant": dominant_component(estimated_component_means),
        },
        "cross_sample_cross_modal_mass": {
            "true": masses_true,
            "estimated": masses_estimated,
        },
        "pair_components_true": {
            f"{left}->{right}": {key: float(value) for key, value in components.items()}
            for (left, right), components in true_pid_by_pair.items()
        },
        "pair_components_estimated": {
            f"{left}->{right}": {key: float(value) for key, value in components.items()}
            for (left, right), components in estimated_pid_by_pair.items()
        },
        "oracle_score_examples": build_score_examples(subset_mask_array, oracle_score_array, predicted_all),
    }


def select_score_target(
    result,
    *,
    target_name: str,
    blend_alpha: float = 0.4,
    eval_x1: torch.Tensor | None = None,
    eval_x2: torch.Tensor | None = None,
    eval_labels: torch.Tensor | None = None,
) -> float:
    values = result.mean_values
    components = result.mean_components
    if target_name == "v1":
        return float(values[0])
    if target_name == "v2":
        return float(values[1])
    if target_name == "v12":
        return float(values[2])
    if target_name == "completion":
        return float(result.mean_projected_completion)
    if target_name == "joint_minus_completion":
        return float(values[2] - result.mean_projected_completion)
    if target_name == "joint_minus_best":
        return float(values[2] - max(float(values[0]), float(values[1])))
    if target_name == "completion_minus_best":
        return float(float(result.mean_projected_completion) - max(float(values[0]), float(values[1])))
    if target_name == "min_single":
        return float(min(float(values[0]), float(values[1])))
    if target_name == "max_single":
        return float(max(float(values[0]), float(values[1])))
    if target_name == "mean_single":
        return float(0.5 * (float(values[0]) + float(values[1])))
    if target_name == "sum_single":
        return float(float(values[0]) + float(values[1]))
    if target_name == "joint_minus_mean_single":
        return float(float(values[2]) - 0.5 * (float(values[0]) + float(values[1])))
    if target_name == "completion_minus_mean_single":
        return float(float(result.mean_projected_completion) - 0.5 * (float(values[0]) + float(values[1])))
    if target_name == "joint_minus_sum_single":
        return float(float(values[2]) - float(values[0]) - float(values[1]))
    if target_name == "completion_minus_sum_single":
        return float(float(result.mean_projected_completion) - float(values[0]) - float(values[1]))
    if target_name == "joint_accuracy":
        accuracy_metrics = compute_accuracy_targets(result, eval_x1=eval_x1, eval_x2=eval_x2, eval_labels=eval_labels)
        return float(accuracy_metrics["joint_accuracy"])
    if target_name == "joint_disagreement_accuracy":
        accuracy_metrics = compute_accuracy_targets(result, eval_x1=eval_x1, eval_x2=eval_x2, eval_labels=eval_labels)
        return float(accuracy_metrics["joint_disagreement_accuracy"])
    if target_name == "joint_rescue_rate":
        accuracy_metrics = compute_accuracy_targets(result, eval_x1=eval_x1, eval_x2=eval_x2, eval_labels=eval_labels)
        return float(accuracy_metrics["joint_rescue_rate"])
    if target_name == "redundancy":
        return float(components[0])
    if target_name == "unique1":
        return float(components[1])
    if target_name == "unique2":
        return float(components[2])
    if target_name == "synergy":
        return float(components[3])
    if target_name == "synergy_minus_redundancy":
        return float(components[3] - components[0])
    if target_name == "blend_v12_completion":
        alpha = float(blend_alpha)
        return float((1.0 - alpha) * values[2] + alpha * result.mean_projected_completion)
    raise ValueError(f"unsupported score target: {target_name}")


def enumerate_masks(num_rows: int, num_modalities: int) -> list[torch.Tensor]:
    masks = []
    total_bits = num_rows * num_modalities
    for index in range(2**total_bits):
        bits = [(index >> bit) & 1 for bit in range(total_bits)]
        masks.append(torch.tensor(bits, dtype=torch.bool).view(num_rows, num_modalities))
    return masks


def train_test_split(num_samples: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_samples)
    train_count = max(1, int(round(train_fraction * num_samples)))
    train_count = min(train_count, num_samples - 1)
    return permutation[:train_count], permutation[train_count:]


def pair_component_arrays(
    true_pid_by_pair: dict[tuple[int, int], dict[str, float]],
    estimated_pid_by_pair: dict[tuple[int, int], dict[str, float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pair_keys = sorted(true_pid_by_pair)
    true_arrays = {
        name: np.asarray([true_pid_by_pair[key][name] for key in pair_keys], dtype=np.float64)
        for name in ["R", "U1", "U2", "S"]
    }
    estimated_arrays = {
        name: np.asarray([estimated_pid_by_pair[key][name] for key in pair_keys], dtype=np.float64)
        for name in ["R", "U1", "U2", "S"]
    }
    return true_arrays, estimated_arrays


def dominant_component(values: dict[str, float]) -> str:
    positive_items = [(name, value) for name, value in values.items() if value > 1e-8]
    if not positive_items:
        ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
        return ordered[0][0]
    positive_items.sort(key=lambda item: item[1], reverse=True)
    return positive_items[0][0]


def build_score_examples(
    masks: np.ndarray,
    oracle_scores: np.ndarray,
    predicted_scores: np.ndarray,
    limit: int = 5,
) -> list[dict[str, object]]:
    examples = []
    for index in range(min(limit, masks.shape[0])):
        examples.append(
            {
                "mask": [int(value) for value in masks[index].astype(int).tolist()],
                "oracle_score": float(oracle_scores[index]),
                "predicted_score": float(predicted_scores[index]),
            }
        )
    return examples


def compute_accuracy_targets(
    result,
    *,
    eval_x1: torch.Tensor | None,
    eval_x2: torch.Tensor | None,
    eval_labels: torch.Tensor | None,
) -> dict[str, float]:
    if eval_x1 is None or eval_x2 is None or eval_labels is None:
        raise ValueError("accuracy-based targets require eval tensors")
    x1 = torch.as_tensor(eval_x1).float()
    x2 = torch.as_tensor(eval_x2).float()
    labels = torch.as_tensor(eval_labels).long().view(-1)
    prior = torch.as_tensor(result.label_prior).float().view(-1)
    majority_class = int(torch.argmax(prior).item())
    baseline_pred = torch.full_like(labels, fill_value=majority_class)

    pred1 = _predict_or_baseline(
        result.predictor_1,
        inputs=x1,
        fallback=baseline_pred,
        available=bool(result.available_counts and result.available_counts.get("modality_1", 0) > 0),
    )
    pred2 = _predict_or_baseline(
        result.predictor_2,
        inputs=x2,
        fallback=baseline_pred,
        available=bool(result.available_counts and result.available_counts.get("modality_2", 0) > 0),
    )
    joint_inputs = torch.cat([x1, x2], dim=1)
    pred12 = _predict_or_baseline(
        result.predictor_12,
        inputs=joint_inputs,
        fallback=baseline_pred,
        available=bool(result.available_counts and result.available_counts.get("any", 0) > 0),
    )

    joint_accuracy = float((pred12 == labels).float().mean().item())
    disagreement_mask = pred1 != pred2
    if bool(disagreement_mask.any()):
        joint_disagreement_accuracy = float((pred12[disagreement_mask] == labels[disagreement_mask]).float().mean().item())
    else:
        joint_disagreement_accuracy = 0.0
    rescue_mask = (pred12 == labels) & ((pred1 != labels) | (pred2 != labels))
    joint_rescue_rate = float(rescue_mask.float().mean().item())
    return {
        "joint_accuracy": joint_accuracy,
        "joint_disagreement_accuracy": joint_disagreement_accuracy,
        "joint_rescue_rate": joint_rescue_rate,
    }


def _predict_or_baseline(
    predictor: nn.Module,
    *,
    inputs: torch.Tensor,
    fallback: torch.Tensor,
    available: bool,
) -> torch.Tensor:
    if not available or predictor.__class__.__name__ == "_NullModule":
        return fallback.clone()
    predictor.eval()
    with torch.no_grad():
        logits = predictor(inputs)
    return torch.argmax(logits, dim=1).detach().cpu()


def to_markdown(payload: dict[str, object]) -> str:
    score_target = payload.get("config", {}).get("score_target", "v12")
    lines = [
        "# Subset PID Kernel Experiment",
        "",
        "This report follows `pid_surrogate_estimator.md`: learn one scalar subset score `g(s)`, then compute PID-compatible coalition gains on the same game.",
        "",
        "## Algorithm",
        "",
        "1. Enumerate subset masks `s in {0,1}^{2N}` over modality-structured sample-modality bits.",
        f"2. Use the subset oracle to compute one scalar outcome `F(s)` for each mask. In this run the scalar target is `{score_target}`.",
        "3. Fit a Hamming-RBF kernel ridge surrogate `g(s) ~= F(s)` on a 50/50 train/test split of masks.",
        "4. For each cross-sample pair `(i,j)` with modality-1 atom from sample `i` and modality-2 atom from sample `j`, compute `g00, g10, g01, g11` on the same learned game.",
        "5. Convert coalition gains into PID-compatible components with the signed-min redundancy rule:",
        "   `R = min(I1+, I2+) - min(I1-, I2-)`, `U1 = I1 - R`, `U2 = I2 - R`, `S = I12 - I1 - I2 + R`.",
        "6. Average these quantities over all backgrounds and aggregate cross-sample cross-modal redundancy/synergy masses.",
        "",
    ]
    for case in payload["cases"]:
        subset_metrics = case["subset_metrics"]
        pid_metrics = case["pid_component_metrics"]
        component_means = case["component_means"]
        masses_true = case["cross_sample_cross_modal_mass"]["true"]
        masses_estimated = case["cross_sample_cross_modal_mass"]["estimated"]
        lines.extend(
            [
                f"## {case['logic_case']}",
                "",
                f"- score target: `{case['score_target']}`",
                f"- total masks: `{case['num_masks_total']}`",
                f"- train masks: `{case['num_masks_train']}`",
                f"- test masks: `{case['num_masks_test']}`",
                "",
                "### Subset Outcome Estimation",
                f"- LDS Spearman (test): `{subset_metrics['lds_spearman_test']:.4f}`",
                f"- LDS Spearman (all): `{subset_metrics['lds_spearman_all']:.4f}`",
                f"- R2 (test): `{subset_metrics['subset_r2_test']:.4f}`",
                f"- RMSE (test): `{subset_metrics['subset_rmse_test']:.4f}`",
                "",
                "### PID-Compatible Recovery on the Same Game",
                "| component | RMSE | Spearman |",
                "| --- | ---: | ---: |",
            ]
        )
        for name in ["R", "U1", "U2", "S"]:
            lines.append(
                f"| {name} | {pid_metrics[name]['rmse']:.4f} | {pid_metrics[name]['spearman']:.4f} |"
            )
        lines.extend(
            [
                "",
                "### Mean Cross-Sample Cross-Modal PID Components",
                f"- True means: `R={component_means['true']['R']:.4f}`, `U1={component_means['true']['U1']:.4f}`, `U2={component_means['true']['U2']:.4f}`, `S={component_means['true']['S']:.4f}`",
                f"- Estimated means: `R={component_means['estimated']['R']:.4f}`, `U1={component_means['estimated']['U1']:.4f}`, `U2={component_means['estimated']['U2']:.4f}`, `S={component_means['estimated']['S']:.4f}`",
                f"- True dominant component: `{component_means['true_dominant']}`",
                f"- Estimated dominant component: `{component_means['estimated_dominant']}`",
                "",
                "### Cross-Sample Cross-Modal Mass",
                f"- True positive redundancy mass: `{masses_true['positive_redundancy_mass']:.4f}`",
                f"- Estimated positive redundancy mass: `{masses_estimated['positive_redundancy_mass']:.4f}`",
                f"- True positive synergy mass: `{masses_true['positive_synergy_mass']:.4f}`",
                f"- Estimated positive synergy mass: `{masses_estimated['positive_synergy_mass']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This experiment no longer supervises PID or value tuples directly. The only learned object is the scalar subset surrogate `g(s)`.",
            "- All redundancy/unique/synergy numbers are computed by applying one PID-compatible coalition decomposition rule to either the oracle game `F` or the learned game `g`.",
            "- This makes the evaluation aligned with the first-principles requirement: `mask -> score` is primary, PID is derived from that same score function rather than predicted separately.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
