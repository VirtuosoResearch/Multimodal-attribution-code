from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pid_estimator.non_interactive_completion import (
    NonInteractiveCompletionConfig,
    estimate_subset_mask_non_interactive_completion_pid,
)
from pid_estimator.synthetic_data import make_binary_logic_dataset


@dataclass(frozen=True)
class SubsetPIDSurrogateExperimentConfig:
    logic_case: str = "xor"
    train_fraction: float = 0.5
    split_seed: int = 0
    kernel_gamma: float | None = None
    ridge_lambda: float = 1e-3
    predictor_model: str = "kernel"
    additive_model: str = "kernel"
    predictor_epochs: int = 20
    additive_epochs: int = 20
    additive_feasibility_lambda: float = 0.0
    additive_rkhs_lambda: float = 0.0
    additive_anchor_lambda: float = 1.0
    batch_size: int = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate subset masks z, fit a kernel surrogate z->targets, and reconstruct PID.")
    parser.add_argument("--logic-cases", nargs="+", default=["redundant", "unique0", "unique1", "xor"])
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--kernel-gamma", type=float, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
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
        default="src/pid_estimator/results/subset_pid_surrogate_logic.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/pid_estimator/results/subset_pid_surrogate_logic.md",
    )
    args = parser.parse_args()

    output = {
        "experiment": "subset_pid_surrogate",
        "cases": [],
    }
    for logic_case in args.logic_cases:
        config = SubsetPIDSurrogateExperimentConfig(
            logic_case=logic_case,
            train_fraction=args.train_fraction,
            split_seed=args.split_seed,
            kernel_gamma=args.kernel_gamma,
            ridge_lambda=args.ridge_lambda,
            predictor_model=args.predictor_model,
            additive_model=args.additive_model,
            predictor_epochs=args.predictor_epochs,
            additive_epochs=args.additive_epochs,
            additive_feasibility_lambda=args.additive_feasibility_lambda,
            additive_rkhs_lambda=args.additive_rkhs_lambda,
            additive_anchor_lambda=args.additive_anchor_lambda,
            batch_size=args.batch_size,
        )
        output["cases"].append(run_logic_case_experiment(config))

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md.write_text(_to_markdown(output), encoding="utf-8")
    print(_to_markdown(output))


def run_logic_case_experiment(config: SubsetPIDSurrogateExperimentConfig) -> dict[str, object]:
    x1, x2, y, _dataset = make_binary_logic_dataset(config.logic_case, repeat=1)
    masks = _enumerate_masks(num_rows=x1.shape[0], num_modalities=2)
    targets = []
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
    for subset_mask in masks:
        result = estimate_subset_mask_non_interactive_completion_pid(
            x1,
            x2,
            y,
            subset_mask=subset_mask,
            num_labels=2,
            config=eval_config,
            device="cpu",
        )
        values = result.mean_values
        completion = result.mean_projected_completion
        targets.append(
            torch.tensor(
                [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                    float(completion),
                    float(result.mean_components[0]),
                    float(result.mean_components[1]),
                    float(result.mean_components[2]),
                    float(result.mean_components[3]),
                ],
                dtype=torch.float32,
            )
        )
    mask_inputs = torch.stack([mask.reshape(-1).float() for mask in masks], dim=0)
    targets_tensor = torch.stack(targets, dim=0)
    train_indices, test_indices = _train_test_split(mask_inputs.shape[0], config.train_fraction, config.split_seed)

    train_masks = mask_inputs[train_indices]
    test_masks = mask_inputs[test_indices]
    train_targets = targets_tensor[train_indices]
    test_targets = targets_tensor[test_indices]

    predicted_values = _kernel_ridge_predict(
        train_inputs=train_masks,
        train_targets=train_targets[:, :4],
        test_inputs=test_masks,
        gamma=config.kernel_gamma,
        ridge_lambda=config.ridge_lambda,
    )
    predicted_pid = _pid_from_value_targets(predicted_values)
    target_pid = test_targets[:, 4:]

    full_predicted_values = _kernel_ridge_predict(
        train_inputs=train_masks,
        train_targets=train_targets[:, :4],
        test_inputs=mask_inputs,
        gamma=config.kernel_gamma,
        ridge_lambda=config.ridge_lambda,
    )
    full_predicted_pid = _pid_from_value_targets(full_predicted_values)
    full_target_pid = targets_tensor[:, 4:]

    dominant_accuracy = (
        (predicted_pid.argmax(dim=1) == target_pid.argmax(dim=1)).float().mean().item() if len(test_indices) > 0 else 1.0
    )
    full_dominant_accuracy = (full_predicted_pid.argmax(dim=1) == full_target_pid.argmax(dim=1)).float().mean().item()
    return {
        "logic_case": config.logic_case,
        "config": asdict(config),
        "num_masks_total": int(mask_inputs.shape[0]),
        "num_masks_train": int(train_masks.shape[0]),
        "num_masks_test": int(test_masks.shape[0]),
        "value_target_rmse": _rmse_dict(predicted_values, test_targets[:, :4], keys=["V1", "V2", "V12", "C"]),
        "pid_rmse": _rmse_dict(predicted_pid, target_pid, keys=["R", "U1", "U2", "S"]),
        "pid_dominant_accuracy_test": float(dominant_accuracy),
        "pid_dominant_accuracy_all_masks": float(full_dominant_accuracy),
        "full_mask_prediction_mean_pid": _component_dict(full_predicted_pid.mean(dim=0)),
        "full_mask_target_mean_pid": _component_dict(full_target_pid.mean(dim=0)),
        "example_test_predictions": _build_examples(test_masks, predicted_values, test_targets[:, :4], predicted_pid, target_pid),
    }


def _enumerate_masks(num_rows: int, num_modalities: int) -> list[torch.Tensor]:
    masks = []
    total_bits = num_rows * num_modalities
    for index in range(2**total_bits):
        bits = [(index >> bit) & 1 for bit in range(total_bits)]
        mask = torch.tensor(bits, dtype=torch.bool).view(num_rows, num_modalities)
        masks.append(mask)
    return masks


def _train_test_split(num_samples: int, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    train_count = max(1, int(round(train_fraction * num_samples)))
    train_count = min(train_count, num_samples - 1)
    return permutation[:train_count], permutation[train_count:]


def _resolve_kernel_gamma(inputs: torch.Tensor, configured_gamma: float | None) -> float:
    if configured_gamma is not None:
        return float(configured_gamma)
    if inputs.shape[0] <= 1:
        return 1.0
    diffs = inputs[:, None, :] - inputs[None, :, :]
    sq_dists = (diffs * diffs).sum(dim=-1)
    positives = sq_dists[sq_dists > 0]
    if positives.numel() == 0:
        return 1.0
    return float(1.0 / positives.median().item())


def _rbf_kernel(lhs: torch.Tensor, rhs: torch.Tensor, gamma: float) -> torch.Tensor:
    lhs = lhs.float()
    rhs = rhs.float()
    lhs_sq = (lhs * lhs).sum(dim=1, keepdim=True)
    rhs_sq = (rhs * rhs).sum(dim=1, keepdim=True).transpose(0, 1)
    sq_dist = lhs_sq + rhs_sq - 2.0 * lhs @ rhs.transpose(0, 1)
    return torch.exp(-gamma * torch.clamp(sq_dist, min=0.0))


def _kernel_ridge_predict(
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    gamma: float | None,
    ridge_lambda: float,
) -> torch.Tensor:
    resolved_gamma = _resolve_kernel_gamma(train_inputs, gamma)
    train_kernel = _rbf_kernel(train_inputs, train_inputs, resolved_gamma)
    regularized = train_kernel + ridge_lambda * torch.eye(train_kernel.shape[0], dtype=train_kernel.dtype)
    dual_weights = torch.linalg.solve(regularized, train_targets.float())
    test_kernel = _rbf_kernel(test_inputs, train_inputs, resolved_gamma)
    return test_kernel @ dual_weights


def _pid_from_value_targets(value_targets: torch.Tensor) -> torch.Tensor:
    v1 = value_targets[:, 0]
    v2 = value_targets[:, 1]
    v12 = value_targets[:, 2]
    c = value_targets[:, 3]
    lower = torch.maximum(v1, v2)
    upper = torch.minimum(torch.maximum(v12, lower), v1 + v2)
    upper = torch.maximum(upper, lower)
    c = torch.minimum(torch.maximum(c, lower), upper)
    redundancy = torch.clamp(v1 + v2 - c, min=0.0)
    unique_1 = torch.clamp(c - v2, min=0.0)
    unique_2 = torch.clamp(c - v1, min=0.0)
    synergy = torch.clamp(v12 - c, min=0.0)
    return torch.stack([redundancy, unique_1, unique_2, synergy], dim=1)


def _rmse_dict(predictions: torch.Tensor, targets: torch.Tensor, keys: list[str]) -> dict[str, float]:
    mse = (predictions - targets).pow(2).mean(dim=0)
    return {key: float(torch.sqrt(mse[idx]).item()) for idx, key in enumerate(keys)}


def _component_dict(values: torch.Tensor) -> dict[str, float]:
    return {
        "R": float(values[0]),
        "U1": float(values[1]),
        "U2": float(values[2]),
        "S": float(values[3]),
    }


def _build_examples(
    test_masks: torch.Tensor,
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
    predicted_pid: torch.Tensor,
    target_pid: torch.Tensor,
    limit: int = 5,
) -> list[dict[str, object]]:
    examples = []
    for idx in range(min(limit, test_masks.shape[0])):
        examples.append(
            {
                "mask": [int(value) for value in test_masks[idx].tolist()],
                "predicted_values": {
                    "V1": float(predicted_values[idx, 0]),
                    "V2": float(predicted_values[idx, 1]),
                    "V12": float(predicted_values[idx, 2]),
                    "C": float(predicted_values[idx, 3]),
                },
                "target_values": {
                    "V1": float(target_values[idx, 0]),
                    "V2": float(target_values[idx, 1]),
                    "V12": float(target_values[idx, 2]),
                    "C": float(target_values[idx, 3]),
                },
                "predicted_pid": _component_dict(predicted_pid[idx]),
                "target_pid": _component_dict(target_pid[idx]),
            }
        )
    return examples


def _to_markdown(payload: dict[str, object]) -> str:
    lines = ["# Subset PID Surrogate Experiment", ""]
    for case in payload["cases"]:
        lines.append(f"## {case['logic_case']}")
        lines.append("")
        lines.append(f"- total masks: `{case['num_masks_total']}`")
        lines.append(f"- train masks: `{case['num_masks_train']}`")
        lines.append(f"- test masks: `{case['num_masks_test']}`")
        lines.append(f"- PID dominant accuracy on test masks: `{case['pid_dominant_accuracy_test']:.4f}`")
        lines.append(f"- PID dominant accuracy on all masks: `{case['pid_dominant_accuracy_all_masks']:.4f}`")
        lines.append("")
        lines.append("| target | RMSE |")
        lines.append("| --- | ---: |")
        for key, value in case["value_target_rmse"].items():
            lines.append(f"| {key} | {value:.4f} |")
        lines.append("")
        lines.append("| PID | RMSE |")
        lines.append("| --- | ---: |")
        for key, value in case["pid_rmse"].items():
            lines.append(f"| {key} | {value:.4f} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
