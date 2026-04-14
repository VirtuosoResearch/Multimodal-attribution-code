from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from pid_estimator.ce_alignment import CEAlignmentConfig, estimate_pid_with_ce_alignment
from pid_estimator.data import MultimodalDataset
from pid_estimator.lsmi import LSMIConfig, estimate_lsmi
from pid_estimator.surrogate import SurrogatePIDConfig, estimate_surrogate_pid
from pid_estimator.synthetic_data import SyntheticGenerationConfig, generate_synthetic_dataset
from kernel_estimator.kernel_surrogate import KernelSurrogateConfig, estimate_kernel_surrogate_pid


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark kernel surrogate vs baseline PID estimators on synthetic data.")
    parser.add_argument("--num-data", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=5)
    parser.add_argument("--transform-dim", type=int, default=16)
    parser.add_argument("--feature-sep", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    # PID CE alignment
    parser.add_argument("--pid-batch-size", type=int, default=64)
    parser.add_argument("--pid-discrim-epochs", type=int, default=5)
    parser.add_argument("--pid-ce-epochs", type=int, default=5)
    # LSMI
    parser.add_argument("--lsmi-batch-size", type=int, default=64)
    parser.add_argument("--lsmi-discriminator-epochs", type=int, default=10)
    parser.add_argument("--lsmi-entropy-epochs", type=int, default=10)
    # Surrogate
    parser.add_argument("--surrogate-batch-size", type=int, default=64)
    parser.add_argument("--surrogate-predictor-epochs", type=int, default=10)
    parser.add_argument("--surrogate-head-epochs", type=int, default=150)
    parser.add_argument("--surrogate-calibration-blocks", type=int, default=8)
    parser.add_argument("--surrogate-calibration-subsets", type=int, default=10)
    parser.add_argument("--surrogate-reference-lsmi-discriminator-epochs", type=int, default=5)
    parser.add_argument("--surrogate-reference-lsmi-entropy-epochs", type=int, default=5)
    # Kernel surrogate
    parser.add_argument("--kernel-num-subcomponents", type=int, default=4)
    parser.add_argument("--kernel-num-masks", type=int, default=128)
    parser.add_argument("--kernel-mask-prob", type=float, default=0.5)
    parser.add_argument("--kernel-ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--kernel-gamma", type=float, default=None)
    parser.add_argument("--kernel-classifier-epochs", type=int, default=15)

    parser.add_argument(
        "--settings",
        nargs="+",
        default=["redundancy", "uniqueness0", "uniqueness1", "synergy"],
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="src/kernel_estimator/results/kernel_benchmark.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/kernel_estimator/results/kernel_benchmark.md",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for offset, setting in enumerate(args.settings):
        print(f"\n{'='*60}")
        print(f"Setting: {setting}")
        print(f"{'='*60}")

        generation_config = SyntheticGenerationConfig(
            num_data=args.num_data,
            num_classes=args.num_classes,
            feature_dim=args.feature_dim,
            transform_dim=args.transform_dim,
            feature_sep=args.feature_sep,
            setting=setting,
            seed=args.seed + offset,
        )
        data = generate_synthetic_dataset(generation_config)
        x1_train, x2_train, y_train, train_dataset = _split_to_dataset(data["train"])
        x1_test, x2_test, y_test, test_dataset = _split_to_dataset(data["test"])

        print("  Running PID CE alignment...")
        pid_result = estimate_pid_with_ce_alignment(
            x1_train, x2_train, y_train,
            num_labels=args.num_classes,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=CEAlignmentConfig(
                batch_size=args.pid_batch_size,
                discrim_epochs=args.pid_discrim_epochs,
                ce_epochs=args.pid_ce_epochs,
                seed=args.seed + offset,
            ),
            device=device,
        )

        print("  Running LSMI...")
        lsmi_result = estimate_lsmi(
            x1_train, x2_train, y_train,
            num_labels=args.num_classes,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=LSMIConfig(
                batch_size=args.lsmi_batch_size,
                discriminator_epochs=args.lsmi_discriminator_epochs,
                entropy_epochs=args.lsmi_entropy_epochs,
                seed=args.seed + offset,
            ),
            device=device,
        )

        print("  Running Surrogate...")
        surrogate_result = estimate_surrogate_pid(
            x1_train, x2_train, y_train,
            num_labels=args.num_classes,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=SurrogatePIDConfig(
                batch_size=args.surrogate_batch_size,
                predictor_epochs=args.surrogate_predictor_epochs,
                head_epochs=args.surrogate_head_epochs,
                calibration_blocks=args.surrogate_calibration_blocks,
                calibration_subsets=args.surrogate_calibration_subsets,
                reference_lsmi_discriminator_epochs=args.surrogate_reference_lsmi_discriminator_epochs,
                reference_lsmi_entropy_epochs=args.surrogate_reference_lsmi_entropy_epochs,
                seed=args.seed + offset,
            ),
            device=device,
        )

        print("  Running Kernel Surrogate...")
        kernel_result = estimate_kernel_surrogate_pid(
            x1_train, x2_train, y_train,
            num_labels=args.num_classes,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=KernelSurrogateConfig(
                num_subcomponents_per_modality=args.kernel_num_subcomponents,
                num_masks=args.kernel_num_masks,
                mask_prob=args.kernel_mask_prob,
                ridge_lambda=args.kernel_ridge_lambda,
                kernel_gamma=args.kernel_gamma,
                classifier_epochs=args.kernel_classifier_epochs,
                seed=args.seed + offset,
                verbose=False,
            ),
            device=device,
        )

        results.append({
            "setting": setting,
            "generation_config": asdict(generation_config),
            "pid_ce_alignment": _components_to_dict(pid_result.mean_components),
            "lsmi": _components_to_dict(lsmi_result.mean_components),
            "surrogate": _components_to_dict(surrogate_result.mean_components),
            "kernel_surrogate": _components_to_dict(kernel_result.mean_components),
        })

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps({"device": device, "results": results}, indent=2), encoding="utf-8")
    md_text = _to_markdown(device=device, results=results)
    output_md.write_text(md_text, encoding="utf-8")

    print(md_text)
    print(f"\nSaved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")


def _split_to_dataset(split: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MultimodalDataset]:
    x1 = torch.as_tensor(split["0"]).float()
    x2 = torch.as_tensor(split["1"]).float()
    y = torch.as_tensor(split["label"]).long()
    dataset = MultimodalDataset([x1, x2], y)
    return x1, x2, y, dataset


def _components_to_dict(components: torch.Tensor) -> dict[str, float]:
    values = components.detach().cpu().tolist()
    return {
        "redundancy": float(values[0]),
        "unique_1": float(values[1]),
        "unique_2": float(values[2]),
        "synergy": float(values[3]),
    }


def _to_markdown(*, device: str, results: list[dict]) -> str:
    lines = [
        "# Kernel Surrogate vs Baseline PID Estimators — Synthetic Benchmark",
        "",
        f"- device: `{device}`",
        "",
        "| setting | estimator | redundancy | unique_1 | unique_2 | synergy | dominant |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    estimator_keys = [
        ("pid_ce_alignment", "pid"),
        ("lsmi", "lsmi"),
        ("surrogate", "surrogate"),
        ("kernel_surrogate", "kernel"),
    ]
    for row in results:
        for key, label in estimator_keys:
            values = row[key]
            dominant = max(values.items(), key=lambda item: item[1])[0]
            lines.append(
                f"| {row['setting']} | {label} | "
                f"{values['redundancy']:.4f} | {values['unique_1']:.4f} | "
                f"{values['unique_2']:.4f} | {values['synergy']:.4f} | {dominant} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
