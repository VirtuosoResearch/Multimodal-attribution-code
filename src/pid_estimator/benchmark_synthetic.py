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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PID, LSMI, and surrogate estimators on synthetic data.")
    parser.add_argument("--num-data", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=5)
    parser.add_argument("--transform-dim", type=int, default=16)
    parser.add_argument("--feature-sep", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pid-batch-size", type=int, default=64)
    parser.add_argument("--pid-discrim-epochs", type=int, default=5)
    parser.add_argument("--pid-ce-epochs", type=int, default=5)
    parser.add_argument("--lsmi-batch-size", type=int, default=64)
    parser.add_argument("--lsmi-discriminator-epochs", type=int, default=10)
    parser.add_argument("--lsmi-entropy-epochs", type=int, default=10)
    parser.add_argument("--surrogate-batch-size", type=int, default=64)
    parser.add_argument("--surrogate-predictor-epochs", type=int, default=10)
    parser.add_argument("--surrogate-head-epochs", type=int, default=150)
    parser.add_argument("--surrogate-calibration-blocks", type=int, default=8)
    parser.add_argument("--surrogate-calibration-subsets", type=int, default=10)
    parser.add_argument("--surrogate-reference-lsmi-discriminator-epochs", type=int, default=5)
    parser.add_argument("--surrogate-reference-lsmi-entropy-epochs", type=int, default=5)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["redundancy", "uniqueness0", "uniqueness1", "synergy"],
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="src/pid_estimator/results/synthetic_estimator_benchmark.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/pid_estimator/results/synthetic_estimator_benchmark.md",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for offset, setting in enumerate(args.settings):
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

        pid_result = estimate_pid_with_ce_alignment(
            x1_train,
            x2_train,
            y_train,
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
        lsmi_result = estimate_lsmi(
            x1_train,
            x2_train,
            y_train,
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
        surrogate_result = estimate_surrogate_pid(
            x1_train,
            x2_train,
            y_train,
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

        results.append(
            {
                "setting": setting,
                "generation_config": asdict(generation_config),
                "train_shape": {
                    "x1": list(x1_train.shape),
                    "x2": list(x2_train.shape),
                    "y": list(y_train.shape),
                },
                "test_shape": {
                    "x1": list(x1_test.shape),
                    "x2": list(x2_test.shape),
                    "y": list(y_test.shape),
                },
                "pid_ce_alignment": _components_to_dict(pid_result.mean_components),
                "lsmi": _components_to_dict(lsmi_result.mean_components),
                "surrogate": _components_to_dict(surrogate_result.mean_components),
            }
        )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"device": device, "results": results}, indent=2), encoding="utf-8")
    output_md.write_text(_to_markdown(device=device, results=results), encoding="utf-8")

    print(_to_markdown(device=device, results=results))
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
        "# Synthetic Estimator Benchmark",
        "",
        f"- device: `{device}`",
        "",
        "| setting | estimator | redundancy | unique_1 | unique_2 | synergy | dominant |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        for estimator_key, label in [("pid_ce_alignment", "pid"), ("lsmi", "lsmi"), ("surrogate", "surrogate")]:
            values = row[estimator_key]
            dominant = _dominant_component(values)
            lines.append(
                f"| {row['setting']} | {label} | "
                f"{values['redundancy']:.4f} | {values['unique_1']:.4f} | "
                f"{values['unique_2']:.4f} | {values['synergy']:.4f} | {dominant} |"
            )
    return "\n".join(lines)


def _dominant_component(values: dict[str, float]) -> str:
    return max(values.items(), key=lambda item: item[1])[0]


if __name__ == "__main__":
    main()
