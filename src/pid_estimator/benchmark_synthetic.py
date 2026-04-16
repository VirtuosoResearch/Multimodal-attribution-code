from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from pid_estimator.ce_alignment import CEAlignmentConfig, estimate_pid_with_ce_alignment
from pid_estimator.data import MultimodalDataset
from pid_estimator.direct_kernel import DirectKernelPIDConfig, estimate_direct_kernel_pid
from pid_estimator.lsmi import LSMIConfig, estimate_lsmi
from pid_estimator.surrogate import SurrogatePIDConfig, estimate_surrogate_pid
from pid_estimator.synthetic_data import (
    SyntheticGenerationConfig,
    generate_synthetic_dataset,
    make_binary_logic_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PID estimators on continuous synthetic data and binary logic data.")
    parser.add_argument("--num-data", type=int, default=2048)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=5)
    parser.add_argument("--transform-dim", type=int, default=16)
    parser.add_argument("--feature-sep", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--noise-degree", type=float, default=0.5)
    parser.add_argument("--logic-repeat", type=int, default=2048)
    parser.add_argument("--logic-cases", nargs="+", default=["redundant", "unique0", "unique1", "xor"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pid-batch-size", type=int, default=64)
    parser.add_argument("--pid-discrim-epochs", type=int, default=5)
    parser.add_argument("--pid-ce-epochs", type=int, default=5)
    parser.add_argument("--lsmi-batch-size", type=int, default=64)
    parser.add_argument("--lsmi-discriminator-epochs", type=int, default=20)
    parser.add_argument("--lsmi-entropy-epochs", type=int, default=20)
    parser.add_argument("--surrogate-batch-size", type=int, default=64)
    parser.add_argument("--surrogate-predictor-epochs", type=int, default=10)
    parser.add_argument("--surrogate-head-epochs", type=int, default=150)
    parser.add_argument("--surrogate-kernel-head-gamma", type=float, default=None)
    parser.add_argument("--surrogate-calibration-blocks", type=int, default=8)
    parser.add_argument("--surrogate-calibration-subsets", type=int, default=10)
    parser.add_argument("--surrogate-reference-lsmi-discriminator-epochs", type=int, default=20)
    parser.add_argument("--surrogate-reference-lsmi-entropy-epochs", type=int, default=20)
    parser.add_argument("--direct-kernel-family", type=str, default="rbf")
    parser.add_argument("--direct-kernel-gamma", type=float, default=None)
    parser.add_argument("--direct-kernel-ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--direct-kernel-refinement-epochs", type=int, default=100)
    parser.add_argument("--direct-kernel-refinement-lr", type=float, default=5e-2)
    parser.add_argument("--direct-kernel-lambda-anchor", type=float, default=0.0)
    parser.add_argument("--direct-kernel-lambda-info", type=float, default=0.0)
    parser.add_argument("--direct-kernel-lambda-redundancy", type=float, default=2.5)
    parser.add_argument("--direct-kernel-lambda-interval", type=float, default=0.0)
    parser.add_argument("--direct-kernel-lambda-rkhs", type=float, default=0.0)
    parser.add_argument(
        "--lsmi-cache-json",
        type=str,
        default="src/pid_estimator/results/lsmi_benchmark_cache.json",
    )
    parser.add_argument("--refresh-lsmi-cache", action="store_true")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["official"],
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
    continuous_results = []
    logic_results = []
    lsmi_cache_path = Path(args.lsmi_cache_json)
    lsmi_cache = _load_json(lsmi_cache_path)

    for offset, setting in enumerate(args.settings):
        generation_config = SyntheticGenerationConfig(
            num_data=args.num_data,
            num_classes=args.num_classes,
            feature_dim=args.feature_dim,
            transform_dim=args.transform_dim,
            feature_sep=args.feature_sep,
            rho=args.rho,
            noise_degree=args.noise_degree,
            setting=setting,
            seed=args.seed + offset,
        )
        data = generate_synthetic_dataset(generation_config)
        x1_train, x2_train, y_train, train_dataset = _split_to_dataset(data["train"])
        x1_test, x2_test, y_test, test_dataset = _split_to_dataset(data["test"])
        estimators = _run_estimators(
            x1_train=x1_train,
            x2_train=x2_train,
            y_train=y_train,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_labels=args.num_classes,
            args=args,
            seed=args.seed + offset,
            device=device,
            benchmark_key=_make_benchmark_key(
                kind="continuous",
                name=setting,
                seed=args.seed + offset,
                num_labels=args.num_classes,
                metadata=asdict(generation_config),
            ),
            lsmi_cache=lsmi_cache,
        )
        continuous_results.append(
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
                **estimators,
            }
        )

    for offset, case in enumerate(args.logic_cases, start=len(args.settings)):
        x1, x2, y, dataset = make_binary_logic_dataset(case, repeat=args.logic_repeat)
        x1_train, x2_train, y_train, train_dataset, x1_test, x2_test, y_test, test_dataset = _split_logic_dataset(
            x1, x2, y, seed=args.seed + offset
        )
        estimators = _run_estimators(
            x1_train=x1_train,
            x2_train=x2_train,
            y_train=y_train,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_labels=2,
            args=args,
            seed=args.seed + offset,
            device=device,
            benchmark_key=_make_benchmark_key(
                kind="logic",
                name=case,
                seed=args.seed + offset,
                num_labels=2,
                metadata={"repeat": args.logic_repeat},
            ),
            lsmi_cache=lsmi_cache,
        )
        logic_results.append(
            {
                "case": case,
                "repeat": args.logic_repeat,
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
                **estimators,
            }
        )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lsmi_cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": device,
        "continuous_results": continuous_results,
        "logic_results": logic_results,
        "lsmi_budget": {
            "num_data": args.num_data,
            "logic_repeat": args.logic_repeat,
            "lsmi_discriminator_epochs": args.lsmi_discriminator_epochs,
            "lsmi_entropy_epochs": args.lsmi_entropy_epochs,
        },
        "lsmi_cache_json": str(lsmi_cache_path),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(_to_markdown(device=device, payload=payload), encoding="utf-8")
    lsmi_cache_path.write_text(json.dumps(lsmi_cache, indent=2), encoding="utf-8")

    print(_to_markdown(device=device, payload=payload))
    print(f"\nSaved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")


def _run_estimators(
    *,
    x1_train: torch.Tensor,
    x2_train: torch.Tensor,
    y_train: torch.Tensor,
    train_dataset: MultimodalDataset,
    test_dataset: MultimodalDataset,
    num_labels: int,
    args: argparse.Namespace,
    seed: int,
    device: str,
    benchmark_key: str,
    lsmi_cache: dict[str, Any],
) -> dict[str, dict[str, float]]:
    pid_result = estimate_pid_with_ce_alignment(
        x1_train,
        x2_train,
        y_train,
        num_labels=num_labels,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=CEAlignmentConfig(
            batch_size=args.pid_batch_size,
            discrim_epochs=args.pid_discrim_epochs,
            ce_epochs=args.pid_ce_epochs,
            seed=seed,
        ),
        device=device,
    )
    lsmi_result_dict = _resolve_lsmi_baseline(
        benchmark_key=benchmark_key,
        x1_train=x1_train,
        x2_train=x2_train,
        y_train=y_train,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_labels=num_labels,
        args=args,
        seed=seed,
        device=device,
        lsmi_cache=lsmi_cache,
    )
    surrogate_result = estimate_surrogate_pid(
        x1_train,
        x2_train,
        y_train,
        num_labels=num_labels,
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
            seed=seed,
        ),
        device=device,
    )
    surrogate_mobius_result = estimate_surrogate_pid(
        x1_train,
        x2_train,
        y_train,
        num_labels=num_labels,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=SurrogatePIDConfig(
            batch_size=args.surrogate_batch_size,
            predictor_epochs=args.surrogate_predictor_epochs,
            head_type="mobius",
            head_epochs=args.surrogate_head_epochs,
            calibration_blocks=args.surrogate_calibration_blocks,
            calibration_subsets=args.surrogate_calibration_subsets,
            reference_lsmi_discriminator_epochs=args.surrogate_reference_lsmi_discriminator_epochs,
            reference_lsmi_entropy_epochs=args.surrogate_reference_lsmi_entropy_epochs,
            seed=seed,
        ),
        device=device,
    )
    surrogate_uncertainty_mobius_result = estimate_surrogate_pid(
        x1_train,
        x2_train,
        y_train,
        num_labels=num_labels,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=SurrogatePIDConfig(
            batch_size=args.surrogate_batch_size,
            predictor_epochs=args.surrogate_predictor_epochs,
            head_type="uncertainty_mobius",
            head_epochs=args.surrogate_head_epochs,
            calibration_blocks=args.surrogate_calibration_blocks,
            calibration_subsets=args.surrogate_calibration_subsets,
            reference_lsmi_discriminator_epochs=args.surrogate_reference_lsmi_discriminator_epochs,
            reference_lsmi_entropy_epochs=args.surrogate_reference_lsmi_entropy_epochs,
            seed=seed,
        ),
        device=device,
    )
    surrogate_kernel_result = estimate_surrogate_pid(
        x1_train,
        x2_train,
        y_train,
        num_labels=num_labels,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=SurrogatePIDConfig(
            batch_size=args.surrogate_batch_size,
            predictor_epochs=args.surrogate_predictor_epochs,
            head_type="kernel",
            kernel_head_gamma=args.surrogate_kernel_head_gamma,
            calibration_blocks=args.surrogate_calibration_blocks,
            calibration_subsets=args.surrogate_calibration_subsets,
            reference_lsmi_discriminator_epochs=args.surrogate_reference_lsmi_discriminator_epochs,
            reference_lsmi_entropy_epochs=args.surrogate_reference_lsmi_entropy_epochs,
            seed=seed,
        ),
        device=device,
    )
    direct_kernel_result = estimate_direct_kernel_pid(
        x1_train,
        x2_train,
        y_train,
        num_labels=num_labels,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=DirectKernelPIDConfig(
            batch_size=args.surrogate_batch_size,
            predictor_epochs=args.surrogate_predictor_epochs,
            kernel=args.direct_kernel_family,
            kernel_gamma=args.direct_kernel_gamma,
            ridge_lambda=args.direct_kernel_ridge_lambda,
            refinement_epochs=args.direct_kernel_refinement_epochs,
            refinement_lr=args.direct_kernel_refinement_lr,
            lambda_anchor=args.direct_kernel_lambda_anchor,
            lambda_info=args.direct_kernel_lambda_info,
            lambda_redundancy=args.direct_kernel_lambda_redundancy,
            lambda_interval=args.direct_kernel_lambda_interval,
            lambda_rkhs=args.direct_kernel_lambda_rkhs,
            calibration_blocks=args.surrogate_calibration_blocks,
            seed=seed,
        ),
        device=device,
    )
    return {
        "pid_ce_alignment": _components_to_dict(pid_result.mean_components),
        "lsmi": lsmi_result_dict,
        "surrogate": _components_to_dict(surrogate_result.mean_components),
        "surrogate_mobius": _components_to_dict(surrogate_mobius_result.mean_components),
        "surrogate_uncertainty_mobius": _components_to_dict(surrogate_uncertainty_mobius_result.mean_components),
        "surrogate_kernel": _components_to_dict(surrogate_kernel_result.mean_components),
        "direct_kernel": _components_to_dict(direct_kernel_result.mean_components),
    }


def _split_to_dataset(split: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MultimodalDataset]:
    x1 = torch.as_tensor(split["0"]).float()
    x2 = torch.as_tensor(split["1"]).float()
    y = torch.as_tensor(split["label"]).long()
    dataset = MultimodalDataset([x1, x2], y)
    return x1, x2, y, dataset


def _split_logic_dataset(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    train_fraction: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MultimodalDataset, torch.Tensor, torch.Tensor, torch.Tensor, MultimodalDataset]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(x1.shape[0], generator=generator)
    train_count = int(train_fraction * x1.shape[0])
    train_indices = permutation[:train_count]
    test_indices = permutation[train_count:]
    x1_train = x1[train_indices].float()
    x2_train = x2[train_indices].float()
    y_train = y[train_indices].long().view(-1, 1)
    x1_test = x1[test_indices].float()
    x2_test = x2[test_indices].float()
    y_test = y[test_indices].long().view(-1, 1)
    train_dataset = MultimodalDataset([x1_train, x2_train], y_train)
    test_dataset = MultimodalDataset([x1_test, x2_test], y_test)
    return x1_train, x2_train, y_train, train_dataset, x1_test, x2_test, y_test, test_dataset


def _components_to_dict(components: torch.Tensor) -> dict[str, float]:
    values = components.detach().cpu().tolist()
    return {
        "redundancy": float(values[0]),
        "unique_1": float(values[1]),
        "unique_2": float(values[2]),
        "synergy": float(values[3]),
    }


def _resolve_lsmi_baseline(
    *,
    benchmark_key: str,
    x1_train: torch.Tensor,
    x2_train: torch.Tensor,
    y_train: torch.Tensor,
    train_dataset: MultimodalDataset,
    test_dataset: MultimodalDataset,
    num_labels: int,
    args: argparse.Namespace,
    seed: int,
    device: str,
    lsmi_cache: dict[str, Any],
) -> dict[str, float]:
    cached = lsmi_cache.get(benchmark_key)
    if cached is not None and not args.refresh_lsmi_cache:
        return {
            "redundancy": float(cached["redundancy"]),
            "unique_1": float(cached["unique_1"]),
            "unique_2": float(cached["unique_2"]),
            "synergy": float(cached["synergy"]),
        }

    lsmi_result = estimate_lsmi(
        x1_train,
        x2_train,
        y_train,
        num_labels=num_labels,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=LSMIConfig(
            batch_size=args.lsmi_batch_size,
            discriminator_epochs=args.lsmi_discriminator_epochs,
            entropy_epochs=args.lsmi_entropy_epochs,
            seed=seed,
        ),
        device=device,
    )
    result = _components_to_dict(lsmi_result.mean_components)
    lsmi_cache[benchmark_key] = {
        **result,
        "budget": {
            "lsmi_discriminator_epochs": args.lsmi_discriminator_epochs,
            "lsmi_entropy_epochs": args.lsmi_entropy_epochs,
            "num_labels": num_labels,
        },
    }
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _make_benchmark_key(*, kind: str, name: str, seed: int, num_labels: int, metadata: dict[str, Any]) -> str:
    return json.dumps(
        {
            "kind": kind,
            "name": name,
            "seed": seed,
            "num_labels": num_labels,
            "metadata": metadata,
        },
        sort_keys=True,
    )


def _to_markdown(*, device: str, payload: dict) -> str:
    lines = [
        "# Synthetic Estimator Benchmark",
        "",
        f"- device: `{device}`",
        f"- continuous num_data: `{payload['lsmi_budget']['num_data']}`",
        f"- logic repeat: `{payload['lsmi_budget']['logic_repeat']}`",
        f"- lsmi discriminator epochs: `{payload['lsmi_budget']['lsmi_discriminator_epochs']}`",
        f"- lsmi entropy epochs: `{payload['lsmi_budget']['lsmi_entropy_epochs']}`",
        "",
        "## Continuous Synthetic",
        "",
        "| setting | estimator | redundancy | unique_1 | unique_2 | synergy | dominant |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["continuous_results"]:
        lines.extend(_format_rows(row_key=row["setting"], row=row))

    lines.extend(
        [
            "",
            "## Binary Logic",
            "",
            "| case | estimator | redundancy | unique_1 | unique_2 | synergy | dominant |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["logic_results"]:
        lines.extend(_format_rows(row_key=row["case"], row=row))
    return "\n".join(lines)


def _format_rows(*, row_key: str, row: dict) -> list[str]:
    lines = []
    for estimator_key, label in [
        ("pid_ce_alignment", "pid"),
        ("lsmi", "lsmi"),
        ("surrogate", "surrogate"),
        ("surrogate_mobius", "surrogate-mobius"),
        ("surrogate_uncertainty_mobius", "surrogate-uncertainty-mobius"),
        ("surrogate_kernel", "surrogate-kernel"),
        ("direct_kernel", "direct-kernel"),
    ]:
        values = row[estimator_key]
        dominant = _dominant_component(values)
        lines.append(
            f"| {row_key} | {label} | "
            f"{values['redundancy']:.4f} | {values['unique_1']:.4f} | "
            f"{values['unique_2']:.4f} | {values['synergy']:.4f} | {dominant} |"
        )
    return lines


def _dominant_component(values: dict[str, float]) -> str:
    return max(values.items(), key=lambda item: item[1])[0]


if __name__ == "__main__":
    main()
