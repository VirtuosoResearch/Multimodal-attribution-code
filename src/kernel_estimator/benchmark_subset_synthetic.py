from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from kernel_estimator.subset_kernel_surrogate import (
    SubsetKernelSurrogateConfig,
    estimate_subset_oracle_pid,
    estimate_subset_kernel_surrogate_pid,
)
from pid_estimator.data import MultimodalDataset
from pid_estimator.lsmi import LSMIConfig, estimate_lsmi
from pid_estimator.surrogate import SurrogatePIDConfig, estimate_surrogate_pid
from pid_estimator.synthetic_data import SyntheticGenerationConfig, generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark subset-level kernel surrogate against surrogate and LSMI oracles.")
    parser.add_argument("--num-data", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=5)
    parser.add_argument("--transform-dim", type=int, default=16)
    parser.add_argument("--feature-sep", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lsmi-batch-size", type=int, default=64)
    parser.add_argument("--lsmi-discriminator-epochs", type=int, default=5)
    parser.add_argument("--lsmi-entropy-epochs", type=int, default=5)
    parser.add_argument("--surrogate-batch-size", type=int, default=64)
    parser.add_argument("--surrogate-predictor-epochs", type=int, default=5)
    parser.add_argument("--surrogate-head-epochs", type=int, default=100)
    parser.add_argument("--surrogate-calibration-blocks", type=int, default=8)
    parser.add_argument("--surrogate-calibration-subsets", type=int, default=12)
    parser.add_argument("--surrogate-reference-lsmi-discriminator-epochs", type=int, default=3)
    parser.add_argument("--surrogate-reference-lsmi-entropy-epochs", type=int, default=3)
    parser.add_argument("--subset-kernel-blocks", type=int, default=8)
    parser.add_argument("--subset-kernel-calibration-subsets", type=int, default=12)
    parser.add_argument("--subset-kernel-ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--subset-kernel-gamma", type=float, default=None)
    parser.add_argument("--subset-kernel-predictor-epochs", type=int, default=8)
    parser.add_argument("--subset-kernel-predictor-batch-size", type=int, default=64)
    parser.add_argument("--eval-subsets", type=int, default=16)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["redundancy", "uniqueness0", "uniqueness1", "synergy"],
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="src/kernel_estimator/results/subset_kernel_benchmark.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="src/kernel_estimator/results/subset_kernel_benchmark.md",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for offset, setting in enumerate(args.settings):
        print(f"\n{'=' * 60}")
        print(f"Setting: {setting}")
        print(f"{'=' * 60}")

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

        print("  Running LSMI oracle...")
        lsmi_full = estimate_lsmi(
            x1_test,
            x2_test,
            y_test,
            num_labels=args.num_classes,
            train_dataset=test_dataset,
            test_dataset=test_dataset,
            config=LSMIConfig(
                batch_size=args.lsmi_batch_size,
                discriminator_epochs=args.lsmi_discriminator_epochs,
                entropy_epochs=args.lsmi_entropy_epochs,
                seed=args.seed + offset,
            ),
            device=device,
        )

        print("  Running sample/block surrogate...")
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

        print("  Running subset-kernel surrogate...")
        subset_kernel_config = SubsetKernelSurrogateConfig(
            num_blocks=args.subset_kernel_blocks,
            num_calibration_subsets=args.subset_kernel_calibration_subsets,
            ridge_lambda=args.subset_kernel_ridge_lambda,
            kernel_gamma=args.subset_kernel_gamma,
            predictor_epochs=args.subset_kernel_predictor_epochs,
            predictor_batch_size=args.subset_kernel_predictor_batch_size,
            seed=args.seed + offset,
        )
        subset_kernel_result = estimate_subset_kernel_surrogate_pid(
            x1_test,
            x2_test,
            y_test,
            num_labels=args.num_classes,
            test_dataset=test_dataset,
            config=subset_kernel_config,
            device=device,
        )
        subset_kernel_oracle_full = estimate_subset_oracle_pid(
            x1_test,
            x2_test,
            y_test,
            num_labels=args.num_classes,
            dataset=test_dataset,
            config=subset_kernel_config,
            device=device,
        )

        surrogate_eval = _evaluate_surrogate_queries(
            result=surrogate_result,
            dataset=test_dataset,
            num_labels=args.num_classes,
            num_eval_subsets=args.eval_subsets,
            oracle_config=subset_kernel_config,
            seed=args.seed + offset,
            device=device,
        )
        subset_kernel_eval = _evaluate_subset_kernel_queries(
            result=subset_kernel_result,
            dataset=test_dataset,
            num_labels=args.num_classes,
            num_eval_subsets=args.eval_subsets,
            oracle_config=subset_kernel_config,
            seed=args.seed + offset,
            device=device,
        )

        rows.append(
            {
                "setting": setting,
                "generation_config": asdict(generation_config),
                "lsmi_full": _components_to_dict(lsmi_full.mean_components),
                "surrogate_full": _components_to_dict(surrogate_result.mean_components),
                "subset_kernel_oracle_full": _components_to_dict(subset_kernel_oracle_full),
                "subset_kernel_full": _components_to_dict(subset_kernel_result.mean_components),
                "surrogate_subset_eval": surrogate_eval,
                "subset_kernel_subset_eval": subset_kernel_eval,
            }
        )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"device": device, "results": rows}, indent=2), encoding="utf-8")
    output_md.write_text(_to_markdown(device=device, rows=rows), encoding="utf-8")
    print(_to_markdown(device=device, rows=rows))
    print(f"\nSaved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")


def _evaluate_surrogate_queries(
    *,
    result,
    dataset: MultimodalDataset,
    num_labels: int,
    num_eval_subsets: int,
    oracle_config: SubsetKernelSurrogateConfig,
    seed: int,
    device: str,
) -> dict[str, float]:
    eval_subsets = _generate_eval_subsets(len(result.block_indices), num_eval_subsets, seed + 1000)
    calibration_set = {tuple(record.subset_weights.tolist()) for record in result.calibration_records}
    sq_errors = []
    abs_errors = []
    oracle_norms = []
    count = 0
    for subset_weights in eval_subsets:
        if tuple(subset_weights.tolist()) in calibration_set:
            continue
        oracle = _subset_oracle(
            dataset=dataset,
            block_indices=result.block_indices,
            subset_weights=subset_weights,
            num_labels=num_labels,
            oracle_config=oracle_config,
            seed=seed,
            device=device,
        )
        prediction = result.query_engine.query(subset_weights).detach().cpu()
        sq_errors.append((prediction - oracle).pow(2))
        abs_errors.append((prediction - oracle).abs())
        oracle_norms.append(oracle.abs().mean())
        count += 1
    return _aggregate_errors(sq_errors, abs_errors, count, oracle_norms)


def _evaluate_subset_kernel_queries(
    *,
    result,
    dataset: MultimodalDataset,
    num_labels: int,
    num_eval_subsets: int,
    oracle_config: SubsetKernelSurrogateConfig,
    seed: int,
    device: str,
) -> dict[str, float]:
    eval_subsets = _generate_eval_subsets(len(result.block_indices), num_eval_subsets, seed + 2000)
    calibration_set = {tuple(record.subset_weights.tolist()) for record in result.calibration_records}
    sq_errors = []
    abs_errors = []
    oracle_norms = []
    count = 0
    for subset_weights in eval_subsets:
        if tuple(subset_weights.tolist()) in calibration_set:
            continue
        oracle = _subset_oracle(
            dataset=dataset,
            block_indices=result.block_indices,
            subset_weights=subset_weights,
            num_labels=num_labels,
            oracle_config=oracle_config,
            seed=seed,
            device=device,
        )
        prediction = result.query_engine.query(subset_weights).detach().cpu()
        sq_errors.append((prediction - oracle).pow(2))
        abs_errors.append((prediction - oracle).abs())
        oracle_norms.append(oracle.abs().mean())
        count += 1
    return _aggregate_errors(sq_errors, abs_errors, count, oracle_norms)


def _subset_oracle(
    *,
    dataset: MultimodalDataset,
    block_indices: list[torch.Tensor],
    subset_weights: torch.Tensor,
    num_labels: int,
    oracle_config: SubsetKernelSurrogateConfig,
    seed: int,
    device: str,
) -> torch.Tensor:
    selected = [block for block, weight in zip(block_indices, subset_weights.tolist()) if weight > 0]
    indices = torch.cat(selected, dim=0) if selected else block_indices[0]
    x1 = dataset.modalities[0][indices].float()
    x2 = dataset.modalities[1][indices].float()
    y = dataset.labels[indices].long()
    subset_dataset = MultimodalDataset([x1, x2], y)
    oracle = estimate_subset_oracle_pid(
        x1,
        x2,
        y,
        num_labels=num_labels,
        dataset=subset_dataset,
        config=SubsetKernelSurrogateConfig(
            num_blocks=oracle_config.num_blocks,
            num_calibration_subsets=oracle_config.num_calibration_subsets,
            kernel_gamma=oracle_config.kernel_gamma,
            ridge_lambda=oracle_config.ridge_lambda,
            predictor_hidden_dim=oracle_config.predictor_hidden_dim,
            predictor_epochs=oracle_config.predictor_epochs,
            predictor_lr=oracle_config.predictor_lr,
            predictor_batch_size=oracle_config.predictor_batch_size,
            oracle_train_fraction=oracle_config.oracle_train_fraction,
            center_targets=oracle_config.center_targets,
            seed=seed,
            verbose=False,
        ),
        device=device,
    )
    return oracle.detach().cpu()


def _aggregate_errors(
    sq_errors: list[torch.Tensor],
    abs_errors: list[torch.Tensor],
    count: int,
    oracle_norms: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    if count == 0:
        return {"num_eval_subsets": 0.0, "rmse": 0.0, "mae": 0.0, "oracle_mean_abs": 0.0}
    sq = torch.stack(sq_errors, dim=0)
    ab = torch.stack(abs_errors, dim=0)
    oracle_mean_abs = 0.0
    if oracle_norms:
        oracle_mean_abs = float(torch.stack(oracle_norms).mean().item())
    return {
        "num_eval_subsets": float(count),
        "rmse": float(torch.sqrt(sq.mean()).item()),
        "mae": float(ab.mean().item()),
        "oracle_mean_abs": oracle_mean_abs,
    }


def _generate_eval_subsets(num_blocks: int, num_subsets: int, seed: int) -> list[torch.Tensor]:
    generator = random.Random(seed)
    candidates = []
    while len(candidates) < num_subsets:
        size = generator.randint(1, num_blocks)
        selected = set(generator.sample(range(num_blocks), k=size))
        candidates.append(torch.tensor([1 if index in selected else 0 for index in range(num_blocks)], dtype=torch.float32))
    unique = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


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


def _to_markdown(*, device: str, rows: list[dict]) -> str:
    lines = [
        "# Subset-Level Kernel Surrogate Benchmark",
        "",
        f"- device: `{device}`",
        "",
        "| setting | estimator | redundancy | unique_1 | unique_2 | synergy | dominant |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        for key, label in [
            ("lsmi_full", "lsmi"),
            ("surrogate_full", "surrogate"),
            ("subset_kernel_oracle_full", "subset-kernel-oracle"),
            ("subset_kernel_full", "subset-kernel"),
        ]:
            values = row[key]
            dominant = max(values.items(), key=lambda item: item[1])[0]
            lines.append(
                f"| {row['setting']} | {label} | "
                f"{values['redundancy']:.4f} | {values['unique_1']:.4f} | "
                f"{values['unique_2']:.4f} | {values['synergy']:.4f} | {dominant} |"
            )
        lines.append(
            f"| {row['setting']} | surrogate-query | rmse={row['surrogate_subset_eval']['rmse']:.4f} | "
            f"mae={row['surrogate_subset_eval']['mae']:.4f} | "
            f"eval={int(row['surrogate_subset_eval']['num_eval_subsets'])}, "
            f"oracle={row['surrogate_subset_eval']['oracle_mean_abs']:.4f} |  |  |"
        )
        lines.append(
            f"| {row['setting']} | subset-kernel-query | rmse={row['subset_kernel_subset_eval']['rmse']:.4f} | "
            f"mae={row['subset_kernel_subset_eval']['mae']:.4f} | "
            f"eval={int(row['subset_kernel_subset_eval']['num_eval_subsets'])}, "
            f"oracle={row['subset_kernel_subset_eval']['oracle_mean_abs']:.4f} |  |  |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
