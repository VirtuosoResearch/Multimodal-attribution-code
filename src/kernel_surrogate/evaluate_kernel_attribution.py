from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_surrogate.kernel_attribution import (
    DEFAULT_METHODS,
    KernelAttributionConfig,
    available_dataset_seeds,
    build_mask_blocks,
    compute_kernel,
    evaluate_methods,
    evaluations_to_records,
    load_result_payload,
    normalize_method_name,
    pairwise_sqeuclidean,
    rankdata,
    spearmanr,
    split_indices,
    split_train_validation_test_indices,
    summarize_records,
)


DEFAULT_RESULT_DIR = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "kernel_attribution_lds.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "kernel_attribution_lds.md"
EXISTING_LDS_DATASETS = ("imdb", "humor", "sarcasm", "mosi", "mosei", "avmnist")


def main() -> None:
    default_config = KernelAttributionConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate hierarchical kernel attribution surrogates on existing LDS subset results. "
            "The default uses the six datasets already present in subset_results/lds_real_train."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(EXISTING_LDS_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--target-metric",
        default=default_config.target_metric,
        help=(
            "Scalar metric to fit, e.g. test_acc/test_loss. For saved per-sample losses, "
            "use test_sample_loss_mean or test_sample_loss:IDX."
        ),
    )
    parser.add_argument("--input-unit", choices=["sample", "group"], default=default_config.input_unit)
    parser.add_argument("--train-fraction", type=float, default=default_config.train_fraction)
    parser.add_argument("--validation-fraction", type=float, default=default_config.validation_fraction)
    parser.add_argument("--split-seed", type=int, default=default_config.split_seed)
    parser.add_argument("--ridge-lambda", type=float, default=default_config.ridge_lambda)
    parser.add_argument("--search-hyperparams", action="store_true")
    parser.add_argument("--search-strategy", choices=["loo", "validation", "test"], default=default_config.search_strategy)
    parser.add_argument(
        "--no-refit-on-train-validation",
        action="store_true",
        help="For validation search, keep the final model fitted on train only instead of refitting on train+validation.",
    )
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=list(default_config.ridge_grid))
    parser.add_argument("--gamma-scale-grid", nargs="+", type=float, default=list(default_config.gamma_scale_grid))
    parser.add_argument("--modality-shared-grid", nargs="+", type=float, default=list(default_config.modality_shared_grid))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
        print("Self-tests passed.")
        return

    dataset_seed_pairs = _resolve_dataset_seed_pairs(args.result_dir, args.datasets, args.seeds)
    config = KernelAttributionConfig(
        ridge_lambda=float(args.ridge_lambda),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        split_seed=int(args.split_seed),
        target_metric=str(args.target_metric),
        input_unit=str(args.input_unit),
        methods=tuple(normalize_method_name(method) for method in args.methods),
        search_hyperparams=bool(args.search_hyperparams),
        search_strategy=str(args.search_strategy),
        refit_on_train_validation=not bool(args.no_refit_on_train_validation),
        ridge_grid=tuple(float(value) for value in args.ridge_grid),
        gamma_scale_grid=tuple(float(value) for value in args.gamma_scale_grid),
        modality_shared_grid=tuple(float(value) for value in args.modality_shared_grid),
    )

    records: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    for dataset, seed in dataset_seed_pairs:
        print(f"[{dataset}] seed={seed}: loading LDS subset results...", flush=True)
        payload = load_result_payload(args.result_dir, dataset, seed)
        blocks = build_mask_blocks(
            payload,
            target_metric=config.target_metric,
            input_unit=config.input_unit,
        )
        print(
            f"[{dataset}] seed={seed}: fitting {len(config.methods)} methods "
            f"on {len(blocks.targets)} subsets"
            + (
                f" with {config.search_strategy} hyperparameter search"
                if config.search_hyperparams
                else ""
            )
            + "...",
            flush=True,
        )
        evaluations = evaluate_methods(blocks, config=config)
        dataset_records = evaluations_to_records(dataset=dataset, seed=seed, evaluations=evaluations)
        records.extend(dataset_records)
        dataset_summaries.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "num_runs": int(len(blocks.targets)),
                "dataset_size": int(blocks.dataset_size),
                "num_modalities": int(blocks.num_modalities),
                "target_metric": config.target_metric,
                "input_unit": config.input_unit,
                "methods": dataset_records,
            }
        )
        for record in dataset_records:
            search_text = ""
            if record.get("loo_lds") is not None:
                search_text = (
                    f", LOO-LDS={record['loo_lds']:.6f}, ridge={record['ridge_lambda']:.1e}"
                    + (
                        f", modality_weight={record['linear_modality_weight']:.3g}"
                        if record.get("linear_modality_weight") is not None
                        else ""
                    )
                    + (
                        f", modality_shared={record['modality_shared_weight']:.3g}"
                        if record.get("modality_shared_weight") is not None
                        else ""
                    )
                )
            elif record.get("validation_lds") is not None:
                search_text = (
                    f", Val-LDS={record['validation_lds']:.6f}, Val-RMSE={record['validation_rmse']:.6f}, "
                    f"ridge={record['ridge_lambda']:.1e}"
                    + (
                        f", modality_weight={record['linear_modality_weight']:.3g}"
                        if record.get("linear_modality_weight") is not None
                        else ""
                    )
                    + (
                        f", modality_shared={record['modality_shared_weight']:.3g}"
                        if record.get("modality_shared_weight") is not None
                        else ""
                    )
                )
            elif record.get("test_search_lds") is not None:
                search_text = (
                    f", TestSearch-LDS={record['test_search_lds']:.6f}, "
                    f"TestSearch-RMSE={record['test_search_rmse']:.6f}, ridge={record['ridge_lambda']:.1e}"
                    + (
                        f", modality_weight={record['linear_modality_weight']:.3g}"
                        if record.get("linear_modality_weight") is not None
                        else ""
                    )
                    + (
                        f", modality_shared={record['modality_shared_weight']:.3g}"
                        if record.get("modality_shared_weight") is not None
                        else ""
                    )
                )
            print(
                f"  {record['method']}: RMSE={record['rmse']:.6f}, LDS={record['lds']:.6f}"
                + search_text,
                flush=True,
            )

    summary = summarize_records(records)
    payload = {
        "config": {
            "result_dir": str(args.result_dir),
            "datasets": [dataset for dataset, _ in dataset_seed_pairs],
            "seeds": sorted({int(seed) for _, seed in dataset_seed_pairs}),
            "target_metric": config.target_metric,
            "input_unit": config.input_unit,
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "split_seed": config.split_seed,
            "ridge_lambda": config.ridge_lambda,
            "search_hyperparams": config.search_hyperparams,
            "search_strategy": config.search_strategy,
            "refit_on_train_validation": config.refit_on_train_validation,
            "ridge_grid": list(config.ridge_grid),
            "gamma_scale_grid": list(config.gamma_scale_grid),
            "modality_shared_grid": list(config.modality_shared_grid),
            "methods": list(config.methods),
        },
        "summary": summary,
        "dataset_results": dataset_summaries,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(_to_markdown(payload), encoding="utf-8")
    print("\n" + _to_markdown(payload), flush=True)
    print(f"\nSaved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def _resolve_dataset_seed_pairs(
    result_dir: Path,
    datasets: list[str],
    seeds: list[int],
) -> list[tuple[str, int]]:
    if len(datasets) == 1 and datasets[0].lower() == "available":
        pairs = available_dataset_seeds(result_dir)
    else:
        pairs = [(dataset, seed) for dataset in datasets for seed in seeds]

    existing = set(available_dataset_seeds(result_dir))
    resolved: list[tuple[str, int]] = []
    missing: list[tuple[str, int]] = []
    for pair in pairs:
        if pair in existing:
            resolved.append(pair)
        else:
            missing.append(pair)
    if missing:
        missing_text = ", ".join(f"{dataset}_seed{seed}" for dataset, seed in missing)
        print(f"Skipping missing LDS result files: {missing_text}", flush=True)
    if not resolved:
        raise FileNotFoundError(f"no matching LDS result files found in {result_dir}")
    return resolved


def _to_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    if config.get("search_hyperparams", False) and config.get("search_strategy") == "validation":
        split_text = (
            f"`{config['train_fraction']:.2f}` / `{config['validation_fraction']:.2f}` / "
            f"`{1.0 - config['train_fraction'] - config['validation_fraction']:.2f}`"
        )
        split_label = "Train/validation/test split"
    else:
        split_text = f"`{config['train_fraction']:.2f}` / `{1.0 - config['train_fraction']:.2f}`"
        split_label = "Train/test split"
    lines = [
        "# Kernel Attribution LDS Evaluation",
        "",
        f"- Target metric: `{config['target_metric']}`",
        f"- Input unit: `{config.get('input_unit', 'sample')}`",
        f"- {split_label}: {split_text}",
        f"- Split seed: `{config['split_seed']}`",
        f"- Hyperparameter search: `{config.get('search_hyperparams', False)}`",
        f"- Search strategy: `{config.get('search_strategy', 'loo')}`",
        f"- Refit on train+validation: `{config.get('refit_on_train_validation', False)}`",
        f"- Ridge lambda: `{config['ridge_lambda']}`",
        "",
        "## Summary",
        "",
        "| Method | Dataset runs | Mean RMSE | Std RMSE | Mean LDS | Std LDS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {method} | {num_dataset_runs} | {mean_rmse:.6f} | {std_rmse:.6f} | {mean_lds:.6f} | {std_lds:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Per Dataset",
            "",
        "| Dataset | Method | Test RMSE | Test LDS | Val RMSE | Val LDS | TestSearch RMSE | TestSearch LDS | LOO-LDS | Ridge | Shared rho | Modality w | Train | Val | Test |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_result in payload["dataset_results"]:
        dataset = dataset_result["dataset"]
        for row in dataset_result["methods"]:
            lines.append(
                f"| {dataset} | {row['method']} | {row['rmse']:.6f} | {row['lds']:.6f} | "
                f"{_format_optional_float(row.get('validation_rmse'))} | "
                f"{_format_optional_float(row.get('validation_lds'))} | "
                f"{_format_optional_float(row.get('test_search_rmse'))} | "
                f"{_format_optional_float(row.get('test_search_lds'))} | "
                f"{_format_optional_float(row.get('loo_lds'))} | {row.get('ridge_lambda', float('nan')):.1e} | "
                f"{_format_optional_float(row.get('modality_shared_weight'))} | "
                f"{_format_optional_float(row.get('linear_modality_weight'))} | "
                f"{row['num_train']} | {row.get('num_validation', 0)} | {row['num_test']} |"
            )
    lines.append("")
    lines.append(
        "Note: version 1 and version 2 hierarchical methods are equivalent under the identity "
        "sample/modality geometry used here, because `(d, E)` equals `(a, B)` with `a = 1 - s` and "
        "`B = s * (1 - G)` for the saved LDS masks."
    )
    return "\n".join(lines) + "\n"


def _run_self_tests() -> None:
    values = np.asarray([3.0, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(rankdata(values), np.asarray([4.0, 1.5, 1.5, 3.0]))
    assert abs(spearmanr(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])) - 1.0) < 1e-12

    train, test = split_indices(50, train_fraction=0.5, seed=0)
    assert train.size == 25
    assert test.size == 25
    assert set(train.tolist()).isdisjoint(test.tolist())
    train, validation, test = split_train_validation_test_indices(
        50,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=0,
    )
    assert train.size == 30
    assert validation.size == 10
    assert test.size == 10
    assert set(train.tolist()).isdisjoint(validation.tolist())
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert set(validation.tolist()).isdisjoint(test.tolist())

    sample = np.asarray([[0, 1], [1, 1]], dtype=np.float32)
    dist = pairwise_sqeuclidean(sample, sample)
    np.testing.assert_allclose(dist, np.asarray([[0.0, 1.0], [1.0, 0.0]]))

    payload = {
        "group_partition": {
            "num_modalities": 2,
            "group_indices": [[0], [1]],
        },
        "runs": [
            {
                "sample_index": 0,
                "data_indices": [0, 1],
                "sample_modality_mask": [[1, 1], [1, 1]],
                "test_acc": 1.0,
                "test_sample_losses": [0.2, 0.4],
            },
            {
                "sample_index": 1,
                "data_indices": [0],
                "sample_modality_mask": [[1, 0]],
                "test_acc": 0.5,
                "test_sample_losses": [0.3, 0.6],
            },
            {
                "sample_index": 2,
                "data_indices": [1],
                "sample_modality_mask": [[0, 1]],
                "test_acc": 0.25,
                "test_sample_losses": [0.5, 0.8],
            },
            {
                "sample_index": 3,
                "data_indices": [0, 1],
                "sample_modality_mask": [[0, 1], [1, 0]],
                "test_acc": 0.75,
                "test_sample_losses": [0.7, 1.0],
            },
        ],
    }
    blocks = build_mask_blocks(payload)
    sample_loss_blocks = build_mask_blocks(payload, target_metric="test_sample_loss:1")
    np.testing.assert_allclose(sample_loss_blocks.targets, np.asarray([0.4, 0.6, 0.8, 1.0]))
    kernel_1, gamma_1 = _kernel_for_self_test(blocks, "kernel_surrogate_1_rbf")
    kernel_2, gamma_2 = _kernel_for_self_test(blocks, "kernel_surrogate_2_rbf")
    np.testing.assert_allclose(kernel_1, kernel_2)
    assert gamma_1 == gamma_2
    structured_identity, _ = compute_kernel(
        blocks,
        lhs_indices=np.arange(len(blocks.targets)),
        rhs_indices=np.arange(len(blocks.targets)),
        method="kernel_surrogate_1_structured_rbf",
        gamma_sample=gamma_1["gamma_sample"],
        gamma_modality=gamma_1["gamma_modality"],
        modality_shared_weight=0.0,
    )
    np.testing.assert_allclose(kernel_1, structured_identity)
    balanced, _ = _kernel_for_self_test(blocks, "kernel_surrogate_hier_linear")
    assert balanced.shape == (4, 4)
    assert np.isfinite(balanced).all()
    centered, _ = _kernel_for_self_test(blocks, "kernel_surrogate_hier_centered_linear")
    assert centered.shape == (4, 4)
    assert np.isfinite(centered).all()


def _format_optional_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _kernel_for_self_test(blocks, method: str):
    from kernel_surrogate.kernel_attribution import compute_kernel

    kernel, gammas = compute_kernel(
        blocks,
        lhs_indices=np.arange(len(blocks.targets)),
        rhs_indices=np.arange(len(blocks.targets)),
        method=method,
    )
    return kernel, gammas


if __name__ == "__main__":
    main()
