from __future__ import annotations

import argparse
import json
import math
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

from kernel_surrogate.kernel_attribution import rmse, spearmanr
from kernel_surrogate.real_data_cmis import (
    build_group_mask_blocks,
    fit_surrogate_method,
    predict_surrogate_method,
    split_indices,
)


DEFAULT_DATASETS = ("avmnist", "humor", "imdb", "mosei", "mosi", "sarcasm")
DEFAULT_SETTINGS = (
    ("kmeans_data80_mod80", "kmeans_grouped25_subset100_data80_mod80"),
    ("kmeans_data90_mod90", "kmeans_grouped25_subset100_data90_mod90"),
    ("random_data80_mod80", "random_grouped25_subset100_data80_mod80"),
    ("random_data90_mod90", "random_grouped25_subset100_data90_mod90"),
)
DEFAULT_RIDGE_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_GAMMA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_SHARED_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "structured_rbf_lds_grid_80_20.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "structured_rbf_lds_grid_80_20.md"
DEFAULT_OUTPUT_PY = ROOT / "src" / "kernel_surrogate" / "structured_rbf_best_hparams.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search the group-level structured_rbf kernel surrogate on existing "
            "LDS subset artifacts using an 80/20 train/test split."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--settings",
        nargs="+",
        default=[name for name, _ in DEFAULT_SETTINGS],
        help="Setting names to include. Known names: kmeans_data80_mod80, kmeans_data90_mod90, random_data80_mod80, random_data90_mod90.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=list(DEFAULT_RIDGE_GRID))
    parser.add_argument("--gamma-grid", nargs="+", type=float, default=list(DEFAULT_GAMMA_GRID))
    parser.add_argument("--shared-grid", nargs="+", type=float, default=list(DEFAULT_SHARED_GRID))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-python", type=Path, default=DEFAULT_OUTPUT_PY)
    return parser.parse_args()


def pearsonr(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denom = float(np.sqrt(np.sum(pred**2) * np.sum(truth**2)))
    if denom <= 1e-12 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(pred * truth) / denom)


def load_payload(result_root: Path, run_name: str, dataset: str, seed: int) -> dict[str, Any] | None:
    path = result_root / run_name / f"{dataset}_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def score_key(row: dict[str, Any]) -> tuple[float, float, float]:
    lds = float(np.nan_to_num(row["test_lds"], nan=-1e18))
    pearson = float(np.nan_to_num(row["test_pearson"], nan=-1e18))
    return lds, pearson, -float(row["test_rmse"])


def evaluate_setting(
    *,
    dataset: str,
    setting_name: str,
    run_name: str,
    payload: dict[str, Any],
    ridge_grid: list[float],
    gamma_grid: list[float],
    shared_grid: list[float],
    train_fraction: float,
    split_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mask_blocks, targets = build_group_mask_blocks(payload)
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    truth = targets[test_indices]
    num_atoms = int(mask_blocks["modality_deleted"].shape[1])
    identity_metric = np.eye(num_atoms, dtype=np.float64)

    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for ridge_lambda in ridge_grid:
        for gamma_scale in gamma_grid:
            for structured_shared_weight in shared_grid:
                fit = fit_surrogate_method(
                    "structured_rbf",
                    mask_blocks,
                    targets,
                    identity_metric,
                    ridge_lambda=float(ridge_lambda),
                    gamma_scale=float(gamma_scale),
                    structured_shared_weight=float(structured_shared_weight),
                    train_indices=train_indices,
                )
                pred = predict_surrogate_method(
                    "structured_rbf",
                    mask_blocks,
                    fit,
                    metric=identity_metric,
                    query_indices=test_indices,
                )
                row = {
                    "dataset": dataset,
                    "setting": setting_name,
                    "run_name": run_name,
                    "method": "structured_rbf",
                    "ridge_lambda": float(ridge_lambda),
                    "gamma_scale": float(gamma_scale),
                    "structured_shared_weight": float(structured_shared_weight),
                    "test_lds": spearmanr(pred, truth),
                    "test_pearson": pearsonr(pred, truth),
                    "test_rmse": rmse(pred, truth),
                    "train_lds": spearmanr(fit["train_predictions"], targets[train_indices]),
                    "train_rmse": rmse(fit["train_predictions"], targets[train_indices]),
                    "gamma": float(fit["gamma"]),
                    "base_gamma": float(fit["base_gamma"]),
                    "num_train": int(train_indices.size),
                    "num_test": int(test_indices.size),
                    "num_subsets": int(len(targets)),
                }
                records.append(row)
                if best is None or score_key(row) > score_key(best):
                    best = row
    assert best is not None
    return best, records


def main() -> None:
    args = parse_args()
    settings_by_name = dict(DEFAULT_SETTINGS)
    requested_settings = []
    for setting_name in args.settings:
        if setting_name not in settings_by_name:
            raise ValueError(f"unknown setting {setting_name!r}; known settings: {sorted(settings_by_name)}")
        requested_settings.append((setting_name, settings_by_name[setting_name]))

    all_records: list[dict[str, Any]] = []
    best_by_setting: list[dict[str, Any]] = []
    best_per_dataset: list[dict[str, Any]] = []
    missing_payloads: list[dict[str, str]] = []

    for dataset in args.datasets:
        dataset_bests: list[dict[str, Any]] = []
        for setting_name, run_name in requested_settings:
            payload = load_payload(args.result_root, run_name, dataset, int(args.seed))
            if payload is None:
                missing_payloads.append({"dataset": dataset, "setting": setting_name, "run_name": run_name})
                continue
            print(f"[{dataset}] searching structured_rbf on {setting_name}...", flush=True)
            best, records = evaluate_setting(
                dataset=dataset,
                setting_name=setting_name,
                run_name=run_name,
                payload=payload,
                ridge_grid=[float(v) for v in args.ridge_grid],
                gamma_grid=[float(v) for v in args.gamma_grid],
                shared_grid=[float(v) for v in args.shared_grid],
                train_fraction=float(args.train_fraction),
                split_seed=int(args.split_seed),
            )
            all_records.extend(records)
            best_by_setting.append(best)
            dataset_bests.append(best)
            print(
                f"  best LDS={best['test_lds']:.6f}, RMSE={best['test_rmse']:.6f}, "
                f"ridge={best['ridge_lambda']}, gamma_scale={best['gamma_scale']}, shared={best['structured_shared_weight']}",
                flush=True,
            )
        if not dataset_bests:
            raise FileNotFoundError(f"no LDS payloads found for dataset {dataset!r}")
        best_per_dataset.append(max(dataset_bests, key=score_key))

    payload = {
        "config": {
            "datasets": [str(dataset) for dataset in args.datasets],
            "settings": [name for name, _ in requested_settings],
            "method": "structured_rbf",
            "split": f"{float(args.train_fraction):.6g}/{1.0 - float(args.train_fraction):.6g}",
            "train_fraction": float(args.train_fraction),
            "split_seed": int(args.split_seed),
            "seed": int(args.seed),
            "ridge_grid": [float(v) for v in args.ridge_grid],
            "gamma_grid": [float(v) for v in args.gamma_grid],
            "shared_grid": [float(v) for v in args.shared_grid],
            "result_root": str(args.result_root),
        },
        "best_per_dataset": best_per_dataset,
        "best_by_setting": best_by_setting,
        "records": all_records,
        "missing_payloads": missing_payloads,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_python.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(payload), encoding="utf-8")
    args.output_python.write_text(to_python_module(payload), encoding="utf-8")
    print(to_markdown(payload), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)
    print(f"Saved fixed hparams to {args.output_python}", flush=True)


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Structured RBF LDS Grid Search",
        "",
        f"- method: `{payload['config']['method']}`",
        f"- split: `{payload['config']['split']}` seed `{payload['config']['split_seed']}`",
        f"- LDS metric: held-out Spearman correlation",
        "",
        "## Best Per Dataset",
        "",
        "| dataset | setting | LDS | Pearson | RMSE | ridge | gamma_scale | shared |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["best_per_dataset"]:
        lines.append(
            f"| {row['dataset']} | {row['setting']} | {row['test_lds']:.6f} | "
            f"{row['test_pearson']:.6f} | {row['test_rmse']:.6f} | "
            f"{row['ridge_lambda']} | {row['gamma_scale']} | {row['structured_shared_weight']} |"
        )
    lines.extend(
        [
            "",
            "## Best By Setting",
            "",
            "| dataset | setting | LDS | RMSE | ridge | gamma_scale | shared |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(payload["best_by_setting"], key=lambda item: (item["dataset"], item["setting"])):
        lines.append(
            f"| {row['dataset']} | {row['setting']} | {row['test_lds']:.6f} | "
            f"{row['test_rmse']:.6f} | {row['ridge_lambda']} | "
            f"{row['gamma_scale']} | {row['structured_shared_weight']} |"
        )
    if payload["missing_payloads"]:
        lines.extend(["", "## Missing Payloads", ""])
        for row in payload["missing_payloads"]:
            lines.append(f"- `{row['dataset']}` / `{row['setting']}` (`{row['run_name']}`)")
    return "\n".join(lines) + "\n"


def to_python_module(payload: dict[str, Any]) -> str:
    fixed = {
        str(row["dataset"]): {
            "method": "structured_rbf",
            "setting": str(row["setting"]),
            "run_name": str(row["run_name"]),
            "ridge_lambda": float(row["ridge_lambda"]),
            "gamma_scale": float(row["gamma_scale"]),
            "structured_shared_weight": float(row["structured_shared_weight"]),
            "test_lds": float(row["test_lds"]),
            "test_pearson": float(row["test_pearson"]),
            "test_rmse": float(row["test_rmse"]),
            "split": str(payload["config"]["split"]),
            "split_seed": int(payload["config"]["split_seed"]),
            "num_train": int(row["num_train"]),
            "num_test": int(row["num_test"]),
        }
        for row in payload["best_per_dataset"]
    }
    return (
        '"""Best structured_rbf surrogate hyperparameters from 80/20 LDS grid search."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from typing import Any\n"
        "\n"
        f"BEST_STRUCTURED_RBF_HPARAMS_80_20: dict[str, dict[str, Any]] = {json.dumps(fixed, indent=4, sort_keys=True, allow_nan=True)}\n"
        "\n"
        "\n"
        "def get_best_structured_rbf_hparams(dataset: str) -> dict[str, Any]:\n"
        "    key = str(dataset).lower()\n"
        "    if key not in BEST_STRUCTURED_RBF_HPARAMS_80_20:\n"
        "        raise KeyError(f\"no fixed structured_rbf hparams for dataset {dataset!r}\")\n"
        "    return dict(BEST_STRUCTURED_RBF_HPARAMS_80_20[key])\n"
    )


if __name__ == "__main__":
    main()
