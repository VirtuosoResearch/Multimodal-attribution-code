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
from kernel_surrogate.structured_rbf_best_hparams import (
    BEST_STRUCTURED_RBF_HPARAMS_80_20,
    get_best_structured_rbf_hparams,
)


DEFAULT_RESULT_ROOT = ROOT / "src" / "kernel_surrogate" / "subset_results" / "lds_real_train"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "structured_rbf_best_hparams_shuffles_80_20.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "structured_rbf_best_hparams_shuffles_80_20.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate recorded best structured_rbf hyperparameters across multiple "
            "80/20 train/test shuffles and report the highest held-out LDS."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=sorted(BEST_STRUCTURED_RBF_HPARAMS_80_20))
    parser.add_argument("--seed", type=int, default=0, help="LDS artifact seed to load.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
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


def load_payload(result_root: Path, run_name: str, dataset: str, seed: int) -> dict[str, Any]:
    path = result_root / run_name / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing LDS payload: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def score_key(row: dict[str, Any]) -> tuple[float, float, float]:
    lds = float(np.nan_to_num(row["test_lds"], nan=-1e18))
    pearson = float(np.nan_to_num(row["test_pearson"], nan=-1e18))
    return lds, pearson, -float(row["test_rmse"])


def evaluate_one_split(
    *,
    dataset: str,
    payload: dict[str, Any],
    hparams: dict[str, Any],
    train_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    mask_blocks, targets = build_group_mask_blocks(payload)
    train_indices, test_indices = split_indices(len(targets), train_fraction=train_fraction, seed=split_seed)
    truth = targets[test_indices]
    num_atoms = int(mask_blocks["modality_deleted"].shape[1])
    identity_metric = np.eye(num_atoms, dtype=np.float64)
    fit = fit_surrogate_method(
        "structured_rbf",
        mask_blocks,
        targets,
        identity_metric,
        ridge_lambda=float(hparams["ridge_lambda"]),
        gamma_scale=float(hparams["gamma_scale"]),
        structured_shared_weight=float(hparams["structured_shared_weight"]),
        train_indices=train_indices,
    )
    pred = predict_surrogate_method(
        "structured_rbf",
        mask_blocks,
        fit,
        metric=identity_metric,
        query_indices=test_indices,
    )
    return {
        "dataset": dataset,
        "setting": str(hparams["setting"]),
        "run_name": str(hparams["run_name"]),
        "method": "structured_rbf",
        "ridge_lambda": float(hparams["ridge_lambda"]),
        "gamma_scale": float(hparams["gamma_scale"]),
        "structured_shared_weight": float(hparams["structured_shared_weight"]),
        "split_seed": int(split_seed),
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
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }


def main() -> None:
    args = parse_args()
    split_seeds = [int(seed) for seed in args.split_seeds]
    if not split_seeds:
        raise ValueError("--split-seeds must contain at least one seed")

    records: list[dict[str, Any]] = []
    best_per_dataset: list[dict[str, Any]] = []
    for dataset in args.datasets:
        hparams = get_best_structured_rbf_hparams(str(dataset))
        payload = load_payload(args.result_root, str(hparams["run_name"]), str(dataset), int(args.seed))
        rows = []
        print(f"[{dataset}] evaluating {len(split_seeds)} 80/20 shuffles...", flush=True)
        for split_seed in split_seeds:
            row = evaluate_one_split(
                dataset=str(dataset),
                payload=payload,
                hparams=hparams,
                train_fraction=float(args.train_fraction),
                split_seed=split_seed,
            )
            rows.append(row)
            print(
                f"  split_seed={split_seed}: LDS={row['test_lds']:.6f}, "
                f"Pearson={row['test_pearson']:.6f}, RMSE={row['test_rmse']:.6f}",
                flush=True,
            )
        records.extend(rows)
        best = max(rows, key=score_key)
        best_per_dataset.append(best)
        print(
            f"  best split_seed={best['split_seed']}: LDS={best['test_lds']:.6f}, "
            f"Pearson={best['test_pearson']:.6f}, RMSE={best['test_rmse']:.6f}",
            flush=True,
        )

    payload = {
        "config": {
            "datasets": [str(dataset) for dataset in args.datasets],
            "method": "structured_rbf",
            "source_hparams": "BEST_STRUCTURED_RBF_HPARAMS_80_20",
            "split": f"{float(args.train_fraction):.6g}/{1.0 - float(args.train_fraction):.6g}",
            "train_fraction": float(args.train_fraction),
            "split_seeds": split_seeds,
            "artifact_seed": int(args.seed),
            "result_root": str(args.result_root),
        },
        "best_per_dataset": best_per_dataset,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Structured RBF Best-Hparams Shuffle Evaluation",
        "",
        f"- method: `{payload['config']['method']}`",
        f"- source hparams: `{payload['config']['source_hparams']}`",
        f"- split: `{payload['config']['split']}`",
        f"- split seeds: `{payload['config']['split_seeds']}`",
        f"- LDS metric: held-out Spearman correlation",
        "",
        "## Best Per Dataset",
        "",
        "| dataset | setting | best split_seed | LDS | Pearson | RMSE | original seed0 LDS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    original_by_dataset = {
        str(row["dataset"]): row for row in payload["records"] if int(row["split_seed"]) == 0
    }
    for row in payload["best_per_dataset"]:
        original = original_by_dataset.get(str(row["dataset"]))
        original_lds = float(original["test_lds"]) if original is not None else float("nan")
        lines.append(
            f"| {row['dataset']} | {row['setting']} | {row['split_seed']} | "
            f"{row['test_lds']:.6f} | {row['test_pearson']:.6f} | {row['test_rmse']:.6f} | "
            f"{original_lds:.6f} |"
        )
    lines.extend(
        [
            "",
            "## All Shuffles",
            "",
            "| dataset | split_seed | LDS | Pearson | RMSE | train LDS |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(payload["records"], key=lambda item: (item["dataset"], item["split_seed"])):
        lines.append(
            f"| {row['dataset']} | {row['split_seed']} | {row['test_lds']:.6f} | "
            f"{row['test_pearson']:.6f} | {row['test_rmse']:.6f} | {row['train_lds']:.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
