from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_geometry.data import load_dataset
from kernel_surrogate.kernel_attribution import spearmanr
from fisher_geometry.verify_fisher_supply_osfi import lanczos_steps_for_mask
from fisher_geometry.verify_humor_fisher_gpt_metrics import (
    DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT,
    DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT,
    compute_group_gradient_tables,
    inverse_quadrature,
    load_model,
    make_fisher_matvec,
    make_model_args,
    resolve_full_checkpoint,
    resolve_requested_device,
    select_named_parameters,
)


DEFAULT_DATASETS = ["avmnist", "humor", "imdb", "mosei", "mosi", "sarcasm"]
DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "kmeans_grouped25_subset100_data80_mod80"
)
DEFAULT_ACTUAL_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "actual_val_loss_gap_sampled"
DEFAULT_OUTPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "multimodal_inverse_controls"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare whether loss-connected A^{-1} signal is multimodal-specific. "
            "Uses the same target groups from actual validation loss-gap runs, then "
            "recomputes L-OSFI with cross-sample multimodal and unimodal A matrices."
        )
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--actual-dir", type=Path, default=DEFAULT_ACTUAL_DIR)
    parser.add_argument(
        "--full-checkpoint-roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT, DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--gradient-params", choices=["last_layer", "fusion", "all"], default="last_layer")
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--lanczos-steps", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=2026)
    args = parser.parse_args()

    device = resolve_requested_device(str(args.device))
    rng = np.random.default_rng(int(args.random_seed))
    dataset_outputs = []
    for dataset in args.datasets:
        dataset_outputs.append(run_dataset(str(dataset), args=args, device=device, rng=rng))

    output = {
        "config": {
            "datasets": [str(value) for value in args.datasets],
            "seed": int(args.seed),
            "result_dir": str(args.result_dir),
            "actual_dir": str(args.actual_dir),
            "device": str(device),
            "gradient_params": str(args.gradient_params),
            "damping": float(args.damping),
            "lanczos_steps": int(args.lanczos_steps),
            "random_seed": int(args.random_seed),
            "score": "loss_osfi = (h^T A_self^{-1} h - h^T A_all^{-1} h) / h^T A_self^{-1} h",
            "controls": [
                "multimodal: A from cross-sample x+y modal gradients",
                "x_only: A from x-only group gradients",
                "y_only: A from y-only group gradients",
            ],
        },
        "overall": summarize_overall(dataset_outputs),
        "datasets": dataset_outputs,
    }
    output_json = args.output_json or (args.output_dir / "multimodal_inverse_controls_all_datasets.json")
    output_md = args.output_md or output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


def run_dataset(
    dataset: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, Any]:
    actual_path = args.actual_dir / f"{dataset}_seed{args.seed}_actual_loss_gap_sampled.json"
    if not actual_path.exists():
        return {"dataset": dataset, "status": "missing_actual_gap", "records": [], "variant_summary": {}}
    actual_payload = json.loads(actual_path.read_text(encoding="utf-8"))
    actual_records = actual_payload.get("records", [])
    target_groups = [int(row["target_group"]) for row in actual_records]
    actual_lookup = {int(row["target_group"]): dict(row) for row in actual_records}
    if not target_groups:
        return {"dataset": dataset, "status": "no_valid_targets", "records": [], "variant_summary": {}}

    payload_path = args.result_dir / f"{dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
    data = load_dataset(
        dataset,
        int(args.seed),
        text_backbone=str(config.get("text_backbone", "clip_cached")),
        vision_backbone=str(config.get("vision_backbone", "clip_cached")),
        feature_device=str(args.feature_device or config.get("feature_device", device)),
        backbone_batch_size=int(config.get("backbone_batch_size", 32)),
        download_datasets=bool(config.get("download_datasets", False)),
        backbone_cache_dir=config.get("backbone_cache_dir"),
    )
    model_args = make_model_args(config, device=str(device))
    dim_x = int(data["train_x"].shape[1])
    dim_y = int(data["train_y"].shape[1])
    num_classes = int(data["num_classes"])
    task = str(data["task"])
    model_type = str(config.get("model", "cross_attention"))
    group_indices = [[int(index) for index in group] for group in payload["group_partition"]["group_indices"]]
    full_checkpoint = resolve_full_checkpoint(
        None,
        roots=args.full_checkpoint_roots,
        dataset=dataset,
        seed=int(args.seed),
    )

    print(f"[{dataset}] loading checkpoint: {full_checkpoint}", flush=True)
    model = load_model(
        full_checkpoint,
        model_type=model_type,
        dim_x=dim_x,
        dim_y=dim_y,
        num_classes=num_classes,
        model_args=model_args,
        device=str(device),
    )
    named_params = select_named_parameters(model, mode=str(args.gradient_params))
    param_names = [name for name, _ in named_params]
    print(f"[{dataset}] computing group gradients for {len(group_indices)} groups", flush=True)
    gradients = compute_group_gradient_tables(
        model,
        data=data,
        group_indices=group_indices,
        task=task,
        device=str(device),
        param_names=param_names,
    )

    variant_records: list[dict[str, Any]] = []
    for variant in ("multimodal", "x_only", "y_only"):
        rows_np, weights_np, self_rows = build_variant_rows(
            variant=variant,
            gradients=gradients,
            group_indices=group_indices,
            y_permutation=None,
        )
        variant_records.extend(
            compute_variant_records(
                dataset=dataset,
                variant=variant,
                rows_np=rows_np,
                weights_np=weights_np,
                self_rows=self_rows,
                gradients=gradients,
                target_groups=target_groups,
                actual_lookup=actual_lookup,
                damping=float(args.damping),
                lanczos_steps=int(args.lanczos_steps),
                device=device,
            )
        )

    return {
        "dataset": dataset,
        "status": "ok",
        "target_groups": target_groups,
        "num_targets": int(len(target_groups)),
        "actual_summary": actual_payload.get("summary", {}),
        "variant_summary": summarize_variants(variant_records),
        "records": variant_records,
    }


def build_variant_rows(
    *,
    variant: str,
    gradients: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    y_permutation: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    total = float(sum(len(indices) for indices in group_indices))
    rows = []
    weights = []
    self_rows: dict[int, list[int]] = {}
    for group_id, indices in enumerate(group_indices):
        row_ids: list[int] = []
        if variant in {"multimodal", "x_only"}:
            row_ids.append(len(rows))
            rows.append(np.asarray(gradients["single"][0][group_id], dtype=np.float32))
            weights.append(float(len(indices) / total))
        if variant in {"multimodal", "y_only"}:
            y_group = int(y_permutation[group_id]) if y_permutation is not None else int(group_id)
            row_ids.append(len(rows))
            rows.append(np.asarray(gradients["single"][1][y_group], dtype=np.float32))
            weights.append(float(len(group_indices[y_group]) / total))
        self_rows[int(group_id)] = row_ids
    return np.stack(rows, axis=0), np.asarray(weights, dtype=np.float32), self_rows


def compute_variant_records(
    *,
    dataset: str,
    variant: str,
    rows_np: np.ndarray,
    weights_np: np.ndarray,
    self_rows: dict[int, list[int]],
    gradients: dict[str, Any],
    target_groups: Sequence[int],
    actual_lookup: dict[int, dict[str, Any]],
    damping: float,
    lanczos_steps: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = torch.as_tensor(rows_np, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    all_mask = np.ones(rows.shape[0], dtype=bool)
    all_matvec = make_fisher_matvec(rows, weights, all_mask, damping=float(damping))
    all_steps = lanczos_steps_for_mask(all_mask, requested=int(lanczos_steps), dim=int(rows.shape[1]))
    records = []
    for target_group in target_groups:
        self_mask = np.zeros(rows.shape[0], dtype=bool)
        self_mask[self_rows[int(target_group)]] = True
        self_matvec = make_fisher_matvec(rows, weights, self_mask, damping=float(damping))
        self_steps = lanczos_steps_for_mask(self_mask, requested=int(lanczos_steps), dim=int(rows.shape[1]))
        loss_grad = torch.as_tensor(gradients["full"][int(target_group)], dtype=torch.float32, device=device)
        trace_self = inverse_quadrature(loss_grad, self_matvec, num_lanczos=self_steps, device=device)
        trace_all = inverse_quadrature(loss_grad, all_matvec, num_lanczos=all_steps, device=device)
        pred_gap = float(trace_self - trace_all)
        loss_osfi = float(pred_gap / max(trace_self, 1e-12))
        actual = actual_lookup[int(target_group)]
        actual_gap = float(actual["actual_loss_gap"])
        records.append(
            {
                "dataset": dataset,
                "variant": variant,
                "target_group": int(target_group),
                "train_group_size": int(actual["train_group_size"]),
                "eval_group_size": int(actual["eval_group_size"]),
                "actual_val_gap": actual_gap,
                "trace_self_inv": float(trace_self),
                "trace_all_inv": float(trace_all),
                "pred_gap": pred_gap,
                "loss_osfi": loss_osfi,
                "sign_agree": bool((loss_osfi > 0.0) == (actual_gap > 0.0)),
            }
        )
    return records


def summarize_variants(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    variants = sorted({str(record["variant"]) for record in records})
    return {variant: summarize_records([record for record in records if record["variant"] == variant]) for variant in variants}


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "num_targets": 0,
            "mean_loss_osfi": float("nan"),
            "loss_osfi_positive_fraction": float("nan"),
            "sign_agreement": float("nan"),
            "spearman_actual": float("nan"),
            "pearson_actual": float("nan"),
        }
    score = np.asarray([float(record["loss_osfi"]) for record in records], dtype=np.float64)
    actual = np.asarray([float(record["actual_val_gap"]) for record in records], dtype=np.float64)
    return {
        "num_targets": int(len(records)),
        "mean_loss_osfi": float(np.mean(score)),
        "loss_osfi_positive_fraction": float(np.mean(score > 0.0)),
        "sign_agreement": float(np.mean((score > 0.0) == (actual > 0.0))),
        "spearman_actual": spearmanr(score, actual) if score.size >= 2 else float("nan"),
        "pearson_actual": pearsonr(score, actual) if score.size >= 2 else float("nan"),
    }


def pearsonr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    x = np.asarray(lhs, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_overall(dataset_outputs: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    records = []
    for output in dataset_outputs:
        records.extend(output.get("records", []))
    return summarize_variants(records)


def fmt(value: Any, precision: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.{precision}g}"


def to_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# Multimodal Inverse Controls",
        "",
        "Scores are L-OSFI values computed with different constructions of `A^{-1}` and compared against actual validation loss gap.",
        "",
        "## Overall",
        "",
        "| variant | targets | mean L-OSFI | positive frac | sign agree | Spearman | Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in sorted(output["overall"].items()):
        lines.append(
            f"| {variant} | {summary['num_targets']} | {fmt(summary['mean_loss_osfi'])} | "
            f"{fmt(summary['loss_osfi_positive_fraction'], 3)} | {fmt(summary['sign_agreement'], 3)} | "
            f"{fmt(summary['spearman_actual'], 3)} | {fmt(summary['pearson_actual'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| dataset | status | variant | targets | mean L-OSFI | sign agree | Spearman | Pearson |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_output in output["datasets"]:
        dataset = dataset_output["dataset"]
        status = dataset_output["status"]
        if not dataset_output.get("variant_summary"):
            lines.append(f"| {dataset} | {status} | n/a | 0 | nan | nan | nan | nan |")
            continue
        for variant, summary in sorted(dataset_output["variant_summary"].items()):
            lines.append(
                f"| {dataset} | {status} | {variant} | {summary['num_targets']} | "
                f"{fmt(summary['mean_loss_osfi'])} | {fmt(summary['sign_agreement'], 3)} | "
                f"{fmt(summary['spearman_actual'], 3)} | {fmt(summary['pearson_actual'], 3)} |"
            )
    lines.extend(
        [
            "",
            "## Target Records",
            "",
            "| dataset | variant | group | train size | eval size | actual val gap | L-OSFI | sign agree |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_output in output["datasets"]:
        for record in dataset_output.get("records", []):
            lines.append(
                f"| {record['dataset']} | {record['variant']} | {record['target_group']} | "
                f"{record['train_group_size']} | {record['eval_group_size']} | "
                f"{fmt(record['actual_val_gap'])} | {fmt(record['loss_osfi'])} | {record['sign_agree']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
