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
    resolve_group_selection,
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
DEFAULT_OUTPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "cross_sample_modal_interaction"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interaction-specific cross-sample Fisher test. "
            "Baseline A contains all unimodal group gradients; the tested signal "
            "adds only other-group non-additive residual gradients c_j=h_j-g_j0-g_j1."
        )
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--full-checkpoint-roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT, DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--gradient-params", choices=["last_layer", "fusion", "all"], default="last_layer")
    parser.add_argument("--target-groups", nargs="*", type=int, default=None)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--lanczos-steps", type=int, default=32)
    parser.add_argument("--random-controls", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=2027)
    args = parser.parse_args()

    device = resolve_requested_device(str(args.device))
    rng = np.random.default_rng(int(args.random_seed))
    datasets = []
    for dataset in args.datasets:
        datasets.append(run_dataset(str(dataset), args=args, device=device, rng=rng))

    output = {
        "config": {
            "datasets": [str(value) for value in args.datasets],
            "seed": int(args.seed),
            "result_dir": str(args.result_dir),
            "device": str(device),
            "gradient_params": str(args.gradient_params),
            "damping": float(args.damping),
            "lanczos_steps": int(args.lanczos_steps),
            "random_controls": int(args.random_controls),
            "random_seed": int(args.random_seed),
            "score": (
                "CSMI(z) = (z^T A_add^{-1} z - z^T (A_add + C_other)^-1 z) "
                "/ z^T A_add^{-1} z"
            ),
            "baseline": "A_add contains all groups' unimodal gradients g_j0 and g_j1.",
            "interaction_rows": "C_other contains only other groups' residual gradients c_j=h_j-g_j0-g_j1.",
            "queries": ["loss_gradient_h_t", "interaction_gradient_r_t"],
        },
        "overall": summarize_overall(datasets),
        "datasets": datasets,
    }
    output_json = args.output_json or (args.output_dir / "cross_sample_modal_interaction_all_datasets.json")
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
    target_groups = resolve_group_selection(args.target_groups, len(group_indices))
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
    records = compute_records(
        dataset=dataset,
        gradients=gradients,
        group_indices=group_indices,
        target_groups=target_groups,
        damping=float(args.damping),
        lanczos_steps=int(args.lanczos_steps),
        random_controls=int(args.random_controls),
        device=device,
        rng=rng,
    )
    return {
        "dataset": dataset,
        "status": "ok",
        "num_groups": int(len(group_indices)),
        "target_groups": [int(value) for value in target_groups],
        "num_parameters": int(sum(param.numel() for _, param in named_params)),
        "records": records,
        "summary": summarize_records(records),
    }


def compute_records(
    *,
    dataset: str,
    gradients: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    target_groups: Sequence[int],
    damping: float,
    lanczos_steps: int,
    random_controls: int,
    device: torch.device,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    add_rows_np, add_weights_np = build_additive_rows(gradients, group_indices)
    residual_np, residual_weights_np = build_residual_rows(gradients, group_indices)
    add_rows = torch.as_tensor(add_rows_np, dtype=torch.float32, device=device)
    add_weights = torch.as_tensor(add_weights_np, dtype=torch.float32, device=device)
    residuals = torch.as_tensor(residual_np, dtype=torch.float32, device=device)
    residual_weights = torch.as_tensor(residual_weights_np, dtype=torch.float32, device=device)
    add_mask = np.ones(add_rows.shape[0], dtype=bool)
    base_matvec = make_fisher_matvec(add_rows, add_weights, add_mask, damping=float(damping))
    base_steps = lanczos_steps_for_mask(add_mask, requested=int(lanczos_steps), dim=int(add_rows.shape[1]))

    records: list[dict[str, Any]] = []
    for target_group in target_groups:
        other_mask = np.ones(residuals.shape[0], dtype=bool)
        other_mask[int(target_group)] = False
        rows_real = torch.cat([add_rows, residuals[torch.as_tensor(other_mask, dtype=torch.bool, device=device)]], dim=0)
        weights_real = torch.cat(
            [add_weights, residual_weights[torch.as_tensor(other_mask, dtype=torch.bool, device=device)]], dim=0
        )
        real_mask = np.ones(rows_real.shape[0], dtype=bool)
        real_matvec = make_fisher_matvec(rows_real, weights_real, real_mask, damping=float(damping))
        real_steps = lanczos_steps_for_mask(real_mask, requested=int(lanczos_steps), dim=int(rows_real.shape[1]))

        queries = {
            "loss": np.asarray(gradients["full"][int(target_group)], dtype=np.float32),
            "interaction": np.asarray(gradients["interaction"][int(target_group)], dtype=np.float32),
        }
        for query_name, query_np in queries.items():
            query = torch.as_tensor(query_np, dtype=torch.float32, device=device)
            base_q = inverse_quadrature(query, base_matvec, num_lanczos=base_steps, device=device)
            real_q = inverse_quadrature(query, real_matvec, num_lanczos=real_steps, device=device)
            real_score = float((base_q - real_q) / max(base_q, 1e-12))
            base_mass = float(torch.dot(query, base_matvec(query)).detach().cpu().item())
            real_mass = float(torch.dot(query, real_matvec(query)).detach().cpu().item())
            real_mass_gain = float((real_mass - base_mass) / max(abs(base_mass), 1e-12))
            random_scores = []
            random_mass_gains = []
            for _ in range(int(random_controls)):
                random_rows = norm_matched_random_rows(residuals[torch.as_tensor(other_mask, dtype=torch.bool, device=device)], rng=rng)
                rows_rand = torch.cat([add_rows, random_rows], dim=0)
                rand_matvec = make_fisher_matvec(rows_rand, weights_real, real_mask, damping=float(damping))
                rand_q = inverse_quadrature(query, rand_matvec, num_lanczos=real_steps, device=device)
                random_scores.append(float((base_q - rand_q) / max(base_q, 1e-12)))
                rand_mass = float(torch.dot(query, rand_matvec(query)).detach().cpu().item())
                random_mass_gains.append(float((rand_mass - base_mass) / max(abs(base_mass), 1e-12)))
            random_mean = float(np.mean(random_scores)) if random_scores else float("nan")
            random_std = float(np.std(random_scores)) if random_scores else float("nan")
            random_mass_mean = float(np.mean(random_mass_gains)) if random_mass_gains else float("nan")
            random_mass_std = float(np.std(random_mass_gains)) if random_mass_gains else float("nan")
            p_value = (
                float((1.0 + np.sum(np.asarray(random_scores, dtype=np.float64) >= real_score)) / (len(random_scores) + 1.0))
                if random_scores
                else float("nan")
            )
            mass_p_value = (
                float(
                    (1.0 + np.sum(np.asarray(random_mass_gains, dtype=np.float64) >= real_mass_gain))
                    / (len(random_mass_gains) + 1.0)
                )
                if random_mass_gains
                else float("nan")
            )
            records.append(
                {
                    "dataset": dataset,
                    "target_group": int(target_group),
                    "query": query_name,
                    "base_trace": float(base_q),
                    "real_trace": float(real_q),
                    "csmi": real_score,
                    "base_direct_mass": base_mass,
                    "real_direct_mass": real_mass,
                    "direct_mass_gain": real_mass_gain,
                    "random_direct_mass_gain_mean": random_mass_mean,
                    "random_direct_mass_gain_std": random_mass_std,
                    "direct_mass_gain_specific": float(real_mass_gain - random_mass_mean),
                    "direct_mass_random_p_value_ge_real": mass_p_value,
                    "random_mean_csmi": random_mean,
                    "random_std_csmi": random_std,
                    "interaction_specific_gain": float(real_score - random_mean),
                    "random_p_value_ge_real": p_value,
                    "residual_other_groups": int(np.sum(other_mask)),
                    "residual_norm_mean": float(torch.linalg.norm(residuals[torch.as_tensor(other_mask, dtype=torch.bool, device=device)], dim=1).mean().detach().cpu()),
                }
            )
    return records


def build_additive_rows(gradients: dict[str, Any], group_indices: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    total = float(sum(len(indices) for indices in group_indices))
    rows = []
    weights = []
    for group_id, indices in enumerate(group_indices):
        weight = float(len(indices) / total)
        rows.append(np.asarray(gradients["single"][0][group_id], dtype=np.float32))
        weights.append(weight)
        rows.append(np.asarray(gradients["single"][1][group_id], dtype=np.float32))
        weights.append(weight)
    return np.stack(rows, axis=0), np.asarray(weights, dtype=np.float32)


def build_residual_rows(gradients: dict[str, Any], group_indices: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    total = float(sum(len(indices) for indices in group_indices))
    full = np.asarray(gradients["full"], dtype=np.float32)
    single0 = np.asarray(gradients["single"][0], dtype=np.float32)
    single1 = np.asarray(gradients["single"][1], dtype=np.float32)
    residuals = full - single0 - single1
    weights = np.asarray([float(len(indices) / total) for indices in group_indices], dtype=np.float32)
    return residuals, weights


def norm_matched_random_rows(rows: torch.Tensor, *, rng: np.random.Generator) -> torch.Tensor:
    noise = torch.as_tensor(rng.standard_normal(tuple(rows.shape)), dtype=rows.dtype, device=rows.device)
    noise = torch.nn.functional.normalize(noise, dim=1)
    norms = torch.linalg.norm(rows, dim=1, keepdim=True)
    return noise * norms


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    queries = sorted({str(record["query"]) for record in records})
    return {query: summarize_query([record for record in records if str(record["query"]) == query]) for query in queries}


def summarize_query(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "num_targets": 0,
            "mean_csmi": float("nan"),
            "csmi_positive_fraction": float("nan"),
            "mean_random_csmi": float("nan"),
            "mean_interaction_specific_gain": float("nan"),
            "gain_positive_fraction": float("nan"),
            "mean_random_p_value_ge_real": float("nan"),
        }
    csmi = np.asarray([float(record["csmi"]) for record in records], dtype=np.float64)
    random_mean = np.asarray([float(record["random_mean_csmi"]) for record in records], dtype=np.float64)
    gain = np.asarray([float(record["interaction_specific_gain"]) for record in records], dtype=np.float64)
    p_values = np.asarray([float(record["random_p_value_ge_real"]) for record in records], dtype=np.float64)
    direct_mass = np.asarray([float(record["direct_mass_gain"]) for record in records], dtype=np.float64)
    random_direct_mass = np.asarray([float(record["random_direct_mass_gain_mean"]) for record in records], dtype=np.float64)
    direct_specific = np.asarray([float(record["direct_mass_gain_specific"]) for record in records], dtype=np.float64)
    direct_p_values = np.asarray([float(record["direct_mass_random_p_value_ge_real"]) for record in records], dtype=np.float64)
    return {
        "num_targets": int(len(records)),
        "mean_csmi": finite_mean(csmi),
        "median_csmi": finite_median(csmi),
        "csmi_positive_fraction": positive_fraction(csmi),
        "mean_random_csmi": finite_mean(random_mean),
        "mean_interaction_specific_gain": finite_mean(gain),
        "gain_positive_fraction": positive_fraction(gain),
        "mean_random_p_value_ge_real": finite_mean(p_values),
        "mean_direct_mass_gain": finite_mean(direct_mass),
        "mean_random_direct_mass_gain": finite_mean(random_direct_mass),
        "mean_direct_mass_specific_gain": finite_mean(direct_specific),
        "direct_mass_specific_positive_fraction": positive_fraction(direct_specific),
        "mean_direct_mass_random_p_value_ge_real": finite_mean(direct_p_values),
    }


def summarize_overall(dataset_outputs: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    records = []
    for output in dataset_outputs:
        records.extend(output.get("records", []))
    return summarize_records(records)


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def positive_fraction(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values > 0.0)) if values.size else float("nan")


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
        "# Cross-Sample Modal Interaction Residual Test",
        "",
        "Baseline `A_add` uses all groups' unimodal gradients. The tested signal adds only other groups' non-additive residual gradients `c_j = h_j - g_j0 - g_j1`.",
        "",
        "## Overall",
        "",
        "| query | targets | mean CSMI | CSMI+ | random mean | interaction gain | gain+ | random p>=real |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for query, summary in sorted(output["overall"].items()):
        lines.append(summary_row(query, summary))
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| dataset | query | targets | mean CSMI | CSMI+ | random mean | interaction gain | gain+ | random p>=real |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_output in output["datasets"]:
        dataset = dataset_output["dataset"]
        for query, summary in sorted(dataset_output["summary"].items()):
            lines.append(summary_row(query, summary, dataset=dataset))
    lines.extend(
        [
            "",
            "## Direct Fisher Mass Control",
            "",
            "`direct mass gain` is `(z^T (A_add + C_other) z - z^T A_add z) / |z^T A_add z|`.",
            "",
            "| query | targets | direct mass gain | random direct mass | specific gain | specific+ | random p>=real |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for query, summary in sorted(output["overall"].items()):
        lines.append(
            f"| {query} | {summary['num_targets']} | {fmt(summary['mean_direct_mass_gain'])} | "
            f"{fmt(summary['mean_random_direct_mass_gain'])} | {fmt(summary['mean_direct_mass_specific_gain'])} | "
            f"{fmt(summary['direct_mass_specific_positive_fraction'], 3)} | "
            f"{fmt(summary['mean_direct_mass_random_p_value_ge_real'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive CSMI means other groups' non-additive residual modal gradients reduce the target query inverse-curvature proxy beyond an all-unimodal baseline.",
            "- Positive interaction gain means the real residual rows beat norm-matched random residual rows, controlling for generic PSD mass and residual norms.",
            "- Direct Fisher mass reports raw target-direction curvature increase. It is a useful alignment control but does not condition on what the additive unimodal baseline already constrains.",
            "- This is a stronger cross-sample modal-interaction signal than all-vs-self or multimodal-vs-unimodal monotonicity, but it is still based on the residual proxy `h_j - g_j0 - g_j1`.",
        ]
    )
    return "\n".join(lines) + "\n"


def summary_row(query: str, summary: dict[str, float], *, dataset: str | None = None) -> str:
    prefix = f"| {dataset} | {query} |" if dataset is not None else f"| {query} |"
    return (
        f"{prefix} {int(summary['num_targets'])} | {fmt(summary['mean_csmi'])} | "
        f"{fmt(summary['csmi_positive_fraction'], 3)} | {fmt(summary['mean_random_csmi'])} | "
        f"{fmt(summary['mean_interaction_specific_gain'])} | {fmt(summary['gain_positive_fraction'], 3)} | "
        f"{fmt(summary['mean_random_p_value_ge_real'], 3)} |"
    )


if __name__ == "__main__":
    main()
