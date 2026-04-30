from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

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
from fisher_geometry.verify_fisher_supply_osfi import (
    direction_variances,
    labels_to_numpy,
    lanczos_steps_for_mask,
    make_label_preserving_random_groups,
    osfi_stats,
    random_orthonormal_basis,
    safe_mean,
    safe_median,
    safe_positive_fraction,
)
from fisher_geometry.verify_humor_fisher_gpt_metrics import (
    DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT,
    DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT,
    build_single_modality_blocks,
    block_id,
    compute_group_gradient_tables,
    inverse_quadrature,
    load_model,
    make_fisher_matvec,
    make_model_args,
    resolve_full_checkpoint,
    resolve_group_selection,
    resolve_requested_device,
    select_named_parameters,
    target_interaction_subspace,
)


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "kmeans_grouped25_subset100_data80_mod80"
)
DEFAULT_OUTPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "fisher_loss_osfi"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Loss-connected OSFI from fisher_test_loss.md. "
            "Uses group-as-sample empirical Fisher H_t and Lanczos inverse-quadrature."
        )
    )
    parser.add_argument("--dataset", default="humor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--full-checkpoint", type=Path, default=None)
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
    parser.add_argument("--target-groups", nargs="*", type=int, default=None)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--target-rank", type=int, default=1)
    parser.add_argument("--lanczos-steps", type=int, default=32)
    parser.add_argument("--random-curvatures", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--skip-label-random", action="store_true")
    args = parser.parse_args()

    device = resolve_requested_device(str(args.device))
    payload_path = args.result_dir / f"{args.dataset}_seed{args.seed}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    config = dict(payload["config"])
    data = load_dataset(
        args.dataset,
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
        args.full_checkpoint,
        roots=args.full_checkpoint_roots,
        dataset=str(args.dataset),
        seed=int(args.seed),
    )

    print(f"Loading checkpoint: {full_checkpoint}", flush=True)
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
    print(
        f"Selected {len(named_params)} tensors / "
        f"{sum(param.numel() for _, param in named_params):,} parameters.",
        flush=True,
    )

    rng = np.random.default_rng(int(args.random_seed))
    print("[kmeans] computing loss-connected OSFI...", flush=True)
    kmeans = compute_loss_osfi_for_grouping(
        "kmeans",
        model=model,
        data=data,
        group_indices=group_indices,
        task=task,
        device=device,
        param_names=param_names,
        target_groups=target_groups,
        damping=float(args.damping),
        target_rank=int(args.target_rank),
        lanczos_steps=int(args.lanczos_steps),
        random_curvatures=int(args.random_curvatures),
        rng=rng,
    )

    controls: dict[str, Any] = {}
    if not args.skip_label_random:
        print("[label_random] computing loss-connected grouping control...", flush=True)
        label_random_groups = make_label_preserving_random_groups(
            group_indices,
            labels=data["train_labels"],
            rng=rng,
        )
        controls["label_random"] = compute_loss_osfi_for_grouping(
            "label_random",
            model=model,
            data=data,
            group_indices=label_random_groups,
            task=task,
            device=device,
            param_names=param_names,
            target_groups=target_groups,
            damping=float(args.damping),
            target_rank=int(args.target_rank),
            lanczos_steps=int(args.lanczos_steps),
            random_curvatures=int(args.random_curvatures),
            rng=rng,
        )

    output = {
        "config": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "payload_path": str(payload_path),
            "result_dir": str(args.result_dir),
            "full_checkpoint": str(full_checkpoint),
            "device": str(device),
            "feature_device": str(args.feature_device or config.get("feature_device", device)),
            "model": model_type,
            "gradient_params": str(args.gradient_params),
            "parameter_names": param_names,
            "num_parameters": int(sum(param.numel() for _, param in named_params)),
            "num_groups": int(len(group_indices)),
            "target_groups": target_groups,
            "damping": float(args.damping),
            "target_rank": int(args.target_rank),
            "lanczos_steps": int(args.lanczos_steps),
            "random_curvatures": int(args.random_curvatures),
            "random_seed": int(args.random_seed),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "unit": "group_as_sample",
            "loss_curvature": "rank1_group_full_loss_empirical_fisher",
        },
        "kmeans": kmeans,
        "controls": controls,
    }
    output["summary"] = summarize_output(output, bootstrap_repeats=int(args.bootstrap_repeats), rng=rng)

    output_json = args.output_json or (args.output_dir / f"{args.dataset}_seed{args.seed}_loss_osfi_cuda1.json")
    output_md = args.output_md or output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


def compute_loss_osfi_for_grouping(
    grouping_name: str,
    *,
    model: torch.nn.Module,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: torch.device,
    param_names: Sequence[str],
    target_groups: Sequence[int],
    damping: float,
    target_rank: int,
    lanczos_steps: int,
    random_curvatures: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    gradients = compute_group_gradient_tables(
        model,
        data=data,
        group_indices=group_indices,
        task=task,
        device=str(device),
        param_names=param_names,
    )
    block_rows, block_weights, _ = build_single_modality_blocks(gradients, group_indices)
    rows = torch.as_tensor(block_rows, dtype=torch.float32, device=device)
    weights = torch.as_tensor(block_weights, dtype=torch.float32, device=device)
    dim = int(rows.shape[1])
    all_mask = np.ones(rows.shape[0], dtype=bool)
    all_matvec = make_fisher_matvec(rows, weights, all_mask, damping=damping)
    all_steps = lanczos_steps_for_mask(all_mask, requested=lanczos_steps, dim=dim)

    records: list[dict[str, Any]] = []
    for target_group in target_groups:
        self_mask = np.zeros(rows.shape[0], dtype=bool)
        self_mask[block_id(target_group, 0)] = True
        self_mask[block_id(target_group, 1)] = True
        self_matvec = make_fisher_matvec(rows, weights, self_mask, damping=damping)
        self_steps = lanczos_steps_for_mask(self_mask, requested=lanczos_steps, dim=dim)

        loss_grad = torch.as_tensor(gradients["full"][target_group], dtype=torch.float32, device=device)
        trace_self = inverse_quadrature(loss_grad, self_matvec, num_lanczos=self_steps, device=device)
        trace_all = inverse_quadrature(loss_grad, all_matvec, num_lanczos=all_steps, device=device)
        pred_gap = float(trace_self - trace_all)
        loss_osfi = float(pred_gap / max(trace_self, 1e-12))

        interaction_basis = target_interaction_subspace(
            torch.as_tensor(gradients["interaction"][target_group], dtype=torch.float32, device=device),
            rank=int(target_rank),
            num_lanczos=min(int(lanczos_steps), dim),
            device=device,
        )
        osfi_self = float(np.sum(direction_variances(interaction_basis, self_matvec, lanczos_steps=self_steps, device=device)))
        osfi_all = float(np.sum(direction_variances(interaction_basis, all_matvec, lanczos_steps=all_steps, device=device)))
        osfi = float((osfi_self - osfi_all) / max(osfi_self, 1e-12))

        h_basis = normalized_vector_as_basis(loss_grad)
        h_self = float(np.sum(direction_variances(h_basis, self_matvec, lanczos_steps=self_steps, device=device)))
        h_all = float(np.sum(direction_variances(h_basis, all_matvec, lanczos_steps=all_steps, device=device)))
        osfi_h = float((h_self - h_all) / max(h_self, 1e-12))

        random_values = []
        for _ in range(int(random_curvatures)):
            random_basis = random_orthonormal_basis(dim, 1, device=device, rng=rng)
            rand_self = float(np.sum(direction_variances(random_basis, self_matvec, lanczos_steps=self_steps, device=device)))
            rand_all = float(np.sum(direction_variances(random_basis, all_matvec, lanczos_steps=all_steps, device=device)))
            random_values.append(float((rand_self - rand_all) / max(rand_self, 1e-12)))
        random_mean = float(np.mean(random_values)) if random_values else float("nan")

        records.append(
            {
                "target_group": int(target_group),
                "trace_h_self_inv": float(trace_self),
                "trace_h_all_inv": float(trace_all),
                "pred_loss_gap": pred_gap,
                "loss_osfi": loss_osfi,
                "osfi": osfi,
                "osfi_h": osfi_h,
                "random_h_loss_osfi_values": random_values,
                "random_h_loss_osfi_mean": random_mean,
                "loss_specific_gain": float(loss_osfi - random_mean),
            }
        )
    return {"grouping": grouping_name, "summary": summarize_grouping(records), "target_records": records}


def normalized_vector_as_basis(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.reshape(-1, 1)
    norm = torch.linalg.norm(vector)
    if float(norm.detach().cpu()) <= 1e-20:
        basis = torch.zeros_like(vector)
        basis[0, 0] = 1.0
        return basis
    return vector / norm


def summarize_grouping(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    loss_osfi = values(records, "loss_osfi")
    pred_gap = values(records, "pred_loss_gap")
    osfi = values(records, "osfi")
    osfi_h = values(records, "osfi_h")
    random_h = values(records, "random_h_loss_osfi_mean")
    loss_specific = values(records, "loss_specific_gain")
    return {
        "mean_loss_osfi": safe_mean(loss_osfi),
        "median_loss_osfi": safe_median(loss_osfi),
        "loss_osfi_positive_fraction": safe_positive_fraction(loss_osfi),
        "mean_pred_loss_gap": safe_mean(pred_gap),
        "pred_loss_gap_positive_fraction": safe_positive_fraction(pred_gap),
        "mean_osfi": safe_mean(osfi),
        "mean_osfi_h": safe_mean(osfi_h),
        "osfi_loss_osfi_spearman": spearmanr(osfi, loss_osfi),
        "osfi_h_loss_osfi_spearman": spearmanr(osfi_h, loss_osfi),
        "osfi_loss_osfi_kendall": kendall_tau(osfi, loss_osfi),
        "osfi_h_loss_osfi_kendall": kendall_tau(osfi_h, loss_osfi),
        "osfi_loss_osfi_sign_agreement": sign_agreement(osfi, loss_osfi),
        "mean_random_h_loss_osfi": safe_mean(random_h),
        "mean_loss_specific_gain": safe_mean(loss_specific),
        "loss_specific_gain_positive_fraction": safe_positive_fraction(loss_specific),
    }


def summarize_output(
    output: dict[str, Any],
    *,
    bootstrap_repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    kmeans = output["kmeans"]
    records = kmeans["target_records"]
    loss_osfi = values(records, "loss_osfi")
    pred_gap = values(records, "pred_loss_gap")
    stats = osfi_stats(loss_osfi, repeats=int(bootstrap_repeats), rng=rng)
    controls = output.get("controls", {})
    control_summary: dict[str, Any] = {}
    if "label_random" in controls:
        label_loss = values(controls["label_random"]["target_records"], "loss_osfi")
        control_summary["label_random_mean_loss_osfi"] = safe_mean(label_loss)
        control_summary["loss_group_gain"] = safe_mean(loss_osfi - label_loss[: loss_osfi.size])
    return {
        "table_1_loss_osfi": {
            **kmeans["summary"],
            "mean_pred_loss_gap": safe_mean(pred_gap),
            **stats,
        },
        "table_2_osfi_vs_loss_osfi": {
            "mean_osfi": kmeans["summary"]["mean_osfi"],
            "mean_loss_osfi": kmeans["summary"]["mean_loss_osfi"],
            "mean_osfi_h": kmeans["summary"]["mean_osfi_h"],
            "osfi_loss_osfi_spearman": kmeans["summary"]["osfi_loss_osfi_spearman"],
            "osfi_h_loss_osfi_spearman": kmeans["summary"]["osfi_h_loss_osfi_spearman"],
            "osfi_loss_osfi_kendall": kmeans["summary"]["osfi_loss_osfi_kendall"],
            "osfi_loss_osfi_sign_agreement": kmeans["summary"]["osfi_loss_osfi_sign_agreement"],
        },
        "table_4_loss_specific_control": {
            "mean_loss_osfi": kmeans["summary"]["mean_loss_osfi"],
            "random_h_mean_loss_osfi": kmeans["summary"]["mean_random_h_loss_osfi"],
            "loss_specific_gain": kmeans["summary"]["mean_loss_specific_gain"],
            "loss_specific_gain_positive_fraction": kmeans["summary"]["loss_specific_gain_positive_fraction"],
        },
        "label_random_control": control_summary,
    }


def values(records: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records], dtype=np.float64)


def safe_positive_fraction(array: np.ndarray) -> float:
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array > 0.0))


def safe_median(array: np.ndarray) -> float:
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.median(array))


def kendall_tau(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    x = np.asarray(list(lhs), dtype=np.float64)
    y = np.asarray(list(rhs), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(x.size):
        for j in range(i + 1, x.size):
            prod = (x[i] - x[j]) * (y[i] - y[j])
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def sign_agreement(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    x = np.asarray(list(lhs), dtype=np.float64)
    y = np.asarray(list(rhs), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.signbit(x) == np.signbit(y)))


def to_markdown(output: dict[str, Any]) -> str:
    cfg = output["config"]
    t1 = output["summary"]["table_1_loss_osfi"]
    t2 = output["summary"]["table_2_osfi_vs_loss_osfi"]
    t4 = output["summary"]["table_4_loss_specific_control"]
    label = output["summary"]["label_random_control"]
    lines = [
        "# Loss-Connected OSFI",
        "",
        f"- dataset: `{cfg['dataset']}`, seed: `{cfg['seed']}`, device: `{cfg['device']}`",
        f"- unit: `{cfg['unit']}`, loss curvature: `{cfg['loss_curvature']}`",
        f"- gradient params: `{cfg['gradient_params']}` ({cfg['num_parameters']:,} scalars)",
        f"- damping: `{cfg['damping']}`, Lanczos steps: `{cfg['lanczos_steps']}`, random curvatures: `{cfg['random_curvatures']}`",
        "",
        "## Table 1: Loss-OSFI Existence",
        "",
        "| Mean L-OSFI | Median L-OSFI | L-OSFI+ | Mean Pred Loss Gap | p-value | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {t1['mean_loss_osfi']:.6g} | {t1['median_loss_osfi']:.6g} | "
        f"{t1['loss_osfi_positive_fraction']:.3f} | {t1['mean_pred_loss_gap']:.6g} | "
        f"{t1['bootstrap_p_value_mean_leq_zero']:.4g} | "
        f"[{t1['mean_osfi_ci_low']:.6g}, {t1['mean_osfi_ci_high']:.6g}] |",
        "",
        "## Table 2: OSFI vs Loss-OSFI",
        "",
        "| Mean OSFI | Mean L-OSFI | Mean OSFI-H | Spearman | OSFI-H Spearman | Kendall | Sign Agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {t2['mean_osfi']:.6g} | {t2['mean_loss_osfi']:.6g} | {t2['mean_osfi_h']:.6g} | "
        f"{t2['osfi_loss_osfi_spearman']:.3f} | {t2['osfi_h_loss_osfi_spearman']:.3f} | "
        f"{t2['osfi_loss_osfi_kendall']:.3f} | {t2['osfi_loss_osfi_sign_agreement']:.3f} |",
        "",
        "## Table 4: Loss-Specific Control",
        "",
        "| Mean L-OSFI | Random-H Mean L-OSFI | LossSpecificGain | Gain+ |",
        "|---:|---:|---:|---:|",
        f"| {t4['mean_loss_osfi']:.6g} | {t4['random_h_mean_loss_osfi']:.6g} | "
        f"{t4['loss_specific_gain']:.6g} | {t4['loss_specific_gain_positive_fraction']:.3f} |",
        "",
        "## Label-Random Control",
        "",
        "| KMeans L-OSFI | Label-Random L-OSFI | L-GroupGain |",
        "|---:|---:|---:|",
        f"| {t1['mean_loss_osfi']:.6g} | {label.get('label_random_mean_loss_osfi', float('nan')):.6g} | "
        f"{label.get('loss_group_gain', float('nan')):.6g} |",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
