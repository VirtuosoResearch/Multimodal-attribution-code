from __future__ import annotations

import argparse
import json
import math
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
    subspace_inverse_variance,
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
DEFAULT_OUTPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "fisher_supply_osfi"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Supplementary OSFI experiment from fisher_test_gpt_supply.md. "
            "Groups are the sample units; inverse-Fisher variances are estimated "
            "with Lanczos/SLQ quadrature."
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
    parser.add_argument("--random-subspaces", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--skip-controls", action="store_true")
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
    print("[kmeans] computing OSFI metrics...", flush=True)
    kmeans_result = compute_grouping_osfi(
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
        random_subspaces=int(args.random_subspaces),
        rng=rng,
    )

    controls: dict[str, Any] = {}
    if not args.skip_controls:
        sizes = [len(indices) for indices in group_indices]
        print("[random] computing same-size random grouping control...", flush=True)
        random_groups = make_random_groups(len(data["train_labels"]), sizes, rng=rng)
        controls["random"] = compute_grouping_osfi(
            "random",
            model=model,
            data=data,
            group_indices=random_groups,
            task=task,
            device=device,
            param_names=param_names,
            target_groups=target_groups,
            damping=float(args.damping),
            target_rank=int(args.target_rank),
            lanczos_steps=int(args.lanczos_steps),
            random_subspaces=int(args.random_subspaces),
            rng=rng,
        )
        print("[label_random] computing label-preserving random grouping control...", flush=True)
        label_random_groups = make_label_preserving_random_groups(
            group_indices,
            labels=data["train_labels"],
            rng=rng,
        )
        controls["label_random"] = compute_grouping_osfi(
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
            random_subspaces=int(args.random_subspaces),
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
            "random_subspaces": int(args.random_subspaces),
            "random_seed": int(args.random_seed),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "unit": "group_as_sample",
        },
        "kmeans": kmeans_result,
        "controls": controls,
    }
    output["summary"] = summarize_output(output, bootstrap_repeats=int(args.bootstrap_repeats), rng=rng)

    output_json = args.output_json or (args.output_dir / f"{args.dataset}_seed{args.seed}_supply_osfi_cuda1.json")
    output_md = args.output_md or output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


def compute_grouping_osfi(
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
    random_subspaces: int,
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
    variance_cache: dict[tuple[tuple[int, ...], int], float] = {}

    def variance_for_basis(basis: torch.Tensor, include_mask: np.ndarray) -> float:
        mask_key = tuple(np.flatnonzero(include_mask).astype(int).tolist())
        basis_key = id(basis)
        key = (mask_key, basis_key)
        cached = variance_cache.get(key)
        if cached is not None:
            return cached
        matvec = make_fisher_matvec(rows, weights, include_mask, damping=damping)
        steps = lanczos_steps_for_mask(include_mask, requested=lanczos_steps, dim=dim)
        value = subspace_inverse_variance(
            basis,
            matvec,
            num_lanczos=steps,
            device=device,
        )
        variance_cache[key] = value
        return value

    target_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    for target_group in target_groups:
        target_basis = target_interaction_subspace(
            torch.as_tensor(gradients["interaction"][target_group], dtype=torch.float32, device=device),
            rank=int(target_rank),
            num_lanczos=min(int(lanczos_steps), dim),
            device=device,
        )
        all_variances = direction_variances(
            target_basis,
            all_matvec,
            lanczos_steps=lanczos_steps_for_mask(all_mask, requested=lanczos_steps, dim=dim),
            device=device,
        )
        self_mask = np.zeros(rows.shape[0], dtype=bool)
        self_mask[block_id(target_group, 0)] = True
        self_mask[block_id(target_group, 1)] = True
        self_matvec = make_fisher_matvec(rows, weights, self_mask, damping=damping)
        self_variances = direction_variances(
            target_basis,
            self_matvec,
            lanczos_steps=lanczos_steps_for_mask(self_mask, requested=lanczos_steps, dim=dim),
            device=device,
        )
        v_all = float(np.sum(all_variances))
        v_self = float(np.sum(self_variances))
        directional = (self_variances - all_variances) / np.maximum(self_variances, 1e-12)
        osfi = float((v_self - v_all) / max(v_self, 1e-12))
        random_osfi_values = []
        for _ in range(int(random_subspaces)):
            rand_basis = random_orthonormal_basis(dim, target_basis.shape[1], device=device, rng=rng)
            v_rand_self = variance_for_basis(rand_basis, self_mask)
            v_rand_all = variance_for_basis(rand_basis, all_mask)
            random_osfi_values.append(float((v_rand_self - v_rand_all) / max(v_rand_self, 1e-12)))
        random_osfi_mean = float(np.mean(random_osfi_values)) if random_osfi_values else float("nan")
        target_rows.append(
            {
                "target_group": int(target_group),
                "v_self": v_self,
                "v_all": v_all,
                "delta_v_other": float(v_self - v_all),
                "osfi": osfi,
                "directional_osfi": [float(value) for value in directional.tolist()],
                "directional_osfi_mean": float(np.mean(directional)),
                "directional_osfi_max": float(np.max(directional)),
                "directional_osfi_positive_fraction": float(np.mean(directional > 0.0)),
                "random_osfi_values": random_osfi_values,
                "random_osfi_mean": random_osfi_mean,
                "target_specific_osfi": float(osfi - random_osfi_mean),
            }
        )
        for source_group in range(len(group_indices)):
            if source_group == target_group:
                continue
            without_source = all_mask.copy()
            without_source[block_id(source_group, 0)] = False
            without_source[block_id(source_group, 1)] = False
            v_without_source = variance_for_basis(target_basis, without_source)
            source_impact = (v_without_source - v_all) / max(v_without_source, 1e-12)
            source_rows.append(
                {
                    "target_group": int(target_group),
                    "source_group": int(source_group),
                    "v_without_source": float(v_without_source),
                    "v_all": v_all,
                    "source_impact": float(source_impact),
                }
            )
            for modality in (0, 1):
                without_modality = all_mask.copy()
                without_modality[block_id(source_group, modality)] = False
                v_without_modality = variance_for_basis(target_basis, without_modality)
                modality_impact = (v_without_modality - v_all) / max(v_without_modality, 1e-12)
                modality_rows.append(
                    {
                        "target_group": int(target_group),
                        "source_group": int(source_group),
                        "source_modality": int(modality),
                        "v_without_modality": float(v_without_modality),
                        "v_all": v_all,
                        "modality_impact": float(modality_impact),
                    }
                )

    return {
        "grouping": grouping_name,
        "summary": summarize_grouping(target_rows, source_rows, modality_rows),
        "target_records": target_rows,
        "source_records": source_rows,
        "modality_records": modality_rows,
    }


def direction_variances(
    basis: torch.Tensor,
    matvec,
    *,
    lanczos_steps: int,
    device: torch.device,
) -> np.ndarray:
    values = []
    for col in range(basis.shape[1]):
        values.append(
            inverse_quadrature(
                basis[:, col],
                matvec,
                num_lanczos=min(int(lanczos_steps), int(basis.shape[0])),
                device=device,
            )
        )
    return np.asarray(values, dtype=np.float64)


def lanczos_steps_for_mask(include_mask: np.ndarray, *, requested: int, dim: int) -> int:
    # The group-as-sample Fisher has rank at most active_blocks plus damping.
    # Using active_blocks + 1 Lanczos steps keeps the computation SLQ/Lanczos
    # while making inverse-quadrature nearly exact for these low-rank operators.
    active_blocks = int(np.sum(include_mask))
    return max(1, min(int(dim), max(int(requested), active_blocks + 1)))


def random_orthonormal_basis(
    dim: int,
    rank: int,
    *,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    matrix = torch.as_tensor(rng.standard_normal((int(dim), int(rank))), dtype=torch.float32, device=device)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, : int(rank)].contiguous()


def make_random_groups(num_samples: int, sizes: Sequence[int], *, rng: np.random.Generator) -> list[list[int]]:
    order = rng.permutation(int(num_samples))
    groups = []
    start = 0
    for size in sizes:
        end = start + int(size)
        groups.append([int(value) for value in order[start:end].tolist()])
        start = end
    return groups


def make_label_preserving_random_groups(
    group_indices: Sequence[Sequence[int]],
    *,
    labels: torch.Tensor,
    rng: np.random.Generator,
) -> list[list[int]]:
    num_samples = int(labels.shape[0])
    original_group = np.empty(num_samples, dtype=np.int64)
    for group_id, indices in enumerate(group_indices):
        original_group[np.asarray(indices, dtype=np.int64)] = int(group_id)
    new_group = original_group.copy()
    label_values = labels_to_numpy(labels)
    for label in np.unique(label_values):
        label_mask = np.flatnonzero(label_values == label)
        permuted = rng.permutation(new_group[label_mask])
        new_group[label_mask] = permuted
    groups = []
    for group_id in range(len(group_indices)):
        groups.append([int(value) for value in np.flatnonzero(new_group == group_id).tolist()])
    return groups


def labels_to_numpy(labels: torch.Tensor) -> np.ndarray:
    values = labels.detach().cpu()
    if values.ndim == 1:
        return values.numpy().astype(np.int64)
    if values.ndim == 2 and values.shape[1] == 1:
        return values.view(-1).numpy().astype(np.int64)
    return values.argmax(dim=1).numpy().astype(np.int64)


def summarize_grouping(
    target_rows: Sequence[dict[str, Any]],
    source_rows: Sequence[dict[str, Any]],
    modality_rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
    osfi = values(target_rows, "osfi")
    target_specific = values(target_rows, "target_specific_osfi")
    delta_v = values(target_rows, "delta_v_other")
    source = values(source_rows, "source_impact")
    modality = values(modality_rows, "modality_impact")
    return {
        "mean_osfi": safe_mean(osfi),
        "median_osfi": safe_median(osfi),
        "osfi_positive_fraction": safe_positive_fraction(osfi),
        "mean_target_specific_osfi": safe_mean(target_specific),
        "target_specific_positive_fraction": safe_positive_fraction(target_specific),
        "mean_delta_v_other": safe_mean(delta_v),
        "other_effect_fraction": safe_positive_fraction(delta_v),
        "mean_source_impact": safe_mean(source),
        "source_impact_positive_fraction": safe_positive_fraction(source),
        "mean_modality_impact": safe_mean(modality),
        "modality_impact_positive_fraction": safe_positive_fraction(modality),
    }


def summarize_output(
    output: dict[str, Any],
    *,
    bootstrap_repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    kmeans = output["kmeans"]
    osfi = values(kmeans["target_records"], "osfi")
    stats = osfi_stats(osfi, repeats=int(bootstrap_repeats), rng=rng)
    controls = output.get("controls", {})
    control_summary: dict[str, Any] = {}
    if "random" in controls:
        random_osfi = values(controls["random"]["target_records"], "osfi")
        control_summary["random_mean_osfi"] = safe_mean(random_osfi)
        control_summary["group_structure_gain"] = safe_mean(osfi - random_osfi[: osfi.size])
    if "label_random" in controls:
        label_random_osfi = values(controls["label_random"]["target_records"], "osfi")
        control_summary["label_random_mean_osfi"] = safe_mean(label_random_osfi)
        control_summary["semantic_group_gain"] = safe_mean(osfi - label_random_osfi[: osfi.size])
    return {
        "table_a": {
            **kmeans["summary"],
            **stats,
        },
        "table_b_controls": control_summary,
        "table_c_source_level": {
            "source_impact_positive_fraction": kmeans["summary"]["source_impact_positive_fraction"],
            "modality_impact_positive_fraction": kmeans["summary"]["modality_impact_positive_fraction"],
            "mean_source_impact": kmeans["summary"]["mean_source_impact"],
            "mean_modality_impact": kmeans["summary"]["mean_modality_impact"],
        },
    }


def osfi_stats(array: np.ndarray, *, repeats: int, rng: np.random.Generator) -> dict[str, float]:
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "mean_osfi_ci_low": float("nan"),
            "mean_osfi_ci_high": float("nan"),
            "bootstrap_p_value_mean_leq_zero": float("nan"),
            "wilcoxon_or_sign_p_value": float("nan"),
        }
    means = []
    for _ in range(max(int(repeats), 1)):
        sample = array[rng.integers(0, array.size, size=array.size)]
        means.append(float(np.mean(sample)))
    means_np = np.asarray(means, dtype=np.float64)
    return {
        "mean_osfi_ci_low": float(np.quantile(means_np, 0.025)),
        "mean_osfi_ci_high": float(np.quantile(means_np, 0.975)),
        "bootstrap_p_value_mean_leq_zero": float((np.sum(means_np <= 0.0) + 1.0) / (means_np.size + 1.0)),
        "wilcoxon_or_sign_p_value": one_sided_wilcoxon_or_sign(array),
    }


def one_sided_wilcoxon_or_sign(array: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(array, alternative="greater", zero_method="wilcox")
        return float(result.pvalue)
    except Exception:
        positives = int(np.sum(array > 0.0))
        nonzero = int(np.sum(array != 0.0))
        if nonzero == 0:
            return 1.0
        return float(sum(math.comb(nonzero, k) for k in range(positives, nonzero + 1)) / (2**nonzero))


def values(records: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records], dtype=np.float64)


def safe_mean(array: Iterable[float] | np.ndarray) -> float:
    values_np = np.asarray(list(array) if not isinstance(array, np.ndarray) else array, dtype=np.float64)
    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        return float("nan")
    return float(np.mean(values_np))


def safe_median(array: np.ndarray) -> float:
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.median(array))


def safe_positive_fraction(array: np.ndarray) -> float:
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array > 0.0))


def to_markdown(output: dict[str, Any]) -> str:
    cfg = output["config"]
    table_a = output["summary"]["table_a"]
    table_b = output["summary"]["table_b_controls"]
    table_c = output["summary"]["table_c_source_level"]
    lines = [
        "# Supplementary OSFI Experiment",
        "",
        f"- dataset: `{cfg['dataset']}`, seed: `{cfg['seed']}`, device: `{cfg['device']}`",
        f"- unit: `{cfg['unit']}`, groups: `{cfg['num_groups']}`, target rank: `{cfg['target_rank']}`",
        f"- gradient params: `{cfg['gradient_params']}` ({cfg['num_parameters']:,} scalars)",
        f"- damping: `{cfg['damping']}`, Lanczos steps: `{cfg['lanczos_steps']}`, random subspaces: `{cfg['random_subspaces']}`",
        "",
        "## Table A: Other-Sample Fisher Impact",
        "",
        "| Mean OSFI | Median OSFI | OSFI+ | Mean TargetSpecificOSFI | Bootstrap p | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {table_a['mean_osfi']:.6g} | {table_a['median_osfi']:.6g} | "
        f"{table_a['osfi_positive_fraction']:.3f} | {table_a['mean_target_specific_osfi']:.6g} | "
        f"{table_a['bootstrap_p_value_mean_leq_zero']:.4g} | "
        f"[{table_a['mean_osfi_ci_low']:.6g}, {table_a['mean_osfi_ci_high']:.6g}] |",
        "",
        "## Table B: Controls",
        "",
        "| KMeans OSFI | Random OSFI | Label-Random OSFI | GroupStructureGain | SemanticGroupGain |",
        "|---:|---:|---:|---:|---:|",
        f"| {table_a['mean_osfi']:.6g} | {table_b.get('random_mean_osfi', float('nan')):.6g} | "
        f"{table_b.get('label_random_mean_osfi', float('nan')):.6g} | "
        f"{table_b.get('group_structure_gain', float('nan')):.6g} | "
        f"{table_b.get('semantic_group_gain', float('nan')):.6g} |",
        "",
        "## Table C: Source-Level Effect",
        "",
        "| SourceImpact+ | ModalityImpact+ | Mean SourceImpact | Mean ModalityImpact |",
        "|---:|---:|---:|---:|",
        f"| {table_c['source_impact_positive_fraction']:.3f} | "
        f"{table_c['modality_impact_positive_fraction']:.3f} | "
        f"{table_c['mean_source_impact']:.6g} | {table_c['mean_modality_impact']:.6g} |",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
