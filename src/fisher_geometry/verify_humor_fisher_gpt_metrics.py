from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
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
from fisher_geometry.experiment import build_model
from fisher_geometry.fisher_metrics import compute_loss
from kernel_surrogate.kernel_attribution import spearmanr
from kernel_surrogate.real_data_cmis import make_model_args, resolve_device
from src.slq.slq import SLQ


DEFAULT_RESULT_DIR = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_results"
    / "lds_real_train"
    / "kmeans_grouped25_subset100_data80_mod80"
)
DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_checkpoints"
    / "lds_real_train"
    / "kmeans_grouped25_full_data_train"
)
DEFAULT_GROUPED_FULL_CHECKPOINT_ROOT = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "subset_checkpoints"
    / "lds_real_train"
    / "grouped25_full_data_train"
)
DEFAULT_FULL_CHECKPOINT = (
    DEFAULT_KMEANS_FULL_CHECKPOINT_ROOT
    / "humor"
    / "seed0"
    / "subset001"
    / "checkpoints"
    / "checkpoint_repeat01.pt"
)
DEFAULT_OUTPUT_JSON = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "verification_results"
    / "humor_fisher_gpt_metrics"
    / "humor_seed0_fisher_gpt_metrics.json"
)
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_JSON.with_suffix(".md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group-as-sample implementation of fisher_test_gpt.md. "
            "Each k-means group contributes one mean-gradient sample per modality. "
            "Target inverse-Fisher variances and natural alignments are estimated "
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
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--gradient-params", choices=["last_layer", "fusion", "all"], default="last_layer")
    parser.add_argument("--target-groups", nargs="*", type=int, default=None)
    parser.add_argument("--source-groups", nargs="*", type=int, default=None)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--target-rank", type=int, default=1)
    parser.add_argument("--lanczos-steps", type=int, default=32)
    parser.add_argument("--global-eig-rank", type=int, default=32)
    parser.add_argument("--global-eig-lanczos", type=int, default=64)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--top-records", type=int, default=12)
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
    num_groups = len(group_indices)
    target_groups = resolve_group_selection(args.target_groups, num_groups)
    source_groups = resolve_group_selection(args.source_groups, num_groups)

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

    print("Computing group-as-sample gradients for single/full/interaction objectives...", flush=True)
    gradients = compute_group_gradient_tables(
        model,
        data=data,
        group_indices=group_indices,
        task=task,
        device=str(device),
        param_names=param_names,
    )
    block_rows, block_weights, block_meta = build_single_modality_blocks(gradients, group_indices)
    metric_rows = torch.as_tensor(block_rows, dtype=torch.float32, device=device)
    metric_weights = torch.as_tensor(block_weights, dtype=torch.float32, device=device)
    damping = float(args.damping)
    if damping <= 0.0:
        raise ValueError("--damping must be positive")

    print("Approximating global Fisher eigenspace with SLQ...", flush=True)
    all_mask = np.ones(metric_rows.shape[0], dtype=bool)
    all_matvec = make_fisher_matvec(metric_rows, metric_weights, all_mask, damping=damping)
    global_evals, global_evecs = SLQ(
        matvec=all_matvec,
        dim=int(metric_rows.shape[1]),
        device=device,
    ).top_k_eigen(
        min(int(args.global_eig_rank), int(metric_rows.shape[1])),
        num_lanczos=min(int(args.global_eig_lanczos), int(metric_rows.shape[1])),
        reorthogonalize=True,
    )

    print("Computing target interaction subspaces with SLQ...", flush=True)
    target_subspaces = {
        target_group: target_interaction_subspace(
            torch.as_tensor(gradients["interaction"][target_group], dtype=torch.float32, device=device),
            rank=int(args.target_rank),
            num_lanczos=min(int(args.lanczos_steps), int(metric_rows.shape[1])),
            device=device,
        )
        for target_group in target_groups
    }
    inverse_sqrt = make_global_invsqrt(
        global_evals.detach(),
        global_evecs.detach(),
        damping=damping,
    )
    variance_cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def variance(target_group: int, excluded_blocks: Iterable[int]) -> float:
        key = (int(target_group), tuple(sorted(int(index) for index in excluded_blocks)))
        cached = variance_cache.get(key)
        if cached is not None:
            return cached
        include_mask = np.ones(metric_rows.shape[0], dtype=bool)
        if key[1]:
            include_mask[np.asarray(key[1], dtype=np.int64)] = False
        matvec = make_fisher_matvec(metric_rows, metric_weights, include_mask, damping=damping)
        value = subspace_inverse_variance(
            target_subspaces[target_group],
            matvec,
            num_lanczos=min(int(args.lanczos_steps), int(metric_rows.shape[1])),
            device=device,
        )
        variance_cache[key] = value
        return value

    print("Computing TFC/FVC/NIH/F-CSMI/MMEx records...", flush=True)
    records: list[dict[str, Any]] = []
    fcsm_records: list[dict[str, Any]] = []
    mmex_records: list[dict[str, Any]] = []
    v_all_by_target = {target_group: variance(target_group, []) for target_group in target_groups}
    denom_by_target = {
        target_group: fisher_trace_on_subspace(
            target_subspaces[target_group],
            metric_rows,
            metric_weights,
            all_mask,
        )
        for target_group in target_groups
    }

    for target_group in target_groups:
        u = target_subspaces[target_group]
        target_int = torch.as_tensor(gradients["interaction"][target_group], dtype=torch.float32, device=device)
        v_all = v_all_by_target[target_group]
        for source_group in source_groups:
            if source_group == target_group:
                continue
            for source_modality in (0, 1):
                source_block = block_id(source_group, source_modality)
                source_grad = metric_rows[source_block]
                v_without = variance(target_group, [source_block])
                fvc = (v_without - v_all) / (v_without + 1e-12)
                block_tfc = fisher_trace_on_subspace(
                    u,
                    metric_rows,
                    metric_weights,
                    np.eye(metric_rows.shape[0], dtype=bool)[source_block],
                )
                tfc = block_tfc / (denom_by_target[target_group] + 1e-12)
                z_u = torch.stack([inverse_sqrt(u[:, i]) for i in range(u.shape[1])], dim=1)
                ntfc = fisher_trace_on_subspace(
                    z_u,
                    metric_rows,
                    metric_weights,
                    np.eye(metric_rows.shape[0], dtype=bool)[source_block],
                ) / max(int(u.shape[1]), 1)
                nih = bilinear_inverse_quadrature(
                    target_int,
                    source_grad,
                    all_matvec,
                    num_lanczos=min(int(args.lanczos_steps), int(metric_rows.shape[1])),
                    device=device,
                )
                target_nat_norm = inverse_quadrature(
                    target_int,
                    all_matvec,
                    num_lanczos=min(int(args.lanczos_steps), int(metric_rows.shape[1])),
                    device=device,
                )
                source_nat_norm = inverse_quadrature(
                    source_grad,
                    all_matvec,
                    num_lanczos=min(int(args.lanczos_steps), int(metric_rows.shape[1])),
                    device=device,
                )
                nalign = nih / (math.sqrt(max(target_nat_norm, 0.0)) * math.sqrt(max(source_nat_norm, 0.0)) + 1e-12)
                records.append(
                    {
                        "target_group": int(target_group),
                        "source_group": int(source_group),
                        "source_modality": int(source_modality),
                        "source_block": int(source_block),
                        "tfc": float(tfc),
                        "ntfc": float(ntfc),
                        "fvc": float(fvc),
                        "nih": float(nih),
                        "nalign": float(nalign),
                        "v_all": float(v_all),
                        "v_without_source": float(v_without),
                    }
                )

        for source_group in source_groups:
            if source_group == target_group:
                continue
            mmex = source_mmex(
                target_subspaces[target_group],
                source_group=source_group,
                gradients=gradients,
                group_indices=group_indices,
                inverse_sqrt=inverse_sqrt,
                device=device,
            )
            mmex_records.append(
                {
                    "target_group": int(target_group),
                    "source_group": int(source_group),
                    "mmex": float(mmex),
                }
            )

        for target_modality in (0, 1):
            target_block = block_id(target_group, target_modality)
            v_without_target = variance(target_group, [target_block])
            for source_group in source_groups:
                if source_group == target_group:
                    continue
                for source_modality in (0, 1):
                    source_block = block_id(source_group, source_modality)
                    v_without_source = variance(target_group, [source_block])
                    v_without_pair = variance(target_group, [target_block, source_block])
                    fcsm = (v_without_source - v_all) - (v_without_pair - v_without_target)
                    fcsm_records.append(
                        {
                            "target_group": int(target_group),
                            "target_modality": int(target_modality),
                            "source_group": int(source_group),
                            "source_modality": int(source_modality),
                            "target_block": int(target_block),
                            "source_block": int(source_block),
                            "f_csmi": float(fcsm),
                            "v_all": float(v_all),
                            "v_without_target": float(v_without_target),
                            "v_without_source": float(v_without_source),
                            "v_without_pair": float(v_without_pair),
                        }
                    )

    lds_effects = estimate_lds_modality_effects(payload, ridge_lambda=float(args.ridge_lambda))
    attach_source_lds(records, lds_effects)
    attach_source_lds(fcsm_records, lds_effects)
    summary = summarize(records, fcsm_records, mmex_records, lds_effects, top_records=int(args.top_records))
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
            "num_parameters": int(metric_rows.shape[1]),
            "num_groups": int(num_groups),
            "target_groups": target_groups,
            "source_groups": source_groups,
            "damping": float(damping),
            "target_rank": int(next(iter(target_subspaces.values())).shape[1]) if target_subspaces else 0,
            "lanczos_steps": int(args.lanczos_steps),
            "global_eig_rank": int(global_evecs.shape[1]),
            "unit": "group_as_sample",
            "fisher_blocks": "single_modality_group_mean_grad_outer_products",
        },
        "global_fisher_slq": {
            "top_eigenvalues": [float(value) for value in global_evals.detach().cpu().tolist()],
        },
        "gradient_stats": gradient_stats(gradients),
        "lds_effect_model": lds_effects["summary"],
        "summary": summary,
        "records": records,
        "f_csmi_records": fcsm_records,
        "mmex_records": mmex_records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {args.output_json}", flush=True)
    if args.output_md is not None:
        print(f"Wrote {args.output_md}", flush=True)


def resolve_requested_device(requested: str) -> torch.device:
    if requested.startswith("cuda"):
        try:
            device = torch.device(requested)
            torch.cuda.get_device_properties(device)
            return device
        except Exception as exc:
            print(f"CUDA probe failed for {requested}: {exc}; falling back.", flush=True)
    return resolve_device(requested)


def resolve_group_selection(values: Sequence[int] | None, num_groups: int) -> list[int]:
    if values is None or len(values) == 0:
        return list(range(num_groups))
    groups = sorted({int(value) for value in values})
    bad = [value for value in groups if value < 0 or value >= num_groups]
    if bad:
        raise ValueError(f"group id(s) out of range 0..{num_groups - 1}: {bad}")
    return groups


def resolve_full_checkpoint(
    explicit: Path | None,
    *,
    roots: Sequence[Path],
    dataset: str,
    seed: int,
) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"explicit full checkpoint does not exist: {explicit}")
        return explicit
    candidates = [
        Path(root)
        / dataset
        / f"seed{seed}"
        / "subset001"
        / "checkpoints"
        / "checkpoint_repeat01.pt"
        for root in roots
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve full checkpoint. Tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def load_model(
    checkpoint_path: Path,
    *,
    model_type: str,
    dim_x: int,
    dim_y: int,
    num_classes: int,
    model_args: SimpleNamespace,
    device: str,
) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    model = build_model(model_type, dim_x, dim_y, num_classes, model_args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def select_named_parameters(model: torch.nn.Module, *, mode: str) -> list[tuple[str, torch.nn.Parameter]]:
    if mode == "all":
        named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    elif mode == "last_layer":
        named = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and name.startswith("classifier.")
        ]
    elif mode == "fusion":
        prefixes = ("layers.", "pool_norm.", "classifier.", "cls_x", "cls_y", "null_embed")
        named = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and (name.startswith(prefixes) or "fusion" in name or "shared_mlp" in name)
        ]
    else:
        raise ValueError(f"unsupported parameter mode: {mode}")
    if not named:
        raise ValueError(f"no trainable parameters selected for mode={mode!r}")
    return named


def parameters_by_name(model: torch.nn.Module, param_names: Sequence[str]) -> list[torch.nn.Parameter]:
    lookup = dict(model.named_parameters())
    return [lookup[name] for name in param_names]


def flatten_grads(
    grads: Sequence[torch.Tensor | None],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    chunks = []
    for grad_value, param in zip(grads, params):
        if grad_value is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float32, device=param.device))
        else:
            chunks.append(grad_value.detach().reshape(-1).float())
    return torch.cat(chunks, dim=0).detach().cpu()


def compute_objective_gradient(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    indices: Sequence[int],
    task: str,
    device: str,
    param_names: Sequence[str],
    objective: str,
    modality: int | None = None,
) -> tuple[torch.Tensor, float]:
    params = parameters_by_name(model, param_names)
    index = torch.as_tensor(indices, dtype=torch.long)
    bx = data["train_x"][index].to(device)
    by = data["train_y"][index].to(device)
    bl = data["train_labels"][index].to(device)
    model.zero_grad(set_to_none=True)
    if objective == "single":
        if modality not in (0, 1):
            raise ValueError("single objective requires modality 0 or 1")
        loss = compute_loss(task, model(bx, by, use_x=modality == 0, use_y=modality == 1), bl)
    elif objective == "full":
        loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
    elif objective == "interaction":
        full_loss = compute_loss(task, model(bx, by, use_x=True, use_y=True), bl)
        x_loss = compute_loss(task, model(bx, by, use_x=True, use_y=False), bl)
        y_loss = compute_loss(task, model(bx, by, use_x=False, use_y=True), bl)
        best_unimodal = x_loss if float(x_loss.detach().cpu()) <= float(y_loss.detach().cpu()) else y_loss
        loss = full_loss - best_unimodal
    else:
        raise ValueError(f"unsupported objective: {objective}")
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return flatten_grads(grads, params), float(loss.detach().cpu().item())


def compute_group_gradient_tables(
    model: torch.nn.Module,
    *,
    data: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    task: str,
    device: str,
    param_names: Sequence[str],
) -> dict[str, Any]:
    single = {0: [], 1: []}
    full = []
    interaction = []
    losses: dict[str, list[float]] = {"single_0": [], "single_1": [], "full": [], "interaction": []}
    for group_id, indices in enumerate(group_indices):
        for modality in (0, 1):
            grad_value, loss_value = compute_objective_gradient(
                model,
                data=data,
                indices=indices,
                task=task,
                device=device,
                param_names=param_names,
                objective="single",
                modality=modality,
            )
            single[modality].append(grad_value)
            losses[f"single_{modality}"].append(loss_value)
        full_grad, full_loss = compute_objective_gradient(
            model,
            data=data,
            indices=indices,
            task=task,
            device=device,
            param_names=param_names,
            objective="full",
        )
        int_grad, int_loss = compute_objective_gradient(
            model,
            data=data,
            indices=indices,
            task=task,
            device=device,
            param_names=param_names,
            objective="interaction",
        )
        full.append(full_grad)
        interaction.append(int_grad)
        losses["full"].append(full_loss)
        losses["interaction"].append(int_loss)
        if (group_id + 1) % 5 == 0 or group_id + 1 == len(group_indices):
            print(f"  group objectives {group_id + 1}/{len(group_indices)}", flush=True)
    return {
        "single": {m: torch.stack(values).numpy().astype(np.float32) for m, values in single.items()},
        "full": torch.stack(full).numpy().astype(np.float32),
        "interaction": torch.stack(interaction).numpy().astype(np.float32),
        "losses": losses,
    }


def build_single_modality_blocks(
    gradients: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    rows = []
    weights = []
    meta = []
    total = float(sum(len(indices) for indices in group_indices))
    for group_id, indices in enumerate(group_indices):
        weight = float(len(indices) / total)
        for modality in (0, 1):
            rows.append(np.asarray(gradients["single"][modality][group_id], dtype=np.float32))
            weights.append(weight)
            meta.append({"group": int(group_id), "modality": int(modality)})
    return np.stack(rows, axis=0), np.asarray(weights, dtype=np.float32), meta


def block_id(group_id: int, modality: int) -> int:
    return int(group_id) * 2 + int(modality)


def make_fisher_matvec(
    rows: torch.Tensor,
    weights: torch.Tensor,
    include_mask: np.ndarray,
    *,
    damping: float,
):
    include = torch.as_tensor(include_mask, dtype=torch.bool, device=rows.device)
    active_rows = rows[include]
    active_weights = weights[include]

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(rows.device, dtype=rows.dtype)
        if active_rows.numel() == 0:
            return float(damping) * vector
        coeff = active_rows @ vector
        return float(damping) * vector + active_rows.T @ (active_weights * coeff)

    return matvec


def target_interaction_subspace(
    interaction_grad: torch.Tensor,
    *,
    rank: int,
    num_lanczos: int,
    device: torch.device,
) -> torch.Tensor:
    norm = torch.linalg.norm(interaction_grad)
    if float(norm.detach().cpu()) <= 1e-12:
        basis = torch.zeros(interaction_grad.numel(), 1, dtype=torch.float32, device=device)
        basis[0, 0] = 1.0
        return basis

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return interaction_grad * torch.dot(interaction_grad, vector)

    slq = SLQ(matvec=matvec, dim=int(interaction_grad.numel()), device=device)
    try:
        _, vecs = slq.top_k_eigen(
            min(max(int(rank), 1), int(interaction_grad.numel())),
            num_lanczos=max(2, min(int(num_lanczos), int(interaction_grad.numel()))),
            reorthogonalize=True,
        )
        return vecs[:, : min(int(rank), vecs.shape[1])].detach()
    except Exception:
        return torch.nn.functional.normalize(interaction_grad.reshape(-1, 1), dim=0)


def lanczos_inverse_quadrature(
    matvec,
    start: torch.Tensor,
    *,
    num_lanczos: int,
    device: torch.device,
    reorthogonalize: bool = True,
) -> float:
    start = start.to(device=device, dtype=torch.float32).reshape(-1)
    start_norm = torch.linalg.norm(start)
    if float(start_norm.detach().cpu()) <= 1e-20:
        return 0.0
    dim = int(start.numel())
    steps = max(1, min(int(num_lanczos), dim))
    q = start / start_norm
    q_prev = torch.zeros_like(q)
    beta_prev = torch.tensor(0.0, device=device, dtype=torch.float32)
    alpha_values = []
    beta_values = []
    basis = []
    for step in range(steps):
        if reorthogonalize:
            basis.append(q)
        v = matvec(q)
        if step > 0:
            v = v - beta_prev * q_prev
        alpha = torch.dot(q, v)
        alpha_values.append(alpha)
        v = v - alpha * q
        if reorthogonalize and basis:
            qmat = torch.stack(basis, dim=0)
            v = v - qmat.T @ (qmat @ v)
        beta = torch.linalg.norm(v)
        if float(beta.detach().cpu()) <= 1e-12 or step == steps - 1:
            break
        beta_values.append(beta)
        q_prev = q
        beta_prev = beta
        q = v / beta
    alpha_t = torch.stack(alpha_values)
    tri = torch.diag(alpha_t)
    if beta_values:
        beta_t = torch.stack(beta_values)
        tri = tri + torch.diag(beta_t, diagonal=1) + torch.diag(beta_t, diagonal=-1)
    e1 = torch.zeros(tri.shape[0], dtype=torch.float32, device=device)
    e1[0] = 1.0
    try:
        solution = torch.linalg.solve(tri, e1)
    except RuntimeError:
        solution = torch.linalg.pinv(tri) @ e1
    return float((start_norm * start_norm * solution[0]).detach().cpu().item())


def inverse_quadrature(
    vector: torch.Tensor,
    matvec,
    *,
    num_lanczos: int,
    device: torch.device,
) -> float:
    return lanczos_inverse_quadrature(matvec, vector, num_lanczos=num_lanczos, device=device)


def bilinear_inverse_quadrature(
    left: torch.Tensor,
    right: torch.Tensor,
    matvec,
    *,
    num_lanczos: int,
    device: torch.device,
) -> float:
    left = left.to(device=device, dtype=torch.float32).reshape(-1)
    right = right.to(device=device, dtype=torch.float32).reshape(-1)
    plus = inverse_quadrature(left + right, matvec, num_lanczos=num_lanczos, device=device)
    minus = inverse_quadrature(left - right, matvec, num_lanczos=num_lanczos, device=device)
    return 0.25 * (plus - minus)


def subspace_inverse_variance(
    basis: torch.Tensor,
    matvec,
    *,
    num_lanczos: int,
    device: torch.device,
) -> float:
    total = 0.0
    for col in range(basis.shape[1]):
        total += inverse_quadrature(basis[:, col], matvec, num_lanczos=num_lanczos, device=device)
    return float(total)


def fisher_trace_on_subspace(
    basis: torch.Tensor,
    rows: torch.Tensor,
    weights: torch.Tensor,
    include_mask: np.ndarray,
) -> float:
    include = torch.as_tensor(include_mask, dtype=torch.bool, device=rows.device)
    active_rows = rows[include]
    active_weights = weights[include]
    if active_rows.numel() == 0:
        return 0.0
    projection = active_rows @ basis
    value = active_weights[:, None] * projection * projection
    return float(value.sum().detach().cpu().item())


def make_global_invsqrt(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    damping: float,
):
    base = 1.0 / math.sqrt(float(damping))
    eigvals = eigenvalues.reshape(-1).float()
    eigvecs = eigenvectors.float()
    coeff = torch.rsqrt(torch.clamp(eigvals, min=float(damping))) - base

    def apply(vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(device=eigvecs.device, dtype=torch.float32).reshape(-1)
        if eigvecs.numel() == 0:
            return base * vector
        return base * vector + eigvecs @ (coeff * (eigvecs.T @ vector))

    return apply


def source_mmex(
    target_basis: torch.Tensor,
    *,
    source_group: int,
    gradients: dict[str, Any],
    group_indices: Sequence[Sequence[int]],
    inverse_sqrt,
    device: torch.device,
) -> float:
    total = float(sum(len(indices) for indices in group_indices))
    weight = float(len(group_indices[source_group]) / total)
    full = torch.as_tensor(gradients["full"][source_group], dtype=torch.float32, device=device)
    single0 = torch.as_tensor(gradients["single"][0][source_group], dtype=torch.float32, device=device)
    single1 = torch.as_tensor(gradients["single"][1][source_group], dtype=torch.float32, device=device)
    value = 0.0
    for col in range(target_basis.shape[1]):
        z = inverse_sqrt(target_basis[:, col])
        full_proj = torch.dot(full, z)
        s0_proj = torch.dot(single0, z)
        s1_proj = torch.dot(single1, z)
        value += float((weight * (full_proj * full_proj - s0_proj * s0_proj - s1_proj * s1_proj)).detach().cpu())
    return value


def estimate_lds_modality_effects(payload: dict[str, Any], *, ridge_lambda: float) -> dict[str, Any]:
    num_groups = int(payload["group_partition"]["num_groups"])
    num_modalities = int(payload["group_partition"]["num_modalities"])
    targets = np.asarray([float(run["test_loss"]) for run in payload["runs"]], dtype=np.float64)
    features = np.zeros((len(payload["runs"]), num_groups * num_modalities), dtype=np.float64)
    group_to_indices = {
        group_id: np.asarray(indices, dtype=np.int64)
        for group_id, indices in enumerate(payload["group_partition"]["group_indices"])
    }
    for row, run in enumerate(payload["runs"]):
        data_indices = np.asarray(run["data_indices"], dtype=np.int64)
        keep_mask = np.asarray(run["sample_modality_mask"], dtype=bool)
        index_to_local = {int(index): local for local, index in enumerate(data_indices.tolist())}
        for group_id in range(num_groups):
            locals_in_group = [
                index_to_local[int(index)]
                for index in group_to_indices[group_id].tolist()
                if int(index) in index_to_local
            ]
            if not locals_in_group:
                features[row, group_id * num_modalities : (group_id + 1) * num_modalities] = 1.0
                continue
            keep_rate = keep_mask[np.asarray(locals_in_group, dtype=np.int64)].mean(axis=0)
            features[row, group_id * num_modalities : (group_id + 1) * num_modalities] = 1.0 - keep_rate
    mean_x = features.mean(axis=0, keepdims=True)
    std_x = features.std(axis=0, keepdims=True)
    standardized = (features - mean_x) / np.maximum(std_x, 1e-8)
    centered_y = targets - float(np.mean(targets))
    lhs = standardized.T @ standardized + float(ridge_lambda) * np.eye(standardized.shape[1], dtype=np.float64)
    coef_standardized = np.linalg.solve(lhs, standardized.T @ centered_y)
    predictions = float(np.mean(targets)) + standardized @ coef_standardized
    coef = coef_standardized / np.maximum(std_x.reshape(-1), 1e-8)
    return {
        "effects": coef.reshape(num_groups, num_modalities),
        "summary": {
            "num_runs": int(len(payload["runs"])),
            "target_loss_mean": float(np.mean(targets)),
            "target_loss_std": float(np.std(targets)),
            "ridge_lambda": float(ridge_lambda),
            "fit_rmse": rmse(predictions, targets),
            "fit_pearson": pearsonr(predictions, targets),
            "fit_spearman": spearmanr(predictions, targets),
            "effect_min": float(np.min(coef)),
            "effect_max": float(np.max(coef)),
        },
    }


def attach_source_lds(records: list[dict[str, Any]], lds_effects: dict[str, Any]) -> None:
    effects = np.asarray(lds_effects["effects"], dtype=np.float64)
    for record in records:
        source_group = int(record["source_group"])
        source_modality = int(record["source_modality"])
        record["source_lds_modality_loss_effect"] = float(effects[source_group, source_modality])


def gradient_stats(gradients: dict[str, Any]) -> dict[str, float]:
    single = np.concatenate([gradients["single"][0], gradients["single"][1]], axis=0)
    full = np.asarray(gradients["full"], dtype=np.float64)
    interaction = np.asarray(gradients["interaction"], dtype=np.float64)
    return {
        "single_gradient_norm_median": float(np.median(np.linalg.norm(single, axis=1))),
        "single_gradient_norm_max": float(np.max(np.linalg.norm(single, axis=1))),
        "full_gradient_norm_median": float(np.median(np.linalg.norm(full, axis=1))),
        "interaction_gradient_norm_median": float(np.median(np.linalg.norm(interaction, axis=1))),
        "interaction_gradient_norm_max": float(np.max(np.linalg.norm(interaction, axis=1))),
    }


def summarize(
    records: Sequence[dict[str, Any]],
    fcsm_records: Sequence[dict[str, Any]],
    mmex_records: Sequence[dict[str, Any]],
    lds_effects: dict[str, Any],
    *,
    top_records: int,
) -> dict[str, Any]:
    fvc = values(records, "fvc")
    nih = values(records, "nih")
    nalign = values(records, "nalign")
    tfc = values(records, "tfc")
    ntfc = values(records, "ntfc")
    source_lds = values(records, "source_lds_modality_loss_effect")
    fcsm = values(fcsm_records, "f_csmi")
    fcsm_lds = values(fcsm_records, "source_lds_modality_loss_effect")
    mmex = values(mmex_records, "mmex")
    return {
        "record_counts": {
            "pair_records": int(len(records)),
            "f_csmi_records": int(len(fcsm_records)),
            "mmex_records": int(len(mmex_records)),
        },
        "metric_summary": {
            "tfc_mean": safe_mean(tfc),
            "ntfc_mean": safe_mean(ntfc),
            "fvc_mean": safe_mean(fvc),
            "fvc_positive_fraction": float(np.mean(fvc > 0.0)) if fvc.size else float("nan"),
            "nih_mean": safe_mean(nih),
            "nih_positive_fraction": float(np.mean(nih > 0.0)) if nih.size else float("nan"),
            "nalign_mean": safe_mean(nalign),
            "f_csmi_mean": safe_mean(fcsm),
            "f_csmi_positive_fraction": float(np.mean(fcsm > 0.0)) if fcsm.size else float("nan"),
            "mmex_mean": safe_mean(mmex),
            "mmex_positive_fraction": float(np.mean(mmex > 0.0)) if mmex.size else float("nan"),
        },
        "lds_proxy_correlations": {
            "lds_proxy_note": (
                "Uses the linear deletion-mask fit on existing subset losses as a source-block proxy; "
                "it is not the leave-group-modality-out retraining target requested in fisher_test_gpt.md."
            ),
            "lds_fit": lds_effects["summary"],
            "tfc_vs_source_lds_spearman": spearmanr(tfc, source_lds),
            "ntfc_vs_source_lds_spearman": spearmanr(ntfc, source_lds),
            "fvc_vs_source_lds_spearman": spearmanr(fvc, source_lds),
            "nih_vs_source_lds_spearman": spearmanr(nih, source_lds),
            "nalign_vs_source_lds_spearman": spearmanr(nalign, source_lds),
            "f_csmi_vs_source_lds_spearman": spearmanr(fcsm, fcsm_lds),
            "fvc_vs_source_lds_pearson": pearsonr(fvc, source_lds),
            "nih_vs_source_lds_pearson": pearsonr(nih, source_lds),
            "nalign_vs_source_lds_pearson": pearsonr(nalign, source_lds),
        },
        "top_fvc": compact_pair_records(sorted(records, key=lambda row: float(row["fvc"]), reverse=True)[:top_records]),
        "top_nih": compact_pair_records(sorted(records, key=lambda row: float(row["nih"]), reverse=True)[:top_records]),
        "most_negative_nih": compact_pair_records(sorted(records, key=lambda row: float(row["nih"]))[:top_records]),
        "top_f_csmi": compact_fcsm_records(sorted(fcsm_records, key=lambda row: float(row["f_csmi"]), reverse=True)[:top_records]),
        "most_negative_f_csmi": compact_fcsm_records(sorted(fcsm_records, key=lambda row: float(row["f_csmi"]))[:top_records]),
        "top_mmex": compact_mmex_records(sorted(mmex_records, key=lambda row: float(row["mmex"]), reverse=True)[:top_records]),
    }


def compact_pair_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("target_group", "source_group", "source_modality", "tfc", "ntfc", "fvc", "nih", "nalign", "source_lds_modality_loss_effect")
    return [{key: row[key] for key in keys} for row in records]


def compact_fcsm_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "target_group",
        "target_modality",
        "source_group",
        "source_modality",
        "f_csmi",
        "source_lds_modality_loss_effect",
    )
    return [{key: row[key] for key in keys} for row in records]


def compact_mmex_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("target_group", "source_group", "mmex")
    return [{key: row[key] for key in keys} for row in records]


def values(records: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records], dtype=np.float64)


def safe_mean(array: np.ndarray) -> float:
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def pearsonr(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    x = np.asarray(list(lhs), dtype=np.float64)
    y = np.asarray(list(rhs), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)) ** 2)))


def to_markdown(output: dict[str, Any]) -> str:
    cfg = output["config"]
    summary = output["summary"]
    metrics = summary["metric_summary"]
    corr = summary["lds_proxy_correlations"]
    lines = [
        "# Fisher GPT Metrics",
        "",
        f"- dataset: `{cfg['dataset']}`, seed: `{cfg['seed']}`, unit: `{cfg['unit']}`",
        f"- groups: `{cfg['num_groups']}`, targets: `{len(cfg['target_groups'])}`, sources: `{len(cfg['source_groups'])}`",
        f"- gradient params: `{cfg['gradient_params']}` ({cfg['num_parameters']:,} scalars)",
        f"- damping: `{cfg['damping']}`, Lanczos steps: `{cfg['lanczos_steps']}`",
        "",
        "## Metric Summary",
        "",
        f"- FVC mean={metrics['fvc_mean']:.6g}, positive_fraction={metrics['fvc_positive_fraction']:.3f}",
        f"- NIH mean={metrics['nih_mean']:.6g}, positive_fraction={metrics['nih_positive_fraction']:.3f}",
        f"- nAlign mean={metrics['nalign_mean']:.6g}",
        f"- F-CSMI mean={metrics['f_csmi_mean']:.6g}, positive_fraction={metrics['f_csmi_positive_fraction']:.3f}",
        f"- MMEx mean={metrics['mmex_mean']:.6g}, positive_fraction={metrics['mmex_positive_fraction']:.3f}",
        "",
        "## LDS Proxy Check",
        "",
        f"- LDS proxy ridge fit RMSE={corr['lds_fit']['fit_rmse']:.6g}, Spearman={corr['lds_fit']['fit_spearman']:.3f}",
        f"- FVC vs source LDS proxy: Spearman={corr['fvc_vs_source_lds_spearman']:.3f}, Pearson={corr['fvc_vs_source_lds_pearson']:.3f}",
        f"- NIH vs source LDS proxy: Spearman={corr['nih_vs_source_lds_spearman']:.3f}, Pearson={corr['nih_vs_source_lds_pearson']:.3f}",
        f"- nAlign vs source LDS proxy: Spearman={corr['nalign_vs_source_lds_spearman']:.3f}, Pearson={corr['nalign_vs_source_lds_pearson']:.3f}",
        f"- F-CSMI vs source LDS proxy: Spearman={corr['f_csmi_vs_source_lds_spearman']:.3f}",
        "",
        "## Top FVC",
        "",
        "| target | source | modality | FVC | NIH | nAlign | LDS proxy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["top_fvc"]:
        lines.append(
            f"| {row['target_group']} | {row['source_group']} | {row['source_modality']} | "
            f"{row['fvc']:.6g} | {row['nih']:.6g} | {row['nalign']:.6g} | "
            f"{row['source_lds_modality_loss_effect']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Top F-CSMI",
            "",
            "| target | target_mod | source | source_mod | F-CSMI | LDS proxy |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["top_f_csmi"]:
        lines.append(
            f"| {row['target_group']} | {row['target_modality']} | {row['source_group']} | "
            f"{row['source_modality']} | {row['f_csmi']:.6g} | {row['source_lds_modality_loss_effect']:.6g} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
