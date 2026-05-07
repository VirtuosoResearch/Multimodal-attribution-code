"""Selection utilities for multimodal data selection experiments.

Selectors return a boolean mask over training samples. ``True`` means the
auxiliary modality is kept for that training sample; ``False`` means the main
modality remains but the auxiliary modality is hidden during training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_RESULT_ROOT = (
    REPO_ROOT / "src" / "attribution" / "results" / "lds_real_train"
)


@dataclass(frozen=True)
class SelectionContext:
    """Information available to data selection methods."""

    dataset_name: str
    seed: int
    budget: float
    labels: torch.Tensor
    main_features: torch.Tensor | None = None
    aux_features: torch.Tensor | None = None
    output_dir: Path | None = None

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class SelectionResult:
    """Result produced by a data selection method."""

    name: str
    mask: torch.BoolTensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def indices(self) -> list[int]:
        return mask_to_indices(self.mask)


class BaseSelector:
    """Base class for pluggable data selection methods."""

    name = "base"

    def select(self, context: SelectionContext) -> SelectionResult:
        raise NotImplementedError


SelectorFactory = Callable[[Any], BaseSelector]
_SELECTOR_REGISTRY: dict[str, SelectorFactory] = {}


def register_selector(name: str, factory: SelectorFactory) -> None:
    if name in _SELECTOR_REGISTRY:
        raise ValueError(f"Selector already registered: {name}")
    _SELECTOR_REGISTRY[name] = factory


def available_selectors() -> tuple[str, ...]:
    return tuple(sorted(_SELECTOR_REGISTRY))


def build_selector(args: Any) -> BaseSelector:
    if args.selection_method not in _SELECTOR_REGISTRY:
        known = ", ".join(available_selectors())
        raise ValueError(
            f"Unknown selection method: {args.selection_method}. "
            f"Available methods: {known}")
    return _SELECTOR_REGISTRY[args.selection_method](args)


def random_budget_mask(
    num_samples: int,
    budget: float,
    *,
    seed: int,
    labels: torch.Tensor | None = None,
    stratified: bool = False,
) -> torch.BoolTensor:
    """Return a boolean mask selecting a random budget of samples.

    When ``stratified`` is true and labels are one-dimensional class labels,
    the budget is applied independently within each class. This is useful for
    classification datasets where a completely uniform random draw can remove
    rare classes at small budgets.
    """
    if not 0.0 <= budget <= 1.0:
        raise ValueError(f"budget must be in [0, 1], got {budget}")
    rng = np.random.default_rng(seed)
    mask = np.zeros(num_samples, dtype=bool)

    if budget == 0.0 or num_samples == 0:
        return torch.from_numpy(mask)
    if budget == 1.0:
        mask[:] = True
        return torch.from_numpy(mask)

    if stratified:
        if labels is None:
            raise ValueError("stratified selection requires labels")
        label_array = labels.detach().cpu().numpy()
        if label_array.ndim != 1:
            raise ValueError("stratified selection only supports 1-D labels")
        for cls in np.unique(label_array):
            cls_indices = np.flatnonzero(label_array == cls)
            cls_count = max(1, int(round(len(cls_indices) * budget)))
            chosen = rng.choice(cls_indices, size=cls_count, replace=False)
            mask[chosen] = True
        return torch.from_numpy(mask)

    selected_count = max(1, int(round(num_samples * budget)))
    selected = rng.choice(num_samples, size=selected_count, replace=False)
    mask[selected] = True
    return torch.from_numpy(mask)


def mask_to_indices(mask: torch.BoolTensor) -> list[int]:
    """Convert a boolean mask to plain Python indices for JSON logging."""
    return mask.nonzero(as_tuple=False).view(-1).cpu().tolist()


def indices_to_mask(indices: list[int] | np.ndarray | torch.Tensor,
                    num_samples: int) -> torch.BoolTensor:
    """Convert selected indices to a validated boolean mask."""
    if isinstance(indices, torch.Tensor):
        index_array = indices.detach().cpu().numpy()
    else:
        index_array = np.asarray(indices)
    if index_array.dtype == np.bool_:
        if index_array.shape != (num_samples,):
            raise ValueError(
                f"Boolean mask has shape {index_array.shape}, expected "
                f"({num_samples},)")
        return torch.from_numpy(index_array.astype(bool))

    index_array = index_array.reshape(-1).astype(np.int64)
    if index_array.size == 0:
        return torch.zeros(num_samples, dtype=torch.bool)
    if np.any(index_array < 0) or np.any(index_array >= num_samples):
        bad = index_array[(index_array < 0) | (index_array >= num_samples)][:10]
        raise IndexError(
            f"Selection indices out of range for {num_samples} samples: "
            f"{bad.tolist()}")
    mask = torch.zeros(num_samples, dtype=torch.bool)
    mask[torch.from_numpy(np.unique(index_array))] = True
    return mask


def _extract_index_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("selected_indices", "indices", "index", "mask"):
            if key in payload:
                return payload[key]
        raise KeyError(
            "Index dict must contain one of: selected_indices, indices, "
            "index, mask")
    return payload


def load_index_mask(path: str | Path, num_samples: int) -> torch.BoolTensor:
    """Load a selection mask from json/txt/csv/npy/npz/pt/pth files."""
    index_path = Path(path)
    suffix = index_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(index_path.read_text())
        return indices_to_mask(_extract_index_payload(payload), num_samples)
    if suffix in {".txt", ".csv"}:
        text = index_path.read_text().replace(",", " ")
        values = [int(part) for part in text.split()]
        return indices_to_mask(values, num_samples)
    if suffix == ".npy":
        payload = np.load(index_path, allow_pickle=True)
        return indices_to_mask(payload, num_samples)
    if suffix == ".npz":
        payload = np.load(index_path, allow_pickle=True)
        for key in ("selected_indices", "indices", "index", "mask"):
            if key in payload:
                return indices_to_mask(payload[key], num_samples)
        raise KeyError(
            "NPZ index file must contain one of: selected_indices, indices, "
            "index, mask")
    if suffix in {".pt", ".pth"}:
        payload = torch.load(index_path, map_location="cpu", weights_only=False)
        return indices_to_mask(_extract_index_payload(payload), num_samples)
    raise ValueError(
        f"Unsupported index file format: {index_path}. Supported: "
        "json, txt, csv, npy, npz, pt, pth")


def _extract_group_ids(payload: Any) -> list[int]:
    if not isinstance(payload, dict):
        raise TypeError("group index selection requires a JSON dict payload")
    for key in ("selected_group_ids", "group_ids", "groups"):
        if key in payload:
            return [int(value) for value in payload[key]]
    raise KeyError("Group index dict must contain selected_group_ids, group_ids, or groups")


def _extract_metadata(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        return dict(payload["metadata"])
    return {}


def _resolve_group_run_name(
    *,
    payload: dict[str, Any],
    explicit_run_name: str | None,
) -> str:
    if explicit_run_name:
        return str(explicit_run_name)
    metadata = _extract_metadata(payload)
    hparams = metadata.get("hparams")
    if isinstance(hparams, dict) and hparams.get("run_name"):
        return str(hparams["run_name"])
    hparams = payload.get("hparams") if isinstance(payload, dict) else None
    if isinstance(hparams, dict) and hparams.get("run_name"):
        return str(hparams["run_name"])
    raise KeyError(
        "Could not resolve LDS run_name. Pass --group-run-name or use an "
        "index JSON with metadata.hparams.run_name."
    )


def _resolve_group_seed(payload: dict[str, Any], explicit_seed: int | None) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    metadata = _extract_metadata(payload)
    if "surrogate_seed" in metadata:
        return int(metadata["surrogate_seed"])
    if "surrogate_seed" in payload:
        return int(payload["surrogate_seed"])
    return 0


def group_ids_to_mask(
    *,
    group_ids: list[int],
    group_payload_path: Path,
    num_samples: int,
) -> tuple[torch.BoolTensor, dict[str, Any]]:
    group_payload = json.loads(group_payload_path.read_text())
    partition = group_payload["group_partition"]
    group_indices = partition["group_indices"]
    selected_indices: list[int] = []
    group_sizes: dict[str, int] = {}
    original_group_sizes: dict[str, int] = {}
    dropped_group_sizes: dict[str, int] = {}
    for group_id in group_ids:
        if group_id < 0 or group_id >= len(group_indices):
            raise IndexError(
                f"group_id={group_id} out of range for {len(group_indices)} groups"
            )
        indices = [int(index) for index in group_indices[group_id]]
        valid_indices = [
            index for index in indices
            if 0 <= index < num_samples
        ]
        selected_indices.extend(valid_indices)
        original_group_sizes[str(group_id)] = len(indices)
        group_sizes[str(group_id)] = len(valid_indices)
        dropped_group_sizes[str(group_id)] = len(indices) - len(valid_indices)
    mask = indices_to_mask(selected_indices, num_samples)
    return mask, {
        "group_payload": str(group_payload_path),
        "selected_group_ids": [int(group_id) for group_id in group_ids],
        "selected_group_sizes": group_sizes,
        "selected_group_original_sizes": original_group_sizes,
        "selected_group_dropped_sizes": dropped_group_sizes,
        "selected_group_dropped_total": int(sum(dropped_group_sizes.values())),
        "num_groups": int(partition.get("num_groups", len(group_indices))),
        "group_method": partition.get("group_method"),
    }


class RandomSelector(BaseSelector):
    """Uniform random budget selector."""

    name = "random"

    def __init__(self, *, stratified: bool = False,
                 seed_offset: int = 1000) -> None:
        self.stratified = stratified
        self.seed_offset = seed_offset

    def select(self, context: SelectionContext) -> SelectionResult:
        mask = random_budget_mask(
            context.num_samples,
            context.budget,
            seed=context.seed + self.seed_offset,
            labels=context.labels,
            stratified=self.stratified,
        )
        return SelectionResult(
            name=self.name,
            mask=mask,
            metadata={
                "budget": context.budget,
                "stratified": self.stratified,
                "selection_seed": context.seed + self.seed_offset,
            },
        )


class IndexFileSelector(BaseSelector):
    """Selector that directly consumes externally written sample indices."""

    name = "index"

    def __init__(self, *, index_file: str | None = None,
                 index_template: str | None = None) -> None:
        self.index_file = index_file
        self.index_template = index_template
        if self.index_file is None and self.index_template is None:
            raise ValueError(
                "selection_method=index requires --index-file or "
                "--index-template")

    def _resolve_path(self, context: SelectionContext) -> Path:
        if self.index_template is not None:
            return Path(self.index_template.format(
                dataset=context.dataset_name,
                seed=context.seed,
                budget=context.budget,
                budget_pct=int(round(context.budget * 100)),
            ))
        if self.index_file is None:
            raise ValueError("Missing index file")
        return Path(self.index_file)

    def select(self, context: SelectionContext) -> SelectionResult:
        index_path = self._resolve_path(context)
        mask = load_index_mask(index_path, context.num_samples)
        return SelectionResult(
            name=self.name,
            mask=mask,
            metadata={
                "index_file": str(index_path),
                "budget": context.budget,
            },
        )


class GroupIndexFileSelector(IndexFileSelector):
    """Selector that expands externally selected LDS group ids to samples."""

    name = "group_index"

    def __init__(
        self,
        *,
        index_file: str | None = None,
        index_template: str | None = None,
        group_result_root: str | Path = DEFAULT_GROUP_RESULT_ROOT,
        group_run_name: str | None = None,
        group_seed: int | None = None,
    ) -> None:
        super().__init__(index_file=index_file, index_template=index_template)
        self.group_result_root = Path(group_result_root)
        self.group_run_name = group_run_name
        self.group_seed = group_seed

    def select(self, context: SelectionContext) -> SelectionResult:
        index_path = self._resolve_path(context)
        if index_path.suffix.lower() != ".json":
            raise ValueError("selection_method=group_index requires a JSON index file")
        payload = json.loads(index_path.read_text())
        group_ids = _extract_group_ids(payload)
        run_name = _resolve_group_run_name(
            payload=payload,
            explicit_run_name=self.group_run_name,
        )
        group_seed = _resolve_group_seed(payload, self.group_seed)
        group_payload_path = (
            self.group_result_root / run_name / f"{context.dataset_name}_seed{group_seed}.json"
        )
        if not group_payload_path.exists():
            raise FileNotFoundError(f"missing LDS group payload: {group_payload_path}")
        mask, group_metadata = group_ids_to_mask(
            group_ids=group_ids,
            group_payload_path=group_payload_path,
            num_samples=context.num_samples,
        )
        return SelectionResult(
            name=self.name,
            mask=mask,
            metadata={
                "index_file": str(index_path),
                "budget": context.budget,
                **group_metadata,
            },
        )


class RandomGroupSelector(IndexFileSelector):
    """Selector that randomly keeps whole LDS groups of auxiliary modality."""

    name = "random_group"

    def __init__(
        self,
        *,
        index_file: str | None = None,
        index_template: str | None = None,
        group_result_root: str | Path = DEFAULT_GROUP_RESULT_ROOT,
        group_run_name: str | None = None,
        group_seed: int | None = None,
        num_select_groups: int = 1,
        seed_offset: int = 3000,
    ) -> None:
        self.index_file = index_file
        self.index_template = index_template
        self.group_result_root = Path(group_result_root)
        self.group_run_name = group_run_name
        self.group_seed = group_seed
        self.num_select_groups = int(num_select_groups)
        self.seed_offset = int(seed_offset)
        if self.num_select_groups <= 0:
            raise ValueError(f"num_select_groups must be positive, got {num_select_groups}")
        if self.group_run_name is None and self.index_file is None and self.index_template is None:
            raise ValueError(
                "selection_method=random_group requires --group-run-name or "
                "an --index-file/--index-template reference with metadata.hparams.run_name"
            )

    def _resolve_reference_payload(self, context: SelectionContext) -> dict[str, Any]:
        if self.index_file is None and self.index_template is None:
            return {}
        index_path = self._resolve_path(context)
        if index_path.suffix.lower() != ".json":
            raise ValueError("random_group reference index must be JSON")
        return json.loads(index_path.read_text())

    def select(self, context: SelectionContext) -> SelectionResult:
        reference_payload = self._resolve_reference_payload(context)
        run_name = _resolve_group_run_name(
            payload=reference_payload,
            explicit_run_name=self.group_run_name,
        )
        group_seed = _resolve_group_seed(reference_payload, self.group_seed)
        group_payload_path = (
            self.group_result_root / run_name / f"{context.dataset_name}_seed{group_seed}.json"
        )
        if not group_payload_path.exists():
            raise FileNotFoundError(f"missing LDS group payload: {group_payload_path}")

        group_payload = json.loads(group_payload_path.read_text())
        num_groups = int(group_payload["group_partition"]["num_groups"])
        if self.num_select_groups > num_groups:
            raise ValueError(
                f"num_select_groups={self.num_select_groups} exceeds available groups={num_groups}"
            )
        rng = np.random.default_rng(context.seed + self.seed_offset)
        group_ids = sorted(
            int(value)
            for value in rng.choice(num_groups, size=self.num_select_groups, replace=False).tolist()
        )
        mask, group_metadata = group_ids_to_mask(
            group_ids=group_ids,
            group_payload_path=group_payload_path,
            num_samples=context.num_samples,
        )
        return SelectionResult(
            name=self.name,
            mask=mask,
            metadata={
                "budget": context.budget,
                "num_select_groups": int(self.num_select_groups),
                "selection_seed": int(context.seed + self.seed_offset),
                **group_metadata,
            },
        )


def _random_factory(args: Any) -> BaseSelector:
    return RandomSelector(
        stratified=args.stratified_random,
        seed_offset=args.selection_seed_offset,
    )


def _index_factory(args: Any) -> BaseSelector:
    return IndexFileSelector(
        index_file=args.index_file,
        index_template=args.index_template,
    )


def _group_index_factory(args: Any) -> BaseSelector:
    return GroupIndexFileSelector(
        index_file=args.index_file,
        index_template=args.index_template,
        group_result_root=args.group_result_root,
        group_run_name=args.group_run_name,
        group_seed=args.group_seed,
    )


def _random_group_factory(args: Any) -> BaseSelector:
    return RandomGroupSelector(
        index_file=args.index_file,
        index_template=args.index_template,
        group_result_root=args.group_result_root,
        group_run_name=args.group_run_name,
        group_seed=args.group_seed,
        num_select_groups=args.num_select_groups,
        seed_offset=args.selection_seed_offset,
    )


register_selector("random", _random_factory)
register_selector("index", _index_factory)
register_selector("group_index", _group_index_factory)
register_selector("random_group", _random_group_factory)
