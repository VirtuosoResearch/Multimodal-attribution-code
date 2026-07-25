from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MultimodalTensorKernelConfig:
    ridge_lambda: float = 1e-3
    sample_gamma: float | None = None
    modality_gamma: float | None = None
    query_gamma: float | None = None


@dataclass
class MultimodalTensorKernelRegressor:
    train_selection_matrices: np.ndarray
    train_query_masks: np.ndarray
    dual_weights: np.ndarray
    sample_gamma: float
    modality_gamma: float
    query_gamma: float

    def __post_init__(self) -> None:
        self.train_selection_matrices = np.asarray(self.train_selection_matrices, dtype=np.float64)
        self.train_query_masks = np.asarray(self.train_query_masks, dtype=np.float64)
        self.dual_weights = np.asarray(self.dual_weights, dtype=np.float64).reshape(-1)
        if self.train_selection_matrices.ndim != 3:
            raise ValueError("train_selection_matrices must have shape [num_queries, num_samples, num_modalities]")
        if self.train_query_masks.ndim != 2:
            raise ValueError("train_query_masks must have shape [num_queries, num_modalities]")
        if self.train_selection_matrices.shape[0] != self.train_query_masks.shape[0]:
            raise ValueError("selection matrices and query masks must have the same number of rows")
        if self.dual_weights.shape[0] != self.train_selection_matrices.shape[0]:
            raise ValueError("dual_weights must align with the number of training queries")
        self.num_queries, self.num_samples, self.num_modalities = self.train_selection_matrices.shape

    def predict(
        self,
        selection_matrix: Sequence[Sequence[bool]] | np.ndarray,
        query_mask: Sequence[bool] | np.ndarray,
    ) -> float:
        selection = _as_selection_batch(selection_matrix, self.num_samples, self.num_modalities)
        query = _as_query_batch(query_mask, self.num_modalities)
        return float(self.predict_batch(selection, query)[0])

    def predict_batch(
        self,
        selection_matrices: Sequence[Sequence[Sequence[bool]]] | np.ndarray,
        query_masks: Sequence[Sequence[bool]] | np.ndarray,
    ) -> np.ndarray:
        selection = _as_selection_batch(selection_matrices, self.num_samples, self.num_modalities)
        query = _as_query_batch(query_masks, self.num_modalities, batch_size=selection.shape[0])
        kernel = _tensor_product_kernel(
            lhs_selection=selection,
            rhs_selection=self.train_selection_matrices,
            lhs_query=query,
            rhs_query=self.train_query_masks,
            sample_gamma=self.sample_gamma,
            modality_gamma=self.modality_gamma,
            query_gamma=self.query_gamma,
        )
        return kernel @ self.dual_weights

    def csmi(
        self,
        *,
        sample_index: int,
        train_modalities: Sequence[int] | Sequence[bool] | np.ndarray,
        query_modalities: Sequence[int] | Sequence[bool] | np.ndarray,
        base_selection_matrix: Sequence[Sequence[bool]] | np.ndarray | None = None,
        base_query_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> float:
        if not (0 <= sample_index < self.num_samples):
            raise IndexError(f"sample_index must lie in [0, {self.num_samples}), got {sample_index}")
        base_selection = (
            np.ones((self.num_samples, self.num_modalities), dtype=bool)
            if base_selection_matrix is None
            else _as_selection_batch(base_selection_matrix, self.num_samples, self.num_modalities)[0].astype(bool)
        )
        base_query = (
            np.ones(self.num_modalities, dtype=bool)
            if base_query_mask is None
            else _as_query_batch(base_query_mask, self.num_modalities)[0].astype(bool)
        )
        train_drop = _as_modality_mask(train_modalities, self.num_modalities)
        query_drop = _as_modality_mask(query_modalities, self.num_modalities)

        dropped_selection = base_selection.copy()
        dropped_selection[sample_index, train_drop] = False
        dropped_query = base_query.copy()
        dropped_query[query_drop] = False

        full_full = self.predict(base_selection, base_query)
        drop_train = self.predict(dropped_selection, base_query)
        drop_query = self.predict(base_selection, dropped_query)
        drop_both = self.predict(dropped_selection, dropped_query)
        return float(full_full - drop_train - drop_query + drop_both)


def fit_multimodal_tensor_kernel_regressor(
    selection_matrices: Sequence[Sequence[Sequence[bool]]] | np.ndarray,
    query_masks: Sequence[Sequence[bool]] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    config: MultimodalTensorKernelConfig | None = None,
) -> MultimodalTensorKernelRegressor:
    config = config or MultimodalTensorKernelConfig()
    selection = np.asarray(selection_matrices, dtype=bool)
    if selection.ndim != 3:
        raise ValueError("selection_matrices must have shape [num_queries, num_samples, num_modalities]")
    num_queries, num_samples, num_modalities = selection.shape
    query = _as_query_batch(query_masks, num_modalities, batch_size=num_queries).astype(bool)
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if values.shape[0] != num_queries:
        raise ValueError("targets must align with the number of query states")

    sample_gamma = _resolve_component_gamma(
        selection.reshape(num_queries * num_samples, num_modalities).astype(np.float64),
        config.sample_gamma,
    )
    modality_gamma = _resolve_component_gamma(
        np.transpose(selection, (0, 2, 1)).reshape(num_queries * num_modalities, num_samples).astype(np.float64),
        config.modality_gamma,
    )
    query_gamma = _resolve_component_gamma(query.astype(np.float64), config.query_gamma)

    kernel = _tensor_product_kernel(
        lhs_selection=selection.astype(np.float64),
        rhs_selection=selection.astype(np.float64),
        lhs_query=query.astype(np.float64),
        rhs_query=query.astype(np.float64),
        sample_gamma=sample_gamma,
        modality_gamma=modality_gamma,
        query_gamma=query_gamma,
    )
    regularized = kernel + float(config.ridge_lambda) * np.eye(num_queries, dtype=np.float64)
    try:
        dual_weights = np.linalg.solve(regularized, values)
    except np.linalg.LinAlgError:
        dual_weights = np.linalg.pinv(regularized) @ values
    return MultimodalTensorKernelRegressor(
        train_selection_matrices=selection.astype(np.float64),
        train_query_masks=query.astype(np.float64),
        dual_weights=dual_weights,
        sample_gamma=sample_gamma,
        modality_gamma=modality_gamma,
        query_gamma=query_gamma,
    )


def enumerate_multimodal_query_states(num_samples: int, num_modalities: int) -> tuple[np.ndarray, np.ndarray]:
    if num_samples <= 0 or num_modalities <= 0:
        raise ValueError("num_samples and num_modalities must be positive")
    states = []
    queries = []
    total_selection_bits = num_samples * num_modalities
    for selection_bits in product([False, True], repeat=total_selection_bits):
        matrix = np.asarray(selection_bits, dtype=bool).reshape(num_samples, num_modalities)
        for query_bits in product([False, True], repeat=num_modalities):
            states.append(matrix.copy())
            queries.append(np.asarray(query_bits, dtype=bool))
    return np.asarray(states, dtype=bool), np.asarray(queries, dtype=bool)


def compute_csmi_from_oracle(
    oracle: callable,
    *,
    num_samples: int,
    num_modalities: int,
    sample_index: int,
    train_modalities: Sequence[int] | Sequence[bool] | np.ndarray,
    query_modalities: Sequence[int] | Sequence[bool] | np.ndarray,
    base_selection_matrix: Sequence[Sequence[bool]] | np.ndarray | None = None,
    base_query_mask: Sequence[bool] | np.ndarray | None = None,
) -> float:
    if not (0 <= sample_index < num_samples):
        raise IndexError(f"sample_index must lie in [0, {num_samples}), got {sample_index}")
    base_selection = (
        np.ones((num_samples, num_modalities), dtype=bool)
        if base_selection_matrix is None
        else _as_selection_batch(base_selection_matrix, num_samples, num_modalities)[0].astype(bool)
    )
    base_query = (
        np.ones(num_modalities, dtype=bool)
        if base_query_mask is None
        else _as_query_batch(base_query_mask, num_modalities)[0].astype(bool)
    )
    train_drop = _as_modality_mask(train_modalities, num_modalities)
    query_drop = _as_modality_mask(query_modalities, num_modalities)
    dropped_selection = base_selection.copy()
    dropped_selection[sample_index, train_drop] = False
    dropped_query = base_query.copy()
    dropped_query[query_drop] = False
    return float(
        oracle(base_selection, base_query)
        - oracle(dropped_selection, base_query)
        - oracle(base_selection, dropped_query)
        + oracle(dropped_selection, dropped_query)
    )


def _as_selection_batch(
    selection_matrices: Sequence[Sequence[Sequence[bool]]] | np.ndarray,
    num_samples: int,
    num_modalities: int,
) -> np.ndarray:
    values = np.asarray(selection_matrices, dtype=bool)
    if values.ndim == 2:
        values = values.reshape(1, num_samples, num_modalities)
    if values.ndim != 3 or values.shape[1:] != (num_samples, num_modalities):
        raise ValueError(f"expected selection matrices with shape [batch, {num_samples}, {num_modalities}]")
    return values.astype(np.float64)


def _as_query_batch(
    query_masks: Sequence[Sequence[bool]] | np.ndarray,
    num_modalities: int,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    values = np.asarray(query_masks, dtype=bool)
    if values.ndim == 1:
        values = values.reshape(1, num_modalities)
    if values.ndim != 2 or values.shape[1] != num_modalities:
        raise ValueError(f"expected query masks with shape [batch, {num_modalities}]")
    if batch_size is not None and values.shape[0] != batch_size:
        raise ValueError(f"expected {batch_size} query masks, got {values.shape[0]}")
    return values.astype(np.float64)


def _as_modality_mask(modalities: Sequence[int] | Sequence[bool] | np.ndarray, num_modalities: int) -> np.ndarray:
    values = np.asarray(modalities)
    if values.dtype == bool:
        mask = values.reshape(-1)
        if mask.shape[0] != num_modalities:
            raise ValueError(f"expected modality mask with {num_modalities} entries")
        return mask.astype(bool)
    indices = np.asarray(modalities, dtype=np.int64).reshape(-1)
    mask = np.zeros(num_modalities, dtype=bool)
    for index in indices:
        if not (0 <= int(index) < num_modalities):
            raise IndexError(f"modality index must lie in [0, {num_modalities}), got {int(index)}")
        mask[int(index)] = True
    return mask


def _tensor_product_kernel(
    *,
    lhs_selection: np.ndarray,
    rhs_selection: np.ndarray,
    lhs_query: np.ndarray,
    rhs_query: np.ndarray,
    sample_gamma: float,
    modality_gamma: float,
    query_gamma: float,
) -> np.ndarray:
    sample_kernel = _mean_axis_rbf_kernel(lhs_selection, rhs_selection, gamma=sample_gamma, axis=1)
    modality_kernel = _mean_axis_rbf_kernel(lhs_selection, rhs_selection, gamma=modality_gamma, axis=2)
    query_kernel = _rbf_kernel(lhs_query, rhs_query, query_gamma)
    return sample_kernel * modality_kernel * query_kernel


def _mean_axis_rbf_kernel(lhs: np.ndarray, rhs: np.ndarray, *, gamma: float, axis: int) -> np.ndarray:
    if axis == 1:
        num_components = lhs.shape[1]
        result = np.zeros((lhs.shape[0], rhs.shape[0]), dtype=np.float64)
        for component_index in range(num_components):
            result += _rbf_kernel(lhs[:, component_index, :], rhs[:, component_index, :], gamma)
        return result / max(num_components, 1)
    if axis == 2:
        num_components = lhs.shape[2]
        result = np.zeros((lhs.shape[0], rhs.shape[0]), dtype=np.float64)
        for component_index in range(num_components):
            result += _rbf_kernel(lhs[:, :, component_index], rhs[:, :, component_index], gamma)
        return result / max(num_components, 1)
    raise ValueError(f"unsupported axis={axis}; expected 1 (samples) or 2 (modalities)")


def _rbf_kernel(lhs: np.ndarray, rhs: np.ndarray, gamma: float) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lhs_sq = np.sum(lhs * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum(rhs * rhs, axis=1, keepdims=True).T
    squared_distance = np.clip(lhs_sq + rhs_sq - 2.0 * (lhs @ rhs.T), a_min=0.0, a_max=None)
    return np.exp(-float(gamma) * squared_distance)


def _resolve_component_gamma(inputs: np.ndarray, configured_gamma: float | None) -> float:
    if configured_gamma is not None:
        return float(configured_gamma)
    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.shape[0] <= 1:
        return 1.0
    lhs_sq = np.sum(inputs * inputs, axis=1, keepdims=True)
    squared_distance = np.clip(lhs_sq + lhs_sq.T - 2.0 * (inputs @ inputs.T), a_min=0.0, a_max=None)
    upper = squared_distance[np.triu_indices_from(squared_distance, k=1)]
    positive = upper[upper > 1e-12]
    if positive.size == 0:
        return 1.0
    return float(1.0 / np.median(positive))


__all__ = [
    "MultimodalTensorKernelConfig",
    "MultimodalTensorKernelRegressor",
    "compute_csmi_from_oracle",
    "enumerate_multimodal_query_states",
    "fit_multimodal_tensor_kernel_regressor",
]
