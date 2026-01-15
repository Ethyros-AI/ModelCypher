# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Shared helpers for probe stage (selection, precision, and activation slicing)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    ceil_scalar,
    log2_scalar,
    machine_epsilon,
    precision_dtype,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _select_probe_text(probe: Any) -> str | None:
    """Pick a usable probe text from the probe definition."""
    probe_text = None
    for candidate in probe.support_texts or []:
        if not candidate or len(candidate.strip()) < 2:
            continue
        probe_text = candidate
        break

    if probe_text is None:
        fallback = None
        if probe.name and probe.description:
            fallback = f"{probe.name}: {probe.description}"
        elif probe.name:
            fallback = probe.name
        elif probe.description:
            fallback = probe.description
        if fallback and len(fallback.strip()) >= 2:
            probe_text = fallback
    return probe_text


def _infer_required_probe_count(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
) -> tuple[int, int, int]:
    """Infer exact probe count from intrinsic rank (weight-space spectrum).

    Closed-form alignment on the shared manifold requires n >= rank(source)
    and n >= rank(target). We estimate rank via singular value support of
    per-layer weight matrices using machine-epsilon thresholds (no heuristics).

    Returns:
        (min_required, source_rank, target_rank)
    """
    from modelcypher.core.use_cases.merge.helpers import extract_layer_index, infer_hidden_dim

    backend = get_default_backend()

    def _spectral_rank(matrix: Any) -> int | None:
        if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
            return None
        arr = backend.array(matrix)
        arr = backend.astype(arr, precision_dtype(backend, reference=arr))
        m = int(arr.shape[0])
        n = int(arr.shape[1])
        if m == 0 or n == 0:
            return 0
        if m >= n:
            gram = backend.matmul(backend.transpose(arr), arr)
        else:
            gram = backend.matmul(arr, backend.transpose(arr))
        backend.eval(gram)
        eigenvalues = backend.eigvalsh(gram)
        eigenvalues = backend.maximum(eigenvalues, backend.zeros_like(eigenvalues))
        s = backend.sqrt(eigenvalues)
        backend.eval(s)
        s_max_arr = backend.max(s)
        backend.eval(s_max_arr)
        s_max = float(backend.to_scalar(s_max_arr))
        if s_max <= 0:
            return 0
        eps = machine_epsilon(backend, s)
        threshold = s_max * sqrt_scalar(eps, backend)
        count_arr = backend.sum(backend.astype(s > threshold, "int32"))
        backend.eval(count_arr)
        return int(backend.to_scalar(count_arr))

    def _model_intrinsic_rank(weights: dict[str, Any]) -> int:
        per_layer: dict[int, int] = {}
        for key, val in weights.items():
            layer_idx = extract_layer_index(key)
            if layer_idx is None:
                continue
            rank = _spectral_rank(val)
            if rank is None:
                continue
            current = per_layer.get(layer_idx, 0)
            per_layer[layer_idx] = rank if rank > current else current
        if not per_layer:
            return 0
        return max(per_layer.values())

    source_rank = _model_intrinsic_rank(source_weights)
    target_rank = _model_intrinsic_rank(target_weights)

    if source_rank <= 0:
        source_rank = infer_hidden_dim(source_weights)
    if target_rank <= 0:
        target_rank = infer_hidden_dim(target_weights)

    min_required = max(source_rank, target_rank)
    return min_required, source_rank, target_rank


def _select_geometry_probes(
    probes: list[tuple[Any, str]],
    required_count: int,
) -> list[tuple[Any, str]]:
    """Select a geometry-sized probe set with broad domain coverage."""
    if required_count <= 0 or required_count >= len(probes):
        return list(probes)

    # De-duplicate by probe_id while preserving deterministic order.
    seen: set[str] = set()
    unique: list[tuple[Any, str]] = []
    for probe, text in probes:
        probe_id = probe.probe_id
        if probe_id in seen:
            continue
        seen.add(probe_id)
        unique.append((probe, text))

    if required_count >= len(unique):
        return unique

    # Round-robin by domain for broad coverage.
    buckets: dict[str, list[tuple[Any, str]]] = {}
    for probe, text in unique:
        domain = probe.domain.value if hasattr(probe.domain, "value") else str(probe.domain)
        buckets.setdefault(domain, []).append((probe, text))

    for domain, items in buckets.items():
        buckets[domain] = sorted(items, key=lambda item: item[0].probe_id)

    domains = sorted(buckets.keys())
    selected: list[tuple[Any, str]] = []
    index = 0
    while len(selected) < required_count:
        added = False
        for domain in domains:
            items = buckets[domain]
            if index < len(items):
                selected.append(items[index])
                added = True
                if len(selected) >= required_count:
                    break
        if not added:
            break
        index += 1

    if len(selected) < required_count:
        selected_ids = {probe.probe_id for probe, _ in selected}
        remaining = [item for item in unique if item[0].probe_id not in selected_ids]
        for probe, text in sorted(remaining, key=lambda item: item[0].probe_id):
            selected.append((probe, text))
            if len(selected) >= required_count:
                break

    return selected


def _proportional_layer_index(
    target_idx: int,
    target_count: int,
    source_count: int,
) -> int:
    """Map a target layer index to a source index by normalized depth."""
    if target_count <= 1 or source_count <= 1:
        return 0
    ratio = target_idx / (target_count - 1)
    mapped = int(round(ratio * (source_count - 1)))
    if mapped < 0:
        return 0
    if mapped >= source_count:
        return source_count - 1
    return mapped


def _dtype_name(dtype: Any) -> str:
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    return name.replace("mlx.core.", "").replace("jax.numpy.", "")


def _promote_precision(
    array: "Array",
    backend: "Backend",
    *,
    min_dtype: str = "float32",
) -> "Array":
    """Promote lower-precision activations to at least float32."""
    if not hasattr(array, "dtype"):
        return backend.array(array, dtype=min_dtype)

    dtype_name = _dtype_name(array.dtype)
    if "float16" in dtype_name or "bfloat16" in dtype_name:
        return backend.astype(array, min_dtype)

    try:
        current_eps = backend.finfo(array.dtype).eps
        min_eps = backend.finfo(min_dtype).eps
    except Exception:
        return backend.astype(array, min_dtype)

    if current_eps > min_eps:
        return backend.astype(array, min_dtype)

    return array


def _precision_reference(
    backend: "Backend",
    *candidates: Any,
    default_dtype: str = "float32",
) -> "Array":
    """Pick a representative array for precision thresholds."""
    def _first_array(container: Any) -> "Array | None":
        if container is None:
            return None
        if hasattr(container, "dtype"):
            return container
        if hasattr(container, "keys") and hasattr(container, "__getitem__"):
            try:
                keys = container.keys()
                first_key = next(iter(keys)) if keys else None
                if first_key is not None:
                    value = container[first_key]
                    if hasattr(value, "dtype"):
                        return value
            except Exception:
                return None
        if isinstance(container, dict):
            for value in container.values():
                if hasattr(value, "dtype"):
                    return value
        if isinstance(container, (list, tuple)):
            for value in container:
                if hasattr(value, "dtype"):
                    return value
        if hasattr(container, "items"):
            try:
                for _key, value in container.items():
                    if hasattr(value, "dtype"):
                        return value
            except Exception:
                return None
        return None

    for candidate in candidates:
        arr = _first_array(candidate)
        if arr is not None:
            return arr

    return backend.array([1.0], dtype=default_dtype)


def _extract_top_k_dims(
    activation_vector: "Array",
    k: int | None = None,
    threshold: float | None = None,
    backend: "Backend | None" = None,
) -> list:
    """Extract top-k activated dimensions by magnitude."""
    from modelcypher.core.domain.geometry.manifold_stitcher import ActivatedDimension

    b = backend or get_default_backend()
    abs_vals = b.abs(activation_vector)
    b.eval(abs_vals)

    dim = b.shape(activation_vector)[0]

    # Derive k from dimensionality: ceil(log2(d)) captures information-theoretic complexity
    if k is None:
        log2_dim = log2_scalar(float(dim + 1), b)
        k = max(1, int(ceil_scalar(log2_dim, b)))

    # Clamp k to array dimension to avoid argpartition out-of-bounds
    k = min(k, dim)

    # Derive threshold from dtype precision scaled by max magnitude (use backend ops)
    max_magnitude_arr = b.max(abs_vals)
    b.eval(max_magnitude_arr)
    max_magnitude = float(b.to_scalar(max_magnitude_arr))
    if threshold is None:
        eps = machine_epsilon(b, activation_vector)
        # Threshold at sqrt(eps) * max - standard numerical tolerance
        threshold = sqrt_scalar(eps, b) * max_magnitude

    # Negate for descending selection
    neg_abs = -abs_vals
    b.eval(neg_abs)
    kth = max(0, k - 1)
    partitioned = b.argpartition(neg_abs, kth)
    top_indices_arr = b.take(partitioned, b.arange(k), axis=0)
    b.eval(top_indices_arr)

    # Convert to Python using native tolist - no NumPy
    top_indices = sorted([int(x) for x in b.tolist(top_indices_arr)])
    indices_arr = b.array(top_indices, dtype="int32")
    selected_acts = b.take(activation_vector, indices_arr, axis=0)
    selected_abs = b.take(abs_vals, indices_arr, axis=0)
    b.eval(selected_acts, selected_abs)

    # Use native tolist() for O(1) extraction
    selected_acts_list = b.tolist(selected_acts)
    selected_abs_list = b.tolist(selected_abs)
    return [
        ActivatedDimension(
            index=int(idx),
            activation=float(selected_acts_list[i]),
        )
        for i, idx in enumerate(top_indices)
        if float(selected_abs_list[i]) > threshold
    ]
