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
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.precision_utils import (
    _promote_precision_float32 as _promote_precision,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _select_probe_text(probe: Any) -> str | None:
    """Pick a usable probe text from the probe definition."""
    probe_text = None
    for candidate in probe.support_texts or []:
        if not candidate or not candidate.strip():
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
        if fallback and fallback.strip():
            probe_text = fallback
    return probe_text


def _infer_required_probe_count(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
) -> tuple[int, int, int]:
    """Infer minimum probe count from hidden dimensions.

    The alignment matrix F = pinv(source_acts) @ target_acts requires numerical
    stability. Geometry requires n_probes > max(source_dim, target_dim) to avoid
    singular Gram matrix.

    We use the strict algebraic minimum: n > max_dim (i.e., n = max_dim + 1).
    This is the smallest count that guarantees a non-singular Gram matrix.

    The runtime condition number check in GramAligner determines actual stability.
    If κ × ε is too large, the alignment will warn and the user can add more probes
    via --full-atlas. We don't pretend to know the "right" overdetermination factor
    for neural activations - that depends on the actual activation structure.

    Returns:
        (min_required, source_dim, target_dim)

    Raises:
        RuntimeError: If hidden dimensions cannot be inferred from weights.
    """
    from modelcypher.core.use_cases.merge.helpers import infer_hidden_dim

    source_dim = infer_hidden_dim(source_weights)
    target_dim = infer_hidden_dim(target_weights)

    if source_dim <= 0 or target_dim <= 0:
        raise RuntimeError(
            "PROBE: Unable to infer hidden dimensions from weights. "
            "Hidden dimensions are required to derive the minimum probe count."
        )

    # Strict algebraic minimum: n > max_dim for non-singular Gram matrix.
    # Runtime condition number check determines actual numerical stability.
    max_dim = max(source_dim, target_dim)
    min_required = max_dim + 1

    return min_required, source_dim, target_dim


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
