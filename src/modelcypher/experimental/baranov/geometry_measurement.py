"""Geometry measurement helpers for Baranov replication.

EXPERIMENTAL: Not validated for production use.

Provides functions for collecting hidden-state activations and computing
CKA drift between pre- and post-intervention model states.  All CKA
computation delegates to ``modelcypher.core.domain.geometry.cka``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeometrySnapshot:
    """Per-layer activation snapshot for a single model state.

    ``activations`` maps layer index to a list of pooled activation
    arrays (one per probe text), each of shape ``[hidden_dim]``.
    """

    activations: dict[int, list[Any]]
    probe_texts: tuple[str, ...]
    n_layers: int


@dataclass(frozen=True)
class CKADriftResult:
    """CKA drift between two model states.

    Attributes
    ----------
    per_layer_cka:
        Linear CKA per layer (1.0 = identical, 0.0 = orthogonal).
    min_cka:
        Worst-case (minimum) per-layer CKA.
    mean_cka:
        Average per-layer CKA.
    cka_drift:
        ``1.0 - min_cka`` — maximum drift across all layers.
    preserved_fraction:
        ``mean_cka`` — average preservation (no arbitrary threshold).
    """

    per_layer_cka: dict[int, float]
    min_cka: float
    mean_cka: float
    cka_drift: float
    preserved_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_layer_cka": self.per_layer_cka,
            "min_cka": self.min_cka,
            "mean_cka": self.mean_cka,
            "cka_drift": self.cka_drift,
            "preserved_fraction": self.preserved_fraction,
        }


def collect_probe_activations(
    model: Any,
    tokenizer: Any,
    probe_texts: list[str],
    backend: Any,
    layer_indices: list[int] | None = None,
) -> GeometrySnapshot:
    """Collect mean-pooled hidden activations on probe texts.

    For each probe text, collects hidden states at every layer (or at
    ``layer_indices`` if specified), mean-pools over the sequence
    dimension, and stores the result.

    This mirrors the activation collection pattern from
    ``DatasetTrainingService._collect_probe_activations``.

    Parameters
    ----------
    model:
        Model object from ``backend.load_model``.
    tokenizer:
        Tokenizer from ``backend.load_model``.
    probe_texts:
        Texts to probe the model with.
    backend:
        Backend instance for tensor operations.
    layer_indices:
        Optional subset of layers to collect (None = all).

    Returns
    -------
    GeometrySnapshot with per-layer activations.
    """
    activations: dict[int, list[Any]] = {}

    for text in probe_texts:
        acts = backend.collect_hidden_activations(
            model,
            tokenizer,
            [text],
            layer_indices=layer_indices,
        )
        # acts: {layer_idx: Array[1, seq_len, hidden_dim]}
        for layer_idx, act in acts.items():
            # Mean-pool over sequence dimension
            pooled = backend.mean(act, axis=1)  # [1, hidden]
            pooled = backend.reshape(pooled, (-1,))  # [hidden]
            backend.eval(pooled)
            activations.setdefault(layer_idx, []).append(pooled)

    n_layers = len(activations)
    logger.info(
        "Collected activations: %d layers × %d probes",
        n_layers,
        len(probe_texts),
    )
    return GeometrySnapshot(
        activations=activations,
        probe_texts=tuple(probe_texts),
        n_layers=n_layers,
    )


def compute_cka_drift(
    pre: GeometrySnapshot,
    post: GeometrySnapshot,
    backend: Any,
) -> CKADriftResult:
    """Compute per-layer CKA drift between two model states.

    Uses linear CKA (fast, O(n²d)) from the shared geometry module.

    Parameters
    ----------
    pre:
        Activations from the pre-intervention model.
    post:
        Activations from the post-intervention model.
    backend:
        Backend instance for tensor operations.

    Returns
    -------
    CKADriftResult with per-layer CKA and aggregate metrics.
    """
    from modelcypher.core.domain.geometry.cka import (
        compute_linear_cka_from_activations,
    )

    common_layers = sorted(set(pre.activations) & set(post.activations))
    if not common_layers:
        raise ValueError("No common layers between pre and post snapshots")

    per_layer_cka: dict[int, float] = {}
    for layer_idx in common_layers:
        pre_stack = backend.stack(pre.activations[layer_idx])
        post_stack = backend.stack(post.activations[layer_idx])
        backend.eval(pre_stack, post_stack)

        cka = compute_linear_cka_from_activations(
            pre_stack,
            post_stack,
            backend,
        )
        per_layer_cka[layer_idx] = float(cka)

    cka_values = list(per_layer_cka.values())
    min_cka = min(cka_values)
    mean_cka = sum(cka_values) / len(cka_values)

    result = CKADriftResult(
        per_layer_cka=per_layer_cka,
        min_cka=min_cka,
        mean_cka=mean_cka,
        cka_drift=1.0 - min_cka,
        preserved_fraction=mean_cka,
    )

    logger.info(
        "CKA drift: min=%.4f mean=%.4f drift=%.4f preserved=%.4f (%d layers)",
        result.min_cka,
        result.mean_cka,
        result.cka_drift,
        result.preserved_fraction,
        len(common_layers),
    )
    return result


__all__ = [
    "CKADriftResult",
    "GeometrySnapshot",
    "collect_probe_activations",
    "compute_cka_drift",
]
