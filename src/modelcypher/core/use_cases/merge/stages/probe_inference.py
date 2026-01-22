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

"""Probe inference loop for activation collection (strict, batched).

Supports two execution paths:
1. Parallel path (legacy): Both models in memory, process together
2. Sequential path (memory-efficient): One model at a time, page to disk between
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.use_cases.merge.stages.probe_activation_storage import (
    _flush_batch_activations,
    _page_activation_space,
    PagedActivations,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.activation_store import ActivationStore
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class SingleModelActivations:
    """Activations collected from a single model."""

    hidden: dict[int, "Array"]
    intermediate: dict[int, "Array"]
    gate: dict[int, "Array"]
    embedding: list["Array"]


def _clear_gpu_memory() -> None:
    """Clear GPU memory caches."""
    gc.collect()
    try:
        import mlx.core as mx

        mx.eval()
        mx.clear_cache()
    except ImportError:
        pass


def run_single_model_probe_inference(
    *,
    valid_probes: list[tuple[Any, str]],
    model: Any,
    tokenizer: Any,
    activation_provider: "ActivationProvider",
    backend: "Backend",
    model_label: str = "model",
) -> SingleModelActivations:
    """Run probe inference on a SINGLE model.

    This is the memory-efficient version that processes one model at a time,
    allowing the caller to unload the model before processing the next one.

    Args:
        valid_probes: List of (probe_id, probe_text) tuples.
        model: The model to collect activations from.
        tokenizer: Tokenizer for the model.
        activation_provider: Provider for collecting activations.
        backend: Compute backend.
        model_label: Label for logging ("source" or "target").

    Returns:
        SingleModelActivations with hidden, intermediate, gate, and embedding activations.
    """
    if not hasattr(activation_provider, "collect_probe_activations_batch"):
        raise RuntimeError(
            "Activation provider must implement collect_probe_activations_batch"
        )

    total_probes = len(valid_probes)
    if total_probes == 0:
        return SingleModelActivations(
            hidden={}, intermediate={}, gate={}, embedding=[]
        )

    # Storage buffers
    layer_activations: dict[int, "Array"] = {}
    intermediate_activations: dict[int, "Array"] = {}
    gate_activations: dict[int, "Array"] = {}
    embedding_activations: list["Array"] = []

    probe_batch_size = 1
    n_batches = (total_probes + probe_batch_size - 1) // probe_batch_size
    logger.info(
        "PROBE %s: %d probes in %d batches...",
        model_label.upper(),
        total_probes,
        n_batches,
    )

    probes_processed = 0
    for batch_start in range(0, total_probes, probe_batch_size):
        batch_end = min(batch_start + probe_batch_size, total_probes)
        batch = valid_probes[batch_start:batch_end]
        batch_texts = [probe_text for _, probe_text in batch]
        batch_size = len(batch_texts)

        batch_data = activation_provider.collect_probe_activations_batch(
            model, tokenizer, batch_texts
        )

        # Validate batch
        if len(batch_data.hidden) != batch_size:
            raise RuntimeError(
                f"{model_label} hidden batch mismatch: {len(batch_data.hidden)} != {batch_size}"
            )

        # Accumulate activations
        hidden_accum: dict[int, list["Array"]] = {}
        hidden_indices: dict[int, list[int]] = {}
        inter_accum: dict[int, list["Array"]] = {}
        inter_indices: dict[int, list[int]] = {}
        gate_accum: dict[int, list["Array"]] = {}
        gate_indices: dict[int, list[int]] = {}

        for i in range(batch_size):
            probe_index = batch_start + i

            for layer_idx, act in batch_data.hidden[i].items():
                hidden_accum.setdefault(layer_idx, []).append(act)
                hidden_indices.setdefault(layer_idx, []).append(probe_index)

            for layer_idx, act in batch_data.intermediate[i].items():
                inter_accum.setdefault(layer_idx, []).append(act)
                inter_indices.setdefault(layer_idx, []).append(probe_index)

            for layer_idx, act in batch_data.gate[i].items():
                gate_accum.setdefault(layer_idx, []).append(act)
                gate_indices.setdefault(layer_idx, []).append(probe_index)

        embedding_activations.extend(batch_data.embedding)

        _flush_batch_activations(
            layer_activations, hidden_accum, hidden_indices, backend, total_probes
        )
        _flush_batch_activations(
            intermediate_activations, inter_accum, inter_indices, backend, total_probes
        )
        _flush_batch_activations(
            gate_activations, gate_accum, gate_indices, backend, total_probes
        )

        probes_processed += batch_size
        if batch_end % 100 <= probe_batch_size:
            logger.info(
                "PROBE %s: %d/%d probes...",
                model_label.upper(),
                probes_processed,
                total_probes,
            )

        _clear_gpu_memory()

    logger.info("PROBE %s: Complete (%d probes)", model_label.upper(), probes_processed)

    return SingleModelActivations(
        hidden=layer_activations,
        intermediate=intermediate_activations,
        gate=gate_activations,
        embedding=embedding_activations,
    )


def page_activations_to_disk(
    activations: SingleModelActivations,
    paging_dir: Path,
    prefix: str,
    activation_store: "ActivationStore",
    backend: "Backend",
) -> tuple[PagedActivations, PagedActivations, PagedActivations, list["Array"]]:
    """Page model activations to disk and return lazy loaders.

    Args:
        activations: Collected activations from a model.
        paging_dir: Directory to store paged activations.
        prefix: Prefix for filenames (e.g., "source" or "target").
        activation_store: Store adapter for persistence.
        backend: Compute backend.

    Returns:
        Tuple of (hidden_paged, intermediate_paged, gate_paged, embedding_list).
        The first three are PagedActivations (lazy disk-backed).
        Embeddings are kept in memory as they're typically small.
    """
    paging_dir.mkdir(parents=True, exist_ok=True)

    hidden_paged = _page_activation_space(
        activation_store,
        paging_dir / "hidden",
        f"{prefix}_hidden",
        activations.hidden,
        backend,
    )

    intermediate_paged = _page_activation_space(
        activation_store,
        paging_dir / "intermediate",
        f"{prefix}_intermediate",
        activations.intermediate,
        backend,
    )

    gate_paged = _page_activation_space(
        activation_store,
        paging_dir / "gate",
        f"{prefix}_gate",
        activations.gate,
        backend,
    )

    logger.info(
        "PAGING %s: Hidden=%d layers, Intermediate=%d layers, Gate=%d layers",
        prefix.upper(),
        len(hidden_paged),
        len(intermediate_paged),
        len(gate_paged),
    )

    return hidden_paged, intermediate_paged, gate_paged, activations.embedding


def run_sequential_probe_inference(
    *,
    valid_probes: list[tuple[Any, str]],
    source_model: Any,
    target_model: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    activation_provider: "ActivationProvider",
    backend: "Backend",
    paging_dir: Path,
    activation_store: "ActivationStore",
    unload_source_callback: Callable[[], None] | None = None,
) -> tuple[
    PagedActivations,  # source_hidden
    PagedActivations,  # target_hidden
    PagedActivations,  # source_intermediate
    PagedActivations,  # target_intermediate
    PagedActivations,  # source_gate
    PagedActivations,  # target_gate
    list["Array"],  # source_embedding
    list["Array"],  # target_embedding
    int,  # probes_processed
]:
    """Run probe inference SEQUENTIALLY to minimize memory usage.

    This processes source model first, pages activations to disk, unloads source,
    then processes target model. Peak memory is ONE model + activations, not two.

    Args:
        valid_probes: List of (probe_id, probe_text) tuples.
        source_model: Source model for probing.
        target_model: Target model for probing.
        source_tokenizer: Source tokenizer.
        target_tokenizer: Target tokenizer.
        activation_provider: Provider for collecting activations.
        backend: Compute backend.
        paging_dir: Directory for paging activations to disk.
        activation_store: Store adapter for persistence.
        unload_source_callback: Optional callback to unload source model after probing.

    Returns:
        Tuple of paged activations and probe count.
    """
    total_probes = len(valid_probes)
    if total_probes == 0:
        empty_paged = PagedActivations(
            paging_dir, "empty", [], activation_store, backend
        )
        return (
            empty_paged,
            empty_paged,
            empty_paged,
            empty_paged,
            empty_paged,
            empty_paged,
            [],
            [],
            0,
        )

    # Phase 1: Process source model
    logger.info("SEQUENTIAL PROBE: Phase 1 - Source model")
    source_activations = run_single_model_probe_inference(
        valid_probes=valid_probes,
        model=source_model,
        tokenizer=source_tokenizer,
        activation_provider=activation_provider,
        backend=backend,
        model_label="source",
    )

    # Page source activations to disk
    (
        source_hidden_paged,
        source_intermediate_paged,
        source_gate_paged,
        source_embedding,
    ) = page_activations_to_disk(
        source_activations,
        paging_dir / "source",
        "source",
        activation_store,
        backend,
    )

    # Clear source activation memory
    del source_activations
    _clear_gpu_memory()

    # Optional: unload source model to free GPU memory
    if unload_source_callback is not None:
        logger.info("SEQUENTIAL PROBE: Unloading source model")
        unload_source_callback()
        _clear_gpu_memory()

    # Phase 2: Process target model
    logger.info("SEQUENTIAL PROBE: Phase 2 - Target model")
    target_activations = run_single_model_probe_inference(
        valid_probes=valid_probes,
        model=target_model,
        tokenizer=target_tokenizer,
        activation_provider=activation_provider,
        backend=backend,
        model_label="target",
    )

    # Page target activations to disk
    (
        target_hidden_paged,
        target_intermediate_paged,
        target_gate_paged,
        target_embedding,
    ) = page_activations_to_disk(
        target_activations,
        paging_dir / "target",
        "target",
        activation_store,
        backend,
    )

    # Clear target activation memory
    del target_activations
    _clear_gpu_memory()

    logger.info("SEQUENTIAL PROBE: Complete (%d probes processed)", total_probes)

    return (
        source_hidden_paged,
        target_hidden_paged,
        source_intermediate_paged,
        target_intermediate_paged,
        source_gate_paged,
        target_gate_paged,
        source_embedding,
        target_embedding,
        total_probes,
    )


def run_probe_inference(
    *,
    valid_probes: list[tuple[Any, str]],
    source_model: Any,
    target_model: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    activation_provider: "ActivationProvider",
    backend: "Backend",
    source_layer_activations: dict[int, "Array"],
    target_layer_activations: dict[int, "Array"],
    source_intermediate_activations: dict[int, "Array"],
    target_intermediate_activations: dict[int, "Array"],
    source_gate_activations: dict[int, "Array"],
    target_gate_activations: dict[int, "Array"],
    source_embedding_activations: list["Array"] | "Array",
    target_embedding_activations: list["Array"] | "Array",
) -> tuple[int, int]:
    """Run batched probing and fill activation buffers."""
    if not hasattr(activation_provider, "collect_probe_activations_batch"):
        raise RuntimeError(
            "Activation provider must implement collect_probe_activations_batch for strict probing."
        )

    probes_processed = 0
    total_probes = len(valid_probes)
    if total_probes == 0:
        return 0, 0

    # Algebraic minimum to avoid heuristic batching.
    probe_batch_size = 1
    n_batches = (total_probes + probe_batch_size - 1) // probe_batch_size
    logger.info(
        "PROBE PRECISE: %d valid probes, processing in %d batches of %d...",
        total_probes, n_batches, probe_batch_size,
    )

    def _validate_batch(label: str, batch_data: Any, expected: int) -> None:
        if len(batch_data.hidden) != expected:
            raise RuntimeError(
                f"{label} hidden batch size mismatch: {len(batch_data.hidden)} != {expected}"
            )
        if len(batch_data.intermediate) != expected:
            raise RuntimeError(
                f"{label} intermediate batch size mismatch: {len(batch_data.intermediate)} != {expected}"
            )
        if len(batch_data.gate) != expected:
            raise RuntimeError(
                f"{label} gate batch size mismatch: {len(batch_data.gate)} != {expected}"
            )
        if len(batch_data.embedding) != expected:
            raise RuntimeError(
                f"{label} embedding batch size mismatch: {len(batch_data.embedding)} != {expected}"
            )

    for batch_start in range(0, total_probes, probe_batch_size):
        batch_end = min(batch_start + probe_batch_size, total_probes)
        batch = valid_probes[batch_start:batch_end]
        batch_texts = [probe_text for _, probe_text in batch]
        batch_size = len(batch_texts)

        source_batch = activation_provider.collect_probe_activations_batch(
            source_model, source_tokenizer, batch_texts
        )
        target_batch = activation_provider.collect_probe_activations_batch(
            target_model, target_tokenizer, batch_texts
        )

        _validate_batch("source", source_batch, batch_size)
        _validate_batch("target", target_batch, batch_size)

        source_hidden_accum: dict[int, list["Array"]] = {}
        source_hidden_indices: dict[int, list[int]] = {}
        target_hidden_accum: dict[int, list["Array"]] = {}
        target_hidden_indices: dict[int, list[int]] = {}
        source_inter_accum: dict[int, list["Array"]] = {}
        source_inter_indices: dict[int, list[int]] = {}
        target_inter_accum: dict[int, list["Array"]] = {}
        target_inter_indices: dict[int, list[int]] = {}
        source_gate_accum: dict[int, list["Array"]] = {}
        source_gate_indices: dict[int, list[int]] = {}
        target_gate_accum: dict[int, list["Array"]] = {}
        target_gate_indices: dict[int, list[int]] = {}

        for i in range(batch_size):
            probe_index = batch_start + i

            source_acts = source_batch.hidden[i]
            target_acts = target_batch.hidden[i]
            source_intermediate_acts = source_batch.intermediate[i]
            target_intermediate_acts = target_batch.intermediate[i]
            source_gate_acts = source_batch.gate[i]
            target_gate_acts = target_batch.gate[i]

            for layer_idx, act in source_acts.items():
                source_hidden_accum.setdefault(layer_idx, []).append(act)
                source_hidden_indices.setdefault(layer_idx, []).append(probe_index)

            for layer_idx, act in target_acts.items():
                target_hidden_accum.setdefault(layer_idx, []).append(act)
                target_hidden_indices.setdefault(layer_idx, []).append(probe_index)

            for layer_idx, act in source_intermediate_acts.items():
                source_inter_accum.setdefault(layer_idx, []).append(act)
                source_inter_indices.setdefault(layer_idx, []).append(probe_index)

            for layer_idx, act in target_intermediate_acts.items():
                target_inter_accum.setdefault(layer_idx, []).append(act)
                target_inter_indices.setdefault(layer_idx, []).append(probe_index)

            for layer_idx, act in source_gate_acts.items():
                source_gate_accum.setdefault(layer_idx, []).append(act)
                source_gate_indices.setdefault(layer_idx, []).append(probe_index)

            for layer_idx, act in target_gate_acts.items():
                target_gate_accum.setdefault(layer_idx, []).append(act)
                target_gate_indices.setdefault(layer_idx, []).append(probe_index)

        if isinstance(source_embedding_activations, list):
            source_embedding_activations.extend(source_batch.embedding)
        if isinstance(target_embedding_activations, list):
            target_embedding_activations.extend(target_batch.embedding)

        _flush_batch_activations(
            source_layer_activations,
            source_hidden_accum,
            source_hidden_indices,
            backend,
            total_probes,
        )
        _flush_batch_activations(
            source_intermediate_activations,
            source_inter_accum,
            source_inter_indices,
            backend,
            total_probes,
        )
        _flush_batch_activations(
            target_layer_activations,
            target_hidden_accum,
            target_hidden_indices,
            backend,
            total_probes,
        )
        _flush_batch_activations(
            target_intermediate_activations,
            target_inter_accum,
            target_inter_indices,
            backend,
            total_probes,
        )
        _flush_batch_activations(
            source_gate_activations,
            source_gate_accum,
            source_gate_indices,
            backend,
            total_probes,
        )
        _flush_batch_activations(
            target_gate_activations,
            target_gate_accum,
            target_gate_indices,
            backend,
            total_probes,
        )

        probes_processed += batch_size
        if batch_end % 50 <= probe_batch_size:
            logger.info(
                "PROBE PRECISE: Processed %d/%d probes...",
                probes_processed,
                total_probes,
            )

        _clear_gpu_memory()

    return probes_processed, 0


__all__ = [
    "SingleModelActivations",
    "run_single_model_probe_inference",
    "page_activations_to_disk",
    "run_sequential_probe_inference",
    "run_probe_inference",
    "PagedActivations",
]
