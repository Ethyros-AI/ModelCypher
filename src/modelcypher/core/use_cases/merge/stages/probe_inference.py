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

"""Probe inference loop for activation collection (strict, batched)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from modelcypher.core.use_cases.merge.stages.probe_activation_storage import (
    _flush_batch_activations,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


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

    probe_batch_size = total_probes
    logger.info(
        "PROBE PRECISE: %d valid probes, processing in a single batch...",
        total_probes,
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
        if batch_end % 50 <= PROBE_BATCH_SIZE:
            logger.info(
                "PROBE PRECISE: Processed %d/%d probes...",
                probes_processed,
                total_probes,
            )

        try:
            import gc
            import mlx.core as mx

            mx.eval()
            mx.clear_cache()
            gc.collect()
        except Exception:
            pass

    return probes_processed, 0
