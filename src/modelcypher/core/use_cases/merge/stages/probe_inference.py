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

"""Probe inference loop for activation collection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.use_cases.merge.stages.probe_activation_storage import (
    _accumulate_activation,
    _flush_batch_activations,
)
from modelcypher.core.use_cases.merge.stages.probe_checkpoint import (
    load_probe_activations as _load_probe_activations,
    load_probe_checkpoint as _load_probe_checkpoint,
    save_probe_activations as _save_probe_activations,
    save_probe_checkpoint as _save_probe_checkpoint,
)
from modelcypher.core.use_cases.merge.stages.probe_helpers import _extract_top_k_dims

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.manifold_stitcher import ActivationFingerprint
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.activation_store import ActivationStore
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def run_probe_inference(
    *,
    valid_probes: list[tuple[Any, str]],
    expected_probe_ids: list[str],
    probe_domain_by_id: dict[str, str],
    source_model: Any,
    target_model: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    activation_provider: "ActivationProvider",
    activation_store: "ActivationStore | None",
    backend: "Backend",
    checkpoint_dir: Path | None,
    source_layer_activations: dict[int, "Array"],
    target_layer_activations: dict[int, "Array"],
    source_intermediate_activations: dict[int, "Array"],
    target_intermediate_activations: dict[int, "Array"],
    source_attention_activations: dict[int, "Array"],
    target_attention_activations: dict[int, "Array"],
    source_k_activations: dict[int, "Array"],
    target_k_activations: dict[int, "Array"],
    source_v_activations: dict[int, "Array"],
    target_v_activations: dict[int, "Array"],
    source_embedding_activations: list["Array"] | "Array",
    target_embedding_activations: list["Array"] | "Array",
    source_fingerprints: list["ActivationFingerprint"],
    target_fingerprints: list["ActivationFingerprint"],
    run_source_inference: bool,
    run_target_inference: bool,
    invalid_probe_count: int,
    checkpoint_interval: int,
) -> tuple[int, int, Path | None]:
    """Run batched probing and fill activation buffers."""
    from modelcypher.core.domain.geometry.manifold_stitcher import ActivationFingerprint

    probes_processed = 0
    probes_failed = invalid_probe_count
    checkpoint_path: Path | None = None

    if checkpoint_dir is not None and activation_store is None:
        raise ValueError("Activation store required for probe checkpointing")

    completed_probe_ids: set[str] = set()
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / ".probe_checkpoint.json"
        existing_checkpoint = _load_probe_checkpoint(checkpoint_path)

        if existing_checkpoint is not None:
            completed_probe_ids = set(existing_checkpoint.get("probe_ids", []))
            probes_processed = len(completed_probe_ids)
            logger.info(
                "PROBE: Found checkpoint with %d completed probes, resuming...",
                len(completed_probe_ids),
            )

            loaded_activations = _load_probe_activations(
                activation_store,
                checkpoint_path,
                backend,
            )
            if loaded_activations is not None:
                (
                    loaded_src_hidden,
                    loaded_tgt_hidden,
                    loaded_src_inter,
                    loaded_tgt_inter,
                    loaded_src_attn_q,
                    loaded_tgt_attn_q,
                    loaded_src_attn_k,
                    loaded_tgt_attn_k,
                    loaded_src_attn_v,
                    loaded_tgt_attn_v,
                ) = loaded_activations
                source_layer_activations.update(loaded_src_hidden)
                target_layer_activations.update(loaded_tgt_hidden)
                source_intermediate_activations.update(loaded_src_inter)
                target_intermediate_activations.update(loaded_tgt_inter)
                source_attention_activations.update(loaded_src_attn_q)
                target_attention_activations.update(loaded_tgt_attn_q)
                source_k_activations.update(loaded_src_attn_k)
                target_k_activations.update(loaded_tgt_attn_k)
                source_v_activations.update(loaded_src_attn_v)
                target_v_activations.update(loaded_tgt_attn_v)
            else:
                logger.warning(
                    "PROBE: Activation checkpoint missing, re-running all probes"
                )
                completed_probe_ids = set()
                probes_processed = 0

    has_batch_hidden = hasattr(activation_provider, "collect_hidden_activations_batch")
    has_batch_intermediate = hasattr(
        activation_provider, "collect_intermediate_activations_batch"
    )
    collect_attention = False
    has_batch_attention = (
        collect_attention
        and hasattr(activation_provider, "collect_attention_activations_batch")
    )

    PROBE_BATCH_SIZE = 4
    logger.info(
        "PROBE PRECISE: %d valid probes, processing in batches of %d...",
        len(valid_probes),
        PROBE_BATCH_SIZE,
    )

    for batch_start in range(0, len(valid_probes), PROBE_BATCH_SIZE):
        batch_end = min(batch_start + PROBE_BATCH_SIZE, len(valid_probes))
        batch = valid_probes[batch_start:batch_end]
        batch_texts = [probe_text for _, probe_text in batch]
        batch_size = len(batch_texts)
        total_probes = len(valid_probes)
        empty_batch = [{} for _ in range(batch_size)]

        try:
            if has_batch_hidden:
                source_hidden_batch = (
                    activation_provider.collect_hidden_activations_batch(
                        source_model, source_tokenizer, batch_texts
                    )
                    if run_source_inference
                    else empty_batch
                )
                target_hidden_batch = (
                    activation_provider.collect_hidden_activations_batch(
                        target_model, target_tokenizer, batch_texts
                    )
                    if run_target_inference
                    else empty_batch
                )
            else:
                source_hidden_batch = (
                    [
                        activation_provider.collect_hidden_activations(
                            source_model, source_tokenizer, text
                        )
                        for text in batch_texts
                    ]
                    if run_source_inference
                    else empty_batch
                )
                target_hidden_batch = (
                    [
                        activation_provider.collect_hidden_activations(
                            target_model, target_tokenizer, text
                        )
                        for text in batch_texts
                    ]
                    if run_target_inference
                    else empty_batch
                )

            has_embedding = hasattr(activation_provider, "collect_embedding_activations")
            if has_embedding:
                for text in batch_texts:
                    if run_source_inference:
                        source_emb = activation_provider.collect_embedding_activations(
                            source_model, source_tokenizer, text
                        )
                        if isinstance(source_embedding_activations, list):
                            source_embedding_activations.append(source_emb)
                    if run_target_inference:
                        target_emb = activation_provider.collect_embedding_activations(
                            target_model, target_tokenizer, text
                        )
                        if isinstance(target_embedding_activations, list):
                            target_embedding_activations.append(target_emb)

            if has_batch_intermediate:
                source_intermediate_batch = (
                    activation_provider.collect_intermediate_activations_batch(
                        source_model, source_tokenizer, batch_texts
                    )
                    if run_source_inference
                    else empty_batch
                )
                target_intermediate_batch = (
                    activation_provider.collect_intermediate_activations_batch(
                        target_model, target_tokenizer, batch_texts
                    )
                    if run_target_inference
                    else empty_batch
                )
            else:
                source_intermediate_batch = (
                    [
                        activation_provider.collect_intermediate_activations(
                            source_model, source_tokenizer, text
                        )
                        for text in batch_texts
                    ]
                    if run_source_inference
                    else empty_batch
                )
                target_intermediate_batch = (
                    [
                        activation_provider.collect_intermediate_activations(
                            target_model, target_tokenizer, text
                        )
                        for text in batch_texts
                    ]
                    if run_target_inference
                    else empty_batch
                )

            if collect_attention and has_batch_attention:
                if run_source_inference:
                    source_q_batch, source_k_batch, source_v_batch = (
                        activation_provider.collect_attention_activations_batch(
                            source_model, source_tokenizer, batch_texts
                        )
                    )
                else:
                    source_q_batch, source_k_batch, source_v_batch = (
                        empty_batch,
                        empty_batch,
                        empty_batch,
                    )
                if run_target_inference:
                    target_q_batch, target_k_batch, target_v_batch = (
                        activation_provider.collect_attention_activations_batch(
                            target_model, target_tokenizer, batch_texts
                        )
                    )
                else:
                    target_q_batch, target_k_batch, target_v_batch = (
                        empty_batch,
                        empty_batch,
                        empty_batch,
                    )
            elif collect_attention:
                source_q_batch, source_k_batch, source_v_batch = [], [], []
                target_q_batch, target_k_batch, target_v_batch = [], [], []
                for text in batch_texts:
                    if run_source_inference:
                        src_q, src_k, src_v = activation_provider.collect_attention_activations(
                            source_model, source_tokenizer, text
                        )
                    else:
                        src_q, src_k, src_v = {}, {}, {}
                    if run_target_inference:
                        tgt_q, tgt_k, tgt_v = activation_provider.collect_attention_activations(
                            target_model, target_tokenizer, text
                        )
                    else:
                        tgt_q, tgt_k, tgt_v = {}, {}, {}
                    source_q_batch.append(src_q)
                    source_k_batch.append(src_k)
                    source_v_batch.append(src_v)
                    target_q_batch.append(tgt_q)
                    target_k_batch.append(tgt_k)
                    target_v_batch.append(tgt_v)
            else:
                source_q_batch, source_k_batch, source_v_batch = [], [], []
                target_q_batch, target_k_batch, target_v_batch = [], [], []

            source_hidden_accum: dict[int, list["Array"]] = {}
            source_hidden_indices: dict[int, list[int]] = {}
            target_hidden_accum: dict[int, list["Array"]] = {}
            target_hidden_indices: dict[int, list[int]] = {}
            source_inter_accum: dict[int, list["Array"]] = {}
            source_inter_indices: dict[int, list[int]] = {}
            target_inter_accum: dict[int, list["Array"]] = {}
            target_inter_indices: dict[int, list[int]] = {}
            source_q_accum: dict[int, list["Array"]] = {}
            source_q_indices: dict[int, list[int]] = {}
            target_q_accum: dict[int, list["Array"]] = {}
            target_q_indices: dict[int, list[int]] = {}
            source_k_accum: dict[int, list["Array"]] = {}
            source_k_indices: dict[int, list[int]] = {}
            target_k_accum: dict[int, list["Array"]] = {}
            target_k_indices: dict[int, list[int]] = {}
            source_v_accum: dict[int, list["Array"]] = {}
            source_v_indices: dict[int, list[int]] = {}
            target_v_accum: dict[int, list["Array"]] = {}
            target_v_indices: dict[int, list[int]] = {}

            for i, (probe, _probe_text) in enumerate(batch):
                if probe.probe_id in completed_probe_ids:
                    continue
                probe_index = batch_start + i

                source_acts = source_hidden_batch[i]
                target_acts = target_hidden_batch[i]
                source_intermediate_acts = source_intermediate_batch[i]
                target_intermediate_acts = target_intermediate_batch[i]
                source_attention_acts = source_q_batch[i] if source_q_batch else {}
                source_k_acts = source_k_batch[i] if source_k_batch else {}
                source_v_acts = source_v_batch[i] if source_v_batch else {}
                target_attention_acts = target_q_batch[i] if target_q_batch else {}
                target_k_acts = target_k_batch[i] if target_k_batch else {}
                target_v_acts = target_v_batch[i] if target_v_batch else {}

                source_activated: dict[int, list[Any]] = {}
                target_activated: dict[int, list[Any]] = {}

                if run_source_inference:
                    for layer_idx, act in source_acts.items():
                        source_activated[layer_idx] = _extract_top_k_dims(
                            act, backend=backend
                        )
                        source_hidden_accum.setdefault(layer_idx, []).append(act)
                        source_hidden_indices.setdefault(layer_idx, []).append(probe_index)

                if run_target_inference:
                    for layer_idx, act in target_acts.items():
                        target_activated[layer_idx] = _extract_top_k_dims(
                            act, backend=backend
                        )
                        target_hidden_accum.setdefault(layer_idx, []).append(act)
                        target_hidden_indices.setdefault(layer_idx, []).append(probe_index)

                if run_source_inference:
                    for layer_idx, act in source_intermediate_acts.items():
                        source_inter_accum.setdefault(layer_idx, []).append(act)
                        source_inter_indices.setdefault(layer_idx, []).append(probe_index)

                if run_target_inference:
                    for layer_idx, act in target_intermediate_acts.items():
                        target_inter_accum.setdefault(layer_idx, []).append(act)
                        target_inter_indices.setdefault(layer_idx, []).append(probe_index)

                if run_source_inference:
                    for layer_idx, act in source_attention_acts.items():
                        source_q_accum.setdefault(layer_idx, []).append(act)
                        source_q_indices.setdefault(layer_idx, []).append(probe_index)

                if run_target_inference:
                    for layer_idx, act in target_attention_acts.items():
                        target_q_accum.setdefault(layer_idx, []).append(act)
                        target_q_indices.setdefault(layer_idx, []).append(probe_index)

                if run_source_inference:
                    for layer_idx, act in source_k_acts.items():
                        source_k_accum.setdefault(layer_idx, []).append(act)
                        source_k_indices.setdefault(layer_idx, []).append(probe_index)

                if run_target_inference:
                    for layer_idx, act in target_k_acts.items():
                        target_k_accum.setdefault(layer_idx, []).append(act)
                        target_k_indices.setdefault(layer_idx, []).append(probe_index)

                if run_source_inference:
                    for layer_idx, act in source_v_acts.items():
                        source_v_accum.setdefault(layer_idx, []).append(act)
                        source_v_indices.setdefault(layer_idx, []).append(probe_index)

                if run_target_inference:
                    for layer_idx, act in target_v_acts.items():
                        target_v_accum.setdefault(layer_idx, []).append(act)
                        target_v_indices.setdefault(layer_idx, []).append(probe_index)

                if run_source_inference and source_activated:
                    source_fingerprints.append(
                        ActivationFingerprint(
                            prime_id=probe.probe_id,
                            prime_text=probe.name,
                            activated_dimensions=source_activated,
                        )
                    )
                if run_target_inference and target_activated:
                    target_fingerprints.append(
                        ActivationFingerprint(
                            prime_id=probe.probe_id,
                            prime_text=probe.name,
                            activated_dimensions=target_activated,
                        )
                    )

                completed_probe_ids.add(probe.probe_id)
                probes_processed += 1

            if run_source_inference:
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
                    source_attention_activations,
                    source_q_accum,
                    source_q_indices,
                    backend,
                    total_probes,
                )
                _flush_batch_activations(
                    source_k_activations,
                    source_k_accum,
                    source_k_indices,
                    backend,
                    total_probes,
                )
                _flush_batch_activations(
                    source_v_activations,
                    source_v_accum,
                    source_v_indices,
                    backend,
                    total_probes,
                )

            if run_target_inference:
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
                    target_attention_activations,
                    target_q_accum,
                    target_q_indices,
                    backend,
                    total_probes,
                )
                _flush_batch_activations(
                    target_k_activations,
                    target_k_accum,
                    target_k_indices,
                    backend,
                    total_probes,
                )
                _flush_batch_activations(
                    target_v_activations,
                    target_v_accum,
                    target_v_indices,
                    backend,
                    total_probes,
                )

            if batch_end % 50 <= PROBE_BATCH_SIZE:
                logger.info(
                    "PROBE PRECISE: Processed %d/%d probes...",
                    probes_processed,
                    len(valid_probes),
                )

            try:
                import gc
                import mlx.core as mx

                mx.eval()
                mx.clear_cache()
                gc.collect()
            except Exception:
                pass

            if checkpoint_path is not None and probes_processed % checkpoint_interval < PROBE_BATCH_SIZE:
                completed_probe_ids_list = [
                    pid for pid in expected_probe_ids if pid in completed_probe_ids
                ]
                completed_probe_domains = [
                    probe_domain_by_id[pid] for pid in completed_probe_ids_list
                ]
                _save_probe_checkpoint(
                    checkpoint_path=checkpoint_path,
                    completed_probes=probes_processed,
                    probe_ids=completed_probe_ids_list,
                    probe_domains=completed_probe_domains,
                    total_probes=len(valid_probes),
                )
                _save_probe_activations(
                    activation_store=activation_store,
                    checkpoint_path=checkpoint_path,
                    source_layer_activations=source_layer_activations,
                    target_layer_activations=target_layer_activations,
                    source_intermediate_activations=source_intermediate_activations,
                    target_intermediate_activations=target_intermediate_activations,
                    source_attention_activations=source_attention_activations,
                    target_attention_activations=target_attention_activations,
                    source_k_activations=source_k_activations,
                    target_k_activations=target_k_activations,
                    source_v_activations=source_v_activations,
                    target_v_activations=target_v_activations,
                    backend=backend,
                )

        except Exception as e:
            logger.warning("Batch processing failed, falling back to sequential: %s", e)
            for i, (probe, probe_text) in enumerate(batch):
                if probe.probe_id in completed_probe_ids:
                    continue
                probe_index = batch_start + i

                try:
                    source_acts = (
                        activation_provider.collect_hidden_activations(
                            source_model, source_tokenizer, probe_text
                        )
                        if run_source_inference
                        else {}
                    )
                    target_acts = (
                        activation_provider.collect_hidden_activations(
                            target_model, target_tokenizer, probe_text
                        )
                        if run_target_inference
                        else {}
                    )
                    source_intermediate_acts = (
                        activation_provider.collect_intermediate_activations(
                            source_model, source_tokenizer, probe_text
                        )
                        if run_source_inference
                        else {}
                    )
                    target_intermediate_acts = (
                        activation_provider.collect_intermediate_activations(
                            target_model, target_tokenizer, probe_text
                        )
                        if run_target_inference
                        else {}
                    )
                    if collect_attention and run_source_inference:
                        (
                            source_attention_acts,
                            source_k_acts,
                            source_v_acts,
                        ) = activation_provider.collect_attention_activations(
                            source_model, source_tokenizer, probe_text
                        )
                    else:
                        source_attention_acts, source_k_acts, source_v_acts = {}, {}, {}
                    if collect_attention and run_target_inference:
                        (
                            target_attention_acts,
                            target_k_acts,
                            target_v_acts,
                        ) = activation_provider.collect_attention_activations(
                            target_model, target_tokenizer, probe_text
                        )
                    else:
                        target_attention_acts, target_k_acts, target_v_acts = {}, {}, {}

                    source_activated_fallback: dict[int, list[Any]] = {}
                    target_activated_fallback: dict[int, list[Any]] = {}

                    if run_source_inference:
                        for layer_idx, act in source_acts.items():
                            source_activated_fallback[layer_idx] = _extract_top_k_dims(
                                act, backend=backend
                            )
                            _accumulate_activation(
                                source_layer_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_target_inference:
                        for layer_idx, act in target_acts.items():
                            target_activated_fallback[layer_idx] = _extract_top_k_dims(
                                act, backend=backend
                            )
                            _accumulate_activation(
                                target_layer_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_source_inference:
                        for layer_idx, act in source_intermediate_acts.items():
                            _accumulate_activation(
                                source_intermediate_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_target_inference:
                        for layer_idx, act in target_intermediate_acts.items():
                            _accumulate_activation(
                                target_intermediate_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_source_inference:
                        for layer_idx, act in source_attention_acts.items():
                            _accumulate_activation(
                                source_attention_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_target_inference:
                        for layer_idx, act in target_attention_acts.items():
                            _accumulate_activation(
                                target_attention_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_source_inference:
                        for layer_idx, act in source_k_acts.items():
                            _accumulate_activation(
                                source_k_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_target_inference:
                        for layer_idx, act in target_k_acts.items():
                            _accumulate_activation(
                                target_k_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_source_inference:
                        for layer_idx, act in source_v_acts.items():
                            _accumulate_activation(
                                source_v_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_target_inference:
                        for layer_idx, act in target_v_acts.items():
                            _accumulate_activation(
                                target_v_activations,
                                layer_idx,
                                act,
                                backend,
                                probe_index,
                                total_probes,
                            )

                    if run_source_inference and source_activated_fallback:
                        source_fingerprints.append(
                            ActivationFingerprint(
                                prime_id=probe.probe_id,
                                prime_text=probe.name,
                                activated_dimensions=source_activated_fallback,
                            )
                        )
                    if run_target_inference and target_activated_fallback:
                        target_fingerprints.append(
                            ActivationFingerprint(
                                prime_id=probe.probe_id,
                                prime_text=probe.name,
                                activated_dimensions=target_activated_fallback,
                            )
                        )

                    completed_probe_ids.add(probe.probe_id)
                    probes_processed += 1

                except Exception as inner_e:
                    logger.warning("Probe '%s' failed: %s", probe.probe_id, inner_e)
                    probes_failed += 1

    return probes_processed, probes_failed, checkpoint_path
