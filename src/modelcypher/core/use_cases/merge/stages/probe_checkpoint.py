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

"""Checkpoint utilities for probe collection.

Provides save/load/clear functions for resuming long-running probe collection.
Activations are saved separately from metadata to allow efficient partial resume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.activation_store import ActivationStore
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def save_probe_checkpoint(
    checkpoint_path: Path,
    completed_probes: int,
    probe_ids: list[str],
    probe_domains: list[str],
    total_probes: int,
) -> None:
    """Save probe progress to checkpoint file.

    Saves probe IDs and metadata. Activation data is saved separately
    via save_probe_activations() for correct checkpoint resume.
    """
    checkpoint = {
        "version": 1,
        "completed_probes": completed_probes,
        "total_probes": total_probes,
        "probe_ids": probe_ids,
        "probe_domains": probe_domains,
    }
    # Write atomically using temp file
    temp_path = checkpoint_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(checkpoint, indent=2))
    temp_path.rename(checkpoint_path)
    logger.debug(
        "PROBE: Saved checkpoint at %d/%d probes to %s",
        completed_probes,
        total_probes,
        checkpoint_path,
    )


def load_probe_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load probe checkpoint if it exists and is valid."""
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("version") != 1:
            logger.warning(
                "PROBE: Checkpoint version mismatch, ignoring checkpoint"
            )
            return None
        return checkpoint
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("PROBE: Failed to load checkpoint: %s", e)
        return None


def clear_probe_checkpoint(checkpoint_path: Path) -> None:
    """Remove checkpoint file after successful completion."""
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.debug("PROBE: Cleared checkpoint file %s", checkpoint_path)
    # Also clear activation NPZ file
    activation_path = checkpoint_path.with_suffix(".activations.npz")
    if activation_path.exists():
        activation_path.unlink()
        logger.debug("PROBE: Cleared activation checkpoint %s", activation_path)


def save_probe_activations(
    activation_store: "ActivationStore",
    checkpoint_path: Path,
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
    backend: "Backend",
) -> None:
    """Save all activation dicts to NPZ file for checkpoint resume.

    Without saving activations, resumed probes skip completed work but lose
    the activation matrices needed for alignment.
    """
    activation_path = checkpoint_path.with_suffix(".activations.npz")

    # Build flat dict with prefixed keys for NPZ storage
    arrays_to_save: dict[str, Any] = {}

    # Helper to flatten activation dict
    def flatten_dict(d: dict[int, "Array"], prefix: str) -> None:
        for layer_idx, arr in d.items():
            arrays_to_save[f"{prefix}_{layer_idx}"] = arr

    flatten_dict(source_layer_activations, "src_hidden")
    flatten_dict(target_layer_activations, "tgt_hidden")
    flatten_dict(source_intermediate_activations, "src_inter")
    flatten_dict(target_intermediate_activations, "tgt_inter")
    flatten_dict(source_attention_activations, "src_attn_q")
    flatten_dict(target_attention_activations, "tgt_attn_q")
    flatten_dict(source_k_activations, "src_attn_k")
    flatten_dict(target_k_activations, "tgt_attn_k")
    flatten_dict(source_v_activations, "src_attn_v")
    flatten_dict(target_v_activations, "tgt_attn_v")

    if not arrays_to_save:
        return  # Nothing to save

    activation_store.save_probe_activations(
        activation_path,
        arrays_to_save,
        backend,
    )
    logger.debug(
        "PROBE: Saved %d activation arrays to %s",
        len(arrays_to_save),
        activation_path,
    )


def load_probe_activations(
    activation_store: "ActivationStore",
    checkpoint_path: Path,
    backend: "Backend",
) -> tuple[
    dict[int, "Array"],  # source_layer_activations
    dict[int, "Array"],  # target_layer_activations
    dict[int, "Array"],  # source_intermediate_activations
    dict[int, "Array"],  # target_intermediate_activations
    dict[int, "Array"],  # source_attention_activations
    dict[int, "Array"],  # target_attention_activations
    dict[int, "Array"],  # source_k_activations
    dict[int, "Array"],  # target_k_activations
    dict[int, "Array"],  # source_v_activations
    dict[int, "Array"],  # target_v_activations
] | None:
    """Load activation dicts from NPZ checkpoint file.

    Returns None if no activation checkpoint exists.
    Otherwise returns tuple of 10 activation dicts.
    """
    activation_path = checkpoint_path.with_suffix(".activations.npz")
    if not activation_path.exists():
        return None

    loaded = activation_store.load_probe_activations(activation_path, backend)
    if loaded is None:
        return None

    # Reconstruct activation dicts from flat keys
    source_layer_activations: dict[int, Any] = {}
    target_layer_activations: dict[int, Any] = {}
    source_intermediate_activations: dict[int, Any] = {}
    target_intermediate_activations: dict[int, Any] = {}
    source_attention_activations: dict[int, Any] = {}
    target_attention_activations: dict[int, Any] = {}
    source_k_activations: dict[int, Any] = {}
    target_k_activations: dict[int, Any] = {}
    source_v_activations: dict[int, Any] = {}
    target_v_activations: dict[int, Any] = {}

    for key, arr in loaded.items():
        if key.startswith("src_hidden_"):
            layer_idx = int(key.split("_")[2])
            source_layer_activations[layer_idx] = arr
        elif key.startswith("tgt_hidden_"):
            layer_idx = int(key.split("_")[2])
            target_layer_activations[layer_idx] = arr
        elif key.startswith("src_inter_"):
            layer_idx = int(key.split("_")[2])
            source_intermediate_activations[layer_idx] = arr
        elif key.startswith("tgt_inter_"):
            layer_idx = int(key.split("_")[2])
            target_intermediate_activations[layer_idx] = arr
        elif key.startswith("src_attn_q_"):
            layer_idx = int(key.split("_")[3])
            source_attention_activations[layer_idx] = arr
        elif key.startswith("tgt_attn_q_"):
            layer_idx = int(key.split("_")[3])
            target_attention_activations[layer_idx] = arr
        elif key.startswith("src_attn_k_"):
            layer_idx = int(key.split("_")[3])
            source_k_activations[layer_idx] = arr
        elif key.startswith("tgt_attn_k_"):
            layer_idx = int(key.split("_")[3])
            target_k_activations[layer_idx] = arr
        elif key.startswith("src_attn_v_"):
            layer_idx = int(key.split("_")[3])
            source_v_activations[layer_idx] = arr
        elif key.startswith("tgt_attn_v_"):
            layer_idx = int(key.split("_")[3])
            target_v_activations[layer_idx] = arr

    total_arrays = sum(
        len(d)
        for d in [
            source_layer_activations,
            target_layer_activations,
            source_intermediate_activations,
            target_intermediate_activations,
            source_attention_activations,
            target_attention_activations,
            source_k_activations,
            target_k_activations,
            source_v_activations,
            target_v_activations,
        ]
    )
    logger.info(
        "PROBE: Loaded %d activation arrays from checkpoint",
        total_arrays,
    )

    return (
        source_layer_activations,
        target_layer_activations,
        source_intermediate_activations,
        target_intermediate_activations,
        source_attention_activations,
        target_attention_activations,
        source_k_activations,
        target_k_activations,
        source_v_activations,
        target_v_activations,
    )
