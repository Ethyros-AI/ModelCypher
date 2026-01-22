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

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import CrossArchitectureInfo, LayerSemanticProfile
from modelcypher.utils.security import trust_remote_code_enabled, warn_trust_remote_code

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


def load_tokenizer(model_path: str, model_loader: "ModelLoaderPort | None" = None) -> Any | None:
    """Load tokenizer for probe execution."""
    try:
        # Try transformers tokenizer first (avoids loading model)
        from transformers import AutoTokenizer

        warn_trust_remote_code(logger)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code_enabled()
        )
        return tokenizer
    except Exception as exc:
        logger.debug("Tokenizer auto-load failed: %s", exc)

    if model_loader is None:
        return None

    try:
        # Fall back to ModelLoaderPort (hexagonal architecture)
        _, tokenizer = model_loader.load_model_for_training(model_path)
        return tokenizer
    except Exception as exc:
        logger.warning("Failed to load tokenizer: %s", exc)
        return None


def load_model_for_probing(
    model_path: str, model_loader: "ModelLoaderPort | None" = None
) -> Any | None:
    """Load model for precise probe execution."""
    if model_loader is None:
        logger.warning("Model loader required for probe execution")
        return None

    try:
        logger.info("Loading model from %s for activation probing...", model_path)
        model, _ = model_loader.load_model_for_training(model_path)
        logger.info("Model loaded successfully: %s", type(model).__name__)
        return model
    except Exception as exc:
        logger.error("Failed to load model for probing: %s", exc)
        import traceback

        logger.debug("Traceback: %s", traceback.format_exc())
        return None


def load_weights(model_loader: "ModelLoaderPort", model_path: str) -> tuple[dict[str, Any], str]:
    """Load model weights as native backend arrays (GPU-accelerated)."""
    weights = model_loader.load_weights(model_path)
    return weights, "safetensors"





def load_weights_as_arrays(
    model_loader: "ModelLoaderPort",
    model_path: str,
) -> tuple[dict[str, "Array"], str]:
    """Load model weights as backend Arrays."""
    weights = model_loader.load_weights(model_path)
    return weights, "safetensors"


def infer_hidden_dim(weights: dict[str, Any]) -> int:
    """Infer hidden dimension from weight shapes.

    Used to determine if permutation alignment is applicable (same hidden dim).
    """
    # Prefer norm weights: they are 1D, remain unquantized, and directly encode hidden size.
    for key, val in weights.items():
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if not hasattr(val, "shape"):
            continue
        if len(val.shape) != 1:
            continue
        if key.endswith(("norm.weight", "layernorm.weight", "rms_norm.weight")):
            return int(val.shape[0])

    # Fall back to projection matrices (avoid quantization metadata like *.scales).
    for key, val in weights.items():
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if not hasattr(val, "shape") or len(val.shape) != 2:
            continue
        if not key.endswith(".weight"):
            continue
        if "q_proj" in key or "o_proj" in key:
            # Usually square [hidden, hidden]
            return int(max(val.shape))
        if "k_proj" in key or "v_proj" in key:
            # GQA: [kv_dim, hidden] -> hidden is the max dim
            return int(max(val.shape))
        if "up_proj" in key or "gate_proj" in key or "down_proj" in key:
            # MLP: [intermediate, hidden] or [hidden, intermediate] -> hidden is the min dim
            return int(min(val.shape))
    # Return 0 for unknown (will disable permutation)
    return 0


def save_weights(
    output_dir: str,
    weights: dict[str, Any],
    output_format: str,
    backend: "Backend",
) -> None:
    """Save merged weights using backend native I/O (hexagonal architecture)."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / "model.safetensors"

    # Backend handles serialization in the format it natively supports
    backend.save_safetensors(str(output_path), weights)
    logger.info("Saved merged weights to %s", output_path)


def load_transplant_occupancy(model_path: str) -> dict[int, list[float]] | None:
    """Load per-layer occupancy weights from a prior merge, if present."""
    path = Path(model_path)
    if path.is_file():
        path = path.parent
    occ_path = path / "transplant_occupancy.json"
    if not occ_path.exists():
        return None
    try:
        payload = json.loads(occ_path.read_text())
        layers = payload.get("layers", {})
        occupancy: dict[int, list[float]] = {}
        for key, values in layers.items():
            try:
                layer_idx = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(values, list):
                occupancy[layer_idx] = [float(v) for v in values]
        if occupancy:
            logger.info(
                "Loaded transplant occupancy from %s (%d layers)",
                occ_path, len(occupancy)
            )
        return occupancy or None
    except Exception as exc:
        logger.warning("Failed to load transplant occupancy: %s", exc)
        return None


def save_transplant_occupancy(
    output_dir: str,
    occupancy_by_layer: dict[str, list[float]] | dict[int, list[float]],
) -> None:
    """Persist per-layer occupancy weights for future merges."""
    if not occupancy_by_layer:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload_layers: dict[str, list[float]] = {}
    for key, values in occupancy_by_layer.items():
        payload_layers[str(key)] = [float(v) for v in values]
    payload = {
        "_schema": "mc.merge.transplant.occupancy.v1",
        "timestamp": datetime.utcnow().isoformat(),
        "layers": payload_layers,
    }
    occ_path = path / "transplant_occupancy.json"
    occ_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved transplant occupancy to %s", occ_path)


def copy_config_files(source_path: str, output_dir: str) -> None:
    """Copy config files from source to output."""
    source = Path(source_path)
    dest = Path(output_dir)

    for config_file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        src_file = source / config_file
        if src_file.exists():
            shutil.copy(src_file, dest / config_file)


def extract_layer_indices(weights: dict[str, "Array"]) -> list[int]:
    """Extract unique layer indices from weight keys."""
    indices = set()
    for key in weights:
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def create_layer_profile(layer_indices: list[int]) -> LayerSemanticProfile:
    """Create a layer profile for geometric analysis.

    NO HEURISTICS. This creates a profile structure that will be populated
    with MEASURED values (intrinsic dimension, Gram rank) during the probe stage.

    The only structural fact we encode:
    - Layer 0 is the embedding layer (this is architecture, not heuristic)

    The semantic highway is identified by MEASURING:
    - Intrinsic dimension per layer (peaks at semantic layers)
    - Gram rank per layer (drops to 2-3 at bottleneck)

    These measurements happen during probe stage via CKA/alignment computation.

    Args:
        layer_indices: List of layer indices (from extract_layer_indices)

    Returns:
        LayerSemanticProfile with structural info (measurements added later)
    """
    if not layer_indices:
        return LayerSemanticProfile()

    profile = LayerSemanticProfile(
        total_layers=len(layer_indices),
        embedding_layer=0,  # Structural: layer 0 is always embedding
    )

    logger.info(
        "LAYER PROFILE: total=%d, embedding_layer=%d (structural)",
        profile.total_layers,
        profile.embedding_layer,
    )
    logger.info(
        "NOTE: Intrinsic dimension and Gram rank will be measured during probe stage"
    )

    return profile


def extract_layer_index(key: str) -> int | None:
    """Extract layer index from weight key."""
    match = re.search(r"layers\.(\d+)\.", key)
    if match:
        return int(match.group(1))
    return None


def detect_cross_architecture(
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
) -> CrossArchitectureInfo:
    """
    Detect if models have different architectures (layer count or hidden dim).

    Cross-architecture merging requires layer correspondence mapping and
    potentially dimension projection. This method detects the mismatch
    and returns information needed for alignment.
    """
    # Extract layer counts
    source_layers = extract_layer_indices(source_weights)
    target_layers = extract_layer_indices(target_weights)

    layer_mismatch = len(source_layers) != len(target_layers)

    # Check dimension mismatch from representative weight matrices
    source_hidden_dim = 0
    target_hidden_dim = 0

    # Look for q_proj weights as they reflect hidden dimension
    for key in source_weights:
        if ".q_proj.weight" in key or ".self_attn.q_proj.weight" in key:
            source_hidden_dim = source_weights[key].shape[-1]
            break

    for key in target_weights:
        if ".q_proj.weight" in key or ".self_attn.q_proj.weight" in key:
            target_hidden_dim = target_weights[key].shape[-1]
            break

    # Fallback to any 2D weight if q_proj not found
    if source_hidden_dim == 0:
        for key in source_weights:
            w = source_weights[key]
            if w.ndim == 2 and "layers.0." in key:
                source_hidden_dim = w.shape[-1]
                break

    if target_hidden_dim == 0:
        for key in target_weights:
            w = target_weights[key]
            if w.ndim == 2 and "layers.0." in key:
                target_hidden_dim = w.shape[-1]
                break

    dim_mismatch = (
        source_hidden_dim != target_hidden_dim
        and source_hidden_dim > 0
        and target_hidden_dim > 0
    )

    is_cross_arch = layer_mismatch or dim_mismatch

    if is_cross_arch:
        logger.info(
            "Cross-architecture detected: source=%d layers/%d dim, target=%d layers/%d dim",
            len(source_layers),
            source_hidden_dim,
            len(target_layers),
            target_hidden_dim,
        )

    return CrossArchitectureInfo(
        is_cross_architecture=is_cross_arch,
        source_layer_count=len(source_layers),
        target_layer_count=len(target_layers),
        source_hidden_dim=source_hidden_dim,
        target_hidden_dim=target_hidden_dim,
        layer_correspondence=None,  # Computed later if needed
    )
