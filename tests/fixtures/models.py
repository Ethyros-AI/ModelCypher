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

"""Test model fixtures for real model testing.

Downloads and caches small models for integration testing.
Uses the production pipeline to stress test actual code paths.

Models are cached in tests/fixtures/.models/ (gitignored).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

# Cache directory for downloaded models
MODELS_CACHE_DIR = Path(__file__).parent / ".models"

# Tiny models for testing (smallest trained models available)
SMOL_LM_135M = "HuggingFaceTB/SmolLM-135M"


def ensure_model(repo_id: str = SMOL_LM_135M) -> Path:
    """Download model if not cached, return path.

    Uses HuggingFace Hub to download the model to a local cache.
    Models are stored in tests/fixtures/.models/ which is gitignored.

    Args:
        repo_id: HuggingFace model repository ID

    Returns:
        Path to the downloaded model directory
    """
    from modelcypher.adapters.hf_hub import HfHubAdapter

    # Create cache directory
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use repo name as local directory name
    model_name = repo_id.replace("/", "--")
    local_dir = MODELS_CACHE_DIR / model_name

    # Check if already cached
    if local_dir.exists() and (local_dir / "config.json").exists():
        logger.debug("Using cached model: %s", local_dir)
        return local_dir

    # Download via HfHubAdapter
    logger.info("Downloading model %s to %s", repo_id, local_dir)
    hub = HfHubAdapter()
    path = hub.fetch(repo_id, local_dir=str(local_dir))

    return Path(path)


def load_model_weights(model_path: Path, backend: "Backend") -> dict[str, "Array"]:
    """Load model weights using the production MLX loader.

    Args:
        model_path: Path to model directory
        backend: Backend for array operations

    Returns:
        Dict of weight name -> Array
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    loader = MLXModelLoader()
    weights = loader.load_weights(str(model_path))

    # Convert to backend arrays
    result = {}
    for name, weight in weights.items():
        arr = backend.array(weight)
        result[name] = arr

    backend.eval(*result.values())
    return result


def load_model_and_tokenizer(model_path: Path):
    """Load model and tokenizer using production loaders.

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (model, tokenizer)
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    loader = MLXModelLoader()
    model, tokenizer = loader.load_model_for_training(str(model_path))
    return model, tokenizer


def collect_real_activations(
    model_path: Path,
    probes: list[str],
    backend: "Backend",
    layer_indices: list[int] | None = None,
) -> dict[int, "Array"]:
    """Collect real activations from a model using probe prompts.

    Uses the production activation collection pipeline.

    Args:
        model_path: Path to model directory
        probes: List of probe prompts to run
        backend: Backend for array operations
        layer_indices: Optional list of layers to collect (default: all)

    Returns:
        Dict mapping layer index -> activation matrix [n_probes, hidden_dim]
    """
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    # Load model and tokenizer
    loader = MLXModelLoader()
    model, tokenizer = loader.load_model_for_training(str(model_path))

    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer for {model_path}")

    # Create activation provider
    provider = MLXActivationProvider()

    # Collect activations for each probe
    activations_by_layer: dict[int, list] = {}

    for probe in probes:
        # Tokenize
        input_ids = tokenizer.encode(probe, add_special_tokens=True)

        # Collect pooled activations per layer
        activations = provider.collect_hidden_activations(
            model, tokenizer, probe, token_ids=input_ids
        )
        for layer_idx, layer_hidden in activations.items():
            if layer_indices is not None and layer_idx not in layer_indices:
                continue

            vec = backend.array(layer_hidden)
            if len(vec.shape) == 1:
                vec = backend.reshape(vec, (1, -1))

            if layer_idx not in activations_by_layer:
                activations_by_layer[layer_idx] = []
            activations_by_layer[layer_idx].append(vec)

    # Stack activations per layer
    result = {}
    for layer_idx, acts_list in activations_by_layer.items():
        stacked = backend.concatenate(acts_list, axis=0)
        backend.eval(stacked)
        result[layer_idx] = stacked

    return result


def get_atlas_probes(n_samples: int = 100, sources: list[str] | None = None) -> list[str]:
    """Get probe prompts from the atlas system.

    Args:
        n_samples: Maximum number of probes to return
        sources: Optional list of atlas sources to filter by

    Returns:
        List of probe prompt strings
    """
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

    all_probes = UnifiedAtlasInventory.all_probes()

    # Filter by sources if specified
    if sources:
        allowed = {s.lower() for s in sources}
        all_probes = [p for p in all_probes if p.source.value.lower() in allowed]

    # Limit to n_samples
    if len(all_probes) > n_samples:
        step = len(all_probes) // n_samples
        all_probes = all_probes[::step][:n_samples]

    # Extract text from probes
    result = []
    for probe in all_probes:
        if probe.support_texts:
            result.append(probe.support_texts[0])
        else:
            result.append(probe.name)

    return result
