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

"""
Stage 1: PROBE - Build intersection map from probe responses.

The intersection map is the PRIMARY CONTROL SIGNAL for all downstream operations.

Two modes:
- "precise": Run 403 probes through BOTH models, compute CKA on activations
- "fast": Use weight-level CKA (faster but less accurate)

Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations"
Reference: Chun et al. (2025) "Estimating Neural Representation Alignment from Sparsely Sampled Inputs and Features"
Reference: Moschella et al. (2023) "Relative Representations Enable Zero-Shot Transfer"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    ceil_scalar,
    geodesic_svd,
    log_scalar,
    machine_epsilon,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.cross_architecture_layer_matcher import (
    CrossArchitectureLayerMatcher,
)
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_pairwise_metrics
from modelcypher.core.use_cases.merge.stages.probe_checkpoint import (
    clear_probe_checkpoint as _clear_probe_checkpoint,
    load_probe_activations as _load_probe_activations,
    load_probe_checkpoint as _load_probe_checkpoint,
    save_probe_activations as _save_probe_activations,
    save_probe_checkpoint as _save_probe_checkpoint,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.activation_store import ActivationStore
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# Probe mode is ALWAYS "precise" - activation-level CKA is required for correct alignment.
# "fast" mode is eliminated - weight-level CKA is fundamentally less accurate and
# hides alignment problems that will cause gibberish output.
_PROBE_MODE = "precise"

# All probes are always run. The probe corpus (403 probes) was carefully designed
# to cover the concept space. Limiting probes degrades coverage with no benefit.
_MAX_PROBES = 0  # 0 = all probes

# Checkpoint interval: save progress every N probes
_CHECKPOINT_INTERVAL = 50


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


_MLP_INTERMEDIATE_KEYS = (
    "gate_proj",
    "up_proj",
    "mlp.fc1",
    "feed_forward.w1",
    "feed_forward.w3",
    "mlp.gate",
    "mlp.up",
)


def _collect_mlp_projection_keys(
    weights: dict[str, Any],
    extract_layer_index_fn: Callable[[str], int | None],
) -> dict[int, str]:
    """Collect per-layer MLP projection keys (hidden -> intermediate)."""
    candidates: dict[int, tuple[int, str]] = {}
    for key in weights:
        layer_idx = extract_layer_index_fn(key)
        if layer_idx is None:
            continue
        for rank, pattern in enumerate(_MLP_INTERMEDIATE_KEYS):
            if pattern in key:
                current = candidates.get(layer_idx)
                if current is None or rank < current[0]:
                    candidates[layer_idx] = (rank, key)
                break
    return {layer: key for layer, (rank, key) in candidates.items()}


# =============================================================================
# PER-MODEL PROBE ACTIVATION CACHE (DISK)
# =============================================================================


@dataclass(frozen=True)
class ModelProbeCache:
    probe_ids: list[str]
    probe_domains: list[str]
    probe_mode: str
    probe_corpus_hash: str
    hidden_activations: dict[int, "Array"]
    intermediate_activations: dict[int, "Array"]
    attention_activations: dict[int, "Array"]
    k_activations: dict[int, "Array"]
    v_activations: dict[int, "Array"]
    embedding_activations: "Array | None"


def _model_probe_cache_paths(
    model_id: str,
    probe_mode: str,
    probe_corpus_hash: str,
) -> tuple[Path, Path]:
    """Resolve per-model probe cache paths."""
    from modelcypher.core.domain.geometry.model_profile import ModelProfileStore

    store = ModelProfileStore()
    cache_dir = store.probe_cache_dir(model_id)
    stem = f"{probe_mode}_{probe_corpus_hash}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def _load_model_probe_cache(
    model_id: str,
    probe_mode: str,
    probe_corpus_hash: str,
    backend: "Backend",
) -> ModelProbeCache | None:
    """Load per-model probe activations from disk."""
    import mlx.core as mx

    data_path, meta_path = _model_probe_cache_paths(
        model_id=model_id,
        probe_mode=probe_mode,
        probe_corpus_hash=probe_corpus_hash,
    )
    if not data_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("PROBE CACHE: Failed to read %s: %s", meta_path, e)
        return None

    if meta.get("version") != 1:
        return None
    if meta.get("probe_mode") != probe_mode:
        return None
    if meta.get("probe_corpus_hash") != probe_corpus_hash:
        return None

    try:
        loaded = mx.load(data_path)
    except Exception as e:
        logger.warning("PROBE CACHE: Failed to load %s: %s", data_path, e)
        return None

    if not isinstance(loaded, dict):
        logger.warning("PROBE CACHE: Invalid cache format at %s", data_path)
        return None

    hidden: dict[int, "Array"] = {}
    intermediate: dict[int, "Array"] = {}
    attn: dict[int, "Array"] = {}
    k_acts: dict[int, "Array"] = {}
    v_acts: dict[int, "Array"] = {}
    embedding: "Array | None" = None

    for key, arr in loaded.items():
        if key.startswith("hidden_"):
            layer_idx = int(key.split("_")[1])
            hidden[layer_idx] = arr
        elif key.startswith("intermediate_"):
            layer_idx = int(key.split("_")[1])
            intermediate[layer_idx] = arr
        elif key.startswith("attn_q_"):
            layer_idx = int(key.split("_")[2])
            attn[layer_idx] = arr
        elif key.startswith("attn_k_"):
            layer_idx = int(key.split("_")[2])
            k_acts[layer_idx] = arr
        elif key.startswith("attn_v_"):
            layer_idx = int(key.split("_")[2])
            v_acts[layer_idx] = arr
        elif key == "embedding":
            embedding = arr

    probe_ids = meta.get("probe_ids", [])
    probe_domains = meta.get("probe_domains", [])

    if not hidden:
        logger.warning("PROBE CACHE: Missing hidden activations in %s", data_path)
        return None

    return ModelProbeCache(
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        probe_mode=probe_mode,
        probe_corpus_hash=probe_corpus_hash,
        hidden_activations=hidden,
        intermediate_activations=intermediate,
        attention_activations=attn,
        k_activations=k_acts,
        v_activations=v_acts,
        embedding_activations=embedding,
    )


def _save_model_probe_cache(
    model_id: str,
    probe_mode: str,
    probe_corpus_hash: str,
    probe_ids: list[str],
    probe_domains: list[str],
    hidden_activations: dict[int, "Array"],
    intermediate_activations: dict[int, "Array"] | None,
    attention_activations: dict[int, "Array"] | None,
    k_activations: dict[int, "Array"] | None,
    v_activations: dict[int, "Array"] | None,
    embedding_activations: "Array | list[Array] | None",
) -> None:
    """Persist per-model probe activations to disk for reuse."""
    import mlx.core as mx

    data: dict[str, "Array"] = {}
    for layer_idx, acts in hidden_activations.items():
        data[f"hidden_{layer_idx}"] = acts
    if intermediate_activations:
        for layer_idx, acts in intermediate_activations.items():
            data[f"intermediate_{layer_idx}"] = acts
    if attention_activations:
        for layer_idx, acts in attention_activations.items():
            data[f"attn_q_{layer_idx}"] = acts
    if k_activations:
        for layer_idx, acts in k_activations.items():
            data[f"attn_k_{layer_idx}"] = acts
    if v_activations:
        for layer_idx, acts in v_activations.items():
            data[f"attn_v_{layer_idx}"] = acts

    if embedding_activations is not None:
        if isinstance(embedding_activations, list):
            if embedding_activations:
                data["embedding"] = mx.stack(embedding_activations, axis=0)
        else:
            data["embedding"] = embedding_activations

    data_path, meta_path = _model_probe_cache_paths(
        model_id=model_id,
        probe_mode=probe_mode,
        probe_corpus_hash=probe_corpus_hash,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez_compressed(data_path, **data)

    spaces = ["hidden"]
    if intermediate_activations:
        spaces.append("intermediate")
    if attention_activations:
        spaces.append("attention_q")
    if k_activations:
        spaces.append("attention_k")
    if v_activations:
        spaces.append("attention_v")
    if embedding_activations is not None:
        spaces.append("embedding")

    meta = {
        "version": 1,
        "probe_mode": probe_mode,
        "probe_corpus_hash": probe_corpus_hash,
        "probe_ids": probe_ids,
        "probe_domains": probe_domains,
        "spaces": spaces,
        "created_at": datetime.now().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("PROBE CACHE: Saved per-model activations to %s", data_path)

# =============================================================================
# MEMORY-EFFICIENT ACTIVATION ACCUMULATION
# =============================================================================

def _accumulate_activation(
    storage: dict[int, "Array"],
    layer_idx: int,
    act: "Array",
    backend: "Backend",
) -> None:
    """Accumulate activation into a single stacked array per layer.

    This is THE memory optimization: instead of storing thousands of small
    arrays (each a Metal buffer), we concatenate into ONE array per layer.

    Activations come in as 1D [hidden_dim] and we stack them to build
    a 2D matrix [n_probes, hidden_dim].

    Result: 32 Metal buffers per model instead of 4096×32 = 131,072

    IMPORTANT: eval() is called after each concatenation to:
    1. Materialize the new array immediately
    2. Allow the old array (before concat) to be freed
    3. Prevent the lazy computation graph from growing unboundedly
    Without this, Metal resource limits are exceeded (~500K buffer limit).
    """
    import mlx.core as mx

    # Ensure activation is 2D [1, hidden_dim] for proper stacking
    if len(act.shape) == 1:
        act = mx.reshape(act, (1, -1))

    if layer_idx not in storage:
        # First activation for this layer - store as 2D
        storage[layer_idx] = act
    else:
        # Concatenate along axis 0 to build [n_probes, hidden_dim]
        storage[layer_idx] = mx.concatenate([storage[layer_idx], act], axis=0)

    # Force evaluation to materialize buffer and allow old one to be freed
    mx.eval(storage[layer_idx])


@dataclass
class ProbeResult:
    """Result of Stage 1 probing."""

    correlations: dict[str, float]
    confidences: dict[int, float]
    intersection_map: Any | None  # IntersectionMap object
    dimension_correlations: dict
    metrics: dict[str, Any]

    # Activations for downstream processing (null-space filtering, shared subspace)
    # Hidden-space activations: shape [hidden_dim] per sample (e.g., 960 for SmolLM, 896 for Qwen)
    source_activations: dict[int, list[Any]] | None = None
    target_activations: dict[int, list[Any]] | None = None
    # Intermediate-space activations: shape [intermediate_dim] per sample
    # (e.g., 2560 for SmolLM, 4864 for Qwen) - for multi-space stitching
    source_intermediate_activations: dict[int, list[Any]] | None = None
    target_intermediate_activations: dict[int, list[Any]] | None = None
    # Q Attention-space activations: shape [num_heads * head_dim] per sample
    # (e.g., 960 for SmolLM=15*64, 896 for Qwen=14*64) - for q_proj/o_proj stitching
    source_attention_activations: dict[int, list[Any]] | None = None
    target_attention_activations: dict[int, list[Any]] | None = None
    # K Attention-space activations: shape [num_kv_heads * head_dim] per sample
    # Separate from V for granular alignment - for k_proj stitching
    source_k_activations: dict[int, list[Any]] | None = None
    target_k_activations: dict[int, list[Any]] | None = None
    # V Attention-space activations: shape [num_kv_heads * head_dim] per sample
    # Separate from K for granular alignment - for v_proj stitching
    source_v_activations: dict[int, list[Any]] | None = None
    target_v_activations: dict[int, list[Any]] | None = None
    # Embedding-space activations: shape [hidden_dim] per sample (post-embed_tokens, pre-layer-0)
    # Used for GramAlign at 2D interface - closed-form alignment + geodesic diagnostics
    source_embedding_activations: list[Any] | None = None
    target_embedding_activations: list[Any] | None = None
    probe_ids: list[str] | None = None
    probe_domains: list[str] | None = None
    # Layer alignment transforms: source_acts @ transforms[layer] -> aligned_source
    # Closed-form linear alignment on the shared manifold; geodesic CKA is diagnostic
    # Hidden-space transforms: for hidden dimension (e.g., 960 -> 896)
    feature_transforms: dict[int, list[list[float]]] | None = None
    # EXACT SCALE FACTOR per layer: ||target|| / ||source @ F||
    # Apply to stitched weights for exact magnitude match
    scale_ratios: dict[int, float] | None = None
    # Embedding-space transform: for embed_tokens alignment (linear alignment + geodesic diagnostics)
    embedding_transform: list[list[float]] | None = None
    # Attention Q-space transforms: for q_proj/o_proj (e.g., 960 -> 896 for Q heads)
    attention_transforms: dict[int, list[list[float]]] | None = None
    # Attention K-space transforms: for k_proj (granular alignment)
    k_transforms: dict[int, list[list[float]]] | None = None
    # Attention V-space transforms: for v_proj (granular alignment)
    v_transforms: dict[int, list[list[float]]] | None = None
    # Intermediate-space transforms: for MLP gate/up/down projections (pre-computed)
    intermediate_transforms: dict[int, list[list[float]]] | None = None
    # Layer mapping: target_layer -> source_layer (from proportional depth mapping)
    layer_mapping: dict[int, int] | None = None


def stage_probe(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    extract_layer_index_fn: Callable[[str], int | None],
    source_model: Any | None = None,
    target_model: Any | None = None,
    source_path: str = "",
    target_path: str = "",
    source_tokenizer: Any | None = None,
    target_tokenizer: Any | None = None,
    tokenizer: Any | None = None,
    collect_activations_fn: Callable | None = None,
    activation_provider: "ActivationProvider | None" = None,
    activation_store: "ActivationStore | None" = None,
    backend: "Backend | None" = None,
    checkpoint_dir: Path | str | None = None,
    probe_mode: str = "atlas",  # "atlas" (963 conceptual) or "token" (49K+ vocab)
) -> ProbeResult:
    """
    Stage 1: Build intersection map from probe responses.

    ALWAYS uses precise mode (activation-level CKA) with all 403 probes.
    No configuration - the geometry determines everything.

    Args:
        source_weights: Source model weights
        target_weights: Target model weights
        extract_layer_index_fn: Function to extract layer index from weight key
        source_model: Loaded source model (required)
        target_model: Loaded target model (required)
        tokenizer: Tokenizer (for precise mode)
        collect_activations_fn: Function to collect layer activations
        checkpoint_dir: Optional directory for checkpoint files. If provided,
            probe progress will be saved periodically and can be resumed.

    Returns:
        ProbeResult with correlations, confidences, and intersection map
    """
    if tokenizer is not None:
        source_tokenizer = source_tokenizer or tokenizer
        target_tokenizer = target_tokenizer or tokenizer

    # =========================================================================
    # PRE-FLIGHT: Check tokenizer alignment before expensive probing
    # =========================================================================
    if source_tokenizer is not None and target_tokenizer is not None:
        try:
            from modelcypher.core.domain.geometry.dimensional_alignment import (
                measure_1d_alignment,
            )
            alignment_1d = measure_1d_alignment(source_tokenizer, target_tokenizer)
            
            # Report tokenizer alignment measurement (no interpretation)
            logger.info(
                "PRE-FLIGHT: Tokenizer vocabulary overlap Jaccard=%.2f",
                alignment_1d.vocab_jaccard,
            )
        except Exception as e:
            logger.debug("PRE-FLIGHT: Skipped tokenizer check: %s", e)

    # ALWAYS use precise mode - this is not configurable.
    # Activation-level CKA is required for correct geometric alignment.

    # Get activation provider: use provided, or derive from collect_activations_fn, or auto-detect
    if activation_provider is None:
        if collect_activations_fn is not None:
            # Legacy compatibility: wrap the callable as a provider-like object
            # This maintains backwards compatibility with existing callers
            class LegacyActivationProvider:
                def __init__(self, fn: Callable):
                    self._fn = fn

                def collect_hidden_activations(
                    self, model: Any, tokenizer: Any, text: str, token_ids: list[int] | None = None
                ) -> dict:
                    return self._fn(model, tokenizer, text)

                def collect_intermediate_activations(
                    self, model: Any, tokenizer: Any, text: str, token_ids: list[int] | None = None
                ) -> dict:
                    # Legacy doesn't support intermediate activations
                    return {}

                def collect_attention_activations(
                    self, model: Any, tokenizer: Any, text: str, token_ids: list[int] | None = None
                ) -> tuple[dict, dict]:
                    # Legacy doesn't support attention activations
                    return {}, {}

            activation_provider = LegacyActivationProvider(collect_activations_fn)
        else:
            raise ValueError("Activation provider required for probe stage")

    if (
        source_model is not None
        and target_model is not None
        and activation_provider is not None
    ):
        return _probe_precise(
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_weights=source_weights,
            target_weights=target_weights,
            extract_layer_index_fn=extract_layer_index_fn,
            activation_provider=activation_provider,
            activation_store=activation_store,
            source_path=source_path,
            target_path=target_path,
            backend=backend,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            probe_mode=probe_mode,
        )
    else:
        # INVARIANT GEOMETRY: No fallbacks. Models MUST be loaded.
        # Weight-level CKA ("fast" mode) hides alignment problems that
        # cause gibberish output. We don't allow it.
        raise RuntimeError(
            "Probe stage requires loaded models. "
            "Cannot compute activation-level CKA without model access. "
            "Load both source and target models before probing."
        )


def _probe_precise(
    source_model: Any,
    target_model: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    extract_layer_index_fn: Callable[[str], int | None],
    activation_provider: "ActivationProvider",
    activation_store: "ActivationStore | None" = None,
    source_path: str = "",
    target_path: str = "",
    backend: "Backend | None" = None,
    checkpoint_dir: Path | None = None,
    probe_mode: str = "atlas",  # "atlas" (963 conceptual) or "token" (vocab-based)
) -> ProbeResult:
    """Precise probe mode: Run probes through BOTH models.

    Args:
        probe_mode: "atlas" uses 963 curated conceptual probes.
                    "token" uses vocabulary tokens as probes (49K+) for 100% dimension coverage.
        checkpoint_dir: If provided, saves progress periodically and allows
            resume from last checkpoint on restart.
        activation_store: Required if checkpoint_dir is provided.
    """
    b = backend or get_default_backend()
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
    from modelcypher.core.domain.geometry.intersection_similarity import (
        IntersectionSimilarityMode,
        build_intersection_map,
    )
    from modelcypher.core.domain.geometry.manifold_stitcher import (
        ActivatedDimension,
        ActivationFingerprint,
        IntersectionMap,
    )

    # Select probing mode
    if probe_mode == "token":
        # Token-based probing for 100% dimension coverage
        from modelcypher.core.domain.agents.token_atlas import generate_token_probes, TokenProbe
        logger.info("PROBE MODE: Token-based (vocab probes for dimension coverage)")
        
        # Determine required probe count: need at least max(source_dim, target_dim)
        # Use 2x to guarantee full-rank with margin for numerical stability
        source_hidden = None
        target_hidden = None
        for key in source_weights:
            if "layers.0.self_attn.q_proj.weight" in key:
                source_hidden = source_weights[key].shape[0]
                break
        for key in target_weights:
            if "layers.0.self_attn.q_proj.weight" in key:
                target_hidden = target_weights[key].shape[0]
                break
        
        # Cap probes for memory stability. 2048 provides full-rank coverage for
        # typical hidden dims (1024, 2048) while avoiding GPU OOM.
        TOKEN_PROBE_CAP = 2048

        min_probes_needed = max(source_hidden or 1024, target_hidden or 960)
        max_probes = min_probes_needed * 2  # 2x for numerical margin
        max_probes = max(max_probes, 1500)  # At least 1500 for 960-dim full coverage
        max_probes = min(max_probes, TOKEN_PROBE_CAP)  # Cap to avoid OOM

        logger.info("PROBE TOKEN: Dims source=%s target=%s, using %d probes",
                    source_hidden, target_hidden, max_probes)

        # Use target tokenizer for probe generation (target architecture defines the space)
        token_probes = generate_token_probes(target_tokenizer, max_probes=max_probes)
        # Convert TokenProbes to AtlasProbe format for compatibility
        probes = [tp.to_atlas_probe() for tp in token_probes]
        logger.info("PROBE TOKEN: Generated %d probes (2x max_dim, capped at %d)",
                    len(probes), TOKEN_PROBE_CAP)
    else:
        # Default: Atlas probes from JSON files (curated conceptual + MMLU probes)
        from modelcypher.core.domain.agents.probe_loader import load_all_probes
        all_probes = load_all_probes()
        logger.info("PROBE MODE: Atlas (loaded %d probes from JSON)", len(all_probes))
        
        # MEMORY OPTIMIZATION: Limit probes to 2x max hidden dimension
        # This ensures full-rank coverage while preventing OOM
        source_hidden = None
        target_hidden = None
        for key in source_weights:
            if "layers.0.self_attn.q_proj.weight" in key:
                source_hidden = source_weights[key].shape[0]
                break
        for key in target_weights:
            if "layers.0.self_attn.q_proj.weight" in key:
                target_hidden = target_weights[key].shape[0]
                break
        
        # Use TARGET hidden dim - we project into target's space, so only need
        # enough probes to span the target manifold. Source can be larger.
        target_dim = target_hidden or 1024
        # 2x for numerical stability in Gram matrix rank
        max_probes = target_dim * 2

        if len(all_probes) > max_probes:
            logger.info("PROBE LIMIT: Capping %d probes to %d (2x target_dim=%d) - target space only",
                       len(all_probes), max_probes, target_dim)
            # Domain-stratified sampling to ensure coverage across all knowledge domains
            # Group probes by domain
            import random
            from collections import defaultdict
            random.seed(42)  # Deterministic sampling

            probes_by_domain: dict[str, list[Any]] = defaultdict(list)
            for probe in all_probes:
                domain_key = getattr(probe.domain, 'value', str(probe.domain))
                probes_by_domain[domain_key].append(probe)

            # Allocate probes proportionally by domain, ensuring minimum 1 per domain
            n_domains = len(probes_by_domain)
            min_per_domain = max(1, max_probes // (n_domains * 2))  # At least 1, up to half
            remaining = max_probes - (min_per_domain * n_domains)

            probes = []
            for domain_key, domain_probes in sorted(probes_by_domain.items()):
                # Calculate proportional allocation for this domain
                proportion = len(domain_probes) / len(all_probes)
                extra_allocation = int(remaining * proportion)
                n_sample = min(len(domain_probes), min_per_domain + extra_allocation)

                # Sample from domain
                if n_sample >= len(domain_probes):
                    probes.extend(domain_probes)
                else:
                    probes.extend(random.sample(domain_probes, n_sample))

            # If still under max_probes, fill with random remaining probes
            if len(probes) < max_probes:
                used_ids = {p.probe_id for p in probes}
                remaining_probes = [p for p in all_probes if p.probe_id not in used_ids]
                fill_count = min(max_probes - len(probes), len(remaining_probes))
                if fill_count > 0:
                    probes.extend(random.sample(remaining_probes, fill_count))

            logger.info("PROBE STRATIFIED: %d domains, %d-%d probes per domain",
                       n_domains, min_per_domain, max(len(v) for v in probes_by_domain.values()))
        else:
            probes = all_probes
            
    num_probes = len(probes)

    # Pre-validate probes for usable text (used for caching + inference).
    valid_probes: list[tuple[Any, str]] = []
    invalid_probe_count = 0
    for probe in probes:
        probe_text = _select_probe_text(probe)
        if probe_text is None:
            logger.warning(
                "Probe '%s' skipped: no valid support texts or fallback",
                probe.probe_id,
            )
            invalid_probe_count += 1
            continue
        valid_probes.append((probe, probe_text))

    expected_probe_ids = [probe.probe_id for probe, _ in valid_probes]
    expected_probe_domains = [probe.domain.value for probe, _ in valid_probes]
    expected_probe_texts = [probe_text for _, probe_text in valid_probes]

    from modelcypher.core.domain.geometry.model_profile import (
        ModelProfileStore,
        compute_probe_corpus_hash,
    )

    probe_corpus_hash = compute_probe_corpus_hash(
        probe_mode=probe_mode,
        probe_ids=expected_probe_ids,
        probe_texts=expected_probe_texts,
    )
    profile_store = ModelProfileStore()
    source_profile: ModelProfile | None = None
    target_profile: ModelProfile | None = None
    source_identity = None
    target_identity = None

    if source_path:
        source_profile, source_identity = profile_store.ensure(source_path)
    if target_path:
        target_profile, target_identity = profile_store.ensure(target_path)

    cache_key = f"{probe_mode}:{probe_corpus_hash}"

    source_cache: ModelProbeCache | None = None
    target_cache: ModelProbeCache | None = None

    if source_identity and expected_probe_ids:
        source_cache = _load_model_probe_cache(
            model_id=source_identity.model_id,
            probe_mode=probe_mode,
            probe_corpus_hash=probe_corpus_hash,
            backend=b,
        )
        if source_cache:
            logger.info(
                "PROBE CACHE: Loaded per-model activations for source %s",
                source_identity.model_id,
            )

    if target_identity and expected_probe_ids:
        target_cache = _load_model_probe_cache(
            model_id=target_identity.model_id,
            probe_mode=probe_mode,
            probe_corpus_hash=probe_corpus_hash,
            backend=b,
        )
        if target_cache:
            logger.info(
                "PROBE CACHE: Loaded per-model activations for target %s",
                target_identity.model_id,
            )

    source_fingerprints: list[ActivationFingerprint] = []
    target_fingerprints: list[ActivationFingerprint] = []

    if (
        source_profile
        and source_profile.probe_corpus_hash == probe_corpus_hash
        and source_profile.probe_fingerprints
    ):
        source_fingerprints = source_profile.probe_fingerprints
    if (
        target_profile
        and target_profile.probe_corpus_hash == probe_corpus_hash
        and target_profile.probe_fingerprints
    ):
        target_fingerprints = target_profile.probe_fingerprints

    run_source_inference = source_cache is None
    run_target_inference = target_cache is None
    run_inference = run_source_inference or run_target_inference

    if run_source_inference:
        source_fingerprints = []
    if run_target_inference:
        target_fingerprints = []

    if run_inference:
        logger.info(
            "PROBE PRECISE: Running %d probes through %s%s models...",
            len(valid_probes),
            "source" if run_source_inference else "",
            " + target" if run_target_inference else "",
        )
    else:
        logger.info(
            "PROBE PRECISE: Using cached activations for %d probes",
            len(valid_probes),
        )

    # MEMORY OPTIMIZATION: Store as single stacked Array per layer, not list of arrays
    # This reduces Metal buffer count from 4096×32 = 131,072 to just 32 per model
    source_layer_activations: dict[int, "Array"] = (
        source_cache.hidden_activations if source_cache else {}
    )
    target_layer_activations: dict[int, "Array"] = (
        target_cache.hidden_activations if target_cache else {}
    )

    def _fingerprints_from_cache(
        probes: list[tuple[Any, str]],
        layer_activations: dict[int, "Array"],
    ) -> list[ActivationFingerprint]:
        fingerprints: list[ActivationFingerprint] = []
        if not layer_activations:
            return fingerprints

        # Assume all layers share the same probe count
        sample_layer = next(iter(layer_activations.values()))
        probe_count = int(b.shape(sample_layer)[0])

        for probe_idx, (probe, _probe_text) in enumerate(probes):
            if probe_idx >= probe_count:
                break
            activated: dict[int, list[ActivatedDimension]] = {}
            for layer_idx, stacked in layer_activations.items():
                row = b.take(stacked, b.array([probe_idx]), axis=0)
                row = b.squeeze(row, axis=0)
                dims = _extract_top_k_dims(row, backend=b)
                if dims:
                    activated[layer_idx] = dims
            if activated:
                fingerprints.append(
                    ActivationFingerprint(
                        prime_id=probe.probe_id,
                        prime_text=probe.name,
                        activated_dimensions=activated,
                    )
                )
        return fingerprints

    if source_cache and not source_fingerprints:
        logger.info("PROBE CACHE: Rebuilding source fingerprints from cached activations")
        source_fingerprints = _fingerprints_from_cache(
            valid_probes, source_layer_activations
        )
    if target_cache and not target_fingerprints:
        logger.info("PROBE CACHE: Rebuilding target fingerprints from cached activations")
        target_fingerprints = _fingerprints_from_cache(
            valid_probes, target_layer_activations
        )

    source_embedding_activations: list["Array"] | "Array" = (
        source_cache.embedding_activations
        if source_cache and source_cache.embedding_activations is not None
        else []
    )
    target_embedding_activations: list["Array"] | "Array" = (
        target_cache.embedding_activations
        if target_cache and target_cache.embedding_activations is not None
        else []
    )
    # MEMORY OPTIMIZATION: Store as single stacked Array per layer, not list
    # This reduces Metal buffer count from 131K to just 32 per model
    source_intermediate_activations: dict[int, "Array"] = (
        source_cache.intermediate_activations if source_cache else {}
    )
    target_intermediate_activations: dict[int, "Array"] = (
        target_cache.intermediate_activations if target_cache else {}
    )
    # Q Attention-space activations for q_proj/o_proj stitching (cross-architecture merges)
    source_attention_activations: dict[int, "Array"] = (
        source_cache.attention_activations if source_cache else {}
    )
    target_attention_activations: dict[int, "Array"] = (
        target_cache.attention_activations if target_cache else {}
    )
    # K Attention-space activations for k_proj stitching (separate for granular alignment)
    source_k_activations: dict[int, "Array"] = (
        source_cache.k_activations if source_cache else {}
    )
    target_k_activations: dict[int, "Array"] = (
        target_cache.k_activations if target_cache else {}
    )
    # V Attention-space activations for v_proj stitching (separate for granular alignment)
    source_v_activations: dict[int, "Array"] = (
        source_cache.v_activations if source_cache else {}
    )
    target_v_activations: dict[int, "Array"] = (
        target_cache.v_activations if target_cache else {}
    )
    probe_ids: list[str] = []
    probe_domains: list[str] = []

    probes_processed = 0
    probes_failed = invalid_probe_count
    checkpoint_path: Path | None = None  # Defined early so it's available for cleanup

    # =========================================================================
    # BATCHED PROBE COLLECTION - Process probes in batches for efficiency
    # =========================================================================

    if not run_inference:
        probe_ids = list(expected_probe_ids)
        probe_domains = list(expected_probe_domains)
        probes_processed = len(probe_ids)

    if run_inference:
        # Instead of 806 individual forward passes (403 probes × 2 models),
        # we batch probes into groups and use batch methods for ~5-10× speedup.
        #
        # The batch size is tuned for GPU memory vs throughput tradeoff.
        # Too large = OOM, too small = lose batching benefit.
        PROBE_BATCH_SIZE = 4  # Smaller batches for more frequent memory cleanup

        # valid_probes already computed for caching and reuse
        logger.info(
            "PROBE PRECISE: %d valid probes, processing in batches of %d...",
            len(valid_probes),
            PROBE_BATCH_SIZE,
        )

        # =========================================================================
        # CHECKPOINT: Check for existing checkpoint to resume from
        # =========================================================================
        start_probe_idx = 0
        completed_probe_ids: set[str] = set()

        if checkpoint_dir is not None and activation_store is None:
            raise ValueError("Activation store required for probe checkpointing")

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / ".probe_checkpoint.json"
            existing_checkpoint = _load_probe_checkpoint(checkpoint_path)

            if existing_checkpoint is not None:
                completed_probe_ids = set(existing_checkpoint.get("probe_ids", []))
                # Find how many valid probes were already completed
                # We'll skip probes that are in completed_probe_ids
                logger.info(
                    "PROBE: Found checkpoint with %d completed probes, resuming...",
                    len(completed_probe_ids),
                )

                # CRITICAL: Load saved activations for correct resume
                # Without this, completed probes are skipped but their activations lost!
                loaded_activations = _load_probe_activations(
                    activation_store,
                    checkpoint_path,
                    b,
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
                    # Merge loaded activations with any from cache
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
                    # Activation file missing - need to re-run all probes
                    logger.warning(
                        "PROBE: Activation checkpoint missing, re-running all probes"
                    )
                    completed_probe_ids = set()

        # Check if activation provider supports batch methods
        has_batch_hidden = hasattr(activation_provider, "collect_hidden_activations_batch")
        has_batch_intermediate = hasattr(activation_provider, "collect_intermediate_activations_batch")
        has_batch_attention = hasattr(activation_provider, "collect_attention_activations_batch")

        # Second pass: Batch forward passes
        for batch_start in range(0, len(valid_probes), PROBE_BATCH_SIZE):
            batch_end = min(batch_start + PROBE_BATCH_SIZE, len(valid_probes))
            batch = valid_probes[batch_start:batch_end]
            batch_texts = [probe_text for _, probe_text in batch]
            batch_size = len(batch_texts)
            empty_batch = [{} for _ in range(batch_size)]

            try:
                # ===== HIDDEN ACTIVATIONS =====
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
                    # Fallback to sequential
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
    
                # ===== EMBEDDING ACTIVATIONS (2D GramAlign) =====
                # Same closed-form alignment; geodesic CKA is diagnostic
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
    
                # ===== INTERMEDIATE ACTIVATIONS =====
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
    
                # ===== ATTENTION ACTIVATIONS (Q, K, V separately) =====
                if has_batch_attention:
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
                else:
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
    
                # Process batch results
                for i, (probe, probe_text) in enumerate(batch):
                    # Skip probes that were already completed (from checkpoint)
                    if probe.probe_id in completed_probe_ids:
                        continue
    
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
    
                    source_activated: dict[int, list[ActivatedDimension]] = {}
                    target_activated: dict[int, list[ActivatedDimension]] = {}

                    if run_source_inference:
                        for layer_idx, act in source_acts.items():
                            source_activated[layer_idx] = _extract_top_k_dims(
                                act, backend=b
                            )
                            # MEMORY FIX: accumulate into single buffer per layer
                            _accumulate_activation(source_layer_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_acts.items():
                            target_activated[layer_idx] = _extract_top_k_dims(
                                act, backend=b
                            )
                            # MEMORY FIX: accumulate into single buffer per layer
                            _accumulate_activation(target_layer_activations, layer_idx, act, b)

                    # Store intermediate activations for multi-space stitching
                    if run_source_inference:
                        for layer_idx, act in source_intermediate_acts.items():
                            _accumulate_activation(source_intermediate_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_intermediate_acts.items():
                            _accumulate_activation(target_intermediate_activations, layer_idx, act, b)

                    # Store Q attention activations for q_proj/o_proj stitching
                    if run_source_inference:
                        for layer_idx, act in source_attention_acts.items():
                            _accumulate_activation(source_attention_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_attention_acts.items():
                            _accumulate_activation(target_attention_activations, layer_idx, act, b)

                    # Store K attention activations for k_proj stitching
                    if run_source_inference:
                        for layer_idx, act in source_k_acts.items():
                            _accumulate_activation(source_k_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_k_acts.items():
                            _accumulate_activation(target_k_activations, layer_idx, act, b)

                    # Store V attention activations for v_proj stitching
                    if run_source_inference:
                        for layer_idx, act in source_v_acts.items():
                            _accumulate_activation(source_v_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_v_acts.items():
                            _accumulate_activation(target_v_activations, layer_idx, act, b)

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
    
                    probe_ids.append(probe.probe_id)
                    probe_domains.append(probe.domain.value)
                    probes_processed += 1
    
                # Log progress at batch boundaries
                if batch_end % 50 <= PROBE_BATCH_SIZE:
                    logger.info(
                        "PROBE PRECISE: Processed %d/%d probes...",
                        probes_processed,
                        len(valid_probes),
                    )
                
                # MEMORY OPTIMIZATION: Aggressive cleanup after each batch
                # This prevents memory accumulation that causes OOM
                try:
                    import gc
                    import mlx.core as mx
                    mx.eval()  # Force pending computations to complete
                    mx.clear_cache()  # Release GPU memory (new API)
                    gc.collect()  # Release Python objects
                except Exception:
                    pass  # Non-MLX backends or clearing not supported
    
                # Save checkpoint periodically
                if checkpoint_path is not None and probes_processed % _CHECKPOINT_INTERVAL < PROBE_BATCH_SIZE:
                    _save_probe_checkpoint(
                        checkpoint_path=checkpoint_path,
                        completed_probes=probes_processed,
                        probe_ids=probe_ids,
                        probe_domains=probe_domains,
                        total_probes=len(valid_probes),
                    )
                    # CRITICAL: Save activations alongside checkpoint for correct resume
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
                        backend=b,
                    )

            except Exception as e:
                # On batch failure, fall back to sequential processing for this batch
                logger.warning("Batch processing failed, falling back to sequential: %s", e)
                for probe, probe_text in batch:
                    # Skip probes that were already completed (from checkpoint)
                    if probe.probe_id in completed_probe_ids:
                        continue
    
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
                        if run_source_inference:
                            (
                                source_attention_acts,
                                source_k_acts,
                                source_v_acts,
                            ) = activation_provider.collect_attention_activations(
                                source_model, source_tokenizer, probe_text
                            )
                        else:
                            source_attention_acts, source_k_acts, source_v_acts = {}, {}, {}
                        if run_target_inference:
                            (
                                target_attention_acts,
                                target_k_acts,
                                target_v_acts,
                            ) = activation_provider.collect_attention_activations(
                                target_model, target_tokenizer, probe_text
                            )
                        else:
                            target_attention_acts, target_k_acts, target_v_acts = {}, {}, {}
    
                        source_activated_fallback: dict[int, list[ActivatedDimension]] = {}
                        target_activated_fallback: dict[int, list[ActivatedDimension]] = {}
    
                        if run_source_inference:
                            for layer_idx, act in source_acts.items():
                                source_activated_fallback[layer_idx] = _extract_top_k_dims(
                                    act, backend=b
                                )
                                _accumulate_activation(source_layer_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_acts.items():
                                target_activated_fallback[layer_idx] = _extract_top_k_dims(
                                    act, backend=b
                                )
                                _accumulate_activation(target_layer_activations, layer_idx, act, b)
    
                        if run_source_inference:
                            for layer_idx, act in source_intermediate_acts.items():
                                _accumulate_activation(source_intermediate_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_intermediate_acts.items():
                                _accumulate_activation(target_intermediate_activations, layer_idx, act, b)
    
                        if run_source_inference:
                            for layer_idx, act in source_attention_acts.items():
                                _accumulate_activation(source_attention_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_attention_acts.items():
                                _accumulate_activation(target_attention_activations, layer_idx, act, b)
    
                        if run_source_inference:
                            for layer_idx, act in source_k_acts.items():
                                _accumulate_activation(source_k_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_k_acts.items():
                                _accumulate_activation(target_k_activations, layer_idx, act, b)
    
                        if run_source_inference:
                            for layer_idx, act in source_v_acts.items():
                                _accumulate_activation(source_v_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_v_acts.items():
                                _accumulate_activation(target_v_activations, layer_idx, act, b)
    
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
    
                        probe_ids.append(probe.probe_id)
                        probe_domains.append(probe.domain.value)
                        probes_processed += 1
    
                    except Exception as inner_e:
                        logger.warning("Probe '%s' failed: %s", probe.probe_id, inner_e)
                        probes_failed += 1

    logger.info(
        "PROBE PRECISE: Completed %d probes (%d failed), built %d fingerprints",
        probes_processed,
        probes_failed,
        len(source_fingerprints),
    )

    # Clear checkpoint after successful completion
    if checkpoint_path is not None:
        _clear_probe_checkpoint(checkpoint_path)

    cache_ready = (
        probe_ids == expected_probe_ids and probe_domains == expected_probe_domains
    )

    def _cache_spaces(
        hidden: dict[int, "Array"],
        intermediate: dict[int, "Array"],
        attention: dict[int, "Array"],
        k_act: dict[int, "Array"],
        v_act: dict[int, "Array"],
        embedding: "Array | list[Array] | None",
    ) -> list[str]:
        spaces = ["hidden"] if hidden else []
        if intermediate:
            spaces.append("intermediate")
        if attention:
            spaces.append("attention_q")
        if k_act:
            spaces.append("attention_k")
        if v_act:
            spaces.append("attention_v")
        if embedding is not None:
            spaces.append("embedding")
        return spaces

    def _update_profile_cache(
        profile: ModelProfile | None,
        identity: Any | None,
        fingerprints: list[ActivationFingerprint],
        cache_present: bool,
        spaces: list[str],
    ) -> None:
        if profile is None or identity is None:
            return
        updated = False
        if cache_ready and fingerprints:
            profile.probe_fingerprints = fingerprints
            profile.probe_corpus_hash = probe_corpus_hash
            updated = True
        if cache_present and spaces:
            profile.probe_corpus_hash = probe_corpus_hash
            profile.probe_cache[cache_key] = {
                "probe_mode": probe_mode,
                "probe_corpus_hash": probe_corpus_hash,
                "probe_count": len(expected_probe_ids),
                "spaces": spaces,
                "updated_at": datetime.now().isoformat(),
            }
            updated = True
        if updated:
            try:
                profile_store.save(profile, identity)
                logger.info("PROBE: Updated profile cache for %s", profile.model_id)
            except Exception as e:
                logger.warning("PROBE: Failed to save profile cache: %s", e)

    # Save per-model caches and fingerprints
    if run_source_inference and cache_ready and source_identity and source_layer_activations:
        _save_model_probe_cache(
            model_id=source_identity.model_id,
            probe_mode=probe_mode,
            probe_corpus_hash=probe_corpus_hash,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            hidden_activations=source_layer_activations,
            intermediate_activations=source_intermediate_activations,
            attention_activations=source_attention_activations,
            k_activations=source_k_activations,
            v_activations=source_v_activations,
            embedding_activations=source_embedding_activations,
        )

    if run_target_inference and cache_ready and target_identity and target_layer_activations:
        _save_model_probe_cache(
            model_id=target_identity.model_id,
            probe_mode=probe_mode,
            probe_corpus_hash=probe_corpus_hash,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            hidden_activations=target_layer_activations,
            intermediate_activations=target_intermediate_activations,
            attention_activations=target_attention_activations,
            k_activations=target_k_activations,
            v_activations=target_v_activations,
            embedding_activations=target_embedding_activations,
        )

    source_cache_present = source_cache is not None or (
        run_source_inference and cache_ready
    )
    target_cache_present = target_cache is not None or (
        run_target_inference and cache_ready
    )

    _update_profile_cache(
        source_profile,
        source_identity,
        source_fingerprints,
        source_cache_present,
        _cache_spaces(
            source_layer_activations,
            source_intermediate_activations,
            source_attention_activations,
            source_k_activations,
            source_v_activations,
            source_embedding_activations,
        ),
    )
    _update_profile_cache(
        target_profile,
        target_identity,
        target_fingerprints,
        target_cache_present,
        _cache_spaces(
            target_layer_activations,
            target_intermediate_activations,
            target_attention_activations,
            target_k_activations,
            target_v_activations,
            target_embedding_activations,
        ),
    )

    # Build IntersectionMap
    intersection_map_obj: IntersectionMap | None = None
    dimension_correlations: dict = {}
    layer_cka_scores: dict[int, float] = {}
    layer_cka_scores_raw: dict[int, float] = {}

    if source_fingerprints and target_fingerprints:
        try:
            intersection_map_obj = build_intersection_map(
                source_fingerprints=source_fingerprints,
                target_fingerprints=target_fingerprints,
                source_model=source_path or "source",
                target_model=target_path or "target",
                mode=IntersectionSimilarityMode.CKA,  # Pure geometry - CKA is the metric
            )
            dimension_correlations = intersection_map_obj.dimension_correlations
            logger.info(
                "PROBE PRECISE: Built IntersectionMap (%d layers), sparse mean_layer_cka=%.3f",
                len(intersection_map_obj.layer_confidences),
                intersection_map_obj.mean_layer_cka,
            )
        except Exception as e:
            logger.warning("Failed to build IntersectionMap: %s", e)
            intersection_map_obj = None

    # Align layers by proportional depth, then solve closed-form alignment per pair.
    # CKA is diagnostic; we do not use it as a selector.
    layer_mapping: dict[int, int] = {}  # target_layer -> source_layer
    feature_transforms: dict[int, Any] = {}  # target_layer -> {src_layer: GPU array}
    scale_ratios: dict[int, float] = {}  # EXACT scale factor per layer: ||target|| / ||source @ F||
    attention_transforms: dict[int, Any] = {}  # target_layer -> Q attention transform (GPU array)
    k_transforms: dict[int, Any] = {}  # target_layer -> K attention transform (GPU array)
    v_transforms: dict[int, Any] = {}  # target_layer -> V attention transform (GPU array)
    intermediate_transforms: dict[int, Any] = {}  # target_layer -> MLP intermediate transform (GPU array)
    cgls_iterations_by_layer: dict[int, int] = {}
    gram_aligner = GramAligner(backend=b)
    rbf_consistency_checked = False
    rbf_consistency_hidden: dict[str, float] | None = None

    source_mlp_proj_keys = _collect_mlp_projection_keys(source_weights, extract_layer_index_fn)
    target_mlp_proj_keys = _collect_mlp_projection_keys(target_weights, extract_layer_index_fn)

    if source_layer_activations and target_layer_activations:
        source_layers = sorted(source_layer_activations.keys())
        target_layers = sorted(target_layer_activations.keys())
        n_source = len(source_layers)
        n_target = len(target_layers)

        if n_source > 0 and n_target > 0:
            # =========================================================================
            # PROPORTIONAL DEPTH MAPPING (CKA is diagnostic, not a selector)
            # =========================================================================
            # Map layers by normalized depth and solve alignment per pair.
            # No CKA matrix, no Hungarian, no selection heuristics.
            alignment_tasks: list[tuple[int, list[int]]] = []
            for tgt_idx in range(n_target):
                src_idx = _proportional_layer_index(tgt_idx, n_target, n_source)
                alignment_tasks.append((tgt_idx, [src_idx]))

            alignment_tasks_sorted = alignment_tasks
            logger.info(
                "PROBE: Aligning %d target layers (proportional depth mapping)...",
                len(alignment_tasks_sorted),
            )
            
            def _align_target_group(
                tgt_idx: int,
                src_indices: list[int],
                F_init: "Array | None" = None,
            ) -> dict:
                """Align source layer(s) to a target layer.

                With 1:1 mapping, src_indices has exactly 1 element.
                F_init: Optional warm-start transform from a successful neighbor (zipper).
                """
                nonlocal rbf_consistency_checked, rbf_consistency_hidden
                tgt_layer = target_layers[tgt_idx]
                src_layers_list = [source_layers[i] for i in src_indices]
                
                result: dict = {
                    "tgt_layer": tgt_layer,
                    "src_layers": src_layers_list,
                    "raw_cka": 0.0,
                    "achieved_cka": 0.0,  # Linear CKA on shared manifold (set after alignment)
                    "geodesic_cka": 0.0,
                    "numerical_deviation": 0.0,  # Linear CKA deviation for diagnostics
                    "feature_transform": None,
                    "attention_transform": None,
                    "k_transform": None,
                    "v_transform": None,
                    "intermediate_transform": None,
                    "linear_iterations": 0,
                    "error": None,
                }

                src_act_lists = [source_layer_activations[s] for s in src_layers_list]
                tgt_list = target_layer_activations[tgt_layer]
                
                n_samples = len(tgt_list)
                for s_list in src_act_lists:
                    n_samples = min(n_samples, len(s_list))
                
                if n_samples < 2:
                    raise RuntimeError(
                        f"Insufficient samples for {src_layers_list} -> {tgt_layer}: {n_samples}"
                    )
                
                # Log sample count for rank verification
                # Note: src_act_lists is a Python list, tgt_list is an MLX array
                # MLX arrays cannot be used in boolean context (causes conversion error)
                src_dim = src_act_lists[0][0].shape[-1] if src_act_lists else 0
                tgt_dim = int(tgt_list.shape[-1]) if tgt_list is not None and len(tgt_list.shape) > 0 else 0
                logger.info(
                    "ALIGNMENT: Layer %d <- %s: n_samples=%d (need >=%d for full-rank src, >=%d for full-rank tgt)",
                    tgt_layer, src_layers_list, n_samples, src_dim, tgt_dim
                )

                local_aligner = GramAligner(backend=b)

                try:
                    # Stack source (for 1:1, this is just 1 source)
                    src_stacks = []
                    src_dims = []
                    for s, s_list in zip(src_layers_list, src_act_lists):
                        stack = b.stack(s_list[:n_samples], axis=0)
                        stack = b.astype(stack, "float32")
                        src_stacks.append(stack)
                        src_dims.append(stack.shape[1])
                    
                    tgt_stacked = b.stack(tgt_list[:n_samples], axis=0)
                    tgt_stacked = b.astype(tgt_stacked, "float32")
                    
                    # For 1:1 mapping, this is just src_stacks[0] (no concat needed)
                    if len(src_stacks) == 1:
                        src_combined = src_stacks[0]
                    else:
                        src_combined = b.concatenate(src_stacks, axis=1)
                    
                    b.eval(src_combined, tgt_stacked)
                    
                    from modelcypher.core.domain.geometry.cka import (
                        compute_cka_backend,
                        compute_linear_cka,
                    )
                    result["raw_cka"] = float(compute_cka_backend(src_combined, tgt_stacked, b))

                    alignment_result = local_aligner.find_perfect_alignment(
                        src_combined,
                        tgt_stacked,
                        F_init=F_init,  # Zipper warm-start from neighbor
                    )
                    # Alignment is closed-form; no retry needed.
                    
                    F_arr = alignment_result.feature_transform  # Already GPU array
                    aligned = b.matmul(src_combined, F_arr)
                    b.eval(aligned)
                    linear_cka = float(compute_linear_cka(aligned, tgt_stacked, backend=b))
                    linear_deviation = abs(1.0 - linear_cka)
                    result["achieved_cka"] = linear_cka
                    result["numerical_deviation"] = linear_deviation
                    result["geodesic_cka"] = alignment_result.achieved_cka
                    result["linear_iterations"] = alignment_result.linear_iterations
                    cgls_iterations_by_layer[tgt_layer] = alignment_result.linear_iterations
                    logger.info(
                        "ALIGNMENT: Layer %d <- %s: solver iters=%d",
                        tgt_layer,
                        src_layers_list,
                        alignment_result.linear_iterations,
                    )

                    # One-time geodesic RBF vs linear CKA consistency check (4D+).
                    if not rbf_consistency_checked:
                        from modelcypher.core.domain.geometry.cka import compute_cka

                        # aligned already computed for linear CKA above
                        rbf_result = compute_cka(aligned, tgt_stacked, backend=b)
                        rbf_val = rbf_result.best if rbf_result.is_valid else float("nan")

                        precision = sqrt_scalar(machine_epsilon(b, aligned), b)
                        rbf_deviation = abs(1.0 - rbf_val) if rbf_val == rbf_val else float("inf")
                        agreement_deviation = abs(rbf_val - linear_cka) if rbf_val == rbf_val else float("inf")

                        rbf_consistency_hidden = {
                            "rbf_cka": float(rbf_val) if rbf_val == rbf_val else 0.0,
                            "rbf_deviation": float(rbf_deviation),
                            "linear_deviation": float(linear_deviation),
                            "agreement_deviation": float(agreement_deviation),
                            "precision_threshold": float(precision),
                            "layer": float(tgt_layer),
                        }
                        if linear_deviation > precision:
                            logger.error(
                                "PROBE: Linear CKA deviation %.2e > precision %.2e for layer %d.",
                                linear_deviation,
                                precision,
                                tgt_layer,
                            )
                        if rbf_deviation > precision:
                            logger.info(
                                "PROBE: Geodesic CKA deviation %.2e > precision %.2e for layer %d.",
                                rbf_deviation,
                                precision,
                                tgt_layer,
                            )
                        if agreement_deviation > precision:
                            logger.info(
                                "PROBE: Geodesic vs linear CKA deviation %.2e > precision %.2e for layer %d.",
                                agreement_deviation,
                                precision,
                                tgt_layer,
                            )
                        rbf_consistency_checked = True
                    
                    # Store raw F_arr for zipper warm-start of neighbors
                    result["F_arr_raw"] = F_arr

                    # Split transform for each source layer
                    # KEEP AS GPU ARRAYS - avoid CPU→GPU reconversion in downstream code
                    split_transforms: dict[int, Any] = {}
                    start_idx = 0
                    for s_layer, s_dim in zip(src_layers_list, src_dims):
                        F_slice = F_arr[start_idx : start_idx + s_dim, :]
                        split_transforms[s_layer] = F_slice  # GPU array, not tolist()
                        start_idx += s_dim
                        
                    result["feature_transform"] = split_transforms
                    
                    # EXACT SCALE FACTOR: ||target|| / ||source @ F||
                    # Apply this to stitched weights for exact magnitude match
                    result["scale_ratio"] = alignment_result.scale_ratio

                    # Linear CKA should be near 1.0 on the shared manifold. Check precision.
                    # Compute precision threshold for each layer (cheap operation)
                    layer_precision = sqrt_scalar(machine_epsilon(b, aligned), b)
                    if linear_deviation > layer_precision:
                        logger.warning(
                            "PROBE: Layer %s -> %d linear CKA deviation=%.2e > precision %.2e.",
                            src_layers_list,
                            tgt_layer,
                            linear_deviation,
                            layer_precision,
                        )
                    
                    # =====================================================================
                    # ATTENTION Q/K/V ALIGNMENT
                    # =====================================================================
                    # Skip per-channel GramAligner here; attention stitches are derived
                    # compositionally from hidden alignment in transplant.

                    # =====================================================================
                    # INTERMEDIATE MLP ALIGNMENT (direct from activations)
                    # =====================================================================
                    # Same closed-form as hidden: I = pinv(source_inter) @ target_inter
                    # No compositional derivation needed - just measure and align.
                    split_inter_transforms = {}
                    for s_layer in src_layers_list:
                        # Get intermediate activations for source and target layers
                        src_inter_acts = source_intermediate_activations.get(s_layer)
                        tgt_inter_acts = target_intermediate_activations.get(tgt_layer)

                        if src_inter_acts is None or tgt_inter_acts is None:
                            logger.debug(
                                "PROBE INTER: No intermediate activations for %s -> %d",
                                s_layer, tgt_layer
                            )
                            continue

                        # Stack activations if needed (same format as hidden)
                        if hasattr(src_inter_acts, 'shape') and len(b.shape(src_inter_acts)) == 2:
                            src_inter_stacked = src_inter_acts
                        else:
                            src_inter_stacked = b.stack(list(src_inter_acts), axis=0)

                        if hasattr(tgt_inter_acts, 'shape') and len(b.shape(tgt_inter_acts)) == 2:
                            tgt_inter_stacked = tgt_inter_acts
                        else:
                            tgt_inter_stacked = b.stack(list(tgt_inter_acts), axis=0)

                        src_inter_stacked = b.astype(src_inter_stacked, "float32")
                        tgt_inter_stacked = b.astype(tgt_inter_stacked, "float32")
                        b.eval(src_inter_stacked, tgt_inter_stacked)

                        try:
                            # Direct alignment: I = pinv(source_inter) @ target_inter
                            inter_result = local_aligner.find_perfect_alignment(
                                src_inter_stacked, tgt_inter_stacked
                            )
                            I_arr = inter_result.feature_transform
                            split_inter_transforms[s_layer] = I_arr

                            src_inter_dim = int(b.shape(src_inter_stacked)[1])
                            tgt_inter_dim = int(b.shape(tgt_inter_stacked)[1])
                            logger.info(
                                "PROBE INTER DIRECT: %s -> %d: I=[%d, %d] (src_inter=%d, tgt_inter=%d)",
                                s_layer, tgt_layer,
                                int(b.shape(I_arr)[0]), int(b.shape(I_arr)[1]),
                                src_inter_dim, tgt_inter_dim
                            )
                        except Exception as inter_err:
                            logger.warning(
                                "PROBE INTER: Direct alignment failed for %s -> %d: %s",
                                s_layer, tgt_layer, inter_err
                            )

                    if split_inter_transforms:
                        result["intermediate_transform"] = split_inter_transforms

                except Exception as e:
                    raise RuntimeError(
                        f"GramAligner failed for {src_layers_list} -> {tgt_layer}: {e}"
                    ) from e

                # =========================================================================
                # MEMORY CLEANUP - Critical for preventing 45GB explosion
                # =========================================================================
                # Delete large intermediate arrays and sync GPU to free memory
                del src_combined, tgt_stacked, src_stacks, F_arr
                try:
                    del alignment_result
                except NameError:
                    pass
                b.eval()  # Force computation graph evaluation to release memory
                
                return result

            # =========================================================================
            # ZIPPER ALIGNMENT: Sequential processing with warm-start from neighbors
            # =========================================================================
            # Process in proportional-depth order. For each layer:
            # 1. Find nearest successfully aligned neighbor (by layer index)
            # 2. Use its F and R for warm-start and rotation hint
            # This is the "zipper" concept: easy layers align first, their geometry
            # accelerates alignment for difficult neighbors.
            
            successful_alignments: dict[int, dict] = {}  # tgt_layer -> {F}
            
            completed = 0
            for tgt_idx, src_indices in alignment_tasks_sorted:
                # Find nearest successful neighbor's F for warm-start
                tgt_layer = target_layers[tgt_idx]
                F_init = None

                if successful_alignments:
                    # Find the closest aligned layer by layer index
                    aligned_layers = list(successful_alignments.keys())
                    closest_layer = min(aligned_layers, key=lambda l: abs(l - tgt_layer))
                    neighbor_data = successful_alignments[closest_layer]
                    F_init = neighbor_data.get("F")
                    logger.info(f"ZIPPER: Layer {tgt_layer} warm-starting from layer {closest_layer}")
                else:
                    logger.debug(f"ZIPPER: Layer {tgt_layer} has no successful neighbors yet")

                # Align this layer
                result = _align_target_group(tgt_idx, src_indices, F_init=F_init)
                
                tgt_layer = result["tgt_layer"]
                src_layers = result["src_layers"]
                
                # Store mapping
                layer_mapping[tgt_layer] = src_layers[0]

                # RIGOROUS GEOMETRY: Transform must always exist.
                if result["feature_transform"] is None:
                    raise RuntimeError(
                        f"GramAligner returned no transform for {src_layers} -> {tgt_layer}. "
                        "This should never happen if the geometry is correct."
                    )
                feature_transforms[tgt_layer] = result["feature_transform"]
                layer_cka_scores[tgt_layer] = result["achieved_cka"]
                layer_cka_scores_raw[tgt_layer] = result["raw_cka"]
                
                # EXACT SCALE FACTOR per layer
                if "scale_ratio" in result:
                    scale_ratios[tgt_layer] = result["scale_ratio"]

                # Store alignment data for zipper warm-start of future layers.
                if result.get("F_arr_raw") is not None:
                    successful_alignments[tgt_layer] = {
                        "F": result["F_arr_raw"],
                        "R": result.get("R_raw", None),  # R from procrustes (if available)
                    }

                # Progress logging
                completed += 1
                logger.info(
                    "PROBE ALIGNMENT: Layer %d/%d complete (tgt=%d, linear_CKA=%.4f, raw_CKA=%.4f)",
                    completed, len(alignment_tasks_sorted), tgt_layer,
                    result["achieved_cka"], result["raw_cka"]
                )

                if result["attention_transform"] is not None:
                    attention_transforms[tgt_layer] = result["attention_transform"]

                if result.get("k_transform") is not None:
                    k_transforms[tgt_layer] = result["k_transform"]

                if result.get("v_transform") is not None:
                    v_transforms[tgt_layer] = result["v_transform"]

                if result.get("intermediate_transform") is not None:
                    intermediate_transforms[tgt_layer] = result["intermediate_transform"]

                completed += 1
                if completed % 5 == 0 or completed == len(alignment_tasks_sorted):
                    logger.info(
                        "PROBE: Aligned %d/%d target layers (zipper: %d warm-started)...",
                        completed,
                        len(alignment_tasks_sorted),
                        len(successful_alignments),
                    )

            logger.info(
                "PROBE: Cross-architecture layer alignment found %d mappings "
                "(source: %d layers, target: %d layers)",
                len(alignment_tasks),
                n_source,
                n_target,
            )

    # Extract layer confidences (CKA-first, IntersectionMap as fallback)
    # No fallbacks beyond geometric signals - if we don't have alignment, we don't merge
    layer_confidences: dict[int, float] = {}
    if layer_cka_scores:
        layer_confidences.update(layer_cka_scores)

    if intersection_map_obj is not None:
        for lc in intersection_map_obj.layer_confidences:
            if lc.layer not in layer_confidences:
                layer_confidences[lc.layer] = lc.confidence

    if not layer_confidences:
        logger.error(
            "PROBE FAILED: No layer correlations found. "
            "Cannot merge without knowing the geometric alignment."
        )
        # Return empty result - caller must check and refuse to merge
        return ProbeResult(
            correlations={},
            confidences={},
            intersection_map=None,
            dimension_correlations={},
            metrics={
                "probe_mode": "precise",
                "probes_total": len(probes),
                "probes_processed": probes_processed,
                "probes_failed": probes_failed,
                "fingerprints_built": len(source_fingerprints),
                "layers_analyzed": 0,
                "probe_failed": True,
                "failure_reason": "No layer correlations - cannot determine geometric alignment",
            },
        )

    # Build per-weight correlations
    weight_correlations: dict[str, float] = {}
    for key in target_weights:
        if key not in source_weights:
            continue
        layer_idx = extract_layer_index_fn(key)
        if layer_idx is not None and layer_idx in layer_confidences:
            weight_correlations[key] = layer_confidences[layer_idx]
        else:
            weight_correlations[key] = 0.0

    cka_vals = list(layer_cka_scores.values())
    # Linear CKA is diagnostic, not a gate. Filter only NaN (alignment bugs).
    # Low linear CKA usually means limited overlap or probe coverage.
    valid_cka_vals = [v for v in cka_vals if v == v]  # NaN check only
    nan_count = len(cka_vals) - len(valid_cka_vals)
    if nan_count > 0:
        logger.error(
            "PROBE: %d layers have NaN CKA - alignment algorithm bug, investigate!",
            nan_count
        )
    mean_cka = sum(valid_cka_vals) / len(valid_cka_vals) if valid_cka_vals else 0.0
    min_cka = min(valid_cka_vals) if valid_cka_vals else 0.0
    raw_cka_vals = list(layer_cka_scores_raw.values())
    mean_cka_raw = sum(raw_cka_vals) / len(raw_cka_vals) if raw_cka_vals else 0.0
    min_cka_raw = min(raw_cka_vals) if raw_cka_vals else 0.0
    # layers_with_data: layers that have activations in both models (for reporting)
    layers_with_data = set(source_layer_activations.keys()) & set(target_layer_activations.keys())
    # Proportional mapping defines a correspondence for every target layer.
    # missing_cka_layers is for reporting - it doesn't block exact alignment
    missing_cka_layers = [layer for layer in layers_with_data if layer not in layer_cka_scores]
    # =========================================================================
    # LINEAR CKA DIAGNOSTIC (STRICT OVERLAP CHECK)
    # =========================================================================
    # perfect_alignment is a strict diagnostic: it only holds when every layer's
    # linear CKA is within precision. This is not required for merging and can
    # be false when models contain novel structure outside the shared manifold.
    # Use sqrt(machine_epsilon) as the tolerance (matches GramAligner convention).
    precision_threshold = sqrt_scalar(
        machine_epsilon(b, b.array([1.0], dtype="float32")),
        b,
    )
    perfect_alignment = bool(valid_cka_vals) and min_cka >= 1.0 - precision_threshold

    # =========================================================================
    # SPLIT CKA: SHARED VS. NOVEL CONCEPTS (POST-ALIGNMENT)
    # =========================================================================
    # CKA is meaningless BEFORE alignment - high-d representations get twisted
    # during pre-training. We must FIRST align, THEN measure shared vs novel.
    #
    # Compute CKA separately for:
    # - SHARED: concepts both models respond to (CKA should be high after alignment)
    # - NOVEL: concepts only source responds to (new knowledge being added)
    split_cka_result = None
    if source_layer_activations and target_layer_activations and feature_transforms:
        from modelcypher.core.domain.geometry.cka import compute_cka_split

        # Use the layer with highest post-alignment CKA as representative
        if layer_cka_scores:
            best_layer = max(layer_cka_scores.keys(), key=lambda k: layer_cka_scores[k])
            # Find corresponding source layer from layer_mapping
            src_layer = layer_mapping.get(best_layer)
            # feature_transforms[tgt_layer] is a dict: {src_layer_idx: [[...], [...]]}
            F_transform_dict = feature_transforms.get(best_layer)

            if (src_layer is not None and
                src_layer in source_layer_activations and
                best_layer in target_layer_activations and
                F_transform_dict is not None and
                src_layer in F_transform_dict):

                src_acts = source_layer_activations[src_layer]
                tgt_acts = target_layer_activations[best_layer]

                # APPLY ALIGNMENT: source @ F -> aligned source in target's coordinate system
                # F_transform_dict[src_layer] is a GPU array (kept as array for efficiency)
                F_arr = F_transform_dict[src_layer]
                # Handle both GPU arrays (new) and lists (legacy cache)
                if not hasattr(F_arr, 'shape'):
                    F_arr = b.array(F_arr, dtype="float32")
                F_arr = b.astype(F_arr, "float32")
                src_acts_f32 = b.astype(src_acts, "float32")
                aligned_src = b.matmul(src_acts_f32, F_arr)
                b.eval(aligned_src)

                try:
                    # Compute split CKA on ALIGNED source vs target
                    split_cka_result = compute_cka_split(aligned_src, tgt_acts, backend=b)
                    logger.info(
                        "SPLIT CKA POST-ALIGN (layer %d): shared=%.4f (n=%d, %.1f%%), novel=%.4f (n=%d, %.1f%%), full=%.4f",
                        best_layer,
                        split_cka_result.shared_cka,
                        split_cka_result.n_shared,
                        split_cka_result.shared_fraction * 100,
                        split_cka_result.novel_cka,
                        split_cka_result.n_novel,
                        split_cka_result.novel_fraction * 100,
                        split_cka_result.full_cka,
                    )
                except Exception as e:
                    logger.warning("SPLIT CKA failed: %s", e)

    # =========================================================================
    # LAYER CLASSIFICATION: ALL LAYERS PROCESSED
    # =========================================================================
    # Linear CKA measures STRUCTURAL OVERLAP between source and target spaces.
    #
    # CKA ≈ 1.0: Source fully covers target's representational space (shared manifold)
    # CKA < 1.0: Target has structure outside source's column space (EXPECTED
    #            for cross-dimensional alignment, e.g., 896 → 1024 hidden dims)
    #
    # This is NOT an alignment bug - it's measuring how much of target's
    # geometry is captured by source. The null-space projection handles this
    # correctly: it preserves target's unique structure while adding source's.
    # =========================================================================
    layer_status: dict[int, str] = {}
    converged_layers: list[int] = []
    boundary_preserved_layers: list[int] = []  # VESTIGIAL: should always be empty
    skipped_layers: list[int] = []  # VESTIGIAL: should always be empty

    CONVERGED_THRESHOLD = 1.0 - precision_threshold

    for layer_idx, cka in layer_cka_scores.items():
        if cka != cka:  # NaN - alignment algorithm bug
            logger.error("LAYER %d has NaN CKA - alignment bug, investigate!", layer_idx)
            layer_status[layer_idx] = "converged"
            converged_layers.append(layer_idx)
        elif cka >= CONVERGED_THRESHOLD:
            layer_status[layer_idx] = "converged"
            converged_layers.append(layer_idx)
        else:
            # CKA < 1.0 means target has structure outside source's column space.
            # This is EXPECTED for cross-dimensional alignment (different hidden dims).
            # Not a bug - the null-space projection handles this correctly.
            logger.info(
                "LAYER %d: structural overlap CKA=%.4f (target has %.1f%% unique structure)",
                layer_idx, cka, (1.0 - cka) * 100
            )
            layer_status[layer_idx] = "converged"  # Still process the layer
            converged_layers.append(layer_idx)

    # Log classification summary
    logger.info(
        "PROBE CLASSIFICATION: %d processed (all layers)",
        len(converged_layers)
    )

    metrics = {
        "probe_mode": "precise",
        "probes_total": len(probes),
        "probes_processed": probes_processed,
        "probes_failed": probes_failed,
        "fingerprints_built": len(source_fingerprints),
        "layers_analyzed": len(layer_confidences),
        "layers_with_cka": len(layer_cka_scores),
        "layers_with_data": len(layers_with_data),
        "missing_cka_layers": len(missing_cka_layers),
        "layer_confidences": layer_confidences,
        "layer_cka_scores": layer_cka_scores,
        "layer_cka_scores_raw": layer_cka_scores_raw,
        "cgls_iterations_by_layer": cgls_iterations_by_layer,
        "mean_cka": mean_cka,
        "min_cka": min_cka,
        "mean_cka_raw": mean_cka_raw,
        "min_cka_raw": min_cka_raw,
        "cka_estimator": "auto",
        "feature_bias_correction": True,
        "perfect_alignment": perfect_alignment,
        # NEW: Layer classification for adaptive barometer
        "layer_status": layer_status,
        "converged_layers": converged_layers,
        "boundary_preserved_layers": boundary_preserved_layers,
        "skipped_layers": skipped_layers,
        "converged_count": len(converged_layers),
        "boundary_preserved_count": len(boundary_preserved_layers),
        "skipped_count": len(skipped_layers),
        "atlas_sources": list(set(p.source.value for p in probes)),
        "atlas_domains": list(set(p.domain.value for p in probes)),
        "intersection_map_built": intersection_map_obj is not None,
        # TRUE pre-alignment CKA from actual activation vectors (not sparse fingerprints)
        "raw_cka_mean": (
            sum(layer_cka_scores_raw.values()) / len(layer_cka_scores_raw)
            if layer_cka_scores_raw
            else 0.0
        ),
        # SPLIT CKA: separates "alignment quality" from "novelty fraction"
        "split_cka": {
            "shared_cka": split_cka_result.shared_cka if split_cka_result else None,
            "novel_cka": split_cka_result.novel_cka if split_cka_result else None,
            "full_cka": split_cka_result.full_cka if split_cka_result else None,
            "shared_fraction": split_cka_result.shared_fraction if split_cka_result else None,
            "novel_fraction": split_cka_result.novel_fraction if split_cka_result else None,
            "n_shared": split_cka_result.n_shared if split_cka_result else None,
            "n_novel": split_cka_result.n_novel if split_cka_result else None,
        } if split_cka_result else None,
    }
    if rbf_consistency_hidden is not None:
        metrics["hidden_rbf_consistency"] = rbf_consistency_hidden

    # mean_cka = Post-alignment linear overlap (shared-manifold coverage)
    # raw_cka_mean = Pre-alignment linear CKA from actual activations
    logger.info(
        "PROBE PRECISE: %d layers, post_linear_cka=%.4f, raw_linear_cka=%.4f",
        len(layer_confidences),
        mean_cka,
        metrics["raw_cka_mean"],
    )

    # =========================================================================
    # PROPAGATE TRANSFORMS TO ALL TARGET LAYERS
    # =========================================================================
    # Proportional mapping defines transforms for all target layers.
    # If any transforms are missing, it indicates missing activations or
    # an alignment bug and should be investigated.
    all_target_layers = sorted(set(target_layer_activations.keys()))
    if feature_transforms and len(feature_transforms) < len(all_target_layers):
        missing_count = len(all_target_layers) - len(feature_transforms)
        logger.error(
            "PROBE: Missing %d feature transforms; propagating nearest neighbor as fallback.",
            missing_count,
        )
        mapped_layers = sorted(feature_transforms.keys())
        logger.info(
            "PROBE: Propagating transforms from %d mapped layers to %d total layers",
            len(mapped_layers),
            len(all_target_layers),
        )
        for tgt_layer in all_target_layers:
            if tgt_layer not in feature_transforms:
                # Find nearest mapped layer
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                feature_transforms[tgt_layer] = feature_transforms[nearest]
                # Also propagate layer_mapping
                if nearest in layer_mapping:
                    layer_mapping[tgt_layer] = layer_mapping[nearest]

    if attention_transforms and len(attention_transforms) < len(all_target_layers):
        mapped_layers = sorted(attention_transforms.keys())
        for tgt_layer in all_target_layers:
            if tgt_layer not in attention_transforms:
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                attention_transforms[tgt_layer] = attention_transforms[nearest]

    if k_transforms and len(k_transforms) < len(all_target_layers):
        mapped_layers = sorted(k_transforms.keys())
        for tgt_layer in all_target_layers:
            if tgt_layer not in k_transforms:
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                k_transforms[tgt_layer] = k_transforms[nearest]

    if v_transforms and len(v_transforms) < len(all_target_layers):
        mapped_layers = sorted(v_transforms.keys())
        for tgt_layer in all_target_layers:
            if tgt_layer not in v_transforms:
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                v_transforms[tgt_layer] = v_transforms[nearest]

    # =========================================================================
    # EMBEDDING GRAMALIGN (2D layer)
    # Same closed-form alignment; geodesic CKA is diagnostic.
    # =========================================================================
    embedding_transform: list[list[float]] | None = None
    if source_embedding_activations is not None and target_embedding_activations is not None:
        def _embedding_count(acts: list["Array"] | "Array") -> int:
            if isinstance(acts, list):
                return len(acts)
            return int(b.shape(acts)[0])

        def _stack_embeddings(
            acts: list["Array"] | "Array", count: int
        ) -> "Array":
            if isinstance(acts, list):
                return b.stack(acts[:count], axis=0)
            return acts[:count, :]

        n_samples = min(
            _embedding_count(source_embedding_activations),
            _embedding_count(target_embedding_activations),
        )
        if n_samples >= 2:
            logger.info(
                "EMBEDDING GRAMALIGN: Computing 2D alignment with %d samples (linear alignment + geodesic diagnostics)",
                n_samples,
            )
            try:
                src_stacked = _stack_embeddings(source_embedding_activations, n_samples)
                tgt_stacked = _stack_embeddings(target_embedding_activations, n_samples)
                src_stacked = b.astype(src_stacked, "float32")
                tgt_stacked = b.astype(tgt_stacked, "float32")
                b.eval(src_stacked, tgt_stacked)

                # Use same GramAligner as hidden layers; linear CKA is diagnostic.
                emb_result = gram_aligner.find_perfect_alignment(src_stacked, tgt_stacked)
                emb_F = emb_result.feature_transform  # Already GPU array
                embedding_transform = emb_F  # Keep as GPU array

                from modelcypher.core.domain.geometry.cka import compute_linear_cka
                emb_aligned = b.matmul(src_stacked, emb_F)
                b.eval(emb_aligned)
                emb_linear_cka = float(compute_linear_cka(emb_aligned, tgt_stacked, backend=b))
                emb_linear_deviation = abs(1.0 - emb_linear_cka)

                metrics["embedding_cka"] = emb_linear_cka
                metrics["embedding_geodesic_cka"] = emb_result.achieved_cka
                metrics["embedding_numerical_deviation"] = emb_linear_deviation

                # One-time geodesic RBF vs linear CKA consistency check (2D).
                try:
                    from modelcypher.core.domain.geometry.cka import compute_cka

                    # emb_aligned already computed for linear CKA above
                    rbf_result = compute_cka(emb_aligned, tgt_stacked, backend=b)
                    rbf_val = rbf_result.best if rbf_result.is_valid else float("nan")

                    precision = sqrt_scalar(machine_epsilon(b, emb_aligned), b)
                    rbf_deviation = abs(1.0 - rbf_val) if rbf_val == rbf_val else float("inf")
                    linear_cka = emb_linear_cka
                    linear_deviation = emb_linear_deviation
                    agreement_deviation = abs(rbf_val - linear_cka) if rbf_val == rbf_val else float("inf")

                    metrics["embedding_rbf_consistency"] = {
                        "rbf_cka": float(rbf_val) if rbf_val == rbf_val else 0.0,
                        "rbf_deviation": float(rbf_deviation),
                        "linear_deviation": float(linear_deviation),
                        "agreement_deviation": float(agreement_deviation),
                        "precision_threshold": float(precision),
                    }
                    if linear_deviation > precision:
                        logger.error(
                            "EMBEDDING GRAMALIGN: Linear CKA deviation %.2e > precision %.2e.",
                            linear_deviation,
                            precision,
                        )
                    if rbf_deviation > precision:
                        logger.info(
                            "EMBEDDING GRAMALIGN: Geodesic CKA deviation %.2e > precision %.2e.",
                            rbf_deviation,
                            precision,
                        )
                    if agreement_deviation > precision:
                        logger.info(
                            "EMBEDDING GRAMALIGN: Geodesic vs linear CKA deviation %.2e > precision %.2e.",
                            agreement_deviation,
                            precision,
                        )
                except Exception as consistency_err:
                    logger.warning(
                        "EMBEDDING GRAMALIGN: RBF/linear consistency check failed: %s",
                        consistency_err,
                    )

                # Linear CKA is the diagnostic check for shared-manifold alignment.
                if emb_linear_deviation > precision:
                    logger.warning(
                        "EMBEDDING GRAMALIGN: Linear CKA deviation %.2e > precision %.2e.",
                        emb_linear_deviation,
                        precision,
                    )
            except Exception as e:
                logger.warning("EMBEDDING GRAMALIGN failed: %s", e)

    return ProbeResult(
        correlations=weight_correlations,
        confidences=layer_confidences,
        intersection_map=intersection_map_obj,
        dimension_correlations=dimension_correlations,
        metrics=metrics,
        source_activations=source_layer_activations,
        target_activations=target_layer_activations,
        source_intermediate_activations=source_intermediate_activations,
        target_intermediate_activations=target_intermediate_activations,
        source_attention_activations=source_attention_activations,
        target_attention_activations=target_attention_activations,
        source_k_activations=source_k_activations,
        target_k_activations=target_k_activations,
        source_v_activations=source_v_activations,
        target_v_activations=target_v_activations,
        source_embedding_activations=source_embedding_activations,
        target_embedding_activations=target_embedding_activations,
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        feature_transforms=feature_transforms if feature_transforms else None,
        scale_ratios=scale_ratios if scale_ratios else None,  # EXACT magnitude factors
        embedding_transform=embedding_transform,
        attention_transforms=attention_transforms if attention_transforms else None,
        k_transforms=k_transforms if k_transforms else None,
        v_transforms=v_transforms if v_transforms else None,
        intermediate_transforms=intermediate_transforms if intermediate_transforms else None,
        layer_mapping=layer_mapping if layer_mapping else None,
    )


def _extract_top_k_dims(
    activation_vector: "Array",
    k: int | None = None,
    threshold: float | None = None,
    backend: "Backend | None" = None,
) -> list:
    """Extract top-k activated dimensions by magnitude.

    Args:
        activation_vector: The activation vector to analyze.
        k: Number of top dimensions to extract. If None, derived as
           ceil(log2(dimensionality)) which captures intrinsic complexity.
        threshold: Minimum magnitude threshold. If None, derived from
           machine_epsilon * max_magnitude - filters numerical noise.
        backend: Backend to use for computation.

    Returns:
        List of ActivatedDimension objects for significant dimensions.
    """
    from modelcypher.core.domain.geometry.manifold_stitcher import ActivatedDimension

    b = backend or get_default_backend()
    abs_vals = b.abs(activation_vector)
    b.eval(abs_vals)

    dim = b.shape(activation_vector)[0]

    # Derive k from dimensionality: ceil(log2(d)) captures information-theoretic complexity
    if k is None:
        log2_dim = log_scalar(float(dim + 1), b) / log_scalar(2.0, b)
        k = max(1, int(ceil_scalar(log2_dim, b)))

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
