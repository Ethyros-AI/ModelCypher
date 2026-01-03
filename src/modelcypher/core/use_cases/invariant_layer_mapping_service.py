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
Invariant Layer Mapping Service.

Service layer for invariant-based layer mapping between models using
the enhanced InvariantLayerMapper with multi-atlas triangulation scoring.

Supports:
- Sequence Invariants: 70 probes (mathematical/logical)
- Semantic Primes: 65 probes (linguistic/mental)
- Computational Gates: 76 probes (computational/structural)
- Emotion Concepts: 32 probes (affective/relational)
- Temporal Concepts: 25 probes (temporal/logical)
- Spatial Concepts: 23 probes (spatial grounding)
- Social Concepts: 25 probes (relational/linguistic)
- Moral Concepts: 30 probes (moral/relational)
- Compositional: 22 probes (semantic prime compositions)
- Philosophical: 30 probes (philosophical/logical)
- Conceptual Genealogy: 29 probes (etymology/lineage)
- Metaphor Invariants: 14 probes (cross-cultural semantics)
- Syntax Concepts: 24 probes (syntax, morphology, word order)
- Safety Ethics: 34 probes (consent, autonomy, coercion, boundaries)

See UnifiedAtlasInventory.total_probe_count() for current total.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from modelcypher.core.domain.agents.unified_atlas import (
    AtlasDomain,
    AtlasProbe,
    UnifiedAtlasInventory,
)
from modelcypher.core.use_cases.atlas_bootstrap import register_default_atlas_inventories
from modelcypher.core.domain.geometry.fingerprint_cache import (
    ModelFingerprintCache,
    make_config_hash,
)
from modelcypher.core.domain.geometry.invariant_layer_mapper import (
    ActivatedDimension,
    ActivationFingerprint,
    InvariantLayerMapper,
    ModelFingerprints,
    Report,
)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    DimensionCorrelation,
    IntersectionMap,
    LayerConfidence,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerMappingResult:
    """Result of layer mapping operation."""

    report: Report


@dataclass(frozen=True)
class CollapseRiskResult:
    """Result of collapse risk analysis."""

    model_path: str
    layer_count: int
    collapsed_layers: int
    collapse_ratio: float


class InvariantLayerMappingService:
    """Service for invariant-based layer mapping between models.

    Uses the enhanced InvariantLayerMapper with multi-atlas triangulation
    scoring for robust layer alignment. Supports:
    - 70 sequence invariants (mathematical/logical)
    - 65 semantic primes (linguistic/mental)
    - 76 computational gates (computational/structural)
    - 32 emotion concepts (affective/relational)
    - 25 temporal concepts (temporal/logical)
    - 25 social concepts (relational/linguistic)
    - 30 moral concepts (moral/relational)
    - 22 compositional probes (semantic prime compositions)
    - 30 philosophical concepts (philosophical/logical)
    - 29 conceptual genealogy probes (etymology/lineage)
    - 14 metaphor invariants (cross-cultural semantics)
    - 24 syntax concepts (syntax, morphology, word order)
    - 34 safety ethics probes (consent, autonomy, coercion, boundaries)

    See UnifiedAtlasInventory.total_probe_count() for current total.

    Fingerprint extraction is cached to ~/Library/Caches/ModelCypher/fingerprints/
    to avoid expensive MLX inference on repeated calls.
    """

    def __init__(self, cache: ModelFingerprintCache | None = None):
        """Initialize the service.

        Args:
            cache: Optional fingerprint cache (uses shared singleton if None)
        """
        register_default_atlas_inventories()
        self._cache = cache or ModelFingerprintCache.shared()

    def map_layers(
        self,
        source_model_path: str,
        target_model_path: str,
    ) -> LayerMappingResult:
        """Map layers between source and target models.

        Uses multi-atlas triangulation to find corresponding layers
        between models with different architectures.

        Args:
            source_model_path: Path to the source model directory
            target_model_path: Path to the target model directory

        Returns:
            LayerMappingResult with raw mapping report

        Raises:
            ValueError: If models cannot be loaded or probe extraction fails
        """
        # Load fingerprints by running probes through models
        logger.info("Extracting fingerprints from source model...")
        source_fingerprints = self._load_fingerprints(source_model_path)
        logger.info("Extracting fingerprints from target model...")
        target_fingerprints = self._load_fingerprints(target_model_path)

        # Run mapping
        report = InvariantLayerMapper.map_layers(source_fingerprints, target_fingerprints)

        return LayerMappingResult(
            report=report,
        )

    def analyze_collapse_risk(self, model_path: str) -> CollapseRiskResult:
        """Analyze layer collapse risk for a single model.

        Identifies layers where invariant activation is too sparse for
        reliable layer correspondence.

        Args:
            model_path: Path to the model directory

        Returns:
            CollapseRiskResult with raw collapse measurements
        """
        # Load fingerprints
        fingerprints = self._load_fingerprints(model_path)

        # Build profile to assess collapse
        invariant_ids, _, _ = InvariantLayerMapper._get_invariants()
        profile = InvariantLayerMapper._build_profile(fingerprints, invariant_ids)

        collapsed_count = profile.collapsed_count
        layer_count = fingerprints.layer_count
        collapse_ratio = collapsed_count / max(1, layer_count)

        return CollapseRiskResult(
            model_path=model_path,
            layer_count=layer_count,
            collapsed_layers=collapsed_count,
            collapse_ratio=collapse_ratio,
        )

    def _load_fingerprints(
        self,
        model_path: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ModelFingerprints:
        """Load fingerprints for a model by running probes.

        Loads the model with MLX and extracts activation fingerprints
        for each probe text in the atlas. Results are cached to avoid
        expensive repeated MLX inference.

        Args:
            model_path: Path to the model directory
            progress_callback: Optional (current, total) progress callback
        """
        path = Path(model_path).expanduser().resolve()

        config_hash = make_config_hash(invariant_scope="all")

        # Check cache first
        cached = self._cache.load(str(path), config_hash)
        if cached is not None:
            logger.info(
                "Using cached fingerprints for %s (%d probes)", path.name, len(cached.fingerprints)
            )
            return cached

        # Get model config for layer count
        layer_count = 32  # Default
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    model_config = json.load(f)
                layer_count = model_config.get("num_hidden_layers", 32)
            except (json.JSONDecodeError, KeyError):
                pass

        # Get probe texts based on config
        probe_texts = self._get_probe_texts()
        if not probe_texts:
            logger.warning("No probe texts found, returning empty fingerprints")
            return ModelFingerprints(
                model_id=str(path),
                layer_count=layer_count,
                fingerprints=[],
            )

        logger.info("Loading model from %s for fingerprinting (%d probes)", path, len(probe_texts))

        try:
            fingerprints = self._extract_fingerprints(
                model_path=str(path),
                probe_texts=probe_texts,
                layer_count=layer_count,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.error("Failed to extract fingerprints: %s", e)
            raise RuntimeError(
                f"Fingerprint extraction failed for {path}: {e}"
            ) from e

        result = ModelFingerprints(
            model_id=str(path),
            layer_count=layer_count,
            fingerprints=fingerprints,
        )

        # Cache the result for future use
        self._cache.save(str(path), config_hash, result)

        return result

    def _get_probe_texts(self) -> dict[str, str]:
        """Get probe texts from the full unified atlas."""
        probes = UnifiedAtlasInventory.all_probes()
        result: dict[str, str] = {}
        for probe in probes:
            probe_id = f"{probe.source.value}:{probe.id}"
            if probe.support_texts:
                result[probe_id] = probe.support_texts[0]
            else:
                result[probe_id] = probe.name
        return result

    def _extract_fingerprints(
        self,
        model_path: str,
        probe_texts: dict[str, str],
        layer_count: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ActivationFingerprint]:
        """Extract activation fingerprints by running probes through model.

        Uses MLX to load the model and capture hidden states at each layer.
        """
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError as e:
            logger.error("MLX not available: %s", e)
            return []

        # Load model
        model, tokenizer = load(model_path)
        inner_model = model.model
        layers = inner_model.layers
        actual_layer_count = len(layers)

        logger.info("Model loaded: %d layers", actual_layer_count)

        fingerprints = []
        total_probes = len(probe_texts)

        for idx, (probe_id, probe_text) in enumerate(probe_texts.items()):
            if progress_callback:
                progress_callback(idx + 1, total_probes)

            try:
                # Tokenize probe text
                tokens = tokenizer.encode(probe_text)
                if not tokens:
                    continue

                input_ids = mx.array([tokens])

                # Forward through model capturing hidden states
                layer_activations: dict[int, list[ActivatedDimension]] = {}

                # Get initial embeddings
                h = inner_model.embed_tokens(input_ids)

                # Forward through each layer
                for layer_idx, layer in enumerate(layers):
                    h_out = layer(h, mask=None, cache=None)
                    if isinstance(h_out, tuple):
                        h = h_out[0]
                    else:
                        h = h_out

                    # Compute activation metrics for this layer
                    # Use L2 norm of the hidden state as activation strength
                    # Take the last token position (most relevant for probe)
                    last_hidden = h[0, -1, :]  # Shape: (hidden_dim,)
                    mx.eval(last_hidden)

                    # Get top-k activated dimensions
                    abs_vals = mx.abs(last_hidden)
                    mx.eval(abs_vals)
                    # Derive top-k from dimensionality (no fixed cap)
                    b = self._backend
                    dim = int(abs_vals.shape[0])
                    from modelcypher.core.domain.geometry.numerical_stability import (
                        ceil_scalar,
                        log2_scalar,
                    )

                    log2_dim = log2_scalar(float(dim + 1), b)
                    top_k = max(1, int(ceil_scalar(log2_dim, b)))
                    if top_k > dim:
                        top_k = dim
                    neg_abs = -abs_vals
                    top_idx = b.argsort(neg_abs)[:top_k]
                    top_vals = b.take(abs_vals, top_idx, axis=0)
                    b.eval(top_idx, top_vals)
                    top_idx_list = [int(x) for x in b.tolist(top_idx)]
                    top_val_list = [float(x) for x in b.tolist(top_vals)]
                    top_dims = list(zip(top_idx_list, top_val_list))

                    # Derive activation threshold from dtype and data range
                    # Use division_epsilon which accounts for machine precision
                    from modelcypher.core.domain.geometry.numerical_stability import (
                        division_epsilon,
                    )

                    eps = division_epsilon(b, abs_vals)
                    max_val_arr = b.max(abs_vals)
                    b.eval(max_val_arr)
                    max_val = float(b.to_scalar(max_val_arr)) if top_k > 0 else 1.0
                    activation_threshold = eps * max_val

                    # Create ActivatedDimension objects
                    activated = [
                        ActivatedDimension(index=dim_idx, activation=float(val))
                        for dim_idx, val in top_dims
                        if val > activation_threshold
                    ]

                    if activated:
                        layer_activations[layer_idx] = activated

                # Create fingerprint for this probe
                if layer_activations:
                    fingerprints.append(
                        ActivationFingerprint(
                            prime_id=probe_id,
                            prime_text=probe_text,
                            activated_dimensions=layer_activations,
                        )
                    )

            except Exception as e:
                logger.warning("Failed to process probe %s: %s", probe_id, e)
                continue

        logger.info("Extracted %d fingerprints from %d probes", len(fingerprints), total_probes)
        return fingerprints

    @staticmethod
    def result_payload(result: LayerMappingResult) -> dict:
        """Convert LayerMappingResult to CLI/MCP payload."""
        report = result.report
        summary = report.summary

        payload = {
            "_schema": "mc.geometry.invariant.map_layers.v1",
            "sourceModel": report.source_model,
            "targetModel": report.target_model,
            "invariantCount": report.invariant_count,
            "mappedLayers": summary.mapped_layers,
            "meanSimilarity": summary.mean_similarity,
            "alignmentQuality": summary.alignment_quality,
            "sourceCollapsedLayers": summary.source_collapsed_layers,
            "targetCollapsedLayers": summary.target_collapsed_layers,
            "meanTriangulationMultiplier": summary.mean_triangulation_multiplier,
            # Multi-atlas metrics
            "atlasSourcesDetected": summary.atlas_sources_detected,
            "atlasDomainsDetected": summary.atlas_domains_detected,
            "totalProbesUsed": summary.total_probes_used,
            "mappings": [
                {
                    "sourceLayer": m.source_layer,
                    "targetLayer": m.target_layer,
                    "similarity": m.similarity,
                }
                for m in report.mappings
            ],
        }

        return payload

    @staticmethod
    def collapse_risk_payload(result: CollapseRiskResult) -> dict:
        """Convert CollapseRiskResult to CLI/MCP payload."""
        return {
            "_schema": "mc.geometry.invariant.collapse_risk.v1",
            "modelPath": result.model_path,
            "layerCount": result.layer_count,
            "collapsedLayers": result.collapsed_layers,
            "collapseRatio": result.collapse_ratio,
        }

    # -------------------------------------------------------------------------
    # Intersection Map Conversion (for merge integration)
    # -------------------------------------------------------------------------

    @staticmethod
    def to_intersection_map(result: LayerMappingResult) -> IntersectionMap:
        """Convert LayerMappingResult to IntersectionMap for merge integration.

        This enables the merge engine to use per-layer similarity from
        multi-atlas triangulation as raw confidence for alignment analysis.

        Args:
            result: Layer mapping result from map_layers()

        Returns:
            IntersectionMap suitable for passing to merge engine
        """
        report = result.report
        summary = report.summary

        # Build per-layer confidences from mappings
        layer_confidences: list[LayerConfidence] = []
        dimension_correlations: dict[int, list[DimensionCorrelation]] = {}

        for mapping in report.mappings:
            layer = mapping.source_layer

            sim = mapping.similarity

            # Create layer confidence from raw similarity
            layer_confidences.append(
                LayerConfidence(
                    layer=layer,
                    confidence=sim,
                    correlation_count=1,
                )
            )

            # Create dimension correlation for this layer
            dimension_correlations[layer] = [
                DimensionCorrelation(
                    source_dim=0,  # Placeholder - full layer mapping
                    target_dim=0,
                    correlation=sim,
                )
            ]

        return IntersectionMap(
            source_model=report.source_model,
            target_model=report.target_model,
            dimension_correlations=dimension_correlations,
            raw_fingerprint_similarity=summary.alignment_quality,
            aligned_dimension_count=summary.mapped_layers,
            total_source_dims=summary.mapped_layers,
            total_target_dims=summary.mapped_layers,
            layer_confidences=layer_confidences,
        )

    @staticmethod
    def intersection_map_payload(intersection_map: IntersectionMap) -> dict:
        """Convert IntersectionMap to JSON-serializable payload."""
        return {
            "_schema": "mc.geometry.intersection_map.v2",
            "sourceModel": intersection_map.source_model,
            "targetModel": intersection_map.target_model,
            "rawFingerprintSimilarity": intersection_map.raw_fingerprint_similarity,
            "alignedDimensionCount": intersection_map.aligned_dimension_count,
            "totalSourceDims": intersection_map.total_source_dims,
            "totalTargetDims": intersection_map.total_target_dims,
            "layerConfidences": [
                {
                    "layer": lc.layer,
                    "confidence": lc.confidence,
                    "correlationCount": lc.correlation_count,
                }
                for lc in intersection_map.layer_confidences
            ],
            "dimensionCorrelations": {
                str(layer): [
                    {
                        "sourceDim": dc.source_dim,
                        "targetDim": dc.target_dim,
                        "correlation": dc.correlation,
                    }
                    for dc in correlations
                ]
                for layer, correlations in intersection_map.dimension_correlations.items()
            },
        }

    # =========================================================================
    # Dimension Analysis Helpers
    # =========================================================================

    @staticmethod
    def build_probe_domain_map(probes: list[AtlasProbe]) -> dict[str, AtlasDomain]:
        """
        Build mapping from probe ID to domain for dimension classification.

        Args:
            probes: List of AtlasProbe objects

        Returns:
            Dict mapping probe_id -> AtlasDomain
        """
        return {f"{probe.source.value}:{probe.id}": probe.domain for probe in probes}

    @staticmethod
    def fingerprints_to_dicts(
        fingerprints: ModelFingerprints,
    ) -> list[dict]:
        """
        Convert ActivationFingerprint objects to dicts for dimension analysis.

        Args:
            fingerprints: ModelFingerprints with list of ActivationFingerprint

        Returns:
            List of dicts with probe_id and activated_dimensions
        """
        result = []
        for fp in fingerprints.fingerprints:
            activated_dims = {}
            for layer_idx, dims in fp.activated_dimensions.items():
                activated_dims[str(layer_idx)] = [
                    {"dimension": d.index, "activation": d.activation} for d in dims
                ]
            result.append(
                {
                    "probe_id": fp.prime_id,
                    "activated_dimensions": activated_dims,
                }
            )
        return result
