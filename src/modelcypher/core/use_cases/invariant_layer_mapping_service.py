"""
Invariant Layer Mapping Service.

Service layer for invariant-based layer mapping between models using
the enhanced InvariantLayerMapper with multi-atlas triangulation scoring.

Supports:
- Sequence Invariants: 68 probes (mathematical/logical)
- Semantic Primes: 65 probes (linguistic/mental)
- Computational Gates: 72 probes (computational/structural)
- Emotion Concepts: 32 probes (affective/relational)

Total: 237 probes for cross-domain triangulation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    SequenceFamily,
    SequenceInvariantInventory,
)
from modelcypher.core.domain.agents.unified_atlas import (
    AtlasSource,
    AtlasDomain,
    AtlasProbe,
    UnifiedAtlasInventory,
    DEFAULT_ATLAS_SOURCES,
)
from modelcypher.core.domain.geometry.invariant_layer_mapper import (
    Config,
    InvariantLayerMapper,
    InvariantScope,
    Report,
    ModelFingerprints,
    ActivationFingerprint,
    ActivatedDimension,
    ConfidenceLevel,
)
from modelcypher.core.domain.geometry.fingerprint_cache import (
    ModelFingerprintCache,
    make_config_hash,
)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    IntersectionMap,
    LayerConfidence,
    DimensionCorrelation,
    Thresholds,
)
from modelcypher.core.domain.geometry.dimension_blender import (
    DimensionBlender,
    DimensionBlendConfig,
    LayerDimensionProfile,
    get_instruct_to_coder_affinity,
    get_coder_to_instruct_affinity,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerMappingConfig:
    """Configuration for layer mapping operations."""
    source_model_path: str
    target_model_path: str
    invariant_scope: str = "sequenceInvariants"  # invariants, logicOnly, sequenceInvariants, multiAtlas
    families: list[str] | None = None
    use_triangulation: bool = True
    collapse_threshold: float = 0.35
    sample_layer_count: int = 12
    # Multi-atlas configuration (only used when invariant_scope="multiAtlas")
    atlas_sources: list[str] | None = None  # sequence_invariant, semantic_prime, computational_gate, emotion_concept
    atlas_domains: list[str] | None = None  # mathematical, logical, linguistic, etc.


@dataclass(frozen=True)
class CollapseRiskConfig:
    """Configuration for collapse risk analysis."""
    model_path: str
    families: list[str] | None = None
    collapse_threshold: float = 0.35
    sample_layer_count: int = 12


@dataclass(frozen=True)
class LayerMappingResult:
    """Result of layer mapping operation."""
    report: Report
    interpretation: str
    recommended_action: str


@dataclass(frozen=True)
class CollapseRiskResult:
    """Result of collapse risk analysis."""
    model_path: str
    layer_count: int
    collapsed_layers: int
    collapse_ratio: float
    risk_level: str  # "low", "medium", "high", "critical"
    interpretation: str
    recommended_action: str


def _parse_scope(scope_str: str) -> InvariantScope:
    """Parse scope string to InvariantScope enum."""
    scope_map = {
        "invariants": InvariantScope.INVARIANTS,
        "logiconly": InvariantScope.LOGIC_ONLY,
        "logic_only": InvariantScope.LOGIC_ONLY,
        "sequenceinvariants": InvariantScope.SEQUENCE_INVARIANTS,
        "sequence_invariants": InvariantScope.SEQUENCE_INVARIANTS,
        "multiatlas": InvariantScope.MULTI_ATLAS,
        "multi_atlas": InvariantScope.MULTI_ATLAS,
    }
    normalized = scope_str.lower().replace("-", "").replace("_", "")
    return scope_map.get(normalized, InvariantScope.SEQUENCE_INVARIANTS)


def _parse_atlas_sources(sources: list[str] | None) -> frozenset[AtlasSource] | None:
    """Parse atlas source string list to frozenset of AtlasSource."""
    if not sources:
        return None

    source_map = {
        "sequence_invariant": AtlasSource.SEQUENCE_INVARIANT,
        "sequenceinvariant": AtlasSource.SEQUENCE_INVARIANT,
        "sequence": AtlasSource.SEQUENCE_INVARIANT,
        "semantic_prime": AtlasSource.SEMANTIC_PRIME,
        "semanticprime": AtlasSource.SEMANTIC_PRIME,
        "semantic": AtlasSource.SEMANTIC_PRIME,
        "computational_gate": AtlasSource.COMPUTATIONAL_GATE,
        "computationalgate": AtlasSource.COMPUTATIONAL_GATE,
        "computational": AtlasSource.COMPUTATIONAL_GATE,
        "gate": AtlasSource.COMPUTATIONAL_GATE,
        "emotion_concept": AtlasSource.EMOTION_CONCEPT,
        "emotionconcept": AtlasSource.EMOTION_CONCEPT,
        "emotion": AtlasSource.EMOTION_CONCEPT,
    }

    result: set[AtlasSource] = set()
    for name in sources:
        normalized = name.strip().lower().replace("-", "").replace("_", "")
        if normalized in source_map:
            result.add(source_map[normalized])

    return frozenset(result) if result else None


def _parse_atlas_domains(domains: list[str] | None) -> frozenset[AtlasDomain] | None:
    """Parse atlas domain string list to frozenset of AtlasDomain."""
    if not domains:
        return None

    domain_map = {
        "mathematical": AtlasDomain.MATHEMATICAL,
        "math": AtlasDomain.MATHEMATICAL,
        "logical": AtlasDomain.LOGICAL,
        "logic": AtlasDomain.LOGICAL,
        "linguistic": AtlasDomain.LINGUISTIC,
        "language": AtlasDomain.LINGUISTIC,
        "mental": AtlasDomain.MENTAL,
        "cognitive": AtlasDomain.MENTAL,
        "computational": AtlasDomain.COMPUTATIONAL,
        "compute": AtlasDomain.COMPUTATIONAL,
        "structural": AtlasDomain.STRUCTURAL,
        "structure": AtlasDomain.STRUCTURAL,
        "affective": AtlasDomain.AFFECTIVE,
        "emotion": AtlasDomain.AFFECTIVE,
        "relational": AtlasDomain.RELATIONAL,
        "social": AtlasDomain.RELATIONAL,
        "temporal": AtlasDomain.TEMPORAL,
        "time": AtlasDomain.TEMPORAL,
        "spatial": AtlasDomain.SPATIAL,
        "space": AtlasDomain.SPATIAL,
    }

    result: set[AtlasDomain] = set()
    for name in domains:
        normalized = name.strip().lower().replace("-", "").replace("_", "")
        if normalized in domain_map:
            result.add(domain_map[normalized])

    return frozenset(result) if result else None


def _parse_families(families: list[str] | None) -> frozenset[SequenceFamily] | None:
    """Parse family string list to frozenset of SequenceFamily."""
    if not families:
        return None

    result: set[SequenceFamily] = set()
    for name in families:
        try:
            result.add(SequenceFamily(name.strip().lower()))
        except ValueError:
            pass  # Skip invalid family names

    return frozenset(result) if result else None


class InvariantLayerMappingService:
    """Service for invariant-based layer mapping between models.

    Uses the enhanced InvariantLayerMapper with multi-atlas triangulation
    scoring for robust layer alignment. Supports:
    - 68 sequence invariants (mathematical/logical)
    - 65 semantic primes (linguistic/mental)
    - 72 computational gates (computational/structural)
    - 32 emotion concepts (affective/relational)

    Total: 237 probes for cross-domain triangulation.

    Fingerprint extraction is cached to ~/Library/Caches/ModelCypher/fingerprints/
    to avoid expensive MLX inference on repeated calls.
    """

    def __init__(self, cache: Optional[ModelFingerprintCache] = None):
        """Initialize the service.

        Args:
            cache: Optional fingerprint cache (uses shared singleton if None)
        """
        self._cache = cache or ModelFingerprintCache.shared()

    def map_layers(self, config: LayerMappingConfig) -> LayerMappingResult:
        """Map layers between source and target models.

        Uses multi-atlas triangulation to find corresponding layers
        between models with different architectures.

        Args:
            config: Layer mapping configuration

        Returns:
            LayerMappingResult with report, interpretation, and recommended action

        Raises:
            ValueError: If models cannot be loaded or have incompatible structure
        """
        # Build mapper config
        scope = _parse_scope(config.invariant_scope)
        families = _parse_families(config.families)
        atlas_sources = _parse_atlas_sources(config.atlas_sources)
        atlas_domains = _parse_atlas_domains(config.atlas_domains)

        mapper_config = Config(
            invariant_scope=scope,
            family_allowlist=families,
            sample_layer_count=config.sample_layer_count,
            collapse_threshold=config.collapse_threshold,
            use_cross_domain_weighting=config.use_triangulation,
            multi_domain_bonus=config.use_triangulation,
            atlas_sources=atlas_sources,
            atlas_domains=atlas_domains,
        )

        # Load fingerprints by running probes through models
        logger.info("Extracting fingerprints from source model...")
        source_fingerprints = self._load_fingerprints(config.source_model_path, mapper_config)
        logger.info("Extracting fingerprints from target model...")
        target_fingerprints = self._load_fingerprints(config.target_model_path, mapper_config)

        # Run mapping
        report = InvariantLayerMapper.map_layers(
            source_fingerprints, target_fingerprints, mapper_config
        )

        # Generate interpretation
        interpretation = self._interpret_mapping(report)
        recommended_action = self._recommend_action(report)

        return LayerMappingResult(
            report=report,
            interpretation=interpretation,
            recommended_action=recommended_action,
        )

    def analyze_collapse_risk(self, config: CollapseRiskConfig) -> CollapseRiskResult:
        """Analyze layer collapse risk for a single model.

        Identifies layers where invariant activation is too sparse for
        reliable layer correspondence.

        Args:
            config: Collapse risk configuration

        Returns:
            CollapseRiskResult with risk assessment and recommendations
        """
        families = _parse_families(config.families)

        mapper_config = Config(
            invariant_scope=InvariantScope.SEQUENCE_INVARIANTS,
            family_allowlist=families,
            sample_layer_count=config.sample_layer_count,
            collapse_threshold=config.collapse_threshold,
        )

        # Load fingerprints
        fingerprints = self._load_fingerprints(config.model_path, mapper_config)

        # Build profile to assess collapse
        invariant_ids, _, _ = InvariantLayerMapper._get_invariants(mapper_config)
        profile = InvariantLayerMapper._build_profile(fingerprints, invariant_ids, mapper_config)

        collapsed_count = profile.collapsed_count
        layer_count = fingerprints.layer_count
        collapse_ratio = collapsed_count / max(1, layer_count)

        # Determine risk level
        if collapse_ratio >= 0.5:
            risk_level = "critical"
        elif collapse_ratio >= 0.3:
            risk_level = "high"
        elif collapse_ratio >= 0.15:
            risk_level = "medium"
        else:
            risk_level = "low"

        interpretation = self._interpret_collapse(collapse_ratio, collapsed_count, layer_count)
        recommended_action = self._recommend_collapse_action(risk_level)

        return CollapseRiskResult(
            model_path=config.model_path,
            layer_count=layer_count,
            collapsed_layers=collapsed_count,
            collapse_ratio=collapse_ratio,
            risk_level=risk_level,
            interpretation=interpretation,
            recommended_action=recommended_action,
        )

    def _load_fingerprints(
        self,
        model_path: str,
        config: Config | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ModelFingerprints:
        """Load fingerprints for a model by running probes.

        Loads the model with MLX and extracts activation fingerprints
        for each probe text in the atlas. Results are cached to avoid
        expensive repeated MLX inference.

        Args:
            model_path: Path to the model directory
            config: Mapper config to determine which probes to use
            progress_callback: Optional (current, total) progress callback
        """
        path = Path(model_path).expanduser().resolve()

        # Build config hash for cache key
        if config is None:
            config = Config()

        families_list = sorted(f.value for f in config.family_allowlist) if config.family_allowlist else None
        sources_list = sorted(s.value for s in config.atlas_sources) if config.atlas_sources else None
        domains_list = sorted(d.value for d in config.atlas_domains) if config.atlas_domains else None

        config_hash = make_config_hash(
            invariant_scope=config.invariant_scope.value,
            families=families_list,
            atlas_sources=sources_list,
            atlas_domains=domains_list,
        )

        # Check cache first
        cached = self._cache.load(str(path), config_hash)
        if cached is not None:
            logger.info("Using cached fingerprints for %s (%d probes)", path.name, len(cached.fingerprints))
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
        probe_texts = self._get_probe_texts(config)
        if not probe_texts:
            logger.warning("No probe texts found for config, returning empty fingerprints")
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
            # Return empty fingerprints on error
            return ModelFingerprints(
                model_id=str(path),
                layer_count=layer_count,
                fingerprints=[],
            )

        result = ModelFingerprints(
            model_id=str(path),
            layer_count=layer_count,
            fingerprints=fingerprints,
        )

        # Cache the result for future use
        self._cache.save(str(path), config_hash, result)

        return result

    def _get_probe_texts(self, config: Config | None) -> dict[str, str]:
        """Get probe texts based on mapper config.

        Returns dict mapping probe_id -> probe_text.
        """
        if config is None:
            config = Config()

        scope = config.invariant_scope

        if scope == InvariantScope.MULTI_ATLAS:
            # Get all atlas probes
            sources = config.atlas_sources or DEFAULT_ATLAS_SOURCES
            probes = UnifiedAtlasInventory.probes_by_source(sources)

            # Filter by domain if specified
            if config.atlas_domains:
                probes = [p for p in probes if p.domain in config.atlas_domains]

            # Build probe texts from support_texts or name
            result = {}
            for probe in probes:
                probe_id = f"{probe.source.value}:{probe.id}"
                # Use first support text if available, else the name
                if probe.support_texts:
                    result[probe_id] = probe.support_texts[0]
                else:
                    result[probe_id] = probe.name
            return result

        elif scope == InvariantScope.SEQUENCE_INVARIANTS:
            # Get sequence invariants
            families = config.family_allowlist or frozenset(SequenceFamily)
            invariants = SequenceInvariantInventory.probes_for_families(set(families))

            result = {}
            for inv in invariants:
                probe_id = f"invariant:{inv.family.value}_{inv.id}"
                # Use support texts from the invariant
                if inv.support_texts:
                    result[probe_id] = inv.support_texts[0]
                else:
                    result[probe_id] = inv.name
            return result

        elif scope == InvariantScope.LOGIC_ONLY:
            invariants = SequenceInvariantInventory.probes_for_families({SequenceFamily.LOGIC})
            result = {}
            for inv in invariants:
                probe_id = f"invariant:{inv.family.value}_{inv.id}"
                if inv.support_texts:
                    result[probe_id] = inv.support_texts[0]
                else:
                    result[probe_id] = inv.name
            return result

        else:
            # Default invariants scope
            invariants = SequenceInvariantInventory.all_probes()[:20]  # Subset for speed
            result = {}
            for inv in invariants:
                probe_id = f"invariant:{inv.family.value}_{inv.id}"
                if inv.support_texts:
                    result[probe_id] = inv.support_texts[0]
                else:
                    result[probe_id] = inv.name
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

                    # Convert to list for processing
                    abs_list = abs_vals.tolist()

                    # Find top 32 activated dimensions
                    indexed = [(i, v) for i, v in enumerate(abs_list)]
                    indexed.sort(key=lambda x: -x[1])
                    top_dims = indexed[:32]

                    # Create ActivatedDimension objects
                    # Note: ActivatedDimension uses 'dimension' not 'index'
                    activated = [
                        ActivatedDimension(dimension=dim_idx, activation=float(val))
                        for dim_idx, val in top_dims
                        if val > 0.01  # Threshold
                    ]

                    if activated:
                        layer_activations[layer_idx] = activated

                # Create fingerprint for this probe
                # Note: ActivationFingerprint uses 'prime_id' and 'activated_dimensions'
                if layer_activations:
                    fingerprints.append(
                        ActivationFingerprint(
                            prime_id=probe_id,
                            activated_dimensions=layer_activations,
                        )
                    )

            except Exception as e:
                logger.warning("Failed to process probe %s: %s", probe_id, e)
                continue

        logger.info("Extracted %d fingerprints from %d probes", len(fingerprints), total_probes)
        return fingerprints

    def _interpret_mapping(self, report: Report) -> str:
        """Generate interpretation of mapping results."""
        summary = report.summary

        if summary.alignment_quality >= 0.7:
            quality = "excellent"
        elif summary.alignment_quality >= 0.5:
            quality = "good"
        elif summary.alignment_quality >= 0.3:
            quality = "moderate"
        else:
            quality = "poor"

        collapsed_total = summary.source_collapsed_layers + summary.target_collapsed_layers

        lines = [
            f"Layer mapping quality: {quality} (alignment {summary.alignment_quality:.3f})",
            f"Mapped {summary.mapped_layers} layers, skipped {summary.skipped_layers}",
        ]

        if collapsed_total > 0:
            lines.append(f"Collapsed layers: {collapsed_total} (source: {summary.source_collapsed_layers}, target: {summary.target_collapsed_layers})")

        if summary.triangulation_quality != "none":
            lines.append(f"Triangulation quality: {summary.triangulation_quality} (multiplier {summary.mean_triangulation_multiplier:.2f})")

        # Add multi-atlas metrics if available
        if summary.total_probes_used > 68:  # More than just sequence invariants
            lines.append(f"Multi-atlas: {summary.atlas_sources_detected} sources, {summary.atlas_domains_detected} domains, {summary.total_probes_used} probes")

        return " | ".join(lines)

    def _recommend_action(self, report: Report) -> str:
        """Generate recommended action based on mapping results."""
        summary = report.summary

        if summary.alignment_quality < 0.3:
            # If not using multi-atlas, suggest upgrading
            if report.config.invariant_scope != InvariantScope.MULTI_ATLAS:
                return "Consider using multiAtlas scope for 237 probes across all atlases; current coverage is too sparse."
            return "Consider using CKA-based layer matching instead; invariant coverage is too sparse."

        if summary.source_collapsed_layers + summary.target_collapsed_layers > summary.mapped_layers * 0.3:
            if report.config.invariant_scope != InvariantScope.MULTI_ATLAS:
                return "High collapse rate. Consider using multiAtlas scope for higher anchor density."
            return "High collapse rate detected. Try lowering collapse threshold or reviewing model compatibility."

        if summary.triangulation_quality == "none" and report.config.invariant_scope in (InvariantScope.SEQUENCE_INVARIANTS, InvariantScope.MULTI_ATLAS):
            return "Enable multi_domain_bonus in config for improved triangulation scoring."

        if summary.alignment_quality >= 0.7:
            return "Proceed with merge using the layer correspondence. High confidence in alignment."

        return "Layer mapping complete. Review correspondence before merge."

    def _interpret_collapse(self, ratio: float, collapsed: int, total: int) -> str:
        """Generate interpretation of collapse risk."""
        return (
            f"{collapsed}/{total} layers ({ratio*100:.1f}%) have insufficient invariant coverage. "
            f"Layers below collapse threshold may produce unreliable mappings."
        )

    def _recommend_collapse_action(self, risk_level: str) -> str:
        """Generate recommended action for collapse risk level."""
        actions = {
            "low": "Collapse risk is acceptable. Proceed with layer mapping.",
            "medium": "Consider using multiAtlas scope for 237 probes across all atlases.",
            "high": "High collapse risk. Use multiAtlas scope for maximum anchor density.",
            "critical": "Critical collapse risk. Use multiAtlas scope or consider alternative alignment methods.",
        }
        return actions.get(risk_level, "Unknown risk level.")

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
            "invariantScope": report.config.invariant_scope.value,
            "mappedLayers": summary.mapped_layers,
            "skippedLayers": summary.skipped_layers,
            "meanSimilarity": summary.mean_similarity,
            "alignmentQuality": summary.alignment_quality,
            "sourceCollapsedLayers": summary.source_collapsed_layers,
            "targetCollapsedLayers": summary.target_collapsed_layers,
            "meanTriangulationMultiplier": summary.mean_triangulation_multiplier,
            "triangulationQuality": summary.triangulation_quality,
            # Multi-atlas metrics
            "atlasSourcesDetected": summary.atlas_sources_detected,
            "atlasDomainsDetected": summary.atlas_domains_detected,
            "totalProbesUsed": summary.total_probes_used,
            "mappings": [
                {
                    "sourceLayer": m.source_layer,
                    "targetLayer": m.target_layer,
                    "similarity": m.similarity,
                    "confidence": m.confidence.value,
                    "isSkipped": m.is_skipped,
                }
                for m in report.mappings
            ],
            "interpretation": result.interpretation,
            "recommendedAction": result.recommended_action,
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
            "riskLevel": result.risk_level,
            "interpretation": result.interpretation,
            "recommendedAction": result.recommended_action,
        }

    # -------------------------------------------------------------------------
    # Intersection Map Conversion (for merge integration)
    # -------------------------------------------------------------------------

    @staticmethod
    def to_intersection_map(result: LayerMappingResult) -> IntersectionMap:
        """Convert LayerMappingResult to IntersectionMap for merge integration.

        This enables the merge engine to use per-layer confidence from
        multi-atlas triangulation to drive adaptive alpha blending.

        The conversion maps:
        - Per-layer similarity → dimension correlation strength
        - Mapping confidence → LayerConfidence (strong/moderate/weak)
        - Triangulation multiplier → correlation boost

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

            # Classify mapping confidence into strong/moderate/weak
            sim = mapping.similarity
            strong = 0
            moderate = 0
            weak = 0

            if mapping.confidence == ConfidenceLevel.HIGH:
                strong = 1
            elif mapping.confidence == ConfidenceLevel.MEDIUM:
                moderate = 1
            else:
                weak = 1

            # Apply triangulation boost if available
            # High triangulation quality means more reliable correlation
            tri_mult = mapping.triangulation_multiplier if hasattr(mapping, 'triangulation_multiplier') else 1.0

            # Create layer confidence
            # The LayerConfidence.__post_init__ computes confidence automatically
            layer_confidences.append(
                LayerConfidence(
                    layer=layer,
                    strong_correlations=strong,
                    moderate_correlations=moderate,
                    weak_correlations=weak,
                )
            )

            # Create dimension correlation for this layer
            # Use similarity as the correlation value, boosted by triangulation
            boosted_correlation = min(1.0, sim * tri_mult)
            dimension_correlations[layer] = [
                DimensionCorrelation(
                    source_dim=0,  # Placeholder - full layer mapping
                    target_dim=0,
                    correlation=boosted_correlation,
                )
            ]

        return IntersectionMap(
            source_model=report.source_model,
            target_model=report.target_model,
            dimension_correlations=dimension_correlations,
            overall_correlation=summary.alignment_quality,
            aligned_dimension_count=summary.mapped_layers,
            total_source_dims=summary.mapped_layers + summary.skipped_layers,
            total_target_dims=summary.mapped_layers + summary.skipped_layers,
            layer_confidences=layer_confidences,
        )

    @staticmethod
    def confidence_based_alpha(layer_confidence: LayerConfidence | None, fallback_alpha: float = 0.5) -> float:
        """Compute adaptive alpha based on layer confidence.

        Ported from TrainingCypher's RotationalModelMerger.confidenceBasedAlpha().

        The formula: alpha = 1.0 - (confidence * 0.8)
        - High confidence (0.7+) → alpha ≈ 0.44 (trust projected weights more)
        - Medium confidence (0.5) → alpha = 0.6 (balanced)
        - Low confidence (0.2) → alpha = 0.84 (trust target weights more)

        Args:
            layer_confidence: Per-layer confidence from intersection map
            fallback_alpha: Alpha to use when no confidence data available

        Returns:
            Adaptive alpha value in [0.2, 1.0]
        """
        if layer_confidence is None:
            return fallback_alpha

        # alpha = 1.0 - (confidence * 0.8)
        # Bounds: confidence=0 → alpha=1.0, confidence=1 → alpha=0.2
        raw_alpha = 1.0 - (layer_confidence.confidence * 0.8)

        # Clamp to [0.2, 1.0] to ensure reasonable blending
        return max(0.2, min(1.0, raw_alpha))

    @staticmethod
    def alpha_by_layer(result: LayerMappingResult, fallback_alpha: float = 0.5) -> dict[int, float]:
        """Compute per-layer adaptive alpha from layer mapping results.

        This is the main entry point for geometry-driven merge alpha.

        Args:
            result: Layer mapping result from map_layers()
            fallback_alpha: Alpha to use for layers without mapping

        Returns:
            Dict mapping layer_index → adaptive_alpha
        """
        intersection_map = InvariantLayerMappingService.to_intersection_map(result)
        confidence_by_layer = {lc.layer: lc for lc in intersection_map.layer_confidences}

        alpha_map: dict[int, float] = {}
        for mapping in result.report.mappings:
            layer = mapping.source_layer
            layer_conf = confidence_by_layer.get(layer)
            alpha_map[layer] = InvariantLayerMappingService.confidence_based_alpha(
                layer_conf, fallback_alpha
            )

        return alpha_map

    @staticmethod
    def intersection_map_payload(intersection_map: IntersectionMap) -> dict:
        """Convert IntersectionMap to JSON-serializable payload."""
        return {
            "_schema": "mc.geometry.intersection_map.v1",
            "sourceModel": intersection_map.source_model,
            "targetModel": intersection_map.target_model,
            "overallCorrelation": intersection_map.overall_correlation,
            "alignedDimensionCount": intersection_map.aligned_dimension_count,
            "totalSourceDims": intersection_map.total_source_dims,
            "totalTargetDims": intersection_map.total_target_dims,
            "layerConfidences": [
                {
                    "layer": lc.layer,
                    "strongCorrelations": lc.strong_correlations,
                    "moderateCorrelations": lc.moderate_correlations,
                    "weakCorrelations": lc.weak_correlations,
                    "confidence": lc.confidence,
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
    # Dimension Blending Methods
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
        return {
            f"{probe.source.value}:{probe.id}": probe.domain
            for probe in probes
        }

    @staticmethod
    def fingerprints_to_dicts(
        fingerprints: ModelFingerprints,
    ) -> list[dict]:
        """
        Convert ActivationFingerprint objects to dicts for dimension blending.

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
                    {"dimension": d.dimension, "activation": d.activation}
                    for d in dims
                ]
            result.append({
                "probe_id": fp.prime_id,
                "activated_dimensions": activated_dims,
            })
        return result

    @classmethod
    def compute_dimension_profiles(
        cls,
        source_fingerprints: ModelFingerprints,
        target_fingerprints: ModelFingerprints,
        probes: list[AtlasProbe],
        layer_indices: list[int] | None = None,
    ) -> tuple[dict[int, LayerDimensionProfile], dict[int, LayerDimensionProfile]]:
        """
        Compute per-dimension domain profiles for both models.

        Args:
            source_fingerprints: Source model fingerprints
            target_fingerprints: Target model fingerprints
            probes: List of atlas probes used for fingerprinting
            layer_indices: Which layers to analyze (default: all available)

        Returns:
            Tuple of (source_profiles, target_profiles)
        """
        import numpy as np

        probe_domain_map = cls.build_probe_domain_map(probes)

        # Determine layer indices from fingerprints
        if layer_indices is None:
            source_layers: set[int] = set()
            for fp in source_fingerprints.fingerprints:
                source_layers.update(fp.activated_dimensions.keys())
            layer_indices = sorted(source_layers)

        # Estimate hidden dimension from first fingerprint with activations
        hidden_dim = 2048  # Default
        for fp in source_fingerprints.fingerprints:
            for layer_dims in fp.activated_dimensions.values():
                if layer_dims:
                    # Get max dimension index seen
                    max_dim = max(d.dimension for d in layer_dims)
                    # Round up to power of 2 (typical hidden dims)
                    hidden_dim = 1 << (max_dim.bit_length())
                    break
            if hidden_dim != 2048:
                break

        # Convert fingerprints to dicts
        source_dicts = cls.fingerprints_to_dicts(source_fingerprints)
        target_dicts = cls.fingerprints_to_dicts(target_fingerprints)

        # Compute profiles
        source_profiles = DimensionBlender.compute_dimension_profiles(
            source_dicts, probe_domain_map, layer_indices, hidden_dim
        )
        target_profiles = DimensionBlender.compute_dimension_profiles(
            target_dicts, probe_domain_map, layer_indices, hidden_dim
        )

        return source_profiles, target_profiles

    @staticmethod
    def compute_dimension_alpha_vectors(
        source_profiles: dict[int, LayerDimensionProfile],
        target_profiles: dict[int, LayerDimensionProfile],
        config: DimensionBlendConfig | None = None,
        merge_direction: str = "instruct_to_coder",
    ) -> dict[int, "np.ndarray"]:
        """
        Compute per-layer alpha vectors for dimension-level blending.

        Args:
            source_profiles: Source model dimension profiles
            target_profiles: Target model dimension profiles
            config: Blend configuration (default: domain-based affinity)
            merge_direction: "instruct_to_coder" or "coder_to_instruct"

        Returns:
            Dict mapping layer_index to alpha vector (shape: hidden_dim,)
        """
        import numpy as np

        if config is None:
            # Use default domain affinity map based on merge direction
            if merge_direction == "instruct_to_coder":
                domain_map = get_instruct_to_coder_affinity()
            else:
                domain_map = get_coder_to_instruct_affinity()

            config = DimensionBlendConfig(
                domain_alpha_map=domain_map,
                default_alpha=0.5,
            )

        # Use source profiles for classification
        # (target profiles can be used for validation/consensus)
        return DimensionBlender.compute_alpha_vectors(source_profiles, config)

    @classmethod
    def dimension_alpha_from_mapping(
        cls,
        result: LayerMappingResult,
        source_fingerprints: ModelFingerprints,
        target_fingerprints: ModelFingerprints,
        probes: list[AtlasProbe],
        merge_direction: str = "instruct_to_coder",
    ) -> dict[int, "np.ndarray"]:
        """
        Compute per-dimension alpha vectors from layer mapping result.

        This is the main entry point for dimension-level blending.

        Args:
            result: Layer mapping result
            source_fingerprints: Source model fingerprints
            target_fingerprints: Target model fingerprints
            probes: Atlas probes used for mapping
            merge_direction: "instruct_to_coder" or "coder_to_instruct"

        Returns:
            Dict mapping layer_index to alpha vector (shape: hidden_dim,)
        """
        # Get layer indices from mapping result
        layer_indices = [m.source_layer for m in result.report.mappings]

        # Compute dimension profiles
        source_profiles, target_profiles = cls.compute_dimension_profiles(
            source_fingerprints,
            target_fingerprints,
            probes,
            layer_indices,
        )

        # Log profile summary
        summary = DimensionBlender.summarize_profiles(source_profiles)
        logger.info(
            "Dimension profiles: %d layers analyzed, avg classification rate %.1f%%",
            summary["layer_count"],
            100.0 * sum(
                l["classification_rate"]
                for l in summary["layers"].values()
            ) / max(1, len(summary["layers"])),
        )

        # Compute alpha vectors
        return cls.compute_dimension_alpha_vectors(
            source_profiles,
            target_profiles,
            merge_direction=merge_direction,
        )
