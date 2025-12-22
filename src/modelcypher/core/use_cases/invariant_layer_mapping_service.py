"""
Invariant Layer Mapping Service.

Service layer for invariant-based layer mapping between models using
the enhanced InvariantLayerMapper with triangulation scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    SequenceFamily,
    SequenceInvariantInventory,
)
from modelcypher.core.domain.geometry.invariant_layer_mapper import (
    Config,
    InvariantLayerMapper,
    InvariantScope,
    Report,
    ModelFingerprints,
    ActivationFingerprint,
    ActivatedDimension,
)


@dataclass(frozen=True)
class LayerMappingConfig:
    """Configuration for layer mapping operations."""
    source_model_path: str
    target_model_path: str
    invariant_scope: str = "sequenceInvariants"
    families: list[str] | None = None
    use_triangulation: bool = True
    collapse_threshold: float = 0.35
    sample_layer_count: int = 12


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
    }
    normalized = scope_str.lower().replace("-", "").replace("_", "")
    return scope_map.get(normalized, InvariantScope.SEQUENCE_INVARIANTS)


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

    Uses the enhanced InvariantLayerMapper with 68 sequence invariants
    and cross-domain triangulation scoring for robust layer alignment.
    """

    def __init__(self):
        """Initialize the service."""
        pass

    def map_layers(self, config: LayerMappingConfig) -> LayerMappingResult:
        """Map layers between source and target models.

        Uses sequence invariant triangulation to find corresponding layers
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

        mapper_config = Config(
            invariant_scope=scope,
            family_allowlist=families,
            sample_layer_count=config.sample_layer_count,
            collapse_threshold=config.collapse_threshold,
            use_cross_domain_weighting=config.use_triangulation,
            multi_domain_bonus=config.use_triangulation,
        )

        # Load fingerprints (stub - would normally load from model)
        source_fingerprints = self._load_fingerprints(config.source_model_path)
        target_fingerprints = self._load_fingerprints(config.target_model_path)

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
        fingerprints = self._load_fingerprints(config.model_path)

        # Build profile to assess collapse
        invariant_ids, _ = InvariantLayerMapper._get_invariants(mapper_config)
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

    def _load_fingerprints(self, model_path: str) -> ModelFingerprints:
        """Load fingerprints for a model.

        This is a stub that creates empty fingerprints. In practice,
        this would load from a fingerprint service or compute them.
        """
        # For now, create stub fingerprints with reasonable defaults
        # A real implementation would use FingerprintService
        path = Path(model_path).expanduser().resolve()

        # Estimate layer count from model config if available
        layer_count = 32  # Default
        config_path = path / "config.json"
        if config_path.exists():
            import json
            try:
                with open(config_path) as f:
                    model_config = json.load(f)
                layer_count = model_config.get("num_hidden_layers", 32)
            except (json.JSONDecodeError, KeyError):
                pass

        return ModelFingerprints(
            model_id=str(path),
            layer_count=layer_count,
            fingerprints=[],  # Empty - no activations computed yet
        )

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

        return " | ".join(lines)

    def _recommend_action(self, report: Report) -> str:
        """Generate recommended action based on mapping results."""
        summary = report.summary

        if summary.alignment_quality < 0.3:
            return "Consider using CKA-based layer matching instead; invariant coverage is too sparse."

        if summary.source_collapsed_layers + summary.target_collapsed_layers > summary.mapped_layers * 0.3:
            return "High collapse rate detected. Try expanding anchor families or lowering collapse threshold."

        if summary.triangulation_quality == "none" and report.config.invariant_scope == InvariantScope.SEQUENCE_INVARIANTS:
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
            "medium": "Consider adding more anchor families to improve coverage.",
            "high": "High collapse risk. Use full SEQUENCE_INVARIANTS scope with all families enabled.",
            "critical": "Critical collapse risk. Model may have incompatible activation patterns. Consider alternative alignment methods.",
        }
        return actions.get(risk_level, "Unknown risk level.")

    @staticmethod
    def result_payload(result: LayerMappingResult) -> dict:
        """Convert LayerMappingResult to CLI/MCP payload."""
        report = result.report
        summary = report.summary

        return {
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
