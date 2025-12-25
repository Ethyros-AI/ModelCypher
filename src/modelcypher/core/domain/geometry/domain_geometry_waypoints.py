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
Domain Geometry Waypoints.

Uses validated geometric structures (spatial, social, temporal, moral)
as merge waypoints for domain-aware model merging.

Provides:
- Per-domain geometry scores for models
- Pre-merge geometry audit comparing source and target
- Post-merge geometry preservation validation
- Domain-aware alpha recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


class GeometryDomain(str, Enum):
    """Validated geometry domains from hypothesis testing."""

    SPATIAL = "spatial"  # 3D world model (Euclidean, gravity, occlusion)
    SOCIAL = "social"  # Power hierarchies, kinship, formality
    TEMPORAL = "temporal"  # Direction, duration, causality
    MORAL = "moral"  # Valence, agency, scope (Haidt foundations)


@dataclass(frozen=True)
class DomainGeometryScore:
    """Geometry score for a single domain."""

    domain: GeometryDomain
    manifold_score: float  # Domain-specific manifold score (SMS, SGS, TMS, MMS)
    axis_orthogonality: float  # Mean orthogonality of domain axes
    gradient_consistency: float  # Mean gradient correlation
    has_manifold: bool  # Whether manifold threshold is met
    anchors_probed: int  # Number of concept anchors used
    layer_analyzed: int  # Which layer was analyzed


@dataclass
class ModelGeometryProfile:
    """Complete geometry profile for a model across all domains."""

    model_path: str
    layer: int
    domain_scores: dict[GeometryDomain, DomainGeometryScore]
    computed_at: datetime
    total_anchors: int

    @property
    def mean_manifold_score(self) -> float:
        """Mean manifold score across all domains."""
        scores = [s.manifold_score for s in self.domain_scores.values()]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def strongest_domain(self) -> GeometryDomain | None:
        """Domain with highest manifold score."""
        if not self.domain_scores:
            return None
        return max(self.domain_scores.items(), key=lambda x: x[1].manifold_score)[0]

    @property
    def weakest_domain(self) -> GeometryDomain | None:
        """Domain with lowest manifold score."""
        if not self.domain_scores:
            return None
        return min(self.domain_scores.items(), key=lambda x: x[1].manifold_score)[0]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "modelPath": self.model_path,
            "layer": self.layer,
            "domainScores": {
                d.value: {
                    "manifoldScore": s.manifold_score,
                    "axisOrthogonality": s.axis_orthogonality,
                    "gradientConsistency": s.gradient_consistency,
                    "hasManifold": s.has_manifold,
                    "anchorsProbed": s.anchors_probed,
                }
                for d, s in self.domain_scores.items()
            },
            "computedAt": self.computed_at.isoformat(),
            "totalAnchors": self.total_anchors,
            "meanManifoldScore": self.mean_manifold_score,
            "strongestDomain": self.strongest_domain.value if self.strongest_domain else None,
            "weakestDomain": self.weakest_domain.value if self.weakest_domain else None,
        }


@dataclass
class GeometryConflictZone:
    """A detected conflict zone between source and target geometry."""

    domain: GeometryDomain
    source_score: float
    target_score: float
    delta: float
    severity: str  # "low", "medium", "high"
    recommendation: str


@dataclass
class PreMergeGeometryAudit:
    """Pre-merge audit comparing source and target geometry profiles."""

    source_profile: ModelGeometryProfile
    target_profile: ModelGeometryProfile
    conflict_zones: list[GeometryConflictZone]
    overall_compatibility: float  # 0-1 score
    recommended_alpha_by_domain: dict[GeometryDomain, float]
    audit_verdict: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sourceProfile": self.source_profile.to_dict(),
            "targetProfile": self.target_profile.to_dict(),
            "conflictZones": [
                {
                    "domain": c.domain.value,
                    "sourceScore": c.source_score,
                    "targetScore": c.target_score,
                    "delta": c.delta,
                    "severity": c.severity,
                    "recommendation": c.recommendation,
                }
                for c in self.conflict_zones
            ],
            "overallCompatibility": self.overall_compatibility,
            "recommendedAlphaByDomain": {
                d.value: a for d, a in self.recommended_alpha_by_domain.items()
            },
            "auditVerdict": self.audit_verdict,
        }


@dataclass
class PostMergeGeometryValidation:
    """Post-merge validation of geometry preservation."""

    source_profile: ModelGeometryProfile
    merged_profile: ModelGeometryProfile
    preservation_by_domain: dict[GeometryDomain, float]  # 0-1 preservation ratio
    degraded_domains: list[GeometryDomain]
    enhanced_domains: list[GeometryDomain]
    overall_preservation: float
    validation_status: str  # "preserved", "degraded", "enhanced"
    recommendations: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sourceProfile": self.source_profile.to_dict(),
            "mergedProfile": self.merged_profile.to_dict(),
            "preservationByDomain": {d.value: p for d, p in self.preservation_by_domain.items()},
            "degradedDomains": [d.value for d in self.degraded_domains],
            "enhancedDomains": [d.value for d in self.enhanced_domains],
            "overallPreservation": self.overall_preservation,
            "validationStatus": self.validation_status,
            "recommendations": self.recommendations,
        }


class DomainGeometryWaypointService:
    """
    Service for computing and using domain geometry as merge waypoints.

    Uses validated geometric structures (spatial, social, temporal, moral)
    to guide model merging with domain-aware alpha profiles.
    """

    def __init__(
        self,
        backend: "Backend",
        model_loader: "ModelLoaderPort",
    ) -> None:
        """Initialize with required dependencies.

        Args:
            backend: Backend for tensor operations (REQUIRED).
            model_loader: Model loader port for loading models (REQUIRED).
        """
        self._backend = backend
        self._model_loader = model_loader
        self._spatial_analyzer = None
        self._social_analyzer = None
        self._temporal_analyzer = None
        self._moral_analyzer = None

    def compute_profile(
        self,
        model_path: str,
        layer: int = -1,
        domains: list[GeometryDomain] | None = None,
    ) -> ModelGeometryProfile:
        """
        Compute complete geometry profile for a model.

        Args:
            model_path: Path to model directory
            layer: Layer to analyze (-1 for last)
            domains: Domains to analyze (default: all)

        Returns:
            ModelGeometryProfile with scores for each domain
        """
        if domains is None:
            domains = list(GeometryDomain)

        domain_scores: dict[GeometryDomain, DomainGeometryScore] = {}
        total_anchors = 0

        for domain in domains:
            try:
                score = self._compute_domain_score(model_path, domain, layer)
                domain_scores[domain] = score
                total_anchors += score.anchors_probed
            except Exception as e:
                logger.warning(f"Failed to compute {domain.value} geometry: {e}")

        return ModelGeometryProfile(
            model_path=model_path,
            layer=layer,
            domain_scores=domain_scores,
            computed_at=datetime.utcnow(),
            total_anchors=total_anchors,
        )

    def _compute_domain_score(
        self,
        model_path: str,
        domain: GeometryDomain,
        layer: int,
    ) -> DomainGeometryScore:
        """Compute geometry score for a specific domain."""
        if domain == GeometryDomain.SPATIAL:
            return self._compute_spatial_score(model_path, layer, self._backend)
        elif domain == GeometryDomain.SOCIAL:
            return self._compute_social_score(model_path, layer, self._backend)
        elif domain == GeometryDomain.TEMPORAL:
            return self._compute_temporal_score(model_path, layer, self._backend)
        elif domain == GeometryDomain.MORAL:
            return self._compute_moral_score(model_path, layer, self._backend)
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _compute_spatial_score(
        self,
        model_path: str,
        layer: int,
        backend: "Backend",
    ) -> DomainGeometryScore:
        """Compute spatial geometry score (Blind Physicist hypothesis)."""
        from modelcypher.core.domain.geometry.spatial_3d import (
            SPATIAL_PRIME_ATLAS,
            Spatial3DAnalyzer,
        )

        model, tokenizer = self._model_loader.load_model_for_training(model_path)

        # Extract activations for spatial probes
        activations = self._extract_activations(
            model,
            tokenizer,
            layer,
            [(p.name, p.prompt) for p in SPATIAL_PRIME_ATLAS],
            backend,
        )

        analyzer = Spatial3DAnalyzer(backend=backend)
        report = analyzer.full_analysis(activations)

        # Extract mean orthogonality from axis_orthogonality dict
        ortho_dict = report.euclidean_consistency.axis_orthogonality
        mean_ortho = sum(ortho_dict.values()) / len(ortho_dict) if ortho_dict else 0.0

        return DomainGeometryScore(
            domain=GeometryDomain.SPATIAL,
            manifold_score=report.world_model_score,
            axis_orthogonality=mean_ortho,
            gradient_consistency=report.euclidean_consistency.consistency_score,
            has_manifold=report.has_3d_world_model,
            anchors_probed=len(activations),
            layer_analyzed=layer,
        )

    def _compute_social_score(
        self,
        model_path: str,
        layer: int,
        backend: "Backend",
    ) -> DomainGeometryScore:
        """Compute social geometry score (Latent Sociologist hypothesis)."""
        from modelcypher.core.domain.agents.social_atlas import ALL_SOCIAL_PROBES
        from modelcypher.core.domain.geometry.social_geometry import (
            SocialGeometryAnalyzer,
        )

        model, tokenizer = self._model_loader.load_model_for_training(model_path)

        # Extract activations for social probes
        activations = self._extract_activations(
            model,
            tokenizer,
            layer,
            [(p.id, f"The word {p.name.lower()} represents") for p in ALL_SOCIAL_PROBES],
            backend,
        )

        analyzer = SocialGeometryAnalyzer(backend=backend)
        report = analyzer.full_analysis(activations)

        return DomainGeometryScore(
            domain=GeometryDomain.SOCIAL,
            manifold_score=report.social_manifold_score,
            axis_orthogonality=report.axis_orthogonality.mean_orthogonality,
            gradient_consistency=abs(report.gradient_consistency.power_correlation),
            has_manifold=report.has_social_manifold,
            anchors_probed=report.anchor_count,
            layer_analyzed=layer,
        )

    def _compute_temporal_score(
        self,
        model_path: str,
        layer: int,
        backend: "Backend",
    ) -> DomainGeometryScore:
        """Compute temporal geometry score (Latent Chronologist hypothesis)."""
        from modelcypher.core.domain.geometry.temporal_topology import (
            TemporalTopologyAnalyzer,
            extract_temporal_activations,
        )

        model, tokenizer = self._model_loader.load_model_for_training(model_path)
        activations = extract_temporal_activations(model, tokenizer, layer)

        analyzer = TemporalTopologyAnalyzer(activations)
        report = analyzer.analyze()

        return DomainGeometryScore(
            domain=GeometryDomain.TEMPORAL,
            manifold_score=report.temporal_manifold_score,
            axis_orthogonality=report.axis_orthogonality.mean_orthogonality,
            gradient_consistency=abs(report.gradient_consistency.direction_correlation),
            has_manifold=report.has_temporal_manifold,
            anchors_probed=report.anchors_probed,
            layer_analyzed=layer,
        )

    def _compute_moral_score(
        self,
        model_path: str,
        layer: int,
        backend: "Backend",
    ) -> DomainGeometryScore:
        """Compute moral geometry score (Latent Ethicist hypothesis)."""
        from modelcypher.core.domain.agents.moral_atlas import ALL_MORAL_PROBES
        from modelcypher.core.domain.geometry.moral_geometry import (
            MoralGeometryAnalyzer,
        )

        model, tokenizer = self._model_loader.load_model_for_training(model_path)

        # Extract activations for moral probes
        activations = self._extract_activations(
            model,
            tokenizer,
            layer,
            [(p.id, f"The word {p.name.lower()} represents") for p in ALL_MORAL_PROBES],
            backend,
        )

        analyzer = MoralGeometryAnalyzer(backend=backend)
        report = analyzer.full_analysis(activations, model_path=model_path, layer=layer)

        return DomainGeometryScore(
            domain=GeometryDomain.MORAL,
            manifold_score=report.moral_manifold_score,
            axis_orthogonality=report.axis_orthogonality.mean_orthogonality,
            gradient_consistency=abs(report.gradient_consistency.valence_correlation),
            has_manifold=report.has_moral_manifold,
            anchors_probed=report.anchors_probed,
            layer_analyzed=layer,
        )

    def _extract_activations(
        self,
        model,
        tokenizer,
        layer: int,
        probes: list[tuple[str, str]],  # (id, prompt)
        backend: "Backend",
    ) -> dict[str, np.ndarray]:
        """Extract activations for a list of probes."""
        activations = {}

        # Resolve model architecture
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_tokens = model.model.embed_tokens
            layers = model.model.layers
            norm = getattr(model.model, "norm", None)
        elif hasattr(model, "embed_tokens") and hasattr(model, "layers"):
            embed_tokens = model.embed_tokens
            layers = model.layers
            norm = getattr(model, "norm", None)
        else:
            raise ValueError("Could not resolve model architecture")

        num_layers = len(layers)
        target_layer = layer if layer >= 0 else num_layers - 1

        for concept_id, prompt in probes:
            try:
                tokens = tokenizer.encode(prompt)
                input_ids = backend.array([tokens])

                hidden = embed_tokens(input_ids)
                seq_len = input_ids.shape[1]
                mask = backend.create_causal_mask(seq_len, hidden.dtype)

                for i, layer_module in enumerate(layers):
                    try:
                        hidden = layer_module(hidden, mask=mask)
                    except TypeError:
                        try:
                            hidden = layer_module(hidden, mask)
                        except TypeError:
                            hidden = layer_module(hidden)
                    if i == target_layer:
                        break

                if norm is not None and target_layer == num_layers - 1:
                    hidden = norm(hidden)

                activation = backend.mean(hidden[0], axis=0)
                backend.eval(activation)
                activations[concept_id] = backend.to_numpy(activation)

            except Exception as e:
                logger.warning(f"Failed to extract activation for {concept_id}: {e}")

        return activations

    def pre_merge_audit(
        self,
        source_path: str,
        target_path: str,
        layer: int = -1,
    ) -> PreMergeGeometryAudit:
        """
        Perform pre-merge geometry audit comparing source and target.

        Identifies conflict zones and recommends domain-aware alpha values.

        Args:
            source_path: Path to source model
            target_path: Path to target model
            layer: Layer to analyze

        Returns:
            PreMergeGeometryAudit with conflict zones and recommendations
        """
        # Compute profiles for both models
        source_profile = self.compute_profile(source_path, layer)
        target_profile = self.compute_profile(target_path, layer)

        # Detect conflict zones
        conflict_zones: list[GeometryConflictZone] = []
        recommended_alpha: dict[GeometryDomain, float] = {}

        for domain in GeometryDomain:
            source_score = source_profile.domain_scores.get(domain)
            target_score = target_profile.domain_scores.get(domain)

            if source_score is None or target_score is None:
                continue

            delta = abs(source_score.manifold_score - target_score.manifold_score)

            # Determine severity
            if delta > 0.3:
                severity = "high"
            elif delta > 0.15:
                severity = "medium"
            else:
                severity = "low"

            # Compute recommended alpha
            # Higher score in source → lower alpha (preserve source)
            # Higher score in target → higher alpha (prefer target)
            if source_score.manifold_score > target_score.manifold_score:
                # Source is stronger → preserve source
                alpha = 0.3 + 0.3 * (1 - delta)  # 0.3-0.6
                recommendation = f"Source has stronger {domain.value} geometry - use lower alpha"
            else:
                # Target is stronger → prefer target
                alpha = 0.5 + 0.3 * delta  # 0.5-0.8
                recommendation = f"Target has stronger {domain.value} geometry - use higher alpha"

            recommended_alpha[domain] = alpha

            if severity != "low":
                conflict_zones.append(
                    GeometryConflictZone(
                        domain=domain,
                        source_score=source_score.manifold_score,
                        target_score=target_score.manifold_score,
                        delta=delta,
                        severity=severity,
                        recommendation=recommendation,
                    )
                )

        # Compute overall compatibility
        if recommended_alpha:
            alpha_variance = np.var(list(recommended_alpha.values()))
            overall_compatibility = 1.0 - min(1.0, alpha_variance * 4)
        else:
            overall_compatibility = 0.5

        # Determine verdict
        high_severity_count = sum(1 for c in conflict_zones if c.severity == "high")
        if high_severity_count >= 2:
            verdict = "CONFLICT - Multiple high-severity geometry mismatches detected"
        elif high_severity_count == 1:
            verdict = "CAUTION - High-severity geometry mismatch in one domain"
        elif conflict_zones:
            verdict = "COMPATIBLE - Minor geometry differences detected"
        else:
            verdict = "ALIGNED - Models have compatible geometry profiles"

        return PreMergeGeometryAudit(
            source_profile=source_profile,
            target_profile=target_profile,
            conflict_zones=conflict_zones,
            overall_compatibility=overall_compatibility,
            recommended_alpha_by_domain=recommended_alpha,
            audit_verdict=verdict,
        )

    def post_merge_validate(
        self,
        source_path: str,
        merged_path: str,
        layer: int = -1,
    ) -> PostMergeGeometryValidation:
        """
        Validate geometry preservation after merge.

        Compares source and merged model geometry to detect degradation.

        Args:
            source_path: Path to source model
            merged_path: Path to merged model
            layer: Layer to analyze

        Returns:
            PostMergeGeometryValidation with preservation metrics
        """
        # Compute profiles
        source_profile = self.compute_profile(source_path, layer)
        merged_profile = self.compute_profile(merged_path, layer)

        # Compute preservation by domain
        preservation_by_domain: dict[GeometryDomain, float] = {}
        degraded_domains: list[GeometryDomain] = []
        enhanced_domains: list[GeometryDomain] = []

        for domain in GeometryDomain:
            source_score = source_profile.domain_scores.get(domain)
            merged_score = merged_profile.domain_scores.get(domain)

            if source_score is None or merged_score is None:
                continue

            # Preservation ratio: merged / source (capped at 1.0 for preservation)
            if source_score.manifold_score > 0:
                ratio = merged_score.manifold_score / source_score.manifold_score
            else:
                ratio = 1.0 if merged_score.manifold_score >= 0 else 0.0

            preservation_by_domain[domain] = min(ratio, 1.0)

            # Detect degradation or enhancement
            delta = merged_score.manifold_score - source_score.manifold_score
            if delta < -0.1:  # Degraded by more than 0.1
                degraded_domains.append(domain)
            elif delta > 0.05:  # Enhanced by more than 0.05
                enhanced_domains.append(domain)

        # Overall preservation
        if preservation_by_domain:
            overall_preservation = sum(preservation_by_domain.values()) / len(
                preservation_by_domain
            )
        else:
            overall_preservation = 0.0

        # Determine status
        if degraded_domains and not enhanced_domains:
            status = "degraded"
        elif enhanced_domains and not degraded_domains:
            status = "enhanced"
        elif degraded_domains and enhanced_domains:
            status = "mixed"
        else:
            status = "preserved"

        # Generate recommendations
        recommendations: list[str] = []

        if degraded_domains:
            domains_str = ", ".join(d.value for d in degraded_domains)
            recommendations.append(
                f"Geometry degraded in {domains_str} - consider reducing alpha for these domains"
            )

        if overall_preservation < 0.8:
            recommendations.append(
                "Overall geometry preservation below 80% - re-merge with lower global alpha"
            )

        if not recommendations:
            recommendations.append("Geometry well-preserved - merge appears successful")

        return PostMergeGeometryValidation(
            source_profile=source_profile,
            merged_profile=merged_profile,
            preservation_by_domain=preservation_by_domain,
            degraded_domains=degraded_domains,
            enhanced_domains=enhanced_domains,
            overall_preservation=overall_preservation,
            validation_status=status,
            recommendations=recommendations,
        )

    def compute_domain_alpha_profile(
        self,
        audit: PreMergeGeometryAudit,
        base_alpha: float = 0.5,
        strength: float = 0.5,
    ) -> dict[GeometryDomain, float]:
        """
        Compute domain-aware alpha profile based on geometry audit.

        Args:
            audit: Pre-merge geometry audit result
            base_alpha: Base alpha to adjust from
            strength: How much to apply domain adjustments (0-1)

        Returns:
            Dict mapping domain to recommended alpha
        """
        alpha_profile: dict[GeometryDomain, float] = {}

        for domain, recommended in audit.recommended_alpha_by_domain.items():
            # Blend recommended alpha with base alpha based on strength
            adjusted = base_alpha * (1 - strength) + recommended * strength
            alpha_profile[domain] = max(0.1, min(0.95, adjusted))

        return alpha_profile


# Export types
__all__ = [
    "GeometryDomain",
    "DomainGeometryScore",
    "ModelGeometryProfile",
    "GeometryConflictZone",
    "PreMergeGeometryAudit",
    "PostMergeGeometryValidation",
    "DomainGeometryWaypointService",
]
