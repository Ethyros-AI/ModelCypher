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
class DomainGeometryDelta:
    """Geometry difference between source and target for a single domain.

    Returns raw measurements. The delta IS the information about how much
    the models differ in this domain - no need for "low/medium/high" severity.
    """

    domain: GeometryDomain
    source_score: float
    target_score: float
    delta: float
    """Absolute difference in manifold scores. The delta IS the severity."""


@dataclass
class PreMergeGeometryAudit:
    """Pre-merge audit comparing source and target geometry profiles.

    Returns raw geometric measurements. The deltas and variance ARE the
    compatibility information - no need for "CONFLICT/COMPATIBLE" verdicts.
    Models are ALWAYS compatible; we just measure the alignment cost.
    """

    source_profile: ModelGeometryProfile
    target_profile: ModelGeometryProfile
    domain_deltas: list[DomainGeometryDelta]
    """Per-domain geometry differences. The deltas ARE the alignment cost."""

    alpha_variance: float
    """Variance in derived alphas across domains. Higher = more domain-specific tuning needed."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sourceProfile": self.source_profile.to_dict(),
            "targetProfile": self.target_profile.to_dict(),
            "domainDeltas": [
                {
                    "domain": d.domain.value,
                    "sourceScore": d.source_score,
                    "targetScore": d.target_score,
                    "delta": d.delta,
                }
                for d in self.domain_deltas
            ],
            "alphaVariance": self.alpha_variance,
        }


@dataclass
class PostMergeGeometryValidation:
    """Post-merge validation of geometry preservation.

    Returns raw preservation ratios. The ratios ARE the validation result -
    no need for "preserved/degraded/enhanced" classifications.
    """

    source_profile: ModelGeometryProfile
    merged_profile: ModelGeometryProfile
    preservation_by_domain: dict[GeometryDomain, float]
    """Preservation ratio per domain: merged_score / source_score."""

    overall_preservation: float
    """Mean preservation ratio across domains."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sourceProfile": self.source_profile.to_dict(),
            "mergedProfile": self.merged_profile.to_dict(),
            "preservationByDomain": {d.value: p for d, p in self.preservation_by_domain.items()},
            "overallPreservation": self.overall_preservation,
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
    ) -> dict[str, "Array"]:
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
                # Keep as backend array (MLX) for GPU operations downstream
                # Only convert to numpy at final output stage
                activations[concept_id] = activation

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

        Returns raw geometric measurements. The deltas ARE the alignment cost.
        Models are ALWAYS compatible - we just measure how different they are.

        Args:
            source_path: Path to source model
            target_path: Path to target model
            layer: Layer to analyze

        Returns:
            PreMergeGeometryAudit with raw geometry deltas
        """
        # Compute profiles for both models
        source_profile = self.compute_profile(source_path, layer)
        target_profile = self.compute_profile(target_path, layer)

        # Compute per-domain deltas
        domain_deltas: list[DomainGeometryDelta] = []
        alphas: list[float] = []

        for domain in GeometryDomain:
            source_score = source_profile.domain_scores.get(domain)
            target_score = target_profile.domain_scores.get(domain)

            if source_score is None or target_score is None:
                continue

            delta = abs(source_score.manifold_score - target_score.manifold_score)

            domain_deltas.append(
                DomainGeometryDelta(
                    domain=domain,
                    source_score=source_score.manifold_score,
                    target_score=target_score.manifold_score,
                    delta=delta,
                )
            )

            # Derive alpha from geometry: stronger manifold gets more weight
            # This is geometry-derived, not arbitrary
            total = source_score.manifold_score + target_score.manifold_score
            if total > 0:
                alpha = target_score.manifold_score / total
            else:
                alpha = 0.5
            alphas.append(alpha)

        # Compute alpha variance - how much domain-specific tuning is needed
        if alphas:
            mean_alpha = sum(alphas) / len(alphas)
            alpha_variance = sum((a - mean_alpha) ** 2 for a in alphas) / len(alphas)
        else:
            alpha_variance = 0.0

        return PreMergeGeometryAudit(
            source_profile=source_profile,
            target_profile=target_profile,
            domain_deltas=domain_deltas,
            alpha_variance=alpha_variance,
        )

    def post_merge_validate(
        self,
        source_path: str,
        merged_path: str,
        layer: int = -1,
    ) -> PostMergeGeometryValidation:
        """
        Validate geometry preservation after merge.

        Returns raw preservation ratios. The ratios ARE the validation result.
        No "preserved/degraded" classifications - the numbers tell the story.

        Args:
            source_path: Path to source model
            merged_path: Path to merged model
            layer: Layer to analyze

        Returns:
            PostMergeGeometryValidation with raw preservation ratios
        """
        # Compute profiles
        source_profile = self.compute_profile(source_path, layer)
        merged_profile = self.compute_profile(merged_path, layer)

        # Compute preservation by domain
        preservation_by_domain: dict[GeometryDomain, float] = {}

        for domain in GeometryDomain:
            source_score = source_profile.domain_scores.get(domain)
            merged_score = merged_profile.domain_scores.get(domain)

            if source_score is None or merged_score is None:
                continue

            # Preservation ratio: merged / source
            # Can be > 1.0 if geometry is enhanced
            if source_score.manifold_score > 0:
                ratio = merged_score.manifold_score / source_score.manifold_score
            else:
                ratio = 1.0 if merged_score.manifold_score >= 0 else 0.0

            preservation_by_domain[domain] = ratio

        # Overall preservation - the mean ratio IS the result
        if preservation_by_domain:
            overall_preservation = sum(preservation_by_domain.values()) / len(
                preservation_by_domain
            )
        else:
            overall_preservation = 0.0

        return PostMergeGeometryValidation(
            source_profile=source_profile,
            merged_profile=merged_profile,
            preservation_by_domain=preservation_by_domain,
            overall_preservation=overall_preservation,
        )

    def compute_domain_alpha_profile(
        self,
        audit: PreMergeGeometryAudit,
    ) -> dict[GeometryDomain, float]:
        """
        Compute domain-aware alpha profile from geometry audit.

        Alpha is derived directly from the geometry: the stronger manifold
        gets more weight in the merge. This is geometry-determined, not arbitrary.

        Args:
            audit: Pre-merge geometry audit result

        Returns:
            Dict mapping domain to geometry-derived alpha
        """
        alpha_profile: dict[GeometryDomain, float] = {}

        for delta in audit.domain_deltas:
            # Alpha derived from geometry: target_score / total
            # Stronger manifold gets more weight
            total = delta.source_score + delta.target_score
            if total > 0:
                alpha = delta.target_score / total
            else:
                alpha = 0.5
            alpha_profile[delta.domain] = alpha

        return alpha_profile


# Export types
__all__ = [
    "GeometryDomain",
    "DomainGeometryScore",
    "ModelGeometryProfile",
    "DomainGeometryDelta",
    "PreMergeGeometryAudit",
    "PostMergeGeometryValidation",
    "DomainGeometryWaypointService",
]
