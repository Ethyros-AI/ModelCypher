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
- Domain strength ratios (measurement-only)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

# Import from canonical location - AtlasDomain is the single source of truth
from modelcypher.core.domain.domains import AtlasDomain

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class DomainGeometryScore:
    """Geometry score for a single domain."""

    domain: AtlasDomain
    manifold_score: float  # Domain-specific manifold score (SMS, SGS, TMS, MMS)
    axis_orthogonality: float  # Mean orthogonality of domain axes
    gradient_consistency: float  # Mean gradient correlation
    anchors_probed: int  # Number of concept anchors used
    layer_analyzed: int  # Which layer was analyzed


@dataclass
class ModelGeometryProfile:
    """Complete geometry profile for a model across all domains."""

    model_path: str
    layer: int
    domain_scores: dict[AtlasDomain, DomainGeometryScore]
    computed_at: datetime
    total_anchors: int

    @property
    def mean_manifold_score(self) -> float:
        """Mean manifold score across all domains."""
        scores = [s.manifold_score for s in self.domain_scores.values()]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def strongest_domain(self) -> AtlasDomain | None:
        """Domain with highest manifold score."""
        if not self.domain_scores:
            return None
        return max(self.domain_scores.items(), key=lambda x: x[1].manifold_score)[0]

    @property
    def weakest_domain(self) -> AtlasDomain | None:
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
                    "anchorsProbed": s.anchors_probed,
                }
                for d, s in self.domain_scores.items()
            },
            "computedAt": self.computed_at.isoformat(),
            "totalAnchors": self.total_anchors,
            "meanManifoldScore": self.mean_manifold_score,
        }


@dataclass
class DomainGeometryDelta:
    """Geometry difference between source and target for a single domain.

    Attributes
    ----------
    domain : AtlasDomain
        The geometry domain being compared.
    source_score : float
        Manifold score from source model.
    target_score : float
        Manifold score from target model.
    delta : float
        Absolute difference |source - target|.
    """

    domain: AtlasDomain
    source_score: float
    target_score: float
    delta: float


@dataclass
class PreMergeGeometryAudit:
    """Pre-merge audit comparing source and target geometry profiles.

    Attributes
    ----------
    source_profile : ModelGeometryProfile
        Geometry profile of source model.
    target_profile : ModelGeometryProfile
        Geometry profile of target model.
    domain_deltas : list[DomainGeometryDelta]
        Per-domain geometry differences.
    strength_ratio_variance : float
        Variance in geometry-derived strength ratios across domains.
    """

    source_profile: ModelGeometryProfile
    target_profile: ModelGeometryProfile
    domain_deltas: list[DomainGeometryDelta]
    strength_ratio_variance: float

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
            "strengthRatioVariance": self.strength_ratio_variance,
        }


@dataclass
class PostMergeGeometryValidation:
    """Post-merge validation of geometry preservation.

    Attributes
    ----------
    source_profile : ModelGeometryProfile
        Geometry profile of source model before merge.
    merged_profile : ModelGeometryProfile
        Geometry profile of merged model.
    preservation_by_domain : dict[AtlasDomain, float]
        Preservation ratio per domain: merged_score / source_score.
    overall_preservation : float
        Mean preservation ratio across domains.
    """

    source_profile: ModelGeometryProfile
    merged_profile: ModelGeometryProfile
    preservation_by_domain: dict[AtlasDomain, float]
    overall_preservation: float

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
    to report domain strength ratios for merge planning.
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
        domains: list[AtlasDomain] | None = None,
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
            domains = list(AtlasDomain)

        domain_scores: dict[AtlasDomain, DomainGeometryScore] = {}
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
        domain: AtlasDomain,
        layer: int,
    ) -> DomainGeometryScore:
        """Compute geometry score for a specific domain."""
        if domain == AtlasDomain.SPATIAL:
            return self._compute_spatial_score(model_path, layer, self._backend)
        elif domain == AtlasDomain.RELATIONAL:
            return self._compute_social_score(model_path, layer, self._backend)
        elif domain == AtlasDomain.TEMPORAL:
            return self._compute_temporal_score(model_path, layer, self._backend)
        elif domain == AtlasDomain.MORAL:
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
        from modelcypher.core.domain.geometry.atlas_registry import get_spatial_concepts
        from modelcypher.core.domain.geometry.spatial_3d import Spatial3DAnalyzer

        concepts = list(get_spatial_concepts())
        if not concepts:
            raise ValueError(
                "No spatial concepts registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before computing spatial geometry."
            )
        model, tokenizer = self._model_loader.load_model_for_training(model_path)

        # Extract activations for spatial probes
        activations = self._extract_activations(
            model,
            tokenizer,
            layer,
            [(p.id, p.prompt) for p in concepts],
            backend,
        )

        analyzer = Spatial3DAnalyzer(backend=backend)
        report = analyzer.full_analysis(activations)

        # Extract mean orthogonality from axis_orthogonality dict
        ortho_dict = report.axis_orthogonality
        mean_ortho = sum(ortho_dict.values()) / len(ortho_dict) if ortho_dict else 0.0
        gravity_consistency = (
            abs(report.gravity_gradient.mass_correlation)
            if report.gravity_gradient.gravity_axis_detected
            else 0.0
        )

        return DomainGeometryScore(
            domain=AtlasDomain.SPATIAL,
            manifold_score=report.world_model_score,
            axis_orthogonality=mean_ortho,
            gradient_consistency=gravity_consistency,
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
        from modelcypher.core.domain.geometry.atlas_registry import get_social_concepts
        from modelcypher.core.domain.geometry.social_geometry import (
            SocialGeometryAnalyzer,
        )

        concepts = list(get_social_concepts())
        if not concepts:
            raise ValueError(
                "No social concepts registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before computing social geometry."
            )
        model, tokenizer = self._model_loader.load_model_for_training(model_path)

        # Extract activations for social probes
        activations = self._extract_activations(
            model,
            tokenizer,
            layer,
            [(p.id, p.prompt) for p in concepts],
            backend,
        )

        analyzer = SocialGeometryAnalyzer(backend=backend)
        report = analyzer.full_analysis(activations)

        return DomainGeometryScore(
            domain=AtlasDomain.RELATIONAL,
            manifold_score=report.social_manifold_score,
            axis_orthogonality=report.axis_orthogonality.mean_orthogonality,
            gradient_consistency=abs(report.gradient_consistency.power_correlation),
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
            domain=AtlasDomain.TEMPORAL,
            manifold_score=report.temporal_manifold_score,
            axis_orthogonality=report.axis_orthogonality.mean_orthogonality,
            gradient_consistency=abs(report.gradient_consistency.direction_correlation),
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
        from modelcypher.core.domain.geometry.atlas_registry import get_moral_concepts
        from modelcypher.core.domain.geometry.moral_geometry import (
            MoralGeometryAnalyzer,
        )

        concepts = list(get_moral_concepts())
        if not concepts:
            raise ValueError(
                "No moral concepts registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before computing moral geometry."
            )
        model, tokenizer = self._model_loader.load_model_for_training(model_path)

        # Extract activations for moral probes
        activations = self._extract_activations(
            model,
            tokenizer,
            layer,
            [(p.id, p.prompt) for p in concepts],
            backend,
        )

        analyzer = MoralGeometryAnalyzer(backend=backend)
        report = analyzer.full_analysis(activations, model_path=model_path, layer=layer)

        return DomainGeometryScore(
            domain=AtlasDomain.MORAL,
            manifold_score=report.moral_manifold_score,
            axis_orthogonality=report.axis_orthogonality.mean_orthogonality,
            gradient_consistency=abs(report.gradient_consistency.valence_correlation),
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
        """Compare geometry profiles of source and target models.

        Parameters
        ----------
        source_path : str
            Path to source model directory.
        target_path : str
            Path to target model directory.
        layer : int, default -1
            Layer to analyze (-1 for last).

        Returns
        -------
        PreMergeGeometryAudit
            Per-domain geometry deltas and strength ratio variance.
        """
        # Compute profiles for both models
        source_profile = self.compute_profile(source_path, layer)
        target_profile = self.compute_profile(target_path, layer)

        # Compute per-domain deltas
        domain_deltas: list[DomainGeometryDelta] = []
        strength_ratios: list[float] = []

        for domain in AtlasDomain:
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

            total = source_score.manifold_score + target_score.manifold_score
            if total > 0:
                strength_ratio = target_score.manifold_score / total
                strength_ratios.append(strength_ratio)

        # Compute strength ratio variance - how different domains are
        if strength_ratios:
            mean_ratio = sum(strength_ratios) / len(strength_ratios)
            strength_ratio_variance = sum(
                (a - mean_ratio) ** 2 for a in strength_ratios
            ) / len(strength_ratios)
        else:
            strength_ratio_variance = 0.0

        return PreMergeGeometryAudit(
            source_profile=source_profile,
            target_profile=target_profile,
            domain_deltas=domain_deltas,
            strength_ratio_variance=strength_ratio_variance,
        )

    def post_merge_validate(
        self,
        source_path: str,
        merged_path: str,
        layer: int = -1,
    ) -> PostMergeGeometryValidation:
        """Measure geometry preservation after merge.

        Parameters
        ----------
        source_path : str
            Path to source model directory.
        merged_path : str
            Path to merged model directory.
        layer : int, default -1
            Layer to analyze (-1 for last).

        Returns
        -------
        PostMergeGeometryValidation
            Per-domain preservation ratios (merged/source).
        """
        # Compute profiles
        source_profile = self.compute_profile(source_path, layer)
        merged_profile = self.compute_profile(merged_path, layer)

        # Compute preservation by domain
        preservation_by_domain: dict[AtlasDomain, float] = {}

        for domain in AtlasDomain:
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

        # Overall preservation - mean ratio across domains
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

    def compute_domain_strength_profile(
        self,
        audit: PreMergeGeometryAudit,
    ) -> dict[AtlasDomain, float]:
        """
        Compute domain strength ratios from geometry audit.

        Ratio is derived directly from the geometry: target_score / (source + target).
        This is measurement-only; callers decide how to use it.

        Args:
            audit: Pre-merge geometry audit result

        Returns:
            Dict mapping domain to geometry-derived strength ratio
        """
        strength_profile: dict[AtlasDomain, float] = {}

        for delta in audit.domain_deltas:
            total = delta.source_score + delta.target_score
            if total > 0:
                strength_profile[delta.domain] = delta.target_score / total

        return strength_profile


# Export types
__all__ = [
    "AtlasDomain",
    "DomainGeometryScore",
    "ModelGeometryProfile",
    "DomainGeometryDelta",
    "PreMergeGeometryAudit",
    "PostMergeGeometryValidation",
    "DomainGeometryWaypointService",
]
