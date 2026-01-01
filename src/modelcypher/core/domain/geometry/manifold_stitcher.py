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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol, enum_key
from modelcypher.core.domain.geometry.atlas_registry import get_atlas_probes

__all__ = [
    # Configuration
    "Thresholds",
    "StitchingConstants",
    "TriangulatedProbingConfig",
    # Enums
    "ProbeSpace",
    # Core dataclasses
    "DimensionCorrelation",
    "LayerConfidence",
    "IntersectionMap",
    "ActivatedDimension",
    "ActivationFingerprint",
    "SparseActivationVector",
    "ModelFingerprints",
    # Continuous fingerprinting
    "ContinuousFingerprint",
    "ContinuousCorrelationResult",
    "ContinuousModelFingerprints",
    "LayerContinuousProfile",
    # Triangulated probing
    "TriangulatedProbeResult",
    "TriangulatedProbeBuilder",
    # Alignment and validation
    "AlignmentCluster",
    "LayerDelta",
    "ValidationResult",
    # Main class
    "ManifoldStitcher",
    "output_layer_marker",
]


class Thresholds:
    strong_correlation: ClassVar[float] = 0.75
    moderate_correlation: ClassVar[float] = 0.5
    strong_weight: ClassVar[float] = 1.0
    moderate_weight: ClassVar[float] = 1.0
    weak_weight: ClassVar[float] = 1.0


@dataclass(frozen=True)
class DimensionCorrelation:
    source_dim: int
    target_dim: int
    correlation: float

    @property
    def is_strong_correlation(self) -> bool:
        return self.correlation > Thresholds.strong_correlation

    @property
    def is_moderate_correlation(self) -> bool:
        return Thresholds.moderate_correlation < self.correlation <= Thresholds.strong_correlation

    @property
    def is_weak_correlation(self) -> bool:
        return self.correlation <= Thresholds.moderate_correlation


@dataclass(frozen=True)
class LayerConfidence:
    layer: int
    strong_correlations: int
    moderate_correlations: int
    weak_correlations: int
    confidence: float = field(init=False)

    def __post_init__(self) -> None:
        total = self.strong_correlations + self.moderate_correlations + self.weak_correlations
        if total > 0:
            weighted = (
                float(self.strong_correlations) * Thresholds.strong_weight
                + float(self.moderate_correlations) * Thresholds.moderate_weight
                + float(self.weak_correlations) * Thresholds.weak_weight
            )
            value = weighted / float(total)
        else:
            value = 0.0
        object.__setattr__(self, "confidence", value)

    @property
    def total_correlations(self) -> int:
        return self.strong_correlations + self.moderate_correlations + self.weak_correlations


@dataclass(frozen=True)
class IntersectionMap:
    source_model: str
    target_model: str
    dimension_correlations: dict[int, list[DimensionCorrelation]]
    overall_correlation: float
    aligned_dimension_count: int
    total_source_dims: int
    total_target_dims: int
    layer_confidences: list[LayerConfidence]


class ProbeSpace(str, Enum):
    prelogits_hidden = "prelogits-hidden"
    output_logits = "output-logits"


output_layer_marker = -1


import logging
import math
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

# Keep runtime imports local to avoid cross-domain cycles.

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _ensure_proper_rotation(
    u: "Array",
    vt: "Array",
    omega: "Array",
    backend: "Backend",
) -> "Array":
    """Ensure rotation matrix has det(R) = +1 (proper rotation, not reflection).

    For SVD-based Procrustes: R = U @ V^T
    If det(R) < 0, we have a reflection. Fix by flipping the last column of U.

    This follows the standard approach from Schönemann (1966) and is used in
    generalized_procrustes.py for consistency.

    Args:
        u: Left singular vectors from SVD
        vt: Right singular vectors (transposed) from SVD
        omega: Current rotation matrix (U @ Vt)
        backend: Backend for array operations

    Returns:
        Proper rotation matrix with det(R) = +1
    """
    try:
        # Compute determinant using backend
        det_arr = backend.det(omega)
        backend.eval(det_arr)
        det_val = float(backend.to_numpy(det_arr).item())

        if det_val < 0:
            # Flip last column of U to change sign of determinant
            u_fixed = backend.concatenate([u[:, :-1], -u[:, -1:]], axis=1)
            omega = backend.matmul(u_fixed, vt)
            logger.debug("Fixed reflection (det=%.3f) to proper rotation", det_val)

        return omega

    except Exception as e:
        # SVD or det computation failed - return original omega with warning
        logger.warning("Could not compute determinant for sign correction: %s", e)
        return omega


@dataclass
class ContinuousFingerprint:
    """
    Continuous activation fingerprint preserving magnitude information.
    """

    prime_id: str
    prime_text: str

    # Layer -> Full activation vector
    activation_vectors: dict[int, list[float]]

    # Layer -> L2 Magnitude
    magnitudes: dict[int, float]

    # Layer -> Entropy (0-1)
    entropies: dict[int, float]

    # Layer -> Sparsity (0-1)
    sparsities: dict[int, float]

    @staticmethod
    def from_activations(
        prime_id: str,
        prime_text: str,
        layer_activations: dict[int, list[float]],
        backend: "Backend | None" = None,
    ) -> "ContinuousFingerprint":
        b = backend or get_default_backend()
        magnitudes = {}
        entropies = {}
        sparsities = {}

        for layer, activations in layer_activations.items():
            arr = b.array(activations)
            magnitudes[layer] = float(b.norm(arr))

            logits = arr
            max_val = b.max(logits)
            exp_acts = b.exp(logits - max_val)
            probs = exp_acts / (b.sum(exp_acts) + 1e-10)

            log_probs = b.log(probs + 1e-10)
            entropy = -float(b.to_numpy(b.sum(probs * log_probs)).item())
            max_entropy = math.log(max(len(activations), 1))
            entropies[layer] = min(max(entropy / max_entropy, 0.0), 1.0) if max_entropy > 0 else 0.0

            abs_acts = b.abs(arr)
            threshold = 0.01 * float(b.to_numpy(b.max(abs_acts)).item())
            near_zero = float(b.to_numpy(b.sum(abs_acts < threshold)).item())
            sparsities[layer] = near_zero / max(len(activations), 1)

        return ContinuousFingerprint(
            prime_id, prime_text, layer_activations, magnitudes, entropies, sparsities
        )


@dataclass(frozen=True)
class StitchingConstants:
    epsilon: float = 1e-8
    similarity_weight: float = 0.25
    cosine_weight: float = 0.25
    magnitude_weight: float = 0.25
    entropy_weight: float = 0.25
    relationship_bonus: float = 0.0
    cross_domain_multiplier: float = 1.0


@dataclass
class ContinuousCorrelationResult:
    cka: float
    cosine_similarity: float
    magnitude_ratio: float
    entropy_delta: float

    @property
    def alignment_score(self) -> float:
        # Note: For 1D vectors (single prime activations), Linear CKA is equivalent to squared cosine similarity.
        # CKA(x, y) = <x, y>^2 / (||x||^2 ||y||^2) = cosine_sim(x, y)^2

        cka_score = self.cka if self.cosine_similarity >= 0 else 0.0

        # Weighted combination of geometric invariants
        return (
            StitchingConstants.similarity_weight * cka_score
            + StitchingConstants.cosine_weight * max(0.0, self.cosine_similarity)
            + StitchingConstants.magnitude_weight
            * (1.0 - min(abs(self.magnitude_ratio - 1.0), 1.0))
            + StitchingConstants.entropy_weight * (1.0 - min(abs(self.entropy_delta), 1.0))
        )


@dataclass
class ContinuousModelFingerprints:
    """
    Collection of continuous fingerprints for a model.
    """

    model_id: str
    hidden_dim: int
    layer_count: int
    fingerprints: list[ContinuousFingerprint]

    @property
    def mean_entropy(self) -> float:
        vals = [e for fp in self.fingerprints for e in fp.entropies.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_sparsity(self) -> float:
        vals = [s for fp in self.fingerprints for s in fp.sparsities.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def from_model_fingerprints(
        source: "ModelFingerprints",
    ) -> "ContinuousModelFingerprints" | None:
        if not hasattr(source, "activation_vectors") or not source.activation_vectors:
            return None

        fingerprints_by_prime: dict[str, dict[int, list[float]]] = {}
        for key, vec in source.activation_vectors.items():
            if "_layer" not in key:
                continue
            idx = key.rfind("_layer")
            try:
                layer = int(key[idx + 6 :])
                prime_id = key[:idx]
                if prime_id not in fingerprints_by_prime:
                    fingerprints_by_prime[prime_id] = {}
                fingerprints_by_prime[prime_id][layer] = vec
            except ValueError:
                continue

        prime_texts = {fp.prime_id: fp.prime_text for fp in source.fingerprints}
        continuous_fps = [
            ContinuousFingerprint.from_activations(pid, prime_texts.get(pid, pid), layers)
            for pid, layers in fingerprints_by_prime.items()
        ]
        return ContinuousModelFingerprints(
            source.model_id, source.hidden_dim, source.layer_count, continuous_fps
        )

    def get_layer_profile(self, layer: int) -> "LayerContinuousProfile" | None:
        """Get aggregated profile for a specific layer."""
        layer_entropies = []
        layer_sparsities = []
        layer_magnitudes = []

        for fp in self.fingerprints:
            if layer in fp.entropies:
                layer_entropies.append(fp.entropies[layer])
            if layer in fp.sparsities:
                layer_sparsities.append(fp.sparsities[layer])
            if layer in fp.magnitudes:
                layer_magnitudes.append(fp.magnitudes[layer])

        if not layer_entropies:
            return None

        return LayerContinuousProfile(
            layer_index=layer,
            mean_entropy=sum(layer_entropies) / len(layer_entropies),
            mean_sparsity=sum(layer_sparsities) / len(layer_sparsities)
            if layer_sparsities
            else 0.0,
            mean_magnitude=sum(layer_magnitudes) / len(layer_magnitudes)
            if layer_magnitudes
            else 0.0,
            probe_count=len(layer_entropies),
            entropy_std=_compute_std(layer_entropies),
            sparsity_std=_compute_std(layer_sparsities) if layer_sparsities else 0.0,
        )

    def get_all_layer_profiles(self) -> dict[int, "LayerContinuousProfile"]:
        """Get profiles for all layers."""
        profiles = {}
        for layer in range(self.layer_count):
            profile = self.get_layer_profile(layer)
            if profile:
                profiles[layer] = profile
        return profiles


@dataclass(frozen=True)
class LayerContinuousProfile:
    """Aggregated continuous profile for a single layer."""

    layer_index: int
    mean_entropy: float
    mean_sparsity: float
    mean_magnitude: float
    probe_count: int
    entropy_std: float = 0.0
    sparsity_std: float = 0.0

    @property
    def is_collapsed(self) -> bool:
        """Layer is considered collapsed if very low entropy or very high sparsity."""
        return self.mean_entropy < 0.1 or self.mean_sparsity > 0.95

    @property
    def confidence(self) -> float:
        """Confidence based on entropy and probe count."""
        # More probes and moderate entropy = higher confidence
        probe_factor = min(self.probe_count / 20.0, 1.0)
        entropy_factor = 1.0 - abs(self.mean_entropy - 0.5) * 2  # Peak at 0.5
        return probe_factor * max(0.0, entropy_factor)


@dataclass(frozen=True)
class TriangulatedProbeResult:
    """Result of triangulated probing across multiple domains."""

    probe_id: str
    primary_domain: str
    activation_score: float
    cross_domain_score: float
    triangulation_multiplier: float
    domains_detected: list[str]
    layer_index: int

    @property
    def triangulated_score(self) -> float:
        """Combined score with triangulation boost."""
        return self.activation_score * self.triangulation_multiplier


@dataclass
class TriangulatedProbingConfig:
    """Configuration for triangulated probing."""

    include_sequence_invariants: bool = True
    include_metaphor_invariants: bool = True
    include_conceptual_genealogy: bool = True
    triangulation_threshold: float = 0.0
    cross_domain_bonus: float = 0.0
    max_domains_for_full_bonus: int = 1


_SOURCE_SEMANTIC_PRIME = "semantic_prime"
_SOURCE_SEQUENCE_INVARIANT = "sequence_invariant"
_SOURCE_COMPUTATIONAL_GATE = "computational_gate"
_SOURCE_METAPHOR_INVARIANT = "metaphor_invariant"
_SOURCE_CONCEPTUAL_GENEALOGY = "conceptual_genealogy"
_SOURCE_MORAL_CONCEPT = "moral_concept"
_SOURCE_SOCIAL_CONCEPT = "social_concept"


class TriangulatedProbeBuilder:
    """
    Builds triangulated probe sets for enhanced fingerprinting.

    Uses atlas registry probes instead of hardcoded probe lists. Outer layers
    register probe inventories so geometry can remain decoupled from agents.
    """

    @staticmethod
    def build_triangulated_probes(
        config: TriangulatedProbingConfig | None = None,
    ) -> list[AtlasProbeProtocol]:
        """
        Build probe set from the atlas registry.

        Returns list of AtlasProbe objects with:
        - probe_id: Unique identifier
        - support_texts: Probe texts for embedding
        - source: Atlas source (SEMANTIC_PRIME, SEQUENCE_INVARIANT, etc.)
        - domain: Triangulation domain
        - cross_domain_weight: Weight for cross-domain detection
        """
        if config is None:
            config = TriangulatedProbingConfig()

        sources: set[str] = set()

        # Semantic primes always included
        sources.add(_SOURCE_SEMANTIC_PRIME)

        # Add sequence invariants if enabled
        if config.include_sequence_invariants:
            sources.add(_SOURCE_SEQUENCE_INVARIANT)

        # Add other sources based on config
        if config.include_metaphor_invariants:
            sources.add(_SOURCE_METAPHOR_INVARIANT)

        if config.include_conceptual_genealogy:
            # Conceptual genealogy probes plus moral/social foundations for lineage anchoring
            sources.add(_SOURCE_CONCEPTUAL_GENEALOGY)
            sources.add(_SOURCE_MORAL_CONCEPT)
            sources.add(_SOURCE_SOCIAL_CONCEPT)

        # Always include computational gates for cross-domain triangulation
        sources.add(_SOURCE_COMPUTATIONAL_GATE)

        probes = list(get_atlas_probes())
        if not probes:
            raise ValueError(
                "No atlas probes registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before building triangulated probes."
            )
        source_keys = {enum_key(source) for source in sources}
        return [probe for probe in probes if enum_key(probe.source) in source_keys]

    @staticmethod
    def build_all_probes() -> list[AtlasProbeProtocol]:
        """Get all probes for full triangulation."""
        probes = list(get_atlas_probes())
        if not probes:
            raise ValueError(
                "No atlas probes registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before building triangulated probes."
            )
        return probes

    @staticmethod
    def build_probes_for_sources(sources: set) -> list[AtlasProbeProtocol]:
        """Get probes from specific atlas sources."""
        probes = list(get_atlas_probes())
        if not probes:
            raise ValueError(
                "No atlas probes registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before building triangulated probes."
            )
        source_keys = {enum_key(source) for source in sources}
        return [probe for probe in probes if enum_key(probe.source) in source_keys]

    @staticmethod
    def compute_triangulation_score(
        activations_by_domain: dict[str, float],
        config: TriangulatedProbingConfig | None = None,
    ) -> tuple[float, float]:
        """
        Compute triangulation score from multi-domain activations.

        Args:
            activations_by_domain: Activation scores keyed by domain
            config: Configuration

        Returns:
            Tuple of (base_score, triangulation_multiplier)
        """
        if config is None:
            config = TriangulatedProbingConfig()

        if not activations_by_domain:
            return 0.0, 1.0

        # Base score is mean of significant activations
        significant = [
            v for v in activations_by_domain.values() if v >= config.triangulation_threshold
        ]
        if not significant:
            return 0.0, 1.0

        base_score = sum(significant) / len(significant)

        # Triangulation bonus based on domain count
        domain_count = len(significant)
        bonus_domains = min(domain_count - 1, config.max_domains_for_full_bonus - 1)
        multiplier = 1.0 + bonus_domains * config.cross_domain_bonus

        return base_score, multiplier


def _compute_std(values: list[float]) -> float:
    """Compute standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


class ManifoldStitcher:
    """
    Manifold stitching for cross-architecture model merging.
    Implementation of Continuous CKA-based stitching.
    """

    OUTPUT_LAYER_MARKER = 999999

    @staticmethod
    def compute_continuous_correlation(
        source: ContinuousFingerprint,
        target: ContinuousFingerprint,
        layer: int,
        backend: "Backend | None" = None,
    ) -> ContinuousCorrelationResult | None:
        b = backend or get_default_backend()
        if layer not in source.activation_vectors or layer not in target.activation_vectors:
            return None
        s_vec = b.array(source.activation_vectors[layer])
        t_vec = b.array(target.activation_vectors[layer])
        if s_vec.size == 0 or t_vec.size == 0:
            return None

        min_len = min(s_vec.size, t_vec.size)
        s_trunc, t_trunc = s_vec[:min_len], t_vec[:min_len]

        dot_prod = float(b.to_numpy(b.sum(s_trunc * t_trunc)).item())
        s_norm = float(b.to_numpy(b.norm(s_vec)).item())
        t_norm = float(b.to_numpy(b.norm(t_vec)).item())
        cosine = dot_prod / (s_norm * t_norm) if (s_norm > 1e-8 and t_norm > 1e-8) else 0.0

        mag_ratio = (
            source.magnitudes.get(layer, 1.0) / target.magnitudes.get(layer, 1.0)
            if target.magnitudes.get(layer, 1.0) > 1e-8
            else 1.0
        )
        entropy_delta = source.entropies.get(layer, 0.0) - target.entropies.get(layer, 0.0)

        return ContinuousCorrelationResult(cosine * cosine, cosine, mag_ratio, entropy_delta)

    @staticmethod
    def compute_cka_matrix(
        source: ContinuousModelFingerprints,
        target: ContinuousModelFingerprints,
        layer: int,
        backend: "Backend | None" = None,
    ) -> tuple[Any, list[str], list[str]]:
        """
        Compute pairwise CKA matrix between all primes at a given layer.
        Returns: (matrix, source_prime_ids, target_prime_ids)
        """
        b = backend or get_default_backend()
        s_fps = [fp for fp in source.fingerprints if layer in fp.activation_vectors]
        t_fps = [fp for fp in target.fingerprints if layer in fp.activation_vectors]
        if not s_fps or not t_fps:
            return (b.array([]), [], [])

        matrix = []
        for s_fp in s_fps:
            row = []
            for t_fp in t_fps:
                res = ManifoldStitcher.compute_continuous_correlation(s_fp, t_fp, layer, backend=b)
                row.append(res.cka if res else 0.0)
            matrix.append(row)
        return (b.array(matrix), [fp.prime_id for fp in s_fps], [fp.prime_id for fp in t_fps])

    @staticmethod
    def compute_intersection_rotation(
        intersection: IntersectionMap,
        layer: int,
        source_basis: Any,
        target_basis: Any,
        backend: "Backend | None" = None,
    ) -> tuple[Any, float]:
        """
        Computes a targeted rotation matrix using the intersection map.
        Strong correlations -> tight rotation (trust mapping).
        """
        b = backend or get_default_backend()
        correlations = intersection.dimension_correlations.get(layer, [])
        if not correlations:
            return (b.eye(source_basis.shape[1]), 0.0)

        dim_s = source_basis.shape[0]
        dim_t = target_basis.shape[0]

        # Filter valid correlations
        filtered = [c for c in correlations if c.source_dim < dim_s and c.target_dim < dim_t]

        if len(filtered) < 2:
            k = min(source_basis.shape[1], target_basis.shape[1])
            return (b.eye(k), 0.0)

        # Build index arrays
        source_indices = [c.source_dim for c in filtered]
        target_indices = [c.target_dim for c in filtered]

        # Take rows from basis matrices using indexing
        z_source = source_basis[source_indices]
        z_target = target_basis[target_indices]

        # Weighting
        weights = [max(0.0, c.correlation) for c in filtered]
        sqrt_weights = b.reshape(b.sqrt(b.array(weights)), (-1, 1))

        z_source = z_source * sqrt_weights
        z_target = z_target * sqrt_weights

        # Procrustes: R = U @ V^T minimizes ||source @ R - target||_F
        m = b.matmul(b.transpose(z_source), z_target)
        u, _, vt = b.svd(m)
        omega = b.matmul(u, vt)

        # Ensure proper rotation (det = +1), not reflection
        omega = _ensure_proper_rotation(u, vt, omega, b)

        confidence = sum(weights) / max(len(weights), 1)
        return (omega, confidence)

    @staticmethod
    def cluster_activations(
        source_activations: dict[str, list[float]],  # PrimeID -> Activation (Layer 0)
        target_activations: dict[str, list[float]],
        cluster_count: int = 8,
        backend: "Backend | None" = None,
        geodesic_k_neighbors: int = 10,
    ) -> list["AlignmentCluster"]:  # Forward ref string since defined later
        """Clusters activations to identify alignment regions.

        Uses Riemannian K-means with geodesic distances and Fréchet centroids.
        In high-dimensional spaces, curvature is inherent - geodesic distance
        is the correct metric.

        Args:
            source_activations: Source model activations (PrimeID -> vector)
            target_activations: Target model activations (PrimeID -> vector)
            cluster_count: Number of clusters
            backend: Compute backend
            geodesic_k_neighbors: k for geodesic distance estimation

        Returns:
            List of alignment clusters with local rotations
        """
        b = backend or get_default_backend()
        keys = sorted(list(set(source_activations.keys()) & set(target_activations.keys())))
        if not keys:
            return []

        source_vecs = [source_activations[k] for k in keys]
        target_vecs = [target_activations[k] for k in keys]

        # Riemannian K-Means with geodesic distances
        assignments, _ = ManifoldStitcher.k_means(
            source_vecs,
            cluster_count,
            backend=b,
            geodesic_k_neighbors=geodesic_k_neighbors,
        )

        # Initialize Riemannian geometry for Fréchet means
        from modelcypher.core.domain.geometry.riemannian_utils import (
            RiemannianGeometry,
        )

        riemannian = RiemannianGeometry(backend=b)

        clusters = []
        shared_dim = min(len(source_vecs[0]), len(target_vecs[0]))

        for cluster_id in range(cluster_count):
            indices = [i for i, a in enumerate(assignments) if a == cluster_id]
            if not indices:
                continue

            s_members = b.array([source_vecs[i][:shared_dim] for i in indices])
            t_members = b.array([target_vecs[i][:shared_dim] for i in indices])

            # Compute cluster centroids using Fréchet mean (Riemannian center of mass)
            s_result = riemannian.frechet_mean(
                s_members, max_iterations=20, tolerance=1e-5
            )
            t_result = riemannian.frechet_mean(
                t_members, max_iterations=20, tolerance=1e-5
            )
            s_mean = s_result.mean
            t_mean = t_result.mean

            # Local rotation via Procrustes
            s_centered = s_members - s_mean
            t_centered = t_members - t_mean

            m = b.matmul(b.transpose(s_centered), t_centered)
            u, _, vt = b.svd(m)
            omega = b.matmul(u, vt)

            # Ensure proper rotation (det = +1), not reflection
            omega = _ensure_proper_rotation(u, vt, omega, b)

            # Error
            projected = b.matmul(s_centered, omega)
            error = projected - t_centered
            error_norm = float(b.to_numpy(b.sqrt(b.sum(error * error))).item())
            target_norm = float(b.to_numpy(b.sqrt(b.sum(t_centered * t_centered))).item())
            procrustes_error = error_norm / target_norm if target_norm > 1e-6 else 0.0

            clusters.append(
                AlignmentCluster(
                    id=cluster_id,
                    centroid_source=b.to_numpy(s_mean).tolist(),
                    centroid_target=b.to_numpy(t_mean).tolist(),
                    local_rotation=omega,
                    procrustes_error=procrustes_error,
                    member_count=len(indices),
                )
            )

        return clusters

    @staticmethod
    def k_means(
        points: list[list[float]],
        k: int,
        max_iterations: int = 50,
        backend: "Backend | None" = None,
        geodesic_k_neighbors: int = 10,
        seed: int | None = 42,
    ) -> tuple[list[int], list[list[float]]]:
        """Riemannian K-means clustering with geodesic distances.

        In high-dimensional spaces, curvature is inherent. Uses geodesic
        distances and Fréchet centroids:
        1. Geodesic distances (via k-NN graph) for assignment step
        2. Fréchet mean (Karcher mean) for centroid updates

        This correctly handles curved manifolds where Euclidean K-means fails:
        - Positive curvature: Euclidean underestimates distances → clusters too spread
        - Negative curvature: Euclidean overestimates distances → clusters too tight

        Args:
            points: [N, D] list of points to cluster
            k: Number of clusters
            max_iterations: Maximum iterations
            backend: Compute backend
            geodesic_k_neighbors: k for k-NN graph in geodesic estimation

        Returns:
            (assignments, centroids) tuple
        """
        b = backend or get_default_backend()
        n = len(points)
        if n == 0 or k <= 0:
            return ([], [])

        if seed is not None:
            b.random_seed(seed)

        pts = b.array(points)
        d_dim = pts.shape[1]

        # Precompute geodesic distances and Riemannian geometry for Fréchet means
        from modelcypher.core.domain.geometry.riemannian_utils import (
            RiemannianGeometry,
        )

        riemannian = RiemannianGeometry(backend=b)
        geodesic_result = riemannian.geodesic_distances(
            pts, k_neighbors=min(geodesic_k_neighbors, n - 1)
        )
        geodesic_dist_matrix = geodesic_result.distances
        # Precompute numpy version for centroid representative updates
        geo_np = b.to_numpy(geodesic_dist_matrix)

        def compute_distance_to_centroids(
            pts_arr: "Array", centroid_indices: list[int]
        ) -> "Array":
            """Compute geodesic distances from all points to selected centroids."""
            num_centroids = len(centroid_indices)
            # Use precomputed geodesic distances
            # Use .copy() to get writable arrays (JAX returns read-only)
            dists = b.zeros((n, num_centroids))
            dists_np = b.to_numpy(dists).copy()
            geo_np = b.to_numpy(geodesic_dist_matrix)
            for ci, idx in enumerate(centroid_indices):
                dists_np[:, ci] = geo_np[:, idx]
            return b.array(dists_np)

        def compute_geodesic_distances_to_centroids(centroids_arr: "Array") -> "Array":
            """Compute geodesic distances from all points to centroids (geodesic-only)."""
            num_centroids = int(centroids_arr.shape[0])
            if num_centroids == 0:
                return b.zeros((n, 0))
            rows = []
            for ci in range(num_centroids):
                geo_from_centroid = riemannian._geodesic_distances_from_query(
                    pts,
                    centroids_arr[ci],
                    geo_result=geodesic_result,
                )
                b.eval(geo_from_centroid)
                rows.append(geo_from_centroid)
            return b.stack(rows, axis=1)

        # K-Means++ Initialization using actual data points as initial centroids
        first_idx = int(b.to_numpy(b.random_randint(0, n, shape=(1,))).item())
        centroid_indices = [first_idx]

        for _ in range(1, min(k, n)):
            # Compute distances to nearest existing centroid
            dists = compute_distance_to_centroids(pts, centroid_indices)
            min_dists = b.min(dists, axis=1)

            # Sample proportional to squared distance
            probs = min_dists**2
            prob_sum = b.sum(probs)
            b.eval(prob_sum)
            if float(b.to_numpy(prob_sum)) < 1e-12:
                # All points are at centroids, pick randomly
                next_idx = int(b.to_numpy(b.random_randint(0, n, shape=(1,))).item())
            else:
                probs = probs / prob_sum
                cumsum = b.cumsum(probs)
                r = b.random_uniform(shape=(1,))
                next_idx = int(
                    b.to_numpy(b.argmax(cumsum > float(b.to_numpy(r).item()))).item()
                )
            centroid_indices.append(next_idx)

        # Initialize centroids from selected points
        # Use array indexing instead of list (JAX doesn't allow list indexing)
        idx_arr = b.array(centroid_indices)
        centroids_np = b.to_numpy(pts[idx_arr])
        centroids = b.array(centroids_np)
        assignments = b.zeros((n,), dtype="int32")
        # Track representative data point for each centroid (for geodesic proxy)
        centroid_reps = list(centroid_indices)

        for _ in range(max_iterations):
            # Assignment step: assign each point to nearest centroid using geodesic distances
            dists = compute_geodesic_distances_to_centroids(centroids)
            new_assignments = b.argmin(dists, axis=1)

            # Check convergence
            if (b.to_numpy(assignments) == b.to_numpy(new_assignments)).all():
                break
            assignments = new_assignments

            # Update step: compute new centroids using Fréchet mean
            assignments_np = b.to_numpy(assignments)
            pts_np = b.to_numpy(pts)

            new_centroids = []
            for c in range(k):
                mask = assignments_np == c
                if mask.sum() > 0:
                    cluster_pts = b.array(pts_np[mask])
                    result = riemannian.frechet_mean(
                        cluster_pts,
                        max_iterations=20,
                        tolerance=1e-5,
                    )
                    new_centroids.append(b.to_numpy(result.mean))
                else:
                    # Empty cluster: keep old centroid
                    new_centroids.append(centroids_np[c])
            centroids_np = b.to_numpy(b.stack([b.array(c) for c in new_centroids]))

            centroids = b.array(centroids_np)

            # Update representatives: find nearest data point to each new centroid
            # Uses geodesic distance from current representative as primary signal
            for ci in range(k):
                old_rep = centroid_reps[ci]
                cent_pt = centroids_np[ci]

                # Find nearest data point using geodesic proxy from old representative
                best_idx = old_rep
                best_dist = float("inf")
                for pi in range(n):
                    pt = pts_np[pi]
                    euc_sq = sum((pt[d] - cent_pt[d]) ** 2 for d in range(d_dim))
                    # Geodesic from old rep to candidate + small Euclidean to new centroid
                    geo_to_old_rep = geo_np[pi, old_rep]
                    total_dist = geo_to_old_rep + 0.1 * (euc_sq**0.5)
                    if total_dist < best_dist:
                        best_dist = total_dist
                        best_idx = pi

                centroid_reps[ci] = best_idx

        return (b.to_numpy(assignments).tolist(), b.to_numpy(centroids).tolist())

    @staticmethod
    def soft_rotation(
        weight: Any,
        clusters: list["AlignmentCluster"],
        temperature: float = 0.3,
        backend: "Backend | None" = None,
    ) -> Any:
        b = backend or get_default_backend()
        if not clusters:
            return weight
        if weight.ndim != 2:
            return weight

        in_dim = weight.shape[1]
        cluster_dim = clusters[0].local_rotation.shape[0]
        if in_dim != cluster_dim:
            return weight

        # Weighted average
        weights = []
        for c in clusters:
            w = math.exp(-c.procrustes_error / temperature) * c.member_count
            weights.append(w)

        total_weight = sum(weights)
        if total_weight <= 0:
            return weight

        weighted_omega = b.zeros((cluster_dim, cluster_dim))
        for i, c in enumerate(clusters):
            norm_w = weights[i] / total_weight
            weighted_omega = weighted_omega + (c.local_rotation * norm_w)

        # Re-orthogonalize
        u, _, vt = b.svd(weighted_omega)
        omega = b.matmul(u, vt)

        return b.matmul(weight, b.transpose(omega))

    @staticmethod
    async def validate_merged_model(
        merged_model_ctx: Any,  # ModelContext
        merged_model_id: str,
        target_fingerprints: ModelFingerprints,
        top_k: int = 32,
    ) -> ValidationResult:
        """
        Validates a merged model by comparing its fingerprints to the original target.
        """
        # Determine layers to probe
        target_layers = set()
        for fp in target_fingerprints.fingerprints:
            target_layers.update(fp.activated_dimensions.keys())

        intermediate_layers = {
            l for l in target_layers if l > 0 and l != ManifoldStitcher.OUTPUT_LAYER_MARKER
        }
        probe_layers = list(intermediate_layers) if intermediate_layers else None

        # Probe merged model
        merged_fingerprints = await ManifoldStitcher.probe_with_primes(
            model_ctx=merged_model_ctx,
            model_id=merged_model_id,
            probe_space=target_fingerprints.probe_space,
            top_k=top_k,
            layer_indices=probe_layers,
        )

        # Compare
        layer_deltas = []
        all_layers = target_layers.union(
            {l for fp in merged_fingerprints.fingerprints for l in fp.activated_dimensions.keys()}
        )

        merged_map = {fp.prime_id: fp for fp in merged_fingerprints.fingerprints}
        target_map = {fp.prime_id: fp for fp in target_fingerprints.fingerprints}

        prime_ids = set(merged_map.keys()) & set(target_map.keys())

        for layer in sorted(list(all_layers)):
            merged_dims = set()
            target_dims = set()

            for pid in prime_ids:
                if layer in merged_map[pid].activated_dimensions:
                    merged_dims.update(
                        [d.index for d in merged_map[pid].activated_dimensions[layer]]
                    )
                if layer in target_map[pid].activated_dimensions:
                    target_dims.update(
                        [d.index for d in target_map[pid].activated_dimensions[layer]]
                    )

            overlap = merged_dims & target_dims
            union = merged_dims | target_dims
            jaccard = len(overlap) / len(union) if union else 0.0

            layer_deltas.append(
                LayerDelta(
                    layer=layer,
                    overlap_count=len(overlap),
                    merged_count=len(merged_dims),
                    target_count=len(target_dims),
                    jaccard_similarity=jaccard,
                )
            )

        mean_jaccard = (
            sum(d.jaccard_similarity for d in layer_deltas) / len(layer_deltas)
            if layer_deltas
            else 0.0
        )

        return ValidationResult(
            merged_model=merged_model_id,
            target_model=target_fingerprints.model_id,
            layer_deltas=layer_deltas,
            overall_similarity=mean_jaccard,
        )

    @staticmethod
    async def probe_with_primes(
        model_ctx: Any,
        model_id: str,
        probe_space: ProbeSpace,
        top_k: int,
        layer_indices: list[int] | None = None,
    ) -> ModelFingerprints:
        # Placeholder for actual probing logic
        # In a real implementation, this would use UnifiedAtlas probes and run inference
        # capturing activations. For now, return empty fingerprint set.

        # We will assume for now this is handled by external service or just return empty for parity structure.
        return ModelFingerprints(
            model_id=model_id,
            probe_space=probe_space,
            probe_capture_key=None,
            hidden_dim=0,
            layer_count=0,
            fingerprints=[],
        )


@dataclass(frozen=True)
class ActivatedDimension:
    index: int
    activation: float

    def __lt__(self, other: "ActivatedDimension") -> bool:
        return self.activation > other.activation


@dataclass(frozen=True)
class ActivationFingerprint:
    prime_id: str
    prime_text: str
    activated_dimensions: dict[int, list[ActivatedDimension]]


@dataclass(frozen=True)
class SparseActivationVector:
    indices: list[int]
    values: list[float]
    length: int

    def dot(self, other: "SparseActivationVector") -> float:
        count_a = min(len(self.indices), len(self.values))
        count_b = min(len(other.indices), len(other.values))
        if count_a <= 0 or count_b <= 0:
            return 0.0
        i = 0
        j = 0
        total = 0.0
        while i < count_a and j < count_b:
            idx_a = self.indices[i]
            idx_b = other.indices[j]
            if idx_a == idx_b:
                total += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif idx_a < idx_b:
                i += 1
            else:
                j += 1
        return total

    def dot_dense(self, dense: list[float]) -> float:
        count = min(len(self.indices), len(self.values))
        if count <= 0:
            return 0.0
        total = 0.0
        for i in range(count):
            idx = self.indices[i]
            if 0 <= idx < len(dense):
                total += self.values[i] * dense[idx]
        return total


@dataclass(frozen=True)
class ModelFingerprints:
    model_id: str
    probe_space: ProbeSpace
    probe_capture_key: str | None
    hidden_dim: int
    layer_count: int
    fingerprints: list[ActivationFingerprint]
    activation_vectors: dict[str, list[float]] | None = None
    activation_sparse_vectors: dict[str, SparseActivationVector] | None = None


@dataclass
class AlignmentCluster:
    """A cluster of aligned activation vectors between source and target models.

    Attributes
    ----------
    id : int
        Cluster identifier.
    centroid_source : list[float]
        Centroid position in source model space.
    centroid_target : list[float]
        Centroid position in target model space.
    local_rotation : Any
        Rotation matrix from backend.
    procrustes_error : float
        Procrustes alignment error. Lower values indicate better alignment.
    member_count : int
        Number of vectors in this cluster.
    """

    id: int
    centroid_source: list[float]
    centroid_target: list[float]
    local_rotation: Any  # Array type from backend
    procrustes_error: float
    member_count: int

    def classification_for_thresholds(
        self,
        aligned_threshold: float = 0.3,
        translatable_threshold: float = 0.7,
    ) -> str:
        """Classify alignment quality using caller-provided thresholds.

        Args:
            aligned_threshold: Error below this is "aligned" (well-aligned)
            translatable_threshold: Error below this is "translatable" (moderately aligned)

        Returns:
            Classification label: "aligned", "translatable", or "divergent"
        """
        if self.procrustes_error < aligned_threshold:
            return "aligned"
        if self.procrustes_error < translatable_threshold:
            return "translatable"
        return "divergent"



@dataclass
class LayerDelta:
    layer: int
    overlap_count: int
    merged_count: int
    target_count: int
    jaccard_similarity: float


@dataclass
class ValidationResult:
    """Result of validating a merged model against a target.

    Attributes
    ----------
    merged_model : str
        Path or identifier of the merged model.
    target_model : str
        Path or identifier of the target model.
    layer_deltas : list[LayerDelta]
        Per-layer similarity deltas.
    overall_similarity : float
        Mean Jaccard similarity across layers. Higher values indicate better merge quality.
    """

    merged_model: str
    target_model: str
    layer_deltas: list[LayerDelta]
    overall_similarity: float

