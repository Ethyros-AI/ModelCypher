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
Cross-architecture model merging via manifold alignment.

Models with different architectures encode knowledge on the same underlying
geometric structure. This module aligns and stitches these representations
by finding the correct transformation between coordinate systems.

Mathematical Foundation:
    Neural network activations define points on a Riemannian manifold. Alignment
    requires computing distances and means on this curved space:

    1. Geodesic Distance: Shortest path along the manifold surface, computed via
       k-NN graph. Euclidean distance systematically errs:
       - Positive curvature: Euclidean underestimates true distance
       - Negative curvature: Euclidean overestimates true distance

    2. Frechet Mean (Karcher Mean): Riemannian center of mass.
       Minimizes sum of squared geodesic distances: mu = argmin_p sum(d^2(p, x_i))

    3. Local Procrustes: SVD-based rotation that minimizes ||X @ R - Y||_F
       subject to det(R) = +1 (proper rotation, not reflection).

Algorithm:
    1. Triangulated Probing: Collect activation fingerprints across multiple
       semantic domains (semantic primes, sequence invariants, metaphor invariants).
       Cross-domain agreement triangulates to robust alignment.

    2. Continuous Fingerprinting: Preserve full activation vectors with magnitude,
       entropy, and sparsity metadata. CKA measures structural similarity.

    3. Riemannian K-Means: Cluster activations using geodesic distances and
       Frechet centroids. Groups activation regions for local alignment.

    4. Local Procrustes: Compute rotation matrices within each cluster using
       SVD. The Schonemann (1966) sign correction ensures det(R) = +1.

Key Principles:
    - Curvature is inherent: Always use geodesic distance, not Euclidean.
      The k-NN graph IS the discrete manifold representation.

    - Proper rotations only: det(R) = +1. Reflections (det = -1) are corrected
      by flipping the sign of the last column of U in the SVD.

    - No interpolation: Model merging is geometric ADDITION via geodesic
      null-space projection, not weighted averaging. See geodesic_null_space.py.

Usage:
    from modelcypher.core.domain.geometry.manifold_stitcher import ManifoldStitcher

    # Compute CKA alignment between models at a specific layer
    cka_matrix, source_ids, target_ids = ManifoldStitcher.compute_cka_matrix(
        source_fingerprints, target_fingerprints, layer=12
    )

    # Cluster activations for local alignment
    clusters = ManifoldStitcher.cluster_activations(
        source_activations, target_activations, cluster_count=8
    )

    # Build triangulated probes for robust fingerprinting
    probes = TriangulatedProbeBuilder.build_triangulated_probes()

References:
    - Schonemann (1966) "A Generalized Solution of the Orthogonal Procrustes Problem"
    - Tenenbaum et al. (2000) "Isomap" - geodesic distance via graph
    - Pennec (2006) "Intrinsic Statistics on Riemannian Manifolds"

See Also:
    - riemannian_utils.py: Geodesic distance, Frechet mean, curvature estimation
    - generalized_procrustes.py: Multi-shape alignment with proper rotations
    - cka.py: Centered Kernel Alignment for representation similarity
    - geodesic_null_space.py: Interference-free weight delta projection (GPU-only)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol, enum_key
from modelcypher.core.domain.geometry.atlas_registry import get_atlas_probes
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    log_scalar,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)

__all__ = [
    # Constants
    "StitchingConstants",
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


@dataclass(frozen=True)
class DimensionCorrelation:
    source_dim: int
    target_dim: int
    correlation: float


@dataclass(frozen=True)
class LayerConfidence:
    layer: int
    confidence: float
    correlation_count: int


@dataclass(frozen=True)
class IntersectionMap:
    """Map of dimension correlations between source and target models.

    Attributes:
        raw_fingerprint_similarity: PRE-ALIGNMENT similarity from raw activation
            fingerprints (0-1). Measures intrinsic similarity before any alignment.
            NOT the same as CKA (which measures post-alignment quality).
            Low values are EXPECTED for different architectures - this does NOT
            indicate alignment failure. Check CKA score for alignment quality.
    """

    source_model: str
    target_model: str
    dimension_correlations: dict[int, list[DimensionCorrelation]]
    raw_fingerprint_similarity: float  # Pre-alignment only. Use CKA for post-alignment.
    aligned_dimension_count: int
    total_source_dims: int
    total_target_dims: int
    layer_confidences: list[LayerConfidence]


class ProbeSpace(str, Enum):
    prelogits_hidden = "prelogits-hidden"
    output_logits = "output-logits"


output_layer_marker = -1


import logging
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
        det_val = float(backend.to_scalar(det_arr))

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
            norms = geodesic_norms(b.reshape(arr, (1, -1)), b)
            b.eval(norms)
            magnitudes[layer] = float(b.to_scalar(norms[0]))

            logits = arr
            max_val = b.max(logits)
            exp_acts = b.exp(logits - max_val)
            eps = division_epsilon(b, exp_acts)
            sum_exp = b.sum(exp_acts)
            probs = exp_acts / (sum_exp + eps)

            log_probs = b.log(probs + eps)
            entropy_arr = -b.sum(probs * log_probs)
            max_entropy = log_scalar(float(max(len(activations), 1)), b)
            abs_acts = b.abs(arr)
            max_abs = b.max(abs_acts)
            eps = division_epsilon(b, abs_acts)
            threshold = b.maximum(max_abs * eps, b.array(eps))
            near_zero_arr = b.sum(abs_acts <= threshold)

            b.eval(sum_exp, entropy_arr, max_abs, near_zero_arr)

            entropy = float(b.to_scalar(entropy_arr))
            entropies[layer] = min(max(entropy / max_entropy, 0.0), 1.0) if max_entropy > 0 else 0.0

            near_zero = float(b.to_scalar(near_zero_arr))
            sparsities[layer] = near_zero / max(len(activations), 1)

        return ContinuousFingerprint(
            prime_id, prime_text, layer_activations, magnitudes, entropies, sparsities
        )


@dataclass
class ContinuousCorrelationResult:
    cka: float
    cosine_similarity: float
    magnitude_ratio: float
    entropy_delta: float


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
            total_probes=len(self.fingerprints),
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
    total_probes: int
    entropy_std: float = 0.0
    sparsity_std: float = 0.0

    @property
    def is_collapsed(self) -> bool:
        """Layer is considered collapsed if very low entropy or very high sparsity."""
        b = get_default_backend()
        stats = b.array([self.mean_entropy, self.mean_sparsity])
        eps = division_epsilon(b, stats)
        return abs(self.mean_entropy) <= eps or abs(1.0 - self.mean_sparsity) <= eps

    @property
    def confidence(self) -> float:
        """Confidence based on entropy and probe count."""
        total = max(self.total_probes, 1)
        probe_factor = self.probe_count / total
        b = get_default_backend()
        stats = b.array(
            [self.mean_entropy, self.mean_sparsity, self.entropy_std, self.sparsity_std]
        )
        eps = division_epsilon(b, stats)
        signal = abs(self.mean_entropy) + abs(self.mean_sparsity)
        noise = self.entropy_std + self.sparsity_std
        stability = signal / (signal + noise + eps)
        return probe_factor * stability


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

    No configuration needed - all probes are included by default for full
    triangulation coverage. Use build_probes_for_sources() if specific
    source filtering is required.
    """

    @staticmethod
    def build_triangulated_probes() -> list[AtlasProbeProtocol]:
        """
        Build complete probe set from the atlas registry.

        Includes all sources for maximum triangulation coverage:
        - Semantic primes
        - Sequence invariants
        - Metaphor invariants
        - Conceptual genealogy (with moral/social foundations)
        - Computational gates

        Returns list of AtlasProbe objects with:
        - probe_id: Unique identifier
        - support_texts: Probe texts for embedding
        - source: Atlas source (SEMANTIC_PRIME, SEQUENCE_INVARIANT, etc.)
        - domain: Triangulation domain
        - cross_domain_weight: Weight for cross-domain detection
        """
        # Include all sources for full triangulation
        sources: set[str] = {
            _SOURCE_SEMANTIC_PRIME,
            _SOURCE_SEQUENCE_INVARIANT,
            _SOURCE_METAPHOR_INVARIANT,
            _SOURCE_CONCEPTUAL_GENEALOGY,
            _SOURCE_MORAL_CONCEPT,
            _SOURCE_SOCIAL_CONCEPT,
            _SOURCE_COMPUTATIONAL_GATE,
        }

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
    ) -> tuple[float, float]:
        """
        Compute triangulation score from multi-domain activations.

        No arbitrary thresholds or bonuses - returns raw mean of all
        activations and multiplier of 1.0. The caller interprets significance.

        Args:
            activations_by_domain: Activation scores keyed by domain

        Returns:
            Tuple of (base_score, triangulation_multiplier=1.0)
        """
        if not activations_by_domain:
            return 0.0, 1.0

        # Base score is mean of all activations (no arbitrary threshold filtering)
        values = list(activations_by_domain.values())
        base_score = sum(values) / len(values)

        # No arbitrary bonus multiplier - always 1.0
        return base_score, 1.0


def _compute_std(values: list[float]) -> float:
    """Compute standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return sqrt_scalar(variance, get_default_backend())


def _array_to_list(backend: "Backend", array: "Array") -> list[float]:
    """Convert 1D array to Python list using native tolist() - O(1) vs O(n)."""
    flat = backend.reshape(array, (-1,))
    return backend.tolist(flat)


def _array_to_2d_list(backend: "Backend", array: "Array") -> list[list[float]]:
    """Convert 2D array to nested Python list using native tolist() - O(1) vs O(n*m)."""
    return backend.tolist(array)


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

        s_mat = b.reshape(s_trunc, (1, -1))
        t_mat = b.reshape(t_trunc, (1, -1))
        cos_arr, _ = geodesic_pairwise_metrics(s_mat, t_mat, b)
        b.eval(cos_arr)
        cosine = float(b.to_scalar(cos_arr))
        norm_eps = division_epsilon(b, s_vec)

        mag_ratio = (
            source.magnitudes.get(layer, 1.0) / target.magnitudes.get(layer, 1.0)
            if target.magnitudes.get(layer, 1.0) > norm_eps
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
        geodesic_k_neighbors: int | None = None,
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
            geodesic_k_neighbors: k for geodesic (None = derive from geometry)

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
        # (k=None triggers connectivity-based selection in k_means)
        assignments, _ = ManifoldStitcher.k_means(
            source_vecs,
            cluster_count,
            backend=b,
            geodesic_k_neighbors=geodesic_k_neighbors,  # Pass through, may be None
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
            # Use regularization_epsilon (sqrt(machine_eps)) for convergence tolerance
            tol = regularization_epsilon(b, s_members)
            s_result = riemannian.frechet_mean(
                s_members, max_iterations=20, tolerance=tol
            )
            t_result = riemannian.frechet_mean(
                t_members, max_iterations=20, tolerance=tol
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

            # Error using geodesic norms
            projected = b.matmul(s_centered, omega)
            error = projected - t_centered
            error_flat = b.reshape(error, (1, -1))
            target_flat = b.reshape(t_centered, (1, -1))
            error_norm_arr = geodesic_norms(error_flat, b)
            target_norm_arr = geodesic_norms(target_flat, b)
            b.eval(error_norm_arr, target_norm_arr)
            error_norm = float(b.to_scalar(error_norm_arr))
            target_norm = float(b.to_scalar(target_norm_arr))
            norm_eps = division_epsilon(b, t_centered)
            procrustes_error = error_norm / target_norm if target_norm > norm_eps else 0.0

            clusters.append(
                AlignmentCluster(
                    id=cluster_id,
                    centroid_source=_array_to_list(b, s_mean),
                    centroid_target=_array_to_list(b, t_mean),
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
        geodesic_k_neighbors: int | None = None,
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
            geodesic_k_neighbors: k for k-NN graph (None = derive from geometry)

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

        # Precompute geodesic distances (k=None triggers connectivity-based selection)
        from modelcypher.core.domain.geometry.riemannian_utils import (
            RiemannianGeometry,
        )

        riemannian = RiemannianGeometry(backend=b)
        # If geodesic_k_neighbors is specified, cap at n-1; otherwise let geodesic_distances
        # do connectivity-based selection
        k_to_use = min(geodesic_k_neighbors, n - 1) if geodesic_k_neighbors is not None else None
        geodesic_result = riemannian.geodesic_distances(pts, k_neighbors=k_to_use)
        # Get actual k from result (may differ if None was passed)
        geodesic_k_neighbors = geodesic_result.k_neighbors
        geodesic_dist_matrix = geodesic_result.distances
        def compute_distance_to_centroids(
            pts_arr: "Array", centroid_indices: list[int]
        ) -> "Array":
            """Compute geodesic distances from all points to selected centroids."""
            num_centroids = len(centroid_indices)
            if num_centroids == 0:
                return b.zeros((n, 0))
            idx_arr = b.array(centroid_indices)
            return b.take(geodesic_dist_matrix, idx_arr, axis=1)

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
        steps = min(k, n)
        rand_uniforms = b.random_uniform(shape=(max(steps - 1, 1),))
        rand_ints = b.random_randint(0, n, shape=(max(steps, 1),))
        b.eval(rand_uniforms, rand_ints)
        rand_uniforms_list = b.tolist(rand_uniforms) if steps > 1 else []
        rand_ints_list = [int(x) for x in b.tolist(rand_ints)]
        first_idx = rand_ints_list[0] if steps > 0 else 0
        centroid_indices = [first_idx]

        for step_idx in range(1, steps):
            # Compute distances to nearest existing centroid
            dists = compute_distance_to_centroids(pts, centroid_indices)
            min_dists = b.min(dists, axis=1)

            # Sample proportional to squared distance
            probs = min_dists**2
            prob_sum = b.sum(probs)
            b.eval(prob_sum)
            prob_eps = division_epsilon(b, probs)
            prob_sum_val = float(b.to_scalar(prob_sum))
            if prob_sum_val <= prob_eps:
                # All points are at centroids, pick randomly
                next_idx = rand_ints_list[step_idx]
            else:
                probs = probs / prob_sum
                cumsum = b.cumsum(probs)
                r_val = float(rand_uniforms_list[step_idx - 1]) if rand_uniforms_list else 0.0
                r_val = min(r_val, 1.0 - prob_eps)
                next_idx_arr = b.argmax(cumsum > r_val)
                b.eval(next_idx_arr)
                next_idx = int(b.to_scalar(next_idx_arr))
            centroid_indices.append(next_idx)

        # Initialize centroids from selected points
        # Use array indexing instead of list (JAX doesn't allow list indexing)
        idx_arr = b.array(centroid_indices)
        centroids = b.take(pts, idx_arr, axis=0)
        assignments = b.zeros((n,), dtype="int32")
        # Track representative data point for each centroid (for geodesic proxy)
        centroid_reps = list(centroid_indices)

        for _ in range(max_iterations):
            # Assignment step: assign each point to nearest centroid using geodesic distances
            dists = compute_geodesic_distances_to_centroids(centroids)
            new_assignments = b.argmin(dists, axis=1)

            # Check convergence
            same = b.sum(b.astype(assignments == new_assignments, "float32"))
            b.eval(same)
            if float(b.to_scalar(same)) == float(n):
                break
            assignments = new_assignments

            # Update step: compute new centroids using Fréchet mean
            assignments_list = [int(x) for x in b.tolist(assignments)]
            new_centroids = []
            for c in range(k):
                indices = [i for i, val in enumerate(assignments_list) if val == c]
                if indices:
                    idx_arr = b.array(indices)
                    cluster_pts = b.take(pts, idx_arr, axis=0)
                    result = riemannian.frechet_mean(
                        cluster_pts,
                        max_iterations=20,
                        tolerance=regularization_epsilon(b, cluster_pts),
                    )
                    new_centroids.append(result.mean)
                else:
                    # Empty cluster: keep old centroid
                    new_centroids.append(centroids[c])
            centroids = b.stack(new_centroids, axis=0)

            # Update representatives: find nearest data point to each new centroid
            # Uses geodesic distance from current representative as primary signal
            for ci in range(k):
                old_rep = centroid_reps[ci]

                # Find nearest data point using pure geodesic distance from old representative
                col = b.take(geodesic_dist_matrix, b.array([old_rep]), axis=1)
                col = b.squeeze(col, axis=1)
                best_idx_arr = b.argmin(col)
                b.eval(best_idx_arr)
                best_idx = int(b.to_scalar(best_idx_arr))
                centroid_reps[ci] = best_idx

        return (_array_to_list(b, assignments), _array_to_2d_list(b, centroids))

    @staticmethod
    async def validate_merged_model(
        merged_model_ctx: Any,  # ModelContext
        merged_model_id: str,
        target_fingerprints: ModelFingerprints,
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
        Procrustes alignment error. Lower values indicate less alignment error.
    member_count : int
        Number of vectors in this cluster.
    """

    id: int
    centroid_source: list[float]
    centroid_target: list[float]
    local_rotation: Any  # Array type from backend
    procrustes_error: float
    member_count: int



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
        Mean Jaccard similarity across layers. Higher values indicate higher similarity.
    """

    merged_model: str
    target_model: str
    layer_deltas: list[LayerDelta]
    overall_similarity: float
