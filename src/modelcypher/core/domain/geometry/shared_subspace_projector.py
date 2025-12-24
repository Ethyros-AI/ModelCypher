from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import math

import numpy as np

from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry.geometry_fingerprint import GeometricFingerprint

# unified_atlas imported lazily in validate_crm_uses_atlas to avoid circular imports

EIGENVALUE_FLOOR = 1e-10
SINGULAR_VALUE_FLOOR = 1e-8


class AlignmentMethod(str, Enum):
    cca = "cca"
    shared_svd = "sharedSVD"
    procrustes = "procrustes"


class PcaMode(str, Enum):
    auto = "auto"
    svd = "svd"
    gram = "gram"


@dataclass(frozen=True)
class Config:
    alignment_method: AlignmentMethod = AlignmentMethod.cca
    variance_threshold: float = 0.95
    pca_variance_threshold: float = 0.95
    max_shared_dimension: int = 256
    cca_regularization: float = 1e-4
    min_samples: int = 10
    min_canonical_correlation: float = 0.1
    pca_mode: PcaMode = PcaMode.auto
    anchor_prefixes: tuple[str, ...] | None = None
    anchor_weights: dict[str, float] | None = None

    @staticmethod
    def default() -> "Config":
        return Config()


def validate_crm_uses_atlas(crm: ConceptResponseMatrix) -> tuple[bool, dict]:
    """Check if ConceptResponseMatrix was built using unified atlas probes.

    The unified atlas provides 321 probes across 7 sources for cross-domain
    triangulation. CRM data built from atlas probes enables more robust
    dimension-agnostic alignment.

    Args:
        crm: The ConceptResponseMatrix to validate

    Returns:
        Tuple of (is_valid, details) where details contains:
        - atlas_ids: Set of atlas probe IDs
        - crm_ids: Set of CRM concept IDs
        - overlap: Number of matching IDs
        - coverage: Fraction of atlas IDs present in CRM
        - is_subset: Whether CRM uses a subset of atlas
    """
    # Lazy import to avoid circular dependency
    from modelcypher.core.domain.agents.unified_atlas import get_probe_ids

    atlas_ids = set(get_probe_ids())
    crm_ids = set(crm.concept_ids) if hasattr(crm, "concept_ids") else set()

    overlap = len(atlas_ids & crm_ids)
    coverage = overlap / len(atlas_ids) if atlas_ids else 0.0

    # Valid if CRM uses atlas probes (either as subset or superset)
    is_valid = (
        atlas_ids.issubset(crm_ids)  # CRM has all atlas probes
        or crm_ids.issubset(atlas_ids)  # CRM uses subset of atlas
        or coverage > 0.5  # At least 50% overlap
    )

    details = {
        "atlas_probe_count": len(atlas_ids),
        "crm_concept_count": len(crm_ids),
        "overlap_count": overlap,
        "coverage": coverage,
        "is_subset": crm_ids.issubset(atlas_ids),
        "is_superset": atlas_ids.issubset(crm_ids),
    }

    return is_valid, details


@dataclass(frozen=True)
class H3ValidationMetrics:
    shared_dimension: int
    top_canonical_correlation: float
    alignment_error: float
    shared_variance_ratio: float

    @property
    def is_h3_validated(self) -> bool:
        return (
            self.shared_dimension >= 32
            and self.top_canonical_correlation > 0.5
            and self.alignment_error < 0.3
            and self.shared_variance_ratio > 0.8
        )

    @property
    def summary(self) -> str:
        status = "PASS" if self.is_h3_validated else "FAIL"
        dim_ok = "OK" if self.shared_dimension >= 32 else "FAIL"
        corr_ok = "OK" if self.top_canonical_correlation > 0.5 else "FAIL"
        err_ok = "OK" if self.alignment_error < 0.3 else "FAIL"
        var_ok = "OK" if self.shared_variance_ratio > 0.8 else "FAIL"
        return (
            f"H3 Validation: {status}\n"
            f"- Shared Dimension: {self.shared_dimension} (target: >=32) {dim_ok}\n"
            f"- Top Correlation: {self.top_canonical_correlation:.3f} (target: >0.5) {corr_ok}\n"
            f"- Alignment Error: {self.alignment_error:.3f} (target: <0.3) {err_ok}\n"
            f"- Shared Variance: {self.shared_variance_ratio * 100:.1f}% (target: >80%) {var_ok}"
        )


@dataclass(frozen=True)
class Result:
    shared_dimension: int
    source_dimension: int
    target_dimension: int
    source_projection: list[list[float]]
    target_projection: list[list[float]]
    alignment_strengths: list[float]
    alignment_error: float
    shared_variance_ratio: float
    sample_count: int
    method: AlignmentMethod

    @property
    def is_valid(self) -> bool:
        return self.shared_dimension > 0 and self.alignment_error < 0.5 and self.shared_variance_ratio > 0.5

    @property
    def h3_metrics(self) -> H3ValidationMetrics:
        return H3ValidationMetrics(
            shared_dimension=self.shared_dimension,
            top_canonical_correlation=self.alignment_strengths[0] if self.alignment_strengths else 0.0,
            alignment_error=self.alignment_error,
            shared_variance_ratio=self.shared_variance_ratio,
        )


class SharedSubspaceProjector:
    @staticmethod
    def discover(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        layer: int,
        target_layer: int | None = None,
        config: Config = Config(),
    ) -> Result | None:
        source_layer = int(layer)
        target_layer = source_layer if target_layer is None else int(target_layer)
        matrices = SharedSubspaceProjector._extract_activation_matrices(
            source_crm,
            target_crm,
            source_layer,
            target_layer,
            config,
        )
        if matrices is None:
            return None
        source_matrix, target_matrix, weights = matrices

        if len(source_matrix) != len(target_matrix) or len(source_matrix) < config.min_samples:
            return None

        n = len(source_matrix)
        d_source = len(source_matrix[0])
        d_target = len(target_matrix[0])

        method = config.alignment_method
        if isinstance(method, str):
            normalized = method.strip().lower().replace("_", "")
            if normalized in {"cca"}:
                method = AlignmentMethod.cca
            elif normalized in {"sharedsvd", "shared-svd"}:
                method = AlignmentMethod.shared_svd
            elif normalized in {"procrustes"}:
                method = AlignmentMethod.procrustes
        if method == AlignmentMethod.cca:
            return SharedSubspaceProjector._discover_with_cca(
                source_matrix, target_matrix, weights, n, d_source, d_target, config
            )
        if method == AlignmentMethod.shared_svd:
            return SharedSubspaceProjector._discover_with_shared_svd(
                source_matrix, target_matrix, weights, n, d_source, d_target, config
            )
        return SharedSubspaceProjector._discover_with_procrustes(
            source_matrix, target_matrix, weights, n, d_source, d_target, config
        )

    @staticmethod
    def _discover_with_cca(
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        weights: list[float] | None,
        n: int,
        d_source: int,
        d_target: int,
        config: Config,
    ) -> Result | None:
        source_array = np.asarray(source_activations, dtype=np.float32)
        target_array = np.asarray(target_activations, dtype=np.float32)
        if source_array.shape[0] != target_array.shape[0]:
            return None

        weight_vector = SharedSubspaceProjector._normalize_weights(weights)
        source_centered, _ = SharedSubspaceProjector._center_array(source_array, weight_vector)
        target_centered, _ = SharedSubspaceProjector._center_array(target_array, weight_vector)

        # SVCCA: reduce to high-variance subspaces before CCA to avoid ill-conditioned covariance.
        max_components_source = min(
            config.max_shared_dimension,
            source_centered.shape[0],
            source_centered.shape[1],
        )
        max_components_target = min(
            config.max_shared_dimension,
            target_centered.shape[0],
            target_centered.shape[1],
        )
        source_reduced, source_components, source_variances = SharedSubspaceProjector._pca_reduce(
            source_centered, config.pca_variance_threshold, max_components_source, config.pca_mode
        )
        target_reduced, target_components, target_variances = SharedSubspaceProjector._pca_reduce(
            target_centered, config.pca_variance_threshold, max_components_target, config.pca_mode
        )
        if source_reduced is None or target_reduced is None:
            return None
        if source_reduced.shape[0] != target_reduced.shape[0]:
            return None

        sample_count = source_reduced.shape[0]
        cxx = (source_reduced.T @ source_reduced) / float(sample_count)
        cyy = (target_reduced.T @ target_reduced) / float(sample_count)
        cxy = (source_reduced.T @ target_reduced) / float(sample_count)

        cxx = SharedSubspaceProjector._regularize_covariance(cxx, config.cca_regularization)
        cyy = SharedSubspaceProjector._regularize_covariance(cyy, config.cca_regularization)
        inv_sqrt_x, x_eigenvalues = SharedSubspaceProjector._whiten_covariance(cxx)
        inv_sqrt_y, y_eigenvalues = SharedSubspaceProjector._whiten_covariance(cyy)
        if inv_sqrt_x is None or inv_sqrt_y is None:
            return None

        cross_cov = inv_sqrt_x @ cxy @ inv_sqrt_y
        u, singular_values, v_t = np.linalg.svd(cross_cov, full_matrices=False)
        canonical = np.clip(singular_values.astype(np.float32), 0.0, 1.0)
        canonical_sq = canonical * canonical

        total_variance = float(canonical_sq.sum())
        shared_dim = 0
        cum_variance = 0.0
        for idx, corr in enumerate(canonical):
            if corr < config.min_canonical_correlation:
                break
            cum_variance += float(canonical_sq[idx])
            shared_dim = idx + 1
            if total_variance > 0 and (cum_variance / total_variance) >= config.variance_threshold:
                break
        shared_dim = min(shared_dim, config.max_shared_dimension)
        if shared_dim <= 0:
            return None

        u_truncated = u[:, :shared_dim]
        v_truncated = v_t.T[:, :shared_dim]

        source_projection = source_components @ (inv_sqrt_x @ u_truncated)
        target_projection = target_components @ (inv_sqrt_y @ v_truncated)

        source_projected = source_reduced @ (inv_sqrt_x @ u_truncated)
        target_projected = target_reduced @ (inv_sqrt_y @ v_truncated)

        alignment_error = SharedSubspaceProjector._compute_procrustes_error(
            source_projected.reshape(sample_count * shared_dim).tolist(),
            target_projected.reshape(sample_count * shared_dim).tolist(),
            sample_count,
            shared_dim,
        )

        shared_variance_ratio = cum_variance / total_variance if total_variance > 0 else 0.0

        return Result(
            shared_dimension=shared_dim,
            source_dimension=d_source,
            target_dimension=d_target,
            source_projection=source_projection.astype(np.float32).tolist(),
            target_projection=target_projection.astype(np.float32).tolist(),
            alignment_strengths=canonical[:shared_dim].tolist(),
            alignment_error=alignment_error,
            shared_variance_ratio=shared_variance_ratio,
            sample_count=sample_count,
            method=AlignmentMethod.cca,
        )

    @staticmethod
    def _discover_with_shared_svd(
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        weights: list[float] | None,
        n: int,
        d_source: int,
        d_target: int,
        config: Config,
    ) -> Result | None:
        centered_source, _ = SharedSubspaceProjector._center_matrix(source_activations, weights)
        centered_target, _ = SharedSubspaceProjector._center_matrix(target_activations, weights)

        source_gram = SharedSubspaceProjector._compute_gram_matrix(centered_source, n, d_source)
        target_gram = SharedSubspaceProjector._compute_gram_matrix(centered_target, n, d_target)

        source_eigen = GeometricFingerprint.symmetric_eigenvalues(source_gram, n)
        target_eigen = GeometricFingerprint.symmetric_eigenvalues(target_gram, n)
        if source_eigen is None or target_eigen is None:
            return None

        source_sorted = sorted([float(val) for val in source_eigen], reverse=True)
        target_sorted = sorted([float(val) for val in target_eigen], reverse=True)

        def effective_rank(values: list[float], threshold: float) -> int:
            total = sum(val for val in values if val > 0)
            if total <= 0:
                return 0
            cumulative = 0.0
            for idx, val in enumerate(values):
                if val <= 0:
                    continue
                cumulative += val
                if cumulative / total >= threshold:
                    return idx + 1
            return len(values)

        source_rank = effective_rank(source_sorted, config.variance_threshold)
        target_rank = effective_rank(target_sorted, config.variance_threshold)
        shared_dim = min(source_rank, target_rank, config.max_shared_dimension, n)
        if shared_dim <= 0:
            return None

        source_cov = SharedSubspaceProjector._compute_covariance(centered_source, centered_source, n)
        target_cov = SharedSubspaceProjector._compute_covariance(centered_target, centered_target, n)

        source_cov_reg = list(source_cov)
        target_cov_reg = list(target_cov)
        for i in range(d_source):
            source_cov_reg[i * d_source + i] += config.cca_regularization
        for i in range(d_target):
            target_cov_reg[i * d_target + i] += config.cca_regularization

        source_eigenvectors = SharedSubspaceProjector._compute_eigenvectors(source_cov_reg, d_source, shared_dim)
        target_eigenvectors = SharedSubspaceProjector._compute_eigenvectors(target_cov_reg, d_target, shared_dim)
        if source_eigenvectors is None or target_eigenvectors is None:
            return None

        source_projected = SharedSubspaceProjector._matrix_multiply(
            centered_source, source_eigenvectors, n, d_source, shared_dim
        )
        target_projected = SharedSubspaceProjector._matrix_multiply(
            centered_target, target_eigenvectors, n, d_target, shared_dim
        )

        alignment_error = SharedSubspaceProjector._compute_procrustes_error(
            source_projected, target_projected, n, shared_dim
        )

        alignment_strengths: list[float] = []
        for k in range(shared_dim):
            sum_prod = 0.0
            sum_sq_s = 0.0
            sum_sq_t = 0.0
            for i in range(n):
                s = source_projected[i * shared_dim + k]
                t = target_projected[i * shared_dim + k]
                sum_prod += s * t
                sum_sq_s += s * s
                sum_sq_t += t * t
            denom = math.sqrt(sum_sq_s * sum_sq_t)
            alignment_strengths.append(abs(sum_prod / denom) if denom > 0 else 0.0)

        source_variance = sum(source_sorted[:shared_dim])
        target_variance = sum(target_sorted[:shared_dim])
        total_source_var = sum(source_sorted)
        total_target_var = sum(target_sorted)
        denom_total = total_source_var + total_target_var
        shared_variance_ratio = (
            (source_variance + target_variance) / denom_total if denom_total > 0 else 0.0
        )

        return Result(
            shared_dimension=shared_dim,
            source_dimension=d_source,
            target_dimension=d_target,
            source_projection=SharedSubspaceProjector._reshape_to_matrix(source_eigenvectors, d_source, shared_dim),
            target_projection=SharedSubspaceProjector._reshape_to_matrix(target_eigenvectors, d_target, shared_dim),
            alignment_strengths=alignment_strengths,
            alignment_error=alignment_error,
            shared_variance_ratio=shared_variance_ratio,
            sample_count=n,
            method=AlignmentMethod.shared_svd,
        )

    @staticmethod
    def _discover_with_procrustes(
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        weights: list[float] | None,
        n: int,
        d_source: int,
        d_target: int,
        config: Config,
    ) -> Result | None:
        if d_source != d_target:
            return SharedSubspaceProjector._discover_with_cca(
                source_activations, target_activations, weights, n, d_source, d_target, config
            )

        d = d_source
        centered_source, _ = SharedSubspaceProjector._center_matrix(source_activations, weights)
        centered_target, _ = SharedSubspaceProjector._center_matrix(target_activations, weights)

        source_flat = [val for row in centered_source for val in row]
        target_flat = [val for row in centered_target for val in row]

        m = [0.0 for _ in range(d * d)]
        for i in range(d):
            for j in range(d):
                total = 0.0
                for sample in range(n):
                    total += source_flat[sample * d + i] * target_flat[sample * d + j]
                m[i * d + j] = total

        svd = SharedSubspaceProjector._svd_decomposition(m, d, d)
        if svd is None:
            return None
        u, singular_values, v_t = svd

        omega = [0.0 for _ in range(d * d)]
        for i in range(d):
            for j in range(d):
                total = 0.0
                for k in range(d):
                    total += u[i * d + k] * v_t[k * d + j]
                omega[i * d + j] = total

        rotated_source = [0.0 for _ in range(n * d)]
        for sample in range(n):
            for j in range(d):
                total = 0.0
                for k in range(d):
                    total += source_flat[sample * d + k] * omega[k * d + j]
                rotated_source[sample * d + j] = total

        error_sum = 0.0
        target_norm = 0.0
        for idx in range(n * d):
            diff = rotated_source[idx] - target_flat[idx]
            error_sum += diff * diff
            target_norm += target_flat[idx] * target_flat[idx]
        alignment_error = math.sqrt(error_sum / target_norm) if target_norm > 0 else 0.0

        total_singular = sum(singular_values)
        cum_sum = 0.0
        shared_dim = 0
        for idx, value in enumerate(singular_values):
            cum_sum += value
            if total_singular > 0 and (cum_sum / total_singular) >= config.variance_threshold:
                shared_dim = idx + 1
                break
        shared_dim = min(max(shared_dim, 1), config.max_shared_dimension, d)

        identity = []
        for i in range(d):
            row = [0.0 for _ in range(d)]
            row[i] = 1.0
            identity.append(row)

        shared_variance_ratio = cum_sum / total_singular if total_singular > 0 else 0.0

        strengths = [
            value / (total_singular if total_singular > 0 else 1.0) for value in singular_values
        ]

        return Result(
            shared_dimension=shared_dim,
            source_dimension=d,
            target_dimension=d,
            source_projection=identity,
            target_projection=SharedSubspaceProjector._reshape_to_matrix(omega, d, d),
            alignment_strengths=strengths,
            alignment_error=alignment_error,
            shared_variance_ratio=shared_variance_ratio,
            sample_count=n,
            method=AlignmentMethod.procrustes,
        )

    @staticmethod
    def _extract_activation_matrix(
        crm: ConceptResponseMatrix,
        layer: int,
    ) -> list[list[float]] | None:
        layer_acts = crm.activations.get(layer)
        if layer_acts is None:
            return None
        sorted_anchors = sorted(layer_acts.keys())
        if not sorted_anchors:
            return None
        matrix: list[list[float]] = []
        for anchor_id in sorted_anchors:
            activation = layer_acts.get(anchor_id)
            if activation is not None:
                matrix.append(activation.activation)
        return matrix if matrix else None

    @staticmethod
    def _extract_activation_matrices(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        source_layer: int,
        target_layer: int,
        config: Config,
    ) -> tuple[list[list[float]], list[list[float]], list[float] | None] | None:
        source_layer_acts = source_crm.activations.get(source_layer)
        target_layer_acts = target_crm.activations.get(target_layer)
        if source_layer_acts is None or target_layer_acts is None:
            return None

        anchor_ids = SharedSubspaceProjector._select_anchor_ids(source_crm, target_crm, config)
        if not anchor_ids:
            return None

        source_matrix: list[list[float]] = []
        target_matrix: list[list[float]] = []
        weights: list[float] = []

        for anchor_id in anchor_ids:
            source_activation = source_layer_acts.get(anchor_id)
            target_activation = target_layer_acts.get(anchor_id)
            if source_activation is None or target_activation is None:
                continue
            source_matrix.append(source_activation.activation)
            target_matrix.append(target_activation.activation)
            weights.append(SharedSubspaceProjector._anchor_weight(anchor_id, config))

        if not source_matrix or not target_matrix:
            return None

        weight_payload = weights if config.anchor_weights or config.anchor_prefixes else None
        return source_matrix, target_matrix, weight_payload

    @staticmethod
    def _select_anchor_ids(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        config: Config,
    ) -> list[str]:
        source_ids = set(source_crm.anchor_metadata.anchor_ids)
        target_ids = set(target_crm.anchor_metadata.anchor_ids)
        common = source_ids.intersection(target_ids)
        if config.anchor_prefixes:
            prefixes = tuple(config.anchor_prefixes)
            common = {anchor_id for anchor_id in common if anchor_id.startswith(prefixes)}
        return sorted(common)

    @staticmethod
    def _anchor_weight(anchor_id: str, config: Config) -> float:
        if not config.anchor_weights:
            return 1.0
        weight = 1.0
        for prefix, value in config.anchor_weights.items():
            if anchor_id.startswith(prefix):
                weight = max(weight, float(value))
        return max(0.0, weight)

    @staticmethod
    def _center_matrix(
        matrix: list[list[float]],
        weights: list[float] | None = None,
    ) -> tuple[list[list[float]], list[float]]:
        if not matrix:
            return [], []
        n = len(matrix)
        d = len(matrix[0])
        if weights is None or len(weights) != n:
            means = [0.0 for _ in range(d)]
            for row in matrix:
                for j, val in enumerate(row):
                    means[j] += val
            for j in range(d):
                means[j] /= float(n)
            centered = []
            for row in matrix:
                centered.append([val - means[j] for j, val in enumerate(row)])
            return centered, means

        weight_sum = sum(max(0.0, float(value)) for value in weights)
        if weight_sum <= 0:
            return SharedSubspaceProjector._center_matrix(matrix, None)

        means = [0.0 for _ in range(d)]
        for idx, row in enumerate(matrix):
            weight = max(0.0, float(weights[idx])) / weight_sum
            for j, val in enumerate(row):
                means[j] += weight * val

        centered: list[list[float]] = []
        for idx, row in enumerate(matrix):
            weight = max(0.0, float(weights[idx])) / weight_sum
            scale = math.sqrt(weight) if weight > 0 else 0.0
            centered.append([(val - means[j]) * scale for j, val in enumerate(row)])
        return centered, means

    @staticmethod
    def _normalize_weights(weights: list[float] | None) -> np.ndarray | None:
        if not weights:
            return None
        values = np.array([max(0.0, float(value)) for value in weights], dtype=np.float32)
        total = float(values.sum())
        if total <= 0.0:
            return None
        return values / total

    @staticmethod
    def _center_array(
        array: np.ndarray,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if array.size == 0:
            return array, np.zeros((array.shape[1],), dtype=np.float32)
        if weights is None or weights.shape[0] != array.shape[0]:
            mean = np.mean(array, axis=0, dtype=np.float32)
            return array - mean, mean
        mean = np.sum(array * weights[:, None], axis=0, dtype=np.float32)
        centered = array - mean
        weighted = centered * np.sqrt(weights[:, None])
        return weighted, mean

    @staticmethod
    def _pca_reduce(
        matrix: np.ndarray,
        variance_threshold: float,
        max_components: int,
        mode: PcaMode,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if matrix.size == 0:
            return None, None, None
        n, d = matrix.shape
        if max_components <= 0:
            return None, None, None
        if isinstance(mode, str):
            normalized = mode.strip().lower()
            if normalized == "gram":
                mode = PcaMode.gram
            elif normalized == "svd":
                mode = PcaMode.svd
            else:
                mode = PcaMode.auto
        if mode == PcaMode.auto:
            # Gram-space PCA avoids forming d x d covariances when n << d.
            mode = PcaMode.gram if d > n else PcaMode.svd

        if mode == PcaMode.gram:
            gram = matrix @ matrix.T
            eigenvalues, eigenvectors = np.linalg.eigh(gram.astype(np.float32))
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
            denom = np.where(singular_values > SINGULAR_VALUE_FLOOR, singular_values, 1.0)
            components = matrix.T @ (eigenvectors / denom)
        else:
            _, singular_values, v_t = np.linalg.svd(matrix.astype(np.float32), full_matrices=False)
            components = v_t.T

        variances = singular_values * singular_values
        k = SharedSubspaceProjector._select_component_count(variances, variance_threshold)
        k = min(k, max_components, components.shape[1])
        if k <= 0:
            return None, None, None
        reduced = matrix @ components[:, :k]
        return reduced.astype(np.float32), components[:, :k].astype(np.float32), variances[:k].astype(np.float32)

    @staticmethod
    def _select_component_count(variances: np.ndarray, threshold: float) -> int:
        if variances.size == 0:
            return 0
        total = float(np.sum(variances))
        if total <= 0.0:
            return 0
        cumulative = 0.0
        for idx, value in enumerate(variances):
            cumulative += float(value)
            if cumulative / total >= threshold:
                return idx + 1
        return int(variances.size)

    @staticmethod
    def _regularize_covariance(cov: np.ndarray, epsilon: float) -> np.ndarray:
        if epsilon <= 0:
            return cov
        dim = cov.shape[0]
        return cov + (epsilon * np.eye(dim, dtype=np.float32))

    @staticmethod
    def _whiten_covariance(cov: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if cov.size == 0:
            return None, None
        eigenvalues, eigenvectors = np.linalg.eigh(cov.astype(np.float32))
        eigenvalues = np.maximum(eigenvalues, EIGENVALUE_FLOOR)
        inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        return inv_sqrt.astype(np.float32), eigenvalues.astype(np.float32)

    @staticmethod
    def _compute_covariance(x: list[list[float]], y: list[list[float]], n: int) -> list[float]:
        d_x = len(x[0])
        d_y = len(y[0])
        cov = [0.0 for _ in range(d_x * d_y)]
        for sample in range(n):
            for i in range(d_x):
                for j in range(d_y):
                    cov[i * d_y + j] += x[sample][i] * y[sample][j]
        scale = 1.0 / float(n)
        for idx in range(d_x * d_y):
            cov[idx] *= scale
        return cov

    @staticmethod
    def _compute_covariance_flat(
        x: list[float],
        y: list[float],
        n: int,
        d_x: int,
        d_y: int,
    ) -> list[float]:
        cov = [0.0 for _ in range(d_x * d_y)]
        for sample in range(n):
            for i in range(d_x):
                for j in range(d_y):
                    cov[i * d_y + j] += x[sample * d_x + i] * y[sample * d_y + j]
        scale = 1.0 / float(n)
        for idx in range(d_x * d_y):
            cov[idx] *= scale
        return cov

    @staticmethod
    def _compute_gram_matrix(x: list[list[float]], n: int, d: int) -> list[float]:
        gram = [0.0 for _ in range(n * n)]
        for i in range(n):
            for j in range(i, n):
                dot = 0.0
                for k in range(d):
                    dot += x[i][k] * x[j][k]
                gram[i * n + j] = dot
                gram[j * n + i] = dot
        return gram

    @staticmethod
    def _compute_whitening_transform(
        cov: list[float],
        dim: int,
    ) -> tuple[list[float], list[float]] | None:
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(cov, dim, max_iterations=100)
        if eigenvalues is None:
            return None
        eigen_float = [float(val) for val in eigenvalues]
        min_eigen = min([val for val in eigen_float if val > 1e-6], default=1e-6)

        inv_sqrt = [0.0 for _ in range(dim * dim)]
        for i in range(dim):
            diag_val = cov[i * dim + i]
            inv_sqrt[i * dim + i] = 1.0 / math.sqrt(max(diag_val, min_eigen))
        return inv_sqrt, eigen_float

    @staticmethod
    def _compute_eigenvectors(
        matrix: list[float],
        dim: int,
        k: int,
    ) -> list[float] | None:
        eigenvectors = [0.0 for _ in range(dim * k)]
        for j in range(k):
            for i in range(dim):
                eigenvectors[i * k + j] = 1.0 if i == j else 0.0

        for _ in range(50):
            new_vecs = [0.0 for _ in range(dim * k)]
            for i in range(dim):
                for j in range(k):
                    total = 0.0
                    for l in range(dim):
                        total += matrix[i * dim + l] * eigenvectors[l * k + j]
                    new_vecs[i * k + j] = total

            for j in range(k):
                norm = 0.0
                for i in range(dim):
                    norm += new_vecs[i * k + j] * new_vecs[i * k + j]
                norm = math.sqrt(max(norm, 1e-12))
                for i in range(dim):
                    eigenvectors[i * k + j] = new_vecs[i * k + j] / norm

            for j in range(1, k):
                for prev in range(j):
                    dot = 0.0
                    for i in range(dim):
                        dot += eigenvectors[i * k + j] * eigenvectors[i * k + prev]
                    for i in range(dim):
                        eigenvectors[i * k + j] -= dot * eigenvectors[i * k + prev]
                norm = 0.0
                for i in range(dim):
                    norm += eigenvectors[i * k + j] * eigenvectors[i * k + j]
                norm = math.sqrt(max(norm, 1e-12))
                for i in range(dim):
                    eigenvectors[i * k + j] /= norm

        return eigenvectors

    @staticmethod
    def _svd_decomposition(
        matrix: list[float],
        m: int,
        n: int,
    ) -> tuple[list[float], list[float], list[float]] | None:
        mtm = [0.0 for _ in range(n * n)]
        for i in range(n):
            for j in range(n):
                total = 0.0
                for k in range(m):
                    total += matrix[k * n + i] * matrix[k * n + j]
                mtm[i * n + j] = total

        k_dim = min(m, n)
        v_vecs = SharedSubspaceProjector._compute_eigenvectors(mtm, n, k_dim)
        if v_vecs is None:
            return None

        singular_values: list[float] = []
        u_vecs = [0.0 for _ in range(m * k_dim)]
        for j in range(k_dim):
            mvj = [0.0 for _ in range(m)]
            for i in range(m):
                total = 0.0
                for l in range(n):
                    total += matrix[i * n + l] * v_vecs[l * k_dim + j]
                mvj[i] = total
            sigma = 0.0
            for i in range(m):
                sigma += mvj[i] * mvj[i]
            sigma = math.sqrt(max(sigma, 1e-12))
            singular_values.append(sigma)
            for i in range(m):
                u_vecs[i * k_dim + j] = mvj[i] / sigma

        v_t = [0.0 for _ in range(k_dim * n)]
        for i in range(k_dim):
            for j in range(n):
                v_t[i * n + j] = v_vecs[j * k_dim + i]
        return u_vecs, singular_values, v_t

    @staticmethod
    def _matrix_multiply(
        a: list[list[float]],
        b: list[float],
        m: int,
        k: int,
        n: int,
    ) -> list[float]:
        result = [0.0 for _ in range(m * n)]
        for i in range(m):
            for j in range(n):
                total = 0.0
                for l in range(k):
                    total += a[i][l] * b[l * n + j]
                result[i * n + j] = total
        return result

    @staticmethod
    def _matrix_multiply_flat(
        a: list[float],
        b: list[float],
        m: int,
        k: int,
        n: int,
    ) -> list[float]:
        result = [0.0 for _ in range(m * n)]
        for i in range(m):
            for j in range(n):
                total = 0.0
                for l in range(k):
                    total += a[i * k + l] * b[l * n + j]
                result[i * n + j] = total
        return result

    @staticmethod
    def _truncate_columns(matrix: list[float], cols: int, total_cols: int) -> list[float]:
        rows = len(matrix) // total_cols
        result = [0.0 for _ in range(rows * cols)]
        for i in range(rows):
            for j in range(cols):
                result[i * cols + j] = matrix[i * total_cols + j]
        return result

    @staticmethod
    def _truncate_rows(matrix: list[float], rows: int, total_rows: int) -> list[float]:
        cols = len(matrix) // total_rows
        return list(matrix[: rows * cols])

    @staticmethod
    def _transpose_matrix(matrix: list[float], m: int, n: int) -> list[float]:
        result = [0.0 for _ in range(m * n)]
        for i in range(m):
            for j in range(n):
                result[j * m + i] = matrix[i * n + j]
        return result

    @staticmethod
    def _reshape_to_matrix(flat: list[float], rows: int, cols: int) -> list[list[float]]:
        result: list[list[float]] = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(flat[i * cols + j])
            result.append(row)
        return result

    @staticmethod
    def _compute_procrustes_error(source: list[float], target: list[float], n: int, k: int) -> float:
        error_sum = 0.0
        target_norm = 0.0
        for i in range(n * k):
            diff = source[i] - target[i]
            error_sum += diff * diff
            target_norm += target[i] * target[i]
        return math.sqrt(error_sum / target_norm) if target_norm > 0 else 0.0
