from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math

from modelcypher.core.domain.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry_fingerprint import GeometricFingerprint


class AlignmentMethod(str, Enum):
    cca = "cca"
    shared_svd = "sharedSVD"
    procrustes = "procrustes"


@dataclass(frozen=True)
class Config:
    alignment_method: AlignmentMethod = AlignmentMethod.cca
    variance_threshold: float = 0.95
    max_shared_dimension: int = 256
    cca_regularization: float = 1e-4
    min_samples: int = 10

    @staticmethod
    def default() -> "Config":
        return Config()


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
        config: Config = Config(),
    ) -> Optional[Result]:
        source_matrix = SharedSubspaceProjector._extract_activation_matrix(source_crm, layer)
        target_matrix = SharedSubspaceProjector._extract_activation_matrix(target_crm, layer)
        if source_matrix is None or target_matrix is None:
            return None

        if len(source_matrix) != len(target_matrix) or len(source_matrix) < config.min_samples:
            return None

        n = len(source_matrix)
        d_source = len(source_matrix[0])
        d_target = len(target_matrix[0])

        if config.alignment_method == AlignmentMethod.cca:
            return SharedSubspaceProjector._discover_with_cca(
                source_matrix, target_matrix, n, d_source, d_target, config
            )
        if config.alignment_method == AlignmentMethod.shared_svd:
            return SharedSubspaceProjector._discover_with_shared_svd(
                source_matrix, target_matrix, n, d_source, d_target, config
            )
        return SharedSubspaceProjector._discover_with_procrustes(
            source_matrix, target_matrix, n, d_source, d_target, config
        )

    @staticmethod
    def _discover_with_cca(
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        n: int,
        d_source: int,
        d_target: int,
        config: Config,
    ) -> Optional[Result]:
        centered_source, _ = SharedSubspaceProjector._center_matrix(source_activations)
        centered_target, _ = SharedSubspaceProjector._center_matrix(target_activations)

        epsilon = config.cca_regularization
        css_raw = SharedSubspaceProjector._compute_covariance(centered_source, centered_source, n)
        ctt_raw = SharedSubspaceProjector._compute_covariance(centered_target, centered_target, n)
        cst = SharedSubspaceProjector._compute_covariance(centered_source, centered_target, n)

        css = list(css_raw)
        ctt = list(ctt_raw)
        for i in range(d_source):
            css[i * d_source + i] += epsilon
        for i in range(d_target):
            ctt[i * d_target + i] += epsilon

        whitening_source = SharedSubspaceProjector._compute_whitening_transform(css, d_source)
        whitening_target = SharedSubspaceProjector._compute_whitening_transform(ctt, d_target)
        if whitening_source is None or whitening_target is None:
            return SharedSubspaceProjector._discover_with_shared_svd(
                source_activations, target_activations, n, d_source, d_target, config
            )
        css_inv_sqrt, css_eigenvalues = whitening_source
        ctt_inv_sqrt, ctt_eigenvalues = whitening_target

        source_whitened = SharedSubspaceProjector._matrix_multiply(
            centered_source, css_inv_sqrt, n, d_source, d_source
        )
        target_whitened = SharedSubspaceProjector._matrix_multiply(
            centered_target, ctt_inv_sqrt, n, d_target, d_target
        )

        cross_cov = SharedSubspaceProjector._compute_covariance_flat(
            source_whitened, target_whitened, n, d_source, d_target
        )

        svd = SharedSubspaceProjector._svd_decomposition(cross_cov, d_source, d_target)
        if svd is None:
            return SharedSubspaceProjector._discover_with_shared_svd(
                source_activations, target_activations, n, d_source, d_target, config
            )
        u, singular_values, v_t = svd

        canonical = [min(1.0, max(0.0, val)) for val in singular_values]
        shared_dim = 0
        cum_variance = 0.0
        total_variance = sum(canonical)
        for idx, corr in enumerate(canonical):
            if corr <= 0.1:
                continue
            cum_variance += corr
            shared_dim = idx + 1
            if total_variance > 0 and (cum_variance / total_variance) >= config.variance_threshold:
                break
        shared_dim = min(shared_dim, config.max_shared_dimension)
        if shared_dim <= 0:
            return None

        u_truncated = SharedSubspaceProjector._truncate_columns(u, shared_dim, min(d_source, d_target))
        v_truncated = SharedSubspaceProjector._truncate_rows(v_t, shared_dim, min(d_source, d_target))

        source_projection = SharedSubspaceProjector._matrix_multiply_flat(
            css_inv_sqrt, u_truncated, d_source, d_source, shared_dim
        )
        target_projection = SharedSubspaceProjector._matrix_multiply_flat(
            ctt_inv_sqrt,
            SharedSubspaceProjector._transpose_matrix(v_truncated, shared_dim, d_target),
            d_target,
            d_target,
            shared_dim,
        )

        source_projected = SharedSubspaceProjector._matrix_multiply(
            centered_source, source_projection, n, d_source, shared_dim
        )
        target_projected = SharedSubspaceProjector._matrix_multiply(
            centered_target, target_projection, n, d_target, shared_dim
        )

        alignment_error = SharedSubspaceProjector._compute_procrustes_error(
            source_projected, target_projected, n, shared_dim
        )

        source_variance_total = sum(css_eigenvalues)
        target_variance_total = sum(ctt_eigenvalues)
        shared_variance = sum(canonical[:shared_dim])
        avg_total_variance = (source_variance_total + target_variance_total) / 2.0
        shared_variance_ratio = shared_variance / float(shared_dim) if avg_total_variance > 0 else 0.0

        return Result(
            shared_dimension=shared_dim,
            source_dimension=d_source,
            target_dimension=d_target,
            source_projection=SharedSubspaceProjector._reshape_to_matrix(source_projection, d_source, shared_dim),
            target_projection=SharedSubspaceProjector._reshape_to_matrix(target_projection, d_target, shared_dim),
            alignment_strengths=canonical[:shared_dim],
            alignment_error=alignment_error,
            shared_variance_ratio=shared_variance_ratio,
            sample_count=n,
            method=AlignmentMethod.cca,
        )

    @staticmethod
    def _discover_with_shared_svd(
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        n: int,
        d_source: int,
        d_target: int,
        config: Config,
    ) -> Optional[Result]:
        centered_source, _ = SharedSubspaceProjector._center_matrix(source_activations)
        centered_target, _ = SharedSubspaceProjector._center_matrix(target_activations)

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
        n: int,
        d_source: int,
        d_target: int,
        config: Config,
    ) -> Optional[Result]:
        if d_source != d_target:
            return SharedSubspaceProjector._discover_with_cca(
                source_activations, target_activations, n, d_source, d_target, config
            )

        d = d_source
        centered_source, _ = SharedSubspaceProjector._center_matrix(source_activations)
        centered_target, _ = SharedSubspaceProjector._center_matrix(target_activations)

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
    ) -> Optional[list[list[float]]]:
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
    def _center_matrix(matrix: list[list[float]]) -> tuple[list[list[float]], list[float]]:
        if not matrix:
            return [], []
        n = len(matrix)
        d = len(matrix[0])
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
    ) -> Optional[tuple[list[float], list[float]]]:
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
    ) -> Optional[list[float]]:
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
    ) -> Optional[tuple[list[float], list[float], list[float]]]:
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
