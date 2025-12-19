from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from modelcypher.core.domain.concept_response_matrix import ConceptResponseMatrix


@dataclass(frozen=True)
class Config:
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    allow_reflections: bool = False
    min_models: int = 2
    allow_scaling: bool = False

    @staticmethod
    def default() -> "Config":
        return Config()


@dataclass(frozen=True)
class Result:
    consensus: list[list[float]]
    rotations: list[list[list[float]]]
    scales: list[float]
    residuals: list[list[list[float]]]
    converged: bool
    iterations: int
    alignment_error: float
    per_model_errors: list[float]
    consensus_variance_ratio: float
    sample_count: int
    dimension: int
    model_count: int

    @property
    def summary(self) -> str:
        return (
            "Generalized Procrustes Analysis\n"
            f"- Models: {self.model_count}\n"
            f"- Samples: {self.sample_count} x {self.dimension}\n"
            f"- Converged: {self.converged} (iterations: {self.iterations})\n"
            f"- Alignment Error: {self.alignment_error:.4f}\n"
            f"- Consensus Variance: {self.consensus_variance_ratio * 100:.1f}%"
        )


class GeneralizedProcrustes:
    @staticmethod
    def align(
        activations: list[list[list[float]]],
        config: Config = Config(),
    ) -> Optional[Result]:
        model_count = len(activations)
        if model_count < config.min_models:
            return None

        first = activations[0] if activations else None
        if not first:
            return None
        n = len(first)
        if n <= 0:
            return None
        k = len(first[0])
        if k <= 0:
            return None

        for matrix in activations:
            if len(matrix) != n:
                return None
            for row in matrix:
                if len(row) != k:
                    return None

        centered = [GeneralizedProcrustes._center_matrix(matrix) for matrix in activations]

        scales = [1.0 for _ in range(model_count)]
        if config.allow_scaling:
            for idx, matrix in enumerate(centered):
                norm = GeneralizedProcrustes._frobenius_norm(matrix)
                if norm > 1e-12:
                    scales[idx] = norm
                    centered[idx] = GeneralizedProcrustes._scale_matrix(matrix, 1.0 / norm)

        consensus = GeneralizedProcrustes._compute_mean(centered)
        rotations = [GeneralizedProcrustes._identity_matrix(k) for _ in range(model_count)]
        aligned = list(centered)

        prev_error = float("inf")
        iterations = 0
        converged = False

        for iter_idx in range(config.max_iterations):
            iterations = iter_idx + 1
            for model_idx in range(model_count):
                rotation, _ = GeneralizedProcrustes._procrustes_alignment(
                    source=aligned[model_idx],
                    target=consensus,
                    allow_reflection=config.allow_reflections,
                )
                rotations[model_idx] = rotation
                aligned[model_idx] = GeneralizedProcrustes._apply_rotation(
                    centered[model_idx],
                    rotation=rotation,
                )

            consensus = GeneralizedProcrustes._compute_mean(aligned)
            current_error = GeneralizedProcrustes._compute_total_error(aligned, consensus)

            relative_change = abs(prev_error - current_error) / max(prev_error, 1e-12)
            if relative_change < config.convergence_threshold:
                converged = True
                break
            prev_error = current_error

        final_error = GeneralizedProcrustes._compute_total_error(aligned, consensus)
        per_model_errors = [GeneralizedProcrustes._compute_model_error(m, consensus) for m in aligned]
        residuals = [GeneralizedProcrustes._compute_residuals(m, consensus) for m in aligned]
        consensus_ratio = GeneralizedProcrustes._compute_consensus_variance_ratio(aligned, consensus)
        rotation_2d = [GeneralizedProcrustes._reshape_to_matrix(r, k) for r in rotations]

        return Result(
            consensus=consensus,
            rotations=rotation_2d,
            scales=scales,
            residuals=residuals,
            converged=converged,
            iterations=iterations,
            alignment_error=final_error,
            per_model_errors=per_model_errors,
            consensus_variance_ratio=consensus_ratio,
            sample_count=n,
            dimension=k,
            model_count=model_count,
        )

    @staticmethod
    def align_crms(
        crms: list[ConceptResponseMatrix],
        layer: int,
        config: Config = Config(),
    ) -> Optional[Result]:
        activations: list[list[list[float]]] = []
        for crm in crms:
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
            if matrix:
                activations.append(matrix)

        if not activations:
            return None

        n = len(activations[0])
        for matrix in activations:
            if len(matrix) != n:
                return None

        dimensions = [len(matrix[0]) for matrix in activations]
        if len(set(dimensions)) > 1:
            min_dim = min(dimensions)
            projected = []
            for matrix in activations:
                d = len(matrix[0])
                if d == min_dim:
                    projected.append(matrix)
                else:
                    projected.append([row[:min_dim] for row in matrix])
            return GeneralizedProcrustes.align(projected, config=config)

        return GeneralizedProcrustes.align(activations, config=config)

    @staticmethod
    def _center_matrix(matrix: list[list[float]]) -> list[list[float]]:
        if not matrix:
            return []
        n = len(matrix)
        k = len(matrix[0])
        means = [0.0 for _ in range(k)]
        for row in matrix:
            for j, val in enumerate(row):
                means[j] += val
        for j in range(k):
            means[j] /= float(n)
        return [[val - means[j] for j, val in enumerate(row)] for row in matrix]

    @staticmethod
    def _frobenius_norm(matrix: list[list[float]]) -> float:
        total = 0.0
        for row in matrix:
            for val in row:
                total += val * val
        return math.sqrt(total)

    @staticmethod
    def _scale_matrix(matrix: list[list[float]], factor: float) -> list[list[float]]:
        return [[val * factor for val in row] for row in matrix]

    @staticmethod
    def _compute_mean(matrices: list[list[list[float]]]) -> list[list[float]]:
        if not matrices:
            return []
        n = len(matrices[0])
        k = len(matrices[0][0])
        count = float(len(matrices))
        mean: list[list[float]] = []
        for i in range(n):
            row = [0.0 for _ in range(k)]
            for matrix in matrices:
                for j in range(k):
                    row[j] += matrix[i][j]
            for j in range(k):
                row[j] /= count
            mean.append(row)
        return mean

    @staticmethod
    def _identity_matrix(size: int) -> list[float]:
        result = [0.0 for _ in range(size * size)]
        for i in range(size):
            result[i * size + i] = 1.0
        return result

    @staticmethod
    def _procrustes_alignment(
        source: list[list[float]],
        target: list[list[float]],
        allow_reflection: bool,
    ) -> tuple[list[float], float]:
        n = len(source)
        k = len(source[0])
        m = [0.0 for _ in range(k * k)]
        for i in range(k):
            for j in range(k):
                total = 0.0
                for sample in range(n):
                    total += source[sample][i] * target[sample][j]
                m[i * k + j] = total

        svd = GeneralizedProcrustes._simple_svd(m, k)
        if svd is None:
            return GeneralizedProcrustes._identity_matrix(k), float("inf")
        u, _, v_t = svd

        omega = [0.0 for _ in range(k * k)]
        for i in range(k):
            for j in range(k):
                total = 0.0
                for l in range(k):
                    total += u[i * k + l] * v_t[l * k + j]
                omega[i * k + j] = total

        if not allow_reflection:
            det = GeneralizedProcrustes._determinant(omega, k)
            if det < 0:
                for i in range(k):
                    u[i * k + (k - 1)] *= -1.0
                for i in range(k):
                    for j in range(k):
                        total = 0.0
                        for l in range(k):
                            total += u[i * k + l] * v_t[l * k + j]
                        omega[i * k + j] = total

        rotated = GeneralizedProcrustes._apply_rotation(source, omega)
        error = GeneralizedProcrustes._compute_model_error(rotated, target)
        return omega, error

    @staticmethod
    def _apply_rotation(matrix: list[list[float]], rotation: list[float]) -> list[list[float]]:
        n = len(matrix)
        k = len(matrix[0])
        result: list[list[float]] = []
        for i in range(n):
            row = [0.0 for _ in range(k)]
            for j in range(k):
                total = 0.0
                for l in range(k):
                    total += matrix[i][l] * rotation[l * k + j]
                row[j] = total
            result.append(row)
        return result

    @staticmethod
    def _compute_total_error(
        aligned: list[list[list[float]]],
        consensus: list[list[float]],
    ) -> float:
        total = 0.0
        for matrix in aligned:
            total += GeneralizedProcrustes._compute_model_error(matrix, consensus)
        return total

    @staticmethod
    def _compute_model_error(matrix: list[list[float]], consensus: list[list[float]]) -> float:
        total = 0.0
        for row, cons_row in zip(matrix, consensus):
            for val, cons_val in zip(row, cons_row):
                diff = val - cons_val
                total += diff * diff
        return total

    @staticmethod
    def _compute_residuals(
        matrix: list[list[float]],
        consensus: list[list[float]],
    ) -> list[list[float]]:
        residuals: list[list[float]] = []
        for row, cons_row in zip(matrix, consensus):
            residuals.append([val - cons_val for val, cons_val in zip(row, cons_row)])
        return residuals

    @staticmethod
    def _compute_consensus_variance_ratio(
        aligned: list[list[list[float]]],
        consensus: list[list[float]],
    ) -> float:
        total_var = 0.0
        for matrix in aligned:
            for row in matrix:
                for val in row:
                    total_var += val * val
        residual_var = 0.0
        for matrix in aligned:
            residual_var += GeneralizedProcrustes._compute_model_error(matrix, consensus)
        if total_var <= 1e-12:
            return 0.0
        return 1.0 - (residual_var / total_var)

    @staticmethod
    def _reshape_to_matrix(flat: list[float], size: int) -> list[list[float]]:
        result: list[list[float]] = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(flat[i * size + j])
            result.append(row)
        return result

    @staticmethod
    def _simple_svd(
        matrix: list[float],
        size: int,
    ) -> Optional[tuple[list[float], list[float], list[float]]]:
        mtm = [0.0 for _ in range(size * size)]
        for i in range(size):
            for j in range(size):
                total = 0.0
                for l in range(size):
                    total += matrix[l * size + i] * matrix[l * size + j]
                mtm[i * size + j] = total

        v = [0.0 for _ in range(size * size)]
        for i in range(size):
            v[i * size + i] = 1.0

        for _ in range(50):
            w = [0.0 for _ in range(size * size)]
            for i in range(size):
                for j in range(size):
                    total = 0.0
                    for l in range(size):
                        total += mtm[i * size + l] * v[l * size + j]
                    w[i * size + j] = total

            for j in range(size):
                for prev in range(j):
                    dot = 0.0
                    for i in range(size):
                        dot += w[i * size + j] * v[i * size + prev]
                    for i in range(size):
                        w[i * size + j] -= dot * v[i * size + prev]
                norm = 0.0
                for i in range(size):
                    norm += w[i * size + j] * w[i * size + j]
                norm = math.sqrt(max(norm, 1e-12))
                for i in range(size):
                    v[i * size + j] = w[i * size + j] / norm

        singular_values: list[float] = []
        u = [0.0 for _ in range(size * size)]
        for j in range(size):
            mv = [0.0 for _ in range(size)]
            for i in range(size):
                total = 0.0
                for l in range(size):
                    total += matrix[i * size + l] * v[l * size + j]
                mv[i] = total
            sigma = 0.0
            for i in range(size):
                sigma += mv[i] * mv[i]
            sigma = math.sqrt(max(sigma, 1e-12))
            singular_values.append(sigma)
            for i in range(size):
                u[i * size + j] = mv[i] / sigma

        v_t = [0.0 for _ in range(size * size)]
        for i in range(size):
            for j in range(size):
                v_t[i * size + j] = v[j * size + i]
        return u, singular_values, v_t

    @staticmethod
    def _determinant(matrix: list[float], size: int) -> float:
        if size == 1:
            return matrix[0]
        if size == 2:
            return matrix[0] * matrix[3] - matrix[1] * matrix[2]
        if size == 3:
            return (
                matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
                - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
                + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
            )
        product = 1.0
        for i in range(size):
            product *= matrix[i * size + i]
        return 1.0 if product > 0 else -1.0
