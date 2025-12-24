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

from dataclasses import dataclass

import math
import random

from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix


@dataclass(frozen=True)
class Config:
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    max_iterations: int = 1000
    forward_weight: float = 0.5
    backward_weight: float = 0.5
    convergence_threshold: float = 1e-5
    min_samples: int = 5
    use_momentum: bool = True
    momentum_coefficient: float = 0.9
    use_procrustes_warm_start: bool = True

    @staticmethod
    def default() -> "Config":
        return Config()


@dataclass(frozen=True)
class H4ValidationMetrics:
    forward_error: float
    backward_error: float
    converged: bool
    iterations: int
    transfer_quality: float

    @property
    def is_h4_validated(self) -> bool:
        return (
            self.forward_error < 0.15
            and self.backward_error < 0.15
            and self.transfer_quality > 0.85
            and self.converged
        )

    @property
    def summary(self) -> str:
        status = "PASS" if self.is_h4_validated else "FAIL"
        forward_ok = "OK" if self.forward_error < 0.15 else "FAIL"
        backward_ok = "OK" if self.backward_error < 0.15 else "FAIL"
        quality_ok = "OK" if self.transfer_quality > 0.85 else "FAIL"
        return (
            f"H4 Validation: {status}\n"
            f"- Forward Error: {self.forward_error:.3f} (target: <0.15) {forward_ok}\n"
            f"- Backward Error: {self.backward_error:.3f} (target: <0.15) {backward_ok}\n"
            f"- Transfer Quality: {self.transfer_quality * 100:.1f}% (target: >85%) {quality_ok}\n"
            f"- Converged: {'Yes' if self.converged else 'No'} in {self.iterations} iterations"
        )


@dataclass(frozen=True)
class Result:
    weights: list[list[float]]
    bias: list[float]
    loss_history: list[float]
    forward_error: float
    backward_error: float
    converged: bool
    iterations: int
    source_dimension: int
    target_dimension: int
    sample_count: int

    @property
    def is_valid(self) -> bool:
        return self.forward_error < 0.5 and self.backward_error < 0.5 and bool(self.weights)

    @property
    def h4_metrics(self) -> H4ValidationMetrics:
        transfer_quality = 1.0 - (self.forward_error + self.backward_error) / 2.0
        return H4ValidationMetrics(
            forward_error=self.forward_error,
            backward_error=self.backward_error,
            converged=self.converged,
            iterations=self.iterations,
            transfer_quality=transfer_quality,
        )


@dataclass(frozen=True)
class AnchorPair:
    source_activation: list[float]
    target_activation: list[float]
    anchor_id: str | None = None


class AffineStitchingLayer:
    @staticmethod
    def train(
        training_data: list[AnchorPair],
        config: Config = Config(),
    ) -> Result | None:
        if len(training_data) < config.min_samples:
            return None

        first = training_data[0]
        d_source = len(first.source_activation)
        d_target = len(first.target_activation)
        if d_source <= 0 or d_target <= 0:
            return None

        for pair in training_data:
            if len(pair.source_activation) != d_source or len(pair.target_activation) != d_target:
                return None

        n = len(training_data)
        source = [pair.source_activation for pair in training_data]
        target = [pair.target_activation for pair in training_data]

        if config.use_procrustes_warm_start and d_source == d_target:
            weights = AffineStitchingLayer._procrustes_initialization(source, target, d_source)
        else:
            scale = math.sqrt(2.0 / float(d_source + d_target))
            weights = [random.uniform(-scale, scale) for _ in range(d_target * d_source)]

        bias = [0.0 for _ in range(d_target)]
        weight_momentum = [0.0 for _ in range(d_target * d_source)]
        bias_momentum = [0.0 for _ in range(d_target)]

        loss_history: list[float] = []
        prev_loss = float("inf")
        converged = False
        iterations = 0

        for iter_idx in range(config.max_iterations):
            iterations = iter_idx + 1

            forward_preds = []
            for i in range(n):
                pred = [0.0 for _ in range(d_target)]
                for j in range(d_target):
                    total = 0.0
                    base = j * d_source
                    for k in range(d_source):
                        total += weights[base + k] * source[i][k]
                    pred[j] = total + bias[j]
                forward_preds.append(pred)

            backward_preds = []
            for i in range(n):
                pred = [0.0 for _ in range(d_source)]
                for k in range(d_source):
                    total = 0.0
                    for j in range(d_target):
                        total += weights[j * d_source + k] * target[i][j]
                    pred[k] = total
                backward_preds.append(pred)

            forward_loss = AffineStitchingLayer._compute_mse(forward_preds, target)
            backward_loss = AffineStitchingLayer._compute_mse(backward_preds, source)
            reg_loss = AffineStitchingLayer._compute_frobenius_norm_squared(weights) * config.weight_decay
            total_loss = (
                config.forward_weight * forward_loss
                + config.backward_weight * backward_loss
                + reg_loss
            )
            loss_history.append(total_loss)

            relative_change = abs(prev_loss - total_loss) / max(prev_loss, 1e-12)
            if relative_change < config.convergence_threshold and iter_idx > 10:
                converged = True
                break
            prev_loss = total_loss

            d_w = [0.0 for _ in range(d_target * d_source)]
            d_b = [0.0 for _ in range(d_target)]

            for i in range(n):
                for j in range(d_target):
                    error = forward_preds[i][j] - target[i][j]
                    base = j * d_source
                    for k in range(d_source):
                        d_w[base + k] += config.forward_weight * 2.0 / float(n) * error * source[i][k]
                    d_b[j] += config.forward_weight * 2.0 / float(n) * error

            for i in range(n):
                for k in range(d_source):
                    error = backward_preds[i][k] - source[i][k]
                    for j in range(d_target):
                        d_w[j * d_source + k] += (
                            config.backward_weight * 2.0 / float(n) * target[i][j] * error
                        )

            for idx in range(d_target * d_source):
                d_w[idx] += 2.0 * config.weight_decay * weights[idx]

            if config.use_momentum:
                for idx in range(d_target * d_source):
                    weight_momentum[idx] = (
                        config.momentum_coefficient * weight_momentum[idx]
                        + config.learning_rate * d_w[idx]
                    )
                    weights[idx] -= weight_momentum[idx]
                for j in range(d_target):
                    bias_momentum[j] = (
                        config.momentum_coefficient * bias_momentum[j]
                        + config.learning_rate * d_b[j]
                    )
                    bias[j] -= bias_momentum[j]
            else:
                for idx in range(d_target * d_source):
                    weights[idx] -= config.learning_rate * d_w[idx]
                for j in range(d_target):
                    bias[j] -= config.learning_rate * d_b[j]

        forward_error = AffineStitchingLayer._compute_reconstruction_error(
            weights=weights,
            bias=bias,
            source=source,
            target=target,
            d_source=d_source,
            d_target=d_target,
        )
        backward_error = AffineStitchingLayer._compute_backward_reconstruction_error(
            weights=weights,
            source=source,
            target=target,
            d_source=d_source,
            d_target=d_target,
        )
        weights_2d = AffineStitchingLayer._reshape_to_matrix(weights, d_target, d_source)

        return Result(
            weights=weights_2d,
            bias=bias,
            loss_history=loss_history,
            forward_error=forward_error,
            backward_error=backward_error,
            converged=converged,
            iterations=iterations,
            source_dimension=d_source,
            target_dimension=d_target,
            sample_count=n,
        )

    @staticmethod
    def train_from_crms(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        layer: int,
        config: Config = Config(),
    ) -> Result | None:
        source_layer = source_crm.activations.get(layer)
        target_layer = target_crm.activations.get(layer)
        if source_layer is None or target_layer is None:
            return None

        common = set(source_layer.keys()).intersection(set(target_layer.keys()))
        if len(common) < config.min_samples:
            return None

        training_data: list[AnchorPair] = []
        for anchor_id in sorted(common):
            source_act = source_layer.get(anchor_id)
            target_act = target_layer.get(anchor_id)
            if source_act is None or target_act is None:
                continue
            training_data.append(
                AnchorPair(
                    source_activation=source_act.activation,
                    target_activation=target_act.activation,
                    anchor_id=anchor_id,
                )
            )

        return AffineStitchingLayer.train(training_data, config=config)

    @staticmethod
    def apply(
        activations: list[list[float]],
        weights: list[list[float]],
        bias: list[float],
    ) -> list[list[float]]:
        if not activations or not weights:
            return []

        d_target = len(weights)
        d_source = len(weights[0])
        transformed: list[list[float]] = []
        for activation in activations:
            if len(activation) != d_source:
                continue
            out = [0.0 for _ in range(d_target)]
            for j in range(d_target):
                total = 0.0
                for k in range(d_source):
                    total += weights[j][k] * activation[k]
                out[j] = total + bias[j]
            transformed.append(out)
        return transformed

    @staticmethod
    def apply_single(
        activation: list[float],
        weights: list[list[float]],
        bias: list[float],
    ) -> list[float] | None:
        result = AffineStitchingLayer.apply([activation], weights, bias)
        return result[0] if result else None

    @staticmethod
    def apply_inverse(
        activations: list[list[float]],
        weights: list[list[float]],
    ) -> list[list[float]]:
        if not activations or not weights:
            return []

        d_target = len(weights)
        d_source = len(weights[0])
        reconstructed: list[list[float]] = []
        for activation in activations:
            if len(activation) != d_target:
                continue
            out = [0.0 for _ in range(d_source)]
            for k in range(d_source):
                total = 0.0
                for j in range(d_target):
                    total += weights[j][k] * activation[j]
                out[k] = total
            reconstructed.append(out)
        return reconstructed

    @staticmethod
    def _procrustes_initialization(
        source: list[list[float]],
        target: list[list[float]],
        size: int,
    ) -> list[float]:
        n = len(source)
        matrix = [0.0 for _ in range(size * size)]
        for i in range(size):
            for j in range(size):
                total = 0.0
                for sample in range(n):
                    total += source[sample][i] * target[sample][j]
                matrix[i * size + j] = total

        svd = AffineStitchingLayer._simple_svd(matrix, size)
        if svd is None:
            identity = [0.0 for _ in range(size * size)]
            for i in range(size):
                identity[i * size + i] = 1.0
            return identity

        u, _, v_t = svd
        omega = [0.0 for _ in range(size * size)]
        for i in range(size):
            for j in range(size):
                total = 0.0
                for k in range(size):
                    total += u[i * size + k] * v_t[k * size + j]
                omega[i * size + j] = total
        return omega

    @staticmethod
    def _simple_svd(
        matrix: list[float],
        size: int,
    ) -> tuple[list[float], list[float], list[float]] | None:
        if size <= 0:
            return None

        mtm = [0.0 for _ in range(size * size)]
        for i in range(size):
            for j in range(size):
                total = 0.0
                for k in range(size):
                    total += matrix[k * size + i] * matrix[k * size + j]
                mtm[i * size + j] = total

        v = [0.0 for _ in range(size * size)]
        for i in range(size):
            v[i * size + i] = 1.0

        for _ in range(30):
            w = [0.0 for _ in range(size * size)]
            for i in range(size):
                for j in range(size):
                    total = 0.0
                    for k in range(size):
                        total += mtm[i * size + k] * v[k * size + j]
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
                for k in range(size):
                    total += matrix[i * size + k] * v[k * size + j]
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
    def _compute_mse(predicted: list[list[float]], actual: list[list[float]]) -> float:
        total = 0.0
        count = 0
        for pred, act in zip(predicted, actual):
            for p, a in zip(pred, act):
                diff = p - a
                total += diff * diff
                count += 1
        return total / float(count) if count > 0 else 0.0

    @staticmethod
    def _compute_frobenius_norm_squared(matrix: list[float]) -> float:
        return sum(value * value for value in matrix)

    @staticmethod
    def _compute_reconstruction_error(
        weights: list[float],
        bias: list[float],
        source: list[list[float]],
        target: list[list[float]],
        d_source: int,
        d_target: int,
    ) -> float:
        error_sum = 0.0
        target_norm = 0.0
        for i in range(len(source)):
            for j in range(d_target):
                pred = 0.0
                base = j * d_source
                for k in range(d_source):
                    pred += weights[base + k] * source[i][k]
                pred += bias[j]
                diff = pred - target[i][j]
                error_sum += diff * diff
                target_norm += target[i][j] * target[i][j]
        return math.sqrt(error_sum / target_norm) if target_norm > 0 else 0.0

    @staticmethod
    def _compute_backward_reconstruction_error(
        weights: list[float],
        source: list[list[float]],
        target: list[list[float]],
        d_source: int,
        d_target: int,
    ) -> float:
        error_sum = 0.0
        source_norm = 0.0
        for i in range(len(source)):
            for k in range(d_source):
                pred = 0.0
                for j in range(d_target):
                    pred += weights[j * d_source + k] * target[i][j]
                diff = pred - source[i][k]
                error_sum += diff * diff
                source_norm += source[i][k] * source[i][k]
        return math.sqrt(error_sum / source_norm) if source_norm > 0 else 0.0

    @staticmethod
    def _reshape_to_matrix(flat: list[float], rows: int, cols: int) -> list[list[float]]:
        result: list[list[float]] = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(flat[i * cols + j])
            result.append(row)
        return result
