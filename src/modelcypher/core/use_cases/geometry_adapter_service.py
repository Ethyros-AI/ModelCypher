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

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modelcypher.core.domain.geometry import ChangeType, DoRAConfiguration, DoRADecomposition
from modelcypher.core.domain.geometry.dare_sparsity import DARESparsityAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterWeights:
    weights: dict[str, np.ndarray]
    scale: float


class GeometryAdapterService:
    def analyze_dare(
        self, checkpoint_path: str, base_path: str | None = None
    ) -> DARESparsityAnalyzer.SparsityAnalysis:
        # Use MLX GPU-accelerated version for performance
        deltas_mlx = self._compute_deltas_mlx(checkpoint_path, base_path)
        if not deltas_mlx:
            raise ValueError("No adapter delta weights found for DARE analysis")
        return DARESparsityAnalyzer.analyze_mlx(delta_weights=deltas_mlx)

    def analyze_dora(
        self,
        checkpoint_path: str,
        base_path: str | None = None,
    ):
        import mlx.core as mx

        base_vectors, current_vectors = self._compute_base_and_current(checkpoint_path, base_path)
        if not base_vectors or not current_vectors:
            raise ValueError(
                "Unable to derive base/current weights for DoRA decomposition. "
                "For LoRA adapters, ensure the --base model is compatible with the adapter "
                "(same architecture and layer count as the model the adapter was trained on)."
            )
        base_mx = {k: mx.array(v) for k, v in base_vectors.items()}
        current_mx = {k: mx.array(v) for k, v in current_vectors.items()}
        decomposer = DoRADecomposition()
        return decomposer.analyze_adapter(base_weights=base_mx, current_weights=current_mx)

    def _compute_deltas(
        self,
        checkpoint_path: str,
        base_path: str | None,
    ) -> tuple[dict[str, list[float]], float]:
        checkpoint = self._load_weights(checkpoint_path)
        deltas = self._lora_deltas_from_weights(checkpoint.weights, checkpoint.scale)
        if deltas:
            return deltas, checkpoint.scale

        if base_path is None:
            return {}, checkpoint.scale

        base = self._load_weights(base_path)
        delta_vectors: dict[str, list[float]] = {}
        for key, current in checkpoint.weights.items():
            base_weight = base.weights.get(key)
            if base_weight is None or base_weight.shape != current.shape:
                continue
            delta = (current.astype(np.float32) - base_weight.astype(np.float32)).ravel()
            delta_vectors[key] = delta.tolist()
        return delta_vectors, checkpoint.scale

    def _compute_deltas_mlx(
        self,
        checkpoint_path: str,
        base_path: str | None,
    ) -> dict[str, "mx.array"]:
        """Compute LoRA deltas as MLX arrays for GPU processing."""
        import mlx.core as mx

        checkpoint = self._load_weights_mlx(checkpoint_path)
        scale = 1.0
        if "lora_scale" in checkpoint:
            scale = float(checkpoint["lora_scale"].reshape(-1)[0])
        deltas = self._lora_deltas_mlx(checkpoint, scale)
        if deltas:
            return deltas

        if base_path is None:
            return {}

        base = self._load_weights_mlx(base_path)
        delta_arrays: dict[str, mx.array] = {}
        for key, current in checkpoint.items():
            if key == "lora_scale":
                continue
            base_weight = base.get(key)
            if base_weight is None or base_weight.shape != current.shape:
                continue
            delta = current.astype(mx.float32) - base_weight.astype(mx.float32)
            delta_arrays[key] = delta
        return delta_arrays

    def _load_weights_mlx(self, path: str) -> dict[str, "mx.array"]:
        """Load weights directly as MLX arrays (stays on GPU)."""
        import mlx.core as mx

        resolved = Path(path).expanduser().resolve()
        weight_path = self._resolve_weight_path(resolved)
        weights = mx.load(str(weight_path))
        mx.eval(weights)  # Ensure loaded to GPU
        return weights

    def _lora_deltas_mlx(
        self,
        weights: dict[str, "mx.array"],
        scale: float,
    ) -> dict[str, "mx.array"]:
        """Compute LoRA deltas as MLX arrays on GPU."""
        import mlx.core as mx

        a_by_prefix: dict[str, mx.array] = {}
        b_by_prefix: dict[str, mx.array] = {}

        for key, value in weights.items():
            lowered = key.lower()
            if lowered.endswith("lora_a"):
                prefix = key[: -len("lora_a")].rstrip(".")
                prefix = prefix if prefix else "W"
                a_by_prefix[prefix] = value.astype(mx.float32)
            elif lowered.endswith("lora_b"):
                prefix = key[: -len("lora_b")].rstrip(".")
                prefix = prefix if prefix else "W"
                b_by_prefix[prefix] = value.astype(mx.float32)

        deltas: dict[str, mx.array] = {}
        for prefix, a in a_by_prefix.items():
            b = b_by_prefix.get(prefix)
            if b is None:
                continue
            # LoRA delta = B @ A where B is [out, rank] and A is [rank, in]
            # But MLX adapters store A as [in, rank] and B as [rank, out]
            # So we need A @ B to get [in, out] or transpose for [out, in]
            a_shape = tuple(a.shape)
            b_shape = tuple(b.shape)

            # Try A @ B first (standard convention for MLX adapters)
            if a_shape[1] == b_shape[0]:
                # A: [in, rank], B: [rank, out] -> A @ B = [in, out]
                delta = a @ b
            elif a_shape[0] == b_shape[1]:
                # Transposed: B.T @ A.T = [out, rank] @ [rank, in] = [out, in]
                delta = b.T @ a.T
            else:
                continue

            if scale != 1.0:
                delta = delta * scale
            deltas[prefix] = delta

        mx.eval(deltas)  # Force GPU computation
        return deltas

    def _compute_base_and_current(
        self,
        checkpoint_path: str,
        base_path: str | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        checkpoint = self._load_weights(checkpoint_path)
        base = self._load_weights(base_path) if base_path else None

        deltas = self._lora_deltas_from_weights(checkpoint.weights, checkpoint.scale)
        if deltas:
            if base is None:
                # DoRA analysis requires a base model for LoRA adapters
                return {}, {}

            base_weights = base.weights
            base_vectors: dict[str, list[float]] = {}
            current_vectors: dict[str, list[float]] = {}

            for prefix, delta_values in deltas.items():
                # LoRA prefix -> base weight key mapping
                # Try common patterns: prefix.weight, prefix, prefix + .weight
                base_key = None
                for candidate in [f"{prefix}.weight", prefix, f"{prefix}weight"]:
                    if candidate in base_weights:
                        base_key = candidate
                        break

                if base_key is None:
                    continue

                base_weight = base_weights[base_key]
                delta_arr = np.array(delta_values, dtype=np.float32)

                # Check if shapes are compatible
                expected_size = base_weight.size
                if delta_arr.size != expected_size:
                    # Shapes don't match - skip this layer
                    continue

                delta = delta_arr.reshape(base_weight.shape)
                current = base_weight.astype(np.float32) + delta
                base_vectors[prefix] = base_weight.astype(np.float32).ravel().tolist()
                current_vectors[prefix] = current.ravel().tolist()

            return base_vectors, current_vectors

        if base is None:
            return {}, {}

        base_vectors: dict[str, list[float]] = {}
        current_vectors: dict[str, list[float]] = {}
        for key, current in checkpoint.weights.items():
            base_weight = base.weights.get(key)
            if base_weight is None or base_weight.shape != current.shape:
                continue
            base_vectors[key] = base_weight.astype(np.float32).ravel().tolist()
            current_vectors[key] = current.astype(np.float32).ravel().tolist()

        return base_vectors, current_vectors

    def _load_weights(self, path: str | None) -> AdapterWeights:
        if path is None:
            raise ValueError("Base path is required for this analysis")

        resolved = Path(path).expanduser().resolve()
        weight_path = self._resolve_weight_path(resolved)
        if weight_path.suffix == ".npz":
            data = np.load(weight_path)
            weights = {key: np.array(value) for key, value in data.items()}
        elif weight_path.suffix == ".safetensors":
            # Use MLX to load safetensors as it handles bfloat16 natively,
            # then convert to numpy float32 for analysis
            import mlx.core as mx

            mx_weights = mx.load(str(weight_path))
            weights = {
                key: np.array(value.astype(mx.float32))
                for key, value in mx_weights.items()
            }
        else:
            raise ValueError(f"Unsupported adapter format: {weight_path.suffix}")

        scale = 1.0
        if "lora_scale" in weights:
            scale = float(np.array(weights["lora_scale"]).reshape(-1)[0])

        return AdapterWeights(weights=weights, scale=scale)

    def _resolve_weight_path(self, path: Path) -> Path:
        if path.is_dir():
            candidates = [
                path / "weights.npz",
                path / "adapters.safetensors",
                path / "adapter_model.safetensors",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            npz_files = sorted(path.glob("*.npz"))
            if npz_files:
                return npz_files[0]
            tensor_files = sorted(path.glob("*.safetensors"))
            if tensor_files:
                return tensor_files[0]
        if not path.exists():
            raise ValueError(f"Checkpoint not found: {path}")
        return path

    def _lora_deltas_from_weights(
        self,
        weights: dict[str, np.ndarray],
        scale: float,
    ) -> dict[str, list[float]]:
        a_by_prefix: dict[str, np.ndarray] = {}
        b_by_prefix: dict[str, np.ndarray] = {}

        for key, value in weights.items():
            lowered = key.lower()
            if lowered.endswith("lora_a"):
                prefix = key[: -len("lora_a")].rstrip(".")
                prefix = prefix if prefix else "W"
                a_by_prefix[prefix] = np.array(value)
            elif lowered.endswith("lora_b"):
                prefix = key[: -len("lora_b")].rstrip(".")
                prefix = prefix if prefix else "W"
                b_by_prefix[prefix] = np.array(value)

        deltas: dict[str, list[float]] = {}
        for prefix, a in a_by_prefix.items():
            b = b_by_prefix.get(prefix)
            if b is None:
                continue
            delta = self._lora_delta(a, b)
            if scale:
                delta = delta * scale
            deltas[prefix] = delta.astype(np.float32).ravel().tolist()

        return deltas

    @staticmethod
    def _lora_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute LoRA delta: A @ B where A is [in, rank] and B is [rank, out].

        MLX adapters store: A: [in_features, rank], B: [rank, out_features]
        Delta = A @ B = [in_features, out_features]
        """
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        # Check A @ B first (standard MLX adapter convention)
        if a.shape[1] == b.shape[0]:
            # A: [in, rank], B: [rank, out] -> A @ B = [in, out]
            return a @ b
        if a.shape[0] == b.shape[1]:
            # Transposed: B.T @ A.T = [out, rank] @ [rank, in] = [out, in]
            return b.T @ a.T

        raise ValueError(f"Unsupported LoRA shapes for delta computation: A={a.shape} B={b.shape}")

    @staticmethod
    def dare_merge_readiness(effective_sparsity: float) -> str:
        if effective_sparsity >= 0.99:
            return "too_sparse"
        if effective_sparsity >= 0.80:
            return "ready"
        return "needs_more_training"

    @staticmethod
    def dora_learning_type(result: DoRADecomposition.DecompositionResult) -> str:
        change_type = result.dominant_change_type
        if change_type == ChangeType.MAGNITUDE_DOMINATED:
            return "magnitude_dominant"
        if change_type == ChangeType.DIRECTION_DOMINATED:
            return "direction_dominant"
        if change_type == ChangeType.MINIMAL:
            return "minimal"
        return "balanced"

    @staticmethod
    def dora_learning_type_confidence(result: DoRADecomposition.DecompositionResult) -> float:
        ratio = result.magnitude_to_direction_ratio
        if ratio <= 0:
            return 0.0
        dominance = max(ratio, 1.0 / ratio)
        config = DoRAConfiguration()
        threshold = max(
            config.magnitude_dominance_threshold,
            config.direction_dominance_threshold,
        )
        if result.dominant_change_type in (
            ChangeType.MAGNITUDE_DOMINATED,
            ChangeType.DIRECTION_DOMINATED,
        ):
            return min(1.0, dominance / threshold)
        if result.dominant_change_type == ChangeType.BALANCED:
            return max(0.0, (threshold - dominance) / (threshold - 1.0))
        return 1.0

    @staticmethod
    def dora_stability_score(result: DoRADecomposition.DecompositionResult) -> float:
        total_layers = len(result.per_layer_metrics)
        if total_layers == 0:
            return 0.0
        significant = set(result.layers_with_significant_direction_change) | set(
            result.layers_with_significant_magnitude_change
        )
        fraction = len(significant) / float(total_layers)
        return max(0.0, 1.0 - fraction)

    @staticmethod
    def dora_overfit_risk(result: DoRADecomposition.DecompositionResult) -> str:
        total_layers = len(result.per_layer_metrics)
        if total_layers == 0:
            return "unknown"
        significant = set(result.layers_with_significant_direction_change) | set(
            result.layers_with_significant_magnitude_change
        )
        fraction = len(significant) / float(total_layers)
        if fraction >= 0.5:
            return "high"
        if fraction >= 0.2:
            return "medium"
        return "low"

    @staticmethod
    def dora_interpretation(result: DoRADecomposition.DecompositionResult) -> str:
        if result.dominant_change_type == ChangeType.MAGNITUDE_DOMINATED:
            if result.overall_magnitude_change > 0:
                return (
                    "Adapter primarily amplifies existing features "
                    f"(magnitude +{int(result.overall_magnitude_change * 100)}%)"
                )
            return (
                "Adapter primarily attenuates existing features "
                f"(magnitude {int(result.overall_magnitude_change * 100)}%)"
            )
        if result.dominant_change_type == ChangeType.DIRECTION_DOMINATED:
            return f"Adapter primarily rotates feature space (drift: {result.overall_directional_drift:.2f})"
        if result.dominant_change_type == ChangeType.MINIMAL:
            return "Adapter has minimal impact on weight geometry"
        return "Adapter combines scaling and rotation (balanced change)"
