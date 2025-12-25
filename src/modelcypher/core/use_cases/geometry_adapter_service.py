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
from typing import TYPE_CHECKING, Any

import numpy as np

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry import ChangeType, DoRAConfiguration, DoRADecomposition
from modelcypher.core.domain.geometry.dare_sparsity import DARESparsityAnalyzer
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterWeights:
    weights: dict[str, np.ndarray]
    scale: float


class GeometryAdapterService:
    """Service for analyzing adapter geometry (DARE sparsity, DoRA decomposition).

    Uses Backend protocol for tensor operations and ModelLoaderPort for weight loading.
    This ensures operations run on GPU and are hot-swappable for different backends
    (MLX, JAX, CUDA).
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        model_loader: "ModelLoaderPort | None" = None,
    ) -> None:
        """Initialize the adapter service.

        Args:
            backend: Compute backend for tensor operations. Auto-detects if None.
            model_loader: Weight loading port. Creates MLXModelLoader if None.
        """
        self._backend = backend or get_default_backend()
        self._model_loader = model_loader

    def _get_model_loader(self) -> "ModelLoaderPort":
        """Lazy-load model loader to avoid import at module level."""
        if self._model_loader is None:
            from modelcypher.adapters.mlx_model_loader import MLXModelLoader

            self._model_loader = MLXModelLoader()
        return self._model_loader

    def analyze_dare(
        self, checkpoint_path: str, base_path: str | None = None
    ) -> DARESparsityAnalyzer.SparsityAnalysis:
        """Analyze DARE sparsity of adapter weights.

        GPU-accelerated via Backend protocol.
        """
        deltas = self._compute_deltas_gpu(checkpoint_path, base_path)
        if not deltas:
            raise ValueError("No adapter delta weights found for DARE analysis")
        return DARESparsityAnalyzer.analyze_with_backend(
            delta_weights=deltas, backend=self._backend
        )

    def analyze_dora(
        self,
        checkpoint_path: str,
        base_path: str | None = None,
    ):
        """GPU-accelerated DoRA decomposition via Backend protocol.

        Hot-swappable: works with MLX, JAX, or CUDA backends.
        """
        from modelcypher.core.domain.geometry.dora_decomposition import DoRADecomposition

        b = self._backend
        loader = self._get_model_loader()

        # Load adapter weights on GPU
        checkpoint = loader.load_weights(checkpoint_path)
        scale = 1.0
        if "lora_scale" in checkpoint:
            scale_arr = b.reshape(b.array(checkpoint["lora_scale"]), (-1,))
            b.eval(scale_arr)
            scale = float(b.to_numpy(scale_arr)[0])

        lora_deltas = self._lora_deltas_gpu(checkpoint, scale)
        if not lora_deltas:
            raise ValueError("No LoRA adapter weights found in checkpoint")

        if base_path is None:
            raise ValueError(
                "DoRA decomposition requires a --base model for LoRA adapters. "
                "Provide the base model the adapter was trained on."
            )

        # Load base model on GPU
        base_raw = loader.load_weights(base_path)

        base_weights: dict[str, Any] = {}
        current_weights: dict[str, Any] = {}
        matched = 0

        for prefix, delta in lora_deltas.items():
            # Find matching base weight
            base_key = None
            for candidate in [f"{prefix}.weight", prefix, f"{prefix}weight"]:
                if candidate in base_raw:
                    base_key = candidate
                    break

            if base_key is None:
                continue

            raw_weight = base_raw[base_key]

            # Dequantize on GPU if needed
            base_weight = self._dequantize_gpu(raw_weight, base_key, base_raw)
            if base_weight is None:
                continue

            # Check shape compatibility
            delta_size = delta.size if hasattr(delta, "size") else len(delta.flatten())
            base_size = base_weight.size if hasattr(base_weight, "size") else len(
                base_weight.flatten()
            )
            if delta_size != base_size:
                logger.debug(
                    "Shape mismatch for %s: delta=%d, base=%d",
                    prefix,
                    delta_size,
                    base_size,
                )
                continue

            # Reshape delta to match base and compute current
            delta_reshaped = b.reshape(delta, base_weight.shape)
            current = base_weight + delta_reshaped

            base_weights[prefix] = base_weight
            current_weights[prefix] = current
            matched += 1

        # Force GPU computation
        b.eval(*base_weights.values(), *current_weights.values())

        if not base_weights or not current_weights:
            raise ValueError(
                "Unable to derive base/current weights for DoRA decomposition. "
                "For LoRA adapters, ensure the --base model is compatible with the adapter "
                "(same architecture and layer count as the model the adapter was trained on)."
            )

        logger.info("DoRA analyzing %d matched layers on GPU", matched)

        # Run DoRA decomposition on GPU
        decomposer = DoRADecomposition(backend=b)
        return decomposer.analyze_adapter(base_weights=base_weights, current_weights=current_weights)

    def _compute_deltas_gpu(
        self,
        checkpoint_path: str,
        base_path: str | None,
    ) -> dict[str, Any]:
        """Compute LoRA deltas as backend arrays for GPU processing.

        Uses Backend protocol for hot-swappable GPU acceleration.
        """
        b = self._backend
        loader = self._get_model_loader()

        checkpoint = loader.load_weights(checkpoint_path)
        scale = 1.0
        if "lora_scale" in checkpoint:
            scale_arr = b.reshape(b.array(checkpoint["lora_scale"]), (-1,))
            b.eval(scale_arr)
            scale = float(b.to_numpy(scale_arr)[0])

        deltas = self._lora_deltas_gpu(checkpoint, scale)
        if deltas:
            return deltas

        if base_path is None:
            return {}

        base = loader.load_weights(base_path)
        delta_arrays: dict[str, Any] = {}
        for key, current in checkpoint.items():
            if key == "lora_scale":
                continue
            base_weight = base.get(key)
            if base_weight is None or base_weight.shape != current.shape:
                continue
            current_f32 = b.astype(current, np.float32)
            base_f32 = b.astype(base_weight, np.float32)
            delta = current_f32 - base_f32
            delta_arrays[key] = delta

        b.eval(*delta_arrays.values())
        return delta_arrays

    def _lora_deltas_gpu(
        self,
        weights: dict[str, Any],
        scale: float,
    ) -> dict[str, Any]:
        """Compute LoRA deltas on GPU via Backend protocol.

        Hot-swappable: works with MLX, JAX, or CUDA backends.
        """
        b = self._backend

        a_by_prefix: dict[str, Any] = {}
        b_by_prefix: dict[str, Any] = {}

        for key, value in weights.items():
            lowered = key.lower()
            if lowered.endswith("lora_a"):
                prefix = key[: -len("lora_a")].rstrip(".")
                prefix = prefix if prefix else "W"
                a_by_prefix[prefix] = b.astype(value, np.float32)
            elif lowered.endswith("lora_b"):
                prefix = key[: -len("lora_b")].rstrip(".")
                prefix = prefix if prefix else "W"
                b_by_prefix[prefix] = b.astype(value, np.float32)

        deltas: dict[str, Any] = {}
        for prefix, a in a_by_prefix.items():
            b_mat = b_by_prefix.get(prefix)
            if b_mat is None:
                continue

            # LoRA delta: A @ B where A is [in, rank] and B is [rank, out]
            a_shape = tuple(a.shape)
            b_shape = tuple(b_mat.shape)

            # Try A @ B first (standard convention for MLX adapters)
            if a_shape[1] == b_shape[0]:
                # A: [in, rank], B: [rank, out] -> A @ B = [in, out]
                delta = b.matmul(a, b_mat)
            elif a_shape[0] == b_shape[1]:
                # Transposed: B.T @ A.T = [out, rank] @ [rank, in] = [out, in]
                b_t = b.transpose(b_mat)
                a_t = b.transpose(a)
                delta = b.matmul(b_t, a_t)
            else:
                continue

            if scale != 1.0:
                delta = delta * scale
            deltas[prefix] = delta

        # Force GPU computation
        if deltas:
            b.eval(*deltas.values())
        return deltas

    def _dequantize_gpu(
        self,
        weight: Any,
        base_key: str,
        all_params: dict[str, Any],
    ) -> Any | None:
        """Dequantize weight on GPU via Backend protocol.

        Returns None if dequantization fails or weight should be skipped.
        """
        b = self._backend

        # Check if already float - no dequantization needed
        weight_np = b.to_numpy(weight)
        if weight_np.dtype in (np.float16, np.float32, np.float64):
            return weight

        if weight_np.dtype.kind not in {"i", "u"}:
            logger.warning(
                "Unsupported dtype for weight %s (dtype=%s); skipping.",
                base_key,
                weight_np.dtype,
            )
            return None

        # Find scales/biases for dequantization
        base = base_key.replace(".weight", "")
        scales_key = f"{base}.scales"
        biases_key = f"{base}.biases"

        scales = all_params.get(scales_key)
        if scales is None:
            logger.warning(
                "Quantized weight %s missing scales; skipping.",
                base_key,
            )
            return None

        biases = all_params.get(biases_key)

        # Infer quantization parameters from shapes
        from modelcypher.core.use_cases.quantization_utils import resolve_quantization

        scales_np = b.to_numpy(scales)
        params = resolve_quantization(
            base_key=base_key,
            weight_shape=weight_np.shape,
            scales_shape=scales_np.shape,
            hint=None,
            biases_present=biases is not None,
        )
        if params is None:
            logger.warning(
                "Unable to infer quantization for %s; skipping.",
                base_key,
            )
            return None

        logger.debug(
            "Dequantizing %s on GPU (bits=%s groupSize=%s mode=%s)",
            base_key,
            params.bits,
            params.group_size,
            params.mode,
        )

        # Dequantize on GPU
        weight_arr = b.array(weight_np)
        scales_arr = b.array(scales_np)
        biases_arr = b.array(b.to_numpy(biases)) if biases is not None else None

        dequantized = b.dequantize(
            weight_arr,
            scales_arr,
            biases=biases_arr,
            group_size=params.group_size,
            bits=params.bits,
            mode=params.mode,
        )
        b.eval(dequantized)
        return dequantized

    def _compute_base_and_current(
        self,
        checkpoint_path: str,
        base_path: str | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        checkpoint = self._load_weights(checkpoint_path)
        base_raw = self._load_weights_raw(base_path) if base_path else None

        deltas = self._lora_deltas_from_weights(checkpoint.weights, checkpoint.scale)
        if deltas:
            if base_raw is None:
                # DoRA analysis requires a base model for LoRA adapters
                return {}, {}

            base_weights = base_raw
            base_vectors: dict[str, list[float]] = {}
            current_vectors: dict[str, list[float]] = {}

            # Get backend for dequantization
            from modelcypher.backends.mlx_backend import MLXBackend

            backend = MLXBackend()

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

                raw_weight = base_weights[base_key]
                # Dequantize if needed (handles quantized base models)
                base_weight = dequantize_if_needed(
                    raw_weight, base_key, base_weights, backend
                )
                delta_arr = np.array(delta_values, dtype=np.float32)

                # Check if shapes are compatible after dequantization
                expected_size = base_weight.size
                if delta_arr.size != expected_size:
                    # Shapes don't match - skip this layer
                    logger.debug(
                        "Shape mismatch for %s: delta=%d, base=%d",
                        prefix,
                        delta_arr.size,
                        expected_size,
                    )
                    continue

                delta = delta_arr.reshape(base_weight.shape)
                current = base_weight.astype(np.float32) + delta
                base_vectors[prefix] = base_weight.astype(np.float32).ravel().tolist()
                current_vectors[prefix] = current.ravel().tolist()

            return base_vectors, current_vectors

        if base_raw is None:
            return {}, {}

        base_vectors: dict[str, list[float]] = {}
        current_vectors: dict[str, list[float]] = {}
        for key, current in checkpoint.weights.items():
            base_weight = base_raw.get(key)
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

    def _load_weights_raw(self, path: str | None) -> dict[str, np.ndarray]:
        """Load model weights without converting dtypes.

        For quantized models, preserves original int/uint types and loads
        all shards including scales/biases needed for dequantization.
        """
        if path is None:
            raise ValueError("Base path is required for this analysis")

        import mlx.core as mx

        resolved = Path(path).expanduser().resolve()

        if resolved.is_dir():
            # Load all safetensors shards in the model directory
            shard_files = sorted(resolved.glob("*.safetensors"))
            if not shard_files:
                raise ValueError(f"No safetensors files found in {resolved}")

            all_weights: dict[str, np.ndarray] = {}
            for shard_file in shard_files:
                mx_weights = mx.load(str(shard_file))
                for key, value in mx_weights.items():
                    # Keep original dtype for quantized weights (int/uint)
                    # Only convert float types to float32
                    if value.dtype in (mx.float16, mx.bfloat16):
                        all_weights[key] = np.array(value.astype(mx.float32))
                    else:
                        all_weights[key] = np.array(value)
            return all_weights
        elif resolved.suffix == ".safetensors":
            mx_weights = mx.load(str(resolved))
            weights: dict[str, np.ndarray] = {}
            for key, value in mx_weights.items():
                if value.dtype in (mx.float16, mx.bfloat16):
                    weights[key] = np.array(value.astype(mx.float32))
                else:
                    weights[key] = np.array(value)
            return weights
        elif resolved.suffix == ".npz":
            data = np.load(resolved)
            return {key: np.array(value) for key, value in data.items()}
        else:
            raise ValueError(f"Unsupported format: {resolved.suffix}")

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
