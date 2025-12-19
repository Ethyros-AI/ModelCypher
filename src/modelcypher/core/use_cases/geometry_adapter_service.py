from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modelcypher.core.domain.dare_sparsity import DARESparsityAnalyzer
from modelcypher.core.domain.geometry import DoRADecomposition


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterWeights:
    weights: dict[str, np.ndarray]
    scale: float


class GeometryAdapterService:
    def analyze_dare(self, checkpoint_path: str, base_path: str | None = None) -> DARESparsityAnalyzer.SparsityAnalysis:
        deltas, _ = self._compute_deltas(checkpoint_path, base_path)
        if not deltas:
            raise ValueError("No adapter delta weights found for DARE analysis")
        return DARESparsityAnalyzer.analyze(delta_weights=deltas)

    def analyze_dora(
        self,
        checkpoint_path: str,
        base_path: str | None = None,
    ) -> DoRADecomposition.DecompositionResult:
        base_vectors, current_vectors = self._compute_base_and_current(checkpoint_path, base_path)
        if not base_vectors or not current_vectors:
            raise ValueError("Unable to derive base/current weights for DoRA decomposition")
        return DoRADecomposition.analyze_adapter(base_weights=base_vectors, current_weights=current_vectors)

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

    def _compute_base_and_current(
        self,
        checkpoint_path: str,
        base_path: str | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        checkpoint = self._load_weights(checkpoint_path)
        base = self._load_weights(base_path) if base_path else None

        deltas = self._lora_deltas_from_weights(checkpoint.weights, checkpoint.scale)
        if deltas:
            base_weights = base.weights if base else checkpoint.weights
            base_vectors: dict[str, list[float]] = {}
            current_vectors: dict[str, list[float]] = {}
            fallback_key = None
            if base_weights:
                fallback_key = next(iter(base_weights.keys()))

            for prefix, delta_values in deltas.items():
                base_key = prefix if prefix in base_weights else fallback_key
                if base_key is None:
                    continue
                base_weight = base_weights.get(base_key)
                if base_weight is None:
                    continue
                delta = np.array(delta_values, dtype=np.float32).reshape(base_weight.shape)
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
            from safetensors.numpy import load_file

            weights = load_file(weight_path)
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
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        if a.shape[0] == b.shape[1]:
            return b @ a
        if a.shape[1] == b.shape[0]:
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
        if change_type == DoRADecomposition.ChangeType.magnitude_dominated:
            return "magnitude_dominant"
        if change_type == DoRADecomposition.ChangeType.direction_dominated:
            return "direction_dominant"
        if change_type == DoRADecomposition.ChangeType.minimal:
            return "minimal"
        return "balanced"

    @staticmethod
    def dora_learning_type_confidence(result: DoRADecomposition.DecompositionResult) -> float:
        ratio = result.magnitude_to_direction_ratio
        if ratio <= 0:
            return 0.0
        dominance = max(ratio, 1.0 / ratio)
        threshold = max(
            DoRADecomposition.Configuration.default().magnitude_dominance_threshold,
            DoRADecomposition.Configuration.default().direction_dominance_threshold,
        )
        if result.dominant_change_type in (
            DoRADecomposition.ChangeType.magnitude_dominated,
            DoRADecomposition.ChangeType.direction_dominated,
        ):
            return min(1.0, dominance / threshold)
        if result.dominant_change_type == DoRADecomposition.ChangeType.balanced:
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
        if result.dominant_change_type == DoRADecomposition.ChangeType.magnitude_dominated:
            if result.overall_magnitude_change > 0:
                return (
                    "Adapter primarily amplifies existing features "
                    f"(magnitude +{int(result.overall_magnitude_change * 100)}%)"
                )
            return (
                "Adapter primarily attenuates existing features "
                f"(magnitude {int(result.overall_magnitude_change * 100)}%)"
            )
        if result.dominant_change_type == DoRADecomposition.ChangeType.direction_dominated:
            return f"Adapter primarily rotates feature space (drift: {result.overall_directional_drift:.2f})"
        if result.dominant_change_type == DoRADecomposition.ChangeType.minimal:
            return "Adapter has minimal impact on weight geometry"
        return "Adapter combines scaling and rotation (balanced change)"
