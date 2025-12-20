"""Model probe service for analyzing model architecture and compatibility."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from safetensors import safe_open

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerInfo:
    """Information about a single model layer."""
    name: str
    type: str
    parameters: int
    shape: list[int]


@dataclass(frozen=True)
class ModelProbeResult:
    """Result of probing a model for architecture details."""
    architecture: str
    parameter_count: int
    layers: list[LayerInfo]
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    quantization: Optional[str] = None


@dataclass(frozen=True)
class MergeValidationResult:
    """Result of validating merge compatibility between two models."""
    compatible: bool
    warnings: list[str]
    architecture_match: bool
    vocab_match: bool
    dimension_match: bool


@dataclass(frozen=True)
class LayerDrift:
    """Drift information for a single layer."""
    layer_name: str
    drift_magnitude: float
    direction: str


@dataclass(frozen=True)
class AlignmentAnalysisResult:
    """Result of analyzing alignment drift between two models."""
    drift_magnitude: float
    layer_drifts: list[LayerDrift]
    assessment: str
    interpretation: str


class ModelProbeService:
    """Service for probing and analyzing model architecture and compatibility."""

    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details.
        
        Args:
            model_path: Path to the model directory containing config.json and weight files.
            
        Returns:
            ModelProbeResult with architecture details.
            
        Raises:
            ValueError: If model path is invalid or config.json is missing.
        """
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Model path is not a directory: {path}")
        
        config_path = path / "config.json"
        if not config_path.exists():
            raise ValueError(f"config.json not found in model directory: {path}")
        
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid config.json: {exc}") from exc
        
        architecture = config.get("model_type", "unknown")
        vocab_size = config.get("vocab_size", 0)
        hidden_size = config.get("hidden_size", 0)
        num_attention_heads = config.get("num_attention_heads", 0)
        quantization = config.get("quantization_config", {}).get("quant_method")
        
        layers, parameter_count = self._analyze_weights(path)
        
        return ModelProbeResult(
            architecture=architecture,
            parameter_count=parameter_count,
            layers=layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            quantization=quantization,
        )

    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        """Validate merge compatibility between two models.
        
        Args:
            source: Path to the source model directory.
            target: Path to the target model directory.
            
        Returns:
            MergeValidationResult with compatibility assessment.
        """
        source_probe = self.probe(source)
        target_probe = self.probe(target)
        
        warnings: list[str] = []
        
        architecture_match = source_probe.architecture == target_probe.architecture
        if not architecture_match:
            warnings.append(
                f"Architecture mismatch: {source_probe.architecture} vs {target_probe.architecture}"
            )
        
        vocab_match = source_probe.vocab_size == target_probe.vocab_size
        if not vocab_match:
            warnings.append(
                f"Vocab size mismatch: {source_probe.vocab_size} vs {target_probe.vocab_size}"
            )
        
        dimension_match = source_probe.hidden_size == target_probe.hidden_size
        if not dimension_match:
            warnings.append(
                f"Hidden dimension mismatch: {source_probe.hidden_size} vs {target_probe.hidden_size}"
            )
        
        compatible = architecture_match and vocab_match and dimension_match
        
        return MergeValidationResult(
            compatible=compatible,
            warnings=warnings,
            architecture_match=architecture_match,
            vocab_match=vocab_match,
            dimension_match=dimension_match,
        )

    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        """Analyze alignment drift between two models.
        
        Computes layer-wise drift using weight comparison.
        
        Args:
            model_a: Path to the first model directory.
            model_b: Path to the second model directory.
            
        Returns:
            AlignmentAnalysisResult with drift metrics bounded in [0.0, 1.0].
        """
        path_a = Path(model_a).expanduser().resolve()
        path_b = Path(model_b).expanduser().resolve()
        
        weights_a = self._load_weight_tensors(path_a)
        weights_b = self._load_weight_tensors(path_b)
        
        common_layers = set(weights_a.keys()) & set(weights_b.keys())
        if not common_layers:
            return AlignmentAnalysisResult(
                drift_magnitude=1.0,
                layer_drifts=[],
                assessment="incompatible",
                interpretation="No common layers found between models.",
            )
        
        layer_drifts: list[LayerDrift] = []
        total_drift = 0.0
        
        for layer_name in sorted(common_layers):
            tensor_a = weights_a[layer_name]
            tensor_b = weights_b[layer_name]
            
            if tensor_a.shape != tensor_b.shape:
                layer_drifts.append(LayerDrift(
                    layer_name=layer_name,
                    drift_magnitude=1.0,
                    direction="shape_mismatch",
                ))
                total_drift += 1.0
                continue
            
            drift = self._compute_layer_drift(tensor_a, tensor_b)
            direction = "divergent" if drift > 0.5 else "aligned"
            
            layer_drifts.append(LayerDrift(
                layer_name=layer_name,
                drift_magnitude=drift,
                direction=direction,
            ))
            total_drift += drift
        
        avg_drift = total_drift / len(common_layers) if common_layers else 1.0
        drift_magnitude = min(1.0, max(0.0, avg_drift))
        
        if drift_magnitude < 0.1:
            assessment = "highly_aligned"
            interpretation = "Models are highly aligned with minimal drift."
        elif drift_magnitude < 0.3:
            assessment = "moderately_aligned"
            interpretation = "Models show moderate alignment with some drift."
        elif drift_magnitude < 0.6:
            assessment = "divergent"
            interpretation = "Models have diverged significantly."
        else:
            assessment = "highly_divergent"
            interpretation = "Models are highly divergent and may not be compatible."
        
        return AlignmentAnalysisResult(
            drift_magnitude=drift_magnitude,
            layer_drifts=layer_drifts,
            assessment=assessment,
            interpretation=interpretation,
        )

    def _analyze_weights(self, model_path: Path) -> tuple[list[LayerInfo], int]:
        """Analyze weight files to extract layer information."""
        layers: list[LayerInfo] = []
        total_params = 0
        
        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            return layers, total_params
        
        for st_file in safetensor_files:
            try:
                with safe_open(st_file, framework="numpy") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        shape = list(tensor.shape)
                        params = 1
                        for dim in shape:
                            params *= dim
                        
                        layer_type = self._infer_layer_type(key)
                        layers.append(LayerInfo(
                            name=key,
                            type=layer_type,
                            parameters=params,
                            shape=shape,
                        ))
                        total_params += params
            except Exception as exc:
                logger.warning("Failed to read safetensors file %s: %s", st_file, exc)
        
        return layers, total_params

    def _load_weight_tensors(self, model_path: Path) -> dict:
        """Load weight tensors from safetensors files."""
        tensors: dict = {}
        
        safetensor_files = list(model_path.glob("*.safetensors"))
        for st_file in safetensor_files:
            try:
                with safe_open(st_file, framework="numpy") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
            except Exception as exc:
                logger.warning("Failed to read safetensors file %s: %s", st_file, exc)
        
        return tensors

    def _compute_layer_drift(self, tensor_a, tensor_b) -> float:
        """Compute normalized drift between two tensors.
        
        Returns a value in [0.0, 1.0] where 0 means identical and 1 means maximally different.
        """
        import numpy as np
        
        diff = tensor_a.astype(np.float32) - tensor_b.astype(np.float32)
        norm_diff = np.linalg.norm(diff.flatten())
        norm_a = np.linalg.norm(tensor_a.astype(np.float32).flatten())
        norm_b = np.linalg.norm(tensor_b.astype(np.float32).flatten())
        
        max_norm = max(norm_a, norm_b, 1e-8)
        relative_drift = norm_diff / max_norm
        
        normalized = 1.0 - np.exp(-relative_drift)
        return float(min(1.0, max(0.0, normalized)))

    @staticmethod
    def _infer_layer_type(key: str) -> str:
        """Infer layer type from weight key name."""
        key_lower = key.lower()
        if "embed" in key_lower:
            return "embedding"
        if "attn" in key_lower or "attention" in key_lower:
            if "q_proj" in key_lower or "query" in key_lower:
                return "attention_query"
            if "k_proj" in key_lower or "key" in key_lower:
                return "attention_key"
            if "v_proj" in key_lower or "value" in key_lower:
                return "attention_value"
            if "o_proj" in key_lower or "out" in key_lower:
                return "attention_output"
            return "attention"
        if "mlp" in key_lower or "ffn" in key_lower:
            if "gate" in key_lower:
                return "mlp_gate"
            if "up" in key_lower:
                return "mlp_up"
            if "down" in key_lower:
                return "mlp_down"
            return "mlp"
        if "norm" in key_lower or "ln" in key_lower:
            return "normalization"
        if "lm_head" in key_lower:
            return "lm_head"
        return "unknown"
