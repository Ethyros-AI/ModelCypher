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

"""Safetensors-header model probe (CPU-only, no MLX/torch/jax required).

This probe reads `config.json` plus the safetensors header(s) to extract:
- architecture fields (model_type, vocab_size, hidden_size, heads, quantization)
- weight tensor names + shapes
- parameter counts (from shapes, without loading tensors)

It is intended as a lightweight fallback when GPU backends are unavailable
or cannot be initialized (e.g. sandboxed environments).
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

from modelcypher.ports.model_probe import (
    AlignmentAnalysisResult,
    BaseModelProbe,
    LayerDrift,
    LayerInfo,
    MergeValidationResult,
    ModelProbeResult,
)
from modelcypher.utils.model_config import (
    resolve_hidden_size,
    resolve_num_hidden_layers,
    resolve_num_attention_heads,
    resolve_vocab_size,
)

logger = logging.getLogger(__name__)


class SafeTensorsModelProbe:
    """ModelProbePort implementation based on safetensors headers."""

    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details using safetensors metadata."""
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

        architecture = str(config.get("model_type", "unknown"))
        vocab_size = resolve_vocab_size(config)
        hidden_size = resolve_hidden_size(config)
        num_attention_heads = resolve_num_attention_heads(config)
        quantization = None
        quant_cfg = config.get("quantization_config")
        if isinstance(quant_cfg, dict):
            quant_method = quant_cfg.get("quant_method")
            if quant_method:
                quantization = str(quant_method)
        layer_count_config = resolve_num_hidden_layers(config)

        layers, parameter_count = self._analyze_safetensors_headers(path)

        return ModelProbeResult(
            architecture=architecture,
            parameter_count=parameter_count,
            layers=layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            quantization=quantization,
            layer_count_config=layer_count_config,
        )

    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        """Validate merge compatibility between two models."""
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

        low_effort = architecture_match and vocab_match and dimension_match

        return MergeValidationResult(
            low_effort=low_effort,
            warnings=warnings,
            architecture_match=architecture_match,
            vocab_match=vocab_match,
            dimension_match=dimension_match,
        )

    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        """Analyze alignment drift between two models.

        This implementation does not load tensors, so it cannot compute numeric drift.
        It reports layer set overlap + shape comparability; drift values are None.
        """
        path_a = Path(model_a).expanduser().resolve()
        path_b = Path(model_b).expanduser().resolve()
        if not path_a.exists() or not path_a.is_dir():
            raise ValueError(f"Model path does not exist: {path_a}")
        if not path_b.exists() or not path_b.is_dir():
            raise ValueError(f"Model path does not exist: {path_b}")

        shapes_a = self._load_weight_shapes(path_a)
        shapes_b = self._load_weight_shapes(path_b)

        set_a = set(shapes_a.keys())
        set_b = set(shapes_b.keys())
        common_layers = set_a & set_b
        missing_layer_count = len(set_a - set_b) + len(set_b - set_a)

        layer_drifts: list[LayerDrift] = []
        comparable_layer_count = 0

        for layer_name in sorted(common_layers):
            shape_a = shapes_a[layer_name]
            shape_b = shapes_b[layer_name]
            comparable = shape_a == shape_b
            if comparable:
                comparable_layer_count += 1
            layer_drifts.append(
                LayerDrift(
                    layer_name=layer_name,
                    drift_magnitude=None,
                    drift_z_score=None,
                    comparable=comparable,
                )
            )

        return AlignmentAnalysisResult(
            drift_magnitude=None,
            drift_std=None,
            drift_min=None,
            drift_max=None,
            drift_p50=None,
            drift_p90=None,
            common_layer_count=len(common_layers),
            comparable_layer_count=comparable_layer_count,
            missing_layer_count=missing_layer_count,
            layer_drifts=layer_drifts,
        )

    @staticmethod
    def _load_weight_shapes(model_path: Path) -> dict[str, list[int]]:
        shapes: dict[str, list[int]] = {}
        for st_file in sorted(model_path.glob("*.safetensors")):
            try:
                header = SafeTensorsModelProbe._read_safetensors_header(st_file)
            except Exception as exc:
                logger.warning("Failed to read safetensors header %s: %s", st_file, exc)
                continue
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                if not isinstance(value, dict):
                    continue
                shape = value.get("shape")
                if not isinstance(shape, list):
                    continue
                try:
                    shapes[key] = [int(dim) for dim in shape]
                except Exception:
                    continue
        return shapes

    @staticmethod
    def _analyze_safetensors_headers(model_path: Path) -> tuple[list[LayerInfo], int]:
        layers: list[LayerInfo] = []
        total_params = 0

        for st_file in sorted(model_path.glob("*.safetensors")):
            try:
                header = SafeTensorsModelProbe._read_safetensors_header(st_file)
            except Exception as exc:
                logger.warning("Failed to read safetensors header %s: %s", st_file, exc)
                continue

            for key, value in header.items():
                if key == "__metadata__":
                    continue
                if not isinstance(value, dict):
                    continue
                shape = value.get("shape")
                if not isinstance(shape, list):
                    continue
                try:
                    shape_ints = [int(dim) for dim in shape]
                except Exception:
                    continue

                params = 1
                for dim in shape_ints:
                    params *= int(dim)

                layers.append(
                    LayerInfo(
                        name=key,
                        type=BaseModelProbe.infer_layer_type(key),
                        parameters=params,
                        shape=shape_ints,
                    )
                )
                total_params += params

        return layers, total_params

    @staticmethod
    def _read_safetensors_header(path: Path) -> dict[str, Any]:
        """Read the JSON header from a safetensors file without loading tensors."""
        with path.open("rb") as handle:
            header_len_raw = handle.read(8)
            if len(header_len_raw) != 8:
                raise ValueError("Invalid safetensors file: missing header length")
            header_len = struct.unpack("<Q", header_len_raw)[0]
            header_bytes = handle.read(header_len)
            if len(header_bytes) != header_len:
                raise ValueError("Invalid safetensors file: truncated header")
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors file: header must be a JSON object")
        return header
