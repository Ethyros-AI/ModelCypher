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
from typing import Any, Mapping

import numpy as np

from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantizationHint:
    bits: int
    group_size: int
    mode: str | None = None


@dataclass(frozen=True)
class QuantizationConfig:
    default: QuantizationHint | None
    overrides: dict[str, QuantizationHint]


@dataclass(frozen=True)
class QuantizationParams:
    bits: int
    group_size: int
    mode: str
    origin: str


def quantization_config_from_payload(payload: Mapping[str, Any]) -> QuantizationConfig | None:
    quant_payload = _select_quantization_payload(payload)
    if not quant_payload:
        return None

    default = _hint_from_entry(quant_payload, fallback=None)
    overrides: dict[str, QuantizationHint] = {}
    for key, entry in quant_payload.items():
        if key in {"bits", "group_size", "mode"}:
            continue
        if not isinstance(entry, Mapping):
            continue
        hint = _hint_from_entry(entry, fallback=default)
        if hint is None:
            continue
        overrides[key] = hint

    return QuantizationConfig(default=default, overrides=overrides)


def quantization_hint_for_key(
    weight_key: str,
    config: QuantizationConfig | None,
) -> QuantizationHint | None:
    if config is None:
        return None

    base_key = weight_key[:-7] if weight_key.endswith(".weight") else weight_key
    if base_key in config.overrides:
        return config.overrides[base_key]
    if base_key.startswith("model.") and base_key[6:] in config.overrides:
        return config.overrides[base_key[6:]]
    if weight_key in config.overrides:
        return config.overrides[weight_key]
    return config.default


def dequantize_if_needed(
    weight: Any,
    base_key: str,
    all_params: dict[str, Any],
    backend: Backend,
    hint: QuantizationHint | None = None,
) -> np.ndarray:
    weight_np = _to_numpy(weight, backend)
    if weight_np.dtype in (np.float16, np.float32, np.float64):
        return weight_np

    if weight_np.dtype.kind not in {"i", "u"}:
        logger.warning(
            "Unsupported dtype for weight %s (dtype=%s); skipping dequantization.",
            base_key,
            weight_np.dtype,
        )
        return weight_np

    base = base_key.replace(".weight", "")
    scales_key = f"{base}.scales"
    biases_key = f"{base}.biases"

    scales = all_params.get(scales_key)
    if scales is None:
        logger.warning(
            "Quantized weight %s missing scales; skipping dequantization.",
            base_key,
        )
        return weight_np

    scales_np = _to_numpy(scales, backend)
    biases = all_params.get(biases_key)
    biases_present = biases is not None
    params = resolve_quantization(
        base_key=base_key,
        weight_shape=weight_np.shape,
        scales_shape=scales_np.shape,
        hint=hint,
        biases_present=biases_present,
    )
    if params is None:
        logger.warning(
            "Unable to infer quantization parameters for %s; skipping dequantization.",
            base_key,
        )
        return weight_np

    logger.debug(
        "Dequantizing %s (bits=%s groupSize=%s mode=%s)",
        base_key,
        params.bits,
        params.group_size,
        params.mode,
    )

    weight_arr = backend.array(weight_np)
    scales_arr = backend.array(scales_np)
    biases_arr = backend.array(_to_numpy(biases, backend)) if biases is not None else None

    dequantized = backend.dequantize(
        weight_arr,
        scales_arr,
        biases=biases_arr,
        group_size=params.group_size,
        bits=params.bits,
        mode=params.mode,
    )
    return np.asarray(backend.to_numpy(dequantized))


def requantize_weights(
    weights: Mapping[str, Any],
    backend: Backend,
    output_hint: QuantizationHint,
    source_quantization: QuantizationConfig | None = None,
) -> dict[str, np.ndarray]:
    if output_hint.bits <= 0 or 32 % output_hint.bits != 0:
        raise ValueError(f"Invalid output quantization bits={output_hint.bits}")
    if output_hint.group_size <= 0:
        raise ValueError(f"Invalid output quantization groupSize={output_hint.group_size}")

    mode = output_hint.mode or "affine"
    quantized: dict[str, np.ndarray] = {}
    for key, value in weights.items():
        if key.endswith(".scales") or key.endswith(".biases"):
            continue

        weight_np = _to_numpy(value, backend)
        if not key.endswith(".weight") or weight_np.ndim != 2:
            quantized[key] = weight_np
            continue

        # Mirror MLX quantization: 2D weights (embeddings/linear) are quantized; norms remain float.
        hint = quantization_hint_for_key(key, source_quantization)
        dequantized = dequantize_if_needed(value, key, weights, backend, hint=hint)
        if dequantized.ndim != 2:
            quantized[key] = dequantized
            continue
        if dequantized.shape[1] % output_hint.group_size != 0:
            logger.warning(
                "Output quantization groupSize=%s does not divide inDim=%s for %s; keeping float.",
                output_hint.group_size,
                dequantized.shape[1],
                key,
            )
            quantized[key] = dequantized.astype(np.float32, copy=False)
            continue

        weight_arr = backend.array(dequantized.astype(np.float32), dtype=np.float32)
        q_weight, q_scales, q_biases = backend.quantize(
            weight_arr,
            group_size=output_hint.group_size,
            bits=output_hint.bits,
            mode=mode,
        )

        base = key.replace(".weight", "")
        quantized[key] = np.asarray(backend.to_numpy(q_weight))
        quantized[f"{base}.scales"] = np.asarray(backend.to_numpy(q_scales))
        if q_biases is not None:
            quantized[f"{base}.biases"] = np.asarray(backend.to_numpy(q_biases))

    return quantized


def resolve_quantization(
    base_key: str,
    weight_shape: tuple[int, ...],
    scales_shape: tuple[int, ...],
    hint: QuantizationHint | None = None,
    biases_present: bool | None = None,
) -> QuantizationParams | None:
    weight_out_dim = weight_shape[0] if len(weight_shape) >= 2 else None
    weight_last_dim = weight_shape[-1]
    scales_last_dim = scales_shape[-1]

    if hint is not None:
        resolved = _resolve_quantization_from_hint(
            base_key=base_key,
            weight_out_dim=weight_out_dim,
            weight_last_dim=weight_last_dim,
            scales_last_dim=scales_last_dim,
            hint=hint,
        )
        if resolved is not None:
            return _adjust_quantization_mode(resolved, biases_present, base_key)

    resolved = _infer_quantization_from_shapes(
        base_key=base_key,
        weight_out_dim=weight_out_dim,
        weight_last_dim=weight_last_dim,
        scales_last_dim=scales_last_dim,
    )
    if resolved is None:
        return None
    return _adjust_quantization_mode(resolved, biases_present, base_key)


def _resolve_quantization_from_hint(
    base_key: str,
    weight_out_dim: int | None,
    weight_last_dim: int,
    scales_last_dim: int,
    hint: QuantizationHint,
) -> QuantizationParams | None:
    bits = hint.bits
    if bits <= 0 or 32 % bits != 0:
        logger.warning(
            "Invalid quantization bits=%s for %s; falling back to inference.",
            bits,
            base_key,
        )
        return None

    group_size = hint.group_size
    if group_size <= 0:
        logger.warning(
            "Invalid quantization groupSize=%s for %s; falling back to inference.",
            group_size,
            base_key,
        )
        return None

    packing_factor = 32 // bits
    original_in_dim = weight_last_dim * packing_factor

    if weight_out_dim is not None and _is_likely_square_projection_key(base_key):
        if original_in_dim != weight_out_dim:
            logger.warning(
                "Quantization hint implies inDim=%s for %s but outDim=%s; falling back to inference.",
                original_in_dim,
                base_key,
                weight_out_dim,
            )
            return None

    if original_in_dim % group_size != 0:
        logger.warning(
            "Quantization hint groupSize=%s does not divide inDim=%s for %s; falling back to inference.",
            group_size,
            original_in_dim,
            base_key,
        )
        return None

    expected_scales = original_in_dim // group_size
    if expected_scales != scales_last_dim:
        logger.warning(
            "Quantization hint mismatch for %s: expected scalesLastDim=%s but got %s; falling back to inference.",
            base_key,
            expected_scales,
            scales_last_dim,
        )
        return None

    return QuantizationParams(
        bits=bits,
        group_size=group_size,
        mode=hint.mode or "affine",
        origin="hint",
    )


def _infer_quantization_from_shapes(
    base_key: str,
    weight_out_dim: int | None,
    weight_last_dim: int,
    scales_last_dim: int,
) -> QuantizationParams | None:
    # Mirrors the reference implementation's shape-based quantization inference heuristics.
    candidates = [4, 8, 2]
    best: tuple[int, QuantizationParams] | None = None

    for bits in candidates:
        if bits <= 0 or 32 % bits != 0:
            continue
        packing_factor = 32 // bits
        original_in_dim = weight_last_dim * packing_factor
        if scales_last_dim <= 0 or original_in_dim % scales_last_dim != 0:
            continue
        group_size = original_in_dim // scales_last_dim
        if group_size <= 0:
            continue

        score = _quantization_candidate_score(
            base_key=base_key,
            weight_out_dim=weight_out_dim,
            original_in_dim=original_in_dim,
            group_size=group_size,
            bits=bits,
        )

        params = QuantizationParams(
            bits=bits,
            group_size=group_size,
            mode="affine",
            origin="shape",
        )

        if best is None or score > best[0]:
            best = (score, params)

    return best[1] if best is not None else None


def _adjust_quantization_mode(
    params: QuantizationParams,
    biases_present: bool | None,
    base_key: str,
) -> QuantizationParams:
    if biases_present is True or biases_present is None:
        return params

    if params.mode != "affine":
        return params

    if params.bits == 4 and params.group_size == 32:
        # MLX mxfp4 uses 4-bit groups of 32 without biases; infer when biases are absent.
        return QuantizationParams(
            bits=params.bits,
            group_size=params.group_size,
            mode="mxfp4",
            origin=f"{params.origin}+mxfp4",
        )

    logger.warning(
        "Quantization mode for %s inferred as affine but biases are missing.",
        base_key,
    )
    return params


def _quantization_candidate_score(
    base_key: str,
    weight_out_dim: int | None,
    original_in_dim: int,
    group_size: int,
    bits: int,
) -> int:
    score = 0

    if _is_power_of_two(group_size):
        score += 3
    if 16 <= group_size <= 256:
        score += 2

    if group_size == 64:
        score += 6
    elif group_size == 32:
        score += 5
    elif group_size == 128:
        score += 4

    if bits in (4, 8):
        score += 1

    if weight_out_dim is not None and _is_likely_square_projection_key(base_key):
        if original_in_dim == weight_out_dim:
            score += 4
        else:
            score -= 4

    return score


def _is_likely_square_projection_key(key: str) -> bool:
    lower = key.lower()
    suffixes = (
        "q_proj.weight",
        "wq.weight",
        "o_proj.weight",
        "wo.weight",
        "out_proj.weight",
    )
    return any(lower.endswith(suffix) or f".{suffix}" in lower for suffix in suffixes)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _to_numpy(value: Any, backend: Backend) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(backend.to_numpy(value))
    except Exception:
        return np.asarray(value)


def _select_quantization_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    candidates = [
        payload.get("quantization"),
        payload.get("quantization_config"),
    ]
    candidates = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer the richer config when both quantization keys are present.
    return max(candidates, key=lambda candidate: len(candidate))


def _hint_from_entry(
    entry: Mapping[str, Any],
    fallback: QuantizationHint | None,
) -> QuantizationHint | None:
    bits = entry.get("bits")
    if bits is None and fallback is not None:
        bits = fallback.bits
    group_size = entry.get("group_size")
    if group_size is None and fallback is not None:
        group_size = fallback.group_size
    if bits is None or group_size is None:
        return None
    mode = entry.get("mode")
    if mode is None and fallback is not None:
        mode = fallback.mode
    mode_value = str(mode) if mode is not None else None
    return QuantizationHint(bits=int(bits), group_size=int(group_size), mode=mode_value)
