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

"""Hypothesis property tests for quantization utils."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.use_cases.quantization_utils import (
    QuantizationHint,
    quantization_hint_for_key,
    quantization_plan_from_payload,
    resolve_quantization,
)

_bits = st.sampled_from([2, 4, 8, 16])
_group_size = st.sampled_from([16, 32, 64])
_valid_quantized_layouts = [
    (bits, group_size, weight_last_dim, weight_out_dim)
    for bits in (2, 4, 8, 16)
    for group_size in (16, 32, 64)
    for weight_last_dim in range(1, 17)
    for weight_out_dim in range(1, 17)
    if (weight_last_dim * (32 // bits)) % group_size == 0
]


@settings(max_examples=10, deadline=None)
@given(
    default_bits=_bits,
    default_group=_group_size,
    override_bits=_bits,
    override_group=_group_size,
    mode=st.sampled_from([None, "affine", "mxfp4"]),
)
def test_quantization_plan_override(
    default_bits: int,
    default_group: int,
    override_bits: int,
    override_group: int,
    mode: str | None,
) -> None:
    payload = {
        "quantization": {
            "bits": default_bits,
            "group_size": default_group,
            "mode": mode,
            "layer": {"bits": override_bits, "group_size": override_group},
        }
    }

    plan = quantization_plan_from_payload(payload)
    assert plan is not None
    default_hint, overrides = plan
    assert default_hint is not None
    assert default_hint.bits == default_bits
    assert default_hint.group_size == default_group
    assert default_hint.mode == (str(mode) if mode is not None else None)

    hint = quantization_hint_for_key("layer.weight", plan)
    assert hint is not None
    assert hint.bits == override_bits
    assert hint.group_size == override_group


@settings(max_examples=10, deadline=None)
@given(
    layout=st.sampled_from(_valid_quantized_layouts),
    biases_present=st.booleans(),
)
def test_resolve_quantization_from_hint(
    layout: tuple[int, int, int, int],
    biases_present: bool,
) -> None:
    bits, group_size, weight_last_dim, weight_out_dim = layout
    packing_factor = 32 // bits
    original_in_dim = weight_last_dim * packing_factor
    scales_last_dim = original_in_dim // group_size

    hint = QuantizationHint(bits=bits, group_size=group_size, mode=None)
    params = resolve_quantization(
        base_key="linear.weight",
        weight_shape=(weight_out_dim, weight_last_dim),
        scales_shape=(1, scales_last_dim),
        hint=hint,
        biases_present=biases_present,
    )
    assert params is not None
    assert params.bits == bits
    assert params.group_size == group_size
    if biases_present is False and bits == 4 and group_size == 32:
        assert params.mode == "mxfp4"
    else:
        assert params.mode == "affine"
