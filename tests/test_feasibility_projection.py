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

from modelcypher.core.use_cases.feasibility_projection import (
    RuntimeOverhead,
    project_memory_gib,
    static_weight_memory_gib,
)


def test_static_weight_memory_monotonic_in_params() -> None:
    small = static_weight_memory_gib(param_count=1_000_000_000, precision_bits=16)
    large = static_weight_memory_gib(param_count=2_000_000_000, precision_bits=16)
    assert large > small


def test_static_weight_memory_monotonic_in_bits() -> None:
    lower_bits = static_weight_memory_gib(param_count=1_000_000_000, precision_bits=4)
    higher_bits = static_weight_memory_gib(param_count=1_000_000_000, precision_bits=8)
    assert higher_bits > lower_bits


def test_projected_memory_monotonic_with_scale() -> None:
    overhead = RuntimeOverhead(
        load_overhead_gib=1.0,
        forward_delta_gib=0.5,
        decode_slope_gib_per_token=0.01,
    )
    p_70b = project_memory_gib(
        param_count=70_000_000_000,
        precision_bits=8,
        overhead=overhead,
        decode_tokens=32,
    )
    p_120b = project_memory_gib(
        param_count=120_000_000_000,
        precision_bits=8,
        overhead=overhead,
        decode_tokens=32,
    )
    assert p_120b["load_active_gib"] > p_70b["load_active_gib"]
    assert p_120b["forward_active_gib"] > p_70b["forward_active_gib"]

