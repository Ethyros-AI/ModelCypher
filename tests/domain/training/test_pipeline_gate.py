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

import dataclasses
import math

from modelcypher.core.domain.training.pipeline_gate import (
    PipelineGateInput,
    evaluate_pipeline_gate,
)

_EPS_F32 = math.ldexp(1.0, -23)
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)


def _healthy_input() -> PipelineGateInput:
    return PipelineGateInput(
        spectral_bounds_ok=True,
        stop_reason="certificate",
        per_layer_cka={0: 0.95, 1: 0.94},
        per_layer_cka_bound={0: 0.90, 1: 0.91},
        adapter_saturation_median_ratio=0.75,
        max_effective_gain_ratio=1.0,
        epoch_metrics=[{"online_eval_stop_basis_degraded_significant": False}],
        strict_fail_closed_core=False,
    )


def test_pipeline_gate_passes_healthy_case():
    verdict = evaluate_pipeline_gate(_healthy_input(), eps=_EPS_F32)
    assert verdict.operator == "pipeline_gate_v1"
    assert verdict.passed is True
    assert verdict.failure_modes == ()
    assert verdict.checks["spectral_bounds"].status == "pass"
    assert verdict.checks["cka_bound_bundle"].status == "pass"


def test_pipeline_gate_fails_each_check_individually():
    spectral_fail = evaluate_pipeline_gate(
        dataclasses.replace(_healthy_input(), spectral_bounds_ok=False),
        eps=_EPS_F32,
    )
    assert "spectral_bounds_violation" in spectral_fail.failure_modes

    safety_cap_fail = evaluate_pipeline_gate(
        dataclasses.replace(_healthy_input(), stop_reason="safety_cap (10 iters)"),
        eps=_EPS_F32,
    )
    assert "safety_cap_hit" in safety_cap_fail.failure_modes

    cka_fail = evaluate_pipeline_gate(
        dataclasses.replace(
            _healthy_input(),
            per_layer_cka={0: 0.1},
            per_layer_cka_bound={0: 0.9},
        ),
        eps=_EPS_F32,
    )
    assert "cka_bound_violation" in cka_fail.failure_modes

    sat_fail = evaluate_pipeline_gate(
        dataclasses.replace(_healthy_input(), adapter_saturation_median_ratio=1.0),
        eps=_EPS_F32,
    )
    assert "adapter_saturation_exceeded" in sat_fail.failure_modes

    gain_fail = evaluate_pipeline_gate(
        dataclasses.replace(
            _healthy_input(),
            max_effective_gain_ratio=1.0 + _SQRT_EPS_F32 + 1e-6,
        ),
        eps=_EPS_F32,
    )
    assert "gain_divergence" in gain_fail.failure_modes

    online_eval_fail = evaluate_pipeline_gate(
        dataclasses.replace(
            _healthy_input(),
            epoch_metrics=[{"online_eval_stop_basis_degraded_significant": True}],
        ),
        eps=_EPS_F32,
    )
    assert "online_eval_degraded_significant" in online_eval_fail.failure_modes


def test_pipeline_gate_boundary_conditions_are_inclusive():
    cka_boundary = evaluate_pipeline_gate(
        dataclasses.replace(
            _healthy_input(),
            per_layer_cka={0: 1.0 - _SQRT_EPS_F32},
            per_layer_cka_bound={0: 1.0},
        ),
        eps=_EPS_F32,
    )
    assert cka_boundary.checks["cka_bound_bundle"].status == "pass"

    gain_boundary = evaluate_pipeline_gate(
        dataclasses.replace(
            _healthy_input(),
            max_effective_gain_ratio=1.0 + _SQRT_EPS_F32,
        ),
        eps=_EPS_F32,
    )
    assert gain_boundary.checks["max_effective_gain_ratio"].status == "pass"


def test_pipeline_gate_strict_mode_fails_closed_for_missing_core_metrics():
    verdict = evaluate_pipeline_gate(
        PipelineGateInput(
            spectral_bounds_ok=None,
            stop_reason="certificate",
            per_layer_cka=None,
            per_layer_cka_bound=None,
            strict_fail_closed_core=True,
        ),
        eps=_EPS_F32,
    )
    assert verdict.passed is False
    assert "spectral_bounds_unavailable" in verdict.failure_modes
    assert "cka_bound_unavailable" in verdict.failure_modes
    assert set(verdict.unresolved_required) == {"spectral_bounds", "cka_bound_bundle"}


def test_pipeline_gate_research_mode_reports_unresolved_without_failing():
    verdict = evaluate_pipeline_gate(
        PipelineGateInput(
            spectral_bounds_ok=None,
            stop_reason="certificate",
            per_layer_cka=None,
            per_layer_cka_bound=None,
            strict_fail_closed_core=False,
        ),
        eps=_EPS_F32,
    )
    assert verdict.passed is True
    assert verdict.failure_modes == ()
    assert verdict.unresolved_required == ()
    assert verdict.checks["spectral_bounds"].status == "unresolved"
    assert verdict.checks["cka_bound_bundle"].status == "unresolved"
