# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_script_module(name: str, relative_path: str) -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module(
    "tangent_subspace_id_mechanism_script",
    "scripts/tangent_subspace_id_mechanism.py",
)


def test_added_direction_signal_detects_orthogonal_append_without_shared_rotation():
    source_basis = SCRIPT.np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=SCRIPT.np.float64,
    )
    target_basis = SCRIPT.np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=SCRIPT.np.float64,
    )

    shared = SCRIPT.shared_rotation_metrics(source_basis, target_basis)
    added = SCRIPT.added_direction_signal_numpy(source_basis, target_basis)

    assert shared["shared_rank"] == 2
    assert shared["shared_rotation_geodesic"] < 1e-12
    assert added["count_above_floor"] == 1
    assert math.isclose(added["total_residual_energy"], 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert added["max_residual_norm"] > 0.99


def test_added_direction_signal_stays_zero_for_identical_span():
    source_basis = SCRIPT.np.array(
        [[1.0, 0.0]],
        dtype=SCRIPT.np.float64,
    )
    target_basis = SCRIPT.np.array(
        [
            [1.0, 0.0],
        ],
        dtype=SCRIPT.np.float64,
    )

    shared = SCRIPT.shared_rotation_metrics(source_basis, target_basis)
    added = SCRIPT.added_direction_signal_numpy(source_basis, target_basis)

    assert shared["shared_rotation_geodesic"] == 0.0
    assert added["count_above_floor"] == 0
    assert added["total_residual_energy"] < 1e-12


def test_measurement_b_payload_preserves_operator_telemetry():
    result = SimpleNamespace(
        anchor_count=256,
        neighbor_count=16,
        tangent_rank=8,
        mean_angle_radians=0.25,
        median_angle_radians=0.2,
        mean_cosine=0.91,
        coverage=0.85,
    )

    payload = SCRIPT._measurement_b_payload(result, 4, 5)

    assert payload["layer_pair"] == [4, 5]
    assert payload["anchor_count"] == 256
    assert payload["neighbor_count"] == 16
    assert payload["tangent_rank"] == 8
    assert payload["coverage"] == 0.85


def test_llama_probe_budget_derives_256_from_reference_results(tmp_path: Path):
    reference = {
        "per_model": [
            {
                "model_name": "Llama-3.2-3B",
                "twonn_ids": [62.5, 4.39, 7.71, 7.77, 7.09],
            }
        ]
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    budget = SCRIPT.derive_llama_probe_budget(path)

    assert budget["required_tangent_rank"] == 8
    assert budget["required_probe_count"] == 256
    assert budget["used_fallback"] is False


def test_observable_correlations_do_not_emit_pass_flags():
    meas_a = [
        {"shared_rotation_geodesic": 0.1, "added_direction_count_eps": 0, "added_direction_total_residual": 0.0},
        {"shared_rotation_geodesic": 0.3, "added_direction_count_eps": 1, "added_direction_total_residual": 1.0},
        {"shared_rotation_geodesic": 0.2, "added_direction_count_eps": 1, "added_direction_total_residual": 0.5},
        {"shared_rotation_geodesic": 0.4, "added_direction_count_eps": 2, "added_direction_total_residual": 1.5},
    ]
    meas_b = [
        {"mean_angle_radians": 0.2},
        {"mean_angle_radians": 0.4},
        {"mean_angle_radians": 0.3},
        {"mean_angle_radians": 0.5},
    ]
    meas_c = [
        {"mean_delta_local_rank": 0.0},
        {"mean_delta_local_rank": 0.1},
        {"mean_delta_local_rank": -0.1},
        {"mean_delta_local_rank": 0.2},
    ]
    twonn_ids = [5.0, 5.5, 6.5, 6.0, 7.0]

    analysis = SCRIPT.compute_observable_correlations(meas_a, meas_b, meas_c, twonn_ids)

    for view_name in ("full", "excluding_stage0"):
        for record in analysis[view_name].values():
            assert "passes" not in record
    assert analysis["full"]["local_rank_vs_delta_id"]["status"] == "[MEASUREMENT_INVALID]"
