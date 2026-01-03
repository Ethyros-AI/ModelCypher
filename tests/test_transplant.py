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

"""Tests for null-space constrained transplant (AlphaEdit-style).

Mathematical guarantees to verify:
    1. A_boundary @ W' = A_boundary @ W_target  (boundary preserved)
    2. Core functionality transplanted via null-space filtered delta
    3. preserved_fraction measures how much delta survived projection
"""

import re

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.constrained_transplant import (
    verify_boundary_invariance,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.transplant import (
    TransplantDeltaResult,
    compute_transplant_delta,
    partition_core_boundary,
)
from modelcypher.core.use_cases.merge.stages.transplant import (
    TransplantStageConfig,
    stage_transplant,
)


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class TestPartitionCoreBoundary:
    """Tests for partition_core_boundary function."""

    def test_partition_empty_activations(self) -> None:
        """Empty activations should return empty partition."""
        backend = get_default_backend()
        activations = backend.zeros((0, 64))
        backend.eval(activations)

        result = partition_core_boundary(
            activations=activations,
            probe_ids=[],
            core_probe_ids=set(),
            backend=backend,
        )

        assert result.core_indices == []
        assert result.boundary_indices == []

    def test_partition_no_core_probes(self) -> None:
        """No core probes should return empty partition."""
        backend = get_default_backend()
        backend.random_seed(42)
        activations = backend.random_normal((10, 64))
        backend.eval(activations)

        result = partition_core_boundary(
            activations=activations,
            probe_ids=[f"probe_{i}" for i in range(10)],
            core_probe_ids=set(),  # empty core
            backend=backend,
        )

        assert result.core_indices == []
        assert result.boundary_indices == []

    def test_partition_finds_boundary_neighbors(self) -> None:
        """Boundary should include every non-core probe."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Create activations where some probes are close together
        activations = backend.random_normal((10, 64))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(10)]
        core_probe_ids = {"probe_0", "probe_1"}  # first two are core

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            backend=backend,
        )

        assert result.core_indices == [0, 1]
        assert len(result.boundary_indices) == 8
        # Boundary should not include core indices
        assert 0 not in result.boundary_indices
        assert 1 not in result.boundary_indices


class TestComputeTransplantDelta:
    """Tests for compute_transplant_delta function."""

    def test_boundary_output_preserved(self) -> None:
        """Core guarantee: A_boundary @ W' = A_boundary @ W_target."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Create test data
        in_dim, out_dim = 64, 32
        n_core, n_boundary = 5, 10

        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        # Compute transplant - null space params derived from spectral properties
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is True

        # Verify boundary preservation: A_boundary @ W' = A_boundary @ W_target
        # W is [out, in], so output = A @ W^T
        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)

        output_before = backend.matmul(activations_boundary, backend.transpose(weight_target))
        output_after = backend.matmul(activations_boundary, backend.transpose(merged_weight))
        backend.eval(output_before, output_after)

        diff = backend.norm(output_after - output_before)
        backend.eval(diff)
        diff_val = float(backend.tolist(diff))

        # Boundary output should be preserved (within numerical tolerance)
        eps = _eps(backend, diff_val)
        assert diff_val <= eps, f"Boundary not preserved: diff={diff_val}"

    def test_boundary_invariance_metric(self) -> None:
        """Boundary invariance metric should report near-zero relative diff."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core, n_boundary = 5, 10

        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        # Null-space params derived from spectral properties - no config needed
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        metrics = verify_boundary_invariance(
            transplanted_weights=result.merged_weight,
            target_weights=weight_target,
            boundary_activations=activations_boundary,
            tolerance=machine_epsilon(backend, weight_target),
            backend=backend,
        )

        assert metrics["passed"] is True
        eps = _eps(backend, metrics["max_relative_diff"])
        assert metrics["max_relative_diff"] <= eps

    def test_non_2d_weight_skipped(self) -> None:
        """Non-2D weights should be skipped (bias vectors, etc)."""
        backend = get_default_backend()
        backend.random_seed(42)

        weight_1d = backend.random_normal((64,))  # 1D bias
        weight_source = backend.random_normal((64,))
        activations_core = backend.random_normal((5, 64))
        activations_boundary = backend.random_normal((10, 64))
        backend.eval(weight_1d, weight_source, activations_core, activations_boundary)

        result = compute_transplant_delta(
            weight_target=weight_1d,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is False
        assert result.null_dim == 0

    def test_dimension_mismatch_skipped(self) -> None:
        """Dimension mismatch between weight and activations should skip."""
        backend = get_default_backend()
        backend.random_seed(42)

        weight_target = backend.random_normal((32, 64))
        weight_source = backend.random_normal((32, 64))
        activations_core = backend.random_normal((5, 128))  # wrong dim
        activations_boundary = backend.random_normal((10, 64))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is False


def test_stage_transplant_emits_alignment_metrics() -> None:
    """Stage transplant should emit core alignment metrics for transplanted layers."""
    backend = get_default_backend()
    backend.random_seed(7)

    in_dim = 8
    out_dim = 4
    probe_ids = ["p0", "p1", "p2", "p3"]
    probe_domains = ["math", "math", "other", "other"]

    source_weights = {
        "model.layers.0.mlp.down_proj.weight": backend.random_normal((out_dim, in_dim)),
    }
    target_weights = {
        "model.layers.0.mlp.down_proj.weight": backend.random_normal((out_dim, in_dim)),
    }
    backend.eval(*source_weights.values(), *target_weights.values())

    source_activations = {
        0: [backend.random_normal((in_dim,)) for _ in probe_ids],
    }
    for act in source_activations[0]:
        backend.eval(act)

    target_activations = {
        0: [backend.random_normal((in_dim,)) for _ in probe_ids],
    }
    for act in target_activations[0]:
        backend.eval(act)

    def extract_layer_index(key: str) -> int | None:
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            return int(match.group(1))
        return None

    graft_mask = {probe_id: {0: True} for probe_id in probe_ids}
    config = TransplantStageConfig(
        core_domains=("math",),
        graft_mask=graft_mask,
    )

    result = stage_transplant(
        source_weights=source_weights,
        target_weights=target_weights,
        layer_indices=[0],
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        source_activations=source_activations,
        target_activations=target_activations,
        config=config,
        extract_layer_index_fn=extract_layer_index,
        backend=backend,
    )

    metrics = result.metrics
    assert metrics.get("alignment_samples", 0) >= 1
    eps = _eps(
        backend,
        metrics.get("mean_cka_before", 0.0),
        metrics.get("mean_cka_after", 0.0),
    )
    assert -eps <= metrics.get("mean_cka_before", 0.0) <= 1.0 + eps
    assert -eps <= metrics.get("mean_cka_after", 0.0) <= 1.0 + eps


def test_stage_transplant_requires_real_activations() -> None:
    """Stage transplant should hard fail without real activations."""
    backend = get_default_backend()
    backend.random_seed(7)

    source_weights = {
        "model.layers.0.mlp.down_proj.weight": backend.random_normal((4, 8)),
    }
    target_weights = {
        "model.layers.0.mlp.down_proj.weight": backend.random_normal((4, 8)),
    }
    backend.eval(*source_weights.values(), *target_weights.values())

    def extract_layer_index(key: str) -> int | None:
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            return int(match.group(1))
        return None

    config = TransplantStageConfig(
        core_domains=("math",),
        graft_mask={"p0": {0: True}},
    )

    with pytest.raises(RuntimeError, match="requires real activations"):
        stage_transplant(
            source_weights=source_weights,
            target_weights=target_weights,
            layer_indices=[0],
            probe_ids=["p0"],
            probe_domains=["math"],
            target_activations=None,
            config=config,
            extract_layer_index_fn=extract_layer_index,
            backend=backend,
        )

def test_insufficient_core_samples_skipped() -> None:
    """Less than 2 core samples should skip transplant."""
    backend = get_default_backend()
    backend.random_seed(42)

    weight_target = backend.random_normal((32, 64))
    weight_source = backend.random_normal((32, 64))
    activations_core = backend.random_normal((1, 64))  # only 1 sample
    activations_boundary = backend.random_normal((10, 64))
    backend.eval(weight_target, weight_source, activations_core, activations_boundary)

    result = compute_transplant_delta(
        weight_target=weight_target,
        weight_source_aligned=weight_source,
        activations_core=activations_core,
        activations_boundary=activations_boundary,
        backend=backend,
    )

    assert result.applied is False


def test_preserved_fraction_computed() -> None:
    """preserved_fraction should measure how much delta survived projection."""
    backend = get_default_backend()
    backend.random_seed(42)

    in_dim, out_dim = 64, 32
    weight_target = backend.random_normal((out_dim, in_dim))
    weight_source = backend.random_normal((out_dim, in_dim))
    activations_core = backend.random_normal((5, in_dim))
    activations_boundary = backend.random_normal((10, in_dim))
    backend.eval(weight_target, weight_source, activations_core, activations_boundary)

    # Null-space params derived from spectral properties - no config needed
    result = compute_transplant_delta(
        weight_target=weight_target,
        weight_source_aligned=weight_source,
        activations_core=activations_core,
        activations_boundary=activations_boundary,
        backend=backend,
    )

    assert result.applied is True
    # preserved_fraction should be between 0 and 1
    eps = _eps(backend, result.preserved_fraction)
    assert -eps <= result.preserved_fraction <= 1.0 + eps
    # projection_loss = 1 - preserved_fraction
    eps = _eps(backend, result.projection_loss, result.preserved_fraction)
    assert abs(result.projection_loss + result.preserved_fraction - 1.0) <= eps


def test_zero_delta_has_full_preservation() -> None:
    """When source == target, delta is zero, preserved_fraction should be 1.0."""
    backend = get_default_backend()
    backend.random_seed(42)

    in_dim, out_dim = 64, 32
    weight = backend.random_normal((out_dim, in_dim))
    activations_core = backend.random_normal((5, in_dim))
    activations_boundary = backend.random_normal((10, in_dim))
    backend.eval(weight, activations_core, activations_boundary)

    result = compute_transplant_delta(
        weight_target=weight,
        weight_source_aligned=weight,  # same as target
        activations_core=activations_core,
        activations_boundary=activations_boundary,
        backend=backend,
    )

    assert result.applied is True
    eps = _eps(backend, result.delta_norm)
    assert result.delta_norm <= eps
    eps = _eps(backend, result.preserved_fraction, 1.0)
    assert abs(result.preserved_fraction - 1.0) <= eps
    eps = _eps(backend, result.projection_loss, 0.0)
    assert abs(result.projection_loss - 0.0) <= eps


class TestTransplantEndToEnd:
    """End-to-end tests for the transplant pipeline."""

    def test_transplant_with_large_null_space(self) -> None:
        """When boundary has large null space, more delta should survive."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 128, 64
        n_core, n_boundary = 10, 5  # few boundary = large null space

        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        # Null-space params derived from spectral properties - no config needed
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is True
        _, singular_vals, _ = backend.svd(activations_boundary)
        backend.eval(singular_vals)
        rank_eps = machine_epsilon(backend, singular_vals)
        rank = int((backend.tolist(singular_vals) > rank_eps).sum())
        expected_null_dim = in_dim - rank
        assert result.null_dim == expected_null_dim
        # Spectral norm bounding enforces compositional stability
        # We use direct scalar scaling (not full Birkhoff) to preserve null-space exactly
        assert result.birkhoff_spectral_clipped is True  # spectral norm was > 1.0
        eps = _eps(backend, result.filtered_norm)
        assert result.filtered_norm >= eps  # some delta survives
        eps = _eps(backend, result.projection_loss)
        assert result.projection_loss <= 1.0 - eps  # not total loss

    def test_transplant_with_small_null_space(self) -> None:
        """When boundary has small null space, less delta survives."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core, n_boundary = 5, 60  # many boundary = small null space

        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        # Null-space params derived from spectral properties - no config needed
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is True
        _, singular_vals, _ = backend.svd(activations_boundary)
        backend.eval(singular_vals)
        rank_eps = machine_epsilon(backend, singular_vals)
        rank = int((backend.tolist(singular_vals) > rank_eps).sum())
        expected_null_dim = in_dim - rank
        assert result.null_dim == expected_null_dim
