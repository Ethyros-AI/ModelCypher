# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for probe_from_profile.py — dimension validation and bottleneck identification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from modelcypher.core.domain.geometry.variance_concentration import (
    VarianceConcentrationResult,
)
from modelcypher.core.domain.profile import ProfileActivations
from modelcypher.experimental.merge.stages.probe_from_profile import (
    compute_alignment_from_profiles,
)

_MODULE = "modelcypher.experimental.merge.stages.probe_from_profile"
_VAR_CONC = "modelcypher.core.domain.geometry.variance_concentration.compute_variance_concentration"


def _arr(shape):
    return np.zeros(shape, dtype=np.float32)


def _fake_profile():
    profile = MagicMock()
    profile.has_activations = True
    profile.probe_ids = []
    profile.probe_domains = []
    return profile


def _fake_align_result():
    result = MagicMock()
    result.feature_transforms = {}
    result.layer_mapping = {0: 0}
    result.scale_ratios = {0: 1.0}
    result.attention_transforms = {}
    result.k_transforms = {}
    result.v_transforms = {}
    result.intermediate_transforms = {}
    result.gate_transforms = {}
    result.layer_cka_scores = {0: 1.0}
    return result


def _fake_backend():
    backend = MagicMock()
    arr = np.array([1.0], dtype=np.float32)
    backend.array.return_value = arr
    backend.eval.return_value = None
    backend.astype.side_effect = lambda a, dtype: a
    finfo = MagicMock()
    finfo.eps = 1.19e-7
    backend.finfo.return_value = finfo
    return backend


def _var_result(var_top1, effective_rank, dim=2048):
    return VarianceConcentrationResult(
        var_top1=var_top1,
        var_top_k={1: var_top1},
        effective_rank=effective_rank,
        n_singular_values=50,
        n_samples=8,
        hidden_dim=dim,
        total_variance=1.0,
    )


class TestDimensionValidation:
    """Dimension validation drops intermediate/gate activations when dims collide."""

    def test_drops_hidden_dim_fallback(self, tmp_path):
        """intermediate_dim == hidden_dim → intermediate and gate cleared for both models."""
        hidden_dim = 512

        source_acts = ProfileActivations(
            hidden={0: _arr((8, hidden_dim))},
            intermediate={0: _arr((8, hidden_dim))},  # wrong: equals hidden_dim
            gate={0: _arr((8, hidden_dim))},
        )
        target_acts = ProfileActivations(
            hidden={0: _arr((8, hidden_dim))},
            intermediate={0: _arr((8, hidden_dim))},  # wrong: equals hidden_dim
            gate={0: _arr((8, hidden_dim))},
        )

        with (
            patch(f"{_MODULE}.GeometricProfile") as mock_gp,
            patch(f"{_MODULE}.load_activations") as mock_load,
            patch(f"{_MODULE}.align_layers") as mock_align,
        ):
            mock_gp.load.return_value = _fake_profile()
            mock_load.side_effect = [source_acts, target_acts]
            mock_align.return_value = _fake_align_result()

            result = compute_alignment_from_profiles(
                source_profile_dir=tmp_path / "source",
                target_profile_dir=tmp_path / "target",
                backend=_fake_backend(),
            )

        assert result.source_intermediate_activations == {}
        assert result.target_intermediate_activations == {}
        assert result.source_gate_activations == {}
        assert result.target_gate_activations == {}

    def test_passes_correct_dims(self, tmp_path):
        """intermediate_dim > hidden_dim → activations retained unchanged."""
        hidden_dim = 512
        intermediate_dim = 2048

        source_acts = ProfileActivations(
            hidden={0: _arr((8, hidden_dim))},
            intermediate={0: _arr((8, intermediate_dim))},
            gate={0: _arr((8, intermediate_dim))},
        )
        target_acts = ProfileActivations(
            hidden={0: _arr((8, hidden_dim))},
            intermediate={0: _arr((8, intermediate_dim))},
            gate={0: _arr((8, intermediate_dim))},
        )

        with (
            patch(f"{_MODULE}.GeometricProfile") as mock_gp,
            patch(f"{_MODULE}.load_activations") as mock_load,
            patch(f"{_MODULE}.align_layers") as mock_align,
            patch(_VAR_CONC, return_value=_var_result(0.5, 40.0, intermediate_dim)),
        ):
            mock_gp.load.return_value = _fake_profile()
            mock_load.side_effect = [source_acts, target_acts]
            mock_align.return_value = _fake_align_result()

            result = compute_alignment_from_profiles(
                source_profile_dir=tmp_path / "source",
                target_profile_dir=tmp_path / "target",
                backend=_fake_backend(),
            )

        # Correct dims → activations preserved
        assert 0 in result.source_intermediate_activations
        assert 0 in result.target_intermediate_activations


class TestBottleneckIdentification:
    """Bottleneck layer selection from variance concentration."""

    def test_bottleneck_and_injection_layer_selected(self, tmp_path):
        """Layer with highest var_top1 is bottleneck; qualifying transmission layer is injection."""
        hidden_dim = 512
        intermediate_dim = 2048

        # 4 layers (indices 0–3), each with distinct intermediate activations
        source_acts = ProfileActivations(
            hidden={i: _arr((8, hidden_dim)) for i in range(4)},
            intermediate={i: _arr((8, intermediate_dim)) for i in range(4)},
        )
        target_acts = ProfileActivations(
            hidden={i: _arr((8, hidden_dim)) for i in range(4)},
            intermediate={i: _arr((8, intermediate_dim)) for i in range(4)},
        )

        # Variance profile — iterated in insertion order (0, 1, 2, 3):
        #   Layer 0: var_top1=0.1, eff_rank=100 — excluded as idx=0
        #   Layer 1: var_top1=0.2, eff_rank=80  — transmission (low var, high rank, idx≠0)
        #   Layer 2: var_top1=0.8, eff_rank=20  — BOTTLENECK (max var_top1)
        #   Layer 3: var_top1=0.6, eff_rank=30  — neither (var_top1 not < median=0.6)
        #
        # median_var = sorted_by_var[n//2=2].var_top1 = 0.6
        # median_rank = sorted_by_rank_desc[n//2=2].eff_rank = 30
        # → injection_layer=1, bottleneck_layer=2, layer_filter=[2,1]
        var_side_effect = [
            _var_result(0.1, 100.0, intermediate_dim),
            _var_result(0.2, 80.0, intermediate_dim),
            _var_result(0.8, 20.0, intermediate_dim),
            _var_result(0.6, 30.0, intermediate_dim),
        ]

        captured_layer_filter = []

        def capturing_align(**kwargs):
            captured_layer_filter.append(kwargs.get("layer_filter"))
            return _fake_align_result()

        with (
            patch(f"{_MODULE}.GeometricProfile") as mock_gp,
            patch(f"{_MODULE}.load_activations") as mock_load,
            patch(f"{_MODULE}.align_layers", side_effect=capturing_align),
            patch(_VAR_CONC, side_effect=var_side_effect),
        ):
            mock_gp.load.return_value = _fake_profile()
            mock_load.side_effect = [source_acts, target_acts]

            result = compute_alignment_from_profiles(
                source_profile_dir=tmp_path / "source",
                target_profile_dir=tmp_path / "target",
                backend=_fake_backend(),
            )

        # Bottleneck = layer 2 (highest var_top1=0.8)
        # Injection = layer 1 (only transmission layer)
        assert result.injection_layer == 1
        assert captured_layer_filter[0] is not None
        assert 2 in captured_layer_filter[0]
        assert 1 in captured_layer_filter[0]
