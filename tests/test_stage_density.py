# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for stage_density contracts.

Covers:
  D1 — Empty source or target activations raise RuntimeError before any computation.
  D2 — Profile-based mode (probe_ids=[]) produces empty graft_mask and
       placeholder per-concept structures; point cloud and density_weights are
       still computed per layer.
  D3 — layer_mapping is honoured: source lookup uses the mapped key, not the
       target layer index directly.
  D4 — filter_core_probes_by_graft_mask correctly interprets the graft mask.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.merge.stages.density import (
    filter_core_probes_by_graft_mask,
    stage_density,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _acts(b, n: int = 16, dim: int = 32):
    """Random float32 activation matrix [n, dim]."""
    return b.random_normal((n, dim))


def _profile_call(b, *, layers=(0,), **kwargs):
    """Profile-based stage_density call with sensible defaults."""
    acts = _acts(b)
    defaults = dict(
        source_activations={l: acts for l in layers},
        target_activations={l: acts for l in layers},
        probe_ids=[],
        probe_domains=[],
        layers=list(layers),
        backend=b,
    )
    defaults.update(kwargs)
    return stage_density(**defaults)


# ---------------------------------------------------------------------------
# D1: Early exit on empty activations
# ---------------------------------------------------------------------------

class TestEarlyExit:
    def test_raises_on_empty_source_activations(self):
        b = get_default_backend()
        with pytest.raises(RuntimeError):
            stage_density(
                source_activations={},
                target_activations={0: _acts(b)},
                probe_ids=[],
                probe_domains=[],
                layers=[0],
                backend=b,
            )

    def test_raises_on_empty_target_activations(self):
        b = get_default_backend()
        with pytest.raises(RuntimeError):
            stage_density(
                source_activations={0: _acts(b)},
                target_activations={},
                probe_ids=[],
                probe_domains=[],
                layers=[0],
                backend=b,
            )


# ---------------------------------------------------------------------------
# D2: Profile-based mode (probe_ids=[])
# ---------------------------------------------------------------------------

class TestProfileBasedMode:
    def test_graft_mask_empty(self):
        """No probe metadata → per-concept analysis skipped → graft_mask == {}."""
        b = get_default_backend()
        result = _profile_call(b)
        assert result.graft_mask == {}

    def test_placeholder_profiles_have_empty_layer_profiles(self):
        """Placeholder ModelDensityProfile objects have no layer_profiles entries."""
        b = get_default_backend()
        result = _profile_call(b)
        assert result.source_profile.layer_profiles == {}
        assert result.target_profile.layer_profiles == {}

    def test_point_cloud_densities_computed_per_layer(self):
        """Point cloud density is computed for each requested layer even in profile mode."""
        b = get_default_backend()
        result = _profile_call(b, layers=(0, 1))
        # Both layers must have a point cloud density entry
        assert 0 in result.point_cloud_densities
        assert 1 in result.point_cloud_densities

    def test_density_weights_computed_per_layer(self):
        """Density weights are derived from point cloud and present for each layer."""
        b = get_default_backend()
        result = _profile_call(b, layers=(0, 1))
        assert 0 in result.density_weights
        assert 1 in result.density_weights

    def test_metrics_has_expected_keys(self):
        """Metrics dict contains the standard set of keys."""
        b = get_default_backend()
        result = _profile_call(b)
        expected_keys = {
            "layers_analyzed",
            "concepts_analyzed",
            "positive_opportunity_count",
            "nonpositive_opportunity_count",
        }
        assert expected_keys.issubset(result.metrics.keys())

    def test_layers_analyzed_metric_matches_layers_arg(self):
        b = get_default_backend()
        result = _profile_call(b, layers=(0, 1, 2))
        assert result.metrics["layers_analyzed"] == 3


# ---------------------------------------------------------------------------
# D3: layer_mapping
# ---------------------------------------------------------------------------

class TestLayerMapping:
    def test_explicit_mapping_uses_mapped_source_layer(self):
        """layer_mapping={0: 1} → source lookup uses key 1.

        source_activations has key 1 only. If the mapping is honoured, the
        stage finds the activation and produces a point cloud for target layer 0.
        If the mapping were ignored, key 0 would be looked up → None → RuntimeError.
        """
        b = get_default_backend()
        acts = _acts(b)
        result = stage_density(
            source_activations={1: acts},   # only key 1 — no key 0
            target_activations={0: acts},   # target layer 0
            probe_ids=[],
            probe_domains=[],
            layers=[0],
            layer_mapping={0: 1},           # target layer 0 → source layer 1
            backend=b,
        )
        assert 0 in result.point_cloud_densities

    def test_unmapped_layer_raises_on_missing_source(self):
        """Without mapping, source lookup uses layer_idx directly.

        source_activations has key 1 only. With no mapping, the stage looks up
        key 0 → None → RuntimeError.
        """
        b = get_default_backend()
        acts = _acts(b)
        with pytest.raises(RuntimeError):
            stage_density(
                source_activations={1: acts},  # key 0 missing
                target_activations={0: acts},
                probe_ids=[],
                probe_domains=[],
                layers=[0],
                layer_mapping=None,            # no remapping — looks up key 0
                backend=b,
            )


# ---------------------------------------------------------------------------
# D4: filter_core_probes_by_graft_mask (pure function — no backend needed)
# ---------------------------------------------------------------------------

class TestFilterCoreProbes:
    def test_none_graft_mask_returns_all_probes(self):
        """graft_mask=None means graft everything."""
        probes = {"p1", "p2", "p3"}
        result = filter_core_probes_by_graft_mask(probes, layer_idx=0, graft_mask=None)
        assert result == probes

    def test_empty_graft_mask_returns_no_probes(self):
        """Empty graft_mask has no True entries → nothing grafted."""
        probes = {"p1", "p2"}
        result = filter_core_probes_by_graft_mask(probes, layer_idx=0, graft_mask={})
        assert result == set()

    def test_filters_to_grafted_probes_only(self):
        """Only probes with graft_mask[probe_id][layer_idx]=True are returned."""
        probes = {"p1", "p2", "p3"}
        graft_mask = {
            "p1": {0: True},
            "p2": {0: False},
            "p3": {0: True},
        }
        result = filter_core_probes_by_graft_mask(probes, layer_idx=0, graft_mask=graft_mask)
        assert result == {"p1", "p3"}

    def test_wrong_layer_excluded(self):
        """True entry at layer 1 does not cause inclusion at layer 0."""
        graft_mask = {"p1": {1: True}}   # True only at layer 1
        result = filter_core_probes_by_graft_mask({"p1"}, layer_idx=0, graft_mask=graft_mask)
        assert result == set()

    def test_probe_absent_from_mask_is_excluded(self):
        """A probe not in graft_mask defaults to False (no graft)."""
        graft_mask = {"p2": {0: True}}
        result = filter_core_probes_by_graft_mask({"p1", "p2"}, layer_idx=0, graft_mask=graft_mask)
        assert result == {"p2"}
