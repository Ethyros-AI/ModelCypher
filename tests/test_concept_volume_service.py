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

"""Tests for ConceptVolumeService."""

from __future__ import annotations

from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.concept_volume_service import (
    ConceptVolumeAnalysisResult,
    ConceptVolumeService,
)


class _MockActivationProvider:
    """Mock activation provider that returns deterministic activations.

    Each prompt maps to a fixed activation vector based on the cluster it belongs to.
    """

    def __init__(self, prompt_to_activation: dict[str, list[float]], num_layers: int = 16):
        self._prompt_to_activation = prompt_to_activation
        self._num_layers = num_layers

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        backend = get_default_backend()
        base = self._prompt_to_activation.get(text)
        if base is None:
            # Fallback: hash-based activation
            h = hash(text) % 1000
            base = [float(h)] * 4
        arr = backend.array(base)
        backend.eval(arr)
        return {i: arr for i in range(self._num_layers)}


def _make_service(prompt_to_activation: dict[str, list[float]]) -> ConceptVolumeService:
    backend = get_default_backend()
    provider = _MockActivationProvider(prompt_to_activation)
    return ConceptVolumeService(activation_provider=provider, backend=backend)


def test_two_separated_clusters():
    """Two structurally different concept clusters produce valid pairwise relations."""
    # Cluster A: variance along dim 0-1 only
    # Cluster B: variance along dim 2-3 only (orthogonal structure)
    prompts = {
        "a1": [5.0, 0.0, 0.0, 0.0],
        "a2": [0.0, 5.0, 0.0, 0.0],
        "a3": [3.0, 4.0, 0.0, 0.0],
        "b1": [0.0, 0.0, 5.0, 0.0],
        "b2": [0.0, 0.0, 0.0, 5.0],
        "b3": [0.0, 0.0, 3.0, 4.0],
    }
    concepts = {
        "cluster_a": ["a1", "a2", "a3"],
        "cluster_b": ["b1", "b2", "b3"],
    }
    service = _make_service(prompts)
    result = service.analyze_concept_volumes(
        model=None, tokenizer=None, concepts=concepts, layer_idx=0,
    )

    assert isinstance(result, ConceptVolumeAnalysisResult)
    assert len(result.concept_stats) == 2
    assert len(result.pairwise_relations) == 1

    rel = result.pairwise_relations[0]
    # Pairwise metrics should be finite and valid
    assert 0.0 <= rel.overlap_coefficient <= 1.0
    assert 0.0 <= rel.subspace_alignment <= 1.0
    assert rel.centroid_distance >= 0.0


def test_identical_concepts():
    """Two identical concept clusters should have high overlap and near-zero distance."""
    prompts = {
        "p1": [1.0, 2.0, 3.0, 4.0],
        "p2": [1.1, 2.1, 3.1, 4.1],
        "p3": [0.9, 1.9, 2.9, 3.9],
    }
    # Both concepts use the SAME prompts
    concepts = {
        "concept_x": ["p1", "p2", "p3"],
        "concept_y": ["p1", "p2", "p3"],
    }
    service = _make_service(prompts)
    result = service.analyze_concept_volumes(
        model=None, tokenizer=None, concepts=concepts, layer_idx=0,
    )

    assert len(result.concept_stats) == 2
    assert len(result.pairwise_relations) == 1

    rel = result.pairwise_relations[0]
    # Identical data should yield high overlap
    assert rel.overlap_coefficient > 0.8
    # Distance should be very small
    assert rel.centroid_distance < 0.1


def test_single_concept_no_pairwise():
    """A single concept should produce stats but no pairwise relations."""
    prompts = {
        "s1": [5.0, 5.0, 5.0, 5.0],
        "s2": [5.1, 4.9, 5.0, 5.0],
        "s3": [4.9, 5.1, 5.0, 5.0],
    }
    concepts = {"only_concept": ["s1", "s2", "s3"]}
    service = _make_service(prompts)
    result = service.analyze_concept_volumes(
        model=None, tokenizer=None, concepts=concepts, layer_idx=0,
    )

    assert len(result.concept_stats) == 1
    assert len(result.pairwise_relations) == 0

    stats = result.concept_stats[0]
    assert stats.concept_id == "only_concept"
    assert stats.num_samples == 3
    assert stats.dimension == 4
    assert stats.geodesic_radius >= 0
    assert stats.volume >= 0


def test_result_serialization():
    """to_dict() should produce expected structure."""
    prompts = {
        "x1": [1.0, 0.0, 0.0, 0.0],
        "x2": [0.0, 1.0, 0.0, 0.0],
        "y1": [0.0, 0.0, 1.0, 0.0],
        "y2": [0.0, 0.0, 0.0, 1.0],
    }
    concepts = {"alpha": ["x1", "x2"], "beta": ["y1", "y2"]}
    service = _make_service(prompts)
    result = service.analyze_concept_volumes(
        model=None, tokenizer=None, concepts=concepts, layer_idx=0,
    )

    d = result.to_dict()
    assert "layer_idx" in d
    assert d["layer_idx"] == 0
    assert "concept_stats" in d
    assert "pairwise_relations" in d
    assert len(d["concept_stats"]) == 2
    assert len(d["pairwise_relations"]) == 1

    # Check concept stats fields
    for cs in d["concept_stats"]:
        assert "concept_id" in cs
        assert "num_samples" in cs
        assert "influence_type" in cs
        assert "geodesic_radius" in cs
        assert "effective_radius" in cs
        assert "volume" in cs
        assert "dimension" in cs

    # Check pairwise fields
    pr = d["pairwise_relations"][0]
    assert "concept_a" in pr
    assert "concept_b" in pr
    assert "overlap_coefficient" in pr
    assert "centroid_distance" in pr
    assert "curvature_divergence" in pr
    assert "subspace_alignment" in pr


def test_empty_concepts():
    """Empty concepts dict should return empty result."""
    service = _make_service({})
    result = service.analyze_concept_volumes(
        model=None, tokenizer=None, concepts={}, layer_idx=0,
    )
    assert len(result.concept_stats) == 0
    assert len(result.pairwise_relations) == 0
    assert result.layer_idx == 0
