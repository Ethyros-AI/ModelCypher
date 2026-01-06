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

"""Tests for intersection_similarity.py."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock
from modelcypher.core.domain.geometry.intersection_similarity import (
    compute_jaccard_similarity,
    compute_weighted_jaccard_similarity,
    compute_cosine_similarity,
    build_intersection_map,
    IntersectionSimilarityMode,
)
from modelcypher.core.domain.geometry.manifold_stitcher import ActivationFingerprint, ActivatedDimension

class TestIntersectionSimilarity:
    """Tests for similarity metrics and map building."""

    def test_compute_jaccard_similarity(self):
        """Standard Jaccard index test."""
        set_a = {1, 2, 3}
        set_b = {2, 3, 4}
        # Intersection: {2, 3} (2)
        # Union: {1, 2, 3, 4} (4)
        # Result: 0.5
        assert compute_jaccard_similarity(set_a, set_b) == 0.5
        
        # Empty sets
        assert compute_jaccard_similarity(set(), set()) == 0.0
        assert compute_jaccard_similarity({1}, set()) == 0.0

    def test_compute_weighted_jaccard_similarity(self):
        """Weighted Jaccard (Ruzicka similarity)."""
        dict_a = {1: 1.0, 2: 0.5}
        dict_b = {1: 0.5, 2: 0.5, 3: 0.2}
        
        # min(a,b): 1->0.5, 2->0.5, 3->0 (implied 0 in a)
        # sum_min = 0.5 + 0.5 + 0 = 1.0
        
        # max(a,b): 1->1.0, 2->0.5, 3->0.2
        # sum_max = 1.0 + 0.5 + 0.2 = 1.7
        
        # 1.0 / 1.7 approx 0.588
        score = compute_weighted_jaccard_similarity(dict_a, dict_b)
        assert abs(score - (1.0 / 1.7)) < 1e-5

    def test_compute_cosine_similarity(self):
        """Cosine similarity for sparse dicts."""
        # Orthogonal
        d1 = {1: 1.0}
        d2 = {2: 1.0}
        assert abs(compute_cosine_similarity(d1, d2)) < 1e-5
        
        # Aligned
        d3 = {1: 1.0, 2: 0.0}
        d4 = {1: 2.0, 2: 0.0}
        assert abs(compute_cosine_similarity(d3, d4) - 1.0) < 1e-5

    def _create_mock_fingerprint(self, dims: list[tuple[int, float]]):
        fp = Mock(spec=ActivationFingerprint)
        act_dims = []
        for idx, val in dims:
            ad = Mock(spec=ActivatedDimension)
            ad.index = idx
            ad.activation = val
            act_dims.append(ad)
        
        # Structure is dict[layer, list[ActivatedDimension]]
        fp.activated_dimensions = {0: act_dims}
        fp.prime_id = "prime_1" # Default ID
        return fp

    def test_build_intersection_map(self):
        """Test building intersection map from fingerprints."""
        # Source: dim 1 active
        src_fp = self._create_mock_fingerprint([(1, 1.0)])
        # Target: dim 2 active
        tgt_fp = self._create_mock_fingerprint([(2, 1.0)])
        
        # If we just test 1 layer
        imap = build_intersection_map(
            source_fingerprints=[src_fp],
            target_fingerprints=[tgt_fp],
            source_model="A",
            target_model="B",
            mode=IntersectionSimilarityMode.JACCARD
        )
        
        assert imap.source_model == "A"
        assert imap.target_model == "B"
        # Since fingerprints match (only 1 pair), we calculate correlation.
        # But wait, correlation is between DIMENSIONS?
        # Manifold Stitcher usually correlates source dim X with target dim Y across MANY fingerprints (probes).
        # Here we have only 1 fingerprint (1 probe).
        # If we have 1 probe where Src Dim 1=1.0 and Tgt Dim 2=1.0
        # Correlation between S1 and T2 is perfect (if we ignore lack of variance or assume boolean presence).
        # But build_intersection_map aggregates across fingerprints.
        
        # Actually `build_layer_correlations` iterates:
        # For each source dim X, for each target dim Y:
        # Compute similarity between vector(X across probes) and vector(Y across probes).
        
        # So providing 1 probe is enough to establish co-occurrence.
        
        # Let's verify results structure
        # Check if layer 0 is present
        assert 0 in imap.dimension_correlations
