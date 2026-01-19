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

"""Tests for interference_predictor.py."""

from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock, Mock, PropertyMock

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.interference_predictor import (
    MergeAnalyzer,
    MergeAnalysisResult,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_density import (
    ConceptVolume,
    ConceptVolumeRelation,
)

class TestInterferencePredictor:
    """Tests for MergeAnalyzer and geometric metrics."""

    def test_analyze_with_precomputed_relation(self):
        """Test analysis logic when relation is provided."""
        
        # Mock volumes
        vol_a = MagicMock(spec=ConceptVolume)
        vol_a.concept_id = "A"
        type(vol_a).effective_radius = PropertyMock(return_value=1.0)
        
        vol_b = MagicMock(spec=ConceptVolume)
        vol_b.concept_id = "B"
        type(vol_b).effective_radius = PropertyMock(return_value=1.0)
        
        # Mock relation
        # High overlap, low distance -> conflict/interference likely if not aligned
        relation = Mock(spec=ConceptVolumeRelation)
        relation.volume_a = vol_a
        relation.volume_b = vol_b
        
        # Overlap metrics (0 to 1)
        relation.overlap_coefficient = 0.8
        relation.jaccard_index = 0.7
        relation.bhattacharyya_coefficient = 0.9
        
        # Distance metrics
        relation.centroid_distance = 0.5 # Close (radius sum is 2.0)
        relation.geodesic_centroid_distance = 0.5
        
        # Geometry
        relation.curvature_divergence = 0.1
        relation.subspace_alignment = 0.95 # Aligned
        
        analyzer = MergeAnalyzer()
        # Mock the density estimator to avoid real init if possible, 
        # but analyze() won't call it if relation is provided.
        # But init creates one. It's fine if it's just instantiated.
        
        result = analyzer.analyze(vol_a, vol_b, relation=relation)
        
        assert isinstance(result, MergeAnalysisResult)
        assert result.volume_a_id == "A"
        assert result.volume_b_id == "B"
        expected_overlap = (0.8 + 0.7 + 0.9) / 3.0
        expected_distance = 0.5 / (1.0 + 1.0)
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([expected_overlap, expected_distance]))
        assert abs(result.overlap_score - expected_overlap) <= eps
        assert abs(result.distance_score - expected_distance) <= eps
        aligned_expected = abs(0.95 - 1.0) <= machine_epsilon(backend, backend.array([1.0]))
        assert result.aligned == aligned_expected

    def test_compute_distance_score(self):
        """Distance score should be normalized by radii."""
        analyzer = MergeAnalyzer()
        
        vol_a = MagicMock(spec=ConceptVolume)
        type(vol_a).effective_radius = PropertyMock(return_value=2.0)
        vol_b = MagicMock(spec=ConceptVolume)
        type(vol_b).effective_radius = PropertyMock(return_value=2.0)
        
        relation = Mock()
        relation.volume_a = vol_a
        relation.volume_b = vol_b
        relation.centroid_distance = 4.0 # Equals sum of radii
        relation.geodesic_centroid_distance = 4.0
        
        # Distance score = dist / (r1 + r2) = 4 / 4 = 1.0
        score = analyzer._compute_distance_score(relation)
        tol = math.ulp(1.0)
        assert abs(score - 1.0) <= tol
