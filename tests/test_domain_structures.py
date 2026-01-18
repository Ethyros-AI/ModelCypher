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

"""Tests for domain data structures and utilities."""

from __future__ import annotations

import math
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass

from modelcypher.core.domain.geometry.domain_signal_profile import DomainSignalProfile, LayerSignal
from modelcypher.core.domain.geometry.manifold_profile import (
    ManifoldPoint,
    ManifoldRegion,
    RegionThresholds,
    ManifoldProfile,
)
from modelcypher.core.domain.geometry.spectral_analysis import (
    compute_spectral_metrics,
    SpectralMetrics,
)
from modelcypher.core.domain.geometry.signature_base import SignatureMixin
from modelcypher.core.domain._backend import get_default_backend

@dataclass
class SimpleSignature(SignatureMixin):
    values: list[float]

class TestSignatureMixin:
    """Tests for signature base class."""
    
    def test_l2_norm(self):
        s = SimpleSignature([3.0, 4.0])
        assert s.l2_norm() == 5.0

    def test_cosine_similarity(self):
        s1 = SimpleSignature([1.0, 0.0])
        s2 = SimpleSignature([0.0, 1.0])
        # Orthogonal
        assert abs(s1.cosine_similarity(s2)) < math.ulp(1.0)
        
        s3 = SimpleSignature([2.0, 0.0])
        # Aligned
        assert abs(s1.cosine_similarity(s3) - 1.0) < math.ulp(1.0)

    def test_normalization(self):
        s = SimpleSignature([3.0, 4.0])
        norm = s.l2_normalized()
        assert abs(norm.values[0] - 0.6) < math.ulp(0.6)
        assert abs(norm.values[1] - 0.8) < math.ulp(0.8)

class TestDomainSignalProfile:
    """Tests for DomainSignalProfile serialization and creation."""

    def test_create_and_serialize(self):
        layer_signals = {
            1: LayerSignal(sparsity=0.5, gradient_variance=0.1)
        }
        profile = DomainSignalProfile.create(
            layer_signals=layer_signals,
            model_id="test_model",
            domain="test_domain",
            baseline_domain="test_baseline",
            total_layers=2,
            prompt_count=10,
            max_tokens_per_prompt=100,
            notes="test notes"
        )
        
        assert profile.model_id == "test_model"
        assert profile.generated_at is not None
        
        data = profile.to_dict()
        assert data["modelId"] == "test_model"
        assert data["layerSignals"]["1"]["sparsity"] == 0.5
        
        # Deserialize
        profile2 = DomainSignalProfile.from_dict(data)
        assert profile2.model_id == profile.model_id
        assert profile2.layer_signals[1].sparsity == 0.5


class TestManifoldStructures:
    """Tests for ManifoldPoint, Region, Profile."""

    def test_region_classification(self):
        """Test region classification logic."""
        # Thresholds: low < 0.2, high > 0.8
        thresholds = Mock(spec=RegionThresholds)
        thresholds.low_entropy = 0.2
        thresholds.high_entropy = 0.8
        thresholds.low_variance = 0.2
        thresholds.high_variance = 0.8
        thresholds.low_coherence = 0.2
        thresholds.high_coherence = 0.8
        
        # Dense point: low entropy, low variance, high coherence
        p_dense = Mock(spec=ManifoldPoint)
        p_dense.mean_entropy = 0.1
        p_dense.entropy_variance = 0.1
        p_dense.mean_gate_similarity = 0.9
        
        # classify is a static method usually taking centroid and thresholds?
        # ManifoldRegion.classify(centroid, thresholds)
        
        # ManifoldRegion.classify implementation logic:
        # if entropy < low and variance < low and coherence > high -> DENSE
        # elif entropy > high and variance > high and coherence < low -> SPARSE
        # else -> TRANSITIONAL
        
        # Mocking values on the point
        # But ManifoldRegion.classify accesses attributes directly.
        
        # I cannot mock `ManifoldPoint` easily if I pass it to real `classify` unless I set attrs.
        # But `classify` is likely checking: centroid.mean_entropy, etc.
        
        # Let's instantiate real ManifoldPoint with dummy values?
        # ManifoldPoint is a dataclass without init arg complications usually.
        # Wait, ManifoldPoint field list is long.
        
        # Easier to use Mock with attrs set.
        p = Mock()
        p.mean_entropy = 0.1
        p.entropy_variance = 0.1
        p.mean_gate_similarity = 0.9 # Coherence
        
        # Need to know attributes accessed logic precisely.
        # Outline says "classify(centroid: ManifoldPoint, thresholds: RegionThresholds)"
        
        region_type = ManifoldRegion.classify(p, thresholds)
        assert region_type == ManifoldRegion.RegionCharacter.DENSE
        
        # Sparse point
        p.mean_entropy = 0.9
        p.entropy_variance = 0.1 # High variance -> Transitional
        p.mean_gate_similarity = 0.1
        region_type = ManifoldRegion.classify(p, thresholds)
        assert region_type == ManifoldRegion.RegionCharacter.SPARSE

    @patch("modelcypher.core.domain.geometry.riemannian_utils.RiemannianGeometry")
    def test_manifold_point_distance(self, MockRG):
        """Test geodesic distance calculation between points."""
        backend = get_default_backend()
        
        # Mock backend behavior for distance
        rg_instance = MockRG.return_value
        # geodesic_distance_result returns a result object with .distance
        mock_res = Mock()
        # Use real array for distances so backend ops work
        # 2x2 matrix: [[0, 1.5], [1.5, 0]]
        mock_res.distances = backend.array([[0.0, 1.5], [1.5, 0.0]])
        rg_instance.geodesic_distances.return_value = mock_res
        
        p1 = ManifoldPoint(
            mean_entropy=0.5, entropy_variance=0.1, first_token_entropy=0.5,
            gate_count=5, mean_gate_similarity=0.5, dominant_gate_category=1.0,
            entropy_path_correlation=0.5, assessment_strength=0.5,
            prompt_hash="h1"
        )
        p2 = ManifoldPoint(
            mean_entropy=0.6, entropy_variance=0.1, first_token_entropy=0.6,
            gate_count=5, mean_gate_similarity=0.6, dominant_gate_category=1.0,
            entropy_path_correlation=0.6, assessment_strength=0.6,
            prompt_hash="h2"
        )
        
        dist = p1.distance(p2)
        assert dist == 1.5
        # Verify call args? Not critical if result matches.

class TestSpectralAnalysis:
    """Tests for spectral metrics computation."""
    
    def test_compute_spectral_metrics(self):
        backend = get_default_backend()
        
        # Simple diagonal arrays
        # Source singular values: [10, 5]
        # Target singular values: [5, 2.5] (Ratio 2.0)
        
        # Create weights. U @ S @ Vt.
        # Identity U, Vt. W = diag(S)
        w_s = backend.eye(2) * 10.0
        # Wait, eye(2) * 10 is [10, 0; 0, 10]. Both SVs are 10.
        # Need [10, 0; 0, 5].
        # Can use backend.diag([10., 5.]) if supported, or manual construction logic?
        # Or standard lists: [[10., 0.], [0., 5.]] converted to array.
        
        w_s = backend.array([[10.0, 0.0], [0.0, 5.0]])
        w_t = backend.array([[5.0, 0.0], [0.0, 2.5]])
        
        # Call compute
        # Note: compute_spectral_metrics calculates ratio on LARGEST singular value.
        # S_max(src) = 10. S_max(tgt) = 5. Ratio = 2.0.
        # Condition number = max/min. Src: 10/5=2. Tgt: 5/2.5=2.
        
        metrics = compute_spectral_metrics(w_s, w_t, backend=backend)
        
        assert metrics.spectral_ratio == 2.0
        assert metrics.spectral_ratio_symmetry == 0.5 # 1/2.0
        assert abs(metrics.condition_number - 2.0) < math.ulp(2.0)
