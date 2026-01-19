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

"""Tests for shared_subspace_projector.py."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, Mock
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
from modelcypher.core.domain.geometry.shared_subspace_projector import SharedSubspaceProjector
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix

class TestSharedSubspaceProjector:
    """Tests for CCA alignment discovery."""

    def setup_method(self):
        self.backend = get_default_backend()

    def _create_mock_crm(self, layer: int, data: dict[str, list[float]], dim: int):
        """Helper to create a mock CRM."""
        crm = MagicMock(spec=ConceptResponseMatrix)
        crm.anchor_metadata = Mock()
        crm.anchor_metadata.anchor_ids = list(data.keys())
        
        layer_acts = {}
        for anchor_id, vec in data.items():
            act = Mock()
            act.activation = vec
            layer_acts[anchor_id] = act
            
        crm.activations = {layer: layer_acts}
        return crm

    def test_discover_exact_alignment(self):
        """CCA should find exact alignment for identical datasets."""
        # Dataset: 3 points in 2D
        # p1: [1, 0]
        # p2: [0, 1]
        # p3: [1, 1]
        data = {
            "a1": [1.0, 0.0],
            "a2": [0.0, 1.0],
            "a3": [1.0, 1.0],
            "a4": [0.0, 0.0], # Need enough points for covariance
            "a5": [0.5, 0.5],
        }
        
        crm1 = self._create_mock_crm(1, data, 2)
        crm2 = self._create_mock_crm(1, data, 2) # Identical
        
        result = SharedSubspaceProjector.discover(crm1, crm2, layer=1)
        
        assert result is not None
        assert result.shared_dimension > 0
        # For identical data, top correlation should be close to 1.0.
        # CCA correlation error scales with O(sqrt(eps) * cond) for small matrices.
        eps = regularization_epsilon(self.backend, self.backend.array([1.0]))
        # Allow 10x tolerance for numerical precision in small-matrix CCA
        assert abs(result.alignment_strengths[0] - 1.0) <= 10 * eps
        # Alignment error should be small but scales with numerical precision
        assert result.alignment_error < 10 * eps

    def test_discover_orthogonal(self):
        """Orthogonal datasets might still have some structure if dimensions align by chance, but here we construct unrelated."""
        # Unlikely to be perfectly 0 unless specific construction, but check it runs.
        data1 = {
            "a1": [1.0, 0.0],
            "a2": [0.0, 1.0],
            "a3": [0.0, 0.0],
            "a4": [1.0, 1.0],
        }
        # Data2 is just noise or different axis
        data2 = {
            "a1": [0.0, 0.1],
            "a2": [0.1, 0.0],
            "a3": [0.0, 0.0],
            "a4": [0.1, 0.1],
        }
        
        crm1 = self._create_mock_crm(1, data1, 2)
        crm2 = self._create_mock_crm(1, data2, 2)
        
        result = SharedSubspaceProjector.discover(crm1, crm2, layer=1)
        # Should run without error
        if result is not None:
             assert result.source_dimension == 2
             assert result.target_dimension == 2

    def test_insufficient_samples(self):
        """Should return None if not enough samples."""
        data = {"a1": [1.0, 0.0]} # Only 1 sample
        crm1 = self._create_mock_crm(1, data, 2)
        crm2 = self._create_mock_crm(1, data, 2)
        
        result = SharedSubspaceProjector.discover(crm1, crm2, layer=1)
        assert result is None
