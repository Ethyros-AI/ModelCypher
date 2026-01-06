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

"""Tests for metaphor_convergence_analyzer.py."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch

from modelcypher.core.domain.geometry.metaphor_convergence_analyzer import (
    MetaphorConvergenceAnalyzer,
    DimensionAlignmentBuilder,
    AlignedDimension,
    DimensionAlignment,
)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    ModelFingerprints,
    ProbeSpace,
    ActivationFingerprint,
    ActivatedDimension,
)


class TestMetaphorConvergenceAnalyzer:
    """Tests for cross-family metaphor convergence."""

    def test_init(self):
        """It initializes."""
        analyzer = MetaphorConvergenceAnalyzer()
        assert analyzer is not None

    def test_alignment_classes(self):
        """Data classes instantiate correctly."""
        ad = AlignedDimension(source_dim=1, target_dim=2, weight=0.5)
        assert ad.source_dim == 1
        
        da = DimensionAlignment(
            by_layer={0: [ad]},
            aligned_counts={0: 1},
            total_aligned=1
        )
        assert da.total_aligned == 1

    @patch("modelcypher.core.domain.geometry.metaphor_convergence_analyzer.get_metaphor_invariants")
    def test_analyze_empty(self, mock_get_inventory):
        """Analyze runs on mock inputs (even if empty results)."""
        # Mock inventory
        mock_inv = Mock()
        mock_inv.id = "Time:01"
        mock_inv.family.value = "Time"  # Mock enum key behavior locally if needed, or string
        # Actually enum_key(inv.family) is called.
        # Let's just mock the object returned by get_metaphor_invariants
        
        # Real inventory has .family (enum) and .id
        # We need to mock that structure
        mock_item = Mock()
        mock_item.id = "Time:01"
        mock_item.family = Mock() 
        # enum_key(inv.family) usually returns string value of enum
        # If we mock enum_key in the module, or just make sure family works.
        
        # Easier: patch enum_key as well? Or just return an object where family.value="Time"
        # Since logic calls enum_key(inv.family), lets assume enum_key returns str(family) or family.value
        
        analyzer = MetaphorConvergenceAnalyzer()
        
        # Mock ModelFingerprints
        fp_source = MagicMock(spec=ModelFingerprints)
        fp_source.model_id = "SourceModel"
        fp_source.layer_count = 2
        fp_source.probe_space = ProbeSpace.prelogits_hidden
        fp_source.hidden_dim = 10
        
        fp_target = MagicMock(spec=ModelFingerprints)
        fp_target.model_id = "TargetModel"
        fp_target.layer_count = 2
        fp_target.probe_space = ProbeSpace.prelogits_hidden
        fp_target.hidden_dim = 10

        # Run analyze - will fail if inventory empty
        mock_get_inventory.return_value = []
        with pytest.raises(ValueError, match="No metaphor invariants registered"):
            analyzer.analyze(fp_source, fp_target)

    @patch("modelcypher.core.domain.geometry.metaphor_convergence_analyzer.get_metaphor_invariants")
    @patch("modelcypher.core.domain.geometry.metaphor_convergence_analyzer.enum_key")
    def test_analyze_success(self, mock_enum_key, mock_get_inventory):
        """Analyze runs successfully with populated fingerprints."""
        mock_enum_key.side_effect = lambda x: str(x)
        
        # Mock Inventory
        mock_inv = Mock()
        mock_inv.id = "Time:01"
        mock_inv.family = "Time"
        mock_get_inventory.return_value = [mock_inv]
        
        # Mock Source Fingerprints
        fp_source = MagicMock(spec=ModelFingerprints)
        fp_source.model_id = "Source"
        fp_source.layer_count = 2
        fp_source.hidden_dim = 4
        fp_source.probe_space = ProbeSpace.prelogits_hidden
        
        # Create a mock fingerprint
        # Need "metaphor_invariant:Time:01" as prime_id
        src_fp = MagicMock(spec=ActivationFingerprint)
        src_fp.prime_id = "metaphor_invariant:Time:01"
        
        # Mock activated dimensions: dict[layer, list[ActivatedDimension]]
        # Layer 0 has dim 0 activated with 1.0
        dims = [ActivatedDimension(index=0, activation=1.0)]
        src_fp.activated_dimensions = {0: dims}
        fp_source.fingerprints = [src_fp]

        # Mock Target Fingerprints
        fp_target = MagicMock(spec=ModelFingerprints)
        fp_target.model_id = "Target"
        fp_target.layer_count = 2
        fp_target.hidden_dim = 4
        fp_target.probe_space = ProbeSpace.prelogits_hidden
        
        tgt_fp = MagicMock(spec=ActivationFingerprint)
        tgt_fp.prime_id = "metaphor_invariant:Time:01"
        tgt_fp.activated_dimensions = {0: [ActivatedDimension(index=0, activation=1.0)]}
        fp_target.fingerprints = [tgt_fp]
        
        analyzer = MetaphorConvergenceAnalyzer()
        report = analyzer.analyze(fp_source, fp_target)
        
        assert report is not None
        assert report.models.model_a == "Source"
        assert report.models.model_b == "Target"
        assert "Time" in report.families
        # Since source/target are identical on layer 0, should match
        assert report.families["Time"].mean_cosine is not None
        
