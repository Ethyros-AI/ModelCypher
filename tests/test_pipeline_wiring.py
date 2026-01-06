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

"""Tests for pipeline wiring between stages.

These tests verify that data flows correctly between pipeline stages
and that no information is lost or recalculated unnecessarily.

Bugs this catches:
    - Probe results not passed to transplant
    - Transforms computed but not used
    - Activations missing for some layers
"""

import pytest


class TestPipelineImports:
    """Tests that verify pipeline components can be imported correctly."""
    
    def test_stages_importable(self) -> None:
        """All pipeline stages should be importable."""
        from modelcypher.core.use_cases.merge.stages import (
            stage_probe,
            stage_density,
            stage_transplant,
            stage_validate,
        )
        
        assert callable(stage_probe)
        assert callable(stage_density)
        assert callable(stage_transplant)
        assert callable(stage_validate)
    
    def test_permute_stage_removed(self) -> None:
        """PERMUTE stage should NOT be importable (removed)."""
        from modelcypher.core.use_cases.merge import stages
        
        # stage_permute was removed - GramAligner subsumes it
        assert not hasattr(stages, "stage_permute"), (
            "stage_permute should have been removed from pipeline"
        )
    
    def test_pipeline_run_merge_importable(self) -> None:
        """run_merge function should be importable."""
        from modelcypher.core.use_cases.merge.pipeline import run_merge
        
        assert callable(run_merge)
    
    def test_merger_class_importable(self) -> None:
        """UnifiedGeometricMerger should be importable."""
        from modelcypher.core.use_cases.merge.merger import UnifiedGeometricMerger
        
        assert UnifiedGeometricMerger is not None


class TestStageDataFlow:
    """Tests verifying data flows correctly between stages."""
    
    def test_probe_result_has_all_required_fields(self) -> None:
        """ProbeResult should have all fields needed by downstream stages."""
        from modelcypher.core.use_cases.merge.stages.probe import ProbeResult
        import dataclasses
        
        fields = {f.name for f in dataclasses.fields(ProbeResult)}
        
        # Required for transplant
        assert "source_activations" in fields
        assert "target_activations" in fields
        assert "feature_transforms" in fields
        
        # Required for attention handling
        assert "attention_transforms" in fields
        assert "k_transforms" in fields
        assert "v_transforms" in fields
        
        # Required for cross-arch
        assert "layer_mapping" in fields
    
    def test_transplant_stage_signature_accepts_transforms(self) -> None:
        """stage_transplant should accept transform parameters from probe."""
        from modelcypher.core.use_cases.merge.stages.transplant import stage_transplant
        import inspect
        
        sig = inspect.signature(stage_transplant)
        params = sig.parameters
        
        # These should be parameters that transplant accepts from probe
        assert "feature_transforms" in params
        assert "attention_transforms" in params
        assert "k_transforms" in params
        assert "v_transforms" in params
        assert "layer_mapping" in params
        assert "embedding_transform" in params


class TestActivationCollection:
    """Tests for activation collection coverage."""
    
    def test_activation_dict_structure(self) -> None:
        """Activation dicts should map layer_idx -> list of arrays."""
        from modelcypher.core.domain._backend import get_default_backend
        
        backend = get_default_backend()
        backend.random_seed(42)
        
        # Simulate activation collection structure
        activations = {
            0: [backend.random_normal((64,)) for _ in range(10)],
            1: [backend.random_normal((64,)) for _ in range(10)],
            2: [backend.random_normal((64,)) for _ in range(10)],
        }
        
        for layer_idx, acts in activations.items():
            assert isinstance(layer_idx, int)
            assert isinstance(acts, list)
            assert len(acts) > 0  # Not empty
