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

"""Tests for the merge pipeline service."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from modelcypher.core.use_cases.merge.service import (
    MergePipelineService,
    PostMergeValidation,
    PreMergeAnalysis,
    PipelineResult,
)


@pytest.fixture
def merge_pipeline_service():
    """Create MergePipelineService with mock dependencies."""
    waypoint_service = MagicMock()
    geometric_merger = MagicMock()
    model_loader = MagicMock()
    return MergePipelineService(
        waypoint_service=waypoint_service,
        geometric_merger=geometric_merger,
        model_loader=model_loader,
    )


class TestPreMergeAnalysis:
    """Tests for PreMergeAnalysis dataclass."""

    def test_create_pre_merge_analysis(self):
        """Test creating a PreMergeAnalysis instance."""
        analysis = PreMergeAnalysis(
            source_model="/path/to/source",
            target_model="/path/to/target",
            timestamp="2025-12-31T00:00:00",
            domains_analyzed=["spatial", "social"],
            domain_results={
                "spatial": {"mean_overlap": 0.8, "mean_subspace_alignment": 0.9},
                "social": {"mean_overlap": 0.7, "mean_subspace_alignment": 0.85},
            },
            mean_overlap=0.75,
            mean_subspace_alignment=0.875,
            mean_curvature_divergence=0.05,
            mean_distance=0.15,
            aligned_pairs=4,
        )

        assert analysis.source_model == "/path/to/source"
        assert analysis.target_model == "/path/to/target"
        assert len(analysis.domains_analyzed) == 2
        assert analysis.mean_overlap == 0.75
        assert analysis.mean_subspace_alignment == 0.875
        assert analysis.mean_curvature_divergence == 0.05

    def test_pre_merge_analysis_is_frozen(self):
        """Test that PreMergeAnalysis is immutable."""
        analysis = PreMergeAnalysis(
            source_model="/path/to/source",
            target_model="/path/to/target",
            timestamp="2025-12-31T00:00:00",
            domains_analyzed=[],
            domain_results={},
            mean_overlap=0.0,
            mean_subspace_alignment=0.0,
            mean_curvature_divergence=0.0,
            mean_distance=0.0,
            aligned_pairs=0,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            analysis.mean_overlap = 0.5  # type: ignore

    def test_pre_merge_analysis_to_dict(self):
        """Test converting PreMergeAnalysis to dict."""
        analysis = PreMergeAnalysis(
            source_model="/path/to/source",
            target_model="/path/to/target",
            timestamp="2025-12-31T00:00:00",
            domains_analyzed=["spatial"],
            domain_results={"spatial": {"mean_overlap": 0.8}},
            mean_overlap=0.8,
            mean_subspace_alignment=0.9,
            mean_curvature_divergence=0.0,
            mean_distance=0.0,
            aligned_pairs=0,
        )

        d = asdict(analysis)
        assert d["source_model"] == "/path/to/source"
        assert d["mean_overlap"] == 0.8
        assert isinstance(d["domains_analyzed"], list)


class TestPostMergeValidation:
    """Tests for PostMergeValidation dataclass."""

    def test_create_post_merge_validation(self):
        """Test creating a PostMergeValidation instance."""
        validation = PostMergeValidation(
            merged_model="/path/to/merged",
            timestamp="2025-12-31T00:00:00",
            geometry_metrics={"mean_preserved_fraction": 0.9},
            layers_transplanted=24,
            weights_transplanted=48,
            mean_preserved_fraction=0.9,
            mean_cka_after=0.95,
        )

        assert validation.merged_model == "/path/to/merged"
        assert validation.layers_transplanted == 24
        assert validation.mean_cka_after == 0.95

    def test_post_merge_validation_is_frozen(self):
        """Test that PostMergeValidation is immutable."""
        validation = PostMergeValidation(
            merged_model="/path/to/merged",
            timestamp="2025-12-31T00:00:00",
            geometry_metrics={},
            layers_transplanted=0,
            weights_transplanted=0,
            mean_preserved_fraction=0.0,
            mean_cka_after=0.0,
        )

        with pytest.raises(Exception):
            validation.mean_preserved_fraction = 0.5  # type: ignore


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_create_pipeline_result(self):
        """Test creating a PipelineResult instance."""
        pre_merge = PreMergeAnalysis(
            source_model="/path/to/source",
            target_model="/path/to/target",
            timestamp="2025-12-31T00:00:00",
            domains_analyzed=["spatial"],
            domain_results={},
            mean_overlap=0.8,
            mean_subspace_alignment=0.9,
            mean_curvature_divergence=0.0,
            mean_distance=0.0,
            aligned_pairs=0,
        )

        post_merge = PostMergeValidation(
            merged_model="/path/to/merged",
            timestamp="2025-12-31T00:00:00",
            geometry_metrics={},
            layers_transplanted=24,
            weights_transplanted=48,
            mean_preserved_fraction=0.9,
            mean_cka_after=0.95,
        )

        result = PipelineResult(
            pipeline_id="pipeline-abc123",
            timestamp="2025-12-31T00:00:00",
            source_model="/path/to/source",
            target_model="/path/to/target",
            output_dir="/path/to/output",
            pre_merge=pre_merge,
            merge_result={"layer_count": 24, "weight_count": 48},
            post_merge=post_merge,
            pre_merge_duration_s=10.5,
            merge_duration_s=120.0,
            validation_duration_s=5.0,
        )

        assert result.pipeline_id == "pipeline-abc123"
        assert result.source_model == "/path/to/source"
        assert result.pre_merge.mean_overlap == 0.8
        assert result.post_merge.mean_cka_after == 0.95
        assert result.merge_duration_s == 120.0

    def test_pipeline_result_with_merge_result(self):
        """Test PipelineResult with merge result dictionary."""
        pre_merge = PreMergeAnalysis(
            source_model="/s",
            target_model="/t",
            timestamp="2025-12-31T00:00:00",
            domains_analyzed=[],
            domain_results={},
            mean_overlap=0.0,
            mean_subspace_alignment=0.0,
            mean_curvature_divergence=0.0,
            mean_distance=0.0,
            aligned_pairs=0,
        )

        post_merge = PostMergeValidation(
            merged_model="/m",
            timestamp="2025-12-31T00:00:00",
            geometry_metrics={},
            layers_transplanted=0,
            weights_transplanted=0,
            mean_preserved_fraction=0.0,
            mean_cka_after=0.0,
        )

        merge_result = {
            "layer_count": 24,
            "weight_count": 48,
            "mean_preserved_fraction": 0.9,
        }

        result = PipelineResult(
            pipeline_id="pipeline-xyz",
            timestamp="2025-12-31T00:00:00",
            source_model="/s",
            target_model="/t",
            output_dir="/o",
            pre_merge=pre_merge,
            merge_result=merge_result,
            post_merge=post_merge,
        )

        assert result.merge_result is not None
        assert result.merge_result["layer_count"] == 24
        assert result.merge_result["mean_preserved_fraction"] == 0.9


class TestMergePipelineService:
    """Tests for MergePipelineService."""

    def test_init_with_dependencies(self, merge_pipeline_service):
        """Test initialization with injected dependencies."""
        assert merge_pipeline_service is not None

    def test_merge_result_to_dict(self, merge_pipeline_service):
        """Test converting merge result to dictionary."""
        # Create a mock merge result
        class MockMergeResult:
            output_path = "/output"
            layer_count = 24
            weight_count = 48
            mean_preserved_fraction = 0.85
            vocab_aligned = True
            mean_procrustes_error = 0.001
            merge_strategy = "null_space"
            probe_metrics = {}
            permute_metrics = {}
            geometry_metrics = {"mean_preserved_fraction": 0.9}
            transplant_metrics = {"layers_transplanted": 24}
            density_metrics = {}
            validation_metrics = {}
            post_merge_density = None
            refusal_preserved = None

        result = merge_pipeline_service._merge_result_to_dict(MockMergeResult())
        assert result["output_path"] == "/output"
        assert result["layer_count"] == 24
        assert result["mean_preserved_fraction"] == 0.85
        assert result["geometry_metrics"]["mean_preserved_fraction"] == 0.9


class TestPipelineServiceInternals:
    """Tests for internal pipeline service methods."""

    def test_merge_result_to_dict_with_geometry_metrics(self, merge_pipeline_service):
        """Test _merge_result_to_dict handles geometry_metrics correctly."""
        class MockMergeResult:
            output_path = "/output"
            layer_count = 12
            weight_count = 24
            mean_preserved_fraction = 0.92
            vocab_aligned = False
            mean_procrustes_error = 0.002
            merge_strategy = "null_space"
            probe_metrics = {}
            permute_metrics = {}
            geometry_metrics = {"mean_cka_after": 0.98}
            transplant_metrics = {"layers_transplanted": 12}
            density_metrics = {}
            validation_metrics = {}
            post_merge_density = None
            refusal_preserved = None

        result = merge_pipeline_service._merge_result_to_dict(MockMergeResult())
        assert result["layer_count"] == 12
        assert result["mean_preserved_fraction"] == 0.92
        assert result["geometry_metrics"]["mean_cka_after"] == 0.98


class TestPipelineTimingFields:
    """Tests for timing fields in pipeline result."""

    def test_timing_fields_default_to_zero(self):
        """Test that timing fields default to zero."""
        pre_merge = PreMergeAnalysis(
            source_model="/s",
            target_model="/t",
            timestamp="2025-12-31T00:00:00",
            domains_analyzed=[],
            domain_results={},
            mean_overlap=0.0,
            mean_subspace_alignment=0.0,
            mean_curvature_divergence=0.0,
            mean_distance=0.0,
            aligned_pairs=0,
        )

        post_merge = PostMergeValidation(
            merged_model="/m",
            timestamp="2025-12-31T00:00:00",
            geometry_metrics={},
            layers_transplanted=0,
            weights_transplanted=0,
            mean_preserved_fraction=0.0,
            mean_cka_after=0.0,
        )

        result = PipelineResult(
            pipeline_id="test",
            timestamp="2025-12-31T00:00:00",
            source_model="/s",
            target_model="/t",
            output_dir="/o",
            pre_merge=pre_merge,
            merge_result={},
            post_merge=post_merge,
        )

        assert result.pre_merge_duration_s == 0.0
        assert result.merge_duration_s == 0.0
        assert result.validation_duration_s == 0.0
