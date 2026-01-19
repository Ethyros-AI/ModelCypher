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

"""Comprehensive tests for AnchorInvarianceAnalyzer module.

Tests cover all public APIs:
- Exception classes (AnchorInvarianceError, NoRunsError, NoAnchorsError)
- Data classes (RunInput, RunModels, RunResult, AnchorScore, TopAnchor, Summary, Report, etc.)
- AnchorInvarianceAnalyzer.analyze() main entry point
- Helper methods (_build_anchor_vectors, _build_layer_alignment, _extract_family, etc.)
"""

from __future__ import annotations

import math
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_sparse
from modelcypher.core.domain.geometry.anchor_invariance_analyzer import (
    AnchorInvarianceAnalyzer,
    AnchorInvarianceError,
    AnchorScore,
    AnchorVectorIndex,
    LayerAlignment,
    NoAnchorsError,
    NoRunsError,
    Report,
    RunInput,
    RunModels,
    RunResult,
    Summary,
    TopAnchor,
)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    ActivatedDimension,
    ActivationFingerprint,
    ModelFingerprints,
    ProbeSpace,
)
from modelcypher.core.domain.geometry.metaphor_convergence_analyzer import (
    AlignedDimension,
    MetaphorConvergenceAnalyzer,
)


@pytest.fixture
def backend():
    """Get compute backend."""
    return get_default_backend()


# =============================================================================
# Test Fixtures: Synthetic Data Builders
# =============================================================================


def _make_fingerprint(
    prime_id: str,
    prime_text: str,
    layer_activations: dict[int, list[tuple[int, float]]],
) -> ActivationFingerprint:
    """Build an ActivationFingerprint from layer -> [(dim_idx, value), ...]."""
    activated_dims = {}
    for layer, dims_list in layer_activations.items():
        activated_dims[layer] = [
            ActivatedDimension(index=idx, activation=val) for idx, val in dims_list
        ]
    return ActivationFingerprint(
        prime_id=prime_id,
        prime_text=prime_text,
        activated_dimensions=activated_dims,
    )


def _make_model_fingerprints(
    model_id: str,
    layer_count: int,
    fingerprints: list[ActivationFingerprint],
    hidden_dim: int = 128,
) -> ModelFingerprints:
    """Build a ModelFingerprints collection."""
    return ModelFingerprints(
        model_id=model_id,
        probe_space=ProbeSpace.prelogits_hidden,
        probe_capture_key=None,
        hidden_dim=hidden_dim,
        layer_count=layer_count,
        fingerprints=fingerprints,
    )


def _make_simple_run_input(
    run_id: str,
    source_model_id: str = "source_model",
    target_model_id: str = "target_model",
    layer_count: int = 4,
    anchor_ids: list[str] | None = None,
) -> RunInput:
    """Create a simple RunInput with matching anchor fingerprints.
    
    Anchors have activations that allow comparison. Source and target have
    identical structure but different activation values to exercise alignment.
    """
    if anchor_ids is None:
        anchor_ids = ["invariant:time_001", "invariant:space_001", "invariant:time_002"]

    source_fps = []
    target_fps = []
    
    for i, anchor_id in enumerate(anchor_ids):
        # Source: activations with offset based on anchor index
        src_layers = {}
        tgt_layers = {}
        for layer in range(layer_count):
            base_val = (i + 1) * 0.1 + layer * 0.01
            # Source: dims 0, 1, 2 activated
            src_layers[layer] = [(0, base_val), (1, base_val + 0.1), (2, base_val + 0.2)]
            # Target: same dims with slight perturbation
            tgt_layers[layer] = [(0, base_val + 0.05), (1, base_val + 0.12), (2, base_val + 0.18)]
        
        source_fps.append(_make_fingerprint(anchor_id, f"Text for {anchor_id}", src_layers))
        target_fps.append(_make_fingerprint(anchor_id, f"Text for {anchor_id}", tgt_layers))

    source = _make_model_fingerprints(source_model_id, layer_count, source_fps)
    target = _make_model_fingerprints(target_model_id, layer_count, target_fps)
    
    return RunInput(id=run_id, source=source, target=target)


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_anchor_invariance_error_is_exception(self):
        """AnchorInvarianceError is a proper Exception subclass."""
        assert issubclass(AnchorInvarianceError, Exception)

    def test_no_runs_error_inherits_anchor_invariance_error(self):
        """NoRunsError inherits from AnchorInvarianceError."""
        assert issubclass(NoRunsError, AnchorInvarianceError)

    def test_no_runs_error_message(self):
        """NoRunsError has correct default message."""
        err = NoRunsError()
        assert "No run inputs were provided" in str(err)

    def test_no_anchors_error_inherits_anchor_invariance_error(self):
        """NoAnchorsError inherits from AnchorInvarianceError."""
        assert issubclass(NoAnchorsError, AnchorInvarianceError)

    def test_no_anchors_error_message_includes_prefix(self):
        """NoAnchorsError message includes the prefix."""
        err = NoAnchorsError("invariant:")
        assert "invariant:" in str(err)
        assert err.prefix == "invariant:"

    def test_no_anchors_error_stores_prefix(self):
        """NoAnchorsError stores prefix as attribute."""
        err = NoAnchorsError("custom_prefix:")
        assert err.prefix == "custom_prefix:"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestRunInput:
    """Tests for RunInput dataclass."""

    def test_run_input_fields(self):
        """RunInput has correct fields."""
        source = _make_model_fingerprints("src", 4, [])
        target = _make_model_fingerprints("tgt", 4, [])
        run = RunInput(id="run_1", source=source, target=target)
        
        assert run.id == "run_1"
        assert run.source.model_id == "src"
        assert run.target.model_id == "tgt"


class TestRunModels:
    """Tests for RunModels dataclass."""

    def test_run_models_fields(self):
        """RunModels has correct fields."""
        models = RunModels(model_a="model_a", model_b="model_b")
        assert models.model_a == "model_a"
        assert models.model_b == "model_b"


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_run_result_fields(self):
        """RunResult has all required fields."""
        models = RunModels(model_a="a", model_b="b")
        result = RunResult(
            id="run_1",
            models=models,
            probe_space=ProbeSpace.prelogits_hidden,
            align_mode=MetaphorConvergenceAnalyzer.AlignMode.LAYER,
            anchor_means={"anchor_1": 0.95},
        )
        
        assert result.id == "run_1"
        assert result.models.model_a == "a"
        assert result.probe_space == ProbeSpace.prelogits_hidden
        assert result.anchor_means["anchor_1"] == 0.95


class TestAnchorScore:
    """Tests for AnchorScore dataclass."""

    def test_anchor_score_fields(self):
        """AnchorScore has all required fields."""
        score = AnchorScore(
            anchor_id="invariant:time_001",
            prompt="Time is...",
            category="invariant",
            family="time",
            mean_cosine=0.95,
            std_cosine=0.02,
            min_cosine=0.92,
            max_cosine=0.98,
            stability_score=0.93,
            run_count=5,
        )
        
        assert score.anchor_id == "invariant:time_001"
        assert score.family == "time"
        b = get_default_backend()
        eps = division_epsilon(b, b.array([0.93]))
        assert abs(score.stability_score - 0.93) <= eps

    def test_anchor_score_stability_formula(self):
        """Stability score is mean - std."""
        mean = 0.9
        std = 0.1
        score = AnchorScore(
            anchor_id="test",
            prompt="",
            category="",
            family=None,
            mean_cosine=mean,
            std_cosine=std,
            min_cosine=0.8,
            max_cosine=1.0,
            stability_score=mean - std,
            run_count=1,
        )
        b = get_default_backend()
        eps = division_epsilon(b, b.array([mean - std]))
        assert abs(score.stability_score - (mean - std)) <= eps


class TestTopAnchor:
    """Tests for TopAnchor dataclass."""

    def test_top_anchor_fields(self):
        """TopAnchor has correct fields."""
        top = TopAnchor(anchor_id="top_1", mean_cosine=0.99, stability_score=0.97)
        assert top.anchor_id == "top_1"
        assert top.mean_cosine == 0.99
        assert top.stability_score == 0.97


class TestSummary:
    """Tests for Summary dataclass."""

    def test_summary_fields(self):
        """Summary has correct fields."""
        summary = Summary(
            anchor_count=10,
            run_count=3,
            overall_mean_cosine=0.85,
            top_anchors=[TopAnchor("a", 0.9, 0.88)],
        )
        assert summary.anchor_count == 10
        assert summary.run_count == 3
        assert len(summary.top_anchors) == 1


class TestReport:
    """Tests for Report dataclass."""

    def test_report_fields(self):
        """Report has all required fields."""
        summary = Summary(anchor_count=5, run_count=2, overall_mean_cosine=0.9, top_anchors=[])
        report = Report(
            align_mode=MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED,
            anchor_prefix="invariant:",
            holdout_prefixes=["exclude:"],
            runs=[],
            anchors=[],
            summary=summary,
        )
        assert report.align_mode == MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED
        assert report.anchor_prefix == "invariant:"
        assert report.holdout_prefixes == ["exclude:"]


class TestAnchorVectorIndex:
    """Tests for AnchorVectorIndex dataclass."""

    def test_anchor_vector_index_fields(self):
        """AnchorVectorIndex has correct fields."""
        index = AnchorVectorIndex(
            vectors={"anchor_1": {0: {0: 0.5, 1: 0.3}}},
            prompts={"anchor_1": "Test prompt"},
            categories={"anchor_1": "invariant"},
            families={"anchor_1": "time"},
        )
        assert "anchor_1" in index.vectors
        assert index.prompts["anchor_1"] == "Test prompt"
        assert index.families["anchor_1"] == "time"


class TestLayerAlignment:
    """Tests for LayerAlignment dataclass."""

    def test_layer_alignment_fields(self):
        """LayerAlignment has correct fields."""
        pairs = [
            MetaphorConvergenceAnalyzer.AlignmentPair(
                index=0, source_layer=0, target_layer=0, normalized_depth=0.0
            ),
        ]
        alignment = LayerAlignment(aligned_pairs=pairs, aligned_indices=[0])
        assert len(alignment.aligned_pairs) == 1
        assert alignment.aligned_indices == [0]


# =============================================================================
# AnchorInvarianceAnalyzer.analyze() Tests
# =============================================================================


class TestAnalyzeErrorCases:
    """Tests for analyze() error conditions."""

    def test_analyze_raises_no_runs_error_on_empty_list(self):
        """analyze() raises NoRunsError when runs list is empty."""
        with pytest.raises(NoRunsError):
            AnchorInvarianceAnalyzer.analyze(runs=[])

    def test_analyze_raises_no_anchors_error_when_no_matching_anchors(self):
        """analyze() raises NoAnchorsError when no anchors match prefix."""
        # Create fingerprints with non-matching prefix
        fp = _make_fingerprint("other:test", "Test", {0: [(0, 0.5)]})
        source = _make_model_fingerprints("src", 4, [fp])
        target = _make_model_fingerprints("tgt", 4, [fp])
        run = RunInput(id="run_1", source=source, target=target)
        
        with pytest.raises(NoAnchorsError) as exc_info:
            AnchorInvarianceAnalyzer.analyze([run], anchor_prefix="invariant:")
        
        assert exc_info.value.prefix == "invariant:"


class TestAnalyzeBasicFunctionality:
    """Tests for analyze() basic functionality."""

    def test_analyze_single_run_returns_report(self):
        """analyze() returns a Report with single run."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze([run])
        
        assert isinstance(report, Report)
        assert len(report.runs) == 1
        assert report.runs[0].id == "run_1"

    def test_analyze_multiple_runs(self):
        """analyze() handles multiple runs correctly."""
        run1 = _make_simple_run_input("run_1")
        run2 = _make_simple_run_input("run_2", source_model_id="src2", target_model_id="tgt2")
        
        report = AnchorInvarianceAnalyzer.analyze([run1, run2])
        
        assert len(report.runs) == 2
        assert report.summary.run_count == 2

    def test_analyze_preserves_anchor_prefix(self):
        """analyze() preserves anchor_prefix in report."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze([run], anchor_prefix="invariant:")
        
        assert report.anchor_prefix == "invariant:"

    def test_analyze_preserves_holdout_prefixes(self):
        """analyze() preserves holdout_prefixes in report."""
        run = _make_simple_run_input("run_1")
        holdouts = ["exclude:", "skip:"]
        
        report = AnchorInvarianceAnalyzer.analyze([run], holdout_prefixes=holdouts)
        
        assert report.holdout_prefixes == holdouts

    def test_analyze_default_holdout_prefixes_empty(self):
        """analyze() uses empty holdout_prefixes by default."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze([run])
        
        assert report.holdout_prefixes == []


class TestAnalyzeAlignModes:
    """Tests for analyze() layer alignment modes."""

    def test_analyze_normalized_mode(self):
        """analyze() works with NORMALIZED align mode."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze(
            [run],
            align_mode=MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED,
        )
        
        assert report.align_mode == MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED

    def test_analyze_layer_mode(self):
        """analyze() works with LAYER align mode."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze(
            [run],
            align_mode=MetaphorConvergenceAnalyzer.AlignMode.LAYER,
        )
        
        assert report.align_mode == MetaphorConvergenceAnalyzer.AlignMode.LAYER


class TestAnalyzeAnchorFiltering:
    """Tests for analyze() anchor filtering."""

    def test_analyze_filters_by_anchor_prefix(self):
        """analyze() only includes anchors matching prefix."""
        # Mix of invariant: and other: prefixes
        fp1 = _make_fingerprint("invariant:time_001", "Time", {0: [(0, 0.5)]})
        fp2 = _make_fingerprint("other:test", "Other", {0: [(0, 0.3)]})
        source = _make_model_fingerprints("src", 4, [fp1, fp2])
        target = _make_model_fingerprints("tgt", 4, [fp1, fp2])
        run = RunInput(id="run_1", source=source, target=target)
        
        report = AnchorInvarianceAnalyzer.analyze([run], anchor_prefix="invariant:")
        
        # Only the invariant: anchor should be in results
        anchor_ids = [a.anchor_id for a in report.anchors]
        assert anchor_ids == ["invariant:time_001"]
        for anchor in report.anchors:
            assert anchor.anchor_id.startswith("invariant:")

    def test_analyze_with_family_allowlist(self):
        """analyze() filters by family allowlist."""
        run = _make_simple_run_input(
            "run_1",
            anchor_ids=["invariant:time_001", "invariant:space_001", "invariant:time_002"],
        )
        
        # Only allow "time" family
        report = AnchorInvarianceAnalyzer.analyze(
            [run],
            anchor_family_allowlist={"time"},
        )
        
        # Only time family anchors should be included
        for anchor in report.anchors:
            if anchor.family:
                assert anchor.family == "time"


class TestAnalyzeAnchorScores:
    """Tests for analyze() anchor score computation."""

    def test_analyze_computes_anchor_scores(self):
        """analyze() computes scores for each anchor."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze([run])
        
        assert len(report.anchors) == len(run.source.fingerprints)
        for anchor in report.anchors:
            assert isinstance(anchor, AnchorScore)
            assert anchor.run_count == 1

    def test_analyze_anchor_scores_sorted_by_stability(self):
        """analyze() sorts anchors by stability score descending."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze([run])
        
        expected_order = sorted(
            report.anchors, key=lambda a: (-a.stability_score, a.anchor_id)
        )
        assert [a.anchor_id for a in report.anchors] == [
            a.anchor_id for a in expected_order
        ]

    def test_analyze_summary_contains_top_anchors(self):
        """analyze() summary contains top 5 anchors."""
        # Create more than 5 anchors
        anchor_ids = [f"invariant:test_{i:03d}" for i in range(8)]
        run = _make_simple_run_input("run_1", anchor_ids=anchor_ids)
        
        report = AnchorInvarianceAnalyzer.analyze([run])
        
        expected_top = [
            TopAnchor(
                anchor_id=a.anchor_id,
                mean_cosine=a.mean_cosine,
                stability_score=a.stability_score,
            )
            for a in report.anchors[: len(report.summary.top_anchors)]
        ]
        assert report.summary.top_anchors == expected_top


class TestAnalyzeSummary:
    """Tests for analyze() summary computation."""

    def test_analyze_summary_anchor_count(self):
        """analyze() summary has correct anchor count."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze([run])
        
        assert report.summary.anchor_count == len(report.anchors)

    def test_analyze_summary_run_count(self):
        """analyze() summary has correct run count."""
        run1 = _make_simple_run_input("run_1")
        run2 = _make_simple_run_input("run_2")
        
        report = AnchorInvarianceAnalyzer.analyze([run1, run2])
        
        assert report.summary.run_count == 2

    def test_analyze_summary_overall_mean_cosine(self):
        """analyze() summary computes overall mean cosine."""
        run = _make_simple_run_input("run_1")
        report = AnchorInvarianceAnalyzer.analyze([run])
        
        # Overall mean should be average of individual anchor means
        if report.anchors:
            expected_mean = sum(a.mean_cosine for a in report.anchors) / len(report.anchors)
            assert report.summary.overall_mean_cosine == pytest.approx(
                expected_mean, abs=math.ulp(expected_mean)
            )


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestExtractFamily:
    """Tests for _extract_family helper."""

    def test_extract_family_basic(self):
        """_extract_family extracts family from invariant: prefix."""
        family = AnchorInvarianceAnalyzer._extract_family("invariant:time_001", "invariant:")
        assert family == "time"

    def test_extract_family_with_underscore_in_id(self):
        """_extract_family handles multiple underscores."""
        family = AnchorInvarianceAnalyzer._extract_family("invariant:space_time_001", "invariant:")
        assert family == "space"

    def test_extract_family_no_underscore(self):
        """_extract_family returns None when no underscore in ID."""
        family = AnchorInvarianceAnalyzer._extract_family("invariant:nounderscore", "invariant:")
        assert family is None

    def test_extract_family_wrong_prefix(self):
        """_extract_family returns None for non-invariant prefix."""
        family = AnchorInvarianceAnalyzer._extract_family("invariant:time_001", "other:")
        assert family is None

    def test_extract_family_non_invariant_prefix_returns_none(self):
        """_extract_family returns None for non-'invariant:' prefix argument."""
        family = AnchorInvarianceAnalyzer._extract_family("custom:time_001", "custom:")
        assert family is None


class TestCollectLayers:
    """Tests for _collect_layers helper."""

    def test_collect_layers_basic(self):
        """_collect_layers collects unique layers."""
        vectors = {
            "anchor_1": {0: {0: 0.5}, 2: {1: 0.3}},
            "anchor_2": {1: {0: 0.4}, 2: {1: 0.2}},
        }
        layers = AnchorInvarianceAnalyzer._collect_layers(vectors)
        
        assert sorted(layers) == [0, 1, 2]

    def test_collect_layers_empty(self):
        """_collect_layers returns empty for empty vectors."""
        layers = AnchorInvarianceAnalyzer._collect_layers({})
        assert layers == []

    def test_collect_layers_sorted(self):
        """_collect_layers returns sorted layers."""
        vectors = {"a": {5: {}, 1: {}, 3: {}}}
        layers = AnchorInvarianceAnalyzer._collect_layers(vectors)
        assert layers == [1, 3, 5]


class TestNormalizeLayerIndex:
    """Tests for _normalize_layer_index helper."""

    def test_normalize_regular_layer(self):
        """_normalize_layer_index returns layer unchanged for regular indices."""
        result = AnchorInvarianceAnalyzer._normalize_layer_index(5, 10)
        assert result == 5

    def test_normalize_output_layer_marker(self):
        """_normalize_layer_index converts output_layer_marker to layer_count."""
        from modelcypher.core.domain.geometry.manifold_stitcher import output_layer_marker
        
        result = AnchorInvarianceAnalyzer._normalize_layer_index(output_layer_marker, 10)
        assert result == 10


class TestScaledIndex:
    """Tests for _scaled_index helper."""

    def test_scaled_index_zero_position(self):
        """_scaled_index returns 0 for position 0."""
        result = AnchorInvarianceAnalyzer._scaled_index(0, 5, 10)
        assert result == 0

    def test_scaled_index_last_position(self):
        """_scaled_index returns last index for last position."""
        result = AnchorInvarianceAnalyzer._scaled_index(4, 5, 10)
        assert result == 9

    def test_scaled_index_middle_position(self):
        """_scaled_index scales middle positions correctly."""
        result = AnchorInvarianceAnalyzer._scaled_index(2, 5, 10)
        # position 2 of 5 -> fraction 2/4 = 0.5 -> scaled to 0.5 * 9 = 4.5 -> round to 4
        assert result == 4

    def test_scaled_index_zero_total_count(self):
        """_scaled_index returns 0 for zero total count."""
        result = AnchorInvarianceAnalyzer._scaled_index(0, 5, 0)
        assert result == 0

    def test_scaled_index_single_aligned(self):
        """_scaled_index returns 0 for single aligned count."""
        result = AnchorInvarianceAnalyzer._scaled_index(0, 1, 10)
        assert result == 0


class TestApplyAlignment:
    """Tests for _apply_alignment helper."""

    def test_apply_alignment_basic(self):
        """_apply_alignment maps dimensions correctly."""
        vector = {0: 1.0, 1: 2.0, 2: 3.0}
        mapping = [
            AlignedDimension(source_dim=0, target_dim=10, weight=1.0),
            AlignedDimension(source_dim=1, target_dim=11, weight=0.5),
        ]
        
        result = AnchorInvarianceAnalyzer._apply_alignment(vector, mapping)
        
        b = get_default_backend()
        eps = division_epsilon(b, b.array([1.0]))
        assert abs(result[10] - 1.0) <= eps
        assert abs(result[11] - 1.0) <= eps  # 2.0 * 0.5 = 1.0

    def test_apply_alignment_empty_vector(self):
        """_apply_alignment returns empty dict for empty vector."""
        mapping = [AlignedDimension(source_dim=0, target_dim=10, weight=1.0)]
        result = AnchorInvarianceAnalyzer._apply_alignment({}, mapping)
        assert result == {}

    def test_apply_alignment_empty_mapping(self):
        """_apply_alignment returns empty dict for empty mapping."""
        vector = {0: 1.0}
        result = AnchorInvarianceAnalyzer._apply_alignment(vector, [])
        assert result == {}

    def test_apply_alignment_accumulates_weights(self):
        """_apply_alignment accumulates weights for same target dim."""
        vector = {0: 1.0, 1: 2.0}
        mapping = [
            AlignedDimension(source_dim=0, target_dim=10, weight=1.0),
            AlignedDimension(source_dim=1, target_dim=10, weight=1.0),  # same target
        ]
        
        result = AnchorInvarianceAnalyzer._apply_alignment(vector, mapping)
        
        b = get_default_backend()
        eps = division_epsilon(b, b.array([3.0]))
        assert abs(result[10] - 3.0) <= eps  # 1.0 + 2.0

    def test_apply_alignment_missing_source_dim(self):
        """_apply_alignment ignores mappings for missing source dims."""
        vector = {0: 1.0}
        mapping = [
            AlignedDimension(source_dim=0, target_dim=10, weight=1.0),
            AlignedDimension(source_dim=5, target_dim=11, weight=1.0),  # not in vector
        ]
        
        result = AnchorInvarianceAnalyzer._apply_alignment(vector, mapping)
        
        assert 10 in result
        assert 11 not in result


class TestCosineSparse:
    """Tests for _cosine_sparse helper."""

    def test_cosine_sparse_identical_vectors(self, backend):
        """_cosine_sparse returns 1.0 for identical vectors."""
        a = {0: 1.0, 1: 0.0, 2: 0.0}
        b = {0: 1.0, 1: 0.0, 2: 0.0}
        
        result = AnchorInvarianceAnalyzer._cosine_sparse(a, b)

        assert result is not None
        eps = division_epsilon(backend, backend.array([1.0]))
        assert abs(result - 1.0) <= eps

    def test_cosine_sparse_orthogonal_vectors(self, backend):
        """_cosine_sparse returns 0.0 for orthogonal vectors."""
        a = {0: 1.0}
        b = {1: 1.0}
        
        result = AnchorInvarianceAnalyzer._cosine_sparse(a, b)
        
        expected = geodesic_cosine_sparse(a, b, backend)
        eps = division_epsilon(backend, backend.array([expected]))
        assert abs(result - expected) <= eps

    def test_cosine_sparse_empty_vectors(self, backend):
        """_cosine_sparse raises ValueError for empty vectors."""
        with pytest.raises(ValueError, match="empty sparse vectors"):
            AnchorInvarianceAnalyzer._cosine_sparse({}, {})


class TestBuildAnchorVectors:
    """Tests for _build_anchor_vectors helper."""

    def test_build_anchor_vectors_filters_by_prefix(self):
        """_build_anchor_vectors filters by anchor prefix."""
        fp1 = _make_fingerprint("invariant:time_001", "Time", {0: [(0, 0.5)]})
        fp2 = _make_fingerprint("other:test", "Other", {0: [(0, 0.3)]})
        fingerprints = _make_model_fingerprints("model", 4, [fp1, fp2])
        
        result = AnchorInvarianceAnalyzer._build_anchor_vectors(
            fingerprints, "invariant:", None
        )
        
        assert "invariant:time_001" in result.vectors
        assert "other:test" not in result.vectors

    def test_build_anchor_vectors_stores_prompts(self):
        """_build_anchor_vectors stores prompts."""
        fp = _make_fingerprint("invariant:time_001", "Time is relative", {0: [(0, 0.5)]})
        fingerprints = _make_model_fingerprints("model", 4, [fp])
        
        result = AnchorInvarianceAnalyzer._build_anchor_vectors(
            fingerprints, "invariant:", None
        )
        
        assert result.prompts.get("invariant:time_001") == "Time is relative"

    def test_build_anchor_vectors_with_family_allowlist(self):
        """_build_anchor_vectors filters by family allowlist."""
        fp1 = _make_fingerprint("invariant:time_001", "Time", {0: [(0, 0.5)]})
        fp2 = _make_fingerprint("invariant:space_001", "Space", {0: [(0, 0.3)]})
        fingerprints = _make_model_fingerprints("model", 4, [fp1, fp2])
        
        result = AnchorInvarianceAnalyzer._build_anchor_vectors(
            fingerprints, "invariant:", {"time"}
        )
        
        assert "invariant:time_001" in result.vectors
        assert "invariant:space_001" not in result.vectors


class TestBuildLayerAlignment:
    """Tests for _build_layer_alignment helper."""

    def test_build_layer_alignment_normalized_mode(self):
        """_build_layer_alignment works in NORMALIZED mode."""
        source_vectors = {"a": {0: {0: 1.0}, 1: {0: 1.0}}}
        target_vectors = {"a": {0: {0: 1.0}, 1: {0: 1.0}}}
        
        result = AnchorInvarianceAnalyzer._build_layer_alignment(
            source_vectors, target_vectors, 4, 4,
            MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED,
        )
        
        assert isinstance(result, LayerAlignment)
        expected_pairs = [
            MetaphorConvergenceAnalyzer.AlignmentPair(
                index=0, source_layer=0, target_layer=0, normalized_depth=0.0
            ),
            MetaphorConvergenceAnalyzer.AlignmentPair(
                index=1, source_layer=1, target_layer=1, normalized_depth=0.25
            ),
        ]
        assert result.aligned_pairs == expected_pairs

    def test_build_layer_alignment_layer_mode(self):
        """_build_layer_alignment works in LAYER mode."""
        source_vectors = {"a": {0: {0: 1.0}, 2: {0: 1.0}}}
        target_vectors = {"a": {0: {0: 1.0}, 2: {0: 1.0}}}
        
        result = AnchorInvarianceAnalyzer._build_layer_alignment(
            source_vectors, target_vectors, 4, 4,
            MetaphorConvergenceAnalyzer.AlignMode.LAYER,
        )
        
        # LAYER mode should use exact matching
        for pair in result.aligned_pairs:
            assert pair.source_layer == pair.target_layer

    def test_build_layer_alignment_mismatched_layer_counts(self):
        """_build_layer_alignment handles different layer counts."""
        source_vectors = {"a": {0: {0: 1.0}, 1: {0: 1.0}, 2: {0: 1.0}}}
        target_vectors = {"a": {0: {0: 1.0}, 1: {0: 1.0}}}
        
        result = AnchorInvarianceAnalyzer._build_layer_alignment(
            source_vectors, target_vectors, 3, 2,
            MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED,
        )
        
        assert isinstance(result, LayerAlignment)


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnalyzeIntegration:
    """Integration tests for the full analyze() workflow."""

    def test_analyze_end_to_end(self):
        """Full end-to-end test with realistic data."""
        # Create two runs with different model pairs
        run1 = _make_simple_run_input(
            "run_1",
            source_model_id="llama-7b",
            target_model_id="mistral-7b",
        )
        run2 = _make_simple_run_input(
            "run_2",
            source_model_id="qwen-7b",
            target_model_id="mistral-7b",
        )
        
        report = AnchorInvarianceAnalyzer.analyze([run1, run2])
        
        # Verify report structure
        assert isinstance(report, Report)
        assert len(report.runs) == 2
        assert report.summary.run_count == 2
        
        # Verify each run has models
        assert report.runs[0].models.model_a == "llama-7b"
        assert report.runs[1].models.model_a == "qwen-7b"

    def test_analyze_produces_stable_results(self):
        """analyze() produces deterministic results."""
        run = _make_simple_run_input("run_1")
        
        report1 = AnchorInvarianceAnalyzer.analyze([run])
        report2 = AnchorInvarianceAnalyzer.analyze([run])
        
        # Results should be identical
        assert len(report1.anchors) == len(report2.anchors)
        for a1, a2 in zip(report1.anchors, report2.anchors):
            assert a1.anchor_id == a2.anchor_id
            b = get_default_backend()
            eps = division_epsilon(b, b.array([a2.mean_cosine]))
            assert abs(a1.mean_cosine - a2.mean_cosine) <= eps
