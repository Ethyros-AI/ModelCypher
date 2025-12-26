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

"""Tests for PathGeometry distance metrics.

Tests mathematical properties of path comparison algorithms:
- Levenshtein (edit distance): d(X,X)=0, non-negativity, triangle inequality
- Frechet distance: d(X,X)=0, captures worst-case deviation
- Dynamic Time Warping: d(X,X)=0, handles time warping
- Path signatures: translation invariance, similarity computation
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.path_geometry import (
    AlignmentOp,
    BackendPathGeometry,
    PathGeometry,
    PathNode,
    PathSignature,
    SimilarityWeights,
    get_path_geometry,
)


def _simple_embeddings():
    """Orthogonal embeddings for gates A, B, C."""
    return {
        "A": [1.0, 0.0],
        "B": [0.0, 1.0],
        "C": [1.0, 1.0],
        "D": [-1.0, 0.0],
    }


def _make_path(gate_ids: list[str], entropies: list[float] | None = None) -> PathSignature:
    """Helper to create a path from gate IDs."""
    if entropies is None:
        entropies = [0.1 * i for i in range(len(gate_ids))]
    nodes = [
        PathNode(gate_id=g, token_index=i, entropy=e)
        for i, (g, e) in enumerate(zip(gate_ids, entropies))
    ]
    return PathSignature(model_id="test", prompt_id="test", nodes=nodes)


class TestLevenshteinDistance:
    """Tests for Levenshtein-based path comparison."""

    def test_identical_paths_zero_distance(self) -> None:
        """Identical paths should have zero distance.

        Mathematical property: d(X, X) = 0.
        """
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.compare(path, path, gate_embeddings=_simple_embeddings())

        assert result.total_distance == pytest.approx(0.0, abs=1e-9)
        assert result.normalized_distance == pytest.approx(0.0, abs=1e-9)

    def test_different_paths_positive_distance(self) -> None:
        """Different paths should have positive distance."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "D", "C"])  # D is different from B

        result = PathGeometry.compare(path_a, path_b, gate_embeddings=_simple_embeddings())

        assert result.total_distance > 0, "Different paths should have positive distance"

    def test_distance_non_negative(self) -> None:
        """Distance should always be non-negative."""
        path_a = _make_path(["A", "B"])
        path_b = _make_path(["C", "D"])

        result = PathGeometry.compare(path_a, path_b, gate_embeddings=_simple_embeddings())

        assert result.total_distance >= 0
        assert result.normalized_distance >= 0

    def test_alignment_tracks_operations(self) -> None:
        """Alignment should track insert/delete/substitute operations."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "C"])  # Missing B

        result = PathGeometry.compare(path_a, path_b, gate_embeddings=_simple_embeddings())

        # Should have a delete operation for B
        ops = [step.op for step in result.alignment]
        assert AlignmentOp.delete in ops or AlignmentOp.insert in ops

    def test_substitution_cost_uses_embedding_similarity(self) -> None:
        """Substituting similar gates should cost less than dissimilar ones."""
        # A and D are opposite directions (cosine = -1)
        path_a = _make_path(["A"])
        path_d = _make_path(["D"])
        # B is orthogonal to A (cosine = 0)
        path_b = _make_path(["B"])

        cost_ad = PathGeometry.compare(path_a, path_d, gate_embeddings=_simple_embeddings())
        cost_ab = PathGeometry.compare(path_a, path_b, gate_embeddings=_simple_embeddings())

        # Substituting A->D (opposite) should cost more than A->B (orthogonal)
        # Because similarity(A,D) < similarity(A,B): -1 < 0
        assert cost_ad.total_distance >= cost_ab.total_distance


class TestFrechetDistance:
    """Tests for discrete Frechet distance."""

    def test_identical_paths_zero_distance(self) -> None:
        """Identical paths should have zero Frechet distance.

        Mathematical property: d(X, X) = 0.
        """
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.frechet_distance(path, path, gate_embeddings=_simple_embeddings())

        assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_optimal_coupling_starts_at_origin(self) -> None:
        """Optimal coupling should start at (0, 0)."""
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.frechet_distance(path, path, gate_embeddings=_simple_embeddings())

        assert result.optimal_coupling[0] == (0, 0)

    def test_empty_path_returns_inf(self) -> None:
        """Empty path should return infinite distance."""
        empty = PathSignature(model_id="m", prompt_id="p", nodes=[])
        path = _make_path(["A"])

        result = PathGeometry.frechet_distance(empty, path, gate_embeddings=_simple_embeddings())

        assert result.distance == float("inf")

    def test_different_lengths_handled(self) -> None:
        """Frechet distance should handle paths of different lengths."""
        short = _make_path(["A", "B"])
        long = _make_path(["A", "B", "C", "C", "C"])

        result = PathGeometry.frechet_distance(short, long, gate_embeddings=_simple_embeddings())

        assert result.distance >= 0
        assert len(result.optimal_coupling) > 0


class TestDTW:
    """Tests for Dynamic Time Warping."""

    def test_identical_paths_zero_cost(self) -> None:
        """Identical paths should have zero DTW cost.

        Mathematical property: d(X, X) = 0.
        """
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.dynamic_time_warping(path, path, gate_embeddings=_simple_embeddings())

        assert result.total_cost == pytest.approx(0.0, abs=1e-9)
        assert result.normalized_cost == pytest.approx(0.0, abs=1e-9)

    def test_warping_path_covers_all_points(self) -> None:
        """Warping path should cover all points in both sequences."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "A", "B", "C"])

        result = PathGeometry.dynamic_time_warping(
            path_a, path_b, gate_embeddings=_simple_embeddings()
        )

        # Warping path should start at (0, 0) and end at (n-1, m-1)
        assert result.warping_path[0] == (0, 0)
        assert result.warping_path[-1] == (len(path_a.nodes) - 1, len(path_b.nodes) - 1)

    def test_empty_path_returns_inf(self) -> None:
        """Empty path should return infinite cost."""
        empty = PathSignature(model_id="m", prompt_id="p", nodes=[])
        path = _make_path(["A"])

        result = PathGeometry.dynamic_time_warping(
            empty, path, gate_embeddings=_simple_embeddings()
        )

        assert result.total_cost == float("inf")

    def test_window_constraint_limits_alignment(self) -> None:
        """Window constraint should limit alignment to nearby indices."""
        path_a = _make_path(["A", "B", "C", "D", "A"])
        path_b = _make_path(["A", "B", "C", "D", "A"])

        result = PathGeometry.dynamic_time_warping(
            path_a, path_b, gate_embeddings=_simple_embeddings(), window_size=1
        )

        # With window_size=1, all pairs in warping path should be within 1 step
        for i, j in result.warping_path:
            assert abs(i - j) <= 1, f"Point ({i}, {j}) violates window constraint"

    def test_compression_ratio_bounded(self) -> None:
        """Compression ratio should be in reasonable range."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "B", "C"])

        result = PathGeometry.dynamic_time_warping(
            path_a, path_b, gate_embeddings=_simple_embeddings()
        )

        assert result.compression_ratio >= 0, "Compression ratio should be non-negative"


class TestPathSignatures:
    """Tests for path signature computation."""

    def test_identical_signatures_high_similarity(self) -> None:
        """Identical signatures should have similarity = 1."""
        path = _make_path(["A", "B", "C"])
        sig = PathGeometry.compute_signature(path, gate_embeddings=_simple_embeddings())

        similarity = PathGeometry.signature_similarity(sig, sig)

        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_single_node_path_zero_signature(self) -> None:
        """Single node path has no increments, so signature components are zero."""
        path = _make_path(["A"])
        sig = PathGeometry.compute_signature(path, gate_embeddings=_simple_embeddings())

        assert sig.signature_norm == pytest.approx(0.0, abs=1e-9)

    def test_signed_area_non_negative(self) -> None:
        """Signed area (magnitude) should be non-negative."""
        path = _make_path(["A", "B", "C"])
        sig = PathGeometry.compute_signature(path, gate_embeddings=_simple_embeddings())

        assert sig.signed_area >= 0

    def test_different_paths_different_signatures(self) -> None:
        """Different paths should produce different signatures."""
        path_abc = _make_path(["A", "B", "C"])
        path_cba = _make_path(["C", "B", "A"])

        sig_abc = PathGeometry.compute_signature(path_abc, gate_embeddings=_simple_embeddings())
        sig_cba = PathGeometry.compute_signature(path_cba, gate_embeddings=_simple_embeddings())

        similarity = PathGeometry.signature_similarity(sig_abc, sig_cba)

        # Reversed path should have different signature
        assert similarity < 1.0


class TestEntropyPathAnalysis:
    """Tests for entropy path analysis."""

    def test_empty_path_defaults(self) -> None:
        """Empty path should return safe defaults."""
        empty = PathSignature(model_id="m", prompt_id="p", nodes=[])
        analysis = PathGeometry.analyze_entropy_path(empty)

        assert analysis.total_entropy == 0.0
        assert analysis.mean_entropy == 0.0
        assert analysis.stability_score == 1.0

    def test_spike_detection(self) -> None:
        """High entropy values should be detected as spikes.

        Spike threshold is mean + 2*std_dev. For a single outlier in
        uniform data, the outlier inflates variance significantly.

        For [1,1,1,1,1,100]: mean=17.5, var=1360.4, std=36.88
        Threshold = 17.5 + 2*36.88 = 91.26
        100 > 91.26 ✓ (detected as spike)
        """
        # Create path with extreme spike that exceeds mean + 2*std
        nodes = [
            PathNode(gate_id="A", token_index=0, entropy=1.0),
            PathNode(gate_id="B", token_index=1, entropy=1.0),
            PathNode(gate_id="C", token_index=2, entropy=1.0),
            PathNode(gate_id="D", token_index=3, entropy=1.0),
            PathNode(gate_id="E", token_index=4, entropy=1.0),
            PathNode(gate_id="F", token_index=5, entropy=100.0),  # Extreme spike
        ]
        path = PathSignature(model_id="m", prompt_id="p", nodes=nodes)
        analysis = PathGeometry.analyze_entropy_path(path)

        assert analysis.spike_count >= 1
        assert 5 in analysis.spike_indices

    def test_max_entropy_tracking(self) -> None:
        """Maximum entropy and its index should be tracked."""
        nodes = [
            PathNode(gate_id="A", token_index=0, entropy=1.0),
            PathNode(gate_id="B", token_index=1, entropy=5.0),  # Max
            PathNode(gate_id="C", token_index=2, entropy=2.0),
        ]
        path = PathSignature(model_id="m", prompt_id="p", nodes=nodes)
        analysis = PathGeometry.analyze_entropy_path(path)

        assert analysis.max_entropy == 5.0
        assert analysis.max_entropy_index == 1


class TestLocalGeometry:
    """Tests for local geometry computation."""

    def test_short_path_empty_curvatures(self) -> None:
        """Paths with < 3 nodes have no curvatures."""
        path = _make_path(["A", "B"])
        geom = PathGeometry.compute_local_geometry(path, gate_embeddings=_simple_embeddings())

        assert geom.curvatures == []
        assert geom.mean_curvature == 0.0

    def test_curvatures_bounded(self) -> None:
        """Curvatures should be bounded angles in [0, π]."""
        path = _make_path(["A", "B", "C", "D"])
        geom = PathGeometry.compute_local_geometry(path, gate_embeddings=_simple_embeddings())

        for curv in geom.curvatures:
            assert 0 <= curv <= math.pi, f"Curvature {curv} out of bounds"


class TestComprehensiveCompare:
    """Tests for comprehensive path comparison."""

    def test_identical_paths_high_similarity(self) -> None:
        """Identical paths should have overall similarity near 1."""
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.comprehensive_compare(
            path, path, gate_embeddings=_simple_embeddings()
        )

        assert result.overall_similarity > 0.99

    def test_weights_must_sum_to_one(self) -> None:
        """SimilarityWeights should validate weights sum to 1."""
        with pytest.raises(ValueError):
            SimilarityWeights(
                levenshtein_weight=0.5,
                frechet_weight=0.5,
                dtw_weight=0.5,  # Sum = 1.5, invalid
                signature_weight=0.5,
            )

    def test_custom_weights_affect_score(self) -> None:
        """Custom weights should affect the overall similarity score."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "D", "C"])

        # Weight Levenshtein heavily
        lev_heavy = SimilarityWeights(
            levenshtein_weight=0.7,
            frechet_weight=0.1,
            dtw_weight=0.1,
            signature_weight=0.1,
        )
        # Weight signature heavily
        sig_heavy = SimilarityWeights(
            levenshtein_weight=0.1,
            frechet_weight=0.1,
            dtw_weight=0.1,
            signature_weight=0.7,
        )

        result_lev = PathGeometry.comprehensive_compare(
            path_a, path_b, gate_embeddings=_simple_embeddings(), similarity_weights=lev_heavy
        )
        result_sig = PathGeometry.comprehensive_compare(
            path_a, path_b, gate_embeddings=_simple_embeddings(), similarity_weights=sig_heavy
        )

        # Different weights should produce different scores
        # (unless paths are identical, which they're not)
        assert result_lev.overall_similarity != result_sig.overall_similarity


class TestBackendPathGeometry:
    """Tests for the GPU-accelerated BackendPathGeometry."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    @pytest.fixture
    def pg(self, backend):
        return BackendPathGeometry(backend)

    def test_compute_signature_identical_to_pure_python(self, pg) -> None:
        """Backend signature computation should match pure Python."""
        path = _make_path(["A", "B", "C"])
        emb = _simple_embeddings()

        pure_sig = PathGeometry.compute_signature(path, emb)
        backend_sig = pg.compute_signature(path, emb)

        # Compare level1
        for i in range(len(pure_sig.level1)):
            assert pure_sig.level1[i] == pytest.approx(backend_sig.level1[i], abs=1e-6)

        # Compare signed area and norm
        assert pure_sig.signed_area == pytest.approx(backend_sig.signed_area, abs=1e-6)
        assert pure_sig.signature_norm == pytest.approx(backend_sig.signature_norm, abs=1e-6)

    def test_signature_similarity_identical_to_pure_python(self, pg) -> None:
        """Backend signature similarity should match pure Python."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "D", "C"])
        emb = _simple_embeddings()

        sig_a = PathGeometry.compute_signature(path_a, emb)
        sig_b = PathGeometry.compute_signature(path_b, emb)

        pure_sim = PathGeometry.signature_similarity(sig_a, sig_b)
        backend_sim = pg.signature_similarity(sig_a, sig_b)

        assert pure_sim == pytest.approx(backend_sim, abs=1e-6)

    def test_analyze_entropy_path_identical_to_pure_python(self, pg) -> None:
        """Backend entropy analysis should match pure Python."""
        nodes = [
            PathNode(gate_id="A", token_index=0, entropy=1.0),
            PathNode(gate_id="B", token_index=1, entropy=5.0),
            PathNode(gate_id="C", token_index=2, entropy=2.0),
            PathNode(gate_id="D", token_index=3, entropy=3.0),
            PathNode(gate_id="E", token_index=4, entropy=1.5),
        ]
        path = PathSignature(model_id="m", prompt_id="p", nodes=nodes)

        pure_analysis = PathGeometry.analyze_entropy_path(path)
        backend_analysis = pg.analyze_entropy_path(path)

        assert pure_analysis.total_entropy == pytest.approx(
            backend_analysis.total_entropy, abs=1e-6
        )
        assert pure_analysis.mean_entropy == pytest.approx(
            backend_analysis.mean_entropy, abs=1e-6
        )
        assert pure_analysis.entropy_variance == pytest.approx(
            backend_analysis.entropy_variance, abs=1e-6
        )
        assert pure_analysis.max_entropy == backend_analysis.max_entropy
        assert pure_analysis.max_entropy_index == backend_analysis.max_entropy_index
        assert pure_analysis.mean_gradient == pytest.approx(
            backend_analysis.mean_gradient, abs=1e-6
        )

    def test_compute_local_geometry_identical_to_pure_python(self, pg) -> None:
        """Backend local geometry should match pure Python."""
        path = _make_path(["A", "B", "C", "D"])
        emb = _simple_embeddings()

        pure_geom = PathGeometry.compute_local_geometry(path, emb)
        backend_geom = pg.compute_local_geometry(path, emb)

        assert len(pure_geom.curvatures) == len(backend_geom.curvatures)
        for i in range(len(pure_geom.curvatures)):
            assert pure_geom.curvatures[i] == pytest.approx(
                backend_geom.curvatures[i], abs=1e-5
            )

        assert pure_geom.mean_curvature == pytest.approx(
            backend_geom.mean_curvature, abs=1e-5
        )
        assert pure_geom.max_curvature == pytest.approx(backend_geom.max_curvature, abs=1e-5)
        assert pure_geom.total_curvature == pytest.approx(
            backend_geom.total_curvature, abs=1e-5
        )

    def test_comprehensive_compare_identical_to_pure_python(self, pg) -> None:
        """Backend comprehensive comparison should match pure Python."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "D", "C"])
        emb = _simple_embeddings()

        pure_result = PathGeometry.comprehensive_compare(path_a, path_b, emb)
        backend_result = pg.comprehensive_compare(path_a, path_b, emb)

        # DP algorithms are identical (use pure Python)
        assert pure_result.levenshtein.total_distance == backend_result.levenshtein.total_distance
        assert pure_result.frechet.distance == backend_result.frechet.distance
        assert pure_result.dtw.total_cost == backend_result.dtw.total_cost

        # Signature similarity computed by Backend
        assert pure_result.signature_similarity == pytest.approx(
            backend_result.signature_similarity, abs=1e-6
        )
        assert pure_result.overall_similarity == pytest.approx(
            backend_result.overall_similarity, abs=1e-6
        )

    def test_empty_path_handling(self, pg) -> None:
        """Backend should handle empty paths correctly."""
        empty = PathSignature(model_id="m", prompt_id="p", nodes=[])
        emb = _simple_embeddings()

        analysis = pg.analyze_entropy_path(empty)
        assert analysis.total_entropy == 0.0
        assert analysis.stability_score == 1.0

        sig = pg.compute_signature(empty, emb)
        assert sig.signature_norm == 0.0

    def test_single_node_path(self, pg) -> None:
        """Backend should handle single node paths correctly."""
        path = _make_path(["A"])
        emb = _simple_embeddings()

        sig = pg.compute_signature(path, emb)
        assert sig.signature_norm == 0.0

        geom = pg.compute_local_geometry(path, emb)
        assert geom.curvatures == []


class TestGetPathGeometry:
    """Tests for the factory function."""

    def test_returns_class_without_backend(self) -> None:
        """Factory should return PathGeometry class without backend."""
        result = get_path_geometry()
        assert result is PathGeometry

    def test_returns_instance_with_backend(self) -> None:
        """Factory should return BackendPathGeometry instance with backend."""
        backend = get_default_backend()
        result = get_path_geometry(backend)
        assert isinstance(result, BackendPathGeometry)
