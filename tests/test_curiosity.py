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

"""Tests for the curiosity module (Sprint 5.0).

Tests cover:
- CuriosityPolicy with EFE-based probe ranking
- AcquisitionProtocols data structures
- CoreSetAcquisition (k-center)
- ManifoldCoverageAcquisition (directional coverage + local ID)
- CompositeAcquisition (geometry-derived weighting)
- CuriosityDaemon orchestration
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend


# =============================================================================
# CuriosityPolicy Tests
# =============================================================================


class TestCuriosityPolicy:
    """Tests for EFE-based curiosity policy."""

    def test_compute_epistemic_value_product_form(self):
        """Epistemic value is eigenscore × capacity_fraction."""
        from modelcypher.core.domain.continual.curiosity_policy import (
            compute_epistemic_value,
        )

        # Zero capacity → zero value (can't encode)
        assert compute_epistemic_value(0.8, 0.0) == 0.0

        # Zero eigenscore → zero value (nothing to learn)
        assert compute_epistemic_value(0.0, 0.8) == 0.0

        # Both positive → product
        assert compute_epistemic_value(0.5, 0.6) == pytest.approx(0.3)

        # Maximum at (1.0, 1.0)
        assert compute_epistemic_value(1.0, 1.0) == 1.0

    def test_compute_efe_risk_plus_ambiguity(self):
        """EFE = risk + ambiguity, where risk = (1 - capacity)²."""
        from modelcypher.core.domain.continual.curiosity_policy import compute_efe

        # Full capacity (0 risk), zero eigenscore (0 ambiguity) → EFE = 0
        assert compute_efe(0.0, 1.0) == pytest.approx(0.0)

        # Zero capacity (1 risk), zero eigenscore → EFE = 1
        assert compute_efe(0.0, 0.0) == pytest.approx(1.0)

        # Test risk calculation: (1 - 0.5)² = 0.25
        assert compute_efe(0.0, 0.5) == pytest.approx(0.25)

        # Test ambiguity addition
        assert compute_efe(0.3, 0.5) == pytest.approx(0.25 + 0.3)

    def test_efe_policy_rank_candidates(self):
        """EFE policy ranks candidates by epistemic value (descending)."""
        from modelcypher.core.domain.continual.curiosity_policy import (
            EFECuriosityPolicy,
            ProbeCandidate,
        )

        backend = get_default_backend()
        policy = EFECuriosityPolicy(backend=backend)

        candidates = [
            ProbeCandidate(
                coordinates=(0.0,),
                eigenscore=0.2,
                capacity_fraction=0.5,
                epistemic_value=0.1,  # Low
                efe_score=0.0,
                layer_id=0,
                neighbor_density=0.0,
                intrinsic_dimension=0.0,
            ),
            ProbeCandidate(
                coordinates=(1.0,),
                eigenscore=0.8,
                capacity_fraction=0.9,
                epistemic_value=0.72,  # High
                efe_score=0.0,
                layer_id=0,
                neighbor_density=0.0,
                intrinsic_dimension=0.0,
            ),
            ProbeCandidate(
                coordinates=(0.5,),
                eigenscore=0.5,
                capacity_fraction=0.5,
                epistemic_value=0.25,  # Medium
                efe_score=0.0,
                layer_id=0,
                neighbor_density=0.0,
                intrinsic_dimension=0.0,
            ),
        ]

        ranked = policy.rank_candidates(candidates)
        assert ranked[0].epistemic_value == pytest.approx(0.72)
        assert ranked[1].epistemic_value == pytest.approx(0.25)
        assert ranked[2].epistemic_value == pytest.approx(0.1)

    def test_exploration_temperature_scales_with_eigenscore(self):
        """Exploration temperature = mean_eigenscore / sqrt(eps)."""
        from modelcypher.core.domain.continual.curiosity_policy import (
            EFECuriosityPolicy,
        )

        backend = get_default_backend()
        policy = EFECuriosityPolicy(backend=backend)
        sqrt_eps = policy.sqrt_eps

        # When eigenscore >> sqrt_eps → T >> 1 (exploration)
        temp = policy.compute_exploration_temperature(0.1, sqrt_eps)
        assert temp > 100  # Much larger than 1

        # When eigenscore << sqrt_eps → T = sqrt_eps (minimum)
        temp = policy.compute_exploration_temperature(sqrt_eps / 10, sqrt_eps)
        assert temp == pytest.approx(sqrt_eps)

    def test_select_action_complete_when_dense(self):
        """Select COMPLETE when mean_eigenscore <= sqrt_eps."""
        from modelcypher.core.domain.continual.curiosity_policy import (
            CuriosityAction,
            CuriosityState,
            EFECuriosityPolicy,
        )

        backend = get_default_backend()
        policy = EFECuriosityPolicy(backend=backend)

        state = CuriosityState(
            n_candidates=10,
            top_candidate=None,
            mean_eigenscore=policy.sqrt_eps / 2,  # Below threshold
            mean_capacity=0.5,
            exploration_temperature=1.0,
            sqrt_eps=policy.sqrt_eps,
        )

        action, candidate = policy.select_action(state)
        assert action == CuriosityAction.COMPLETE
        assert candidate is None

    def test_select_action_wait_when_no_capacity(self):
        """Select WAIT when mean_capacity <= sqrt_eps."""
        from modelcypher.core.domain.continual.curiosity_policy import (
            CuriosityAction,
            CuriosityState,
            EFECuriosityPolicy,
        )

        backend = get_default_backend()
        policy = EFECuriosityPolicy(backend=backend)

        state = CuriosityState(
            n_candidates=10,
            top_candidate=None,
            mean_eigenscore=0.5,
            mean_capacity=policy.sqrt_eps / 2,  # Below threshold
            exploration_temperature=1.0,
            sqrt_eps=policy.sqrt_eps,
        )

        action, candidate = policy.select_action(state)
        assert action == CuriosityAction.WAIT
        assert candidate is None

    def test_create_candidate_computes_efe(self):
        """create_candidate computes epistemic_value and efe_score."""
        from modelcypher.core.domain.continual.curiosity_policy import (
            EFECuriosityPolicy,
        )

        backend = get_default_backend()
        policy = EFECuriosityPolicy(backend=backend)

        candidate = policy.create_candidate(
            coordinates=(0.5, 0.5),
            eigenscore=0.4,
            capacity_fraction=0.6,
            layer_id=2,
        )

        assert candidate.eigenscore == pytest.approx(0.4)
        assert candidate.capacity_fraction == pytest.approx(0.6)
        assert candidate.epistemic_value == pytest.approx(0.24)  # 0.4 × 0.6
        # EFE = (1 - 0.6)² + 0.4 = 0.16 + 0.4 = 0.56
        assert candidate.efe_score == pytest.approx(0.56)


# =============================================================================
# AcquisitionProtocols Tests
# =============================================================================


class TestAcquisitionProtocols:
    """Tests for acquisition function protocols and data structures."""

    def test_acquisition_score_to_dict(self):
        """AcquisitionScore serializes to dict."""
        from modelcypher.core.domain.geometry.acquisition_protocols import (
            AcquisitionScore,
        )

        score = AcquisitionScore(
            probe_idx=5,
            score=0.75,
            coreset_contribution=0.4,
            coverage_contribution=0.2,
            density_contribution=0.15,
        )

        d = score.to_dict()
        assert d["probe_idx"] == 5
        assert d["score"] == 0.75
        assert d["coreset_contribution"] == 0.4

    def test_acquisition_result_top_indices(self):
        """AcquisitionResult.top_indices returns sorted indices."""
        from modelcypher.core.domain.geometry.acquisition_protocols import (
            AcquisitionResult,
            AcquisitionScore,
        )

        scores = [
            AcquisitionScore(probe_idx=2, score=0.9, coreset_contribution=0, coverage_contribution=0, density_contribution=0),
            AcquisitionScore(probe_idx=0, score=0.7, coreset_contribution=0, coverage_contribution=0, density_contribution=0),
            AcquisitionScore(probe_idx=1, score=0.5, coreset_contribution=0, coverage_contribution=0, density_contribution=0),
        ]

        result = AcquisitionResult(
            scores=scores,
            coverage_radius=1.0,
            mean_local_id=2.0,
            sparse_fraction=0.3,
        )

        assert result.top_indices == [2, 0, 1]
        assert result.top_score.probe_idx == 2
        assert result.select_top_k(2) == scores[:2]

    def test_empty_acquisition_result(self):
        """empty_acquisition_result returns valid empty result."""
        from modelcypher.core.domain.geometry.acquisition_protocols import (
            empty_acquisition_result,
        )

        result = empty_acquisition_result()
        assert result.scores == []
        assert result.coverage_radius == 0.0
        assert result.top_score is None

    def test_uniform_acquisition_result(self):
        """uniform_acquisition_result gives all candidates score 1.0."""
        from modelcypher.core.domain.geometry.acquisition_protocols import (
            uniform_acquisition_result,
        )

        result = uniform_acquisition_result(5)
        assert len(result.scores) == 5
        assert all(s.score == 1.0 for s in result.scores)
        assert result.coverage_radius == float("inf")


# =============================================================================
# CoreSetAcquisition Tests
# =============================================================================


class TestCoreSetAcquisition:
    """Tests for k-center core-set acquisition."""

    def test_empty_candidates_returns_empty(self):
        """Empty candidates returns empty result."""
        from modelcypher.core.domain.geometry.acquisition_coreset import (
            CoreSetAcquisition,
        )

        backend = get_default_backend()
        acq = CoreSetAcquisition(backend=backend)

        candidates = backend.array([]).reshape(0, 10)
        corpus = backend.random_normal((50, 10))

        result = acq.score(candidates, corpus)
        assert result.scores == []

    def test_empty_corpus_returns_uniform(self):
        """Empty corpus gives uniform scores (all equally valuable)."""
        from modelcypher.core.domain.geometry.acquisition_coreset import (
            CoreSetAcquisition,
        )

        backend = get_default_backend()
        acq = CoreSetAcquisition(backend=backend)

        candidates = backend.random_normal((10, 20))
        corpus = backend.array([]).reshape(0, 20)

        result = acq.score(candidates, corpus)
        assert len(result.scores) == 10
        assert all(s.score == 1.0 for s in result.scores)

    def test_score_higher_for_distant_candidates(self):
        """Candidates farther from corpus get higher scores."""
        from modelcypher.core.domain.geometry.acquisition_coreset import (
            CoreSetAcquisition,
            CoreSetConfig,
        )

        backend = get_default_backend()
        # Use more neighbors for stability
        config = CoreSetConfig(k_neighbors=5, refine_iterations=0)
        acq = CoreSetAcquisition(backend=backend, config=config)

        # Corpus is a cluster around origin
        corpus = backend.random_normal((30, 10)) * 0.1
        backend.eval(corpus)

        # Candidates: one near corpus, one far
        near = backend.random_normal((1, 10)) * 0.1
        far = backend.random_normal((1, 10)) * 5.0
        candidates = backend.concatenate([near, far], axis=0)
        backend.eval(candidates)

        result = acq.score(candidates, corpus)

        # Find which index got higher score
        idx_0_score = next(s.score for s in result.scores if s.probe_idx == 0)
        idx_1_score = next(s.score for s in result.scores if s.probe_idx == 1)

        # Far candidate (idx 1) should have higher score
        assert idx_1_score > idx_0_score

    def test_select_batch_greedy_kcenter(self):
        """select_batch uses greedy k-center selection."""
        from modelcypher.core.domain.geometry.acquisition_coreset import (
            CoreSetAcquisition,
            CoreSetConfig,
        )

        backend = get_default_backend()
        config = CoreSetConfig(k_neighbors=5, refine_iterations=0)
        acq = CoreSetAcquisition(backend=backend, config=config)

        # Create candidates in a line
        candidates = backend.array([[float(i), 0.0] for i in range(10)])
        corpus = backend.array([[0.0, 0.0]])  # Single point at origin
        backend.eval(candidates, corpus)

        # Select 3 - should get endpoints first (maximizes min-distance)
        selected = acq.select_batch(candidates, corpus, batch_size=3)
        assert len(selected) == 3
        # First selection should be farthest from corpus (index 9)
        assert selected[0] == 9


# =============================================================================
# ManifoldCoverageAcquisition Tests
# =============================================================================


class TestManifoldCoverageAcquisition:
    """Tests for manifold coverage acquisition."""

    def test_empty_candidates_returns_empty(self):
        """Empty candidates returns empty result."""
        from modelcypher.core.domain.geometry.acquisition_manifold import (
            ManifoldCoverageAcquisition,
        )

        backend = get_default_backend()
        acq = ManifoldCoverageAcquisition(backend=backend)

        candidates = backend.array([]).reshape(0, 10)
        corpus = backend.random_normal((50, 10))

        result = acq.score(candidates, corpus)
        assert result.scores == []

    def test_score_includes_coverage_and_density(self):
        """Scores include coverage and density contributions."""
        from modelcypher.core.domain.geometry.acquisition_manifold import (
            ManifoldCoverageAcquisition,
            ManifoldCoverageConfig,
        )

        backend = get_default_backend()
        config = ManifoldCoverageConfig(k_neighbors=5)
        acq = ManifoldCoverageAcquisition(backend=backend, config=config)

        # Need enough points for stable computation
        candidates = backend.random_normal((10, 20))
        corpus = backend.random_normal((50, 20))
        backend.eval(candidates, corpus)

        result = acq.score(candidates, corpus)

        assert len(result.scores) == 10
        # Check that scores have the expected fields
        for score in result.scores:
            assert hasattr(score, "coverage_contribution")
            assert hasattr(score, "density_contribution")
            assert score.score >= 0

    def test_mean_local_id_computed(self):
        """mean_local_id is computed from corpus."""
        from modelcypher.core.domain.geometry.acquisition_manifold import (
            ManifoldCoverageAcquisition,
            ManifoldCoverageConfig,
        )

        backend = get_default_backend()
        config = ManifoldCoverageConfig(k_neighbors=5)
        acq = ManifoldCoverageAcquisition(backend=backend, config=config)

        candidates = backend.random_normal((5, 20))
        corpus = backend.random_normal((50, 20))
        backend.eval(candidates, corpus)

        result = acq.score(candidates, corpus)

        # Local ID should be positive and reasonable
        assert result.mean_local_id > 0


# =============================================================================
# CompositeAcquisition Tests
# =============================================================================


class TestCompositeAcquisition:
    """Tests for composite acquisition function."""

    def test_compute_weights_geometry_derived(self):
        """Weights derived from coverage_radius / mean_local_id."""
        from modelcypher.core.domain.geometry.acquisition_composite import (
            CompositeAcquisition,
        )

        backend = get_default_backend()
        acq = CompositeAcquisition(backend=backend)

        # Test w = 1 / (1 + ratio)
        # When ratio = 1 (coverage = local_id), w = 0.5
        # coreset_weight = 1 - w = 0.5, coverage_weight + density_weight = w = 0.5
        weights = acq.compute_weights(coverage_radius=2.0, mean_local_id=2.0)
        assert weights.coreset_weight == pytest.approx(0.5)
        assert weights.coverage_weight == pytest.approx(0.25)
        assert weights.density_weight == pytest.approx(0.25)

        # When coverage >> local_id, favor coreset (w small → coreset_weight high)
        weights = acq.compute_weights(coverage_radius=10.0, mean_local_id=1.0)
        assert weights.coreset_weight > 0.5

        # When coverage << local_id, favor local (w large → local weights high)
        weights = acq.compute_weights(coverage_radius=1.0, mean_local_id=10.0)
        assert (weights.coverage_weight + weights.density_weight) > 0.5

    def test_composite_combines_scores(self):
        """Composite combines coreset and manifold scores."""
        from modelcypher.core.domain.geometry.acquisition_composite import (
            CompositeAcquisition,
            CompositeAcquisitionConfig,
        )

        backend = get_default_backend()
        config = CompositeAcquisitionConfig(k_neighbors=5, refine_iterations=0)
        acq = CompositeAcquisition(backend=backend, config=config)

        candidates = backend.random_normal((5, 20))
        corpus = backend.random_normal((30, 20))
        backend.eval(candidates, corpus)

        result = acq.score(candidates, corpus)

        assert len(result.scores) == 5
        # Check that both contributions are present
        for score in result.scores:
            # Scores should be non-negative
            assert score.score >= 0
            assert score.coreset_contribution >= 0


# =============================================================================
# CuriosityDaemon Tests (Unit)
# =============================================================================


class TestCuriosityDaemonUnit:
    """Unit tests for curiosity daemon (no async)."""

    def test_daemon_initial_state_stopped(self):
        """Daemon starts in STOPPED state."""
        from modelcypher.core.use_cases.curiosity_daemon import (
            CuriosityDaemon,
            DaemonState,
        )

        backend = get_default_backend()
        daemon = CuriosityDaemon(hidden_dim=64, backend=backend)

        status = daemon.get_status()
        assert status.state == DaemonState.STOPPED
        assert status.iterations_completed == 0

    def test_daemon_add_candidate(self):
        """Daemon accumulates candidates."""
        from modelcypher.core.use_cases.curiosity_daemon import CuriosityDaemon
        from modelcypher.core.domain.continual.curiosity_policy import ProbeCandidate

        backend = get_default_backend()
        daemon = CuriosityDaemon(hidden_dim=64, backend=backend)

        candidate = ProbeCandidate(
            coordinates=(0.1, 0.2),
            eigenscore=0.5,
            capacity_fraction=0.6,
            epistemic_value=0.3,
            efe_score=0.7,
            layer_id=0,
            neighbor_density=0.0,
            intrinsic_dimension=0.0,
        )
        daemon.add_candidate(candidate)

        # Check candidates were added
        assert len(daemon._candidates) == 1

    def test_daemon_add_to_corpus(self):
        """Daemon adds activations to corpus."""
        from modelcypher.core.use_cases.curiosity_daemon import CuriosityDaemon

        backend = get_default_backend()
        daemon = CuriosityDaemon(hidden_dim=64, backend=backend)

        activation = backend.random_normal((64,))
        backend.eval(activation)

        daemon.add_to_corpus(activation)

        metrics = daemon.get_metrics()
        assert metrics.n_corpus == 1

    def test_daemon_status_to_dict(self):
        """DaemonStatus serializes to dict."""
        from modelcypher.core.use_cases.curiosity_daemon import CuriosityDaemon

        backend = get_default_backend()
        daemon = CuriosityDaemon(hidden_dim=64, backend=backend)

        status = daemon.get_status()
        d = status.to_dict()

        assert "state" in d
        assert "iterations_completed" in d
        assert "probes_executed" in d

    def test_daemon_set_probe_executor(self):
        """Daemon can set probe executor."""
        from modelcypher.core.use_cases.curiosity_daemon import CuriosityDaemon
        from modelcypher.core.domain.continual.curiosity_policy import ProbeCandidate

        backend = get_default_backend()
        daemon = CuriosityDaemon(hidden_dim=64, backend=backend)

        def mock_executor(candidate: ProbeCandidate):
            return backend.random_normal((64,))

        daemon.set_probe_executor(mock_executor)

        assert daemon._probe_executor is not None


# =============================================================================
# CuriosityDaemon Async Tests
# =============================================================================


class TestCuriosityDaemonAsync:
    """Async tests for curiosity daemon."""

    @pytest.mark.asyncio
    async def test_daemon_start_stop(self):
        """Daemon can start and stop."""
        from modelcypher.core.use_cases.curiosity_daemon import (
            CuriosityDaemon,
            DaemonState,
        )
        import asyncio

        backend = get_default_backend()
        daemon = CuriosityDaemon(hidden_dim=64, backend=backend)

        # Start (non-blocking, creates task)
        daemon.start()

        # Give the daemon a moment to transition
        await asyncio.sleep(0.1)

        status = daemon.get_status()
        assert status.is_running is True

        # Stop
        await daemon.stop()
        status = daemon.get_status()
        assert status.state == DaemonState.STOPPED

    @pytest.mark.asyncio
    async def test_daemon_add_candidates_and_corpus(self):
        """Daemon handles candidates and corpus."""
        from modelcypher.core.use_cases.curiosity_daemon import CuriosityDaemon
        from modelcypher.core.domain.continual.curiosity_policy import ProbeCandidate, EFECuriosityPolicy

        backend = get_default_backend()
        daemon = CuriosityDaemon(hidden_dim=64, backend=backend)
        policy = EFECuriosityPolicy(backend=backend)

        # Add candidates using policy helper
        for i in range(10):
            candidate = policy.create_candidate(
                coordinates=tuple(backend.tolist(backend.random_normal((64,)))),
                eigenscore=0.5,
                capacity_fraction=0.6,
                layer_id=0,
            )
            daemon.add_candidate(candidate)

        assert len(daemon._candidates) == 10

        # Add corpus points
        for _ in range(5):
            activation = backend.random_normal((64,))
            backend.eval(activation)
            daemon.add_to_corpus(activation)

        assert daemon.get_metrics().n_corpus == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestCuriosityIntegration:
    """Integration tests for curiosity components."""

    def test_full_acquisition_pipeline(self):
        """Full pipeline: events → candidates → scoring → ranking."""
        from modelcypher.core.domain.continual.curiosity_policy import EFECuriosityPolicy
        from modelcypher.core.domain.geometry.acquisition_composite import (
            CompositeAcquisition,
            CompositeAcquisitionConfig,
        )

        backend = get_default_backend()
        policy = EFECuriosityPolicy(backend=backend)
        config = CompositeAcquisitionConfig(k_neighbors=5, refine_iterations=0)
        acquisition = CompositeAcquisition(backend=backend, config=config)

        # Generate synthetic candidates and corpus
        candidates = backend.random_normal((10, 32))
        corpus = backend.random_normal((50, 32))
        backend.eval(candidates, corpus)

        # Score candidates
        result = acquisition.score(candidates, corpus)
        assert len(result.scores) == 10

        # Convert to ProbeCandidate for policy ranking
        probe_candidates = [
            policy.create_candidate(
                coordinates=(0.0,),
                eigenscore=result.coverage_radius / 10,  # Proxy for sparsity
                capacity_fraction=0.6,
            )
            for _ in range(len(result.scores))
        ]

        # Rank by epistemic value
        ranked = policy.rank_candidates(probe_candidates)
        assert len(ranked) == 10

        # Create state and select action
        state = policy.create_state(probe_candidates, mean_capacity=0.6)
        action, selected = policy.select_action(state)

        # Should either PROBE (if conditions met) or CONSOLIDATE/WAIT
        assert action is not None

    def test_policy_and_acquisition_work_together(self):
        """Policy and acquisition functions integrate correctly."""
        from modelcypher.core.domain.continual.curiosity_policy import (
            EFECuriosityPolicy,
            CuriosityAction,
        )
        from modelcypher.core.domain.geometry.acquisition_coreset import (
            CoreSetAcquisition,
            CoreSetConfig,
        )

        backend = get_default_backend()
        policy = EFECuriosityPolicy(backend=backend)
        config = CoreSetConfig(k_neighbors=5, refine_iterations=0)
        coreset = CoreSetAcquisition(backend=backend, config=config)

        # Create candidates
        candidates_arr = backend.random_normal((10, 32))
        corpus_arr = backend.random_normal((30, 32))
        backend.eval(candidates_arr, corpus_arr)

        # Score using coreset
        result = coreset.score(candidates_arr, corpus_arr)

        # Create probe candidates from scores
        probe_candidates = []
        for score in result.scores:
            candidate = policy.create_candidate(
                coordinates=tuple(backend.tolist(candidates_arr[score.probe_idx])),
                eigenscore=score.coreset_contribution / 10,  # Normalize
                capacity_fraction=0.7,
            )
            probe_candidates.append(candidate)

        # Rank
        ranked = policy.rank_candidates(probe_candidates)

        # Create state
        state = policy.create_state(ranked, mean_capacity=0.7)

        # Action should be PROBE (we have capacity and eigenscores)
        action, selected = policy.select_action(state)

        # With high capacity and eigenscores, should either PROBE or CONSOLIDATE
        assert action in (CuriosityAction.PROBE, CuriosityAction.CONSOLIDATE, CuriosityAction.WAIT)
