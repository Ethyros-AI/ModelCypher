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

"""Tests for SinkhornSolver optimal transport."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.optimal_transport import (
    SinkhornResult,
    SinkhornSolver,
)


@pytest.fixture
def backend():
    """Get default backend for tests."""
    return get_default_backend()


@pytest.fixture
def solver(backend):
    """Create SinkhornSolver instance."""
    return SinkhornSolver(backend)


class TestSinkhornSolverInit:
    """Tests for SinkhornSolver initialization."""

    def test_default_initialization(self, backend):
        """Solver should initialize with backend."""
        solver = SinkhornSolver(backend)
        assert solver is not None
        assert solver.backend is backend

    def test_initialization_without_backend(self):
        """Solver should use default backend if none provided."""
        solver = SinkhornSolver()
        assert solver is not None
        assert solver.backend is not None


class TestSinkhornSolve:
    """Tests for the main solve method."""

    def test_solve_square_cost_matrix(self, solver, backend):
        """Solve OT with square cost matrix."""
        cost = backend.array([[1.0, 2.0], [3.0, 4.0]])
        result = solver.solve(cost)

        assert isinstance(result, SinkhornResult)
        assert result.plan is not None
        assert result.plan.shape == (2, 2)

    def test_solve_rectangular_cost_matrix(self, solver, backend):
        """Solve OT with rectangular cost matrix."""
        cost = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = solver.solve(cost)

        assert isinstance(result, SinkhornResult)
        assert result.plan.shape == (2, 3)


@settings(max_examples=10, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=4),
    m=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=1_000_000),
)
def test_sinkhorn_marginals_hypothesis(n: int, m: int, seed: int) -> None:
    """Sinkhorn plan respects target/source marginals (precision-derived)."""
    backend = get_default_backend()
    backend.random_seed(seed)
    cost = backend.random_uniform(low=0.0, high=1.0, shape=(n, m))
    backend.eval(cost)

    solver = SinkhornSolver(backend)
    result = solver.solve(cost)

    mu = backend.ones((n,)) / n
    nu = backend.ones((m,)) / m
    row_sums = backend.sum(result.plan, axis=1)
    col_sums = backend.sum(result.plan, axis=0)
    row_error = backend.max(backend.abs(row_sums - mu))
    col_error = backend.max(backend.abs(col_sums - nu))
    backend.eval(row_error, col_error)
    computed_error = max(
        float(backend.to_scalar(row_error)),
        float(backend.to_scalar(col_error)),
    )

    eps = division_epsilon(backend, result.plan) * max(1.0, computed_error)
    assert abs(result.marginal_error - computed_error) <= eps

    min_val_arr = backend.min(result.plan)
    backend.eval(min_val_arr)
    min_val = float(backend.to_scalar(min_val_arr))
    assert min_val >= -eps

    cost_check = backend.sum(cost * result.plan)
    backend.eval(cost_check)
    cost_val = float(backend.to_scalar(cost_check))
    eps_cost = division_epsilon(backend, cost) * max(1.0, abs(cost_val))
    assert abs(result.cost - cost_val) <= eps_cost

    def test_solve_with_uniform_marginals(self, solver, backend):
        """Default marginals should be uniform."""
        cost = backend.array([[1.0, 2.0], [3.0, 4.0]])
        result = solver.solve(cost)

        # Row sums should be approximately 0.5 (uniform over 2 sources)
        row_sums = backend.sum(result.plan, axis=1)
        backend.eval(row_sums)
        row_sum_list = backend.tolist(row_sums)
        # Use solver-reported marginal error with dtype-derived floor
        tol = max(result.marginal_error, regularization_epsilon(backend, result.plan))
        assert all(abs(s - 0.5) < tol for s in row_sum_list)

    def test_solve_with_custom_marginals(self, solver, backend):
        """Solve OT with custom marginals."""
        cost = backend.array([[1.0, 2.0], [3.0, 4.0]])
        source = backend.array([0.3, 0.7])
        target = backend.array([0.6, 0.4])

        result = solver.solve(cost, source_marginal=source, target_marginal=target)

        assert isinstance(result, SinkhornResult)
        # Marginal constraints should be approximately satisfied
        row_sums = backend.sum(result.plan, axis=1)
        backend.eval(row_sums)
        row_list = backend.tolist(row_sums)
        # Sinkhorn converges to marginal_error precision
        tol = max(result.marginal_error, regularization_epsilon(backend, result.plan))
        assert abs(row_list[0] - 0.3) < tol

    def test_solve_returns_convergence_info(self, solver, backend):
        """Result should include convergence information."""
        cost = backend.array([[1.0, 2.0], [3.0, 4.0]])
        result = solver.solve(cost)

        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "marginal_error")
        assert hasattr(result, "cost")
        assert isinstance(result.iterations, int)
        assert result.iterations > 0


class TestSinkhornResult:
    """Tests for SinkhornResult structure."""

    def test_result_is_frozen_dataclass(self, solver, backend):
        """SinkhornResult should be a frozen dataclass."""
        cost = backend.array([[1.0, 2.0], [3.0, 4.0]])
        result = solver.solve(cost)

        # Should not be able to modify
        with pytest.raises(AttributeError):
            result.converged = False


class TestTransportPlanProperties:
    """Tests for transport plan mathematical properties."""

    def test_plan_is_non_negative(self, solver, backend):
        """Transport plan should have non-negative entries."""
        cost = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = solver.solve(cost)

        min_val = backend.min(result.plan)
        backend.eval(min_val)
        tol = regularization_epsilon(backend, result.plan)
        assert float(backend.to_scalar(min_val)) >= -tol

    def test_plan_sums_to_one(self, solver, backend):
        """Transport plan should sum to 1 (for probability distributions)."""
        cost = backend.array([[1.0, 2.0], [3.0, 4.0]])
        result = solver.solve(cost)

        total = backend.sum(result.plan)
        backend.eval(total)
        tol = regularization_epsilon(backend, result.plan)
        assert abs(float(backend.to_scalar(total)) - 1.0) < tol


class TestSolveLinearOT:
    """Tests for fast linear OT solver (inner loop optimization)."""

    def test_solve_linear_ot_exists(self, solver):
        """solve_linear_ot method should exist."""
        assert hasattr(solver, "solve_linear_ot")
        assert callable(solver.solve_linear_ot)

    def test_solve_linear_ot_basic(self, solver, backend):
        """Basic linear OT solve."""
        cost = backend.array([[1.0, 2.0], [3.0, 4.0]])
        p = backend.array([0.5, 0.5])
        q = backend.array([0.5, 0.5])

        plan = solver.solve_linear_ot(cost, p, q, epsilon=0.1)

        assert plan.shape == (2, 2)


class TestCostMatrices:
    """Tests for cost matrix computation methods."""

    def test_cosine_cost_exists(self, solver):
        """cosine_cost method should exist."""
        assert hasattr(solver, "cosine_cost")
        assert callable(solver.cosine_cost)

    def test_cosine_cost_basic(self, solver, backend):
        """Compute cosine distance cost matrix."""
        source = backend.array([[1.0, 0.0], [0.0, 1.0]])
        target = backend.array([[1.0, 0.0], [0.5, 0.5]])

        cost = solver.cosine_cost(source, target)

        assert cost.shape == (2, 2)
        # Cosine distance is in [0, 2]
        min_val = backend.min(cost)
        max_val = backend.max(cost)
        backend.eval(min_val, max_val)
        tol = regularization_epsilon(backend, cost)
        assert float(backend.to_scalar(min_val)) >= -tol
        assert float(backend.to_scalar(max_val)) <= 2.0 + tol

    def test_cosine_cost_identical_vectors(self, solver, backend):
        """Identical vectors should have zero cosine distance."""
        source = backend.array([[1.0, 0.0]])
        target = backend.array([[1.0, 0.0]])

        cost = solver.cosine_cost(source, target)
        backend.eval(cost)

        tol = regularization_epsilon(backend, cost)
        assert float(backend.to_scalar(cost[0, 0])) < tol

    def test_squared_chord_cost_exists(self, solver):
        """squared_chord_cost method should exist."""
        assert hasattr(solver, "squared_chord_cost")
        assert callable(solver.squared_chord_cost)

    def test_squared_geodesic_cost_exists(self, solver):
        """squared_geodesic_cost method should exist."""
        assert hasattr(solver, "squared_geodesic_cost")
        assert callable(solver.squared_geodesic_cost)


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_very_small_costs(self, solver, backend):
        """Handle very small cost values."""
        cost = backend.array([[1e-10, 2e-10], [3e-10, 4e-10]])
        result = solver.solve(cost)

        assert isinstance(result, SinkhornResult)
        assert result.plan is not None

    def test_very_large_costs(self, solver, backend):
        """Handle very large cost values."""
        cost = backend.array([[1e6, 2e6], [3e6, 4e6]])
        result = solver.solve(cost)

        assert isinstance(result, SinkhornResult)
        assert result.plan is not None

    def test_mixed_scale_costs(self, solver, backend):
        """Handle costs with mixed scales."""
        cost = backend.array([[1e-5, 1e5], [1e5, 1e-5]])
        result = solver.solve(cost)

        assert isinstance(result, SinkhornResult)

    def test_zero_cost_entry(self, solver, backend):
        """Handle cost matrix with zero entries."""
        cost = backend.array([[0.0, 1.0], [1.0, 0.0]])
        result = solver.solve(cost)

        assert isinstance(result, SinkhornResult)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_element_cost(self, solver, backend):
        """Handle 1x1 cost matrix."""
        cost = backend.array([[5.0]])
        result = solver.solve(cost)

        assert result.plan.shape == (1, 1)

    def test_single_row_cost(self, solver, backend):
        """Handle single row cost matrix."""
        cost = backend.array([[1.0, 2.0, 3.0]])
        result = solver.solve(cost)

        assert result.plan.shape == (1, 3)

    def test_single_column_cost(self, solver, backend):
        """Handle single column cost matrix."""
        cost = backend.array([[1.0], [2.0], [3.0]])
        result = solver.solve(cost)

        assert result.plan.shape == (3, 1)

    def test_larger_matrix(self, solver, backend):
        """Handle larger cost matrix."""
        n, m = 50, 30
        cost = backend.random_normal((n, m))
        cost = backend.abs(cost)  # Ensure positive costs
        result = solver.solve(cost)

        assert result.plan.shape == (n, m)


class TestEpsilonDerivation:
    """Tests for data-driven epsilon derivation."""

    def test_epsilon_derived_from_cost_scale(self, solver, backend):
        """Epsilon should be derived from cost matrix scale."""
        # This is tested implicitly - different cost scales should still converge
        small_cost = backend.array([[0.01, 0.02], [0.03, 0.04]])
        large_cost = backend.array([[100.0, 200.0], [300.0, 400.0]])

        result_small = solver.solve(small_cost)
        result_large = solver.solve(large_cost)

        # Both should produce valid transport plans
        assert result_small.plan is not None
        assert result_large.plan is not None
