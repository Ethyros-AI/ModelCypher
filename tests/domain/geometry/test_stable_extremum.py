# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.stable_extremum import (
    StableExtremumResult,
    find_stable_inflection,
    find_stable_minimum,
)


class TestFindStableMinimum:
    def test_clear_minimum(self) -> None:
        """One value much lower than the rest → stable minimum."""
        values = [10.0, 10.0, 10.0, 2.0, 10.0, 10.0, 10.0, 10.0]
        result = find_stable_minimum(values, seed=42)

        assert result.index == 3
        assert result.value == 2.0
        assert result.is_stable is True
        assert result.frequency > 0.5
        assert result.n_values == 8

    def test_ambiguous_minimum(self) -> None:
        """Multiple near-minimum values → may be unstable."""
        values = [10.0, 2.0, 2.1, 2.0, 10.0, 10.0, 10.0, 10.0]
        result = find_stable_minimum(values, seed=42)

        # Minimum could be at index 1 or 3 (both 2.0)
        assert result.index in (1, 3)
        # Frequency likely < 1.0 because bootstrap can pick either
        assert result.n_values == 8

    def test_flat_values_unstable(self) -> None:
        """All identical values → minimum is at index 0, unstable.

        Pairs bootstrap: all values tied, so min(sampled_indices, key=v)
        returns first element of sampled_indices (Python min on ties keeps
        first occurrence). Random first elements → no index dominates.
        """
        values = [5.0] * 10
        result = find_stable_minimum(values, seed=42)

        # Index 0 (first occurrence of minimum in original)
        assert result.index == 0
        # With all ties, bootstrap selects random indices → unstable
        assert result.is_stable is False

    def test_single_value(self) -> None:
        """Single value → trivially stable."""
        result = find_stable_minimum([7.0], seed=42)
        assert result.index == 0
        assert result.value == 7.0
        assert result.is_stable is True
        assert result.frequency == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            find_stable_minimum([])

    def test_ci_range(self) -> None:
        """CI range should span only observed bootstrap argmins."""
        values = [10.0, 10.0, 1.0, 10.0, 10.0]
        result = find_stable_minimum(values, seed=42)

        lo, hi = result.ci_range
        assert lo >= 0
        assert hi < len(values)

    def test_descending_minimum_at_end(self) -> None:
        """Descending values → minimum at last index."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = find_stable_minimum(values, seed=42)
        assert result.index == 4
        assert result.value == 2.0


class TestFindStableInflection:
    def test_v_shaped_trajectory(self) -> None:
        """Decreasing then increasing → inflection at the bottom."""
        values = [10.0, 8.0, 6.0, 4.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        result = find_stable_inflection(values, seed=42)

        # Inflection should be near index 4 (transition from decrease to increase)
        assert 3 <= result.index <= 5
        assert result.n_values == 9

    def test_monotone_no_inflection(self) -> None:
        """Strictly increasing → no sign change, inflection not found."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        result = find_stable_inflection(values, seed=42)

        # No inflection found → is_stable should be False
        # (internal _find_inflection returns -1, gets clamped to 0)
        assert result.is_stable is False

    def test_sharp_inflection(self) -> None:
        """Sharp V → inflection is stable."""
        values = [10.0, 5.0, 1.0, 0.5, 1.0, 5.0, 10.0]
        result = find_stable_inflection(values, seed=42)

        assert 2 <= result.index <= 4

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError, match="Need >= 3"):
            find_stable_inflection([1.0, 2.0])

    def test_exactly_three_values(self) -> None:
        """Minimum viable: 3 values with one inflection."""
        values = [5.0, 1.0, 5.0]
        result = find_stable_inflection(values, seed=42)
        assert result.index == 1
