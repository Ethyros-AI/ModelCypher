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

from modelcypher.core.domain.geometry.spectral_changepoint import (
    ChangePointResult,
    detect_spectral_changepoint,
)


class TestDetectSpectralChangepoint:
    def test_clear_two_segment_break(self) -> None:
        """Values with a clear step → changepoint at the step."""
        # 5 low values, then 5 high values
        values = [1.0, 1.1, 1.2, 1.3, 1.4, 5.0, 5.1, 5.2, 5.3, 5.4]
        result = detect_spectral_changepoint(values, seed=42)

        assert result.k == 4  # Break between index 4 and 5
        assert result.strength > 5.0  # Step >> background noise
        assert result.rss_reduction > 0.5  # Two lines much better than one
        assert result.n_values == 10

    def test_linear_no_break(self) -> None:
        """Perfectly linear values → changepoint exists but low strength."""
        values = [float(i) for i in range(10)]
        result = detect_spectral_changepoint(values, seed=42)

        # Changepoint found but strength should be low (no real break)
        assert result.rss_reduction < 0.1

    def test_bootstrap_stability_strong_break(self) -> None:
        """Strong break with non-degenerate values → stable changepoint.

        Non-constant segments so bootstrap resampling doesn't just shuffle
        identical values — each segment has a clear linear trend.
        """
        values = [1.0, 1.2, 1.4, 1.6, 1.8, 8.0, 8.2, 8.4, 8.6, 8.8]
        result = detect_spectral_changepoint(values, n_bootstrap=500, seed=42)

        assert result.k == 4  # Break at step boundary
        assert result.is_stable is True
        assert result.frequency > 0.5

    def test_minimum_input_size(self) -> None:
        """Exactly 5 values — minimum viable."""
        values = [1.0, 1.0, 1.0, 5.0, 5.0]
        result = detect_spectral_changepoint(values, seed=42)
        assert isinstance(result.k, int)
        assert result.n_values == 5

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError, match="Need >= 5"):
            detect_spectral_changepoint([1.0, 2.0, 3.0, 4.0])

    def test_use_log(self) -> None:
        """Log transform for singular value spectra."""
        # Exponential decay with a break
        import math
        values = [math.exp(-0.1 * i) for i in range(5)] + [
            math.exp(-2.0 * i) for i in range(5, 10)
        ]
        result = detect_spectral_changepoint(values, use_log=True, seed=42)
        assert isinstance(result.k, int)
        assert result.rss_reduction > 0.0

    def test_ci_bounds(self) -> None:
        """CI bounds are within valid index range."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 8.0, 8.0, 8.0, 8.0]
        result = detect_spectral_changepoint(values, seed=42)
        assert result.ci_lower >= 0
        assert result.ci_upper < result.n_values
        assert result.ci_lower <= result.ci_upper

    def test_gradual_transition(self) -> None:
        """Gradual slope change — changepoint near the inflection."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 5.8, 5.9, 5.95, 5.99]
        result = detect_spectral_changepoint(values, seed=42)
        # Changepoint should be somewhere in the middle where slope changes
        assert 2 <= result.k <= 7
