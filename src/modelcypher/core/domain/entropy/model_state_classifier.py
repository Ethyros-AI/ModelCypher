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

"""Model state measurements for entropy-based analysis.

Provides calibrated baseline statistics and raw signal bundles. No
classification or thresholding logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibratedBaseline:
    """Calibrated entropy baseline from empirical measurement.

    Percentiles are stored as raw distribution statistics (not thresholds).
    """

    mean: float
    """Mean entropy from calibration (measured, not assumed)."""

    std_dev: float
    """Standard deviation from calibration (measured, not assumed)."""

    percentile_25: float
    """25th percentile of measured entropy distribution."""

    percentile_75: float
    """75th percentile of measured entropy distribution."""

    percentile_95: float
    """95th percentile of measured entropy distribution."""

    vocab_size: int
    """Model vocabulary size."""

    model_id: str
    """Model identifier for this baseline."""

    sample_count: int
    """Number of samples used in calibration."""

    def z_score(self, entropy: float) -> float:
        """Compute z-score (standard deviations from mean)."""
        if self.std_dev < 1e-10:
            return 0.0 if abs(entropy - self.mean) < 1e-10 else float("inf")
        return (entropy - self.mean) / self.std_dev


@dataclass(frozen=True)
class ModelStateSignals:
    """Raw entropy and variance signals with baseline-relative z-score."""

    entropy: float
    variance: float
    z_score: float
    entropy_trend: float
    entropy_variance_correlation: float
