# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DEPRECATED: Use temporal_geometry instead.

This module name is retained for backwards compatibility.
The implementation has been moved to temporal_geometry.py with:
- Neutral terminology (no "Latent Chronologist Hypothesis")
- Removed interpretive thresholds
- Measurement-only outputs

Import from the canonical location:
    from modelcypher.core.domain.geometry.temporal_geometry import (
        TemporalGeometryAnalyzer,
        TemporalGeometryReport,
    )
"""

# Re-export from canonical location for backwards compatibility
from modelcypher.core.domain.geometry.temporal_geometry import (
    TemporalAxisOrthogonality,
    TemporalDirectionResult,
    TemporalGeometryAnalyzer,
    TemporalGeometryComponents,
    TemporalGeometryReport,
    TemporalGradientConsistency,
    extract_temporal_activations,
)

__all__ = [
    "TemporalAxisOrthogonality",
    "TemporalDirectionResult",
    "TemporalGeometryAnalyzer",
    "TemporalGeometryComponents",
    "TemporalGeometryReport",
    "TemporalGradientConsistency",
    "extract_temporal_activations",
]
