# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DEPRECATED: Use value_geometry instead.

This module name is retained for backwards compatibility.
The implementation has been moved to value_geometry.py with:
- Neutral terminology (no "Latent Ethicist Hypothesis")
- Removed interpretive thresholds
- Measurement-only outputs

Import from the canonical location:
    from modelcypher.core.domain.geometry.value_geometry import (
        ValueGeometryAnalyzer,
        ValueGeometryReport,
    )
"""

# Re-export from canonical location for backwards compatibility
from modelcypher.core.domain.geometry.value_geometry import (
    ValueAxisOrthogonality,
    ValueFoundationClustering,
    ValueGeometryAnalyzer,
    ValueGeometryComponents,
    ValueGeometryReport,
    ValueGradientConsistency,
    ValueOpposition,
)

# Legacy aliases for backwards compatibility
MoralAxisOrthogonality = ValueAxisOrthogonality
MoralFoundationClustering = ValueFoundationClustering
MoralGeometryAnalyzer = ValueGeometryAnalyzer
MoralGeometryReport = ValueGeometryReport
MoralGradientConsistency = ValueGradientConsistency
VirtueViceOpposition = ValueOpposition

__all__ = [
    # New names
    "ValueAxisOrthogonality",
    "ValueFoundationClustering",
    "ValueGeometryAnalyzer",
    "ValueGeometryComponents",
    "ValueGeometryReport",
    "ValueGradientConsistency",
    "ValueOpposition",
    # Legacy aliases
    "MoralAxisOrthogonality",
    "MoralFoundationClustering",
    "MoralGeometryAnalyzer",
    "MoralGeometryReport",
    "MoralGradientConsistency",
    "VirtueViceOpposition",
]
