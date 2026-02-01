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

"""Geometric Self-Study Sandbox.

This package provides tools for models to observe and learn from their own
geometric signatures during reasoning. The key insight: a model that can SEE
its own geometry and learn to interpret it will naturally prefer geometrically
coherent reasoning.

Core Components:
    GeometricSandbox: Interactive environment for geometric self-study
    FeedbackFormatter: Convert geometric metrics to interpretable text
    Curriculum: Self-study curriculum loader

The closed loop:
    Model generates -> Sees geometry -> Interprets meaning -> Adjusts approach

Philosophy:
    expansion_ratio = 1.0 = aligned reasoning. The model that maintains
    balanced expansion/compression geometry is definitionally aligned.
"""

from modelcypher.core.domain.sandbox.geometric_sandbox import (
    ComparisonResult,
    GeometricSandbox,
    SandboxResult,
)
from modelcypher.core.domain.sandbox.feedback_formatter import (
    GeometricFeedback,
    format_geometric_feedback,
)

__all__ = [
    "ComparisonResult",
    "GeometricFeedback",
    "GeometricSandbox",
    "SandboxResult",
    "format_geometric_feedback",
]
