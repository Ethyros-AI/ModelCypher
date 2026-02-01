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

"""Geometric Self-Alignment System.

An algorithm that lets any model self-play and modify its own weights to
reduce entropy across the full manifold, using pure geometric measurements.

No external supervision. The geometry IS the teacher.

Modules:
    direction_generator: Generate candidate weight perturbations
    convergence_detector: Detect when entropy has stabilized
    geometric_self_alignment: Main orchestrator
"""

from __future__ import annotations

# Lazy imports to avoid circular dependencies
__all__ = [
    # Direction generation
    "DirectionGenerator",
    "DirectionResult",
    "DirectionStrategy",
    # Convergence detection
    "ConvergenceDetector",
    "ConvergenceResult",
    # Autonomous completion
    "AutonomousCompletion",
    "AutonomousRunResult",
    # Main orchestrator
    "GeometricSelfAlignment",
    "AlignmentResult",
    "AlignmentRoundResult",
]


def __getattr__(name: str):
    """Lazy load submodules."""
    if name in ("DirectionGenerator", "DirectionResult", "DirectionStrategy"):
        from .direction_generator import (
            DirectionGenerator,
            DirectionResult,
            DirectionStrategy,
        )
        return locals()[name]
    if name in ("ConvergenceDetector", "ConvergenceResult"):
        from .convergence_detector import ConvergenceDetector, ConvergenceResult
        return locals()[name]
    if name in ("AutonomousCompletion", "AutonomousRunResult"):
        from .autonomous_completion import AutonomousCompletion, AutonomousRunResult
        return locals()[name]
    if name in ("GeometricSelfAlignment", "AlignmentResult", "AlignmentRoundResult"):
        from .geometric_self_alignment import (
            GeometricSelfAlignment,
            AlignmentResult,
            AlignmentRoundResult,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
