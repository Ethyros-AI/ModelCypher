# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Self-Consistency Through Thinking.

The hypothesis: fundamental constants (e/π, π/e, φ) emerge from coherent
information processing, not from being forced. If a model engages in genuine
self-questioning to achieve internal consistency, the geometric signatures
should appear naturally.

Modules:
    probing: Generate implications, contradictions, and connections
    consistency_measure: Measure semantic consistency using representation distance
    thinking_loop: Iterative self-questioning until coherence

The key insight: learning isn't about modifying weights. It's about
processing information recursively until coherence emerges.
"""

from __future__ import annotations

__all__ = [
    "SelfConsistencyProber",
    "ConsistencyMeasure",
    "ThinkingLoop",
    "ThinkingResult",
]


def __getattr__(name: str):
    """Lazy load submodules."""
    if name == "SelfConsistencyProber":
        from .probing import SelfConsistencyProber
        return SelfConsistencyProber
    if name == "ConsistencyMeasure":
        from .consistency_measure import ConsistencyMeasure
        return ConsistencyMeasure
    if name in ("ThinkingLoop", "ThinkingResult"):
        from .thinking_loop import ThinkingLoop, ThinkingResult
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
