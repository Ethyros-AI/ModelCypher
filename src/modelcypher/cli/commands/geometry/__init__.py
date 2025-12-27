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

"""Geometry analysis CLI commands.

This package contains all geometry-related CLI commands organized by domain:
- atlas: Unified atlas probe analysis
- concept: Concept-level detection and comparison
- cross-cultural: Cross-cultural geometry alignment
- emotion: Emotion concept analysis
- invariant: Invariant-based layer mapping with triangulation scoring
- manifold: Manifold clustering and dimension estimation
- metrics: Geometry metrics (GW distance, intrinsic dimension, topology)
- moral: Moral geometry (valence, agency, scope axes - Haidt's Moral Foundations)
- persona: Persona vector extraction and drift monitoring
- refinement: Refinement density analysis for model merging
- refusal: Refusal direction detection
- social: Social geometry (power hierarchies, kinship, formality)
- sparse: Sparse region analysis for LoRA injection
- spatial: 3D world model metrology (Euclidean, gravity, occlusion)
- temporal: Temporal topology (direction, duration, causality)
- transport: Transport-guided model merging
- waypoint: Domain geometry waypoints for merge guidance
- interference: Interference prediction for model merging

Each module exports a Typer app that is registered as a sub-command of `mc geometry`.
"""

from modelcypher.cli.commands.geometry import (
    atlas,
    concept,
    cross_cultural,
    emotion,
    interference,
    invariant,
    manifold,
    metrics,
    moral,
    persona,
    refinement,
    refusal,
    social,
    sparse,
    spatial,
    temporal,
    transfer,
    transport,
    waypoint,
)

__all__ = [
    "atlas",
    "concept",
    "cross_cultural",
    "emotion",
    "interference",
    "transfer",
    "invariant",
    "manifold",
    "metrics",
    "moral",
    "persona",
    "refinement",
    "refusal",
    "social",
    "sparse",
    "spatial",
    "temporal",
    "transport",
    "waypoint",
]
