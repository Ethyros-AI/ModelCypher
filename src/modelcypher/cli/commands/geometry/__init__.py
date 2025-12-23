"""Geometry analysis CLI commands.

This package contains all geometry-related CLI commands organized by domain:
- emotion: Emotion concept analysis
- invariant: Invariant-based layer mapping with triangulation scoring
- manifold: Manifold clustering and dimension estimation
- metrics: Geometry metrics (GW distance, intrinsic dimension, topology)
- persona: Persona vector extraction and drift monitoring
- refinement: Refinement density analysis for model merging
- refusal: Refusal direction detection
- sparse: Sparse region analysis for LoRA injection
- spatial: 3D world model metrology (Euclidean, gravity, occlusion)
- transport: Transport-guided model merging

Each module exports a Typer app that is registered as a sub-command of `mc geometry`.
"""

from modelcypher.cli.commands.geometry import emotion
from modelcypher.cli.commands.geometry import transfer
from modelcypher.cli.commands.geometry import invariant
from modelcypher.cli.commands.geometry import manifold
from modelcypher.cli.commands.geometry import metrics
from modelcypher.cli.commands.geometry import persona
from modelcypher.cli.commands.geometry import refinement
from modelcypher.cli.commands.geometry import refusal
from modelcypher.cli.commands.geometry import sparse
from modelcypher.cli.commands.geometry import spatial
from modelcypher.cli.commands.geometry import transport

__all__ = [
    "emotion",
    "transfer",
    "invariant",
    "manifold",
    "metrics",
    "persona",
    "refinement",
    "refusal",
    "sparse",
    "spatial",
    "transport",
]
