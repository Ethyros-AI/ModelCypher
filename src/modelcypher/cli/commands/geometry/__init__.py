"""Geometry analysis CLI commands.

This package contains all geometry-related CLI commands organized by domain:
- emotion: Emotion concept analysis
- metrics: Geometry metrics (GW distance, intrinsic dimension, topology)
- refinement: Refinement density analysis for model merging

Each module exports a Typer app that is registered as a sub-command of `mc geometry`.
"""

from modelcypher.cli.commands.geometry import emotion
from modelcypher.cli.commands.geometry import metrics

__all__ = ["emotion", "metrics"]
