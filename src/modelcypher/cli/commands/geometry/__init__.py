"""Geometry analysis CLI commands.

This package contains all geometry-related CLI commands organized by domain:
- emotion: Emotion concept analysis
- refinement: Refinement density analysis for model merging

Each module exports a Typer app that is registered as a sub-command of `mc geometry`.
"""

import typer

from modelcypher.cli.commands.geometry import emotion

# Create the main geometry app
geometry_app = typer.Typer(no_args_is_help=True)

# Register sub-command apps
geometry_app.add_typer(emotion.app, name="emotion")

__all__ = ["geometry_app", "emotion"]
