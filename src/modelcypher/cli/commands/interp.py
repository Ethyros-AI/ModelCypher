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

"""Mechanistic Interpretability CLI commands.

Commands for SOTA interpretability tools:
- SAE (Sparse Autoencoders): Extract monosemantic features
- Patching: Causal intervention experiments
- Steering: Direction-based behavior modification

Usage:
    mc interp sae train --model PATH --layer 16 --output sae.json
    mc interp sae encode --model PATH --sae sae.json --prompt "Hello"
    mc interp sae features --sae sae.json --top-k 10
    mc interp patch --model PATH --layer 10 --clean "good" --corrupt "bad"
    mc interp steer --model PATH --direction refusal --strength -0.5
    mc interp diff --base PATH --finetuned PATH --layer 16
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)
sae_app = typer.Typer(no_args_is_help=True)
app.add_typer(sae_app, name="sae", help="Sparse Autoencoder operations")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@sae_app.command("info")
def sae_info(ctx: typer.Context) -> None:
    """Show SAE module information and capabilities."""
    context = _context(ctx)

    payload = {
        "module": "Sparse Autoencoder (SAE)",
        "purpose": "Extract monosemantic features from polysemantic neurons",
        "capabilities": [
            "Feature extraction via encoding",
            "Reconstruction via decoding",
            "Feature analysis (top-k activation)",
            "Feature direction extraction for steering",
        ],
        "config_options": {
            "hidden_dim": "Model activation dimension",
            "expansion_factor": "Latent dimension multiplier (4-32x typical)",
            "sparsity_coefficient": "L1 penalty (derived from data if None)",
            "normalize_decoder": "Normalize decoder columns to unit norm",
        },
        "geodesic": True,
        "backend_agnostic": True,
    }

    if context.output_format == "text":
        lines = [
            "SPARSE AUTOENCODER (SAE)",
            "",
            "Purpose: Extract monosemantic features from polysemantic neurons",
            "",
            "Capabilities:",
            "  - Feature extraction via encoding",
            "  - Reconstruction via decoding",
            "  - Feature analysis (top-k activation)",
            "  - Feature direction extraction for steering",
            "",
            "Configuration:",
            "  - hidden_dim: Model activation dimension",
            "  - expansion_factor: Latent dimension multiplier (4-32x)",
            "  - sparsity_coefficient: L1 penalty (auto-derived)",
            "  - normalize_decoder: Unit norm decoder columns",
            "",
            "Properties:",
            "  - Geodesic reconstruction loss (not Euclidean)",
            "  - Backend-agnostic (MLX/JAX/CUDA)",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@sae_app.command("config")
def sae_config(
    ctx: typer.Context,
    hidden_dim: int = typer.Option(768, help="Model hidden dimension"),
    expansion_factor: int = typer.Option(8, help="Latent expansion factor"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output config file"),
) -> None:
    """Generate SAE configuration file."""
    context = _context(ctx)

    from modelcypher.core.domain.interpretability.sae import SAEConfig

    config = SAEConfig(
        hidden_dim=hidden_dim,
        expansion_factor=expansion_factor,
    )

    payload = {
        "hidden_dim": config.hidden_dim,
        "expansion_factor": config.expansion_factor,
        "latent_dim": config.latent_dim,
        "sparsity_coefficient": config.sparsity_coefficient,
        "normalize_decoder": config.normalize_decoder,
        "tied_weights": config.tied_weights,
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        write_output(f"Config written to {output_file}", context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("patch-info")
def patch_info(ctx: typer.Context) -> None:
    """Show activation patching module information."""
    context = _context(ctx)

    payload = {
        "module": "Activation Patching",
        "purpose": "Causal intervention to localize computation",
        "capabilities": [
            "Clean/corrupt run capture",
            "Single-layer patching",
            "Path patching (multi-layer trace)",
            "KL divergence and causal effect measurement",
        ],
        "patch_components": ["residual", "attention", "mlp", "attention_output", "mlp_output"],
        "geodesic": True,
        "backend_agnostic": True,
    }

    if context.output_format == "text":
        lines = [
            "ACTIVATION PATCHING",
            "",
            "Purpose: Causal intervention to localize computation",
            "",
            "Capabilities:",
            "  - Clean/corrupt run capture",
            "  - Single-layer patching",
            "  - Path patching (multi-layer trace)",
            "  - KL divergence and causal effect measurement",
            "",
            "Patch Components:",
            "  - residual: Main residual stream",
            "  - attention: Attention output",
            "  - mlp: MLP output",
            "",
            "Properties:",
            "  - Geodesic distance for effect measurement",
            "  - Backend-agnostic (MLX/JAX/CUDA)",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("steer-info")
def steer_info(ctx: typer.Context) -> None:
    """Show feature steering module information."""
    context = _context(ctx)

    payload = {
        "module": "Feature Steering",
        "purpose": "Modify behavior via activation intervention",
        "capabilities": [
            "Contrastive direction extraction",
            "SAE feature direction steering",
            "Null-space constrained steering (AlphaSteer)",
            "Position-specific steering",
        ],
        "steering_sources": ["contrastive", "sae_feature", "refusal", "mean_difference", "custom"],
        "geodesic": True,
        "backend_agnostic": True,
    }

    if context.output_format == "text":
        lines = [
            "FEATURE STEERING",
            "",
            "Purpose: Modify behavior via activation intervention",
            "",
            "Capabilities:",
            "  - Contrastive direction extraction",
            "  - SAE feature direction steering",
            "  - Null-space constrained steering (AlphaSteer)",
            "  - Position-specific steering",
            "",
            "Steering Sources:",
            "  - contrastive: From paired prompts",
            "  - sae_feature: From SAE decoder",
            "  - refusal: Refusal direction",
            "  - mean_difference: Between concept sets",
            "",
            "Properties:",
            "  - Geodesic null-space projection",
            "  - Backend-agnostic (MLX/JAX/CUDA)",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("transcoder-info")
def transcoder_info(ctx: typer.Context) -> None:
    """Show transcoder module information."""
    context = _context(ctx)

    payload = {
        "module": "Transcoder",
        "purpose": "Cross-layer MLP replacement for circuit tracing",
        "capabilities": [
            "MLP input-to-output transcoding",
            "Sparse feature extraction",
            "Feature contribution analysis",
            "Runtime MLP replacement",
        ],
        "geodesic": True,
        "backend_agnostic": True,
    }

    if context.output_format == "text":
        lines = [
            "TRANSCODER",
            "",
            "Purpose: Cross-layer MLP replacement for circuit tracing",
            "",
            "Capabilities:",
            "  - MLP input-to-output transcoding",
            "  - Sparse feature extraction",
            "  - Feature contribution analysis",
            "  - Runtime MLP replacement for tracing",
            "",
            "Architecture:",
            "  - Input: MLP layer input",
            "  - Output: Predicted MLP output",
            "  - Latent: Sparse interpretable features",
            "",
            "Properties:",
            "  - Geodesic reconstruction loss",
            "  - Backend-agnostic (MLX/JAX/CUDA)",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("crosscoder-info")
def crosscoder_info(ctx: typer.Context) -> None:
    """Show crosscoder module information."""
    context = _context(ctx)

    payload = {
        "module": "Crosscoder",
        "purpose": "Model diffing between base and fine-tuned",
        "capabilities": [
            "Joint SAE on two related models",
            "Shared vs exclusive feature identification",
            "Change magnitude quantification",
            "CKA alignment measurement",
        ],
        "feature_types": ["shared", "base_exclusive", "ft_exclusive"],
        "geodesic": True,
        "backend_agnostic": True,
    }

    if context.output_format == "text":
        lines = [
            "CROSSCODER",
            "",
            "Purpose: Model diffing between base and fine-tuned",
            "",
            "Capabilities:",
            "  - Joint SAE on two related models",
            "  - Shared vs exclusive feature identification",
            "  - Change magnitude quantification",
            "  - CKA alignment measurement",
            "",
            "Feature Types:",
            "  - shared: Present in both models",
            "  - base_exclusive: Only in base model",
            "  - ft_exclusive: Only in fine-tuned model",
            "",
            "Properties:",
            "  - Geodesic reconstruction loss",
            "  - Backend-agnostic (MLX/JAX/CUDA)",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("modules")
def list_modules(ctx: typer.Context) -> None:
    """List all interpretability modules and their status."""
    context = _context(ctx)

    payload = {
        "modules": [
            {
                "name": "sae",
                "description": "Sparse Autoencoders - monosemantic feature extraction",
                "status": "implemented",
                "subcommands": ["info", "config"],
            },
            {
                "name": "sae_training",
                "description": "SAE training loop with Adam optimizer",
                "status": "implemented",
                "subcommands": [],
            },
            {
                "name": "activation_patching",
                "description": "Causal intervention for circuit discovery",
                "status": "implemented",
                "subcommands": ["info"],
            },
            {
                "name": "feature_steering",
                "description": "Direction-based behavior modification",
                "status": "implemented",
                "subcommands": ["info"],
            },
            {
                "name": "transcoder",
                "description": "Cross-layer MLP replacement",
                "status": "implemented",
                "subcommands": ["info"],
            },
            {
                "name": "crosscoder",
                "description": "Model diffing via joint SAE",
                "status": "implemented",
                "subcommands": ["info"],
            },
        ],
        "principles": {
            "geodesic": "All modules use geodesic distances (not Euclidean)",
            "backend_agnostic": "All modules work with MLX/JAX/CUDA backends",
            "threshold_free": "No hardcoded thresholds - derived from data",
        },
    }

    if context.output_format == "text":
        lines = [
            "MECHANISTIC INTERPRETABILITY MODULES",
            "",
            "Modules:",
        ]
        for mod in payload["modules"]:
            status_icon = "✓" if mod["status"] == "implemented" else "○"
            lines.append(f"  {status_icon} {mod['name']}: {mod['description']}")
            if mod["subcommands"]:
                lines.append(f"      Commands: {', '.join(mod['subcommands'])}")
        lines.extend([
            "",
            "Principles:",
            "  - Geodesic: All distances are geodesic (not Euclidean)",
            "  - Backend-agnostic: MLX/JAX/CUDA via Backend protocol",
            "  - Threshold-free: All values derived from data",
            "",
            "Usage:",
            "  mc interp sae info",
            "  mc interp sae config --hidden-dim 768",
            "  mc interp patch-info",
            "  mc interp steer-info",
            "  mc interp transcoder-info",
            "  mc interp crosscoder-info",
        ])
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app", "sae_app"]
