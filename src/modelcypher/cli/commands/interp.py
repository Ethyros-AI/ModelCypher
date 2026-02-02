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
- Label: Token-level domain labeling via SAE latents

Usage:
    mc interp sae train --model PATH --layer 16 --output sae.json
    mc interp sae encode --model PATH --sae sae.json --prompt "Hello"
    mc interp sae features --sae sae.json --top-k 10
    mc interp patch --model PATH --layer 10 --clean "good" --corrupt "bad"
    mc interp steer --model PATH --direction refusal --strength -0.5
    mc interp diff --base PATH --finetuned PATH --layer 16
    mc interp label run --sae sae.json --latents domain.json --input texts.jsonl
    mc interp label calibrate --sae sae.json --latents domain.json --target-rate 0.1
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
label_app = typer.Typer(no_args_is_help=True)
app.add_typer(sae_app, name="sae", help="Sparse Autoencoder operations")
app.add_typer(label_app, name="label", help="Token-level labeling via SAE latents")


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


@label_app.command("info")
def label_info(ctx: typer.Context) -> None:
    """Show token labeling module information."""
    context = _context(ctx)

    payload = {
        "module": "Token Labeling",
        "purpose": "Label tokens via SAE latent activations for domain filtering",
        "capabilities": [
            "Domain-specific latent identification",
            "Token-level labeling with 4σ threshold",
            "Adjacent token expansion",
            "Threshold calibration for target positive rate",
        ],
        "reference": "arXiv:2601.21571v1",
        "config_options": {
            "min_active_latents": "Minimum domain latents required (default: 2)",
            "activation_threshold_sigma": "Sigma threshold for activation (default: 4.0)",
            "expand_adjacent": "Expand labels to adjacent tokens",
            "expansion_radius": "Number of tokens to expand (default: 1)",
        },
    }

    if context.output_format == "text":
        lines = [
            "TOKEN LABELING (SAE-based)",
            "",
            "Purpose: Label tokens via SAE latent activations for domain filtering",
            "",
            "Capabilities:",
            "  - Domain-specific latent identification",
            "  - Token-level labeling with 4σ threshold",
            "  - Adjacent token expansion",
            "  - Threshold calibration for target positive rate",
            "",
            "Configuration:",
            "  - min_active_latents: Minimum domain latents required (default: 2)",
            "  - activation_threshold_sigma: Sigma threshold (default: 4.0)",
            "  - expand_adjacent: Expand labels to neighbors",
            "  - expansion_radius: Expansion distance (default: 1)",
            "",
            "Reference: arXiv:2601.21571v1",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@label_app.command("run")
def label_run(
    ctx: typer.Context,
    sae_activations_file: str = typer.Option(..., "--activations", "-a", help="JSONL file with SAE activations"),
    domain_latents_file: str = typer.Option(..., "--latents", "-l", help="JSON file with domain latent indices"),
    output_file: Optional[str] = typer.Option(None, "--out-file", "-o", help="Output JSONL file for labels"),
    min_active_latents: int = typer.Option(2, "--min-latents", help="Minimum domain latents required"),
    sigma_threshold: float = typer.Option(4.0, "--sigma", help="Activation sigma threshold"),
    expand: bool = typer.Option(True, "--expand/--no-expand", help="Expand labels to adjacent tokens"),
    expansion_radius: int = typer.Option(1, "--radius", help="Token expansion radius"),
) -> None:
    """Run token labeling on SAE activations.

    Reads pre-computed SAE activations and labels tokens based on
    domain-specific latent activation patterns.

    Examples:
        mc interp label run --activations acts.jsonl --latents domain.json
        mc interp label run -a acts.jsonl -l domain.json --sigma 3.0 --out-file labels.jsonl
    """
    context = _context(ctx)

    try:
        from modelcypher.backends import initialize_default_backend
        from modelcypher.core.domain.interpretability.token_labeling import TokenLabelingConfig
        from modelcypher.core.use_cases.token_labeling_service import TokenLabelingService

        backend = initialize_default_backend()

        # Load activations from JSONL
        activations_data = []
        text_lengths = []
        with open(sae_activations_file, "r") as f:
            for line in f:
                record = json.loads(line)
                activations_data.extend(record.get("activations", []))
                text_lengths.append(len(record.get("activations", [])))

        if not activations_data:
            write_error(
                ErrorDetail(
                    code="MC-3001",
                    title="Empty activations file",
                    detail="No activations found in the input file",
                    hint="Check that the file contains valid JSONL records with 'activations' field",
                    trace_id=context.trace_id,
                ).as_dict(),
                context.output_format,
                context.pretty,
            )
            raise typer.Exit(code=1)

        sae_activations = backend.array(activations_data)

        # Load domain latent indices
        service = TokenLabelingService(backend)
        domain_latents = service.load_domain_latents(domain_latents_file)

        config = TokenLabelingConfig(
            min_active_latents=min_active_latents,
            activation_threshold_sigma=sigma_threshold,
            expand_adjacent=expand,
            expansion_radius=expansion_radius,
        )

        summary, _ = service.run_labeling(
            sae_activations=sae_activations,
            domain_latent_indices=domain_latents,
            text_lengths=text_lengths,
            config=config,
            output_path=output_file,
        )

        payload = TokenLabelingService.label_run_payload(summary)

        if context.output_format == "text":
            lines = [
                "TOKEN LABELING COMPLETE",
                f"Total tokens: {summary.total_tokens}",
                f"Positive tokens: {summary.positive_tokens}",
                f"Positive rate: {summary.positive_rate:.2%}",
                f"Texts processed: {summary.texts_processed}",
            ]
            if summary.output_path:
                lines.append(f"Output: {summary.output_path}")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3002",
            title="Token labeling failed",
            detail=str(exc),
            hint="Check activations and latent indices files",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@label_app.command("calibrate")
def label_calibrate(
    ctx: typer.Context,
    sae_activations_file: str = typer.Option(..., "--activations", "-a", help="JSONL file with SAE activations"),
    domain_latents_file: str = typer.Option(..., "--latents", "-l", help="JSON file with domain latent indices"),
    target_rate: float = typer.Option(0.1, "--target-rate", "-t", help="Target positive rate"),
    output_file: Optional[str] = typer.Option(None, "--out-file", "-o", help="Output JSON file for calibration"),
) -> None:
    """Calibrate labeling threshold for target positive rate.

    Finds the sigma threshold that achieves approximately the target
    fraction of tokens labeled as positive.

    Examples:
        mc interp label calibrate --activations acts.jsonl --latents domain.json --target-rate 0.05
    """
    context = _context(ctx)

    try:
        from modelcypher.backends import initialize_default_backend
        from modelcypher.core.use_cases.token_labeling_service import TokenLabelingService

        backend = initialize_default_backend()

        # Load activations from JSONL
        activations_data = []
        with open(sae_activations_file, "r") as f:
            for line in f:
                record = json.loads(line)
                activations_data.extend(record.get("activations", []))

        if not activations_data:
            write_error(
                ErrorDetail(
                    code="MC-3001",
                    title="Empty activations file",
                    detail="No activations found in the input file",
                    hint="Check that the file contains valid JSONL records",
                    trace_id=context.trace_id,
                ).as_dict(),
                context.output_format,
                context.pretty,
            )
            raise typer.Exit(code=1)

        sae_activations = backend.array(activations_data)

        # Load domain latent indices
        service = TokenLabelingService(backend)
        domain_latents = service.load_domain_latents(domain_latents_file)

        result = service.calibrate(
            sae_activations=sae_activations,
            domain_latent_indices=domain_latents,
            target_positive_rate=target_rate,
        )

        if output_file:
            service.save_calibration(result, output_file)

        payload = TokenLabelingService.calibration_payload(result)

        if context.output_format == "text":
            lines = [
                "THRESHOLD CALIBRATION COMPLETE",
                f"Calibrated sigma: {result.calibrated_sigma:.4f}",
                f"Target rate: {result.target_positive_rate:.2%}",
                f"Achieved rate: {result.achieved_positive_rate:.2%}",
                f"Sample count: {result.sample_count}",
            ]
            if output_file:
                lines.append(f"Saved to: {output_file}")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3003",
            title="Calibration failed",
            detail=str(exc),
            hint="Check activations and latent indices files",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


__all__ = ["app", "sae_app", "label_app"]
