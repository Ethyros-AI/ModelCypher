import json
import logging
import typer
import click
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)

def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj

def _resolve_text_backbone(model, model_type: str):
    """
    Resolve the text backbone components from various model architectures.
    Returns: Tuple of (embed_tokens, layers, norm) or None if resolution fails.
    """
    embed_tokens = None
    layers = None
    norm = None

    # Strategy 1: Standard mlx_lm structure
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        typer.echo("DEBUG: Using Strategy 1 (Standard)")
        embed_tokens = model.model.embed_tokens
        layers = model.model.layers
        norm = getattr(model.model, "norm", None)
        return (embed_tokens, layers, norm)

    # Strategy 2: Multimodal VL wrapper
    if hasattr(model, "language_model"):
        typer.echo("DEBUG: Using Strategy 2 (Multimodal)")
        lm = model.language_model
        if hasattr(lm, "transformer"):
            typer.echo("DEBUG: Detected transformer in language_model")
            transformer = lm.transformer
            embed_tokens = getattr(transformer, "embedding", None)
            if embed_tokens is not None:
                embed_tokens = getattr(embed_tokens, "word_embeddings", embed_tokens)
            layers = getattr(transformer, "encoder", None)
            if layers is not None:
                layers = getattr(layers, "layers", layers)
            norm = getattr(transformer, "output_layer_norm", None)
            if embed_tokens is not None and layers is not None:
                return (embed_tokens, layers, norm)
        
        if hasattr(lm, "model"):
            typer.echo("DEBUG: Detected model in language_model")
            embed_tokens = getattr(lm.model, "embed_tokens", None)
            layers = getattr(lm.model, "layers", None)
            norm = getattr(lm.model, "norm", None)
            if embed_tokens is not None and layers is not None:
                return (embed_tokens, layers, norm)

    # Strategy 3: Deep Search for lists of modules (layers)
    def find_deep(module):
        found_embed = None
        found_layers = None
        found_norm = None
        
        # Look for layers first (ModuleList or list of transformer blocks)
        for name, m in module.named_modules():
            # Most common layer collection names
            if any(x in name.lower() for x in ["layers", "blocks", "h", "encoder"]):
                # It should be a collection of modules
                if hasattr(m, "__getitem__") and hasattr(m, "__len__") and len(m) > 0:
                    first = m[0]
                    # Duck typing check for a transformer layer
                    if hasattr(first, "self_attn") or hasattr(first, "attention") or hasattr(first, "mlp"):
                        found_layers = m
                        typer.echo(f"DEBUG: Found layers at {name}")
                        break
        
        # Look for embedding
        for name, m in module.named_modules():
            if "embed" in name.lower() and hasattr(m, "__call__") and not hasattr(m, "__getitem__"):
                found_embed = m
                typer.echo(f"DEBUG: Found embedding at {name}")
                break
                
        # Look for final norm
        for name, m in module.named_modules():
            if "norm" in name.lower() and "layer" not in name.lower():
                found_norm = m
                # Keep looking for the last one usually
        
        if found_norm:
            typer.echo(f"DEBUG: Found norm")
                
        return found_embed, found_layers, found_norm

    embed_tokens, layers, norm = find_deep(model)
    if embed_tokens and layers:
        return (embed_tokens, layers, norm)

    return None

def _forward_text_backbone(input_ids, embed_tokens, layers, norm, target_layer: int, backend):
    """Execution engine for hidden state extraction."""
    # Special case: GLM-4V internal model handles everything (RoPE, mask)
    # We just need to iterate through its layers and break at target
    if hasattr(embed_tokens, "__obj__") and "language_model" in str(embed_tokens.__obj__):
        # If we passed model.language_model.model.embed_tokens, 
        # its parent is usually the correct execution context
        pass

    # Generic execution loop
    hidden = embed_tokens(input_ids)
    
    seq_len = input_ids.shape[1]
    mask = backend.create_causal_mask(seq_len, hidden.dtype)

    for i, layer in enumerate(layers):
        try:
            # GLM-4V layers in MLX-VLM require (h, mask, cache, position_embeddings)
            # If we don't have position_embeddings, we must use the parent __call__
            # Let's try a safer approach for these complex models:
            # If it's a known complex layer, try to call it via its parent's logic if possible
            hidden = layer(hidden, mask=mask)
        except Exception:
            try:
                hidden = layer(hidden)
            except Exception as e:
                # If everything fails, it's likely missing the multimodal context (RoPE)
                raise RuntimeError(f"Layer {i} forward failed. Complex architecture requires full model call.") from e
        
        if i == target_layer or (target_layer == -1 and i == len(layers) - 1):
            break

    if norm is not None and (target_layer == -1 or target_layer == len(layers) - 1):
        hidden = norm(hidden)

    return hidden

@app.command("probe-model")
def spatial_probe_model(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    layer: int = typer.Option(-1, help="Layer to analyze (default is last)"),
    output_file: str = typer.Option(None, "--output", "-o", help="File to save activations"),
) -> None:
    """Probe a model for 3D world model geometry."""
    context = _context(ctx)

    from modelcypher.core.domain.geometry.spatial_3d import (
        SPATIAL_PRIME_ATLAS,
        Spatial3DAnalyzer,
    )
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend

    typer.echo(f"Loading model from {model_path}...")
    model, tokenizer = load_model_for_training(model_path)

    model_type = getattr(model, "model_type", "unknown")
    resolved = _resolve_text_backbone(model, model_type)
    
    if not resolved:
        typer.echo("Error: Could not resolve architecture.", err=True)
        raise typer.Exit(1)
        
    embed_tokens, layers, norm = resolved
    typer.echo(f"Architecture resolved: {len(layers)} layers")

    backend = MLXBackend()
    anchor_activations = {}

    for anchor in SPATIAL_PRIME_ATLAS:
        tokens = tokenizer.encode(anchor.prompt)
        input_ids = backend.array([tokens])

        try:
            hidden = _forward_text_backbone(input_ids, embed_tokens, layers, norm, layer, backend)
            activation = backend.mean(hidden[0], axis=0)
            backend.eval(activation)
            anchor_activations[anchor.name] = activation
        except Exception as e:
            typer.echo(f"  Warning: Failed anchor {anchor.name}: {e}", err=True)

    if not anchor_activations:
        typer.echo("Error: No activations extracted.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracted {len(anchor_activations)} activations.")

    analyzer = Spatial3DAnalyzer(backend=backend)
    report = analyzer.full_analysis(anchor_activations)

    payload = {
        "_schema": "mc.geometry.spatial.probe_model.v1",
        "model_path": model_path,
        **report.to_dict(),
        "verdict": (
            "HIGH VISUAL GROUNDING" if report.has_3d_world_model and report.physics_engine_detected else
            "MODERATE GROUNDING" if report.has_3d_world_model else "ALTERNATIVE GROUNDING"
        ),
    }

    if context.output_format == "text":
        typer.echo("=" * 60)
        typer.echo(f"3D WORLD MODEL ANALYSIS: {Path(model_path).name}")
        typer.echo("=" * 60)
        typer.echo(f"World Model Score: {report.world_model_score:.2f}")
        typer.echo(f"Verdict: {payload['verdict']}")
        return

    write_output(payload, context.output_format, context.pretty)

@app.command("analyze")
def spatial_analyze(
    ctx: typer.Context,
    activations_file: str = typer.Argument(..., help="JSON file with activations"),
):
    """Run full 3D world model analysis from saved activations."""
    context = _context(ctx)
    from modelcypher.core.domain.geometry.spatial_3d import Spatial3DAnalyzer
    from modelcypher.backends.mlx_backend import MLXBackend

    data = json.loads(Path(activations_file).read_text())
    backend = MLXBackend()
    anchor_activations = {k: backend.array(v) for k, v in data.items()}

    analyzer = Spatial3DAnalyzer(backend=backend)
    report = analyzer.full_analysis(anchor_activations)
    write_output(report.to_dict(), context.output_format, context.pretty)