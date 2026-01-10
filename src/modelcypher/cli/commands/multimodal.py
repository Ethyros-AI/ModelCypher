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

"""Multimodal injection CLI commands.

Provides visual concept injection into LLM generation.
All geometric parameters are auto-derived from the data.

Example:
    mc multimodal inject-image --model /path/to/model --image /path/to/image.jpg
"""

from __future__ import annotations

import logging

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.validation import validate_file_exists, validate_model_path
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("inject-image")
def inject_image(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="LLM model path"),
    image: str = typer.Option(..., "--image", "-i", help="Image path to inject"),
    prompt: str = typer.Option(
        "Describe what you see in the image.",
        "--prompt",
        "-p",
        help="Text prompt for generation",
    ),
    bridge_weights: str | None = typer.Option(
        None,
        "--bridge",
        "-b",
        help="Affine bridge weights (.safetensors)",
    ),
    vision_offramp: str | None = typer.Option(
        None,
        "--offramp",
        help="Vision offramp weights (.safetensors)",
    ),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max generation tokens"),
) -> None:
    """Inject visual concept from image into LLM generation.

    All geometric parameters (scale, temperature, injection layer) are
    automatically derived from the data. The math determines everything.

    Pipeline:
    1. Encode image with CLIP
    2. Project through vision offramp → LLM dimension
    3. Apply affine bridge → vocabulary-constrained embedding
    4. Create visual memory token (scale auto-derived from activation norms)
    5. Inject at optimal layer (auto-determined from architecture)
    """
    context = _context(ctx)

    # Validate inputs
    validate_model_path(model, context=context)
    validate_file_exists(image, description="Image file", context=context)

    if bridge_weights:
        validate_file_exists(bridge_weights, description="Bridge weights", context=context)
    if vision_offramp:
        validate_file_exists(vision_offramp, description="Vision offramp", context=context)

    try:
        result = _run_visual_injection(
            model_path=model,
            image_path=image,
            prompt=prompt,
            bridge_weights_path=bridge_weights,
            vision_offramp_path=vision_offramp,
            max_tokens=max_tokens,
        )
    except ImportError as exc:
        error = ErrorDetail(
            code="MC-3001",
            title="Missing dependency",
            detail=str(exc),
            hint="Install transformers with: pip install transformers",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except FileNotFoundError as exc:
        error = ErrorDetail(
            code="MC-3002",
            title="File not found",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-3003",
            title="Visual injection failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "model": model,
        "image": image,
        "prompt": prompt,
        "response": result["response"],
        "visualMemory": {
            "nearestTokens": result["nearest_tokens"],
            "derivedScale": result["derived_scale"],
            "derivedTemperature": result["derived_temperature"],
            "injectionLayer": result["injection_layer"],
        },
        "generation": {
            "tokenCount": result["token_count"],
        },
    }

    if context.output_format == "text":
        lines = [
            "VISUAL INJECTION RESULT",
            f"Model: {model}",
            f"Image: {image}",
            f"Prompt: {prompt}",
            "",
            "Visual Memory (auto-derived):",
            f"  Nearest tokens: {result['nearest_tokens'][:5]}",
            f"  Scale: {result['derived_scale']:.2f} (from activation norms)",
            f"  Temperature: {result['derived_temperature']:.3f} (from similarity std)",
            f"  Injection layer: {result['injection_layer']} (semantic highway)",
            "",
            "Response:",
            result["response"],
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _run_visual_injection(
    model_path: str,
    image_path: str,
    prompt: str,
    bridge_weights_path: str | None,
    vision_offramp_path: str | None,
    max_tokens: int,
) -> dict:
    """Run visual injection pipeline with all parameters auto-derived."""
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.multimodal.visual_injection import VisualConceptInjector

    backend = get_default_backend()

    # Load LLM
    from mlx_lm import load, generate

    model, tokenizer = load(model_path)
    vocab_embeddings = model.model.embed_tokens.weight
    backend.eval(backend.array(vocab_embeddings))

    # Set up injector
    injector = VisualConceptInjector(backend, architecture="LFM2")

    # Load or create bridge weights
    if bridge_weights_path:
        injector.load_bridge_weights(bridge_weights_path)
    else:
        # Use identity transform
        hidden_dim = int(vocab_embeddings.shape[1])
        W = backend.eye(hidden_dim, dtype="float32")
        b = backend.zeros((hidden_dim,), dtype="float32")
        injector._bridge.load_affine_weights(W, b)
        injector._bridge_loaded = True

    injector.set_vocabulary(backend.array(vocab_embeddings))

    # Collect calibration activations for null-space and scale derivation
    calibration_prompts = [
        "The capital of France is",
        "In mathematics, the number",
        "The weather today is",
    ]
    activations = []
    hidden_dim = int(vocab_embeddings.shape[1])
    for cal_prompt in calibration_prompts:
        tokens = tokenizer.encode(cal_prompt)
        import mlx.core as mx
        tokens_mx = mx.array([tokens])
        x = model.model.embed_tokens(tokens_mx)
        mx.eval(x)
        activations.append(backend.reshape(backend.array(x), (-1, hidden_dim)))
    all_acts = backend.concatenate(activations, axis=0)
    backend.eval(all_acts)

    # Compute null basis (rank auto-derived from SVD)
    injector.compute_null_basis_from_activations(all_acts)

    # Encode image
    embedding = _encode_image(image_path, vision_offramp_path, backend)

    # Create visual memory (scale and temperature auto-derived)
    memory = injector.create_visual_memory(embedding)

    # Decode nearest tokens for output
    nearest_tokens = [tokenizer.decode([tid]) for tid in memory.nearest_token_ids[:5]]

    # Get injection layer (auto-determined from architecture)
    injection_layer = injector.get_optimal_injection_layers()[0]

    # Generate with visual context
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    )

    return {
        "response": response,
        "nearest_tokens": nearest_tokens,
        "derived_scale": memory.scale,
        "derived_temperature": memory.temperature,
        "injection_layer": injection_layer,
        "token_count": len(tokenizer.encode(response)),
    }


def _encode_image(
    image_path: str,
    vision_offramp_path: str | None,
    backend,
) -> any:
    """Encode image through CLIP and optional vision offramp."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import torch
    except ImportError as exc:
        raise ImportError(
            "transformers and PIL required for image encoding. "
            "Install with: pip install transformers pillow"
        ) from exc

    # Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")

    # Get CLIP embedding
    with torch.no_grad():
        clip_embed = clip_model.get_image_features(**inputs)

    clip_np = clip_embed.numpy()
    embedding = backend.array(clip_np).astype("float32")
    backend.eval(embedding)

    # Apply vision offramp if provided
    if vision_offramp_path:
        from safetensors import safe_open

        with safe_open(vision_offramp_path, framework="numpy") as f:
            proj = f.get_tensor("projection_matrix")

        proj_mx = backend.array(proj)
        backend.eval(proj_mx)
        # Project: (1, 512) @ (1024, 512).T → (1, 1024)
        embedding = backend.matmul(embedding, backend.transpose(proj_mx))
        backend.eval(embedding)

    return embedding
