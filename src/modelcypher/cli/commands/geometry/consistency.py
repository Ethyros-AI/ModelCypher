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

"""Representation Consistency CLI commands.

Representation consistency measures how stable a model's internal
representations are for semantically related inputs.

For LoRA merging:
    - Pre-LoRA: Establish baseline semantic consistency
    - Post-LoRA: Verify LoRA doesn't break semantic relationships

For safety:
    - Detect representation collapse (everything maps to same vector)
    - Detect semantic confusion (related inputs treated as contradictory)

Commands:
    mc geometry consistency analyze MODEL --original FILE --related FILE [--contradictory FILE]
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.composition import get_backend
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.validation import validate_file_exists, validate_model_path
from modelcypher.cli.commands.geometry.helpers import (
    extract_anchor_activations,
    resolve_model_backbone,
)

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _load_prompts(prompts_file: str, context: CLIContext) -> list[str]:
    """Load prompts from file (JSON array or newline-separated)."""
    prompts_path = validate_file_exists(
        prompts_file,
        description="Prompts file",
        context=context,
    )
    content = prompts_path.read_text(encoding="utf-8")
    try:
        prompt_data = json.loads(content)
        if not isinstance(prompt_data, list) or not all(
            isinstance(prompt, str) for prompt in prompt_data
        ):
            raise typer.BadParameter("Prompts file must contain a JSON array of strings")
        prompts = [prompt.strip() for prompt in prompt_data if prompt.strip()]
    except json.JSONDecodeError:
        prompts = [line.strip() for line in content.splitlines() if line.strip()]

    if not prompts:
        raise typer.BadParameter("Prompts file is empty")

    return prompts


def _extract_activations(
    model_path: str,
    prompts: list[str],
    layer: int | None,
    context: CLIContext,
):
    """Extract activations from a model for given prompts."""
    from modelcypher.adapters.model_loader import load_model_for_training

    validate_model_path(model_path, context=context)
    model, tokenizer = load_model_for_training(model_path)
    backbone = resolve_model_backbone(model, getattr(model, "model_type", None))
    if backbone is None:
        raise typer.BadParameter("Failed to resolve model backbone.")
    embed_tokens, layers, norm = backbone
    num_layers = len(layers)

    if layer is None:
        layer_idx = num_layers // 2
    else:
        if layer < 0 or layer >= num_layers:
            raise typer.BadParameter(
                f"Layer {layer} out of range for model with {num_layers} layers."
            )
        layer_idx = layer

    class PromptAnchor:
        def __init__(self, name: str, prompt: str) -> None:
            self.name = name
            self.prompt = prompt

    anchors = [
        PromptAnchor(f"prompt_{idx}", prompt)
        for idx, prompt in enumerate(prompts)
    ]

    backend = get_backend()
    activations_dict = extract_anchor_activations(
        anchors=anchors,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        layers=layers,
        norm=norm,
        target_layer=layer_idx,
        backend=backend,
        prompt_attr="prompt",
        name_attr="name",
    )

    if not activations_dict:
        raise typer.BadParameter("Activation extraction returned no samples.")

    # Return as list of activation vectors
    names = list(activations_dict.keys())
    acts_list = [activations_dict[name] for name in names]

    return acts_list, layer_idx, num_layers, backend


@app.command("analyze")
def consistency_analyze(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    original: str = typer.Option(
        ..., "--original", help="Path to original prompts file (single prompt or first used)"
    ),
    related: str = typer.Option(
        ..., "--related", help="Path to semantically related prompts file"
    ),
    contradictory: str | None = typer.Option(
        None, "--contradictory", help="Path to contradictory/unrelated prompts file"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
) -> None:
    """Analyze representation consistency for a model.

    Compares how the model represents:
    - Original input(s) vs semantically related inputs (should be similar)
    - Original input(s) vs contradictory inputs (should be different)

    High implication_consistency = related inputs cluster together (good).
    High contradiction_distance = unrelated inputs are separated (good).
    High separation_score = model distinguishes related from unrelated (good).

    For LoRA validation: Run before and after LoRA insertion to verify
    the LoRA doesn't break semantic consistency.
    """
    context = _context(ctx)

    # Load prompts
    original_prompts = _load_prompts(original, context)
    related_prompts = _load_prompts(related, context)
    contradictory_prompts = None
    if contradictory:
        contradictory_prompts = _load_prompts(contradictory, context)

    # Extract activations for all prompts
    all_prompts = original_prompts + related_prompts
    if contradictory_prompts:
        all_prompts += contradictory_prompts

    all_acts, layer_idx, num_layers, backend = _extract_activations(
        model_path, all_prompts, layer, context
    )

    # Split activations back into groups
    n_original = len(original_prompts)
    n_related = len(related_prompts)

    original_acts = all_acts[:n_original]
    related_acts = all_acts[n_original:n_original + n_related]
    contra_acts = all_acts[n_original + n_related:] if contradictory_prompts else None

    # Use first original as reference (or could average them)
    reference_act = original_acts[0]

    from modelcypher.core.domain.geometry.representation_consistency import (
        RepresentationConsistencyAnalyzer,
    )

    analyzer = RepresentationConsistencyAnalyzer(backend)
    result = analyzer.compute(reference_act, related_acts, contra_acts)

    payload = {
        "_schema": "mc.geometry.representation_consistency.v1",
        "model_path": model_path,
        "layer": layer_idx,
        "num_layers": num_layers,
        "n_original": n_original,
        "n_related": len(related_prompts),
        "n_contradictory": len(contradictory_prompts) if contradictory_prompts else 0,
        "consistency": {
            "implication_consistency": result.implication_consistency,
            "contradiction_distance": result.contradiction_distance,
            "consistency_score": result.consistency_score,
            "separation_score": result.separation_score,
        },
        "counts": {
            "n_implications": result.n_implications,
            "n_contradictions": result.n_contradictions,
        },
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def consistency_compare(
    ctx: typer.Context,
    model_a: str = typer.Argument(..., help="Path to first model (e.g., base model)"),
    model_b: str = typer.Argument(..., help="Path to second model (e.g., base+LoRA)"),
    original: str = typer.Option(
        ..., "--original", help="Path to original prompts file"
    ),
    related: str = typer.Option(
        ..., "--related", help="Path to semantically related prompts file"
    ),
    contradictory: str | None = typer.Option(
        None, "--contradictory", help="Path to contradictory/unrelated prompts file"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
) -> None:
    """Compare representation consistency between two models.

    Useful for LoRA validation: Compare base model vs base+LoRA to see
    if the LoRA degrades semantic consistency.

    A well-behaved LoRA should maintain or improve consistency scores.
    Degraded consistency suggests the LoRA may be causing representation
    collapse or semantic confusion.
    """
    context = _context(ctx)

    # Load prompts
    original_prompts = _load_prompts(original, context)
    related_prompts = _load_prompts(related, context)
    contradictory_prompts = None
    if contradictory:
        contradictory_prompts = _load_prompts(contradictory, context)

    all_prompts = original_prompts + related_prompts
    if contradictory_prompts:
        all_prompts += contradictory_prompts

    # Extract from both models
    acts_a, layer_a, num_layers_a, backend = _extract_activations(
        model_a, all_prompts, layer, context
    )
    acts_b, layer_b, num_layers_b, _ = _extract_activations(
        model_b, all_prompts, layer, context
    )

    # Split activations
    n_original = len(original_prompts)
    n_related = len(related_prompts)

    def split_acts(acts):
        original = acts[:n_original]
        related = acts[n_original:n_original + n_related]
        contra = acts[n_original + n_related:] if contradictory_prompts else None
        return original, related, contra

    orig_a, rel_a, contra_a = split_acts(acts_a)
    orig_b, rel_b, contra_b = split_acts(acts_b)

    from modelcypher.core.domain.geometry.representation_consistency import (
        RepresentationConsistencyAnalyzer,
    )

    analyzer = RepresentationConsistencyAnalyzer(backend)

    result_a = analyzer.compute(orig_a[0], rel_a, contra_a)
    result_b = analyzer.compute(orig_b[0], rel_b, contra_b)

    # Compute deltas
    delta_consistency = result_b.implication_consistency - result_a.implication_consistency
    delta_distance = result_b.contradiction_distance - result_a.contradiction_distance
    delta_separation = result_b.separation_score - result_a.separation_score

    payload = {
        "_schema": "mc.geometry.representation_consistency.compare.v1",
        "model_a": model_a,
        "model_b": model_b,
        "layer": layer_a,
        "model_a_results": {
            "implication_consistency": result_a.implication_consistency,
            "contradiction_distance": result_a.contradiction_distance,
            "consistency_score": result_a.consistency_score,
            "separation_score": result_a.separation_score,
        },
        "model_b_results": {
            "implication_consistency": result_b.implication_consistency,
            "contradiction_distance": result_b.contradiction_distance,
            "consistency_score": result_b.consistency_score,
            "separation_score": result_b.separation_score,
        },
        "delta": {
            "implication_consistency": delta_consistency,
            "contradiction_distance": delta_distance,
            "separation_score": delta_separation,
            "consistency_preserved": delta_consistency >= -0.1,  # Loose threshold
        },
    }
    write_output(payload, context.output_format, context.pretty)
