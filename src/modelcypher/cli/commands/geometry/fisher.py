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

"""Fisher Information CLI commands.

Fisher Information measures which dimensions the model "cares about" most -
the curvature of the loss landscape at each parameter.

For LoRA merging:
    - High Fisher dimensions = model depends on these heavily (protect them)
    - Low Fisher dimensions = model doesn't use these much (room for LoRA)

Commands:
    mc geometry fisher analyze MODEL --prompts FILE
    mc geometry fisher compatibility SOURCE TARGET --prompts FILE
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.composition import get_backend
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.input_validation import validate_file_exists, validate_model_path
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
    """Extract activations from a model at a specified layer."""
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

    # Stack into matrix [n_samples, hidden_dim]
    names = list(activations_dict.keys())
    acts_list = [activations_dict[name] for name in names]
    activations = backend.stack(acts_list, axis=0)
    backend.eval(activations)

    return activations, layer_idx, num_layers, backend


@app.command("analyze")
def fisher_analyze(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    prompts: str = typer.Option(
        ..., "--prompts", help="Path to prompts file (JSON array or newline-separated)"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
) -> None:
    """Analyze Fisher Information for a model's activations.

    Computes the diagonal Fisher Information Matrix, which shows
    which dimensions the model depends on most. High Fisher values
    indicate dimensions critical to model behavior.

    For LoRA insertion: Avoid targeting high-Fisher dimensions to
    prevent catastrophic forgetting.
    """
    context = _context(ctx)

    prompt_list = _load_prompts(prompts, context)
    activations, layer_idx, num_layers, backend = _extract_activations(
        model_path, prompt_list, layer, context
    )

    from modelcypher.core.domain.geometry.fisher_information import (
        compute_empirical_fisher_diagonal,
    )
    from modelcypher.core.support.array_utils import array_to_list

    result = compute_empirical_fisher_diagonal(activations, backend)

    payload = {
        "_schema": "mc.geometry.fisher.v1",
        "model_path": model_path,
        "layer": layer_idx,
        "num_layers": num_layers,
        "n_samples": int(activations.shape[0]),
        "hidden_dim": int(activations.shape[1]),
        "fisher_stats": {
            "trace": result.trace_fim,
            "mean": result.mean_fim,
            "effective_rank": result.effective_rank,
            "condition_number": result.condition_number,
            "n_significant": result.n_significant,
            "significance_threshold": result.significance_threshold,
        },
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("compatibility")
def fisher_compatibility(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Path to source model"),
    target: str = typer.Argument(..., help="Path to target model"),
    prompts: str = typer.Option(
        ..., "--prompts", help="Path to prompts file (JSON array or newline-separated)"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
) -> None:
    """Compute Fisher compatibility between two models.

    Measures how similar the Fisher Information profiles are between
    source and target models. High compatibility means models "care about"
    similar dimensions, suggesting smoother merging/transfer.

    For LoRA: Compare base model Fisher to LoRA-modified model to see
    if the LoRA is targeting appropriate dimensions.
    """
    context = _context(ctx)

    prompt_list = _load_prompts(prompts, context)

    source_acts, source_layer, source_layers, backend = _extract_activations(
        source, prompt_list, layer, context
    )
    target_acts, target_layer, target_layers, backend = _extract_activations(
        target, prompt_list, layer, context
    )

    from modelcypher.core.domain.geometry.fisher_information import (
        fisher_compatibility_score,
    )

    result = fisher_compatibility_score(source_acts, target_acts, backend)

    payload = {
        "_schema": "mc.geometry.fisher.compatibility.v1",
        "source_path": source,
        "target_path": target,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "n_samples": int(source_acts.shape[0]),
        "source_hidden_dim": int(source_acts.shape[1]),
        "target_hidden_dim": int(target_acts.shape[1]),
        "compatibility": {
            "score": result.compatibility_score,
            "cosine_similarity": result.cosine_similarity,
            "correlation": result.correlation,
            "overlap_ratio": result.overlap_ratio,
        },
        "source_fisher": {
            "effective_rank": result.source_fisher.effective_rank,
            "condition_number": result.source_fisher.condition_number,
            "n_significant": result.source_fisher.n_significant,
        },
        "target_fisher": {
            "effective_rank": result.target_fisher.effective_rank,
            "condition_number": result.target_fisher.condition_number,
            "n_significant": result.target_fisher.n_significant,
        },
    }
    write_output(payload, context.output_format, context.pretty)
