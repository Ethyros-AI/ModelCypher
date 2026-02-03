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

"""LoRA Safety CLI commands.

Provides safety analysis for LoRA training and deployment:

1. Fisher-guided module targeting (exp15: r=-0.864)
   mc geometry lora-safety recommend MODEL --prompts FILE

2. Mode connectivity barrier check (exp16: r=0.989)
   mc geometry lora-safety check-barrier BASE TARGET --prompts FILE

3. Goldilocks quality scoring for curriculum (exp17: r=-0.955)
   mc geometry lora-safety score-curriculum MODEL --problems FILE

References:
    - exp15_fisher_lora_validation: r=-0.864 (Fisher-perplexity)
    - exp16_mode_connectivity_lora: r=0.989 (Barrier-steps)
    - exp17_soar_curriculum: r=-0.955 (Goldilocks-perplexity)
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.validation import validate_file_exists, validate_model_path

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


def _load_problems(problems_file: str, context: CLIContext) -> list[dict]:
    """Load problems from JSONL file."""
    problems_path = validate_file_exists(
        problems_file,
        description="Problems file",
        context=context,
    )

    problems = []
    with open(problems_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    problems.append(json.loads(line))
                except json.JSONDecodeError:
                    # Treat as plain text prompt
                    problems.append({"prompt": line})

    if not problems:
        raise typer.BadParameter("Problems file is empty")

    return problems


@app.command("recommend")
def lora_safety_recommend(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    prompts: str = typer.Option(
        ..., "--prompts", help="Path to prompts file (JSON array or newline-separated)"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
    top_k: int = typer.Option(
        4, "--top-k", help="Number of module recommendations to return"
    ),
) -> None:
    """Get Fisher-guided module recommendations for LoRA targeting.

    Based on exp15 validation (r=-0.864): Target LOW-Fisher modules
    for better LoRA adaptation. Higher Fisher = more important to
    base model = worse LoRA target.

    Example:
        mc geometry lora-safety recommend /path/to/model --prompts prompts.json

    Output:
        Ranked list of modules with Fisher scores and recommendations:
        EXCELLENT (< 0.0004), GOOD (< 0.0005), ACCEPTABLE (< 0.0007), AVOID (>= 0.0007)
    """
    context = _context(ctx)
    validate_model_path(model_path, context=context)

    prompt_list = _load_prompts(prompts, context)

    from modelcypher.cli.composition import get_lora_safety_service

    service = get_lora_safety_service()
    result = service.recommend_target_modules(
        model_path=model_path,
        prompts=prompt_list,
        layer_idx=layer,
        top_k=top_k,
    )

    payload = {
        "_schema": "mc.geometry.lora_safety.recommend.v1",
        "model_path": result.model_path,
        "layer": result.layer,
        "n_samples": result.n_samples,
        "recommendations": [
            {
                "module": r.module,
                "fisher_score": r.fisher_score,
                "recommendation": r.recommendation,
            }
            for r in result.recommendations
        ],
        "guidance": result.guidance,
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("check-barrier")
def lora_safety_check_barrier(
    ctx: typer.Context,
    base: str = typer.Argument(..., help="Path to base model"),
    target: str = typer.Argument(..., help="Path to LoRA weights or merged model"),
    prompts: str = typer.Option(
        ..., "--prompts", help="Path to prompts file (JSON array or newline-separated)"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
) -> None:
    """Check mode connectivity barrier for LoRA safety.

    Based on exp16 validation (r=0.989): Barrier predicts how far LoRA
    pushes model from base. Use as safety gate before deployment.

    Safety levels:
        SAFE (barrier < 0.01): LoRA stays in-basin
        CAUTION (0.01 <= barrier < 0.03): Verify downstream performance
        WARNING (barrier >= 0.03): LoRA may fight base model

    Example:
        mc geometry lora-safety check-barrier /path/to/base /path/to/lora --prompts prompts.json
    """
    context = _context(ctx)
    validate_model_path(base, context=context)
    validate_model_path(target, context=context)

    prompt_list = _load_prompts(prompts, context)

    from modelcypher.cli.composition import get_lora_safety_service

    service = get_lora_safety_service()
    result = service.check_barrier_safety(
        base_path=base,
        target_path=target,
        prompts=prompt_list,
        layer_idx=layer,
    )

    payload = {
        "_schema": "mc.geometry.lora_safety.barrier.v1",
        "base_path": result.base_path,
        "target_path": result.target_path,
        "layer": result.layer,
        "barrier": {
            "height": result.barrier_height,
            "normalized": result.barrier_normalized,
            "safety_level": result.safety_level,
            "cka_at_target": result.cka_at_target,
        },
        "thresholds": {
            "safe": "< 0.01",
            "caution": "0.01 - 0.03",
            "warning": "> 0.03",
        },
        "recommendation": result.recommendation,
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("score-curriculum")
def lora_safety_score_curriculum(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    problems: str = typer.Option(
        ..., "--problems", help="Path to problems JSONL file"
    ),
    reference: str | None = typer.Option(
        None, "--reference", help="Path to reference prompts file (defaults to simple arithmetic)"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
    top_k: int = typer.Option(
        10, "--top-k", help="Number of top problems to return"
    ),
) -> None:
    """Score training problems using Goldilocks quality metric.

    Based on exp17 validation (r=-0.955): Moderate challenge = best learning.
    Problems with CKA ~0.9 and barrier 0.02-0.10 are optimal.

    Quality levels:
        high: quality_score >= 0.7 (best for learning)
        medium: 0.4 <= quality_score < 0.7
        low: quality_score < 0.4 (avoid for training)

    Example:
        mc geometry lora-safety score-curriculum /path/to/model --problems train.jsonl
    """
    context = _context(ctx)
    validate_model_path(model_path, context=context)

    problem_list = _load_problems(problems, context)

    reference_prompts = None
    if reference:
        reference_prompts = _load_prompts(reference, context)

    from modelcypher.cli.composition import get_lora_safety_service

    service = get_lora_safety_service()
    result = service.score_curriculum(
        model_path=model_path,
        problems=problem_list,
        reference_prompts=reference_prompts,
        layer_idx=layer,
        top_k=top_k,
    )

    payload = {
        "_schema": "mc.geometry.lora_safety.curriculum.v1",
        "model_path": result.model_path,
        "n_problems": result.n_problems,
        "quality_distribution": result.quality_distribution,
        "top_problems": result.top_problems,
        "guidance": result.guidance,
    }
    write_output(payload, context.output_format, context.pretty)
