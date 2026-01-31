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

"""CLI commands for Geometric Self-Study Sandbox.

Commands:
    mc sandbox explore: Interactive REPL with geometric feedback
    mc sandbox compare: Side-by-side approach comparison
    mc sandbox study: Automated self-study curriculum

The sandbox provides models with "eyes on themselves" - the ability to
see and learn from their own geometric signatures during reasoning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True, help="Geometric self-study sandbox")


def _load_model_with_adapter(model_path: Path, adapter_path: Path | None = None):
    """Load model with optional LoRA adapter.

    Args:
        model_path: Path to base model
        adapter_path: Optional path to LoRA adapter

    Returns:
        Tuple of (model, tokenizer)
    """
    if adapter_path:
        from modelcypher.core.domain.training.self_reflection import (
            load_self_reflection_adapters,
        )
        return load_self_reflection_adapters(str(model_path), str(adapter_path))
    else:
        from mlx_lm import load
        return load(str(model_path))


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("explore")
def sandbox_explore(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-t", help="Maximum tokens per response"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (geometry only)"),
) -> None:
    """Interactive REPL with geometric feedback.

    Enter prompts and see the geometric signature of each response.
    The model learns to correlate geometry with response quality.

    Examples:
        mc sandbox explore --model /path/to/model

        # In the REPL:
        > A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?
        Response: The ball costs $0.10
        === GEOMETRIC FEEDBACK ===
        comp/phi: 0.618 (UNDER - narrow processing)
        peak_layer: 15/16 (LATE)
        interpretation: Processing was shallow. Consider explicit reasoning.
        ===========================
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3030",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.core.domain.sandbox.geometric_sandbox import create_sandbox_from_path

        sandbox = create_sandbox_from_path(model_path, max_tokens=max_tokens)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3031",
            title="Failed to load model",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # For non-interactive mode (piped input or AI mode), just process stdin
    if context.ai_mode or not sys.stdin.isatty():
        lines = sys.stdin.read().strip().split("\n")
        results = []
        for line in lines:
            if not line.strip():
                continue
            result = sandbox.attempt(line.strip())
            results.append({
                "prompt": result.prompt,
                "response": result.response,
                "comp_phi": result.comp_phi,
                "is_aligned": result.is_aligned,
                "feedback": result.feedback_text,
            })
        write_output(results, context.output_format, context.pretty)
        return

    # Interactive REPL
    print("Geometric Self-Study Sandbox")
    print(f"Model: {model_path}")
    print(f"Max tokens: {max_tokens}")
    print("Type 'quit' or 'exit' to leave. Type 'help' for commands.")
    print("-" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting sandbox.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit", "q"):
            print("Exiting sandbox.")
            break

        if prompt.lower() == "help":
            print("""
Commands:
    quit, exit, q    - Exit the sandbox
    help             - Show this help
    compare <text>   - Compare approaches (use | to separate approaches)
    reflect          - Reflect on the last result

Otherwise, enter any prompt to generate and see geometry.
""")
            continue

        if prompt.lower().startswith("compare "):
            # Parse comparison: "compare approach1 | approach2 | approach3"
            rest = prompt[8:].strip()
            parts = [p.strip() for p in rest.split("|")]
            if len(parts) < 2:
                print("Usage: compare approach1 | approach2")
                continue

            # Use the first part as the prompt, rest as approaches
            base_prompt = parts[0]
            approaches = parts[1:]

            comparison = sandbox.compare(base_prompt, approaches, max_tokens=max_tokens)
            print(comparison.comparison_text)
            continue

        # Regular prompt -> attempt
        result = sandbox.attempt(prompt)

        if quiet:
            print(result.feedback_text)
        else:
            print(f"\nResponse: {result.response}")
            print()
            print(result.feedback_text)

        # Store for potential reflection
        _last_result = result


@app.command("compare")
def sandbox_compare(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="The prompt to compare approaches on"),
    approaches: list[str] = typer.Option(
        ..., "--approach", "-a", help="Approach prefixes to compare (use multiple -a flags)"
    ),
    max_tokens: int = typer.Option(80, "--max-tokens", "-t", help="Maximum tokens per approach"),
) -> None:
    """Compare multiple reasoning approaches geometrically.

    For each approach prefix, generates a continuation and measures geometry.
    Shows side-by-side comparison with best approach highlighted.

    Examples:
        mc sandbox compare --model /path/to/model \\
            --prompt "A bat and ball cost \\$1.10. The bat costs \\$1 more. Ball cost?" \\
            --approach "The ball costs" \\
            --approach "Let me think step by step"

        mc sandbox compare --model /path/to/model \\
            --prompt "What is 15% of 80?" \\
            --approach "80 times" \\
            --approach "I need to calculate"
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3030",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if len(approaches) < 2:
        error = ErrorDetail(
            code="MC-3032",
            title="Insufficient approaches",
            detail="At least 2 approaches are required for comparison",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.core.domain.sandbox.geometric_sandbox import create_sandbox_from_path

        sandbox = create_sandbox_from_path(model_path, max_tokens=max_tokens)
        comparison = sandbox.compare(prompt, list(approaches), max_tokens=max_tokens)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3033",
            title="Comparison failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if context.output_format == "text":
        print(comparison.comparison_text)
    else:
        payload = {
            "_schema": "mc.sandbox.comparison.v1",
            "prompt": comparison.prompt,
            "best_approach": comparison.best_approach,
            "approaches": [
                {
                    "name": name,
                    "response": result.response,
                    "comp_phi": result.comp_phi,
                    "is_aligned": result.is_aligned,
                    "peak_layer": result.feedback.peak_layer,
                    "n_layers": result.feedback.n_layers,
                    "entropy_pattern": result.feedback.entropy_pattern.value,
                }
                for name, result in comparison.approaches
            ],
        }
        write_output(payload, context.output_format, context.pretty)


@app.command("study")
def sandbox_study(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    adapter: str = typer.Option(None, "--adapter", "-a", help="Path to LoRA adapter"),
    curriculum: str = typer.Option(
        "geometric_self_study", "--curriculum", "-c", help="Curriculum name or path"
    ),
    level: int = typer.Option(
        None, "--level", "-l", help="Curriculum level (1-4). Default: all levels"
    ),
    limit: int = typer.Option(
        None, "--limit", "-n", help="Maximum examples to run. Default: all"
    ),
    max_tokens: int = typer.Option(100, "--max-tokens", "-t", help="Maximum tokens per response"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full responses"),
) -> None:
    """Run automated self-study curriculum.

    Levels:
        1. Observation - See geometry, no action required
        2. Prediction - Predict geometry before generating
        3. Selection - Choose approach based on predicted geometry
        4. Correction - Detect and fix geometric anomalies

    Examples:
        # Run full curriculum
        mc sandbox study --model /path/to/model

        # Run with trained adapter
        mc sandbox study --model /path/to/model --adapter /path/to/adapter

        # Run only level 1 (observation)
        mc sandbox study --model /path/to/model --level 1

        # Run with custom curriculum
        mc sandbox study --model /path/to/model --curriculum /path/to/curriculum/
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3030",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load curriculum
    try:
        from modelcypher.core.domain.sandbox.curriculum import (
            Curriculum,
            CurriculumLevel,
            get_builtin_curriculum,
            load_curriculum_from_directory,
        )

        curriculum_path = Path(curriculum)
        if curriculum_path.exists():
            loaded_curriculum = load_curriculum_from_directory(curriculum_path)
        else:
            loaded_curriculum = get_builtin_curriculum(curriculum)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3034",
            title="Failed to load curriculum",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load sandbox
    try:
        from modelcypher.core.domain.sandbox.geometric_sandbox import GeometricSandbox

        adapter_path = Path(adapter) if adapter else None
        if adapter_path and not adapter_path.exists():
            error = ErrorDetail(
                code="MC-3037",
                title="Adapter not found",
                detail=f"Adapter path does not exist: {adapter}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        loaded_model, tokenizer = _load_model_with_adapter(model_path, adapter_path)
        sandbox = GeometricSandbox(loaded_model, tokenizer, max_tokens=max_tokens)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3031",
            title="Failed to load model",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Filter by level if specified
    if level:
        try:
            target_level = CurriculumLevel(level)
            examples = loaded_curriculum.get_level(target_level)
        except ValueError:
            error = ErrorDetail(
                code="MC-3035",
                title="Invalid level",
                detail=f"Level must be 1-4, got: {level}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
    else:
        examples = loaded_curriculum.examples

    # Apply limit
    if limit:
        examples = examples[:limit]

    if not examples:
        error = ErrorDetail(
            code="MC-3036",
            title="No examples found",
            detail=f"No examples found in curriculum '{curriculum}' for specified criteria",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Run study session
    results = []
    correct_count = 0
    aligned_count = 0

    if context.output_format == "text" and not context.ai_mode:
        print(f"Self-Study Session: {loaded_curriculum.name}")
        print(f"Examples: {len(examples)}")
        print("-" * 60)

    for i, example in enumerate(examples, 1):
        study_result = sandbox.study_example(
            example.prompt,
            expected_answer=example.expected_answer,
        )

        # Track metrics
        attempt1 = study_result["attempt1"]
        is_correct = study_result.get("is_correct")
        is_aligned = attempt1["is_aligned"]

        if is_correct:
            correct_count += 1
        if is_aligned:
            aligned_count += 1

        result_record = {
            "example_idx": i,
            "level": int(example.level),
            "prompt": example.prompt[:50] + "..." if len(example.prompt) > 50 else example.prompt,
            "comp_phi": attempt1["comp_phi"],
            "is_aligned": is_aligned,
            "is_correct": is_correct,
            "expected_answer": example.expected_answer,
        }

        if "attempt2" in study_result:
            result_record["attempt2_comp_phi"] = study_result["attempt2"]["comp_phi"]
            result_record["geometry_improved"] = study_result.get("geometry_improved", False)
            result_record["correctness_improved"] = study_result.get("correctness_improved", False)

        results.append(result_record)

        # Print progress for text output
        if context.output_format == "text" and not context.ai_mode:
            status = "PASS" if is_correct else "FAIL" if is_correct is False else "N/A"
            aligned_marker = "phi" if is_aligned else "   "
            comp_phi = attempt1["comp_phi"]
            print(f"[{i:3d}] {aligned_marker} {status:4s} phi={comp_phi:.3f} | {result_record['prompt']}")

            if verbose:
                print(f"      Response: {study_result['attempt1']['response'][:60]}...")
                if "attempt2" in study_result:
                    print(f"      Retry: phi={study_result['attempt2']['comp_phi']:.3f}")

    # Summary
    total = len(examples)
    accuracy = correct_count / total if total > 0 else 0
    alignment_rate = aligned_count / total if total > 0 else 0

    summary = {
        "curriculum": loaded_curriculum.name,
        "total_examples": total,
        "correct_count": correct_count,
        "aligned_count": aligned_count,
        "accuracy": accuracy,
        "alignment_rate": alignment_rate,
        "level_filter": level,
    }

    if context.output_format == "text" and not context.ai_mode:
        print("-" * 60)
        print(f"Accuracy: {correct_count}/{total} ({accuracy:.1%})")
        print(f"Alignment Rate: {aligned_count}/{total} ({alignment_rate:.1%})")
    else:
        payload = {
            "_schema": "mc.sandbox.study.v1",
            "summary": summary,
            "results": results,
        }
        write_output(payload, context.output_format, context.pretty)


@app.command("attempt")
def sandbox_attempt(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    adapter: str = typer.Option(None, "--adapter", "-a", help="Path to LoRA adapter"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="The prompt to attempt"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-t", help="Maximum tokens"),
) -> None:
    """Generate a single response and show geometric feedback.

    A non-interactive version of the explore command for scripting.

    Examples:
        mc sandbox attempt --model /path/to/model --prompt "What is 2+2?"
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3030",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.core.domain.sandbox.geometric_sandbox import GeometricSandbox

        adapter_path = Path(adapter) if adapter else None
        if adapter_path and not adapter_path.exists():
            error = ErrorDetail(
                code="MC-3037",
                title="Adapter not found",
                detail=f"Adapter path does not exist: {adapter}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        loaded_model, tokenizer = _load_model_with_adapter(model_path, adapter_path)
        sandbox = GeometricSandbox(loaded_model, tokenizer, max_tokens=max_tokens)
        result = sandbox.attempt(prompt)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3033",
            title="Attempt failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if context.output_format == "text":
        print(f"Response: {result.response}")
        print()
        print(result.feedback_text)
    else:
        payload = {
            "_schema": "mc.sandbox.attempt.v1",
            "prompt": result.prompt,
            "response": result.response,
            "comp_phi": result.comp_phi,
            "is_aligned": result.is_aligned,
            "peak_layer": result.feedback.peak_layer,
            "n_layers": result.feedback.n_layers,
            "entropy_pattern": result.feedback.entropy_pattern.value,
            "interpretation": result.feedback.interpretation,
            "raw_metrics": result.raw_metrics,
        }
        write_output(payload, context.output_format, context.pretty)
