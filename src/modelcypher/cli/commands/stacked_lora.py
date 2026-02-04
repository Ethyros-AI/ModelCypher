# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Stacked LoRA CLI commands.

Commands for managing stacked LoRA self-improvement:

    mc stack init         Initialize a new stack
    mc stack status       View stack status
    mc stack train        Train and add adapter to stack
    mc stack merge        Merge adapters in stack
    mc stack improve      Run iterative improvement loop
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True, help="Stacked LoRA self-improvement commands")


def _default_policy():
    """Create a policy with documented default values.

    These values are NOT magic - they're explicit starting points.
    Adjust based on your use case.
    """
    from modelcypher.core.use_cases.self_improve import StackerPolicy

    return StackerPolicy(
        barrier_merge_threshold=0.03,  # Merge when cumulative barrier exceeds 3%
        cka_drift_threshold=0.1,  # Merge when CKA drift exceeds 10%
        max_adapters=5,  # Hard limit on stack depth
        convergence_ratio_threshold=1.0,  # Trigger if adapter more converged than base
        convergence_barrier_multiplier=0.5,  # Halve barrier threshold when converged
    )


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("init")
def stack_init(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to base model"),
    state_path: str = typer.Option(
        None, "--state", "-s", help="Path to save stack state (default: auto-generated)"
    ),
) -> None:
    """Initialize a new LoRA stack for a base model.

    Examples:
        mc stack init /path/to/model
        mc stack init /path/to/model --state ./stack_state.json
    """
    context = _context(ctx)
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_improve import LoRAStacker
    
    try:
        initialize_default_backend()
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        stacker = LoRAStacker(model_path, policy=_default_policy())

        # Determine state path
        if state_path:
            state_file = Path(state_path)
        else:
            state_file = model_path.parent / f"{model_path.name}_stack_state.json"
        
        stacker.save_state(state_file)
        
        payload = {
            "baseModel": str(model_path),
            "statePath": str(state_file),
            "nAdapters": 0,
            "message": "Stack initialized successfully",
        }
        
        if context.output_format == "text":
            lines = [
                "Stack initialized:",
                f"  Base model: {model_path}",
                f"  State file: {state_file}",
            ]
            write_output("\n".join(lines), context.output_format, context.pretty)
            return
        
        write_output(payload, context.output_format, context.pretty)
        
    except Exception as exc:
        error = ErrorDetail(
            code="MC-STACK-001",
            title="Stack initialization failed",
            detail=str(exc),
            hint="Ensure the model path is valid",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("status")
def stack_status(
    ctx: typer.Context,
    state_path: str = typer.Argument(..., help="Path to stack state file"),
) -> None:
    """View status of a LoRA stack.

    Examples:
        mc stack status ./stack_state.json
    """
    context = _context(ctx)
    from modelcypher.core.use_cases.self_improve import LoRAStacker
    
    try:
        state_path = Path(state_path)
        if not state_path.exists():
            raise ValueError(f"State file not found: {state_path}")
        
        # Load state to get base model path
        with open(state_path) as f:
            state_data = json.load(f)
        
        base_model = Path(state_data.get("base_model_path", "."))
        # Policy is loaded from state file
        stacker = LoRAStacker(base_model, policy=_default_policy(), state_path=state_path)
        status = stacker.get_status()
        
        if context.output_format == "text":
            lines = [
                "STACK STATUS",
                f"Base model: {status['base_model']}",
                f"Adapters: {status['n_adapters']}",
                f"Cumulative barrier: {status['cumulative_barrier']:.4f}",
                f"Cumulative CKA drift: {status['cumulative_cka_drift']:.4f}",
                f"Current difficulty: {status['current_difficulty']}",
                f"Should merge: {'YES' if status['should_merge'] else 'no'}",
            ]
            
            # Get adapter paths from stacker state directly
            if stacker.state.adapters:
                lines.append("\nAdapters:")
                for i, adapter in enumerate(stacker.state.adapters, 1):
                    name = str(adapter.path).split('/')[-1]
                    lines.append(
                        f"  {i}. {name} "
                        f"(barrier={adapter.barrier:.4f}, cka={adapter.cka_from_base:.4f})"
                    )
            
            write_output("\n".join(lines), context.output_format, context.pretty)
            return
        
        write_output(status, context.output_format, context.pretty)
        
    except Exception as exc:
        error = ErrorDetail(
            code="MC-STACK-002",
            title="Stack status failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("train")
def stack_train(
    ctx: typer.Context,
    state_path: str = typer.Argument(..., help="Path to stack state file"),
    data_path: str = typer.Option(..., "--data", "-d", help="Path to training data"),
    output_dir: str = typer.Option(..., "--output", "-o", help="Output directory for adapter"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Training epochs"),
    rank: int = typer.Option(8, "--rank", "-r", help="LoRA rank"),
) -> None:
    """Train a LoRA adapter and add to stack.

    Examples:
        mc stack train ./stack_state.json --data ./data.jsonl --output ./adapter1
        mc stack train ./stack_state.json -d ./data.jsonl -o ./adapter1 --epochs 5
    """
    context = _context(ctx)
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_improve import LoRAStacker
    from modelcypher.core.use_cases.lora_training_service import LoRATrainingService
    
    try:
        initialize_default_backend()
        
        state_path = Path(state_path)
        data_path = Path(data_path)
        output_dir = Path(output_dir)
        
        # Load stacker
        with open(state_path) as f:
            state_data = json.load(f)
        
        base_model = Path(state_data.get("base_model_path", "."))
        stacker = LoRAStacker(base_model, policy=_default_policy(), state_path=state_path)

        # Train adapter
        service = LoRATrainingService()
        result = service.train_lora(
            model_path=base_model,
            training_data_path=data_path,
            output_path=output_dir,
            epochs=epochs,
            rank=rank,
        )
        
        if not result.success:
            raise ValueError(f"Training failed: {result.error}")
        
        # Add to stack
        stack_result = stacker.add_adapter(
            adapter_path=result.adapter_path,
            barrier=result.barrier_to_base,
            cka_from_base=result.cka_from_base,
            difficulty_level=stacker.state.current_difficulty,
            training_samples=result.samples_used,
            target_modules=result.target_modules,
        )
        
        # Save updated state
        stacker.save_state(state_path)
        
        payload = {
            "success": True,
            "adapterPath": str(result.adapter_path),
            "trainingLoss": result.final_loss,
            "barrier": result.barrier_to_base,
            "ckaFromBase": result.cka_from_base,
            "cumulativeBarrier": stack_result.cumulative_barrier,
            "shouldMerge": stack_result.should_merge,
            "mergeReason": stack_result.merge_reason,
        }
        
        if context.output_format == "text":
            lines = [
                "TRAINING COMPLETE",
                f"Adapter: {result.adapter_path}",
                f"Loss: {result.final_loss:.4f}",
                f"Barrier: {result.barrier_to_base:.4f}",
                f"CKA: {result.cka_from_base:.4f}",
                f"Stack depth: {stacker.state.n_adapters}",
            ]
            if stack_result.should_merge:
                lines.append(f"\n⚠️  Merge recommended: {stack_result.merge_reason}")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return
        
        write_output(payload, context.output_format, context.pretty)
        
    except Exception as exc:
        error = ErrorDetail(
            code="MC-STACK-003",
            title="Stack training failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("merge")
def stack_merge(
    ctx: typer.Context,
    state_path: str = typer.Argument(..., help="Path to stack state file"),
    output_path: str = typer.Option(..., "--output", "-o", help="Output path for merged adapter"),
) -> None:
    """Merge all adapters in stack into a single adapter.

    Examples:
        mc stack merge ./stack_state.json --output ./merged_adapter
    """
    context = _context(ctx)
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_improve import LoRAStacker
    
    try:
        initialize_default_backend()
        
        state_path = Path(state_path)
        output_path = Path(output_path)
        
        # Load stacker
        with open(state_path) as f:
            state_data = json.load(f)
        
        base_model = Path(state_data.get("base_model_path", "."))
        stacker = LoRAStacker(base_model, policy=_default_policy(), state_path=state_path)

        if stacker.state.n_adapters == 0:
            raise ValueError("No adapters to merge")
        
        # Merge
        result = stacker.merge_stack(output_path)
        
        if not result.success:
            raise ValueError(f"Merge failed: {result.message}")
        
        # Save updated state
        stacker.save_state(state_path)
        
        payload = {
            "success": True,
            "mergedPath": str(result.merged_path),
            "adaptersMerged": result.adapters_merged,
            "preBarrier": result.cumulative_barrier_before,
            "message": result.message,
        }
        
        if context.output_format == "text":
            lines = [
                "MERGE COMPLETE",
                f"Output: {result.merged_path}",
                f"Adapters merged: {result.adapters_merged}",
                f"Pre-merge barrier: {result.cumulative_barrier_before:.4f}",
            ]
            write_output("\n".join(lines), context.output_format, context.pretty)
            return
        
        write_output(payload, context.output_format, context.pretty)
        
    except Exception as exc:
        error = ErrorDetail(
            code="MC-STACK-004",
            title="Stack merge failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("improve")
def stack_improve(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to base model"),
    output_dir: str = typer.Option(..., "--output", "-o", help="Output directory"),
    rounds: int = typer.Option(5, "--rounds", "-n", help="Max improvement rounds"),
    samples: int = typer.Option(100, "--samples", help="Training samples per round"),
) -> None:
    """Run iterative self-improvement loop.

    This command runs the full self-improvement loop:
    1. Scan for capability gaps
    2. Generate training data
    3. Train LoRA adapter
    4. Check geometry, stack or merge
    5. Repeat

    Examples:
        mc stack improve /path/to/model --output ./improvement
        mc stack improve /path/to/model -o ./improvement --rounds 10
    """
    context = _context(ctx)
    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.use_cases.self_improve import LoRAStacker, AutonomousSelfImprover
    from modelcypher.core.use_cases.self_improve.types import Capability
    
    try:
        initialize_default_backend()
        
        model_path = Path(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create stacker
        stacker = LoRAStacker(model_path, policy=_default_policy())

        # Load model
        model, tokenizer = load_model_for_training(str(model_path))
        
        # Define capabilities to test
        # These are example arithmetic capabilities
        capabilities = [
            Capability.from_lists(
                name="arithmetic_addition",
                questions=[
                    "What is 23 + 45?",
                    "Calculate 156 + 289",
                    "What is 1234 + 5678?",
                ],
                answers=["68", "445", "6912"],
            ),
            Capability.from_lists(
                name="word_problems",
                questions=[
                    "If I have 5 apples and get 3 more, how many do I have?",
                    "A store has 20 items. 7 are sold. How many remain?",
                ],
                answers=["8", "13"],
            ),
        ]
        
        # Run improvement
        improver = AutonomousSelfImprover(model, tokenizer)
        result = improver.improve_iterative(
            capabilities=capabilities,
            output_dir=output_dir,
            max_rounds=rounds,
            n_samples_per_round=samples,
            stacker=stacker,
        )
        
        # Save final state
        state_file = output_dir / "stack_state.json"
        stacker.save_state(state_file)
        
        payload = {
            "success": result.get("success", False),
            "roundsCompleted": result.get("rounds_completed", 0),
            "adaptersTrained": result.get("adapters_trained", 0),
            "mergesPerformed": result.get("merges_performed", 0),
            "statePath": str(state_file),
        }
        
        if context.output_format == "text":
            lines = [
                "IMPROVEMENT COMPLETE",
                f"Rounds: {result.get('rounds_completed', 0)}",
                f"Adapters trained: {result.get('adapters_trained', 0)}",
                f"Merges: {result.get('merges_performed', 0)}",
                f"State saved: {state_file}",
            ]
            write_output("\n".join(lines), context.output_format, context.pretty)
            return
        
        write_output(payload, context.output_format, context.pretty)
        
    except Exception as exc:
        error = ErrorDetail(
            code="MC-STACK-005",
            title="Improvement loop failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("profile")
def stack_profile(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model"),
    problems_file: str = typer.Option(..., "--problems", "-p", help="Path to problems file (one per line)"),
    output_file: str = typer.Option(None, "--output", "-o", help="Output JSON file for profiles"),
    layer: int = typer.Option(None, "--layer", "-l", help="Layer index to profile (default: middle layer)"),
) -> None:
    """Profile problems geometrically for curriculum design.

    Measures difficulty using CKA, barrier, curvature, density, and intrinsic dimension.
    No heuristics - all values are raw geometric measurements.

    Examples:
        mc stack profile /path/to/model --problems ./questions.txt
        mc stack profile /path/to/model -p ./questions.txt -o ./profiles.json
    """
    context = _context(ctx)
    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.use_cases.curriculum_profiler import CurriculumProfiler
    
    try:
        initialize_default_backend()
        
        model_path = Path(model_path)
        problems_file = Path(problems_file)
        
        # Load problems
        with open(problems_file) as f:
            problems = [line.strip() for line in f if line.strip()]
        
        if not problems:
            raise ValueError("No problems found in file")
        
        # Load model
        model, tokenizer = load_model_for_training(str(model_path))
        
        # Create profiler
        profiler = CurriculumProfiler(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer,
        )
        
        # Profile problems
        profiles = profiler.profile_problems(
            problems=problems,
            progress_callback=lambda i, n: print(f"\rProfiled {i}/{n}", end="") if not context.quiet else None,
        )
        
        # Save if output specified
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(json.dumps(profiles.as_dict(), indent=2))
        
        # Build summary
        if profiles.profiles:
            cka_values = [p.cka_similarity for p in profiles.profiles]
            barrier_values = [p.barrier_height for p in profiles.profiles]
            goldilocks_zone = profiles.filter_by_goldilocks()
            
            payload = {
                "totalProblems": len(profiles.profiles),
                "successfulProfiles": len(profiles.profiles),
                "cka": {
                    "mean": sum(cka_values) / len(cka_values),
                    "min": min(cka_values),
                    "max": max(cka_values),
                },
                "barrier": {
                    "mean": sum(barrier_values) / len(barrier_values),
                    "min": min(barrier_values),
                    "max": max(barrier_values),
                },
                "goldilocksCount": len(goldilocks_zone),
                "outputFile": str(output_file) if output_file else None,
            }
        else:
            payload = {"totalProblems": 0, "error": "No profiles generated"}
        
        if context.output_format == "text":
            if profiles.profiles:
                lines = [
                    "CURRICULUM PROFILE",
                    f"Problems: {len(profiles.profiles)}",
                    f"Layer: {profiles.profiles[0].layer_idx}",
                    "",
                    "CKA Similarity:",
                    f"  Mean: {payload['cka']['mean']:.4f}",
                    f"  Range: [{payload['cka']['min']:.4f}, {payload['cka']['max']:.4f}]",
                    "",
                    "Barrier Height:",
                    f"  Mean: {payload['barrier']['mean']:.4f}",
                    f"  Range: [{payload['barrier']['min']:.4f}, {payload['barrier']['max']:.4f}]",
                    "",
                    f"Goldilocks zone (CKA 0.85-0.95): {len(goldilocks_zone)} problems",
                ]
                if output_file:
                    lines.append(f"\nSaved to: {output_file}")
            else:
                lines = ["No profiles generated"]
            print()  # Clear progress line
            write_output("\n".join(lines), context.output_format, context.pretty)
            return
        
        write_output(payload, context.output_format, context.pretty)
        
    except Exception as exc:
        error = ErrorDetail(
            code="MC-STACK-006",
            title="Profile failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("select")
def stack_select(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model"),
    problems_file: str = typer.Option(..., "--problems", "-p", help="Path to problems file (one per line)"),
    output_file: str = typer.Option(..., "--output", "-o", help="Output file for selected curriculum"),
    n_samples: int = typer.Option(100, "--n", "-n", help="Number of samples to select"),
    strategy: str = typer.Option("balanced", "--strategy", "-s", help="Selection strategy: balanced, hardest, goldilocks, highway_first"),
    layer: int = typer.Option(None, "--layer", "-l", help="Layer index to profile (default: middle layer)"),
) -> None:
    """Select training curriculum based on geometric difficulty.

    Profiles problems and selects optimal training set using composite difficulty score.

    Strategies:
      - balanced: Mix of easy (20%), medium (60%), hard (20%)
      - hardest: Focus on highest difficulty problems
      - goldilocks: Moderate difficulty only (score 0.3-0.7)
      - highway_first: Order by intrinsic dimension (low ID first).
        Problems that activate geometric highways are trained first.

    Examples:
        mc stack select /path/to/model -p ./all_problems.txt -o ./curriculum.txt -n 50
        mc stack select /path/to/model -p ./problems.txt -o ./hard.txt -s hardest -n 20
    """
    context = _context(ctx)
    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.use_cases.curriculum_profiler import CurriculumProfiler
    
    try:
        initialize_default_backend()
        
        model_path = Path(model_path)
        problems_file = Path(problems_file)
        output_path = Path(output_file)
        
        # Load problems
        with open(problems_file) as f:
            problems = [line.strip() for line in f if line.strip()]
        
        if not problems:
            raise ValueError("No problems found in file")
        
        if len(problems) < n_samples:
            n_samples = len(problems)
        
        # Load model
        model, tokenizer = load_model_for_training(str(model_path))
        
        # Create profiler
        profiler = CurriculumProfiler(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer,
        )
        
        # Profile problems
        profiles = profiler.profile_problems(
            problems=problems,
            progress_callback=lambda i, n: print(f"\rProfiled {i}/{n}", end="") if not context.quiet else None,
        )
        
        # Select curriculum
        selected = profiles.select_curriculum(n_samples=n_samples, strategy=strategy)
        
        # Write selected prompts
        with open(output_path, "w") as f:
            for profile in selected:
                f.write(profile.prompt + "\n")
        
        # Build summary
        difficulty_scores = [p.difficulty_score for p in selected]
        payload = {
            "strategy": strategy,
            "inputProblems": len(problems),
            "selectedProblems": len(selected),
            "difficultyRange": {
                "min": min(difficulty_scores),
                "max": max(difficulty_scores),
                "mean": sum(difficulty_scores) / len(difficulty_scores),
            },
            "outputFile": str(output_path),
        }
        
        if context.output_format == "text":
            lines = [
                "CURRICULUM SELECTION",
                f"Strategy: {strategy}",
                f"Input: {len(problems)} problems",
                f"Selected: {len(selected)} problems",
                "",
                "Difficulty Score:",
                f"  Mean: {payload['difficultyRange']['mean']:.3f}",
                f"  Range: [{payload['difficultyRange']['min']:.3f}, {payload['difficultyRange']['max']:.3f}]",
                "",
                f"Saved to: {output_path}",
            ]
            print()
            write_output("\n".join(lines), context.output_format, context.pretty)
            return
        
        write_output(payload, context.output_format, context.pretty)
        
    except Exception as exc:
        error = ErrorDetail(
            code="MC-STACK-007",
            title="Selection failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


__all__ = ["app"]
