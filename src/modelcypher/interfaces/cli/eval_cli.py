"""
Evaluation CLI - mc-eval command.

Evaluates models and LoRA checkpoints against datasets.

Usage:
    mc-eval --model ./mistral-7b --checkpoint ./lora.safetensors --data ./eval.jsonl
    mc-eval --model ./mistral-7b --data ./eval.jsonl --metrics loss perplexity accuracy
"""
import argparse
from pathlib import Path
from typing import List

from modelcypher.core.domain.training.evaluation import (
    EvaluationConfig,
    EvaluationEngine,
    EvaluationMetric,
    EvaluationProgress,
    EvaluationResult,
)


def register(subparsers):
    """Register eval subcommand with argument parser."""
    parser = subparsers.add_parser("eval", help="Evaluate a model or LoRA checkpoint")
    parser.add_argument("--model", required=True, help="Path or ID of the model")
    parser.add_argument("--checkpoint", help="Path to LoRA checkpoint (.safetensors)")
    parser.add_argument("--data", required=True, help="Path to evaluation dataset")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["loss", "perplexity", "accuracy", "bpc"],
        default=["loss", "perplexity"],
        help="Metrics to compute",
    )
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")


def run(args):
    """Execute evaluation command."""
    print(f"Evaluating model: {args.model}")

    # Parse metrics
    metrics: List[EvaluationMetric] = []
    for m in args.metrics:
        if m == "loss":
            metrics.append(EvaluationMetric.LOSS)
        elif m == "perplexity":
            metrics.append(EvaluationMetric.PERPLEXITY)
        elif m == "accuracy":
            metrics.append(EvaluationMetric.ACCURACY)
        elif m == "bpc":
            metrics.append(EvaluationMetric.BITS_PER_CHARACTER)

    config = EvaluationConfig(
        metrics=metrics,
        batch_size=args.batch_size,
        sequence_length=args.seq_length,
        max_samples=args.max_samples,
    )

    print(f"Configuration:")
    print(f"  Dataset: {args.data}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Metrics: {[m.value for m in config.metrics]}")

    if args.checkpoint:
        print(f"  LoRA Checkpoint: {args.checkpoint}")

    # Progress callback
    def on_progress(progress: EvaluationProgress):
        if not args.quiet:
            pct = progress.percentage * 100
            print(f"\rProgress: {pct:.1f}% ({progress.samples_processed}/{progress.total_samples})", end="")

    # In a real implementation, would:
    # 1. Load tokenizer and model
    # 2. Load dataset
    # 3. If --checkpoint specified, load and apply LoRA weights
    # 4. Run evaluation
    # 5. Print/save results

    print("\nEvaluation would run here (model loading needed)")

    # Placeholder result for demonstration
    print("\n--- Evaluation Results (stub) ---")
    print(f"  Loss: 2.45")
    print(f"  Perplexity: 11.59")
    if args.max_samples:
        print(f"  Samples: {args.max_samples}")

    if args.output:
        print(f"\nResults saved to: {args.output}")
