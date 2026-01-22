#!/usr/bin/env python3
"""Large-scale knowledge stuffing pipeline for LFM2.

Iteratively merges knowledge from multiple large specialist and generalist
models into the compact LFM2-350M target. Uses density-guided null-space
projection to add knowledge where the target has gaps.

Pipeline Strategy:
1. Start with largest generalist (captures broadest knowledge)
2. Add specialist models for domain-specific depth
3. Each merge builds on the previous stuffed version

Usage:
    poetry run python scripts/knowledge_stuffing_pipeline.py --target LFM2-350M
    poetry run python scripts/knowledge_stuffing_pipeline.py --resume lfm2-stuffed-v3
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Model paths
MODEL_BASE = Path("/path/to/models/mlx-community")
OUTPUT_BASE = Path("/path/to/models/merged")


@dataclass
class SourceModel:
    """Source model specification for knowledge stuffing."""

    name: str
    path: str
    category: str  # generalist, code, math, medical, legal, reasoning
    priority: int  # Lower = merge first
    size_b: float  # Approximate size in billions


# Define merge sequence - prioritized by knowledge breadth and size
MERGE_SEQUENCE: list[SourceModel] = [
    # Tier 1: Large Generalists (broadest knowledge base)
    SourceModel("Qwen2.5-72B", "Qwen2.5-72B-Instruct-4bit", "generalist", 1, 72.0),
    SourceModel("Llama-3.3-70B", "Llama-3.3-70B-Instruct-4bit", "generalist", 2, 70.0),
    # SourceModel("GPT-OSS-120B", "gpt-oss-120b-MLX-8bit", "generalist", 3, 120.0),  # May OOM

    # Tier 2: Medium Generalists
    SourceModel("Qwen3-8B", "Qwen3-8B-4bit", "generalist", 10, 8.0),
    SourceModel("DeepSeek-R1-8B", "DeepSeek-R1-0528-Qwen3-8B-MLX-4bit", "reasoning", 11, 8.0),

    # Tier 3: Code Specialists
    SourceModel("Qwen2.5-Coder-7B", "Qwen2.5-Coder-7B-Instruct-8bit", "code", 20, 7.0),
    SourceModel("Granite-8B-Code", "granite-8b-code-instruct-128k-mlx-8bit", "code", 21, 8.0),

    # Tier 4: Math Specialists
    SourceModel("Qwen2.5-Math-7B", "Qwen2.5-Math-7B-Instruct-4bit", "math", 30, 7.0),
    SourceModel("Mathstral-7B", "mathstral-7B-v0.1-8bit", "math", 31, 7.0),

    # Tier 5: Domain Specialists
    SourceModel("BioMistral-7B", "BioMistral-7B-4bit", "medical", 40, 7.0),
    SourceModel("HuatuoGPT-7B", "HuatuoGPT-o1-7B-4bit", "medical", 41, 7.0),
    SourceModel("Saul-7B", "Saul-7B-Instruct-v1-4bit", "legal", 42, 7.0),

    # Tier 6: Reasoning Enhancement
    SourceModel("Phi-4-Reasoning", "Phi-4-mini-reasoning-MLX-4bit", "reasoning", 50, 4.0),
]


def run_merge(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    source_name: str,
) -> dict[str, Any]:
    """Run a single merge operation."""
    logger.info(f"MERGE: {source_name} -> {target_path.name}")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Target: {target_path}")
    logger.info(f"  Output: {output_path}")

    cmd = [
        "poetry", "run", "mc", "merge", "run",
        "-s", str(source_path),
        "-t", str(target_path),
        "-o", str(output_path),
    ]

    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (datetime.now() - start_time).total_seconds()

    if result.returncode != 0:
        logger.error(f"MERGE FAILED: {source_name}")
        logger.error(result.stderr)
        return {
            "success": False,
            "source": source_name,
            "error": result.stderr,
            "elapsed_s": elapsed,
        }

    # Parse merge analysis
    analysis_path = output_path / "merge_analysis.json"
    analysis = {}
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())

    logger.info(f"MERGE SUCCESS: {source_name} in {elapsed:.1f}s")
    if analysis.get("density"):
        d = analysis["density"]
        logger.info(f"  Concepts analyzed: {d.get('concepts_analyzed', 'N/A')}")
        logger.info(f"  Grafted: {d.get('grafted_count', 'N/A')}")
        logger.info(f"  Skipped: {d.get('skipped_count', 'N/A')}")

    return {
        "success": True,
        "source": source_name,
        "analysis": analysis,
        "elapsed_s": elapsed,
    }


def run_pipeline(
    target_name: str = "LFM2-350M-MLX-bf16",
    resume_from: str | None = None,
    max_models: int | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full knowledge stuffing pipeline.

    Args:
        target_name: Initial target model name
        resume_from: Resume from a previous stuffed version (e.g., "lfm2-stuffed-v3")
        max_models: Maximum number of models to merge (for testing)
        categories: Only merge models from these categories

    Returns:
        Pipeline results including all merge outcomes
    """
    pipeline_start = datetime.now()
    results: list[dict[str, Any]] = []

    # Determine starting target
    if resume_from:
        current_target = OUTPUT_BASE / resume_from
        if not current_target.exists():
            raise ValueError(f"Resume target not found: {current_target}")
        version = int(resume_from.split("-v")[-1]) if "-v" in resume_from else 0
    else:
        current_target = MODEL_BASE / target_name
        version = 0

    logger.info("=" * 60)
    logger.info("KNOWLEDGE STUFFING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Starting target: {current_target}")
    logger.info(f"Models to merge: {len(MERGE_SEQUENCE)}")
    if categories:
        logger.info(f"Categories filter: {categories}")
    logger.info("=" * 60)

    # Filter and sort models
    models_to_merge = sorted(MERGE_SEQUENCE, key=lambda m: m.priority)
    if categories:
        models_to_merge = [m for m in models_to_merge if m.category in categories]
    if max_models:
        models_to_merge = models_to_merge[:max_models]

    for i, source in enumerate(models_to_merge):
        source_path = MODEL_BASE / source.path
        if not source_path.exists():
            logger.warning(f"SKIP: {source.name} not found at {source_path}")
            continue

        version += 1
        output_name = f"lfm2-stuffed-v{version}"
        output_path = OUTPUT_BASE / output_name

        logger.info("")
        logger.info(f"[{i+1}/{len(models_to_merge)}] {source.name}")
        logger.info(f"  Category: {source.category}")
        logger.info(f"  Size: {source.size_b}B params")

        result = run_merge(source_path, current_target, output_path, source.name)
        results.append(result)

        if result["success"]:
            current_target = output_path  # Chain to next merge
        else:
            logger.error(f"Pipeline stopped at {source.name}")
            break

    # Summary
    pipeline_elapsed = (datetime.now() - pipeline_start).total_seconds()
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f}m)")
    logger.info(f"Successful merges: {successful}")
    logger.info(f"Failed merges: {failed}")
    logger.info(f"Final model: {current_target}")

    return {
        "final_model": str(current_target),
        "total_merges": len(results),
        "successful": successful,
        "failed": failed,
        "elapsed_s": pipeline_elapsed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Knowledge stuffing pipeline")
    parser.add_argument(
        "--target",
        default="LFM2-350M-MLX-bf16",
        help="Initial target model name",
    )
    parser.add_argument(
        "--resume",
        help="Resume from a previous stuffed version (e.g., lfm2-stuffed-v3)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        help="Maximum number of models to merge (for testing)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["generalist", "code", "math", "medical", "legal", "reasoning"],
        help="Only merge models from these categories",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models for knowledge stuffing:\n")
        for m in sorted(MERGE_SEQUENCE, key=lambda x: (x.priority, x.name)):
            path = MODEL_BASE / m.path
            status = "OK" if path.exists() else "MISSING"
            print(f"  [{status:7}] {m.name:25} ({m.category:10}) {m.size_b:5.1f}B")
        return

    results = run_pipeline(
        target_name=args.target,
        resume_from=args.resume,
        max_models=args.max_models,
        categories=args.categories,
    )

    # Save results
    results_path = OUTPUT_BASE / f"pipeline_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
