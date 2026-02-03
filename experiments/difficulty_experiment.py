# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Experiment: Correlate geometric metrics with model accuracy.

This script validates which geometric signals predict problem difficulty
by measuring actual model performance on problems with known answers.

Output: Correlation matrix showing which metrics predict difficulty.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """A problem with known answer for evaluation."""
    prompt: str
    expected: str
    category: str = "general"


# Test problems with known answers - ranging from trivial to hard
TEST_PROBLEMS = [
    # Easy arithmetic (model should get ~100%)
    Problem("What is 2+2?", "4", "arithmetic"),
    Problem("What is 5+3?", "8", "arithmetic"),
    Problem("What is 10-4?", "6", "arithmetic"),
    Problem("What is 7*2?", "14", "arithmetic"),
    Problem("What is 20/4?", "5", "arithmetic"),
    
    # Medium arithmetic (model should get ~70-90%)
    Problem("What is 123+456?", "579", "arithmetic"),
    Problem("What is 15*12?", "180", "arithmetic"),
    Problem("What is 256-128?", "128", "arithmetic"),
    Problem("What is 144/12?", "12", "arithmetic"),
    Problem("What is 25*25?", "625", "arithmetic"),
    
    # Hard arithmetic (model should get ~30-60%)
    Problem("What is 1234+5678?", "6912", "arithmetic"),
    Problem("What is 789*123?", "97047", "arithmetic"),
    Problem("What is 10000-3456?", "6544", "arithmetic"),
    Problem("What is 2024*2024?", "4096576", "arithmetic"),
    Problem("What is 999*999?", "998001", "arithmetic"),
    
    # Factual - capital cities (easy facts)
    Problem("What is the capital of France?", "Paris", "factual"),
    Problem("What is the capital of Japan?", "Tokyo", "factual"),
    Problem("What is the capital of Germany?", "Berlin", "factual"),
    Problem("What is the capital of Italy?", "Rome", "factual"),
    Problem("What is the capital of Spain?", "Madrid", "factual"),
    
    # Factual - harder facts
    Problem("What is the capital of Australia?", "Canberra", "factual"),
    Problem("What is the capital of Brazil?", "Brasilia", "factual"),
    Problem("What is the capital of Myanmar?", "Naypyidaw", "factual"),
    Problem("What is the capital of Kazakhstan?", "Astana", "factual"),
    Problem("What is the capital of Bhutan?", "Thimphu", "factual"),
    
    # Reasoning - simple
    Problem("If I have 5 apples and give away 2, how many do I have?", "3", "reasoning"),
    Problem("A train travels 60 miles in 1 hour. How far in 2 hours?", "120", "reasoning"),
    Problem("If a book costs $10 and I have $25, how much change do I get?", "15", "reasoning"),
    
    # Reasoning - harder
    Problem("If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?", "3", "reasoning"),
    Problem("A bat and ball cost $1.10. The bat costs $1 more than the ball. What does the ball cost in cents?", "5", "reasoning"),
]


def evaluate_model_accuracy(
    model,
    tokenizer,
    problems: list[Problem],
) -> list[tuple[Problem, bool, str]]:
    """Evaluate model accuracy on problems.
    
    Returns:
        List of (problem, correct, model_answer) tuples
    """
    import mlx.core as mx
    
    results = []
    
    for problem in problems:
        try:
            # Generate answer
            tokens = tokenizer.encode(problem.prompt)
            input_ids = mx.array([tokens])
            
            # Simple greedy generation
            for _ in range(20):  # Max 20 new tokens
                logits = model(input_ids)
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
                input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)
                
                # Stop at newline or EOS
                if int(next_token[0]) in [tokenizer.eos_token_id, 
                                           tokenizer.encode("\n")[0] if "\n" in tokenizer.get_vocab() else -1]:
                    break
            
            mx.eval(input_ids)
            
            # Decode and extract answer
            full_response = tokenizer.decode(input_ids[0].tolist())
            # Get just the generated part
            model_answer = full_response[len(problem.prompt):].strip()
            
            # Check if correct (flexible matching)
            correct = _check_answer(model_answer, problem.expected)
            
            results.append((problem, correct, model_answer))
            logger.debug(f"Q: {problem.prompt[:50]}... A: {model_answer[:30]} ({'✓' if correct else '✗'})")
            
        except Exception as e:
            logger.warning(f"Failed to evaluate: {e}")
            results.append((problem, False, f"ERROR: {e}"))
    
    return results


def _check_answer(model_answer: str, expected: str) -> bool:
    """Check if model answer matches expected (flexible matching)."""
    # Clean up
    model_clean = model_answer.lower().strip()
    expected_clean = expected.lower().strip()
    
    # Direct match
    if expected_clean in model_clean:
        return True
    
    # Number extraction
    import re
    model_numbers = re.findall(r'\d+', model_clean)
    expected_numbers = re.findall(r'\d+', expected_clean)
    
    if model_numbers and expected_numbers:
        return model_numbers[0] == expected_numbers[0]
    
    return False


def run_experiment(
    model_path: str,
    output_path: str = "/tmp/difficulty_correlation.json",
) -> dict[str, Any]:
    """Run the full correlation experiment."""
    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.use_cases.curriculum_profiler import CurriculumProfiler
    
    initialize_default_backend()
    
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model_for_training(model_path)
    
    # Step 1: Get geometric profiles
    logger.info("Step 1: Computing geometric profiles...")
    profiler = CurriculumProfiler(model, tokenizer)
    
    prompts = [p.prompt for p in TEST_PROBLEMS]
    profiles = profiler.profile_problems(prompts)
    
    # Step 2: Evaluate model accuracy
    logger.info("Step 2: Evaluating model accuracy...")
    eval_results = evaluate_model_accuracy(model, tokenizer, TEST_PROBLEMS)
    
    accuracy_map = {p.prompt: correct for p, correct, _ in eval_results}
    
    # Step 3: Compute correlations
    logger.info("Step 3: Computing correlations...")
    
    # Prepare data for correlation
    data = []
    for profile in profiles.profiles:
        correct = accuracy_map.get(profile.prompt, False)
        data.append({
            "prompt": profile.prompt[:50],
            "correct": 1 if correct else 0,
            "cka_similarity": profile.cka_similarity,
            "barrier_height": profile.barrier_height,
            "fisher_mean": profile.fisher_mean,
            "goldilocks_score": profile.goldilocks_score,
            "trajectory_curvature": profile.trajectory_curvature_mean,
            "local_density": profile.local_density,
        })
    
    # Compute correlations (Pearson's r)
    metrics = ["cka_similarity", "barrier_height", "fisher_mean", 
               "goldilocks_score", "trajectory_curvature", "local_density"]
    
    correlations = {}
    accuracy = [d["correct"] for d in data]
    
    for metric in metrics:
        values = [d[metric] for d in data]
        # Filter NaN
        valid_pairs = [(a, v) for a, v in zip(accuracy, values) 
                       if v is not None and not (isinstance(v, float) and v != v)]
        
        if len(valid_pairs) >= 3:
            acc_valid = [p[0] for p in valid_pairs]
            val_valid = [p[1] for p in valid_pairs]
            
            # Pearson correlation
            n = len(acc_valid)
            sum_a = sum(acc_valid)
            sum_v = sum(val_valid)
            sum_av = sum(a * v for a, v in valid_pairs)
            sum_a2 = sum(a * a for a in acc_valid)
            sum_v2 = sum(v * v for v in val_valid)
            
            numerator = n * sum_av - sum_a * sum_v
            denominator = ((n * sum_a2 - sum_a**2) * (n * sum_v2 - sum_v**2)) ** 0.5
            
            if denominator > 0:
                r = numerator / denominator
            else:
                r = 0.0
            
            correlations[metric] = round(r, 4)
        else:
            correlations[metric] = None
    
    # Compute overall accuracy
    total_correct = sum(1 for d in data if d["correct"])
    overall_accuracy = total_correct / len(data) if data else 0
    
    # Summary
    result = {
        "model_path": model_path,
        "n_problems": len(data),
        "overall_accuracy": round(overall_accuracy, 4),
        "correlations": correlations,
        "interpretation": _interpret_correlations(correlations),
        "detailed_results": data,
    }
    
    # Save
    Path(output_path).write_text(json.dumps(result, indent=2))
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DIFFICULTY CORRELATION EXPERIMENT")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Problems: {len(data)}")
    print(f"Overall accuracy: {overall_accuracy:.1%}")
    print()
    print("Correlations with accuracy (higher = better predictor):")
    for metric, r in sorted(correlations.items(), key=lambda x: abs(x[1] or 0), reverse=True):
        if r is not None:
            direction = "↑" if r > 0 else "↓"
            strength = "STRONG" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
            print(f"  {metric:25} r={r:+.3f} {direction} ({strength})")
    print("="*60)
    
    return result


def _interpret_correlations(correlations: dict[str, float | None]) -> str:
    """Generate interpretation of correlation results."""
    interpretations = []
    
    cka = correlations.get("cka_similarity")
    if cka is not None:
        if cka > 0.3:
            interpretations.append(
                f"CKA similarity positively correlates with accuracy (r={cka:.3f}): "
                "Problems similar to reference are easier."
            )
        elif cka < -0.3:
            interpretations.append(
                f"CKA similarity negatively correlates (r={cka:.3f}): "
                "This is unexpected and suggests reference set issues."
            )
    
    barrier = correlations.get("barrier_height")
    if barrier is not None:
        if barrier < -0.3:
            interpretations.append(
                f"Barrier height negatively correlates (r={barrier:.3f}): "
                "Higher barriers = harder problems. Validated!"
            )
    
    goldilocks = correlations.get("goldilocks_score")
    if goldilocks is not None:
        if goldilocks > 0.3:
            interpretations.append(
                f"Goldilocks score positively correlates (r={goldilocks:.3f}): "
                "Score predicts problem difficulty well."
            )
    
    if not interpretations:
        interpretations.append("No strong correlations found. More data may be needed.")
    
    return " | ".join(interpretations)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python difficulty_experiment.py /path/to/model")
        sys.exit(1)
    
    run_experiment(sys.argv[1])
