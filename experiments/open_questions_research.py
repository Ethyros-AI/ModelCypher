# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Research experiments for open questions in curriculum design.

Open Questions:
1. Which layer's activations matter most? — Need to test middle vs late layers
2. Aggregation method? — Mean, max, or per-problem?
3. Reference set for CKA? — What defines "known knowledge"?
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test problems for experiments
TEST_PROBLEMS = [
    # Easy arithmetic
    "What is 2+2?",
    "What is 5+3?",
    "What is 10-4?",
    "What is 7*2?",
    "What is 20/4?",
    
    # Medium arithmetic
    "What is 123+456?",
    "What is 15*12?",
    "What is 256-128?",
    
    # Hard arithmetic
    "What is 789*123?",
    "What is 999*999?",
    
    # Factual
    "What is the capital of France?",
    "What is the capital of Brazil?",
    "What is the capital of Bhutan?",
    
    # Reasoning
    "If I have 5 apples and give away 2, how many do I have?",
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. What does the ball cost?",
]


def experiment_1_layer_importance(model_path: str) -> dict:
    """Q1: Which layer's activations matter most for difficulty prediction?
    
    Tests: early (25%), middle (50%), late (75%), final (100%) layers.
    """
    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.use_cases.curriculum_profiler import CurriculumProfiler
    
    initialize_default_backend()
    model, tokenizer = load_model_for_training(model_path)
    
    # Determine layer count
    n_layers = len(model.model.layers)
    test_layers = {
        "early_25%": int(n_layers * 0.25),
        "middle_50%": int(n_layers * 0.50),
        "late_75%": int(n_layers * 0.75),
        "final": n_layers - 1,
    }
    
    results = {}
    
    for layer_name, layer_idx in test_layers.items():
        logger.info(f"Testing layer {layer_idx} ({layer_name})...")
        
        profiler = CurriculumProfiler(model, tokenizer, layer_idx=layer_idx)
        profiles = profiler.profile_problems(TEST_PROBLEMS)
        profiles.compute_difficulty_scores()
        
        # Collect metrics
        cka_values = [p.cka_similarity for p in profiles.profiles]
        fisher_values = [p.fisher_mean for p in profiles.profiles]
        difficulty_values = [p.difficulty_score for p in profiles.profiles]
        
        # Calculate spread (variance indicates discriminative power)
        import statistics
        results[layer_name] = {
            "layer_idx": layer_idx,
            "cka_variance": statistics.variance(cka_values),
            "cka_range": max(cka_values) - min(cka_values),
            "fisher_variance": statistics.variance(fisher_values),
            "fisher_range": max(fisher_values) - min(fisher_values),
            "difficulty_variance": statistics.variance(difficulty_values),
            "difficulty_range": max(difficulty_values) - min(difficulty_values),
        }
    
    # Best layer = highest variance (most discriminative)
    best_layer = max(results.items(), key=lambda x: x[1]["difficulty_variance"])
    
    return {
        "question": "Which layer's activations matter most?",
        "results": results,
        "best_layer": best_layer[0],
        "recommendation": f"Use layer at {best_layer[0]} position (idx={best_layer[1]['layer_idx']})",
    }


def experiment_2_aggregation_method(model_path: str) -> dict:
    """Q2: Aggregation method — Mean, max, or per-problem?
    
    Tests how different aggregation affects separation between easy/hard problems.
    """
    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
    
    initialize_default_backend()
    model, tokenizer = load_model_for_training(model_path)
    activation_provider = MLXActivationProvider()
    
    import mlx.core as mx
    import numpy as np
    
    n_layers = len(model.model.layers)
    
    # Collect activations across ALL layers for each problem
    all_activations = []
    for problem in TEST_PROBLEMS:
        acts = activation_provider.collect_hidden_activations(
            model=model,
            tokenizer=tokenizer,
            text=problem,
        )
        all_activations.append(acts)
    
    # Test different aggregation methods
    results = {}
    
    # Method 1: Mean across tokens, then mean across layers
    method1_values = []
    for acts in all_activations:
        layer_means = []
        for layer_idx in sorted(acts.keys()):
            layer_act = acts[layer_idx]
            mx.eval(layer_act)
            mean_act = float(mx.mean(mx.abs(layer_act)))
            layer_means.append(mean_act)
        method1_values.append(sum(layer_means) / len(layer_means))
    
    results["mean_mean"] = {
        "values": method1_values,
        "variance": float(np.var(method1_values)),
        "range": max(method1_values) - min(method1_values),
    }
    
    # Method 2: Max across tokens, then max across layers
    method2_values = []
    for acts in all_activations:
        layer_maxes = []
        for layer_idx in sorted(acts.keys()):
            layer_act = acts[layer_idx]
            mx.eval(layer_act)
            max_act = float(mx.max(mx.abs(layer_act)))
            layer_maxes.append(max_act)
        method2_values.append(max(layer_maxes))
    
    results["max_max"] = {
        "values": method2_values,
        "variance": float(np.var(method2_values)),
        "range": max(method2_values) - min(method2_values),
    }
    
    # Method 3: Middle layer only (current approach)
    mid_layer = n_layers // 2
    method3_values = []
    for acts in all_activations:
        layer_act = acts[mid_layer]
        mx.eval(layer_act)
        method3_values.append(float(mx.mean(mx.abs(layer_act))))
    
    results["single_middle"] = {
        "values": method3_values,
        "variance": float(np.var(method3_values)),
        "range": max(method3_values) - min(method3_values),
    }
    
    # Best method = highest variance (most discriminative)
    best_method = max(results.items(), key=lambda x: x[1]["variance"])
    
    return {
        "question": "Which aggregation method is best?",
        "results": {k: {kk: v for kk, v in v.items() if kk != "values"} for k, v in results.items()},
        "best_method": best_method[0],
        "recommendation": f"Use {best_method[0]} aggregation for maximum discrimination",
    }


def experiment_3_reference_set(model_path: str) -> dict:
    """Q3: Reference set for CKA — What defines "known knowledge"?
    
    Tests different reference sets and their effect on CKA scores.
    """
    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.use_cases.curriculum_profiler import CurriculumProfiler
    
    initialize_default_backend()
    model, tokenizer = load_model_for_training(model_path)
    
    # Different reference set strategies
    reference_strategies = {
        "first_1": TEST_PROBLEMS[:1],  # Just "2+2"
        "first_3": TEST_PROBLEMS[:3],  # First 3 easy problems
        "easy_5": TEST_PROBLEMS[:5],   # All easy arithmetic
        "mixed_5": [TEST_PROBLEMS[0], TEST_PROBLEMS[5], TEST_PROBLEMS[10], 
                    TEST_PROBLEMS[13], TEST_PROBLEMS[14]],  # Mix of types
        "all_easy": TEST_PROBLEMS[:8],  # All arithmetic
    }
    
    results = {}
    
    for strategy_name, reference_set in reference_strategies.items():
        logger.info(f"Testing reference strategy: {strategy_name}...")
        
        profiler = CurriculumProfiler(model, tokenizer)
        
        # Profile with specific reference
        profiles = profiler.profile_problems(
            problems=TEST_PROBLEMS,
            reference_prompts=reference_set,
        )
        
        # Compute CKA spread
        cka_values = [p.cka_similarity for p in profiles.profiles]
        
        # Check separation: do easy problems have high CKA, hard problems low CKA?
        easy_cka = cka_values[:5]  # First 5 are easy
        hard_cka = cka_values[-5:]  # Last 5 are hard
        
        separation = sum(easy_cka)/len(easy_cka) - sum(hard_cka)/len(hard_cka)
        
        import statistics
        results[strategy_name] = {
            "reference_count": len(reference_set),
            "cka_variance": statistics.variance(cka_values),
            "cka_range": max(cka_values) - min(cka_values),
            "easy_hard_separation": separation,
            "easy_mean_cka": sum(easy_cka)/len(easy_cka),
            "hard_mean_cka": sum(hard_cka)/len(hard_cka),
        }
    
    # Best reference = highest easy/hard separation
    best = max(results.items(), key=lambda x: x[1]["easy_hard_separation"])
    
    return {
        "question": "What reference set for CKA?",
        "results": results,
        "best_strategy": best[0],
        "recommendation": f"Use {best[0]} strategy ({best[1]['reference_count']} examples) for best separation",
    }


def run_all_experiments(model_path: str) -> dict:
    """Run all open question experiments."""
    results = {}
    
    print("\n" + "="*60)
    print("OPEN QUESTIONS RESEARCH")
    print("="*60)
    
    print("\n[1/3] Testing layer importance...")
    results["q1_layer_importance"] = experiment_1_layer_importance(model_path)
    print(f"  → Best: {results['q1_layer_importance']['recommendation']}")
    
    print("\n[2/3] Testing aggregation methods...")
    results["q2_aggregation"] = experiment_2_aggregation_method(model_path)
    print(f"  → Best: {results['q2_aggregation']['recommendation']}")
    
    print("\n[3/3] Testing reference sets...")
    results["q3_reference_set"] = experiment_3_reference_set(model_path)
    print(f"  → Best: {results['q3_reference_set']['recommendation']}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for q, r in results.items():
        print(f"{q}: {r['recommendation']}")
    print("="*60)
    
    # Save
    Path("/tmp/open_questions_research.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to /tmp/open_questions_research.json")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python open_questions_research.py /path/to/model")
        sys.exit(1)
    
    run_all_experiments(sys.argv[1])
