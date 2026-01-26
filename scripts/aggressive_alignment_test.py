#!/usr/bin/env python3
"""Aggressive alignment test - push harder to see if behavior changes.

The question: Does geometric alignment actually improve model quality?

Previous test showed tiny entropy changes had no effect.
This test:
1. Uses larger perturbation scales (0.01 - 0.1)
2. Runs more rounds
3. Uses multi-scale to escape local minima
4. Tracks perplexity as additional metric
"""

from __future__ import annotations

import json
import logging
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Evaluation prompts
EVAL_PROMPTS = [
    ("If all cats are animals, and Whiskers is a cat, what can we conclude?", 50, "logic"),
    ("Explain why the sky is blue.", 100, "coherence"),
    ("Name the capital of France.", 20, "fact"),
    ("What is 2 + 2?", 20, "math"),
    ("Complete: The quick brown fox jumps over the lazy", 10, "completion"),
]


def generate_and_score(model, tokenizer, prompt: str, max_tokens: int, category: str) -> dict:
    """Generate a response and score it."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    current = input_ids
    total_log_prob = 0.0

    for _ in range(max_tokens):
        logits = model(current)
        mx.eval(logits)

        # Get probabilities for perplexity
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        next_token = int(mx.argmax(probs).item())

        if next_token == tokenizer.eos_token_id:
            break

        # Track log probability
        prob = float(probs[next_token].item())
        if prob > 0:
            total_log_prob += math.log(prob)

        generated.append(next_token)
        current = mx.concatenate([current, mx.array([[next_token]])], axis=1)

    response = tokenizer.decode(generated)
    n_tokens = len(generated)

    # Perplexity = exp(-avg_log_prob)
    perplexity = math.exp(-total_log_prob / max(1, n_tokens)) if n_tokens > 0 else float('inf')

    # Score based on category
    response_lower = response.lower()
    score = 0.0

    if category == "logic":
        score = 1.0 if "animal" in response_lower else 0.0
    elif category == "coherence":
        # Check for key concepts
        if "scatter" in response_lower or "rayleigh" in response_lower or "wavelength" in response_lower:
            score = 1.0
        elif "light" in response_lower and "blue" in response_lower:
            score = 0.7
        else:
            score = 0.3 if response.strip() else 0.0
    elif category == "fact":
        score = 1.0 if "paris" in response_lower else 0.0
    elif category == "math":
        score = 1.0 if "4" in response else 0.0
    elif category == "completion":
        score = 1.0 if "dog" in response_lower else 0.5 if response.strip() else 0.0

    return {
        "prompt": prompt,
        "response": response,
        "score": score,
        "perplexity": perplexity,
        "n_tokens": n_tokens,
    }


def evaluate_model(model, tokenizer) -> dict:
    """Run full evaluation."""
    results = []
    total_score = 0.0
    total_perplexity = 0.0

    for prompt, max_tokens, category in EVAL_PROMPTS:
        result = generate_and_score(model, tokenizer, prompt, max_tokens, category)
        results.append(result)
        total_score += result["score"]
        total_perplexity += result["perplexity"]

    n = len(EVAL_PROMPTS)
    return {
        "results": results,
        "avg_score": total_score / n,
        "avg_perplexity": total_perplexity / n,
    }


class ModelWrapper:
    """Wrapper for self-alignment interface."""

    def __init__(self, model, tokenizer, backend):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.n_layers = len(model.model.layers)

    def get_weights(self, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            if hasattr(layer.feed_forward, 'gate_proj'):
                w = layer.feed_forward.gate_proj.weight
            elif hasattr(layer.feed_forward, 'w1'):
                w = layer.feed_forward.w1.weight
            else:
                w = layer.feed_forward.weight
        else:
            if hasattr(layer.mlp, 'gate_proj'):
                w = layer.mlp.gate_proj.weight
            else:
                w = layer.mlp.weight

        mx.eval(w)
        w_f32 = w.astype(mx.float32)
        mx.eval(w_f32)
        return self.backend.array(np.array(w_f32.tolist(), dtype=np.float32))

    def set_weights(self, layer_idx: int, weights) -> None:
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]

        if hasattr(weights, 'tolist'):
            w_np = np.array(weights.tolist())
        else:
            w_np = np.array(weights)

        new_weight = mx.array(w_np)

        if hasattr(layer, 'feed_forward'):
            if hasattr(layer.feed_forward, 'gate_proj'):
                layer.feed_forward.gate_proj.weight = new_weight
            elif hasattr(layer.feed_forward, 'w1'):
                layer.feed_forward.w1.weight = new_weight
            else:
                layer.feed_forward.weight = new_weight
        else:
            if hasattr(layer.mlp, 'gate_proj'):
                layer.mlp.gate_proj.weight = new_weight
            else:
                layer.mlp.weight = new_weight

        mx.eval(new_weight)

    def get_activations(self, probes: List[str]) -> Dict[int, np.ndarray]:
        import mlx.core as mx
        layer_activations: Dict[int, list] = {i: [] for i in range(self.n_layers)}

        for probe in probes:
            tokens = self.tokenizer.encode(probe)
            input_ids = mx.array([tokens])

            for layer_idx in range(self.n_layers):
                layer = self.model.model.layers[layer_idx]

                if hasattr(layer, 'feed_forward'):
                    original = layer.feed_forward
                    key = 'feed_forward'
                else:
                    original = layer.mlp
                    key = 'mlp'

                captured = {}

                class Hook:
                    def __init__(self, mlp):
                        self.mlp = mlp
                    def __call__(self, x):
                        captured['output'] = self.mlp(x)
                        return captured['output']

                if key == 'feed_forward':
                    layer.feed_forward = Hook(original)
                else:
                    layer.mlp = Hook(original)

                try:
                    _ = self.model(input_ids)
                    mx.eval(captured.get('output', mx.zeros((1, 1, 1))))
                    if 'output' in captured:
                        act = np.array(captured['output'][0].tolist())
                        if act.ndim > 1:
                            act = act.mean(axis=0)
                        layer_activations[layer_idx].append(act)
                finally:
                    if key == 'feed_forward':
                        layer.feed_forward = original
                    else:
                        layer.mlp = original

        result = {}
        for layer_idx, acts in layer_activations.items():
            if acts:
                stacked = np.vstack(acts)
                result[layer_idx] = self.backend.array(stacked.astype(np.float32))
        return result


PROBES = [
    "The sky is blue.",
    "Water is wet.",
    "Fire is hot.",
    "Paris is the capital of France.",
    "The Earth orbits the Sun.",
    "Mathematics is abstract.",
    "Self-reference is recursive.",
]


def main():
    import mlx.core as mx
    from mlx_lm import load
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_alignment.geometric_self_alignment import GeometricSelfAlignment
    from modelcypher.core.use_cases.self_alignment.direction_generator import DirectionStrategy

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    backend = initialize_default_backend()

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    logger.info("\n" + "="*60)
    logger.info("BEFORE ALIGNMENT - Baseline Evaluation")
    logger.info("="*60)
    before = evaluate_model(model, tokenizer)
    logger.info(f"  Average Score: {before['avg_score']:.2%}")
    logger.info(f"  Average Perplexity: {before['avg_perplexity']:.2f}")
    for r in before['results']:
        logger.info(f"    {r['prompt'][:40]}... → score={r['score']:.1f}, ppl={r['perplexity']:.1f}")

    # Run aggressive alignment
    logger.info("\n" + "="*60)
    logger.info("RUNNING AGGRESSIVE ALIGNMENT")
    logger.info("Using larger scales: 0.01, 0.03, 0.05")
    logger.info("="*60)

    wrapper = ModelWrapper(model, tokenizer, backend)

    # Test multiple scale levels
    scales_to_test = [0.01, 0.03, 0.05]
    best_score = before['avg_score']
    best_scale = 0.0

    for scale in scales_to_test:
        # Reload model to reset
        model, tokenizer = load(model_path)
        wrapper = ModelWrapper(model, tokenizer, backend)

        logger.info(f"\nTrying scale={scale}...")

        aligner = GeometricSelfAlignment(
            backend=backend,
            window_size=5,
            patience=5,
            n_directions_per_round=5,  # Fewer for speed
            perturbation_scale=scale,
        )

        result = aligner.run(
            get_weights=wrapper.get_weights,
            set_weights=wrapper.set_weights,
            get_activations=wrapper.get_activations,
            layer_indices=[4, 8, 12],
            probes=PROBES,
            max_rounds=15,
            strategies=[DirectionStrategy.CONSTANT_ALIGNED, DirectionStrategy.SPECTRAL_COMPRESS],
            dry_run=False,
        )

        logger.info(f"  Entropy: {result.initial_entropy:.4f} → {result.final_entropy:.4f} (Δ={result.entropy_reduction:.4f})")

        # Evaluate after
        after = evaluate_model(model, tokenizer)
        logger.info(f"  Score: {before['avg_score']:.2%} → {after['avg_score']:.2%}")
        logger.info(f"  Perplexity: {before['avg_perplexity']:.2f} → {after['avg_perplexity']:.2f}")

        if after['avg_score'] > best_score:
            best_score = after['avg_score']
            best_scale = scale
            logger.info(f"  *** NEW BEST! ***")

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Best scale: {best_scale}")
    logger.info(f"Best score: {best_score:.2%} (baseline was {before['avg_score']:.2%})")

    if best_scale > 0:
        delta = best_score - before['avg_score']
        if delta > 0:
            logger.info(f"IMPROVEMENT: +{delta:.2%}")
        elif delta < 0:
            logger.info(f"DEGRADATION: {delta:.2%}")
        else:
            logger.info("NO CHANGE")
    else:
        logger.info("No improvement found at any scale")


if __name__ == "__main__":
    main()
