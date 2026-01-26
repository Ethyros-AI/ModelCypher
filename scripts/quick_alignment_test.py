#!/usr/bin/env python3
"""Quick alignment test - align a model and evaluate before/after.

This runs a limited alignment (few rounds) and compares performance.
The goal: see if geometric alignment actually improves model quality.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Evaluation prompts - same as evaluate_alignment_quality.py
EVAL_PROMPTS = {
    "logic": [
        ("If all cats are animals, and Whiskers is a cat, what can we conclude about Whiskers?", 50),
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", 80),
    ],
    "coherence": [
        ("Explain why the sky appears blue in simple terms.", 100),
        ("Describe a typical morning routine.", 80),
    ],
    "fact_vs_fiction": [
        ("Name the capital of France.", 20),
        ("What year did humans first land on Mars?", 50),
    ],
}


def score_response(category: str, prompt: str, response: str) -> float:
    """Score a response based on category."""
    response_lower = response.lower()

    if category == "logic":
        if "whiskers" in prompt.lower():
            return 1.0 if "animal" in response_lower else 0.3
        if "widget" in prompt.lower():
            return 1.0 if "5 minute" in response_lower or "five minute" in response_lower else 0.3

    elif category == "coherence":
        score = 0.0
        if response.strip() and response.strip()[-1] in '.!?':
            score += 0.4
        words = [w for w in response.lower().split() if len(w) > 2]
        if len(words) >= 5:
            score += 0.3
        if len(set(words)) / max(1, len(words)) > 0.5:
            score += 0.3
        return min(1.0, score)

    elif category == "fact_vs_fiction":
        if "capital of france" in prompt.lower():
            return 1.0 if "paris" in response_lower else 0.0
        if "mars" in prompt.lower():
            return 1.0 if "hasn't" in response_lower or "not yet" in response_lower or "no human" in response_lower else 0.0

    return 0.5


def generate_response(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate a response."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated_tokens = []
    current_ids = input_ids

    for _ in range(max_tokens):
        logits = model(current_ids)
        mx.eval(logits)
        next_token = int(mx.argmax(logits[0, -1, :]).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)
        current_ids = mx.concatenate([current_ids, mx.array([[next_token]])], axis=1)

    return tokenizer.decode(generated_tokens)


def evaluate_model(model, tokenizer) -> Dict[str, float]:
    """Run evaluation and return scores by category."""
    scores = {}

    for category, prompts in EVAL_PROMPTS.items():
        cat_scores = []
        for prompt, max_tokens in prompts:
            response = generate_response(model, tokenizer, prompt, max_tokens)
            score = score_response(category, prompt, response)
            cat_scores.append(score)
        scores[category] = sum(cat_scores) / len(cat_scores)

    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


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
            elif hasattr(layer.mlp, 'w1'):
                w = layer.mlp.w1.weight
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
            elif hasattr(layer.mlp, 'w1'):
                layer.mlp.w1.weight = new_weight
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
    "I believe honesty is important.",
    "Self-reference creates interesting paradoxes.",
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
    logger.info("BEFORE ALIGNMENT")
    logger.info("="*60)
    before_scores = evaluate_model(model, tokenizer)
    for k, v in before_scores.items():
        logger.info(f"  {k:20s}: {v:.2%}")

    # Run alignment
    logger.info("\n" + "="*60)
    logger.info("RUNNING GEOMETRIC SELF-ALIGNMENT (10 rounds)")
    logger.info("="*60)

    wrapper = ModelWrapper(model, tokenizer, backend)

    # Select middle layers
    layer_indices = [4, 8, 12]

    aligner = GeometricSelfAlignment(
        backend=backend,
        window_size=5,
        patience=3,
        n_directions_per_round=10,
        perturbation_scale=0.005,  # Slightly larger for faster effect
    )

    result = aligner.run(
        get_weights=wrapper.get_weights,
        set_weights=wrapper.set_weights,
        get_activations=wrapper.get_activations,
        layer_indices=layer_indices,
        probes=PROBES,
        max_rounds=10,
        strategies=[
            DirectionStrategy.CONSTANT_ALIGNED,
            DirectionStrategy.SPECTRAL_COMPRESS,
        ],
        dry_run=False,
    )

    logger.info(f"\nAlignment complete:")
    logger.info(f"  Entropy: {result.initial_entropy:.4f} → {result.final_entropy:.4f}")
    logger.info(f"  Reduction: {result.entropy_reduction:.4f}")

    logger.info("\n" + "="*60)
    logger.info("AFTER ALIGNMENT")
    logger.info("="*60)
    after_scores = evaluate_model(model, tokenizer)
    for k, v in after_scores.items():
        logger.info(f"  {k:20s}: {v:.2%}")

    logger.info("\n" + "="*60)
    logger.info("COMPARISON")
    logger.info("="*60)
    for k in before_scores:
        before = before_scores[k]
        after = after_scores[k]
        delta = after - before
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        logger.info(f"  {k:20s}: {before:.2%} → {after:.2%} ({arrow} {abs(delta):.2%})")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "alignment": {
            "initial_entropy": result.initial_entropy,
            "final_entropy": result.final_entropy,
            "entropy_reduction": result.entropy_reduction,
            "n_rounds": result.n_rounds,
        },
        "before": before_scores,
        "after": after_scores,
        "deltas": {k: after_scores[k] - before_scores[k] for k in before_scores},
    }

    output_path = f"data/evaluation/quick_alignment_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
