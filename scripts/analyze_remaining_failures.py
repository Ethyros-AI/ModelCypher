#!/usr/bin/env python3
"""Analyze the 2 remaining GSM8K failures with the unified adapter.

Problems 8 and 25 still fail. Let's understand:
1. What's structurally different about these problems?
2. What does the model output?
3. What's the entropy trajectory?
4. How can we target these patterns?
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_spectral_entropy(activations: np.ndarray, sqrt_eps: float) -> float:
    if len(activations) < 2:
        return 0.0
    centered = activations - activations.mean(axis=0)
    _, S, _ = svd(centered, full_matrices=False)
    S_valid = S[S > sqrt_eps * S[0]]
    if len(S_valid) < 2:
        return 0.0
    p = S_valid ** 2
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def analyze_problem_structure(question: str) -> dict:
    """Analyze structural features of a problem."""
    numbers = re.findall(r'\d+\.?\d*', question)

    has_fraction_words = any(w in question.lower() for w in ['third', 'quarter', 'half', 'percent', '%', 'fraction', 'twice', 'double', 'triple'])
    has_conditional = any(w in question.lower() for w in ['if', 'when', 'after', 'before', 'while', 'until'])
    has_comparison = any(w in question.lower() for w in ['more than', 'less than', 'twice', 'half', 'ratio', 'times as'])
    has_multi_step = any(w in question.lower() for w in ['then', 'after that', 'next', 'finally', 'first', 'second'])
    has_rate = any(w in question.lower() for w in ['per', 'each', 'every', '/hour', '/day', '/week', '/minute'])
    has_total = any(w in question.lower() for w in ['total', 'altogether', 'combined', 'sum', 'all together'])

    n_sentences = len([s for s in question.split('.') if s.strip()])
    n_words = len(question.split())

    return {
        "n_numbers": len(numbers),
        "n_words": n_words,
        "n_sentences": n_sentences,
        "has_fraction_words": has_fraction_words,
        "has_conditional": has_conditional,
        "has_comparison": has_comparison,
        "has_multi_step": has_multi_step,
        "has_rate": has_rate,
        "has_total": has_total,
    }


def main():
    import mlx.core as mx
    from mlx_lm import load, generate
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader

    logger.info("=" * 70)
    logger.info("ANALYZING REMAINING FAILURES")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/unified_expansion_lora"

    # Load model
    logger.info(f"\nLoading model with adapter: {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    n_layers = len(model.model.layers)
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    # Load GSM8K
    loader = BenchmarkLoader()
    gsm_test = loader.load("gsm8k", split="test", limit=30)

    # Problems 8 and 25 failed (0-indexed: 7 and 24)
    failed_indices = [7, 24]

    results = {"timestamp": datetime.now().isoformat(), "failures": []}

    for idx in failed_indices:
        sample = gsm_test.samples[idx]
        question = sample.prompt.replace("Answer:", "").strip()
        expected = sample.answer

        logger.info(f"\n{'=' * 70}")
        logger.info(f"PROBLEM {idx + 1} (index {idx})")
        logger.info(f"{'=' * 70}")
        logger.info(f"\nQuestion:\n{question}")
        logger.info(f"\nExpected answer: {expected}")

        # Generate model output
        prompt = f"Question: {question}\n\nAnswer:"
        output = generate(model, tokenizer, prompt=prompt, max_tokens=600, verbose=False)

        # Extract predicted answer
        if "####" in output:
            nums = re.findall(r'-?\d+', output.split("####")[-1].replace(",", ""))
        else:
            nums = re.findall(r'-?\d+', output.replace(",", ""))
        predicted = nums[-1] if nums else ""

        logger.info(f"\nModel output:\n{output}")
        logger.info(f"\nPredicted: {predicted}, Expected: {expected}")

        # Analyze structure
        structure = analyze_problem_structure(question)
        logger.info(f"\nStructural analysis:")
        for k, v in structure.items():
            logger.info(f"  {k}: {v}")

        # Get entropy trajectory
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = model.model.embed_tokens(input_ids)

        trajectory = []
        for layer in model.model.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            mx.eval(hidden)
            # Use std as proxy for entropy (single sample)
            std = float(np.std(np.array(hidden[0, -1, :].tolist())))
            trajectory.append(std)

        peak_idx = np.argmax(trajectory)
        peak = trajectory[peak_idx]
        initial = trajectory[0]
        final = trajectory[-1]
        expansion = (peak - initial) / (peak_idx + 1) if peak_idx > 0 else 0
        compression_layers = n_layers - peak_idx - 1
        compression = (peak - final) / max(compression_layers, 1)
        ratio = compression / expansion if expansion > 1e-10 else float('inf')

        logger.info(f"\nEntropy trajectory:")
        logger.info(f"  Initial: {initial:.4f}")
        logger.info(f"  Peak (layer {peak_idx}): {peak:.4f}")
        logger.info(f"  Final: {final:.4f}")
        logger.info(f"  Expansion rate: {expansion:.4f}")
        logger.info(f"  Compression rate: {compression:.4f}")
        logger.info(f"  Ratio/φ: {ratio/PHI:.4f}")

        results["failures"].append({
            "index": idx,
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "output": output,
            "structure": structure,
            "trajectory": {
                "initial": initial,
                "peak": peak,
                "peak_layer": int(peak_idx),
                "final": final,
                "expansion_rate": expansion,
                "compression_rate": compression,
                "ratio_vs_phi": ratio / PHI if ratio != float('inf') else float('inf'),
            }
        })

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("FAILURE PATTERN SUMMARY")
    logger.info(f"{'=' * 70}")

    for failure in results["failures"]:
        logger.info(f"\nProblem {failure['index'] + 1}:")
        logger.info(f"  Expected: {failure['expected']}, Got: {failure['predicted']}")
        logger.info(f"  Ratio/φ: {failure['trajectory']['ratio_vs_phi']:.4f}")
        s = failure['structure']
        patterns = []
        if s['has_fraction_words']:
            patterns.append("fraction_words")
        if s['has_conditional']:
            patterns.append("conditional")
        if s['has_comparison']:
            patterns.append("comparison")
        if s['has_multi_step']:
            patterns.append("multi_step")
        if s['has_rate']:
            patterns.append("rate")
        logger.info(f"  Patterns: {', '.join(patterns) if patterns else 'basic arithmetic'}")

    # Save results
    output_path = Path("data/experiments/remaining_failures_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
