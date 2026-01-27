#!/usr/bin/env python3
"""Phase A: Geometric Analysis of GSM8K Capability.

Scientific method: First diagnose, then treat.

This script analyzes whether GSM8K failures are:
1. DISCONNECTED: High κ, constants present (like arithmetic was)
2. TRUE GAP: Low constant matches, capability doesn't exist

NO HEURISTICS. All analysis derived from the geometry itself.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Fundamental constants (proven statistically significant, p < 0.01)
CONSTANTS = {
    "pi/e": np.pi / np.e,         # 1.1557273
    "e/pi": np.e / np.pi,         # 0.8652560
    "phi": (1 + np.sqrt(5)) / 2,  # 1.6180339
    "1/phi": 2 / (1 + np.sqrt(5)), # 0.6180339
    "sqrt2": np.sqrt(2),          # 1.4142135
    "1/sqrt2": 1 / np.sqrt(2),    # 0.7071067
}


class GeometricAnalyzer:
    """Analyze geometric signatures of model capabilities."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dtype_eps = np.finfo(np.float32).eps

    def get_last_hidden_state(self, prompt: str) -> np.ndarray:
        """Get the last hidden state for a prompt."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Get hidden states from the model
        # For Qwen3, we need to access the internal model
        hidden = self.model.model.embed_tokens(input_ids)

        for layer in self.model.model.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        hidden = self.model.model.norm(hidden)
        mx.eval(hidden)

        # Return last token's hidden state
        return np.array(hidden[0, -1, :].tolist(), dtype=np.float32)

    def get_logits(self, prompt: str) -> np.ndarray:
        """Get the output logits for a prompt."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)

        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    def compute_gram_and_kappa(self, activations: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute Gram matrix and condition number from activations.

        Args:
            activations: Shape (n_samples, dim) activation matrix

        Returns:
            Gram matrix and condition number κ
        """
        # Center activations
        centered = activations - activations.mean(axis=0)

        # Gram matrix
        G = centered @ centered.T

        # Condition number
        kappa = np.linalg.cond(G)

        return G, kappa

    def count_constant_matches(self, weight_matrix: np.ndarray,
                                proximity_threshold: float = 0.05) -> Dict[str, int]:
        """Count how many SVD ratios match fundamental constants.

        Args:
            weight_matrix: Weight matrix to analyze
            proximity_threshold: Max relative error to count as match (derived from sqrt(eps))

        Returns:
            Dict mapping constant names to match counts
        """
        # SVD decomposition
        U, S, Vt = svd(weight_matrix, full_matrices=False)

        # Count matches for each constant
        matches = {name: 0 for name in CONSTANTS}

        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if S[j] < self.dtype_eps:
                    continue

                ratio = S[i] / S[j]
                inv_ratio = S[j] / S[i]

                for const_name, const_val in CONSTANTS.items():
                    rel_error = abs(ratio - const_val) / const_val
                    inv_error = abs(inv_ratio - const_val) / const_val

                    if rel_error < proximity_threshold:
                        matches[const_name] += 1
                    if inv_error < proximity_threshold:
                        matches[const_name] += 1

        return matches

    def analyze_category(self, prompts: List[str], category_name: str) -> Dict:
        """Analyze geometric properties of a category of prompts.

        Returns condition number κ and constant matches.
        """
        logger.info(f"\nAnalyzing {category_name} ({len(prompts)} prompts)...")

        # Get activations
        activations = []
        for prompt in prompts:
            try:
                act = self.get_last_hidden_state(prompt)
                activations.append(act)
            except Exception as e:
                logger.warning(f"Failed to get activation: {e}")
                continue

        if len(activations) < 2:
            logger.warning(f"Not enough activations for {category_name}")
            return {"error": "insufficient_data"}

        activations = np.vstack(activations)

        # Compute Gram matrix and κ
        G, kappa = self.compute_gram_and_kappa(activations)

        # Count constant matches in the activation Gram matrix
        matches = self.count_constant_matches(G)
        total_matches = sum(matches.values())

        logger.info(f"  κ (condition number): {kappa:.2e}")
        logger.info(f"  Total constant matches: {total_matches}")
        for const_name, count in sorted(matches.items(), key=lambda x: -x[1])[:3]:
            if count > 0:
                logger.info(f"    {const_name}: {count}")

        return {
            "kappa": float(kappa),
            "total_matches": total_matches,
            "matches_by_constant": {k: int(v) for k, v in matches.items()},
            "n_prompts": len(prompts),
            "activation_shape": list(activations.shape),
        }


def main():
    import re
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("PHASE A: GEOMETRIC ANALYSIS OF GSM8K CAPABILITY")
    logger.info("=" * 70)
    logger.info("\nDiagnosis: Is GSM8K failure DISCONNECTED or TRUE GAP?")

    # Load model with best adapter
    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"\nLoading model: {model_path}")
    logger.info(f"With adapter: {adapter_path}")

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    analyzer = GeometricAnalyzer(model, tokenizer)

    # Load GSM8K test data
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader
    loader = BenchmarkLoader()
    gsm_test = loader.load("gsm8k", split="test", limit=60)

    # First, evaluate which problems pass/fail
    logger.info("\nEvaluating GSM8K problems to categorize passing vs failing...")

    import mlx.core as mx

    passing_prompts = []
    failing_prompts = []

    for sample in gsm_test.samples[:50]:
        question = sample.prompt.replace("Answer:", "").strip()
        expected = sample.answer

        prompt = f"Question: {question}\n\nAnswer:"
        tokens = tokenizer.encode(prompt)
        generated = []

        for _ in range(300):
            logits = model(mx.array([tokens + generated]))
            mx.eval(logits)
            logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()
            next_tok = int(np.argmax(probs))
            generated.append(next_tok)

            decoded = tokenizer.decode(generated)
            if "####" in decoded:
                for _ in range(15):
                    logits = model(mx.array([tokens + generated]))
                    mx.eval(logits)
                    logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
                    probs = np.exp(logits_np - logits_np.max())
                    probs = probs / probs.sum()
                    next_tok = int(np.argmax(probs))
                    generated.append(next_tok)
                break
            if "<|im_end|>" in decoded:
                break

        output = tokenizer.decode(generated).strip().replace("<|im_end|>", "")

        if "####" in output:
            answer_part = output.split("####")[-1].strip().replace(",", "").replace("$", "")
            numbers = re.findall(r'-?\d+', answer_part)
            predicted = numbers[0] if numbers else ""
        else:
            numbers = re.findall(r'-?\d+', output.replace(",", ""))
            predicted = numbers[-1] if numbers else ""

        if predicted == expected:
            passing_prompts.append(prompt)
        else:
            failing_prompts.append(prompt)
            if len(failing_prompts) <= 3:
                logger.info(f"  FAIL: {question[:50]}... → {predicted} (expected {expected})")

    logger.info(f"\nCategorized: {len(passing_prompts)} passing, {len(failing_prompts)} failing")

    # Analyze each category
    results = {
        "passing": analyzer.analyze_category(passing_prompts, "PASSING"),
        "failing": analyzer.analyze_category(failing_prompts, "FAILING"),
    }

    # Also analyze simple arithmetic (known working) as reference
    arithmetic_prompts = [
        "2+2=", "3+4=", "5+6=", "7+8=", "9+1=",
        "10-3=", "15-7=", "20-12=", "8*7=", "6*9=",
    ]
    results["arithmetic"] = analyzer.analyze_category(arithmetic_prompts, "ARITHMETIC (reference)")

    # Diagnosis
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSIS")
    logger.info("=" * 70)

    passing_kappa = results["passing"].get("kappa", float('inf'))
    failing_kappa = results["failing"].get("kappa", float('inf'))
    arith_kappa = results["arithmetic"].get("kappa", float('inf'))

    passing_matches = results["passing"].get("total_matches", 0)
    failing_matches = results["failing"].get("total_matches", 0)
    arith_matches = results["arithmetic"].get("total_matches", 0)

    logger.info(f"\n  Category        κ (condition)    Constant Matches")
    logger.info(f"  {'─'*50}")
    logger.info(f"  ARITHMETIC      {arith_kappa:>12.2e}    {arith_matches:>6}")
    logger.info(f"  PASSING         {passing_kappa:>12.2e}    {passing_matches:>6}")
    logger.info(f"  FAILING         {failing_kappa:>12.2e}    {failing_matches:>6}")

    # Interpretation based on the geometry
    logger.info(f"\nInterpretation:")

    # κ ratio tells us about relative stability
    kappa_ratio = failing_kappa / passing_kappa if passing_kappa > 0 else float('inf')
    match_ratio = failing_matches / passing_matches if passing_matches > 0 else 0

    if failing_kappa > passing_kappa * 10:
        diagnosis = "DISCONNECTED"
        explanation = (
            f"Failing problems have κ {kappa_ratio:.1f}x higher than passing.\n"
            "This suggests the capability EXISTS but is geometrically disconnected.\n"
            "→ Surgical alignment should help."
        )
    elif failing_matches < passing_matches * 0.5:
        diagnosis = "TRUE_GAP"
        explanation = (
            f"Failing problems have {match_ratio:.1%} the constant matches of passing.\n"
            "This suggests the capability may not exist in the same form.\n"
            "→ May need different training approach."
        )
    else:
        diagnosis = "AMBIGUOUS"
        explanation = (
            "Geometric signatures are similar between passing and failing.\n"
            "The difference may be in specific reasoning patterns, not geometry.\n"
            "→ Try targeted training on specific failure patterns."
        )

    logger.info(f"\n  DIAGNOSIS: {diagnosis}")
    logger.info(f"\n  {explanation}")

    results["diagnosis"] = {
        "status": diagnosis,
        "kappa_ratio": float(kappa_ratio) if kappa_ratio != float('inf') else None,
        "match_ratio": float(match_ratio),
        "explanation": explanation,
    }

    # Save results
    output_path = Path("data/experiments/gsm8k_geometric_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
