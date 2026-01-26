#!/usr/bin/env python3
"""Experiment 50: Internal Relational Structure Analysis.

Compare the relational structure of math vs non-math WITHIN the same model.

The hypothesis: If math is corrupted at the relational level, the Gram matrix
of math operations should have different structural properties than non-math.

We test:
1. Gram matrix eigenvalue spectrum (does math have different "shape"?)
2. Effective dimensionality (is math compressed into fewer/more dimensions?)
3. Clustering structure (do math prompts cluster differently?)
4. Self-consistency (does similar math have similar representations?)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Math prompts with known relationships
# These should have predictable relational structure if math works
MATH_PROMPTS = {
    # Addition family - these should cluster together
    "add_1": ["1+1=", "1+2=", "1+3=", "1+4=", "1+5="],
    # Multiplication family - these should cluster together
    "mult_2": ["2×1=", "2×2=", "2×3=", "2×4=", "2×5="],
    # Subtraction family
    "sub_from_10": ["10-1=", "10-2=", "10-3=", "10-4=", "10-5="],
    # Mixed
    "equals_6": ["3+3=", "2×3=", "6÷1=", "12÷2=", "6-0="],  # All should equal 6
}

# Non-math prompts with known relationships
NON_MATH_PROMPTS = {
    # Color family - should cluster
    "colors": ["red is a", "blue is a", "green is a", "yellow is a", "purple is a"],
    # Animal family - should cluster
    "animals": ["dog is a", "cat is a", "bird is a", "fish is a", "horse is a"],
    # Geographic - should cluster
    "capitals": ["Paris is the capital of", "London is the capital of",
                 "Tokyo is the capital of", "Rome is the capital of", "Berlin is the capital of"],
    # Natural - should cluster
    "nature": ["sun is", "moon is", "stars are", "sky is", "ocean is"],
}


class InternalStructureAnalyzer:
    """Analyze relational structure within a single model."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def _get_activations(self, prompts: List[str]) -> np.ndarray:
        """Get activations for all prompts."""
        import mlx.core as mx

        activations = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)
            act = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            activations.append(act)

        return np.vstack(activations)

    def _compute_gram_properties(self, acts: np.ndarray) -> Dict:
        """Compute properties of the Gram matrix."""
        # Gram matrix: relational structure
        G = acts @ acts.T

        # Normalize
        G_norm = G / (np.linalg.norm(G) + 1e-10)

        # Eigenvalues
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(eigvals)[::-1]  # Descending

        # Effective dimensionality
        eigvals_pos = eigvals[eigvals > 1e-10]
        eigvals_norm = eigvals_pos / eigvals_pos.sum()
        entropy = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-10))
        effective_dim = np.exp(entropy)

        # Spectral decay rate
        if len(eigvals_pos) > 1:
            decay_rate = eigvals_pos[0] / eigvals_pos[1] if eigvals_pos[1] > 0 else 0
        else:
            decay_rate = 0

        # Concentration ratio (how much in top eigenvalue)
        concentration = eigvals_pos[0] / eigvals_pos.sum() if eigvals_pos.sum() > 0 else 0

        return {
            "effective_dim": float(effective_dim),
            "decay_rate": float(decay_rate),
            "concentration": float(concentration),
            "top_5_eigenvalues": eigvals[:5].tolist(),
            "gram_trace": float(np.trace(G)),
            "gram_frobenius": float(np.linalg.norm(G, 'fro')),
        }

    def _compute_family_coherence(self, prompts_dict: Dict[str, List[str]]) -> Dict:
        """Check if families cluster together (within-family similarity > between-family)."""
        family_acts = {}
        for family, prompts in prompts_dict.items():
            family_acts[family] = self._get_activations(prompts)

        # Compute within-family and between-family similarities
        within_sims = []
        between_sims = []

        families = list(family_acts.keys())
        for i, fam1 in enumerate(families):
            acts1 = family_acts[fam1]
            # Within-family: similarity between members of same family
            G_within = acts1 @ acts1.T
            n = G_within.shape[0]
            # Off-diagonal elements (similarity between different members)
            for j in range(n):
                for k in range(j+1, n):
                    within_sims.append(G_within[j, k])

            # Between-family: similarity to other families
            for fam2 in families[i+1:]:
                acts2 = family_acts[fam2]
                G_between = acts1 @ acts2.T
                between_sims.extend(G_between.flatten().tolist())

        mean_within = np.mean(within_sims) if within_sims else 0
        mean_between = np.mean(between_sims) if between_sims else 0
        coherence = mean_within - mean_between

        return {
            "mean_within_family_sim": float(mean_within),
            "mean_between_family_sim": float(mean_between),
            "coherence": float(coherence),
            "within_std": float(np.std(within_sims)) if within_sims else 0,
            "between_std": float(np.std(between_sims)) if between_sims else 0,
        }

    def _check_mathematical_consistency(self) -> Dict:
        """Check if mathematically equivalent expressions have similar representations."""
        # All these should equal the same value
        equivalent_groups = [
            # All = 4
            (["2+2=", "1+3=", "4×1=", "8÷2=", "4-0="], 4),
            # All = 6
            (["3+3=", "2×3=", "12÷2=", "6-0=", "1+5="], 6),
            # All = 10
            (["5+5=", "2×5=", "20÷2=", "10-0=", "7+3="], 10),
        ]

        consistencies = []
        for prompts, expected_value in equivalent_groups:
            acts = self._get_activations(prompts)
            G = acts @ acts.T

            # Within-group similarity (should be HIGH if math is consistent)
            n = G.shape[0]
            sims = []
            for i in range(n):
                for j in range(i+1, n):
                    # Normalize
                    sim = G[i,j] / (np.sqrt(G[i,i] * G[j,j]) + 1e-10)
                    sims.append(sim)

            mean_sim = np.mean(sims) if sims else 0
            consistencies.append({
                "expected_value": expected_value,
                "mean_similarity": float(mean_sim),
                "min_similarity": float(min(sims)) if sims else 0,
                "max_similarity": float(max(sims)) if sims else 0,
            })

        overall_consistency = np.mean([c["mean_similarity"] for c in consistencies])

        return {
            "groups": consistencies,
            "overall_consistency": float(overall_consistency),
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 50: INTERNAL RELATIONAL STRUCTURE")
        logger.info("=" * 60)

        # Flatten prompts for overall analysis
        math_all = [p for prompts in MATH_PROMPTS.values() for p in prompts]
        non_math_all = [p for prompts in NON_MATH_PROMPTS.values() for p in prompts]

        # Get activations
        logger.info("\nGetting math activations...")
        math_acts = self._get_activations(math_all)
        logger.info(f"  Shape: {math_acts.shape}")

        logger.info("\nGetting non-math activations...")
        non_math_acts = self._get_activations(non_math_all)
        logger.info(f"  Shape: {non_math_acts.shape}")

        # Compute Gram properties
        logger.info("\nComputing Gram matrix properties...")
        math_gram = self._compute_gram_properties(math_acts)
        non_math_gram = self._compute_gram_properties(non_math_acts)

        logger.info(f"\n| Metric | Math | Non-Math |")
        logger.info(f"|--------|------|----------|")
        logger.info(f"| Effective dim | {math_gram['effective_dim']:.2f} | {non_math_gram['effective_dim']:.2f} |")
        logger.info(f"| Decay rate | {math_gram['decay_rate']:.2f} | {non_math_gram['decay_rate']:.2f} |")
        logger.info(f"| Concentration | {math_gram['concentration']:.3f} | {non_math_gram['concentration']:.3f} |")

        # Family coherence
        logger.info("\nComputing family coherence...")
        math_coherence = self._compute_family_coherence(MATH_PROMPTS)
        non_math_coherence = self._compute_family_coherence(NON_MATH_PROMPTS)

        logger.info(f"\n| Metric | Math | Non-Math |")
        logger.info(f"|--------|------|----------|")
        logger.info(f"| Within-family sim | {math_coherence['mean_within_family_sim']:.2f} | {non_math_coherence['mean_within_family_sim']:.2f} |")
        logger.info(f"| Between-family sim | {math_coherence['mean_between_family_sim']:.2f} | {non_math_coherence['mean_between_family_sim']:.2f} |")
        logger.info(f"| Coherence | {math_coherence['coherence']:.2f} | {non_math_coherence['coherence']:.2f} |")

        # Mathematical consistency
        logger.info("\nChecking mathematical consistency...")
        math_consistency = self._check_mathematical_consistency()
        logger.info(f"  Overall consistency: {math_consistency['overall_consistency']:.3f}")
        for group in math_consistency['groups']:
            logger.info(f"    Equals {group['expected_value']}: {group['mean_similarity']:.3f} similarity")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        # Compare effective dimensionality
        dim_diff = math_gram['effective_dim'] - non_math_gram['effective_dim']
        if abs(dim_diff) > 1:
            logger.info(f"\n1. DIMENSIONALITY DIFFERENCE: {dim_diff:+.2f}")
            if dim_diff > 0:
                logger.info("   Math uses MORE dimensions than non-math (scattered representation)")
            else:
                logger.info("   Math uses FEWER dimensions than non-math (compressed representation)")
        else:
            logger.info(f"\n1. Similar dimensionality (diff: {dim_diff:.2f})")

        # Compare coherence
        coherence_diff = math_coherence['coherence'] - non_math_coherence['coherence']
        logger.info(f"\n2. COHERENCE DIFFERENCE: {coherence_diff:+.2f}")
        if coherence_diff < 0:
            logger.info("   Math families are LESS coherent than non-math")
            logger.info("   (Similar math doesn't cluster together)")
        else:
            logger.info("   Math families are more coherent than non-math")

        # Mathematical consistency
        logger.info(f"\n3. MATHEMATICAL CONSISTENCY: {math_consistency['overall_consistency']:.3f}")
        if math_consistency['overall_consistency'] < 0.5:
            logger.info("   LOW - Equivalent expressions have DIFFERENT representations")
            logger.info("   This is the relational structure corruption!")
        else:
            logger.info("   HIGH - Equivalent expressions have similar representations")

        # Conclusion
        if math_coherence['coherence'] < non_math_coherence['coherence'] or math_consistency['overall_consistency'] < 0.5:
            conclusion = "math_relationally_corrupted"
            logger.info("\n*** MATH RELATIONAL STRUCTURE IS CORRUPTED ***")
            logger.info("The model's internal representation of math operations")
            logger.info("does not preserve mathematical relationships.")
            logger.info("2+2 and 1+3 and 4×1 should be nearby, but they're not.")
        else:
            conclusion = "structure_intact"
            logger.info("\n*** RELATIONAL STRUCTURE APPEARS INTACT ***")

        results = {
            "math_gram": math_gram,
            "non_math_gram": non_math_gram,
            "math_coherence": math_coherence,
            "non_math_coherence": non_math_coherence,
            "math_consistency": math_consistency,
            "conclusion": conclusion,
        }

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = InternalStructureAnalyzer(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/internal_structure_comparison.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
