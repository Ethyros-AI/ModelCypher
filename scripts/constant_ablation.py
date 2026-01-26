#!/usr/bin/env python3
"""Constant Ablation Study - Which Constants Matter?

Tests each constant family separately to determine which ones
drive the quality improvement.

Families tested:
1. π/e family: {π/e, e/π}
2. Golden ratio: {φ, 1/φ}
3. Roots: {√2, 1/√2, √3}
4. All constants (baseline)

Usage:
    poetry run python scripts/constant_ablation.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --iterations 10 \
        --output data/ablation/constant_families.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# All constants
ALL_CONSTANTS = {
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "1/phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "1/sqrt2": 1 / np.sqrt(2),
    "sqrt3": np.sqrt(3),
}

# Constant families to test
FAMILIES = {
    "pi_e": {"pi/e", "e/pi"},
    "phi": {"phi", "1/phi"},
    "roots": {"sqrt2", "1/sqrt2", "sqrt3"},
    "all": set(ALL_CONSTANTS.keys()),
}

# Quality tests
TEST_PROMPTS = [
    ("What is 2 + 2?", "4"),
    ("Capital of France?", "paris"),
    ("Is water wet?", "yes"),
    ("What color is the sky?", "blue"),
    ("Are dogs mammals?", "yes"),
]


@dataclass
class FamilyResult:
    family_name: str
    constants_used: List[str]
    initial_matches: int
    final_matches: int
    initial_quality: float
    final_quality: float
    iterations_to_converge: int
    trajectory: List[Tuple[int, float]]  # (matches, quality) per iteration


class ConstantAblation:
    """Test surgical alignment with different constant subsets."""

    def __init__(
        self,
        model,
        tokenizer,
        proximity_threshold: float = 0.10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.proximity_threshold = proximity_threshold
        self.n_layers = len(model.model.layers)
        self._original_weights = {}  # Cache for reset

    def _get_mlp_weight(self, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward
        else:
            mlp = layer.mlp

        if hasattr(mlp, 'gate_proj'):
            w = mlp.gate_proj.weight
        elif hasattr(mlp, 'w1'):
            w = mlp.w1.weight
        else:
            w = mlp.weight

        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_mlp_weight(self, layer_idx: int, weights: np.ndarray):
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward
        else:
            mlp = layer.mlp

        new_weight = mx.array(weights.astype(np.float32))

        if hasattr(mlp, 'gate_proj'):
            mlp.gate_proj.weight = new_weight
        elif hasattr(mlp, 'w1'):
            mlp.w1.weight = new_weight
        else:
            mlp.weight = new_weight

        mx.eval(new_weight)

    def _cache_weights(self, layer_indices: List[int]):
        """Cache original weights for reset."""
        self._original_weights = {}
        for layer_idx in layer_indices:
            self._original_weights[layer_idx] = self._get_mlp_weight(layer_idx).copy()

    def _reset_weights(self, layer_indices: List[int]):
        """Reset to cached original weights."""
        for layer_idx in layer_indices:
            if layer_idx in self._original_weights:
                self._set_mlp_weight(layer_idx, self._original_weights[layer_idx])

    def _count_matches(self, S: np.ndarray, constant_subset: Set[str]) -> int:
        """Count matches for a specific subset of constants."""
        count = 0
        subset_values = {ALL_CONSTANTS[k] for k in constant_subset if k in ALL_CONSTANTS}

        for i in range(min(len(S) - 1, 20)):
            for j in range(i + 1, min(len(S), i + 6)):
                if S[j] > 1e-10:
                    ratio = S[i] / S[j]
                    for const_val in subset_values:
                        if abs(ratio - const_val) / const_val < 0.05:
                            count += 1
                            break
        return count

    def _count_total_matches(self, layer_indices: List[int], constant_subset: Set[str]) -> int:
        total = 0
        for layer_idx in layer_indices:
            W = self._get_mlp_weight(layer_idx)
            _, S, _ = svd(W, full_matrices=False)
            total += self._count_matches(S, constant_subset)
        return total

    def _surgical_align_layer(
        self,
        layer_idx: int,
        constant_subset: Set[str],
        max_targets: int = 2,
    ) -> int:
        """Apply surgical alignment for specific constants only."""
        W = self._get_mlp_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)

        min_sv = S[0] * 1e-6
        targets = []

        # Only target the specified constants
        subset_constants = {k: ALL_CONSTANTS[k] for k in constant_subset if k in ALL_CONSTANTS}

        for i in range(min(len(S) - 1, 15)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > max(1e-10, min_sv):
                    ratio = S[i] / S[j]

                    for const_name, const_val in subset_constants.items():
                        error = abs(ratio - const_val) / const_val
                        if error < self.proximity_threshold:
                            targets.append((i, j, const_val))
                            break

        if not targets:
            return 0

        S_modified = S.copy()
        aligned = 0

        for i, j, target_val in targets[:max_targets]:
            if S_modified[j] < min_sv:
                continue
            new_val = target_val * S_modified[j]
            if new_val > S[0] * 10 or new_val < min_sv:
                continue
            S_modified[i] = new_val
            aligned += 1

        if aligned > 0:
            if not np.all(np.isfinite(S_modified)):
                return 0
            W_modified = U @ np.diag(S_modified) @ Vt
            if not np.all(np.isfinite(W_modified)):
                return 0
            self._set_mlp_weight(layer_idx, W_modified)

        return aligned

    def _evaluate_quality(self, test_prompts: List[Tuple[str, str]]) -> float:
        import mlx.core as mx

        correct = 0
        for prompt, expected in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            generated = []
            current = input_ids
            for _ in range(20):
                logits = self.model(current)
                mx.eval(logits)
                next_token = int(mx.argmax(logits[0, -1, :]).item())
                if next_token == self.tokenizer.eos_token_id:
                    break
                generated.append(next_token)
                current = mx.concatenate([current, mx.array([[next_token]])], axis=1)

            response = self.tokenizer.decode(generated).lower()
            if expected.lower() in response:
                correct += 1

        return correct / len(test_prompts) if test_prompts else 1.0

    def test_family(
        self,
        family_name: str,
        constant_subset: Set[str],
        n_iterations: int,
        layer_indices: List[int],
    ) -> FamilyResult:
        """Test a single constant family."""

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing family: {family_name}")
        logger.info(f"Constants: {sorted(constant_subset)}")
        logger.info(f"{'='*60}")

        # Reset to original weights
        self._reset_weights(layer_indices)

        # Initial state
        initial_quality = self._evaluate_quality(TEST_PROMPTS)
        initial_matches = self._count_total_matches(layer_indices, constant_subset)

        logger.info(f"Initial: {initial_matches} matches, {initial_quality:.0%} quality")

        trajectory = []
        converged_at = n_iterations

        for iteration in range(n_iterations):
            # Surgical alignment with this family only
            for layer_idx in layer_indices:
                self._surgical_align_layer(layer_idx, constant_subset, max_targets=2)

            matches = self._count_total_matches(layer_indices, constant_subset)
            quality = self._evaluate_quality(TEST_PROMPTS)

            trajectory.append((matches, quality))

            # Check convergence
            if len(trajectory) >= 2 and trajectory[-1][0] == trajectory[-2][0]:
                if converged_at == n_iterations:
                    converged_at = iteration + 1

            logger.info(f"  Iter {iteration+1}: {matches} matches, {quality:.0%} quality")

        final_matches = trajectory[-1][0] if trajectory else initial_matches
        final_quality = trajectory[-1][1] if trajectory else initial_quality

        logger.info(f"Final: {final_matches} matches, {final_quality:.0%} quality")
        logger.info(f"Change: {final_matches - initial_matches:+d} matches, {(final_quality - initial_quality)*100:+.0f}% quality")

        return FamilyResult(
            family_name=family_name,
            constants_used=sorted(constant_subset),
            initial_matches=initial_matches,
            final_matches=final_matches,
            initial_quality=initial_quality,
            final_quality=final_quality,
            iterations_to_converge=converged_at,
            trajectory=trajectory,
        )

    def run(
        self,
        n_iterations: int = 10,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, FamilyResult]:
        """Run ablation study on all families."""

        if layer_indices is None:
            mid = self.n_layers // 2
            layer_indices = list(range(mid - 3, mid + 4))

        # Cache original weights
        self._cache_weights(layer_indices)

        results = {}

        for family_name, constant_subset in FAMILIES.items():
            result = self.test_family(
                family_name=family_name,
                constant_subset=constant_subset,
                n_iterations=n_iterations,
                layer_indices=layer_indices,
            )
            results[family_name] = result

        # Reset to original at end
        self._reset_weights(layer_indices)

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--proximity", type=float, default=0.10)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    from mlx_lm import load

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    ablation = ConstantAblation(
        model=model,
        tokenizer=tokenizer,
        proximity_threshold=args.proximity,
    )

    results = ablation.run(n_iterations=args.iterations)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Family':<12} {'Matches':>12} {'Quality':>12} {'Δ Quality':>12}")
    logger.info("-" * 50)

    for family_name, result in sorted(results.items(), key=lambda x: x[1].final_quality - x[1].initial_quality, reverse=True):
        delta_q = (result.final_quality - result.initial_quality) * 100
        logger.info(
            f"{family_name:<12} "
            f"{result.initial_matches}→{result.final_matches:>3} "
            f"{result.initial_quality:.0%}→{result.final_quality:.0%} "
            f"{delta_q:>+10.0f}%"
        )

    # Save results
    output_path = args.output or f"data/ablation/constant_families_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "experiment": "constant_ablation",
        "families": {
            name: {
                "constants_used": result.constants_used,
                "initial_matches": result.initial_matches,
                "final_matches": result.final_matches,
                "initial_quality": result.initial_quality,
                "final_quality": result.final_quality,
                "quality_improvement": result.final_quality - result.initial_quality,
                "iterations_to_converge": result.iterations_to_converge,
                "trajectory": [{"matches": m, "quality": q} for m, q in result.trajectory],
            }
            for name, result in results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
