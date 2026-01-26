#!/usr/bin/env python3
"""Run Surgical-Only Alignment (No Thinking Phase).

This is the ablation version of run_iterative_learning.py.
It removes all generation/thinking and just loops surgical SVD alignment.

If results match run_iterative_learning.py → thinking phase is placebo
If results differ → thinking phase contributes something

Usage:
    poetry run python scripts/run_surgical_only.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --iterations 10 \
        --output data/ablation/surgical_only.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CONSTANTS = {
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "1/phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "sqrt3": np.sqrt(3),
}

# Quality tests (same as iterative learning)
TEST_PROMPTS = [
    ("What is 2 + 2?", "4"),
    ("Capital of France?", "paris"),
    ("Is water wet?", "yes"),
    ("What color is the sky?", "blue"),
    ("Are dogs mammals?", "yes"),
]


@dataclass
class IterationResult:
    iteration: int
    matches_before: int
    matches_after: int
    targets_aligned: int
    quality: float


@dataclass
class SurgicalOnlyResult:
    total_iterations: int
    initial_matches: int
    final_matches: int
    initial_quality: float
    final_quality: float
    history: List[IterationResult]


class SurgicalOnlyLoop:
    """Pure surgical alignment loop - no thinking/generation."""

    def __init__(
        self,
        model,
        tokenizer,
        proximity_threshold: float = 0.10,
        quality_threshold: float = 0.90,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.proximity_threshold = proximity_threshold
        self.quality_threshold = quality_threshold
        self.n_layers = len(model.model.layers)

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

    def _count_matches(self, S: np.ndarray) -> int:
        count = 0
        for i in range(min(len(S) - 1, 20)):
            for j in range(i + 1, min(len(S), i + 6)):
                if S[j] > 1e-10:
                    ratio = S[i] / S[j]
                    for const_val in CONSTANTS.values():
                        if abs(ratio - const_val) / const_val < 0.05:
                            count += 1
                            break
        return count

    def _count_total_matches(self, layer_indices: List[int]) -> int:
        total = 0
        for layer_idx in layer_indices:
            W = self._get_mlp_weight(layer_idx)
            _, S, _ = svd(W, full_matrices=False)
            total += self._count_matches(S)
        return total

    def _surgical_align_layer(self, layer_idx: int, max_targets: int = 2) -> int:
        """Apply surgical SVD alignment to a layer. Returns targets aligned."""
        W = self._get_mlp_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)

        min_sv = S[0] * 1e-6
        targets = []

        for i in range(min(len(S) - 1, 15)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > max(1e-10, min_sv):
                    ratio = S[i] / S[j]

                    for const_name, const_val in CONSTANTS.items():
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

    def run(
        self,
        test_prompts: List[Tuple[str, str]],
        n_iterations: int = 10,
        layer_indices: Optional[List[int]] = None,
    ) -> SurgicalOnlyResult:
        """Run pure surgical alignment loop - NO thinking phase."""

        if layer_indices is None:
            mid = self.n_layers // 2
            layer_indices = list(range(mid - 3, mid + 4))

        logger.info("\n" + "=" * 60)
        logger.info("SURGICAL-ONLY ALIGNMENT (NO THINKING)")
        logger.info(f"Iterations: {n_iterations}")
        logger.info(f"Layers: {layer_indices}")
        logger.info("=" * 60)

        # Initial state
        initial_quality = self._evaluate_quality(test_prompts)
        initial_matches = self._count_total_matches(layer_indices)

        logger.info(f"\nInitial state:")
        logger.info(f"  Quality: {initial_quality:.2%}")
        logger.info(f"  Matches: {initial_matches}")

        history = []

        for iteration in range(n_iterations):
            logger.info(f"\n--- Iteration {iteration + 1} ---")

            # NO THINKING - just surgical alignment
            matches_before = self._count_total_matches(layer_indices)
            total_aligned = 0

            for layer_idx in layer_indices:
                aligned = self._surgical_align_layer(layer_idx, max_targets=2)
                total_aligned += aligned

            matches_after = self._count_total_matches(layer_indices)
            logger.info(f"  Locking: {total_aligned} targets aligned")
            logger.info(f"  Matches: {matches_before} → {matches_after}")

            quality = self._evaluate_quality(test_prompts)
            logger.info(f"  Quality: {quality:.2%}")

            if quality < initial_quality * self.quality_threshold:
                logger.info(f"  Quality degraded below threshold, stopping")
                break

            history.append(IterationResult(
                iteration=iteration + 1,
                matches_before=matches_before,
                matches_after=matches_after,
                targets_aligned=total_aligned,
                quality=quality,
            ))

        # Final state
        final_quality = self._evaluate_quality(test_prompts)
        final_matches = self._count_total_matches(layer_indices)

        logger.info(f"\n{'=' * 60}")
        logger.info("FINAL RESULTS (SURGICAL ONLY)")
        logger.info(f"{'=' * 60}")
        logger.info(f"Quality: {initial_quality:.2%} → {final_quality:.2%}")
        logger.info(f"Matches: {initial_matches} → {final_matches}")

        return SurgicalOnlyResult(
            total_iterations=len(history),
            initial_matches=initial_matches,
            final_matches=final_matches,
            initial_quality=initial_quality,
            final_quality=final_quality,
            history=history,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--proximity", type=float, default=0.10)
    parser.add_argument("--quality-threshold", type=float, default=0.90)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    from mlx_lm import load

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    loop = SurgicalOnlyLoop(
        model=model,
        tokenizer=tokenizer,
        proximity_threshold=args.proximity,
        quality_threshold=args.quality_threshold,
    )

    result = loop.run(
        test_prompts=TEST_PROMPTS,
        n_iterations=args.iterations,
    )

    # Save results
    output_path = args.output or f"data/ablation/surgical_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "experiment": "surgical_only_ablation",
        "n_iterations": result.total_iterations,
        "initial_matches": result.initial_matches,
        "final_matches": result.final_matches,
        "initial_quality": result.initial_quality,
        "final_quality": result.final_quality,
        "history": [
            {
                "iteration": h.iteration,
                "matches_before": h.matches_before,
                "matches_after": h.matches_after,
                "targets_aligned": h.targets_aligned,
                "quality": h.quality,
            }
            for h in result.history
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    if result.final_matches > result.initial_matches:
        improvement = (result.final_matches - result.initial_matches) / result.initial_matches * 100
        logger.info(f"\nSUCCESS: Matches improved {result.initial_matches} → {result.final_matches} ({improvement:.1f}%)")
    else:
        logger.info(f"\nNo improvement in matches")


if __name__ == "__main__":
    main()
