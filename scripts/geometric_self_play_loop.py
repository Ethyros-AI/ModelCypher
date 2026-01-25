#!/usr/bin/env python3
"""Geometric Self-Play Loop.

The core idea: Let the model explore its own manifold, guided only by geometry.
No external labels. No human feedback. Just mathematical invariants.

Convergence criteria for "manifold completion":
1. Mean kurtosis stabilizes (variance < sqrt(eps) over N rounds)
2. Spectral entropy reaches minimum (no further compression possible)
3. Layer consistency approaches 1.0 (representation stable through depth)
4. Invariant score distribution separates into clusters (facts vs opinions)

The loop:
1. Generate diverse prompts via the model itself (self-prompted exploration)
2. Measure geometric signature of each prompt's representation
3. Identify "weak spots" - low invariant score, high entropy regions
4. Apply direction boosts that improve geometry without hurting accuracy
5. Track convergence metrics
6. Stop when manifold is "complete" (geometry stabilizes)

Usage:
    python geometric_self_play_loop.py --model /path/to/model
    python geometric_self_play_loop.py --model /path/to/model --max-rounds 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import numpy as np
from scipy.linalg import svd
from scipy.stats import kurtosis as scipy_kurtosis
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Iterator
import json
import pickle
import signal
from datetime import datetime
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Geometric Metrics (same as invariant_discovery)
# =============================================================================

def compute_kurtosis(activations: np.ndarray) -> float:
    flat = activations.flatten()
    if flat.std() < 1e-10:
        return 0.0
    return float(scipy_kurtosis(flat, fisher=True))


def compute_spectral_entropy(activations: np.ndarray) -> float:
    if activations.ndim == 1:
        activations = activations.reshape(1, -1)
    centered = activations - activations.mean(axis=0)
    try:
        _, S, _ = svd(centered, full_matrices=False)
        S_sum = S.sum()
        if S_sum < 1e-10:
            return 0.0
        S_norm = S / S_sum
        return -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
    except:
        return 0.0


def compute_effective_rank(activations: np.ndarray) -> float:
    return float(np.exp(compute_spectral_entropy(activations)))


# =============================================================================
# Convergence Tracking
# =============================================================================

@dataclass
class ConvergenceState:
    """Track convergence toward manifold completion."""

    # Rolling windows for stability detection
    kurtosis_history: deque = field(default_factory=lambda: deque(maxlen=20))
    entropy_history: deque = field(default_factory=lambda: deque(maxlen=20))
    score_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Convergence thresholds (derived from dtype)
    sqrt_eps: float = 1e-4

    # Best seen values
    best_mean_kurtosis: float = -np.inf
    best_min_entropy: float = np.inf
    best_mean_score: float = 0.0

    # Stagnation tracking
    rounds_without_improvement: int = 0

    def update(self, kurtosis: float, entropy: float, score: float):
        """Update with new round metrics."""
        self.kurtosis_history.append(kurtosis)
        self.entropy_history.append(entropy)
        self.score_history.append(score)

        improved = False

        if kurtosis > self.best_mean_kurtosis + self.sqrt_eps:
            self.best_mean_kurtosis = kurtosis
            improved = True

        if entropy < self.best_min_entropy - self.sqrt_eps:
            self.best_min_entropy = entropy
            improved = True

        if score > self.best_mean_score + self.sqrt_eps:
            self.best_mean_score = score
            improved = True

        if improved:
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

    @property
    def kurtosis_stable(self) -> bool:
        """Is kurtosis stable (variance below threshold)?"""
        if len(self.kurtosis_history) < 10:
            return False
        return np.std(list(self.kurtosis_history)) < self.sqrt_eps * 10

    @property
    def entropy_stable(self) -> bool:
        """Is spectral entropy stable?"""
        if len(self.entropy_history) < 10:
            return False
        return np.std(list(self.entropy_history)) < self.sqrt_eps * 10

    @property
    def score_stable(self) -> bool:
        """Is invariant score stable?"""
        if len(self.score_history) < 10:
            return False
        return np.std(list(self.score_history)) < self.sqrt_eps * 10

    @property
    def is_converged(self) -> bool:
        """Has the manifold converged?"""
        return self.kurtosis_stable and self.entropy_stable and self.score_stable

    def summary(self) -> Dict[str, Any]:
        """Get convergence summary."""
        return {
            'kurtosis_mean': np.mean(list(self.kurtosis_history)) if self.kurtosis_history else 0,
            'kurtosis_std': np.std(list(self.kurtosis_history)) if self.kurtosis_history else 0,
            'kurtosis_stable': self.kurtosis_stable,
            'entropy_mean': np.mean(list(self.entropy_history)) if self.entropy_history else 0,
            'entropy_std': np.std(list(self.entropy_history)) if self.entropy_history else 0,
            'entropy_stable': self.entropy_stable,
            'score_mean': np.mean(list(self.score_history)) if self.score_history else 0,
            'score_std': np.std(list(self.score_history)) if self.score_history else 0,
            'score_stable': self.score_stable,
            'is_converged': self.is_converged,
            'rounds_without_improvement': self.rounds_without_improvement,
        }


# =============================================================================
# Self-Prompted Exploration
# =============================================================================

class SelfPromptedExplorer:
    """Generate exploration prompts from the model itself."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Seed prompts to bootstrap exploration
        self.seed_prompts = [
            "The most important thing to know about",
            "A fact that most people don't realize is",
            "The mathematical relationship between",
            "If we consider the logical implications of",
            "The fundamental principle underlying",
            "One thing that is absolutely certain is",
            "The evidence strongly suggests that",
            "Based on first principles,",
        ]

        self.explored_prompts = set()

    def generate_prompts(self, n_prompts: int = 10) -> List[str]:
        """Generate prompts by having the model complete seed prompts."""
        from mlx_lm import generate

        prompts = []

        for seed in self.seed_prompts[:n_prompts]:
            try:
                completion = generate(
                    self.model,
                    self.tokenizer,
                    prompt=seed,
                    max_tokens=20,
                    verbose=False
                )
                full_prompt = seed + completion.split('.')[0]  # Take first sentence
                if full_prompt not in self.explored_prompts:
                    prompts.append(full_prompt)
                    self.explored_prompts.add(full_prompt)
            except:
                pass

        return prompts if prompts else self.seed_prompts[:n_prompts]


# =============================================================================
# Geometric Self-Play Loop
# =============================================================================

class GeometricSelfPlayLoop:
    """Main self-play loop for manifold completion."""

    def __init__(
        self,
        model,
        tokenizer,
        target_layers: List[int],
        config: Dict[str, Any]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.config = config

        self.explorer = SelfPromptedExplorer(model, tokenizer)
        self.convergence = ConvergenceState(
            sqrt_eps=config.get('sqrt_eps', 1e-4)
        )

        # Exploration parameters
        self.directions = config.get('directions', list(range(10)))
        self.boost_factors = config.get('boost_factors', [0.5, 0.7, 1.3, 1.5, 2.0])

        # State
        self.round_num = 0
        self.total_improvements = 0
        self.improvement_log = []

        # Interrupt handling
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Checkpoint
        self.checkpoint_dir = Path(config.get(
            'checkpoint_dir',
            Path(__file__).parent.parent / "data" / "geometric_self_play"
        ))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _handle_interrupt(self, signum, frame):
        logger.info("\n Interrupt received. Saving and exiting...")
        self.interrupted = True

    def get_layer_activations(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get MLP output activations for a prompt at a layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        captured = {}
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            original = layer.feed_forward
            key = 'feed_forward'
        else:
            original = layer.mlp
            key = 'mlp'

        class Hook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['input'] = x
                captured['output'] = self.mlp(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = self.model(input_ids)
            mx.eval(captured['input'], captured['output'])
            inp = np.array(captured['input'][0, -1, :].tolist(), dtype=np.float64)
            out = np.array(captured['output'][0, -1, :].tolist(), dtype=np.float64)
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

        return inp, out

    def compute_round_geometry(self, prompts: List[str]) -> Dict[str, float]:
        """Compute aggregate geometry metrics for a set of prompts."""
        kurtoses = []
        entropies = []

        mid_layer = self.target_layers[len(self.target_layers) // 2]

        for prompt in prompts:
            try:
                _, acts = self.get_layer_activations(prompt, mid_layer)
                kurtoses.append(compute_kurtosis(acts))
                entropies.append(compute_spectral_entropy(acts))
            except:
                pass

        if not kurtoses:
            return {'kurtosis': 0, 'entropy': 0, 'score': 0}

        mean_kurt = np.mean(kurtoses)
        mean_ent = np.mean(entropies)
        score = mean_kurt / 100 - mean_ent  # Same formula as autonomous_manifold_completion

        return {
            'kurtosis': mean_kurt,
            'entropy': mean_ent,
            'score': score,
        }

    def try_improvement(
        self,
        prompts: List[str],
        layer_idx: int,
        direction: int,
        boost: float
    ) -> Optional[Dict[str, Any]]:
        """Try a geometric improvement and return if successful."""
        import mlx.core as mx

        # Collect activations for all prompts
        inputs = []
        outputs = []

        for prompt in prompts:
            try:
                inp, out = self.get_layer_activations(prompt, layer_idx)
                inputs.append(inp)
                outputs.append(out)
            except:
                pass

        if len(inputs) < 3:
            return None

        S_X = np.stack(inputs)
        S_Y = np.stack(outputs)

        # Baseline geometry
        baseline_kurt = compute_kurtosis(S_Y)
        baseline_ent = compute_spectral_entropy(S_Y)
        baseline_score = baseline_kurt / 100 - baseline_ent

        # SVD for direction
        S_Y_centered = S_Y - S_Y.mean(axis=0)
        try:
            _, S, Vh = svd(S_Y_centered, full_matrices=False)
        except:
            return None

        if direction >= len(Vh):
            return None

        # Apply boost
        coefs = S_Y_centered @ Vh[direction]
        proj = np.outer(coefs, Vh[direction])
        Y_new = S_Y + proj * (boost - 1)

        # Check geometry improvement
        new_kurt = compute_kurtosis(Y_new)
        new_ent = compute_spectral_entropy(Y_new)
        new_score = new_kurt / 100 - new_ent

        if new_score <= baseline_score + 1e-4:
            return None

        # Compute replacement weight matrix
        S_X_scale = np.abs(S_X).max()
        Y_scale = np.abs(Y_new).max()
        if S_X_scale < 1e-10 or Y_scale < 1e-10:
            return None

        S_X_norm = S_X / S_X_scale
        Y_norm = Y_new / Y_scale
        reg = 1e-3
        ATA = S_X_norm.T @ S_X_norm + reg * np.eye(S_X_norm.shape[1])
        ATB = S_X_norm.T @ Y_norm

        try:
            W_norm, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)
        except:
            return None

        W = (W_norm * Y_scale / S_X_scale).T

        if np.isnan(W).any() or np.isinf(W).any():
            return None

        return {
            'layer': layer_idx,
            'direction': direction,
            'boost': boost,
            'W': W,
            'baseline_score': baseline_score,
            'new_score': new_score,
            'improvement': new_score - baseline_score,
            'new_kurtosis': new_kurt,
            'new_entropy': new_ent,
        }

    def apply_improvement(self, improvement: Dict[str, Any]):
        """Permanently apply a weight improvement."""
        import mlx.core as mx

        W = improvement['W']
        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class NewMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        layer = self.model.model.layers[improvement['layer']]
        if hasattr(layer, 'feed_forward'):
            layer.feed_forward = NewMLP(W_mx)
        else:
            layer.mlp = NewMLP(W_mx)

        self.total_improvements += 1
        self.improvement_log.append({
            'round': self.round_num,
            'layer': improvement['layer'],
            'direction': improvement['direction'],
            'boost': improvement['boost'],
            'score_improvement': improvement['improvement'],
            'timestamp': datetime.now().isoformat(),
        })

    def run_round(self) -> bool:
        """Run one round of self-play. Returns True if improved."""
        # Generate exploration prompts
        prompts = self.explorer.generate_prompts(n_prompts=10)

        # Compute baseline geometry
        baseline = self.compute_round_geometry(prompts)

        # Try improvements across layers and directions
        best = None

        for layer_idx in self.target_layers:
            if self.interrupted:
                break

            for direction in self.directions:
                if self.interrupted:
                    break

                for boost in self.boost_factors:
                    if self.interrupted:
                        break

                    candidate = self.try_improvement(
                        prompts, layer_idx, direction, boost
                    )

                    if candidate is not None:
                        if best is None or candidate['new_score'] > best['new_score']:
                            best = candidate

        # Apply best improvement if found
        if best is not None:
            self.apply_improvement(best)
            logger.info(
                f"  IMPROVED: L{best['layer']} d{best['direction']} b{best['boost']:.1f} "
                f"score: {best['baseline_score']:.4f} -> {best['new_score']:.4f} "
                f"(+{best['improvement']:.4f})"
            )
            return True

        return False

    def run(self, max_rounds: Optional[int] = None, max_stagnant: int = 30):
        """Main loop."""
        logger.info("=" * 80)
        logger.info("GEOMETRIC SELF-PLAY LOOP")
        logger.info("=" * 80)
        logger.info(f"Target layers: {self.target_layers}")
        logger.info(f"Directions: {len(self.directions)}, Boosts: {len(self.boost_factors)}")
        logger.info(f"Max stagnant rounds: {max_stagnant}")

        # Initial geometry
        prompts = self.explorer.generate_prompts(10)
        initial_geo = self.compute_round_geometry(prompts)
        logger.info(f"\nInitial geometry:")
        logger.info(f"  Kurtosis: {initial_geo['kurtosis']:.4f}")
        logger.info(f"  Entropy:  {initial_geo['entropy']:.4f}")
        logger.info(f"  Score:    {initial_geo['score']:.4f}")

        while not self.interrupted:
            self.round_num += 1

            # Check limits
            if max_rounds and self.round_num > max_rounds:
                logger.info(f"\nReached max rounds ({max_rounds})")
                break

            if self.convergence.rounds_without_improvement >= max_stagnant:
                logger.info(f"\nStagnated for {max_stagnant} rounds")
                break

            if self.convergence.is_converged:
                logger.info("\n MANIFOLD CONVERGED!")
                break

            logger.info(f"\nRound {self.round_num}:")

            # Run round
            improved = self.run_round()

            # Update convergence tracking
            prompts = self.explorer.generate_prompts(5)
            geo = self.compute_round_geometry(prompts)
            self.convergence.update(geo['kurtosis'], geo['entropy'], geo['score'])

            if not improved:
                logger.info(f"  No improvement (stagnant: {self.convergence.rounds_without_improvement})")

            # Log convergence state
            conv = self.convergence.summary()
            logger.info(
                f"  Geometry: kurt={conv['kurtosis_mean']:.3f}±{conv['kurtosis_std']:.3f} "
                f"ent={conv['entropy_mean']:.3f}±{conv['entropy_std']:.3f}"
            )

        # Final report
        self.report_final()

    def report_final(self):
        """Print final report."""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL REPORT")
        logger.info("=" * 80)

        prompts = self.explorer.generate_prompts(20)
        final_geo = self.compute_round_geometry(prompts)

        logger.info(f"\nFinal geometry:")
        logger.info(f"  Kurtosis: {final_geo['kurtosis']:.4f}")
        logger.info(f"  Entropy:  {final_geo['entropy']:.4f}")
        logger.info(f"  Score:    {final_geo['score']:.4f}")

        logger.info(f"\nTotal rounds: {self.round_num}")
        logger.info(f"Total improvements: {self.total_improvements}")

        conv = self.convergence.summary()
        logger.info(f"\nConvergence state:")
        logger.info(f"  Kurtosis stable: {conv['kurtosis_stable']}")
        logger.info(f"  Entropy stable:  {conv['entropy_stable']}")
        logger.info(f"  Score stable:    {conv['score_stable']}")
        logger.info(f"  CONVERGED:       {conv['is_converged']}")

        # Save results
        output_path = self.checkpoint_dir / "final_results.json"
        output = {
            'timestamp': datetime.now().isoformat(),
            'rounds': self.round_num,
            'improvements': self.total_improvements,
            'final_geometry': final_geo,
            'convergence': conv,
            'improvement_log': self.improvement_log,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Geometric Self-Play Loop")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="4,6,8,10,12,14",
        help="Layer indices to explore"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Max rounds (None = until convergence)"
    )
    parser.add_argument(
        "--max-stagnant",
        type=int,
        default=30,
        help="Max stagnant rounds before stopping"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode for testing"
    )
    args = parser.parse_args()

    # Parse layers
    target_layers = [int(x) for x in args.layers.split(',')]

    # Load model
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Config
    if args.fast:
        config = {
            'directions': list(range(5)),
            'boost_factors': [0.5, 1.5, 2.0],
            'sqrt_eps': 1e-3,
        }
        target_layers = target_layers[:3]
    else:
        config = {
            'directions': list(range(10)),
            'boost_factors': [0.5, 0.7, 1.3, 1.5, 2.0],
            'sqrt_eps': 1e-4,
        }

    # Run
    loop = GeometricSelfPlayLoop(model, tokenizer, target_layers, config)
    loop.run(max_rounds=args.max_rounds, max_stagnant=args.max_stagnant)


if __name__ == "__main__":
    main()
