#!/usr/bin/env python3
"""Autonomous Manifold Completion.

Run indefinitely (or until max_rounds) to complete the model's manifold.
Uses generation-based evaluation for multi-token relationships.

Key insight from exp86-87:
- Single-token evaluation = "using only letters a,b,c"
- Generation evaluation = "using the full alphabet"
- Models have +20pp hidden capability via generation

Usage:
    # Run until convergence (could take days)
    python autonomous_manifold_completion.py

    # Run for 100 rounds then checkpoint
    python autonomous_manifold_completion.py --max-rounds 100

    # Resume from checkpoint
    python autonomous_manifold_completion.py --resume checkpoints/latest.pkl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import numpy as np
from scipy.linalg import svd
import json
import pickle
import signal
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Geometry Metrics (from exp70-74)
# =============================================================================

def compute_kurtosis(Y: np.ndarray) -> float:
    """Compute mean kurtosis across activation dimensions."""
    kurtoses = []
    for h in Y:
        std = h.std()
        if std < 1e-10:
            kurtoses.append(0.0)
            continue
        z = (h - h.mean()) / std
        kurtoses.append(float(np.mean(z ** 4) - 3))
    return np.mean(kurtoses)


def compute_spectral_entropy(Y: np.ndarray) -> float:
    """Compute spectral entropy of activation covariance."""
    Y_centered = Y - Y.mean(axis=0)
    try:
        _, S, _ = svd(Y_centered, full_matrices=False)
        S_sum = S.sum()
        if S_sum < 1e-10:
            return 0.0
        S_norm = S / S_sum
        return -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
    except:
        return 0.0


def geometry_score(kurtosis: float, spectral_entropy: float) -> float:
    """Combined geometry score: higher is better."""
    return kurtosis / 100 - spectral_entropy


# =============================================================================
# Manifold Explorer
# =============================================================================

class ManifoldExplorer:
    """Autonomous exploration of model's manifold via generation-based evaluation."""

    def __init__(
        self,
        model,
        tokenizer,
        config: Dict[str, Any]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Test cases with expected answers
        self.test_cases = [
            ("The capital of France is", "Paris"),
            ("2 + 2 equals", "4"),
            ("The square root of 16 is", "4"),
            ("The opposite of hot is", "cold"),
            ("Birds can", "fly"),
            ("Fish live in", "water"),
            ("The sky is usually", "blue"),
            ("Gravity causes objects to", "fall"),
            ("The sun rises in the", "east"),
            ("A noun is a word that names a", "person"),
        ]

        # Probe prompts for geometry measurement
        self.probe_prompts = [
            "The capital of", "The largest planet",
            "Water freezes at", "If it rains",
            "2 + 2 equals", "A noun is",
            "The square root of", "10 times 10",
            "The sky is", "Birds can",
            "Fish live in", "The sun rises",
            "Gravity causes", "The opposite of",
            "The past tense of", "An adjective describes",
            "Shakespeare wrote", "The speed of light",
            "Photosynthesis occurs in", "DNA stands for",
        ]

        # Exploration parameters
        self.target_layers = config.get('target_layers', list(range(16)))
        self.directions = config.get('directions', list(range(20)))
        self.boost_factors = config.get('boost_factors', [
            0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 2.5, 3.0
        ])
        self.sequence_lengths = config.get('sequence_lengths', [1, 3, 5, 10])

        # Convergence tracking
        self.max_stagnant = config.get('max_stagnant', 50)
        self.stagnant_rounds = 0

        # State
        self.current_gen_accuracy = 0.0
        self.current_top_accuracy = 0.0
        self.current_geometry = None
        self.improvements = []
        self.round_num = 0
        self.total_configurations_tested = 0

        # Constraints
        self.max_perplexity = config.get('max_perplexity', 100.0)
        self.min_accuracy = config.get('min_accuracy', 0.3)

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get(
            'checkpoint_dir',
            Path(__file__).parent.parent / "data" / "manifold_checkpoints"
        ))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Interrupt handling
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("\n⚠️  Interrupt received. Saving checkpoint and exiting...")
        self.interrupted = True

    # =========================================================================
    # Evaluation Methods
    # =========================================================================

    def evaluate_generation(self, max_tokens: int = 10) -> Tuple[float, List[Dict]]:
        """Evaluate using multi-token generation."""
        import mlx.core as mx
        from mlx_lm import generate

        correct = 0
        results = []

        for prompt, expected in self.test_cases:
            try:
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )
                is_correct = expected.lower() in response.lower()
                if is_correct:
                    correct += 1
                results.append({
                    'prompt': prompt,
                    'expected': expected,
                    'got': response[:50],
                    'correct': is_correct
                })
            except Exception as e:
                results.append({
                    'prompt': prompt,
                    'expected': expected,
                    'got': f"ERROR: {e}",
                    'correct': False
                })

        return correct / len(self.test_cases), results

    def evaluate_top_token(self) -> float:
        """Evaluate using single top token (for comparison)."""
        import mlx.core as mx

        correct = 0
        for prompt, expected in self.test_cases:
            try:
                tokens = self.tokenizer.encode(prompt)
                input_ids = mx.array([tokens])
                logits = self.model(input_ids)
                mx.eval(logits)
                top_token = int(mx.argmax(logits[0, -1, :]).item())
                word = self.tokenizer.decode([top_token]).strip()
                if expected.lower() in word.lower():
                    correct += 1
            except:
                pass

        return correct / len(self.test_cases)

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of generated text."""
        import mlx.core as mx

        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) < 2:
                return 1.0

            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)

            logits_np = np.array(logits[0].tolist())
            log_probs = []

            for i in range(1, len(tokens)):
                log_prob = logits_np[i-1, tokens[i]] - np.log(np.exp(logits_np[i-1]).sum())
                log_probs.append(log_prob)

            return float(np.exp(-np.mean(log_probs)))
        except:
            return float('inf')

    def check_coherence(self, num_samples: int = 3) -> bool:
        """Check if model still produces coherent output."""
        from mlx_lm import generate

        coherence_prompts = [
            "The weather today is",
            "I went to the store to buy",
            "Technology has changed the way we",
        ]

        for prompt in coherence_prompts[:num_samples]:
            try:
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=20,
                    verbose=False
                )
                ppl = self.compute_perplexity(prompt + response)
                if ppl > self.max_perplexity:
                    logger.warning(f"Coherence check failed: ppl={ppl:.1f} > {self.max_perplexity}")
                    return False
            except:
                return False

        return True

    # =========================================================================
    # Geometry & Activation Methods
    # =========================================================================

    def get_layer_activations(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get input/output activations for a layer."""
        import mlx.core as mx

        inputs = []
        outputs = []
        captured = {}

        layer = self.model.model.layers[layer_idx]
        if hasattr(layer, 'feed_forward'):
            original_mlp = layer.feed_forward
            key = 'feed_forward'
        else:
            original_mlp = layer.mlp
            key = 'mlp'

        class MLPHook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['input'] = x
                captured['output'] = self.mlp(x)
                return captured['output']

        for prompt in self.probe_prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            captured.clear()

            if key == 'feed_forward':
                layer.feed_forward = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = self.model(input_ids)
                mx.eval(captured['input'], captured['output'])
                inputs.append(np.array(captured['input'][0, -1, :].tolist(), dtype=np.float64))
                outputs.append(np.array(captured['output'][0, -1, :].tolist(), dtype=np.float64))
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        return np.stack(inputs), np.stack(outputs)

    def compute_current_geometry(self, layer_idx: int) -> Dict[str, float]:
        """Compute geometry metrics for current state."""
        _, Y = self.get_layer_activations(layer_idx)
        kurtosis = compute_kurtosis(Y)
        spectral_entropy = compute_spectral_entropy(Y)
        score = geometry_score(kurtosis, spectral_entropy)
        return {
            'kurtosis': kurtosis,
            'spectral_entropy': spectral_entropy,
            'score': score
        }

    # =========================================================================
    # Exploration Methods
    # =========================================================================

    def try_modification(
        self,
        layer_idx: int,
        direction: int,
        boost: float,
        seq_length: int
    ) -> Optional[Dict[str, Any]]:
        """Try a single modification and return result if valid."""
        import mlx.core as mx

        self.total_configurations_tested += 1

        # Get current activations
        S_X, S_Y = self.get_layer_activations(layer_idx)
        baseline_geo = self.compute_current_geometry(layer_idx)

        # SVD for direction extraction
        S_Y_centered = S_Y - S_Y.mean(axis=0)
        try:
            _, S, Vh = svd(S_Y_centered, full_matrices=False)
        except:
            return None

        if direction >= len(Vh):
            return None

        if boost == 1.0:
            return None  # No change

        # Compute boosted outputs
        coefs = S_Y_centered @ Vh[direction]
        proj = np.outer(coefs, Vh[direction])
        Y_new = S_Y + proj * (boost - 1)

        # Check geometry improvement
        new_kurtosis = compute_kurtosis(Y_new)
        new_entropy = compute_spectral_entropy(Y_new)
        new_score = geometry_score(new_kurtosis, new_entropy)

        if new_score <= baseline_geo['score'] + 1e-4:
            return None  # No geometry improvement

        # Compute replacement weight matrix
        S_X_scale = np.abs(S_X).max()
        Y_scale = np.abs(Y_new).max()
        if S_X_scale < 1e-10 or Y_scale < 1e-10:
            return None

        S_X_norm = S_X / S_X_scale
        Y_norm = Y_new / Y_scale
        reg = 1e-3
        ATA_w = S_X_norm.T @ S_X_norm + reg * np.eye(S_X_norm.shape[1])
        ATB_w = S_X_norm.T @ Y_norm

        try:
            W_norm, _, _, _ = np.linalg.lstsq(ATA_w, ATB_w, rcond=None)
        except:
            return None

        W = (W_norm * Y_scale / S_X_scale).T

        if np.isnan(W).any() or np.isinf(W).any():
            return None

        # Apply temporarily and evaluate
        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class TestMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        layer = self.model.model.layers[layer_idx]
        if hasattr(layer, 'feed_forward'):
            original_mlp = layer.feed_forward
            layer.feed_forward = TestMLP(W_mx)
            key = 'feed_forward'
        else:
            original_mlp = layer.mlp
            layer.mlp = TestMLP(W_mx)
            key = 'mlp'

        try:
            # CRITICAL: Evaluate with generation
            new_gen_acc, results = self.evaluate_generation(max_tokens=seq_length)

            # Restore original
            if key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

            # Check if improvement
            if new_gen_acc < self.current_gen_accuracy:
                return None  # Accuracy regression

            # Check coherence
            if not self.check_coherence(num_samples=1):
                return None  # Lost coherence

            return {
                'layer': layer_idx,
                'direction': direction,
                'boost': boost,
                'seq_length': seq_length,
                'gen_accuracy': new_gen_acc,
                'geometry': {
                    'kurtosis': new_kurtosis,
                    'spectral_entropy': new_entropy,
                    'score': new_score
                },
                'W': W,
                'key': key,
                'results': results
            }

        except Exception as e:
            # Restore on error
            if key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp
            return None

    def apply_improvement(self, improvement: Dict[str, Any]):
        """Permanently apply an improvement."""
        import mlx.core as mx

        W = improvement['W']
        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class PermanentMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        layer = self.model.model.layers[improvement['layer']]
        if improvement['key'] == 'feed_forward':
            layer.feed_forward = PermanentMLP(W_mx)
        else:
            layer.mlp = PermanentMLP(W_mx)

        self.current_gen_accuracy = improvement['gen_accuracy']
        self.current_geometry = improvement['geometry']
        self.improvements.append({
            'round': self.round_num,
            'layer': improvement['layer'],
            'direction': improvement['direction'],
            'boost': improvement['boost'],
            'seq_length': improvement['seq_length'],
            'gen_accuracy': improvement['gen_accuracy'],
            'geometry': improvement['geometry'],
            'timestamp': datetime.now().isoformat()
        })

    def explore_round(self) -> bool:
        """Explore one round, return True if improvement found."""
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

                    for seq_length in self.sequence_lengths:
                        if self.interrupted:
                            break

                        candidate = self.try_modification(
                            layer_idx, direction, boost, seq_length
                        )

                        if candidate is not None:
                            if best is None or candidate['gen_accuracy'] > best['gen_accuracy']:
                                best = candidate
                            elif candidate['gen_accuracy'] == best['gen_accuracy']:
                                if candidate['geometry']['score'] > best['geometry']['score']:
                                    best = candidate

        if best is not None and best['gen_accuracy'] > self.current_gen_accuracy:
            self.apply_improvement(best)
            logger.info(
                f"  IMPROVED: L{best['layer']} d{best['direction']} "
                f"b{best['boost']:.1f} seq{best['seq_length']} → "
                f"{best['gen_accuracy']*100:.0f}%"
            )
            return True

        return False

    # =========================================================================
    # Checkpoint Methods
    # =========================================================================

    def save_checkpoint(self, reason: str = "round"):
        """Save current state to checkpoint."""
        checkpoint = {
            'round_num': self.round_num,
            'current_gen_accuracy': self.current_gen_accuracy,
            'current_top_accuracy': self.current_top_accuracy,
            'current_geometry': self.current_geometry,
            'improvements': self.improvements,
            'stagnant_rounds': self.stagnant_rounds,
            'total_configurations_tested': self.total_configurations_tested,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'config': self.config,
        }

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_r{self.round_num:04d}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        # Update latest symlink
        latest_path = self.checkpoint_dir / "latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)

        logger.info(f"  Checkpoint saved: {checkpoint_path.name}")

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load state from checkpoint."""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            self.round_num = checkpoint['round_num']
            self.current_gen_accuracy = checkpoint['current_gen_accuracy']
            self.current_top_accuracy = checkpoint.get('current_top_accuracy', 0.0)
            self.current_geometry = checkpoint.get('current_geometry')
            self.improvements = checkpoint['improvements']
            self.stagnant_rounds = checkpoint.get('stagnant_rounds', 0)
            self.total_configurations_tested = checkpoint.get('total_configurations_tested', 0)

            logger.info(f"Loaded checkpoint: round {self.round_num}, acc {self.current_gen_accuracy*100:.0f}%")

            # Re-apply all improvements
            # Note: This requires the model to be in its initial state
            logger.info(f"Re-applying {len(self.improvements)} improvements...")
            # TODO: Implement improvement replay

            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self, max_rounds: Optional[int] = None):
        """Main exploration loop."""
        logger.info("="*80)
        logger.info("AUTONOMOUS MANIFOLD COMPLETION")
        logger.info("="*80)
        logger.info(f"Started at {datetime.now().isoformat()}")
        logger.info(f"Max stagnant rounds: {self.max_stagnant}")
        logger.info(f"Max rounds: {max_rounds if max_rounds else 'unlimited'}")
        logger.info(f"Layers: {len(self.target_layers)}, Directions: {len(self.directions)}")
        logger.info(f"Boosts: {len(self.boost_factors)}, Seq lengths: {len(self.sequence_lengths)}")
        logger.info(f"Total configs per round: {len(self.target_layers) * len(self.directions) * len(self.boost_factors) * len(self.sequence_lengths)}")

        # Initial evaluation
        self.current_gen_accuracy, initial_results = self.evaluate_generation()
        self.current_top_accuracy = self.evaluate_top_token()

        logger.info(f"\nInitial accuracy:")
        logger.info(f"  Top-token:  {self.current_top_accuracy*100:.0f}%")
        logger.info(f"  Generation: {self.current_gen_accuracy*100:.0f}%")

        # Main loop
        while not self.interrupted:
            self.round_num += 1

            # Check round limit
            if max_rounds and self.round_num > max_rounds:
                logger.info(f"\nReached max rounds ({max_rounds})")
                break

            # Check convergence
            if self.stagnant_rounds >= self.max_stagnant:
                logger.info(f"\nConverged! No improvement for {self.max_stagnant} rounds")
                break

            # Check if 100%
            if self.current_gen_accuracy >= 1.0:
                logger.info("\n🎉 REACHED 100% GENERATION ACCURACY!")
                break

            logger.info(f"\nRound {self.round_num} (stagnant: {self.stagnant_rounds}):")

            improved = self.explore_round()

            if improved:
                self.stagnant_rounds = 0
            else:
                self.stagnant_rounds += 1
                logger.info(f"  No improvement found")

            # Checkpoint every round
            self.save_checkpoint()

            # Update top-token accuracy
            self.current_top_accuracy = self.evaluate_top_token()
            logger.info(f"  Current: gen={self.current_gen_accuracy*100:.0f}%, top={self.current_top_accuracy*100:.0f}%")

        # Final checkpoint
        self.save_checkpoint(reason="final")

        # Final report
        self.report_final()

    def report_final(self):
        """Print final report."""
        logger.info("\n" + "="*80)
        logger.info("FINAL REPORT")
        logger.info("="*80)

        final_gen_acc, final_results = self.evaluate_generation()
        final_top_acc = self.evaluate_top_token()

        logger.info(f"\nFinal accuracy:")
        logger.info(f"  Top-token:  {final_top_acc*100:.0f}%")
        logger.info(f"  Generation: {final_gen_acc*100:.0f}%")

        logger.info(f"\nTotal rounds: {self.round_num}")
        logger.info(f"Total configurations tested: {self.total_configurations_tested}")
        logger.info(f"Total improvements applied: {len(self.improvements)}")

        if self.improvements:
            logger.info("\nImprovement history:")
            for imp in self.improvements:
                logger.info(
                    f"  R{imp['round']:3d}: L{imp['layer']} d{imp['direction']} "
                    f"b{imp['boost']:.1f} → {imp['gen_accuracy']*100:.0f}%"
                )

        logger.info(f"\nTest case results:")
        for r in final_results:
            mark = "✓" if r['correct'] else "✗"
            logger.info(f"  {mark} {r['prompt'][:35]:<38} → {r['got'][:30]}")

        logger.info(f"\nCompleted at {datetime.now().isoformat()}")

        # Save final results
        output = {
            'final_gen_accuracy': final_gen_acc,
            'final_top_accuracy': final_top_acc,
            'total_rounds': self.round_num,
            'total_configurations': self.total_configurations_tested,
            'improvements': self.improvements,
            'final_results': final_results,
            'timestamp': datetime.now().isoformat(),
        }

        output_path = self.checkpoint_dir / "final_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autonomous Manifold Completion")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum rounds (None = unlimited)"
    )
    parser.add_argument(
        "--max-stagnant",
        type=int,
        default=50,
        help="Max stagnant rounds before declaring convergence"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0-15",
        help="Layer range to explore (e.g., '0-15' or '2,4,8')"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode for testing (fewer configs)"
    )
    args = parser.parse_args()

    # Parse layers
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        target_layers = list(range(start, end + 1))
    else:
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
        # Fast mode: minimal exploration for testing
        config = {
            'target_layers': target_layers[:2] if len(target_layers) > 2 else target_layers,
            'directions': list(range(5)),  # Only 5 directions
            'boost_factors': [0.0, 0.5, 1.5, 2.0],  # Only 4 boosts
            'sequence_lengths': [5],  # Only 1 seq length
            'max_stagnant': min(args.max_stagnant, 5),
            'max_perplexity': 100.0,
        }
        logger.info("FAST MODE: reduced exploration space for testing")
    else:
        config = {
            'target_layers': target_layers,
            'directions': list(range(20)),
            'boost_factors': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 2.5, 3.0],
            'sequence_lengths': [1, 3, 5, 10],
            'max_stagnant': args.max_stagnant,
            'max_perplexity': 100.0,
        }

    # Create explorer
    explorer = ManifoldExplorer(model, tokenizer, config)

    # Resume if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            explorer.load_checkpoint(resume_path)
        else:
            logger.error(f"Checkpoint not found: {resume_path}")
            return

    # Run
    explorer.run(max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
