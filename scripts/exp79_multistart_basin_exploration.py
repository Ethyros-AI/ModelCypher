#!/usr/bin/env python3
"""Experiment 79: Multi-Start Basin Exploration.

Test the disconnected manifold hypothesis:
- If different starting points all converge to ~70%, the basin is architecturally fixed
- If ceilings vary, multiple basins exist and starting point matters

Method:
1. Create N perturbations of Layer 2 (the key improvement layer)
2. Perturb along principal directions (stays on manifold)
3. Run self-improvement from each starting point
4. Compare ceilings reached
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.linalg import svd
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_kurtosis(Y):
    """Compute average kurtosis over samples."""
    kurtoses = []
    for h in Y:
        std = h.std()
        if std < 1e-10:
            kurtoses.append(0.0)
            continue
        z = (h - h.mean()) / std
        kurtoses.append(float(np.mean(z ** 4) - 3))
    return np.mean(kurtoses)


def compute_spectral_entropy(Y):
    """Compute spectral entropy of a manifold."""
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


def geometry_score(kurtosis, spectral_entropy):
    """Higher = more 'correct-like' geometry."""
    return kurtosis / 100 - spectral_entropy


def run_experiment():
    """Multi-start basin exploration."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    logger.info("="*80)
    logger.info("MULTI-START BASIN EXPLORATION")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")
    logger.info("\nTesting the disconnected manifold hypothesis...")

    # Configuration
    N_STARTS = 10
    PERTURBATION_SCALE = 0.01  # 1% of weight magnitude
    TARGET_LAYER = 2  # The key improvement layer
    MAX_IMPROVEMENT_ROUNDS = 20  # Limit for speed

    logger.info(f"\nConfiguration:")
    logger.info(f"  Number of starts: {N_STARTS}")
    logger.info(f"  Perturbation scale: {PERTURBATION_SCALE*100}%")
    logger.info(f"  Target layer: {TARGET_LAYER}")
    logger.info(f"  Max improvement rounds: {MAX_IMPROVEMENT_ROUNDS}")

    # Test cases
    test_cases = [
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

    # Probe prompts
    probe_prompts = [
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

    def get_prediction(model, tokenizer, prompt):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        top_token = int(mx.argmax(logits[0, -1, :]).item())
        return tokenizer.decode([top_token]).strip()

    def evaluate_accuracy(model, tokenizer):
        correct = 0
        for prompt, expected in test_cases:
            word = get_prediction(model, tokenizer, prompt)
            if expected.lower() in word.lower():
                correct += 1
        return correct / len(test_cases)

    def get_layer_activations(model, tokenizer, layer_idx, prompts):
        inputs = []
        outputs = []
        captured = {}

        layer = model.model.layers[layer_idx]
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

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            captured.clear()

            if key == 'feed_forward':
                layer.feed_forward = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = model(input_ids)
                mx.eval(captured['input'], captured['output'])
                inputs.append(np.array(captured['input'][0, -1, :].tolist(), dtype=np.float64))
                outputs.append(np.array(captured['output'][0, -1, :].tolist(), dtype=np.float64))
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        return np.stack(inputs), np.stack(outputs)

    def boost_direction(S_X, S_Y, direction_idx, boost_factor):
        """Boost a specific direction. Returns new Y only."""
        if np.isnan(S_X).any() or np.isinf(S_X).any():
            return None
        if np.isnan(S_Y).any() or np.isinf(S_Y).any():
            return None

        S_Y_centered = S_Y - S_Y.mean(axis=0)

        try:
            _, S, Vh = svd(S_Y_centered, full_matrices=False)
        except:
            return None

        d = direction_idx
        if d >= len(Vh):
            return None

        if S[d] < 1e-6 * S[0]:
            return None

        coefs = S_Y_centered @ Vh[d]
        if np.isnan(coefs).any() or np.isinf(coefs).any():
            return None

        proj = np.outer(coefs, Vh[d])
        result = S_Y + proj * (boost_factor - 1)

        if np.isnan(result).any() or np.isinf(result).any():
            return None

        return result

    def compute_weight_transform(S_X, Y_new):
        """Compute the weight matrix for a given transformation."""
        S_X_scale = np.abs(S_X).max()
        Y_scale = np.abs(Y_new).max()
        if S_X_scale < 1e-10 or Y_scale < 1e-10:
            return None

        S_X_norm = S_X / S_X_scale
        Y_norm = Y_new / Y_scale

        alpha = 1e-3
        ATA = S_X_norm.T @ S_X_norm + alpha * np.eye(S_X_norm.shape[1])
        ATB = S_X_norm.T @ Y_norm

        try:
            W_norm, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)
            W = (W_norm * Y_scale / S_X_scale).T
        except:
            return None

        if np.isnan(W).any() or np.isinf(W).any():
            return None

        return W

    def run_self_improvement(model, tokenizer, layer_idx, max_rounds):
        """Run self-improvement loop, return ceiling accuracy reached."""
        current_acc = evaluate_accuracy(model, tokenizer)
        best_acc = current_acc
        stagnant = 0
        max_stagnant = 3

        directions = list(range(12))
        boosts = [0.0, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]

        for round_num in range(max_rounds):
            if stagnant >= max_stagnant:
                break

            S_X, S_Y = get_layer_activations(model, tokenizer, layer_idx, probe_prompts)
            baseline_score = geometry_score(compute_kurtosis(S_Y), compute_spectral_entropy(S_Y))

            best_this_round = None

            for d in directions:
                for boost in boosts:
                    if boost == 1.0:
                        continue

                    Y_new = boost_direction(S_X, S_Y, d, boost)
                    if Y_new is None:
                        continue

                    new_score = geometry_score(compute_kurtosis(Y_new), compute_spectral_entropy(Y_new))
                    if new_score <= baseline_score + 1e-4:
                        continue

                    W = compute_weight_transform(S_X, Y_new)
                    if W is None:
                        continue

                    W_mx = mx.array(W.astype(np.float32))
                    mx.eval(W_mx)

                    class ModifiedMLP:
                        def __init__(self, W):
                            self.W = W
                        def __call__(self, x):
                            return mx.matmul(x, self.W.T)

                    layer = model.model.layers[layer_idx]
                    if hasattr(layer, 'feed_forward'):
                        original_mlp = layer.feed_forward
                        layer.feed_forward = ModifiedMLP(W_mx)
                        mlp_key = 'feed_forward'
                    else:
                        original_mlp = layer.mlp
                        layer.mlp = ModifiedMLP(W_mx)
                        mlp_key = 'mlp'

                    new_acc = evaluate_accuracy(model, tokenizer)

                    if new_acc >= current_acc:
                        if best_this_round is None or new_acc > best_this_round['acc']:
                            best_this_round = {
                                'W': W,
                                'acc': new_acc,
                                'mlp_key': mlp_key,
                            }

                    # Restore
                    if mlp_key == 'feed_forward':
                        layer.feed_forward = original_mlp
                    else:
                        layer.mlp = original_mlp

            if best_this_round:
                W_mx = mx.array(best_this_round['W'].astype(np.float32))
                mx.eval(W_mx)

                class PermanentMLP:
                    def __init__(self, W):
                        self.W = W
                    def __call__(self, x):
                        return mx.matmul(x, self.W.T)

                layer = model.model.layers[layer_idx]
                if best_this_round['mlp_key'] == 'feed_forward':
                    layer.feed_forward = PermanentMLP(W_mx)
                else:
                    layer.mlp = PermanentMLP(W_mx)

                if best_this_round['acc'] > best_acc:
                    best_acc = best_this_round['acc']
                    current_acc = best_this_round['acc']
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                stagnant += 1

        return best_acc

    # ========================================
    # MAIN EXPERIMENT
    # ========================================

    results = []

    for trial in range(N_STARTS):
        logger.info(f"\n{'='*60}")
        logger.info(f"TRIAL {trial + 1}/{N_STARTS}")
        logger.info(f"{'='*60}")

        # Load fresh model for each trial
        logger.info("\nLoading fresh LFM2-1.2B...")
        model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
        model, tokenizer = load(model_path)

        initial_acc = evaluate_accuracy(model, tokenizer)
        logger.info(f"Initial accuracy: {initial_acc*100:.0f}%")

        if trial == 0:
            # Trial 0: Original model (baseline)
            logger.info("\nTrial 0: BASELINE (no perturbation)")
        else:
            # Apply perturbation to Layer 2
            logger.info(f"\nApplying perturbation {trial} to Layer {TARGET_LAYER}...")

            # Get current activations to find principal directions
            S_X, S_Y = get_layer_activations(model, tokenizer, TARGET_LAYER, probe_prompts)

            # SVD to get principal directions
            S_Y_centered = S_Y - S_Y.mean(axis=0)
            _, S_vals, Vh = svd(S_Y_centered, full_matrices=False)

            # Perturb along a different principal direction for each trial
            direction_to_perturb = (trial - 1) % len(Vh)
            perturbation_direction = Vh[direction_to_perturb]

            # Compute perturbation magnitude
            weight_scale = np.abs(S_Y).mean()
            perturbation_magnitude = weight_scale * PERTURBATION_SCALE

            # Create perturbed target outputs
            # We shift all outputs along this principal direction
            shift_sign = 1 if trial % 2 == 1 else -1  # Alternate positive/negative
            shift_amount = perturbation_magnitude * shift_sign * (1 + trial * 0.5)

            Y_perturbed = S_Y + shift_amount * perturbation_direction

            # Compute weight matrix for this perturbation
            W_perturbed = compute_weight_transform(S_X, Y_perturbed)

            if W_perturbed is not None:
                W_mx = mx.array(W_perturbed.astype(np.float32))
                mx.eval(W_mx)

                class PerturbedMLP:
                    def __init__(self, W):
                        self.W = W
                    def __call__(self, x):
                        return mx.matmul(x, self.W.T)

                layer = model.model.layers[TARGET_LAYER]
                if hasattr(layer, 'feed_forward'):
                    layer.feed_forward = PerturbedMLP(W_mx)
                else:
                    layer.mlp = PerturbedMLP(W_mx)

                perturbed_acc = evaluate_accuracy(model, tokenizer)
                logger.info(f"After perturbation: {perturbed_acc*100:.0f}%")
                logger.info(f"  Direction: {direction_to_perturb}, Shift: {shift_sign}×{shift_amount:.4f}")
            else:
                logger.info("  Perturbation failed, using original")

        # Run self-improvement
        logger.info("\nRunning self-improvement...")
        ceiling = run_self_improvement(model, tokenizer, TARGET_LAYER, MAX_IMPROVEMENT_ROUNDS)
        logger.info(f"Ceiling reached: {ceiling*100:.0f}%")

        results.append({
            'trial': trial,
            'is_baseline': trial == 0,
            'ceiling': ceiling,
        })

    # ========================================
    # ANALYSIS
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS")
    logger.info(f"{'='*80}")

    ceilings = [r['ceiling'] for r in results]
    baseline_ceiling = results[0]['ceiling']
    perturbed_ceilings = [r['ceiling'] for r in results[1:]]

    logger.info(f"\nResults:")
    logger.info(f"  Baseline ceiling: {baseline_ceiling*100:.0f}%")
    logger.info(f"  Perturbed ceilings: {[f'{c*100:.0f}%' for c in perturbed_ceilings]}")
    logger.info(f"\n  Mean: {np.mean(ceilings)*100:.1f}%")
    logger.info(f"  Std:  {np.std(ceilings)*100:.1f}%")
    logger.info(f"  Min:  {np.min(ceilings)*100:.0f}%")
    logger.info(f"  Max:  {np.max(ceilings)*100:.0f}%")

    # Hypothesis test
    if np.std(ceilings) < 0.05:  # Less than 5pp variance
        logger.info(f"\n{'='*60}")
        logger.info("CONCLUSION: SINGLE BASIN")
        logger.info(f"{'='*60}")
        logger.info("""
All starting points converge to ~{:.0f}% ceiling.
The basin is ARCHITECTURALLY DETERMINED.

Random starts don't help. The ceiling is fixed by the model structure.
To break through 70%, we need EXTERNAL information (teacher bridge).
""".format(np.mean(ceilings)*100))
    else:
        better_starts = [r for r in results if r['ceiling'] > baseline_ceiling + 0.05]
        if better_starts:
            logger.info(f"\n{'='*60}")
            logger.info("CONCLUSION: MULTIPLE BASINS EXIST!")
            logger.info(f"{'='*60}")
            logger.info(f"""
Found {len(better_starts)} starts that reached higher ceilings!
Best ceiling: {max(ceilings)*100:.0f}%

The starting point MATTERS. Different basins exist.
Next: investigate what makes "good" starting points.
""")
        else:
            logger.info(f"\n{'='*60}")
            logger.info("CONCLUSION: SOME VARIANCE BUT NO IMPROVEMENT")
            logger.info(f"{'='*60}")
            logger.info("""
There's variance in ceilings, but none exceeded baseline.
Some perturbations led to WORSE basins.

This suggests:
- The baseline is already in a "good" basin
- Random perturbations can move us to WORSE basins
- Need targeted perturbation (teacher bridge) to find BETTER basins
""")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save results
    output = {
        'configuration': {
            'n_starts': N_STARTS,
            'perturbation_scale': PERTURBATION_SCALE,
            'target_layer': TARGET_LAYER,
            'max_improvement_rounds': MAX_IMPROVEMENT_ROUNDS,
        },
        'results': results,
        'statistics': {
            'mean': float(np.mean(ceilings)),
            'std': float(np.std(ceilings)),
            'min': float(np.min(ceilings)),
            'max': float(np.max(ceilings)),
        },
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "multistart_basin_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
