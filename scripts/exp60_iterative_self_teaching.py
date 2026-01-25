#!/usr/bin/env python3
"""Experiment 60: Iterative Self-Teaching Loop.

The full cycle: Run entropy-guided teaching until convergence.

This is the culmination of experiments 56-59:
- Exp 56: Entropy reduction is prompt-specific
- Exp 57: Apply teaching only when H_trans < H_orig
- Exp 58: Per-prompt optimal pairs extract 2x more entropy
- Exp 59: Pure manifold teaching via spectral entropy

Now we close the loop:
1. Measure spectral entropy at each layer pair
2. Find pairs where teacher < student (transfer opportunities)
3. Transfer the cleanest direction
4. Update student model weights
5. Repeat until no more entropy reduction possible

This is SELF-TEACHING through pure geometry.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.linalg import svd
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def spectral_entropy(Y):
    """Compute entropy from singular value spectrum."""
    Y_centered = Y - Y.mean(axis=0)
    _, S, _ = svd(Y_centered, full_matrices=False)
    S_norm = S / np.sum(S)
    S_norm = S_norm[S_norm > 1e-10]
    return -np.sum(S_norm * np.log(S_norm))


def run_experiment():
    """Run iterative self-teaching until convergence."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    # Load models
    source_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    target_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"

    logger.info("Loading teacher (DeepSeek-R1-8B)...")
    from mlx_lm import load
    teacher_model, teacher_tokenizer = load(source_path)

    logger.info("Loading student (LFM2-1.2B)...")
    student_model, student_tokenizer = load(target_path)

    # Probe prompts for entropy measurement
    probe_prompts = [
        "The capital of France is",
        "Water freezes at",
        "The largest planet is",
        "DNA stands for",
        "The speed of light",
        "Photosynthesis occurs in",
        "The periodic table",
        "Machine learning uses",
        "The theory of relativity",
        "Quantum mechanics describes",
        "Shakespeare wrote",
        "The human brain",
        "Evolution explains",
        "Gravity attracts",
        "The internet connects",
        "Vaccines prevent",
    ]

    # Test prompts for accuracy measurement
    test_prompts = [
        "The moon is", "Trees are", "Birds can", "Mountains are",
        "Electrons orbit", "Genes contain", "Stories tell", "Poetry expresses",
        "Therefore we", "Because of", "Technology enables", "Culture shapes",
    ]

    # Layer pairs to consider (teacher golden zone → student corresponding)
    teacher_layers = [22, 23, 24, 25, 26, 28, 30]
    student_layers = [9, 10, 11, 12, 13]

    def get_layer_activations(model, tokenizer, layer_idx, prompts):
        """Get MLP input and output activations for a layer."""
        inputs = []
        outputs = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            mlp_input = None
            mlp_output = None

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
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            if key == 'feed_forward':
                layer.feed_forward = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(np.array(mlp_input[0, -1, :].tolist(), dtype=np.float64))
                outputs.append(np.array(mlp_output[0, -1, :].tolist(), dtype=np.float64))
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        return np.stack(inputs), np.stack(outputs)

    def compute_transplant_weight(source_Y, target_X, target_Y, direction_idx):
        """Compute new MLP weight that replaces direction d with source's."""
        # SVD of both
        source_Y_centered = source_Y - source_Y.mean(axis=0)
        target_Y_centered = target_Y - target_Y.mean(axis=0)

        _, _, Vh_s = svd(source_Y_centered, full_matrices=False)
        _, _, Vh_t = svd(target_Y_centered, full_matrices=False)

        # Translation matrix
        F = np.linalg.lstsq(source_Y, target_Y, rcond=1e-10)[0]

        # Start with target, replace direction d
        result = target_Y.copy()

        d = direction_idx
        if d < len(Vh_s) and d < len(Vh_t):
            # Remove target's direction d
            target_coefs_d = target_Y_centered @ Vh_t[d]
            target_proj_d = np.outer(target_coefs_d, Vh_t[d])
            result -= target_proj_d

            # Add source's direction d (translated)
            source_coefs_d = source_Y_centered @ Vh_s[d]
            source_proj_d = np.outer(source_coefs_d, Vh_s[d])
            result += source_proj_d @ F

        # Solve for new weight matrix: target_X @ W.T = result
        alpha = 1e-6
        ATA = target_X.T @ target_X + alpha * np.eye(target_X.shape[1])
        ATB = target_X.T @ result
        W = np.linalg.solve(ATA, ATB).T

        return W

    def install_transplant(model, layer_idx, W_new):
        """Install new MLP weight matrix (as linear approximation)."""
        W_mx = mx.array(W_new.astype(np.float32))
        mx.eval(W_mx)

        class TransplantedMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'feed_forward'):
            layer.feed_forward = TransplantedMLP(W_mx)
        else:
            layer.mlp = TransplantedMLP(W_mx)

    def evaluate_accuracy(model, tokenizer, test_prompts, reference_model, ref_tokenizer):
        """Evaluate token agreement with reference model."""
        matches = 0
        for prompt in test_prompts:
            # Reference prediction
            ref_tokens = ref_tokenizer.encode(prompt)
            ref_ids = mx.array([ref_tokens])
            ref_logits = reference_model(ref_ids)
            mx.eval(ref_logits)
            ref_top = int(mx.argmax(ref_logits[0, -1, :]).item())

            # Model prediction
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = model(input_ids)
            mx.eval(logits)
            top = int(mx.argmax(logits[0, -1, :]).item())

            if top == ref_top:
                matches += 1

        return matches / len(test_prompts)

    # ========================================
    # PHASE 1: Initial state
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Initial State")
    logger.info(f"{'='*80}")

    # Measure initial entropies
    initial_entropies = {}
    for s_layer in student_layers:
        _, S_acts = get_layer_activations(student_model, student_tokenizer, s_layer, probe_prompts)
        initial_entropies[s_layer] = spectral_entropy(S_acts)
        logger.info(f"Student L{s_layer}: H = {initial_entropies[s_layer]:.4f}")

    total_initial_entropy = sum(initial_entropies.values())
    logger.info(f"\nTotal initial entropy: {total_initial_entropy:.4f}")

    # Store original MLPs for potential rollback
    original_mlps = {}
    for s_layer in student_layers:
        layer = student_model.model.layers[s_layer]
        if hasattr(layer, 'feed_forward'):
            original_mlps[s_layer] = layer.feed_forward
        else:
            original_mlps[s_layer] = layer.mlp

    # ========================================
    # PHASE 2: Iterative self-teaching loop
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Iterative Self-Teaching Loop")
    logger.info(f"{'='*80}")

    max_iterations = 1  # CRITICAL: Compression quantum = 1 layer!
    min_improvement = 0.001  # Minimum entropy reduction to continue
    direction_to_transfer = 5  # Direction 6 (0-indexed = 5) was best in exp52-54

    history = []
    modified_layers = set()

    # NOTE: We learned from exp43 and exp49 that only ONE layer can be modified
    # at full accuracy. The "compression quantum" = 1 layer.
    # Multi-layer modification causes 26% manifold shift and accuracy collapse.

    for iteration in range(max_iterations):
        logger.info(f"\n--- Iteration {iteration + 1} ---")

        # Find all transfer opportunities
        opportunities = []

        for t_layer in teacher_layers:
            for s_layer in student_layers:
                if s_layer in modified_layers:
                    # Skip already-modified layers (compression quantum = 1)
                    continue

                # Get current activations
                T_X, T_Y = get_layer_activations(teacher_model, teacher_tokenizer, t_layer, probe_prompts)
                S_X, S_Y = get_layer_activations(student_model, student_tokenizer, s_layer, probe_prompts)

                t_entropy = spectral_entropy(T_Y)
                s_entropy = spectral_entropy(S_Y)
                delta_h = s_entropy - t_entropy  # Positive = student has MORE entropy

                if delta_h > min_improvement:
                    opportunities.append({
                        't_layer': t_layer,
                        's_layer': s_layer,
                        't_entropy': t_entropy,
                        's_entropy': s_entropy,
                        'delta_h': delta_h,
                        'T_Y': T_Y,
                        'S_X': S_X,
                        'S_Y': S_Y,
                    })

        if not opportunities:
            logger.info("No more transfer opportunities. Converged!")
            break

        # Sort by entropy reduction potential
        opportunities.sort(key=lambda x: -x['delta_h'])

        best = opportunities[0]
        logger.info(f"Best opportunity: T{best['t_layer']}→S{best['s_layer']}")
        logger.info(f"  Teacher H: {best['t_entropy']:.4f}")
        logger.info(f"  Student H: {best['s_entropy']:.4f}")
        logger.info(f"  Potential ΔH: {best['delta_h']:.4f}")

        # Compute and install the transplant
        W_new = compute_transplant_weight(
            best['T_Y'], best['S_X'], best['S_Y'], direction_to_transfer
        )
        install_transplant(student_model, best['s_layer'], W_new)
        modified_layers.add(best['s_layer'])

        # Measure new entropy
        _, S_Y_new = get_layer_activations(student_model, student_tokenizer, best['s_layer'], probe_prompts)
        new_entropy = spectral_entropy(S_Y_new)
        actual_delta = best['s_entropy'] - new_entropy

        logger.info(f"  After transplant: H = {new_entropy:.4f}")
        logger.info(f"  Actual ΔH: {actual_delta:.4f}")

        history.append({
            'iteration': iteration + 1,
            'pair': f"T{best['t_layer']}→S{best['s_layer']}",
            'before': best['s_entropy'],
            'after': new_entropy,
            'delta': actual_delta,
        })

        if actual_delta < min_improvement:
            logger.info("Minimal improvement. Stopping.")
            break

    # ========================================
    # PHASE 3: Final evaluation
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Final Evaluation")
    logger.info(f"{'='*80}")

    # Measure final entropies
    final_entropies = {}
    for s_layer in student_layers:
        _, S_acts = get_layer_activations(student_model, student_tokenizer, s_layer, probe_prompts)
        final_entropies[s_layer] = spectral_entropy(S_acts)

    total_final_entropy = sum(final_entropies.values())
    total_reduction = total_initial_entropy - total_final_entropy

    logger.info(f"\nEntropy comparison:")
    logger.info(f"{'Layer':>8} {'Initial':>12} {'Final':>12} {'ΔH':>12}")
    logger.info("-" * 48)
    for s_layer in student_layers:
        delta = initial_entropies[s_layer] - final_entropies[s_layer]
        logger.info(f"L{s_layer:>6} {initial_entropies[s_layer]:>12.4f} {final_entropies[s_layer]:>12.4f} {delta:>+12.4f}")

    logger.info("-" * 48)
    logger.info(f"{'TOTAL':>8} {total_initial_entropy:>12.4f} {total_final_entropy:>12.4f} {total_reduction:>+12.4f}")

    # Evaluate token accuracy (self-agreement after modification)
    logger.info(f"\n{'='*80}")
    logger.info("Token Prediction Analysis")
    logger.info(f"{'='*80}")

    # Reload original student for comparison
    logger.info("\nReloading original student for comparison...")
    original_student, _ = load(target_path)

    logger.info(f"\n{'Prompt':<25} {'Original':>12} {'Modified':>12} {'Match':>8}")
    logger.info("-" * 60)

    matches = 0
    for prompt in test_prompts:
        tokens = student_tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Original prediction
        orig_logits = original_student(input_ids)
        mx.eval(orig_logits)
        orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())
        orig_word = student_tokenizer.decode([orig_top]).strip()

        # Modified prediction
        mod_logits = student_model(input_ids)
        mx.eval(mod_logits)
        mod_top = int(mx.argmax(mod_logits[0, -1, :]).item())
        mod_word = student_tokenizer.decode([mod_top]).strip()

        match = orig_top == mod_top
        if match:
            matches += 1
        mark = "✓" if match else "✗"

        logger.info(f"{prompt:<25} {orig_word:>12} {mod_word:>12} {mark:>8}")

    self_agreement = matches / len(test_prompts)
    logger.info(f"\nSelf-agreement: {self_agreement*100:.1f}% ({matches}/{len(test_prompts)})")

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Iterative Self-Teaching")
    logger.info(f"{'='*80}")

    logger.info(f"""
ITERATIONS: {len(history)}

TRANSFER HISTORY:
""")
    for h in history:
        logger.info(f"  {h['iteration']}. {h['pair']}: {h['before']:.4f} → {h['after']:.4f} (ΔH = {h['delta']:+.4f})")

    logger.info(f"""
RESULTS:

Layers modified: {len(modified_layers)} ({', '.join(f'L{l}' for l in sorted(modified_layers))})
Total entropy reduction: {total_reduction:+.4f} nats
Self-agreement: {self_agreement*100:.1f}%

THE SELF-TEACHING CYCLE:

1. MEASURE: Spectral entropy at each layer pair
2. COMPARE: Find where teacher entropy < student entropy
3. TRANSFER: Replace student's direction with teacher's cleaner one
4. REPEAT: Until no more entropy reduction possible

CONVERGENCE:

The loop converges when the student's manifold is as "clean"
as the teacher's at all transferable layer pairs.

This is TEACHING WITHOUT TOKENS:
- No logits compared
- No cross-entropy computed
- Just manifold geometry: entropy reduction = knowledge transfer
""")


if __name__ == "__main__":
    run_experiment()
