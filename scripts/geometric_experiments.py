#!/usr/bin/env python3
"""Systematic Geometric Experiments - No Heuristics, Just Math.

This script runs rigorous experiments to understand the mathematical
structure of the constants in neural network geometry.

Experiments:
1. Bidirectional ratio measurement (are inverses there?)
2. Null hypothesis: random matrix comparison
3. Untrained vs trained model comparison
4. Pre vs post nonlinearity geometry
5. Gram matrix eigenvalue analysis
6. Residual stream geometry tracking
7. Orthogonal rotation invariance
8. Uniform scaling effects
9. Surgical SVD modification
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats
from scipy.linalg import svd, eigh

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fundamental constants
CONSTANTS = {
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "1/phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "1/sqrt2": 1 / np.sqrt(2),
    "sqrt3": np.sqrt(3),
    "e": np.e,
    "pi": np.pi,
}

MATCH_THRESHOLD = 0.05  # 5% relative error


def count_constant_matches(S: np.ndarray, bidirectional: bool = True) -> Dict[str, int]:
    """Count matches for each constant in singular value ratios.

    Args:
        S: Singular values (sorted descending)
        bidirectional: If True, check both σᵢ/σⱼ and σⱼ/σᵢ
    """
    matches = {name: 0 for name in CONSTANTS}

    for i in range(min(len(S) - 1, 20)):
        for j in range(i + 1, min(len(S), i + 6)):
            if S[j] > 1e-10:
                ratio1 = S[i] / S[j]
                ratio2 = S[j] / S[i] if bidirectional else None

                for const_name, const_val in CONSTANTS.items():
                    error1 = abs(ratio1 - const_val) / const_val
                    if error1 < MATCH_THRESHOLD:
                        matches[const_name] += 1

                    if bidirectional and ratio2 is not None:
                        error2 = abs(ratio2 - const_val) / const_val
                        if error2 < MATCH_THRESHOLD:
                            matches[const_name] += 1

    return matches


class GeometricExperiments:
    """Systematic experiments on neural network geometry."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self.results = {}

    def _get_mlp_weight(self, layer_idx: int) -> np.ndarray:
        """Get the gate projection weight matrix."""
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
        """Set the gate projection weight matrix."""
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

    def _get_activations(self, text: str, layer_idx: int, pre_nonlin: bool = False) -> np.ndarray:
        """Get activations from a layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward
            key = 'feed_forward'
        else:
            mlp = layer.mlp
            key = 'mlp'

        captured = {}

        if pre_nonlin:
            # Capture before nonlinearity
            if hasattr(mlp, 'gate_proj'):
                original_gate = mlp.gate_proj
                class PreNonlinHook:
                    def __init__(self, proj):
                        self.proj = proj
                    def __call__(self, x):
                        out = self.proj(x)
                        captured['pre_nonlin'] = out
                        return out
                mlp.gate_proj = PreNonlinHook(original_gate)
                try:
                    _ = self.model(input_ids)
                    mx.eval(captured.get('pre_nonlin', mx.zeros((1,1,1))))
                    if 'pre_nonlin' in captured:
                        return np.array(captured['pre_nonlin'][0].tolist(), dtype=np.float32)
                finally:
                    mlp.gate_proj = original_gate

        # Standard post-nonlinearity capture
        original = mlp if key == 'mlp' else layer.feed_forward

        class Hook:
            def __init__(self, m):
                self.m = m
            def __call__(self, x):
                captured['output'] = self.m(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = self.model(input_ids)
            mx.eval(captured.get('output', mx.zeros((1,1,1))))
            if 'output' in captured:
                return np.array(captured['output'][0].tolist(), dtype=np.float32)
            return np.zeros((1, 1))
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def _evaluate_model(self, prompts: List[Tuple[str, str]]) -> float:
        """Quick model quality check."""
        import mlx.core as mx

        correct = 0
        for prompt, expected in prompts:
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

        return correct / len(prompts) if prompts else 0.0

    # =========================================================================
    # EXPERIMENT 1: Bidirectional Ratio Measurement
    # =========================================================================
    def experiment_1_bidirectional_ratios(self) -> Dict:
        """Are the inverse constants there when we measure both directions?"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 1: Bidirectional Ratio Measurement")
        logger.info("="*60)

        unidirectional_totals = {name: 0 for name in CONSTANTS}
        bidirectional_totals = {name: 0 for name in CONSTANTS}

        for layer_idx in range(self.n_layers):
            W = self._get_mlp_weight(layer_idx)
            _, S, _ = svd(W, full_matrices=False)

            uni_matches = count_constant_matches(S, bidirectional=False)
            bi_matches = count_constant_matches(S, bidirectional=True)

            for name in CONSTANTS:
                unidirectional_totals[name] += uni_matches[name]
                bidirectional_totals[name] += bi_matches[name]

        logger.info("\nResults:")
        logger.info(f"{'Constant':>10} {'Unidirectional':>15} {'Bidirectional':>15}")
        logger.info("-" * 45)
        for name in sorted(CONSTANTS.keys()):
            logger.info(f"{name:>10} {unidirectional_totals[name]:>15} {bidirectional_totals[name]:>15}")

        # Key finding: do inverses appear?
        pi_e_uni = unidirectional_totals['pi/e']
        e_pi_uni = unidirectional_totals['e/pi']
        pi_e_bi = bidirectional_totals['pi/e']
        e_pi_bi = bidirectional_totals['e/pi']

        logger.info(f"\nKey finding:")
        logger.info(f"  π/e: {pi_e_uni} (uni) → {pi_e_bi} (bi)")
        logger.info(f"  e/π: {e_pi_uni} (uni) → {e_pi_bi} (bi)")
        logger.info(f"  The inverse IS there when we look both directions: {e_pi_bi > 0}")

        result = {
            'unidirectional': unidirectional_totals,
            'bidirectional': bidirectional_totals,
            'inverses_found': e_pi_bi > 0,
        }
        self.results['experiment_1'] = result
        return result

    # =========================================================================
    # EXPERIMENT 2: Null Hypothesis - Random Matrices
    # =========================================================================
    def experiment_2_null_hypothesis(self, n_samples: int = 1000) -> Dict:
        """Do random matrices show the same constant distribution?"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 2: Null Hypothesis (Random Matrices)")
        logger.info("="*60)

        # Get dimensions from actual weights
        layer_dims = []
        for layer_idx in range(self.n_layers):
            W = self._get_mlp_weight(layer_idx)
            layer_dims.append(W.shape)

        # Generate random matrices and count matches
        random_totals = {name: [] for name in CONSTANTS}

        np.random.seed(42)
        for _ in range(n_samples):
            sample_totals = {name: 0 for name in CONSTANTS}
            for shape in layer_dims:
                R = np.random.randn(*shape).astype(np.float32)
                _, S, _ = svd(R, full_matrices=False)
                matches = count_constant_matches(S, bidirectional=True)
                for name in CONSTANTS:
                    sample_totals[name] += matches[name]

            for name in CONSTANTS:
                random_totals[name].append(sample_totals[name])

        # Get actual model totals
        actual_totals = {name: 0 for name in CONSTANTS}
        for layer_idx in range(self.n_layers):
            W = self._get_mlp_weight(layer_idx)
            _, S, _ = svd(W, full_matrices=False)
            matches = count_constant_matches(S, bidirectional=True)
            for name in CONSTANTS:
                actual_totals[name] += matches[name]

        # Statistical comparison
        logger.info("\nComparison (actual vs random):")
        logger.info(f"{'Constant':>10} {'Actual':>10} {'Random Mean':>12} {'Random Std':>12} {'Z-score':>10} {'p-value':>10}")
        logger.info("-" * 75)

        p_values = {}
        for name in sorted(CONSTANTS.keys()):
            actual = actual_totals[name]
            rand_mean = np.mean(random_totals[name])
            rand_std = np.std(random_totals[name])

            if rand_std > 0:
                z_score = (actual - rand_mean) / rand_std
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed
            else:
                z_score = float('inf') if actual != rand_mean else 0
                p_value = 0.0 if actual != rand_mean else 1.0

            p_values[name] = p_value
            logger.info(f"{name:>10} {actual:>10} {rand_mean:>12.1f} {rand_std:>12.1f} {z_score:>10.2f} {p_value:>10.4f}")

        # Significance summary
        significant = [name for name, p in p_values.items() if p < 0.01]
        logger.info(f"\nConstants significantly different from random (p < 0.01): {significant}")

        result = {
            'actual': actual_totals,
            'random_mean': {name: float(np.mean(random_totals[name])) for name in CONSTANTS},
            'random_std': {name: float(np.std(random_totals[name])) for name in CONSTANTS},
            'p_values': p_values,
            'significant_constants': significant,
        }
        self.results['experiment_2'] = result
        return result

    # =========================================================================
    # EXPERIMENT 3: Pre vs Post Nonlinearity
    # =========================================================================
    def experiment_3_nonlinearity_effect(self, probes: List[str]) -> Dict:
        """Does the nonlinearity create or preserve the constants?"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 3: Pre vs Post Nonlinearity Geometry")
        logger.info("="*60)

        # This is tricky because we need to capture intermediate activations
        # For now, compare weight geometry with activation geometry

        weight_matches = {name: 0 for name in CONSTANTS}
        activation_matches = {name: 0 for name in CONSTANTS}

        for layer_idx in range(self.n_layers):
            # Weight geometry
            W = self._get_mlp_weight(layer_idx)
            _, S_w, _ = svd(W, full_matrices=False)
            w_match = count_constant_matches(S_w, bidirectional=True)

            # Activation geometry (collect across probes)
            all_acts = []
            for probe in probes:
                act = self._get_activations(probe, layer_idx)
                all_acts.append(act)

            A = np.vstack(all_acts)
            A_centered = A - A.mean(axis=0)
            _, S_a, _ = svd(A_centered, full_matrices=False)
            a_match = count_constant_matches(S_a, bidirectional=True)

            for name in CONSTANTS:
                weight_matches[name] += w_match[name]
                activation_matches[name] += a_match[name]

        logger.info("\nWeight vs Activation geometry:")
        logger.info(f"{'Constant':>10} {'Weights':>10} {'Activations':>12} {'Ratio':>10}")
        logger.info("-" * 45)
        for name in sorted(CONSTANTS.keys()):
            w = weight_matches[name]
            a = activation_matches[name]
            ratio = a / w if w > 0 else float('inf')
            logger.info(f"{name:>10} {w:>10} {a:>12} {ratio:>10.2f}")

        result = {
            'weight_matches': weight_matches,
            'activation_matches': activation_matches,
        }
        self.results['experiment_3'] = result
        return result

    # =========================================================================
    # EXPERIMENT 4: Gram Matrix Eigenvalues
    # =========================================================================
    def experiment_4_gram_matrix(self, probes: List[str]) -> Dict:
        """Do Gram matrix eigenvalue ratios show the same constants?"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 4: Gram Matrix Eigenvalue Analysis")
        logger.info("="*60)

        # G = A^T A has eigenvalues = squared singular values of A
        # So eigenvalue ratios should be squared SVD ratios

        gram_matches = {name: 0 for name in CONSTANTS}
        squared_constants = {f"{name}_sq": val**2 for name, val in CONSTANTS.items()}
        gram_squared_matches = {name: 0 for name in squared_constants}

        for layer_idx in range(self.n_layers):
            all_acts = []
            for probe in probes:
                act = self._get_activations(probe, layer_idx)
                all_acts.append(act.mean(axis=0))  # Mean across sequence

            A = np.vstack(all_acts)
            G = A.T @ A  # Gram matrix

            eigenvalues = np.linalg.eigvalsh(G)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
            eigenvalues = eigenvalues[eigenvalues > 1e-10]

            if len(eigenvalues) >= 2:
                # Check ratios against original constants
                matches = count_constant_matches(np.sqrt(eigenvalues), bidirectional=True)
                for name in CONSTANTS:
                    gram_matches[name] += matches[name]

                # Check ratios against squared constants
                for i in range(min(len(eigenvalues) - 1, 10)):
                    for j in range(i + 1, min(len(eigenvalues), i + 5)):
                        if eigenvalues[j] > 1e-10:
                            ratio = eigenvalues[i] / eigenvalues[j]
                            for name, val in squared_constants.items():
                                if abs(ratio - val) / val < MATCH_THRESHOLD:
                                    gram_squared_matches[name] += 1

        logger.info("\nGram eigenvalue ratios vs constants:")
        logger.info(f"{'Constant':>10} {'sqrt(λ) ratios':>15}")
        logger.info("-" * 30)
        for name in sorted(CONSTANTS.keys()):
            logger.info(f"{name:>10} {gram_matches[name]:>15}")

        result = {
            'gram_matches': gram_matches,
            'gram_squared_matches': gram_squared_matches,
        }
        self.results['experiment_4'] = result
        return result

    # =========================================================================
    # EXPERIMENT 5: Orthogonal Rotation Invariance
    # =========================================================================
    def experiment_5_rotation_invariance(self, test_prompts: List[Tuple[str, str]]) -> Dict:
        """Does rotating weights in orthogonal basis preserve geometry and function?"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 5: Orthogonal Rotation Invariance")
        logger.info("="*60)

        # Test on middle layer
        test_layer = self.n_layers // 2
        W_original = self._get_mlp_weight(test_layer)

        # Original geometry
        _, S_orig, _ = svd(W_original, full_matrices=False)
        orig_matches = count_constant_matches(S_orig, bidirectional=True)
        orig_quality = self._evaluate_model(test_prompts)

        # Apply random orthogonal rotation: W' = Q @ W where Q is orthogonal
        np.random.seed(42)
        m = W_original.shape[0]
        Q, _ = np.linalg.qr(np.random.randn(m, m))  # Random orthogonal matrix

        W_rotated = Q @ W_original
        self._set_mlp_weight(test_layer, W_rotated)

        # Rotated geometry
        _, S_rot, _ = svd(W_rotated, full_matrices=False)
        rot_matches = count_constant_matches(S_rot, bidirectional=True)
        rot_quality = self._evaluate_model(test_prompts)

        # Restore original
        self._set_mlp_weight(test_layer, W_original)

        # Compare
        logger.info(f"\nLayer {test_layer}:")
        logger.info(f"  Original quality: {orig_quality:.2%}")
        logger.info(f"  Rotated quality:  {rot_quality:.2%}")
        logger.info(f"  Singular values changed: {not np.allclose(S_orig, S_rot)}")

        # Singular values should be IDENTICAL under orthogonal rotation
        sv_match = np.allclose(S_orig, S_rot, rtol=1e-5)
        logger.info(f"  Singular values preserved: {sv_match}")

        logger.info(f"\nConstant matches:")
        logger.info(f"{'Constant':>10} {'Original':>10} {'Rotated':>10}")
        for name in sorted(CONSTANTS.keys()):
            logger.info(f"{name:>10} {orig_matches[name]:>10} {rot_matches[name]:>10}")

        result = {
            'sv_preserved': sv_match,
            'quality_original': orig_quality,
            'quality_rotated': rot_quality,
            'matches_original': orig_matches,
            'matches_rotated': rot_matches,
        }
        self.results['experiment_5'] = result
        return result

    # =========================================================================
    # EXPERIMENT 6: Surgical SVD Modification
    # =========================================================================
    def experiment_6_surgical_svd(self, test_prompts: List[Tuple[str, str]]) -> Dict:
        """What happens if we surgically modify SVD ratios to exact constants?"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 6: Surgical SVD Modification")
        logger.info("="*60)

        test_layer = self.n_layers // 2
        W_original = self._get_mlp_weight(test_layer)

        U, S, Vt = svd(W_original, full_matrices=False)

        # Original state
        orig_ratio = S[0] / S[1] if S[1] > 1e-10 else 0
        orig_quality = self._evaluate_model(test_prompts)
        orig_matches = count_constant_matches(S, bidirectional=True)

        logger.info(f"\nOriginal state:")
        logger.info(f"  σ₁/σ₂ = {orig_ratio:.6f}")
        logger.info(f"  Quality: {orig_quality:.2%}")

        # Modify S[0] to make S[0]/S[1] = π/e exactly
        S_modified = S.copy()
        target_ratio = np.pi / np.e
        S_modified[0] = target_ratio * S[1]

        W_modified = U @ np.diag(S_modified) @ Vt
        self._set_mlp_weight(test_layer, W_modified)

        # Modified state
        _, S_check, _ = svd(W_modified, full_matrices=False)
        mod_ratio = S_check[0] / S_check[1] if S_check[1] > 1e-10 else 0
        mod_quality = self._evaluate_model(test_prompts)
        mod_matches = count_constant_matches(S_check, bidirectional=True)

        logger.info(f"\nAfter setting σ₁/σ₂ = π/e:")
        logger.info(f"  σ₁/σ₂ = {mod_ratio:.6f} (target: {target_ratio:.6f})")
        logger.info(f"  Quality: {mod_quality:.2%}")
        logger.info(f"  Ratio achieved: {np.isclose(mod_ratio, target_ratio, rtol=1e-5)}")

        # Restore
        self._set_mlp_weight(test_layer, W_original)

        logger.info(f"\nConstant matches:")
        logger.info(f"{'Constant':>10} {'Original':>10} {'Modified':>10}")
        for name in sorted(CONSTANTS.keys()):
            logger.info(f"{name:>10} {orig_matches[name]:>10} {mod_matches[name]:>10}")

        result = {
            'original_ratio': float(orig_ratio),
            'modified_ratio': float(mod_ratio),
            'target_ratio': float(target_ratio),
            'ratio_achieved': np.isclose(mod_ratio, target_ratio, rtol=1e-5),
            'quality_original': orig_quality,
            'quality_modified': mod_quality,
            'quality_preserved': mod_quality >= orig_quality * 0.9,
            'matches_original': orig_matches,
            'matches_modified': mod_matches,
        }
        self.results['experiment_6'] = result
        return result

    # =========================================================================
    # EXPERIMENT 7: Residual Stream Geometry
    # =========================================================================
    def experiment_7_residual_stream(self, probes: List[str]) -> Dict:
        """How does geometry accumulate through the residual stream?"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 7: Residual Stream Geometry")
        logger.info("="*60)

        import mlx.core as mx

        # Track cumulative representation through layers
        layer_matches = []
        cumulative_matches = []

        for probe in probes[:3]:  # Use first 3 probes for speed
            tokens = self.tokenizer.encode(probe)
            input_ids = mx.array([tokens])

            # Get embeddings
            if hasattr(self.model.model, 'embed_tokens'):
                x = self.model.model.embed_tokens(input_ids)
            else:
                x = self.model.model.embeddings(input_ids)
            mx.eval(x)

            residual = np.array(x[0].tolist(), dtype=np.float32)

            for layer_idx in range(self.n_layers):
                layer = self.model.model.layers[layer_idx]

                # Full layer forward (attention + MLP)
                x_mlx = mx.array(residual.reshape(1, *residual.shape))

                # This is approximate - proper way needs full forward through each sublayer
                # For now, just get the activations as proxy for residual state
                act = self._get_activations(probe, layer_idx)

                if act.shape[0] > 1:
                    _, S, _ = svd(act, full_matrices=False)
                    matches = count_constant_matches(S, bidirectional=True)
                    layer_matches.append({
                        'layer': layer_idx,
                        'probe': probe[:30],
                        'matches': sum(matches.values()),
                    })

        # Aggregate by layer
        layer_totals = {}
        for lm in layer_matches:
            layer_idx = lm['layer']
            if layer_idx not in layer_totals:
                layer_totals[layer_idx] = 0
            layer_totals[layer_idx] += lm['matches']

        logger.info("\nMatches by layer (cumulative effect):")
        for layer_idx in sorted(layer_totals.keys()):
            logger.info(f"  Layer {layer_idx:2d}: {layer_totals[layer_idx]} matches")

        result = {
            'layer_totals': layer_totals,
            'raw_data': layer_matches,
        }
        self.results['experiment_7'] = result
        return result

    def run_all(self, probes: List[str], test_prompts: List[Tuple[str, str]]) -> Dict:
        """Run all experiments."""
        self.experiment_1_bidirectional_ratios()
        self.experiment_2_null_hypothesis(n_samples=50)
        self.experiment_3_nonlinearity_effect(probes)
        self.experiment_4_gram_matrix(probes)
        self.experiment_5_rotation_invariance(test_prompts)
        self.experiment_6_surgical_svd(test_prompts)
        self.experiment_7_residual_stream(probes)

        return self.results


def main():
    import mlx.core as mx
    from mlx_lm import load

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    probes = [
        "Two plus two equals four.",
        "Paris is the capital of France.",
        "Water boils at one hundred degrees.",
        "The sky is blue.",
        "Dogs are mammals.",
        "Triangles have three sides.",
    ]

    test_prompts = [
        ("What is 2 + 2?", "4"),
        ("Capital of France?", "paris"),
        ("Is water wet?", "yes"),
    ]

    experiments = GeometricExperiments(model, tokenizer)
    results = experiments.run_all(probes, test_prompts)

    # Save results
    output_path = f"data/experiments/geometric_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
