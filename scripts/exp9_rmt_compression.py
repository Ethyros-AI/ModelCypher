#!/usr/bin/env python3
"""Experiment 9: Test RMT-based rank detection for compression.

Hypothesis: Using Marchenko-Pastur signal/noise separation gives better
compression than naive pinv because it filters noise from the lstsq solution.

Compares:
1. Naive pinv: T = Y @ pinv(X)
2. RMT-aware: T = Y @ pinv_k(X) where k = signal_rank from MP distribution
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of compression attempt."""
    layer_idx: int
    method: str
    rank_used: int
    token_accuracy: float
    reconstruction_error: float
    margin_preserved: float  # fraction of samples where argmax margin sign preserved


def run_experiment():
    """Compare naive pinv vs RMT-aware compression."""
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.rmt_signal_separation import (
        compute_signal_rank_from_singular_values,
    )
    from modelcypher.core.domain.geometry.numerical_stability import (
        division_epsilon,
        geodesic_svd,
    )

    import mlx.core as mx

    initialize_default_backend()
    backend = get_default_backend()
    logger.info(f"Using backend: {type(backend).__name__}")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    # Get calibration prompts
    calibration_prompts = [
        "The capital of France is",
        "In mathematics, the derivative of",
        "The largest planet in our solar system is",
        "Water freezes at",
        "The speed of light is approximately",
        "Photosynthesis is the process by which",
        "The human heart has",
        "DNA stands for",
        "The chemical symbol for gold is",
        "Shakespeare wrote",
        "The Great Wall of China was built",
        "E = mc² was discovered by",
        "The mitochondria is",
        "Python is a programming language that",
        "Machine learning algorithms",
        "The stock market",
        "Climate change refers to",
        "Quantum mechanics describes",
        "The Renaissance was a period",
        "Artificial intelligence",
    ]

    held_out_prompts = [
        "The theory of relativity states",
        "Neurons in the brain",
        "The periodic table",
        "Evolution by natural selection",
    ]

    # Tokenize
    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_out_prompts]

    # Test layers - mix of known good and bad
    test_layers = [1, 2, 5, 6, 7, 10, 14]

    results = []

    for layer_idx in test_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer_idx}")
        logger.info(f"{'='*60}")

        # Collect activations for calibration
        cal_inputs = []
        cal_outputs = []

        for tokens in cal_tokens:
            input_ids = mx.array([tokens])

            # Hook to capture MLP input/output
            mlp_input = None
            mlp_output = None

            layer = model.model.layers[layer_idx]
            original_mlp = layer.mlp

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp

                def __call__(self, x):
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            layer.mlp = MLPHook(original_mlp)
            try:
                _ = model(input_ids)
                mx.eval(mlp_input, mlp_output)

                # Take last token position
                cal_inputs.append(mlp_input[0, -1, :])
                cal_outputs.append(mlp_output[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X_cal = mx.stack(cal_inputs)  # [n_cal, d_in]
        Y_cal = mx.stack(cal_outputs)  # [n_cal, d_out]
        mx.eval(X_cal, Y_cal)

        n_samples, d_in = X_cal.shape
        _, d_out = Y_cal.shape

        logger.info(f"Calibration: X={X_cal.shape}, Y={Y_cal.shape}")

        # Method 1: Naive pinv
        # T = Y @ pinv(X) = Y @ X.T @ inv(X @ X.T)
        X_cal_f32 = X_cal.astype(mx.float32)
        Y_cal_f32 = Y_cal.astype(mx.float32)

        # Full SVD for naive pinv (MLX requires CPU stream for SVD)
        # X is [n_samples, d_in] = [20, 4096]
        # SVD: X = U @ diag(S) @ Vt where U=[20,20], S=[20], Vt=[20, 4096]
        U, S, Vt = mx.linalg.svd(X_cal_f32, stream=mx.cpu)
        mx.eval(U, S, Vt)

        eps = 1e-6
        k = int(S.shape[0])  # Number of singular values = min(n, d) = 20
        S_inv_naive = 1.0 / (S + eps)  # [k]

        # MLX SVD returns full Vt [d, d] but only first k rows correspond to non-zero S
        # pinv(X) = V[:,:k] @ diag(1/S) @ U.T
        # V[:,:k] = Vt[:k,:].T = [d, k]
        Vt_k = Vt[:k, :]  # [k, d_in] = [20, 4096]
        V_k = Vt_k.T  # [4096, 20]
        VS = V_k * S_inv_naive  # [4096, 20] * [20] broadcast along last axis
        pinv_naive = mx.matmul(VS, U.T)  # [4096, 20] @ [20, 20] = [4096, 20]

        # We want T: [d_out, d_in] such that X @ T.T = Y
        # T.T = pinv(X) @ Y => T = (pinv(X) @ Y).T
        T_T_naive = mx.matmul(pinv_naive, Y_cal_f32)  # [4096, 20] @ [20, 4096] = [4096, 4096]
        T_naive = T_T_naive.T  # [4096, 4096]
        mx.eval(T_naive)

        naive_rank = k  # All singular values used

        # Method 2: RMT-aware (Marchenko-Pastur signal rank)
        # Convert to backend arrays for RMT function
        S_backend = backend.array(S)
        mp_result = compute_signal_rank_from_singular_values(
            S_backend, n_samples=n_samples, n_features=d_in, backend=backend
        )
        signal_rank = max(1, min(int(mp_result.signal_rank), int(S.shape[0])))

        logger.info(f"RMT: signal_rank={signal_rank}/{S.shape[0]}, MP_edge={mp_result.mp_upper_edge:.4f}")
        logger.info(f"RMT: signal_var_fraction={mp_result.signal_variance_fraction:.2%}")

        # Rank-truncated pinv using RMT signal rank
        U_sr = U[:, :signal_rank]  # [20, signal_rank]
        S_sr = S[:signal_rank]      # [signal_rank]
        Vt_sr = Vt[:signal_rank, :] # [signal_rank, 4096]

        S_inv_rmt = 1.0 / (S_sr + eps)  # [signal_rank]
        V_sr = Vt_sr.T  # [4096, signal_rank]
        VS_sr = V_sr * S_inv_rmt  # [4096, signal_rank]
        pinv_rmt = mx.matmul(VS_sr, U_sr.T)  # [4096, signal_rank] @ [signal_rank, 20] = [4096, 20]

        T_T_rmt = mx.matmul(pinv_rmt, Y_cal_f32)  # [4096, 20] @ [20, 4096] = [4096, 4096]
        T_rmt = T_T_rmt.T  # [4096, 4096]
        mx.eval(T_rmt)

        # Evaluate on held-out prompts
        def evaluate_compression(T, method_name, rank_used):
            correct = 0
            total = 0
            margins_preserved = 0
            total_recon_error = 0.0

            for tokens in held_tokens:
                input_ids = mx.array([tokens])

                # Get original logits
                orig_logits = model(input_ids)
                mx.eval(orig_logits)
                orig_logits_last = orig_logits[0, -1, :]
                orig_top = int(mx.argmax(orig_logits_last).item())

                # Get original margin (top1 - top2)
                sorted_orig = mx.sort(orig_logits_last)[::-1]
                orig_margin = float((sorted_orig[0] - sorted_orig[1]).item())

                # Now run with compression
                layer = model.model.layers[layer_idx]
                original_mlp = layer.mlp

                class CompressedMLP:
                    def __init__(self, T):
                        self.T = T

                    def __call__(self, x):
                        # Linear approximation: y = x @ T.T
                        return mx.matmul(x, self.T.T)

                layer.mlp = CompressedMLP(T)
                try:
                    comp_logits = model(input_ids)
                    mx.eval(comp_logits)
                    comp_logits_last = comp_logits[0, -1, :]
                    comp_top = int(mx.argmax(comp_logits_last).item())

                    # Compressed margin
                    sorted_comp = mx.sort(comp_logits_last)[::-1]
                    comp_margin = float((sorted_comp[0] - sorted_comp[1]).item())

                    if comp_top == orig_top:
                        correct += 1

                    # Margin sign preserved?
                    if (orig_margin > 0 and comp_margin > 0) or (orig_margin <= 0 and comp_margin <= 0):
                        margins_preserved += 1

                    total += 1

                    # Reconstruction error on this sample's MLP
                    mlp_in = None
                    mlp_out_orig = None

                    class CaptureHook:
                        def __init__(self, mlp):
                            self.mlp = mlp
                        def __call__(self, x):
                            nonlocal mlp_in, mlp_out_orig
                            mlp_in = x
                            mlp_out_orig = self.mlp(x)
                            return mlp_out_orig

                    layer.mlp = CaptureHook(original_mlp)
                    _ = model(input_ids)
                    mx.eval(mlp_in, mlp_out_orig)

                    mlp_out_comp = mx.matmul(mlp_in, T.T)
                    mx.eval(mlp_out_comp)

                    error = float(mx.sqrt(mx.sum((mlp_out_orig - mlp_out_comp) ** 2)).item())
                    norm = float(mx.sqrt(mx.sum(mlp_out_orig ** 2)).item())
                    total_recon_error += error / (norm + 1e-8)

                finally:
                    layer.mlp = original_mlp

            accuracy = correct / total if total > 0 else 0.0
            margin_frac = margins_preserved / total if total > 0 else 0.0
            avg_error = total_recon_error / total if total > 0 else 0.0

            return CompressionResult(
                layer_idx=layer_idx,
                method=method_name,
                rank_used=rank_used,
                token_accuracy=accuracy,
                reconstruction_error=avg_error,
                margin_preserved=margin_frac,
            )

        # Evaluate both methods
        result_naive = evaluate_compression(T_naive, "naive_pinv", naive_rank)
        result_rmt = evaluate_compression(T_rmt, "rmt_pinv", signal_rank)

        results.append(result_naive)
        results.append(result_rmt)

        logger.info(f"\nNaive pinv (rank={naive_rank}):")
        logger.info(f"  Token accuracy: {result_naive.token_accuracy:.1%}")
        logger.info(f"  Recon error: {result_naive.reconstruction_error:.4f}")
        logger.info(f"  Margin preserved: {result_naive.margin_preserved:.1%}")

        logger.info(f"\nRMT pinv (rank={signal_rank}):")
        logger.info(f"  Token accuracy: {result_rmt.token_accuracy:.1%}")
        logger.info(f"  Recon error: {result_rmt.reconstruction_error:.4f}")
        logger.info(f"  Margin preserved: {result_rmt.margin_preserved:.1%}")

        # Compare
        if result_rmt.token_accuracy > result_naive.token_accuracy:
            logger.info(f"\n>>> RMT WINS by {(result_rmt.token_accuracy - result_naive.token_accuracy)*100:.1f}pp")
        elif result_rmt.token_accuracy < result_naive.token_accuracy:
            logger.info(f"\n>>> NAIVE WINS by {(result_naive.token_accuracy - result_rmt.token_accuracy)*100:.1f}pp")
        else:
            logger.info(f"\n>>> TIE")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    naive_wins = 0
    rmt_wins = 0
    ties = 0

    for i in range(0, len(results), 2):
        naive = results[i]
        rmt = results[i+1]

        if rmt.token_accuracy > naive.token_accuracy:
            rmt_wins += 1
            winner = "RMT"
        elif naive.token_accuracy > rmt.token_accuracy:
            naive_wins += 1
            winner = "NAIVE"
        else:
            ties += 1
            winner = "TIE"

        logger.info(f"Layer {naive.layer_idx}: naive={naive.token_accuracy:.0%} vs rmt={rmt.token_accuracy:.0%} -> {winner}")

    logger.info(f"\nOverall: Naive wins={naive_wins}, RMT wins={rmt_wins}, Ties={ties}")

    if rmt_wins > naive_wins:
        logger.info("\n>>> RMT-BASED RANK DETECTION HELPS COMPRESSION")
    elif naive_wins > rmt_wins:
        logger.info("\n>>> NAIVE PINV IS SUFFICIENT")
    else:
        logger.info("\n>>> NO CLEAR WINNER")


if __name__ == "__main__":
    run_experiment()
