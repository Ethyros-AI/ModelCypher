#!/usr/bin/env python3
"""Experiment 10: Test active subspace projection for compression.

Hypothesis: The attention transformation operates in ~465 dimensions (not full 4096).
Compressing in the active subspace might preserve ranking better because we're
working in the space where the model actually computes.

Compares:
1. RMT pinv (baseline from exp9)
2. Active subspace: project X,Y into variance-defined subspace, then compress
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    layer_idx: int
    method: str
    rank_used: int
    token_accuracy: float
    active_rank: int  # dimensions in active subspace


def run_experiment():
    """Compare RMT pinv vs active subspace compression."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.rmt_signal_separation import (
        compute_signal_rank_from_singular_values,
    )
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
        compute_variance_null_space,
    )

    initialize_default_backend()
    backend = get_default_backend()
    logger.info(f"Using backend: {type(backend).__name__}")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    # Calibration and held-out prompts
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

    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_out_prompts]

    # Test layers
    test_layers = [1, 2, 5, 6, 7, 10, 14]

    results = []

    for layer_idx in test_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer_idx}")
        logger.info(f"{'='*60}")

        # Collect activations
        cal_inputs = []
        cal_outputs = []

        for tokens in cal_tokens:
            input_ids = mx.array([tokens])
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
                cal_inputs.append(mlp_input[0, -1, :])
                cal_outputs.append(mlp_output[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X_cal = mx.stack(cal_inputs).astype(mx.float32)
        Y_cal = mx.stack(cal_outputs).astype(mx.float32)
        mx.eval(X_cal, Y_cal)

        n_samples, d_in = X_cal.shape
        _, d_out = Y_cal.shape
        logger.info(f"Calibration: X={X_cal.shape}, Y={Y_cal.shape}")

        # Method 1: RMT pinv (baseline)
        U, S, Vt = mx.linalg.svd(X_cal, stream=mx.cpu)
        mx.eval(U, S, Vt)

        S_backend = backend.array(S)
        mp_result = compute_signal_rank_from_singular_values(
            S_backend, n_samples=n_samples, n_features=d_in, backend=backend
        )
        signal_rank = max(1, min(int(mp_result.signal_rank), int(S.shape[0])))

        eps = 1e-6
        U_sr = U[:, :signal_rank]
        S_sr = S[:signal_rank]
        Vt_sr = Vt[:signal_rank, :]
        S_inv = 1.0 / (S_sr + eps)
        V_sr = Vt_sr.T
        VS_sr = V_sr * S_inv
        pinv_rmt = mx.matmul(VS_sr, U_sr.T)
        T_rmt = mx.matmul(pinv_rmt, Y_cal).T
        mx.eval(T_rmt)

        logger.info(f"RMT: signal_rank={signal_rank}")

        # Method 2: Active subspace compression
        # Step 1: Find the active subspace of X (high variance directions)
        X_backend = backend.array(X_cal)
        null_result = compute_variance_null_space(X_backend, backend=backend)
        active_rank = int(null_result.utilized_rank)
        utilized_basis = null_result.utilized_basis  # [d, active_rank]

        logger.info(f"Active subspace: rank={active_rank} (of {d_in})")

        # Step 2: Project X and Y into active subspace
        # X_proj = X @ U_active, Y_proj = Y @ U_active (if Y is in same space)
        # But Y might be in a different subspace... let's think about this
        #
        # Actually the MLP does: y = MLP(x) where x,y are in hidden space
        # So both X and Y are in the same 4096-dim space
        # We can project both into the active subspace of X

        # Convert backend array to MLX array via list serialization
        # (to_numpy is disabled in this codebase)
        U_active_list = backend.tolist(utilized_basis)
        U_active = mx.array(U_active_list).astype(mx.float32)  # [4096, active_rank]
        mx.eval(U_active)

        X_proj = mx.matmul(X_cal, U_active)  # [n, active_rank]
        Y_proj = mx.matmul(Y_cal, U_active)  # [n, active_rank]
        mx.eval(X_proj, Y_proj)

        # Step 3: Compute T in the projected space
        # T_proj: [active_rank, active_rank]
        U_p, S_p, Vt_p = mx.linalg.svd(X_proj, stream=mx.cpu)
        mx.eval(U_p, S_p, Vt_p)

        # Use RMT on projected space too
        S_p_backend = backend.array(S_p)
        mp_proj = compute_signal_rank_from_singular_values(
            S_p_backend, n_samples=n_samples, n_features=active_rank, backend=backend
        )
        proj_rank = max(1, min(int(mp_proj.signal_rank), int(S_p.shape[0])))

        U_pr = U_p[:, :proj_rank]
        S_pr = S_p[:proj_rank]
        Vt_pr = Vt_p[:proj_rank, :]
        S_inv_p = 1.0 / (S_pr + eps)
        V_pr = Vt_pr.T
        VS_pr = V_pr * S_inv_p
        pinv_proj = mx.matmul(VS_pr, U_pr.T)
        T_proj = mx.matmul(pinv_proj, Y_proj).T  # [active_rank, active_rank]
        mx.eval(T_proj)

        # Step 4: The full transform is: x -> project -> T_proj -> unproject
        # T_full = U_active @ T_proj @ U_active.T
        T_active = mx.matmul(mx.matmul(U_active, T_proj), U_active.T)  # [4096, 4096]
        mx.eval(T_active)

        logger.info(f"Active proj_rank={proj_rank}")

        # Evaluate both methods
        def evaluate_compression(T, method_name, rank_used, active_r):
            correct = 0
            total = 0

            for tokens in held_tokens:
                input_ids = mx.array([tokens])

                # Original logits
                orig_logits = model(input_ids)
                mx.eval(orig_logits)
                orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

                # Compressed
                layer = model.model.layers[layer_idx]
                original_mlp = layer.mlp

                class CompressedMLP:
                    def __init__(self, T):
                        self.T = T
                    def __call__(self, x):
                        return mx.matmul(x, self.T.T)

                layer.mlp = CompressedMLP(T)
                try:
                    comp_logits = model(input_ids)
                    mx.eval(comp_logits)
                    comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())

                    if comp_top == orig_top:
                        correct += 1
                    total += 1
                finally:
                    layer.mlp = original_mlp

            accuracy = correct / total if total > 0 else 0.0
            return CompressionResult(
                layer_idx=layer_idx,
                method=method_name,
                rank_used=rank_used,
                token_accuracy=accuracy,
                active_rank=active_r,
            )

        result_rmt = evaluate_compression(T_rmt, "rmt_pinv", signal_rank, d_in)
        result_active = evaluate_compression(T_active, "active_subspace", proj_rank, active_rank)

        results.append(result_rmt)
        results.append(result_active)

        logger.info(f"\nRMT pinv (rank={signal_rank}):")
        logger.info(f"  Token accuracy: {result_rmt.token_accuracy:.1%}")

        logger.info(f"\nActive subspace (active_dim={active_rank}, proj_rank={proj_rank}):")
        logger.info(f"  Token accuracy: {result_active.token_accuracy:.1%}")

        if result_active.token_accuracy > result_rmt.token_accuracy:
            logger.info(f"\n>>> ACTIVE WINS by {(result_active.token_accuracy - result_rmt.token_accuracy)*100:.1f}pp")
        elif result_active.token_accuracy < result_rmt.token_accuracy:
            logger.info(f"\n>>> RMT WINS by {(result_rmt.token_accuracy - result_active.token_accuracy)*100:.1f}pp")
        else:
            logger.info(f"\n>>> TIE")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    rmt_wins = 0
    active_wins = 0
    ties = 0

    for i in range(0, len(results), 2):
        rmt = results[i]
        active = results[i+1]

        if active.token_accuracy > rmt.token_accuracy:
            active_wins += 1
            winner = "ACTIVE"
        elif rmt.token_accuracy > active.token_accuracy:
            rmt_wins += 1
            winner = "RMT"
        else:
            ties += 1
            winner = "TIE"

        logger.info(f"Layer {rmt.layer_idx}: rmt={rmt.token_accuracy:.0%} vs active={active.token_accuracy:.0%} (dim={active.active_rank}) -> {winner}")

    logger.info(f"\nOverall: RMT wins={rmt_wins}, Active wins={active_wins}, Ties={ties}")

    if active_wins > rmt_wins:
        logger.info("\n>>> ACTIVE SUBSPACE HELPS COMPRESSION")
    elif rmt_wins > active_wins:
        logger.info("\n>>> RMT IS SUFFICIENT, ACTIVE SUBSPACE DOESN'T HELP")
    else:
        logger.info("\n>>> NO CLEAR WINNER")


if __name__ == "__main__":
    run_experiment()
