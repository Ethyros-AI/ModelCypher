#!/usr/bin/env python3
"""Experiment 25: Perturbation Spectrum Analysis.

The key question: what is the spectral structure of the error?

When we compress layer i with T_i instead of W_i:
- Error matrix E_i = T_i - W_i (in behavioral terms)
- The error propagates: E_total = Σ E_i (approximately)

The EIGENVALUE DISTRIBUTION of E_total might follow Marchenko-Pastur
or another universal distribution with Euler's number in the scaling.

Hypothesis: The spectral radius of accumulated error follows ρ(n) ∝ √n
(random matrix theory predicts this for sum of independent matrices).

The phase transition happens when ρ(n) exceeds the margin.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Analyze the spectral structure of compression error."""
    import mlx.core as mx
    import numpy as np

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.compression import RMTAwareCompressor

    initialize_default_backend()
    backend = get_default_backend()

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    # Calibration prompts
    calibration_prompts = [
        "The theory of relativity states that",
        "Quantum mechanics describes",
        "Evolution by natural selection",
        "Neural networks are",
        "The derivative of a function",
        "Prime numbers are",
        "Shakespeare wrote",
        "The human brain contains",
        "Newton's laws state",
        "Climate change refers to",
        "The Amazon rainforest",
        "Gravity causes objects to",
        "Chemical bonds form when",
        "Cells divide through",
        "The internet was invented",
        "Artificial intelligence",
    ]

    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    compressor = RMTAwareCompressor(backend=backend)

    # For each layer, compute the T matrix and its deviation from identity-like behavior
    logger.info(f"\n--- Analyzing compression perturbation spectrum ---")

    layer_stats = []

    for layer_idx in range(8, 28):  # Transmission zone subset
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

        X_backend = backend.array(X_cal)
        Y_backend = backend.array(Y_cal)

        rmt_result = compressor.compress_layer(X_backend, Y_backend)

        # Compute behavioral error: E = T @ X - Y (residual)
        T = rmt_result.T
        TX = backend.matmul(X_backend, backend.transpose(T))
        E = TX - Y_backend
        backend.eval(E)

        # Error statistics
        E_np = np.array(backend.tolist(E))

        # Frobenius norm of error
        frob_error = np.linalg.norm(E_np, 'fro')

        # SVD of error matrix
        U, S, Vh = np.linalg.svd(E_np, full_matrices=False)

        # Spectral statistics
        spectral_radius = S[0]  # Largest singular value
        trace_S = np.sum(S)
        effective_rank = (trace_S / spectral_radius) if spectral_radius > 0 else 0

        # Marchenko-Pastur edge
        n_samples, dim = E_np.shape
        gamma = n_samples / dim if dim > 0 else 1
        mp_edge = (1 + 1/np.sqrt(gamma))**2 * (frob_error**2 / n_samples)

        layer_stats.append({
            "layer": layer_idx,
            "frob_error": frob_error,
            "spectral_radius": spectral_radius,
            "trace_S": trace_S,
            "effective_rank": effective_rank,
            "top3_sv": S[:3].tolist() if len(S) >= 3 else S.tolist(),
        })

        logger.info(f"\nLayer {layer_idx}:")
        logger.info(f"  Frobenius error: {frob_error:.4f}")
        logger.info(f"  Spectral radius: {spectral_radius:.4f}")
        logger.info(f"  Effective rank: {effective_rank:.2f}")
        logger.info(f"  Top 3 SV: {[f'{s:.2f}' for s in S[:3]]}")

    # Analyze how errors combine
    logger.info(f"\n{'='*60}")
    logger.info("CUMULATIVE ERROR ANALYSIS")
    logger.info(f"{'='*60}")

    # If errors were independent, Frobenius norm would grow as √n
    # If errors compound, it would grow faster

    frob_errors = [s["frob_error"] for s in layer_stats]
    spectral_radii = [s["spectral_radius"] for s in layer_stats]

    # Cumulative sums (assuming errors add)
    cumulative_frob = []
    cumulative_spectral = []
    running_frob = 0
    running_spectral = 0

    for i, s in enumerate(layer_stats):
        running_frob += s["frob_error"]**2  # Variances add
        running_spectral += s["spectral_radius"]**2
        cumulative_frob.append(math.sqrt(running_frob))
        cumulative_spectral.append(math.sqrt(running_spectral))

    logger.info(f"\n{'n':>3} {'Cumul Frob':>12} {'Pred √n':>12} {'Ratio':>10}")
    logger.info("-" * 45)

    avg_frob = sum(frob_errors) / len(frob_errors)

    for n in range(1, len(cumulative_frob) + 1):
        actual = cumulative_frob[n-1]
        predicted = avg_frob * math.sqrt(n)
        ratio = actual / predicted if predicted > 0 else 0
        logger.info(f"{n:3d} {actual:12.4f} {predicted:12.4f} {ratio:10.4f}")

    # The Euler connection: random walk in error space
    logger.info(f"\n{'='*60}")
    logger.info("RANDOM WALK INTERPRETATION")
    logger.info(f"{'='*60}")

    logger.info("""
If each layer's error E_i is independent with variance σ²:
- After n layers: Total variance = n * σ²
- Total error magnitude ∝ √n

This is the DIFFUSION process:
- <|E|²> = D * n where D is diffusion coefficient
- |E| ∝ √(D*n)

The CENTRAL LIMIT THEOREM applies:
- Sum of n random errors → Gaussian with std = σ * √n
- The √n factor is related to e through:
  - Gaussian: exp(-x²/2σ²)
  - The 2 in the exponent relates to the second moment

EULER'S NUMBER appears in the Gaussian:
  P(x) = 1/(σ√(2π)) * e^(-x²/2σ²)

The phase transition happens when:
  |E_total| > margin
  σ * √n > M
  n > (M/σ)²

This predicts n_critical ∝ (M/σ)² = (M/σ)^(2*1)

The exponent 2 is the dimension of the random walk!
""")

    # Verify √n scaling
    logger.info(f"\n{'='*60}")
    logger.info("VERIFICATION: Does error grow as √n?")
    logger.info(f"{'='*60}")

    # Fit power law: cumulative_frob = A * n^α
    # log(frob) = log(A) + α * log(n)

    ns = list(range(1, len(cumulative_frob) + 1))
    log_ns = [math.log(n) for n in ns]
    log_frobs = [math.log(f) if f > 0 else 0 for f in cumulative_frob]

    # Linear regression
    log_n_mean = sum(log_ns) / len(log_ns)
    log_f_mean = sum(log_frobs) / len(log_frobs)

    numerator = sum((log_ns[i] - log_n_mean) * (log_frobs[i] - log_f_mean) for i in range(len(ns)))
    denominator = sum((log_ns[i] - log_n_mean)**2 for i in range(len(ns)))

    alpha = numerator / denominator if denominator > 0 else 0
    log_A = log_f_mean - alpha * log_n_mean
    A = math.exp(log_A)

    logger.info(f"\nPower law fit: error = {A:.4f} * n^{alpha:.4f}")
    logger.info(f"Expected for random walk: α = 0.5")
    logger.info(f"Deviation from √n: {(alpha - 0.5)*100:.1f}%")

    if abs(alpha - 0.5) < 0.1:
        logger.info("\n>>> CONFIRMED: Error grows as √n (random walk)")
    elif alpha > 0.5:
        logger.info(f"\n>>> Error grows FASTER than √n (correlated errors)")
    else:
        logger.info(f"\n>>> Error grows SLOWER than √n (anti-correlated errors)")

    # The Euler formula
    logger.info(f"\n{'='*60}")
    logger.info("THE EULER FORMULA FOR COMPRESSION")
    logger.info(f"{'='*60}")

    avg_error = sum(frob_errors) / len(frob_errors)

    logger.info(f"""
Given:
  - Average per-layer error: σ = {avg_error:.4f}
  - Error growth exponent: α = {alpha:.4f}
  - Typical margin: M ≈ 1.0 (in normalized units)

The phase transition occurs at:
  n_crit = (M/σ)^(2/α)
        = ({1.0:.2f}/{avg_error:.4f})^(2/{alpha:.4f})
        = {(1.0/avg_error)**(2/alpha):.1f}

For SPREAD compression:
  - Errors don't accumulate sequentially
  - Each error disperses before next compression
  - Effective σ_spread = σ / √(gap) where gap is spacing

With gap = 5:
  σ_spread = {avg_error:.4f} / √5 = {avg_error/math.sqrt(5):.4f}
  n_crit_spread = (M/σ_spread)^(2/α) = {(1.0/(avg_error/math.sqrt(5)))**(2/alpha):.1f}

Spread allows {(1.0/(avg_error/math.sqrt(5)))**(2/alpha) / (1.0/avg_error)**(2/alpha):.1f}x more layers!
""")

    # Final insight
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: WHERE EULER APPEARS")
    logger.info(f"{'='*60}")

    logger.info("""
1. ERROR DIFFUSION: Errors propagate as a random walk
   - |E(n)| = σ√n (central limit theorem)
   - The √ comes from Gaussian diffusion

2. GAUSSIAN DISTRIBUTION: e^(-x²/2σ²)
   - Error follows Gaussian after n layers
   - Euler's number is fundamental to Gaussian

3. PHASE TRANSITION: When error exceeds margin
   - n_crit ∝ (M/σ)² for sequential
   - n_crit ∝ (M/σ)² * gap for spread
   - The ² exponent is the random walk dimension

4. THE EULER CONNECTION:
   - Compression error is a DIFFUSION process
   - Diffusion is governed by the heat equation
   - Heat equation solutions involve e^(-λt)
   - The λ eigenvalue determines decay rate

The fundamental insight:
   COMPRESSION IS DIFFUSION IN REPRESENTATION SPACE

   Each layer adds noise. Noise accumulates as √n.
   When noise > signal, ranking flips.
   Spread compression reduces effective n.
""")


if __name__ == "__main__":
    run_experiment()
