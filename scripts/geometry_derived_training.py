#!/usr/bin/env python3
"""Experiment 59: Geometry-Derived Training.

ALL parameters derived from the geometry itself:
- Learning rate from Gram matrix condition number
- Stopping threshold from numerical precision
- Convergence from gradient/loss ratio

NO arbitrary constants. NO heuristics.
The geometry tells us what to do.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# The prompts we're aligning
ALIGNMENT_PAIRS = [
    # (counting_prompt, symbolic_prompt, expected_output)
    ("1, 2, 3, 4,", "4+1=", "5"),
    ("2, 3, 4, 5,", "5+1=", "6"),
    ("3, 4, 5, 6,", "6+1=", "7"),
    ("4, 5, 6, 7,", "7+1=", "8"),
    ("5, 6, 7, 8,", "8+1=", "9"),
    ("6, 7, 8, 9,", "9+1=", "10"),
    ("Count to 5: 1, 2, 3, 4,", "4+1=", "5"),
    ("Count: one, two, three,", "3+1=", "four"),
]


class GeometryDerivedTrainer:
    """Training where ALL hyperparameters come from the geometry."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dtype_eps = np.finfo(np.float32).eps  # 1.19e-7

    def _get_logits(self, prompt: str) -> np.ndarray:
        import mlx.core as mx
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    def get_gram_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get Gram matrices for counting and symbolic prompts."""
        counting_acts = []
        symbolic_acts = []

        for counting_prompt, symbolic_prompt, _ in ALIGNMENT_PAIRS:
            counting_acts.append(self._get_logits(counting_prompt))
            symbolic_acts.append(self._get_logits(symbolic_prompt))

        C = np.vstack(counting_acts)
        S = np.vstack(symbolic_acts)

        # Center
        C_c = C - C.mean(axis=0)
        S_c = S - S.mean(axis=0)

        # Gram matrices
        G_count = C_c @ C_c.T
        G_symb = S_c @ S_c.T

        return G_count, G_symb

    def compute_geometry_params(self) -> Dict:
        """Derive ALL training parameters from the geometry."""
        G_count, G_symb = self.get_gram_matrices()

        # Condition numbers tell us the numerical stability
        kappa_count = np.linalg.cond(G_count)
        kappa_symb = np.linalg.cond(G_symb)
        kappa = max(kappa_count, kappa_symb)

        # Frobenius norms give us the scale
        norm_count = np.linalg.norm(G_count, 'fro')
        norm_symb = np.linalg.norm(G_symb, 'fro')
        scale = (norm_count + norm_symb) / 2

        # Learning rate: 1/κ ensures we stay in linear regime
        # But for full model, we need to scale by the loss curvature
        # The Gram curvature ≈ 1/scale, so LR ≈ 1/(κ × scale)
        lr = 1.0 / (kappa * scale)

        # Stopping threshold: when loss is within numerical precision of zero
        # Given condition number κ, achievable precision is κ × eps
        stop_threshold = kappa * np.sqrt(self.dtype_eps)

        # Convergence criterion: relative change in loss
        # Should be < sqrt(eps) to be numerically stable
        convergence_threshold = np.sqrt(self.dtype_eps)

        logger.info("=== GEOMETRY-DERIVED PARAMETERS ===")
        logger.info(f"Gram condition (counting): {kappa_count:.4e}")
        logger.info(f"Gram condition (symbolic): {kappa_symb:.4e}")
        logger.info(f"Max condition number κ: {kappa:.4e}")
        logger.info(f"Gram scale: {scale:.4e}")
        logger.info(f"dtype eps: {self.dtype_eps:.4e}")
        logger.info(f"")
        logger.info(f"DERIVED:")
        logger.info(f"  Learning rate = 1/(κ×scale) = {lr:.4e}")
        logger.info(f"  Stop threshold = κ×√eps = {stop_threshold:.4e}")
        logger.info(f"  Convergence = √eps = {convergence_threshold:.4e}")

        return {
            "kappa_count": float(kappa_count),
            "kappa_symb": float(kappa_symb),
            "kappa": float(kappa),
            "scale": float(scale),
            "lr": float(lr),
            "stop_threshold": float(stop_threshold),
            "convergence_threshold": float(convergence_threshold),
        }

    def compute_alignment_loss(self) -> float:
        """Compute alignment loss between Gram matrices."""
        G_count, G_symb = self.get_gram_matrices()

        # Normalize
        G_count_norm = G_count / (np.linalg.norm(G_count, 'fro') + 1e-10)
        G_symb_norm = G_symb / (np.linalg.norm(G_symb, 'fro') + 1e-10)

        # Alignment loss
        diff = G_count_norm - G_symb_norm
        return float(np.linalg.norm(diff, 'fro'))

    def compute_supervised_loss_and_grad(self):
        """Compute supervised loss on symbolic → correct answer."""
        import mlx.core as mx

        def loss_fn(model):
            total_loss = mx.array(0.0)

            for _, symbolic_prompt, expected in ALIGNMENT_PAIRS:
                # Get target token
                expected_tokens = self.tokenizer.encode(expected)
                if not expected_tokens:
                    continue
                target_token = expected_tokens[0]

                # Forward pass
                tokens = self.tokenizer.encode(symbolic_prompt)
                input_ids = mx.array([tokens])
                logits = model(input_ids)
                next_logits = logits[0, -1, :]

                # Cross-entropy
                log_probs = mx.log(mx.softmax(next_logits) + 1e-10)
                loss = -log_probs[target_token]
                total_loss = total_loss + loss

            return total_loss / len(ALIGNMENT_PAIRS)

        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        mx.eval(loss, grads)

        return float(loss.item()), grads

    def compute_gradient_norm(self, grads) -> float:
        """Compute the Frobenius norm of all gradients."""
        import mlx.core as mx

        total_sq = 0.0
        for name, param in grads.items():
            if hasattr(param, 'items'):
                for sub_name, sub_param in param.items():
                    if hasattr(sub_param, 'shape'):
                        total_sq += float(mx.sum(sub_param ** 2).item())
            elif hasattr(param, 'shape'):
                total_sq += float(mx.sum(param ** 2).item())

        return np.sqrt(total_sq)

    def training_step(self, lr: float):
        """Single training step."""
        import mlx.core as mx
        import mlx.optimizers as optim

        loss, grads = self.compute_supervised_loss_and_grad()
        grad_norm = self.compute_gradient_norm(grads)

        optimizer = optim.SGD(learning_rate=lr)
        optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())

        return loss, grad_norm

    def evaluate(self) -> Dict:
        """Evaluate counting and symbolic accuracy."""
        import mlx.core as mx

        results = {"counting": [], "symbolic": []}

        for counting_prompt, symbolic_prompt, expected in ALIGNMENT_PAIRS:
            # Counting
            tokens = self.tokenizer.encode(counting_prompt)
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            output = self.tokenizer.decode([next_token]).strip()
            results["counting"].append(expected in output)

            # Symbolic
            tokens = self.tokenizer.encode(symbolic_prompt)
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            output = self.tokenizer.decode([next_token]).strip()
            results["symbolic"].append(expected in output)

        return {
            "counting_accuracy": sum(results["counting"]) / len(results["counting"]),
            "symbolic_accuracy": sum(results["symbolic"]) / len(results["symbolic"]),
            "counting_correct": sum(results["counting"]),
            "symbolic_correct": sum(results["symbolic"]),
            "total": len(ALIGNMENT_PAIRS),
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 59: GEOMETRY-DERIVED TRAINING")
        logger.info("=" * 60)
        logger.info("\nALL parameters from geometry. NO heuristics.\n")

        # Get geometry-derived parameters
        geo_params = self.compute_geometry_params()

        # Initial state
        initial_loss = self.compute_alignment_loss()
        initial_eval = self.evaluate()

        logger.info("\n=== INITIAL STATE ===")
        logger.info(f"Alignment loss: {initial_loss:.4f}")
        logger.info(f"Counting: {initial_eval['counting_correct']}/{initial_eval['total']}")
        logger.info(f"Symbolic: {initial_eval['symbolic_correct']}/{initial_eval['total']}")

        results = {
            "geometry_params": geo_params,
            "initial": {
                "alignment_loss": initial_loss,
                **initial_eval,
            },
            "iterations": [],
        }

        # Training loop - geometry determines when to stop
        logger.info("\n=== GEOMETRY-DERIVED TRAINING ===")
        lr = geo_params["lr"]
        stop_threshold = geo_params["stop_threshold"]
        convergence_threshold = geo_params["convergence_threshold"]

        prev_loss = initial_loss
        iteration = 0

        while True:
            # Training step
            supervised_loss, grad_norm = self.training_step(lr)

            # Check alignment loss (the true objective)
            current_loss = self.compute_alignment_loss()
            eval_result = self.evaluate()

            # Compute relative change
            rel_change = abs(current_loss - prev_loss) / (prev_loss + 1e-10)

            logger.info(f"Iter {iteration}: loss={current_loss:.4f}, "
                       f"grad={grad_norm:.4e}, rel_change={rel_change:.4e}, "
                       f"count={eval_result['counting_correct']}/{eval_result['total']}, "
                       f"symb={eval_result['symbolic_correct']}/{eval_result['total']}")

            results["iterations"].append({
                "iteration": iteration,
                "alignment_loss": current_loss,
                "supervised_loss": supervised_loss,
                "gradient_norm": grad_norm,
                "relative_change": rel_change,
                **eval_result,
            })

            # Stopping criteria - ALL from geometry
            # 1. Loss below achievable precision
            if current_loss < stop_threshold:
                logger.info(f"STOPPED: loss {current_loss:.4e} < threshold {stop_threshold:.4e}")
                results["stop_reason"] = "below_threshold"
                break

            # 2. Converged (relative change below numerical precision)
            if rel_change < convergence_threshold:
                logger.info(f"STOPPED: converged (rel_change {rel_change:.4e} < {convergence_threshold:.4e})")
                results["stop_reason"] = "converged"
                break

            # 3. Gradient vanished (no more information)
            if grad_norm < np.sqrt(self.dtype_eps):
                logger.info(f"STOPPED: gradient vanished ({grad_norm:.4e})")
                results["stop_reason"] = "gradient_vanished"
                break

            # 4. Loss increased significantly (wrong direction)
            if current_loss > prev_loss * (1 + np.sqrt(self.dtype_eps)):
                logger.info(f"STOPPED: loss increased ({current_loss:.4f} > {prev_loss:.4f})")
                results["stop_reason"] = "loss_increased"
                break

            # Safety: but even this should be geometry-derived in principle
            # The number of iterations should relate to κ
            if iteration > geo_params["kappa"]:
                logger.info(f"STOPPED: exceeded κ iterations ({iteration} > {geo_params['kappa']:.0f})")
                results["stop_reason"] = "max_iterations"
                break

            prev_loss = current_loss
            iteration += 1

        # Final state
        final_loss = self.compute_alignment_loss()
        final_eval = self.evaluate()

        logger.info("\n=== FINAL STATE ===")
        logger.info(f"Alignment loss: {final_loss:.4f}")
        logger.info(f"Counting: {final_eval['counting_correct']}/{final_eval['total']}")
        logger.info(f"Symbolic: {final_eval['symbolic_correct']}/{final_eval['total']}")

        results["final"] = {
            "alignment_loss": final_loss,
            **final_eval,
        }

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Alignment loss: {initial_loss:.4f} → {final_loss:.4f}")
        logger.info(f"Symbolic: {initial_eval['symbolic_accuracy']:.0%} → {final_eval['symbolic_accuracy']:.0%}")
        logger.info(f"Counting: {initial_eval['counting_accuracy']:.0%} → {final_eval['counting_accuracy']:.0%}")

        if final_loss < initial_loss and final_eval['symbolic_accuracy'] > initial_eval['symbolic_accuracy']:
            logger.info("\n*** GEOMETRY-DERIVED TRAINING WORKED ***")
            results["conclusion"] = "success"
        else:
            logger.info("\n*** GEOMETRY-DERIVED TRAINING DID NOT IMPROVE ***")
            results["conclusion"] = "failed"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    trainer = GeometryDerivedTrainer(model, tokenizer)
    results = trainer.run_experiment()

    output_path = "data/experiments/geometry_derived_training.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
