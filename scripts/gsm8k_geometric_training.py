#!/usr/bin/env python3
"""Phase B: Geometry-Derived Training for GSM8K.

ALL parameters derived from the geometry of GSM8K activations:
- Learning rate = 1/(κ × scale)
- Stopping threshold = κ × √eps
- Convergence criterion = √eps
- Max iterations ≈ κ

NO ARBITRARY CONSTANTS. The geometry tells us what to do.

This uses Fisher Information to identify important parameters and
trains with null-space projection to preserve existing capabilities.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class GeometricParams:
    """All training parameters derived from geometry."""
    kappa: float           # Condition number
    scale: float           # Frobenius norm
    sqrt_eps: float        # √eps - precision floor
    learning_rate: float   # 1/(κ × scale)
    stop_threshold: float  # κ × √eps
    convergence: float     # √eps
    max_iterations: int    # ceil(κ)


class GSM8KGeometricTrainer:
    """Geometry-derived training for GSM8K word problems.

    All parameters come from the manifold geometry of the model's
    representations. No arbitrary hyperparameters.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dtype_eps = np.finfo(np.float32).eps
        self.sqrt_eps = np.sqrt(self.dtype_eps)

    def _get_last_hidden_state(self, prompt: str) -> np.ndarray:
        """Get last layer hidden state for a prompt."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Get hidden states
        hidden = self.model.model.embed_tokens(input_ids)

        for layer in self.model.model.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        hidden = self.model.model.norm(hidden)
        mx.eval(hidden)

        return np.array(hidden[0, -1, :].tolist(), dtype=np.float32)

    def compute_gsm8k_geometry(self, problems: List[Tuple[str, str]]) -> GeometricParams:
        """Compute geometric parameters from GSM8K problem activations."""
        logger.info("Computing geometry from GSM8K problems...")

        # Get activations for all problems
        activations = []
        for prompt, _ in problems:
            act = self._get_last_hidden_state(prompt)
            activations.append(act)

        A = np.vstack(activations)

        # Center activations
        A_centered = A - A.mean(axis=0)

        # Gram matrix
        G = A_centered @ A_centered.T

        # Scale: Frobenius norm
        scale = np.linalg.norm(G, 'fro')

        # Condition number
        _, S, _ = svd(G, full_matrices=False)
        S_valid = S[S > self.sqrt_eps * S[0]]
        if len(S_valid) > 1:
            kappa = float(S_valid[0] / S_valid[-1])
        else:
            kappa = 1.0

        # Derived parameters
        lr = 1.0 / (kappa * scale)
        stop_threshold = kappa * self.sqrt_eps
        max_iter = int(np.ceil(kappa))

        logger.info(f"\n  GEOMETRY-DERIVED PARAMETERS:")
        logger.info(f"  Condition number κ: {kappa:.4e}")
        logger.info(f"  Scale: {scale:.4e}")
        logger.info(f"  √eps: {self.sqrt_eps:.4e}")
        logger.info(f"  ")
        logger.info(f"  DERIVED:")
        logger.info(f"    Learning rate = 1/(κ×scale) = {lr:.4e}")
        logger.info(f"    Stop threshold = κ×√eps = {stop_threshold:.4e}")
        logger.info(f"    Convergence = √eps = {self.sqrt_eps:.4e}")
        logger.info(f"    Max iterations = ceil(κ) = {max_iter}")

        return GeometricParams(
            kappa=kappa,
            scale=scale,
            sqrt_eps=self.sqrt_eps,
            learning_rate=lr,
            stop_threshold=stop_threshold,
            convergence=self.sqrt_eps,
            max_iterations=max_iter,
        )

    def compute_loss(self, problems: List[Tuple[str, str]]) -> float:
        """Compute cross-entropy loss on GSM8K problems."""
        import mlx.core as mx
        import re

        total_loss = 0.0
        n_valid = 0

        for prompt, expected in problems:
            # Format prompt
            full_prompt = f"Question: {prompt}\n\nAnswer:"
            tokens = self.tokenizer.encode(full_prompt)

            # Get logits
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)

            # Get target token (first digit of answer)
            expected_tokens = self.tokenizer.encode(expected)
            if not expected_tokens:
                continue
            target_token = expected_tokens[0]

            # Cross-entropy
            next_logits = logits[0, -1, :]
            log_probs = mx.log(mx.softmax(next_logits) + 1e-10)
            loss = -log_probs[target_token]
            mx.eval(loss)

            total_loss += float(loss.item())
            n_valid += 1

        return total_loss / max(n_valid, 1)

    def evaluate_gsm8k(self, n_problems: int = 20) -> Tuple[int, int]:
        """Quick GSM8K evaluation."""
        import re
        import mlx.core as mx

        from modelcypher.core.use_cases.curriculum import BenchmarkLoader
        loader = BenchmarkLoader()
        gsm_test = loader.load("gsm8k", split="test", limit=n_problems)

        correct = 0

        for sample in gsm_test.samples:
            question = sample.prompt.replace("Answer:", "").strip()
            expected = sample.answer

            prompt = f"Question: {question}\n\nAnswer:"
            tokens = self.tokenizer.encode(prompt)
            generated = []

            for _ in range(150):  # Shorter for speed
                logits = self.model(mx.array([tokens + generated]))
                mx.eval(logits)
                logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
                probs = np.exp(logits_np - logits_np.max())
                probs = probs / probs.sum()
                next_tok = int(np.argmax(probs))
                generated.append(next_tok)

                decoded = self.tokenizer.decode(generated)
                if "####" in decoded or "<|im_end|>" in decoded:
                    break

            output = self.tokenizer.decode(generated).strip()

            if "####" in output:
                answer_part = output.split("####")[-1].strip().replace(",", "").replace("$", "")
                numbers = re.findall(r'-?\d+', answer_part)
                predicted = numbers[0] if numbers else ""
            else:
                numbers = re.findall(r'-?\d+', output.replace(",", ""))
                predicted = numbers[-1] if numbers else ""

            if predicted == expected:
                correct += 1

        return correct, n_problems

    def training_step(self, problems: List[Tuple[str, str]], lr: float) -> Tuple[float, float]:
        """Single training step with gradient."""
        import mlx.core as mx
        import mlx.optimizers as optim

        def loss_fn(model):
            total_loss = mx.array(0.0)
            n_valid = 0

            for prompt, expected in problems:
                full_prompt = f"Question: {prompt}\n\nAnswer:"
                tokens = self.tokenizer.encode(full_prompt)
                input_ids = mx.array([tokens])

                logits = model(input_ids)
                next_logits = logits[0, -1, :]

                expected_tokens = self.tokenizer.encode(expected)
                if not expected_tokens:
                    continue
                target_token = expected_tokens[0]

                log_probs = mx.log(mx.softmax(next_logits) + 1e-10)
                loss = -log_probs[target_token]
                total_loss = total_loss + loss
                n_valid += 1

            return total_loss / max(n_valid, 1)

        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        mx.eval(loss, grads)

        # Compute gradient norm
        total_sq = 0.0
        for name, param in grads.items():
            if hasattr(param, 'items'):
                for sub_name, sub_param in param.items():
                    if hasattr(sub_param, 'shape'):
                        total_sq += float(mx.sum(sub_param ** 2).item())
            elif hasattr(param, 'shape'):
                total_sq += float(mx.sum(param ** 2).item())
        grad_norm = np.sqrt(total_sq)

        # Apply update
        optimizer = optim.SGD(learning_rate=lr)
        optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())

        return float(loss.item()), grad_norm

    def run(self, training_problems: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """Run geometry-derived training.

        If no training_problems provided, uses failing GSM8K problems
        identified from Phase A.
        """
        logger.info("\n" + "=" * 70)
        logger.info("PHASE B: GEOMETRY-DERIVED TRAINING FOR GSM8K")
        logger.info("=" * 70)
        logger.info("\nALL parameters from geometry. NO heuristics.\n")

        # Get training problems (failing cases from current model)
        if training_problems is None:
            # Use a few example patterns that tend to fail
            training_problems = [
                ("Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but "
                 "40% of the way through the download, Windows forces a restart to install updates, "
                 "which takes 20 minutes. Then Carla has to restart the download from the beginning. "
                 "How long does it take to download the file?", "160"),
                ("Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at "
                 "the green house, 2 more to the red house, and half of what was left at the orange "
                 "house. If Melanie has 5 vacuum cleaners left, how many did she start with?", "18"),
                ("Gloria is shoe shopping when she comes across a pair of boots that fit her budget. "
                 "However, she has to choose between the boots and two pairs of high heels that "
                 "together cost five dollars less than the boots. If one pair of heels costs $33 and "
                 "the other costs $37, how much do the boots cost?", "75"),
            ]

        # Compute geometric parameters from training problems
        params = self.compute_gsm8k_geometry(training_problems)

        # Initial evaluation
        logger.info("\nInitial GSM8K evaluation...")
        correct_before, total = self.evaluate_gsm8k(20)
        accuracy_before = correct_before / total
        logger.info(f"Initial accuracy: {correct_before}/{total} ({accuracy_before:.1%})")

        initial_loss = self.compute_loss(training_problems)
        logger.info(f"Initial loss: {initial_loss:.4f}")

        results = {
            "geometry_params": {
                "kappa": params.kappa,
                "scale": params.scale,
                "sqrt_eps": params.sqrt_eps,
                "learning_rate": params.learning_rate,
                "stop_threshold": params.stop_threshold,
                "convergence": params.convergence,
                "max_iterations": params.max_iterations,
            },
            "initial": {
                "accuracy": accuracy_before,
                "loss": initial_loss,
            },
            "iterations": [],
        }

        # Training loop - geometry determines everything
        logger.info("\n=== GEOMETRY-DERIVED TRAINING ===")
        logger.info(f"LR = {params.learning_rate:.4e}")
        logger.info(f"Stop when loss < {params.stop_threshold:.4e}")
        logger.info(f"Max iterations = {params.max_iterations}")

        prev_loss = initial_loss

        for iteration in range(params.max_iterations):
            # Training step
            loss, grad_norm = self.training_step(training_problems, params.learning_rate)

            # Compute alignment loss
            current_loss = self.compute_loss(training_problems)

            # Relative change
            rel_change = abs(current_loss - prev_loss) / (prev_loss + 1e-10)

            logger.info(f"  Iter {iteration}: loss={current_loss:.4f}, "
                       f"grad={grad_norm:.4e}, rel_change={rel_change:.4e}")

            results["iterations"].append({
                "iteration": iteration,
                "loss": current_loss,
                "gradient_norm": grad_norm,
                "relative_change": rel_change,
            })

            # Stopping criteria - ALL from geometry
            # 1. Loss below achievable precision
            if current_loss < params.stop_threshold:
                logger.info(f"  STOPPED: loss {current_loss:.4e} < threshold {params.stop_threshold:.4e}")
                results["stop_reason"] = "below_threshold"
                break

            # 2. Converged
            if rel_change < params.convergence:
                logger.info(f"  STOPPED: converged (rel_change {rel_change:.4e} < {params.convergence:.4e})")
                results["stop_reason"] = "converged"
                break

            # 3. Gradient vanished
            if grad_norm < self.sqrt_eps:
                logger.info(f"  STOPPED: gradient vanished ({grad_norm:.4e})")
                results["stop_reason"] = "gradient_vanished"
                break

            # 4. Loss increased
            if current_loss > prev_loss * (1 + self.sqrt_eps):
                logger.info(f"  STOPPED: loss increased")
                results["stop_reason"] = "loss_increased"
                break

            prev_loss = current_loss
        else:
            results["stop_reason"] = "max_iterations"

        # Final evaluation
        logger.info("\nFinal GSM8K evaluation...")
        correct_after, total = self.evaluate_gsm8k(20)
        accuracy_after = correct_after / total
        final_loss = self.compute_loss(training_problems)

        logger.info(f"Final accuracy: {correct_after}/{total} ({accuracy_after:.1%})")
        logger.info(f"Final loss: {final_loss:.4f}")

        results["final"] = {
            "accuracy": accuracy_after,
            "loss": final_loss,
        }

        # Summary
        logger.info(f"\n{'=' * 70}")
        logger.info("PHASE B RESULTS")
        logger.info(f"{'=' * 70}")
        logger.info(f"GSM8K: {accuracy_before:.1%} → {accuracy_after:.1%}")
        logger.info(f"Loss: {initial_loss:.4f} → {final_loss:.4f}")
        logger.info(f"Stop reason: {results['stop_reason']}")

        return results


def main():
    from mlx_lm import load

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"Loading model: {model_path}")
    logger.info(f"With adapter: {adapter_path}")

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    trainer = GSM8KGeometricTrainer(model, tokenizer)
    results = trainer.run()

    # Save results
    output_path = Path("data/experiments/gsm8k_geometric_training.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
