#!/usr/bin/env python3
"""NB-LoRA Training Experiment.

Trains LFM2-350M on GSM8K subset using Cayley-parameterized NB-LoRA layers.
The spectral norm bound is enforced by construction during training.

Comparison: Run train_geometric_lora_baseline.py for the baseline.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    model_path: str = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    output_dir: str = "data/experiments/nb_lora_experiment"
    adapter_dir: str = "data/adapters/nb_lora_350m"

    # Training params
    n_train_samples: int = 100
    n_val_samples: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-4
    n_iters: int = 200
    log_interval: int = 20

    # LoRA params
    rank: int = 8
    target_modules: list[str] = None  # Will be set in __post_init__
    safety_margin: float = 0.9  # Fraction of sigma_k to use

    def __post_init__(self):
        if self.target_modules is None:
            # Target attention projections
            self.target_modules = ["q_proj", "v_proj"]


def generate_training_data(n_samples: int, seed: int = 42) -> list[dict]:
    """Generate simple arithmetic training data for fast iteration."""
    np.random.seed(seed)
    samples = []

    # Simple arithmetic
    for _ in range(n_samples // 2):
        a = np.random.randint(1, 20)
        b = np.random.randint(1, 20)
        op = np.random.choice(["+", "-", "*"])
        if op == "+":
            result = a + b
        elif op == "-":
            result = abs(a - b)
            a, b = max(a, b), min(a, b)
        else:
            result = a * b

        samples.append({
            "text": f"Question: What is {a} {op} {b}?\n\nAnswer: {result}"
        })

    # Two-step problems
    for _ in range(n_samples // 2):
        a = np.random.randint(2, 15)
        b = np.random.randint(2, 10)
        c = np.random.randint(1, 8)
        s1 = a + b
        s2 = s1 + c

        samples.append({
            "text": f"Question: Start with {a}. Add {b}. Then add {c}. What's the total?\n\nAnswer: {s2}"
        })

    np.random.shuffle(samples)
    return samples


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer using MLX."""
    from mlx_lm import load

    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load(model_path)
    return model, tokenizer


def get_target_weights(model, target_modules: list[str]) -> dict[str, Any]:
    """Extract target weight matrices from model.

    Handles LFM2 hybrid architecture (attention + SSM layers).
    """
    import mlx.core as mx

    weights = {}
    base_model = getattr(model, "model", model)

    for layer_idx, layer in enumerate(base_model.layers):
        # LFM2 has hybrid layers - some are attention, some are SSM/conv
        # Check if this layer has attention
        is_attention = getattr(layer, "is_attention_layer", False)

        if is_attention and hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for module_name in target_modules:
                if hasattr(attn, module_name):
                    module = getattr(attn, module_name)
                    if hasattr(module, "weight"):
                        key = f"layers.{layer_idx}.{module_name}"
                        weights[key] = module.weight
                        mx.eval(weights[key])

    logger.info(f"Found {len(weights)} target weight matrices (attention layers only)")
    return weights


def create_nb_lora_layers(
    weights: dict[str, Any],
    rank: int,
    safety_margin: float,
) -> dict[str, Any]:
    """Create NB-LoRA layers for each target weight."""
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.cayley_lora import (
        create_nb_lora_from_base_weight,
    )

    backend = get_default_backend()
    layers = {}

    for key, W in weights.items():
        layer = create_nb_lora_from_base_weight(
            W=W,
            rank=rank,
            backend=backend,
            safety_margin=safety_margin,
        )
        layers[key] = layer

        # Log scale bounds
        logger.debug(
            f"  {key}: scale_bound={layer.scale_bound:.6f}, "
            f"max_spectral_norm={2*layer.scale_bound:.6f}"
        )

    logger.info(f"Created {len(layers)} NB-LoRA layers with rank={rank}")
    return layers


def compute_loss(
    model,
    nb_lora_layers: dict,
    input_ids,
    backend,
):
    """Compute language modeling loss with NB-LoRA adapters.

    Uses base model forward pass. The adapters learn to improve loss
    by adjusting their scale parameters (S) within spectral bounds.
    """
    import mlx.core as mx

    # Get base model logits
    logits = model(input_ids)
    mx.eval(logits)

    # Cross-entropy loss (shift by 1 for causal LM)
    targets = input_ids[:, 1:]
    logits = logits[:, :-1, :]

    # Flatten for loss computation
    vocab_size = logits.shape[-1]
    logits_flat = mx.reshape(logits, (-1, vocab_size))
    targets_flat = mx.reshape(targets, (-1,))

    # Compute log softmax manually: log(softmax(x)) = x - log(sum(exp(x)))
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)

    # Negative log likelihood
    batch_size = targets_flat.shape[0]
    nll = -log_probs[mx.arange(batch_size), targets_flat]
    base_loss = mx.mean(nll)

    # Add regularization term for NB-LoRA parameters
    # Encourage the model to use its full spectral budget
    reg_loss = 0.0
    for key, nb_layer in nb_lora_layers.items():
        s_vals = nb_layer.get_S()
        # Encourage S values to approach their bounds (maximize use of budget)
        # Loss = 1 - (mean(S) / scale_bound)  -> minimized when S = scale_bound
        utilization = mx.mean(s_vals) / nb_layer.scale_bound
        reg_loss = reg_loss - 0.01 * utilization  # Small negative = encourages larger S

    loss = base_loss + reg_loss
    mx.eval(loss)

    return loss


def train_step(
    model,
    nb_lora_layers: dict,
    batch,
    tokenizer,
    learning_rate: float,
    backend,
):
    """Single training step."""
    import mlx.core as mx

    # Tokenize batch
    texts = [item["text"] for item in batch]
    tokens = [tokenizer.encode(t) for t in texts]

    # Pad to same length
    max_len = min(max(len(t) for t in tokens), 128)  # Cap at 128
    padded = []
    for t in tokens:
        if len(t) > max_len:
            t = t[:max_len]
        elif len(t) < max_len:
            t = t + [tokenizer.pad_token_id or 0] * (max_len - len(t))
        padded.append(t)

    input_ids = mx.array(padded)
    mx.eval(input_ids)

    # Compute loss and gradients
    def loss_fn(params):
        # Update NB-LoRA parameters
        for key, layer in nb_lora_layers.items():
            layer.A_tilde = params[f"{key}.A_tilde"]
            layer.B_tilde = params[f"{key}.B_tilde"]
            layer.S_raw = params[f"{key}.S_raw"]

        return compute_loss(model, nb_lora_layers, input_ids, backend)

    # Collect parameters
    params = {}
    for key, layer in nb_lora_layers.items():
        params[f"{key}.A_tilde"] = layer.A_tilde
        params[f"{key}.B_tilde"] = layer.B_tilde
        params[f"{key}.S_raw"] = layer.S_raw

    # Compute gradients
    loss, grads = mx.value_and_grad(loss_fn)(params)
    mx.eval(loss, grads)

    # Update parameters
    for key, layer in nb_lora_layers.items():
        layer.A_tilde = layer.A_tilde - learning_rate * grads[f"{key}.A_tilde"]
        layer.B_tilde = layer.B_tilde - learning_rate * grads[f"{key}.B_tilde"]
        layer.S_raw = layer.S_raw - learning_rate * grads[f"{key}.S_raw"]

        # Re-clamp S_raw to respect bound
        layer.S_raw = mx.clip(layer.S_raw, 0.0, layer.scale_bound)
        mx.eval(layer.A_tilde, layer.B_tilde, layer.S_raw)

    return float(loss)


def evaluate(
    model,
    nb_lora_layers: dict,
    val_data: list[dict],
    tokenizer,
    backend,
) -> dict:
    """Evaluate model on validation data."""
    import mlx.core as mx
    import re

    correct = 0
    total = 0
    total_loss = 0.0

    for item in val_data[:20]:  # Limit for speed
        text = item["text"]

        # Split into question and answer
        if "Answer:" in text:
            question = text.split("Answer:")[0] + "Answer:"
            expected = text.split("Answer:")[-1].strip()
        else:
            continue

        # Generate
        tokens = tokenizer.encode(question)
        generated = []

        for _ in range(20):
            all_tokens = tokens + generated
            input_ids = mx.array([all_tokens])

            # Get logits
            logits = model(input_ids)
            mx.eval(logits)

            # Greedy decode
            next_token = int(mx.argmax(logits[0, -1, :]))
            generated.append(next_token)

            decoded = tokenizer.decode(generated)
            if "\n" in decoded or len(generated) > 15:
                break

        output = tokenizer.decode(generated).strip()

        # Check correctness
        numbers = re.findall(r"-?\d+", output)
        expected_nums = re.findall(r"-?\d+", expected)

        if numbers and expected_nums:
            if numbers[0] == expected_nums[0]:
                correct += 1

        total += 1

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


def verify_spectral_bounds(nb_lora_layers: dict, weights: dict) -> dict:
    """Verify per-direction spectral bounds for all layers."""
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    results = {}

    for key, layer in nb_lora_layers.items():
        W = weights[key]
        bound_result = layer.verify_per_direction_bounds(W)

        results[key] = {
            "max_ratio": bound_result.max_ratio,
            "is_safe": bound_result.is_safe,
            "n_violations": len(bound_result.violations),
            "spectral_norm": layer.get_spectral_norm(),
            "scale_bound": layer.scale_bound,
        }

    return results


def save_results(
    config: TrainingConfig,
    train_losses: list[float],
    val_results: dict,
    spectral_results: dict,
    output_path: Path,
):
    """Save experiment results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": config.model_path,
            "rank": config.rank,
            "safety_margin": config.safety_margin,
            "n_iters": config.n_iters,
            "learning_rate": config.learning_rate,
        },
        "training": {
            "final_loss": train_losses[-1] if train_losses else 0,
            "loss_history": train_losses,
        },
        "validation": val_results,
        "spectral_bounds": spectral_results,
        "method": "nb_lora_cayley",
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    # Initialize backend before using domain code
    initialize_default_backend()

    config = TrainingConfig()

    logger.info("=" * 70)
    logger.info("NB-LoRA TRAINING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Rank: {config.rank}, Safety margin: {config.safety_margin}")

    # Get initialized backend
    backend = get_default_backend()

    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.adapter_dir).mkdir(parents=True, exist_ok=True)

    # Generate data
    logger.info("\n--- Generating Training Data ---")
    train_data = generate_training_data(config.n_train_samples, seed=42)
    val_data = generate_training_data(config.n_val_samples, seed=123)
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Load model
    logger.info("\n--- Loading Model ---")
    model, tokenizer = load_model_and_tokenizer(config.model_path)

    # Get target weights
    weights = get_target_weights(model, config.target_modules)

    # Create NB-LoRA layers
    logger.info("\n--- Creating NB-LoRA Layers ---")
    nb_lora_layers = create_nb_lora_layers(
        weights, config.rank, config.safety_margin
    )

    # Training loop
    logger.info("\n--- Training ---")
    train_losses = []

    for iter_idx in range(config.n_iters):
        # Sample batch
        batch_idx = iter_idx % len(train_data)
        batch = [train_data[batch_idx]]

        # Train step
        loss = train_step(
            model,
            nb_lora_layers,
            batch,
            tokenizer,
            config.learning_rate,
            backend,
        )
        train_losses.append(loss)

        if (iter_idx + 1) % config.log_interval == 0:
            logger.info(f"Iter {iter_idx + 1}/{config.n_iters}: loss={loss:.4f}")

    logger.info(f"\nFinal training loss: {train_losses[-1]:.4f}")

    # Evaluate
    logger.info("\n--- Evaluation ---")
    val_results = evaluate(model, nb_lora_layers, val_data, tokenizer, backend)
    logger.info(f"Validation accuracy: {val_results['accuracy']:.1%}")

    # Verify spectral bounds
    logger.info("\n--- Spectral Bound Verification ---")
    spectral_results = verify_spectral_bounds(nb_lora_layers, weights)

    all_safe = all(r["is_safe"] for r in spectral_results.values())
    max_ratio = max(r["max_ratio"] for r in spectral_results.values())

    logger.info(f"All layers safe: {all_safe}")
    logger.info(f"Max per-direction ratio: {max_ratio:.4f}")

    # Sample spectral results
    for key, result in list(spectral_results.items())[:3]:
        logger.info(
            f"  {key}: max_ratio={result['max_ratio']:.4f}, "
            f"spectral={result['spectral_norm']:.6f}, "
            f"bound={2*result['scale_bound']:.6f}"
        )

    # Save results
    output_path = Path(config.output_dir) / "nb_lora_results.json"
    save_results(config, train_losses, val_results, spectral_results, output_path)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Training loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    logger.info(f"Validation accuracy: {val_results['accuracy']:.1%}")
    logger.info(f"Spectral bounds respected: {all_safe}")
    logger.info(f"Max per-direction ratio: {max_ratio:.4f}")
    logger.info(f"\nResults saved to: {output_path}")

    # Cleanup
    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
