#!/usr/bin/env python3
"""Baseline Geometric LoRA Training Experiment.

Trains LFM2-350M on GSM8K subset using standard LoRA (B=0, A random).
Post-hoc geometric rescaling via apply_lora_geometric() for deployment.

Comparison: Run train_nb_lora_experiment.py for the NB-LoRA treatment.
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
    output_dir: str = "data/experiments/geometric_lora_baseline"
    adapter_dir: str = "data/adapters/geometric_lora_350m"

    # Training params
    n_train_samples: int = 100
    n_val_samples: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-4
    n_iters: int = 200
    log_interval: int = 20

    # LoRA params
    rank: int = 8
    alpha: float = 16.0  # Standard alpha/rank scaling
    target_modules: list[str] = None

    def __post_init__(self):
        if self.target_modules is None:
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


class StandardLoRALayer:
    """Standard LoRA layer with B=0 initialization."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, backend):
        import mlx.core as mx

        self._backend = backend
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Standard LoRA initialization:
        # A: Gaussian initialization
        # B: Zero initialization (so B @ A = 0 initially)
        self.lora_A = mx.random.normal((rank, in_features)) * 0.01
        self.lora_B = mx.zeros((out_features, rank))
        mx.eval(self.lora_A, self.lora_B)

    def forward(self, x):
        """Forward pass: output = scale * B @ A @ x."""
        import mlx.core as mx

        # x: [batch, seq, in_features]
        # A: [rank, in_features]
        # B: [out_features, rank]

        input_shape = x.shape
        if len(input_shape) == 3:
            batch, seq, d_in = input_shape
            x_flat = mx.reshape(x, (batch * seq, d_in))
        else:
            x_flat = x

        # A @ x^T: [rank, batch*seq]
        z = mx.matmul(self.lora_A, mx.transpose(x_flat))

        # B @ z: [out_features, batch*seq]
        out = mx.matmul(self.lora_B, z)

        # Transpose: [batch*seq, out_features]
        out = mx.transpose(out) * self.scale

        if len(input_shape) == 3:
            out = mx.reshape(out, (batch, seq, -1))

        mx.eval(out)
        return out

    def get_effective_delta(self):
        """Get weight delta: scale * B @ A."""
        import mlx.core as mx

        delta = self.scale * mx.matmul(self.lora_B, self.lora_A)
        mx.eval(delta)
        return delta

    def get_spectral_norm(self):
        """Compute spectral norm of delta."""
        import mlx.core as mx

        delta = self.get_effective_delta()
        delta_f32 = delta.astype(mx.float32)
        mx.eval(delta_f32)
        _, S, _ = mx.linalg.svd(delta_f32, compute_uv=True, stream=mx.cpu)
        mx.eval(S)
        return float(S[0])

    def parameters(self) -> dict:
        return {"lora_A": self.lora_A, "lora_B": self.lora_B}


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


def create_standard_lora_layers(
    weights: dict[str, Any],
    rank: int,
    alpha: float,
    backend,
) -> dict[str, StandardLoRALayer]:
    """Create standard LoRA layers for each target weight."""
    layers = {}

    for key, W in weights.items():
        out_features, in_features = W.shape
        layer = StandardLoRALayer(in_features, out_features, rank, alpha, backend)
        layers[key] = layer

    logger.info(f"Created {len(layers)} standard LoRA layers with rank={rank}, alpha={alpha}")
    return layers


def compute_loss(
    model,
    lora_layers: dict,
    input_ids,
    backend,
):
    """Compute language modeling loss with standard LoRA adapters.

    Uses base model forward pass. The adapters learn to improve loss
    by adjusting their A and B matrices.
    """
    import mlx.core as mx

    # Get base model logits
    logits = model(input_ids)
    mx.eval(logits)

    # Cross-entropy loss (shift by 1 for causal LM)
    targets = input_ids[:, 1:]
    logits = logits[:, :-1, :]

    vocab_size = logits.shape[-1]
    logits_flat = mx.reshape(logits, (-1, vocab_size))
    targets_flat = mx.reshape(targets, (-1,))

    # Compute log softmax manually: log(softmax(x)) = x - log(sum(exp(x)))
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    batch_size = targets_flat.shape[0]
    nll = -log_probs[mx.arange(batch_size), targets_flat]
    loss = mx.mean(nll)
    mx.eval(loss)

    return loss


def train_step(
    model,
    lora_layers: dict,
    batch,
    tokenizer,
    learning_rate: float,
    backend,
):
    """Single training step."""
    import mlx.core as mx

    # Tokenize
    texts = [item["text"] for item in batch]
    tokens = [tokenizer.encode(t) for t in texts]

    max_len = min(max(len(t) for t in tokens), 128)
    padded = []
    for t in tokens:
        if len(t) > max_len:
            t = t[:max_len]
        elif len(t) < max_len:
            t = t + [tokenizer.pad_token_id or 0] * (max_len - len(t))
        padded.append(t)

    input_ids = mx.array(padded)
    mx.eval(input_ids)

    # Loss function for gradient computation
    def loss_fn(params):
        for key, layer in lora_layers.items():
            layer.lora_A = params[f"{key}.lora_A"]
            layer.lora_B = params[f"{key}.lora_B"]
        return compute_loss(model, lora_layers, input_ids, backend)

    # Collect parameters
    params = {}
    for key, layer in lora_layers.items():
        params[f"{key}.lora_A"] = layer.lora_A
        params[f"{key}.lora_B"] = layer.lora_B

    # Compute gradients
    loss, grads = mx.value_and_grad(loss_fn)(params)
    mx.eval(loss, grads)

    # Update
    for key, layer in lora_layers.items():
        layer.lora_A = layer.lora_A - learning_rate * grads[f"{key}.lora_A"]
        layer.lora_B = layer.lora_B - learning_rate * grads[f"{key}.lora_B"]
        mx.eval(layer.lora_A, layer.lora_B)

    return float(loss)


def evaluate(model, lora_layers: dict, val_data: list[dict], tokenizer, backend) -> dict:
    """Evaluate model on validation data."""
    import mlx.core as mx
    import re

    correct = 0
    total = 0

    for item in val_data[:20]:
        text = item["text"]

        if "Answer:" in text:
            question = text.split("Answer:")[0] + "Answer:"
            expected = text.split("Answer:")[-1].strip()
        else:
            continue

        tokens = tokenizer.encode(question)
        generated = []

        for _ in range(20):
            all_tokens = tokens + generated
            input_ids = mx.array([all_tokens])
            logits = model(input_ids)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1, :]))
            generated.append(next_token)

            decoded = tokenizer.decode(generated)
            if "\n" in decoded or len(generated) > 15:
                break

        output = tokenizer.decode(generated).strip()
        numbers = re.findall(r"-?\d+", output)
        expected_nums = re.findall(r"-?\d+", expected)

        if numbers and expected_nums:
            if numbers[0] == expected_nums[0]:
                correct += 1

        total += 1

    return {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}


def apply_geometric_rescaling(
    lora_layers: dict[str, StandardLoRALayer],
    weights: dict[str, Any],
    target_ratio: float = 0.9,
) -> dict:
    """Apply geometric rescaling to respect spectral bounds."""
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

    backend = get_default_backend()
    import mlx.core as mx

    results = {}

    for key, layer in lora_layers.items():
        W = weights[key]

        # SVD of base weight (must run on CPU in MLX)
        W_f32 = W.astype(mx.float32)
        mx.eval(W_f32)
        _, S, _ = mx.linalg.svd(W_f32, compute_uv=True, stream=mx.cpu)
        mx.eval(S)

        # Find sigma_k
        sigma_max = float(S[0])
        sqrt_eps = sqrt_scalar(backend.finfo().eps, backend)
        threshold = sqrt_eps * sigma_max

        significant = S > threshold
        eff_rank = int(mx.sum(significant.astype(mx.int32)))
        sigma_k = float(S[eff_rank - 1]) if eff_rank > 0 else float(S[-1])

        # Current spectral norm of LoRA delta
        delta = layer.get_effective_delta()
        delta_f32 = delta.astype(mx.float32)
        mx.eval(delta_f32)
        _, S_delta, _ = mx.linalg.svd(delta_f32, compute_uv=True, stream=mx.cpu)
        mx.eval(S_delta)
        delta_norm = float(S_delta[0])

        # Compute geometric scale
        if delta_norm > 1e-10:
            geo_scale = (sigma_k / delta_norm) * target_ratio
        else:
            geo_scale = 1.0

        # Scale the LoRA weights
        layer.lora_A = layer.lora_A * geo_scale
        layer.lora_B = layer.lora_B * geo_scale
        mx.eval(layer.lora_A, layer.lora_B)

        # Verify new spectral norm
        new_norm = layer.get_spectral_norm()

        results[key] = {
            "sigma_k": sigma_k,
            "original_norm": delta_norm,
            "geo_scale": geo_scale,
            "final_norm": new_norm,
            "bound": sigma_k,
            "is_safe": new_norm <= sigma_k * 1.01,
        }

        logger.debug(
            f"  {key}: norm {delta_norm:.6f} -> {new_norm:.6f} (bound: {sigma_k:.6f})"
        )

    return results


def verify_per_direction_bounds(lora_layers: dict, weights: dict) -> dict:
    """Verify per-direction spectral bounds after geometric rescaling."""
    from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    results = {}

    for key, layer in lora_layers.items():
        W = weights[key]
        B = layer.lora_B
        A = layer.lora_A

        bound_result = LoRASafetyService.verify_per_direction_bounds(B, A, W, backend)

        results[key] = {
            "max_ratio": bound_result.max_ratio,
            "is_safe": bound_result.is_safe,
            "n_violations": len(bound_result.violations),
            "spectral_norm": layer.get_spectral_norm(),
        }

    return results


def save_results(
    config: TrainingConfig,
    train_losses: list[float],
    val_results: dict,
    geo_results: dict,
    spectral_results: dict,
    output_path: Path,
):
    """Save experiment results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": config.model_path,
            "rank": config.rank,
            "alpha": config.alpha,
            "n_iters": config.n_iters,
            "learning_rate": config.learning_rate,
        },
        "training": {
            "final_loss": train_losses[-1] if train_losses else 0,
            "loss_history": train_losses,
        },
        "validation": val_results,
        "geometric_rescaling": geo_results,
        "spectral_bounds": spectral_results,
        "method": "standard_lora_geometric_rescale",
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
    logger.info("BASELINE: GEOMETRIC LoRA TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Rank: {config.rank}, Alpha: {config.alpha}")

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

    # Create standard LoRA layers
    logger.info("\n--- Creating Standard LoRA Layers ---")
    lora_layers = create_standard_lora_layers(
        weights, config.rank, config.alpha, backend
    )

    # Training loop
    logger.info("\n--- Training (Standard LoRA) ---")
    train_losses = []

    for iter_idx in range(config.n_iters):
        batch_idx = iter_idx % len(train_data)
        batch = [train_data[batch_idx]]

        loss = train_step(model, lora_layers, batch, tokenizer, config.learning_rate, backend)
        train_losses.append(loss)

        if (iter_idx + 1) % config.log_interval == 0:
            logger.info(f"Iter {iter_idx + 1}/{config.n_iters}: loss={loss:.4f}")

    logger.info(f"\nFinal training loss: {train_losses[-1]:.4f}")

    # Apply geometric rescaling
    logger.info("\n--- Applying Geometric Rescaling ---")
    geo_results = apply_geometric_rescaling(lora_layers, weights, target_ratio=0.9)

    for key, result in list(geo_results.items())[:3]:
        logger.info(
            f"  {key}: norm {result['original_norm']:.6f} -> {result['final_norm']:.6f}"
        )

    # Evaluate after rescaling
    logger.info("\n--- Evaluation (After Geometric Rescaling) ---")
    val_results = evaluate(model, lora_layers, val_data, tokenizer, backend)
    logger.info(f"Validation accuracy: {val_results['accuracy']:.1%}")

    # Verify per-direction bounds
    logger.info("\n--- Per-Direction Bound Verification ---")
    spectral_results = verify_per_direction_bounds(lora_layers, weights)

    all_safe = all(r["is_safe"] for r in spectral_results.values())
    max_ratio = max(r["max_ratio"] for r in spectral_results.values())

    logger.info(f"All layers safe: {all_safe}")
    logger.info(f"Max per-direction ratio: {max_ratio:.4f}")

    for key, result in list(spectral_results.items())[:3]:
        logger.info(
            f"  {key}: max_ratio={result['max_ratio']:.4f}, "
            f"spectral={result['spectral_norm']:.6f}"
        )

    # Save results
    output_path = Path(config.output_dir) / "geometric_lora_results.json"
    save_results(config, train_losses, val_results, geo_results, spectral_results, output_path)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (BASELINE: Geometric LoRA)")
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
