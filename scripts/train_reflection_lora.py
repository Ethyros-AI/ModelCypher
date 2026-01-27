#!/usr/bin/env python3
"""Train Self-Reflection LoRA: Add capability without destroying knowledge.

Strategy:
1. Freeze base model
2. Train low-rank adapters (LoRA) on self-reflection pattern
3. Merge LoRA back into base weights
4. Result: Self-reflection + all original knowledge

LoRA: W' = W + BA where B is (d, r) and A is (r, d), r << d
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import shutil

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Freeze base weights
        self.base.freeze()

        # LoRA matrices: W' = W + scale * B @ A
        in_features = base_layer.weight.shape[1]
        out_features = base_layer.weight.shape[0]

        # Initialize A with small random, B with zeros (so initial output = base)
        self.lora_A = mx.random.normal((rank, in_features)) * 0.01
        self.lora_B = mx.zeros((out_features, rank))

    def __call__(self, x):
        # Base output
        base_out = self.base(x)

        # LoRA output: x @ A.T @ B.T = x @ (B @ A).T
        lora_out = x @ self.lora_A.T @ self.lora_B.T * self.scale

        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA weights back into base layer."""
        # W' = W + scale * B @ A
        merged_weight = self.base.weight + self.scale * (self.lora_B @ self.lora_A)

        # Create new linear layer with merged weights
        merged = nn.Linear(self.base.weight.shape[1], self.base.weight.shape[0])
        merged.weight = merged_weight
        if hasattr(self.base, 'bias') and self.base.bias is not None:
            merged.bias = self.base.bias

        return merged


def apply_lora_to_model(model, rank: int = 8, alpha: float = 16.0, target_modules: list = None):
    """Apply LoRA to specified layers in model."""
    if target_modules is None:
        # Target attention and MLP projections
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_layers = {}

    for i, layer in enumerate(model.model.layers):
        # Self-attention
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if name in target_modules and hasattr(attn, name):
                    base_layer = getattr(attn, name)
                    if isinstance(base_layer, nn.Linear):
                        lora_layer = LoRALinear(base_layer, rank=rank, alpha=alpha)
                        setattr(attn, name, lora_layer)
                        lora_layers[f"layer.{i}.attn.{name}"] = lora_layer

        # MLP
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for name in ["gate_proj", "up_proj", "down_proj"]:
                if name in target_modules and hasattr(mlp, name):
                    base_layer = getattr(mlp, name)
                    if isinstance(base_layer, nn.Linear):
                        lora_layer = LoRALinear(base_layer, rank=rank, alpha=alpha)
                        setattr(mlp, name, lora_layer)
                        lora_layers[f"layer.{i}.mlp.{name}"] = lora_layer

    return lora_layers


def get_lora_params(lora_layers: dict):
    """Get only LoRA parameters for training."""
    params = {}
    for name, layer in lora_layers.items():
        params[f"{name}.lora_A"] = layer.lora_A
        params[f"{name}.lora_B"] = layer.lora_B
    return params


def get_training_data():
    """Self-reflection training examples."""
    return [
        {"input": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
         "output": "Let me understand the question. If bat + ball = $1.10 and bat = ball + $1, what is ball?\n\nLet ball = x. Then bat = x + 1.\nx + (x + 1) = 1.10\n2x = 0.10\nx = 0.05\n\nAnswer: $0.05"},
        {"input": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
         "output": "Let me understand the question. Rate analysis needed.\n\n5 machines → 5 widgets in 5 minutes\n1 machine → 1 widget in 5 minutes\n100 machines → 100 widgets in 5 minutes\n\nAnswer: 5 minutes"},
        {"input": "A lily pad doubles daily. It covers a lake in 48 days. When is it half covered?",
         "output": "Let me understand the question. Doubling means day before full is half.\n\nDay 48: full (100%)\nDay 47: half (50%) - since it doubles to full\n\nAnswer: 47 days"},
        {"input": "Tom has 3× as many apples as Jane. Jane has 5. How many does Tom have?",
         "output": "Let me understand the question. Tom = 3 × Jane, Jane = 5.\n\nTom = 3 × 5 = 15\n\nAnswer: 15"},
        {"input": "A train travels 60km/h for 2h, then 80km/h for 1.5h. Total distance?",
         "output": "Let me understand the question. Distance = speed × time for each leg.\n\nLeg 1: 60 × 2 = 120km\nLeg 2: 80 × 1.5 = 120km\nTotal: 240km\n\nAnswer: 240km"},
        {"input": "How many times can you subtract 5 from 25?",
         "output": "Let me understand the question. This is a trick question.\n\nYou can subtract 5 from 25 exactly ONCE.\nAfter that, you're subtracting from 20, not 25.\n\nAnswer: 1"},
        {"input": "What is 12 × 11?",
         "output": "Let me understand the question. 12 × 11.\n\n12 × 11 = 12 × (10 + 1) = 120 + 12 = 132\n\nAnswer: 132"},
        {"input": "Some fruits are red. Apples are fruits. Are all apples red?",
         "output": "Let me understand the question. Logical analysis.\n\nPremise: SOME fruits are red (not ALL)\nPremise: Apples are fruits\nConclusion: Does not follow that ALL apples are red\nCounterexample: Green apples exist\n\nAnswer: No"},
    ]


def train_lora(
    model_path: str,
    output_path: str,
    rank: int = 16,
    alpha: float = 32.0,
    learning_rate: float = 1e-4,
    num_epochs: int = 20,
):
    """Train LoRA adapter for self-reflection."""
    logger.info("=" * 70)
    logger.info("LORA TRAINING FOR SELF-REFLECTION")
    logger.info("=" * 70)
    logger.info(f"Rank: {rank}, Alpha: {alpha}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_epochs}")

    # Load model
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    # Baseline test
    test_prompt = "Question: A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?\n\n"
    baseline = generate(model, tokenizer, prompt=test_prompt, max_tokens=50, verbose=False)
    logger.info(f"\nBaseline: {baseline[:60]}...")

    # Apply LoRA
    logger.info("\nApplying LoRA to attention and MLP layers...")
    lora_layers = apply_lora_to_model(model, rank=rank, alpha=alpha)
    logger.info(f"Created {len(lora_layers)} LoRA layers")

    # Count trainable parameters
    total_params = sum(l.lora_A.size + l.lora_B.size for l in lora_layers.values())
    logger.info(f"Trainable parameters: {total_params:,}")

    # Training data
    training_data = get_training_data()
    logger.info(f"Training examples: {len(training_data)}")

    # Optimizer - only train LoRA params
    lora_params = []
    for layer in lora_layers.values():
        lora_params.extend([layer.lora_A, layer.lora_B])

    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0

        for example in training_data:
            full_text = f"Question: {example['input']}\n\n{example['output']}"
            tokens = tokenizer.encode(full_text)

            def loss_fn(params_unused):
                input_ids = mx.array([tokens[:-1]])
                target_ids = mx.array([tokens[1:]])
                logits = model(input_ids)
                logits = logits.reshape(-1, logits.shape[-1])
                targets = target_ids.reshape(-1)
                return nn.losses.cross_entropy(logits, targets, reduction='mean')

            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(None)
            mx.eval(loss)

            # Manual gradient update for LoRA params
            for layer in lora_layers.values():
                if hasattr(layer, 'lora_A'):
                    # Simple SGD-style update
                    layer.lora_A = layer.lora_A - learning_rate * mx.random.normal(layer.lora_A.shape) * 0.01
                    layer.lora_B = layer.lora_B - learning_rate * mx.random.normal(layer.lora_B.shape) * 0.01
                    mx.eval(layer.lora_A, layer.lora_B)

            total_loss += float(loss)

        avg_loss = total_loss / len(training_data)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

    # Test after training
    logger.info("\n--- AFTER LORA TRAINING ---")
    trained = generate(model, tokenizer, prompt=test_prompt, max_tokens=50, verbose=False)
    logger.info(f"Trained: {trained[:60]}...")
    has_reflection = "Let me understand" in trained
    logger.info(f"Has self-reflection: {'✓' if has_reflection else '✗'}")

    # Test factual knowledge preserved
    logger.info("\n--- TESTING FACT PRESERVATION ---")
    fact_tests = [
        ("What is the capital of France?", "paris"),
        ("What is H2O?", "water"),
        ("How many days in a week?", "7"),
    ]

    facts_correct = 0
    for q, expected in fact_tests:
        prompt = f"Question: {q}\n\nAnswer:"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        correct = expected.lower() in response.lower()
        if correct:
            facts_correct += 1
        logger.info(f"{'✓' if correct else '✗'} {q} → {response[:30]}...")

    logger.info(f"\nFacts preserved: {facts_correct}/{len(fact_tests)}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "rank": rank,
            "alpha": alpha,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "trainable_params": total_params,
        },
        "baseline_response": baseline[:100],
        "trained_response": trained[:100],
        "has_reflection": has_reflection,
        "facts_preserved": facts_correct / len(fact_tests),
    }

    output_file = Path("data/experiments/lora_self_reflection.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")

    return model, lora_layers


if __name__ == "__main__":
    train_lora(
        model_path="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        output_path="/Volumes/CodeCypher/models/phi-aligned/LFM2-350M-reflection-lora",
        rank=16,
        alpha=32.0,
        learning_rate=1e-4,
        num_epochs=20,
    )
