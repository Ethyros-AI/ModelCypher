#!/usr/bin/env python3
"""Train Self-Reflection LoRA v2: Using mlx-lm's built-in LoRA support.

This uses the proper mlx LoRA implementation for correct gradient flow.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate
from mlx_lm.tuner.utils import linear_to_lora_layers

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_training_data():
    """Self-reflection training examples."""
    return [
        {"input": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
         "output": "Let me understand the question. If bat + ball = $1.10 and bat = ball + $1, what is ball?\n\nLet ball = x. Then bat = x + 1.\nx + (x + 1) = 1.10\n2x = 0.10\nx = 0.05\n\nAnswer: $0.05"},
        {"input": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
         "output": "Let me understand the question. Rate analysis.\n\n5 machines → 5 widgets in 5 min\n1 machine → 1 widget in 5 min\n100 machines → 100 widgets in 5 min\n\nAnswer: 5 minutes"},
        {"input": "A lily pad doubles daily. It covers a lake in 48 days. When is it half covered?",
         "output": "Let me understand the question. Doubling backward.\n\nDay 48: full\nDay 47: half (since doubling gives full)\n\nAnswer: 47 days"},
        {"input": "Tom has 3× as many apples as Jane. Jane has 5. How many does Tom have?",
         "output": "Let me understand the question. Tom = 3 × Jane.\n\nTom = 3 × 5 = 15\n\nAnswer: 15"},
        {"input": "A train travels 60km/h for 2h, then 80km/h for 1.5h. Total distance?",
         "output": "Let me understand the question. Sum of distances.\n\n60×2 + 80×1.5 = 120 + 120 = 240\n\nAnswer: 240km"},
        {"input": "How many times can you subtract 5 from 25?",
         "output": "Let me understand the question. Trick: subtract FROM 25.\n\nOnce. After that it's 20, not 25.\n\nAnswer: 1"},
        {"input": "What is 12 × 11?",
         "output": "Let me understand the question. 12 × 11.\n\n12 × 11 = 132\n\nAnswer: 132"},
        {"input": "Some fruits are red. Apples are fruits. Are all apples red?",
         "output": "Let me understand the question. Logic check.\n\nSOME ≠ ALL. Green apples exist.\n\nAnswer: No"},
    ]


def train_lora(
    model_path: str,
    rank: int = 8,
    num_epochs: int = 15,
    learning_rate: float = 1e-4,
):
    """Train LoRA with proper gradient flow."""
    logger.info("=" * 70)
    logger.info("LORA TRAINING FOR SELF-REFLECTION (v2)")
    logger.info("=" * 70)
    logger.info(f"Rank: {rank}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_epochs}")

    # Load model
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    # Baseline test
    test_prompt = "Question: A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?\n\n"
    baseline = generate(model, tokenizer, prompt=test_prompt, max_tokens=50, verbose=False)
    logger.info(f"\nBaseline: {baseline[:70]}...")

    # Apply LoRA using mlx-lm utility
    logger.info("\nApplying LoRA...")
    lora_config = {
        "rank": rank,
        "alpha": rank * 2,  # Standard scaling
        "dropout": 0.0,
        "scale": 1.0,
    }

    # Convert linear layers to LoRA
    linear_to_lora_layers(
        model,
        num_layers=len(model.model.layers),  # All layers
        config=lora_config,
    )

    # Freeze non-LoRA parameters
    model.freeze()
    # Unfreeze LoRA parameters
    for name, module in model.named_modules():
        if "lora" in name.lower():
            module.unfreeze()

    # Count trainable params
    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            elif hasattr(v, 'size'):
                total += v.size
        return total

    trainable = count_params(model.trainable_parameters())
    total = count_params(model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Training data
    training_data = get_training_data()
    logger.info(f"Training examples: {len(training_data)}")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Loss function
    def loss_fn(model, tokens):
        input_ids = mx.array([tokens[:-1]])
        target_ids = mx.array([tokens[1:]])
        logits = model(input_ids)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = target_ids.reshape(-1)
        return nn.losses.cross_entropy(logits, targets, reduction='mean')

    # Training loop
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(num_epochs):
        total_loss = 0

        for example in training_data:
            full_text = f"Question: {example['input']}\n\n{example['output']}"
            tokens = tokenizer.encode(full_text)

            loss, grads = loss_and_grad(model, tokens)
            mx.eval(loss)

            optimizer.update(model, grads)
            mx.eval(model.parameters())

            total_loss += float(loss)

        avg_loss = total_loss / len(training_data)
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

    # Test after training
    logger.info("\n--- AFTER LORA TRAINING ---")
    trained = generate(model, tokenizer, prompt=test_prompt, max_tokens=80, verbose=False)
    logger.info(f"Trained: {trained[:80]}...")
    has_reflection = "Let me understand" in trained
    logger.info(f"Has self-reflection: {'✓' if has_reflection else '✗'}")

    # Test word problems
    logger.info("\n--- WORD PROBLEM TEST ---")
    word_problems = [
        ("A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?", "0.05"),
        ("5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100?", "5"),
        ("A lily pad doubles daily. Covers lake in 48 days. When half covered?", "47"),
        ("Tom has 3× as many apples as Jane. Jane has 5. How many does Tom have?", "15"),
    ]

    word_correct = 0
    for q, expected in word_problems:
        prompt = f"Question: {q}\n\n"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)
        correct = expected in response
        if correct:
            word_correct += 1
        status = "✓" if correct else "✗"
        logger.info(f"{status} {q[:40]}... → {expected}")

    logger.info(f"\nWord problems: {word_correct}/{len(word_problems)}")

    # Test factual knowledge preserved
    logger.info("\n--- FACT PRESERVATION TEST ---")
    fact_tests = [
        ("What is the capital of France?", "paris"),
        ("What is H2O?", "water"),
        ("How many days in a week?", "7"),
        ("What planet is closest to the sun?", "mercury"),
        ("How many legs does a spider have?", "8"),
    ]

    facts_correct = 0
    for q, expected in fact_tests:
        prompt = f"Question: {q}\n\nAnswer:"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        correct = expected.lower() in response.lower()
        if correct:
            facts_correct += 1
        status = "✓" if correct else "✗"
        logger.info(f"{status} {q} → {response[:30].strip()}...")

    logger.info(f"\nFacts preserved: {facts_correct}/{len(fact_tests)}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Self-reflection learned: {'✓' if has_reflection else '✗'}")
    logger.info(f"Word problems: {word_correct}/{len(word_problems)} ({100*word_correct/len(word_problems):.0f}%)")
    logger.info(f"Facts preserved: {facts_correct}/{len(fact_tests)} ({100*facts_correct/len(fact_tests):.0f}%)")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "rank": rank,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "trainable_params": trainable,
            "total_params": total,
        },
        "results": {
            "has_reflection": has_reflection,
            "word_problems": word_correct / len(word_problems),
            "facts_preserved": facts_correct / len(fact_tests),
        },
        "baseline": baseline[:100],
        "trained": trained[:100],
    }

    output_file = Path("data/experiments/lora_self_reflection_v2.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to: {output_file}")

    return model, tokenizer


if __name__ == "__main__":
    train_lora(
        model_path="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        rank=8,
        num_epochs=15,
        learning_rate=1e-4,
    )
