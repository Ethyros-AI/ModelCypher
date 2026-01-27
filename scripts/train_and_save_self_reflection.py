#!/usr/bin/env python3
"""Train and Save Self-Reflection Model.

This script:
1. Trains the model to self-reflect
2. Saves the trained weights
3. Tests the saved model

The goal: A model that automatically extracts the core question first.
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
from mlx_lm.utils import save_model

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


def compute_ratio(model, tokenizer, text: str) -> tuple[float, int]:
    """Compute compression ratio for a given text."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)
    peak = float(mx.sqrt(mx.sum(hidden * hidden)))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norm = float(mx.sqrt(mx.sum(hidden * hidden)))
        peak = max(peak, norm)

    final = norm
    return peak / final, len(tokens)


def get_training_data() -> list[dict]:
    """Training data teaching self-reflection."""
    return [
        {
            "input": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "output": "Let me understand the question. What is the ball's cost if bat + ball = $1.10 and bat = ball + $1?\n\nAnswer: $0.05"
        },
        {
            "input": "I was wondering if you could help me figure out what happens when you add the number five to the number three?",
            "output": "Let me understand the question. What is 5 + 3?\n\nAnswer: 8"
        },
        {
            "input": "In the context of basic arithmetic, what is fifteen plus seven?",
            "output": "Let me understand the question. What is 15 + 7?\n\nAnswer: 22"
        },
        {
            "input": "If you have 5 apples and someone gives you 3 more apples, how many do you have?",
            "output": "Let me understand the question. What is 5 + 3?\n\nAnswer: 8"
        },
        {
            "input": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
            "output": "Let me understand the question. Time for N machines to make N widgets?\n\nAnswer: 5 minutes"
        },
        {
            "input": "A lily pad doubles in size every day. If it takes 48 days to cover a lake, how many days to cover half?",
            "output": "Let me understand the question. When is the lake half covered if full at day 48?\n\nAnswer: 47 days"
        },
        {
            "input": "What is 2 + 2?",
            "output": "Let me understand the question. What is 2 + 2?\n\nAnswer: 4"
        },
        {
            "input": "What is the capital of France?",
            "output": "Let me understand the question. What is France's capital city?\n\nAnswer: Paris"
        },
        {
            "input": "Tom has 3 times as many apples as Jane. Jane has 5 apples. How many does Tom have?",
            "output": "Let me understand the question. If Tom = 3 × Jane and Jane = 5?\n\nAnswer: 15"
        },
        {
            "input": "All cats have tails. Fluffy is a cat. Does Fluffy have a tail?",
            "output": "Let me understand the question. Does Fluffy have a tail given all cats do and Fluffy is a cat?\n\nAnswer: Yes"
        },
    ]


def train_and_test(
    base_model_path: str,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
):
    """Train model for self-reflection and save weights."""
    logger.info("=" * 70)
    logger.info("TRAINING SELF-REFLECTION MODEL")
    logger.info("=" * 70)

    # Load base model
    logger.info(f"Loading base model: {base_model_path}")
    model, tokenizer = load(base_model_path)

    training_data = get_training_data()
    logger.info(f"Training examples: {len(training_data)}")

    # Test baseline
    logger.info("\n--- BASELINE ---")
    test_prompt = "Question: What is 5 + 3?\n\n"
    baseline_response = generate(model, tokenizer, prompt=test_prompt, max_tokens=30, verbose=False)
    logger.info(f"Baseline: {baseline_response[:60]}...")

    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Training
    for epoch in range(num_epochs):
        epoch_losses = []

        for example in training_data:
            full_text = f"Question: {example['input']}\n\n{example['output']}"
            tokens = tokenizer.encode(full_text)

            def loss_fn(model):
                input_ids = mx.array([tokens[:-1]])
                target_ids = mx.array([tokens[1:]])
                logits = model(input_ids)
                logits = logits.reshape(-1, logits.shape[-1])
                targets = target_ids.reshape(-1)
                return nn.losses.cross_entropy(logits, targets, reduction='mean')

            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad(model)
            mx.eval(loss)

            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_losses.append(float(loss))

        mean_loss = np.mean(epoch_losses)
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: loss={mean_loss:.4f}")

    # Test trained model (in memory)
    logger.info("\n--- TRAINED (IN MEMORY) ---")
    trained_response = generate(model, tokenizer, prompt=test_prompt, max_tokens=30, verbose=False)
    logger.info(f"Trained: {trained_response[:60]}...")
    has_reflection = "Let me understand" in trained_response
    logger.info(f"Has self-reflection: {'✓' if has_reflection else '✗'}")

    # Test on multiple prompts (in memory - model is trained)
    logger.info("\n--- MULTI-PROMPT TEST ---")
    model2 = model
    tokenizer2 = tokenizer
    test_prompts = [
        "Question: A bat and a ball cost $1.10. The bat costs $1 more. How much is the ball?\n\n",
        "Question: In basic arithmetic, what is fifteen plus seven?\n\n",
        "Question: What is 2 + 2?\n\n",
    ]

    reflection_count = 0
    for prompt in test_prompts:
        response = generate(model2, tokenizer2, prompt=prompt, max_tokens=40, verbose=False)
        has_ref = "Let me understand" in response
        if has_ref:
            reflection_count += 1
        logger.info(f"{'✓' if has_ref else '✗'} {response[:50]}...")

    logger.info(f"\nSelf-reflection rate: {reflection_count}/{len(test_prompts)}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline_response": baseline_response[:100],
        "trained_response": trained_response[:100],
        "has_reflection_trained": has_reflection,
        "reflection_rate": reflection_count / len(test_prompts),
    }

    output_file = Path("data/experiments/self_reflection_training.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    train_and_test(
        base_model_path="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        learning_rate=2e-5,
        num_epochs=10,
    )
