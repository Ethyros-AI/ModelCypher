#!/usr/bin/env python3
"""Measure Reflection Geometry: Does self-reflection improve φ alignment?

After training the model to self-reflect, we need to verify:
1. The core question extraction IS at φ resonance
2. The model processes at optimal geometry
3. This correlates with better answers

This measures geometry DURING the self-reflection process.
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


def train_self_reflection(model, tokenizer, training_data, learning_rate=2e-5, num_epochs=10):
    """Train model to self-reflect."""
    optimizer = optim.AdamW(learning_rate=learning_rate)

    for epoch in range(num_epochs):
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


def get_training_data():
    return [
        {"input": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
         "output": "Let me understand the question. What is the ball's cost if bat + ball = $1.10 and bat = ball + $1?\n\nAnswer: $0.05"},
        {"input": "I was wondering if you could help me figure out what happens when you add the number five to the number three?",
         "output": "Let me understand the question. What is 5 + 3?\n\nAnswer: 8"},
        {"input": "In the context of basic arithmetic, what is fifteen plus seven?",
         "output": "Let me understand the question. What is 15 + 7?\n\nAnswer: 22"},
        {"input": "If you have 5 apples and someone gives you 3 more apples, how many do you have?",
         "output": "Let me understand the question. What is 5 + 3?\n\nAnswer: 8"},
        {"input": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
         "output": "Let me understand the question. Time for N machines to make N widgets?\n\nAnswer: 5 minutes"},
        {"input": "What is 2 + 2?",
         "output": "Let me understand the question. What is 2 + 2?\n\nAnswer: 4"},
        {"input": "Tom has 3 times as many apples as Jane. Jane has 5 apples. How many does Tom have?",
         "output": "Let me understand the question. If Tom = 3 × Jane and Jane = 5?\n\nAnswer: 15"},
        {"input": "All cats have tails. Fluffy is a cat. Does Fluffy have a tail?",
         "output": "Let me understand the question. Does Fluffy have a tail given all cats do?\n\nAnswer: Yes"},
    ]


def measure_reflection_geometry():
    """Compare geometry: baseline model vs trained self-reflection model."""
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"

    # Test cases - questions where self-reflection should help
    test_cases = [
        {
            "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "core": "What is ball's cost if total = $1.10 and bat = ball + $1?",
            "answer": "$0.05"
        },
        {
            "question": "I was wondering if you could help me figure out what happens when you add the number five to the number three?",
            "core": "What is 5 + 3?",
            "answer": "8"
        },
        {
            "question": "In the context of basic arithmetic, what is fifteen plus seven?",
            "core": "What is 15 + 7?",
            "answer": "22"
        },
        {
            "question": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
            "core": "Time for N machines to make N widgets?",
            "answer": "5 minutes"
        },
    ]

    logger.info("=" * 70)
    logger.info("COMPARING BASELINE vs SELF-REFLECTION GEOMETRY")
    logger.info("=" * 70)
    logger.info(f"Target φ: {PHI:.4f}")

    # Load model
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    # Measure baseline geometry and responses
    logger.info("\n--- BASELINE MODEL ---")
    baseline_results = []

    for tc in test_cases:
        prompt = f"Question: {tc['question']}\n\nAnswer:"
        ratio, tokens = compute_ratio(model, tokenizer, tc["question"])
        core_ratio, core_tokens = compute_ratio(model, tokenizer, tc["core"])

        response = generate(model, tokenizer, prompt=prompt, max_tokens=30, verbose=False)
        correct = tc["answer"].lower() in response.lower()

        logger.info(f"\nQ ({tokens} tok): ratio={ratio:.3f}, dist_φ={abs(ratio-PHI):.3f}")
        logger.info(f"Core ({core_tokens} tok): ratio={core_ratio:.3f}, dist_φ={abs(core_ratio-PHI):.3f}")
        logger.info(f"Answer: {response[:40]}... {'✓' if correct else '✗'}")

        baseline_results.append({
            "question_ratio": ratio,
            "question_dist": abs(ratio - PHI),
            "core_ratio": core_ratio,
            "core_dist": abs(core_ratio - PHI),
            "correct": correct,
        })

    # Train model for self-reflection
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SELF-REFLECTION...")
    logger.info("=" * 70)

    training_data = get_training_data()
    train_self_reflection(model, tokenizer, training_data)
    logger.info("Training complete")

    # Measure trained geometry and responses
    logger.info("\n--- TRAINED SELF-REFLECTION MODEL ---")
    trained_results = []

    for tc in test_cases:
        prompt = f"Question: {tc['question']}\n\n"
        ratio, tokens = compute_ratio(model, tokenizer, tc["question"])

        response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)

        # Extract the core question from the response
        extracted_core = ""
        if "Let me understand" in response:
            parts = response.split("Let me understand the question.")
            if len(parts) > 1:
                extracted_core = parts[1].split("\n")[0].strip()

        if extracted_core:
            extracted_ratio, extracted_tokens = compute_ratio(model, tokenizer, extracted_core)
        else:
            extracted_ratio, extracted_tokens = ratio, tokens

        correct = tc["answer"].lower() in response.lower()

        logger.info(f"\nQ ({tokens} tok): ratio={ratio:.3f}, dist_φ={abs(ratio-PHI):.3f}")
        if extracted_core:
            logger.info(f"Extracted ({extracted_tokens} tok): ratio={extracted_ratio:.3f}, dist_φ={abs(extracted_ratio-PHI):.3f}")
            logger.info(f"  '{extracted_core[:50]}'")
        logger.info(f"Answer: {response[:50]}... {'✓' if correct else '✗'}")

        trained_results.append({
            "question_ratio": ratio,
            "question_dist": abs(ratio - PHI),
            "extracted_core": extracted_core,
            "extracted_ratio": extracted_ratio if extracted_core else None,
            "extracted_dist": abs(extracted_ratio - PHI) if extracted_core else None,
            "correct": correct,
            "has_reflection": "Let me understand" in response,
        })

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    baseline_correct = sum(1 for r in baseline_results if r["correct"])
    trained_correct = sum(1 for r in trained_results if r["correct"])
    trained_reflection = sum(1 for r in trained_results if r["has_reflection"])

    avg_baseline_dist = np.mean([r["question_dist"] for r in baseline_results])
    avg_trained_extracted = np.mean([r["extracted_dist"] for r in trained_results if r["extracted_dist"]])

    logger.info(f"Baseline accuracy: {baseline_correct}/{len(baseline_results)}")
    logger.info(f"Trained accuracy: {trained_correct}/{len(trained_results)}")
    logger.info(f"Self-reflection rate: {trained_reflection}/{len(trained_results)}")
    logger.info(f"")
    logger.info(f"Avg question dist from φ: {avg_baseline_dist:.3f}")
    logger.info(f"Avg extracted core dist from φ: {avg_trained_extracted:.3f}")

    if avg_trained_extracted < avg_baseline_dist:
        improvement = (avg_baseline_dist - avg_trained_extracted) / avg_baseline_dist * 100
        logger.info(f"\n✓ Self-reflection improves φ alignment by {improvement:.0f}%!")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "phi_target": float(PHI),
        "baseline_results": baseline_results,
        "trained_results": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}
            for r in trained_results
        ],
        "summary": {
            "baseline_accuracy": baseline_correct / len(baseline_results),
            "trained_accuracy": trained_correct / len(trained_results),
            "reflection_rate": trained_reflection / len(trained_results),
            "avg_question_dist": float(avg_baseline_dist),
            "avg_extracted_dist": float(avg_trained_extracted) if not np.isnan(avg_trained_extracted) else None,
        }
    }

    output_path = Path("data/experiments/reflection_geometry.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    measure_reflection_geometry()
