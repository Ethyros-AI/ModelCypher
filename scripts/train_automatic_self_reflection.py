#!/usr/bin/env python3
"""Train Automatic Self-Reflection: Model learns to ask itself first.

The discovery:
    Question normalization improves φ alignment by 73%.
    But we don't want to prompt it - we want the model to LEARN this.

Training approach:
    Input: Raw question
    Target: "Let me understand the question. [Core question]. [Answer]"

    The model learns that the FIRST thing to do is clarify what's being asked.
    This naturally puts processing at the φ resonance point.

This is alignment through learned behavior, not prompting.
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


def get_self_reflection_training_data() -> list[dict]:
    """Training data that teaches self-reflection first.

    Format: question -> self-reflection -> answer

    The model learns to ALWAYS clarify the question before answering.
    """
    return [
        # Math problems - verbose to core
        {
            "input": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "output": "Let me understand the question. What is the cost of the ball if bat + ball = $1.10 and bat = ball + $1.00?\n\nSolving: ball + (ball + 1) = 1.10, so 2*ball = 0.10, ball = $0.05."
        },
        {
            "input": "I was wondering if you could help me figure out what happens when you add the number five to the number three?",
            "output": "Let me understand the question. What is 5 + 3?\n\n5 + 3 = 8."
        },
        {
            "input": "In the context of basic arithmetic operations commonly taught in elementary school mathematics, if we consider the simple addition of two positive integers, specifically the numbers fifteen and seven, what would be the resulting sum?",
            "output": "Let me understand the question. What is 15 + 7?\n\n15 + 7 = 22."
        },
        {
            "input": "If you have 5 apples and someone gives you 3 more apples, how many apples do you have in total?",
            "output": "Let me understand the question. What is 5 + 3?\n\n5 + 3 = 8 apples."
        },
        {
            "input": "5 machines take 5 minutes to make 5 widgets. How long would it take 100 machines to make 100 widgets?",
            "output": "Let me understand the question. If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?\n\n1 machine makes 1 widget in 5 minutes. So 100 machines make 100 widgets in 5 minutes."
        },
        {
            "input": "A lily pad doubles in size every day. If it takes 48 days to cover a lake, how many days to cover half?",
            "output": "Let me understand the question. If doubling daily covers lake in 48 days, when is it half covered?\n\nIf day 48 is full, day 47 was half (since it doubles). Answer: 47 days."
        },
        # Logic problems
        {
            "input": "All cats have tails. Fluffy is a cat. Does Fluffy have a tail?",
            "output": "Let me understand the question. Given all cats have tails and Fluffy is a cat, does Fluffy have a tail?\n\nYes, by syllogism (modus ponens)."
        },
        {
            "input": "Some fruits are red. Apples are fruits. Are all apples red?",
            "output": "Let me understand the question. Does 'some fruits are red' plus 'apples are fruits' imply all apples are red?\n\nNo. 'Some' doesn't mean 'all'. Green apples exist."
        },
        # Simple questions (already at resonance)
        {
            "input": "What is 2 + 2?",
            "output": "Let me understand the question. What is 2 + 2?\n\n2 + 2 = 4."
        },
        {
            "input": "What color is the sky?",
            "output": "Let me understand the question. What color is the sky?\n\nThe sky appears blue due to Rayleigh scattering."
        },
        # Word problems with relationships
        {
            "input": "Tom has 3 times as many apples as Jane. Jane has 5 apples. How many does Tom have?",
            "output": "Let me understand the question. If Tom = 3 × Jane and Jane = 5, what is Tom?\n\nTom = 3 × 5 = 15 apples."
        },
        {
            "input": "A train travels 60 km/h for 2 hours, then 80 km/h for 1.5 hours. What is the total distance?",
            "output": "Let me understand the question. Distance = 60×2 + 80×1.5?\n\n120 + 120 = 240 km."
        },
    ]


def train_self_reflection(
    model_path: str,
    learning_rate: float = 1e-5,
    num_epochs: int = 5,
):
    """Train the model to automatically self-reflect.

    Loss = task_loss (the model learns the self-reflection pattern)

    We monitor comp/φ to verify training improves geometric alignment.
    """
    logger.info("=" * 70)
    logger.info("TRAINING AUTOMATIC SELF-REFLECTION")
    logger.info("=" * 70)
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Target: Model learns 'Let me understand the question' pattern")

    # Load model
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    # Training data
    training_data = get_self_reflection_training_data()
    logger.info(f"Training examples: {len(training_data)}")

    # Measure baseline
    logger.info("\n" + "-" * 40)
    logger.info("BASELINE MEASUREMENTS")
    logger.info("-" * 40)

    baseline_ratios = []
    for example in training_data[:3]:
        ratio, tokens = compute_ratio(model, tokenizer, example["input"])
        dist = abs(ratio - PHI)
        baseline_ratios.append(dist)
        logger.info(f"Input dist from φ: {dist:.3f} ({tokens} tokens)")

    avg_baseline = np.mean(baseline_ratios)
    logger.info(f"Average baseline distance from φ: {avg_baseline:.3f}")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Training loop
    history = []
    step = 0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 40)

        epoch_losses = []
        epoch_ratios = []

        for example in training_data:
            # Full sequence: input + output
            full_text = f"Question: {example['input']}\n\n{example['output']}"
            tokens = tokenizer.encode(full_text)

            # Compute gradients on task loss
            def loss_fn(model):
                input_ids = mx.array([tokens[:-1]])
                target_ids = mx.array([tokens[1:]])
                logits = model(input_ids)
                logits = logits.reshape(-1, logits.shape[-1])
                targets = target_ids.reshape(-1)
                return nn.losses.cross_entropy(logits, targets, reduction='mean')

            loss_and_grad = nn.value_and_grad(model, loss_fn)
            task_loss, grads = loss_and_grad(model)
            mx.eval(task_loss)

            # Measure ratio on the OUTPUT (what we want the model to produce)
            ratio, _ = compute_ratio(model, tokenizer, example["output"])

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            step += 1
            loss_val = float(task_loss)
            dist_phi = abs(ratio - PHI)

            epoch_losses.append(loss_val)
            epoch_ratios.append(dist_phi)

            if step % 4 == 0:
                logger.info(f"Step {step}: loss={loss_val:.4f}, ratio={ratio:.3f}, dist_φ={dist_phi:.3f}")

            history.append({
                "step": step,
                "epoch": epoch + 1,
                "loss": loss_val,
                "ratio": float(ratio),
                "distance_from_phi": float(dist_phi),
            })

        # Epoch summary
        mean_loss = np.mean(epoch_losses)
        mean_dist = np.mean(epoch_ratios)
        logger.info(f"Epoch {epoch + 1}: loss={mean_loss:.4f}, avg dist_φ={mean_dist:.3f}")

    # Test generation
    logger.info("\n" + "=" * 70)
    logger.info("TESTING LEARNED SELF-REFLECTION")
    logger.info("=" * 70)

    test_prompts = [
        "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\n\n",
        "Question: What is 5 + 3?\n\n",
        "Question: In basic arithmetic, what is fifteen plus seven?\n\n",
    ]

    test_results = []
    for prompt in test_prompts:
        response = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)

        # Check if it starts with self-reflection
        has_reflection = "Let me understand" in response or "understand the question" in response.lower()

        # Measure geometry of response
        ratio, tokens = compute_ratio(model, tokenizer, response)
        dist = abs(ratio - PHI)

        logger.info(f"\nPrompt: {prompt[:50]}...")
        logger.info(f"Response: {response[:100]}...")
        logger.info(f"Has self-reflection: {'✓' if has_reflection else '✗'}")
        logger.info(f"Ratio: {ratio:.3f}, dist_φ: {dist:.3f}")

        test_results.append({
            "prompt": prompt[:50],
            "response": response[:200],
            "has_reflection": has_reflection,
            "ratio": float(ratio),
            "distance_from_phi": float(dist),
        })

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)

    reflection_rate = sum(1 for r in test_results if r["has_reflection"]) / len(test_results)
    avg_final_dist = np.mean([r["distance_from_phi"] for r in test_results])

    logger.info(f"Self-reflection rate: {reflection_rate*100:.0f}%")
    logger.info(f"Baseline avg dist from φ: {avg_baseline:.3f}")
    logger.info(f"Final avg dist from φ: {avg_final_dist:.3f}")

    if avg_final_dist < avg_baseline:
        improvement = (avg_baseline - avg_final_dist) / avg_baseline * 100
        logger.info(f"✓ Improved φ alignment by {improvement:.1f}%!")
    else:
        logger.info("✗ No improvement in φ alignment")

    if reflection_rate > 0.5:
        logger.info(f"✓ Model learned self-reflection pattern ({reflection_rate*100:.0f}% of responses)")
    else:
        logger.info(f"? Model needs more training on self-reflection")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "target_phi": float(PHI),
        },
        "baseline_avg_dist": float(avg_baseline),
        "history": history,
        "test_results": test_results,
        "summary": {
            "reflection_rate": float(reflection_rate),
            "final_avg_dist": float(avg_final_dist),
            "improvement_pct": float((avg_baseline - avg_final_dist) / avg_baseline * 100) if avg_final_dist < avg_baseline else 0,
        },
    }

    output_path = Path("data/experiments/automatic_self_reflection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return history, test_results


if __name__ == "__main__":
    train_self_reflection(
        model_path="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        learning_rate=1e-5,
        num_epochs=5,
    )
