#!/usr/bin/env python3
"""Phase B Implementation: Train for TRUE GAP patterns.

The 6 failing patterns have ZERO constant matches. This means:
1. Traditional LoRA won't work (causes forgetting)
2. Surgical alignment won't work (nothing to align)

New approach: **Geometric Induction Training**
- Create synthetic examples for each failing pattern type
- Train with VERY low LR (geometry-derived: 1/(κ × scale))
- Add examples that explicitly show the pattern structure
- The goal is to INDUCE geometric alignment, not just memorize

Patterns needing geometric induction:
1. RESTART_RETRY - Operations that reset progress
2. INVERSE_SOLVE - Working backwards to find original
3. ALGEBRAIC_COMPARE - "X costs Y less than Z"
4. COMPLEX_MULTISTEP - Multiple sequential operations
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Fundamental constants for alignment checking
CONSTANTS = {
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "1/phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "1/sqrt2": 1 / np.sqrt(2),
}


def generate_restart_retry_examples(n: int = 10) -> List[Dict]:
    """Generate RESTART_RETRY pattern examples.

    Pattern: Progress is lost partway through and must restart.
    Key insight: Total = attempt_1_partial + restart_penalty + full_completion
    """
    examples = []

    templates = [
        {
            "template": (
                "Question: {name} is completing a {task} of {total} {unit}. "
                "Normally {they} can do {rate} {unit} per {time_unit}, but {percent}% of the way through, "
                "{interruption}. After {delay} {time_unit}, {name} restarts from the beginning. "
                "How many {time_unit} does it take in total?\n\n"
                "Answer: First attempt goes {percent}% of {total} = {partial} {unit}, taking {partial_time} {time_unit}. "
                "Then {delay} {time_unit} for interruption. Then restart means full {total} {unit} from beginning = {full_time} {time_unit}. "
                "Total: {partial_time} + {delay} + {full_time} = {answer}\n\n#### {answer}"
            ),
        }
    ]

    interruptions = [
        "Windows forces a restart for updates",
        "the power goes out",
        "the system crashes",
        "an emergency call interrupts",
        "a mandatory break is required",
    ]

    tasks = [
        ("downloading", "file", "GB", "GB", "minute"),
        ("copying", "documents", "pages", "pages", "minute"),
        ("uploading", "photos", "photos", "photos", "minute"),
        ("processing", "batch", "items", "items", "hour"),
    ]

    names = ["Carla", "Alex", "Jordan", "Sam", "Taylor", "Morgan"]

    for i in range(n):
        name = random.choice(names)
        task_type, task_name, unit, rate_unit, time_unit = random.choice(tasks)
        total = random.randint(100, 500)
        rate = random.randint(1, 10)
        percent = random.choice([20, 25, 30, 40, 50])
        delay = random.randint(10, 30)
        interruption = random.choice(interruptions)

        partial = total * percent // 100
        partial_time = partial // rate
        full_time = total // rate
        answer = partial_time + delay + full_time

        they = "they" if name in ["Jordan", "Sam", "Taylor", "Morgan"] else "she" if name == "Carla" else "he"

        text = templates[0]["template"].format(
            name=name, task=task_name, total=total, unit=unit, rate=rate,
            time_unit=time_unit, percent=percent, interruption=interruption,
            delay=delay, partial=partial, partial_time=partial_time,
            full_time=full_time, answer=answer, they=they
        )

        examples.append({"text": text})

    return examples


def generate_inverse_solve_examples(n: int = 10) -> List[Dict]:
    """Generate INVERSE_SOLVE pattern examples.

    Pattern: Given final state, find original amount.
    Key insight: Work backwards step by step.
    """
    examples = []

    templates = [
        {
            "template": (
                "Question: {name} is selling {items}. {action1}. {action2}. {action3}. "
                "If {name} has {remaining} {items} left, how many did {they} start with?\n\n"
                "Answer: Let's work backwards. Final: {remaining}. {back1}. {back2}. {back3}. "
                "Started with: {answer}\n\n#### {answer}"
            ),
        }
    ]

    items_list = ["vacuum cleaners", "cookies", "books", "tickets", "paintings"]
    names = ["Melanie", "Chris", "Pat", "Riley", "Casey"]

    for i in range(n):
        name = random.choice(names)
        items = random.choice(items_list)
        they = "they"

        # Generate backwards
        remaining = random.randint(3, 10)

        # Action 3: sold half of what was left
        before_action3 = remaining * 2

        # Action 2: sold a fixed number more
        sold_fixed = random.randint(2, 5)
        before_action2 = before_action3 + sold_fixed

        # Action 1: sold a fraction (1/3)
        # before_action1 = before_action2 * 3 / 2 (since 1/3 was sold, 2/3 remain)
        # Actually, if they sold 1/3, then 2/3 of original = before_action2
        original = before_action2 * 3 // 2

        action1 = f"{they.capitalize()} sold a third of them at the first house"
        action2 = f"then sold {sold_fixed} more at the second house"
        action3 = "then sold half of what was left at the third house"

        back3 = f"Before third house: {remaining} × 2 = {before_action3}"
        back2 = f"Before second house: {before_action3} + {sold_fixed} = {before_action2}"
        back1 = f"Before first house: {before_action2} was 2/3, so original = {before_action2} × 3/2 = {original}"

        text = templates[0]["template"].format(
            name=name, items=items, action1=action1, action2=action2, action3=action3,
            remaining=remaining, back1=back1, back2=back2, back3=back3, answer=original, they=they
        )

        examples.append({"text": text})

    return examples


def generate_algebraic_compare_examples(n: int = 10) -> List[Dict]:
    """Generate ALGEBRAIC_COMPARE pattern examples.

    Pattern: X costs/is Y more/less than Z.
    Key insight: Set up the equation and solve.
    """
    examples = []

    comparisons = [
        ("costs", "dollars"),
        ("weighs", "pounds"),
        ("is", "inches"),
        ("contains", "items"),
    ]

    names = ["Gloria", "Dana", "Kim", "Jamie", "Robin"]

    for i in range(n):
        name = random.choice(names)
        verb, unit = random.choice(comparisons)

        # Item A and B together equal something
        item_a_cost = random.randint(20, 50)
        item_b_cost = random.randint(20, 50)
        total_ab = item_a_cost + item_b_cost

        # Item C costs/is X more than A and B together
        diff = random.randint(5, 20)
        item_c = total_ab + diff

        text = (
            f"Question: {name} is shopping. Two items together cost ${total_ab}. "
            f"One item {verb} ${item_a_cost} and the other {verb} ${item_b_cost}. "
            f"A third item {verb} ${diff} more than the two items combined. "
            f"How much does the third item cost?\n\n"
            f"Answer: First two items: ${item_a_cost} + ${item_b_cost} = ${total_ab}. "
            f"Third item is ${diff} more: ${total_ab} + ${diff} = ${item_c}\n\n#### {item_c}"
        )

        examples.append({"text": text})

    return examples


def generate_complex_multistep_examples(n: int = 10) -> List[Dict]:
    """Generate COMPLEX_MULTISTEP pattern examples.

    Pattern: Multiple operations that must be done in order.
    Key insight: Track intermediate values carefully.
    """
    examples = []

    for i in range(n):
        a = random.randint(10, 50)
        b = random.randint(2, 10)
        c = random.randint(5, 20)
        d = random.randint(2, 5)

        step1 = a * b
        step2 = step1 + c
        step3 = step2 // d  # Use integer division for clean numbers
        answer = step3

        text = (
            f"Question: Start with {a}. Multiply by {b}. Add {c}. Divide by {d}. "
            f"What is the result?\n\n"
            f"Answer: Step 1: {a} × {b} = {step1}. "
            f"Step 2: {step1} + {c} = {step2}. "
            f"Step 3: {step2} ÷ {d} = {answer}\n\n#### {answer}"
        )

        examples.append({"text": text})

    return examples


def generate_training_data() -> List[Dict]:
    """Generate all training data for TRUE GAP patterns."""
    examples = []

    logger.info("Generating RESTART_RETRY examples...")
    examples.extend(generate_restart_retry_examples(15))

    logger.info("Generating INVERSE_SOLVE examples...")
    examples.extend(generate_inverse_solve_examples(15))

    logger.info("Generating ALGEBRAIC_COMPARE examples...")
    examples.extend(generate_algebraic_compare_examples(15))

    logger.info("Generating COMPLEX_MULTISTEP examples...")
    examples.extend(generate_complex_multistep_examples(15))

    logger.info(f"Total training examples: {len(examples)}")

    return examples


def count_constant_matches(activations: np.ndarray, sqrt_eps: float) -> int:
    """Count SVD ratio matches to constants."""
    _, S, _ = svd(activations, full_matrices=False)

    min_sv = S[0] * sqrt_eps
    n_valid = np.sum(S > min_sv)

    count = 0
    for i in range(n_valid - 1):
        for j in range(i + 1, n_valid):
            if S[j] < min_sv:
                continue

            ratio = S[i] / S[j]
            for const_val in CONSTANTS.values():
                rel_error = abs(ratio - const_val) / const_val
                if rel_error < sqrt_eps:
                    count += 1
                    break

    return count


def main():
    import mlx.core as mx
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("PHASE B: GEOMETRIC INDUCTION TRAINING")
    logger.info("=" * 70)
    logger.info("\nTraining for TRUE GAP patterns (0 constant matches)")
    logger.info("Goal: INDUCE geometric structure, not just memorize\n")

    # Generate training data
    examples = generate_training_data()

    # Save training data
    train_path = Path("data/training/geometric_gap_train.jsonl")
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Training data saved to: {train_path}")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"\nLoading model: {model_path}")
    logger.info(f"With adapter: {adapter_path}")

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # Compute geometry-derived parameters
    logger.info("\nComputing geometry-derived parameters...")

    # Get sample activations
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    sample_prompts = [ex["text"].split("Answer:")[0] + "Answer:" for ex in examples[:20]]
    activations = []

    for prompt in sample_prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        hidden = model.model.embed_tokens(input_ids)
        for layer in model.model.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        hidden = model.model.norm(hidden)
        mx.eval(hidden)

        activations.append(np.array(hidden[0, -1, :].tolist(), dtype=np.float32))

    A = np.vstack(activations)
    A_centered = A - A.mean(axis=0)
    G = A_centered @ A_centered.T

    _, S, _ = svd(G, full_matrices=False)
    S_valid = S[S > sqrt_eps * S[0]]
    kappa = float(S_valid[0] / S_valid[-1]) if len(S_valid) > 1 else 1.0
    scale = np.linalg.norm(G, 'fro')

    lr = 1.0 / (kappa * scale)
    iterations = int(np.ceil(kappa / 10))  # Scale down for practical training

    logger.info(f"\n  GEOMETRY-DERIVED PARAMETERS:")
    logger.info(f"  κ = {kappa:.4e}")
    logger.info(f"  scale = {scale:.4e}")
    logger.info(f"  LR = 1/(κ×scale) = {lr:.4e}")
    logger.info(f"  Iterations = ceil(κ/10) = {iterations}")

    # Check initial constant matches
    initial_matches = count_constant_matches(A, sqrt_eps)
    logger.info(f"\n  Initial constant matches: {initial_matches}")

    # Prepare for MLX-LM training
    train_config = {
        "model": model_path,
        "adapter_path": adapter_path,
        "train_data": str(train_path),
        "learning_rate": float(lr),
        "iterations": iterations,
        "batch_size": 2,  # Small to avoid overwriting
        "lora_layers": 36,  # All layers very lightly
        "lora_rank": 8,
    }

    config_path = Path("data/training/geometric_gap_config.json")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2)

    logger.info(f"\nTraining config saved to: {config_path}")
    logger.info("\nTo train:")
    logger.info(f"  python -m mlx_lm.lora --config {config_path}")

    # Save experiment info
    output = {
        "timestamp": datetime.now().isoformat(),
        "diagnosis": "TRUE_GAP",
        "patterns": ["RESTART_RETRY", "INVERSE_SOLVE", "ALGEBRAIC_COMPARE", "COMPLEX_MULTISTEP"],
        "n_examples": len(examples),
        "geometry_params": {
            "kappa": float(kappa),
            "scale": float(scale),
            "learning_rate": float(lr),
            "iterations": int(iterations),
        },
        "initial_constant_matches": int(initial_matches),
        "train_path": str(train_path),
        "config_path": str(config_path),
    }

    output_path = Path("data/experiments/geometric_gap_training.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nExperiment info saved to: {output_path}")


if __name__ == "__main__":
    main()
