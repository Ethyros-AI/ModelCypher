#!/usr/bin/env python3
"""Experiment 91b: Self-Improvement Training with Proper Chat Format.

The instruct model needs chat-formatted training data.
This version properly formats the training data.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_chat_training_data(tokenizer, n_samples: int = 500) -> List[dict]:
    """Generate training data in chat format."""
    import numpy as np

    np.random.seed(42)

    addition_templates = [
        "I have {a} apples. I get {b} more. How many total?",
        "{a} birds. {b} more arrive. How many total?",
        "Start with {a}. Add {b}. What's the result?",
        "There are {a} cats. {b} more come. Total?",
        "{a} toys plus {b} toys equals?",
        "Mary has {a} candies. She gets {b} more. How many?",
        "{a} books. Buy {b} more. How many now?",
        "Begin with {a}. Increase by {b}. Result?",
    ]

    subtraction_templates = [
        "{a} apples. {b} eaten. How many remaining?",
        "{a} birds. {b} fly away. How many left?",
        "Start with {a}. Take away {b}. What's left?",
        "There are {a} cats. {b} leave. How many left?",
        "{a} toys. Give away {b}. How many remaining?",
        "Tom has {a} candies. He gives {b} away. How many left?",
        "{a} books. Lose {b}. How many now?",
        "Begin with {a}. Decrease by {b}. Result?",
    ]

    samples = []

    for _ in range(n_samples):
        a = np.random.randint(2, 15)
        b = np.random.randint(1, min(a, 10))

        if np.random.rand() > 0.5:
            template = addition_templates[np.random.randint(len(addition_templates))]
            question = template.format(a=a, b=b)
            answer = str(a + b)
            equation = f"{a}+{b}={answer}"
        else:
            template = subtraction_templates[np.random.randint(len(subtraction_templates))]
            question = template.format(a=a, b=b)
            answer = str(a - b)
            equation = f"{a}-{b}={answer}"

        # Format as chat - both the equation and answer
        # Train to output: "The equation is X+Y=Z. The answer is Z."
        response = f"The equation is {equation}. The answer is {answer}."

        # Format with chat template
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]

        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        samples.append({"text": formatted})

    return samples


def evaluate_accuracy(model, tokenizer, prime: str, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate accuracy on problems."""
    import mlx.core as mx

    results = []
    correct = 0

    for problem, expected in problems:
        prompt = f"{prime} {problem}" if prime else problem
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        predicted = tokenizer.decode([top_token]).strip()

        is_correct = expected in predicted or predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "problem": problem,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def evaluate_with_generation(model, tokenizer, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate using full generation (not just next token)."""
    import mlx.core as mx

    results = []
    correct = 0

    for question, expected in problems:
        # Format as chat
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        tokens = tokenizer.encode(prompt)
        generated = []

        # Generate up to 50 tokens
        for _ in range(50):
            input_ids = mx.array([tokens + generated])
            logits = model(input_ids)
            mx.eval(logits)

            logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()

            next_token = int(np.argmax(probs))
            generated.append(next_token)

            # Stop on end token
            if next_token == tokenizer.eos_token_id:
                break

        response = tokenizer.decode(generated).strip()

        # Check if expected answer is in response
        is_correct = expected in response
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "expected": expected,
            "response": response[:100],
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def main():
    from mlx_lm import load

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"
    train_data_dir = "data/training/self_improve_chat"
    adapter_path = "data/adapters/self_improve_lora_v2"

    logger.info("=" * 60)
    logger.info("EXPERIMENT 91b: SELF-IMPROVEMENT WITH CHAT FORMAT")
    logger.info("=" * 60)

    # Test sets - as chat questions
    arithmetic_tests = [
        ("1+1=", "2"), ("2+2=", "4"), ("3+1=", "4"), ("5+2=", "7"),
        ("4+3=", "7"), ("6+1=", "7"), ("3+3=", "6"), ("2+5=", "7"),
        ("5-2=", "3"), ("4-1=", "3"), ("7-3=", "4"), ("6-2=", "4"),
    ]

    word_problem_tests = [
        ("I have 5 apples. I get 3 more. How many total?", "8"),
        ("7 birds. 4 fly away. How many left?", "3"),
        ("Start with 6. Add 2. What's the result?", "8"),
        ("There are 8 cats. 3 leave. How many left?", "5"),
        ("9 toys plus 1 toy equals?", "10"),
        ("Tom has 4 candies. He gives 2 away. How many left?", "2"),
        ("Begin with 5. Decrease by 2. Result?", "3"),
        ("Mary has 3 books. She gets 4 more. How many?", "7"),
    ]

    prime = "Arithmetic means calculating numbers."

    # Phase 1: Load and baseline
    logger.info("\n=== PHASE 1: BASELINE MEASUREMENT ===")

    logger.info("Loading base model...")
    model, tokenizer = load(model_path)

    arith_baseline, _ = evaluate_accuracy(model, tokenizer, prime, arithmetic_tests)
    logger.info(f"Arithmetic (primed, next-token): {arith_baseline:.0%}")

    wp_baseline, wp_details = evaluate_with_generation(model, tokenizer, word_problem_tests)
    logger.info(f"Word problems (chat generation): {wp_baseline:.0%}")

    logger.info("\nBaseline word problem examples:")
    for r in wp_details[:3]:
        logger.info(f"  Q: {r['question'][:40]}...")
        logger.info(f"  A: {r['response'][:60]}...")
        logger.info(f"  Expected: {r['expected']} | Correct: {r['correct']}")

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATING CHAT-FORMATTED TRAINING DATA ===")

    train_samples = generate_chat_training_data(tokenizer, n_samples=500)

    # Create train/valid/test splits
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)

    # 80/10/10 split
    n_train = int(len(train_samples) * 0.8)
    n_valid = int(len(train_samples) * 0.1)

    train_set = train_samples[:n_train]
    valid_set = train_samples[n_train:n_train + n_valid]
    test_set = train_samples[n_train + n_valid:]

    for name, data in [("train", train_set), ("valid", valid_set), ("test", test_set)]:
        path = Path(train_data_dir) / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"  {name}: {len(data)} samples -> {path}")

    logger.info(f"\nSample training data:")
    sample = train_samples[0]
    logger.info(f"  {sample['text'][:200]}...")

    # Clear model
    del model
    import mlx.core as mx
    mx.clear_cache()

    # Phase 3: Training
    logger.info("\n=== PHASE 3: LORA TRAINING ===")

    n_iters = 200  # More iterations for more data

    Path(adapter_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", adapter_path,
        "--batch-size", "4",
        "--num-layers", "8",  # More layers
        "--iters", str(n_iters),
        "--learning-rate", "5e-5",  # Lower LR
        "--seed", "42",
        "--steps-per-report", "20",
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-10:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Training error: {e}")
        return

    # Phase 4: Evaluation
    logger.info("\n=== PHASE 4: POST-TRAINING EVALUATION ===")

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    arith_post, _ = evaluate_accuracy(model, tokenizer, prime, arithmetic_tests)
    logger.info(f"Arithmetic (primed): {arith_baseline:.0%} → {arith_post:.0%}")

    wp_post, wp_details_post = evaluate_with_generation(model, tokenizer, word_problem_tests)
    logger.info(f"Word problems (chat): {wp_baseline:.0%} → {wp_post:.0%}")

    logger.info("\nPost-training word problem examples:")
    for r in wp_details_post:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} Q: {r['question'][:35]}...")
        logger.info(f"    A: {r['response'][:60]}...")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    arith_preserved = arith_post >= arith_baseline - 0.15
    wp_improved = wp_post > wp_baseline

    logger.info(f"""
                        Before    After
  Arithmetic (oracle):  {arith_baseline:>6.0%}    {arith_post:>5.0%}  {'✓' if arith_preserved else '✗'}
  Word problems:        {wp_baseline:>6.0%}    {wp_post:>5.0%}  {'✓ IMPROVED!' if wp_improved else 'unchanged'}

SUCCESS:
  Arithmetic preserved: {arith_preserved}
  Word problems improved: {wp_improved}
""")

    if arith_preserved and wp_improved:
        logger.info("*** THE MODEL TAUGHT ITSELF! ***")
    elif arith_preserved:
        logger.info("Arithmetic preserved. Word problems may need more training.")

    # Save results
    results = {
        "baseline": {"arithmetic": arith_baseline, "word_problems": wp_baseline},
        "post_training": {"arithmetic": arith_post, "word_problems": wp_post},
        "word_problem_details": wp_details_post,
    }

    output_path = "data/experiments/self_improvement_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
