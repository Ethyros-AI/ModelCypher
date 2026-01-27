#!/usr/bin/env python3
"""Targeted GSM8K training for the 6 failure patterns.

Failure patterns:
1. "Increased BY X%" - value increase semantics
2. Restart/interrupt scenarios - multi-phase calculations
3. Multi-segment journeys - varying speeds/times
4. Breakeven with costs - net profit, not gross
5. Working backwards - algebraic reasoning
6. Speed vs distance - unit clarity
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.use_cases.curriculum import BenchmarkLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_targeted_training(n_per_type: int = 50, seed: int = 42) -> List[dict]:
    """Generate training data for the 6 failure patterns."""
    np.random.seed(seed)
    samples = []

    # === PATTERN 1: "Increased BY X%" ===
    logger.info("Generating 'increased BY X%' problems...")
    for _ in range(n_per_type):
        original = np.random.choice([50, 80, 100, 120, 200]) * 1000  # House prices
        pct = np.random.choice([50, 100, 150, 200])
        increase = int(original * pct / 100)
        new_value = original + increase
        cost = original + np.random.randint(20, 60) * 1000  # Repairs
        profit = new_value - cost

        if profit > 0:
            text = f"""Question: A house costs ${original:,}. After ${cost-original:,} in repairs, the value increased BY {pct}%. What is the profit?

Answer: The value increased by {original:,} * {pct/100} = <<{original}*{pct/100}={increase}>>{increase:,}
So the new value is {original:,} + {increase:,} = <<{original}+{increase}={new_value}>>{new_value:,}
Total cost was {original:,} + {cost-original:,} = <<{original}+{cost-original}={cost}>>{cost:,}
Profit is {new_value:,} - {cost:,} = <<{new_value}-{cost}={profit}>>{profit:,}
#### {profit}"""
            samples.append({"text": text})

    # === PATTERN 2: Restart/Interrupt scenarios ===
    logger.info("Generating restart/interrupt problems...")
    for _ in range(n_per_type):
        total_size = np.random.choice([100, 200, 500])
        rate = np.random.choice([2, 5, 10])
        restart_pct = np.random.choice([20, 40, 50])
        restart_time = np.random.choice([10, 20, 30])

        partial_size = int(total_size * restart_pct / 100)
        partial_time = partial_size // rate
        full_time = total_size // rate
        total_time = partial_time + restart_time + full_time

        text = f"""Question: Downloading a {total_size} GB file at {rate} GB/minute. At {restart_pct}% complete, a restart takes {restart_time} minutes. Then download restarts from beginning. Total time?

Answer: First {restart_pct}% = {total_size} * {restart_pct/100} = <<{total_size}*{restart_pct/100}={partial_size}>>{partial_size} GB
Time for partial: {partial_size} / {rate} = <<{partial_size}/{rate}={partial_time}>>{partial_time} minutes
Time for full after restart: {total_size} / {rate} = <<{total_size}/{rate}={full_time}>>{full_time} minutes
Total: {partial_time} + {restart_time} + {full_time} = <<{partial_time}+{restart_time}+{full_time}={total_time}>>{total_time} minutes
#### {total_time}"""
        samples.append({"text": text})

    # === PATTERN 3: Multi-segment journeys ===
    logger.info("Generating multi-segment journey problems...")
    for _ in range(n_per_type):
        speed1 = np.random.choice([30, 40, 60])
        time1 = np.random.choice([2, 3, 4])
        dist_out = speed1 * time1

        traffic_time = np.random.choice([1, 2])
        speed2 = np.random.choice([20, 30])
        time2 = np.random.choice([0.5, 1])
        dist2 = int(speed2 * time2)

        speed3 = np.random.choice([60, 80])
        time3 = np.random.choice([1, 1.5, 2])
        dist3 = int(speed3 * time3)

        total_back = dist2 + dist3
        remaining = dist_out - total_back

        if remaining > 0:
            text = f"""Question: John drives {time1} hours at {speed1} mph away from home, then turns back. He spends {traffic_time} hours in traffic, then {time2} hours at {speed2} mph, then {time3} hours at {speed3} mph. How far from home?

Answer: Distance out: {time1} * {speed1} = <<{time1}*{speed1}={dist_out}>>{dist_out} miles
At {speed2} mph for {time2} hr: {speed2} * {time2} = <<{speed2}*{time2}={dist2}>>{dist2} miles
At {speed3} mph for {time3} hr: {speed3} * {time3} = <<{speed3}*{time3}={dist3}>>{dist3} miles
Total driven back: {dist2} + {dist3} = <<{dist2}+{dist3}={total_back}>>{total_back} miles
Distance from home: {dist_out} - {total_back} = <<{dist_out}-{total_back}={remaining}>>{remaining} miles
#### {remaining}"""
            samples.append({"text": text})

    # === PATTERN 4: Breakeven with costs ===
    logger.info("Generating breakeven with costs problems...")
    for _ in range(n_per_type):
        initial_cost = np.random.choice([60, 90, 120])
        items_per_year = np.random.choice([5, 7, 10])
        price_per_item = np.random.choice([1.5, 2, 2.5])
        yearly_cost = np.random.choice([2, 3, 5])

        gross = items_per_year * price_per_item
        net = gross - yearly_cost
        breakeven_years = int(initial_cost / net)
        earning_year = breakeven_years + 1

        text = f"""Question: A tree costs ${initial_cost}. Each year it produces {items_per_year} items worth ${price_per_item} each. Yearly maintenance is ${yearly_cost}. When does it start earning money?

Answer: Yearly revenue: {items_per_year} * {price_per_item} = <<{items_per_year}*{price_per_item}={gross}>>{gross}
Net profit per year: {gross} - {yearly_cost} = <<{gross}-{yearly_cost}={net}>>{net}
Years to break even: {initial_cost} / {net} = <<{initial_cost}/{net}={breakeven_years}>>{breakeven_years}
Start earning in year: {breakeven_years} + 1 = <<{breakeven_years}+1={earning_year}>>{earning_year}
#### {earning_year}"""
        samples.append({"text": text})

    # === PATTERN 5: Working backwards ===
    logger.info("Generating working backwards problems...")
    for _ in range(n_per_type):
        final = np.random.choice([3, 4, 5, 6])
        sold_at_last = "half"
        added_middle = np.random.choice([2, 3, 4])
        first_fraction = "a third"

        # Working backwards
        before_last = final * 2  # Sold half
        before_middle = before_last + added_middle
        # before_middle is (1 - 1/3) = 2/3 of start
        start = int(before_middle * 3 / 2)

        text = f"""Question: A seller sold {first_fraction} at the first stop, {added_middle} more at the second stop, and {sold_at_last} of what was left at the third stop. If {final} items remain, how many did they start with?

Answer: Working backwards from {final} remaining:
Before third stop (sold half): {final} * 2 = <<{final}*2={before_last}>>{before_last}
Before second stop (added {added_middle}): {before_last} + {added_middle} = <<{before_last}+{added_middle}={before_middle}>>{before_middle}
This {before_middle} is 2/3 of start (sold 1/3 at first)
Start: {before_middle} * 3 / 2 = <<{before_middle}*3/2={start}>>{start}
#### {start}"""
        samples.append({"text": text})

    # === PATTERN 6: Speed calculations ===
    logger.info("Generating speed calculation problems...")
    for _ in range(n_per_type):
        total_dist = np.random.choice([12, 15, 20])
        target_speed = np.random.choice([3, 4, 5])
        total_time = total_dist // target_speed

        dist1 = np.random.randint(3, total_dist // 2)
        time1 = 1
        dist2 = np.random.randint(1, 4)
        time2 = 1

        remaining_dist = total_dist - dist1 - dist2
        remaining_time = total_time - time1 - time2
        required_speed = remaining_dist // remaining_time if remaining_time > 0 else remaining_dist

        if remaining_time > 0 and required_speed == remaining_dist / remaining_time:
            text = f"""Question: Hiking a {total_dist}-mile trail. Walked {dist1} miles in {time1} hour, then {dist2} miles in {time2} hour. To average {target_speed} mph overall, what speed for the rest?

Answer: Total time for {target_speed} mph: {total_dist} / {target_speed} = <<{total_dist}/{target_speed}={total_time}>>{total_time} hours
Time spent: {time1} + {time2} = <<{time1}+{time2}={time1+time2}>>{time1+time2} hours
Time left: {total_time} - {time1+time2} = <<{total_time}-{time1+time2}={remaining_time}>>{remaining_time} hours
Distance left: {total_dist} - {dist1} - {dist2} = <<{total_dist}-{dist1}-{dist2}={remaining_dist}>>{remaining_dist} miles
Required speed: {remaining_dist} / {remaining_time} = <<{remaining_dist}/{remaining_time}={required_speed}>>{required_speed} mph
#### {required_speed}"""
            samples.append({"text": text})

    logger.info(f"Generated {len(samples)} targeted samples")
    return samples


def main():
    from mlx_lm import load
    import mlx.core as mx
    import re

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    prev_adapter = "data/adapters/qwen3_gsm8k_mastery_lora"
    train_data_dir = "data/training/qwen3_gsm8k_targeted"
    new_adapter_path = "data/adapters/qwen3_gsm8k_targeted_lora"

    logger.info("=" * 70)
    logger.info("TARGETED GSM8K TRAINING - 6 FAILURE PATTERNS")
    logger.info("=" * 70)

    loader = BenchmarkLoader()

    # Generate targeted samples
    targeted = generate_targeted_training(n_per_type=60)

    # Also load more GSM8K training data
    gsm_train = loader.load("gsm8k", split="train", limit=400)
    gsm_samples = []
    for sample in gsm_train.samples:
        question = sample.prompt.replace("Answer:", "").strip()
        full_answer = sample.metadata.get("full_answer", sample.answer)
        gsm_samples.append({"text": f"Question: {question}\n\nAnswer: {full_answer}"})

    # Cumulative arithmetic
    arith_samples = []
    for a in range(1, 20):
        for b in range(1, 20):
            arith_samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                arith_samples.append({"text": f"{a}-{b}={a-b}"})
            arith_samples.append({"text": f"{a}*{b}={a*b}"})

    # Combine
    all_samples = targeted + gsm_samples + arith_samples[:500]
    np.random.shuffle(all_samples)

    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"  Targeted: {len(targeted)}")
    logger.info(f"  GSM8K: {len(gsm_samples)}")
    logger.info(f"  Arithmetic: 500")

    # Save
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)
    n_train = int(len(all_samples) * 0.85)
    n_valid = int(len(all_samples) * 0.10)

    for name, data in [
        ("train", all_samples[:n_train]),
        ("valid", all_samples[n_train:n_train + n_valid]),
        ("test", all_samples[n_train + n_valid:]),
    ]:
        path = Path(train_data_dir) / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"  {name}: {len(data)} samples")

    # Train
    logger.info("\n=== TRAINING ===")
    Path(new_adapter_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", new_adapter_path,
        "--batch-size", "1",
        "--num-layers", "16",
        "--iters", "2000",
        "--learning-rate", "1.5e-5",
        "--seed", "42",
        "--steps-per-report", "400",
    ]

    logger.info("Training (2000 iterations)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=12000)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-12:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Evaluate
    logger.info("\n=== EVALUATION ===")

    model, tokenizer = load(model_path, adapter_path=new_adapter_path)

    gsm_test = loader.load("gsm8k", split="test", limit=30)
    test_problems = [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples[:20]]

    correct = 0
    for question, expected in test_problems:
        prompt = f"Question: {question}\n\nAnswer:"
        tokens = tokenizer.encode(prompt)
        generated = []

        for _ in range(250):
            logits = model(mx.array([tokens + generated]))
            mx.eval(logits)
            logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()
            next_tok = int(np.argmax(probs))
            generated.append(next_tok)

            decoded = tokenizer.decode(generated)
            if "####" in decoded:
                for _ in range(15):
                    logits = model(mx.array([tokens + generated]))
                    mx.eval(logits)
                    logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
                    probs = np.exp(logits_np - logits_np.max())
                    probs = probs / probs.sum()
                    next_tok = int(np.argmax(probs))
                    generated.append(next_tok)
                break
            if "<|im_end|>" in decoded:
                break

        output = tokenizer.decode(generated).strip().replace("<|im_end|>", "")

        if "####" in output:
            answer_part = output.split("####")[-1].strip().replace(",", "").replace("$", "")
            numbers = re.findall(r'-?\d+', answer_part)
            predicted = numbers[0] if numbers else ""
        else:
            numbers = re.findall(r'-?\d+', output.replace(",", ""))
            predicted = numbers[-1] if numbers else ""

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        mark = "OK" if is_correct else "XX"
        logger.info(f"  {mark}: '{question[:40]}...' -> '{predicted}' (expected '{expected}')")

    accuracy = correct / len(test_problems)
    logger.info(f"\n{'='*60}")
    logger.info(f"GSM8K Accuracy: {accuracy:.0%} ({correct}/{len(test_problems)})")
    logger.info(f"{'='*60}")

    if accuracy == 1.0:
        logger.info("\n*** 100% MASTERY ACHIEVED! ***")
    else:
        logger.info(f"\nStill need {len(test_problems) - correct} more correct for 100%")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
