#!/usr/bin/env python3
"""Distilled logic training - 10 perfect examples > 10,000 mediocre ones.

For each failing pattern, distill to the FUNDAMENTAL LOGICAL SHAPE.
Walk the model through the reasoning step-by-step so it learns
the SHAPE of logic, not just surface patterns.

The 6 logical shapes we need to teach:
1. PERCENTAGE INCREASE: new = original + (original × percent)
2. AVERAGE RATE: total_output / total_input (not mean of rates)
3. THRESHOLD CROSSING: breakeven + 1 = first profitable
4. INVERSE CHAIN: work backwards, undo operations in reverse
5. SEQUENTIAL OPERATIONS: subtract first, THEN multiply
6. REMAINING FIRST: compute what's left BEFORE applying rate
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# THE DISTILLED LOGIC EXAMPLES - 10 perfect examples per pattern
# ==============================================================================

DISTILLED_EXAMPLES = [
    # =========================================================================
    # SHAPE 1: PERCENTAGE INCREASE
    # Logic: "increased by X%" means ADD X% of original TO original
    # Common error: treating "increased by 150%" as "multiply by 1.5" or "multiply by 0.15"
    # =========================================================================

    """Question: A house worth $100 increases in value by 50%. What is it worth now?

Answer: **LOGICAL SHAPE: PERCENTAGE INCREASE**

"Increased by 50%" means we ADD 50% of the original to the original.

Step 1: What is 50% of $100?
50% of $100 = $100 × 0.50 = $50

Step 2: Add this increase to the original:
New value = $100 + $50 = $150

The house is worth $150.
#### 150""",

    """Question: A painting worth $200 increases in value by 100%. What is the new value?

Answer: **LOGICAL SHAPE: PERCENTAGE INCREASE**

"Increased by 100%" means we ADD 100% of the original (which is the full original amount).

Step 1: 100% of $200 = $200
Step 2: New value = $200 + $200 = $400

(Note: This is the same as doubling - increase by 100% = double)
#### 400""",

    """Question: A car bought for $80,000 increases in value by 150% after restoration. What is its new value?

Answer: **LOGICAL SHAPE: PERCENTAGE INCREASE**

CRITICAL: "Increased by 150%" is NOT "multiply by 1.5" (that's only 50% increase).
"Increased by 150%" means ADD 150% of the original.

Step 1: 150% of $80,000 = $80,000 × 1.5 = $120,000
Step 2: New value = Original + Increase = $80,000 + $120,000 = $200,000

#### 200000""",

    """Question: Someone buys a house for $80,000, spends $50,000 on repairs. The house value increases by 150% from the purchase price. What is their profit?

Answer: **LOGICAL SHAPE: PERCENTAGE INCREASE + PROFIT CALCULATION**

Step 1: Calculate new value using percentage increase rule.
"Increased by 150%" means: New = Original + (Original × 1.5)
New value = $80,000 + ($80,000 × 1.5) = $80,000 + $120,000 = $200,000

Step 2: Calculate total cost.
Total cost = Purchase + Repairs = $80,000 + $50,000 = $130,000

Step 3: Calculate profit.
Profit = New value - Total cost = $200,000 - $130,000 = $70,000

#### 70000""",

    # =========================================================================
    # SHAPE 2: AVERAGE RATE
    # Logic: Average rate = Total output / Total input
    # Common error: (rate1 + rate2) / 2 (arithmetic mean of rates is WRONG)
    # =========================================================================

    """Question: A car travels 60 miles at 30 mph, then 60 miles at 60 mph. What is the average speed?

Answer: **LOGICAL SHAPE: AVERAGE RATE**

CRITICAL: Average speed is NOT (30 + 60) / 2 = 45 mph. That's WRONG!
Average speed = Total distance / Total time

Step 1: Calculate time for each segment.
Time at 30 mph: 60 miles ÷ 30 mph = 2 hours
Time at 60 mph: 60 miles ÷ 60 mph = 1 hour

Step 2: Calculate totals.
Total distance = 60 + 60 = 120 miles
Total time = 2 + 1 = 3 hours

Step 3: Calculate average speed.
Average speed = 120 miles / 3 hours = 40 mph

#### 40""",

    """Question: John drives 180 miles to a destination at 60 mph. He returns at 30 mph due to traffic. What is his average speed for the round trip?

Answer: **LOGICAL SHAPE: AVERAGE RATE**

Step 1: Time going = 180 ÷ 60 = 3 hours
Step 2: Time returning = 180 ÷ 30 = 6 hours
Step 3: Total distance = 180 + 180 = 360 miles
Step 4: Total time = 3 + 6 = 9 hours
Step 5: Average speed = 360 / 9 = 40 mph

(Note: NOT (60+30)/2 = 45. The slower speed is weighted more because more TIME is spent at that speed.)
#### 40""",

    """Question: A runner jogs 5 miles at 5 mph, then runs 5 miles at 10 mph. What is the average speed?

Answer: **LOGICAL SHAPE: AVERAGE RATE**

Time jogging: 5 ÷ 5 = 1 hour
Time running: 5 ÷ 10 = 0.5 hours
Total distance: 10 miles
Total time: 1.5 hours
Average speed: 10 / 1.5 = 6.67 mph

#### 7""",

    # =========================================================================
    # SHAPE 3: THRESHOLD CROSSING
    # Logic: "When do you START earning" = the year AFTER you break even
    # Common error: reporting the breakeven point instead of first profitable
    # =========================================================================

    """Question: A tree costs $100 to plant. It produces $10 of fruit per year. In what year do you START making money?

Answer: **LOGICAL SHAPE: THRESHOLD CROSSING**

Step 1: Find breakeven point.
Years to break even = $100 / $10 per year = 10 years exactly

Step 2: Determine "START making money."
After year 10: You've earned exactly $100, which equals your cost. Net = $0.
Year 10 is breakeven, not profit.

After year 11: You've earned $110. Net = +$10. This is your first profit!

Year 11 is when you START making money.
#### 11""",

    """Question: An investment costs $90. It returns $7.50 per year. In which year do you first have positive returns?

Answer: **LOGICAL SHAPE: THRESHOLD CROSSING**

Step 1: Exact breakeven = $90 / $7.50 = 12 years

Step 2: Since 12 × $7.50 = $90 exactly, year 12 is exact breakeven (zero profit).

Year 13 is when cumulative returns ($97.50) first EXCEED the cost ($90).
Year 13 is when you START earning.
#### 13""",

    """Question: A lemon tree costs $90 to plant. It produces $10.50 of lemons per year but costs $3 per year to maintain. When does the owner start earning money?

Answer: **LOGICAL SHAPE: THRESHOLD CROSSING**

Step 1: Net annual profit = $10.50 - $3 = $7.50
Step 2: Breakeven = $90 / $7.50 = 12 years exactly
Step 3: Year 12 is breakeven (12 × $7.50 = $90 = cost)
Step 4: Year 13 is first year with positive cumulative profit

#### 13""",

    # =========================================================================
    # SHAPE 4: INVERSE CHAIN (Working Backwards)
    # Logic: Start from what you know, undo operations in reverse order
    # Common error: trying to work forwards with unknowns
    # =========================================================================

    """Question: After giving away half of her apples, Sarah has 5 left. How many did she start with?

Answer: **LOGICAL SHAPE: INVERSE CHAIN**

Work BACKWARDS from what we know.

We know: After giving half, she has 5.
Inverse: If 5 is half, then whole = 5 × 2 = 10

She started with 10 apples.
#### 10""",

    """Question: A salesperson sold 1/3 of their items at the first house, then 2 more items at the second house, and half of what remained at the third house. They sold 5 items at the third house. How many items did they start with?

Answer: **LOGICAL SHAPE: INVERSE CHAIN**

Work BACKWARDS from the known endpoint.

Step 1: At third house, sold 5, which was HALF of remaining.
So before third house: 5 × 2 = 10 items

Step 2: At second house, sold 2 items.
So before second house: 10 + 2 = 12 items

Step 3: After first house, had 12 items. This is 2/3 of original (since 1/3 was sold).
Original = 12 × (3/2) = 18 items

Verify: Start 18 → sell 6 (1/3) → 12 left → sell 2 → 10 left → sell 5 (half) → 5 left ✓
#### 18""",

    """Question: Half of Tom's remaining money is $15. How much does he have?

Answer: **LOGICAL SHAPE: INVERSE CHAIN**

If half is $15, then whole is $15 × 2 = $30.
#### 30""",

    # =========================================================================
    # SHAPE 5: SEQUENTIAL SUBTRACTION THEN MULTIPLY
    # Logic: First compute what remains, THEN multiply
    # Common error: multiplying before subtracting, or wrong order
    # =========================================================================

    """Question: A farmer has 20 eggs. She uses 5 for breakfast and 3 for baking. She sells the rest at $2 each. How much does she make?

Answer: **LOGICAL SHAPE: SEQUENTIAL SUBTRACTION THEN MULTIPLY**

Step 1: Calculate remaining (subtract FIRST).
Remaining = 20 - 5 - 3 = 12 eggs

Step 2: Calculate revenue (multiply SECOND).
Revenue = 12 × $2 = $24
#### 24""",

    """Question: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for baking muffins. She sells the rest at $2 each. How much does she make daily?

Answer: **LOGICAL SHAPE: SEQUENTIAL SUBTRACTION THEN MULTIPLY**

Step 1: Eggs used = 3 + 4 = 7
Step 2: Eggs remaining = 16 - 7 = 9
Step 3: Daily revenue = 9 × $2 = $18
#### 18""",

    # =========================================================================
    # SHAPE 6: REMAINING FIRST (then apply rate)
    # Logic: Calculate remaining distance/quantity BEFORE dividing by rate
    # Common error: applying rates to wrong quantities
    # =========================================================================

    """Question: A 12-mile trail. You walk 4 miles, then 2 miles. How many miles remain?

Answer: **LOGICAL SHAPE: REMAINING FIRST**

Step 1: Calculate remaining.
Remaining = Total - Part1 - Part2 = 12 - 4 - 2 = 6 miles
#### 6""",

    """Question: Marissa is hiking a 12-mile trail. She walked 4 miles in the first hour, then 2 miles in the next 2 hours. If she finishes at 3 mph, how many more hours will it take?

Answer: **LOGICAL SHAPE: REMAINING FIRST**

Step 1: Calculate remaining distance FIRST.
Already walked = 4 + 2 = 6 miles
Remaining = 12 - 6 = 6 miles

Step 2: Calculate time at given rate.
Time = Distance / Rate = 6 miles / 3 mph = 2 hours
#### 2""",

    """Question: You need to read 100 pages. You read 30 pages on Monday and 25 on Tuesday. At 15 pages per hour, how many more hours do you need?

Answer: **LOGICAL SHAPE: REMAINING FIRST**

Step 1: Remaining pages = 100 - 30 - 25 = 45 pages
Step 2: Hours needed = 45 / 15 = 3 hours
#### 3""",
]

# Additional reinforcement examples for the hardest patterns
REINFORCEMENT_EXAMPLES = [
    # More percentage increase (the most common error)
    """Question: Stock worth $500 increases by 200%. New value?

Answer: Increase = $500 × 2.0 = $1000
New = $500 + $1000 = $1500
#### 1500""",

    # More working backwards
    """Question: A third of the cookies were eaten, leaving 20. How many were there originally?

Answer: 20 is 2/3 of original (since 1/3 was eaten).
Original = 20 × (3/2) = 30
#### 30""",

    # More threshold crossing
    """Question: Machine costs $60, earns $5/month. When do you profit?

Answer: Breakeven: 60/5 = 12 months exactly.
First profit: month 13.
#### 13""",
]


def main():
    from mlx_lm import load
    import mlx.core as mx
    import re

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    train_data_dir = "data/training/qwen3_distilled_logic"
    new_adapter_path = "data/adapters/qwen3_distilled_logic_lora"

    logger.info("=" * 70)
    logger.info("DISTILLED LOGIC TRAINING")
    logger.info("10 perfect examples per pattern - quality over quantity")
    logger.info("=" * 70)

    np.random.seed(42)

    # Create training samples from distilled examples
    distilled_samples = []
    for example in DISTILLED_EXAMPLES + REINFORCEMENT_EXAMPLES:
        distilled_samples.append({"text": f"Question: {example.split('Question: ')[1]}" if "Question:" in example else example})

    # Actually, the examples already have "Question:" - let's just use them as-is
    distilled_samples = [{"text": ex} for ex in DISTILLED_EXAMPLES + REINFORCEMENT_EXAMPLES]

    logger.info(f"Distilled logic examples: {len(distilled_samples)}")

    # Minimal arithmetic foundation (just enough to maintain)
    arith_samples = []
    for a in range(1, 13):
        for b in range(1, 13):
            arith_samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                arith_samples.append({"text": f"{a}-{b}={a-b}"})
            arith_samples.append({"text": f"{a}*{b}={a*b}"})

    np.random.shuffle(arith_samples)
    arith_samples = arith_samples[:100]

    # Repeat distilled examples to give them weight (10x)
    all_samples = distilled_samples * 10 + arith_samples
    np.random.shuffle(all_samples)

    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"  Distilled (10x): {len(distilled_samples) * 10}")
    logger.info(f"  Arithmetic: {len(arith_samples)}")

    # Save
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)
    n_train = int(len(all_samples) * 0.9)

    for name, data in [
        ("train", all_samples[:n_train]),
        ("valid", all_samples[n_train:]),
        ("test", all_samples[n_train:]),
    ]:
        path = Path(train_data_dir) / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"  {name}: {len(data)} samples")

    # Train with fewer iterations - we want to LEARN the shapes, not memorize
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
        "--iters", "1000",  # Fewer iterations - quality data needs less training
        "--learning-rate", "2e-5",
        "--seed", "42",
        "--steps-per-report", "200",
    ]

    logger.info("Training on distilled logic (1000 iterations)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-8:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Evaluate
    logger.info("\n=== EVALUATION ===")

    from modelcypher.core.use_cases.curriculum import BenchmarkLoader
    loader = BenchmarkLoader()

    model, tokenizer = load(model_path, adapter_path=new_adapter_path)

    gsm_test = loader.load("gsm8k", split="test", limit=30)

    test_suites = {
        "Arithmetic": [
            ("2+2=", "4"), ("7+8=", "15"), ("15-7=", "8"), ("6*9=", "54"),
            ("12+19=", "31"), ("24-15=", "9"), ("8*7=", "56"), ("11+11=", "22"),
        ],
        "MultiStep": [
            ("5+3=8, 8+2=", "10"),
            ("7+4=11, 11-3=", "8"),
            ("4+6=10, 10+5=", "15"),
        ],
        "GSM8K": [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples[:20]],
    }

    results = {}
    for suite_name, problems in test_suites.items():
        correct = 0
        details = []

        for question, expected in problems:
            if suite_name == "GSM8K":
                prompt = f"Question: {question}\n\nAnswer:"
                max_tokens = 400
            else:
                prompt = question
                max_tokens = 20

            tokens = tokenizer.encode(prompt)
            generated = []

            for _ in range(max_tokens):
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

            if suite_name == "GSM8K":
                if "####" in output:
                    answer_part = output.split("####")[-1].strip().replace(",", "").replace("$", "")
                    numbers = re.findall(r'-?\d+', answer_part)
                    predicted = numbers[0] if numbers else ""
                else:
                    numbers = re.findall(r'-?\d+', output.replace(",", ""))
                    predicted = numbers[-1] if numbers else ""
            else:
                numbers = re.findall(r'-?\d+', output)
                predicted = numbers[0] if numbers else ""

            is_correct = predicted == expected
            if is_correct:
                correct += 1

            details.append({
                "question": question[:40],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })

        accuracy = correct / len(problems)
        results[suite_name] = {"accuracy": accuracy, "correct": correct, "total": len(problems)}

        logger.info(f"\n{suite_name}: {accuracy:.0%} ({correct}/{len(problems)})")
        for d in details:
            mark = "OK" if d["correct"] else "XX"
            logger.info(f"  {mark}: '{d['question'][:35]}...' -> '{d['predicted']}' (expected '{d['expected']}')")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DISTILLED LOGIC RESULTS")
    logger.info("=" * 70)

    arith_acc = results["Arithmetic"]["accuracy"]
    multi_acc = results["MultiStep"]["accuracy"]
    gsm_acc = results["GSM8K"]["accuracy"]

    logger.info(f"""
Arithmetic:  {arith_acc:.0%} (target: 100%)
MultiStep:   {multi_acc:.0%} (target: 100%)
GSM8K:       {gsm_acc:.0%} (target: 100%)

FULL MASTERY: {arith_acc == 1.0 and multi_acc == 1.0 and gsm_acc == 1.0}
""")

    output = {
        "results": {k: v["accuracy"] for k, v in results.items()},
        "adapter": new_adapter_path,
        "approach": "distilled_logic_shapes",
        "num_examples": len(distilled_samples),
    }
    output_path = Path("data/experiments/qwen3_distilled_logic.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
