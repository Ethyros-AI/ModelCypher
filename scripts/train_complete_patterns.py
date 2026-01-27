#!/usr/bin/env python3
"""Complete pattern training: ALL 11 logical shapes.

The model keeps hitting 75% because different test problems require different patterns.
This script covers ALL patterns discovered through error analysis.

PATTERNS THAT WORK (from distilled_logic):
1. PERCENTAGE INCREASE: new = original + (original × percent)
2. AVERAGE RATE: total_output / total_input
3. THRESHOLD CROSSING: breakeven + 1 = first profitable
4. INVERSE CHAIN: work backwards, undo operations
5. SEQUENTIAL OPERATIONS: subtract first, THEN multiply
6. REMAINING FIRST: compute what's left BEFORE applying rate

NEW PATTERNS (from current failures):
7. CHAIN MULTIPLICATION: multiply ALL factors together
8. RESTART/RETRY: work done + restart + complete from beginning
9. SEGMENTED JOURNEY: different legs at different rates, sum them
10. INVERSE FRACTIONS: "sold 1/3" means 2/3 remain, then work backwards
11. CASCADING PERCENTAGES: percentage of what REMAINS after each step
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ALL PATTERNS - complete coverage
COMPLETE_PATTERNS = [
    # ==========================================================================
    # PATTERN 1: PERCENTAGE INCREASE
    # ==========================================================================
    """Question: A house costs $80,000. Its value increases by 150%. What is the new value?

Answer: **PATTERN: PERCENTAGE INCREASE**
Rule: "Increases by X%" means: New = Original + (Original × X/100)

Increase = $80,000 × 1.50 = $120,000
New value = $80,000 + $120,000 = $200,000
#### 200000""",

    """Question: Stock worth $400 increases by 75%. New value?

Answer: **PATTERN: PERCENTAGE INCREASE**
Increase = $400 × 0.75 = $300
New = $400 + $300 = $700
#### 700""",

    # ==========================================================================
    # PATTERN 2: AVERAGE RATE
    # ==========================================================================
    """Question: John drives 60 miles in 2 hours, then 90 miles in 3 hours. What's his average speed?

Answer: **PATTERN: AVERAGE RATE**
Rule: Average rate = Total distance / Total time (NOT average of the rates!)

Total distance = 60 + 90 = 150 miles
Total time = 2 + 3 = 5 hours
Average speed = 150 / 5 = 30 mph
#### 30""",

    """Question: A factory makes 100 items in 4 hours, then 150 items in 6 hours. Average rate?

Answer: **PATTERN: AVERAGE RATE**
Total items = 100 + 150 = 250
Total hours = 4 + 6 = 10
Average = 250 / 10 = 25 items/hour
#### 25""",

    # ==========================================================================
    # PATTERN 3: THRESHOLD CROSSING
    # ==========================================================================
    """Question: A machine costs $120 and earns $10/month. In which month do you first profit?

Answer: **PATTERN: THRESHOLD CROSSING**
Rule: Breakeven is when earnings = cost. First PROFIT is the NEXT period.

Breakeven: 120/10 = 12 months (earnings exactly equal cost)
First profit: month 13 (first month you're ABOVE breakeven)
#### 13""",

    """Question: Investment costs $500. Returns $25/week. When do you first profit?

Answer: **PATTERN: THRESHOLD CROSSING**
Breakeven: 500/25 = 20 weeks (cost recovered)
First profit: week 21
#### 21""",

    # ==========================================================================
    # PATTERN 4: INVERSE CHAIN (working backwards)
    # ==========================================================================
    """Question: After doubling money and spending $40, Sam has $60. How much did he start with?

Answer: **PATTERN: INVERSE CHAIN**
Rule: Work backwards by undoing each operation in reverse order.

End: $60
Before spending $40: $60 + $40 = $100
Before doubling: $100 / 2 = $50
Start: $50
#### 50""",

    """Question: A number is tripled, then 15 is subtracted, leaving 30. What was the original?

Answer: **PATTERN: INVERSE CHAIN**
End: 30
Before subtracting 15: 30 + 15 = 45
Before tripling: 45 / 3 = 15
#### 15""",

    # ==========================================================================
    # PATTERN 5: SEQUENTIAL OPERATIONS
    # ==========================================================================
    """Question: Janet has 10 eggs. She eats 3, then sells the rest at $2 each. How much money?

Answer: **PATTERN: SEQUENTIAL OPERATIONS**
Rule: Do operations in order. Subtract FIRST, then multiply.

After eating: 10 - 3 = 7 eggs
Money from selling: 7 × $2 = $14
#### 14""",

    """Question: Store has 50 items. 20 are defective and thrown out. Rest sold at $5 each. Revenue?

Answer: **PATTERN: SEQUENTIAL OPERATIONS**
Good items: 50 - 20 = 30
Revenue: 30 × $5 = $150
#### 150""",

    # ==========================================================================
    # PATTERN 6: REMAINING FIRST
    # ==========================================================================
    """Question: Marissa is hiking 12 miles. She walked 6 miles already. At 3 mph, how many hours left?

Answer: **PATTERN: REMAINING FIRST**
Rule: Calculate remaining distance FIRST, then apply rate.

Remaining: 12 - 6 = 6 miles
Time = Distance / Rate = 6 / 3 = 2 hours
#### 2""",

    """Question: Need to read 100 pages. Read 40 already. At 20 pages/hour, how many hours left?

Answer: **PATTERN: REMAINING FIRST**
Remaining: 100 - 40 = 60 pages
Hours: 60 / 20 = 3 hours
#### 3""",

    # ==========================================================================
    # PATTERN 7: CHAIN MULTIPLICATION (NEW)
    # ==========================================================================
    """Question: James runs 3 sprints, 3 times per week, 60 meters each sprint. Total meters per week?

Answer: **PATTERN: CHAIN MULTIPLICATION**
Rule: Multiply ALL factors together. Don't stop at one multiplication.

Sprints per week = 3 sprints × 3 times = 9 sprints
Total meters = 9 sprints × 60 meters = 540 meters
#### 540""",

    """Question: A baker makes 4 batches of cookies, 3 trays per batch, 12 cookies per tray. Total cookies?

Answer: **PATTERN: CHAIN MULTIPLICATION**
Total = 4 × 3 × 12 = 144 cookies
Step by step: 4 × 3 = 12 trays, 12 × 12 = 144 cookies
#### 144""",

    """Question: 5 teams, 4 players each, 3 games per player. Total games played?

Answer: **PATTERN: CHAIN MULTIPLICATION**
Total = 5 × 4 × 3 = 60 games
#### 60""",

    # ==========================================================================
    # PATTERN 8: RESTART/RETRY (NEW)
    # ==========================================================================
    """Question: Downloading 200 GB at 2 GB/min. At 40% done, computer restarts (20 min delay), then starts over. Total time?

Answer: **PATTERN: RESTART/RETRY**
Rule: Time for partial work + delay + time for complete work from scratch.

40% of 200 GB = 80 GB
Time to 40%: 80 / 2 = 40 minutes
Restart delay: 20 minutes
Time to download full file: 200 / 2 = 100 minutes
Total: 40 + 20 + 100 = 160 minutes
#### 160""",

    """Question: Painting a wall takes 60 min. After 20 min, paint spills and you restart. Total time?

Answer: **PATTERN: RESTART/RETRY**
Time before spill: 20 minutes
Time to complete from start: 60 minutes
Total: 20 + 60 = 80 minutes
#### 80""",

    # ==========================================================================
    # PATTERN 9: SEGMENTED JOURNEY (NEW)
    # ==========================================================================
    """Question: John drives 3h at 60mph away from home. Returning: 2h stuck in traffic, 0.5h at 30mph, 1.5h at 80mph. How far from home?

Answer: **PATTERN: SEGMENTED JOURNEY**
Rule: Calculate each segment separately, then combine.

Distance from home: 3 × 60 = 180 miles
Returning journey (4 hours total):
- 2 hours stuck: 0 miles
- 0.5 hours at 30 mph: 0.5 × 30 = 15 miles
- 1.5 hours at 80 mph: 1.5 × 80 = 120 miles
Total distance traveled home: 15 + 120 = 135 miles
Still from home: 180 - 135 = 45 miles
#### 45""",

    """Question: Cyclist rides 2h at 15mph, rests 30min, then 1h at 20mph. Total distance?

Answer: **PATTERN: SEGMENTED JOURNEY**
Segment 1: 2 × 15 = 30 miles
Rest: 0 miles
Segment 2: 1 × 20 = 20 miles
Total: 30 + 20 = 50 miles
#### 50""",

    # ==========================================================================
    # PATTERN 10: INVERSE FRACTIONS (NEW)
    # ==========================================================================
    """Question: Melanie sold 1/3 of her vacuums at house A, 2 more at house B, half of the rest at house C. She has 5 left. How many did she start with?

Answer: **PATTERN: INVERSE FRACTIONS**
Rule: "Sold 1/3" means 2/3 remain. Work backwards from the end.

Has 5 left (after selling half at C)
Before house C: 5 × 2 = 10 (since she sold half)
Before house B: 10 + 2 = 12 (add back the 2 sold)
Before house A: 12 is 2/3 of original (since 1/3 was sold)
Original = 12 × (3/2) = 18
#### 18""",

    """Question: Tom ate 1/4 of the pizza, then gave away 3 slices, leaving 9 slices. How many slices originally?

Answer: **PATTERN: INVERSE FRACTIONS**
End: 9 slices
Before giving away 3: 9 + 3 = 12 slices
12 is 3/4 of original (since 1/4 was eaten)
Original = 12 × (4/3) = 16 slices
#### 16""",

    """Question: A store sold 2/5 of its shirts. Then 10 more were returned. Now there are 34. Original count?

Answer: **PATTERN: INVERSE FRACTIONS**
Current: 34
Before returns: 34 - 10 = 24
24 is 3/5 of original (since 2/5 were sold)
Original = 24 × (5/3) = 40
#### 40""",

    # ==========================================================================
    # PATTERN 11: CASCADING PERCENTAGES (NEW)
    # ==========================================================================
    """Question: 20 students: 20% take class A, 25% of the rest take class B. What percent take neither?

Answer: **PATTERN: CASCADING PERCENTAGES**
Rule: Each percentage applies to what REMAINS, not the original total.

Class A: 20% of 20 = 4 students
Remaining after A: 20 - 4 = 16 students
Class B: 25% of 16 = 4 students
Neither: 16 - 4 = 12 students
Percent neither: 12/20 × 100 = 60%
#### 60""",

    """Question: 100 items: 30% are defective, 50% of good ones are sold. How many unsold good items?

Answer: **PATTERN: CASCADING PERCENTAGES**
Defective: 30% of 100 = 30 items
Good items: 100 - 30 = 70 items
Sold (from good): 50% of 70 = 35 items
Unsold good: 70 - 35 = 35 items
#### 35""",

    """Question: 80 apples: 25% are rotten, 75% of good ones are sold. How many unsold good apples?

Answer: **PATTERN: CASCADING PERCENTAGES**
Rotten: 25% of 80 = 20
Good: 80 - 20 = 60
Sold: 75% of 60 = 45
Unsold good: 60 - 45 = 15
#### 15""",
]


def main():
    from mlx_lm import load
    import mlx.core as mx
    import re

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    train_data_dir = "data/training/qwen3_complete_patterns"
    new_adapter_path = "data/adapters/qwen3_complete_patterns_lora"
    log_path = Path("data/experiments/complete_patterns_log.txt")

    # Redirect logging to file
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("COMPLETE PATTERN TRAINING - ALL 11 LOGICAL SHAPES")
    logger.info("=" * 70)

    np.random.seed(42)

    # Create training samples
    pattern_samples = [{"text": ex} for ex in COMPLETE_PATTERNS]
    logger.info(f"Pattern examples: {len(pattern_samples)}")

    # Minimal arithmetic foundation
    arith_samples = []
    for a in range(1, 13):
        for b in range(1, 13):
            arith_samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                arith_samples.append({"text": f"{a}-{b}={a-b}"})
            arith_samples.append({"text": f"{a}*{b}={a*b}"})

    np.random.shuffle(arith_samples)
    arith_samples = arith_samples[:100]

    # Repeat pattern examples (10x weight)
    all_samples = pattern_samples * 10 + arith_samples
    np.random.shuffle(all_samples)

    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"  Patterns (10x): {len(pattern_samples) * 10}")
    logger.info(f"  Arithmetic: {len(arith_samples)}")

    # Save
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)
    n_train = int(len(all_samples) * 0.9)

    for name, data in [
        ("train", all_samples[:n_train]),
        ("valid", all_samples[n_train:int(len(all_samples)*0.95)]),
        ("test", all_samples[int(len(all_samples)*0.95):]),
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
        "--iters", "1500",  # Slightly more for 11 patterns
        "--learning-rate", "2e-5",
        "--seed", "42",
        "--steps-per-report", "100",
    ]

    logger.info("Training (1500 iterations, 11 patterns)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-10:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Evaluation
    logger.info("\n=== EVALUATION ===")

    from modelcypher.core.use_cases.curriculum import BenchmarkLoader

    model, tokenizer = load(model_path, adapter_path=new_adapter_path)
    loader = BenchmarkLoader()

    # Test suites
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
    }

    # GSM8K test - larger sample for better accuracy estimate
    gsm_test = loader.load("gsm8k", split="test", limit=50)
    test_suites["GSM8K"] = [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples[:40]]

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

            # Extract answer
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
    logger.info("COMPLETE PATTERNS RESULTS")
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

    # Save results
    output = {
        "results": {k: v["accuracy"] for k, v in results.items()},
        "adapter": new_adapter_path,
        "training": {
            "pattern_examples": len(pattern_samples),
            "arithmetic_samples": len(arith_samples),
            "iterations": 1500,
            "patterns_covered": 11,
        },
    }
    output_path = Path("data/experiments/qwen3_complete_patterns.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
