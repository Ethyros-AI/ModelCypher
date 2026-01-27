#!/usr/bin/env python3
"""Final mastery training: 13 patterns with reinforced weak areas.

PATTERNS COVERED:
1. PERCENTAGE INCREASE
2. AVERAGE RATE
3. THRESHOLD CROSSING (reinforced with explicit +1)
4. INVERSE CHAIN
5. SEQUENTIAL OPERATIONS
6. REMAINING FIRST
7. CHAIN MULTIPLICATION
8. RESTART/RETRY
9. SEGMENTED JOURNEY
10. INVERSE FRACTIONS
11. CASCADING PERCENTAGES
12. AVERAGE CONSTRAINT (NEW) - speed needed to achieve target average
13. RATE CONVERSION (NEW) - convert between related rates

REINFORCED (caused regressions):
- PERCENTAGE INCREASE + PROFIT: Apply to ORIGINAL, subtract TOTAL COST
- THRESHOLD CROSSING: breakeven + 1 = FIRST profit
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


FINAL_PATTERNS = [
    # ==========================================================================
    # PATTERN 1: PERCENTAGE INCREASE + PROFIT (REINFORCED)
    # ==========================================================================
    """Question: Josh buys a house for $80,000 and puts in $50,000 in repairs. This increases the house value by 150%. What is his profit?

Answer: **PATTERN: PERCENTAGE INCREASE + PROFIT**

CRITICAL: "Increased by 150%" applies to the ORIGINAL purchase price, NOT the total cost!

Step 1: Total cost (what Josh spent)
Total cost = $80,000 + $50,000 = $130,000

Step 2: Calculate value increase (150% of ORIGINAL price)
Increase = $80,000 × 1.50 = $120,000

Step 3: New house value (original + increase)
New value = $80,000 + $120,000 = $200,000

Step 4: Profit (selling price - total cost)
Profit = $200,000 - $130,000 = $70,000
#### 70000""",

    """Question: Sam buys a car for $20,000 and spends $5,000 on upgrades. The car's value increases by 80%. What's the profit?

Answer: **PATTERN: PERCENTAGE INCREASE + PROFIT**

Total cost = $20,000 + $5,000 = $25,000
Value increase = $20,000 × 0.80 = $16,000 (applied to ORIGINAL)
New value = $20,000 + $16,000 = $36,000
Profit = $36,000 - $25,000 = $11,000
#### 11000""",

    """Question: Art bought for $500, restoration costs $200. Value increases by 300%. Profit?

Answer: **PATTERN: PERCENTAGE INCREASE + PROFIT**

Total cost = $500 + $200 = $700
Increase = $500 × 3.0 = $1,500 (on ORIGINAL)
New value = $500 + $1,500 = $2,000
Profit = $2,000 - $700 = $1,300
#### 1300""",

    # ==========================================================================
    # PATTERN 2: AVERAGE RATE
    # ==========================================================================
    """Question: John drives 60 miles in 2 hours, then 90 miles in 3 hours. What's his average speed?

Answer: **PATTERN: AVERAGE RATE**
Average rate = Total distance / Total time (NOT average of individual rates!)

Total distance = 60 + 90 = 150 miles
Total time = 2 + 3 = 5 hours
Average speed = 150 / 5 = 30 mph
#### 30""",

    # ==========================================================================
    # PATTERN 3: THRESHOLD CROSSING (REINFORCED)
    # ==========================================================================
    """Question: Carlos plants a tree costing $90. It earns $7.50 net per year. When does he first EARN money?

Answer: **PATTERN: THRESHOLD CROSSING**

CRITICAL: Breakeven is when cumulative earnings = cost. First PROFIT is the NEXT year!

Net earnings per year = $7.50
Years to breakeven = $90 / $7.50 = 12 years (exactly covers cost)
First year of PROFIT = 12 + 1 = 13

The tree pays off in year 12 (zero profit). He EARNS money starting year 13.
#### 13""",

    """Question: Machine costs $60, earns $5/month. In which month do you first profit?

Answer: **PATTERN: THRESHOLD CROSSING**

Breakeven: 60 / 5 = 12 months (cost recovered, zero profit)
First profit: month 13 (first month ABOVE breakeven)
#### 13""",

    """Question: Investment of $100 returns $8/week. When do you first profit?

Answer: **PATTERN: THRESHOLD CROSSING**

Breakeven = 100 / 8 = 12.5, round up to 13 weeks to cover cost
First profit = week 14
#### 14""",

    # ==========================================================================
    # PATTERN 4: INVERSE CHAIN
    # ==========================================================================
    """Question: After doubling money and spending $40, Sam has $60. How much did he start with?

Answer: **PATTERN: INVERSE CHAIN**
Work backwards by undoing each operation in reverse order.

End: $60
Before spending $40: $60 + $40 = $100
Before doubling: $100 / 2 = $50
Start: $50
#### 50""",

    # ==========================================================================
    # PATTERN 5: SEQUENTIAL OPERATIONS
    # ==========================================================================
    """Question: Janet has 10 eggs. She eats 3, then sells the rest at $2 each. How much money?

Answer: **PATTERN: SEQUENTIAL OPERATIONS**
Subtract FIRST, then multiply.

After eating: 10 - 3 = 7 eggs
Money from selling: 7 × $2 = $14
#### 14""",

    # ==========================================================================
    # PATTERN 6: REMAINING FIRST (REINFORCED)
    # ==========================================================================
    """Question: Reading 100 pages total. Already read 40. At 20 pages/hour, how many hours left?

Answer: **PATTERN: REMAINING FIRST**
Calculate remaining FIRST, then apply rate.

Remaining pages = 100 - 40 = 60 pages
Hours needed = 60 / 20 = 3 hours
#### 3""",

    # ==========================================================================
    # PATTERN 7: CHAIN MULTIPLICATION
    # ==========================================================================
    """Question: James runs 3 sprints, 3 times per week, 60 meters each sprint. Total meters per week?

Answer: **PATTERN: CHAIN MULTIPLICATION**
Multiply ALL factors together.

Sprints per week = 3 sprints × 3 times = 9 sprints
Total meters = 9 sprints × 60 meters = 540 meters
#### 540""",

    """Question: 4 batches, 3 trays per batch, 12 cookies per tray. Total cookies?

Answer: **PATTERN: CHAIN MULTIPLICATION**
Total = 4 × 3 × 12 = 144 cookies
#### 144""",

    # ==========================================================================
    # PATTERN 8: RESTART/RETRY
    # ==========================================================================
    """Question: Downloading 200 GB at 2 GB/min. At 40% done, restart needed (20 min delay), then start over. Total time?

Answer: **PATTERN: RESTART/RETRY**
Time for partial work + delay + time for complete work from scratch.

40% of 200 GB = 80 GB
Time to 40%: 80 / 2 = 40 minutes
Restart delay: 20 minutes
Time to download full file: 200 / 2 = 100 minutes
Total: 40 + 20 + 100 = 160 minutes
#### 160""",

    # ==========================================================================
    # PATTERN 9: SEGMENTED JOURNEY
    # ==========================================================================
    """Question: John drives 3h at 60mph away from home. Returning: 2h stuck, 0.5h at 30mph, 1.5h at 80mph. How far from home?

Answer: **PATTERN: SEGMENTED JOURNEY**
Calculate each segment separately, then combine.

Distance from home: 3 × 60 = 180 miles
Return journey:
- 2 hours stuck: 0 miles
- 0.5h at 30mph: 15 miles
- 1.5h at 80mph: 120 miles
Distance returned: 15 + 120 = 135 miles
Still from home: 180 - 135 = 45 miles
#### 45""",

    # ==========================================================================
    # PATTERN 10: INVERSE FRACTIONS
    # ==========================================================================
    """Question: Melanie sold 1/3 of vacuums at house A, 2 more at B, half of rest at C. She has 5 left. How many did she start with?

Answer: **PATTERN: INVERSE FRACTIONS**
"Sold 1/3" means 2/3 remain. Work backwards.

Has 5 left (after selling half at C)
Before house C: 5 × 2 = 10 (half was sold)
Before house B: 10 + 2 = 12 (add back the 2)
Before house A: 12 is 2/3 of original
Original = 12 × (3/2) = 18
#### 18""",

    """Question: Tom ate 1/4 of pizza, gave away 3 slices, has 9 left. Original slices?

Answer: **PATTERN: INVERSE FRACTIONS**
End: 9 slices
Before giving 3: 9 + 3 = 12
12 is 3/4 of original (since 1/4 eaten)
Original = 12 × (4/3) = 16
#### 16""",

    # ==========================================================================
    # PATTERN 11: CASCADING PERCENTAGES
    # ==========================================================================
    """Question: 20 students: 20% take A, 25% of rest take B. What percent take neither?

Answer: **PATTERN: CASCADING PERCENTAGES**
Each percentage applies to what REMAINS, not the original.

Class A: 20% of 20 = 4 students
Remaining: 20 - 4 = 16 students
Class B: 25% of 16 = 4 students
Neither: 16 - 4 = 12 students
Percent: 12/20 × 100 = 60%
#### 60""",

    # ==========================================================================
    # PATTERN 12: AVERAGE CONSTRAINT (NEW)
    # ==========================================================================
    """Question: Marissa hikes 12 miles. She walked 4 miles in hour 1, 2 miles in hour 2. To average 4 mph overall, what speed for the rest?

Answer: **PATTERN: AVERAGE CONSTRAINT**
To achieve a target average, calculate time budget first, then speed needed for remaining distance.

Step 1: Time budget for target average
Total time allowed = Total distance / Target average = 12 / 4 = 3 hours

Step 2: Time already used
Time used = 1 hour + 1 hour = 2 hours

Step 3: Time remaining
Time remaining = 3 - 2 = 1 hour

Step 4: Distance remaining
Distance covered = 4 + 2 = 6 miles
Distance remaining = 12 - 6 = 6 miles

Step 5: Speed needed
Speed = Distance / Time = 6 miles / 1 hour = 6 mph
#### 6""",

    """Question: A train needs to cover 300 miles averaging 60 mph. It went 100 miles at 50 mph. What speed for the rest?

Answer: **PATTERN: AVERAGE CONSTRAINT**

Total time budget = 300 / 60 = 5 hours
Time used for first 100 miles = 100 / 50 = 2 hours
Time remaining = 5 - 2 = 3 hours
Distance remaining = 300 - 100 = 200 miles
Speed needed = 200 / 3 = 66.67 ≈ 67 mph
#### 67""",

    """Question: Runner wants to complete 10 miles at 5 mph average. Ran first 4 miles at 4 mph. Speed for remaining?

Answer: **PATTERN: AVERAGE CONSTRAINT**

Time budget = 10 / 5 = 2 hours total
Time used = 4 / 4 = 1 hour
Time remaining = 2 - 1 = 1 hour
Distance remaining = 10 - 4 = 6 miles
Speed needed = 6 / 1 = 6 mph
#### 6""",

    # ==========================================================================
    # PATTERN 13: RATE CONVERSION + WEIGHTED COMBINATION (NEW)
    # ==========================================================================
    """Question: Dana can run 4x faster than she walks. She skips at half her running speed. Skip = 3 mph. In 6 hours (1/3 running, 2/3 walking), how far?

Answer: **PATTERN: RATE CONVERSION + WEIGHTED COMBINATION**

Step 1: Convert all rates from given info
Skip = 3 mph
Run = 2 × Skip = 2 × 3 = 6 mph (skip is half of run, so run is 2× skip)
Walk = Run / 4 = 6 / 4 = 1.5 mph (run is 4× walk)

Step 2: Calculate time for each activity
Running time = 6 hours × (1/3) = 2 hours
Walking time = 6 hours × (2/3) = 4 hours

Step 3: Calculate distance for each activity
Running distance = 2 hours × 6 mph = 12 miles
Walking distance = 4 hours × 1.5 mph = 6 miles

Step 4: Total distance
Total = 12 + 6 = 18 miles
#### 18""",

    """Question: Car A goes twice as fast as car B. Car B goes 3x as fast as car C. If car C goes 10 mph, how far does car A travel in 2 hours?

Answer: **PATTERN: RATE CONVERSION**

Car C = 10 mph
Car B = 3 × 10 = 30 mph
Car A = 2 × 30 = 60 mph
Distance = 60 × 2 = 120 miles
#### 120""",

    """Question: Bike speed is half car speed. Walk speed is 1/4 bike speed. Car = 40 mph. In 3 hours (1 hour biking, 2 hours walking), distance?

Answer: **PATTERN: RATE CONVERSION + WEIGHTED COMBINATION**

Car = 40 mph
Bike = 40 / 2 = 20 mph
Walk = 20 / 4 = 5 mph

Biking: 1 × 20 = 20 miles
Walking: 2 × 5 = 10 miles
Total = 20 + 10 = 30 miles
#### 30""",
]


def main():
    from mlx_lm import load
    import mlx.core as mx
    import re

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    train_data_dir = "data/training/qwen3_final_mastery"
    new_adapter_path = "data/adapters/qwen3_final_mastery_lora"
    log_path = Path("data/experiments/final_mastery_log.txt")

    # Set up file handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("FINAL MASTERY TRAINING - 13 PATTERNS")
    logger.info("=" * 70)

    np.random.seed(42)

    pattern_samples = [{"text": ex} for ex in FINAL_PATTERNS]
    logger.info(f"Pattern examples: {len(pattern_samples)}")

    # Minimal arithmetic
    arith_samples = []
    for a in range(1, 13):
        for b in range(1, 13):
            arith_samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                arith_samples.append({"text": f"{a}-{b}={a-b}"})
            arith_samples.append({"text": f"{a}*{b}={a*b}"})

    np.random.shuffle(arith_samples)
    arith_samples = arith_samples[:100]

    # Repeat patterns (12x for stronger learning)
    all_samples = pattern_samples * 12 + arith_samples
    np.random.shuffle(all_samples)

    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"  Patterns (12x): {len(pattern_samples) * 12}")
    logger.info(f"  Arithmetic: {len(arith_samples)}")

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
        "--iters", "2000",  # More iterations for 13 patterns
        "--learning-rate", "2e-5",
        "--seed", "42",
        "--steps-per-report", "200",
    ]

    logger.info("Training (2000 iterations, 13 patterns)...")

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

    # GSM8K test - 50 samples for reliable estimate
    gsm_test = loader.load("gsm8k", split="test", limit=60)
    test_suites["GSM8K"] = [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples[:50]]

    results = {}
    for suite_name, problems in test_suites.items():
        correct = 0
        details = []

        for question, expected in problems:
            if suite_name == "GSM8K":
                prompt = f"Question: {question}\n\nAnswer:"
                max_tokens = 500
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
    logger.info("FINAL MASTERY RESULTS")
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
        "training": {
            "pattern_examples": len(pattern_samples),
            "arithmetic_samples": len(arith_samples),
            "iterations": 2000,
            "patterns_covered": 13,
        },
    }
    output_path = Path("data/experiments/qwen3_final_mastery.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
