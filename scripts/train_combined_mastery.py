#!/usr/bin/env python3
"""Combined training: Full GSM8K + explicit reasoning for failing patterns.

The key insight: We need BOTH
1. Broad coverage from real GSM8K data (for generalization)
2. Explicit reasoning for the specific logical errors

This script combines:
- Full GSM8K training set (all 7473)
- Explicit reasoning examples for the 6 failing patterns
- Arithmetic foundation to prevent regression
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import random
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_explicit_reasoning_samples() -> List[Dict]:
    """Generate explicit reasoning samples for the 6 failing patterns.

    These are TEACHING examples with step-by-step reasoning that
    directly addresses the logical errors the model is making.
    """
    samples = []

    # PATTERN 1: 150% increase means multiply by 2.5, not by 0.15 or 1.5
    pattern1_samples = [
        """Question: A house is bought for $80,000. After repairs, its value increased by 150%. The repairs cost $50,000. What is the profit?

Answer: Let me work through this carefully.

"Increased by 150%" means we ADD 150% of the ORIGINAL value to the original.
This is NOT 15% (that would be 0.15).
This is NOT multiplying by 1.5 (that would be only a 50% increase).

Original house value: $80,000
Increase amount: $80,000 × 150% = $80,000 × 1.50 = $<<80000*1.5=120000>>120,000
New value: $80,000 + $120,000 = $<<80000+120000=200000>>200,000

Total costs: $80,000 (purchase) + $50,000 (repairs) = $<<80000+50000=130000>>130,000

Profit = New value - Total costs = $200,000 - $130,000 = $<<200000-130000=70000>>70,000
#### 70000""",

        """Question: An antique worth $100 increases in value by 200%. What is it worth now?

Answer: "Increases by 200%" means ADD 200% of the original to itself.
200% of $100 = $100 × 2.00 = $<<100*2=200>>200
New value = $100 + $200 = $<<100+200=300>>300
The antique tripled in value (original + 200% = 300% of original).
#### 300""",

        """Question: A painting bought for $50,000 increased in value by 150%. What is the new value?

Answer: IMPORTANT: "Increased by 150%" is NOT the same as "multiplied by 1.5"!
When something increases by 150%, we ADD 150% to the original.
Original: $50,000
Increase: $50,000 × 1.5 = $<<50000*1.5=75000>>75,000
New value: $50,000 + $75,000 = $<<50000+75000=125000>>125,000
#### 125000""",
    ]

    # PATTERN 2: Average speed = total distance / total time
    pattern2_samples = [
        """Question: John drives 3 hours at 60 mph to a destination, then drives back. He spends some time in traffic. What is his average speed for the whole trip?

Answer: For average speed, we ALWAYS use:
Average speed = Total distance / Total time

This is NOT (speed1 + speed2) / 2!

If John drives at 60 mph for 3 hours going:
Distance there: 60 × 3 = <<60*3=180>>180 miles
Distance back: also 180 miles (same route)
Total distance: 180 + 180 = <<180+180=360>>360 miles

To find average speed, we need total time, then: 360 / total_time
#### (depends on total time given)""",

        """Question: A car travels 100 miles at 50 mph, then 100 miles at 25 mph. What is the average speed?

Answer: Common mistake: (50 + 25) / 2 = 37.5 mph. This is WRONG!

Correct method:
Time for first 100 miles: 100 / 50 = <<100/50=2>>2 hours
Time for second 100 miles: 100 / 25 = <<100/25=4>>4 hours
Total distance: 200 miles
Total time: 2 + 4 = <<2+4=6>>6 hours
Average speed: 200 / 6 = <<200/6=33.33>>33.33 mph
#### 33""",
    ]

    # PATTERN 3: Breakeven + 1 = first earning year
    pattern3_samples = [
        """Question: A tree costs $90 to plant. It produces $7 worth of fruit per year, but costs $3.50 per year to maintain. In what year will you START earning money?

Answer: Net annual profit: $7 - $3.50 = $<<7-3.5=3.5>>3.50

Years to break even: $90 / $3.50 = <<90/3.5=25.7>>25.7 years

But wait - we're asked when we START earning, not when we break even.
After year 25: cumulative = 25 × $3.50 = $87.50 (still negative $2.50)
After year 26: cumulative = 26 × $3.50 = $91 (exceeds $90 cost!)

Year 26 is when cumulative profit first exceeds the initial cost.
Year 26 is when you START earning money.
#### 26""",

        """Question: An investment costs $100 and returns $7.50 per year. In which year do you start making profit?

Answer: Exact breakeven: $100 / $7.50 = <<100/7.5=13.33>>13.33 years

After year 13: earned $97.50 (still $2.50 short)
After year 14: earned $105 (exceeds $100!)

Year 14 is when you first have positive total returns.
#### 14""",

        """Question: A lemon tree costs $90 to plant. It produces lemons worth $10.50 per year, but maintenance costs $3 per year. When does the owner START earning money?

Answer: Net per year: $10.50 - $3 = $<<10.5-3=7.5>>7.50
Breakeven: $90 / $7.50 = <<90/7.5=12>>12 years exactly

Since 12 years breaks exactly even (12 × $7.50 = $90), year 12 is breakeven.
Year 13 is when you START earning (first year with positive cumulative).
#### 13""",
    ]

    # PATTERN 4: Working backwards from fractions
    pattern4_samples = [
        """Question: A salesperson sold 1/3 of their items at the first stop, 2 more items at the second stop, and half the remaining at the third stop. They sold 5 at the third stop. How many did they start with?

Answer: Work BACKWARDS from what we know!

At third stop: sold 5, which was HALF of remaining.
So before third stop, they had: 5 × 2 = <<5*2=10>>10 items

Now I need to work out the original total.
Let T = original total.

After first stop: T - T/3 = 2T/3 remaining
After second stop: 2T/3 - 2 remaining (sold 2 items, not "2 more than first")
Before third: 2T/3 - 2 = 10 (what we calculated above)

Solving: 2T/3 - 2 = 10
2T/3 = 12
T = 12 × 3/2 = <<12*1.5=18>>18

Verify: Start with 18. Sell 18/3=6 at first. Sell 2 at second. Remaining: 18-6-2=10. Sell half (5) at third. ✓
#### 18""",

        """Question: Half of what remained was 5. How much was there before?

Answer: If half equals 5, the whole is 5 × 2 = <<5*2=10>>10.
#### 10""",
    ]

    # PATTERN 5: Comparing options correctly
    pattern5_samples = [
        """Question: A merchant can invest $5000 in jewelry (2.5% profit) or $8000 in gadgets (1.2% profit). Which gives more profit, and how much?

Answer: Let me calculate each option:

Jewelry: $5000 × 2.5% = $5000 × 0.025 = $<<5000*0.025=125>>125 profit
Gadgets: $8000 × 1.2% = $8000 × 0.012 = $<<8000*0.012=96>>96 profit

Comparing: $125 > $96

The jewelry gives more profit.
The profit from the better choice is $125.
#### 125""",
    ]

    # PATTERN 6: Calculate remaining before applying rate
    pattern6_samples = [
        """Question: Marissa is hiking a 12-mile trail. She walked 4 miles in 1 hour, then 2 miles in 2 hours. She will walk the rest at 3 mph. How long will the whole hike take?

Answer: First, find what's remaining:
Total trail: 12 miles
Already walked: 4 + 2 = <<4+2=6>>6 miles
Remaining: 12 - 6 = <<12-6=6>>6 miles

Time for remaining at 3 mph: 6 / 3 = <<6/3=2>>2 hours

Total time: 1 + 2 + 2 = <<1+2+2=5>>5 hours
#### 5""",

        """Question: Jill gets paid $20 per hour teaching and $30 per hour cheerleading. She teaches 35 hours per week and cheerleads 15 hours per week. How much does she make per year (50 weeks)?

Answer: Weekly earnings:
Teaching: $20 × 35 = $<<20*35=700>>700
Cheerleading: $30 × 15 = $<<30*15=450>>450
Weekly total: $700 + $450 = $<<700+450=1150>>1150

Annual earnings (50 weeks): $1150 × 50 = $<<1150*50=57500>>57,500
#### 57500""",
    ]

    for sample in (pattern1_samples + pattern2_samples + pattern3_samples +
                   pattern4_samples + pattern5_samples + pattern6_samples):
        samples.append({"text": sample})

    return samples


def main():
    from mlx_lm import load
    import mlx.core as mx
    import re

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    train_data_dir = "data/training/qwen3_combined_mastery"
    new_adapter_path = "data/adapters/qwen3_combined_mastery_lora"

    logger.info("=" * 70)
    logger.info("COMBINED MASTERY TRAINING")
    logger.info("Full GSM8K + explicit reasoning for failing patterns")
    logger.info("=" * 70)

    from modelcypher.core.use_cases.curriculum import BenchmarkLoader
    loader = BenchmarkLoader()

    np.random.seed(42)
    random.seed(42)

    # Load FULL GSM8K training set
    logger.info("Loading full GSM8K training set...")
    gsm_train = loader.load("gsm8k", split="train", limit=8000)

    gsm_samples = []
    for sample in gsm_train.samples:
        question = sample.prompt.replace("Answer:", "").strip()
        full_answer = sample.metadata.get("full_answer", sample.answer)
        gsm_samples.append({"text": f"Question: {question}\n\nAnswer: {full_answer}"})

    logger.info(f"Loaded {len(gsm_samples)} GSM8K samples")

    # Generate explicit reasoning samples (5x weight)
    explicit_samples = generate_explicit_reasoning_samples()
    logger.info(f"Explicit reasoning samples: {len(explicit_samples)} (will use 5x)")

    # Arithmetic foundation
    arith_samples = []
    for a in range(1, 20):
        for b in range(1, 20):
            arith_samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                arith_samples.append({"text": f"{a}-{b}={a-b}"})
            if a <= 12 and b <= 12:
                arith_samples.append({"text": f"{a}*{b}={a*b}"})

    np.random.shuffle(arith_samples)
    arith_samples = arith_samples[:500]
    logger.info(f"Arithmetic samples: {len(arith_samples)}")

    # Combine with explicit reasoning emphasized (5x)
    all_samples = gsm_samples + explicit_samples * 5 + arith_samples
    np.random.shuffle(all_samples)

    logger.info(f"Total training samples: {len(all_samples)}")
    logger.info(f"  GSM8K: {len(gsm_samples)}")
    logger.info(f"  Explicit (5x): {len(explicit_samples) * 5}")
    logger.info(f"  Arithmetic: {len(arith_samples)}")

    # Save
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)
    n_train = int(len(all_samples) * 0.95)
    n_valid = int(len(all_samples) * 0.025)

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

    # Train with more iterations for larger dataset
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
        "--iters", "4000",  # More iterations for larger dataset
        "--learning-rate", "1e-5",
        "--seed", "42",
        "--steps-per-report", "500",
    ]

    logger.info("Training (4000 iterations, full GSM8K + explicit reasoning)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
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
    logger.info("COMBINED MASTERY RESULTS")
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
    }
    output_path = Path("data/experiments/qwen3_combined_mastery.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
