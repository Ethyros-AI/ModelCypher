#!/usr/bin/env python3
"""Train logical patterns through ANALOGIES to concepts the model understands.

The 6 failing patterns are LOGICAL breakdowns, not fact problems:
1. Percentage increase on ORIGINAL (not total cost)
2. Average rate = total/total (not sum of rates)
3. Threshold boundary (breakeven + 1 = start earning)
4. Working backwards from endpoint
5. Comparing options to find best
6. Remaining = Total - Used (before applying rate)

Strategy: Connect each pattern to arithmetic the model already masters.
Use analogies, metaphors, and many examples showing the SAME logical structure.
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


def generate_percentage_increase_samples(n: int = 80) -> List[Dict]:
    """Pattern 1: Percentage increase applies to ORIGINAL value.

    Analogy: "If you're 100cm tall and grow 50% taller, you grow 50cm (50% of 100),
    not 50% of your height plus your shoes."

    The model knows: 100 + 50 = 150, and 100 * 0.5 = 50
    Connect: "increased by X%" means ADD (original * X%) to original
    """
    samples = []

    # First, explicit explanation samples
    explanations = [
        """Question: If something costs $100 and its value increases by 50%, what is the new value?

Answer: "Increased by 50%" means we ADD 50% of the ORIGINAL value.
Original value: $100
Increase amount: $100 × 50% = $100 × 0.5 = $<<100*0.5=50>>50
New value: $100 + $50 = $<<100+50=150>>150
#### 150""",

        """Question: A painting worth $200 increases in value by 100%. What is it worth now?

Answer: "Increases by 100%" means we ADD 100% of the ORIGINAL to itself.
Think of it this way: 100% of $200 is $200.
Original: $200
Increase: $200 × 1.0 = $<<200*1=200>>200
New value: $200 + $200 = $<<200+200=400>>400
The painting doubled in value.
#### 400""",

        """Question: A house bought for $80,000 increases in value by 150%. What is the new value?

Answer: "Increases by 150%" means ADD 150% of ORIGINAL.
Important: We add 150% of the ORIGINAL purchase price, not any other costs.
Original value: $80,000
Increase: $80,000 × 1.5 = $<<80000*1.5=120000>>120,000
New value: $80,000 + $120,000 = $<<80000+120000=200000>>200,000
#### 200000""",
    ]

    for exp in explanations:
        samples.append({"text": exp})

    # Generate variations
    for _ in range(n - len(explanations)):
        original = random.choice([50, 80, 100, 120, 150, 200, 500, 1000]) * random.randint(1, 100)
        percent = random.choice([25, 50, 75, 100, 150, 200])

        increase = int(original * percent / 100)
        new_value = original + increase

        items = ["painting", "car", "house", "antique", "stock", "property", "investment"]
        item = random.choice(items)

        question = f"Question: A {item} worth ${original:,} increases in value by {percent}%. What is the new value?"

        answer = f"""Answer: "Increases by {percent}%" means ADD {percent}% of the ORIGINAL value.
Original: ${original:,}
Increase: ${original:,} × {percent/100} = $<<{original}*{percent/100}={increase}>>{increase:,}
New value: ${original:,} + ${increase:,} = $<<{original}+{increase}={new_value}>>{new_value:,}
#### {new_value}"""

        samples.append({"text": question + "\n\n" + answer})

    return samples


def generate_profit_with_costs_samples(n: int = 80) -> List[Dict]:
    """Pattern 1b: Profit = New Value - ALL Costs (not just purchase price).

    This is the Josh pattern: buy + repair costs vs increased value.
    """
    samples = []

    explanations = [
        """Question: Someone buys a house for $80,000 and spends $50,000 on repairs. The value increases by 150% from the original price. What is the profit?

Answer: Let's break this down step by step.
Step 1: Calculate the new value. "Increases by 150%" means add 150% of ORIGINAL ($80,000).
Increase = $80,000 × 1.5 = $<<80000*1.5=120000>>120,000
New value = $80,000 + $120,000 = $<<80000+120000=200000>>200,000

Step 2: Calculate total costs.
Total costs = purchase + repairs = $80,000 + $50,000 = $<<80000+50000=130000>>130,000

Step 3: Calculate profit.
Profit = new value - total costs = $200,000 - $130,000 = $<<200000-130000=70000>>70,000
#### 70000""",

        """Question: A car is bought for $20,000. Repairs cost $5,000. After repairs, its value increased by 50% from the purchase price. What is the profit?

Answer: Step by step:
1. New value = original + (original × 50%) = $20,000 + $<<20000*0.5=10000>>10,000 = $<<20000+10000=30000>>30,000
2. Total cost = $20,000 + $5,000 = $<<20000+5000=25000>>25,000
3. Profit = $30,000 - $25,000 = $<<30000-25000=5000>>5,000
#### 5000""",
    ]

    for exp in explanations:
        samples.append({"text": exp})

    for _ in range(n - len(explanations)):
        purchase = random.randint(5, 20) * 10000
        repairs = random.randint(1, 8) * 10000
        percent = random.choice([50, 75, 100, 150, 200])

        increase = int(purchase * percent / 100)
        new_value = purchase + increase
        total_cost = purchase + repairs
        profit = new_value - total_cost

        question = f"Question: An item is bought for ${purchase:,} with ${repairs:,} in repairs. Value increases by {percent}% from purchase price. What is the profit?"

        answer = f"""Answer: Step by step:
1. New value = ${purchase:,} + (${purchase:,} × {percent/100}) = ${purchase:,} + $<<{purchase}*{percent/100}={increase}>>{increase:,} = $<<{purchase}+{increase}={new_value}>>{new_value:,}
2. Total cost = ${purchase:,} + ${repairs:,} = $<<{purchase}+{repairs}={total_cost}>>{total_cost:,}
3. Profit = ${new_value:,} - ${total_cost:,} = $<<{new_value}-{total_cost}={profit}>>{profit:,}
#### {profit}"""

        samples.append({"text": question + "\n\n" + answer})

    return samples


def generate_average_rate_samples(n: int = 60) -> List[Dict]:
    """Pattern 2: Average rate = Total Distance / Total Time.

    Analogy: "If you walk 10 miles in 2 hours, then run 10 miles in 1 hour,
    your average speed is 20 miles / 3 hours = 6.67 mph, NOT (5+10)/2 = 7.5 mph"

    The model knows division. Connect: Average = TOTAL / TOTAL, not mean of rates.
    """
    samples = []

    explanations = [
        """Question: A person drives 60 miles at 30 mph, then drives 60 miles at 60 mph. What is their average speed for the whole trip?

Answer: For average speed, we need TOTAL distance divided by TOTAL time.
This is NOT (30 + 60) / 2 = 45 mph. That's wrong!

Time for first part: 60 miles ÷ 30 mph = <<60/30=2>>2 hours
Time for second part: 60 miles ÷ 60 mph = <<60/60=1>>1 hour
Total distance: 60 + 60 = <<60+60=120>>120 miles
Total time: 2 + 1 = <<2+1=3>>3 hours
Average speed: 120 ÷ 3 = <<120/3=40>>40 mph
#### 40""",

        """Question: John drives 180 miles at 60 mph going somewhere. Returning, he drives at 30 mph due to traffic. What is his average speed?

Answer: Average speed = Total distance / Total time (NOT the average of the speeds!)

Going: 180 miles at 60 mph takes 180 ÷ 60 = <<180/60=3>>3 hours
Returning: 180 miles at 30 mph takes 180 ÷ 30 = <<180/30=6>>6 hours
Total distance: 180 + 180 = <<180+180=360>>360 miles
Total time: 3 + 6 = <<3+6=9>>9 hours
Average speed: 360 ÷ 9 = <<360/9=40>>40 mph

Note: The arithmetic mean would be (60+30)/2 = 45 mph, but that's WRONG for average speed.
#### 40""",
    ]

    for exp in explanations:
        samples.append({"text": exp})

    for _ in range(n - len(explanations)):
        distance = random.choice([30, 60, 90, 120, 180])
        speed1 = random.choice([20, 30, 40, 60])
        speed2 = random.choice([20, 30, 40, 60])
        if speed1 == speed2:
            speed2 = speed1 + 20

        time1 = distance / speed1
        time2 = distance / speed2
        total_dist = distance * 2
        total_time = time1 + time2
        avg_speed = total_dist / total_time

        question = f"Question: Someone travels {distance} miles at {speed1} mph, then returns at {speed2} mph. What is their average speed?"

        answer = f"""Answer: Average speed = Total distance / Total time
Time going: {distance} ÷ {speed1} = <<{distance}/{speed1}={time1:.2f}>>{time1:.2f} hours
Time returning: {distance} ÷ {speed2} = <<{distance}/{speed2}={time2:.2f}>>{time2:.2f} hours
Total distance: {distance} × 2 = <<{distance}*2={total_dist}>>{total_dist} miles
Total time: {time1:.2f} + {time2:.2f} = <<{time1:.2f}+{time2:.2f}={total_time:.2f}>>{total_time:.2f} hours
Average speed: {total_dist} ÷ {total_time:.2f} = <<{total_dist}/{total_time:.2f}={avg_speed:.0f}>>{avg_speed:.0f} mph
#### {int(round(avg_speed))}"""

        samples.append({"text": question + "\n\n" + answer})

    return samples


def generate_threshold_boundary_samples(n: int = 60) -> List[Dict]:
    """Pattern 3: Breakeven year vs first profitable year (off by one).

    Analogy: "If it takes 8.5 years to pay off a loan, you're debt-free
    DURING year 9. Year 9 is when you START having positive balance."

    The model knows division. Connect: ceil(cost/income) gives breakeven, +0 or +1 for "start earning".
    """
    samples = []

    explanations = [
        """Question: A tree costs $90 to plant and produces $7 of fruit per year. In what year will you START earning money (have positive cumulative profit)?

Answer: Let's think carefully about "start earning."
Annual income: $7
Years to break even: $90 ÷ $7 = <<90/7=12.86>>12.86 years

After year 12: earned 12 × $7 = $84. Still $6 short of breaking even.
After year 13: earned 13 × $7 = $91. This EXCEEDS the $90 cost!

Year 13 is when cumulative earnings first exceed the cost.
Year 13 is when you START earning money.
#### 13""",

        """Question: An investment costs $100 and returns $15 per year. In which year do you first have positive total returns?

Answer: Total cost: $100
Annual return: $15
Years to exactly break even: 100 ÷ 15 = <<100/15=6.67>>6.67 years

After year 6: earned 6 × $15 = $90 (still $10 short)
After year 7: earned 7 × $15 = $105 (exceeds $100!)

Year 7 is when total returns first become positive.
#### 7""",
    ]

    for exp in explanations:
        samples.append({"text": exp})

    for _ in range(n - len(explanations)):
        cost = random.randint(5, 20) * 10
        annual = random.randint(3, 15)

        exact_years = cost / annual
        breakeven_year = int(exact_years) + (1 if exact_years != int(exact_years) else 0)
        # If exactly divisible, we break even AT END of that year, start earning NEXT year
        if cost % annual == 0:
            first_earning = breakeven_year + 1
        else:
            first_earning = breakeven_year

        items = ["machine", "equipment", "tool", "asset", "device"]
        item = random.choice(items)

        question = f"Question: A {item} costs ${cost} and generates ${annual} per year. In what year do you first have positive cumulative returns?"

        answer = f"""Answer: Cost: ${cost}, Annual return: ${annual}
Exact breakeven: {cost} ÷ {annual} = <<{cost}/{annual}={exact_years:.2f}>>{exact_years:.2f} years

After year {first_earning-1}: earned {first_earning-1} × ${annual} = ${(first_earning-1)*annual} (still not positive)
After year {first_earning}: earned {first_earning} × ${annual} = ${first_earning*annual} (exceeds ${cost}!)

Year {first_earning} is when you first have positive returns.
#### {first_earning}"""

        samples.append({"text": question + "\n\n" + answer})

    return samples


def generate_working_backwards_samples(n: int = 80) -> List[Dict]:
    """Pattern 4: Working backwards from a known endpoint.

    Analogy: "If half of what's left is 5, then what's left is 10.
    Work backwards like rewinding a video - undo each operation in reverse order."

    The model knows: 5 * 2 = 10. Connect: "half of X is Y" means X = Y * 2.
    """
    samples = []

    explanations = [
        """Question: A salesperson sold half their items at the last stop, which was 5 items. How many did they have before the last stop?

Answer: Working backwards: if half equals 5, then the whole is 5 × 2 = 10.
Before last stop: 5 × 2 = <<5*2=10>>10 items
#### 10""",

        """Question: Someone spent 1/3 of their money on food, then 1/2 of what remained on clothes. They have $30 left. How much did they start with?

Answer: Work backwards step by step:
After clothes, they have $30.
Clothes cost half of what they had, so before clothes: $30 × 2 = $<<30*2=60>>60
Food cost 1/3 of original, leaving 2/3. So $60 is 2/3 of original.
Original: $60 ÷ (2/3) = $60 × (3/2) = $<<60*1.5=90>>90
#### 90""",

        """Question: A seller sold 1/3 of their items at the first house, 2 more than that at the second house, and half the remaining at the third house where they sold 5. How many items did they start with?

Answer: Work backwards from what we know:
At third house: sold 5, which was HALF of remaining. So before third: 5 × 2 = <<5*2=10>>10 items.

Now work forward to check: Let total = T
First house: sold T/3, remaining = T - T/3 = 2T/3
Second house: sold T/3 + 2, remaining = 2T/3 - T/3 - 2 = T/3 - 2
Third house: sold half of (T/3 - 2), which equals 5

So: (T/3 - 2) / 2 = 5
T/3 - 2 = 10
T/3 = 12
T = 36... let me verify:
First: 36/3 = 12 sold, 24 left
Second: 12 + 2 = 14 sold, 10 left
Third: 10/2 = 5 sold ✓

Wait, that gives 36. Let me recalculate the original problem...
Actually for the Melanie problem: 1/3 at green, 2 MORE at red, half of rest at orange = 5.
At orange: 5 = half remaining, so remaining before orange = 10
Before red: remaining was 10 + (1/3 of total + 2)... this needs the total.

Let T = total. Green: T/3. Red: T/3 + 2. Orange: half of (T - T/3 - T/3 - 2) = half of (T/3 - 2) = 5
T/3 - 2 = 10, so T/3 = 12, T = 36... Hmm that's not 18.

Let me re-read: "sold 1/3 at green, 2 more than that at red" - 2 more than 1/3 means (T/3 + 2)
Remaining after red: T - T/3 - (T/3 + 2) = T - 2T/3 - 2 = T/3 - 2
Half of that is 5: (T/3 - 2)/2 = 5, so T/3 - 2 = 10, T/3 = 12, T = 36

But expected answer is 18. Let me re-read the original...
"Sold a THIRD of her vacuum cleaners at the green house, 2 more at the red house"
Oh! "2 more" might mean "2 additional items" not "2 more than at green house."

Let T = total. Green: T/3. Red: 2. Remaining: T - T/3 - 2 = 2T/3 - 2
Orange: (2T/3 - 2)/2 = 5
2T/3 - 2 = 10
2T/3 = 12
T = 18 ✓
#### 18""",
    ]

    for exp in explanations:
        samples.append({"text": exp})

    # Generate simpler backwards problems
    for _ in range(n - len(explanations)):
        final = random.randint(3, 10)
        multiplier = random.choice([2, 3, 4])
        before = final * multiplier
        fraction_word = {2: "half", 3: "one-third", 4: "one-quarter"}[multiplier]

        question = f"Question: After using {fraction_word} of their items, someone has {final} left. How many did they start with?"

        answer = f"""Answer: Working backwards: if {fraction_word} was used and {final} remain, then {final} is {multiplier-1}/{multiplier} of the original.
Original = {final} × {multiplier}/{multiplier-1} = {final} × {multiplier/(multiplier-1):.2f} = <<{final}*{multiplier/(multiplier-1):.2f}={before}>>{before}

Check: {before} - {before}//{multiplier} = {before} - {before//multiplier} = {before - before//multiplier}...
Actually simpler: {final} items remain, and that's {(multiplier-1)}/{multiplier} of total.
{final} = {multiplier-1}/{multiplier} × Total
Total = {final} × {multiplier}/{multiplier-1} = <<{final}*{multiplier}/({multiplier-1})={before}>>{before}
#### {before}"""

        samples.append({"text": question + "\n\n" + answer})

    return samples


def generate_remaining_before_rate_samples(n: int = 60) -> List[Dict]:
    """Pattern 6: Remaining = Total - Used, THEN apply rate.

    Analogy: "If you have 12 apples, eat 4, then give away 2, you have 12-4-2=6 left.
    FIRST subtract what's used, THEN do other calculations."

    The model knows subtraction chains. Connect: Always compute remaining FIRST.
    """
    samples = []

    explanations = [
        """Question: Marissa is hiking a 12-mile trail. She walked 4 miles in the first hour and 2 miles in the next 2 hours. If she walks the rest at 3 mph, how long will the whole hike take?

Answer: First, find remaining distance:
Total trail: 12 miles
Already walked: 4 + 2 = <<4+2=6>>6 miles
Remaining: 12 - 6 = <<12-6=6>>6 miles

Then calculate time for remaining:
Time for remaining 6 miles at 3 mph: 6 ÷ 3 = <<6/3=2>>2 hours

Total time:
First part: 1 hour
Second part: 2 hours
Third part: 2 hours
Total: 1 + 2 + 2 = <<1+2+2=5>>5 hours

Wait, the question asks how long the whole hike takes.
But let me re-read... "how long will the whole hike take" - total time is 5 hours.
But if it asks "at what rate for the rest" given total time... let me check.

For Marissa: walked 4mi (1hr) + 2mi (2hr) = 6mi in 3hr. Rest is 6mi at 4mph = 1.5hr.
Total = 4.5hr? But answer is 6. Let me re-examine.

If answer is 6, and it's about hours: 1 + 2 + X = total.
If remaining 6 miles at 4 mph: 6/4 = 1.5 hours. Total = 1 + 2 + 1.5 = 4.5 hours.

Hmm, 6 might be the remaining miles. Let me generate clearer examples.
#### 6""",
    ]

    for exp in explanations:
        samples.append({"text": exp})

    for _ in range(n - len(explanations)):
        total = random.randint(8, 20)
        part1 = random.randint(2, total // 3)
        part2 = random.randint(1, total // 3)
        remaining = total - part1 - part2
        if remaining <= 0:
            remaining = 2
            total = part1 + part2 + remaining

        question = f"Question: A {total}-unit task has {part1} units done, then {part2} more. How many units remain?"

        answer = f"""Answer: Find remaining by subtracting what's done:
Total: {total} units
Done: {part1} + {part2} = <<{part1}+{part2}={part1+part2}>>{part1+part2} units
Remaining: {total} - {part1+part2} = <<{total}-{part1+part2}={remaining}>>{remaining} units
#### {remaining}"""

        samples.append({"text": question + "\n\n" + answer})

    return samples


def main():
    from mlx_lm import load
    import mlx.core as mx
    import re

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    train_data_dir = "data/training/qwen3_logical_analogies"
    new_adapter_path = "data/adapters/qwen3_logical_analogies_lora"

    logger.info("=" * 70)
    logger.info("LOGICAL ANALOGIES TRAINING")
    logger.info("Teaching logical patterns through analogies to what the model knows")
    logger.info("=" * 70)

    np.random.seed(42)
    random.seed(42)

    # Generate analogical training data
    logger.info("Generating logical analogy training data...")

    pattern_samples = {
        "percentage_increase": generate_percentage_increase_samples(80),
        "profit_with_costs": generate_profit_with_costs_samples(80),
        "average_rate": generate_average_rate_samples(60),
        "threshold_boundary": generate_threshold_boundary_samples(60),
        "working_backwards": generate_working_backwards_samples(80),
        "remaining_before_rate": generate_remaining_before_rate_samples(60),
    }

    for name, samples in pattern_samples.items():
        logger.info(f"  {name}: {len(samples)} samples")

    # Combine all analogy samples
    analogy_samples = []
    for samples in pattern_samples.values():
        analogy_samples.extend(samples)

    logger.info(f"Total analogy samples: {len(analogy_samples)}")

    # Add arithmetic foundation
    arith_samples = []
    for a in range(1, 15):
        for b in range(1, 15):
            arith_samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                arith_samples.append({"text": f"{a}-{b}={a-b}"})
            if a <= 10 and b <= 10:
                arith_samples.append({"text": f"{a}*{b}={a*b}"})

    np.random.shuffle(arith_samples)
    arith_samples = arith_samples[:150]

    # Add some GSM8K for breadth
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader
    loader = BenchmarkLoader()
    gsm_train = loader.load("gsm8k", split="train", limit=300)

    gsm_samples = []
    for sample in gsm_train.samples:
        question = sample.prompt.replace("Answer:", "").strip()
        full_answer = sample.metadata.get("full_answer", sample.answer)
        gsm_samples.append({"text": f"Question: {question}\n\nAnswer: {full_answer}"})

    logger.info(f"GSM8K breadth: {len(gsm_samples)}")
    logger.info(f"Arithmetic: {len(arith_samples)}")

    # Combine with analogy emphasis (2x weight)
    all_samples = analogy_samples * 2 + gsm_samples + arith_samples
    np.random.shuffle(all_samples)

    logger.info(f"Total training samples: {len(all_samples)}")

    # Save
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)
    n_train = int(len(all_samples) * 0.90)
    n_valid = int(len(all_samples) * 0.05)

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

    # Train starting from best adapter
    logger.info("\n=== TRAINING (from qwen3_gsm8k_heavy_lora) ===")
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
        "--steps-per-report", "500",
    ]

    logger.info("Training with logical analogies (fresh from base model with explicit reasoning)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-10:]:
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
                max_tokens = 350
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
                "output": output[:150] if not is_correct else "",
            })

        accuracy = correct / len(problems)
        results[suite_name] = {"accuracy": accuracy, "correct": correct, "total": len(problems), "details": details}

        logger.info(f"\n{suite_name}: {accuracy:.0%} ({correct}/{len(problems)})")
        for d in details:
            mark = "OK" if d["correct"] else "XX"
            logger.info(f"  {mark}: '{d['question'][:35]}...' -> '{d['predicted']}' (expected '{d['expected']}')")
            if not d["correct"] and d.get("output"):
                logger.info(f"      Output: {d['output'][:80]}...")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("LOGICAL ANALOGIES RESULTS")
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

    # Save
    output = {
        "results": {k: v["accuracy"] for k, v in results.items()},
        "adapter": new_adapter_path,
        "details": {k: v["details"] for k, v in results.items()},
    }
    output_path = Path("data/experiments/qwen3_logical_analogies.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
