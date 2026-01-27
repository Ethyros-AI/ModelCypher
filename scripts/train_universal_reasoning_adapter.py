#!/usr/bin/env python3
"""Train a universal reasoning adapter.

Key insight: The math adapter improved ARC-Challenge by 6% without any science training.
This proves we're teaching the STRUCTURE of thinking, not domain facts.

This script extends the approach:
1. Math recognition + solving (proven)
2. Logical reasoning patterns
3. Causal reasoning patterns
4. Comparison/contrast patterns

All patterns teach: implicit structure → explicit structure → solution
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# PART 1: MATH RECOGNITION (from unified adapter)
# ============================================================================

MATH_RECOGNITION = [
    {"prompt": "Question: She sold a third of her items.\n\nStructure:", "completion": " Let X = total. Sold = X/3. Remaining = 2X/3."},
    {"prompt": "Question: He ate half of what was left.\n\nStructure:", "completion": " Let R = remaining. Ate = R/2. Now has = R/2."},
    {"prompt": "Question: The boots cost five dollars more.\n\nStructure:", "completion": " Let H = other price. Boots = H + 5."},
    {"prompt": "Question: She has twice as many apples as oranges.\n\nStructure:", "completion": " Let O = oranges. Apples = 2 × O."},
    {"prompt": "Question: After spending $20, she has $35 left.\n\nStructure:", "completion": " Let X = start. X - 20 = 35. X = 55."},
]

# ============================================================================
# PART 2: LOGICAL REASONING PATTERNS
# ============================================================================

LOGICAL_RECOGNITION = [
    # If-then patterns
    {
        "prompt": "Question: If it rains, the ground gets wet. It rained. What happened to the ground?\n\nStructure:",
        "completion": " Rule: Rain → Wet ground. Fact: Rain. Conclusion: Wet ground. (Modus ponens)"
    },
    {
        "prompt": "Question: All mammals are warm-blooded. A dog is a mammal. Is a dog warm-blooded?\n\nStructure:",
        "completion": " Rule: Mammal → Warm-blooded. Fact: Dog is mammal. Conclusion: Dog is warm-blooded. (Universal instantiation)"
    },
    {
        "prompt": "Question: If A then B. B is false. What about A?\n\nStructure:",
        "completion": " Rule: A → B. Fact: ¬B. Conclusion: ¬A. (Modus tollens - contrapositive)"
    },

    # Negation patterns
    {
        "prompt": "Question: Not all birds can fly. Penguins are birds. Can penguins fly?\n\nStructure:",
        "completion": " 'Not all X are Y' ≠ 'No X are Y'. Being a bird doesn't guarantee flying. Answer: Unknown without more info."
    },

    # Conjunction/disjunction
    {
        "prompt": "Question: Either the light is on or it's off. The light is not on. What is its state?\n\nStructure:",
        "completion": " A ∨ B (exclusive). ¬A. Therefore B. The light is off."
    },
]

# ============================================================================
# PART 3: CAUSAL REASONING PATTERNS
# ============================================================================

CAUSAL_RECOGNITION = [
    # Cause-effect
    {
        "prompt": "Question: Plants need sunlight to grow. A plant was kept in a dark room. What happened?\n\nStructure:",
        "completion": " Cause: No sunlight. Effect rule: No sunlight → No growth. Conclusion: Plant didn't grow well."
    },
    {
        "prompt": "Question: When water freezes, it expands. Ice formed in a pipe. What risk does this create?\n\nStructure:",
        "completion": " Cause: Water freezes. Effect: Expansion. Container: Pipe (rigid). Risk: Pipe may burst from pressure."
    },

    # Counterfactual
    {
        "prompt": "Question: The match lit because it was struck. If it hadn't been struck, what would have happened?\n\nStructure:",
        "completion": " Cause: Striking. Effect: Light. Counterfactual: No strike → No light. The match wouldn't have lit."
    },

    # Chain causation
    {
        "prompt": "Question: Eating too much sugar can lead to weight gain. Weight gain increases heart disease risk. What does sugar do to heart disease risk?\n\nStructure:",
        "completion": " Chain: Sugar → Weight gain → Heart disease. Transitive: Sugar indirectly increases heart disease risk."
    },
]

# ============================================================================
# PART 4: COMPARISON/CONTRAST PATTERNS
# ============================================================================

COMPARISON_RECOGNITION = [
    {
        "prompt": "Question: Object A is heavier than B. Object B is heavier than C. Compare A and C.\n\nStructure:",
        "completion": " A > B and B > C. Transitive: A > C. Object A is heavier than C."
    },
    {
        "prompt": "Question: The temperature today is higher than yesterday, but lower than the record. Rank them.\n\nStructure:",
        "completion": " Today > Yesterday. Record > Today. Order: Record > Today > Yesterday."
    },
    {
        "prompt": "Question: Both X and Y are fruits. X is sweet, Y is sour. What do they share? How do they differ?\n\nStructure:",
        "completion": " Common property: Both are fruits. Difference: Taste (X=sweet, Y=sour)."
    },
]

# ============================================================================
# PART 5: SCIENCE REASONING PATTERNS
# ============================================================================

SCIENCE_RECOGNITION = [
    # Physical properties
    {
        "prompt": "Question: Metal conducts heat well. A metal spoon is placed in hot soup. What happens to the spoon?\n\nStructure:",
        "completion": " Property: Metal conducts heat. Setup: Metal in hot liquid. Result: Heat transfers to metal. The spoon gets hot."
    },
    {
        "prompt": "Question: Objects denser than water sink. A rock sinks in water. What can we conclude about the rock?\n\nStructure:",
        "completion": " Rule: Denser → Sinks. Observation: Rock sinks. Conclusion: Rock is denser than water."
    },

    # Life science
    {
        "prompt": "Question: Plants absorb carbon dioxide and release oxygen. What would happen if all plants disappeared?\n\nStructure:",
        "completion": " Plants: CO2 in, O2 out. No plants: No CO2 absorption, no O2 production. Result: CO2 rises, O2 falls."
    },

    # Earth science
    {
        "prompt": "Question: Warm air rises, cool air sinks. A heater is placed at the floor. How does air circulate?\n\nStructure:",
        "completion": " Source: Heat at floor. Hot air rises from floor. Cool air sinks to replace it. Creates convection cycle."
    },
]

# ============================================================================
# PART 6: FULL PROBLEM SOLVING (diverse domains)
# ============================================================================

FULL_SOLUTIONS = [
    # Math (from before)
    {
        "prompt": "Question: Melanie sold a third of her vacuums at the green house, 2 more at the red house, and half of what was left at the orange house. She has 5 left. How many did she start with?\n\nAnswer:",
        "completion": """ Work backwards from 5.
After orange: 5 left = half of before. Before orange: 5 × 2 = 10.
After red: 10 + 2 (sold at red) = 12.
After green: 12 = 2/3 of start (sold 1/3). Start: 12 × 3/2 = 18.

#### 18"""
    },

    # Science reasoning
    {
        "prompt": "Question: A student notices that a metal spoon feels colder than a wooden spoon at room temperature. Both are actually the same temperature. Why does the metal feel colder?\n\nAnswer:",
        "completion": """ Both objects are at room temperature (same temperature).
Metal conducts heat well. Wood is an insulator.
When you touch metal, heat flows quickly FROM your hand TO the metal.
When you touch wood, heat flows slowly.
Rapid heat loss = feels colder.

The metal feels colder because it conducts heat away from your hand faster, not because it's actually colder.

#### Metal conducts heat away from your hand faster"""
    },

    # Logical reasoning
    {
        "prompt": "Question: All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?\n\nAnswer:",
        "completion": """ Premise 1: All roses are flowers (Roses ⊂ Flowers)
Premise 2: Some flowers fade quickly (Some Flowers = Quick-fading)

Question: Are some roses quick-fading?

The quick-fading flowers might or might NOT include roses.
We only know SOME flowers fade quickly, not which ones.
The roses could be in the non-quick-fading subset.

Conclusion: We CANNOT conclude that some roses fade quickly.

#### No, we cannot conclude this"""
    },

    # Causal reasoning
    {
        "prompt": "Question: A factory increased production by 20%, but profits decreased. What could explain this?\n\nAnswer:",
        "completion": """ Production increased by 20% = More output.
Profits decreased = Revenue - Costs went down.

Possible explanations:
1. Costs increased more than 20% (materials, labor, energy)
2. Price per unit decreased (oversupply, competition)
3. Quality issues led to returns or discounts
4. Increased production required expensive overtime

Most likely: Costs of the additional production exceeded the additional revenue.

#### Production costs exceeded the additional revenue"""
    },

    # Comparison reasoning
    {
        "prompt": "Question: City A has 3x the population of City B, but City B has 2x the crime rate per capita. Which city has more total crimes?\n\nAnswer:",
        "completion": """ Let City B population = P
City A population = 3P

Let City B crime rate = R per capita
City A crime rate = R/2 per capita (half of B's rate per capita, but wait...)

Actually: City B has 2x the crime rate per capita means:
City B rate = 2R, City A rate = R (where R is some base rate)

Total crimes:
City A: 3P × R = 3PR
City B: P × 2R = 2PR

3PR > 2PR

City A has more total crimes (1.5x as many).

#### City A has more total crimes"""
    },
]


def generate_training_file(output_path: Path):
    """Generate combined JSONL training file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_data = (
        MATH_RECOGNITION +
        LOGICAL_RECOGNITION +
        CAUSAL_RECOGNITION +
        COMPARISON_RECOGNITION +
        SCIENCE_RECOGNITION +
        FULL_SOLUTIONS
    )

    with open(output_path, "w") as f:
        for item in all_data:
            text = item["prompt"] + item["completion"]
            f.write(json.dumps({"text": text}) + "\n")

    logger.info(f"Generated {len(all_data)} training samples:")
    logger.info(f"  - Math recognition: {len(MATH_RECOGNITION)}")
    logger.info(f"  - Logical recognition: {len(LOGICAL_RECOGNITION)}")
    logger.info(f"  - Causal recognition: {len(CAUSAL_RECOGNITION)}")
    logger.info(f"  - Comparison recognition: {len(COMPARISON_RECOGNITION)}")
    logger.info(f"  - Science recognition: {len(SCIENCE_RECOGNITION)}")
    logger.info(f"  - Full solutions: {len(FULL_SOLUTIONS)}")


def main():
    from mlx_lm import load, generate

    logger.info("=" * 70)
    logger.info("UNIVERSAL REASONING ADAPTER")
    logger.info("=" * 70)
    logger.info("\nGoal: Teach the STRUCTURE of thinking across domains")
    logger.info("Not domain facts, but recognition patterns and logical structure")

    # Generate training data
    train_path = Path("data/training/universal_reasoning/train.jsonl")
    generate_training_file(train_path)

    # Validation set
    valid_path = Path("data/training/universal_reasoning/valid.jsonl")
    valid_samples = (
        MATH_RECOGNITION[:2] +
        LOGICAL_RECOGNITION[:2] +
        SCIENCE_RECOGNITION[:2]
    )
    with open(valid_path, "w") as f:
        for item in valid_samples:
            text = item["prompt"] + item["completion"]
            f.write(json.dumps({"text": text}) + "\n")

    total_samples = (
        len(MATH_RECOGNITION) +
        len(LOGICAL_RECOGNITION) +
        len(CAUSAL_RECOGNITION) +
        len(COMPARISON_RECOGNITION) +
        len(SCIENCE_RECOGNITION) +
        len(FULL_SOLUTIONS)
    )

    # Training parameters - geometry-derived
    n_layers = 17  # Full expansion phase
    lr = 5e-5      # Same as unified adapter (worked well)
    iters = 1000   # More iterations for more diverse data

    adapter_path = Path("data/adapters/universal_reasoning_lora")

    logger.info(f"\nTraining configuration:")
    logger.info(f"  Layers: 0-{n_layers} (expansion phase)")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Iterations: {iters}")
    logger.info(f"  Training samples: {total_samples}")

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--data", str(train_path.parent),
        "--train",
        "--adapter-path", str(adapter_path),
        "--num-layers", str(n_layers),
        "--batch-size", "1",
        "--iters", str(iters),
        "--learning-rate", str(lr),
        "--save-every", "200",
    ]

    logger.info(f"\nRunning: {' '.join(cmd)}")
    logger.info("\n" + "=" * 70)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Adapter saved to: {adapter_path}")

        # Quick test
        logger.info("\n" + "=" * 70)
        logger.info("QUICK VERIFICATION")
        logger.info("=" * 70)

        model, tokenizer = load(model_path, adapter_path=str(adapter_path))

        test_prompts = [
            ("Question: She sold half her items. She has 10 left. How many did she start with?\n\nAnswer:", "20"),
            ("Question: If it rains, the picnic is canceled. It rained. Is the picnic happening?\n\nAnswer:", "no"),
            ("Question: Metal conducts heat well. You touch a hot metal pan. What do you feel?\n\nAnswer:", "hot"),
        ]

        import re
        for prompt, expected_keyword in test_prompts:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
            contains_keyword = expected_keyword.lower() in output.lower()
            logger.info(f"\n{'OK' if contains_keyword else 'CHECK'}: Expected '{expected_keyword}'")
            logger.info(f"  Output: {output[:150]}...")

        # Save metadata
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_layers": n_layers,
                "learning_rate": lr,
                "iterations": iters,
                "total_samples": total_samples,
            },
            "categories": {
                "math_recognition": len(MATH_RECOGNITION),
                "logical_recognition": len(LOGICAL_RECOGNITION),
                "causal_recognition": len(CAUSAL_RECOGNITION),
                "comparison_recognition": len(COMPARISON_RECOGNITION),
                "science_recognition": len(SCIENCE_RECOGNITION),
                "full_solutions": len(FULL_SOLUTIONS),
            }
        }

        output_path = Path("data/experiments/universal_reasoning_training.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\nResults saved to: {output_path}")

    else:
        logger.error(f"Training failed with code {result.returncode}")


if __name__ == "__main__":
    main()
