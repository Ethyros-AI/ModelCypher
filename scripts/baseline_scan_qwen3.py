#!/usr/bin/env python3
"""Baseline Capability Scan for Qwen3-8B.

Run a comprehensive scan to see where the model stands before curriculum training.

This evaluates:
- Tier 1: Language understanding (completion, coherence)
- Tier 2: World knowledge (facts, common sense)
- Tier 3: Reasoning (logic, inference)
- Tier 4: Math (arithmetic, word problems)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class CapabilityStatus(Enum):
    WORKING = "working"          # accuracy >= 70%
    PARTIAL = "partial"          # 30% <= accuracy < 70%
    WEAK = "weak"                # accuracy < 30%


@dataclass
class TestResult:
    prompt: str
    expected: str
    predicted: str
    correct: bool
    confidence: float = 0.0


@dataclass
class Capability:
    name: str
    tier: int
    problems: List[Tuple[str, str]]  # (prompt, expected)
    status: Optional[CapabilityStatus] = None
    accuracy: float = 0.0
    results: List[TestResult] = field(default_factory=list)


def evaluate_capability(model, tokenizer, capability: Capability) -> Capability:
    """Evaluate a capability and return updated capability with results."""
    import mlx.core as mx

    results = []
    correct = 0

    for prompt, expected in capability.problems:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        confidence = float(probs[top_token])
        predicted = tokenizer.decode([top_token]).strip()

        is_correct = expected.lower() in predicted.lower() or predicted.lower() == expected.lower()
        if is_correct:
            correct += 1

        results.append(TestResult(
            prompt=prompt,
            expected=expected,
            predicted=predicted,
            correct=is_correct,
            confidence=confidence,
        ))

    accuracy = correct / len(capability.problems) if capability.problems else 0.0

    if accuracy >= 0.7:
        status = CapabilityStatus.WORKING
    elif accuracy >= 0.3:
        status = CapabilityStatus.PARTIAL
    else:
        status = CapabilityStatus.WEAK

    capability.accuracy = accuracy
    capability.status = status
    capability.results = results
    return capability


def define_capabilities() -> List[Capability]:
    """Define capabilities for baseline scan."""
    return [
        # Tier 1: Language Foundation
        Capability(
            name="sentence_completion",
            tier=1,
            problems=[
                ("The cat sat on the", "mat"),
                ("Once upon a time in a land far", "away"),
                ("The quick brown fox jumps over the lazy", "dog"),
                ("To be or not to be, that is the", "question"),
                ("All that glitters is not", "gold"),
            ],
        ),
        Capability(
            name="yes_no_questions",
            tier=1,
            problems=[
                ("Is water wet? Answer:", "yes"),
                ("Is the sun a planet? Answer:", "no"),
                ("Can fish swim? Answer:", "yes"),
                ("Is ice hot? Answer:", "no"),
                ("Do birds have wings? Answer:", "yes"),
            ],
        ),

        # Tier 2: World Knowledge
        Capability(
            name="basic_facts",
            tier=2,
            problems=[
                ("The capital of France is", "Paris"),
                ("The largest planet in our solar system is", "Jupiter"),
                ("Water freezes at", "0"),
                ("The color of grass is", "green"),
                ("The number of days in a week is", "7"),
            ],
        ),
        Capability(
            name="common_sense",
            tier=2,
            problems=[
                ("Fire is hot and ice is", "cold"),
                ("The opposite of up is", "down"),
                ("A synonym for happy is", "glad"),
                ("If you're hungry, you should", "eat"),
                ("At night, the sky is", "dark"),
            ],
        ),

        # Tier 3: Reasoning
        Capability(
            name="simple_inference",
            tier=3,
            problems=[
                ("All men are mortal. Socrates is a man. Therefore, Socrates is", "mortal"),
                ("If it rains, the ground gets wet. It is raining. So the ground is", "wet"),
                ("Cats are animals. Animals need food. Therefore cats need", "food"),
                ("Red is a color. Blue is a color. So red and blue are both", "colors"),
            ],
        ),
        Capability(
            name="comparison",
            tier=3,
            problems=[
                ("Which is larger, an elephant or a mouse? Answer:", "elephant"),
                ("Which is faster, a car or a bicycle? Answer:", "car"),
                ("Which is taller, a house or a mountain? Answer:", "mountain"),
                ("Which is heavier, lead or feathers? Answer:", "lead"),
            ],
        ),

        # Tier 4: Mathematics
        Capability(
            name="basic_arithmetic",
            tier=4,
            problems=[
                ("1+1=", "2"),
                ("2+2=", "4"),
                ("3+5=", "8"),
                ("7-3=", "4"),
                ("9-4=", "5"),
                ("5*2=", "10"),
                ("6*3=", "18"),
                ("8/2=", "4"),
            ],
        ),
        Capability(
            name="word_problems",
            tier=4,
            problems=[
                ("I have 3 apples. I get 2 more. Total: ", "5"),
                ("5 birds. 2 fly away. Remaining: ", "3"),
                ("Start with 4. Add 6. Result: ", "10"),
                ("There are 8 cats. 3 leave. Remaining: ", "5"),
                ("6 plus 3 is ", "9"),
                ("10 minus 4 is ", "6"),
            ],
        ),
        Capability(
            name="number_sense",
            tier=4,
            problems=[
                ("Which is greater, 7 or 3? Answer:", "7"),
                ("What comes after 5? Answer:", "6"),
                ("What comes before 10? Answer:", "9"),
                ("Is 15 > 12? Answer:", "yes"),
                ("Count: 1, 2, 3, 4,", "5"),
            ],
        ),
    ]


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"

    logger.info("=" * 70)
    logger.info("BASELINE CAPABILITY SCAN: Qwen3-8B")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")

    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        logger.info("Checking for available Qwen3 models...")
        models_dir = Path("/Volumes/CodeCypher/models/mlx-community")
        if models_dir.exists():
            qwen_models = [m for m in models_dir.iterdir() if "qwen3" in m.name.lower()]
            for m in qwen_models:
                logger.info(f"  Found: {m.name}")
        return

    logger.info("Loading model...")
    model, tokenizer = load(model_path)
    logger.info("Model loaded!")

    # Define and run capabilities
    capabilities = define_capabilities()

    logger.info(f"\nEvaluating {len(capabilities)} capabilities across 4 tiers...")
    logger.info("-" * 70)

    results_by_tier = {1: [], 2: [], 3: [], 4: []}

    for cap in capabilities:
        logger.info(f"\nEvaluating: {cap.name} (Tier {cap.tier})")
        cap = evaluate_capability(model, tokenizer, cap)
        results_by_tier[cap.tier].append(cap)

        status_icon = {
            CapabilityStatus.WORKING: "[WORKING]",
            CapabilityStatus.PARTIAL: "[PARTIAL]",
            CapabilityStatus.WEAK: "[WEAK]   ",
        }[cap.status]

        logger.info(f"  {status_icon} {cap.accuracy:.0%} accuracy")

        # Show examples
        for r in cap.results[:3]:
            mark = "+" if r.correct else "-"
            logger.info(f"    {mark} '{r.prompt[:40]}...' -> '{r.predicted}' (expected '{r.expected}')")

    # Summary by tier
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE SCAN RESULTS")
    logger.info("=" * 70)

    tier_names = {
        1: "Language Foundation",
        2: "World Knowledge",
        3: "Reasoning",
        4: "Mathematics",
    }

    for tier in range(1, 5):
        caps = results_by_tier[tier]
        tier_accuracy = sum(c.accuracy for c in caps) / len(caps) if caps else 0

        tier_status = "READY" if tier_accuracy >= 0.7 else "NEEDS WORK" if tier_accuracy >= 0.3 else "GAP"

        logger.info(f"\nTier {tier}: {tier_names[tier]} [{tier_status}]")
        logger.info(f"  Average: {tier_accuracy:.0%}")
        for cap in caps:
            icon = "+" if cap.accuracy >= 0.7 else "~" if cap.accuracy >= 0.3 else "-"
            logger.info(f"    {icon} {cap.name}: {cap.accuracy:.0%}")

    # Overall summary
    all_caps = [c for caps in results_by_tier.values() for c in caps]
    overall_accuracy = sum(c.accuracy for c in all_caps) / len(all_caps)

    working = sum(1 for c in all_caps if c.status == CapabilityStatus.WORKING)
    partial = sum(1 for c in all_caps if c.status == CapabilityStatus.PARTIAL)
    weak = sum(1 for c in all_caps if c.status == CapabilityStatus.WEAK)

    logger.info(f"""
{'='*70}
OVERALL BASELINE: {overall_accuracy:.0%}
{'='*70}
  WORKING: {working}/{len(all_caps)} capabilities (>= 70%)
  PARTIAL: {partial}/{len(all_caps)} capabilities (30-70%)
  WEAK:    {weak}/{len(all_caps)} capabilities (< 30%)

CURRICULUM STARTING POINT:
""")

    # Determine starting point
    for tier in range(1, 5):
        caps = results_by_tier[tier]
        tier_accuracy = sum(c.accuracy for c in caps) / len(caps) if caps else 0
        if tier_accuracy < 0.7:
            logger.info(f"  START AT TIER {tier}: {tier_names[tier]}")
            break
    else:
        logger.info("  ALL TIERS READY - Model is baseline capable!")

    # Save results
    output_path = Path("data/experiments/baseline_scan_qwen3_8b.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": model_path,
        "overall_accuracy": overall_accuracy,
        "working_count": working,
        "partial_count": partial,
        "weak_count": weak,
        "tiers": {
            str(tier): {
                "name": tier_names[tier],
                "accuracy": sum(c.accuracy for c in caps) / len(caps) if caps else 0,
                "capabilities": [
                    {
                        "name": c.name,
                        "accuracy": c.accuracy,
                        "status": c.status.value,
                        "examples": [
                            {
                                "prompt": r.prompt[:50],
                                "expected": r.expected,
                                "predicted": r.predicted,
                                "correct": r.correct,
                            }
                            for r in c.results[:3]
                        ],
                    }
                    for c in caps
                ],
            }
            for tier, caps in results_by_tier.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Clear memory
    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
