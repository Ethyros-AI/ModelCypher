#!/usr/bin/env python3
"""Evaluate mastery with FIXED number extraction.

Key fix: For arithmetic, extract FIRST number (the direct answer).
For GSM8K, extract number after ####.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.use_cases.curriculum import BenchmarkLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_tier(model, tokenizer, tier_name: str, problems: List[Tuple[str, str]], max_tokens: int) -> Dict:
    """Evaluate a tier with correct extraction logic."""
    import mlx.core as mx
    import re

    correct = 0
    details = []

    for prompt, expected in problems:
        # GSM8K gets "Question: ... Answer:" format
        if "GSM8K" in tier_name:
            full_prompt = f"Question: {prompt}\n\nAnswer:"
            gen_tokens = max_tokens
        else:
            full_prompt = prompt
            gen_tokens = 20  # Enough to capture continuation

        tokens = tokenizer.encode(full_prompt)
        generated = []

        for _ in range(gen_tokens):
            logits = model(mx.array([tokens + generated]))
            mx.eval(logits)

            logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()

            next_tok = int(np.argmax(probs))
            generated.append(next_tok)

            decoded = tokenizer.decode(generated)

            # Stop conditions
            if "####" in decoded:
                # Get a bit more for the final answer
                for _ in range(15):
                    logits = model(mx.array([tokens + generated]))
                    mx.eval(logits)
                    logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
                    probs = np.exp(logits_np - logits_np.max())
                    probs = probs / probs.sum()
                    next_tok = int(np.argmax(probs))
                    generated.append(next_tok)
                break

            if "<|im_end|>" in decoded or "\n\n\n" in decoded:
                break

        output = tokenizer.decode(generated).strip()
        output = output.replace("<|im_end|>", "").replace("!", "")

        # FIXED EXTRACTION LOGIC
        if "GSM8K" in tier_name:
            # For GSM8K: extract number after ####
            if "####" in output:
                answer_part = output.split("####")[-1].strip()
                answer_part = answer_part.replace(",", "").replace("$", "")
                numbers = re.findall(r'-?\d+\.?\d*', answer_part)
                predicted = numbers[0].split('.')[0] if numbers else ""  # Integer part
            else:
                # Fallback: last number
                numbers = re.findall(r'-?\d+', output.replace(",", ""))
                predicted = numbers[-1] if numbers else ""
        else:
            # For arithmetic/multi-step: extract FIRST number (the direct answer)
            numbers = re.findall(r'-?\d+', output)
            predicted = numbers[0] if numbers else ""

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        details.append({
            "prompt": prompt[:50],
            "expected": expected,
            "predicted": predicted,
            "output": output[:100],
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "details": details,
    }


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_gsm8k_mastery_lora"

    logger.info("=" * 70)
    logger.info("MASTERY EVALUATION - FIXED EXTRACTION")
    logger.info("=" * 70)

    loader = BenchmarkLoader()
    gsm_test = loader.load("gsm8k", split="test", limit=30)

    test_suite = {
        "Tier1_Arithmetic": [
            ("2+2=", "4"), ("3+5=", "8"), ("9-4=", "5"), ("7+8=", "15"),
            ("12-7=", "5"), ("6*4=", "24"), ("8*3=", "24"), ("5*9=", "45"),
            ("15+6=", "21"), ("18-9=", "9"),
        ],
        "Tier2_MultiStep": [
            ("5+3=8, 8+2=", "10"),
            ("7+4=11, 11-3=", "8"),
            ("4+6=10, 10+5=", "15"),
            ("3*4=12, 12+5=", "17"),
            ("6+8=14, 14-6=", "8"),
        ],
        "Tier3_GSM8K": [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples[:20]],
    }

    logger.info(f"\nLoading adapter: {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    results = {}
    for tier_name, problems in test_suite.items():
        max_tokens = 200 if "GSM8K" in tier_name else 20
        results[tier_name] = evaluate_tier(model, tokenizer, tier_name, problems, max_tokens)

    # Display results
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS (FIXED EXTRACTION)")
    logger.info(f"{'='*60}")

    for tier_name, data in results.items():
        logger.info(f"\n{tier_name}: {data['accuracy']:.0%} ({data['correct']}/{data['total']})")
        for d in data["details"]:
            mark = "OK" if d["correct"] else "XX"
            logger.info(f"  {mark}: '{d['prompt'][:35]}...' -> '{d['predicted']}' (expected '{d['expected']}')")
            if not d["correct"]:
                logger.info(f"       Output: {d['output'][:60]}...")

    # Mastery check
    t1 = results["Tier1_Arithmetic"]["accuracy"]
    t2 = results["Tier2_MultiStep"]["accuracy"]
    t3 = results["Tier3_GSM8K"]["accuracy"]

    logger.info("\n" + "=" * 70)
    logger.info("MASTERY STATUS")
    logger.info("=" * 70)

    arithmetic_mastered = t1 >= 0.9
    multistep_mastered = t2 >= 0.9
    gsm8k_mastered = t3 >= 0.7

    logger.info(f"""
Tier 1 - Arithmetic:  {'MASTERED' if arithmetic_mastered else 'NOT YET'} ({t1:.0%})
Tier 2 - Multi-step:  {'MASTERED' if multistep_mastered else 'NOT YET'} ({t2:.0%})
Tier 3 - GSM8K:       {'MASTERED' if gsm8k_mastered else 'NOT YET'} ({t3:.0%})

Foundation preserved: {arithmetic_mastered and multistep_mastered}
GSM8K target met: {gsm8k_mastered}

READY FOR TIER 4 (ARC): {arithmetic_mastered and multistep_mastered and gsm8k_mastered}
""")

    # Save
    output = {
        "adapter": adapter_path,
        "results": {k: v["accuracy"] for k, v in results.items()},
        "mastery": {
            "arithmetic": arithmetic_mastered,
            "multistep": multistep_mastered,
            "gsm8k": gsm8k_mastered,
            "ready_for_arc": arithmetic_mastered and multistep_mastered and gsm8k_mastered,
        },
        "details": {k: v["details"] for k, v in results.items()},
    }

    output_path = Path("data/experiments/mastery_evaluation_fixed.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
