#!/usr/bin/env python3
"""Verify Intermediate Geometry: Check φ alignment at the self-reflection step.

The model now says "Let me understand the question. [core question]."

We need to verify that:
1. The core question part IS at φ resonance
2. The model processes at the optimal geometry point
3. Then expands to the answer

This measures geometry at the INTERMEDIATE state, not the final output.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


def compute_ratio(model, tokenizer, text: str) -> tuple[float, int]:
    """Compute compression ratio for a given text."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)
    peak = float(mx.sqrt(mx.sum(hidden * hidden)))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norm = float(mx.sqrt(mx.sum(hidden * hidden)))
        peak = max(peak, norm)

    final = norm
    return peak / final, len(tokens)


def extract_core_from_reflection(response: str) -> str:
    """Extract just the core question from a self-reflection response.

    Input: "Let me understand the question. What is 5 + 3?\n\n5 + 3 = 8."
    Output: "What is 5 + 3?"
    """
    # Find the core question between "Let me understand the question." and "\n\n"
    if "Let me understand" in response:
        parts = response.split("Let me understand the question.")
        if len(parts) > 1:
            after_reflection = parts[1].strip()
            # Core question ends at first double newline or period followed by newline
            if "\n\n" in after_reflection:
                core = after_reflection.split("\n\n")[0].strip()
            else:
                core = after_reflection.split("\n")[0].strip()
            return core
    return response


def verify_intermediate_geometry():
    """Verify the geometry at the self-reflection intermediate step."""
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"Loading: {model_path}")
    model, tokenizer = load(model_path)

    # Test prompts - the model should self-reflect on these
    test_cases = [
        "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\n\n",
        "Question: I was wondering if you could help me figure out what happens when you add the number five to the number three?\n\n",
        "Question: In the context of basic arithmetic, what is fifteen plus seven?\n\n",
        "Question: If you have 5 apples and someone gives you 3 more, how many do you have?\n\n",
        "Question: What is 2 + 2?\n\n",
    ]

    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING INTERMEDIATE GEOMETRY")
    logger.info("=" * 70)
    logger.info(f"Target: Core question should have ratio ≈ φ ({PHI:.3f})")

    results = []

    for prompt in test_cases:
        # Generate response
        response = generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)

        # Extract just the original question (input)
        original_q = prompt.replace("Question: ", "").replace("\n\n", "").strip()

        # Extract the core question from the self-reflection
        core_q = extract_core_from_reflection(response)

        # Measure geometries
        orig_ratio, orig_tokens = compute_ratio(model, tokenizer, original_q)
        core_ratio, core_tokens = compute_ratio(model, tokenizer, core_q)
        full_ratio, full_tokens = compute_ratio(model, tokenizer, response)

        orig_dist = abs(orig_ratio - PHI)
        core_dist = abs(core_ratio - PHI)
        full_dist = abs(full_ratio - PHI)

        logger.info(f"\n{'-' * 60}")
        logger.info(f"Original ({orig_tokens} tokens): ratio={orig_ratio:.3f}, dist_φ={orig_dist:.3f}")
        logger.info(f"  '{original_q[:50]}...'")
        logger.info(f"Core ({core_tokens} tokens): ratio={core_ratio:.3f}, dist_φ={core_dist:.3f}")
        logger.info(f"  '{core_q[:50]}...'")
        logger.info(f"Full ({full_tokens} tokens): ratio={full_ratio:.3f}, dist_φ={full_dist:.3f}")

        # Check if core is closer to φ than original
        improved = core_dist < orig_dist
        at_resonance = core_tokens >= 8 and core_tokens <= 20 and core_dist < 0.2

        if improved:
            logger.info(f"✓ Core improved φ alignment by {(orig_dist - core_dist)/orig_dist*100:.0f}%")
        if at_resonance:
            logger.info(f"✓ Core is at φ resonance!")

        results.append({
            "original": {
                "text": original_q[:100],
                "tokens": orig_tokens,
                "ratio": float(orig_ratio),
                "distance_phi": float(orig_dist),
            },
            "core": {
                "text": core_q[:100],
                "tokens": core_tokens,
                "ratio": float(core_ratio),
                "distance_phi": float(core_dist),
            },
            "full": {
                "text": response[:100],
                "tokens": full_tokens,
                "ratio": float(full_ratio),
                "distance_phi": float(full_dist),
            },
            "improved": bool(improved),
            "at_resonance": bool(at_resonance),
        })

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    improved_count = sum(1 for r in results if r["improved"])
    resonance_count = sum(1 for r in results if r["at_resonance"])

    avg_orig_dist = np.mean([r["original"]["distance_phi"] for r in results])
    avg_core_dist = np.mean([r["core"]["distance_phi"] for r in results])

    logger.info(f"Improved φ alignment: {improved_count}/{len(results)}")
    logger.info(f"At φ resonance: {resonance_count}/{len(results)}")
    logger.info(f"Avg original dist from φ: {avg_orig_dist:.3f}")
    logger.info(f"Avg core dist from φ: {avg_core_dist:.3f}")

    if avg_core_dist < avg_orig_dist:
        pct = (avg_orig_dist - avg_core_dist) / avg_orig_dist * 100
        logger.info(f"\n✓ CONFIRMED: Self-reflection improves φ alignment by {pct:.0f}%!")
        logger.info("The model processes at optimal geometry in the intermediate step.")
    else:
        logger.info("\n? Core questions not improving φ alignment")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "phi_target": float(PHI),
        "results": results,
        "summary": {
            "improved_count": improved_count,
            "resonance_count": resonance_count,
            "total": len(results),
            "avg_original_dist": float(avg_orig_dist),
            "avg_core_dist": float(avg_core_dist),
        },
    }

    output_path = Path("data/experiments/intermediate_geometry.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to: {output_path}")

    return results


if __name__ == "__main__":
    verify_intermediate_geometry()
