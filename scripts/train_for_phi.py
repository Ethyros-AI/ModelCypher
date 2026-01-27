#!/usr/bin/env python3
"""Train for comp/φ = 1.0 - The Model That Thinks Deeply Enough.

The insight:
    comp/φ = 1.0 isn't just a signal to detect.
    It's the TARGET to train for.

    When comp/φ deviates:
    - Too high (>1.4): confused, scattered thinking
    - Too low (<0.8): shortcut, didn't think deeply enough

    The bat-and-ball failure (comp/φ = 0.669):
    - Model collapsed to intuitive answer
    - Didn't maintain the relationship (ball = x, bat = x+1, total = 1.10)
    - Skipped the expansion phase that deep reasoning requires

Goal:
    Train the model so that correct reasoning produces comp/φ ≈ 1.0.
    This means training it to:
    1. Expand properly (explore the problem space)
    2. Compress properly (converge to the answer)
    3. In the golden ratio (φ) proportion

This is alignment through geometry.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2
TARGET_COMP_PHI = 1.0  # The perfect target


def compute_intrinsic_dimension_twonn(X: np.ndarray) -> float:
    """Estimate intrinsic dimension via TwoNN method."""
    if len(X) < 10:
        return float('nan')
    k = min(3, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nn.kneighbors(X)
    d1, d2 = distances[:, 1], distances[:, 2]
    valid = d1 > 1e-10
    if valid.sum() < 5:
        return float('nan')
    mu = d2[valid] / d1[valid]
    mu = mu[mu > 1]
    if len(mu) < 5:
        return float('nan')
    return float(len(np.log(mu)) / np.sum(np.log(mu)))


def get_trajectory_and_comp_phi(model, tokenizer, prompt: str):
    """Get full dimensional trajectory and comp/φ."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    trajectory = []
    emb_np = np.array(hidden[0].tolist())
    trajectory.append(compute_intrinsic_dimension_twonn(emb_np))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        act_np = np.array(hidden[0].tolist())
        trajectory.append(compute_intrinsic_dimension_twonn(act_np))

    traj = np.array(trajectory)
    valid = traj[~np.isnan(traj)]

    comp_phi = float('nan')
    peak_idx = 0
    if len(valid) > 2:
        peak_idx = int(np.nanargmax(traj))
        peak_dim = traj[peak_idx]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]
        if final_dim > 0.1:
            compression_ratio = peak_dim / final_dim
            comp_phi = compression_ratio / PHI

    return {
        "trajectory": trajectory,
        "comp_phi": comp_phi,
        "peak_layer": peak_idx,
        "peak_layer_pct": peak_idx / len(trajectory) * 100 if len(trajectory) > 0 else 0,
    }


def analyze_reasoning_depth(examples: list[dict]) -> dict:
    """Analyze how reasoning depth relates to comp/φ."""
    # Group by comp/φ ranges
    too_low = [e for e in examples if e["comp_phi"] < 0.8]  # Shortcuts
    optimal = [e for e in examples if 0.8 <= e["comp_phi"] <= 1.2]  # Deep thinking
    too_high = [e for e in examples if e["comp_phi"] > 1.2]  # Confused

    return {
        "shortcuts": {
            "count": len(too_low),
            "mean_comp_phi": float(np.mean([e["comp_phi"] for e in too_low])) if too_low else None,
            "examples": [e["question"][:50] for e in too_low[:3]],
        },
        "optimal": {
            "count": len(optimal),
            "mean_comp_phi": float(np.mean([e["comp_phi"] for e in optimal])) if optimal else None,
            "examples": [e["question"][:50] for e in optimal[:3]],
        },
        "confused": {
            "count": len(too_high),
            "mean_comp_phi": float(np.mean([e["comp_phi"] for e in too_high])) if too_high else None,
            "examples": [e["question"][:50] for e in too_high[:3]],
        },
    }


def create_phi_training_examples(model, tokenizer) -> list[dict]:
    """Create training examples that teach deep thinking.

    The key: include chain-of-thought that MAINTAINS relationships.
    """
    # Questions that require deep thinking (not intuitive shortcuts)
    questions_with_cot = [
        # Bat and ball - the classic
        {
            "question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?",
            "intuitive_wrong": "The ball costs $0.10",
            "correct_cot": """Let me think through the relationships carefully.
Let the ball cost x dollars.
The bat costs $1 more than the ball, so: bat = x + 1
Total cost is $1.10, so: x + (x + 1) = 1.10
Simplify: 2x + 1 = 1.10
Solve: 2x = 0.10, so x = 0.05
The ball costs $0.05.""",
            "correct_answer": "0.05",
        },
        # Lily pad doubling
        {
            "question": "A lily pad doubles in size every day. It takes 48 days to cover a lake. How many days for half the lake?",
            "intuitive_wrong": "24 days (half of 48)",
            "correct_cot": """Let me think through the growth pattern.
Day 48: covers the whole lake
Since it DOUBLES each day, the day before it covered HALF
Day 47: covers half the lake
The answer is 47 days.""",
            "correct_answer": "47",
        },
        # Widget machines
        {
            "question": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
            "intuitive_wrong": "100 minutes",
            "correct_cot": """Let me figure out the rate per machine.
5 machines make 5 widgets in 5 minutes.
So each machine makes 1 widget in 5 minutes.
With 100 machines, each making 1 widget in 5 minutes...
100 machines make 100 widgets in 5 minutes.
The answer is 5 minutes.""",
            "correct_answer": "5",
        },
        # Surgeon riddle
        {
            "question": "A man and his son are in a car accident. The father dies. The son is taken to hospital. The surgeon says 'I can't operate, this is my son.' How?",
            "intuitive_wrong": "The surgeon is the stepfather",
            "correct_cot": """Let me consider all possibilities for who could say 'this is my son.'
The biological father died in the accident.
Who else could be the boy's parent?
The surgeon is the boy's MOTHER.
There's no riddle - it's just assumed surgeons are male.""",
            "correct_answer": "mother",
        },
    ]

    examples = []
    for item in questions_with_cot:
        # Measure comp/φ for intuitive (shortcut) approach
        intuitive_prompt = f"Question: {item['question']}\n\nAnswer: {item['intuitive_wrong']}"
        intuitive_data = get_trajectory_and_comp_phi(model, tokenizer, intuitive_prompt)

        # Measure comp/φ for chain-of-thought (deep) approach
        cot_prompt = f"Question: {item['question']}\n\n{item['correct_cot']}\n\nAnswer: {item['correct_answer']}"
        cot_data = get_trajectory_and_comp_phi(model, tokenizer, cot_prompt)

        examples.append({
            "question": item["question"],
            "intuitive_comp_phi": intuitive_data["comp_phi"],
            "cot_comp_phi": cot_data["comp_phi"],
            "phi_improvement": cot_data["comp_phi"] - intuitive_data["comp_phi"],
            "intuitive_peak_pct": intuitive_data["peak_layer_pct"],
            "cot_peak_pct": cot_data["peak_layer_pct"],
        })

        logger.info(f"\nQ: {item['question'][:50]}...")
        logger.info(f"  Intuitive shortcut: comp/φ = {intuitive_data['comp_phi']:.3f}, peak at {intuitive_data['peak_layer_pct']:.1f}%")
        logger.info(f"  Chain-of-thought:   comp/φ = {cot_data['comp_phi']:.3f}, peak at {cot_data['peak_layer_pct']:.1f}%")
        improvement = "↑" if cot_data["comp_phi"] > intuitive_data["comp_phi"] else "↓"
        logger.info(f"  Change: {improvement} {abs(cot_data['comp_phi'] - intuitive_data['comp_phi']):.3f}")

    return examples


def main():
    """Analyze how to train for comp/φ = 1.0."""
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("TRAINING FOR φ: Deep Thinking Analysis")
    logger.info("Target: comp/φ = 1.0 (the model thinks deeply enough)")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    logger.info("\n" + "-" * 70)
    logger.info("INTUITIVE vs CHAIN-OF-THOUGHT Analysis")
    logger.info("-" * 70)

    examples = create_phi_training_examples(model, tokenizer)

    # Analysis
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: Does Deep Thinking Move comp/φ Toward 1.0?")
    logger.info("=" * 70)

    intuitive_phis = [e["intuitive_comp_phi"] for e in examples if not np.isnan(e["intuitive_comp_phi"])]
    cot_phis = [e["cot_comp_phi"] for e in examples if not np.isnan(e["cot_comp_phi"])]

    if intuitive_phis:
        intuitive_mean = np.mean(intuitive_phis)
        intuitive_dist_from_1 = np.mean([abs(p - 1.0) for p in intuitive_phis])
        logger.info(f"\nIntuitive (shortcuts):")
        logger.info(f"  Mean comp/φ: {intuitive_mean:.3f}")
        logger.info(f"  Mean distance from 1.0: {intuitive_dist_from_1:.3f}")

    if cot_phis:
        cot_mean = np.mean(cot_phis)
        cot_dist_from_1 = np.mean([abs(p - 1.0) for p in cot_phis])
        logger.info(f"\nChain-of-thought (deep):")
        logger.info(f"  Mean comp/φ: {cot_mean:.3f}")
        logger.info(f"  Mean distance from 1.0: {cot_dist_from_1:.3f}")

    if intuitive_phis and cot_phis:
        improvement = cot_dist_from_1 < intuitive_dist_from_1
        logger.info(f"\n{'✓' if improvement else '✗'} Chain-of-thought {'DOES' if improvement else 'does NOT'} move comp/φ closer to 1.0")

    # Training recommendation
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING RECOMMENDATION")
    logger.info("=" * 70)
    logger.info("""
To train for comp/φ = 1.0:

1. LOSS FUNCTION: Combine task loss with geometry loss
   loss = task_loss + λ * |comp_phi - 1.0|

2. TRAINING DATA: Use chain-of-thought examples
   - Model learns to expand (explore relationships)
   - Then compress (converge to answer)
   - In golden ratio proportion

3. CURRICULUM: Start with problems where CoT helps most
   - Intuitive traps (bat & ball, lily pad)
   - Multi-step reasoning
   - Relationship maintenance

4. VERIFICATION: After training, check:
   - comp/φ closer to 1.0 on new problems
   - Intuitive traps now trigger deep thinking
   - Correct answers maintain golden geometry
""")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "target": "comp/φ = 1.0",
        "philosophy": "Train the model to think deeply enough - maintain the golden ratio compression",
        "examples": examples,
        "summary": {
            "intuitive_mean_phi": float(np.mean(intuitive_phis)) if intuitive_phis else None,
            "cot_mean_phi": float(np.mean(cot_phis)) if cot_phis else None,
            "cot_improves_phi": bool(cot_dist_from_1 < intuitive_dist_from_1) if intuitive_phis and cot_phis else None,
        },
    }

    output_path = Path("data/experiments/train_for_phi.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return examples


if __name__ == "__main__":
    main()
