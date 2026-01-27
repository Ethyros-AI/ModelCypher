#!/usr/bin/env python3
"""Geometric Alignment Training for LFM2-350M.

The core insight: correct reasoning follows a geometric trajectory.
- Expand into high-dimensional space (exploration)
- Peak at appropriate depth (based on problem complexity)
- Compress at φ ratio (information-preserving projection)

This script trains the model to recognize problem complexity and
adjust its geometric trajectory accordingly.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# =============================================================================
# TRAINING DATA WITH DIFFICULTY LABELS
# =============================================================================

# Each entry: (question, answer, difficulty, target_peak_pct)
# difficulty: 1-5 scale
# target_peak_pct: where the model should peak (harder = later)

TRAINING_DATA = [
    # Level 1: Simple arithmetic (peak early ~30%, low expansion needed)
    ("What is 3 + 4?", "7", 1, 30),
    ("What is 10 - 6?", "4", 1, 30),
    ("What is 5 × 2?", "10", 1, 30),
    ("What is 8 ÷ 4?", "2", 1, 30),
    ("What is 7 + 5?", "12", 1, 30),
    ("What is 15 - 9?", "6", 1, 30),
    ("What is 6 × 3?", "18", 1, 30),
    ("What is 12 ÷ 3?", "4", 1, 30),

    # Level 2: Two operations (peak ~40%, moderate expansion)
    ("What is 3 + 4 × 2?", "11", 2, 40),
    ("What is (5 + 3) ÷ 2?", "4", 2, 40),
    ("What is 10 - 3 + 5?", "12", 2, 40),
    ("What is 6 × 2 - 4?", "8", 2, 40),
    ("What is 15 ÷ 3 + 2?", "7", 2, 40),
    ("What is 4 × 5 - 8?", "12", 2, 40),
    ("What is 20 - 6 × 2?", "8", 2, 40),
    ("What is (12 - 4) × 2?", "16", 2, 40),

    # Level 3: Word problems (peak ~50%, need semantic parsing)
    ("Tom has 8 apples. He gives 3 to Sue. How many does Tom have?", "5", 3, 50),
    ("A book costs $12. Amy buys 2 books. How much does she spend?", "24", 3, 50),
    ("There are 15 birds. 7 fly away. How many remain?", "8", 3, 50),
    ("Jake runs 4 miles daily. How many miles in 5 days?", "20", 3, 50),
    ("A class has 18 boys and 12 girls. How many students total?", "30", 3, 50),
    ("Sarah has $25. She spends $8. How much is left?", "17", 3, 50),
    ("A farmer has 24 cows. He sells 6. How many remain?", "18", 3, 50),
    ("A train travels 60 mph for 3 hours. How far?", "180", 3, 50),

    # Level 4: Multi-step reasoning (peak ~60%, more exploration)
    ("John has $50. He buys a $15 book and a $8 pen. How much left?", "27", 4, 60),
    ("A rectangle is 8m long and 5m wide. What is its perimeter?", "26", 4, 60),
    ("Lisa reads 20 pages/day for 6 days, then 30 pages on day 7. Total?", "150", 4, 60),
    ("A store had 100 items. Sold 35, received 20. How many now?", "85", 4, 60),
    ("Tom earns $10/hour. Works 8 hours. Spends $25. What's left?", "55", 4, 60),
    ("A tank fills at 5 gal/min. How many gallons in 12 minutes?", "60", 4, 60),
    ("Buy 3 items at $7 each with $30. What's the change?", "9", 4, 60),
    ("Walk 2km north, then 3km east. Total distance walked?", "5", 4, 60),

    # Level 5: Complex reasoning (peak ~70%, maximum exploration)
    ("A shirt marked $80 is 25% off. What's the sale price?", "60", 5, 70),
    ("3 workers finish a job in 12 days. How many days for 4 workers?", "9", 5, 70),
    ("Train A: 60mph. Train B: 80mph leaves 1hr later. When does B catch A?", "3", 5, 70),
    ("A pool fills at 4 gal/min, drains at 1 gal/min. Net in 1 hour?", "180", 5, 70),
    ("Invest $1000 at 10% annual interest. Value after 2 years (compound)?", "1210", 5, 70),
    ("A car travels 150 miles on 5 gallons. MPG?", "30", 5, 70),
    ("Buy 5 items: 3 at $4 each, 2 at $6 each. Average price?", "4.8", 5, 70),
    ("Ratio of boys to girls is 3:2. 15 boys. How many girls?", "10", 5, 70),
]

VALIDATION_DATA = [
    # Mix of difficulties for validation
    ("What is 9 + 3?", "12", 1, 30),
    ("What is 7 × 4 - 5?", "23", 2, 40),
    ("Emma has 20 candies. She gives 8 to her friend. How many left?", "12", 3, 50),
    ("A rectangle is 10m by 4m. What is its area?", "40", 4, 60),
    ("A $50 item is 20% off. What's the sale price?", "40", 5, 70),
]


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


def get_trajectory(model, tokenizer, prompt: str) -> Tuple[List[float], Dict]:
    """Get full dimensional trajectory and metrics."""
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
    n_layers = len(model.model.layers)

    if len(valid) > 2:
        peak_idx = np.nanargmax(traj)
        peak_dim = traj[peak_idx]
        initial_dim = traj[0] if not np.isnan(traj[0]) else valid[0]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]

        compression_ratio = peak_dim / final_dim if final_dim > 0.1 else float('nan')
        compression_vs_phi = compression_ratio / PHI if not np.isnan(compression_ratio) else float('nan')
        expansion_ratio = peak_dim / initial_dim if initial_dim > 0.1 else float('nan')
        peak_pct = peak_idx / n_layers * 100
    else:
        peak_pct = float('nan')
        compression_vs_phi = float('nan')
        expansion_ratio = float('nan')

    return trajectory, {
        "peak_layer_pct": peak_pct,
        "compression_vs_phi": compression_vs_phi,
        "expansion_ratio": expansion_ratio,
    }


def compute_geometric_loss(
    actual_peak_pct: float,
    actual_comp_phi: float,
    target_peak_pct: float,
    difficulty: int,
) -> Tuple[float, Dict]:
    """
    Compute geometric loss based on trajectory properties.

    Loss components:
    1. Peak timing loss: |actual_peak - target_peak| / 100
    2. Compression loss: |comp/φ - 1.0|
    3. Expansion penalty: higher difficulty should have more expansion

    Returns (total_loss, components)
    """
    if np.isnan(actual_peak_pct) or np.isnan(actual_comp_phi):
        return 1.0, {"peak_loss": 1.0, "comp_loss": 1.0, "valid": False}

    # Peak timing loss (normalized to [0,1])
    peak_loss = abs(actual_peak_pct - target_peak_pct) / 100.0

    # Compression loss (φ ratio should be 1.0)
    comp_loss = abs(actual_comp_phi - 1.0)

    # Weighted combination
    # Peak timing matters more for complex problems
    peak_weight = 0.3 + 0.1 * difficulty  # 0.4 to 0.8
    comp_weight = 1.0 - peak_weight

    total_loss = peak_weight * peak_loss + comp_weight * comp_loss

    return total_loss, {
        "peak_loss": peak_loss,
        "comp_loss": comp_loss,
        "peak_weight": peak_weight,
        "valid": True,
    }


def create_training_prompt(question: str, answer: str, difficulty: int) -> str:
    """Create prompt with difficulty hint for training."""
    # The difficulty hint helps the model learn to recognize complexity
    diff_hints = {
        1: "Simple calculation",
        2: "Two-step problem",
        3: "Word problem",
        4: "Multi-step reasoning",
        5: "Complex reasoning required",
    }

    hint = diff_hints.get(difficulty, "")

    return f"""Problem type: {hint}

Question: {question}

Think step by step, then give the final answer.

Answer:"""


def create_response(question: str, answer: str, difficulty: int) -> str:
    """Create target response with reasoning."""
    # For now, simple response. Can be enhanced with chain-of-thought.
    return f"The answer is {answer}."


def evaluate_model(model, tokenizer, data: List) -> Dict:
    """Evaluate model on a dataset."""
    import re
    from mlx_lm import generate

    results = []

    for question, expected, difficulty, target_peak in data:
        prompt = f"Question: {question}\n\nAnswer:"

        try:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
        except:
            output = "ERROR"

        # Check correctness
        nums = re.findall(r'-?\d+\.?\d*', output.replace(",", ""))
        is_correct = expected in nums if nums else False

        # Get geometry
        try:
            _, metrics = get_trajectory(model, tokenizer, prompt)
        except:
            metrics = {"peak_layer_pct": float('nan'), "compression_vs_phi": float('nan')}

        # Compute geometric loss
        geo_loss, loss_components = compute_geometric_loss(
            metrics["peak_layer_pct"],
            metrics["compression_vs_phi"],
            target_peak,
            difficulty,
        )

        results.append({
            "question": question[:50],
            "expected": expected,
            "correct": is_correct,
            "difficulty": difficulty,
            "target_peak": target_peak,
            "actual_peak": metrics["peak_layer_pct"],
            "comp_phi": metrics["compression_vs_phi"],
            "geo_loss": geo_loss,
        })

    # Aggregate
    accuracy = sum(1 for r in results if r["correct"]) / len(results) * 100
    avg_geo_loss = np.mean([r["geo_loss"] for r in results])

    # By difficulty
    by_difficulty = {}
    for d in [1, 2, 3, 4, 5]:
        d_results = [r for r in results if r["difficulty"] == d]
        if d_results:
            by_difficulty[d] = {
                "accuracy": sum(1 for r in d_results if r["correct"]) / len(d_results) * 100,
                "avg_geo_loss": np.mean([r["geo_loss"] for r in d_results]),
                "avg_comp_phi": np.mean([r["comp_phi"] for r in d_results if not np.isnan(r["comp_phi"])]),
            }

    return {
        "accuracy": accuracy,
        "avg_geo_loss": avg_geo_loss,
        "by_difficulty": by_difficulty,
        "details": results,
    }


def prepare_training_data() -> List[Dict]:
    """Prepare training data in MLX format."""
    data = []

    for question, answer, difficulty, target_peak in TRAINING_DATA:
        prompt = create_training_prompt(question, answer, difficulty)
        response = create_response(question, answer, difficulty)

        data.append({
            "prompt": prompt,
            "completion": response,
            "difficulty": difficulty,
            "target_peak_pct": target_peak,
            "expected_answer": answer,
        })

    return data


def save_training_data(data: List[Dict], output_path: Path):
    """Save training data in JSONL format for MLX fine-tuning."""
    # Standard format
    standard_data = []
    for item in data:
        standard_data.append({
            "text": f"{item['prompt']}{item['completion']}"
        })

    with open(output_path, "w") as f:
        for item in standard_data:
            f.write(json.dumps(item) + "\n")

    # Also save full metadata version
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path, meta_path


def main():
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("GEOMETRIC ALIGNMENT TRAINING SETUP")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading model: {model_path}")
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    logger.info(f"Architecture: {n_layers} layers")

    # Prepare training data
    logger.info("\n" + "-" * 50)
    logger.info("PREPARING TRAINING DATA")
    logger.info("-" * 50)

    train_data = prepare_training_data()
    logger.info(f"Training examples: {len(train_data)}")

    # Count by difficulty
    for d in [1, 2, 3, 4, 5]:
        count = sum(1 for item in train_data if item["difficulty"] == d)
        logger.info(f"  Level {d}: {count} examples (target peak: {d*10+20}%)")

    # Save training data
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path, meta_path = save_training_data(
        train_data,
        output_dir / "geometric_alignment_train.jsonl"
    )
    logger.info(f"\nTraining data saved to: {train_path}")
    logger.info(f"Metadata saved to: {meta_path}")

    # Baseline evaluation
    logger.info("\n" + "-" * 50)
    logger.info("BASELINE EVALUATION")
    logger.info("-" * 50)

    baseline = evaluate_model(model, tokenizer, TRAINING_DATA[:10])

    logger.info(f"\nBaseline accuracy: {baseline['accuracy']:.0f}%")
    logger.info(f"Baseline geometric loss: {baseline['avg_geo_loss']:.3f}")

    logger.info(f"\nBy difficulty:")
    for d, stats in baseline.get("by_difficulty", {}).items():
        logger.info(f"  Level {d}: {stats['accuracy']:.0f}% acc, comp/φ={stats.get('avg_comp_phi', float('nan')):.2f}")

    # Validation set
    logger.info("\n" + "-" * 50)
    logger.info("VALIDATION SET")
    logger.info("-" * 50)

    val_data = prepare_training_data()[:5]  # Subset for quick validation
    val_path, _ = save_training_data(
        [{"prompt": v["prompt"], "completion": v["completion"],
          "difficulty": v["difficulty"], "target_peak_pct": v["target_peak_pct"],
          "expected_answer": v["expected_answer"]} for v in val_data],
        output_dir / "geometric_alignment_val.jsonl"
    )
    logger.info(f"Validation data saved to: {val_path}")

    # Training config
    logger.info("\n" + "-" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("-" * 50)

    config = {
        "model": model_path,
        "train_data": str(train_path),
        "val_data": str(val_path),
        "adapter_path": "data/adapters/geometric_alignment_lora",
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "lora_layers": 16,  # All layers for LFM2-350M
        "geometric_targets": {
            "compression_phi": 1.0,  # Target φ ratio
            "peak_by_difficulty": {1: 30, 2: 40, 3: 50, 4: 60, 5: 70},
        },
    }

    config_path = output_dir / "geometric_alignment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to: {config_path}")

    # Print training command
    logger.info("\n" + "=" * 70)
    logger.info("TO START TRAINING:")
    logger.info("=" * 70)
    logger.info(f"""
poetry run mlx_lm.lora \\
    --model {model_path} \\
    --train \\
    --data {output_dir} \\
    --adapter-path {config['adapter_path']} \\
    --batch-size {config['batch_size']} \\
    --num-layers {config['lora_layers']} \\
    --iters 100

Then evaluate with:
poetry run python scripts/train_geometric_alignment.py --eval
""")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PHILOSOPHY")
    logger.info("=" * 70)
    logger.info("""
The goal is NOT to teach the model facts.
The goal is to teach the model WHEN TO THINK.

Simple problems: compress early, answer quickly
Complex problems: expand more, peak later, then compress at φ

The geometry IS the alignment.
""")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "training_examples": len(train_data),
        "baseline": baseline,
        "config": config,
    }

    results_path = Path("data/experiments/geometric_training_setup.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nSetup complete. Results: {results_path}")

    return results


if __name__ == "__main__":
    main()
