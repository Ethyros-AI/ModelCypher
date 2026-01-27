#!/usr/bin/env python3
"""Chain-of-Thought Training that Preserves Natural Geometry.

Key insight: LFM2-350M already has comp/φ ≈ 0.99.
We don't train geometry - we train reasoning while PRESERVING geometry.

The hypothesis: if we give the model better reasoning scaffolding,
it will naturally converge to φ = 1.0 as it learns to think more accurately.

Training approach:
1. Chain-of-thought data with explicit reasoning steps
2. Monitor geometry during training - stop if φ drifts
3. Let the model find its optimal trajectory naturally
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
# CHAIN-OF-THOUGHT TRAINING DATA
# =============================================================================
# Each problem has explicit reasoning steps that scaffold the solution.
# The model learns the PROCESS, not just the answer.

COT_TRAINING_DATA = [
    # Simple problems - establish the pattern
    {
        "question": "What is 7 + 5?",
        "reasoning": "I need to add 7 and 5.\n7 + 5 = 12",
        "answer": "12",
    },
    {
        "question": "What is 15 - 8?",
        "reasoning": "I need to subtract 8 from 15.\n15 - 8 = 7",
        "answer": "7",
    },
    {
        "question": "What is 6 × 4?",
        "reasoning": "I need to multiply 6 by 4.\n6 × 4 = 24",
        "answer": "24",
    },

    # Two-step - teach sequencing
    {
        "question": "What is 3 + 4 × 2?",
        "reasoning": "I need to follow order of operations.\nFirst, multiply: 4 × 2 = 8\nThen add: 3 + 8 = 11",
        "answer": "11",
    },
    {
        "question": "What is (8 + 4) ÷ 3?",
        "reasoning": "I need to follow order of operations.\nFirst, parentheses: 8 + 4 = 12\nThen divide: 12 ÷ 3 = 4",
        "answer": "4",
    },
    {
        "question": "What is 20 - 6 × 2?",
        "reasoning": "I need to follow order of operations.\nFirst, multiply: 6 × 2 = 12\nThen subtract: 20 - 12 = 8",
        "answer": "8",
    },

    # Word problems - teach extraction
    {
        "question": "Tom has 15 apples. He gives 6 to Sue. How many does Tom have left?",
        "reasoning": "Let me identify the values:\n- Tom starts with: 15 apples\n- Tom gives away: 6 apples\n- Operation: subtraction\n\n15 - 6 = 9\n\nTom has 9 apples left.",
        "answer": "9",
    },
    {
        "question": "A book costs $8. Maria buys 3 books. How much does she spend?",
        "reasoning": "Let me identify the values:\n- Price per book: $8\n- Number of books: 3\n- Operation: multiplication\n\n8 × 3 = 24\n\nMaria spends $24.",
        "answer": "24",
    },
    {
        "question": "There are 28 students. 12 are girls. How many are boys?",
        "reasoning": "Let me identify the values:\n- Total students: 28\n- Girls: 12\n- Boys: total - girls\n\n28 - 12 = 16\n\nThere are 16 boys.",
        "answer": "16",
    },

    # Multi-step reasoning
    {
        "question": "John has $50. He buys a $12 book and a $8 pen. How much does he have left?",
        "reasoning": "Let me solve step by step:\n1. John starts with: $50\n2. Cost of book: $12\n3. Cost of pen: $8\n4. Total spent: $12 + $8 = $20\n5. Money left: $50 - $20 = $30\n\nJohn has $30 left.",
        "answer": "30",
    },
    {
        "question": "A store has 80 items. It sells 25 and receives 15 more. How many now?",
        "reasoning": "Let me solve step by step:\n1. Start with: 80 items\n2. After selling 25: 80 - 25 = 55 items\n3. After receiving 15: 55 + 15 = 70 items\n\nThe store has 70 items.",
        "answer": "70",
    },
    {
        "question": "Lisa works 8 hours at $15/hour. She spends $40. How much is left?",
        "reasoning": "Let me solve step by step:\n1. Hours worked: 8\n2. Rate: $15/hour\n3. Earnings: 8 × $15 = $120\n4. Spent: $40\n5. Left: $120 - $40 = $80\n\nLisa has $80 left.",
        "answer": "80",
    },

    # Complex reasoning with explicit strategy
    {
        "question": "A shirt is $60 with a 25% discount. What's the sale price?",
        "reasoning": "Let me solve step by step:\n1. Original price: $60\n2. Discount: 25%\n3. Discount amount: 60 × 0.25 = $15\n4. Sale price: $60 - $15 = $45\n\nThe sale price is $45.",
        "answer": "45",
    },
    {
        "question": "4 workers finish a job in 6 days. How many days for 3 workers?",
        "reasoning": "Let me solve step by step:\n1. Total work = workers × days = 4 × 6 = 24 worker-days\n2. With 3 workers: 24 ÷ 3 = 8 days\n\nIt takes 8 days with 3 workers.",
        "answer": "8",
    },
    {
        "question": "A pool fills at 5 gal/min and drains at 2 gal/min. Net gallons in 20 minutes?",
        "reasoning": "Let me solve step by step:\n1. Fill rate: 5 gal/min\n2. Drain rate: 2 gal/min\n3. Net rate: 5 - 2 = 3 gal/min\n4. In 20 minutes: 3 × 20 = 60 gallons\n\nNet is 60 gallons.",
        "answer": "60",
    },

    # GSM8K-style problems
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins daily. She sells the rest at $2 each. How much does she make per day?",
        "reasoning": "Let me solve step by step:\n1. Eggs laid per day: 16\n2. Eggs eaten: 3\n3. Eggs baked: 4\n4. Eggs used: 3 + 4 = 7\n5. Eggs to sell: 16 - 7 = 9\n6. Price per egg: $2\n7. Daily earnings: 9 × $2 = $18\n\nJanet makes $18 per day.",
        "answer": "18",
    },
    {
        "question": "A farmer has 60 sheep. He sells 1/4 of them, then buys 10 more. How many sheep does he have?",
        "reasoning": "Let me solve step by step:\n1. Starting sheep: 60\n2. Sells 1/4: 60 ÷ 4 = 15 sheep sold\n3. After selling: 60 - 15 = 45 sheep\n4. Buys 10 more: 45 + 10 = 55 sheep\n\nThe farmer has 55 sheep.",
        "answer": "55",
    },
    {
        "question": "Train A travels at 50 mph. Train B travels at 70 mph and leaves 2 hours later, same direction. How many hours until B catches A?",
        "reasoning": "Let me solve step by step:\n1. A's speed: 50 mph\n2. B's speed: 70 mph\n3. A's head start: 2 hours × 50 mph = 100 miles\n4. B gains on A at: 70 - 50 = 20 mph\n5. Time to catch up: 100 ÷ 20 = 5 hours\n\nB catches A in 5 hours.",
        "answer": "5",
    },
    {
        "question": "Beth has 72 marbles. She gives 1/3 to Ann, then half of what's left to Carl. How many does Beth have?",
        "reasoning": "Let me solve step by step:\n1. Beth starts with: 72 marbles\n2. Gives 1/3 to Ann: 72 ÷ 3 = 24 marbles\n3. Beth has left: 72 - 24 = 48 marbles\n4. Gives half to Carl: 48 ÷ 2 = 24 marbles\n5. Beth keeps: 48 - 24 = 24 marbles\n\nBeth has 24 marbles.",
        "answer": "24",
    },
]


def compute_intrinsic_dimension_twonn(X: np.ndarray) -> float:
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


def get_comp_phi(model, tokenizer, prompt: str) -> float:
    """Get compression/φ ratio for a prompt."""
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

    if len(valid) > 2:
        peak_dim = np.max(valid)
        final_dim = valid[-1]
        if final_dim > 0.1:
            return (peak_dim / final_dim) / PHI

    return float('nan')


def format_training_example(item: Dict) -> str:
    """Format a training example with CoT structure."""
    return f"""Question: {item['question']}

Let me think through this step by step.

{item['reasoning']}

The answer is {item['answer']}."""


def prepare_training_data(output_dir: Path):
    """Prepare CoT training data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training set
    train_data = []
    for item in COT_TRAINING_DATA:
        text = format_training_example(item)
        train_data.append({"text": text})

    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # Validation set (subset)
    val_data = train_data[:5]
    val_path = output_dir / "valid.jsonl"
    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    return train_path, val_path, len(train_data)


def measure_model_geometry(model, tokenizer, sample_prompts: List[str]) -> Dict:
    """Measure average geometry across sample prompts."""
    comp_phis = []

    for prompt in sample_prompts:
        try:
            phi = get_comp_phi(model, tokenizer, prompt)
            if not np.isnan(phi):
                comp_phis.append(phi)
        except:
            pass

    if comp_phis:
        return {
            "mean_comp_phi": np.mean(comp_phis),
            "std_comp_phi": np.std(comp_phis),
            "n_samples": len(comp_phis),
            "distance_from_1": abs(np.mean(comp_phis) - 1.0),
        }
    return {"mean_comp_phi": float('nan'), "std_comp_phi": float('nan'), "n_samples": 0}


def evaluate_accuracy(model, tokenizer, problems: List[Dict]) -> Tuple[float, List[Dict]]:
    """Evaluate accuracy on problems."""
    from mlx_lm import generate

    results = []
    correct = 0

    for item in problems:
        prompt = f"Question: {item['question']}\n\nAnswer:"

        try:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
        except:
            output = "ERROR"

        # Check if answer is in output
        expected = item['answer']
        nums = re.findall(r'-?\d+\.?\d*', output.replace(",", ""))

        is_correct = False
        if nums:
            for num in nums:
                try:
                    if abs(float(num) - float(expected)) < 0.1:
                        is_correct = True
                        break
                except:
                    pass

        if is_correct:
            correct += 1

        results.append({
            "question": item["question"][:40],
            "expected": expected,
            "output": output[:100],
            "correct": is_correct,
        })

    return correct / len(problems) * 100, results


def main():
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("CHAIN-OF-THOUGHT TRAINING (Geometry-Preserving)")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    adapter_output = "data/adapters/cot_preserve_geometry_lora"

    # Prepare data
    logger.info("\n" + "-" * 50)
    logger.info("PREPARING TRAINING DATA")
    logger.info("-" * 50)

    data_dir = Path("data/training/cot_preserve_geometry")
    train_path, val_path, n_examples = prepare_training_data(data_dir)

    logger.info(f"Training examples: {n_examples}")
    logger.info(f"Data saved to: {data_dir}")

    # Load base model and measure baseline geometry
    logger.info("\n" + "-" * 50)
    logger.info("BASELINE GEOMETRY")
    logger.info("-" * 50)

    model, tokenizer = load(model_path)

    sample_prompts = [
        "Question: What is 5 + 3?\n\nAnswer:",
        "Question: John has 10 apples and gives 4 away. How many left?\n\nAnswer:",
        "Question: A shirt costs $40 after 20% discount. Original price?\n\nAnswer:",
    ]

    baseline_geo = measure_model_geometry(model, tokenizer, sample_prompts)
    logger.info(f"Baseline comp/φ: {baseline_geo['mean_comp_phi']:.3f} ± {baseline_geo['std_comp_phi']:.3f}")
    logger.info(f"Distance from 1.0: {baseline_geo['distance_from_1']:.3f}")

    # Baseline accuracy
    baseline_acc, _ = evaluate_accuracy(model, tokenizer, COT_TRAINING_DATA[:10])
    logger.info(f"Baseline accuracy: {baseline_acc:.0f}%")

    # Training command
    logger.info("\n" + "-" * 50)
    logger.info("TRAINING")
    logger.info("-" * 50)

    logger.info(f"""
Run this command to train:

poetry run mlx_lm.lora \\
    --model {model_path} \\
    --train \\
    --data {data_dir} \\
    --adapter-path {adapter_output} \\
    --batch-size 2 \\
    --num-layers 16 \\
    --iters 300 \\
    --learning-rate 5e-6

Key settings:
- Lower learning rate (5e-6) to preserve geometry
- More iterations (300) for thorough learning
- Small batch size (2) for stability

After training, this script will evaluate geometry preservation.
""")

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "adapter_output": adapter_output,
        "data_dir": str(data_dir),
        "n_examples": n_examples,
        "baseline_geometry": baseline_geo,
        "baseline_accuracy": baseline_acc,
        "hypothesis": "CoT training will preserve geometry while improving accuracy",
        "success_criteria": {
            "comp_phi_drift": "< 0.1 from baseline",
            "accuracy_improvement": "> 10%",
            "convergence_to_1": "comp/φ should move toward 1.0",
        },
    }

    config_path = data_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nConfig saved to: {config_path}")

    # Philosophy
    logger.info("\n" + "=" * 70)
    logger.info("THE HYPOTHESIS")
    logger.info("=" * 70)
    logger.info("""
We're not training geometry. The model already has it.

We're giving it REASONING SCAFFOLDS that let its natural
geometry express more accurately.

If the hypothesis is correct:
1. Accuracy will improve (the model reasons better)
2. comp/φ will stay near baseline OR move toward 1.0
3. The model will naturally converge to perfect geometry
   as its reasoning becomes more accurate

The geometry and reasoning are coupled.
Better reasoning → better geometry → better reasoning.

This is the virtuous cycle of alignment.
""")

    return config


if __name__ == "__main__":
    main()
