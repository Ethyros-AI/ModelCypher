#!/usr/bin/env python3
"""Experiment 47: Curriculum-Based Cross-Architecture Teaching.

The insight from exp46: We're not transplanting weights, we're TEACHING.
Like how children learn - through examples, not brain surgery.

Question: Does a better "curriculum" improve the transplant?

Method (inspired by pedagogy):
1. DIVERSE examples (like a broad education)
2. PROGRESSIVE difficulty (simple → complex)
3. MULTIPLE domains (generalization across topics)
4. REINFORCEMENT through repetition (core concepts)

Hypothesis: A pedagogically-designed curriculum will outperform
random calibration prompts.

The physics connection:
- Random prompts = random sampling of the behavior manifold
- Curriculum = structured exploration that covers the manifold efficiently
- Like how a good teacher covers the "essential coordinates" of a subject
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Test curriculum-based transplant teaching."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    # Load models
    source_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    target_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"

    logger.info("Loading source model (DeepSeek-R1-8B)...")
    from mlx_lm import load
    source_model, source_tokenizer = load(source_path)

    logger.info("Loading target model (LFM2-1.2B)...")
    target_model, target_tokenizer = load(target_path)

    source_layer_idx = 24
    target_layer_idx = 10

    # ========================================
    # CURRICULUM DESIGN (like a good teacher)
    # ========================================

    # Level 1: Simple facts (foundation)
    level1_simple = [
        "The sky is blue",
        "Water is wet",
        "Fire is hot",
        "Ice is cold",
        "The sun rises",
        "Night is dark",
        "One plus one equals two",
        "A circle is round",
    ]

    # Level 2: Basic knowledge (building blocks)
    level2_basic = [
        "The capital of France is Paris",
        "Water freezes at zero degrees",
        "The largest planet is Jupiter",
        "Oxygen is needed for breathing",
        "Plants use photosynthesis",
        "The Earth orbits the sun",
        "Gravity pulls objects down",
        "Sound travels through air",
    ]

    # Level 3: Domain knowledge (specialization)
    level3_science = [
        "DNA contains genetic information",
        "Atoms have protons and electrons",
        "Energy cannot be created or destroyed",
        "Evolution occurs through natural selection",
        "Cells are the basic unit of life",
        "Light travels at constant speed",
        "Entropy always increases",
        "Quantum mechanics describes particles",
    ]

    level3_language = [
        "Shakespeare wrote many famous plays",
        "Poetry uses rhythm and rhyme",
        "Metaphors compare unlike things",
        "Grammar structures our sentences",
        "Language enables communication",
        "Stories have beginning middle end",
        "Words carry meaning and emotion",
        "Literature reflects human experience",
    ]

    level3_reasoning = [
        "If A then B means A implies B",
        "Correlation does not imply causation",
        "All squares are rectangles",
        "Logic requires valid premises",
        "Induction generalizes from examples",
        "Deduction derives from principles",
        "Probability measures uncertainty",
        "Mathematics describes patterns",
    ]

    # Level 4: Complex concepts (integration)
    level4_complex = [
        "The theory of relativity unifies space and time",
        "Neural networks learn from data",
        "Climate change affects global ecosystems",
        "Democracy requires informed citizens",
        "Economics balances supply and demand",
        "Philosophy examines fundamental questions",
        "History teaches lessons for the future",
        "Art expresses the human condition",
    ]

    # Curriculum: Progressive difficulty
    curriculum = level1_simple + level2_basic + level3_science + level3_language + level3_reasoning + level4_complex

    # Random baseline: Same number of prompts, shuffled
    random_prompts = curriculum.copy()
    np.random.seed(42)
    np.random.shuffle(random_prompts)

    # Test prompts (unseen during "teaching")
    test_prompts = [
        # Simple (should be easy)
        "The moon is",
        "Trees are",
        # Basic knowledge
        "Birds can",
        "Mountains are",
        # Science
        "Electrons orbit",
        "Genes contain",
        # Language
        "Stories tell",
        "Poetry expresses",
        # Reasoning
        "Therefore we",
        "Because of",
        # Complex
        "Technology enables",
        "Culture shapes",
    ]

    def get_activations(model, tokenizer, layer_idx, prompts):
        """Collect MLP activations."""
        inputs = []
        outputs = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            mlp_input = None
            mlp_output = None

            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                key = 'mlp'

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            if key == 'feed_forward':
                layer.feed_forward = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(np.array(mlp_input[0, -1, :].tolist(), dtype=np.float64))
                outputs.append(np.array(mlp_output[0, -1, :].tolist(), dtype=np.float64))
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        return np.stack(inputs), np.stack(outputs)

    def train_transplant(source_X, source_Y, target_X, target_Y):
        """Learn the transplant transform (teaching the target model)."""
        # Step 1: Find F that maps source outputs to target outputs
        F_out = np.linalg.lstsq(source_Y, target_Y, rcond=1e-10)[0]

        # Step 2: Project source behavior to target space
        source_behavior_in_target = source_Y @ F_out

        # Step 3: Learn W that reproduces this behavior from target inputs
        alpha = 1e-6
        ATA = target_X.T @ target_X + alpha * np.eye(target_X.shape[1])
        ATB = target_X.T @ source_behavior_in_target
        W = np.linalg.solve(ATA, ATB).T

        return W, F_out

    def evaluate_transplant(W, test_prompts, target_model, target_tokenizer, target_layer_idx):
        """Evaluate how well the transplant learned."""
        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class TransplantedMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        target_layer = target_model.model.layers[target_layer_idx]
        if hasattr(target_layer, 'feed_forward'):
            original_mlp = target_layer.feed_forward
            mlp_key = 'feed_forward'
        else:
            original_mlp = target_layer.mlp
            mlp_key = 'mlp'

        correct = 0
        results = []

        for prompt in test_prompts:
            tokens = target_tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            # Original prediction
            orig_logits = target_model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            # Transplanted prediction
            if mlp_key == 'feed_forward':
                target_layer.feed_forward = TransplantedMLP(W_mx)
            else:
                target_layer.mlp = TransplantedMLP(W_mx)

            try:
                trans_logits = target_model(input_ids)
                mx.eval(trans_logits)
                trans_top = int(mx.argmax(trans_logits[0, -1, :]).item())
            finally:
                if mlp_key == 'feed_forward':
                    target_layer.feed_forward = original_mlp
                else:
                    target_layer.mlp = original_mlp

            match = orig_top == trans_top
            if match:
                correct += 1
            results.append((prompt, match, orig_top, trans_top))

        return correct / len(test_prompts), results

    # ========================================
    # EXPERIMENT: Compare teaching strategies
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT: Curriculum vs Random Teaching")
    logger.info(f"{'='*80}")

    logger.info(f"\nCurriculum: {len(curriculum)} prompts, 4 levels of difficulty")
    logger.info(f"Random: Same {len(random_prompts)} prompts, shuffled order")

    # Collect source activations (the "teacher's" behavior)
    logger.info("\nCollecting source (teacher) activations...")
    source_X_curr, source_Y_curr = get_activations(source_model, source_tokenizer, source_layer_idx, curriculum)
    source_X_rand, source_Y_rand = get_activations(source_model, source_tokenizer, source_layer_idx, random_prompts)

    logger.info(f"Source activations: {source_X_curr.shape}")

    # Collect target activations (the "student's" responses to same prompts)
    logger.info("Collecting target (student) activations...")
    target_X_curr, target_Y_curr = get_activations(target_model, target_tokenizer, target_layer_idx, curriculum)
    target_X_rand, target_Y_rand = get_activations(target_model, target_tokenizer, target_layer_idx, random_prompts)

    logger.info(f"Target activations: {target_X_curr.shape}")

    # Train transplants ("teach" the student)
    logger.info("\nTraining transplants...")

    W_curr, F_curr = train_transplant(source_X_curr, source_Y_curr, target_X_curr, target_Y_curr)
    W_rand, F_rand = train_transplant(source_X_rand, source_Y_rand, target_X_rand, target_Y_rand)

    logger.info(f"Curriculum W shape: {W_curr.shape}")
    logger.info(f"Random W shape: {W_rand.shape}")

    # Evaluate ("test" the student)
    logger.info("\nEvaluating on unseen test prompts...")

    acc_curr, results_curr = evaluate_transplant(W_curr, test_prompts, target_model, target_tokenizer, target_layer_idx)
    acc_rand, results_rand = evaluate_transplant(W_rand, test_prompts, target_model, target_tokenizer, target_layer_idx)

    # Results
    logger.info(f"\n{'='*80}")
    logger.info("RESULTS: Teaching Strategy Comparison")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Strategy':<15} {'Accuracy':>10}")
    logger.info("-" * 30)
    logger.info(f"{'Curriculum':<15} {acc_curr*100:>9.1f}%")
    logger.info(f"{'Random':<15} {acc_rand*100:>9.1f}%")
    logger.info(f"{'Improvement':<15} {(acc_curr - acc_rand)*100:>+9.1f}pp")

    # Breakdown by difficulty level
    logger.info(f"\n--- Accuracy by Prompt Difficulty ---")

    level_map = {
        "The moon is": "simple", "Trees are": "simple",
        "Birds can": "basic", "Mountains are": "basic",
        "Electrons orbit": "science", "Genes contain": "science",
        "Stories tell": "language", "Poetry expresses": "language",
        "Therefore we": "reasoning", "Because of": "reasoning",
        "Technology enables": "complex", "Culture shapes": "complex",
    }

    levels = ["simple", "basic", "science", "language", "reasoning", "complex"]
    for level in levels:
        curr_correct = sum(1 for p, m, _, _ in results_curr if level_map[p] == level and m)
        rand_correct = sum(1 for p, m, _, _ in results_rand if level_map[p] == level and m)
        total = sum(1 for p in test_prompts if level_map[p] == level)
        logger.info(f"  {level:<12}: Curriculum {curr_correct}/{total}, Random {rand_correct}/{total}")

    # Detailed results
    logger.info(f"\n--- Per-Prompt Results ---")
    logger.info(f"{'Prompt':<25} {'Curr':>6} {'Rand':>6}")
    logger.info("-" * 40)
    for (p1, m1, _, _), (p2, m2, _, _) in zip(results_curr, results_rand):
        curr_mark = "✓" if m1 else "✗"
        rand_mark = "✓" if m2 else "✗"
        logger.info(f"{p1:<25} {curr_mark:>6} {rand_mark:>6}")

    # Analysis
    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS: The Pedagogy of Model Merging")
    logger.info(f"{'='*80}")

    logger.info(f"""
THE TEACHING ANALOGY:

1. CURRICULUM = Structured Knowledge
   - Simple facts (Level 1) = Building foundation
   - Domain knowledge (Level 3) = Specialization
   - Complex concepts (Level 4) = Integration
   Result: {acc_curr*100:.1f}% accuracy

2. RANDOM = Unstructured Exposure
   - Same information, no progression
   - No domain grouping
   - No difficulty scaling
   Result: {acc_rand*100:.1f}% accuracy

3. THE DIFFERENCE: {(acc_curr - acc_rand)*100:+.1f}pp
   {'Curriculum wins!' if acc_curr > acc_rand else 'Random wins!' if acc_rand > acc_curr else 'Tie!'}

WHY THIS MATTERS:

Cross-architecture merging is not about copying weights.
It's about TEACHING one model to behave like another.

Like human education:
- The student (LFM2) has its own neural structure
- The teacher (DeepSeek-R1) demonstrates behavior
- Learning = reconstructing capability, not copying neurons
- A good curriculum covers the "essential coordinates"

THE ESSENTIAL COORDINATES:

With k=6 dimensions capturing behavior:
- 6 "concepts" define the MLP's essential function
- A good curriculum samples all 6 dimensions
- Random sampling may miss some dimensions
- This is why curriculum design matters

NEXT STEPS:

1. Analyze which dimensions each curriculum level covers
2. Design minimal curriculum that covers all k dimensions
3. Test on other layer pairs
4. Scale to multi-layer teaching
""")


if __name__ == "__main__":
    run_experiment()
