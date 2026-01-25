#!/usr/bin/env python3
"""Experiment 62: Reciprocal Teaching.

The discovery from exp61: Teaching is BIDIRECTIONAL.

- DeepSeek-R1-8B is better at: reasoning, science, language
- LFM2-1.2B is better at: math, world_knowledge

This is like a study group where each student has different strengths!

The experiment:
1. DeepSeek teaches LFM2 its strong domains
2. LFM2 teaches DeepSeek its strong domains
3. Both models improve through geometric knowledge exchange

No tokens. No training. Just manifold geometry.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.linalg import svd
from scipy.special import softmax

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def spectral_entropy(Y):
    """Compute entropy from singular value spectrum."""
    Y_centered = Y - Y.mean(axis=0)
    _, S, _ = svd(Y_centered, full_matrices=False)
    S_norm = S / np.sum(S)
    S_norm = S_norm[S_norm > 1e-10]
    return -np.sum(S_norm * np.log(S_norm))


def run_experiment():
    """Explore reciprocal teaching between models."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    # Load both models
    model_a_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    model_b_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"

    logger.info("Loading Model A (DeepSeek-R1-8B)...")
    from mlx_lm import load
    model_a, tokenizer_a = load(model_a_path)

    logger.info("Loading Model B (LFM2-1.2B)...")
    model_b, tokenizer_b = load(model_b_path)

    # Domain expertise mapping (from exp61)
    # Positive gap = A teaches B, Negative gap = B teaches A
    domain_expertise = {
        "reasoning": {"gap": +0.009, "teacher": "A", "probes": [
            "If A implies B and B implies C, then",
            "The logical conclusion is",
            "Therefore, we can deduce that",
            "By contrapositive reasoning,",
        ]},
        "science": {"gap": +0.004, "teacher": "A", "probes": [
            "The second law of thermodynamics",
            "Quantum entanglement occurs when",
            "The speed of light in a vacuum",
            "Entropy always",
        ]},
        "language": {"gap": +0.021, "teacher": "A", "probes": [
            "The grammatical structure of",
            "A metaphor differs from a simile",
            "The passive voice is used",
            "Semantic meaning differs from",
        ]},
        "math": {"gap": -0.023, "teacher": "B", "probes": [
            "The derivative of x squared is",
            "The integral of 1/x is",
            "The Pythagorean theorem states",
            "A prime number is",
        ]},
        "world_knowledge": {"gap": -0.003, "teacher": "B", "probes": [
            "The capital of France is",
            "World War II ended in",
            "The largest ocean is",
            "Shakespeare wrote",
        ]},
    }

    # Layer pairs for each model
    # A (DeepSeek-R1): 36 layers, golden zone 22-30
    # B (LFM2): 16 layers, golden zone 9-13
    layer_a = 24
    layer_b = 10

    def get_layer_activations(model, tokenizer, layer_idx, prompts):
        """Get MLP input and output activations."""
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

    def compute_teaching_weights(teacher_Y, student_X, student_Y, direction_idx):
        """Compute weights after teaching direction d."""
        teacher_centered = teacher_Y - teacher_Y.mean(axis=0)
        student_centered = student_Y - student_Y.mean(axis=0)

        _, _, Vh_t = svd(teacher_centered, full_matrices=False)
        _, _, Vh_s = svd(student_centered, full_matrices=False)

        F = np.linalg.lstsq(teacher_Y, student_Y, rcond=1e-10)[0]

        result = student_Y.copy()
        d = direction_idx

        if d < len(Vh_t) and d < len(Vh_s):
            # Remove student's direction d
            student_coefs_d = student_centered @ Vh_s[d]
            student_proj_d = np.outer(student_coefs_d, Vh_s[d])
            result -= student_proj_d

            # Add teacher's direction d (translated)
            teacher_coefs_d = teacher_centered @ Vh_t[d]
            teacher_proj_d = np.outer(teacher_coefs_d, Vh_t[d])
            result += teacher_proj_d @ F

        # Solve for weights
        alpha = 1e-6
        ATA = student_X.T @ student_X + alpha * np.eye(student_X.shape[1])
        ATB = student_X.T @ result
        W = np.linalg.solve(ATA, ATB).T

        return W

    def get_prediction(model, tokenizer, prompt):
        """Get model's top prediction."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        top_token = int(mx.argmax(logits[0, -1, :]).item())
        word = tokenizer.decode([top_token]).strip()
        return top_token, word

    # ========================================
    # PHASE 1: Baseline analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Domain Expertise Analysis")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Domain':<20} {'Gap':>10} {'Teacher':>10} {'Can Teach':>15}")
    logger.info("-" * 60)

    a_teaches = []
    b_teaches = []

    for domain, info in domain_expertise.items():
        teacher = info['teacher']
        gap = info['gap']
        can_teach = "A → B" if teacher == "A" else "B → A"

        if teacher == "A":
            a_teaches.append(domain)
        else:
            b_teaches.append(domain)

        logger.info(f"{domain:<20} {gap:>+10.4f} {teacher:>10} {can_teach:>15}")

    logger.info(f"\nModel A (DeepSeek-R1) teaches: {', '.join(a_teaches)}")
    logger.info(f"Model B (LFM2) teaches: {', '.join(b_teaches)}")

    # ========================================
    # PHASE 2: A teaches B
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Model A Teaches Model B")
    logger.info(f"{'='*80}")

    direction_to_teach = 5  # Direction 6

    for domain in a_teaches:
        probes = domain_expertise[domain]['probes']

        logger.info(f"\n--- Teaching: {domain.upper()} ---")

        # Get activations
        A_X, A_Y = get_layer_activations(model_a, tokenizer_a, layer_a, probes)
        B_X, B_Y = get_layer_activations(model_b, tokenizer_b, layer_b, probes)

        # Before entropy
        before_entropy = spectral_entropy(B_Y)

        # Compute teaching
        W_taught = compute_teaching_weights(A_Y, B_X, B_Y, direction_to_teach)

        # Create taught MLP
        W_mx = mx.array(W_taught.astype(np.float32))
        mx.eval(W_mx)

        class TaughtMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        # Get new activations with taught MLP
        layer_b_obj = model_b.model.layers[layer_b]
        if hasattr(layer_b_obj, 'feed_forward'):
            original_mlp = layer_b_obj.feed_forward
            mlp_key = 'feed_forward'
        else:
            original_mlp = layer_b_obj.mlp
            mlp_key = 'mlp'

        # Measure after entropy
        if mlp_key == 'feed_forward':
            layer_b_obj.feed_forward = TaughtMLP(W_mx)
        else:
            layer_b_obj.mlp = TaughtMLP(W_mx)

        try:
            _, B_Y_after = get_layer_activations(model_b, tokenizer_b, layer_b, probes)
            after_entropy = spectral_entropy(B_Y_after)
        finally:
            if mlp_key == 'feed_forward':
                layer_b_obj.feed_forward = original_mlp
            else:
                layer_b_obj.mlp = original_mlp

        delta = before_entropy - after_entropy
        logger.info(f"Entropy: {before_entropy:.4f} → {after_entropy:.4f} (ΔH = {delta:+.4f})")

    # ========================================
    # PHASE 3: B teaches A
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Model B Teaches Model A")
    logger.info(f"{'='*80}")

    for domain in b_teaches:
        probes = domain_expertise[domain]['probes']

        logger.info(f"\n--- Teaching: {domain.upper()} ---")

        # Get activations (note: B is now teacher, A is student)
        B_X, B_Y = get_layer_activations(model_b, tokenizer_b, layer_b, probes)
        A_X, A_Y = get_layer_activations(model_a, tokenizer_a, layer_a, probes)

        # Before entropy
        before_entropy = spectral_entropy(A_Y)

        # Compute teaching (B teaches A)
        W_taught = compute_teaching_weights(B_Y, A_X, A_Y, direction_to_teach)

        # Create taught MLP
        W_mx = mx.array(W_taught.astype(np.float32))
        mx.eval(W_mx)

        class TaughtMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        # Get layer for A
        layer_a_obj = model_a.model.layers[layer_a]
        if hasattr(layer_a_obj, 'feed_forward'):
            original_mlp = layer_a_obj.feed_forward
            mlp_key = 'feed_forward'
        else:
            original_mlp = layer_a_obj.mlp
            mlp_key = 'mlp'

        # Measure after entropy
        if mlp_key == 'feed_forward':
            layer_a_obj.feed_forward = TaughtMLP(W_mx)
        else:
            layer_a_obj.mlp = TaughtMLP(W_mx)

        try:
            _, A_Y_after = get_layer_activations(model_a, tokenizer_a, layer_a, probes)
            after_entropy = spectral_entropy(A_Y_after)
        finally:
            if mlp_key == 'feed_forward':
                layer_a_obj.feed_forward = original_mlp
            else:
                layer_a_obj.mlp = original_mlp

        delta = before_entropy - after_entropy
        logger.info(f"Entropy: {before_entropy:.4f} → {after_entropy:.4f} (ΔH = {delta:+.4f})")

    # ========================================
    # PHASE 4: Knowledge pool demonstration
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: The Knowledge Pool")
    logger.info(f"{'='*80}")

    # Test prompts across all domains
    test_cases = [
        ("reasoning", "If all cats are mammals, then cats are"),
        ("science", "The nucleus of an atom contains"),
        ("math", "The square root of 144 is"),
        ("language", "A verb describes an"),
        ("world_knowledge", "The Eiffel Tower is located in"),
    ]

    logger.info(f"\n{'Domain':<15} {'Prompt':<40} {'A says':>12} {'B says':>12} {'Expert':>8}")
    logger.info("-" * 95)

    for domain, prompt in test_cases:
        _, word_a = get_prediction(model_a, tokenizer_a, prompt)
        _, word_b = get_prediction(model_b, tokenizer_b, prompt)
        expert = domain_expertise[domain]['teacher']

        logger.info(f"{domain:<15} {prompt:<40} {word_a[:10]:>12} {word_b[:10]:>12} {expert:>8}")

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Reciprocal Teaching")
    logger.info(f"{'='*80}")

    logger.info(f"""
THE RECIPROCAL TEACHING PARADIGM:

Each model has DIFFERENT strengths:

Model A (DeepSeek-R1-8B):
  - Strong: reasoning, science, language
  - Can teach these to B

Model B (LFM2-1.2B):
  - Strong: math, world_knowledge
  - Can teach these to A

THE KNOWLEDGE POOL:

Instead of:
  A teaches B (one-way distillation)

We have:
  A ⇄ B (reciprocal exchange)

Each model contributes its expertise.
The result: both models improve!

THE MATH:

For each domain D:
  1. Identify teacher T and student S based on entropy gap
  2. Extract teacher's principal directions for D
  3. Replace student's corresponding directions
  4. Both models gain knowledge

IMPLICATIONS:

1. SIZE ≠ CAPABILITY
   - Smaller models can be "experts" in specific domains
   - Larger models aren't universally better

2. KNOWLEDGE IS MODULAR
   - Different domains live in different directions
   - Directions can be transferred independently

3. ENSEMBLE THROUGH GEOMETRY
   - No need to run both models at inference
   - Transfer knowledge once, use forever

4. SCALABLE TO N MODELS
   - Each model contributes its strengths
   - Pool grows with diversity, not size

This is the future: Geometric Knowledge Pooling.
""")


if __name__ == "__main__":
    run_experiment()
