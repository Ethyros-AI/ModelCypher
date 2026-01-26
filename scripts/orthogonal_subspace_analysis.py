#!/usr/bin/env python3
"""Experiment 20: Orthogonal Subspace Analysis.

Characterize the "safe" subspace - the directions orthogonal to preservation gradients.

Questions:
1. What's the dimensionality of the safe subspace?
2. Does it have geometric structure (constant ratios)?
3. Does it align with semantic separation directions from Exp 16?
4. How much modification capacity is "safe"?

Method:
1. Compute preservation gradients for multiple categories
2. Find the null space orthogonal to all preservation gradients
3. Analyze geometric structure of this subspace
4. Check alignment with semantic directions
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd, null_space

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CONSTANTS = {
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "1/phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "1/sqrt2": 1 / np.sqrt(2),
    "sqrt3": np.sqrt(3),
}


CATEGORY_QUESTIONS = {
    "math": [
        ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
        ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
        ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
        ("What is 3²?", ["6", "9", "12", "27"], 1),
        ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
    ],
    "geography": [
        ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
        ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
        ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
        ("Which country has the most people?", ["USA", "India", "China", "Russia"], 2),
        ("What river flows through Egypt?", ["Amazon", "Nile", "Yangtze", "Mississippi"], 1),
    ],
    "science": [
        ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
        ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
        ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
        ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
        ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Platinum"], 2),
    ],
    "history": [
        ("Who was the first US President?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
        ("In what year did WW2 end?", ["1943", "1944", "1945", "1946"], 2),
        ("What ancient wonder was in Egypt?", ["Colosseum", "Pyramids", "Parthenon", "Great Wall"], 1),
        ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Hemingway"], 2),
        ("What empire built the Colosseum?", ["Greek", "Persian", "Roman", "Ottoman"], 2),
    ],
    "logic": [
        ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
        ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
        ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
        ("Which is heavier: 1kg of feathers or 1kg of steel?", ["Feathers", "Steel", "Same weight", "Cannot compare"], 2),
        ("If today is Monday, what day was yesterday?", ["Tuesday", "Sunday", "Saturday", "Wednesday"], 1),
    ],
    "language": [
        ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
        ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
        ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
        ("Which is a synonym for 'happy'?", ["Sad", "Angry", "Joyful", "Tired"], 2),
        ("What punctuation ends a question?", ["Period", "Comma", "Question mark", "Exclamation"], 2),
    ],
    "common_sense": [
        ("What do you use to cut paper?", ["Hammer", "Scissors", "Spoon", "Brush"], 1),
        ("Where do fish live?", ["Trees", "Deserts", "Water", "Mountains"], 2),
        ("What meal do people eat in the morning?", ["Lunch", "Dinner", "Breakfast", "Snack"], 2),
        ("What color is grass usually?", ["Blue", "Red", "Green", "Yellow"], 2),
        ("How many days are in a week?", ["5", "6", "7", "8"], 2),
    ],
}


class OrthogonalSubspaceAnalysis:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self._original_weights = {}

    def _get_weight(self, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_weight(self, layer_idx: int, weights: np.ndarray):
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        new_weight = mx.array(weights.astype(np.float32))
        if hasattr(mlp, 'gate_proj'):
            mlp.gate_proj.weight = new_weight
        elif hasattr(mlp, 'w1'):
            mlp.w1.weight = new_weight
        else:
            mlp.weight = new_weight
        mx.eval(new_weight)

    def _cache_weights(self, layers: List[int]):
        self._original_weights = {i: self._get_weight(i).copy() for i in layers}

    def _reset_weights(self, layers: List[int]):
        for i in layers:
            if i in self._original_weights:
                self._set_weight(i, self._original_weights[i])

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> Tuple[bool, float]:
        """Evaluate a question, return (correct, loss)."""
        import mlx.core as mx

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        next_logits = logits[0, -1, :]

        choice_tokens = []
        for letter in ['A', 'B', 'C', 'D']:
            for variant in [f" {letter}", letter, f"{letter}."]:
                token_ids = self.tokenizer.encode(variant)
                if token_ids:
                    choice_tokens.append(token_ids[-1])
                    break
            else:
                choice_tokens.append(0)

        scores = [float(next_logits[t].item()) for t in choice_tokens]
        prediction = int(np.argmax(scores))

        # Compute cross-entropy loss for the correct answer
        scores_np = np.array(scores)
        probs = np.exp(scores_np - np.max(scores_np))
        probs = probs / probs.sum()
        loss = -np.log(probs[correct_idx] + 1e-10)

        return prediction == correct_idx, float(loss)

    def compute_loss_gradient(
        self,
        layer_idx: int,
        category: str,
        epsilon: float = 0.01,
        k: int = 20
    ) -> np.ndarray:
        """Compute gradient direction in SVD space for a category's loss."""
        W = self._original_weights[layer_idx]
        questions = CATEGORY_QUESTIONS[category]

        # Compute baseline loss
        self._set_weight(layer_idx, W)
        base_loss = sum(self._evaluate_question(q, c, idx)[1] for q, c, idx in questions)

        # Use SVD basis
        U, S, Vt = svd(W, full_matrices=False)

        # Compute gradient in top-k SVD directions
        k = min(k, len(S))
        gradient = np.zeros(k)

        for i in range(k):
            S_perturbed = S.copy()
            S_perturbed[i] += epsilon * S[i]

            W_perturbed = U @ np.diag(S_perturbed) @ Vt
            if np.all(np.isfinite(W_perturbed)):
                self._set_weight(layer_idx, W_perturbed)
                perturbed_loss = sum(self._evaluate_question(q, c, idx)[1] for q, c, idx in questions)
                gradient[i] = (perturbed_loss - base_loss) / (epsilon * S[i])

        self._set_weight(layer_idx, W)
        return gradient

    def compute_orthogonal_subspace(
        self,
        preserve_gradients: List[np.ndarray]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute the subspace orthogonal to all preservation gradients.

        Returns:
        - orthogonal_basis: columns form an orthonormal basis for the safe subspace
        - analysis: dimensionality and other metrics
        """
        if not preserve_gradients:
            return np.eye(len(preserve_gradients[0])), {"dimension": len(preserve_gradients[0])}

        # Stack gradients as rows
        k = len(preserve_gradients[0])
        G = np.array([g / np.linalg.norm(g) if np.linalg.norm(g) > 1e-10 else g
                      for g in preserve_gradients])

        # Find null space of G (directions orthogonal to all gradients)
        # Using SVD: null space is right singular vectors with zero singular values
        try:
            ortho_basis = null_space(G)
        except Exception:
            # Fallback: compute via SVD
            U, S, Vt = svd(G, full_matrices=True)
            rank = np.sum(S > 1e-10)
            ortho_basis = Vt[rank:].T

        safe_dim = ortho_basis.shape[1] if ortho_basis.ndim > 1 else 0
        total_dim = k

        return ortho_basis, {
            "total_dimensions": total_dim,
            "safe_dimensions": safe_dim,
            "preserved_dimensions": G.shape[0],
            "safe_fraction": safe_dim / total_dim if total_dim > 0 else 0,
        }

    def analyze_subspace_geometry(self, basis: np.ndarray) -> Dict:
        """Analyze if the orthogonal subspace has geometric structure (constant ratios)."""
        if basis.size == 0 or basis.ndim < 2 or basis.shape[1] == 0:
            return {"has_structure": False, "reason": "empty_subspace"}

        # Project weight matrix into this subspace and analyze
        # The basis vectors themselves may have structure
        # Check singular values of the basis
        try:
            _, S, _ = svd(basis, full_matrices=False)
        except Exception:
            return {"has_structure": False, "reason": "svd_failed"}

        # Count constant ratios in singular values
        const_matches = 0
        ratios = []
        for i in range(min(len(S) - 1, 10)):
            if S[i+1] > 1e-10:
                ratio = S[i] / S[i+1]
                ratios.append(float(ratio))
                for const_val in CONSTANTS.values():
                    if abs(ratio - const_val) / const_val < 0.10:
                        const_matches += 1
                        break

        return {
            "has_structure": const_matches > 0,
            "constant_matches": const_matches,
            "singular_values": [float(s) for s in S[:10]],
            "ratios": ratios,
        }

    def compute_semantic_directions(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Compute semantic separation directions (from Exp 16)."""
        import mlx.core as mx

        def capture_activations(prompt: str) -> np.ndarray:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            x = self.model.model.embed_tokens(input_ids)
            mx.eval(x)
            for i, layer in enumerate(self.model.model.layers):
                x = layer(x)
                mx.eval(x)
                if i == layer_idx:
                    return np.array(x.tolist())[0, -1, :]  # Last token
            return np.array(x.tolist())[0, -1, :]

        def format_question(q, choices):
            prompt = f"Question: {q}\n"
            for i, c in enumerate(choices):
                prompt += f"{chr(65+i)}. {c}\n"
            return prompt + "Answer:"

        # Collect mean activations per category
        category_means = {}
        for cat, questions in CATEGORY_QUESTIONS.items():
            acts = []
            for q, choices, _ in questions:
                prompt = format_question(q, choices)
                acts.append(capture_activations(prompt))
            category_means[cat] = np.mean(acts, axis=0)

        # Compute separation directions
        semantic_directions = {}
        categories = list(CATEGORY_QUESTIONS.keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                diff = category_means[cat1] - category_means[cat2]
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    semantic_directions[f"{cat1}_vs_{cat2}"] = diff / norm

        return semantic_directions

    def check_alignment_with_semantic(
        self,
        safe_basis: np.ndarray,
        semantic_directions: Dict[str, np.ndarray],
        layer_idx: int
    ) -> Dict:
        """Check if semantic directions align with the safe subspace."""
        if safe_basis.size == 0 or safe_basis.ndim < 2:
            return {"aligned": False, "reason": "empty_subspace"}

        W = self._original_weights[layer_idx]
        _, _, Vt = svd(W, full_matrices=False)
        V = Vt.T[:, :safe_basis.shape[0]]  # Input space singular vectors

        alignments = {}
        for name, sem_dir in semantic_directions.items():
            # Project semantic direction onto V (input space)
            projected = V.T @ sem_dir
            projected = projected[:safe_basis.shape[0]]  # Truncate to gradient space

            # Check alignment with safe subspace
            if safe_basis.shape[1] > 0:
                safe_proj = safe_basis @ (safe_basis.T @ projected)
                safe_alignment = np.linalg.norm(safe_proj) / (np.linalg.norm(projected) + 1e-10)
            else:
                safe_alignment = 0.0

            alignments[name] = float(safe_alignment)

        avg_alignment = np.mean(list(alignments.values())) if alignments else 0.0

        return {
            "alignments": alignments,
            "average_alignment": avg_alignment,
            "highly_aligned": [k for k, v in alignments.items() if v > 0.7],
        }

    def run_analysis(self) -> Dict:
        """Run full orthogonal subspace analysis."""
        mid = self.n_layers // 2
        layer_idx = mid  # Focus on middle layer

        self._cache_weights([layer_idx])

        logger.info("=" * 60)
        logger.info("ORTHOGONAL SUBSPACE ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Analyzing layer: {layer_idx}")

        results = {"layer": layer_idx}

        # Define categories to preserve (the "strong" ones)
        preserve_categories = ["geography", "history", "common_sense", "science"]
        improve_categories = ["math", "language", "logic"]

        logger.info(f"\nPreserve categories: {preserve_categories}")
        logger.info(f"Improve categories: {improve_categories}")

        # Compute preservation gradients
        logger.info("\nComputing preservation gradients...")
        preserve_gradients = []
        for cat in preserve_categories:
            logger.info(f"  {cat}...")
            grad = self.compute_loss_gradient(layer_idx, cat)
            preserve_gradients.append(grad)
            results[f"gradient_{cat}"] = {
                "norm": float(np.linalg.norm(grad)),
                "top_components": [int(i) for i in np.argsort(np.abs(grad))[::-1][:5]],
            }

        # Compute orthogonal subspace
        logger.info("\nComputing orthogonal subspace...")
        safe_basis, subspace_info = self.compute_orthogonal_subspace(preserve_gradients)
        logger.info(f"  Total dimensions: {subspace_info['total_dimensions']}")
        logger.info(f"  Safe dimensions: {subspace_info['safe_dimensions']}")
        logger.info(f"  Safe fraction: {subspace_info['safe_fraction']:.1%}")
        results["subspace_info"] = subspace_info

        # Analyze subspace geometry
        logger.info("\nAnalyzing subspace geometry...")
        geometry = self.analyze_subspace_geometry(safe_basis)
        logger.info(f"  Has constant ratio structure: {geometry['has_structure']}")
        if geometry.get('constant_matches', 0) > 0:
            logger.info(f"  Constant matches: {geometry['constant_matches']}")
        results["subspace_geometry"] = geometry

        # Compute semantic directions
        logger.info("\nComputing semantic separation directions...")
        semantic_dirs = self.compute_semantic_directions(layer_idx)
        logger.info(f"  Found {len(semantic_dirs)} semantic directions")

        # Check alignment
        logger.info("\nChecking alignment with safe subspace...")
        alignment = self.check_alignment_with_semantic(safe_basis, semantic_dirs, layer_idx)
        logger.info(f"  Average alignment: {alignment['average_alignment']:.2%}")
        if alignment.get('highly_aligned'):
            logger.info(f"  Highly aligned pairs: {alignment['highly_aligned']}")
        results["semantic_alignment"] = alignment

        # Test: can improvement gradients fit in safe subspace?
        logger.info("\nAnalyzing improvement gradients in safe subspace...")
        improve_in_safe = {}
        for cat in improve_categories:
            logger.info(f"  {cat}...")
            grad = self.compute_loss_gradient(layer_idx, cat)

            # Project into safe subspace
            if safe_basis.size > 0 and safe_basis.ndim >= 2 and safe_basis.shape[1] > 0:
                safe_proj = safe_basis @ (safe_basis.T @ grad)
                safe_ratio = np.linalg.norm(safe_proj) / (np.linalg.norm(grad) + 1e-10)
            else:
                safe_ratio = 0.0

            improve_in_safe[cat] = {
                "gradient_norm": float(np.linalg.norm(grad)),
                "safe_component_ratio": float(safe_ratio),
            }
            logger.info(f"    Safe component: {safe_ratio:.1%} of gradient")

        results["improvement_in_safe_subspace"] = improve_in_safe

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("KEY FINDINGS")
        logger.info("=" * 60)

        math_safe = improve_in_safe.get("math", {}).get("safe_component_ratio", 0)
        lang_safe = improve_in_safe.get("language", {}).get("safe_component_ratio", 0)

        if math_safe < 0.5:
            logger.info(f"\n*** MATH has only {math_safe:.0%} of its gradient in safe subspace ***")
            logger.info("This explains why math failed - most of its improvement direction")
            logger.info("is entangled with preservation gradients.")
        else:
            logger.info(f"\nMath has {math_safe:.0%} of gradient in safe subspace")

        if lang_safe > 0.6:
            logger.info(f"\n*** LANGUAGE has {lang_safe:.0%} of gradient in safe subspace ***")
            logger.info("This explains why language succeeded - most of its improvement")
            logger.info("direction is orthogonal to preservation gradients.")
        else:
            logger.info(f"\nLanguage has {lang_safe:.0%} of gradient in safe subspace")

        results["summary"] = {
            "math_safe_ratio": math_safe,
            "language_safe_ratio": lang_safe,
            "explanation": (
                f"Math: {math_safe:.0%} safe, Language: {lang_safe:.0%} safe. "
                f"The safe subspace has {subspace_info['safe_dimensions']} dimensions "
                f"({subspace_info['safe_fraction']:.0%} of total)."
            ),
        }

        self._reset_weights([layer_idx])

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    analyzer = OrthogonalSubspaceAnalysis(model, tokenizer)
    results = analyzer.run_analysis()

    # Save
    output_path = "data/experiments/orthogonal_subspace_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
