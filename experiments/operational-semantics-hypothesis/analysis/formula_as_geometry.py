#!/usr/bin/env python3
"""
Test the strong form of the hypothesis:

The Pythagorean formula a² + b² = c² is LITERALLY encoded in the latent geometry.

Not just clustering - the actual positions satisfy the mathematical relationship.

Possible encodings:
1. ||embed(a)||² + ||embed(b)||² ≈ ||embed(c)||²  (norm encodes magnitude)
2. The "squared numbers" form a manifold, and valid triples lie on a constraint surface
3. There exists a transformation T such that T(embed(a), embed(b)) = embed(c) for all valid triples
4. The cosine angle between directions encodes the relationship

We're looking for: the formula is COMPUTABLE from the geometry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

# Add ModelCypher to path
sys.path.insert(0, str(Path(__file__).parents[3] / "ModelCypher" / "src"))

SCRIPT_DIR = Path(__file__).parent.parent


def load_model(model_path: str):
    """Load model and tokenizer."""
    from mlx_lm import load
    return load(model_path)


def extract_embedding(model, tokenizer, text: str) -> np.ndarray:
    """Extract last-token embedding."""
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    # Forward pass
    if hasattr(model, 'model'):
        inner = model.model
    else:
        inner = model

    # Get embeddings
    if hasattr(inner, 'embed_tokens'):
        x = inner.embed_tokens(input_ids)
    else:
        x = inner.wte(input_ids)

    # Pass through layers
    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    for layer in layers:
        x = layer(x)
        if isinstance(x, tuple):
            x = x[0]

    # Extract last token, evaluate, and convert to float32 for numpy
    result = x[0, -1, :]
    mx.eval(result)
    # Convert to float32 to avoid bf16 numpy issues
    result = result.astype(mx.float32)
    mx.eval(result)
    return np.array(result, dtype=np.float32)


class FormulaGeometryTester:
    """Test if the Pythagorean formula is encoded in latent geometry."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.number_embeddings = {}
        self.squared_embeddings = {}

    def embed_number(self, n: int) -> np.ndarray:
        """Get embedding for a number."""
        if n not in self.number_embeddings:
            text = f"The number {n}."
            self.number_embeddings[n] = extract_embedding(
                self.model, self.tokenizer, text
            )
        return self.number_embeddings[n]

    def embed_squared(self, n: int) -> np.ndarray:
        """Get embedding for n² as a concept."""
        if n not in self.squared_embeddings:
            text = f"{n} squared equals {n*n}."
            self.squared_embeddings[n] = extract_embedding(
                self.model, self.tokenizer, text
            )
        return self.squared_embeddings[n]

    def test_norm_encoding(self) -> dict:
        """
        Test if ||embed(n)||² encodes n² proportionally.

        If true, then for valid triples:
        ||embed(a)||² + ||embed(b)||² ≈ ||embed(c)||²
        """
        print("\n=== Test: Norm² Encodes Magnitude ===")

        numbers = list(range(1, 26))
        norms = []
        for n in numbers:
            emb = self.embed_number(n)
            norms.append(np.linalg.norm(emb))

        # Check if norm correlates with n or n²
        corr_linear, _ = spearmanr(numbers, norms)
        corr_squared, _ = spearmanr([n**2 for n in numbers], [norm**2 for norm in norms])

        print(f"Correlation(n, ||embed(n)||): {corr_linear:.4f}")
        print(f"Correlation(n², ||embed(n)||²): {corr_squared:.4f}")

        # Test Pythagorean formula directly
        triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
        formula_errors = []

        for a, b, c in triples:
            norm_a = np.linalg.norm(self.embed_number(a))
            norm_b = np.linalg.norm(self.embed_number(b))
            norm_c = np.linalg.norm(self.embed_number(c))

            lhs = norm_a**2 + norm_b**2
            rhs = norm_c**2
            error = abs(lhs - rhs) / rhs
            formula_errors.append(error)
            print(f"  ({a},{b},{c}): ||a||² + ||b||² = {lhs:.2f}, ||c||² = {rhs:.2f}, error = {error:.2%}")

        return {
            "correlation_linear": float(corr_linear),
            "correlation_squared": float(corr_squared),
            "formula_errors": formula_errors,
            "mean_formula_error": float(np.mean(formula_errors)),
            "encodes_formula": np.mean(formula_errors) < 0.1
        }

    def test_direction_encoding(self) -> dict:
        """
        Test if there's a consistent "squaring direction" in latent space.

        If embed(n²) - embed(n) has a consistent direction, then squaring
        is encoded as a geometric operation.
        """
        print("\n=== Test: Squaring as Direction ===")

        # Compute "squaring vectors" for several numbers
        squaring_vectors = []
        for n in [2, 3, 4, 5, 6, 7]:
            n_emb = self.embed_number(n)
            n_sq_emb = self.embed_number(n * n)
            diff = n_sq_emb - n_emb
            diff_norm = diff / (np.linalg.norm(diff) + 1e-8)
            squaring_vectors.append(diff_norm)

        # Check consistency
        similarities = []
        for i in range(len(squaring_vectors)):
            for j in range(i + 1, len(squaring_vectors)):
                sim = np.dot(squaring_vectors[i], squaring_vectors[j])
                similarities.append(sim)

        mean_consistency = np.mean(similarities)
        print(f"Mean direction consistency: {mean_consistency:.4f}")

        return {
            "direction_consistency": float(mean_consistency),
            "consistent_squaring_direction": mean_consistency > 0.5
        }

    def test_pythagorean_surface(self) -> dict:
        """
        Test if valid Pythagorean triples lie on a specific surface
        that invalid triples don't.

        For valid (a,b,c): there should be a consistent relationship
        between embed(a), embed(b), embed(c) that doesn't hold for invalid.
        """
        print("\n=== Test: Pythagorean Constraint Surface ===")

        valid_triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (9, 40, 41)]
        invalid_triples = [(3, 4, 6), (5, 12, 14), (8, 15, 18), (7, 24, 26), (9, 40, 42)]

        def compute_triple_features(a, b, c) -> np.ndarray:
            """Compute geometric features for a triple."""
            emb_a = self.embed_number(a)
            emb_b = self.embed_number(b)
            emb_c = self.embed_number(c)

            # Feature 1: ||a + b - c|| (if addition maps, this should be small for valid)
            add_residual = np.linalg.norm(emb_a + emb_b - emb_c)

            # Feature 2: Cosine(a+b, c)
            ab_sum = emb_a + emb_b
            cos_abc = np.dot(ab_sum, emb_c) / (np.linalg.norm(ab_sum) * np.linalg.norm(emb_c) + 1e-8)

            # Feature 3: ||a||² + ||b||² - ||c||² (Pythagorean in norm space)
            norm_residual = np.linalg.norm(emb_a)**2 + np.linalg.norm(emb_b)**2 - np.linalg.norm(emb_c)**2

            # Feature 4: Triangle inequality ratios
            d_ab = np.linalg.norm(emb_a - emb_b)
            d_ac = np.linalg.norm(emb_a - emb_c)
            d_bc = np.linalg.norm(emb_b - emb_c)

            return np.array([add_residual, cos_abc, norm_residual, d_ab, d_ac, d_bc])

        valid_features = [compute_triple_features(a, b, c) for a, b, c in valid_triples]
        invalid_features = [compute_triple_features(a, b, c) for a, b, c in invalid_triples]

        valid_mean = np.mean(valid_features, axis=0)
        invalid_mean = np.mean(invalid_features, axis=0)

        print("Feature means (valid vs invalid):")
        feature_names = ["||a+b-c||", "cos(a+b,c)", "||a||²+||b||²-||c||²", "d(a,b)", "d(a,c)", "d(b,c)"]
        separations = []
        for i, name in enumerate(feature_names):
            diff = abs(valid_mean[i] - invalid_mean[i])
            avg = (abs(valid_mean[i]) + abs(invalid_mean[i])) / 2 + 1e-8
            separation = diff / avg
            separations.append(separation)
            print(f"  {name}: valid={valid_mean[i]:.4f}, invalid={invalid_mean[i]:.4f}, sep={separation:.2%}")

        # Train a simple classifier
        from sklearn.linear_model import LogisticRegression
        X = np.vstack(valid_features + invalid_features)
        y = [1] * len(valid_features) + [0] * len(invalid_features)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        accuracy = clf.score(X, y)

        print(f"\nClassifier accuracy (valid vs invalid): {accuracy:.2%}")

        return {
            "feature_separations": {name: float(sep) for name, sep in zip(feature_names, separations)},
            "classifier_accuracy": float(accuracy),
            "surface_separates_valid": accuracy > 0.8
        }

    def test_formula_recovery(self) -> dict:
        """
        The strongest test: Can we RECOVER the Pythagorean formula from embeddings?

        Given embed(a), embed(b), embed(c), can we find a function f such that:
        f(embed(a), embed(b), embed(c)) = 0 iff a² + b² = c²
        """
        print("\n=== Test: Formula Recovery ===")

        # Generate training data: valid and invalid triples
        valid = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
        invalid = [(3, 4, 6), (5, 12, 14), (8, 15, 18), (7, 24, 26)]

        # For each triple, compute: a² + b² - c² (should be 0 for valid)
        def actual_residual(a, b, c):
            return a**2 + b**2 - c**2

        # Try to predict this residual from embeddings
        X = []
        y = []
        for a, b, c in valid + invalid:
            emb_a = self.embed_number(a)
            emb_b = self.embed_number(b)
            emb_c = self.embed_number(c)
            # Concatenate embeddings as features
            X.append(np.concatenate([emb_a, emb_b, emb_c]))
            y.append(actual_residual(a, b, c))

        X = np.array(X)
        y = np.array(y)

        # Linear regression: can we predict the Pythagorean residual?
        X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        W, residuals, rank, s = np.linalg.lstsq(X_with_bias, y, rcond=None)

        y_pred = X_with_bias @ W
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)

        print(f"R² for predicting a² + b² - c² from embeddings: {r_squared:.4f}")

        # Test on novel triples
        novel_valid = [(9, 40, 41), (11, 60, 61)]
        novel_invalid = [(9, 40, 42), (11, 60, 62)]

        novel_errors = []
        for a, b, c in novel_valid + novel_invalid:
            emb_a = self.embed_number(a)
            emb_b = self.embed_number(b)
            emb_c = self.embed_number(c)
            x = np.concatenate([emb_a, emb_b, emb_c, [1]])
            pred = x @ W
            actual = actual_residual(a, b, c)
            error = abs(pred - actual)
            novel_errors.append(error)
            print(f"  ({a},{b},{c}): predicted={pred:.1f}, actual={actual}, error={error:.1f}")

        return {
            "r_squared": float(r_squared),
            "novel_errors": [float(e) for e in novel_errors],
            "formula_recoverable": r_squared > 0.7
        }

    def test_concept_position_encoding(self) -> dict:
        """
        The deepest test: Is the POSITION of "5" relative to "3" and "4"
        geometrically constrained by 3² + 4² = 5²?

        If the universe is information, then the position of 5 in conceptual
        space should be DETERMINED by its relationships to other numbers.
        """
        print("\n=== Test: Concept Position Encoding ===")
        print("(Testing if the position of 5 is constrained by 3² + 4² = 5²)")

        # Hypothesis: embed(5) lies in a specific position relative to embed(3) and embed(4)
        # that encodes the Pythagorean relationship

        emb_3 = self.embed_number(3)
        emb_4 = self.embed_number(4)
        emb_5 = self.embed_number(5)
        emb_6 = self.embed_number(6)  # Control: 6 is NOT the hypotenuse

        # The "Pythagorean position" for 5 given 3 and 4
        # If embedding norms encode magnitude proportionally:
        # The direction from origin to 5 should be related to 3 and 4

        # Test: Is 5 more "aligned" with the (3,4) pair than 6 is?
        # Compute: similarity of 5 to the span of {3, 4} vs 6 to that span

        # Project onto subspace spanned by emb_3 and emb_4
        basis = np.stack([emb_3, emb_4]).T  # d x 2
        # Orthonormalize
        Q, R = np.linalg.qr(basis)

        proj_5 = Q @ (Q.T @ emb_5)
        proj_6 = Q @ (Q.T @ emb_6)

        # Residual: how much of the embedding is NOT in the 3-4 subspace
        resid_5 = np.linalg.norm(emb_5 - proj_5) / np.linalg.norm(emb_5)
        resid_6 = np.linalg.norm(emb_6 - proj_6) / np.linalg.norm(emb_6)

        print(f"Residual from (3,4)-subspace for 5: {resid_5:.4f}")
        print(f"Residual from (3,4)-subspace for 6: {resid_6:.4f}")
        print(f"5 is {'more' if resid_5 < resid_6 else 'less'} aligned with (3,4) than 6")

        # Alternative: The angle in the (3,4,5) triangle
        # In high-D, the angle at 5 between directions to 3 and 4
        dir_5_to_3 = emb_3 - emb_5
        dir_5_to_4 = emb_4 - emb_5
        cos_angle_at_5 = np.dot(dir_5_to_3, dir_5_to_4) / (
            np.linalg.norm(dir_5_to_3) * np.linalg.norm(dir_5_to_4) + 1e-8
        )
        angle_at_5 = np.arccos(np.clip(cos_angle_at_5, -1, 1)) * 180 / np.pi

        # For a right triangle, the angle opposite the hypotenuse should be 90°
        # But this is in embedding space, not concept space...
        print(f"Angle at 5 in embedding space: {angle_at_5:.1f}°")

        return {
            "residual_5_from_34_subspace": float(resid_5),
            "residual_6_from_34_subspace": float(resid_6),
            "5_more_aligned_than_6": resid_5 < resid_6,
            "angle_at_5": float(angle_at_5),
            "position_encodes_relationship": resid_5 < resid_6
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results" / "formula_geometry.json"))
    args = parser.parse_args()

    print("=" * 70)
    print("FORMULA AS GEOMETRY")
    print("Testing if Pythagorean theorem is LITERALLY encoded in latent space")
    print("=" * 70)

    model, tokenizer = load_model(args.model)
    tester = FormulaGeometryTester(model, tokenizer)

    results = {}
    results["norm_encoding"] = tester.test_norm_encoding()
    results["direction_encoding"] = tester.test_direction_encoding()
    results["pythagorean_surface"] = tester.test_pythagorean_surface()
    results["formula_recovery"] = tester.test_formula_recovery()
    results["concept_position"] = tester.test_concept_position_encoding()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Is the formula encoded in geometry?")
    print("=" * 70)

    tests_passed = sum([
        results["norm_encoding"]["encodes_formula"],
        results["direction_encoding"]["consistent_squaring_direction"],
        results["pythagorean_surface"]["surface_separates_valid"],
        results["formula_recovery"]["formula_recoverable"],
        results["concept_position"]["position_encodes_relationship"],
    ])

    for test_name, test_results in results.items():
        key = [k for k in test_results.keys() if k.startswith("encodes") or
               k.startswith("consistent") or k.startswith("surface") or
               k.startswith("formula") or k.startswith("position")][0]
        status = "✓" if test_results[key] else "✗"
        print(f"{status} {test_name}: {key} = {test_results[key]}")

    print(f"\n{tests_passed}/5 tests support the hypothesis")

    if tests_passed >= 3:
        print("\n>>> HYPOTHESIS SUPPORTED: Mathematical relationships appear to be")
        print("    encoded in the geometric structure of the latent space!")
    else:
        print("\n>>> HYPOTHESIS NOT SUPPORTED: No strong evidence of formula encoding")

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)

    def convert_for_json(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(args.output, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
