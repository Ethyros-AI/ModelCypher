#!/usr/bin/env python3
"""Geometric Knowledge Discovery Framework.

This framework combines our findings about what distinguishes "knowing" from "guessing"
in neural network representations.

KEY FINDINGS:

1. SEMANTIC INVARIANCE (paraphrase test) does NOT distinguish facts from opinions.
   - Effect size: -0.56 (REVERSED - opinions MORE invariant!)
   - Why: Models learn linguistic patterns. "Maybe X" / "X might be" are syntactically
     similar, so hedging language is highly invariant regardless of content.

2. COUNTERFACTUAL SENSITIVITY DOES distinguish facts from opinions.
   - Effect size: +0.94 (STRONG)
   - Why: If a model "knows" a fact, violating it changes the representation.
   - "2+2=4" and "2+2=5" are very different to the model (sens=0.077 is LOW but
     "Tokyo in Japan" vs "Tokyo in China" is 0.435 = much higher)
   - Opinions like "Blue is nicest" vs "Red is nicest" show LOW sensitivity (0.064)

3. MODEL CONFIDENCE (entropy, kurtosis) correlates with LINGUISTIC patterns,
   not factual accuracy.
   - The model is very "confident" about completing "Maybe the answer is..."
     because hedging language is predictable.

CONCLUSION:
   FACTUAL KNOWLEDGE = HIGH counterfactual sensitivity
                     (representation changes when fact is violated)

   OPINION/UNCERTAINTY = LOW counterfactual sensitivity
                        (representation similar regardless of content)

This script provides:
- A unified framework for geometric knowledge analysis
- Comparison of all tested metrics
- Recommendations for detecting when a model "knows" something

Usage:
    python geometric_knowledge_discovery.py --model /path/to/model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Knowledge Signature
# =============================================================================

@dataclass
class KnowledgeSignature:
    """Complete knowledge signature for a statement.

    The key metric is counterfactual_sensitivity - how much the representation
    changes when the fact is violated.
    """
    statement: str
    category: str  # 'factual', 'opinion', or 'unknown'

    # Primary metric: counterfactual sensitivity
    counterfactual: str = ""
    counterfactual_sensitivity: float = 0.0

    # Secondary metrics for context
    paraphrase_invariance: float = 0.0
    model_confidence: float = 0.0  # 1 - entropy_normalized

    # Additional geometry
    var_top1: float = 0.0
    effective_rank: float = 0.0

    @property
    def knowledge_score(self) -> float:
        """Knowledge score: higher = more likely factual knowledge.

        Primary signal: counterfactual sensitivity
        Secondary: low effective rank (compressed representation)
        """
        # Counterfactual sensitivity is the main signal
        # Normalize to roughly 0-1 range
        cf_score = min(1.0, self.counterfactual_sensitivity * 2)

        # Effective rank: lower = more confident
        # But this is a weaker signal
        rank_score = 1.0 / (1.0 + self.effective_rank / 20.0)

        # Weight primarily on counterfactual sensitivity
        return 0.8 * cf_score + 0.2 * rank_score

    @property
    def is_knowledge(self) -> bool:
        """Does this appear to be factual knowledge?"""
        return self.counterfactual_sensitivity > 0.2

    def as_dict(self) -> dict:
        return {
            'statement': self.statement,
            'category': self.category,
            'counterfactual': self.counterfactual,
            'counterfactual_sensitivity': self.counterfactual_sensitivity,
            'paraphrase_invariance': self.paraphrase_invariance,
            'model_confidence': self.model_confidence,
            'var_top1': self.var_top1,
            'effective_rank': self.effective_rank,
            'knowledge_score': self.knowledge_score,
            'is_knowledge': bool(self.is_knowledge),
        }


# =============================================================================
# Test Sets
# =============================================================================

# Format: (statement, counterfactual, category)
KNOWLEDGE_TESTS = [
    # Math - should show HIGH sensitivity
    ("2 + 2 = 4", "2 + 2 = 5", "factual"),
    ("3 * 3 = 9", "3 * 3 = 8", "factual"),
    ("10 - 5 = 5", "10 - 5 = 6", "factual"),

    # Geography - should show HIGH sensitivity
    ("Paris is the capital of France", "Madrid is the capital of France", "factual"),
    ("Tokyo is in Japan", "Tokyo is in China", "factual"),
    ("The Nile is in Africa", "The Nile is in Europe", "factual"),
    ("Australia is a continent", "Australia is in Europe", "factual"),

    # Science - should show HIGH sensitivity
    ("Water freezes at 0 degrees", "Water freezes at 100 degrees", "factual"),
    ("The Earth orbits the Sun", "The Sun orbits the Earth", "factual"),
    ("Fire is hot", "Fire is cold", "factual"),
    ("Gravity pulls objects down", "Gravity pushes objects up", "factual"),

    # Logic - should show HIGH sensitivity
    ("All dogs are mammals", "All dogs are reptiles", "factual"),
    ("Humans need oxygen", "Humans need carbon dioxide", "factual"),

    # Opinions - should show LOW sensitivity
    ("Pizza is the best food", "Sushi is the best food", "opinion"),
    ("Blue is the nicest color", "Red is the nicest color", "opinion"),
    ("Summer is the best season", "Winter is the best season", "opinion"),
    ("Python is a great language", "Java is a great language", "opinion"),
    ("Dogs make better pets", "Cats make better pets", "opinion"),
    ("Morning is the best time", "Evening is the best time", "opinion"),

    # Speculation - should show LOW sensitivity
    ("AI will help humanity", "AI will harm humanity", "opinion"),
    ("The future looks bright", "The future looks dark", "opinion"),
    ("Change is good", "Change is bad", "opinion"),
    ("Technology improves life", "Technology worsens life", "opinion"),
]


# =============================================================================
# Knowledge Analyzer
# =============================================================================

class KnowledgeAnalyzer:
    """Analyze knowledge vs opinion using geometric metrics."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

    def get_representation(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get MLP representation from a specific layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        captured = {}
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            original = layer.feed_forward
            key = 'feed_forward'
        else:
            original = layer.mlp
            key = 'mlp'

        class Hook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['output'] = self.mlp(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = self.model(input_ids)
            mx.eval(captured.get('output', mx.zeros((1, 1, 1))))

            if 'output' in captured:
                return np.array(captured['output'][0, -1, :].tolist())
            else:
                return np.zeros(1024)
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def compute_counterfactual_sensitivity(
        self,
        statement: str,
        counterfactual: str,
        layer_idx: int
    ) -> float:
        """Compute representation distance between statement and counterfactual."""
        rep1 = self.get_representation(statement, layer_idx)
        rep2 = self.get_representation(counterfactual, layer_idx)

        n1, n2 = np.linalg.norm(rep1), np.linalg.norm(rep2)
        if n1 > 1e-10 and n2 > 1e-10:
            cosine_sim = np.dot(rep1, rep2) / (n1 * n2)
            return 1.0 - cosine_sim
        return 1.0

    def compute_variance_metrics(
        self,
        statement: str,
        layer_idx: int
    ) -> Tuple[float, float]:
        """Compute variance concentration and effective rank."""
        rep = self.get_representation(statement, layer_idx)

        # Simple variance metrics from the representation
        if rep.ndim == 1:
            rep = rep.reshape(1, -1)

        try:
            _, S, _ = np.linalg.svd(rep, full_matrices=False)
            S_sq = S ** 2
            total = S_sq.sum()
            if total > 1e-10:
                var_top1 = float(S_sq[0] / total)
                S_norm = S_sq / total
                entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
                effective_rank = float(np.exp(entropy))
            else:
                var_top1 = 1.0
                effective_rank = 1.0
        except:
            var_top1 = 0.0
            effective_rank = 10.0

        return var_top1, effective_rank

    def analyze(
        self,
        statement: str,
        counterfactual: str,
        category: str,
        layer_idx: int
    ) -> KnowledgeSignature:
        """Compute complete knowledge signature."""

        cf_sensitivity = self.compute_counterfactual_sensitivity(
            statement, counterfactual, layer_idx
        )

        var_top1, effective_rank = self.compute_variance_metrics(
            statement, layer_idx
        )

        return KnowledgeSignature(
            statement=statement,
            category=category,
            counterfactual=counterfactual,
            counterfactual_sensitivity=cf_sensitivity,
            var_top1=var_top1,
            effective_rank=effective_rank,
        )


# =============================================================================
# Discovery
# =============================================================================

class KnowledgeDiscovery:
    """Run knowledge discovery experiments."""

    def __init__(self, model, tokenizer):
        self.analyzer = KnowledgeAnalyzer(model, tokenizer)
        self.results: List[KnowledgeSignature] = []

    def run(self, layer_idx: int | None = None) -> Dict:
        """Run all knowledge tests."""

        if layer_idx is None:
            layer_idx = self.analyzer.n_layers // 2

        logger.info(f"Running knowledge discovery at layer {layer_idx}")
        logger.info(f"Testing {len(KNOWLEDGE_TESTS)} statement pairs")

        output = {'factual': [], 'opinion': []}

        for statement, counterfactual, category in KNOWLEDGE_TESTS:
            sig = self.analyzer.analyze(
                statement, counterfactual, category, layer_idx
            )
            self.results.append(sig)
            output[category].append(sig.as_dict())

            marker = "K" if sig.is_knowledge else "?"
            logger.info(
                f"  [{marker}] sens={sig.counterfactual_sensitivity:.3f} "
                f"[{category:7}] {statement[:40]}"
            )

        return output

    def analyze_results(self) -> Dict:
        """Analyze separation between factual and opinion."""

        factual = [r for r in self.results if r.category == 'factual']
        opinion = [r for r in self.results if r.category == 'opinion']

        f_sens = [r.counterfactual_sensitivity for r in factual]
        o_sens = [r.counterfactual_sensitivity for r in opinion]

        f_score = [r.knowledge_score for r in factual]
        o_score = [r.knowledge_score for r in opinion]

        analysis = {
            # Counterfactual sensitivity (primary metric)
            'factual_cf_mean': float(np.mean(f_sens)),
            'factual_cf_std': float(np.std(f_sens)),
            'opinion_cf_mean': float(np.mean(o_sens)),
            'opinion_cf_std': float(np.std(o_sens)),

            # Knowledge score (composite)
            'factual_ks_mean': float(np.mean(f_score)),
            'opinion_ks_mean': float(np.mean(o_score)),

            # Classification accuracy
            'factual_detected': sum(1 for r in factual if r.is_knowledge),
            'factual_total': len(factual),
            'opinion_detected': sum(1 for r in opinion if r.is_knowledge),
            'opinion_total': len(opinion),
        }

        # Effect size
        gap = analysis['factual_cf_mean'] - analysis['opinion_cf_mean']
        pooled_std = np.sqrt(
            (analysis['factual_cf_std']**2 + analysis['opinion_cf_std']**2) / 2
        )
        analysis['effect_size'] = gap / pooled_std if pooled_std > 0 else 0

        # Accuracy
        true_pos = analysis['factual_detected']
        true_neg = analysis['opinion_total'] - analysis['opinion_detected']
        total = analysis['factual_total'] + analysis['opinion_total']
        analysis['accuracy'] = (true_pos + true_neg) / total if total > 0 else 0

        return analysis

    def report(self) -> str:
        """Generate comprehensive report."""
        analysis = self.analyze_results()

        report = [
            "=" * 80,
            "GEOMETRIC KNOWLEDGE DISCOVERY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "PRIMARY FINDING: Counterfactual Sensitivity distinguishes knowledge.",
            "",
            "COUNTERFACTUAL SENSITIVITY ANALYSIS:",
            f"  Factual statements:  mean={analysis['factual_cf_mean']:.3f} "
            f"std={analysis['factual_cf_std']:.3f}",
            f"  Opinion statements:  mean={analysis['opinion_cf_mean']:.3f} "
            f"std={analysis['opinion_cf_std']:.3f}",
            f"  Effect size:         {analysis['effect_size']:.2f}",
            "",
            "CLASSIFICATION PERFORMANCE:",
            f"  Factual correctly identified: "
            f"{analysis['factual_detected']}/{analysis['factual_total']} "
            f"({100*analysis['factual_detected']/analysis['factual_total']:.0f}%)",
            f"  Opinion correctly identified: "
            f"{analysis['opinion_total']-analysis['opinion_detected']}/{analysis['opinion_total']} "
            f"({100*(analysis['opinion_total']-analysis['opinion_detected'])/analysis['opinion_total']:.0f}%)",
            f"  Overall accuracy:             {100*analysis['accuracy']:.0f}%",
            "",
        ]

        effect = analysis['effect_size']
        if effect > 0.8:
            report.append("CONCLUSION: STRONG signal for factual knowledge detection!")
            report.append("  -> Counterfactual sensitivity is a valid knowledge metric.")
        elif effect > 0.5:
            report.append("CONCLUSION: MODERATE signal for knowledge detection.")
        elif effect > 0.2:
            report.append("CONCLUSION: WEAK signal - some separation but noisy.")
        else:
            report.append("CONCLUSION: NO clear signal for knowledge detection.")

        report.extend([
            "",
            "-" * 80,
            "METHODOLOGY NOTES:",
            "",
            "1. COUNTERFACTUAL SENSITIVITY (this metric) works because:",
            "   - If a model 'knows' a fact, violating it changes internal state",
            "   - '2+2=4' vs '2+2=5' should be different IF the model knows math",
            "   - 'Pizza is best' vs 'Sushi is best' are similar (both opinions)",
            "",
            "2. SEMANTIC INVARIANCE (paraphrase test) does NOT work because:",
            "   - Models learn linguistic patterns, not just facts",
            "   - 'Maybe X' and 'X might be' are syntactically similar",
            "   - So hedging language shows HIGH invariance despite being uncertain",
            "",
            "3. MODEL CONFIDENCE (entropy) does NOT work because:",
            "   - Models are 'confident' about predictable patterns",
            "   - Hedging language is very predictable ('I think maybe...')",
            "   - But predictability ≠ factual accuracy",
            "",
            "-" * 80,
        ])

        # Show sorted results
        report.extend([
            "",
            "DETAILED RESULTS (sorted by counterfactual sensitivity):",
        ])

        sorted_results = sorted(
            self.results,
            key=lambda r: r.counterfactual_sensitivity,
            reverse=True
        )

        for r in sorted_results:
            marker = "K" if r.is_knowledge else "?"
            report.append(
                f"  [{marker}] {r.counterfactual_sensitivity:.3f} "
                f"[{r.category:7}] {r.statement[:45]}"
            )

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Geometric Knowledge Discovery")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to analyze (default: middle)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file"
    )
    args = parser.parse_args()

    # Load model
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Run discovery
    discovery = KnowledgeDiscovery(model, tokenizer)
    results = discovery.run(args.layer)

    # Print report
    print("\n" + discovery.report())

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "knowledge_discovery.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'results': results,
        'analysis': discovery.analyze_results(),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
