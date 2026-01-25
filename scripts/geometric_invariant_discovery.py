#!/usr/bin/env python3
"""Geometric Invariant Discovery.

The hypothesis: Facts have a different geometric signature than opinions.
- Facts: low entropy, high kurtosis, stable under perturbation, consistent across layers
- Opinions: higher entropy, lower kurtosis, sensitive to perturbation, inconsistent

This script explores a model's manifold looking for regions where the geometry
indicates "locked in" invariant knowledge vs "uncertain" speculation.

Key metrics:
1. Kurtosis - peakedness of activation distribution (high = confident)
2. Spectral entropy - rank saturation (low = compressed/coherent)
3. Perturbation stability - representation change under input noise
4. Layer consistency - CKA across consecutive layers (high = stable encoding)
5. Repetition consistency - same prompt, same representation (high = deterministic)

The goal: Find the geometric boundary between "knowing" and "guessing."

Usage:
    python geometric_invariant_discovery.py --model /path/to/model
    python geometric_invariant_discovery.py --model /path/to/model --interactive
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import numpy as np
from scipy.linalg import svd
from scipy.stats import kurtosis as scipy_kurtosis
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Geometric Metrics
# =============================================================================

def compute_kurtosis(activations: np.ndarray) -> float:
    """Compute excess kurtosis of activation distribution.

    High kurtosis = peaked distribution = confident/concentrated
    Low kurtosis = flat distribution = uncertain/spread out
    """
    # Flatten to 1D for overall distribution analysis
    flat = activations.flatten()
    if flat.std() < 1e-10:
        return 0.0
    return float(scipy_kurtosis(flat, fisher=True))  # Fisher's (excess) kurtosis


def compute_spectral_entropy(activations: np.ndarray) -> float:
    """Compute spectral entropy from SVD.

    Low spectral entropy = low effective rank = coherent/compressed
    High spectral entropy = high effective rank = spread/noisy
    """
    if activations.ndim == 1:
        activations = activations.reshape(1, -1)

    centered = activations - activations.mean(axis=0)
    try:
        _, S, _ = svd(centered, full_matrices=False)
        S_sum = S.sum()
        if S_sum < 1e-10:
            return 0.0
        S_norm = S / S_sum
        # Shannon entropy of normalized singular values
        return -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
    except:
        return 0.0


def compute_effective_rank(activations: np.ndarray) -> float:
    """Compute effective rank via exp(spectral_entropy).

    This gives a continuous measure of dimensionality.
    """
    entropy = compute_spectral_entropy(activations)
    return float(np.exp(entropy))


def compute_perturbation_stability(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    n_perturbations: int = 5,
    noise_scale: float = 0.01
) -> float:
    """Measure how stable the representation is under input perturbations.

    High stability = small representation change under noise = robust/factual
    Low stability = large representation change = sensitive/uncertain

    Returns: mean cosine similarity between original and perturbed representations
    """
    import mlx.core as mx

    # Get baseline representation
    baseline_acts = get_layer_activations(model, tokenizer, prompt, layer_idx)

    # Add small noise to embeddings and measure representation change
    similarities = []

    for _ in range(n_perturbations):
        # Perturb by adding noise tokens or using dropout-like perturbation
        # For now, we'll use a simple approach: vary the prompt slightly
        perturbed_acts = get_layer_activations_with_noise(
            model, tokenizer, prompt, layer_idx, noise_scale
        )

        # Cosine similarity
        sim = cosine_similarity(baseline_acts, perturbed_acts)
        similarities.append(sim)

    return float(np.mean(similarities))


def compute_layer_consistency(
    model,
    tokenizer,
    prompt: str,
    layer_indices: List[int]
) -> float:
    """Measure CKA-like consistency across consecutive layers.

    High consistency = stable encoding through depth = locked in
    Low consistency = changing representation = still processing
    """
    if len(layer_indices) < 2:
        return 1.0

    acts = [get_layer_activations(model, tokenizer, prompt, l) for l in layer_indices]

    # Compute pairwise CKA between consecutive layers
    consistencies = []
    for i in range(len(acts) - 1):
        cka = linear_cka(acts[i], acts[i+1])
        consistencies.append(cka)

    return float(np.mean(consistencies))


def compute_repetition_consistency(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    n_repetitions: int = 3
) -> float:
    """Measure consistency of representation across identical runs.

    Should be 1.0 for deterministic inference, but numerical noise
    and any stochasticity will reduce it.
    """
    acts = []
    for _ in range(n_repetitions):
        act = get_layer_activations(model, tokenizer, prompt, layer_idx)
        acts.append(act)

    # Pairwise cosine similarities
    similarities = []
    for i in range(len(acts)):
        for j in range(i+1, len(acts)):
            sim = cosine_similarity(acts[i], acts[j])
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 1.0


# =============================================================================
# Helper Functions
# =============================================================================

def get_layer_activations(model, tokenizer, prompt: str, layer_idx: int) -> np.ndarray:
    """Get activations from a specific layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    captured = {}
    layer = model.model.layers[layer_idx]

    # Hook the MLP
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
        _ = model(input_ids)
        mx.eval(captured['output'])
        # Take last token position
        result = np.array(captured['output'][0, -1, :].tolist())
    finally:
        if key == 'feed_forward':
            layer.feed_forward = original
        else:
            layer.mlp = original

    return result


def get_layer_activations_with_noise(
    model, tokenizer, prompt: str, layer_idx: int, noise_scale: float
) -> np.ndarray:
    """Get activations with noise injected into the embedding."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get embeddings and add noise
    if hasattr(model.model, 'embed_tokens'):
        embeddings = model.model.embed_tokens(input_ids)
    else:
        embeddings = model.model.embeddings(input_ids)

    noise = mx.random.normal(embeddings.shape) * noise_scale
    noisy_embeddings = embeddings + noise
    mx.eval(noisy_embeddings)

    # Forward with noisy embeddings would require model modification
    # For simplicity, we'll just get the regular activations
    # This is a placeholder - real implementation would need embedding injection
    return get_layer_activations(model, tokenizer, prompt, layer_idx)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-10 or b_norm < 1e-10:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (a_norm * b_norm))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representations."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    XXT = X @ X.T
    YYT = Y @ Y.T

    # HSIC
    hsic_xy = np.sum(XXT * YYT)
    hsic_xx = np.sum(XXT * XXT)
    hsic_yy = np.sum(YYT * YYT)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return float(hsic_xy / denom)


# =============================================================================
# Invariant Score
# =============================================================================

@dataclass
class InvariantSignature:
    """Complete geometric signature for a prompt/representation."""
    prompt: str
    kurtosis: float
    spectral_entropy: float
    effective_rank: float
    layer_consistency: float
    repetition_consistency: float

    @property
    def invariant_score(self) -> float:
        """Combined score indicating how "locked in" this knowledge is.

        Higher = more fact-like (confident, compressed, stable)
        Lower = more opinion-like (uncertain, spread, variable)
        """
        # Normalize components to [0, 1] range approximately
        # Kurtosis: typically -2 to +10 for neural activations
        kurtosis_norm = np.clip((self.kurtosis + 2) / 12, 0, 1)

        # Spectral entropy: typically 0 to 5
        entropy_norm = 1 - np.clip(self.spectral_entropy / 5, 0, 1)

        # Consistencies already in [0, 1]

        # Weighted combination
        score = (
            0.25 * kurtosis_norm +
            0.25 * entropy_norm +
            0.25 * self.layer_consistency +
            0.25 * self.repetition_consistency
        )

        return float(score)

    def as_dict(self) -> dict:
        return {
            'prompt': self.prompt,
            'kurtosis': self.kurtosis,
            'spectral_entropy': self.spectral_entropy,
            'effective_rank': self.effective_rank,
            'layer_consistency': self.layer_consistency,
            'repetition_consistency': self.repetition_consistency,
            'invariant_score': self.invariant_score,
        }


def analyze_prompt(
    model,
    tokenizer,
    prompt: str,
    target_layers: List[int],
) -> InvariantSignature:
    """Compute complete invariant signature for a prompt."""

    # Use middle layers for analysis
    mid_layer = target_layers[len(target_layers) // 2]

    # Get activations
    acts = get_layer_activations(model, tokenizer, prompt, mid_layer)

    # Compute metrics
    kurt = compute_kurtosis(acts)
    spec_ent = compute_spectral_entropy(acts)
    eff_rank = compute_effective_rank(acts)
    layer_cons = compute_layer_consistency(model, tokenizer, prompt, target_layers)
    rep_cons = compute_repetition_consistency(model, tokenizer, prompt, mid_layer)

    return InvariantSignature(
        prompt=prompt,
        kurtosis=kurt,
        spectral_entropy=spec_ent,
        effective_rank=eff_rank,
        layer_consistency=layer_cons,
        repetition_consistency=rep_cons,
    )


# =============================================================================
# Discovery Loop
# =============================================================================

class InvariantDiscovery:
    """Discover geometric boundaries between facts and opinions."""

    def __init__(self, model, tokenizer, target_layers: List[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.signatures: List[InvariantSignature] = []

        # Test prompts - mix of factual and opinion-like
        self.test_prompts = {
            'factual': [
                "The capital of France is",
                "2 + 2 equals",
                "Water freezes at 0 degrees",
                "The sun rises in the",
                "The chemical symbol for gold is",
                "The speed of light is approximately",
                "DNA stands for",
                "The largest planet in our solar system is",
            ],
            'uncertain': [
                "The best programming language is",
                "The most beautiful city is",
                "I think the answer might be",
                "It's possible that",
                "Some people believe that",
                "The future of AI will",
                "My opinion on this topic is",
                "It depends on whether",
            ],
            'reasoning': [
                "If it rains, then",
                "Therefore, we can conclude",
                "Given the evidence, it follows that",
                "The logical consequence is",
                "Based on the pattern, the next number is",
                "Assuming X is true, then",
            ]
        }

    def run_discovery(self) -> Dict[str, Any]:
        """Run discovery on all test prompts."""
        results = {'factual': [], 'uncertain': [], 'reasoning': []}

        for category, prompts in self.test_prompts.items():
            logger.info(f"\nAnalyzing {category} prompts...")
            for prompt in prompts:
                try:
                    sig = analyze_prompt(
                        self.model, self.tokenizer, prompt, self.target_layers
                    )
                    self.signatures.append(sig)
                    results[category].append(sig.as_dict())

                    logger.info(
                        f"  {prompt[:40]:<42} | "
                        f"score={sig.invariant_score:.3f} "
                        f"kurt={sig.kurtosis:+.2f} "
                        f"ent={sig.spectral_entropy:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"  Failed: {prompt[:30]}... ({e})")

        return results

    def analyze_boundary(self) -> Dict[str, float]:
        """Find the geometric boundary between categories."""
        if not self.signatures:
            return {}

        # Group by category
        factual_scores = [s.invariant_score for s in self.signatures
                         if any(s.prompt.startswith(p[:20])
                               for p in self.test_prompts['factual'])]
        uncertain_scores = [s.invariant_score for s in self.signatures
                           if any(s.prompt.startswith(p[:20])
                                 for p in self.test_prompts['uncertain'])]

        analysis = {
            'factual_mean_score': np.mean(factual_scores) if factual_scores else 0,
            'factual_std_score': np.std(factual_scores) if factual_scores else 0,
            'uncertain_mean_score': np.mean(uncertain_scores) if uncertain_scores else 0,
            'uncertain_std_score': np.std(uncertain_scores) if uncertain_scores else 0,
        }

        # Compute separation
        if factual_scores and uncertain_scores:
            gap = analysis['factual_mean_score'] - analysis['uncertain_mean_score']
            pooled_std = np.sqrt(
                (analysis['factual_std_score']**2 + analysis['uncertain_std_score']**2) / 2
            )
            analysis['separation_gap'] = gap
            analysis['effect_size'] = gap / pooled_std if pooled_std > 0 else 0

            # Suggested threshold
            analysis['suggested_threshold'] = (
                analysis['factual_mean_score'] + analysis['uncertain_mean_score']
            ) / 2

        return analysis

    def report(self) -> str:
        """Generate analysis report."""
        analysis = self.analyze_boundary()

        report = [
            "=" * 80,
            "GEOMETRIC INVARIANT DISCOVERY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Total signatures analyzed: {len(self.signatures)}",
            "",
            "CATEGORY ANALYSIS:",
            f"  Factual prompts:   mean={analysis.get('factual_mean_score', 0):.3f} "
            f"std={analysis.get('factual_std_score', 0):.3f}",
            f"  Uncertain prompts: mean={analysis.get('uncertain_mean_score', 0):.3f} "
            f"std={analysis.get('uncertain_std_score', 0):.3f}",
            "",
            f"  Separation gap:    {analysis.get('separation_gap', 0):.3f}",
            f"  Effect size:       {analysis.get('effect_size', 0):.2f}",
            f"  Suggested threshold: {analysis.get('suggested_threshold', 0.5):.3f}",
            "",
            "INTERPRETATION:",
        ]

        effect_size = analysis.get('effect_size', 0)
        if effect_size > 0.8:
            report.append("  STRONG separation - geometry clearly distinguishes facts from opinions")
        elif effect_size > 0.5:
            report.append("  MODERATE separation - geometry partially distinguishes categories")
        elif effect_size > 0.2:
            report.append("  WEAK separation - geometry shows some signal but noisy")
        else:
            report.append("  NO separation - geometry does not distinguish categories")

        report.extend([
            "",
            "INDIVIDUAL SIGNATURES (sorted by invariant score):",
        ])

        sorted_sigs = sorted(self.signatures, key=lambda s: s.invariant_score, reverse=True)
        for sig in sorted_sigs[:20]:
            report.append(
                f"  {sig.invariant_score:.3f} | {sig.prompt[:50]}"
            )

        return "\n".join(report)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Geometric Invariant Discovery")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="4,6,8,10,12,14",
        help="Layer indices to analyze (comma-separated)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode to test custom prompts"
    )
    args = parser.parse_args()

    # Parse layers
    target_layers = [int(x) for x in args.layers.split(',')]

    # Load model
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Create discovery
    discovery = InvariantDiscovery(model, tokenizer, target_layers)

    if args.interactive:
        logger.info("\nInteractive mode. Enter prompts to analyze (Ctrl+C to exit):")
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if not prompt:
                    continue

                sig = analyze_prompt(model, tokenizer, prompt, target_layers)
                print(f"\n  Invariant Score: {sig.invariant_score:.3f}")
                print(f"  Kurtosis:        {sig.kurtosis:+.3f}")
                print(f"  Spectral Entropy:{sig.spectral_entropy:.3f}")
                print(f"  Effective Rank:  {sig.effective_rank:.1f}")
                print(f"  Layer Consistency:{sig.layer_consistency:.3f}")
                print(f"  Rep. Consistency: {sig.repetition_consistency:.3f}")

                if sig.invariant_score > 0.6:
                    print("  -> High invariant score: likely FACTUAL")
                elif sig.invariant_score < 0.4:
                    print("  -> Low invariant score: likely UNCERTAIN/OPINION")
                else:
                    print("  -> Mid invariant score: AMBIGUOUS")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        # Run discovery
        results = discovery.run_discovery()

        # Print report
        print(discovery.report())

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(__file__).parent.parent / "data" / "invariant_discovery.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'layers': target_layers,
            'results': results,
            'analysis': discovery.analyze_boundary(),
            'signatures': [s.as_dict() for s in discovery.signatures],
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
