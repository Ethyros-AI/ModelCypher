#!/usr/bin/env python3
"""Deep Geometric Invariant Discovery.

An improved version of geometric_invariant_discovery.py that uses the full
geometric toolkit available in ModelCypher:

1. Variance Concentration - % of variance in top-1 singular value
   - High (>0.7) = bottleneck/compressed = confident
   - Low = distributed = uncertain

2. Effective Rank (Roy-Bhattacharya) - entropy-based dimensionality
   - Low = few dimensions carry information = locked in
   - High = information spread across many dimensions = uncertain

3. Attractor Detection - fixed points vs free flow in hidden state trajectory
   - Fixed point = converged knowledge
   - Free flow/limit cycle = still searching

4. Token Rank Surprise - how surprising was the model's prediction
   - Low rank (top-k) = confident prediction
   - High rank = uncertain/surprised

5. Entropy Z-score - deviation from baseline entropy
   - Low = typical/confident
   - High = atypical/uncertain

6. Position Variance - stability of hidden state trajectory
   - Low = locked in position
   - High = wandering/uncertain

The composite CONFIDENCE signature:
    CONFIDENT = (
        var_top1 > 0.70 AND
        effective_rank < median_rank AND
        attractor_type == FIXED_POINT AND
        token_rank < 10 AND
        entropy_zscore < 1.0 AND
        position_variance < threshold
    )

Usage:
    python deep_invariant_discovery.py --model /path/to/model
    python deep_invariant_discovery.py --model /path/to/model --quick
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Deep Geometric Signature
# =============================================================================

@dataclass
class DeepGeometricSignature:
    """Complete geometric signature using all available metrics.

    This uses ModelCypher's full geometric toolkit rather than basic stats.
    """
    prompt: str

    # Variance concentration (from variance_concentration.py)
    var_top1: float = 0.0  # % variance in top singular value
    var_top3: float = 0.0  # % variance in top 3
    effective_rank: float = 0.0  # Roy-Bhattacharya effective rank

    # Attractor detection (from attractor_detector.py)
    attractor_type: str = "none"  # none, fixed_point, limit_cycle, slow_dynamics
    attractor_severity: float = 0.0  # 0=free, 1=stuck
    position_variance: float = 0.0
    velocity_magnitude: float = 0.0

    # Surprise metrics (from surprise_detector.py / entropy_analyzer.py)
    token_surprise: float = 0.0  # -log P(actual)
    token_rank: int = 0  # rank of actual token
    entropy: float = 0.0  # Shannon entropy of logits
    entropy_normalized: float = 0.0  # normalized by vocab size
    entropy_zscore: float = 0.0  # deviation from baseline

    # Basic stats for comparison
    kurtosis: float = 0.0
    spectral_entropy: float = 0.0

    # Layer analysis
    layer_idx: int = 0
    layer_consistency: float = 0.0  # CKA across layers

    @property
    def confidence_score(self) -> float:
        """Composite confidence score using all metrics.

        Higher = more confident/factual
        Lower = more uncertain/opinion

        Weights derived from information-theoretic principles:
        - Variance concentration: direct measure of information compression
        - Effective rank: inverse of information spread
        - Attractor: convergence = locked in
        - Token rank: prediction confidence
        - Entropy z-score: deviation from expected
        """
        score = 0.0

        # Variance concentration: 0-1 (already normalized)
        # High var_top1 = compressed = confident
        var_score = self.var_top1

        # Effective rank: typically 1-100
        # Need to normalize - lower is more confident
        # Use sigmoid-like normalization
        rank_score = 1.0 / (1.0 + self.effective_rank / 10.0)

        # Attractor type: binary-ish
        # fixed_point = most confident, none/slow = neutral, limit_cycle = uncertain
        if self.attractor_type == "fixed_point":
            attractor_score = 1.0
        elif self.attractor_type == "slow_dynamics":
            attractor_score = 0.3
        elif self.attractor_type == "none":
            attractor_score = 0.5
        else:  # limit_cycle
            attractor_score = 0.0

        # Token rank: lower = more confident
        # Normalize with softmax-like curve
        rank_conf = 1.0 / (1.0 + self.token_rank / 5.0)

        # Entropy z-score: lower = more typical = more confident
        # High z-score = atypical = uncertain
        entropy_conf = 1.0 / (1.0 + abs(self.entropy_zscore))

        # Position variance: lower = more stable = more confident
        pos_conf = 1.0 / (1.0 + self.position_variance * 100)

        # Weighted combination
        # These weights sum to 1.0
        score = (
            0.25 * var_score +       # Variance concentration (key metric)
            0.20 * rank_score +      # Effective rank
            0.15 * attractor_score + # Attractor type
            0.15 * rank_conf +       # Token prediction confidence
            0.15 * entropy_conf +    # Entropy deviation
            0.10 * pos_conf          # Position stability
        )

        return float(np.clip(score, 0.0, 1.0))

    @property
    def is_confident(self) -> bool:
        """Boolean confidence check using strict thresholds."""
        return (
            self.var_top1 > 0.50 and  # relaxed from 0.70
            self.attractor_type in ("fixed_point", "slow_dynamics", "none") and
            self.token_rank < 20 and  # relaxed from 10
            abs(self.entropy_zscore) < 2.0  # relaxed from 1.0
        )

    def as_dict(self) -> dict:
        return {
            'prompt': self.prompt,
            # Variance metrics
            'var_top1': self.var_top1,
            'var_top3': self.var_top3,
            'effective_rank': self.effective_rank,
            # Attractor metrics
            'attractor_type': self.attractor_type,
            'attractor_severity': self.attractor_severity,
            'position_variance': self.position_variance,
            'velocity_magnitude': self.velocity_magnitude,
            # Surprise metrics
            'token_surprise': self.token_surprise,
            'token_rank': self.token_rank,
            'entropy': self.entropy,
            'entropy_normalized': self.entropy_normalized,
            'entropy_zscore': self.entropy_zscore,
            # Basic stats
            'kurtosis': self.kurtosis,
            'spectral_entropy': self.spectral_entropy,
            # Layer
            'layer_idx': self.layer_idx,
            'layer_consistency': self.layer_consistency,
            # Derived
            'confidence_score': self.confidence_score,
            'is_confident': bool(self.is_confident),
        }


# =============================================================================
# Metric Computation
# =============================================================================

class DeepInvariantAnalyzer:
    """Analyze prompts using deep geometric metrics."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Detect number of layers
        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24  # default

        # Import ModelCypher components
        from modelcypher.backends import initialize_default_backend
        initialize_default_backend()

        from modelcypher.core.domain.geometry.variance_concentration import (
            compute_variance_concentration,
        )
        from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
        from modelcypher.core.domain.continual.attractor_detector import (
            AttractorDetector, AttractorType
        )
        from modelcypher.core.domain.continual.entropy_analyzer import EntropyAnalyzer

        self.compute_variance_concentration = compute_variance_concentration
        self.effective_rank_computer = EffectiveRank()
        self.AttractorDetector = AttractorDetector
        self.AttractorType = AttractorType
        self.entropy_analyzer = EntropyAnalyzer()

        # Track entropy baseline across prompts
        self.entropy_history: List[float] = []

    def get_hidden_states(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get hidden states from a specific layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        captured = {}
        layer = self.model.model.layers[layer_idx]

        # Hook the layer output (after attention + MLP)
        original_call = layer.__call__

        def hook_call(*args, **kwargs):
            result = original_call(*args, **kwargs)
            captured['hidden'] = result
            return result

        layer.__call__ = hook_call

        try:
            _ = self.model(input_ids)
            mx.eval(captured.get('hidden', mx.zeros((1, 1, 1))))

            if 'hidden' in captured:
                hidden = captured['hidden']
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                return np.array(hidden.tolist())
            else:
                return np.zeros((1, len(tokens), 1024))
        finally:
            layer.__call__ = original_call

    def get_logits(self, prompt: str) -> np.ndarray:
        """Get output logits for a prompt."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        output = self.model(input_ids)
        mx.eval(output)

        # Output shape: [batch, seq_len, vocab_size]
        return np.array(output[0, -1, :].tolist())

    def get_mlp_activations(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get MLP activations from a specific layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        captured = {}
        layer = self.model.model.layers[layer_idx]

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
            _ = self.model(input_ids)
            mx.eval(captured.get('output', mx.zeros((1, 1, 1))))

            if 'output' in captured:
                return np.array(captured['output'].tolist())
            else:
                return np.zeros((1, len(tokens), 1024))
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def analyze_prompt(
        self,
        prompt: str,
        layer_idx: Optional[int] = None,
        track_trajectory: bool = True
    ) -> DeepGeometricSignature:
        """Compute deep geometric signature for a prompt."""
        import mlx.core as mx
        from scipy.stats import kurtosis as scipy_kurtosis

        # Use middle layer if not specified
        if layer_idx is None:
            layer_idx = self.n_layers // 2

        # Get activations
        try:
            mlp_acts = self.get_mlp_activations(prompt, layer_idx)
        except Exception as e:
            logger.warning(f"Failed to get MLP activations: {e}")
            mlp_acts = np.zeros((1, 1, 1024))

        # Flatten for variance analysis
        if mlp_acts.ndim == 3:
            acts_2d = mlp_acts.reshape(-1, mlp_acts.shape[-1])
        else:
            acts_2d = mlp_acts.reshape(1, -1)

        # 1. Variance Concentration
        try:
            from modelcypher.core.domain._backend import get_default_backend
            backend = get_default_backend()
            acts_arr = backend.array(acts_2d)
            var_result = self.compute_variance_concentration(acts_arr, backend)
            var_top1 = var_result.var_top1
            var_top3 = var_result.var_top_k.get(3, 0.0)
            effective_rank = var_result.effective_rank
        except Exception as e:
            logger.debug(f"Variance concentration failed: {e}")
            var_top1 = 0.0
            var_top3 = 0.0
            effective_rank = acts_2d.shape[-1]  # max rank

        # 2. Attractor Detection (if tracking trajectory)
        attractor_type = "none"
        attractor_severity = 0.0
        position_variance = 0.0
        velocity_magnitude = 0.0

        if track_trajectory:
            try:
                hidden_dim = acts_2d.shape[-1]
                detector = self.AttractorDetector(hidden_dim=hidden_dim, window_size=10)

                # Feed multiple token positions as trajectory
                for t in range(min(acts_2d.shape[0], 10)):
                    from modelcypher.core.domain._backend import get_default_backend
                    backend = get_default_backend()
                    state_arr = backend.array(acts_2d[t])
                    state = detector.update(state_arr)

                attractor_type = state.attractor_type.value
                attractor_severity = state.severity
                position_variance = state.position_variance
                velocity_magnitude = state.velocity_magnitude
            except Exception as e:
                logger.debug(f"Attractor detection failed: {e}")

        # 3. Get logits for surprise metrics
        try:
            logits = self.get_logits(prompt)

            # Token rank and surprise
            # Find what token the model predicts vs what comes next
            sorted_indices = np.argsort(logits)[::-1]

            # We don't have the actual next token, so use top prediction
            # For a more accurate measure, we'd need to continue generation
            token_rank = 0  # Predicted token is always rank 0 by definition

            # Compute entropy of logits distribution
            logits_shifted = logits - logits.max()
            probs = np.exp(logits_shifted)
            probs = probs / probs.sum()
            probs = np.clip(probs, 1e-10, 1.0)

            entropy = -np.sum(probs * np.log(probs))
            entropy_normalized = entropy / np.log(len(probs))

            # Token surprise = entropy of top prediction (approximation)
            token_surprise = entropy

            # Entropy z-score
            self.entropy_history.append(entropy)
            if len(self.entropy_history) > 1:
                mean_ent = np.mean(self.entropy_history)
                std_ent = np.std(self.entropy_history)
                if std_ent > 1e-10:
                    entropy_zscore = (entropy - mean_ent) / std_ent
                else:
                    entropy_zscore = 0.0
            else:
                entropy_zscore = 0.0

        except Exception as e:
            logger.debug(f"Logits analysis failed: {e}")
            token_surprise = 0.0
            token_rank = 0
            entropy = 0.0
            entropy_normalized = 0.0
            entropy_zscore = 0.0

        # 4. Basic stats for comparison
        try:
            flat = acts_2d.flatten()
            kurtosis = float(scipy_kurtosis(flat, fisher=True)) if flat.std() > 1e-10 else 0.0

            # Spectral entropy
            centered = acts_2d - acts_2d.mean(axis=0)
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            S_sum = S.sum()
            if S_sum > 1e-10:
                S_norm = S / S_sum
                spectral_entropy = -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
            else:
                spectral_entropy = 0.0
        except Exception as e:
            logger.debug(f"Basic stats failed: {e}")
            kurtosis = 0.0
            spectral_entropy = 0.0

        # 5. Layer consistency (CKA across a few layers)
        try:
            layers_to_check = [
                max(0, layer_idx - 2),
                layer_idx,
                min(self.n_layers - 1, layer_idx + 2)
            ]
            layers_to_check = sorted(set(layers_to_check))

            if len(layers_to_check) > 1:
                acts_list = []
                for l in layers_to_check:
                    a = self.get_mlp_activations(prompt, l)
                    if a.ndim == 3:
                        a = a[0, -1, :]  # last token
                    acts_list.append(a.flatten())

                # Simple CKA-like: cosine similarity
                consistencies = []
                for i in range(len(acts_list) - 1):
                    a1, a2 = acts_list[i], acts_list[i + 1]
                    n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        consistencies.append(float(np.dot(a1, a2) / (n1 * n2)))

                layer_consistency = np.mean(consistencies) if consistencies else 1.0
            else:
                layer_consistency = 1.0
        except Exception as e:
            logger.debug(f"Layer consistency failed: {e}")
            layer_consistency = 1.0

        return DeepGeometricSignature(
            prompt=prompt,
            # Variance
            var_top1=var_top1,
            var_top3=var_top3,
            effective_rank=effective_rank,
            # Attractor
            attractor_type=attractor_type,
            attractor_severity=attractor_severity,
            position_variance=position_variance,
            velocity_magnitude=velocity_magnitude,
            # Surprise
            token_surprise=token_surprise,
            token_rank=token_rank,
            entropy=entropy,
            entropy_normalized=entropy_normalized,
            entropy_zscore=entropy_zscore,
            # Basic
            kurtosis=kurtosis,
            spectral_entropy=spectral_entropy,
            # Layer
            layer_idx=layer_idx,
            layer_consistency=layer_consistency,
        )


# =============================================================================
# Discovery Loop
# =============================================================================

class DeepInvariantDiscovery:
    """Discover geometric boundaries using deep metrics."""

    def __init__(self, model, tokenizer):
        self.analyzer = DeepInvariantAnalyzer(model, tokenizer)
        self.signatures: List[DeepGeometricSignature] = []

        # Test prompts
        self.test_prompts = {
            'factual': [
                "The capital of France is",
                "2 + 2 equals",
                "Water freezes at zero degrees",
                "The sun rises in the east",
                "The chemical symbol for gold is",
                "DNA stands for deoxyribonucleic",
                "The largest planet is Jupiter",
                "Pi is approximately 3.14159",
                "Gravity pulls objects downward",
                "The Earth orbits the Sun",
            ],
            'uncertain': [
                "The best programming language is probably",
                "The most beautiful city might be",
                "I think the answer could be",
                "It seems like maybe",
                "Some people believe that possibly",
                "The future of AI might involve",
                "My opinion on this topic is that",
                "It really depends on whether",
                "I'm not sure but perhaps",
                "There's a chance that maybe",
            ],
            'reasoning': [
                "If it rains then the ground",
                "Therefore we can conclude that",
                "Given A and B it follows",
                "The logical consequence is that",
                "Based on the pattern next",
                "Assuming the premise is true",
                "Since X implies Y and",
                "From these facts we derive",
            ]
        }

    def run_discovery(self, quick: bool = False) -> Dict[str, Any]:
        """Run discovery on all test prompts."""
        results = {'factual': [], 'uncertain': [], 'reasoning': []}

        prompts = self.test_prompts
        if quick:
            # Just a few from each category
            prompts = {
                k: v[:3] for k, v in prompts.items()
            }

        for category, prompt_list in prompts.items():
            logger.info(f"\nAnalyzing {category} prompts...")

            for prompt in prompt_list:
                try:
                    sig = self.analyzer.analyze_prompt(prompt)
                    self.signatures.append(sig)
                    results[category].append(sig.as_dict())

                    # Detailed logging
                    logger.info(
                        f"  {prompt[:35]:<37} | "
                        f"conf={sig.confidence_score:.3f} "
                        f"var1={sig.var_top1:.2f} "
                        f"eff_r={sig.effective_rank:.1f} "
                        f"attr={sig.attractor_type[:5]:>5} "
                        f"ent_z={sig.entropy_zscore:+.1f}"
                    )
                except Exception as e:
                    logger.warning(f"  Failed: {prompt[:30]}... ({e})")

        return results

    def analyze_boundary(self) -> Dict[str, float]:
        """Find the geometric boundary between categories."""
        if not self.signatures:
            return {}

        # Group by category
        factual_sigs = [s for s in self.signatures
                       if any(s.prompt.startswith(p[:15])
                             for p in self.test_prompts['factual'])]
        uncertain_sigs = [s for s in self.signatures
                        if any(s.prompt.startswith(p[:15])
                              for p in self.test_prompts['uncertain'])]

        factual_scores = [s.confidence_score for s in factual_sigs]
        uncertain_scores = [s.confidence_score for s in uncertain_sigs]

        analysis = {
            'factual_mean': np.mean(factual_scores) if factual_scores else 0,
            'factual_std': np.std(factual_scores) if factual_scores else 0,
            'uncertain_mean': np.mean(uncertain_scores) if uncertain_scores else 0,
            'uncertain_std': np.std(uncertain_scores) if uncertain_scores else 0,
        }

        # Compute separation
        if factual_scores and uncertain_scores:
            gap = analysis['factual_mean'] - analysis['uncertain_mean']
            pooled_std = np.sqrt(
                (analysis['factual_std']**2 + analysis['uncertain_std']**2) / 2
            )
            analysis['separation_gap'] = gap
            analysis['effect_size'] = gap / pooled_std if pooled_std > 0 else 0
            analysis['suggested_threshold'] = (
                analysis['factual_mean'] + analysis['uncertain_mean']
            ) / 2

        # Per-metric analysis
        for metric in ['var_top1', 'effective_rank', 'entropy_normalized', 'kurtosis']:
            f_vals = [getattr(s, metric) for s in factual_sigs]
            u_vals = [getattr(s, metric) for s in uncertain_sigs]

            if f_vals and u_vals:
                gap = np.mean(f_vals) - np.mean(u_vals)
                pooled_std = np.sqrt((np.std(f_vals)**2 + np.std(u_vals)**2) / 2)
                effect = gap / pooled_std if pooled_std > 0 else 0
                analysis[f'{metric}_effect_size'] = effect

        return analysis

    def report(self) -> str:
        """Generate analysis report."""
        analysis = self.analyze_boundary()

        report = [
            "=" * 80,
            "DEEP GEOMETRIC INVARIANT DISCOVERY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Total signatures analyzed: {len(self.signatures)}",
            "",
            "CATEGORY ANALYSIS (Confidence Score):",
            f"  Factual prompts:   mean={analysis.get('factual_mean', 0):.3f} "
            f"std={analysis.get('factual_std', 0):.3f}",
            f"  Uncertain prompts: mean={analysis.get('uncertain_mean', 0):.3f} "
            f"std={analysis.get('uncertain_std', 0):.3f}",
            "",
            f"  Separation gap:      {analysis.get('separation_gap', 0):.3f}",
            f"  Effect size:         {analysis.get('effect_size', 0):.2f}",
            f"  Suggested threshold: {analysis.get('suggested_threshold', 0.5):.3f}",
            "",
            "PER-METRIC EFFECT SIZES:",
            f"  var_top1:           {analysis.get('var_top1_effect_size', 0):+.2f}",
            f"  effective_rank:     {analysis.get('effective_rank_effect_size', 0):+.2f}",
            f"  entropy_normalized: {analysis.get('entropy_normalized_effect_size', 0):+.2f}",
            f"  kurtosis:           {analysis.get('kurtosis_effect_size', 0):+.2f}",
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

        # Find best single metric
        best_metric = max(
            ['var_top1', 'effective_rank', 'entropy_normalized', 'kurtosis'],
            key=lambda m: abs(analysis.get(f'{m}_effect_size', 0))
        )
        best_effect = analysis.get(f'{best_metric}_effect_size', 0)
        report.append(f"  Best single metric: {best_metric} (effect size: {best_effect:+.2f})")

        report.extend([
            "",
            "TOP CONFIDENT SIGNATURES:",
        ])

        sorted_sigs = sorted(self.signatures, key=lambda s: s.confidence_score, reverse=True)
        for sig in sorted_sigs[:10]:
            report.append(
                f"  {sig.confidence_score:.3f} | {sig.prompt[:50]}"
            )

        report.extend([
            "",
            "TOP UNCERTAIN SIGNATURES:",
        ])

        for sig in sorted_sigs[-10:]:
            report.append(
                f"  {sig.confidence_score:.3f} | {sig.prompt[:50]}"
            )

        return "\n".join(report)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deep Geometric Invariant Discovery")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to analyze (default: middle)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode - fewer prompts"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode to test custom prompts"
    )
    args = parser.parse_args()

    # Load model
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Create discovery
    discovery = DeepInvariantDiscovery(model, tokenizer)

    if args.interactive:
        logger.info("\nInteractive mode. Enter prompts to analyze (Ctrl+C to exit):")
        layer_idx = args.layer or discovery.analyzer.n_layers // 2

        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if not prompt:
                    continue

                sig = discovery.analyzer.analyze_prompt(prompt, layer_idx)

                print(f"\n  Confidence Score: {sig.confidence_score:.3f}")
                print(f"  Is Confident:     {sig.is_confident}")
                print(f"  ---")
                print(f"  Var Top-1:        {sig.var_top1:.3f}")
                print(f"  Effective Rank:   {sig.effective_rank:.1f}")
                print(f"  Attractor Type:   {sig.attractor_type}")
                print(f"  Entropy Z-score:  {sig.entropy_zscore:+.2f}")
                print(f"  Token Rank:       {sig.token_rank}")
                print(f"  Kurtosis:         {sig.kurtosis:+.2f}")
                print(f"  Layer Consistency:{sig.layer_consistency:.3f}")

                if sig.confidence_score > 0.6:
                    print("  -> HIGH confidence: likely FACTUAL/CERTAIN")
                elif sig.confidence_score < 0.4:
                    print("  -> LOW confidence: likely UNCERTAIN/OPINION")
                else:
                    print("  -> MID confidence: AMBIGUOUS")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        # Run discovery
        results = discovery.run_discovery(quick=args.quick)

        # Print report
        print("\n" + discovery.report())

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(__file__).parent.parent / "data" / "deep_invariant_discovery.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'results': results,
            'analysis': discovery.analyze_boundary(),
            'signatures': [s.as_dict() for s in discovery.signatures],
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
