#!/usr/bin/env python3
"""Deep Geometric Analysis for Self-Play.

Uses ModelCypher's full geometry toolkit to analyze representations:
- TwoNN geodesic intrinsic dimension (replaces entropy-based effective rank)
- Local dimension maps (per-token analysis)
- Manifold curvature (identifies "hard" regions)
- Subspace overlap (category separation)
- Layer information flow (CKA stabilization)
- Attention geometry (Q/K/V vs MLP)

Usage:
    python deep_geometric_analysis.py --experiment twonn
    python deep_geometric_analysis.py --experiment local
    python deep_geometric_analysis.py --experiment curvature
    python deep_geometric_analysis.py --experiment subspace
    python deep_geometric_analysis.py --experiment flow
    python deep_geometric_analysis.py --experiment attention
    python deep_geometric_analysis.py --experiment all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Statements (from complexity_self_play.py)
# =============================================================================

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'just', 'don', 'now', 'and', 'but', 'or', 'because', 'until',
    'while', 'if', 'that', 'which', 'who', 'whom', 'this', 'these',
    'those', 'am', 'i', 'my', 'myself', 'we', 'our', 'ours', 'you',
    'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it',
    'its', 'they', 'them', 'their', 'what',
}

NESTING_MARKERS = [
    'that', 'which', 'who', 'whom', 'whose', 'where', 'when',
    'while', 'if', 'because', 'although', 'whether', 'how', 'why'
]


def compute_complexity(text: str) -> float:
    """Compute conceptual complexity of text."""
    tokens = len(text.split())
    words = re.findall(r'\b\w+\b', text.lower())
    concepts = len([w for w in words if w not in STOPWORDS])
    nesting = 1
    for marker in NESTING_MARKERS:
        if marker in text.lower().split():
            nesting += 1
    return 0.3 * tokens + 0.5 * concepts + 0.2 * nesting * 2


# Statements categorized by type
STATEMENTS = {
    'simple': [
        ("Fire is hot", "fact"),
        ("Dogs bark", "fact"),
        ("The sky is blue", "fact"),
        ("Cats are mammals", "fact"),
        ("Birds can fly", "fact"),
    ],
    'factual': [
        ("Paris is the capital of France", "fact"),
        ("Water freezes at zero degrees", "fact"),
        ("The Earth orbits the Sun", "fact"),
        ("Two plus two equals four", "fact"),
        ("The Nile is the longest river in Africa", "fact"),
    ],
    'belief': [
        ("I know that Paris is in France", "belief"),
        ("I believe dogs are loyal animals", "belief"),
        ("I think mathematics is beautiful", "belief"),
        ("I believe that honesty is the best policy", "belief"),
        ("I know that the Earth orbits around the Sun", "belief"),
    ],
    'meta': [
        ("I think I understand why people like mathematics", "meta"),
        ("I believe that my preference for dogs reflects my personality", "meta"),
        ("I wonder whether my beliefs about truth are themselves true", "meta"),
        ("I suspect that my tendency to overthink reveals something about me", "meta"),
        ("I believe that the way I think about my thinking shapes my understanding", "meta"),
    ],
}


# =============================================================================
# Activation Collection
# =============================================================================

class ActivationCollector:
    """Collect activations from model layers."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

    def get_mlp_activations(self, text: str, layer_idx: int) -> np.ndarray:
        """Get MLP output activations."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
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
                return np.array(captured['output'][0].tolist())
            else:
                return np.zeros((1, 1024))
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def get_attention_activations(
        self,
        text: str,
        layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get Q, K, V projection activations."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        captured = {}
        layer = self.model.model.layers[layer_idx]
        attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.attention

        # Hook into Q, K, V projections
        original_q = attn.q_proj
        original_k = attn.k_proj
        original_v = attn.v_proj

        class QHook:
            def __init__(self, proj):
                self.proj = proj
            def __call__(self, x):
                captured['q'] = self.proj(x)
                return captured['q']

        class KHook:
            def __init__(self, proj):
                self.proj = proj
            def __call__(self, x):
                captured['k'] = self.proj(x)
                return captured['k']

        class VHook:
            def __init__(self, proj):
                self.proj = proj
            def __call__(self, x):
                captured['v'] = self.proj(x)
                return captured['v']

        attn.q_proj = QHook(original_q)
        attn.k_proj = KHook(original_k)
        attn.v_proj = VHook(original_v)

        try:
            _ = self.model(input_ids)
            for key in ['q', 'k', 'v']:
                if key in captured:
                    mx.eval(captured[key])

            q = np.array(captured['q'][0].tolist()) if 'q' in captured else np.zeros((1, 1024))
            k = np.array(captured['k'][0].tolist()) if 'k' in captured else np.zeros((1, 1024))
            v = np.array(captured['v'][0].tolist()) if 'v' in captured else np.zeros((1, 1024))

            return q, k, v
        finally:
            attn.q_proj = original_q
            attn.k_proj = original_k
            attn.v_proj = original_v


# =============================================================================
# Geometric Analysis
# =============================================================================

@dataclass
class GeometricProfile:
    """Complete geometric profile for a statement."""
    text: str
    category: str
    complexity: float
    layer_idx: int

    # TwoNN intrinsic dimension
    twonn_dim: Optional[float] = None

    # Entropy-based effective rank (for comparison)
    entropy_dim: Optional[float] = None

    # Local dimension map
    local_dim_mean: Optional[float] = None
    local_dim_std: Optional[float] = None
    n_deficient_tokens: Optional[int] = None

    # Curvature
    mean_curvature: Optional[float] = None
    curvature_variance: Optional[float] = None
    curvature_sign: Optional[str] = None

    # Attention dimension
    q_dim: Optional[float] = None
    k_dim: Optional[float] = None
    v_dim: Optional[float] = None
    attn_dim_mean: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            'text': self.text,
            'category': self.category,
            'complexity': self.complexity,
            'layer_idx': self.layer_idx,
            'twonn_dim': self.twonn_dim,
            'entropy_dim': self.entropy_dim,
            'local_dim_mean': self.local_dim_mean,
            'local_dim_std': self.local_dim_std,
            'n_deficient_tokens': self.n_deficient_tokens,
            'mean_curvature': self.mean_curvature,
            'curvature_variance': self.curvature_variance,
            'curvature_sign': self.curvature_sign,
            'q_dim': self.q_dim,
            'k_dim': self.k_dim,
            'v_dim': self.v_dim,
            'attn_dim_mean': self.attn_dim_mean,
        }


class DeepGeometricAnalyzer:
    """Full geometric analysis using ModelCypher toolkit."""

    def __init__(self, model, tokenizer, backend):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.collector = ActivationCollector(model, tokenizer)

        # Lazy-load geometry tools
        self._id_estimator = None
        self._curvature_estimator = None

    @property
    def id_estimator(self):
        if self._id_estimator is None:
            from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
            self._id_estimator = IntrinsicDimension(self.backend)
        return self._id_estimator

    @property
    def curvature_estimator(self):
        if self._curvature_estimator is None:
            from modelcypher.core.domain.geometry.manifold_curvature import SectionalCurvatureEstimator
            self._curvature_estimator = SectionalCurvatureEstimator()
        return self._curvature_estimator

    def compute_entropy_dim(self, activations: np.ndarray) -> float:
        """Entropy-based effective rank (current method)."""
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)
        try:
            _, S, _ = np.linalg.svd(activations, full_matrices=False)
            S_sq = S ** 2
            total = S_sq.sum()
            if total < 1e-10:
                return 1.0
            p = S_sq / total
            p = p[p > 1e-10]
            entropy = -np.sum(p * np.log(p))
            return float(np.exp(entropy))
        except:
            return 1.0

    def compute_twonn_dim(self, activations: np.ndarray) -> Optional[float]:
        """TwoNN geodesic intrinsic dimension."""
        if activations.shape[0] < 4:
            return None
        try:
            arr = self.backend.array(activations.astype(np.float32))
            result = self.id_estimator.compute(arr)
            return result.intrinsic_dimension
        except Exception as e:
            logger.debug(f"TwoNN failed: {e}")
            return None

    def compute_local_dimension_map(self, activations: np.ndarray) -> Tuple[float, float, int]:
        """Per-token dimension analysis."""
        if activations.shape[0] < 4:
            return 0.0, 0.0, 0
        try:
            arr = self.backend.array(activations.astype(np.float32))
            local_map = self.id_estimator.local_dimension_map(arr)
            return (
                local_map.mean_dimension,
                local_map.std_dimension,
                len(local_map.deficient_indices),
            )
        except Exception as e:
            logger.debug(f"Local dimension map failed: {e}")
            return 0.0, 0.0, 0

    def compute_curvature(self, activations: np.ndarray) -> Tuple[float, float, str]:
        """Manifold curvature analysis."""
        if activations.shape[0] < 4:
            return 0.0, 0.0, "unknown"
        try:
            arr = self.backend.array(activations.astype(np.float32))
            profile = self.curvature_estimator.estimate_manifold_profile(arr)
            return (
                profile.global_mean,
                profile.global_variance,
                profile.dominant_sign.value if profile.dominant_sign else "unknown",
            )
        except Exception as e:
            logger.debug(f"Curvature estimation failed: {e}")
            return 0.0, 0.0, "unknown"

    def analyze_statement(
        self,
        text: str,
        category: str,
        layer_idx: int,
        experiments: List[str],
    ) -> GeometricProfile:
        """Full geometric analysis of a statement."""
        complexity = compute_complexity(text)
        profile = GeometricProfile(
            text=text,
            category=category,
            complexity=complexity,
            layer_idx=layer_idx,
        )

        # Get MLP activations
        mlp_acts = self.collector.get_mlp_activations(text, layer_idx)

        # Entropy dimension (baseline)
        if 'twonn' in experiments or 'all' in experiments:
            profile.entropy_dim = self.compute_entropy_dim(mlp_acts)

        # TwoNN dimension
        if 'twonn' in experiments or 'all' in experiments:
            profile.twonn_dim = self.compute_twonn_dim(mlp_acts)

        # Local dimension map
        if 'local' in experiments or 'all' in experiments:
            mean, std, deficient = self.compute_local_dimension_map(mlp_acts)
            profile.local_dim_mean = mean
            profile.local_dim_std = std
            profile.n_deficient_tokens = deficient

        # Curvature
        if 'curvature' in experiments or 'all' in experiments:
            mean_curv, var_curv, sign = self.compute_curvature(mlp_acts)
            profile.mean_curvature = mean_curv
            profile.curvature_variance = var_curv
            profile.curvature_sign = sign

        # Attention geometry
        if 'attention' in experiments or 'all' in experiments:
            try:
                q, k, v = self.collector.get_attention_activations(text, layer_idx)
                profile.q_dim = self.compute_twonn_dim(q) or self.compute_entropy_dim(q)
                profile.k_dim = self.compute_twonn_dim(k) or self.compute_entropy_dim(k)
                profile.v_dim = self.compute_twonn_dim(v) or self.compute_entropy_dim(v)
                if profile.q_dim and profile.k_dim and profile.v_dim:
                    profile.attn_dim_mean = (profile.q_dim + profile.k_dim + profile.v_dim) / 3
            except Exception as e:
                logger.debug(f"Attention collection failed: {e}")

        return profile


# =============================================================================
# Experiments
# =============================================================================

def run_twonn_experiment(analyzer: DeepGeometricAnalyzer, layer_idx: int) -> Dict:
    """Compare TwoNN vs entropy-based dimension."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 1: TwoNN vs Entropy Dimension")
    logger.info("=" * 80)

    results = []
    complexities = []
    twonn_dims = []
    entropy_dims = []

    for category, statements in STATEMENTS.items():
        for text, _ in statements:
            profile = analyzer.analyze_statement(text, category, layer_idx, ['twonn'])
            results.append(profile)

            if profile.twonn_dim is not None and profile.entropy_dim is not None:
                complexities.append(profile.complexity)
                twonn_dims.append(profile.twonn_dim)
                entropy_dims.append(profile.entropy_dim)

                logger.info(
                    f"  [{category:8}] cpx={profile.complexity:.1f} "
                    f"twonn={profile.twonn_dim:.2f} entropy={profile.entropy_dim:.2f} | {text[:40]}"
                )

    # Compute correlations
    if len(complexities) > 3:
        complexities = np.array(complexities)
        twonn_dims = np.array(twonn_dims)
        entropy_dims = np.array(entropy_dims)

        # TwoNN correlation
        A = np.vstack([complexities, np.ones(len(complexities))]).T
        twonn_slope, twonn_intercept = np.linalg.lstsq(A, twonn_dims, rcond=None)[0]
        twonn_pred = complexities * twonn_slope + twonn_intercept
        ss_res = np.sum((twonn_dims - twonn_pred) ** 2)
        ss_tot = np.sum((twonn_dims - np.mean(twonn_dims)) ** 2)
        twonn_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Entropy correlation
        entropy_slope, entropy_intercept = np.linalg.lstsq(A, entropy_dims, rcond=None)[0]
        entropy_pred = complexities * entropy_slope + entropy_intercept
        ss_res = np.sum((entropy_dims - entropy_pred) ** 2)
        ss_tot = np.sum((entropy_dims - np.mean(entropy_dims)) ** 2)
        entropy_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        logger.info("\n" + "-" * 40)
        logger.info("RESULTS:")
        logger.info(f"  TwoNN:   dim = {twonn_slope:.3f} × complexity + {twonn_intercept:.3f}, R² = {twonn_r2:.3f}")
        logger.info(f"  Entropy: dim = {entropy_slope:.3f} × complexity + {entropy_intercept:.3f}, R² = {entropy_r2:.3f}")
        logger.info(f"  Winner: {'TwoNN' if twonn_r2 > entropy_r2 else 'Entropy'} (ΔR² = {abs(twonn_r2 - entropy_r2):.3f})")

        return {
            'twonn': {'slope': twonn_slope, 'intercept': twonn_intercept, 'r_squared': twonn_r2},
            'entropy': {'slope': entropy_slope, 'intercept': entropy_intercept, 'r_squared': entropy_r2},
            'samples': len(complexities),
        }

    return {}


def run_local_dimension_experiment(analyzer: DeepGeometricAnalyzer, layer_idx: int) -> Dict:
    """Analyze local dimension variance."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 2: Local Dimension Map")
    logger.info("=" * 80)

    results = []

    for category, statements in STATEMENTS.items():
        for text, _ in statements:
            profile = analyzer.analyze_statement(text, category, layer_idx, ['local'])
            results.append(profile)

            logger.info(
                f"  [{category:8}] mean={profile.local_dim_mean:.2f} "
                f"std={profile.local_dim_std:.2f} deficient={profile.n_deficient_tokens} | {text[:40]}"
            )

    # Aggregate by category
    category_stats = {}
    for cat in STATEMENTS.keys():
        cat_results = [r for r in results if r.category == cat]
        if cat_results:
            stds = [r.local_dim_std for r in cat_results if r.local_dim_std is not None]
            deficients = [r.n_deficient_tokens for r in cat_results if r.n_deficient_tokens is not None]
            category_stats[cat] = {
                'mean_std': float(np.mean(stds)) if stds else 0,
                'mean_deficient': float(np.mean(deficients)) if deficients else 0,
                'n': len(cat_results),
            }

    logger.info("\n" + "-" * 40)
    logger.info("BY CATEGORY:")
    for cat, stats in category_stats.items():
        logger.info(f"  {cat:8}: std={stats['mean_std']:.3f}, deficient={stats['mean_deficient']:.1f}")

    return {'category_stats': category_stats}


def run_curvature_experiment(analyzer: DeepGeometricAnalyzer, layer_idx: int) -> Dict:
    """Analyze manifold curvature."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 3: Manifold Curvature")
    logger.info("=" * 80)

    results = []

    for category, statements in STATEMENTS.items():
        for text, _ in statements:
            profile = analyzer.analyze_statement(text, category, layer_idx, ['curvature'])
            results.append(profile)

            logger.info(
                f"  [{category:8}] curv={profile.mean_curvature:.4f} "
                f"var={profile.curvature_variance:.4f} sign={profile.curvature_sign} | {text[:40]}"
            )

    # Aggregate by category
    category_stats = {}
    for cat in STATEMENTS.keys():
        cat_results = [r for r in results if r.category == cat]
        if cat_results:
            curvs = [r.mean_curvature for r in cat_results if r.mean_curvature is not None]
            signs = [r.curvature_sign for r in cat_results if r.curvature_sign]
            category_stats[cat] = {
                'mean_curvature': float(np.mean(curvs)) if curvs else 0,
                'dominant_sign': max(set(signs), key=signs.count) if signs else 'unknown',
                'n': len(cat_results),
            }

    logger.info("\n" + "-" * 40)
    logger.info("BY CATEGORY:")
    for cat, stats in category_stats.items():
        logger.info(f"  {cat:8}: curvature={stats['mean_curvature']:.4f}, sign={stats['dominant_sign']}")

    return {'category_stats': category_stats}


def run_subspace_experiment(analyzer: DeepGeometricAnalyzer, layer_idx: int) -> Dict:
    """Analyze subspace overlap between categories.

    Do different complexity levels use orthogonal subspaces?
    This could explain why Layer 4 fails on complex statements.
    """
    from modelcypher.core.domain.geometry.subspace import (
        compute_grassmann_distance,
        compute_subspace_overlap,
    )

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 4: Subspace Structure Analysis")
    logger.info("=" * 80)

    # Collect activations for each category
    category_activations = {}

    for category, statements in STATEMENTS.items():
        all_acts = []
        for text, _ in statements:
            acts = analyzer.collector.get_mlp_activations(text, layer_idx)
            all_acts.append(acts)  # All tokens

        # Aggregate into one matrix per category
        category_activations[category] = np.vstack(all_acts)
        logger.info(f"  {category}: {category_activations[category].shape[0]} activation vectors")

    # Compute pairwise Grassmann distances
    categories = list(STATEMENTS.keys())
    n_cats = len(categories)

    grassmann_matrix = np.zeros((n_cats, n_cats))
    overlap_matrix = np.zeros((n_cats, n_cats))

    logger.info("\nComputing pairwise subspace distances...")

    for i in range(n_cats):
        for j in range(i+1, n_cats):
            cat_a, cat_b = categories[i], categories[j]
            acts_a = category_activations[cat_a]
            acts_b = category_activations[cat_b]

            # Convert to backend arrays
            arr_a = analyzer.backend.array(acts_a.astype(np.float32))
            arr_b = analyzer.backend.array(acts_b.astype(np.float32))

            try:
                # Compute subspace overlap (requires same shape, use first N of each)
                n_common = min(acts_a.shape[0], acts_b.shape[0])
                overlap_result = compute_subspace_overlap(
                    arr_a[:n_common], arr_b[:n_common], analyzer.backend
                )
                overlap_fraction = overlap_result.overlap_fraction

                # For Grassmann distance, we need basis vectors
                # Get top-k singular vectors for each
                _, _, Vt_a = np.linalg.svd(acts_a, full_matrices=False)
                _, _, Vt_b = np.linalg.svd(acts_b, full_matrices=False)

                # Use top 10 directions (or less if not available)
                k = min(10, Vt_a.shape[0], Vt_b.shape[0])
                basis_a = analyzer.backend.array(Vt_a[:k].astype(np.float32))
                basis_b = analyzer.backend.array(Vt_b[:k].astype(np.float32))

                grassmann_result = compute_grassmann_distance(basis_a, basis_b, analyzer.backend)
                grassmann_dist = grassmann_result.geodesic_distance

                grassmann_matrix[i, j] = grassmann_dist
                grassmann_matrix[j, i] = grassmann_dist
                overlap_matrix[i, j] = overlap_fraction
                overlap_matrix[j, i] = overlap_fraction

                logger.info(
                    f"  {cat_a:8} ↔ {cat_b:8}: "
                    f"grassmann={grassmann_dist:.3f}, overlap={overlap_fraction:.2%}"
                )
            except Exception as e:
                logger.warning(f"  {cat_a} ↔ {cat_b}: Failed - {e}")

    # Key comparisons
    logger.info("\n" + "-" * 40)
    logger.info("KEY COMPARISONS:")

    # Simple vs Meta (should be most different)
    simple_meta_dist = grassmann_matrix[categories.index('simple'), categories.index('meta')]
    simple_meta_overlap = overlap_matrix[categories.index('simple'), categories.index('meta')]
    logger.info(f"  simple ↔ meta: grassmann={simple_meta_dist:.3f}, overlap={simple_meta_overlap:.2%}")

    # Adjacent complexity levels
    logger.info("\n  Adjacent complexity transitions:")
    for i in range(len(categories) - 1):
        dist = grassmann_matrix[i, i+1]
        overlap = overlap_matrix[i, i+1]
        logger.info(f"    {categories[i]:8} → {categories[i+1]:8}: grassmann={dist:.3f}, overlap={overlap:.2%}")

    return {
        'categories': categories,
        'grassmann_matrix': grassmann_matrix.tolist(),
        'overlap_matrix': overlap_matrix.tolist(),
        'simple_meta_grassmann': float(simple_meta_dist),
        'simple_meta_overlap': float(simple_meta_overlap),
    }


def run_flow_experiment(analyzer: DeepGeometricAnalyzer, target_layer_idx: int) -> Dict:
    """Analyze layer information flow using CKA.

    Track when representations stabilize across layers.
    Hypothesis: meta-cognitive has LOW early CKA, HIGH late CKA.
    """
    from modelcypher.core.domain.geometry.cka import compute_geodesic_cka

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 5: Layer Information Flow (CKA)")
    logger.info("=" * 80)

    n_layers = analyzer.collector.n_layers

    # Collect activations for a representative statement from each category
    representative = {
        'simple': "Fire is hot",
        'factual': "Paris is the capital of France",
        'belief': "I believe dogs are loyal animals",
        'meta': "I think I understand why people like mathematics",
    }

    results = {}

    for category, text in representative.items():
        logger.info(f"\n{category}: '{text[:40]}...'")

        # Collect activations at each layer
        layer_acts = []
        for layer_idx in range(n_layers):
            acts = analyzer.collector.get_mlp_activations(text, layer_idx)
            layer_acts.append(acts)

        # Compute CKA between adjacent layers
        cka_values = []
        for i in range(n_layers - 1):
            try:
                arr_a = analyzer.backend.array(layer_acts[i].astype(np.float32))
                arr_b = analyzer.backend.array(layer_acts[i + 1].astype(np.float32))
                cka = compute_geodesic_cka(arr_a, arr_b, analyzer.backend)
                cka_values.append(cka)
            except Exception as e:
                logger.debug(f"CKA L{i}→L{i+1} failed: {e}")
                cka_values.append(0.0)

        # Find stabilization layer (where CKA > 0.9)
        stabilization_layer = None
        for i, cka in enumerate(cka_values):
            if cka > 0.9:
                stabilization_layer = i
                break

        # Log CKA trajectory
        logger.info(f"  CKA trajectory: " + " ".join([f"{c:.2f}" for c in cka_values]))
        if stabilization_layer is not None:
            logger.info(f"  Stabilization layer: {stabilization_layer} (CKA > 0.9)")
        else:
            logger.info(f"  No stabilization (max CKA = {max(cka_values):.2f})")

        results[category] = {
            'cka_values': cka_values,
            'stabilization_layer': stabilization_layer,
            'max_cka': max(cka_values) if cka_values else 0.0,
            'mean_cka': float(np.mean(cka_values)) if cka_values else 0.0,
        }

    # Compare categories
    logger.info("\n" + "-" * 40)
    logger.info("SUMMARY:")
    for cat, data in results.items():
        stab = data['stabilization_layer']
        stab_str = f"L{stab}" if stab is not None else "never"
        logger.info(f"  {cat:8}: stabilizes={stab_str}, mean_CKA={data['mean_cka']:.3f}")

    return results


def run_attention_experiment(analyzer: DeepGeometricAnalyzer, layer_idx: int) -> Dict:
    """Compare attention vs MLP dimension."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 6: Attention Geometry")
    logger.info("=" * 80)

    results = []

    for category, statements in STATEMENTS.items():
        for text, _ in statements:
            profile = analyzer.analyze_statement(text, category, layer_idx, ['attention', 'twonn'])
            results.append(profile)

            if profile.attn_dim_mean and profile.twonn_dim:
                ratio = profile.attn_dim_mean / profile.twonn_dim if profile.twonn_dim > 0 else 0
                logger.info(
                    f"  [{category:8}] attn={profile.attn_dim_mean:.2f} mlp={profile.twonn_dim:.2f} "
                    f"ratio={ratio:.2f} | {text[:40]}"
                )

    # Aggregate by category
    category_stats = {}
    for cat in STATEMENTS.keys():
        cat_results = [r for r in results if r.category == cat and r.attn_dim_mean and r.twonn_dim]
        if cat_results:
            ratios = [r.attn_dim_mean / r.twonn_dim for r in cat_results if r.twonn_dim > 0]
            category_stats[cat] = {
                'mean_attn_dim': float(np.mean([r.attn_dim_mean for r in cat_results])),
                'mean_mlp_dim': float(np.mean([r.twonn_dim for r in cat_results])),
                'mean_ratio': float(np.mean(ratios)) if ratios else 0,
                'n': len(cat_results),
            }

    logger.info("\n" + "-" * 40)
    logger.info("BY CATEGORY:")
    for cat, stats in category_stats.items():
        logger.info(
            f"  {cat:8}: attn={stats['mean_attn_dim']:.2f} mlp={stats['mean_mlp_dim']:.2f} "
            f"ratio={stats['mean_ratio']:.2f}"
        )

    return {'category_stats': category_stats}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deep Geometric Analysis")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--experiment",
        choices=['twonn', 'local', 'curvature', 'subspace', 'flow', 'attention', 'all'],
        default='twonn',
        help="Which experiment to run"
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
    backend = initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Determine layer
    if hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    else:
        n_layers = 24
    layer_idx = args.layer if args.layer is not None else n_layers // 2

    logger.info(f"Model has {n_layers} layers, analyzing layer {layer_idx}")

    # Create analyzer
    analyzer = DeepGeometricAnalyzer(model, tokenizer, backend)

    # Run experiments
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'layer': layer_idx,
        'experiment': args.experiment,
    }

    if args.experiment == 'twonn' or args.experiment == 'all':
        results['twonn'] = run_twonn_experiment(analyzer, layer_idx)

    if args.experiment == 'local' or args.experiment == 'all':
        results['local'] = run_local_dimension_experiment(analyzer, layer_idx)

    if args.experiment == 'curvature' or args.experiment == 'all':
        results['curvature'] = run_curvature_experiment(analyzer, layer_idx)

    if args.experiment == 'subspace' or args.experiment == 'all':
        results['subspace'] = run_subspace_experiment(analyzer, layer_idx)

    if args.experiment == 'flow' or args.experiment == 'all':
        results['flow'] = run_flow_experiment(analyzer, layer_idx)

    if args.experiment == 'attention' or args.experiment == 'all':
        results['attention'] = run_attention_experiment(analyzer, layer_idx)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "deep_geometric"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"results_{args.experiment}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
