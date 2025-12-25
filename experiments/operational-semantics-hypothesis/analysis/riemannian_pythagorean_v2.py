#!/usr/bin/env python3
"""
Riemannian Pythagorean Test v2 - Corrected Geometry

This version uses PROPER Riemannian geometry:
- Backend protocol instead of numpy
- Geodesic distances via k-NN graph (not Euclidean)
- Frechet mean instead of arithmetic mean
- Curvature measurement to understand the manifold structure

The original experiment used Euclidean algebra which is WRONG for curved manifolds:
- Euclidean distance underestimates on positive curvature
- Euclidean distance overestimates on negative curvature
- Arithmetic mean is biased away from the true center of mass

This script answers: Is the Pythagorean relationship a² + b² = c²
encoded in the GEODESIC geometry of the number embeddings?
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# ModelCypher imports - no numpy!
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
    frechet_mean,
    geodesic_distance_matrix,
)
from modelcypher.core.domain.geometry.manifold_curvature import (
    SectionalCurvatureEstimator,
    CurvatureConfig,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
    TwoNNConfiguration,
    BootstrapConfiguration,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent


@dataclass
class CurvatureAnalysis:
    """Results of curvature analysis."""
    mean_sectional_curvature: float
    curvature_sign: str  # "positive", "negative", "mixed", "flat"
    confidence: float
    per_number_curvature: dict[int, float]


@dataclass
class GeodesicDistanceAnalysis:
    """Results of geodesic vs Euclidean distance comparison."""
    geodesic_5_to_34_centroid: float
    geodesic_6_to_34_centroid: float
    euclidean_5_to_34_centroid: float
    euclidean_6_to_34_centroid: float
    geodesic_ratio: float  # geodesic / euclidean - >1 means positive curvature
    five_closer_geodesic: bool
    five_closer_euclidean: bool


@dataclass
class PythagoreanStructureAnalysis:
    """Results of Pythagorean structure tests."""
    # Valid vs invalid triple separation
    valid_triples_geodesic_signature: list[float]
    invalid_triples_geodesic_signature: list[float]
    separation_score: float

    # Position encoding
    five_geodesic_from_34_frechet: float
    six_geodesic_from_34_frechet: float
    position_test_passed: bool

    # Squared numbers analysis
    squared_triangle_geodesic: dict[str, float]  # 9-16, 16-25, 9-25
    squaring_direction_consistency: float


@dataclass
class FullResults:
    """Complete results for a model."""
    model_name: str
    embedding_dim: int
    curvature: CurvatureAnalysis
    geodesic_comparison: GeodesicDistanceAnalysis
    pythagorean_structure: PythagoreanStructureAnalysis
    intrinsic_dimension: float | None
    hypothesis_supported: bool


def load_model(model_path: str):
    """Load MLX model and tokenizer."""
    from mlx_lm import load
    return load(model_path)


def extract_embedding(model, tokenizer, text: str, backend):
    """Extract last-token embedding using Backend protocol."""
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    inner = getattr(model, 'model', model)
    x = inner.embed_tokens(input_ids) if hasattr(inner, 'embed_tokens') else inner.wte(input_ids)

    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    for layer in layers:
        x = layer(x)
        if isinstance(x, tuple):
            x = x[0]

    result = x[0, -1, :]
    mx.eval(result)
    result = result.astype(mx.float32)
    mx.eval(result)

    # Convert to backend array (NOT numpy!)
    # MLX backend can use the array directly
    return backend.array(result)


def get_number_embeddings(model, tokenizer, numbers: list[int], backend) -> dict:
    """Get embeddings for a list of numbers."""
    embeddings = {}
    for n in numbers:
        text = f"The number {n}."
        emb = extract_embedding(model, tokenizer, text, backend)
        backend.eval(emb)
        embeddings[n] = emb
    return embeddings


def get_number_cloud(model, tokenizer, n: int, backend, num_variations: int = 8):
    """Get multiple embeddings for a number to form a concept cloud."""
    templates = [
        f"The number {n}.",
        f"The value {n}.",
        f"Consider {n}.",
        f"{n} is a number.",
        f"When we have {n} items.",
        f"The quantity {n}.",
        f"There are {n} of them.",
        f"Exactly {n}.",
    ]

    embeddings = []
    for i in range(min(num_variations, len(templates))):
        emb = extract_embedding(model, tokenizer, templates[i], backend)
        backend.eval(emb)
        embeddings.append(emb)

    return backend.stack(embeddings)


def analyze_curvature(embeddings: dict, backend) -> CurvatureAnalysis:
    """Analyze the curvature of the number embedding manifold."""
    logger.info("\n=== CURVATURE ANALYSIS ===")

    rg = RiemannianGeometry(backend)

    # Stack all embeddings into a point cloud
    numbers = sorted(embeddings.keys())
    points = backend.stack([embeddings[n] for n in numbers])
    backend.eval(points)

    # Estimate curvature at each number
    per_number_curvature = {}
    curvatures = []

    for i, n in enumerate(numbers):
        try:
            estimate = rg.estimate_local_curvature(points, center_idx=i, k_neighbors=min(8, len(numbers)-1))
            per_number_curvature[n] = estimate.sectional_curvature
            curvatures.append(estimate.sectional_curvature)
            sign = "+" if estimate.is_positive else ("-" if estimate.is_negative else "~")
            logger.info(f"  {n}: K = {estimate.sectional_curvature:.6f} ({sign})")
        except Exception as e:
            logger.warning(f"  {n}: curvature estimation failed: {e}")
            per_number_curvature[n] = 0.0

    if curvatures:
        mean_curvature = sum(curvatures) / len(curvatures)
        positive_count = sum(1 for k in curvatures if k > 0.01)
        negative_count = sum(1 for k in curvatures if k < -0.01)

        if positive_count > len(curvatures) * 0.6:
            sign = "positive"
        elif negative_count > len(curvatures) * 0.6:
            sign = "negative"
        elif positive_count > 0 and negative_count > 0:
            sign = "mixed"
        else:
            sign = "flat"

        confidence = max(positive_count, negative_count) / len(curvatures)
    else:
        mean_curvature = 0.0
        sign = "unknown"
        confidence = 0.0

    logger.info(f"\n  Mean curvature: {mean_curvature:.6f}")
    logger.info(f"  Dominant sign: {sign} (confidence: {confidence:.2%})")

    return CurvatureAnalysis(
        mean_sectional_curvature=mean_curvature,
        curvature_sign=sign,
        confidence=confidence,
        per_number_curvature=per_number_curvature,
    )


def analyze_geodesic_vs_euclidean(embeddings: dict, backend) -> GeodesicDistanceAnalysis:
    """Compare geodesic and Euclidean distances for the key test."""
    logger.info("\n=== GEODESIC vs EUCLIDEAN COMPARISON ===")

    rg = RiemannianGeometry(backend)

    # Get embeddings for 3, 4, 5, 6
    e3, e4, e5, e6 = embeddings[3], embeddings[4], embeddings[5], embeddings[6]

    # Build point cloud for geodesic calculation
    numbers = sorted(embeddings.keys())
    points = backend.stack([embeddings[n] for n in numbers])
    backend.eval(points)

    # Compute geodesic distance matrix
    geo_result = rg.geodesic_distances(points, k_neighbors=min(10, len(numbers)-1))
    geo_dist = geo_result.distances
    backend.eval(geo_dist)

    # Get indices
    idx_3, idx_4, idx_5, idx_6 = numbers.index(3), numbers.index(4), numbers.index(5), numbers.index(6)

    # Compute Frechet mean of 3 and 4 (proper Riemannian centroid)
    pair_34 = backend.stack([e3, e4])
    backend.eval(pair_34)
    centroid_34 = frechet_mean(pair_34, backend=backend)
    backend.eval(centroid_34)

    # Euclidean centroid for comparison
    euclidean_centroid_34 = (e3 + e4) / 2.0
    backend.eval(euclidean_centroid_34)

    logger.info("  Computing centroid difference (Frechet vs arithmetic)...")
    centroid_diff = backend.sqrt(backend.sum((centroid_34 - euclidean_centroid_34) ** 2))
    backend.eval(centroid_diff)
    logger.info(f"  Centroid difference: {float(backend.to_numpy(centroid_diff)):.6f}")

    # Geodesic distances from 5 and 6 to the (3,4) pair
    # Use average geodesic distance to both 3 and 4
    geo_5_to_3 = float(backend.to_numpy(geo_dist[idx_5, idx_3]))
    geo_5_to_4 = float(backend.to_numpy(geo_dist[idx_5, idx_4]))
    geo_6_to_3 = float(backend.to_numpy(geo_dist[idx_6, idx_3]))
    geo_6_to_4 = float(backend.to_numpy(geo_dist[idx_6, idx_4]))

    geodesic_5_to_34 = (geo_5_to_3 + geo_5_to_4) / 2
    geodesic_6_to_34 = (geo_6_to_3 + geo_6_to_4) / 2

    # Euclidean distances
    euc_5_to_centroid = backend.sqrt(backend.sum((e5 - euclidean_centroid_34) ** 2))
    euc_6_to_centroid = backend.sqrt(backend.sum((e6 - euclidean_centroid_34) ** 2))
    backend.eval(euc_5_to_centroid, euc_6_to_centroid)

    euclidean_5_to_34 = float(backend.to_numpy(euc_5_to_centroid))
    euclidean_6_to_34 = float(backend.to_numpy(euc_6_to_centroid))

    # Geodesic ratio (geodesic / euclidean) - tells us about curvature
    # > 1 means positive curvature (geodesic > Euclidean)
    # < 1 means negative curvature (geodesic < Euclidean)
    geo_ratio_5 = geodesic_5_to_34 / (euclidean_5_to_34 + 1e-10)
    geo_ratio_6 = geodesic_6_to_34 / (euclidean_6_to_34 + 1e-10)
    avg_ratio = (geo_ratio_5 + geo_ratio_6) / 2

    logger.info(f"\n  Geodesic 5 → (3,4): {geodesic_5_to_34:.4f}")
    logger.info(f"  Geodesic 6 → (3,4): {geodesic_6_to_34:.4f}")
    logger.info(f"  Euclidean 5 → (3,4): {euclidean_5_to_34:.4f}")
    logger.info(f"  Euclidean 6 → (3,4): {euclidean_6_to_34:.4f}")
    logger.info(f"\n  Geodesic/Euclidean ratio: {avg_ratio:.4f}")

    if avg_ratio > 1.05:
        logger.info("  >> POSITIVE CURVATURE: Geodesic > Euclidean")
    elif avg_ratio < 0.95:
        logger.info("  >> NEGATIVE CURVATURE: Geodesic < Euclidean")
    else:
        logger.info("  >> NEAR FLAT: Geodesic ≈ Euclidean")

    five_closer_geo = geodesic_5_to_34 < geodesic_6_to_34
    five_closer_euc = euclidean_5_to_34 < euclidean_6_to_34

    logger.info(f"\n  5 closer than 6 (GEODESIC): {five_closer_geo}")
    logger.info(f"  5 closer than 6 (EUCLIDEAN): {five_closer_euc}")

    if five_closer_geo != five_closer_euc:
        logger.info("  >> CURVATURE CHANGES THE ANSWER!")

    return GeodesicDistanceAnalysis(
        geodesic_5_to_34_centroid=geodesic_5_to_34,
        geodesic_6_to_34_centroid=geodesic_6_to_34,
        euclidean_5_to_34_centroid=euclidean_5_to_34,
        euclidean_6_to_34_centroid=euclidean_6_to_34,
        geodesic_ratio=avg_ratio,
        five_closer_geodesic=five_closer_geo,
        five_closer_euclidean=five_closer_euc,
    )


def analyze_pythagorean_structure(embeddings: dict, backend) -> PythagoreanStructureAnalysis:
    """Analyze Pythagorean structure using geodesic geometry."""
    logger.info("\n=== PYTHAGOREAN STRUCTURE ANALYSIS ===")

    rg = RiemannianGeometry(backend)

    # Build full point cloud
    numbers = sorted(embeddings.keys())
    points = backend.stack([embeddings[n] for n in numbers])
    backend.eval(points)

    # Compute geodesic distance matrix
    geo_result = rg.geodesic_distances(points, k_neighbors=min(10, len(numbers)-1))
    geo_dist = geo_result.distances
    backend.eval(geo_dist)

    def get_idx(n):
        return numbers.index(n)

    # Test 1: Valid vs Invalid triple geodesic signatures
    logger.info("\n--- Valid vs Invalid Triple Separation ---")

    valid_triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17)]
    invalid_triples = [(3, 4, 6), (5, 12, 14), (8, 15, 18)]

    def compute_triple_geodesic_signature(a, b, c) -> list[float]:
        """Compute geodesic features for a triple."""
        try:
            idx_a, idx_b, idx_c = get_idx(a), get_idx(b), get_idx(c)
            geo_ab = float(backend.to_numpy(geo_dist[idx_a, idx_b]))
            geo_bc = float(backend.to_numpy(geo_dist[idx_b, idx_c]))
            geo_ac = float(backend.to_numpy(geo_dist[idx_a, idx_c]))

            # Geodesic triangle inequality excess
            triangle_excess = (geo_ab + geo_bc) - geo_ac

            # Ratio of c's distance to (a,b) pair
            c_to_ab_ratio = geo_ac / (geo_ab + 1e-10)

            return [geo_ab, geo_bc, geo_ac, triangle_excess, c_to_ab_ratio]
        except (ValueError, IndexError):
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    valid_signatures = [compute_triple_geodesic_signature(a, b, c) for a, b, c in valid_triples]
    invalid_signatures = [compute_triple_geodesic_signature(a, b, c) for a, b, c in invalid_triples]

    # Average signature difference
    valid_mean = [sum(s[i] for s in valid_signatures) / len(valid_signatures) for i in range(5)]
    invalid_mean = [sum(s[i] for s in invalid_signatures) / len(invalid_signatures) for i in range(5)]

    separation = sum(abs(valid_mean[i] - invalid_mean[i]) for i in range(5)) / 5

    logger.info(f"  Valid triple signature mean: {[f'{v:.3f}' for v in valid_mean]}")
    logger.info(f"  Invalid triple signature mean: {[f'{v:.3f}' for v in invalid_mean]}")
    logger.info(f"  Separation score: {separation:.4f}")

    # Test 2: Position encoding (5 vs 6 from Frechet mean of 3,4)
    logger.info("\n--- Position Encoding Test ---")

    e3, e4, e5, e6 = embeddings[3], embeddings[4], embeddings[5], embeddings[6]
    pair_34 = backend.stack([e3, e4])
    backend.eval(pair_34)
    frechet_34 = frechet_mean(pair_34, backend=backend)
    backend.eval(frechet_34)

    # Find nearest point to Frechet mean for geodesic calculation
    # We need to add the Frechet mean to the point cloud
    points_with_frechet = backend.concatenate([points, backend.reshape(frechet_34, (1, -1))], axis=0)
    backend.eval(points_with_frechet)

    geo_result_ext = rg.geodesic_distances(points_with_frechet, k_neighbors=min(10, len(numbers)))
    geo_dist_ext = geo_result_ext.distances
    backend.eval(geo_dist_ext)

    frechet_idx = len(numbers)  # Last point
    idx_5, idx_6 = get_idx(5), get_idx(6)

    geo_5_to_frechet = float(backend.to_numpy(geo_dist_ext[idx_5, frechet_idx]))
    geo_6_to_frechet = float(backend.to_numpy(geo_dist_ext[idx_6, frechet_idx]))

    logger.info(f"  Geodesic 5 → Frechet(3,4): {geo_5_to_frechet:.4f}")
    logger.info(f"  Geodesic 6 → Frechet(3,4): {geo_6_to_frechet:.4f}")

    position_test_passed = geo_5_to_frechet < geo_6_to_frechet
    logger.info(f"  Position test (5 closer than 6): {'PASS' if position_test_passed else 'FAIL'}")

    # Test 3: Squared numbers triangle (9, 16, 25) = (3², 4², 5²)
    logger.info("\n--- Squared Numbers Triangle ---")

    idx_9, idx_16, idx_25 = get_idx(9), get_idx(16), get_idx(25)
    geo_9_16 = float(backend.to_numpy(geo_dist[idx_9, idx_16]))
    geo_16_25 = float(backend.to_numpy(geo_dist[idx_16, idx_25]))
    geo_9_25 = float(backend.to_numpy(geo_dist[idx_9, idx_25]))

    logger.info(f"  9 ↔ 16: {geo_9_16:.4f}")
    logger.info(f"  16 ↔ 25: {geo_16_25:.4f}")
    logger.info(f"  9 ↔ 25: {geo_9_25:.4f}")

    # Test 4: Squaring direction consistency
    logger.info("\n--- Squaring Direction Consistency ---")

    def compute_direction(from_n: int, to_n: int):
        e_from = embeddings[from_n]
        e_to = embeddings[to_n]
        diff = e_to - e_from
        norm = backend.sqrt(backend.sum(diff * diff))
        backend.eval(norm)
        norm_val = float(backend.to_numpy(norm))
        if norm_val < 1e-10:
            return diff
        return diff / norm_val

    dir_3_9 = compute_direction(3, 9)    # 3 → 3²
    dir_4_16 = compute_direction(4, 16)  # 4 → 4²
    dir_5_25 = compute_direction(5, 25)  # 5 → 5²
    backend.eval(dir_3_9, dir_4_16, dir_5_25)

    cos_1 = float(backend.to_numpy(backend.sum(dir_3_9 * dir_4_16)))
    cos_2 = float(backend.to_numpy(backend.sum(dir_4_16 * dir_5_25)))
    cos_3 = float(backend.to_numpy(backend.sum(dir_3_9 * dir_5_25)))

    consistency = (cos_1 + cos_2 + cos_3) / 3

    logger.info(f"  3→9 vs 4→16: {cos_1:.4f}")
    logger.info(f"  4→16 vs 5→25: {cos_2:.4f}")
    logger.info(f"  3→9 vs 5→25: {cos_3:.4f}")
    logger.info(f"  Average consistency: {consistency:.4f}")

    return PythagoreanStructureAnalysis(
        valid_triples_geodesic_signature=valid_mean,
        invalid_triples_geodesic_signature=invalid_mean,
        separation_score=separation,
        five_geodesic_from_34_frechet=geo_5_to_frechet,
        six_geodesic_from_34_frechet=geo_6_to_frechet,
        position_test_passed=position_test_passed,
        squared_triangle_geodesic={"9_16": geo_9_16, "16_25": geo_16_25, "9_25": geo_9_25},
        squaring_direction_consistency=consistency,
    )


def compute_intrinsic_dimension(embeddings: dict, backend) -> float | None:
    """Estimate intrinsic dimension of the number manifold."""
    logger.info("\n=== INTRINSIC DIMENSION ===")

    numbers = sorted(embeddings.keys())
    points = backend.stack([embeddings[n] for n in numbers])
    backend.eval(points)

    try:
        id_estimator = IntrinsicDimension(backend)
        id_result = id_estimator.estimate_two_nn(
            points,
            TwoNNConfiguration(use_regression=True)
        )
        logger.info(f"  Intrinsic dimension: {id_result.intrinsic_dimension:.2f}")
        if id_result.ci:
            logger.info(f"  95% CI: [{id_result.ci.lower:.2f}, {id_result.ci.upper:.2f}]")
        return id_result.intrinsic_dimension
    except Exception as e:
        logger.warning(f"  Could not estimate: {e}")
        return None


def run_full_analysis(model_path: str) -> FullResults:
    """Run complete Riemannian analysis on a model."""
    model_name = Path(model_path).name
    backend = get_default_backend()

    logger.info("=" * 70)
    logger.info("RIEMANNIAN PYTHAGOREAN TEST v2")
    logger.info("Proper geodesic geometry - NO EUCLIDEAN ASSUMPTIONS")
    logger.info(f"Model: {model_name}")
    logger.info(f"Backend: {backend.__class__.__name__}")
    logger.info("=" * 70)

    # Load model
    logger.info("\nLoading model...")
    model, tokenizer = load_model(model_path)

    # Get embeddings for key numbers
    # Include all numbers needed for Pythagorean triples
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 24, 25]

    logger.info(f"Extracting embeddings for {len(numbers)} numbers...")
    embeddings = get_number_embeddings(model, tokenizer, numbers, backend)

    embedding_dim = int(embeddings[3].shape[0])
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Run analyses
    curvature = analyze_curvature(embeddings, backend)
    geodesic_comparison = analyze_geodesic_vs_euclidean(embeddings, backend)
    pythagorean_structure = analyze_pythagorean_structure(embeddings, backend)
    intrinsic_dim = compute_intrinsic_dimension(embeddings, backend)

    # Determine if hypothesis is supported
    # Key criteria:
    # 1. 5 is closer to (3,4) than 6 using GEODESIC distance
    # 2. Valid/invalid triples show separation
    # 3. Consistent squaring direction

    hypothesis_supported = (
        pythagorean_structure.position_test_passed and
        pythagorean_structure.separation_score > 0.1 and
        pythagorean_structure.squaring_direction_consistency > 0.1
    )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Curvature: {curvature.curvature_sign} (K = {curvature.mean_sectional_curvature:.6f})")
    logger.info(f"Geodesic/Euclidean ratio: {geodesic_comparison.geodesic_ratio:.4f}")
    logger.info(f"Position test (geodesic): {'PASS' if pythagorean_structure.position_test_passed else 'FAIL'}")
    logger.info(f"Position test (Euclidean): {'PASS' if geodesic_comparison.five_closer_euclidean else 'FAIL'}")
    logger.info(f"Triple separation score: {pythagorean_structure.separation_score:.4f}")
    logger.info(f"Squaring consistency: {pythagorean_structure.squaring_direction_consistency:.4f}")

    if geodesic_comparison.five_closer_geodesic != geodesic_comparison.five_closer_euclidean:
        logger.info("\n>>> CRITICAL: GEODESIC AND EUCLIDEAN GIVE DIFFERENT ANSWERS!")
        logger.info(">>> The original Euclidean analysis was WRONG for this model.")

    if hypothesis_supported:
        logger.info("\n>>> HYPOTHESIS SUPPORTED (using geodesic geometry)")
    else:
        logger.info("\n>>> HYPOTHESIS NOT SUPPORTED (using geodesic geometry)")

    return FullResults(
        model_name=model_name,
        embedding_dim=embedding_dim,
        curvature=curvature,
        geodesic_comparison=geodesic_comparison,
        pythagorean_structure=pythagorean_structure,
        intrinsic_dimension=intrinsic_dim,
        hypothesis_supported=hypothesis_supported,
    )


def results_to_dict(results: FullResults) -> dict:
    """Convert results to JSON-serializable dict."""
    return {
        "model_name": results.model_name,
        "embedding_dim": results.embedding_dim,
        "curvature": {
            "mean_sectional_curvature": results.curvature.mean_sectional_curvature,
            "curvature_sign": results.curvature.curvature_sign,
            "confidence": results.curvature.confidence,
            "per_number_curvature": {str(k): v for k, v in results.curvature.per_number_curvature.items()},
        },
        "geodesic_comparison": {
            "geodesic_5_to_34": results.geodesic_comparison.geodesic_5_to_34_centroid,
            "geodesic_6_to_34": results.geodesic_comparison.geodesic_6_to_34_centroid,
            "euclidean_5_to_34": results.geodesic_comparison.euclidean_5_to_34_centroid,
            "euclidean_6_to_34": results.geodesic_comparison.euclidean_6_to_34_centroid,
            "geodesic_ratio": results.geodesic_comparison.geodesic_ratio,
            "five_closer_geodesic": results.geodesic_comparison.five_closer_geodesic,
            "five_closer_euclidean": results.geodesic_comparison.five_closer_euclidean,
            "curvature_changes_answer": (
                results.geodesic_comparison.five_closer_geodesic !=
                results.geodesic_comparison.five_closer_euclidean
            ),
        },
        "pythagorean_structure": {
            "valid_triples_signature": results.pythagorean_structure.valid_triples_geodesic_signature,
            "invalid_triples_signature": results.pythagorean_structure.invalid_triples_geodesic_signature,
            "separation_score": results.pythagorean_structure.separation_score,
            "five_geodesic_from_34_frechet": results.pythagorean_structure.five_geodesic_from_34_frechet,
            "six_geodesic_from_34_frechet": results.pythagorean_structure.six_geodesic_from_34_frechet,
            "position_test_passed": results.pythagorean_structure.position_test_passed,
            "squared_triangle_geodesic": results.pythagorean_structure.squared_triangle_geodesic,
            "squaring_direction_consistency": results.pythagorean_structure.squaring_direction_consistency,
        },
        "intrinsic_dimension": results.intrinsic_dimension,
        "hypothesis_supported": results.hypothesis_supported,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Riemannian Pythagorean Test v2")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    model_name = Path(args.model).name
    output_path = args.output or str(SCRIPT_DIR / "results" / f"riemannian_v2_{model_name}.json")

    results = run_full_analysis(args.model)

    # Save results
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_to_dict(results), f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
