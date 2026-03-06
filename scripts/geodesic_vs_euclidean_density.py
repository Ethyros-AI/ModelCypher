#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Geodesic vs Euclidean distance experiment with controls and gated decisions.
#
# This script implements a 6-phase protocol:
#   Phase 1: Flat-space null hypothesis
#   Phase 2: Known-curvature positive control
#   Phase 3: Sample-size sensitivity on real activations
#   Phase 4: Permutation control
#   Phase 5: Direct curvature measurement
#   Phase 6: Downstream merge validation
#
# Each phase writes JSON artifacts under:
#   results/geodesic_vs_euclidean/phase_{N}/
#
# The final decision is codified (not interpreted manually) and written to:
#   results/geodesic_vs_euclidean/summary.json
#   results/geodesic_vs_euclidean/decision.json

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from modelcypher.adapters.model_backbone import forward_through_backbone, resolve_model_backbone
from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.manifold_curvature import SectionalCurvatureEstimator
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.ollivier_ricci import OllivierRicciCurvature
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.geometry.riemannian_validation import derive_k_neighbors


LOG = logging.getLogger("geodesic_vs_euclidean_density")

PHASE1_N_VALUES = [15, 30, 50, 100, 200]
PHASE2_N_VALUES = [15, 50, 100, 200, 500]
PHASE2_D_VALUES = [10, 100, 1024]
PHASE2_RADII = [1.0, 10.0]
PHASE3_N_VALUES = [15, 20, 30, 50, 75, 100, 150, 200]
PHASE3_LAYERS = [4, 8, 12]
PHASE_REPEATS = 10

SCIENCE_KEYWORDS = {
    "equation", "theorem", "derivative", "integral", "probability", "matrix", "vector",
    "physics", "chemistry", "biology", "photosynthesis", "mitochondria", "gravity",
    "temperature", "celsius", "meters", "kilogram", "electron", "atom", "molecule",
    "dna", "genome", "light", "speed", "newton", "ohm", "pi", "algebra", "geometry",
    "prime", "integer", "logic", "proof", "inference",
}
CREATIVE_KEYWORDS = {
    "story", "once upon", "character", "hero", "villain", "forest", "castle", "dragon",
    "whispered", "twilight", "midnight", "sunset", "melody", "dream", "poem", "poetry",
    "narrative", "dialogue", "scene", "emotion", "heartfelt", "imagined", "fiction",
    "adventure", "mystery", "ancient", "library", "beach", "rain", "stars",
}

SPLIT_MARKERS = ("answer:", "rule:", "classification:", "conclusion:")


@dataclass(frozen=True)
class EvalCase:
    index: int
    prompt: str
    expected: str
    split_method: str


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def sqrt_eps_f32() -> float:
    return math.sqrt(math.ldexp(1.0, -23))


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    if mu == 0.0:
        return 0.0
    return float(pstdev(values) / abs(mu))


def _ranks(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_corr(values_a: list[float], values_b: list[float], eps: float) -> float:
    n = len(values_a)
    if n != len(values_b) or n < 2:
        return 1.0

    ra = _ranks(values_a)
    rb = _ranks(values_b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n

    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for i in range(n):
        da = ra[i] - mean_a
        db = rb[i] - mean_b
        num += da * db
        den_a += da * da
        den_b += db * db

    den = math.sqrt(den_a * den_b)
    if den <= eps:
        return 1.0
    return num / den


def upper_triangle_values(matrix: Any, backend: Any, eps: float) -> list[float]:
    n = int(matrix.shape[0])
    values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = float(backend.to_scalar(matrix[i][j]))
            if abs(v) > eps:
                values.append(v)
    return values


def distortion_values(chord: Any, geodesic: Any, backend: Any, eps: float) -> list[float]:
    n = int(chord.shape[0])
    values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(backend.to_scalar(chord[i][j]))
            if c <= eps:
                continue
            g = float(backend.to_scalar(geodesic[i][j]))
            values.append(abs(g - c) / c)
    return values


def flatten_upper_triangle(matrix: Any, backend: Any) -> list[float]:
    n = int(matrix.shape[0])
    values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(float(backend.to_scalar(matrix[i][j])))
    return values


def compute_knn_density_from_distance_matrix(
    dist_matrix: Any,
    k: int,
    backend: Any,
    eps: float,
) -> list[float]:
    sorted_dists = backend.sort(dist_matrix, axis=1)
    backend.eval(sorted_dists)
    k_dists = sorted_dists[:, 1 : k + 1]
    mean_k_dist = backend.mean(k_dists, axis=1)
    backend.eval(mean_k_dist)
    values = backend.tolist(mean_k_dist)
    if not isinstance(values, list):
        values = [float(values)]
    return [1.0 / max(float(v), eps) for v in values]


def sample_rows(matrix: Any, indices: list[int], backend: Any) -> Any:
    idx = backend.array(indices, dtype="int32")
    sampled = backend.take(matrix, idx, axis=0)
    backend.eval(sampled)
    return sampled


def measure_two_clouds(source_points: Any, target_points: Any, backend: Any) -> dict[str, Any]:
    source = backend.array(source_points)
    target = backend.array(target_points)
    backend.eval(source, target)
    n_source = int(source.shape[0])
    n_target = int(target.shape[0])

    rg = RiemannianGeometry(backend)

    eps = float(division_epsilon(backend, source))
    source_chord = rg._chord_distance_matrix(source, use_cache=False)
    target_chord = rg._chord_distance_matrix(target, use_cache=False)
    source_geo_result = rg.geodesic_distances(source, use_cache=False)
    target_geo_result = rg.geodesic_distances(target, use_cache=False)
    source_geo = source_geo_result.distances
    target_geo = target_geo_result.distances
    backend.eval(source_chord, target_chord, source_geo, target_geo)

    src_distortion = distortion_values(source_chord, source_geo, backend, eps)
    tgt_distortion = distortion_values(target_chord, target_geo, backend, eps)
    all_distortion = src_distortion + tgt_distortion

    k_source = derive_k_neighbors(source, backend)
    k_target = derive_k_neighbors(target, backend)
    k = max(k_source, k_target, 1)
    k = min(k, n_source - 1, n_target - 1)

    src_density_chord = compute_knn_density_from_distance_matrix(source_chord, k, backend, eps)
    src_density_geo = compute_knn_density_from_distance_matrix(source_geo, k, backend, eps)
    tgt_density_chord = compute_knn_density_from_distance_matrix(target_chord, k, backend, eps)
    tgt_density_geo = compute_knn_density_from_distance_matrix(target_geo, k, backend, eps)

    rho_source = spearman_rank_corr(src_density_chord, src_density_geo, eps)
    rho_target = spearman_rank_corr(tgt_density_chord, tgt_density_geo, eps)

    n_compare = min(len(src_density_chord), len(tgt_density_chord))
    sign_changes = 0
    if n_compare > 0:
        for i in range(n_compare):
            chord_total = src_density_chord[i] + tgt_density_chord[i]
            geo_total = src_density_geo[i] + tgt_density_geo[i]
            w_chord = src_density_chord[i] / max(chord_total, eps)
            w_geo = src_density_geo[i] / max(geo_total, eps)
            if (w_chord > 0.5) != (w_geo > 0.5):
                sign_changes += 1

    source_pairwise = upper_triangle_values(source_chord, backend, eps)
    target_pairwise = upper_triangle_values(target_chord, backend, eps)

    source_flat_chord = flatten_upper_triangle(source_chord, backend)
    source_flat_geo = flatten_upper_triangle(source_geo, backend)
    target_flat_chord = flatten_upper_triangle(target_chord, backend)
    target_flat_geo = flatten_upper_triangle(target_geo, backend)

    return {
        "n_source": n_source,
        "n_target": n_target,
        "k": int(k),
        "source_k_neighbors_geodesic": int(source_geo_result.k_neighbors),
        "target_k_neighbors_geodesic": int(target_geo_result.k_neighbors),
        "source_distortion_mean": float(mean(src_distortion)) if src_distortion else 0.0,
        "source_distortion_max": float(max(src_distortion)) if src_distortion else 0.0,
        "target_distortion_mean": float(mean(tgt_distortion)) if tgt_distortion else 0.0,
        "target_distortion_max": float(max(tgt_distortion)) if tgt_distortion else 0.0,
        "overall_mean_distortion": float(mean(all_distortion)) if all_distortion else 0.0,
        "overall_max_distortion": float(max(all_distortion)) if all_distortion else 0.0,
        "spearman_rho_source_density": float(rho_source),
        "spearman_rho_target_density": float(rho_target),
        "sign_changes": int(sign_changes),
        "sign_change_rate": float(sign_changes / max(n_compare, 1)),
        "pairwise_distance_cv_source": coefficient_of_variation(source_pairwise),
        "pairwise_distance_cv_target": coefficient_of_variation(target_pairwise),
        "spearman_rho_source_pairwise": float(
            spearman_rank_corr(source_flat_chord, source_flat_geo, eps)
        ),
        "spearman_rho_target_pairwise": float(
            spearman_rank_corr(target_flat_chord, target_flat_geo, eps)
        ),
    }


def aggregate_measurements(measurements: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(m[key]) for m in measurements]
        result[key] = stats(values)
    return result


def load_previous_activation_reference() -> dict[str, Any]:
    baseline_path = REPO_ROOT / "results" / "geodesic_vs_euclidean" / "LFM2-350M_density_comparison.json"
    if not baseline_path.exists():
        return {
            "source": "fallback_constant",
            "activation_distortion_reference": 0.46,
            "path": str(baseline_path),
        }

    with open(baseline_path) as f:
        payload = json.load(f)
    means = []
    for _, layer_data in payload.items():
        if isinstance(layer_data, dict) and "mean_distortion" in layer_data:
            means.append(float(layer_data["mean_distortion"]))
    if not means:
        return {
            "source": "fallback_constant",
            "activation_distortion_reference": 0.46,
            "path": str(baseline_path),
        }
    return {
        "source": "results_file",
        "activation_distortion_reference": float(max(means)),
        "path": str(baseline_path),
        "layer_mean_distortions": means,
    }


def run_phase_1_flat_null_hypothesis(
    backend: Any,
    phase_dir: Path,
    activation_reference: float,
    base_seed: int,
) -> dict[str, Any]:
    start = time.time()
    LOG.info("Phase 1: Flat-space null hypothesis")
    phase_results: dict[str, Any] = {
        "phase": 1,
        "name": "flat_space_null_hypothesis",
        "n_values": PHASE1_N_VALUES,
        "dimension": 1024,
        "repeats_per_n": PHASE_REPEATS,
        "by_n": {},
    }

    metrics_to_aggregate = [
        "overall_mean_distortion",
        "overall_max_distortion",
        "spearman_rho_source_density",
        "spearman_rho_target_density",
        "spearman_rho_source_pairwise",
        "spearman_rho_target_pairwise",
        "sign_changes",
        "sign_change_rate",
        "pairwise_distance_cv_source",
        "pairwise_distance_cv_target",
    ]

    for n_points in PHASE1_N_VALUES:
        per_seed: list[dict[str, Any]] = []
        for rep in range(PHASE_REPEATS):
            seed = base_seed + n_points * 100 + rep
            backend.random_seed(seed)
            source = backend.random_normal((n_points, 1024))
            target = backend.random_normal((n_points, 1024))
            backend.eval(source, target)
            measurement = measure_two_clouds(source, target, backend)
            measurement["seed"] = seed
            per_seed.append(measurement)

        aggregates = aggregate_measurements(per_seed, metrics_to_aggregate)
        cv_reliability_hits = 0
        for m in per_seed:
            if m["pairwise_distance_cv_source"] < 0.05:
                cv_reliability_hits += 1
            if m["pairwise_distance_cv_target"] < 0.05:
                cv_reliability_hits += 1

        phase_results["by_n"][str(n_points)] = {
            "per_seed": per_seed,
            "aggregates": aggregates,
            "cv_lt_0_05_fraction_across_clouds": cv_reliability_hits / (2.0 * PHASE_REPEATS),
        }

    n15_mean = phase_results["by_n"]["15"]["aggregates"]["overall_mean_distortion"]["mean"]
    pass_threshold = 0.10 * activation_reference
    fail_threshold = 0.50 * activation_reference

    status = "PASS" if n15_mean < pass_threshold else "FAIL"
    if status == "PASS":
        reason = (
            f"Flat-space N=15 mean distortion {n15_mean:.6f} < 10% of activation reference "
            f"({pass_threshold:.6f})."
        )
    elif n15_mean >= fail_threshold:
        reason = (
            f"Flat-space N=15 mean distortion {n15_mean:.6f} >= 50% of activation reference "
            f"({fail_threshold:.6f})."
        )
    else:
        reason = (
            f"Flat-space N=15 mean distortion {n15_mean:.6f} did not satisfy the pass criterion "
            f"(< {pass_threshold:.6f})."
        )

    phase_results["activation_distortion_reference"] = activation_reference
    phase_results["n15_mean_flat_distortion"] = n15_mean
    phase_results["pass_threshold"] = pass_threshold
    phase_results["fail_threshold"] = fail_threshold
    phase_results["status"] = status
    phase_results["reason"] = reason
    phase_results["elapsed_seconds"] = time.time() - start

    write_json(phase_dir / "phase_1_results.json", phase_results)
    return phase_results


def generate_hypersphere_points(n_points: int, dimension: int, radius: float, backend: Any) -> Any:
    points = backend.random_normal((n_points, dimension))
    norms = backend.sqrt(backend.sum(points * points, axis=1, keepdims=True))
    eps = float(division_epsilon(backend, points))
    norms_safe = backend.maximum(norms, backend.full(norms.shape, eps))
    unit = points / norms_safe
    sphere = unit * radius
    backend.eval(sphere)
    return sphere


def analytic_hypersphere_geodesic(points: Any, radius: float, backend: Any) -> Any:
    gram = backend.matmul(points, backend.transpose(points))
    denom = radius * radius
    cos_theta = gram / denom
    cos_theta = backend.clip(cos_theta, -1.0, 1.0)
    distances = backend.arccos(cos_theta) * radius
    backend.eval(distances)
    return distances


def run_phase_2_positive_control(
    backend: Any,
    phase_dir: Path,
    base_seed: int,
) -> dict[str, Any]:
    start = time.time()
    LOG.info("Phase 2: Known-curvature positive control")
    phase_results: dict[str, Any] = {
        "phase": 2,
        "name": "known_curvature_positive_control",
        "radii": PHASE2_RADII,
        "n_values": PHASE2_N_VALUES,
        "d_values": PHASE2_D_VALUES,
        "measurements": [],
    }

    rg = RiemannianGeometry(backend)
    for radius in PHASE2_RADII:
        for dimension in PHASE2_D_VALUES:
            for n_points in PHASE2_N_VALUES:
                seed = base_seed + int(radius * 1000) + dimension * 10 + n_points
                backend.random_seed(seed)
                points = generate_hypersphere_points(n_points, dimension, radius, backend)

                chord = rg._chord_distance_matrix(points, use_cache=False)
                estimated_geo_result = rg.geodesic_distances(points, use_cache=False)
                estimated_geo = estimated_geo_result.distances
                true_geo = analytic_hypersphere_geodesic(points, radius, backend)
                backend.eval(chord, estimated_geo, true_geo)

                eps = float(division_epsilon(backend, chord))
                true_distortion = distortion_values(chord, true_geo, backend, eps)
                measured_distortion = distortion_values(chord, estimated_geo, backend, eps)

                true_mean = float(mean(true_distortion)) if true_distortion else 0.0
                measured_mean = float(mean(measured_distortion)) if measured_distortion else 0.0
                true_max = float(max(true_distortion)) if true_distortion else 0.0
                measured_max = float(max(measured_distortion)) if measured_distortion else 0.0

                rel_err_mean = abs(measured_mean - true_mean) / max(abs(true_mean), eps)
                rel_err_max = abs(measured_max - true_max) / max(abs(true_max), eps)

                measurement = {
                    "radius": radius,
                    "curvature": 1.0 / (radius * radius),
                    "dimension": dimension,
                    "n_points": n_points,
                    "seed": seed,
                    "k_neighbors": int(estimated_geo_result.k_neighbors),
                    "true_distortion_mean": true_mean,
                    "measured_distortion_mean": measured_mean,
                    "true_distortion_max": true_max,
                    "measured_distortion_max": measured_max,
                    "relative_error_mean": float(rel_err_mean),
                    "relative_error_max": float(rel_err_max),
                    "spearman_rho_true_vs_measured_geodesic": float(
                        spearman_rank_corr(
                            flatten_upper_triangle(true_geo, backend),
                            flatten_upper_triangle(estimated_geo, backend),
                            eps,
                        )
                    ),
                }
                phase_results["measurements"].append(measurement)

    checks_500_10 = [
        m for m in phase_results["measurements"] if m["n_points"] == 500 and m["dimension"] == 10
    ]
    per_radius_pass: dict[str, bool] = {}
    for radius in PHASE2_RADII:
        matches = [m for m in checks_500_10 if m["radius"] == radius]
        per_radius_pass[str(radius)] = (
            bool(matches) and all(m["relative_error_mean"] <= 0.20 for m in matches)
        )

    minimum_working_pair: dict[str, dict[str, int] | None] = {}
    for radius in PHASE2_RADII:
        candidates = [
            m for m in phase_results["measurements"]
            if m["radius"] == radius and m["relative_error_mean"] <= 0.20
        ]
        candidates = sorted(candidates, key=lambda x: (x["n_points"], x["dimension"]))
        if not candidates:
            minimum_working_pair[str(radius)] = None
        else:
            minimum_working_pair[str(radius)] = {
                "n_points": int(candidates[0]["n_points"]),
                "dimension": int(candidates[0]["dimension"]),
            }

    status = "PASS" if all(per_radius_pass.values()) else "FAIL"
    if status == "PASS":
        reason = "Measured distortion matches theoretical prediction within 20% at N=500, d=10."
    else:
        reason = (
            "Method failed to recover theoretical distortion within 20% at N=500, d=10 "
            f"(per-radius: {per_radius_pass})."
        )

    phase_results["per_radius_pass_at_n500_d10"] = per_radius_pass
    phase_results["minimum_working_pair_per_radius"] = minimum_working_pair
    phase_results["status"] = status
    phase_results["reason"] = reason
    phase_results["elapsed_seconds"] = time.time() - start

    write_json(phase_dir / "phase_2_results.json", phase_results)
    return phase_results


def split_prompt_and_expected(text: str) -> tuple[str, str, str]:
    stripped = text.strip()
    lower = stripped.lower()

    marker_idx = -1
    marker = ""
    for candidate in SPLIT_MARKERS:
        idx = lower.rfind(candidate)
        if idx > marker_idx:
            marker_idx = idx
            marker = candidate
    if marker_idx >= 0:
        split_point = marker_idx + len(marker)
        prompt = stripped[:split_point].strip()
        expected = stripped[split_point:].strip()
        return prompt, expected, f"marker:{marker}"

    q_idx = stripped.rfind("?")
    if q_idx >= 0 and q_idx < len(stripped) - 1:
        prompt = stripped[: q_idx + 1].strip()
        expected = stripped[q_idx + 1 :].strip()
        return prompt, expected, "question_split"

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        prompt = " ".join(sentences[:-1]).strip()
        expected = sentences[-1].strip()
        return prompt, expected, "sentence_split"

    return stripped, "", "unsplittable"


def extract_prompt_only(text: str) -> str:
    prompt, _, _ = split_prompt_and_expected(text)
    return prompt.strip()


def keyword_score(text: str, keywords: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for kw in keywords if kw in lowered)


def build_prompt_pools(
    train_path: Path,
    per_domain_count: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    with open(train_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    candidates: list[tuple[str, int, int]] = []
    seen_prompts: set[str] = set()
    for row in rows:
        text = row.get("text", "")
        prompt = extract_prompt_only(text)
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        science_score = keyword_score(prompt, SCIENCE_KEYWORDS)
        creative_score = keyword_score(prompt, CREATIVE_KEYWORDS)
        candidates.append((prompt, science_score, creative_score))

    science_sorted = sorted(
        candidates,
        key=lambda item: (item[1] - item[2], item[1], -len(item[0])),
        reverse=True,
    )
    creative_sorted = sorted(
        candidates,
        key=lambda item: (item[2] - item[1], item[2], -len(item[0])),
        reverse=True,
    )

    science: list[str] = []
    creative: list[str] = []
    used_science: set[str] = set()
    used_creative: set[str] = set()

    for prompt, s_score, c_score in science_sorted:
        if len(science) >= per_domain_count:
            break
        if prompt in used_science:
            continue
        if s_score > 0 and s_score >= c_score:
            science.append(prompt)
            used_science.add(prompt)
    for prompt, _, _ in science_sorted:
        if len(science) >= per_domain_count:
            break
        if prompt in used_science:
            continue
        science.append(prompt)
        used_science.add(prompt)

    for prompt, s_score, c_score in creative_sorted:
        if len(creative) >= per_domain_count:
            break
        if prompt in used_creative:
            continue
        if c_score > 0 and c_score >= s_score:
            creative.append(prompt)
            used_creative.add(prompt)
    for prompt, _, _ in creative_sorted:
        if len(creative) >= per_domain_count:
            break
        if prompt in used_creative:
            continue
        creative.append(prompt)
        used_creative.add(prompt)

    diagnostics = {
        "candidate_count": len(candidates),
        "science_selected": len(science),
        "creative_selected": len(creative),
        "science_keyword_hits": sum(
            1 for p in science if keyword_score(p, SCIENCE_KEYWORDS) > 0
        ),
        "creative_keyword_hits": sum(
            1 for p in creative if keyword_score(p, CREATIVE_KEYWORDS) > 0
        ),
    }
    return science, creative, diagnostics


def collect_layer_activations(
    model_path: str,
    prompts: list[str],
    layers_to_collect: list[int],
    backend: Any,
) -> dict[int, Any]:
    model, tokenizer = ModelLoader(backend).load_model(model_path)
    backbone = resolve_model_backbone(model)
    if not backbone:
        raise RuntimeError(f"Could not resolve model backbone for {model_path}")
    embed_tokens, layers, final_norm = backbone

    valid_layers = [layer for layer in layers_to_collect if 0 <= layer < len(layers)]
    if len(valid_layers) != len(layers_to_collect):
        raise RuntimeError(
            f"Requested layers {layers_to_collect} but model has {len(layers)} layers."
        )

    per_layer_vectors: dict[int, list[Any]] = {layer: [] for layer in valid_layers}
    for idx, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        input_ids = backend.array([token_ids])
        for layer_idx in valid_layers:
            hidden = forward_through_backbone(
                input_ids,
                embed_tokens,
                layers,
                final_norm,
                target_layer=layer_idx,
                backend=backend,
            )
            pooled = backend.mean(hidden[0], axis=0)
            backend.eval(pooled)
            per_layer_vectors[layer_idx].append(pooled)
        if (idx + 1) % 20 == 0:
            LOG.info(
                "Activation extraction %d/%d prompts (%s)",
                idx + 1,
                len(prompts),
                model_path,
            )

    per_layer_matrix: dict[int, Any] = {}
    for layer_idx, vectors in per_layer_vectors.items():
        matrix = backend.stack(vectors, axis=0)
        backend.eval(matrix)
        per_layer_matrix[layer_idx] = matrix
    return per_layer_matrix


def detect_convergence(by_n: dict[str, dict[str, Any]], eps: float) -> int | None:
    sequence = [(int(k), v) for k, v in by_n.items()]
    sequence.sort(key=lambda item: item[0])
    for i in range(1, len(sequence)):
        n_curr, current = sequence[i]
        _, prev = sequence[i - 1]
        prev_dist = prev["aggregates"]["overall_mean_distortion"]["mean"]
        curr_dist = current["aggregates"]["overall_mean_distortion"]["mean"]
        prev_rho = prev["aggregates"]["spearman_rho_source_density"]["mean"]
        curr_rho = current["aggregates"]["spearman_rho_source_density"]["mean"]
        prev_sign = prev["aggregates"]["sign_change_rate"]["mean"]
        curr_sign = current["aggregates"]["sign_change_rate"]["mean"]

        dist_rel = abs(curr_dist - prev_dist) / max(abs(prev_dist), eps)
        rho_rel = abs(curr_rho - prev_rho) / max(abs(prev_rho), eps)
        sign_rel = abs(curr_sign - prev_sign) / max(abs(prev_sign), eps)

        if dist_rel < 0.05 and rho_rel < 0.05 and sign_rel < 0.05:
            return n_curr
    return None


def run_phase_3_sample_size_sensitivity(
    backend: Any,
    phase_dir: Path,
    model_path: str,
    benchmark_train_path: Path,
    base_seed: int,
    per_domain_pool_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.time()
    LOG.info("Phase 3: Sample-size sensitivity on real activations")
    science_prompts, creative_prompts, pool_diag = build_prompt_pools(
        benchmark_train_path,
        per_domain_pool_size,
    )

    max_n = max(PHASE3_N_VALUES)
    if len(science_prompts) < max_n or len(creative_prompts) < max_n:
        raise RuntimeError(
            f"Need at least {max_n} prompts per domain; got "
            f"science={len(science_prompts)} creative={len(creative_prompts)}"
        )

    science_prompts = science_prompts[:per_domain_pool_size]
    creative_prompts = creative_prompts[:per_domain_pool_size]

    source_activations = collect_layer_activations(
        model_path=model_path,
        prompts=science_prompts,
        layers_to_collect=PHASE3_LAYERS,
        backend=backend,
    )
    target_activations = collect_layer_activations(
        model_path=model_path,
        prompts=creative_prompts,
        layers_to_collect=PHASE3_LAYERS,
        backend=backend,
    )

    phase_results: dict[str, Any] = {
        "phase": 3,
        "name": "sample_size_sensitivity_real_activations",
        "model_path": model_path,
        "layers": PHASE3_LAYERS,
        "n_values": PHASE3_N_VALUES,
        "repeats_per_n": PHASE_REPEATS,
        "prompt_pool": pool_diag,
        "science_prompt_count": len(science_prompts),
        "creative_prompt_count": len(creative_prompts),
        "layers_results": {},
    }

    aggregate_keys = [
        "overall_mean_distortion",
        "overall_max_distortion",
        "spearman_rho_source_density",
        "spearman_rho_target_density",
        "sign_changes",
        "sign_change_rate",
    ]
    eps = float(division_epsilon(backend, source_activations[PHASE3_LAYERS[0]]))
    layer_convergence: dict[int, int | None] = {}

    for layer_idx in PHASE3_LAYERS:
        by_n: dict[str, Any] = {}
        src_matrix = source_activations[layer_idx]
        tgt_matrix = target_activations[layer_idx]
        src_count = int(src_matrix.shape[0])
        tgt_count = int(tgt_matrix.shape[0])
        for n_points in PHASE3_N_VALUES:
            per_repeat: list[dict[str, Any]] = []
            for rep in range(PHASE_REPEATS):
                rng = random.Random(base_seed + layer_idx * 10000 + n_points * 100 + rep)
                src_idx = rng.sample(range(src_count), n_points)
                tgt_idx = rng.sample(range(tgt_count), n_points)
                src_sample = sample_rows(src_matrix, src_idx, backend)
                tgt_sample = sample_rows(tgt_matrix, tgt_idx, backend)
                measurement = measure_two_clouds(src_sample, tgt_sample, backend)
                measurement["repeat"] = rep
                per_repeat.append(measurement)

            by_n[str(n_points)] = {
                "per_repeat": per_repeat,
                "aggregates": aggregate_measurements(per_repeat, aggregate_keys),
            }

        n_converged = detect_convergence(by_n, eps)
        layer_convergence[layer_idx] = n_converged
        phase_results["layers_results"][str(layer_idx)] = {
            "by_n": by_n,
            "n_converged": n_converged,
        }

    converged_values = [v for v in layer_convergence.values() if v is not None]
    all_converged = len(converged_values) == len(PHASE3_LAYERS)
    global_n_converged = max(converged_values) if all_converged else None
    status = "PASS" if all_converged and global_n_converged is not None and global_n_converged <= 200 else "FAIL"

    if status == "PASS":
        reason = f"Distortion/rank/sign metrics converged by N_c={global_n_converged}."
    else:
        reason = "No stable convergence across all layers by N<=200."

    phase_results["layer_convergence"] = {str(k): v for k, v in layer_convergence.items()}
    phase_results["global_n_converged"] = global_n_converged
    phase_results["status"] = status
    phase_results["reason"] = reason
    phase_results["elapsed_seconds"] = time.time() - start

    write_json(phase_dir / "phase_3_results.json", phase_results)

    context = {
        "source_activations": source_activations,
        "target_activations": target_activations,
        "global_n_converged": global_n_converged,
        "science_prompt_count": len(science_prompts),
        "creative_prompt_count": len(creative_prompts),
    }
    return phase_results, context


def permute_coordinates_per_point(points: Any, backend: Any, rng: random.Random) -> Any:
    n_points = int(points.shape[0])
    dimension = int(points.shape[1])
    rows = []
    for i in range(n_points):
        perm = list(range(dimension))
        rng.shuffle(perm)
        perm_arr = backend.array(perm, dtype="int32")
        row = backend.take(points[i], perm_arr, axis=0)
        rows.append(row)
    out = backend.stack(rows, axis=0)
    backend.eval(out)
    return out


def run_phase_4_permutation_control(
    backend: Any,
    phase_dir: Path,
    phase3_context: dict[str, Any],
    selected_n: int,
    base_seed: int,
) -> dict[str, Any]:
    start = time.time()
    LOG.info("Phase 4: Permutation control")
    source_activations = phase3_context["source_activations"]
    target_activations = phase3_context["target_activations"]

    phase_results: dict[str, Any] = {
        "phase": 4,
        "name": "permutation_control",
        "selected_n": selected_n,
        "layers": PHASE3_LAYERS,
        "per_layer": {},
    }

    layer_pass: dict[int, bool] = {}
    for layer_idx in PHASE3_LAYERS:
        src_matrix = source_activations[layer_idx]
        tgt_matrix = target_activations[layer_idx]
        src_count = int(src_matrix.shape[0])
        tgt_count = int(tgt_matrix.shape[0])
        n_points = min(selected_n, src_count, tgt_count)
        rng_sample = random.Random(base_seed + layer_idx * 997)
        src_idx = rng_sample.sample(range(src_count), n_points)
        tgt_idx = rng_sample.sample(range(tgt_count), n_points)

        src_sample = sample_rows(src_matrix, src_idx, backend)
        tgt_sample = sample_rows(tgt_matrix, tgt_idx, backend)
        original = measure_two_clouds(src_sample, tgt_sample, backend)
        original_distortion = original["overall_mean_distortion"]

        permuted_measurements: list[dict[str, Any]] = []
        permuted_distortions: list[float] = []
        for rep in range(PHASE_REPEATS):
            rng_perm = random.Random(base_seed + layer_idx * 10000 + rep)
            src_perm = permute_coordinates_per_point(src_sample, backend, rng_perm)
            tgt_perm = permute_coordinates_per_point(tgt_sample, backend, rng_perm)
            measured = measure_two_clouds(src_perm, tgt_perm, backend)
            measured["repeat"] = rep
            permuted_measurements.append(measured)
            permuted_distortions.append(float(measured["overall_mean_distortion"]))

        perm_mean = mean(permuted_distortions) if permuted_distortions else 0.0
        ratio = perm_mean / max(original_distortion, float(division_epsilon(backend, src_sample)))
        passed = ratio < 0.5
        layer_pass[layer_idx] = passed
        phase_results["per_layer"][str(layer_idx)] = {
            "original": original,
            "permuted": permuted_measurements,
            "permuted_mean_distortion": float(perm_mean),
            "permuted_to_original_ratio": float(ratio),
            "passed": passed,
        }

    status = "PASS" if all(layer_pass.values()) else "FAIL"
    if status == "PASS":
        reason = "Permuted distortion is <50% of original distortion."
    else:
        reason = "Permuted distortion is comparable to original distortion."

    phase_results["layer_pass"] = {str(k): v for k, v in layer_pass.items()}
    phase_results["status"] = status
    phase_results["reason"] = reason
    phase_results["elapsed_seconds"] = time.time() - start

    write_json(phase_dir / "phase_4_results.json", phase_results)
    return phase_results


def run_phase_5_direct_curvature(
    backend: Any,
    phase_dir: Path,
    phase3_context: dict[str, Any],
    phase3_results: dict[str, Any],
    selected_n: int,
    base_seed: int,
) -> dict[str, Any]:
    start = time.time()
    LOG.info("Phase 5: Direct curvature measurement")
    source_activations = phase3_context["source_activations"]
    target_activations = phase3_context["target_activations"]

    phase_results: dict[str, Any] = {
        "phase": 5,
        "name": "direct_curvature_measurement",
        "selected_n": selected_n,
        "layers": PHASE3_LAYERS,
        "per_layer": {},
    }

    id_estimator = IntrinsicDimension(backend)
    ollivier_estimator = OllivierRicciCurvature(backend)
    sectional_estimator = SectionalCurvatureEstimator()
    rg = RiemannianGeometry(backend)

    layer_pass: dict[int, bool] = {}
    eps_global = sqrt_eps_f32()
    for layer_idx in PHASE3_LAYERS:
        src_matrix = source_activations[layer_idx]
        tgt_matrix = target_activations[layer_idx]
        src_count = int(src_matrix.shape[0])
        tgt_count = int(tgt_matrix.shape[0])
        n_points = min(selected_n, src_count, tgt_count)

        rng_sample = random.Random(base_seed + layer_idx * 123)
        src_idx = rng_sample.sample(range(src_count), n_points)
        tgt_idx = rng_sample.sample(range(tgt_count), n_points)
        src_sample = sample_rows(src_matrix, src_idx, backend)
        tgt_sample = sample_rows(tgt_matrix, tgt_idx, backend)
        points = backend.concatenate([src_sample, tgt_sample], axis=0)
        backend.eval(points)

        id_result = id_estimator.compute(points)
        intrinsic_dim = float(id_result.intrinsic_dimension)

        chord = rg._chord_distance_matrix(points, use_cache=False)
        k_neighbors = derive_k_neighbors(points, backend)
        k_neighbors = max(1, min(k_neighbors, int(points.shape[0]) - 1))
        sorted_chord = backend.sort(chord, axis=1)
        k_dists = sorted_chord[:, 1 : k_neighbors + 1]
        mean_k = backend.mean(k_dists, axis=1)
        mean_radius_arr = backend.mean(mean_k)
        backend.eval(mean_radius_arr)
        r_k = float(backend.to_scalar(mean_radius_arr))

        ollivier_result = ollivier_estimator.compute(points)
        ollivier_mean = float(ollivier_result.mean_edge_curvature)

        sectional_profile = sectional_estimator.estimate_manifold_profile(points)
        sectional_mean = float(sectional_profile.global_mean)

        observed = phase3_results["layers_results"][str(layer_idx)]["by_n"][str(selected_n)][
            "aggregates"
        ]["overall_mean_distortion"]["mean"]

        kappa_prediction = 0.5 * (abs(ollivier_mean) + abs(sectional_mean))
        bernstein_predicted_distortion = (r_k * r_k * kappa_prediction) / 24.0
        rel_error = abs(bernstein_predicted_distortion - observed) / max(abs(observed), eps_global)

        nonzero_ollivier = abs(ollivier_mean) > eps_global
        nonzero_sectional = abs(sectional_mean) > eps_global
        sign_agreement = (
            nonzero_ollivier
            and nonzero_sectional
            and (ollivier_mean * sectional_mean > 0.0)
        )
        bernstein_match = rel_error <= 0.20
        passed = nonzero_ollivier and nonzero_sectional and sign_agreement and bernstein_match
        layer_pass[layer_idx] = passed

        phase_results["per_layer"][str(layer_idx)] = {
            "intrinsic_dimension": intrinsic_dim,
            "k_neighbors": int(k_neighbors),
            "r_k_mean": float(r_k),
            "ollivier_mean_edge_curvature": ollivier_mean,
            "sectional_global_mean": sectional_mean,
            "observed_distortion": float(observed),
            "bernstein_predicted_distortion": float(bernstein_predicted_distortion),
            "bernstein_relative_error": float(rel_error),
            "nonzero_ollivier": nonzero_ollivier,
            "nonzero_sectional": nonzero_sectional,
            "sign_agreement": sign_agreement,
            "bernstein_match": bernstein_match,
            "passed": passed,
        }

    status = "PASS" if all(layer_pass.values()) else "FAIL"
    if status == "PASS":
        reason = "Curvature is non-zero across methods and Bernstein prediction matches measured distortion."
    else:
        reason = "Curvature methods disagree or Bernstein prediction does not match measured distortion."

    phase_results["layer_pass"] = {str(k): v for k, v in layer_pass.items()}
    phase_results["status"] = status
    phase_results["reason"] = reason
    phase_results["elapsed_seconds"] = time.time() - start

    write_json(phase_dir / "phase_5_results.json", phase_results)
    return phase_results


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9 ]", "", lowered)
    return lowered.strip()


def response_matches_expected(response: str, expected: str) -> bool:
    norm_resp = normalize_text(response)
    norm_exp = normalize_text(expected)
    if not norm_exp:
        return False
    exp_tokens = norm_exp.split()
    resp_tokens = norm_resp.split()
    if len(exp_tokens) == 1:
        return exp_tokens[0] in resp_tokens
    return norm_exp in norm_resp


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            prompt, expected, split_method = split_prompt_and_expected(text)
            if not prompt or not expected:
                continue
            cases.append(EvalCase(index=idx, prompt=prompt, expected=expected, split_method=split_method))
    return cases


def evaluate_model_accuracy(
    backend: Any,
    model_path: str,
    cases: list[EvalCase],
    max_tokens: int,
) -> dict[str, Any]:
    model, tokenizer = ModelLoader().load_model(model_path)
    correct = 0
    by_split_method: dict[str, dict[str, int]] = {}
    sample_errors: list[dict[str, Any]] = []

    for i, case in enumerate(cases):
        response = backend.generate(model, tokenizer, case.prompt, max_tokens=max_tokens)
        ok = response_matches_expected(response, case.expected)
        if ok:
            correct += 1
        else:
            if len(sample_errors) < 12:
                sample_errors.append(
                    {
                        "case_index": case.index,
                        "split_method": case.split_method,
                        "expected": case.expected[:160],
                        "response": response[:160],
                    }
                )
        bucket = by_split_method.setdefault(case.split_method, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if ok:
            bucket["correct"] += 1
        if (i + 1) % 25 == 0:
            LOG.info(
                "Accuracy eval %d/%d on %s",
                i + 1,
                len(cases),
                model_path,
            )

    total = len(cases)
    accuracy = correct / total if total > 0 else 0.0
    split_acc = {
        method: {
            "correct": stats_dict["correct"],
            "total": stats_dict["total"],
            "accuracy": (stats_dict["correct"] / stats_dict["total"]) if stats_dict["total"] > 0 else 0.0,
        }
        for method, stats_dict in by_split_method.items()
    }
    return {
        "model_path": model_path,
        "correct": correct,
        "total": total,
        "accuracy": float(accuracy),
        "by_split_method": split_acc,
        "sample_errors": sample_errors,
    }


def run_merge_command(
    mode: str,
    source_model_path: str,
    target_model_path: str,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "poetry",
        "run",
        "mc",
        "merge",
        "run",
        "-s",
        source_model_path,
        "-t",
        target_model_path,
        "-o",
        str(output_path),
    ]
    env = os.environ.copy()
    env["MC_DENSITY_DISTANCE_MODE"] = mode
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.time() - started
    return {
        "mode": mode,
        "command": cmd,
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "output_path": str(output_path),
    }


def run_phase_6_downstream_validation(
    backend: Any,
    phase_dir: Path,
    source_model_path: str,
    target_model_path: str,
    benchmark_val_path: Path,
    max_tokens: int,
) -> dict[str, Any]:
    start = time.time()
    LOG.info("Phase 6: Downstream validation")
    phase_results: dict[str, Any] = {
        "phase": 6,
        "name": "downstream_validation",
        "source_model_path": source_model_path,
        "target_model_path": target_model_path,
        "benchmark_val_path": str(benchmark_val_path),
    }

    merge_geo_out = phase_dir / "merged_geodesic"
    merge_euc_out = phase_dir / "merged_euclidean"

    merge_geo = run_merge_command("geodesic", source_model_path, target_model_path, merge_geo_out)
    merge_euc = run_merge_command("euclidean", source_model_path, target_model_path, merge_euc_out)
    phase_results["merge_commands"] = {
        "geodesic": merge_geo,
        "euclidean": merge_euc,
    }

    if merge_geo["returncode"] != 0 or merge_euc["returncode"] != 0:
        phase_results["status"] = "FAIL"
        phase_results["reason"] = "Merge command failed for one or both modes."
        phase_results["elapsed_seconds"] = time.time() - start
        write_json(phase_dir / "phase_6_results.json", phase_results)
        return phase_results

    cases = load_eval_cases(benchmark_val_path)
    if not cases:
        phase_results["status"] = "FAIL"
        phase_results["reason"] = "No evaluable benchmark cases were extracted."
        phase_results["elapsed_seconds"] = time.time() - start
        write_json(phase_dir / "phase_6_results.json", phase_results)
        return phase_results

    eval_geo = evaluate_model_accuracy(
        backend=backend,
        model_path=str(merge_geo_out),
        cases=cases,
        max_tokens=max_tokens,
    )
    eval_euc = evaluate_model_accuracy(
        backend=backend,
        model_path=str(merge_euc_out),
        cases=cases,
        max_tokens=max_tokens,
    )
    phase_results["accuracy_eval"] = {
        "geodesic": eval_geo,
        "euclidean": eval_euc,
        "case_count": len(cases),
    }

    prompts = [case.prompt for case in cases]
    acts_target = collect_layer_activations(target_model_path, prompts, PHASE3_LAYERS, backend)
    acts_geo = collect_layer_activations(str(merge_geo_out), prompts, PHASE3_LAYERS, backend)
    acts_euc = collect_layer_activations(str(merge_euc_out), prompts, PHASE3_LAYERS, backend)

    cka_by_layer: dict[str, Any] = {}
    for layer_idx in PHASE3_LAYERS:
        cka_geo = compute_cka(acts_geo[layer_idx], acts_target[layer_idx], backend=backend).cka
        cka_euc = compute_cka(acts_euc[layer_idx], acts_target[layer_idx], backend=backend).cka
        cka_by_layer[str(layer_idx)] = {
            "geodesic_vs_target": float(cka_geo),
            "euclidean_vs_target": float(cka_euc),
            "absolute_diff": float(abs(cka_geo - cka_euc)),
        }
    phase_results["cka_eval"] = cka_by_layer

    threshold = sqrt_eps_f32()
    metric_differences = {
        "accuracy_abs_diff": abs(
            eval_geo["accuracy"] - eval_euc["accuracy"]
        ),
    }
    for layer_idx in PHASE3_LAYERS:
        metric_differences[f"cka_layer_{layer_idx}_abs_diff"] = cka_by_layer[str(layer_idx)][
            "absolute_diff"
        ]

    merge_diff = any(diff > threshold for diff in metric_differences.values())
    phase_results["sqrt_eps_f32"] = threshold
    phase_results["metric_differences"] = metric_differences
    phase_results["merge_diff_detected"] = merge_diff
    phase_results["status"] = "PASS" if merge_diff else "FAIL"
    if merge_diff:
        phase_results["reason"] = (
            "At least one quality metric differs by more than sqrt(eps_f32)."
        )
    else:
        phase_results["reason"] = (
            "Quality metrics are identical within sqrt(eps_f32)."
        )
    phase_results["elapsed_seconds"] = time.time() - start

    write_json(phase_dir / "phase_6_results.json", phase_results)
    return phase_results


def skipped_phase(phase_num: int, name: str, reason: str) -> dict[str, Any]:
    return {
        "phase": phase_num,
        "name": name,
        "status": "SKIPPED",
        "reason": reason,
    }


def summarize(phase_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    p1 = phase_results.get(1, {})
    p2 = phase_results.get(2, {})
    p3 = phase_results.get(3, {})
    p4 = phase_results.get(4, {})
    p5 = phase_results.get(5, {})
    p6 = phase_results.get(6, {})

    if p1.get("status") != "PASS":
        decision = {
            "matrix_row": "FAIL-P1",
            "production_distance_mode": "euclidean",
            "adaptive_min_samples": None,
            "reason": "Phase 1 failed: flat-space control invalidates prior geodesic claim.",
        }
    elif p2.get("status") != "PASS":
        decision = {
            "matrix_row": "FAIL-P2",
            "production_distance_mode": "euclidean",
            "adaptive_min_samples": None,
            "reason": "Phase 2 failed: method cannot recover known curvature.",
        }
    elif p3.get("status") != "PASS":
        decision = {
            "matrix_row": "FAIL-P3",
            "production_distance_mode": "euclidean",
            "adaptive_min_samples": None,
            "reason": "Phase 3 failed: no stable convergence at practical sample sizes.",
        }
    elif p4.get("status") != "PASS":
        decision = {
            "matrix_row": "FAIL-P4",
            "production_distance_mode": "euclidean",
            "adaptive_min_samples": None,
            "reason": "Phase 4 failed: distortion behaves like distributional artifact.",
        }
    elif p5.get("status") != "PASS":
        decision = {
            "matrix_row": "FAIL-P5",
            "production_distance_mode": "euclidean",
            "adaptive_min_samples": None,
            "reason": "Phase 5 failed: curvature evidence is not internally consistent.",
        }
    elif p6.get("status") != "PASS":
        decision = {
            "matrix_row": "FAIL-P6",
            "production_distance_mode": "euclidean",
            "adaptive_min_samples": None,
            "reason": "Phase 6 failed: merge quality does not change under metric choice.",
        }
    else:
        decision = {
            "matrix_row": "PASS-ALL",
            "production_distance_mode": "adaptive",
            "adaptive_min_samples": p3.get("global_n_converged"),
            "reason": "All gates passed and merge quality differs between metrics.",
        }

    summary = {
        "phase_status": {str(k): v.get("status") for k, v in phase_results.items()},
        "decision": decision,
        "phase_results": phase_results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled geodesic-vs-euclidean experiment with phase gating.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "geodesic_vs_euclidean",
    )
    parser.add_argument(
        "--phase3-model",
        type=str,
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    )
    parser.add_argument(
        "--phase6-source-model",
        type=str,
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    )
    parser.add_argument(
        "--phase6-target-model",
        type=str,
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    )
    parser.add_argument(
        "--benchmark-train",
        type=Path,
        default=REPO_ROOT / "data" / "training" / "benchmark_train.jsonl",
    )
    parser.add_argument(
        "--benchmark-val",
        type=Path,
        default=REPO_ROOT / "data" / "training" / "benchmark_val.jsonl",
    )
    parser.add_argument(
        "--phase3-pool-per-domain",
        type=int,
        default=220,
        help="Prompt pool size per domain for phase 3 (must be >= max N=200).",
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=96,
        help="Max generation tokens for phase 6 inference accuracy.",
    )
    parser.add_argument(
        "--run-until-phase",
        type=int,
        default=6,
        choices=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    backend = initialize_default_backend()
    results_root: Path = args.results_root
    results_root.mkdir(parents=True, exist_ok=True)

    reference_info = load_previous_activation_reference()
    activation_reference = float(reference_info["activation_distortion_reference"])
    LOG.info(
        "Activation distortion reference: %.6f (source=%s)",
        activation_reference,
        reference_info.get("source"),
    )

    phases: dict[int, dict[str, Any]] = {}
    phase3_context: dict[str, Any] | None = None

    # Phase 1
    phase1_dir = results_root / "phase_1"
    phases[1] = run_phase_1_flat_null_hypothesis(
        backend=backend,
        phase_dir=phase1_dir,
        activation_reference=activation_reference,
        base_seed=args.seed,
    )

    # Phase 2
    if args.run_until_phase >= 2 and phases[1]["status"] == "PASS":
        phases[2] = run_phase_2_positive_control(
            backend=backend,
            phase_dir=results_root / "phase_2",
            base_seed=args.seed + 1000,
        )
    else:
        phases[2] = skipped_phase(
            2,
            "known_curvature_positive_control",
            "Phase 1 failed or run-until-phase < 2.",
        )

    # Phase 3
    if args.run_until_phase >= 3 and phases[2]["status"] == "PASS":
        phase3_result, phase3_context = run_phase_3_sample_size_sensitivity(
            backend=backend,
            phase_dir=results_root / "phase_3",
            model_path=args.phase3_model,
            benchmark_train_path=args.benchmark_train,
            base_seed=args.seed + 2000,
            per_domain_pool_size=max(args.phase3_pool_per_domain, max(PHASE3_N_VALUES)),
        )
        phases[3] = phase3_result
    else:
        phases[3] = skipped_phase(
            3,
            "sample_size_sensitivity_real_activations",
            "Phase 2 failed or run-until-phase < 3.",
        )

    # Phase 4
    if args.run_until_phase >= 4 and phases[3]["status"] == "PASS" and phase3_context is not None:
        selected_n = int(phases[3]["global_n_converged"])
        phases[4] = run_phase_4_permutation_control(
            backend=backend,
            phase_dir=results_root / "phase_4",
            phase3_context=phase3_context,
            selected_n=selected_n,
            base_seed=args.seed + 3000,
        )
    else:
        phases[4] = skipped_phase(
            4,
            "permutation_control",
            "Phase 3 failed or run-until-phase < 4.",
        )

    # Phase 5
    if args.run_until_phase >= 5 and phases[4]["status"] == "PASS" and phase3_context is not None:
        selected_n = int(phases[3]["global_n_converged"])
        phases[5] = run_phase_5_direct_curvature(
            backend=backend,
            phase_dir=results_root / "phase_5",
            phase3_context=phase3_context,
            phase3_results=phases[3],
            selected_n=selected_n,
            base_seed=args.seed + 4000,
        )
    else:
        phases[5] = skipped_phase(
            5,
            "direct_curvature_measurement",
            "Phase 4 failed or run-until-phase < 5.",
        )

    # Phase 6
    if args.run_until_phase >= 6 and phases[5]["status"] == "PASS":
        phases[6] = run_phase_6_downstream_validation(
            backend=backend,
            phase_dir=results_root / "phase_6",
            source_model_path=args.phase6_source_model,
            target_model_path=args.phase6_target_model,
            benchmark_val_path=args.benchmark_val,
            max_tokens=args.max_eval_tokens,
        )
    else:
        phases[6] = skipped_phase(
            6,
            "downstream_validation",
            "Phase 5 failed or run-until-phase < 6.",
        )

    summary = summarize(phases)
    summary["activation_reference"] = reference_info
    summary["generated_at_unix_seconds"] = time.time()
    write_json(results_root / "summary.json", summary)

    decision_payload = {
        "production_distance_mode": summary["decision"]["production_distance_mode"],
        "adaptive_min_samples": summary["decision"]["adaptive_min_samples"],
        "decision_reason": summary["decision"]["reason"],
        "decision_matrix_row": summary["decision"]["matrix_row"],
        "phase_status": summary["phase_status"],
        "summary_path": str(results_root / "summary.json"),
        "generated_at_unix_seconds": summary["generated_at_unix_seconds"],
    }
    write_json(results_root / "decision.json", decision_payload)

    print("=" * 80)
    print("GEODESIC VS EUCLIDEAN DECISION")
    print("=" * 80)
    print(f"Decision row: {summary['decision']['matrix_row']}")
    print(f"Production mode: {summary['decision']['production_distance_mode']}")
    print(f"Adaptive min samples: {summary['decision']['adaptive_min_samples']}")
    print(f"Reason: {summary['decision']['reason']}")
    print(f"Summary: {results_root / 'summary.json'}")
    print(f"Decision: {results_root / 'decision.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
