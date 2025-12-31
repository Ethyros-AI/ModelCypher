# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _sample_embedding_similarities(
    source_embed: "object",
    target_embed: "object",
    backend: "object",
    sample_size: int = 500,
) -> list[float]:
    """Sample cosine similarities between source and target embeddings.

    Returns a list of similarity values that can be used to derive
    thresholds via spectral gap detection.
    """

    n_source = source_embed.shape[0]
    n_target = target_embed.shape[0]

    if n_source == 0 or n_target == 0:
        return []

    # Sample indices uniformly
    sample_size = min(sample_size, n_source, n_target)
    step_source = max(1, n_source // sample_size)
    step_target = max(1, n_target // sample_size)

    source_indices = list(range(0, n_source, step_source))[:sample_size]
    target_indices = list(range(0, n_target, step_target))[:sample_size]

    if not source_indices or not target_indices:
        return []

    # Get sample embeddings
    source_sample = backend.take(source_embed, backend.array(source_indices), axis=0)
    target_sample = backend.take(target_embed, backend.array(target_indices), axis=0)

    # Normalize for cosine similarity
    eps = machine_epsilon(backend, source_sample)
    source_norms = backend.norm(source_sample, axis=1, keepdims=True)
    target_norms = backend.norm(target_sample, axis=1, keepdims=True)

    source_normed = source_sample / (source_norms + eps)
    target_normed = target_sample / (target_norms + eps)

    # Compute pairwise cosine similarities (sample x sample matrix)
    sim_matrix = backend.matmul(source_normed, backend.transpose(target_normed))
    backend.eval(sim_matrix)

    # Flatten and convert to list
    sim_flat = backend.reshape(sim_matrix, (-1,))
    return list(backend.to_numpy(sim_flat).tolist())


def _derive_thresholds_from_similarities(
    similarities: list[float],
) -> dict[str, float]:
    """Derive similarity and confidence thresholds from spectral gap.

    Finds natural boundaries in the similarity distribution using
    the largest gap between consecutive sorted values.

    Returns dict with 'similarity_threshold' and 'confidence_threshold'.
    """
    import math as _math

    if len(similarities) < 2:
        return {"similarity_threshold": 0.0, "confidence_threshold": 0.0}

    sorted_sims = sorted(similarities)

    # Compute gaps between consecutive values
    gaps = [sorted_sims[i + 1] - sorted_sims[i] for i in range(len(sorted_sims) - 1)]

    if not gaps:
        return {"similarity_threshold": 0.0, "confidence_threshold": 0.0}

    # Find the gap that separates "low" from "high" similarity
    # Use mean + 2*stddev as significance threshold
    mean_gap = sum(gaps) / len(gaps)
    if len(gaps) > 1:
        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        stddev = _math.sqrt(variance)
        significance_threshold = mean_gap + 2.0 * stddev
    else:
        significance_threshold = 0.0

    # Find largest significant gap
    max_gap = max(gaps)
    max_gap_idx = gaps.index(max_gap)

    if max_gap <= significance_threshold:
        # No significant gap - use median as threshold
        mid_idx = len(sorted_sims) // 2
        threshold = sorted_sims[mid_idx]
    else:
        # Threshold at geometric mean of values around the gap
        lower_val = sorted_sims[max_gap_idx]
        upper_val = sorted_sims[max_gap_idx + 1]
        if lower_val <= 0:
            threshold = upper_val / 2.0
        else:
            threshold = _math.sqrt(lower_val * upper_val)

    # similarity_threshold is for "high quality" alignment
    # confidence_threshold is for "acceptable" alignment
    # Use spectral gap for similarity, and half that for confidence
    return {
        "similarity_threshold": threshold,
        "confidence_threshold": threshold / 2.0 if threshold > 0 else 0.0,
    }
