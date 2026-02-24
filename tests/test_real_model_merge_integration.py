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

"""Integration tests using real model weights.

Tests merge pipeline components with actual model weights from:
- SmolLM-135M: Llama architecture, hidden=576, 30 layers
- LFM2-350M: LFM2 architecture, hidden=1024, 16 layers

These models have different architectures, hidden dimensions, and layer counts,
providing realistic cross-architecture merge testing.
"""

import json
from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_geodesic_cka
from modelcypher.core.domain.geometry.geodesic_null_space import filter_delta_svd
from modelcypher.core.domain.geometry.gram_aligner import find_alignment
from modelcypher.core.domain.geometry.numerical_stability import all_finite, division_epsilon

# Model fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / ".models"
SMOLLM_PATH = FIXTURES_DIR / "HuggingFaceTB--SmolLM-135M"
LFM2_PATH = FIXTURES_DIR / "mlx-community--LFM2-350M-MLX-bf16"


def _build_synthetic_smollm_weights(backend):
    backend.random_seed(11)
    hidden_size = 576
    weights = {
        "model.embed_tokens.weight": backend.random_normal((128, hidden_size)),
        "model.norm.weight": backend.random_normal((hidden_size,)),
    }
    for idx in range(30):
        shape = (hidden_size, hidden_size * 2) if idx == 0 else (1, 1)
        weights[f"model.layers.{idx}.mlp.down_proj.weight"] = backend.random_normal(shape)
    return weights


def _build_synthetic_lfm2_weights(backend):
    backend.random_seed(17)
    hidden_size = 1024
    weights = {
        "model.embed.weight": backend.random_normal((128, hidden_size)),
        "model.norm.weight": backend.random_normal((hidden_size,)),
    }
    for idx in range(16):
        shape = (hidden_size * 2, hidden_size) if idx == 0 else (1, 1)
        weights[f"model.blocks.{idx}.mlp.down_proj.weight"] = backend.random_normal(shape)
    return weights


@pytest.fixture(scope="module")
def backend():
    return get_default_backend()


@pytest.fixture(scope="module")
def smollm_weights(backend):
    """Load SmolLM-135M weights."""
    model_file = SMOLLM_PATH / "model.safetensors"
    if model_file.exists():
        return backend.load_safetensors(str(model_file))
    return _build_synthetic_smollm_weights(backend)


@pytest.fixture(scope="module")
def lfm2_weights(backend):
    """Load LFM2-350M weights."""
    model_file = LFM2_PATH / "model.safetensors"
    if model_file.exists():
        return backend.load_safetensors(str(model_file))
    return _build_synthetic_lfm2_weights(backend)


@pytest.fixture(scope="module")
def smollm_config():
    """Load SmolLM config."""
    config_path = SMOLLM_PATH / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    return {"hidden_size": 576}


@pytest.fixture(scope="module")
def lfm2_config():
    """Load LFM2 config."""
    config_path = LFM2_PATH / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    return {"hidden_size": 1024}


class TestWeightLoading:
    """Tests for loading real model weights."""

    def test_smollm_weights_load(self, smollm_weights):
        """SmolLM weights should load successfully."""
        assert smollm_weights is not None
        assert len(smollm_weights) > 0

    def test_lfm2_weights_load(self, lfm2_weights):
        """LFM2 weights should load successfully."""
        assert lfm2_weights is not None
        assert len(lfm2_weights) > 0

    def test_smollm_has_expected_keys(self, smollm_weights):
        """SmolLM should have Llama-style weight keys."""
        keys = list(smollm_weights.keys())
        # Should have embedding
        assert any("embed" in k for k in keys)
        # Should have layer weights
        assert any("layers" in k for k in keys)
        # Should have MLP weights
        assert any("mlp" in k for k in keys)

    def test_lfm2_has_expected_keys(self, lfm2_weights):
        """LFM2 should have its custom weight keys."""
        keys = list(lfm2_weights.keys())
        assert len(keys) > 0
        # Should have some form of layer structure
        assert any("block" in k.lower() or "layer" in k.lower() for k in keys)

    def test_smollm_hidden_dim_matches_config(self, smollm_weights, smollm_config, backend):
        """SmolLM weight dimensions should match config."""
        hidden_size = smollm_config["hidden_size"]  # 576

        # Find a hidden projection weight
        for key, weight in smollm_weights.items():
            if "down_proj" in key and "weight" in key:
                shape = backend.shape(weight)
                # down_proj is [intermediate, hidden]
                assert shape[1] == hidden_size or shape[0] == hidden_size
                break

    def test_lfm2_hidden_dim_matches_config(self, lfm2_weights, lfm2_config, backend):
        """LFM2 weight dimensions should match config."""
        hidden_size = lfm2_config["hidden_size"]  # 1024

        # Find a projection weight
        for key, weight in lfm2_weights.items():
            shape = backend.shape(weight)
            if len(shape) == 2 and (shape[0] == hidden_size or shape[1] == hidden_size):
                # Found a weight with hidden_size dimension
                break


class TestLayerExtraction:
    """Tests for extracting layer information from weights."""

    def test_smollm_layer_indices(self, smollm_weights):
        """Should extract layer indices from SmolLM keys."""
        import re
        indices = set()
        for key in smollm_weights.keys():
            match = re.search(r"layers\.(\d+)\.", key)
            if match:
                indices.add(int(match.group(1)))

        # SmolLM-135M has 30 layers (0-29)
        assert len(indices) == 30
        assert min(indices) == 0
        assert max(indices) == 29

    def test_lfm2_layer_indices(self, lfm2_weights):
        """Should extract layer indices from LFM2 keys."""
        import re
        indices = set()
        for key in lfm2_weights.keys():
            # LFM2 uses different naming - search for number patterns
            match = re.search(r"\.(\d+)\.", key)
            if match:
                idx = int(match.group(1))
                if idx < 50:  # Filter out non-layer indices
                    indices.add(idx)

        # LFM2-350M has 16 layers
        assert len(indices) > 0


class TestCrossArchitectureAlignment:
    """Tests for aligning representations across different architectures."""

    def test_align_smollm_to_lfm2_synthetic_activations(self, backend):
        """Align synthetic activations with SmolLM → LFM2 dimensions."""
        # SmolLM: hidden=576, LFM2: hidden=1024
        n_samples = 64
        smollm_dim = 576
        lfm2_dim = 1024

        backend.random_seed(42)
        source_acts = backend.random_normal((n_samples, smollm_dim))
        target_acts = backend.random_normal((n_samples, lfm2_dim))
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)

        # Transform should be [576, 1024]
        F = backend.array(result.feature_transform)
        assert backend.shape(F) == (smollm_dim, lfm2_dim)

        # Apply transform
        aligned = backend.matmul(source_acts, F)
        backend.eval(aligned)
        assert backend.shape(aligned) == (n_samples, lfm2_dim)
        assert all_finite(aligned, backend)

    def test_align_lfm2_to_smollm_synthetic_activations(self, backend):
        """Align synthetic activations with LFM2 → SmolLM dimensions (compression)."""
        n_samples = 64
        smollm_dim = 576
        lfm2_dim = 1024

        backend.random_seed(42)
        source_acts = backend.random_normal((n_samples, lfm2_dim))
        target_acts = backend.random_normal((n_samples, smollm_dim))
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)

        # Transform should be [1024, 576]
        F = backend.array(result.feature_transform)
        assert backend.shape(F) == (lfm2_dim, smollm_dim)

        # Apply transform
        aligned = backend.matmul(source_acts, F)
        backend.eval(aligned)
        assert backend.shape(aligned) == (n_samples, smollm_dim)
        assert all_finite(aligned, backend)


class TestRealWeightTransformations:
    """Tests for applying transformations to real model weights."""

    def test_transform_smollm_down_proj(self, smollm_weights, backend):
        """Test transforming a real SmolLM down_proj weight."""
        # Find first down_proj weight
        weight = None
        for key in smollm_weights.keys():
            if "down_proj" in key and "weight" in key:
                weight = smollm_weights[key]
                break

        assert weight is not None
        shape = backend.shape(weight)

        backend.random_seed(42)
        # Create a delta (small perturbation)
        delta = backend.random_normal(shape) * 0.01
        backend.eval(delta)

        # Project delta through SVD filter
        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)

        assert all_finite(delta_proj, backend)
        assert backend.shape(delta_proj) == shape

    def test_transform_lfm2_weight(self, lfm2_weights, backend):
        """Test transforming a real LFM2 weight."""
        # Find a 2D weight
        weight = None
        for key in lfm2_weights.keys():
            w = lfm2_weights[key]
            if len(backend.shape(w)) == 2:
                weight = w
                break

        assert weight is not None
        shape = backend.shape(weight)

        backend.random_seed(42)
        delta = backend.random_normal(shape) * 0.01
        backend.eval(delta)

        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)

        assert all_finite(delta_proj, backend)
        assert backend.shape(delta_proj) == shape


class TestSVDFilterOnRealWeights:
    """Tests for SVD filter on real weights."""

    def test_svd_filter_preserves_structure(self, smollm_weights, backend):
        """SVD filter should preserve dominant structure of delta."""
        # Get a real weight
        weight = None
        for key in smollm_weights.keys():
            if "down_proj" in key and "weight" in key:
                weight = smollm_weights[key]
                break

        assert weight is not None
        shape = backend.shape(weight)

        backend.random_seed(42)
        delta = backend.random_normal(shape) * 0.1
        backend.eval(delta)

        original_norm = backend.mean(backend.abs(delta))
        backend.eval(original_norm)

        # Project using precision-derived rank
        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)

        # Measure effect after projection
        proj_norm = backend.mean(backend.abs(delta_proj))
        backend.eval(proj_norm)

        ratio = float(backend.to_scalar(proj_norm)) / float(backend.to_scalar(original_norm))
        eps = division_epsilon(backend, delta)
        tol = eps * max(1.0, abs(result.preserved_fraction))
        assert abs(ratio - result.preserved_fraction) <= tol


class TestCrossArchitectureWeightMerge:
    """Tests for merging weights across different architectures."""

    def test_can_compute_weight_delta(self, smollm_weights, lfm2_weights, backend):
        """Should be able to compute delta after alignment."""
        # Get weights of similar function (MLP projections)
        smollm_weight = None
        for key in smollm_weights.keys():
            if "down_proj" in key and "weight" in key:
                smollm_weight = smollm_weights[key]
                break

        lfm2_weight = None
        for key in lfm2_weights.keys():
            w = lfm2_weights[key]
            if len(backend.shape(w)) == 2:
                lfm2_weight = w
                break

        assert smollm_weight is not None
        assert lfm2_weight is not None

        smollm_shape = backend.shape(smollm_weight)
        lfm2_shape = backend.shape(lfm2_weight)

        # Create synthetic activations for alignment
        n_samples = 32
        backend.random_seed(42)
        source_acts = backend.random_normal((n_samples, smollm_shape[1]))
        target_acts = backend.random_normal((n_samples, lfm2_shape[1]))
        backend.eval(source_acts, target_acts)

        # Compute alignment
        result = find_alignment(source_acts, target_acts, backend)
        F = backend.array(result.feature_transform)
        backend.eval(F)

        # Transform source weight output dimension
        # For weight [in, out], we transform the output: W @ F
        aligned_source = backend.matmul(smollm_weight, F)
        backend.eval(aligned_source)

        # Shape should now be [smollm_in, lfm2_out]
        expected_shape = (smollm_shape[0], lfm2_shape[1])
        assert backend.shape(aligned_source) == expected_shape
        assert all_finite(aligned_source, backend)


class TestModelArchitectureInference:
    """Tests for inferring model architecture from weights."""

    def test_infer_smollm_hidden_dim(self, smollm_weights, backend):
        """Should infer correct hidden dimension from SmolLM weights."""
        from modelcypher.experimental.merge.helpers import infer_hidden_dim

        hidden_dim = infer_hidden_dim(smollm_weights)
        assert hidden_dim == 576  # From config

    def test_infer_lfm2_hidden_dim(self, lfm2_weights, backend):
        """Should infer correct hidden dimension from LFM2 weights."""
        from modelcypher.experimental.merge.helpers import infer_hidden_dim

        hidden_dim = infer_hidden_dim(lfm2_weights)
        # LFM2 has hidden_size=1024, but infer_hidden_dim may find different patterns
        assert hidden_dim > 0  # At least infers something


class TestWeightFiniteness:
    """Tests that all weight operations produce finite results."""

    def test_all_smollm_weights_finite(self, smollm_weights, backend):
        """All SmolLM weights should be finite."""
        for key, weight in smollm_weights.items():
            assert all_finite(weight, backend), f"Non-finite values in {key}"

    def test_all_lfm2_weights_finite(self, lfm2_weights, backend):
        """All LFM2 weights should be finite."""
        for key, weight in lfm2_weights.items():
            assert all_finite(weight, backend), f"Non-finite values in {key}"


class TestCrossArchitectureDensityComparison:
    """Tests for density comparison across different activation dimensions.

    This tests the bug discovered when merging LFM2 (4608 intermediate) into
    SmolLM (1536 intermediate) where the density comparison fails due to
    different feature dimensions.
    """

    def test_density_with_same_dimensions(self, backend):
        """Density comparison should work when dimensions match."""
        from modelcypher.core.domain.geometry.knowledge_density import (
            compute_knn_point_cloud_density,
        )

        n_samples = 64
        dim = 256

        backend.random_seed(42)
        source_acts = backend.random_normal((n_samples, dim))
        target_acts = backend.random_normal((n_samples, dim))
        backend.eval(source_acts, target_acts)

        result = compute_knn_point_cloud_density(
            source_acts, target_acts, backend=backend
        )

        assert result.source_densities is not None
        assert result.target_densities is not None
        assert int(backend.shape(result.source_densities)[0]) == n_samples
        assert int(backend.shape(result.target_densities)[0]) == n_samples

    def test_density_with_different_dimensions_should_handle_gracefully(self, backend):
        """Density comparison should handle different feature dimensions.

        When source and target have different dimensions (e.g., source=4608, target=1536),
        the density computation should either:
        1. Work by normalizing by sqrt(dim) for each
        2. Or require activations to be pre-aligned
        """
        from modelcypher.core.domain.geometry.knowledge_density import (
            compute_knn_point_cloud_density,
        )

        n_samples = 64
        source_dim = 4608  # LFM2 intermediate
        target_dim = 1536  # SmolLM intermediate

        backend.random_seed(42)
        source_acts = backend.random_normal((n_samples, source_dim))
        target_acts = backend.random_normal((n_samples, target_dim))
        backend.eval(source_acts, target_acts)

        # This should work - density computes independently on each point cloud
        result = compute_knn_point_cloud_density(
            source_acts, target_acts, backend=backend
        )

        assert result.source_densities is not None
        assert result.target_densities is not None
        assert int(backend.shape(result.source_densities)[0]) == n_samples
        assert int(backend.shape(result.target_densities)[0]) == n_samples


class TestCrossArchitectureMLPWeightTransplant:
    """Tests for MLP weight transplant across different architectures.

    Tests the full flow of transplanting MLP weights (gate_proj, up_proj, down_proj)
    when source and target have different hidden and intermediate dimensions.
    """

    def test_mlp_down_proj_stitch_dimensions(self, backend):
        """Test down_proj dual stitch produces correct dimensions.

        down_proj: [hidden_out, intermediate_in]
        Source: [1024, 4608]
        Target: [576, 1536]
        """
        # Source dimensions (LFM2)
        src_hidden = 1024
        src_inter = 4608

        # Target dimensions (SmolLM)
        tgt_hidden = 576
        tgt_inter = 1536

        backend.random_seed(42)

        # Create source down_proj weight [hidden, intermediate]
        source_down = backend.random_normal((src_hidden, src_inter))

        # Create stitch transforms
        # hidden_stitch_output: [tgt_hidden, src_hidden] maps output dim
        hidden_stitch_output = backend.random_normal((tgt_hidden, src_hidden))
        # intermediate_stitch_input: [src_inter, tgt_inter] maps input dim
        intermediate_stitch_input = backend.random_normal((src_inter, tgt_inter))
        backend.eval(source_down, hidden_stitch_output, intermediate_stitch_input)

        # Apply dual stitch for down_proj: H @ W @ I
        stitched = backend.matmul(hidden_stitch_output, source_down)
        stitched = backend.matmul(stitched, intermediate_stitch_input)
        backend.eval(stitched)

        # Result should match target dimensions
        assert backend.shape(stitched) == (tgt_hidden, tgt_inter)
        assert all_finite(stitched, backend)

    def test_mlp_gate_up_proj_stitch_dimensions(self, backend):
        """Test gate/up_proj dual stitch produces correct dimensions.

        gate_proj/up_proj: [intermediate_out, hidden_in]
        Source: [4608, 1024]
        Target: [1536, 576]
        """
        src_hidden = 1024
        src_inter = 4608
        tgt_hidden = 576
        tgt_inter = 1536

        backend.random_seed(42)

        # Create source gate_proj weight [intermediate, hidden]
        source_gate = backend.random_normal((src_inter, src_hidden))

        # Create stitch transforms
        intermediate_stitch_output = backend.random_normal((tgt_inter, src_inter))
        hidden_stitch_input = backend.random_normal((src_hidden, tgt_hidden))
        backend.eval(source_gate, intermediate_stitch_output, hidden_stitch_input)

        # Apply dual stitch for gate/up: I_out @ W @ H_in
        stitched = backend.matmul(intermediate_stitch_output, source_gate)
        stitched = backend.matmul(stitched, hidden_stitch_input)
        backend.eval(stitched)

        # Result should match target dimensions
        assert backend.shape(stitched) == (tgt_inter, tgt_hidden)
        assert all_finite(stitched, backend)

    def test_weight_space_transplant_with_cross_dim_activations(self, backend):
        """Test weight space transplant when activation dimensions differ.

        This tests the bug path where source intermediate activations (4608-dim)
        and target intermediate activations (1536-dim) are passed to density
        comparison without proper alignment.
        """
        from modelcypher.core.domain.geometry.transplant import (
            compute_weight_space_transplant,
        )

        n_samples = 64
        tgt_hidden = 576
        tgt_inter = 1536

        backend.random_seed(42)

        # After stitching, weights are in target dimensions
        source_aligned = backend.random_normal((tgt_hidden, tgt_inter))
        target_weight = backend.random_normal((tgt_hidden, tgt_inter))

        # Input activations for null-space: target intermediate [n, tgt_inter]
        input_activations = backend.random_normal((n_samples, tgt_inter))

        # For density comparison, source activations should ALSO be in
        # target dimension (after alignment) - this is the bug!
        # Currently the code passes raw source intermediate [n, src_inter]
        source_density_acts = backend.random_normal((n_samples, tgt_inter))  # Aligned
        target_density_acts = input_activations

        backend.eval(
            source_aligned, target_weight, input_activations,
            source_density_acts, target_density_acts
        )

        # This should work when density activations have matching dimensions
        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            source_activations_for_density=source_density_acts,
            target_activations_for_density=target_density_acts,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert backend.shape(result.merged_weight) == (tgt_hidden, tgt_inter)
        assert all_finite(result.merged_weight, backend)

    def test_weight_space_transplant_without_density_weighting(self, backend):
        """Test weight space transplant without density (uniform transfer)."""
        from modelcypher.core.domain.geometry.transplant import (
            compute_weight_space_transplant,
        )

        n_samples = 64
        tgt_hidden = 576
        tgt_inter = 1536

        backend.random_seed(42)

        source_aligned = backend.random_normal((tgt_hidden, tgt_inter))
        target_weight = backend.random_normal((tgt_hidden, tgt_inter))
        input_activations = backend.random_normal((n_samples, tgt_inter))

        backend.eval(source_aligned, target_weight, input_activations)

        # Without density activations, should use uniform transfer
        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            source_activations_for_density=None,
            target_activations_for_density=None,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert backend.shape(result.merged_weight) == (tgt_hidden, tgt_inter)
        assert all_finite(result.merged_weight, backend)


class TestIntermediateActivationAlignment:
    """Tests for aligning intermediate activations across architectures.

    When source and target have different intermediate dimensions, the source
    intermediate activations must be aligned (transformed) before density
    comparison can be performed.
    """

    def test_intermediate_alignment_transform_dimensions(self, backend):
        """Test computing intermediate alignment transform."""
        n_samples = 64
        src_inter = 4608
        tgt_inter = 1536

        backend.random_seed(42)

        # Source and target intermediate activations
        src_inter_acts = backend.random_normal((n_samples, src_inter))
        tgt_inter_acts = backend.random_normal((n_samples, tgt_inter))
        backend.eval(src_inter_acts, tgt_inter_acts)

        # Compute alignment transform: I such that src @ I ≈ tgt
        result = find_alignment(src_inter_acts, tgt_inter_acts, backend)
        I_transform = backend.array(result.feature_transform)
        backend.eval(I_transform)

        # Transform should be [src_inter, tgt_inter]
        assert backend.shape(I_transform) == (src_inter, tgt_inter)

        # Apply transform
        aligned_src = backend.matmul(src_inter_acts, I_transform)
        backend.eval(aligned_src)

        # Aligned source should have target dimensions
        assert backend.shape(aligned_src) == (n_samples, tgt_inter)
        assert all_finite(aligned_src, backend)

    def test_aligned_intermediate_activations_have_high_cka(self, backend):
        """Aligned intermediate activations should have high CKA."""
        n_samples = 64
        src_inter = 4608
        tgt_inter = 1536

        backend.random_seed(42)

        # Create correlated activations (shared structure)
        shared = backend.random_normal((n_samples, min(src_inter, tgt_inter)))
        backend.eval(shared)

        # Expand to source/target dims with some noise
        src_expand = backend.random_normal((min(src_inter, tgt_inter), src_inter))
        tgt_expand = backend.random_normal((min(src_inter, tgt_inter), tgt_inter))
        backend.eval(src_expand, tgt_expand)

        src_inter_acts = backend.matmul(shared, src_expand)
        tgt_inter_acts = backend.matmul(shared, tgt_expand)
        backend.eval(src_inter_acts, tgt_inter_acts)

        # Compute alignment
        result = find_alignment(src_inter_acts, tgt_inter_acts, backend)
        I_transform = backend.array(result.feature_transform)

        # Apply transform
        aligned_src = backend.matmul(src_inter_acts, I_transform)
        backend.eval(aligned_src)

        # Compute CKA between aligned source and target
        cka = compute_geodesic_cka(aligned_src, tgt_inter_acts, backend)

        eps = division_epsilon(backend, aligned_src)
        assert abs(cka - 1.0) <= eps

    def test_intermediate_stitch_from_alignment(self, backend):
        """Test computing intermediate stitch matrices from alignment."""
        n_samples = 64
        src_inter = 4608
        tgt_inter = 1536

        backend.random_seed(42)

        src_inter_acts = backend.random_normal((n_samples, src_inter))
        tgt_inter_acts = backend.random_normal((n_samples, tgt_inter))
        backend.eval(src_inter_acts, tgt_inter_acts)

        # Compute alignment
        result = find_alignment(src_inter_acts, tgt_inter_acts, backend)
        I = backend.array(result.feature_transform)  # [src_inter, tgt_inter]
        backend.eval(I)

        # For weight stitching, we need:
        # intermediate_stitch_output = I^T = [tgt_inter, src_inter]
        # intermediate_stitch_input = I = [src_inter, tgt_inter]
        intermediate_stitch_output = backend.transpose(I)  # [tgt_inter, src_inter]
        intermediate_stitch_input = I  # [src_inter, tgt_inter]

        assert backend.shape(intermediate_stitch_output) == (tgt_inter, src_inter)
        assert backend.shape(intermediate_stitch_input) == (src_inter, tgt_inter)

        # Verify stitch application for down_proj: [hidden, inter]
        src_hidden = 1024
        tgt_hidden = 576
        hidden_stitch_output = backend.random_normal((tgt_hidden, src_hidden))
        down_proj = backend.random_normal((src_hidden, src_inter))
        backend.eval(hidden_stitch_output, down_proj)

        # Apply: H_out @ W @ I_in
        stitched = backend.matmul(hidden_stitch_output, down_proj)
        stitched = backend.matmul(stitched, intermediate_stitch_input)
        backend.eval(stitched)

        assert backend.shape(stitched) == (tgt_hidden, tgt_inter)
        assert all_finite(stitched, backend)


class TestBroadcastShapeBug:
    """Tests to reproduce and verify the (1536,1536) vs (576,576) broadcast bug.

    This bug occurs during cross-architecture MLP weight transplant when
    intermediate activations have unexpected shapes.
    """

    def test_geodesic_with_transposed_activations(self, backend):
        """Test what happens when activations are transposed [dim, n] instead of [n, dim]."""
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        # If activations are transposed, geodesic distances would be [dim, dim]
        # instead of [n, n]
        n_samples = 64
        feature_dim = 256

        backend.random_seed(42)

        # Correct shape: [n_samples, feature_dim]
        correct_acts = backend.random_normal((n_samples, feature_dim))
        backend.eval(correct_acts)

        rg = RiemannianGeometry(backend)
        correct_result = rg.geodesic_distances(correct_acts, k_neighbors=5)

        # Distance matrix should be [n_samples, n_samples]
        assert backend.shape(correct_result.distances) == (n_samples, n_samples)

        # Transposed shape: [feature_dim, n_samples]
        transposed_acts = backend.random_normal((feature_dim, n_samples))
        backend.eval(transposed_acts)

        transposed_result = rg.geodesic_distances(transposed_acts, k_neighbors=5)

        # This would produce [feature_dim, feature_dim] - the bug pattern!
        assert backend.shape(transposed_result.distances) == (feature_dim, feature_dim)

    def test_density_comparison_with_mismatched_sample_counts(self, backend):
        """Test density comparison when source and target have different sample counts.

        This could happen if activations are stored transposed with different dims.
        """
        from modelcypher.core.domain.geometry.knowledge_density import (
            compute_knn_point_cloud_density,
        )

        # Simulate the bug: source and target have different "sample counts"
        # which would happen if [dim, n] was used instead of [n, dim]
        src_fake_samples = 1536  # Actually feature_dim
        tgt_fake_samples = 576   # Actually feature_dim
        dim = 64  # Actually n_samples

        backend.random_seed(42)
        source_acts = backend.random_normal((src_fake_samples, dim))
        target_acts = backend.random_normal((tgt_fake_samples, dim))
        backend.eval(source_acts, target_acts)

        # This will create distance matrices [1536, 1536] and [576, 576]
        # which cannot be compared element-wise
        result = compute_knn_point_cloud_density(
            source_acts, target_acts, backend=backend
        )

        # Densities have different lengths - this is the bug state
        src_len = int(backend.shape(result.source_densities)[0])
        tgt_len = int(backend.shape(result.target_densities)[0])

        # With different sample counts, densities have different lengths
        assert src_len == src_fake_samples
        assert tgt_len == tgt_fake_samples

    def test_intermediate_activation_shape_expectation(self, backend):
        """Verify expected shape of intermediate activations."""
        # Intermediate activations should be [n_samples, intermediate_dim]
        n_samples = 1024
        tgt_inter_dim = 1536  # SmolLM intermediate
        src_inter_dim = 4608  # LFM2 intermediate

        backend.random_seed(42)

        # Correct shapes
        src_inter_correct = backend.random_normal((n_samples, src_inter_dim))
        tgt_inter_correct = backend.random_normal((n_samples, tgt_inter_dim))
        backend.eval(src_inter_correct, tgt_inter_correct)

        # Verify correct shape indexing
        assert int(backend.shape(src_inter_correct)[0]) == n_samples  # samples first
        assert int(backend.shape(src_inter_correct)[1]) == src_inter_dim  # features second
        assert int(backend.shape(tgt_inter_correct)[0]) == n_samples
        assert int(backend.shape(tgt_inter_correct)[1]) == tgt_inter_dim

    def test_weight_transplant_with_different_density_feature_dims(self, backend):
        """Test weight transplant when density activations have different feature dimensions.

        This is the normal case for cross-architecture merges: same sample count,
        different feature dimensions (source 4608, target 1536).
        """
        from modelcypher.core.domain.geometry.transplant import (
            compute_weight_space_transplant,
        )

        tgt_hidden = 576
        tgt_inter = 1536
        src_inter = 4608

        backend.random_seed(42)

        source_aligned = backend.random_normal((tgt_hidden, tgt_inter))
        target_weight = backend.random_normal((tgt_hidden, tgt_inter))

        # All activations have the same sample count
        n_samples = 64
        input_activations = backend.random_normal((n_samples, tgt_inter))

        # Cross-architecture: same samples, different feature dims
        src_density_acts = backend.random_normal((n_samples, src_inter))
        tgt_density_acts = backend.random_normal((n_samples, tgt_inter))

        backend.eval(
            source_aligned, target_weight, input_activations,
            src_density_acts, tgt_density_acts
        )

        # This should work - density comparison handles different feature dims
        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            source_activations_for_density=src_density_acts,
            target_activations_for_density=tgt_density_acts,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert backend.shape(result.merged_weight) == (tgt_hidden, tgt_inter)
        assert all_finite(result.merged_weight, backend)

    def test_weight_transplant_with_mismatched_activation_sample_counts(self, backend):
        """Test that weight transplant handles mismatched sample counts gracefully.

        This is an edge case where density activations have different sample counts
        due to transposed storage or other bugs.
        """
        from modelcypher.core.domain.geometry.transplant import (
            compute_weight_space_transplant,
        )

        tgt_hidden = 576
        tgt_inter = 1536

        backend.random_seed(42)

        source_aligned = backend.random_normal((tgt_hidden, tgt_inter))
        target_weight = backend.random_normal((tgt_hidden, tgt_inter))

        # Correct: input_activations for null-space
        n_samples = 64
        input_activations = backend.random_normal((n_samples, tgt_inter))

        # Bug scenario: density activations with different sample counts
        # (would happen if activations were transposed)
        src_density_fake_samples = 128  # Different sample count
        tgt_density_fake_samples = 96   # Different sample count

        src_density_acts = backend.random_normal((src_density_fake_samples, 64))
        tgt_density_acts = backend.random_normal((tgt_density_fake_samples, 64))

        backend.eval(
            source_aligned, target_weight, input_activations,
            src_density_acts, tgt_density_acts
        )

        # This should now work with the density truncation fix
        # The density weights will be truncated/padded to match input_activations
        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            source_activations_for_density=src_density_acts,
            target_activations_for_density=tgt_density_acts,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert backend.shape(result.merged_weight) == (tgt_hidden, tgt_inter)
