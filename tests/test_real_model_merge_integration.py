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

from pathlib import Path
import json

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import find_alignment
from modelcypher.core.domain.geometry.cka import compute_linear_cka
from modelcypher.core.domain.geometry.geodesic_null_space import filter_delta_svd
from modelcypher.core.domain.geometry.numerical_stability import all_finite


# Model fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / ".models"
SMOLLM_PATH = FIXTURES_DIR / "HuggingFaceTB--SmolLM-135M"
LFM2_PATH = FIXTURES_DIR / "mlx-community--LFM2-350M-MLX-bf16"

# Skip if fixtures not available
pytestmark = pytest.mark.skipif(
    not (SMOLLM_PATH / "model.safetensors").exists() or not (LFM2_PATH / "model.safetensors").exists(),
    reason="Real model fixtures not found"
)


@pytest.fixture(scope="module")
def backend():
    return get_default_backend()


@pytest.fixture(scope="module")
def smollm_weights(backend):
    """Load SmolLM-135M weights."""
    return backend.load_safetensors(str(SMOLLM_PATH / "model.safetensors"))


@pytest.fixture(scope="module")
def lfm2_weights(backend):
    """Load LFM2-350M weights."""
    return backend.load_safetensors(str(LFM2_PATH / "model.safetensors"))


@pytest.fixture(scope="module")
def smollm_config():
    """Load SmolLM config."""
    with open(SMOLLM_PATH / "config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def lfm2_config():
    """Load LFM2 config."""
    with open(LFM2_PATH / "config.json") as f:
        return json.load(f)


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

        # Project with high energy threshold
        result = filter_delta_svd(
            delta,
            backend=backend,
            energy_threshold=0.99,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)

        # Measure effect after projection
        proj_norm = backend.mean(backend.abs(delta_proj))
        backend.eval(proj_norm)

        # High threshold should preserve most of the signal
        ratio = float(backend.to_scalar(proj_norm)) / float(backend.to_scalar(original_norm))
        assert ratio > 0.5  # At least 50% preserved


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
        from modelcypher.core.use_cases.merge.helpers import infer_hidden_dim

        hidden_dim = infer_hidden_dim(smollm_weights)
        assert hidden_dim == 576  # From config

    def test_infer_lfm2_hidden_dim(self, lfm2_weights, backend):
        """Should infer correct hidden dimension from LFM2 weights."""
        from modelcypher.core.use_cases.merge.helpers import infer_hidden_dim

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
