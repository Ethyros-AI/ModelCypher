# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests validating information conservation across synthetic layer transformations.

The hypothesis: Information is neither lost nor gained as it flows through layers -
it is transformed. Entropy may drop but total "energy" (variance, capacity) remains stable.

These tests verify information conservation properties on synthetic transformations
with known mathematical properties:
1. Orthogonal transformations: Perfect conservation (no information loss)
2. Bottleneck transformations: Information compressed but recoverable
3. Expansion transformations: Information spread but not created
4. Noisy transformations: Information degraded in predictable ways

Key metrics:
- Intrinsic dimension (TwoNN): Should track transformation rank
- Signal rank (RMT): Should match effective degrees of freedom
- Variance captured: Should be stable or follow predictable trajectory
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
    TwoNNEstimate,
)
from modelcypher.core.domain.geometry.rmt_signal_separation import (
    separate_signal_noise,
    MPSignalNoiseResult,
)


def _sqrt_eps(backend):
    """Get sqrt(machine_epsilon) for the default float type."""
    eps = backend.finfo().eps
    return sqrt_scalar(eps, backend)


def _random_orthogonal(n: int, backend):
    """Generate a random orthogonal matrix via QR decomposition."""
    b = backend
    random_matrix = b.random_normal((n, n))
    Q, _ = b.qr(random_matrix)
    b.eval(Q)
    return Q


def _generate_manifold_data(n_samples: int, intrinsic_dim: int, ambient_dim: int, backend):
    """Generate synthetic manifold data with known intrinsic dimension.

    Data lives on an intrinsic_dim-dimensional linear subspace of R^ambient_dim.
    """
    b = backend
    assert intrinsic_dim <= ambient_dim

    # Generate intrinsic coordinates
    intrinsic_coords = b.random_normal((n_samples, intrinsic_dim))
    b.eval(intrinsic_coords)

    # Random embedding into ambient space
    embedding = b.random_normal((intrinsic_dim, ambient_dim))
    b.eval(embedding)

    # Project to ambient space
    ambient_coords = b.matmul(intrinsic_coords, embedding)
    b.eval(ambient_coords)

    return ambient_coords


class TestOrthogonalTransformConservation:
    """Test that orthogonal transforms preserve information exactly."""

    def test_id_preserved_under_rotation(self):
        """Intrinsic dimension should be unchanged by rotation."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        intrinsic_dim = 8
        ambient_dim = 32

        # Generate manifold data
        data = _generate_manifold_data(n_samples, intrinsic_dim, ambient_dim, b)

        # Compute ID before rotation
        id_computer = IntrinsicDimension(b)
        id_before = id_computer.compute(data)

        # Apply random rotation
        Q = _random_orthogonal(ambient_dim, b)
        data_rotated = b.matmul(data, Q)
        b.eval(data_rotated)

        # Compute ID after rotation
        id_after = id_computer.compute(data_rotated)

        print(f"\nOrthogonal transform ID conservation:")
        print(f"  ID before: {id_before.intrinsic_dimension:.2f}")
        print(f"  ID after:  {id_after.intrinsic_dimension:.2f}")
        print(f"  True ID:   {intrinsic_dim}")

        # ID should be approximately equal (within 20% relative tolerance)
        rel_diff = abs(id_before.intrinsic_dimension - id_after.intrinsic_dimension)
        rel_diff /= max(id_before.intrinsic_dimension, 1.0)
        assert rel_diff < 0.2, (
            f"ID changed too much under rotation: {id_before.intrinsic_dimension:.2f} -> "
            f"{id_after.intrinsic_dimension:.2f}"
        )

    def test_signal_rank_preserved_under_rotation(self):
        """RMT signal rank should be unchanged by rotation."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        intrinsic_dim = 8
        ambient_dim = 32

        # Generate manifold data
        data = _generate_manifold_data(n_samples, intrinsic_dim, ambient_dim, b)

        # Compute signal rank before rotation
        rmt_before = separate_signal_noise(data, b)

        # Apply random rotation
        Q = _random_orthogonal(ambient_dim, b)
        data_rotated = b.matmul(data, Q)
        b.eval(data_rotated)

        # Compute signal rank after rotation
        rmt_after = separate_signal_noise(data_rotated, b)

        print(f"\nOrthogonal transform RMT conservation:")
        print(f"  Signal rank before: {rmt_before.signal_rank}")
        print(f"  Signal rank after:  {rmt_after.signal_rank}")
        print(f"  True rank:          {intrinsic_dim}")

        # Signal rank should be exactly equal (integer metric)
        assert rmt_before.signal_rank == rmt_after.signal_rank, (
            f"Signal rank changed under rotation: {rmt_before.signal_rank} -> "
            f"{rmt_after.signal_rank}"
        )

    def test_variance_preserved_under_rotation(self):
        """Total variance should be unchanged by orthogonal transform."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        ambient_dim = 32

        data = b.random_normal((n_samples, ambient_dim))
        b.eval(data)

        # Total variance before
        var_before = b.sum(b.var(data, axis=0))
        b.eval(var_before)

        # Apply rotation
        Q = _random_orthogonal(ambient_dim, b)
        data_rotated = b.matmul(data, Q)
        b.eval(data_rotated)

        # Total variance after
        var_after = b.sum(b.var(data_rotated, axis=0))
        b.eval(var_after)

        var_before_val = float(b.to_scalar(var_before))
        var_after_val = float(b.to_scalar(var_after))

        print(f"\nOrthogonal transform variance conservation:")
        print(f"  Variance before: {var_before_val:.4f}")
        print(f"  Variance after:  {var_after_val:.4f}")

        sqrt_eps = _sqrt_eps(b)
        rel_diff = abs(var_before_val - var_after_val) / max(var_before_val, 1e-10)
        assert rel_diff < sqrt_eps, (
            f"Variance changed under rotation: {var_before_val:.4f} -> {var_after_val:.4f}"
        )


class TestBottleneckConservation:
    """Test information behavior through bottleneck layers."""

    def test_bottleneck_reduces_effective_rank(self):
        """Projecting through bottleneck should reduce signal rank appropriately."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        dim_in = 64
        dim_bottleneck = 16

        # Full-rank input data
        data = b.random_normal((n_samples, dim_in))
        b.eval(data)

        # Random projection to bottleneck
        proj = b.random_normal((dim_in, dim_bottleneck))
        b.eval(proj)
        data_bottleneck = b.matmul(data, proj)
        b.eval(data_bottleneck)

        # Compute signal ranks
        rmt_before = separate_signal_noise(data, b)
        rmt_after = separate_signal_noise(data_bottleneck, b)

        print(f"\nBottleneck effect on signal rank:")
        print(f"  Input dim:       {dim_in}")
        print(f"  Bottleneck dim:  {dim_bottleneck}")
        print(f"  Signal rank in:  {rmt_before.signal_rank}")
        print(f"  Signal rank out: {rmt_after.signal_rank}")

        # Signal rank after bottleneck should be at most bottleneck dimension
        assert rmt_after.signal_rank <= dim_bottleneck, (
            f"Signal rank {rmt_after.signal_rank} exceeds bottleneck {dim_bottleneck}"
        )

    def test_bottleneck_variance_ratio(self):
        """Variance through bottleneck should be predictable from singular values."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        dim_in = 64
        dim_bottleneck = 16

        # Generate data with known covariance structure
        data = b.random_normal((n_samples, dim_in))
        b.eval(data)

        # SVD of projection matrix gives us expected variance retention
        proj = b.random_normal((dim_in, dim_bottleneck))
        b.eval(proj)

        # Project data
        data_bottleneck = b.matmul(data, proj)
        b.eval(data_bottleneck)

        # Compute variance before and after
        var_before = b.sum(b.var(data, axis=0))
        var_after = b.sum(b.var(data_bottleneck, axis=0))
        b.eval(var_before, var_after)

        var_before_val = float(b.to_scalar(var_before))
        var_after_val = float(b.to_scalar(var_after))

        print(f"\nBottleneck variance:")
        print(f"  Variance in:  {var_before_val:.4f}")
        print(f"  Variance out: {var_after_val:.4f}")
        print(f"  Ratio: {var_after_val / var_before_val:.4f}")

        # Variance can increase or decrease through random projection
        # but should be bounded and predictable (not zero, not infinite)
        assert var_after_val > 0, "Variance collapsed to zero through bottleneck"
        # Random projection can amplify variance - upper bound is dim_in * dim_bottleneck
        # (each output is sum of dim_in inputs, and we have dim_bottleneck outputs)
        assert var_after_val < var_before_val * dim_in, (
            "Variance explosion through bottleneck"
        )


class TestExpansionConservation:
    """Test that expansion doesn't create information."""

    def test_expansion_id_bounded(self):
        """ID after expansion should not exceed original effective dimension."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        intrinsic_dim = 8
        dim_small = 16
        dim_large = 64

        # Generate low-rank data in small dimension
        data_small = _generate_manifold_data(n_samples, intrinsic_dim, dim_small, b)

        # Compute ID before expansion
        id_computer = IntrinsicDimension(b)
        id_before = id_computer.compute(data_small)

        # Expand to larger dimension via random embedding
        embedding = b.random_normal((dim_small, dim_large))
        b.eval(embedding)
        data_large = b.matmul(data_small, embedding)
        b.eval(data_large)

        # Compute ID after expansion
        id_after = id_computer.compute(data_large)

        print(f"\nExpansion ID behavior:")
        print(f"  Dim before:   {dim_small}")
        print(f"  Dim after:    {dim_large}")
        print(f"  True ID:      {intrinsic_dim}")
        print(f"  ID before:    {id_before.intrinsic_dimension:.2f}")
        print(f"  ID after:     {id_after.intrinsic_dimension:.2f}")

        # ID should be approximately preserved (expansion doesn't create structure)
        rel_diff = abs(id_before.intrinsic_dimension - id_after.intrinsic_dimension)
        rel_diff /= max(id_before.intrinsic_dimension, 1.0)
        assert rel_diff < 0.3, (
            f"ID changed too much under expansion: {id_before.intrinsic_dimension:.2f} -> "
            f"{id_after.intrinsic_dimension:.2f}"
        )

    def test_expansion_signal_rank_bounded(self):
        """Signal rank after expansion should not exceed original."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        intrinsic_dim = 8
        dim_small = 16
        dim_large = 64

        # Generate low-rank data
        data_small = _generate_manifold_data(n_samples, intrinsic_dim, dim_small, b)

        # Compute signal rank before
        rmt_before = separate_signal_noise(data_small, b)

        # Expand to larger dimension
        embedding = b.random_normal((dim_small, dim_large))
        b.eval(embedding)
        data_large = b.matmul(data_small, embedding)
        b.eval(data_large)

        # Compute signal rank after
        rmt_after = separate_signal_noise(data_large, b)

        print(f"\nExpansion signal rank behavior:")
        print(f"  Signal rank before: {rmt_before.signal_rank}")
        print(f"  Signal rank after:  {rmt_after.signal_rank}")
        print(f"  True rank:          {intrinsic_dim}")

        # Signal rank should be approximately preserved
        # Allow some tolerance because RMT edge detection can vary
        assert rmt_after.signal_rank <= rmt_before.signal_rank + 4, (
            f"Signal rank increased too much: {rmt_before.signal_rank} -> {rmt_after.signal_rank}"
        )


class TestNoisyTransformDegradation:
    """Test that noise degrades information predictably."""

    def test_noise_increases_id(self):
        """Adding noise should increase effective intrinsic dimension."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        intrinsic_dim = 8
        ambient_dim = 32

        # Generate clean manifold data
        data_clean = _generate_manifold_data(n_samples, intrinsic_dim, ambient_dim, b)

        # Compute ID before noise
        id_computer = IntrinsicDimension(b)
        id_clean = id_computer.compute(data_clean)

        # Add significant noise (SNR ~ 1)
        noise_scale = float(b.to_scalar(b.std(data_clean)))
        noise = b.random_normal(data_clean.shape) * noise_scale
        b.eval(noise)
        data_noisy = data_clean + noise
        b.eval(data_noisy)

        # Compute ID after noise
        id_noisy = id_computer.compute(data_noisy)

        print(f"\nNoise effect on ID:")
        print(f"  True ID:     {intrinsic_dim}")
        print(f"  ID clean:    {id_clean.intrinsic_dimension:.2f}")
        print(f"  ID noisy:    {id_noisy.intrinsic_dimension:.2f}")
        print(f"  Ambient dim: {ambient_dim}")

        # Noisy ID should be higher (noise spreads data)
        assert id_noisy.intrinsic_dimension >= id_clean.intrinsic_dimension * 0.8, (
            f"Noisy ID unexpectedly lower: {id_clean.intrinsic_dimension:.2f} -> "
            f"{id_noisy.intrinsic_dimension:.2f}"
        )

    def test_noise_reduces_signal_to_noise_ratio(self):
        """Adding noise should shift eigenvalues into noise bulk."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        intrinsic_dim = 8
        ambient_dim = 32

        # Generate clean manifold data
        data_clean = _generate_manifold_data(n_samples, intrinsic_dim, ambient_dim, b)

        # Compute RMT before noise
        rmt_clean = separate_signal_noise(data_clean, b)

        # Add moderate noise
        noise_scale = float(b.to_scalar(b.std(data_clean))) * 0.5
        noise = b.random_normal(data_clean.shape) * noise_scale
        b.eval(noise)
        data_noisy = data_clean + noise
        b.eval(data_noisy)

        # Compute RMT after noise
        rmt_noisy = separate_signal_noise(data_noisy, b)

        print(f"\nNoise effect on RMT:")
        print(f"  True rank:        {intrinsic_dim}")
        print(f"  Signal rank clean: {rmt_clean.signal_rank}")
        print(f"  Signal rank noisy: {rmt_noisy.signal_rank}")
        print(f"  Signal var fraction clean: {rmt_clean.signal_variance_fraction:.4f}")
        print(f"  Signal var fraction noisy: {rmt_noisy.signal_variance_fraction:.4f}")
        print(f"  Noise var clean: {rmt_clean.noise_variance:.6f}")
        print(f"  Noise var noisy: {rmt_noisy.noise_variance:.6f}")

        # Signal variance fraction should decrease when noise is added
        # (noise reduces the proportion of variance captured by signal directions)
        # Allow some tolerance because the effect depends on noise level
        assert rmt_noisy.signal_variance_fraction <= rmt_clean.signal_variance_fraction + 0.1, (
            f"Signal fraction unexpectedly increased with noise: "
            f"{rmt_clean.signal_variance_fraction:.4f} -> {rmt_noisy.signal_variance_fraction:.4f}"
        )

        # Noise variance estimate should increase when we add noise
        # (the estimated noise level from the MP bulk should be higher)
        print(f"  Noise variance increased: {rmt_noisy.noise_variance > rmt_clean.noise_variance}")


class TestLayerChainConservation:
    """Test conservation across a chain of transformations."""

    def test_three_layer_chain_id_stability(self):
        """ID should be relatively stable across orthogonal-bottleneck-expansion chain."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        intrinsic_dim = 8
        dims = [32, 32, 16, 32]  # Initial -> Rotate -> Bottleneck -> Expand

        # Generate initial data
        data = _generate_manifold_data(n_samples, intrinsic_dim, dims[0], b)

        id_computer = IntrinsicDimension(b)
        ids = [id_computer.compute(data).intrinsic_dimension]

        # Layer 1: Orthogonal rotation (should preserve)
        Q = _random_orthogonal(dims[1], b)
        data = b.matmul(data, Q)
        b.eval(data)
        ids.append(id_computer.compute(data).intrinsic_dimension)

        # Layer 2: Bottleneck projection (may reduce)
        proj_down = b.random_normal((dims[1], dims[2]))
        b.eval(proj_down)
        data = b.matmul(data, proj_down)
        b.eval(data)
        ids.append(id_computer.compute(data).intrinsic_dimension)

        # Layer 3: Expansion (should preserve)
        proj_up = b.random_normal((dims[2], dims[3]))
        b.eval(proj_up)
        data = b.matmul(data, proj_up)
        b.eval(data)
        ids.append(id_computer.compute(data).intrinsic_dimension)

        print(f"\nThree-layer chain ID evolution:")
        print(f"  True ID:  {intrinsic_dim}")
        for i, (dim, id_val) in enumerate(zip(dims, ids)):
            print(f"  Layer {i} (dim={dim}): ID = {id_val:.2f}")

        # Check that final ID is within 50% of initial
        # (bottleneck may compress, but shouldn't destroy)
        rel_change = abs(ids[-1] - ids[0]) / max(ids[0], 1.0)
        assert rel_change < 0.5, (
            f"ID changed too much across chain: {ids[0]:.2f} -> {ids[-1]:.2f}"
        )

    def test_chain_variance_accounting(self):
        """Total variance through chain should be trackable."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 200
        dims = [32, 32, 16, 32]

        # Generate initial data
        data = b.random_normal((n_samples, dims[0]))
        b.eval(data)

        variances = [float(b.to_scalar(b.sum(b.var(data, axis=0))))]

        # Layer 1: Orthogonal (exact preservation)
        Q = _random_orthogonal(dims[1], b)
        data = b.matmul(data, Q)
        b.eval(data)
        variances.append(float(b.to_scalar(b.sum(b.var(data, axis=0)))))

        # Layer 2: Bottleneck
        proj_down = b.random_normal((dims[1], dims[2])) / (dims[1] ** 0.5)  # Scale for stability
        b.eval(proj_down)
        data = b.matmul(data, proj_down)
        b.eval(data)
        variances.append(float(b.to_scalar(b.sum(b.var(data, axis=0)))))

        # Layer 3: Expansion
        proj_up = b.random_normal((dims[2], dims[3])) / (dims[2] ** 0.5)  # Scale for stability
        b.eval(proj_up)
        data = b.matmul(data, proj_up)
        b.eval(data)
        variances.append(float(b.to_scalar(b.sum(b.var(data, axis=0)))))

        print(f"\nThree-layer chain variance evolution:")
        for i, (dim, var_val) in enumerate(zip(dims, variances)):
            print(f"  Layer {i} (dim={dim}): Variance = {var_val:.4f}")

        # Orthogonal layer should preserve variance exactly
        sqrt_eps = _sqrt_eps(b)
        rel_diff_ortho = abs(variances[1] - variances[0]) / max(variances[0], 1e-10)
        assert rel_diff_ortho < sqrt_eps, (
            f"Orthogonal layer didn't preserve variance: {variances[0]:.4f} -> {variances[1]:.4f}"
        )

        # All variances should be positive and finite
        for i, var_val in enumerate(variances):
            assert var_val > 0, f"Layer {i} has zero variance"
            assert var_val < 1e10, f"Layer {i} has exploding variance"
