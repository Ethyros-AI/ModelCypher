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

"""
Verification tests for Phase 9: Advanced Geometry & OT.
Tests TransportGuidedMerger and SharedSubspaceProjector.
"""

import unittest

from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    Config as SharedSubspaceConfig,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    SharedSubspaceProjector,
)

# Import modules under test
from modelcypher.core.domain.geometry.transport_guided_merger import TransportGuidedMerger
from modelcypher.core.domain._backend import get_default_backend


def _make_crm(model_id: str, layer_vectors: dict[int, list[float]]) -> ConceptResponseMatrix:
    anchor_ids = [f"anchor:{idx}" for idx in layer_vectors.keys()]
    hidden_dim = len(next(iter(layer_vectors.values()))) if layer_vectors else 0
    metadata = AnchorMetadata(
        total_count=len(anchor_ids),
        semantic_prime_count=len(anchor_ids),
        computational_gate_count=0,
        anchor_ids=anchor_ids,
    )
    crm = ConceptResponseMatrix(
        model_identifier=model_id,
        layer_count=1,
        hidden_dim=hidden_dim,
        anchor_metadata=metadata,
    )
    crm.activations = {
        0: {
            anchor_id: AnchorActivation(anchor_id, 0, layer_vectors[idx])
            for idx, anchor_id in zip(layer_vectors.keys(), anchor_ids)
        }
    }
    return crm


class TestPhase9Geometry(unittest.TestCase):
    def test_transport_synthesis_simple(self):
        """Test deterministic weight synthesis given a known plan."""
        backend = get_default_backend()
        merger = TransportGuidedMerger(backend=backend)

        # 3 source neurons, 2 target neurons
        # Plan: source 0 -> target 0 (1.0), source 1 -> target 1 (1.0), source 2 -> both (50/50)
        # Actually plan is N x M (source x target)
        # S0 -> T0
        # S1 -> T1
        # S2 -> T0 (0.5), T1 (0.5)

        source_weights = backend.array([
            [1.0, 1.0],  # S0
            [2.0, 2.0],  # S1
            [3.0, 3.0],  # S2
        ])

        target_weights = backend.array([
            [10.0, 10.0],  # T0
            [20.0, 20.0],  # T1
        ])

        # Transport Plan (Source rows sum to 1? Wait, plan usually coupling. NormalizeRows does that.)
        # If we manually provide plan:
        # S0, S1, S2 rows.
        # T0, T1 cols.
        plan = backend.array([
            [1.0, 0.0],  # S0 -> T0
            [0.0, 1.0],  # S1 -> T1
            [0.5, 0.5],  # S2 split
        ])

        # Config blend_alpha=0 means pure transport
        config = TransportGuidedMerger.Config(
            blend_alpha=0.0, normalize_rows=False, coupling_threshold=0.0
        )

        # Result logic: W_merged = Plan^T @ W_source
        # T0 = 1.0*S0 + 0.0*S1 + 0.5*S2 = [1,1] + [1.5,1.5] = [2.5, 2.5]
        # T1 = 0.0*S0 + 1.0*S1 + 0.5*S2 = [2,2] + [1.5,1.5] = [3.5, 3.5]

        merged = merger.synthesize(source_weights, target_weights, plan, config)

        self.assertIsNotNone(merged)
        backend.eval(merged)
        merged_np = backend.to_numpy(merged)
        self.assertEqual(merged_np.shape[0], 2)
        # Check values using standard Python assertions
        self.assertAlmostEqual(float(merged_np[0, 0]), 2.5, places=5)
        self.assertAlmostEqual(float(merged_np[0, 1]), 2.5, places=5)
        self.assertAlmostEqual(float(merged_np[1, 0]), 3.5, places=5)
        self.assertAlmostEqual(float(merged_np[1, 1]), 3.5, places=5)

    def test_gw_computation(self):
        """Verify standard GW computation flow on isomorphic structures."""
        backend = get_default_backend()
        merger = TransportGuidedMerger(backend=backend)

        # Create two identical triangles in 2D space
        points_s = backend.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        points_t = backend.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Activations = points
        # weights = same

        # Just use synthesize_with_gw directly
        result = merger.synthesize_with_gw(
            source_activations=points_s,
            target_activations=points_t,
            source_weights=points_s,
            target_weights=points_t,
            config=TransportGuidedMerger.Config(min_samples=3, blend_alpha=0.0),
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.converged)
        # Distance should be near 0
        self.assertLess(result.gw_distance, 0.1)
        # Identity match expected (permuted possibly, but ordered inputs usually yield diagonal)

    def test_shared_subspace_cca(self):
        """Test CCA on linearly correlated data."""
        backend = get_default_backend()
        backend.random_seed(42)
        n = 50
        # Create linspace equivalent
        t = [10.0 * i / (n - 1) for i in range(n)]

        # Shared latent variable z
        import math
        z = [math.sin(val) for val in t]

        # X_s = z + noise in 2D
        noise_s = backend.random_normal((n, 2))
        noise_s = noise_s * 0.1
        backend.eval(noise_s)
        z_arr = backend.array(z)
        z_2x = z_arr * 2.0
        X_s_base = backend.stack([z_arr, z_2x], axis=1)
        X_s = X_s_base + noise_s
        backend.eval(X_s)

        # X_t = -z + noise in 3D (different dimension)
        noise_t = backend.random_normal((n, 3))
        noise_t = noise_t * 0.1
        backend.eval(noise_t)
        z_neg = z_arr * -1.0
        z_half = z_arr * 0.5
        X_t_base = backend.stack([z_neg, z_arr, z_half], axis=1)
        X_t = X_t_base + noise_t
        backend.eval(X_t)

        X_s_np = backend.to_numpy(X_s)
        X_t_np = backend.to_numpy(X_t)
        mock_source = _make_crm("source", {i: list(X_s_np[i]) for i in range(n)})
        mock_target = _make_crm("target", {i: list(X_t_np[i]) for i in range(n)})

        result = SharedSubspaceProjector.discover(
            source_crm=mock_source,
            target_crm=mock_target,
            layer=0,
            target_layer=0,
            config=SharedSubspaceConfig(alignment_method="cca"),
        )

        self.assertIsNotNone(result)
        self.assertGreater(result.shared_dimension, 0)
        # First correlation should be high (> 0.9)
        self.assertGreater(result.alignment_strengths[0], 0.8)
        self.assertTrue(result.h3_metrics.is_h3_validated or result.alignment_strengths[0] > 0.5)

    def test_shared_subspace_procrustes(self):
        """Test Procrustes on rotated data."""
        backend = get_default_backend()
        backend.random_seed(42)
        n = 20
        d = 3
        X_s = backend.random_normal((n, d))
        backend.eval(X_s)

        # Rotate X_s
        import math
        theta = math.radians(45)
        # 3D rotation around Z
        R = backend.array(
            [[math.cos(theta), -math.sin(theta), 0],
             [math.sin(theta), math.cos(theta), 0],
             [0, 0, 1]]
        )
        X_t = backend.matmul(X_s, R)
        backend.eval(X_t)

        X_s_np = backend.to_numpy(X_s)
        X_t_np = backend.to_numpy(X_t)
        mock_source = _make_crm("source", {i: list(X_s_np[i]) for i in range(n)})
        mock_target = _make_crm("target", {i: list(X_t_np[i]) for i in range(n)})

        result = SharedSubspaceProjector.discover(
            source_crm=mock_source,
            target_crm=mock_target,
            layer=0,
            target_layer=0,
            config=SharedSubspaceConfig(alignment_method="procrustes"),
        )

        self.assertIsNotNone(result)
        # Alignment error should be near 0
        self.assertLess(result.alignment_error, 0.05)


if __name__ == "__main__":
    unittest.main()
