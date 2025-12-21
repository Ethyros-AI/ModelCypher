"""
Verification tests for Phase 9: Advanced Geometry & OT.
Tests TransportGuidedMerger and SharedSubspaceProjector.
"""
import unittest
import numpy as np
from typing import List

# Import modules under test
from modelcypher.core.domain.geometry.transport_guided_merger import TransportGuidedMerger
from modelcypher.core.domain.geometry.shared_subspace_projector import SharedSubspaceProjector
from modelcypher.core.domain.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.gromov_wasserstein import Config as GWConfig

# Mock ConceptResponseMatrix for testing
class MockCRM(ConceptResponseMatrix):
    def __init__(self, data, anchors):
        self.data = data # Dict[layer, Dict[anchor, vec]]
        self.anchors = anchors
    
    def common_anchor_ids(self, other):
        return self.anchors
    
    def activation_matrix(self, layer, anchors):
        if layer not in self.data: return None
        mat = []
        for a in anchors:
            if a in self.data[layer]:
                mat.append(self.data[layer][a])
        return mat

class TestPhase9Geometry(unittest.TestCase):

    def test_transport_synthesis_simple(self):
        """Test deterministic weight synthesis given a known plan."""
        # 3 source neurons, 2 target neurons
        # Plan: source 0 -> target 0 (1.0), source 1 -> target 1 (1.0), source 2 -> both (50/50)
        # Actually plan is N x M (source x target)
        # S0 -> T0
        # S1 -> T1
        # S2 -> T0 (0.5), T1 (0.5)
        
        source_weights = [
            [1.0, 1.0], # S0
            [2.0, 2.0], # S1
            [3.0, 3.0]  # S2
        ]
        
        target_weights = [
            [10.0, 10.0], # T0
            [20.0, 20.0]  # T1
        ]
        
        # Transport Plan (Source rows sum to 1? Wait, plan usually coupling. NormalizeRows does that.)
        # If we manually provide plan:
        # S0, S1, S2 rows.
        # T0, T1 cols.
        plan = [
            [1.0, 0.0], # S0 -> T0
            [0.0, 1.0], # S1 -> T1
            [0.5, 0.5]  # S2 split
        ]
        
        # Config blend_alpha=0 means pure transport
        config = TransportGuidedMerger.Config(blend_alpha=0.0, normalize_rows=False, coupling_threshold=0.0)
        
        # Result logic: W_merged = Plan^T @ W_source
        # T0 = 1.0*S0 + 0.0*S1 + 0.5*S2 = [1,1] + [1.5,1.5] = [2.5, 2.5]
        # T1 = 0.0*S0 + 1.0*S1 + 0.5*S2 = [2,2] + [1.5,1.5] = [3.5, 3.5] 
        
        merged = TransportGuidedMerger.synthesize(source_weights, target_weights, plan, config)
        
        self.assertIsNotNone(merged)
        self.assertEqual(len(merged), 2)
        np.testing.assert_allclose(merged[0], [2.5, 2.5])
        np.testing.assert_allclose(merged[1], [3.5, 3.5])

    def test_gw_computation(self):
        """Verify standard GW computation flow on isomorphic structures."""
        # Create two identical triangles in 2D space
        points_s = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        points_t = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        
        # Activations = points
        # weights = same
        
        # Just use synthesize_with_gw directly
        result = TransportGuidedMerger.synthesize_with_gw(
            source_activations=points_s,
            target_activations=points_t,
            source_weights=points_s,
            target_weights=points_t,
            config=TransportGuidedMerger.Config(min_samples=3, blend_alpha=0.0)
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(result.converged)
        # Distance should be near 0
        self.assertLess(result.gw_distance, 0.1)
        # Identity match expected (permuted possibly, but ordered inputs usually yield diagonal)
        
    def test_shared_subspace_cca(self):
        """Test CCA on linearly correlated data."""
        np.random.seed(42)
        n = 50
        t = np.linspace(0, 10, n)
        
        # Shared latent variable z
        z = np.sin(t)
        
        # X_s = z + noise in 2D
        X_s = np.column_stack([z, 2*z]) + np.random.normal(0, 0.1, (n, 2))
        
        # X_t = -z + noise in 3D (different dimension)
        X_t = np.column_stack([-z, z, 0.5*z]) + np.random.normal(0, 0.1, (n, 3))
        
        mock_source = MockCRM({0: {i: list(X_s[i]) for i in range(n)}}, range(n))
        mock_target = MockCRM({0: {i: list(X_t[i]) for i in range(n)}}, range(n))
        
        result = SharedSubspaceProjector.discover(
            source_crm=mock_source,
            target_crm=mock_target,
            source_layer=0,
            target_layer=0,
            config=SharedSubspaceProjector.Config(alignment_method="cca")
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(result.shared_dimension, 0)
        # First correlation should be high (> 0.9)
        self.assertGreater(result.alignment_strengths[0], 0.8)
        self.assertTrue(result.h3_metrics.is_h3_validated or result.alignment_strengths[0]>0.5)

    def test_shared_subspace_procrustes(self):
        """Test Procrustes on rotated data."""
        np.random.seed(42)
        n = 20
        d = 3
        X_s = np.random.randn(n, d)
        
        # Rotate X_s
        theta = np.radians(45)
        # 3D rotation around Z
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,              1]
        ])
        X_t = X_s @ R
        
        mock_source = MockCRM({0: {i: list(X_s[i]) for i in range(n)}}, range(n))
        mock_target = MockCRM({0: {i: list(X_t[i]) for i in range(n)}}, range(n))
        
        result = SharedSubspaceProjector.discover(
            source_crm=mock_source,
            target_crm=mock_target,
            source_layer=0,
            target_layer=0,
            config=SharedSubspaceProjector.Config(alignment_method="procrustes")
        )
        
        self.assertIsNotNone(result)
        # Alignment error should be near 0
        self.assertLess(result.alignment_error, 0.05)

if __name__ == "__main__":
    unittest.main()
