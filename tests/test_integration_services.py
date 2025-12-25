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
Integration tests for geometry services.

Tests end-to-end functionality of geometry analysis pipelines.
Uses actual module APIs discovered from existing tests.
"""

from __future__ import annotations

import numpy as np
import pytest


# Test that all key geometry modules can be imported and work together
class TestGeometryIntegration:
    """Integration tests for geometry module interoperability."""

    def test_vector_math_cosine_similarity_chain(self):
        """VectorMath can be used in a processing chain."""
        from modelcypher.core.domain.geometry.vector_math import VectorMath

        # Create test vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [1.0, 1.0, 0.0]

        # Normalize v3
        v3_norm = VectorMath.l2_normalized(v3)

        # Compute similarities
        sim_v1_v2 = VectorMath.cosine_similarity(v1, v2)
        sim_v1_v3 = VectorMath.cosine_similarity(v1, v3_norm)

        assert sim_v1_v2 == pytest.approx(0.0)
        assert sim_v1_v3 == pytest.approx(1.0 / np.sqrt(2))

    def test_dora_decomposition_import(self):
        """DoRA decomposition module can be imported."""
        from modelcypher.core.domain.geometry.dora_decomposition import (
            DoRADecomposition,
        )

        # Just verify imports work
        assert DoRADecomposition is not None

    def test_dare_sparsity_import(self):
        """DARE sparsity module can be imported."""
        from modelcypher.core.domain.geometry.dare_sparsity import (
            DARESparsityAnalyzer,
        )

        # Verify import works
        assert DARESparsityAnalyzer is not None

    def test_intrinsic_dimension_import(self):
        """Intrinsic dimension estimator can be imported."""
        from modelcypher.core.domain.geometry.intrinsic_dimension_estimator import (
            IntrinsicDimensionEstimator,
        )

        assert IntrinsicDimensionEstimator is not None

    def test_path_geometry_import(self):
        """PathGeometry types can be imported."""
        from modelcypher.core.domain.geometry.path_geometry import (
            PathNode,
        )

        # Verify import works
        assert PathNode is not None

    def test_manifold_clusterer_import(self):
        """ManifoldClusterer can be imported."""
        from modelcypher.core.domain.geometry.manifold_clusterer import ManifoldClusterer

        assert ManifoldClusterer is not None


class TestEntropyIntegration:
    """Integration tests for entropy analysis pipeline."""

    def test_entropy_tracker_import(self):
        """EntropyTracker can be imported and created."""
        from modelcypher.core.domain.entropy.entropy_tracker import (
            EntropyTracker,
        )

        tracker = EntropyTracker()
        assert tracker is not None

    def test_logit_entropy_calculator_import(self):
        """LogitEntropyCalculator can be imported."""
        from modelcypher.core.domain.entropy.logit_entropy_calculator import (
            LogitEntropyCalculator,
        )

        calculator = LogitEntropyCalculator()
        assert calculator is not None

    def test_entropy_window_import(self):
        """EntropyWindow can be imported and created."""
        from modelcypher.core.domain.entropy.entropy_window import (
            EntropyWindow,
            EntropyWindowConfig,
        )

        config = EntropyWindowConfig(window_size=5)
        window = EntropyWindow(config)

        assert window is not None
        assert window.config.window_size == 5


class TestSafetyIntegration:
    """Integration tests for safety analysis pipeline."""

    def test_adapter_safety_probe_import(self):
        """AdapterSafetyProbe can be imported."""
        from modelcypher.core.domain.safety.adapter_safety_probe import (
            AdapterSafetyProbe,
        )

        assert AdapterSafetyProbe is not None

    def test_capability_guard_import(self):
        """CapabilityGuard can be imported."""
        from modelcypher.core.domain.safety.capability_guard import (
            CapabilityGuard,
        )

        assert CapabilityGuard is not None

    def test_safety_models_import(self):
        """Safety models can be imported."""
        from modelcypher.core.domain.safety.safety_models import (
            SafetyStatus,
        )

        assert SafetyStatus is not None


class TestAgentsIntegration:
    """Integration tests for agent observability."""

    def test_agent_trace_import(self):
        """AgentTrace can be imported."""
        from modelcypher.core.domain.agents.agent_trace import (
            TraceKind,
            TraceStatus,
        )

        # Check available trace kinds
        assert len(list(TraceKind)) > 0
        assert len(list(TraceStatus)) > 0

    def test_agent_trace_analytics_empty(self):
        """AgentTraceAnalytics can create empty analytics."""
        from modelcypher.core.domain.agents.agent_trace_analytics import (
            AgentTraceAnalytics,
        )

        analytics = AgentTraceAnalytics.empty(requested_count=10)

        assert analytics.requested_trace_count == 10
        assert analytics.loaded_trace_count == 0

    def test_semantic_prime_atlas_import(self):
        """SemanticPrimeAtlas can be imported."""
        from modelcypher.core.domain.agents.semantic_prime_atlas import (
            AtlasConfiguration,
            SemanticPrimeAtlas,
        )

        config = AtlasConfiguration()
        atlas = SemanticPrimeAtlas(configuration=config)

        # Get inventory
        primes = atlas.inventory
        assert len(primes) > 0


class TestCrossModuleIntegration:
    """Tests that verify modules work together correctly."""

    def test_vector_math_operations(self):
        """Vector math operations work correctly."""
        from modelcypher.core.domain.geometry.vector_math import VectorMath

        # Create activation vectors and compute similarities
        activations = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ]

        similarities = []
        for i in range(1, len(activations)):
            sim = VectorMath.cosine_similarity(activations[0], activations[i])
            similarities.append(sim)

        assert len(similarities) == 3
        # Similarity should decrease as vectors diverge
        assert similarities[0] > similarities[2]

    def test_safety_agents_action_validation(self):
        """Safety and agents modules work together."""
        from modelcypher.core.domain.agents.agent_action import (
            ActionKind,
            ActionResponse,
            AgentActionEnvelope,
        )
        from modelcypher.core.domain.agents.agent_action_validator import (
            AgentActionValidator,
        )

        # Create an action envelope
        envelope = AgentActionEnvelope.create(
            kind=ActionKind.RESPOND,
            response=ActionResponse(text="Hello, how can I help?"),
        )

        # Validate it
        result = AgentActionValidator.validate(envelope)

        assert result.is_valid

    def test_all_domain_packages_importable(self):
        """All domain packages can be imported."""
        from modelcypher.core.domain import (
            agents,
            dynamics,
            entropy,
            geometry,
            merging,
            safety,
            training,
        )

        assert geometry is not None
        assert entropy is not None
        assert safety is not None
        assert agents is not None
        assert training is not None
        assert dynamics is not None
        assert merging is not None
