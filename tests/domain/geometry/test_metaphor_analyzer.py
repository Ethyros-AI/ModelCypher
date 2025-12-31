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

"""Metaphor analyzer tests (requires MLX)."""

import unittest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

import pytest

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain.geometry.manifold_stitcher import (
    ActivatedDimension,
    ActivationFingerprint,
    ModelFingerprints,
    ProbeSpace,
)
from modelcypher.core.domain.geometry.metaphor_convergence_analyzer import (
    MetaphorConvergenceAnalyzer,
    MetaphorFamily,
    MetaphorInvariantInventory,
)


class TestMetaphorConvergenceAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create dummy fingerprints for a source model
        self.source_fingerprints = self._create_dummy_fingerprints("source-model", offset=0.0)
        # Create dummy fingerprints for a target model (slightly shifted)
        self.target_fingerprints = self._create_dummy_fingerprints("target-model", offset=0.1)

    def _create_dummy_fingerprints(self, model_id: str, offset: float) -> ModelFingerprints:
        fingerprints = []
        # Use known probes from inventory
        probes = MetaphorInvariantInventory.ALL_PROBES

        for i, probe in enumerate(probes):
            # Simulate activation in layer 5 and 10
            activations = {
                5: [ActivatedDimension(index=i, activation=1.0 + offset)],  # higher activation
                10: [ActivatedDimension(index=i, activation=0.5 + offset)],
            }

            # prime_id must be prefixed with "metaphor_invariant:" for the analyzer to recognize it
            fp = ActivationFingerprint(
                prime_id=f"metaphor_invariant:{probe.id}",
                prime_text=probe.universal_concept,
                activated_dimensions=activations,
            )
            fingerprints.append(fp)

        return ModelFingerprints(
            model_id=model_id,
            layer_count=12,
            hidden_dim=768,
            probe_space=ProbeSpace.prelogits_hidden,
            probe_capture_key=None,
            fingerprints=fingerprints,
        )

    def test_analyzer_initialization(self):
        # Just ensure static analyze method exists and module is loaded
        self.assertTrue(hasattr(MetaphorConvergenceAnalyzer, "analyze"))

    def test_analysis_report_structure(self):
        # Run analysis
        report = MetaphorConvergenceAnalyzer.analyze(
            self.source_fingerprints,
            self.target_fingerprints,
            align_mode=MetaphorConvergenceAnalyzer.AlignMode.LAYER,
        )

        # Check basic report structure
        self.assertEqual(report.models.model_a, "source-model")
        self.assertEqual(report.models.model_b, "target-model")
        self.assertEqual(report.source_layer_count, 12)
        self.assertTrue(len(report.layers) > 0)

        # Check families are present (FUTILITY is the first probe family)
        self.assertIn(MetaphorFamily.FUTILITY.value, report.families)

        # Check convergence values (should be high since vectors are similar)
        # We used identity mapping shim in DimensionAlignmentBuilder,
        # so dimensions i <-> i align perfectly if values are close.
        # But we used sparse vectors.
        # Cosine similarity between [1.0] and [1.1] is 1.0.

        futility_family = report.families[MetaphorFamily.FUTILITY.value]
        # In layer 5, both have activation on dimension i.
        # Cosine should be 1.0

        # Check label for layer 5
        self.assertIn("5", futility_family.layers)
        self.assertAlmostEqual(futility_family.layers["5"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
