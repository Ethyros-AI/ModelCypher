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

"""Unit tests for knowledge signature dataclasses.

Design Principles Tested:
- Raw metrics only (no composite scores)
- Frozen immutability
- NaN values preserved in serialization
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.knowledge_signature import (
    InvariantSignature,
    KnowledgeSignature,
)


class TestKnowledgeSignature:
    """Tests for KnowledgeSignature dataclass."""

    def test_frozen_dataclass(self):
        """KnowledgeSignature should be immutable."""
        sig = KnowledgeSignature(
            statement="2 + 2 = 4",
            counterfactual="2 + 2 = 5",
            counterfactual_sensitivity=0.25,
            effective_rank=5.0,
            spectral_entropy=1.5,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            sig.counterfactual_sensitivity = 0.5

    def test_as_dict(self):
        """as_dict should include all fields."""
        sig = KnowledgeSignature(
            statement="test statement",
            counterfactual="test counterfactual",
            counterfactual_sensitivity=0.3,
            effective_rank=8.0,
            spectral_entropy=2.0,
        )

        d = sig.as_dict()

        assert d["statement"] == "test statement"
        assert d["counterfactual"] == "test counterfactual"
        assert d["counterfactual_sensitivity"] == 0.3
        assert d["effective_rank"] == 8.0
        assert d["spectral_entropy"] == 2.0
        # No composite score - raw metrics only
        assert len(d) == 5

    def test_nan_values_preserved(self):
        """NaN values should be preserved in dataclass and as_dict."""
        sig = KnowledgeSignature(
            statement="test",
            counterfactual="test cf",
            counterfactual_sensitivity=float("nan"),
            effective_rank=5.0,
            spectral_entropy=1.5,
        )

        assert math.isnan(sig.counterfactual_sensitivity)
        assert math.isnan(sig.as_dict()["counterfactual_sensitivity"])


class TestInvariantSignature:
    """Tests for InvariantSignature dataclass."""

    def test_frozen_dataclass(self):
        """InvariantSignature should be immutable."""
        sig = InvariantSignature(
            prompt="What is 2 + 2?",
            kurtosis=2.0,
            spectral_entropy=1.5,
            effective_rank=5.0,
            layer_consistency=0.9,
            repetition_consistency=0.95,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            sig.kurtosis = 3.0

    def test_as_dict(self):
        """as_dict should include all fields."""
        sig = InvariantSignature(
            prompt="test prompt",
            kurtosis=2.5,
            spectral_entropy=1.5,
            effective_rank=5.0,
            layer_consistency=0.8,
            repetition_consistency=0.9,
        )

        d = sig.as_dict()

        assert d["prompt"] == "test prompt"
        assert d["kurtosis"] == 2.5
        assert d["spectral_entropy"] == 1.5
        assert d["effective_rank"] == 5.0
        assert d["layer_consistency"] == 0.8
        assert d["repetition_consistency"] == 0.9
        # No composite score - raw metrics only
        assert len(d) == 6

    def test_nan_values_preserved(self):
        """NaN values should be preserved for degenerate cases."""
        sig = InvariantSignature(
            prompt="test",
            kurtosis=float("nan"),  # e.g., zero variance
            spectral_entropy=1.5,
            effective_rank=5.0,
            layer_consistency=float("nan"),  # e.g., single layer
            repetition_consistency=float("nan"),  # e.g., single repetition
        )

        assert math.isnan(sig.kurtosis)
        assert math.isnan(sig.layer_consistency)
        assert math.isnan(sig.repetition_consistency)

        d = sig.as_dict()
        assert math.isnan(d["kurtosis"])
        assert math.isnan(d["layer_consistency"])
        assert math.isnan(d["repetition_consistency"])
