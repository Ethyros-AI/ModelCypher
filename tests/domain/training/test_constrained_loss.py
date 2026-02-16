# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for constrained geometric training infrastructure.

Tests the pure-Python components (constraint config, dataset loading,
pair groups) without requiring MLX or model loading.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.constraint_config import (
    ConstraintConfig,
    ConstraintState,
    derive_constraint_thresholds,
)
from modelcypher.core.domain.dataset_loading import (
    build_pair_groups,
    is_paired_dataset,
    load_jsonl_dataset,
)


# =============================================================================
# Constraint Config Tests
# =============================================================================


class TestConstraintConfig:
    def test_to_dict_roundtrip(self):
        config = ConstraintConfig(
            epsilon_inv=0.5,
            margin_sep=1.0,
            epsilon_tail=0.0,
            target_layers=[5, 8, 15],
            baseline_entropy={5: 2.1, 8: 1.9, 15: 1.5},
        )
        d = config.to_dict()
        assert d["epsilon_inv"] == 0.5
        assert d["margin_sep"] == 1.0
        assert d["epsilon_tail"] == 0.0
        assert d["target_layers"] == [5, 8, 15]
        assert d["baseline_entropy"] == {5: 2.1, 8: 1.9, 15: 1.5}

    def test_derive_thresholds_from_distances(self):
        inv_dists = [1.0, 2.0, 3.0]  # mean = 2.0, std = 1.0
        sep_dists = [4.0, 5.0, 6.0]  # mean = 5.0, std = 1.0
        layer_ent = {5: 2.1, 8: 1.9}

        config = derive_constraint_thresholds(inv_dists, sep_dists, layer_ent)

        # epsilon_inv = mean - 1*std = 2.0 - 1.0 = 1.0
        assert config.epsilon_inv == pytest.approx(1.0)
        # margin_sep = mean - 1*std = 5.0 - 1.0 = 4.0
        assert config.margin_sep == pytest.approx(4.0)
        # epsilon_tail = 0.05 * min(1.9, 2.1) = 0.05 * 1.9 = 0.095
        assert config.epsilon_tail == pytest.approx(0.095)
        assert config.target_layers == [5, 8]
        assert config.baseline_entropy == {5: 2.1, 8: 1.9}

    def test_derive_thresholds_insufficient_distances(self):
        """Fail fast when not enough measurements for reliable statistics."""
        with pytest.raises(ValueError, match="invariance distance"):
            derive_constraint_thresholds([1.0], [3.0, 4.0, 5.0], {5: 1.0})
        with pytest.raises(ValueError, match="separation distance"):
            derive_constraint_thresholds([1.0, 2.0, 3.0], [4.0], {5: 1.0})
        with pytest.raises(ValueError, match="No layer entropies"):
            derive_constraint_thresholds([1.0, 2.0, 3.0], [3.0, 4.0, 5.0], {})


class TestConstraintState:
    def test_dual_update_satisfied(self):
        """When all constraints are satisfied, multipliers should decrease."""
        config = ConstraintConfig(
            epsilon_inv=1.0, margin_sep=1.0, epsilon_tail=0.0,
            target_layers=[5], baseline_entropy={5: 2.0},
        )
        state = ConstraintState(mu_inv=1.0, mu_sep=1.0, mu_geo=1.0)

        # All constraints satisfied: C_inv < ε_inv, C_sep > m_sep, C_geo < ε_tail
        state.dual_update(
            C_inv=0.5,  # below ε_inv=1.0 → negative violation
            C_sep=2.0,  # above m_sep=1.0 → satisfied
            C_geo=0.0,  # zero
            config=config,
            alpha_dual=0.1,
        )

        # mu_inv should decrease: max(0, 1.0 + 0.1 * (0.5 - 1.0)) = max(0, 0.95) = 0.95
        assert state.mu_inv == pytest.approx(0.95)
        # mu_sep should decrease: max(0, 1.0 + 0.1 * (1.0 - 2.0)) = max(0, 0.9) = 0.9
        assert state.mu_sep == pytest.approx(0.9)
        # mu_geo stays: max(0, 1.0 + 0.1 * (0.0 - 0.0)) = 1.0
        assert state.mu_geo == pytest.approx(1.0)

    def test_dual_update_violated(self):
        """When constraints are violated, multipliers should increase."""
        config = ConstraintConfig(
            epsilon_inv=0.5, margin_sep=3.0, epsilon_tail=0.0,
            target_layers=[5], baseline_entropy={5: 2.0},
        )
        state = ConstraintState(mu_inv=1.0, mu_sep=1.0, mu_geo=1.0)

        state.dual_update(
            C_inv=1.0,  # above ε_inv=0.5 → violated
            C_sep=1.0,  # below m_sep=3.0 → violated
            C_geo=0.5,  # above ε_tail=0.0 → violated
            config=config,
            alpha_dual=0.1,
        )

        # mu_inv increases: 1.0 + 0.1 * (1.0 - 0.5) = 1.05
        assert state.mu_inv == pytest.approx(1.05)
        # mu_sep increases: 1.0 + 0.1 * (3.0 - 1.0) = 1.2
        assert state.mu_sep == pytest.approx(1.2)
        # mu_geo increases: 1.0 + 0.1 * (0.5 - 0.0) = 1.05
        assert state.mu_geo == pytest.approx(1.05)

    def test_multiplier_non_negative(self):
        """Multipliers should never go negative (projected dual ascent)."""
        config = ConstraintConfig(
            epsilon_inv=10.0, margin_sep=0.0, epsilon_tail=0.0,
            target_layers=[], baseline_entropy={},
        )
        state = ConstraintState(mu_inv=0.1, mu_sep=0.1, mu_geo=0.1)

        # Large negative violations should push mu toward 0 but not below
        state.dual_update(
            C_inv=0.0,
            C_sep=100.0,
            C_geo=0.0,
            config=config,
            alpha_dual=1.0,
        )

        assert state.mu_inv >= 0.0
        assert state.mu_sep >= 0.0
        assert state.mu_geo >= 0.0

    def test_frozen_multipliers_unchanged(self):
        """Frozen multipliers should not be updated by dual_update()."""
        config = ConstraintConfig(
            epsilon_inv=0.5, margin_sep=3.0, epsilon_tail=0.0,
            target_layers=[5], baseline_entropy={5: 2.0},
        )
        # Freeze mu_inv and mu_sep — only mu_geo should update
        state = ConstraintState(
            mu_inv=0.0, mu_sep=0.0, mu_geo=1.0,
            frozen=frozenset({"mu_inv", "mu_sep"}),
        )

        state.dual_update(
            C_inv=1.0,  # would increase mu_inv if not frozen
            C_sep=1.0,  # would increase mu_sep if not frozen
            C_geo=0.5,  # above ε_tail=0.0 → mu_geo increases
            config=config,
            alpha_dual=0.1,
        )

        # Frozen multipliers stay at initial values
        assert state.mu_inv == 0.0
        assert state.mu_sep == 0.0
        # Unfrozen multiplier updates normally
        assert state.mu_geo == pytest.approx(1.05)
        # Constraint values are always tracked regardless of frozen
        assert state.last_C_inv == 1.0
        assert state.last_C_sep == 1.0
        assert state.last_C_geo == 0.5

    def test_frozen_all_multipliers(self):
        """Freezing all multipliers disables dual ascent entirely."""
        config = ConstraintConfig(
            epsilon_inv=0.5, margin_sep=3.0, epsilon_tail=0.0,
            target_layers=[], baseline_entropy={},
        )
        state = ConstraintState(
            mu_inv=0.0, mu_sep=0.0, mu_geo=0.0,
            frozen=frozenset({"mu_inv", "mu_sep", "mu_geo"}),
        )

        state.dual_update(C_inv=10.0, C_sep=0.0, C_geo=10.0,
                          config=config, alpha_dual=1.0)

        assert state.mu_inv == 0.0
        assert state.mu_sep == 0.0
        assert state.mu_geo == 0.0

    def test_frozen_empty_is_default(self):
        """Empty frozen set means all multipliers update normally."""
        config = ConstraintConfig(
            epsilon_inv=0.5, margin_sep=3.0, epsilon_tail=0.0,
            target_layers=[], baseline_entropy={},
        )
        state = ConstraintState(mu_inv=1.0, mu_sep=1.0, mu_geo=1.0)
        assert state.frozen == frozenset()

        state.dual_update(C_inv=1.0, C_sep=1.0, C_geo=0.5,
                          config=config, alpha_dual=0.1)

        assert state.mu_inv == pytest.approx(1.05)
        assert state.mu_sep == pytest.approx(1.2)
        assert state.mu_geo == pytest.approx(1.05)

    def test_to_dict(self):
        state = ConstraintState(mu_inv=1.5, mu_sep=0.8, mu_geo=2.1)
        state.last_C_inv = 0.3
        state.last_C_sep = 1.2
        state.last_C_geo = 0.1
        state.last_ce_loss = 2.5

        d = state.to_dict()
        assert d["mu_inv"] == 1.5
        assert d["mu_sep"] == 0.8
        assert d["mu_geo"] == 2.1
        assert d["frozen"] == []
        assert d["C_inv"] == 0.3
        assert d["C_sep"] == 1.2
        assert d["C_geo"] == 0.1
        assert d["ce_loss"] == 2.5

    def test_to_dict_with_frozen(self):
        state = ConstraintState(
            mu_inv=0.0, mu_sep=0.0, mu_geo=1.0,
            frozen=frozenset({"mu_inv", "mu_sep"}),
        )
        d = state.to_dict()
        assert d["frozen"] == ["mu_inv", "mu_sep"]


# =============================================================================
# Dataset Loading Tests
# =============================================================================


class TestPairedDataDetection:
    def test_is_paired_empty(self):
        assert is_paired_dataset([]) is False

    def test_is_paired_regular(self):
        samples = [{"text": "hello"}, {"text": "world"}]
        assert is_paired_dataset(samples) is False

    def test_is_paired_true(self):
        samples = [
            {
                "text": "If A then B",
                "answer_start": "B",
                "logic_id": "mp",
                "template_id": "abstract",
            }
        ]
        assert is_paired_dataset(samples) is True

    def test_is_paired_partial_returns_false(self):
        """If some samples have paired fields but others don't, returns False."""
        samples = [
            {"text": "a", "answer_start": "a", "logic_id": "mp", "template_id": "abs"},
            {"text": "b"},  # missing paired fields
        ]
        assert is_paired_dataset(samples) is False


class TestBuildPairGroups:
    def test_groups(self):
        samples = [
            {"text": "a", "logic_id": "mp", "template_id": "abstract"},
            {"text": "b", "logic_id": "mp", "template_id": "concrete"},
            {"text": "c", "logic_id": "mt", "template_id": "abstract"},
            {"text": "d", "logic_id": "mt", "template_id": "concrete"},
        ]

        logic_g, tmpl_g = build_pair_groups(samples)

        assert set(logic_g["mp"]) == {0, 1}
        assert set(logic_g["mt"]) == {2, 3}
        assert set(tmpl_g["abstract"]) == {0, 2}
        assert set(tmpl_g["concrete"]) == {1, 3}

    def test_empty(self):
        logic_g, tmpl_g = build_pair_groups([])
        assert logic_g == {}
        assert tmpl_g == {}

    def test_missing_field_raises(self):
        """Samples missing logic_id or template_id should raise ValueError."""
        samples = [
            {"text": "a", "logic_id": "mp"},  # missing template_id
        ]
        with pytest.raises(ValueError, match="template_id.*MISSING"):
            build_pair_groups(samples)


# =============================================================================
# Generated Data Tests
# =============================================================================


class TestGeneratedData:
    """Test that the generated paired data has correct structure."""

    def test_train_data_loads(self):
        """Generated train data should load and be detected as paired."""
        import os
        path = "data/training/paired_reasoning_train.jsonl"
        if not os.path.exists(path):
            pytest.skip("Generated data not found")

        samples = load_jsonl_dataset(path)
        assert len(samples) > 0
        assert is_paired_dataset(samples)

    def test_train_data_has_required_fields(self):
        import os
        path = "data/training/paired_reasoning_train.jsonl"
        if not os.path.exists(path):
            pytest.skip("Generated data not found")

        samples = load_jsonl_dataset(path)
        for s in samples:
            assert "text" in s
            assert "answer_start" in s
            assert "logic_id" in s
            assert "template_id" in s
            # answer_start should appear in text
            assert s["answer_start"] in s["text"], (
                f"answer_start '{s['answer_start']}' not found in text"
            )

    def test_train_data_has_pairs(self):
        import os
        path = "data/training/paired_reasoning_train.jsonl"
        if not os.path.exists(path):
            pytest.skip("Generated data not found")

        samples = load_jsonl_dataset(path)
        logic_g, tmpl_g = build_pair_groups(samples)

        # Should have multiple logic groups (different reasoning patterns)
        assert len(logic_g) >= 3
        # Should have invariance pairs (same logic, multiple templates)
        has_invariance = any(len(v) >= 2 for v in logic_g.values())
        assert has_invariance, "No invariance pairs found"
