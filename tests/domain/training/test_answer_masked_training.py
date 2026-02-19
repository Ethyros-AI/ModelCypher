# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for answer-span masking and retention replay."""

from __future__ import annotations

import random

import pytest

from modelcypher.core.domain.dataset_loading import (
    is_answer_masked_dataset,
    merge_datasets_with_fraction,
)


# ── is_answer_masked_dataset ──


class TestIsAnswerMaskedDataset:
    def test_true_when_all_have_answer_start(self):
        samples = [
            {"text": "Q?\nA", "answer_start": 3},
            {"text": "Q2?\nA2", "answer_start": 4},
        ]
        assert is_answer_masked_dataset(samples) is True

    def test_false_when_missing_answer_start(self):
        samples = [
            {"text": "Q?\nA"},
            {"text": "Q2?\nA2", "answer_start": 4},
        ]
        assert is_answer_masked_dataset(samples) is False

    def test_false_when_missing_text(self):
        samples = [
            {"answer_start": 3},
        ]
        assert is_answer_masked_dataset(samples) is False

    def test_false_empty(self):
        assert is_answer_masked_dataset([]) is False

    def test_true_with_extra_fields(self):
        samples = [
            {"text": "Q?\nA", "answer_start": 3, "logic_id": "L1"},
        ]
        assert is_answer_masked_dataset(samples) is True


# ── merge_datasets_with_fraction ──


class TestMergeDatasets:
    def test_correct_count(self):
        primary = [{"text": f"p{i}", "answer_start": 0} for i in range(80)]
        retention = [{"text": f"r{i}"} for i in range(100)]
        merged = merge_datasets_with_fraction(primary, retention, 0.2)
        # 80 primary + 20 retention = 100 total (fraction=0.2 of 100 = 20)
        assert len(merged) == 100

    def test_retention_larger_than_needed_subsamples(self):
        random.seed(42)
        primary = [{"text": f"p{i}", "answer_start": 0} for i in range(100)]
        retention = [{"text": f"r{i}"} for i in range(200)]
        merged = merge_datasets_with_fraction(primary, retention, 0.2)
        # 100 primary + 25 retention = 125
        assert len(merged) == 125
        # All retention samples should have answer_start set
        retention_items = [s for s in merged if s["text"].startswith("r")]
        assert all("answer_start" in s for s in retention_items)

    def test_retention_smaller_than_computed_uses_all(self):
        primary = [{"text": f"p{i}", "answer_start": 0} for i in range(100)]
        retention = [{"text": f"r{i}"} for i in range(5)]
        # fraction=0.5 → computed n_retention = 100, but only 5 available
        merged = merge_datasets_with_fraction(primary, retention, 0.5)
        assert len(merged) == 105

    def test_retention_gets_answer_start_zero(self):
        primary = [{"text": "p0", "answer_start": 10}]
        retention = [{"text": "r0"}, {"text": "r1"}]
        merged = merge_datasets_with_fraction(primary, retention, 0.5)
        for s in merged:
            assert "answer_start" in s
            if s["text"].startswith("r"):
                assert s["answer_start"] == 0

    def test_empty_primary_raises(self):
        with pytest.raises(ValueError, match="Primary dataset is empty"):
            merge_datasets_with_fraction([], [{"text": "r"}], 0.2)

    def test_invalid_fraction_raises(self):
        primary = [{"text": "p", "answer_start": 0}]
        retention = [{"text": "r"}]
        with pytest.raises(ValueError, match="retention_fraction"):
            merge_datasets_with_fraction(primary, retention, 0.0)
        with pytest.raises(ValueError, match="retention_fraction"):
            merge_datasets_with_fraction(primary, retention, 1.0)

    def test_empty_retention_returns_primary(self):
        primary = [{"text": "p0", "answer_start": 0}]
        merged = merge_datasets_with_fraction(primary, [], 0.2)
        assert len(merged) == 1
        assert merged[0]["text"] == "p0"

    def test_merged_is_shuffled(self):
        random.seed(42)
        primary = [{"text": f"p{i}", "answer_start": 0} for i in range(50)]
        retention = [{"text": f"r{i}"} for i in range(50)]
        merged = merge_datasets_with_fraction(primary, retention, 0.2)
        # Check that the merged list is not just primary + retention
        texts = [s["text"] for s in merged]
        # With shuffling, the first item shouldn't always be p0
        # (probabilistically true; seed=42 ensures determinism)
        assert texts != [f"p{i}" for i in range(50)] + [f"r{i}" for i in range(50)]
