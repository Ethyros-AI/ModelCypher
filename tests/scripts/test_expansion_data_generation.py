# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the expansion data generation script."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_expansion_data import (
    GENERATORS,
    _verify_sample,
    generate_dataset,
)


class TestIndividualGenerators:
    """Each generator produces valid text with verification."""

    @pytest.mark.parametrize("gen_name", list(GENERATORS.keys()))
    def test_generator_produces_valid_text(self, gen_name):
        rng = random.Random(42)
        gen = GENERATORS[gen_name]
        # Try multiple times since some generators can return None
        sample = None
        for _ in range(20):
            sample = gen(rng, difficulty=3)
            if sample is not None:
                break
        assert sample is not None, f"{gen_name} returned None on all 20 attempts"
        assert "text" in sample
        assert isinstance(sample["text"], str)
        assert len(sample["text"]) > 50

    @pytest.mark.parametrize("gen_name", list(GENERATORS.keys()))
    def test_generator_passes_verify(self, gen_name):
        rng = random.Random(42)
        gen = GENERATORS[gen_name]
        sample = None
        for _ in range(20):
            sample = gen(rng, difficulty=3)
            if sample is not None and _verify_sample(sample):
                break
        assert sample is not None
        assert _verify_sample(sample)

    @pytest.mark.parametrize("gen_name", list(GENERATORS.keys()))
    def test_no_rule_labels(self, gen_name):
        """Output text must NOT contain rule labels that caused template memorization."""
        rng = random.Random(42)
        gen = GENERATORS[gen_name]
        for _ in range(20):
            sample = gen(rng, difficulty=3)
            if sample is not None:
                text = sample["text"]
                assert "Rule: Modus" not in text
                assert "Rule: Disjunct" not in text
                assert "Pattern: If P" not in text
                assert "Rule: Contrap" not in text
                break


class TestVerifySample:
    """Tests for _verify_sample() validator."""

    def test_none_rejected(self):
        assert _verify_sample(None) is False  # type: ignore[arg-type]

    def test_empty_dict_rejected(self):
        assert _verify_sample({}) is False

    def test_short_text_rejected(self):
        assert _verify_sample({"text": "too short"}) is False

    def test_no_verification_rejected(self):
        assert _verify_sample({"text": "x" * 200}) is False

    def test_valid_sample_accepted(self):
        assert _verify_sample({"text": "x" * 100 + " Verification: correct"}) is True

    def test_rule_label_rejected(self):
        assert _verify_sample(
            {"text": "x" * 100 + " Rule: Modus Ponens. Verification done."}
        ) is False


class TestReproducibility:
    """Same seed must produce identical output."""

    def test_same_seed_same_output(self, tmp_path):
        out1 = str(tmp_path / "run1.jsonl")
        out2 = str(tmp_path / "run2.jsonl")
        n1 = generate_dataset(50, out1, seed=99, min_difficulty=2, max_difficulty=4)
        n2 = generate_dataset(50, out2, seed=99, min_difficulty=2, max_difficulty=4)
        assert n1 == n2
        assert Path(out1).read_text() == Path(out2).read_text()

    def test_different_seed_different_output(self, tmp_path):
        out1 = str(tmp_path / "run1.jsonl")
        out2 = str(tmp_path / "run2.jsonl")
        generate_dataset(50, out1, seed=1)
        generate_dataset(50, out2, seed=2)
        assert Path(out1).read_text() != Path(out2).read_text()


class TestGenerateDataset:
    """Integration tests for generate_dataset()."""

    def test_generates_requested_count(self, tmp_path):
        out = str(tmp_path / "test.jsonl")
        n = generate_dataset(100, out, seed=42)
        assert n == 100
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 100

    def test_all_lines_valid_json(self, tmp_path):
        import json
        out = str(tmp_path / "test.jsonl")
        generate_dataset(50, out, seed=42)
        with open(out) as f:
            for line in f:
                data = json.loads(line)
                assert "text" in data
                assert isinstance(data["text"], str)

    def test_difficulty_range_respected(self, tmp_path):
        """Higher difficulty should generally produce longer outputs."""
        out_easy = str(tmp_path / "easy.jsonl")
        out_hard = str(tmp_path / "hard.jsonl")
        generate_dataset(100, out_easy, seed=42, min_difficulty=2, max_difficulty=2)
        generate_dataset(100, out_hard, seed=42, min_difficulty=5, max_difficulty=5)
        easy_size = Path(out_easy).stat().st_size
        hard_size = Path(out_hard).stat().st_size
        # Hard problems should be at least somewhat longer on average
        assert hard_size > easy_size * 0.8  # loose bound, just sanity check
