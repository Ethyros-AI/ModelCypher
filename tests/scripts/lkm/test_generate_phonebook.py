# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the PhoneBook dataset generator (LKM validation protocol)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.lkm.generate_phonebook import (
    format_eval_prompt,
    format_qa_text,
    generate_pairs,
    slice_pairs_by_tokens,
)


class MockTokenizer:
    """Mock tokenizer: encode returns list(range(len(text))).

    Each character maps to one token, so token count == character count.
    """

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class TestGeneratePairsDeterministic:
    """Same seed must produce identical output."""

    def test_generate_pairs_deterministic(self):
        pairs_a = generate_pairs(100, seed=42)
        pairs_b = generate_pairs(100, seed=42)
        assert pairs_a == pairs_b

    def test_different_seeds_differ(self):
        pairs_a = generate_pairs(100, seed=42)
        pairs_b = generate_pairs(100, seed=99)
        assert pairs_a != pairs_b


class TestGeneratePairsUniqueNames:
    """All generated names must be unique."""

    def test_generate_pairs_unique_names(self):
        pairs = generate_pairs(800, seed=42)
        names = [name for name, _phone in pairs]
        assert len(names) == len(set(names)), "Duplicate names found"

    def test_generate_800_pairs(self):
        pairs = generate_pairs(800, seed=42)
        assert len(pairs) == 800


class TestPhoneFormat:
    """Phone numbers must match XXX-XXX-XXXX format."""

    def test_phone_format(self):
        pairs = generate_pairs(100, seed=42)
        pattern = re.compile(r"^\d{3}-\d{3}-\d{4}$")
        for name, phone in pairs:
            assert pattern.match(phone), f"Bad phone format for {name}: {phone}"


class TestFormatQaPair:
    """Exact text format for QA training pairs."""

    def test_format_qa_pair(self):
        text = format_qa_text("Alice Smith", "555-123-4567")
        expected = (
            "Question: What is the phone number of Alice Smith? "
            "Answer: 555-123-4567"
        )
        assert text == expected


class TestFormatEvalPrompt:
    """Exact prompt format for evaluation."""

    def test_format_eval_prompt(self):
        prompt = format_eval_prompt("Alice Smith")
        expected = "Question: What is the phone number of Alice Smith? Answer:"
        assert prompt == expected


class TestSliceByTokensPrefixProperty:
    """Smaller slices must be strict prefixes of larger slices."""

    def test_slice_by_tokens_prefix_property(self):
        pairs = generate_pairs(200, seed=42)
        tokenizer = MockTokenizer()
        targets = [500, 1000, 2000, 5000]
        slices = slice_pairs_by_tokens(pairs, targets, tokenizer)

        # Should return one dict per target
        assert len(slices) == len(targets)

        # Each slice should have the required keys
        for s in slices:
            assert "target_tokens" in s
            assert "actual_tokens" in s
            assert "n_pairs" in s
            assert "pairs" in s

        # Prefix property: each smaller slice's pairs must be a prefix of
        # every larger slice's pairs
        for i in range(len(slices) - 1):
            smaller = slices[i]["pairs"]
            larger = slices[i + 1]["pairs"]
            assert len(smaller) <= len(larger), (
                f"Slice {slices[i]['target_tokens']} has more pairs "
                f"({len(smaller)}) than slice {slices[i + 1]['target_tokens']} "
                f"({len(larger)})"
            )
            assert larger[: len(smaller)] == smaller, (
                f"Slice {slices[i]['target_tokens']} is not a prefix of "
                f"slice {slices[i + 1]['target_tokens']}"
            )

    def test_n_pairs_matches_pairs_length(self):
        pairs = generate_pairs(50, seed=7)
        tokenizer = MockTokenizer()
        slices = slice_pairs_by_tokens(pairs, [200, 500], tokenizer)
        for s in slices:
            assert s["n_pairs"] == len(s["pairs"])

    def test_actual_tokens_does_not_exceed_target(self):
        pairs = generate_pairs(100, seed=7)
        tokenizer = MockTokenizer()
        slices = slice_pairs_by_tokens(pairs, [500, 1000, 3000], tokenizer)
        for s in slices:
            assert s["actual_tokens"] <= s["target_tokens"], (
                f"actual_tokens ({s['actual_tokens']}) exceeds "
                f"target_tokens ({s['target_tokens']})"
            )
