# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the PhoneBook evaluator (LKM validation protocol).

Tests pure Python functions only -- no GPU or model loading required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.lkm.evaluate_phonebook import (
    check_exact_match,
    compute_score,
    extract_phone,
)


class TestExtractPhoneFromGeneration:
    """extract_phone should find the first XXX-XXX-XXXX pattern in text."""

    def test_phone_with_leading_space_and_newline(self):
        assert extract_phone(" 555-123-4567\n") == "555-123-4567"

    def test_phone_embedded_in_sentence(self):
        assert extract_phone("The number is 555-123-4567.") == "555-123-4567"

    def test_no_phone_returns_none(self):
        assert extract_phone("I don't know") is None


class TestExactMatchCorrect:
    """check_exact_match returns True when generated text contains the target."""

    def test_exact_match_correct(self):
        assert check_exact_match("555-123-4567", " 555-123-4567") is True


class TestExactMatchWrong:
    """check_exact_match returns False when generated text has a different number."""

    def test_exact_match_wrong(self):
        assert check_exact_match("555-123-4567", "555-999-0000") is False


class TestExactMatchWithExtraction:
    """check_exact_match works when the answer has trailing text after the number."""

    def test_exact_match_with_extraction(self):
        assert check_exact_match(
            "555-123-4567", " 555-123-4567 is the number for Alice."
        ) is True


class TestScoreResults:
    """compute_score returns the fraction of exact matches."""

    def test_score_results(self):
        results = [
            {"exact_match": True},
            {"exact_match": True},
            {"exact_match": False},
            {"exact_match": True},
        ]
        assert compute_score(results) == 0.75


class TestScoreEmpty:
    """compute_score returns 0.0 for an empty list."""

    def test_score_empty(self):
        assert compute_score([]) == 0.0
