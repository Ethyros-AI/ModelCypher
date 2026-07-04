"""Unit tests for fact_dataset module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelcypher.experimental.baranov.fact_dataset import (
    fact_to_training_text,
    facts_to_training_samples,
    write_fact_training_jsonl,
)
from modelcypher.experimental.baranov.models import FactTriple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fact(
    subject: str = "Paris",
    relation: str = "capital_of",
    obj: str = "France",
    fact_id: str = "f1",
) -> FactTriple:
    return FactTriple(subject=subject, relation=relation, object=obj, fact_id=fact_id)


# ---------------------------------------------------------------------------
# fact_to_training_text
# ---------------------------------------------------------------------------


class TestFactToTrainingText:
    def test_basic_fact(self):
        fact = _make_fact()
        text = fact_to_training_text(fact)
        assert text == "Paris capital of France"

    def test_relation_underscore_normalization(self):
        fact = _make_fact(relation="chemical_formula")
        text = fact_to_training_text(fact)
        assert text == "Paris chemical formula France"

    def test_relation_already_clean(self):
        fact = _make_fact(relation="wrote")
        text = fact_to_training_text(fact)
        assert text == "Paris wrote France"

    def test_preserves_subject_and_object(self):
        fact = _make_fact(subject="H2O", relation="is_a", obj="molecule")
        text = fact_to_training_text(fact)
        assert "H2O" in text
        assert "molecule" in text

    def test_multi_word_object(self):
        fact = _make_fact(obj="Turing machine")
        text = fact_to_training_text(fact)
        assert text == "Paris capital of Turing machine"


# ---------------------------------------------------------------------------
# facts_to_training_samples
# ---------------------------------------------------------------------------


class TestFactsToTrainingSamples:
    def test_returns_list_of_dicts(self):
        facts = [_make_fact(fact_id="f1"), _make_fact(fact_id="f2")]
        samples = facts_to_training_samples(facts)
        assert len(samples) == 2
        assert all(isinstance(s, dict) for s in samples)
        assert all("text" in s for s in samples)

    def test_text_format(self):
        samples = facts_to_training_samples([_make_fact()])
        assert samples[0]["text"] == "Paris capital of France"

    def test_empty_list(self):
        samples = facts_to_training_samples([])
        assert samples == []

    def test_only_text_key(self):
        """Samples should only have a 'text' key (DatasetTrainingService format)."""
        samples = facts_to_training_samples([_make_fact()])
        assert set(samples[0].keys()) == {"text"}


# ---------------------------------------------------------------------------
# write_fact_training_jsonl
# ---------------------------------------------------------------------------


class TestWriteFactTrainingJsonl:
    def test_writes_jsonl(self, tmp_path: Path):
        facts = [
            _make_fact(subject="Paris", obj="France", fact_id="f1"),
            _make_fact(subject="Berlin", obj="Germany", fact_id="f2"),
        ]
        output = tmp_path / "facts.jsonl"
        result = write_fact_training_jsonl(facts, output)

        assert result == output
        assert output.exists()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "text" in data

    def test_content_matches_training_text(self, tmp_path: Path):
        fact = _make_fact()
        output = tmp_path / "facts.jsonl"
        write_fact_training_jsonl([fact], output)

        line = output.read_text().strip()
        data = json.loads(line)
        assert data["text"] == fact_to_training_text(fact)

    def test_refuses_overwrite_by_default(self, tmp_path: Path):
        output = tmp_path / "facts.jsonl"
        output.write_text("existing content")

        with pytest.raises(FileExistsError, match="already exists"):
            write_fact_training_jsonl([_make_fact()], output)

    def test_overwrite_flag(self, tmp_path: Path):
        output = tmp_path / "facts.jsonl"
        output.write_text("old")
        write_fact_training_jsonl([_make_fact()], output, overwrite=True)

        data = json.loads(output.read_text().strip())
        assert "text" in data

    def test_creates_parent_dirs(self, tmp_path: Path):
        output = tmp_path / "a" / "b" / "c" / "facts.jsonl"
        write_fact_training_jsonl([_make_fact()], output)
        assert output.exists()

    def test_utf8_encoding(self, tmp_path: Path):
        fact = _make_fact(subject="München", obj="Bayerñ")
        output = tmp_path / "facts.jsonl"
        write_fact_training_jsonl([fact], output)

        data = json.loads(output.read_text(encoding="utf-8").strip())
        assert "München" in data["text"]
        assert "Bayerñ" in data["text"]

    def test_trailing_newline(self, tmp_path: Path):
        output = tmp_path / "facts.jsonl"
        write_fact_training_jsonl([_make_fact()], output)
        content = output.read_text()
        assert content.endswith("\n")
