# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for curriculum skill DAG contracts and generator output schema.

Covers:
  P3a — Five newly wired train_files paths are non-empty and follow expected naming.
  P3b — Generator functions produce correct schema (text, answer_start, logic_id).
  P3c — Arithmetic answer_start is a correct integer offset into the text string.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers to load scripts/ modules without installing them
# ---------------------------------------------------------------------------

def _load_script(name: str):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# P3a: DAG contract — five skill nodes have non-empty train_files
# ---------------------------------------------------------------------------

EXPECTED_SKILL_TRAIN_FILES = {
    "modus_tollens":            "data/training/modus_tollens_train.jsonl",
    "disjunctive_syllogism":    "data/training/disj_syllogism_train.jsonl",
    "universal_instantiation":  "data/training/universal_instantiation_train.jsonl",
    "arithmetic_add":           "data/training/arithmetic_add_train.jsonl",
    "arithmetic_divide":        "data/training/arithmetic_div_train.jsonl",
}


def test_five_skill_nodes_have_train_files():
    """Each of the five previously-empty skill nodes now references a train file."""
    from modelcypher.core.use_cases.curriculum.skill_dag import CURRICULUM_DAG

    for skill_name, expected_path in EXPECTED_SKILL_TRAIN_FILES.items():
        node = CURRICULUM_DAG.get(skill_name)
        assert node is not None, f"Skill '{skill_name}' not found in DAG"
        assert node.train_files, f"Skill '{skill_name}' still has empty train_files"
        assert expected_path in node.train_files, (
            f"Skill '{skill_name}' train_files={node.train_files!r} "
            f"does not contain '{expected_path}'"
        )


# ---------------------------------------------------------------------------
# P3b: Generator schema — logic files produce expected fields
# ---------------------------------------------------------------------------

def _minimal_pairs():
    """A minimal set of premise pairs for fast generator tests."""
    return [
        ("biology", "a cell divides", "two daughter cells form"),
        ("physics", "a force is applied", "acceleration occurs"),
    ]


def _minimal_ui_pairs():
    return [
        ("biology", "All cells contain DNA", "Neurons are cells", "Neurons contain DNA"),
    ]


def test_generate_mt_schema():
    gen = _load_script("generate_curriculum_data")
    samples = gen.generate_mt(_minimal_pairs(), seed=0)
    assert len(samples) > 0
    for s in samples:
        assert "text" in s and isinstance(s["text"], str)
        assert "answer_start" in s and isinstance(s["answer_start"], str)
        assert s["logic_id"] == "modus_tollens"
        assert s["answer_start"] in s["text"], (
            f"answer_start string not found in text:\n"
            f"  answer_start={s['answer_start']!r}\n"
            f"  text={s['text']!r}"
        )


def test_generate_ds_schema():
    gen = _load_script("generate_curriculum_data")
    samples = gen.generate_ds(_minimal_pairs(), seed=0)
    assert len(samples) > 0
    for s in samples:
        assert "text" in s and isinstance(s["text"], str)
        assert "answer_start" in s and isinstance(s["answer_start"], str)
        assert s["logic_id"] == "disjunctive_syllogism"
        assert s["answer_start"] in s["text"]


def test_generate_ui_schema():
    gen = _load_script("generate_curriculum_data")
    samples = gen.generate_ui(_minimal_ui_pairs(), seed=0)
    assert len(samples) > 0
    for s in samples:
        assert "text" in s and isinstance(s["text"], str)
        assert "answer_start" in s
        assert s["logic_id"] == "universal_instantiation"


# ---------------------------------------------------------------------------
# P3c: Arithmetic answer_start is a correct integer offset
# ---------------------------------------------------------------------------

def test_generate_add_answer_start_offset():
    """answer_start is a valid integer index into the text pointing at the answer."""
    gen = _load_script("generate_curriculum_data")
    samples = gen.generate_add(seed=0)
    assert len(samples) > 0
    for s in samples:
        assert isinstance(s["answer_start"], int), (
            f"answer_start should be int, got {type(s['answer_start'])}"
        )
        text = s["text"]
        offset = s["answer_start"]
        answer_in_text = text[offset:]
        # After the offset, the text should be the numeric answer (no leading question)
        assert answer_in_text.isdigit(), (
            f"text[answer_start:] should be a digit string, got {answer_in_text!r}"
        )
        assert s["logic_id"] == "arithmetic_add"


def test_generate_div_answer_start_offset():
    """answer_start correctly indexes into division examples."""
    gen = _load_script("generate_curriculum_data")
    samples = gen.generate_div(seed=0)
    assert len(samples) > 0
    for s in samples:
        assert isinstance(s["answer_start"], int)
        text = s["text"]
        offset = s["answer_start"]
        answer_in_text = text[offset:]
        assert answer_in_text.isdigit(), (
            f"text[answer_start:] should be a digit string, got {answer_in_text!r}"
        )
        assert s["logic_id"] == "arithmetic_divide"
