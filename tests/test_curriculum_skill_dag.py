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
    # arithmetic_add was decomposed into three formal sub-skills:
    "single_digit_add":         "data/training/single_digit_add_train.jsonl",
    "carry_rule":               "data/training/carry_rule_train.jsonl",
    "multi_digit_add":          "data/training/multi_digit_add_train.jsonl",
    "arithmetic_multiply":      "data/training/arithmetic_multiply_train.jsonl",
    "arithmetic_divide":        "data/training/arithmetic_div_train.jsonl",
}


def test_five_skill_nodes_have_train_files():
    """Each wired skill node references its expected train file."""
    from modelcypher.core.use_cases.curriculum.skill_dag import CURRICULUM_DAG

    for skill_name, expected_path in EXPECTED_SKILL_TRAIN_FILES.items():
        node = CURRICULUM_DAG.get(skill_name)
        assert node is not None, f"Skill '{skill_name}' not found in DAG"
        assert node.train_files, f"Skill '{skill_name}' still has empty train_files"
        assert expected_path in node.train_files, (
            f"Skill '{skill_name}' train_files={node.train_files!r} "
            f"does not contain '{expected_path}'"
        )


def test_arithmetic_add_node_removed():
    """arithmetic_add has been decomposed — it must not exist in the DAG."""
    from modelcypher.core.use_cases.curriculum.skill_dag import CURRICULUM_DAG
    names = {n.name for n in CURRICULUM_DAG.nodes}
    assert "arithmetic_add" not in names, (
        "arithmetic_add was decomposed into single_digit_add / carry_rule / multi_digit_add "
        "and must not exist as a node"
    )
    assert "single_digit_add" in names
    assert "carry_rule" in names
    assert "multi_digit_add" in names


def test_arithmetic_dag_dependency_chain():
    """single_digit_add → carry_rule → multi_digit_add → arithmetic_multiply chain holds."""
    from modelcypher.core.use_cases.curriculum.skill_dag import CURRICULUM_DAG
    assert "single_digit_add" in CURRICULUM_DAG.get("carry_rule").prerequisites
    assert "carry_rule" in CURRICULUM_DAG.get("multi_digit_add").prerequisites
    assert "multi_digit_add" in CURRICULUM_DAG.get("arithmetic_multiply").prerequisites
    assert "multi_digit_add" in CURRICULUM_DAG.get("word_problem_1step").prerequisites


def test_arithmetic_nodes_have_numeric_answer_mode():
    """Arithmetic nodes use answer_mode='numeric' or 'procedural'."""
    from modelcypher.core.use_cases.curriculum.skill_dag import CURRICULUM_DAG
    numeric_nodes = ["single_digit_add", "arithmetic_divide"]
    for name in numeric_nodes:
        node = CURRICULUM_DAG.get(name)
        assert node.answer_mode == "numeric", (
            f"Expected {name}.answer_mode == 'numeric', got {node.answer_mode!r}"
        )
    # carry_rule, multi_digit_add, arithmetic_multiply use 'procedural':
    # requires carry-indicator tokens ("write"/"carry") + correct final answer.
    # Formal claim is procedure execution — numeric check alone is insufficient.
    procedural_nodes = ["carry_rule", "multi_digit_add", "arithmetic_multiply"]
    for name in procedural_nodes:
        node = CURRICULUM_DAG.get(name)
        assert node.answer_mode == "procedural", (
            f"Expected {name}.answer_mode == 'procedural', got {node.answer_mode!r}"
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
