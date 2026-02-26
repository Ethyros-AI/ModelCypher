"""Fact-to-training-data converter for Baranov replication.

EXPERIMENTAL: Not validated for production use.

Converts ``FactTriple`` instances into training-format JSONL suitable for
``DatasetTrainingService``.  Each fact becomes a single training sample
whose ``text`` field contains the complete triple in natural-language form.
"""

from __future__ import annotations

import json
from pathlib import Path

from modelcypher.experimental.baranov.models import FactTriple
from modelcypher.experimental.baranov.simple_recall_evaluator import (
    _normalize_relation_text,
)


def fact_to_training_text(fact: FactTriple) -> str:
    """Convert a single fact to a training text string.

    Uses the structured triple format:
    ``"{subject} {relation_normalized} {object}"``

    The relation is normalized (underscores → spaces) for consistency
    with how the model will be probed at evaluation time.
    """
    relation = _normalize_relation_text(fact.relation)
    return f"{fact.subject} {relation} {fact.object}"


def facts_to_training_samples(
    facts: list[FactTriple],
) -> list[dict[str, str]]:
    """Convert a list of facts to training samples.

    Returns a list of ``{"text": "..."}`` dicts matching the format
    expected by ``DatasetTrainingService``.
    """
    return [{"text": fact_to_training_text(f)} for f in facts]


def write_fact_training_jsonl(
    facts: list[FactTriple],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write facts as a JSONL training file.

    Each line is a JSON object ``{"text": "..."}`` suitable for
    ``DatasetTrainingService``.

    Raises ``FileExistsError`` if *output_path* exists and *overwrite*
    is ``False``.

    Returns the written path.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output path already exists: {output_path}. "
            "Pass overwrite=True to replace.",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = facts_to_training_samples(facts)
    lines = [json.dumps(s, ensure_ascii=False) for s in samples]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "fact_to_training_text",
    "facts_to_training_samples",
    "write_fact_training_jsonl",
]
