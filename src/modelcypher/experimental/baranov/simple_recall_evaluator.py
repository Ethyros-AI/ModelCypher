"""Simple string-match recall evaluator for Baranov replication.

EXPERIMENTAL: Not validated for production use.

Implements ``RecallEvaluator`` via case-insensitive substring match of the
fact object in the model's generated output.  Supports both ``raw_completion``
(bare prompt) and ``chat_template`` (formatted via ``ChatTemplate``) modes.

No heuristic thresholds -- recall is a binary yes/no per fact based on
whether the expected object string appears in the output.
"""

from __future__ import annotations

from typing import Any

from modelcypher.core.domain.chat_template import ChatMessage, ChatTemplate
from modelcypher.experimental.baranov.models import FactTriple
from modelcypher.experimental.baranov.recall_evaluator import (
    GenerateFn,
    RecallMode,
    RecallOutcome,
    RecallResult,
    compute_recall_aggregate,
)


def _build_raw_prompt(fact: FactTriple) -> str:
    """Build a bare completion prompt for a single fact.

    Format: ``"<subject> <relation>"`` — the model should complete with the
    object.  This is the simplest possible probe.
    """
    return f"{fact.subject} {fact.relation}"


def _normalize_relation_text(relation: str) -> str:
    """Normalize relation tokens for chat prompts.

    Replaces underscores with spaces and collapses repeated whitespace so
    prompts like ``capital_of`` become ``capital of``.
    """
    return " ".join(relation.replace("_", " ").split())


def _build_chat_prompt(
    fact: FactTriple,
    template: ChatTemplate,
) -> str:
    """Build a chat-formatted prompt for a single fact.

    Uses a system message establishing the task, then a user message
    asking the fact question.  The template formats them per the model's
    expected chat format.
    """
    relation_text = _normalize_relation_text(fact.relation)
    messages = [
        ChatMessage(
            role="system",
            content="Answer the following question with only the answer, no explanation.",
        ),
        ChatMessage(
            role="user",
            content=(
                "Return only the object for this fact triple.\n"
                f"Subject: {fact.subject}\n"
                f"Relation: {relation_text}\n"
                "Object:"
            ),
        ),
    ]
    return template.format_messages(messages)


def _check_recall(fact: FactTriple, output: str) -> bool:
    """Case-insensitive substring match of fact.object in output."""
    return fact.object.lower() in output.lower()


class SimpleRecallEvaluator:
    """String-match recall evaluator.

    Probes a model for each fact and checks whether the expected object
    appears (case-insensitive) in the generated output.

    This is a minimal first implementation -- it catches the gross
    success/failure signal.  More sophisticated evaluators (semantic
    similarity, structured extraction) are future work.

    Satisfies the ``RecallEvaluator`` protocol.
    """

    def __init__(self, *, max_tokens: int = 64) -> None:
        """
        Parameters
        ----------
        max_tokens:
            Maximum tokens to generate per fact probe.  Kept short since
            we only need the object string to appear.
        """
        self._max_tokens = max_tokens

    def evaluate_recall(
        self,
        facts: list[FactTriple],
        generate_fn: GenerateFn,
        model: Any,
        tokenizer: Any,
        mode: RecallMode = RecallMode.raw_completion,
        chat_template: str | None = None,
    ) -> RecallResult:
        """Evaluate recall of *facts* using string-match scoring.

        Parameters
        ----------
        facts:
            Facts to probe.
        generate_fn:
            Model generation callback:
            ``generate_fn(model, tokenizer, prompt, max_tokens, verbose) -> str``
        model:
            The model object (passed through to generate_fn).
        tokenizer:
            The tokenizer (passed through to generate_fn).
        mode:
            ``raw_completion`` or ``chat_template``.
        chat_template:
            Template name when mode is ``chat_template``.  Must be a valid
            ``ChatTemplate`` enum value.  Ignored when mode is
            ``raw_completion``.

        Returns
        -------
        RecallResult with per-fact outcomes and aggregate statistics.
        """
        template: ChatTemplate | None = None
        if mode == RecallMode.chat_template:
            if chat_template is None:
                raise ValueError(
                    "chat_template must be provided when mode is chat_template",
                )
            template = ChatTemplate(chat_template)

        outcomes: list[RecallOutcome] = []
        for fact in facts:
            if mode == RecallMode.raw_completion:
                prompt = _build_raw_prompt(fact)
            else:
                assert template is not None
                prompt = _build_chat_prompt(fact, template)

            raw_output = generate_fn(
                model,
                tokenizer,
                prompt,
                self._max_tokens,
                False,
            )

            recalled = _check_recall(fact, raw_output)
            outcomes.append(
                RecallOutcome(
                    fact_id=fact.fact_id,
                    recalled=recalled,
                    raw_output=raw_output,
                    confidence=None,
                ),
            )

        aggregate = compute_recall_aggregate(outcomes)
        return RecallResult(
            per_fact_outcomes=tuple(outcomes),
            aggregate=aggregate,
        )


__all__ = ["SimpleRecallEvaluator"]
