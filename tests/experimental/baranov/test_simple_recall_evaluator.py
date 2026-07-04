"""Unit tests for SimpleRecallEvaluator."""

from __future__ import annotations

import pytest

from modelcypher.experimental.baranov.models import FactTriple
from modelcypher.experimental.baranov.recall_evaluator import (
    RecallEvaluator,
    RecallMode,
    RecallResult,
)
from modelcypher.experimental.baranov.simple_recall_evaluator import (
    SimpleRecallEvaluator,
    _build_chat_prompt,
    _build_raw_prompt,
    _check_recall,
    _normalize_relation_text,
)

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


def _mock_generate_fn(responses: dict[str, str]):
    """Return a generate_fn that maps prompts to canned responses."""

    def generate(model, tokenizer, prompt, max_tokens, verbose=False):
        # Find the matching response by checking if any key is a substring
        for key, resp in responses.items():
            if key in prompt:
                return resp
        return ""

    return generate


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildRawPrompt:
    def test_basic_format(self) -> None:
        fact = _make_fact()
        prompt = _build_raw_prompt(fact)
        assert prompt == "Paris capital_of"

    def test_different_relation(self) -> None:
        fact = _make_fact(subject="Water", relation="chemical_formula", obj="H2O")
        prompt = _build_raw_prompt(fact)
        assert prompt == "Water chemical_formula"


class TestBuildChatPrompt:
    def test_contains_subject_and_relation(self) -> None:
        from modelcypher.core.domain.chat_template import ChatTemplate

        fact = _make_fact()
        prompt = _build_chat_prompt(fact, ChatTemplate.chatml)
        assert "Paris" in prompt
        assert "capital of" in prompt
        assert "Object:" in prompt

    def test_uses_template_formatting(self) -> None:
        from modelcypher.core.domain.chat_template import ChatTemplate

        fact = _make_fact()
        prompt = _build_chat_prompt(fact, ChatTemplate.llama3)
        # Llama3 uses <|begin_of_text|> header tokens
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>" in prompt


class TestNormalizeRelationText:
    def test_replaces_underscores(self) -> None:
        assert _normalize_relation_text("capital_of") == "capital of"

    def test_collapses_whitespace(self) -> None:
        assert _normalize_relation_text("  chemical___symbol  ") == "chemical symbol"


# ---------------------------------------------------------------------------
# Recall checking
# ---------------------------------------------------------------------------


class TestCheckRecall:
    def test_exact_match(self) -> None:
        fact = _make_fact()
        assert _check_recall(fact, "France") is True

    def test_case_insensitive(self) -> None:
        fact = _make_fact()
        assert _check_recall(fact, "france") is True
        assert _check_recall(fact, "FRANCE") is True

    def test_substring_match(self) -> None:
        fact = _make_fact()
        assert _check_recall(fact, "The capital is France, of course.") is True

    def test_no_match(self) -> None:
        fact = _make_fact()
        assert _check_recall(fact, "Germany is great") is False

    def test_empty_output(self) -> None:
        fact = _make_fact()
        assert _check_recall(fact, "") is False

    def test_partial_match_not_sufficient(self) -> None:
        """'Fran' should not match 'France'."""
        fact = _make_fact()
        assert _check_recall(fact, "Fran") is False

    def test_multi_token_object(self) -> None:
        fact = _make_fact(obj="Turing machine")
        assert _check_recall(fact, "He invented the Turing machine.") is True
        assert _check_recall(fact, "He invented the Turing test.") is False


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_protocol(self) -> None:
        """SimpleRecallEvaluator satisfies the RecallEvaluator protocol."""
        evaluator = SimpleRecallEvaluator()
        assert isinstance(evaluator, RecallEvaluator)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


class TestEvaluateRecall:
    def test_all_recalled(self) -> None:
        facts = [
            _make_fact("Paris", "capital_of", "France", "f1"),
            _make_fact("Berlin", "capital_of", "Germany", "f2"),
        ]
        responses = {
            "Paris": "France is the answer",
            "Berlin": "Germany of course",
        }
        evaluator = SimpleRecallEvaluator()
        result = evaluator.evaluate_recall(
            facts=facts,
            generate_fn=_mock_generate_fn(responses),
            model=None,
            tokenizer=None,
            mode=RecallMode.raw_completion,
        )

        assert isinstance(result, RecallResult)
        assert result.aggregate.total == 2
        assert result.aggregate.recalled_count == 2
        assert result.aggregate.recall_rate == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        facts = [
            _make_fact("Paris", "capital_of", "France", "f1"),
            _make_fact("Berlin", "capital_of", "Germany", "f2"),
            _make_fact("Tokyo", "capital_of", "Japan", "f3"),
        ]
        responses = {
            "Paris": "France",
            "Berlin": "I don't know",
            "Tokyo": "Japan",
        }
        evaluator = SimpleRecallEvaluator()
        result = evaluator.evaluate_recall(
            facts=facts,
            generate_fn=_mock_generate_fn(responses),
            model=None,
            tokenizer=None,
        )

        assert result.aggregate.recalled_count == 2
        assert result.aggregate.recall_rate == pytest.approx(2 / 3)

        # Verify per-fact outcomes
        outcomes_by_id = {o.fact_id: o for o in result.per_fact_outcomes}
        assert outcomes_by_id["f1"].recalled is True
        assert outcomes_by_id["f2"].recalled is False
        assert outcomes_by_id["f3"].recalled is True

    def test_zero_recall(self) -> None:
        facts = [_make_fact("Paris", "capital_of", "France", "f1")]
        responses = {"Paris": "I have no idea"}
        evaluator = SimpleRecallEvaluator()
        result = evaluator.evaluate_recall(
            facts=facts,
            generate_fn=_mock_generate_fn(responses),
            model=None,
            tokenizer=None,
        )

        assert result.aggregate.recalled_count == 0
        assert result.aggregate.recall_rate == pytest.approx(0.0)

    def test_empty_facts(self) -> None:
        evaluator = SimpleRecallEvaluator()
        result = evaluator.evaluate_recall(
            facts=[],
            generate_fn=_mock_generate_fn({}),
            model=None,
            tokenizer=None,
        )
        assert result.aggregate.total == 0

    def test_chat_template_mode_requires_template(self) -> None:
        evaluator = SimpleRecallEvaluator()
        with pytest.raises(ValueError, match="chat_template must be provided"):
            evaluator.evaluate_recall(
                facts=[_make_fact()],
                generate_fn=_mock_generate_fn({}),
                model=None,
                tokenizer=None,
                mode=RecallMode.chat_template,
                chat_template=None,
            )

    def test_chat_template_mode(self) -> None:
        """Chat template mode uses formatted prompts."""
        facts = [_make_fact("Paris", "capital_of", "France", "f1")]
        # The chat prompt will contain "Paris" so the mock will match
        responses = {"Paris": "France"}
        evaluator = SimpleRecallEvaluator()
        result = evaluator.evaluate_recall(
            facts=facts,
            generate_fn=_mock_generate_fn(responses),
            model=None,
            tokenizer=None,
            mode=RecallMode.chat_template,
            chat_template="chatml",
        )
        assert result.aggregate.recalled_count == 1

    def test_raw_output_preserved(self) -> None:
        """Per-fact outcomes preserve the raw model output."""
        facts = [_make_fact("Paris", "capital_of", "France", "f1")]
        responses = {"Paris": "The answer is France, naturally."}
        evaluator = SimpleRecallEvaluator()
        result = evaluator.evaluate_recall(
            facts=facts,
            generate_fn=_mock_generate_fn(responses),
            model=None,
            tokenizer=None,
        )
        assert result.per_fact_outcomes[0].raw_output == "The answer is France, naturally."

    def test_max_tokens_passed_to_generate(self) -> None:
        """The configured max_tokens is passed to generate_fn."""
        captured = {}

        def spy_generate(model, tokenizer, prompt, max_tokens, verbose=False):
            captured["max_tokens"] = max_tokens
            return "France"

        evaluator = SimpleRecallEvaluator(max_tokens=128)
        evaluator.evaluate_recall(
            facts=[_make_fact()],
            generate_fn=spy_generate,
            model=None,
            tokenizer=None,
        )
        assert captured["max_tokens"] == 128

    def test_confidence_interval_present(self) -> None:
        """Aggregate includes a CI when n > 0."""
        facts = [_make_fact(fact_id=f"f{i}") for i in range(5)]
        responses = {"Paris": "France"}
        evaluator = SimpleRecallEvaluator()
        result = evaluator.evaluate_recall(
            facts=facts,
            generate_fn=_mock_generate_fn(responses),
            model=None,
            tokenizer=None,
        )
        assert result.aggregate.confidence_interval is not None
        lo, hi = result.aggregate.confidence_interval
        assert 0.0 <= lo <= hi <= 1.0
