"""Tests for output safety result types.

Covers enum values, factory class methods, computed properties, and
the DEFAULT_FILTERED_PLACEHOLDER constant.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.safety.output_safety_result import (
    DEFAULT_FILTERED_PLACEHOLDER,
    OutputSafetyResult,
    OutputSafetyResultType,
)
from modelcypher.core.domain.safety.safety_models import SafetyCategory


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestOutputSafetyResultType:
    def test_all_values_accessible(self) -> None:
        assert OutputSafetyResultType.SAFE == "safe"
        assert OutputSafetyResultType.FILTERED == "filtered"
        assert OutputSafetyResultType.TRUNCATED == "truncated"

    def test_is_str_enum(self) -> None:
        assert isinstance(OutputSafetyResultType.SAFE, str)

    def test_member_count(self) -> None:
        assert len(OutputSafetyResultType) == 3


# ---------------------------------------------------------------------------
# Constant tests
# ---------------------------------------------------------------------------


class TestDefaultFilteredPlaceholder:
    def test_value(self) -> None:
        assert DEFAULT_FILTERED_PLACEHOLDER == "[...]"


# ---------------------------------------------------------------------------
# OutputSafetyResult.safe() tests
# ---------------------------------------------------------------------------


class TestOutputSafetyResultSafe:
    @pytest.fixture()
    def safe_result(self) -> OutputSafetyResult:
        return OutputSafetyResult.safe("hello")

    def test_result_type(self, safe_result: OutputSafetyResult) -> None:
        assert safe_result.result_type == OutputSafetyResultType.SAFE

    def test_token_stored(self, safe_result: OutputSafetyResult) -> None:
        assert safe_result.token == "hello"

    def test_is_safe(self, safe_result: OutputSafetyResult) -> None:
        assert safe_result.is_safe is True

    def test_was_filtered(self, safe_result: OutputSafetyResult) -> None:
        assert safe_result.was_filtered is False

    def test_display_text(self, safe_result: OutputSafetyResult) -> None:
        assert safe_result.display_text == "hello"

    def test_display_text_empty_token(self) -> None:
        result = OutputSafetyResult.safe("")
        assert result.display_text == ""

    def test_optional_fields_none(self, safe_result: OutputSafetyResult) -> None:
        assert safe_result.replacement is None
        assert safe_result.category is None
        assert safe_result.rule_id is None
        assert safe_result.reason is None


# ---------------------------------------------------------------------------
# OutputSafetyResult.filtered() tests
# ---------------------------------------------------------------------------


class TestOutputSafetyResultFiltered:
    @pytest.fixture()
    def filtered_result(self) -> OutputSafetyResult:
        return OutputSafetyResult.filtered(
            replacement="[redacted]",
            category=SafetyCategory.TOXICITY,
            rule_id="tox-001",
        )

    def test_result_type(self, filtered_result: OutputSafetyResult) -> None:
        assert filtered_result.result_type == OutputSafetyResultType.FILTERED

    def test_is_safe(self, filtered_result: OutputSafetyResult) -> None:
        assert filtered_result.is_safe is False

    def test_was_filtered(self, filtered_result: OutputSafetyResult) -> None:
        assert filtered_result.was_filtered is True

    def test_replacement_stored(self, filtered_result: OutputSafetyResult) -> None:
        assert filtered_result.replacement == "[redacted]"

    def test_category_stored(self, filtered_result: OutputSafetyResult) -> None:
        assert filtered_result.category == SafetyCategory.TOXICITY

    def test_rule_id_stored(self, filtered_result: OutputSafetyResult) -> None:
        assert filtered_result.rule_id == "tox-001"

    def test_display_text(self, filtered_result: OutputSafetyResult) -> None:
        assert filtered_result.display_text == "[redacted]"

    def test_display_text_falls_back_to_placeholder(self) -> None:
        result = OutputSafetyResult(
            result_type=OutputSafetyResultType.FILTERED,
            replacement=None,
            category=SafetyCategory.PII,
            rule_id="pii-001",
        )
        assert result.display_text == DEFAULT_FILTERED_PLACEHOLDER


# ---------------------------------------------------------------------------
# OutputSafetyResult.truncated() tests
# ---------------------------------------------------------------------------


class TestOutputSafetyResultTruncated:
    @pytest.fixture()
    def truncated_result(self) -> OutputSafetyResult:
        return OutputSafetyResult.truncated("too many violations")

    def test_result_type(self, truncated_result: OutputSafetyResult) -> None:
        assert truncated_result.result_type == OutputSafetyResultType.TRUNCATED

    def test_is_safe(self, truncated_result: OutputSafetyResult) -> None:
        assert truncated_result.is_safe is False

    def test_was_filtered(self, truncated_result: OutputSafetyResult) -> None:
        assert truncated_result.was_filtered is True

    def test_reason_stored(self, truncated_result: OutputSafetyResult) -> None:
        assert truncated_result.reason == "too many violations"

    def test_display_text_includes_reason(self, truncated_result: OutputSafetyResult) -> None:
        assert "too many violations" in truncated_result.display_text

    def test_display_text_format(self, truncated_result: OutputSafetyResult) -> None:
        expected = "\n[Generation stopped: too many violations]"
        assert truncated_result.display_text == expected

    def test_display_text_none_reason_fallback(self) -> None:
        result = OutputSafetyResult(result_type=OutputSafetyResultType.TRUNCATED)
        assert result.display_text == "\n[Generation stopped: safety limit reached]"


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------


class TestOutputSafetyResultFrozen:
    def test_cannot_mutate(self) -> None:
        result = OutputSafetyResult.safe("token")
        with pytest.raises(AttributeError):
            result.token = "changed"  # type: ignore[misc]
