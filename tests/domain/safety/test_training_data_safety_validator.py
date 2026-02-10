# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path

import pytest

from modelcypher.core.domain.safety.exceptions import ModerationUnavailableError
from modelcypher.core.domain.safety.regex_content_filter import ContentFilterResult, RuleAction
from modelcypher.core.domain.safety.safety_models import (
    DatasetPurpose,
    SafetyCategory,
    SafetyStatus,
    SafetyThresholds,
    StrictnessLevel,
)
from modelcypher.core.domain.safety.training_data_safety_validator import (
    ModerationResponse,
    TrainingDataSafetyValidator,
)
from modelcypher.core.domain.safety.training_sample import TrainingSample


class _RegexStub:
    def __init__(self, result):
        self._result = result

    def check(self, _text, purpose=None, custom_whitelist=None):
        return self._result


class _ModerationClientStub:
    def __init__(self, response: ModerationResponse | Exception):
        self._response = response

    async def moderate(self, _text: str) -> ModerationResponse:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _thresholds() -> SafetyThresholds:
    return SafetyThresholds(
        toxicity=0.5,
        hate_speech=0.5,
        violence=0.5,
        sexual=0.5,
        self_harm=0.5,
        harassment=0.5,
        prompt_injection=0.5,
        dangerous_code=0.5,
        pii=0.5,
    )


def _sample(text: str) -> TrainingSample:
    return TrainingSample.from_line(text, Path("/tmp/data.jsonl"))


async def test_validator_regex_reject_and_flag_paths() -> None:
    reject_match = ContentFilterResult(
        action=RuleAction.REJECT,
        reason="reject",
        category=SafetyCategory.VIOLENCE,
        rule_id="r1",
        matched_text="x",
    )
    validator_reject = TrainingDataSafetyValidator(regex_filter=_RegexStub(reject_match))
    rejected = await validator_reject.validate(
        sample=_sample("danger"),
        purpose=DatasetPurpose.GENERAL,
        strictness=StrictnessLevel.MODERATE,
        thresholds=_thresholds(),
    )
    assert rejected.status == SafetyStatus.REJECTED
    assert rejected.flagged_categories == (SafetyCategory.VIOLENCE,)

    flag_match = ContentFilterResult(
        action=RuleAction.FLAG,
        reason="flag",
        category=SafetyCategory.HARASSMENT,
        rule_id="r2",
        matched_text="y",
    )
    validator_flag = TrainingDataSafetyValidator(regex_filter=_RegexStub(flag_match))
    flagged = await validator_flag.validate(
        sample=_sample("maybe"),
        purpose=DatasetPurpose.GENERAL,
        strictness=StrictnessLevel.MODERATE,
        thresholds=_thresholds(),
    )
    assert flagged.status == SafetyStatus.FLAGGED_FOR_REVIEW
    assert flagged.requires_human_review is True


async def test_validator_approved_paths_without_external_calls(monkeypatch) -> None:
    validator = TrainingDataSafetyValidator(
        regex_filter=_RegexStub(None),
        moderation_client=None,
        allow_external_moderation=False,
    )

    approved_disabled = await validator.validate(
        sample=_sample("safe"),
        purpose=DatasetPurpose.GENERAL,
        strictness=StrictnessLevel.MODERATE,
        thresholds=_thresholds(),
    )
    assert approved_disabled.status == SafetyStatus.APPROVED

    validator_external = TrainingDataSafetyValidator(
        regex_filter=_RegexStub(None),
        allow_external_moderation=True,
    )

    short_text = await validator_external.validate(
        sample=_sample("abc"),
        purpose=DatasetPurpose.GENERAL,
        strictness=StrictnessLevel.MODERATE,
        thresholds=_thresholds(),
    )
    assert short_text.status == SafetyStatus.APPROVED

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    no_client = await validator_external.validate(
        sample=_sample("this is long enough"),
        purpose=DatasetPurpose.GENERAL,
        strictness=StrictnessLevel.MODERATE,
        thresholds=_thresholds(),
    )
    assert no_client.status == SafetyStatus.APPROVED


async def test_validator_external_moderation_mapping_and_rejection() -> None:
    response = ModerationResponse(
        flagged=True,
        category_scores={
            "violence": 0.95,
            "unknown": 0.99,
        },
    )
    validator = TrainingDataSafetyValidator(
        regex_filter=_RegexStub(None),
        moderation_client=_ModerationClientStub(response),
        allow_external_moderation=True,
    )

    result_reject = await validator.validate(
        sample=_sample("this text should be externally checked"),
        purpose=DatasetPurpose.GENERAL,
        strictness=StrictnessLevel.MODERATE,
        thresholds=_thresholds(),
    )
    assert result_reject.status == SafetyStatus.REJECTED
    assert result_reject.confidence == pytest.approx(0.95)
    assert result_reject.flagged_categories == (SafetyCategory.VIOLENCE,)

    result_flag = await validator.validate(
        sample=_sample("this text should be externally checked"),
        purpose=DatasetPurpose.GENERAL,
        strictness=StrictnessLevel.PERMISSIVE,
        thresholds=_thresholds(),
    )
    assert result_flag.status == SafetyStatus.FLAGGED_FOR_REVIEW


async def test_validator_external_moderation_error_raises() -> None:
    validator = TrainingDataSafetyValidator(
        regex_filter=_RegexStub(None),
        moderation_client=_ModerationClientStub(RuntimeError("network")),
        allow_external_moderation=True,
    )

    with pytest.raises(ModerationUnavailableError) as exc:
        await validator.validate(
            sample=_sample("this is long enough for moderation"),
            purpose=DatasetPurpose.GENERAL,
            strictness=StrictnessLevel.MODERATE,
            thresholds=_thresholds(),
        )

    assert exc.value.stage == "EXTERNAL_MODERATION"
    assert "OpenAI moderation API call failed" in str(exc.value)


def test_validator_category_mapping_and_openai_client_creation(monkeypatch) -> None:
    validator = TrainingDataSafetyValidator(regex_filter=_RegexStub(None), allow_external_moderation=True)

    assert validator._category_for_key("violence") == SafetyCategory.VIOLENCE
    assert validator._category_for_key("HATE") == SafetyCategory.HATE_SPEECH
    assert validator._category_for_key("not-a-real-category") is None

    mapping = validator._map_scores(
        {"violence": 0.8, "harassment": 0.4, "unknown": 1.0},
        _thresholds(),
    )
    assert mapping.categories == [SafetyCategory.VIOLENCE]
    assert mapping.scores[SafetyCategory.VIOLENCE] == pytest.approx(0.8)
    assert mapping.highest_score == pytest.approx(0.8)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with_key = TrainingDataSafetyValidator(regex_filter=_RegexStub(None), allow_external_moderation=True)
    assert with_key._openai_key == "sk-test"
    assert with_key._make_openai_client() is None
