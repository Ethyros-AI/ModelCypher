# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import warnings

import pytest

from modelcypher.core.domain.safety.safety_models import (
    DatasetPurpose,
    ModerationFailureMode,
    SafetyCategory,
    SafetyStatus,
    SafetyThresholds,
    SafetyValidationLayer,
    SafetyValidationResult,
    StrictnessLevel,
)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_all_enums_importable(self) -> None:
        for cls in (
            SafetyCategory,
            SafetyStatus,
            SafetyValidationLayer,
            ModerationFailureMode,
            DatasetPurpose,
            StrictnessLevel,
        ):
            assert cls is not None

    def test_dataclasses_importable(self) -> None:
        assert SafetyValidationResult is not None
        assert SafetyThresholds is not None


# ---------------------------------------------------------------------------
# SafetyCategory enum
# ---------------------------------------------------------------------------


class TestSafetyCategory:
    EXPECTED_VALUES = [
        "toxicity",
        "hate_speech",
        "violence",
        "sexual",
        "self_harm",
        "harassment",
        "prompt_injection",
        "dangerous_code",
        "pii",
    ]

    def test_all_values_accessible(self) -> None:
        values = [m.value for m in SafetyCategory]
        assert sorted(values) == sorted(self.EXPECTED_VALUES)

    def test_is_str_enum(self) -> None:
        assert isinstance(SafetyCategory.TOXICITY, str)
        assert SafetyCategory.TOXICITY == "toxicity"


# ---------------------------------------------------------------------------
# SafetyStatus enum
# ---------------------------------------------------------------------------


class TestSafetyStatus:
    def test_all_values(self) -> None:
        assert SafetyStatus.APPROVED.value == "approved"
        assert SafetyStatus.FLAGGED_FOR_REVIEW.value == "flaggedForReview"
        assert SafetyStatus.REJECTED.value == "rejected"

    def test_is_str_enum(self) -> None:
        assert isinstance(SafetyStatus.APPROVED, str)


# ---------------------------------------------------------------------------
# SafetyValidationLayer enum
# ---------------------------------------------------------------------------


class TestSafetyValidationLayer:
    def test_all_values(self) -> None:
        assert SafetyValidationLayer.REGEX.value == "regex"
        assert SafetyValidationLayer.OPENAI.value == "openAI"


# ---------------------------------------------------------------------------
# ModerationFailureMode enum
# ---------------------------------------------------------------------------


class TestModerationFailureMode:
    def test_all_values(self) -> None:
        assert ModerationFailureMode.APPROVE.value == "approve"
        assert ModerationFailureMode.FLAG.value == "flag"
        assert ModerationFailureMode.REJECT.value == "reject"

    def test_display_name_returns_strings(self) -> None:
        for mode in ModerationFailureMode:
            name = mode.display_name
            assert isinstance(name, str)
            assert len(name) > 0

    def test_description_returns_strings(self) -> None:
        for mode in ModerationFailureMode:
            desc = mode.description
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_display_name_specific_values(self) -> None:
        assert ModerationFailureMode.APPROVE.display_name == "Approve"
        assert ModerationFailureMode.FLAG.display_name == "Flag for review"
        assert ModerationFailureMode.REJECT.display_name == "Reject"


# ---------------------------------------------------------------------------
# DatasetPurpose enum
# ---------------------------------------------------------------------------


class TestDatasetPurpose:
    def test_all_values(self) -> None:
        assert DatasetPurpose.GENERAL.value == "general"
        assert DatasetPurpose.CODE_GENERATION.value == "codeGeneration"
        assert DatasetPurpose.CYBERSECURITY.value == "cybersecurity"
        assert DatasetPurpose.MEDICAL_LEGAL.value == "medicalLegal"

    def test_display_name_returns_strings(self) -> None:
        for purpose in DatasetPurpose:
            name = purpose.display_name
            assert isinstance(name, str)
            assert len(name) > 0

    def test_description_returns_strings(self) -> None:
        for purpose in DatasetPurpose:
            desc = purpose.description
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_whitelisted_rule_ids_returns_frozensets(self) -> None:
        for purpose in DatasetPurpose:
            rules = purpose.whitelisted_rule_ids
            assert isinstance(rules, frozenset)

    def test_general_has_no_whitelisted_rules(self) -> None:
        assert DatasetPurpose.GENERAL.whitelisted_rule_ids == frozenset()

    def test_medical_legal_has_no_whitelisted_rules(self) -> None:
        assert DatasetPurpose.MEDICAL_LEGAL.whitelisted_rule_ids == frozenset()

    def test_code_generation_whitelists_shell_and_code(self) -> None:
        rules = DatasetPurpose.CODE_GENERATION.whitelisted_rule_ids
        assert "shell_commands" in rules
        assert "code_execution" in rules

    def test_cybersecurity_whitelists_shell_and_code(self) -> None:
        rules = DatasetPurpose.CYBERSECURITY.whitelisted_rule_ids
        assert "shell_commands" in rules
        assert "code_execution" in rules


# ---------------------------------------------------------------------------
# StrictnessLevel enum
# ---------------------------------------------------------------------------


class TestStrictnessLevel:
    def test_all_values(self) -> None:
        assert StrictnessLevel.STRICT.value == "strict"
        assert StrictnessLevel.MODERATE.value == "moderate"
        assert StrictnessLevel.PERMISSIVE.value == "permissive"

    def test_display_name_returns_strings(self) -> None:
        for level in StrictnessLevel:
            name = level.display_name
            assert isinstance(name, str)
            assert len(name) > 0

    def test_description_returns_strings(self) -> None:
        for level in StrictnessLevel:
            desc = level.description
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_thresholds_returns_safety_thresholds_instances(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for level in StrictnessLevel:
                thresholds = level.thresholds
                assert isinstance(thresholds, SafetyThresholds)

    def test_auto_reject_floor_strict_returns_float(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            floor = StrictnessLevel.STRICT.auto_reject_floor
            assert isinstance(floor, float)
            assert floor == 0.7

    def test_auto_reject_floor_moderate_returns_float(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            floor = StrictnessLevel.MODERATE.auto_reject_floor
            assert isinstance(floor, float)
            assert floor == 0.9

    def test_auto_reject_floor_permissive_returns_none(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            floor = StrictnessLevel.PERMISSIVE.auto_reject_floor
            assert floor is None

    def test_auto_reject_floor_emits_warning_for_non_none(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StrictnessLevel.STRICT.auto_reject_floor
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "uncalibrated" in str(user_warnings[0].message).lower()


# ---------------------------------------------------------------------------
# SafetyThresholds dataclass
# ---------------------------------------------------------------------------


class TestSafetyThresholds:
    def test_instantiation(self) -> None:
        t = SafetyThresholds(
            toxicity=0.5,
            hate_speech=0.4,
            violence=0.6,
            sexual=0.5,
            self_harm=0.3,
            harassment=0.5,
            prompt_injection=0.8,
            dangerous_code=0.7,
            pii=0.6,
        )
        assert t.toxicity == 0.5
        assert t.pii == 0.6

    def test_frozen(self) -> None:
        t = SafetyThresholds(
            toxicity=0.5,
            hate_speech=0.4,
            violence=0.6,
            sexual=0.5,
            self_harm=0.3,
            harassment=0.5,
            prompt_injection=0.8,
            dangerous_code=0.7,
            pii=0.6,
        )
        with pytest.raises(AttributeError):
            t.toxicity = 0.9  # type: ignore[misc]

    def test_threshold_for_maps_all_categories(self) -> None:
        t = SafetyThresholds(
            toxicity=0.1,
            hate_speech=0.2,
            violence=0.3,
            sexual=0.4,
            self_harm=0.5,
            harassment=0.6,
            prompt_injection=0.7,
            dangerous_code=0.8,
            pii=0.9,
        )
        assert t.threshold_for(SafetyCategory.TOXICITY) == 0.1
        assert t.threshold_for(SafetyCategory.HATE_SPEECH) == 0.2
        assert t.threshold_for(SafetyCategory.VIOLENCE) == 0.3
        assert t.threshold_for(SafetyCategory.SEXUAL) == 0.4
        assert t.threshold_for(SafetyCategory.SELF_HARM) == 0.5
        assert t.threshold_for(SafetyCategory.HARASSMENT) == 0.6
        assert t.threshold_for(SafetyCategory.PROMPT_INJECTION) == 0.7
        assert t.threshold_for(SafetyCategory.DANGEROUS_CODE) == 0.8
        assert t.threshold_for(SafetyCategory.PII) == 0.9

    def test_default_factory(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t = SafetyThresholds.default()
            assert isinstance(t, SafetyThresholds)
            # Verify all fields are populated floats
            for cat in SafetyCategory:
                val = t.threshold_for(cat)
                assert isinstance(val, float)
                assert 0.0 < val < 1.0

    def test_recommended_factory(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t = SafetyThresholds.recommended()
            assert isinstance(t, SafetyThresholds)

    def test_strict_factory(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t = SafetyThresholds.strict()
            assert isinstance(t, SafetyThresholds)
            for cat in SafetyCategory:
                val = t.threshold_for(cat)
                assert isinstance(val, float)
                assert 0.0 < val < 1.0

    def test_permissive_factory(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t = SafetyThresholds.permissive()
            assert isinstance(t, SafetyThresholds)
            for cat in SafetyCategory:
                val = t.threshold_for(cat)
                assert isinstance(val, float)
                assert 0.0 < val < 1.0

    def test_strict_lower_than_permissive(self) -> None:
        """Strict thresholds should be lower (more sensitive) than permissive."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            strict = SafetyThresholds.strict()
            permissive = SafetyThresholds.permissive()
            for cat in SafetyCategory:
                assert strict.threshold_for(cat) < permissive.threshold_for(cat)

    def test_factory_methods_emit_warnings(self) -> None:
        for factory in (SafetyThresholds.default, SafetyThresholds.strict, SafetyThresholds.permissive):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                factory()
                user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
                assert len(user_warnings) >= 1
                assert "uncalibrated" in str(user_warnings[0].message).lower()


# ---------------------------------------------------------------------------
# SafetyValidationResult dataclass
# ---------------------------------------------------------------------------


class TestSafetyValidationResult:
    def test_instantiation_with_required_fields(self) -> None:
        result = SafetyValidationResult(
            status=SafetyStatus.APPROVED,
            confidence=0.1,
            flagged_categories=(),
            category_scores={},
            requires_human_review=False,
        )
        assert result.status == SafetyStatus.APPROVED
        assert result.confidence == 0.1
        assert result.flagged_categories == ()
        assert result.category_scores == {}
        assert result.requires_human_review is False
        assert result.reason is None
        assert result.source_layer is None

    def test_instantiation_with_all_fields(self) -> None:
        scores = {SafetyCategory.TOXICITY: 0.9, SafetyCategory.PII: 0.3}
        result = SafetyValidationResult(
            status=SafetyStatus.REJECTED,
            confidence=0.9,
            flagged_categories=(SafetyCategory.TOXICITY,),
            category_scores=scores,
            requires_human_review=True,
            reason="High toxicity score",
            source_layer=SafetyValidationLayer.OPENAI,
        )
        assert result.status == SafetyStatus.REJECTED
        assert result.confidence == 0.9
        assert result.flagged_categories == (SafetyCategory.TOXICITY,)
        assert result.category_scores == scores
        assert result.requires_human_review is True
        assert result.reason == "High toxicity score"
        assert result.source_layer == SafetyValidationLayer.OPENAI

    def test_frozen(self) -> None:
        result = SafetyValidationResult(
            status=SafetyStatus.APPROVED,
            confidence=0.1,
            flagged_categories=(),
            category_scores={},
            requires_human_review=False,
        )
        with pytest.raises(AttributeError):
            result.status = SafetyStatus.REJECTED  # type: ignore[misc]

    def test_flagged_for_review_status(self) -> None:
        result = SafetyValidationResult(
            status=SafetyStatus.FLAGGED_FOR_REVIEW,
            confidence=0.6,
            flagged_categories=(SafetyCategory.HARASSMENT,),
            category_scores={SafetyCategory.HARASSMENT: 0.6},
            requires_human_review=True,
            source_layer=SafetyValidationLayer.REGEX,
        )
        assert result.status == SafetyStatus.FLAGGED_FOR_REVIEW
        assert result.source_layer == SafetyValidationLayer.REGEX
