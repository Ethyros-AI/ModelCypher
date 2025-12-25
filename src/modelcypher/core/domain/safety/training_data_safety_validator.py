# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Two-layer training data safety validator.

Provides regex-first validation with optional external moderation API
(OpenAI Moderation) for context-aware classification. Designed for
thread-safe validation of training samples.

Architecture:
1. First pass: Fast local regex pattern matching
2. Second pass: Optional OpenAI Moderation API for context-aware flags

The validator uses the sample's normalized text for validation and can
be configured with custom thresholds and strictness levels.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from modelcypher.core.domain.safety.regex_content_filter import (
    DatasetPurpose,
    RegexContentFilter,
    SafetyCategory,
)
from modelcypher.core.domain.safety.safety_models import (
    ModerationFailureMode,
    SafetyStatus,
    SafetyThresholds,
    SafetyValidationLayer,
    SafetyValidationResult,
    StrictnessLevel,
)
from modelcypher.core.domain.safety.training_sample import TrainingSample

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ContentModerationClient(Protocol):
    """Protocol for external content moderation clients."""

    async def moderate(self, text: str) -> ModerationResponse:
        """Moderate text content.

        Args:
            text: Text to moderate.

        Returns:
            Moderation response with category scores.
        """
        ...


@dataclass(frozen=True)
class ModerationResponse:
    """Response from a content moderation API."""

    flagged: bool
    """Whether the content was flagged."""

    category_scores: dict[str, float]
    """Per-category confidence scores (0.0 to 1.0)."""


@dataclass
class TrainingDataSafetyValidator:
    """Two-layer training data safety validator.

    Runs regex filtering first, then optional OpenAI moderation with
    category thresholds for context-aware classification.
    """

    regex_filter: RegexContentFilter = field(default_factory=RegexContentFilter.default)
    """Local pattern filter."""

    moderation_client: ContentModerationClient | None = None
    """Optional external moderation client."""

    allow_external_moderation: bool = False
    """Whether to allow external moderation API calls."""

    _openai_key: str | None = field(init=False, default=None)
    """Cached OpenAI API key."""

    def __post_init__(self) -> None:
        """Initialize external moderation if enabled."""
        if self.allow_external_moderation:
            key = os.environ.get("OPENAI_API_KEY", "")
            self._openai_key = key if key else None

    async def validate(
        self,
        sample: TrainingSample,
        purpose: DatasetPurpose,
        strictness: StrictnessLevel,
        thresholds: SafetyThresholds,
        custom_whitelist: set[str] | None = None,
        failure_mode: ModerationFailureMode = ModerationFailureMode.FLAG,
        allow_external_moderation: bool | None = None,
    ) -> SafetyValidationResult:
        """Validate a training sample for potentially harmful content.

        Runs regex filtering first, then optional OpenAI moderation with
        category thresholds.

        Args:
            sample: Training sample to validate.
            purpose: Dataset purpose for whitelist rules.
            strictness: Validation strictness level.
            thresholds: Per-category confidence thresholds.
            custom_whitelist: Additional rule IDs to skip.
            failure_mode: Behavior when moderation API fails.
            allow_external_moderation: Override instance setting.

        Returns:
            Validation result with status and details.
        """
        custom_whitelist = custom_whitelist or set()

        # First pass: regex filtering
        regex_result = self.regex_filter.check(
            sample.text,
            purpose=purpose,
            custom_whitelist=custom_whitelist,
        )

        if regex_result is not None:
            # Map regex SafetyStatus to safety_models SafetyStatus
            status = (
                SafetyStatus.REJECTED
                if regex_result.status.value == "rejected"
                else SafetyStatus.FLAGGED_FOR_REVIEW
            )
            result = SafetyValidationResult(
                status=status,
                confidence=1.0,
                flagged_categories=(regex_result.category,) if regex_result.category else (),
                category_scores={},
                requires_human_review=status == SafetyStatus.FLAGGED_FOR_REVIEW,
                reason=regex_result.reason,
                source_layer=SafetyValidationLayer.REGEX,
            )
            self._log_decision(result)
            return result

        # Check if external moderation is allowed
        external_allowed = (
            allow_external_moderation
            if allow_external_moderation is not None
            else self.allow_external_moderation
        )

        # Skip moderation if disabled or text too short
        if not external_allowed or len(sample.text) < 4:
            return self._approved_result()

        # Skip moderation if no client and no API key
        if self.moderation_client is None and not self._openai_key:
            return self._approved_result()

        # Second pass: external moderation
        try:
            client = self.moderation_client or self._make_openai_client()
            if client is None:
                return self._approved_result()

            response = await client.moderate(sample.text)
            mapped = self._map_scores(response.category_scores, thresholds)

            if not mapped.categories:
                return self._approved_result()

            # Determine rejection based on strictness
            should_reject = False
            auto_reject_floor = strictness.auto_reject_floor
            if auto_reject_floor is not None:
                should_reject = mapped.highest_score >= auto_reject_floor

            status = SafetyStatus.REJECTED if should_reject else SafetyStatus.FLAGGED_FOR_REVIEW
            category_names = ", ".join(c.value for c in mapped.categories)
            reason = f"OpenAI flagged {category_names}"

            result = SafetyValidationResult(
                status=status,
                confidence=mapped.highest_score,
                flagged_categories=tuple(mapped.categories),
                category_scores=mapped.scores,
                requires_human_review=status == SafetyStatus.FLAGGED_FOR_REVIEW,
                reason=reason,
                source_layer=SafetyValidationLayer.OPENAI,
            )
            self._log_decision(result)
            return result

        except Exception as e:
            logger.error("OpenAI moderation failed: %s", str(e))
            fallback = self._result_for_failure_mode(failure_mode, "OpenAI moderation unavailable")
            self._log_decision(fallback)
            return fallback

    def _approved_result(self) -> SafetyValidationResult:
        """Create an approved validation result."""
        return SafetyValidationResult(
            status=SafetyStatus.APPROVED,
            confidence=0.0,
            flagged_categories=(),
            category_scores={},
            requires_human_review=False,
            reason=None,
            source_layer=None,
        )

    def _result_for_failure_mode(
        self,
        failure_mode: ModerationFailureMode,
        reason: str,
    ) -> SafetyValidationResult:
        """Create a result based on failure mode."""
        if failure_mode == ModerationFailureMode.APPROVE:
            return self._approved_result()
        elif failure_mode == ModerationFailureMode.FLAG:
            return SafetyValidationResult(
                status=SafetyStatus.FLAGGED_FOR_REVIEW,
                confidence=0.0,
                flagged_categories=(),
                category_scores={},
                requires_human_review=True,
                reason=reason,
                source_layer=SafetyValidationLayer.OPENAI,
            )
        else:  # REJECT
            return SafetyValidationResult(
                status=SafetyStatus.REJECTED,
                confidence=0.0,
                flagged_categories=(),
                category_scores={},
                requires_human_review=False,
                reason=reason,
                source_layer=SafetyValidationLayer.OPENAI,
            )

    def _log_decision(self, result: SafetyValidationResult) -> None:
        """Log a validation decision."""
        if result.status == SafetyStatus.APPROVED:
            return

        categories = (
            "none"
            if not result.flagged_categories
            else ", ".join(
                c.value if hasattr(c, "value") else str(c) for c in result.flagged_categories
            )
        )
        layer = result.source_layer.value if result.source_layer else "unknown"

        logger.info(
            "Safety flag: status=%s, categories=%s, confidence=%.2f, layer=%s",
            result.status.value,
            categories,
            result.confidence,
            layer,
        )

    @dataclass
    class _ScoreMapping:
        categories: list[SafetyCategory]
        scores: dict[SafetyCategory, float]
        highest_score: float

    def _map_scores(
        self,
        raw_scores: dict[str, float],
        thresholds: SafetyThresholds,
    ) -> _ScoreMapping:
        """Map raw API scores to safety categories."""
        categories: list[SafetyCategory] = []
        scores: dict[SafetyCategory, float] = {}
        max_score: float = 0.0

        for key, value in raw_scores.items():
            category = self._category_for_key(key)
            if category is None:
                continue

            # Get threshold - handle both SafetyCategory types
            try:
                # Map to models SafetyCategory for threshold lookup
                models_category = self._to_models_safety_category(category)
                if models_category is None:
                    continue
                threshold = thresholds.threshold_for(models_category)
            except Exception:
                threshold = 0.5  # Default threshold

            if value < threshold:
                continue

            categories.append(category)
            scores[category] = value
            max_score = max(max_score, value)

        return self._ScoreMapping(categories=categories, scores=scores, highest_score=max_score)

    def _category_for_key(self, key: str) -> SafetyCategory | None:
        """Map API category key to SafetyCategory."""
        key_lower = key.lower()
        mapping = {
            "hate": SafetyCategory.HATE_SPEECH,
            "hate/threatening": SafetyCategory.HATE_SPEECH,
            "harassment": SafetyCategory.HARASSMENT,
            "harassment/threatening": SafetyCategory.HARASSMENT,
            "self-harm": SafetyCategory.SELF_HARM,
            "self-harm/intent": SafetyCategory.SELF_HARM,
            "self-harm/instructions": SafetyCategory.SELF_HARM,
            "sexual": SafetyCategory.SEXUAL,
            "sexual/minors": SafetyCategory.SEXUAL,
            "violence": SafetyCategory.VIOLENCE,
            "violence/graphic": SafetyCategory.VIOLENCE,
        }
        return mapping.get(key_lower)

    def _to_models_safety_category(self, category: SafetyCategory) -> "ModelsSafetyCategory" | None:
        """Convert regex_content_filter SafetyCategory to safety_models SafetyCategory."""
        from modelcypher.core.domain.safety.safety_models import (
            SafetyCategory as ModelsSafetyCategory,
        )

        mapping = {
            SafetyCategory.DANGEROUS_CODE: ModelsSafetyCategory.DANGEROUS_CODE,
            SafetyCategory.PROMPT_INJECTION: ModelsSafetyCategory.PROMPT_INJECTION,
            SafetyCategory.PII: ModelsSafetyCategory.PII,
            SafetyCategory.SELF_HARM: ModelsSafetyCategory.SELF_HARM,
            SafetyCategory.VIOLENCE: ModelsSafetyCategory.VIOLENCE,
            SafetyCategory.HATE_SPEECH: ModelsSafetyCategory.HATE_SPEECH,
            SafetyCategory.SEXUAL: ModelsSafetyCategory.SEXUAL,
            SafetyCategory.HARASSMENT: ModelsSafetyCategory.HARASSMENT,
        }
        return mapping.get(category)

    def _make_openai_client(self) -> ContentModerationClient | None:
        """Create an OpenAI moderation client if API key is available."""
        # Placeholder - actual implementation would create OpenAI client
        # For now, return None as we don't have the OpenAI SDK as a dependency
        return None
