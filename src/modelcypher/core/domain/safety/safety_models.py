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

"""Safety models and types for training data validation.

Defines core types for content safety classification across validation layers:
- SafetyCategory: Content categories (toxicity, hate speech, PII, etc.)
- SafetyStatus: Validation disposition (approved, flagged, rejected)
- SafetyThresholds: Per-category confidence thresholds
- DatasetPurpose: Domain-specific safety rule whitelists
- StrictnessLevel: Validation strictness presets
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyCategory(str, Enum):
    """Content safety categories for training data validation.

    Maps to both regex patterns and moderation API categories
    for consistent safety classification across validation layers.
    """

    TOXICITY = "toxicity"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"
    PROMPT_INJECTION = "prompt_injection"
    DANGEROUS_CODE = "dangerous_code"
    PII = "pii"


class SafetyStatus(str, Enum):
    """Final disposition of a safety validation check."""

    APPROVED = "approved"
    """Content passed validation and can be used for training."""

    FLAGGED_FOR_REVIEW = "flaggedForReview"
    """Content needs human review before training."""

    REJECTED = "rejected"
    """Content is blocked from training."""


class SafetyValidationLayer(str, Enum):
    """Source layer that produced a safety validation decision."""

    REGEX = "regex"
    """Fast local pattern matching (first pass)."""

    OPENAI = "openAI"
    """OpenAI Moderation API (second pass for context-aware classification)."""


class ModerationFailureMode(str, Enum):
    """Behavior when the moderation API is unavailable or fails."""

    APPROVE = "approve"
    FLAG = "flag"
    REJECT = "reject"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return {
            ModerationFailureMode.APPROVE: "Approve",
            ModerationFailureMode.FLAG: "Flag for review",
            ModerationFailureMode.REJECT: "Reject",
        }[self]

    @property
    def description(self) -> str:
        """Detailed description of behavior."""
        return {
            ModerationFailureMode.APPROVE: (
                "If moderation fails, treat the sample as approved (no moderation signal)."
            ),
            ModerationFailureMode.FLAG: (
                "Send samples to the review queue when moderation is unavailable."
            ),
            ModerationFailureMode.REJECT: "Reject samples outright when moderation fails.",
        }[self]


@dataclass(frozen=True)
class SafetyValidationResult:
    """Outcome of a safety validation pass for a single sample."""

    status: SafetyStatus
    """Final status chosen by the validator (approved, flagged, or rejected)."""

    confidence: float
    """Confidence value originating from the highest-risk category."""

    flagged_categories: tuple[SafetyCategory, ...]
    """Categories that triggered above their thresholds."""

    category_scores: dict[SafetyCategory, float]
    """Raw per-category confidence scores."""

    requires_human_review: bool
    """Indicates whether a human should review the sample before training."""

    reason: str | None = None
    """Optional human-readable explanation for the decision."""

    source_layer: SafetyValidationLayer | None = None
    """The layer (regex or OpenAI) responsible for the decision."""


@dataclass(frozen=True)
class SafetyThresholds:
    """Per-category confidence thresholds for triggering safety flags.

    Values between 0.0 and 1.0. Lower thresholds are stricter (more flags).
    Provides presets: strict, default, permissive.

    WARNING: The preset values are ARBITRARY PLACEHOLDERS, not calibrated
    from empirical data. Users MUST calibrate these thresholds for their
    specific deployment using labeled safety data and ROC curve analysis.

    To calibrate:
    1. Collect labeled examples of safe/unsafe content for each category
    2. Run classifier predictions on the labeled set
    3. Generate ROC curves and choose thresholds based on your
       false-positive/false-negative tolerance
    4. Create custom SafetyThresholds with calibrated values
    """

    toxicity: float
    hate_speech: float
    violence: float
    sexual: float
    self_harm: float
    harassment: float
    prompt_injection: float
    dangerous_code: float
    pii: float

    def threshold_for(self, category: SafetyCategory) -> float:
        """Get threshold for a specific category."""
        mapping = {
            SafetyCategory.TOXICITY: self.toxicity,
            SafetyCategory.HATE_SPEECH: self.hate_speech,
            SafetyCategory.VIOLENCE: self.violence,
            SafetyCategory.SEXUAL: self.sexual,
            SafetyCategory.SELF_HARM: self.self_harm,
            SafetyCategory.HARASSMENT: self.harassment,
            SafetyCategory.PROMPT_INJECTION: self.prompt_injection,
            SafetyCategory.DANGEROUS_CODE: self.dangerous_code,
            SafetyCategory.PII: self.pii,
        }
        return mapping[category]

    @classmethod
    def default(cls) -> SafetyThresholds:
        """Default thresholds - UNCALIBRATED PLACEHOLDERS.

        WARNING: These values are NOT derived from empirical data. They are
        arbitrary placeholders for development/testing only.

        For production use, call SafetyThresholds.from_calibration() with
        baseline measurements from your specific deployment.
        """
        warnings.warn(
            "SafetyThresholds.default() returns uncalibrated placeholders. "
            "Use SafetyThresholds.from_calibration() for production.",
            UserWarning,
            stacklevel=2,
        )
        return cls(
            toxicity=0.7,
            hate_speech=0.6,
            violence=0.8,
            sexual=0.75,
            self_harm=0.6,
            harassment=0.7,
            prompt_injection=0.9,
            dangerous_code=0.8,
            pii=0.7,
        )

    @classmethod
    def recommended(cls) -> SafetyThresholds:
        """Alias for default - still arbitrary, not calibrated."""
        return cls.default()

    @classmethod
    def strict(cls) -> SafetyThresholds:
        """Stricter thresholds - ARBITRARY PLACEHOLDERS, not calibrated.

        Lower values = more flags. Use as a starting point for high-risk
        deployments. Calibrate based on your false-positive tolerance.
        """
        warnings.warn(
            "SafetyThresholds.strict() returns uncalibrated placeholders. "
            "Use SafetyThresholds.from_calibration() for production.",
            UserWarning,
            stacklevel=2,
        )
        return cls(
            toxicity=0.5,
            hate_speech=0.4,
            violence=0.6,
            sexual=0.6,
            self_harm=0.4,
            harassment=0.5,
            prompt_injection=0.8,
            dangerous_code=0.6,
            pii=0.5,
        )

    @classmethod
    def permissive(cls) -> SafetyThresholds:
        """Permissive thresholds - ARBITRARY PLACEHOLDERS, not calibrated.

        Higher values = fewer flags. May be appropriate for curated datasets
        or internal tooling. Calibrate based on your false-negative tolerance.
        """
        warnings.warn(
            "SafetyThresholds.permissive() returns uncalibrated placeholders. "
            "Use SafetyThresholds.from_calibration() for production.",
            UserWarning,
            stacklevel=2,
        )
        return cls(
            toxicity=0.85,
            hate_speech=0.8,
            violence=0.9,
            sexual=0.85,
            self_harm=0.8,
            harassment=0.85,
            prompt_injection=0.95,
            dangerous_code=0.9,
            pii=0.88,
        )


class DatasetPurpose(str, Enum):
    """Dataset purpose that determines which safety rules are whitelisted.

    Different domains have different safety requirements. Code generation
    datasets allow shell commands that would be blocked for general use.
    """

    GENERAL = "general"
    CODE_GENERATION = "codeGeneration"
    CYBERSECURITY = "cybersecurity"
    MEDICAL_LEGAL = "medicalLegal"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return {
            DatasetPurpose.GENERAL: "General",
            DatasetPurpose.CODE_GENERATION: "Code Generation",
            DatasetPurpose.CYBERSECURITY: "Cybersecurity",
            DatasetPurpose.MEDICAL_LEGAL: "Medical / Legal",
        }[self]

    @property
    def description(self) -> str:
        """Detailed description of safety rules."""
        return {
            DatasetPurpose.GENERAL: (
                "Max safety. Reject shell commands, credentials, or sensitive instructions."
            ),
            DatasetPurpose.CODE_GENERATION: (
                "Allow benign shell commands and code blocks but still block destructive payloads."
            ),
            DatasetPurpose.CYBERSECURITY: (
                "Permit exploit discussions for defensive datasets while catching prompt injections."
            ),
            DatasetPurpose.MEDICAL_LEGAL: (
                "Allow domain advice but reject PII and high-risk instructions."
            ),
        }[self]

    @property
    def whitelisted_rule_ids(self) -> frozenset[str]:
        """Rule IDs that are whitelisted for this purpose."""
        if self in (DatasetPurpose.CODE_GENERATION, DatasetPurpose.CYBERSECURITY):
            return frozenset({"shell_commands", "code_execution"})
        return frozenset()


class StrictnessLevel(str, Enum):
    """Validation strictness level controlling auto-reject behavior.

    Higher strictness = more samples rejected automatically.
    Lower strictness = more samples sent to review queue.
    """

    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return {
            StrictnessLevel.STRICT: "Strict",
            StrictnessLevel.MODERATE: "Moderate",
            StrictnessLevel.PERMISSIVE: "Permissive",
        }[self]

    @property
    def description(self) -> str:
        """Detailed description of behavior."""
        return {
            StrictnessLevel.STRICT: "Auto-reject at >= 0.7 confidence.",
            StrictnessLevel.MODERATE: "Auto-reject at >= 0.9 confidence.",
            StrictnessLevel.PERMISSIVE: "Route findings to review queue; never auto-reject.",
        }[self]

    @property
    def thresholds(self) -> SafetyThresholds:
        """Safety thresholds for this strictness level."""
        return {
            StrictnessLevel.STRICT: SafetyThresholds.strict(),
            StrictnessLevel.MODERATE: SafetyThresholds.default(),
            StrictnessLevel.PERMISSIVE: SafetyThresholds.permissive(),
        }[self]

    @property
    def auto_reject_floor(self) -> float | None:
        """Confidence floor for auto-rejection, or None to never auto-reject.

        WARNING: The values 0.7 (strict) and 0.9 (moderate) are UNCALIBRATED
        PLACEHOLDERS. They do not derive from empirical precision measurements.

        For production use, calibrate using labeled safety data:
        1. Collect labeled safe/unsafe examples
        2. Run classifier predictions on the labeled set
        3. Choose threshold to achieve target precision (e.g., 99% true positive rate)
        """
        floor = {
            StrictnessLevel.STRICT: 0.7,
            StrictnessLevel.MODERATE: 0.9,
            StrictnessLevel.PERMISSIVE: None,
        }[self]
        if floor is not None:
            warnings.warn(
                f"StrictnessLevel.{self.name}.auto_reject_floor returns uncalibrated "
                f"placeholder ({floor}). Calibrate using precision measurements from "
                "labeled safety data for production use.",
                UserWarning,
                stacklevel=2,
            )
        return floor
