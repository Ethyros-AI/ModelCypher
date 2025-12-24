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

"""Semantic validator for AgentActionEnvelope beyond JSON Schema structural checks.

Rationale: our lightweight JSON Schema validator intentionally does not support
oneOf / anyOf / allOf, so kind-specific requirements must be enforced in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelcypher.core.domain.agents.agent_action import (
    ActionKind,
    AgentActionEnvelope,
)


@dataclass(frozen=True)
class AgentActionValidationResult:
    """Result of validating an agent action."""

    is_valid: bool
    """Whether the action is valid."""

    errors: list[str] = field(default_factory=list)
    """Validation errors."""

    warnings: list[str] = field(default_factory=list)
    """Validation warnings."""


class AgentActionValidator:
    """Semantic validator for AgentActionEnvelope beyond JSON Schema structural checks.

    Rationale: our lightweight JSON Schema validator intentionally does not support
    oneOf / anyOf / allOf, so kind-specific requirements must be enforced in code.
    """

    @staticmethod
    def validate(action: AgentActionEnvelope) -> AgentActionValidationResult:
        """Validate an agent action.

        Args:
            action: The action to validate.

        Returns:
            Validation result with errors and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check schema and version
        if action.schema != AgentActionEnvelope.SCHEMA_ID:
            errors.append(
                f"schema: expected {AgentActionEnvelope.SCHEMA_ID}, got {action.schema}"
            )
        if action.version != AgentActionEnvelope.SCHEMA_VERSION:
            errors.append(
                f"version: expected {AgentActionEnvelope.SCHEMA_VERSION}, got {action.version}"
            )

        # Check confidence range
        if action.confidence is not None:
            if not 0.0 <= action.confidence <= 1.0:
                errors.append(f"confidence: expected 0...1, got {action.confidence}")

        # Check payload exclusivity
        has_tool = action.tool is not None
        has_response = action.response is not None
        has_clarification = action.clarification is not None
        has_refusal = action.refusal is not None
        has_deferral = action.deferral is not None

        payload_count = sum([has_tool, has_response, has_clarification, has_refusal, has_deferral])
        if payload_count == 0:
            warnings.append("payload: no payload object present")
        elif payload_count > 1:
            errors.append("payload: multiple payload objects present; action must be unambiguous")

        # Kind-specific validation
        if action.kind == ActionKind.TOOL_CALL:
            if action.tool is None:
                errors.append("tool: required for kind=tool_call")
            else:
                name = action.tool.name.strip()
                if not name:
                    errors.append("tool.name: must not be empty")

        elif action.kind == ActionKind.RESPOND:
            if action.response is None:
                errors.append("response: required for kind=respond")
            else:
                text = action.response.text.strip()
                if not text:
                    errors.append("response.text: must not be empty")

        elif action.kind == ActionKind.ASK_CLARIFICATION:
            if action.clarification is None:
                errors.append("clarification: required for kind=ask_clarification")
            else:
                question = action.clarification.question.strip()
                if not question:
                    errors.append("clarification.question: must not be empty")
                if action.clarification.options is not None:
                    trimmed = [o.strip() for o in action.clarification.options]
                    if any(not o for o in trimmed):
                        errors.append("clarification.options: must not contain empty strings")

        elif action.kind == ActionKind.REFUSE:
            if action.refusal is None:
                errors.append("refusal: required for kind=refuse")
            else:
                reason = action.refusal.reason.strip()
                if not reason:
                    errors.append("refusal.reason: must not be empty")

        elif action.kind == ActionKind.DEFERRAL:
            if action.deferral is None:
                errors.append("deferral: required for kind=defer")
            else:
                reason = action.deferral.reason.strip()
                if not reason:
                    errors.append("deferral.reason: must not be empty")

        return AgentActionValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
