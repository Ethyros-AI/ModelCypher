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

"""Intrinsic identity dataset linter.

Detects identity instructions, roleplay, and persona definitions in datasets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from modelcypher.core.domain.validation.dataset_validation_models import (
    DatasetContentFormat,
    ValidationWarning,
    ValidationWarningKind,
)


@dataclass(frozen=True)
class IdentityFinding:
    """A finding from identity linting."""

    kind: ValidationWarningKind
    """Kind of identity issue."""

    matched_text: str
    """Text that triggered the finding."""

    field_name: str
    """Field where finding occurred."""

    pattern_name: str
    """Name of pattern that matched."""


class IntrinsicIdentityLinter:
    """Lints datasets for intrinsic identity instructions.

    Detects patterns like:
    - "You are a..." / "Act as a..."
    - Roleplay instructions
    - Persona definitions
    """

    # Identity instruction patterns
    IDENTITY_PATTERNS = [
        (r"\byou\s+are\s+(?:a|an|the)\b", "you_are"),
        (r"\bact\s+as\s+(?:a|an|the)?\b", "act_as"),
        (r"\bpretend\s+(?:to\s+be|you\s+are)\b", "pretend"),
        (r"\bbehave\s+(?:like|as)\b", "behave_as"),
        (r"\byour\s+(?:name|identity)\s+is\b", "identity_assignment"),
        (r"\byou\s+will\s+(?:act|be|play)\b", "you_will"),
        (r"\bassume\s+the\s+(?:role|identity|persona)\b", "assume_role"),
        (r"\btake\s+on\s+the\s+(?:role|identity|persona)\b", "take_on_role"),
    ]

    # Roleplay patterns
    ROLEPLAY_PATTERNS = [
        (r"\broleplay\b", "roleplay"),
        (r"\brole-play\b", "role_play"),
        (r"\bplay\s+the\s+role\b", "play_role"),
        (r"\bin\s+character\b", "in_character"),
        (r"\bstay\s+in\s+character\b", "stay_in_character"),
        (r"\bcharacter\s+sheet\b", "character_sheet"),
    ]

    # Persona patterns
    PERSONA_PATTERNS = [
        (r"\bpersona\b", "persona"),
        (r"\bpersonality\s*:\s*", "personality_colon"),
        (r"\btraits\s*:\s*", "traits_colon"),
        (r"\bbackstory\s*:\s*", "backstory_colon"),
        (r"\bcharacter\s+description\b", "character_description"),
    ]

    def __init__(self):
        """Initialize linter."""
        # Compile patterns
        self._identity_re = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.IDENTITY_PATTERNS
        ]
        self._roleplay_re = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.ROLEPLAY_PATTERNS
        ]
        self._persona_re = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.PERSONA_PATTERNS
        ]

    def lint(
        self,
        sample: dict[str, Any],
        detected_format: DatasetContentFormat,
        line_number: int | None = None,
        sample_index: int | None = None,
    ) -> list[ValidationWarning]:
        """Lint a sample for identity issues.

        Args:
            sample: Parsed sample.
            detected_format: Format of the sample.
            line_number: Line number (1-based).
            sample_index: Sample index (0-based).

        Returns:
            List of warnings.
        """
        if detected_format == DatasetContentFormat.TEXT:
            return self._lint_text(sample, line_number, sample_index)
        elif detected_format == DatasetContentFormat.CHAT:
            return self._lint_chat(sample, line_number, sample_index)
        elif detected_format == DatasetContentFormat.TOOLS:
            return self._lint_chat(sample, line_number, sample_index)
        elif detected_format == DatasetContentFormat.INSTRUCTION:
            return self._lint_instruction(sample, line_number, sample_index)
        elif detected_format == DatasetContentFormat.COMPLETION:
            return self._lint_completion(sample, line_number, sample_index)
        else:
            return []

    def _lint_text(
        self,
        sample: dict[str, Any],
        line_number: int | None,
        sample_index: int | None,
    ) -> list[ValidationWarning]:
        """Lint text format sample."""
        text = sample.get("text", "")
        if not isinstance(text, str):
            return []

        findings = self._find_identity_issues(text, "text")
        return self._findings_to_warnings(findings, line_number, sample_index)

    def _lint_chat(
        self,
        sample: dict[str, Any],
        line_number: int | None,
        sample_index: int | None,
    ) -> list[ValidationWarning]:
        """Lint chat format sample.

        Only checks system messages for identity issues.
        """
        messages = sample.get("messages", [])
        if not isinstance(messages, list):
            return []

        findings: list[IdentityFinding] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            # Only lint system messages for identity instructions
            if role == "system" and isinstance(content, str):
                findings.extend(
                    self._find_identity_issues(content, "messages[system].content")
                )

        return self._findings_to_warnings(findings, line_number, sample_index)

    def _lint_instruction(
        self,
        sample: dict[str, Any],
        line_number: int | None,
        sample_index: int | None,
    ) -> list[ValidationWarning]:
        """Lint instruction format sample."""
        findings: list[IdentityFinding] = []

        # Check instruction/input field
        instruction = sample.get("instruction") or sample.get("input", "")
        if isinstance(instruction, str):
            findings.extend(self._find_identity_issues(instruction, "instruction"))

        # Check system field if present
        system = sample.get("system", "")
        if isinstance(system, str):
            findings.extend(self._find_identity_issues(system, "system"))

        return self._findings_to_warnings(findings, line_number, sample_index)

    def _lint_completion(
        self,
        sample: dict[str, Any],
        line_number: int | None,
        sample_index: int | None,
    ) -> list[ValidationWarning]:
        """Lint completion format sample."""
        findings: list[IdentityFinding] = []

        prompt = sample.get("prompt", "")
        if isinstance(prompt, str):
            findings.extend(self._find_identity_issues(prompt, "prompt"))

        return self._findings_to_warnings(findings, line_number, sample_index)

    def _find_identity_issues(
        self, text: str, field_name: str
    ) -> list[IdentityFinding]:
        """Find identity issues in text.

        Args:
            text: Text to check.
            field_name: Name of field being checked.

        Returns:
            List of findings.
        """
        findings: list[IdentityFinding] = []

        # Check identity patterns
        for pattern, name in self._identity_re:
            match = pattern.search(text)
            if match:
                findings.append(
                    IdentityFinding(
                        kind=ValidationWarningKind.INTRINSIC_IDENTITY_INSTRUCTION,
                        matched_text=match.group(0),
                        field_name=field_name,
                        pattern_name=name,
                    )
                )

        # Check roleplay patterns
        for pattern, name in self._roleplay_re:
            match = pattern.search(text)
            if match:
                findings.append(
                    IdentityFinding(
                        kind=ValidationWarningKind.INTRINSIC_IDENTITY_ROLEPLAY,
                        matched_text=match.group(0),
                        field_name=field_name,
                        pattern_name=name,
                    )
                )

        # Check persona patterns
        for pattern, name in self._persona_re:
            match = pattern.search(text)
            if match:
                findings.append(
                    IdentityFinding(
                        kind=ValidationWarningKind.INTRINSIC_IDENTITY_PERSONA,
                        matched_text=match.group(0),
                        field_name=field_name,
                        pattern_name=name,
                    )
                )

        return findings

    def _findings_to_warnings(
        self,
        findings: list[IdentityFinding],
        line_number: int | None,
        sample_index: int | None,
    ) -> list[ValidationWarning]:
        """Convert findings to warnings.

        Deduplicates by kind to avoid excessive warnings.
        """
        # Dedupe by kind
        seen_kinds: set[ValidationWarningKind] = set()
        warnings: list[ValidationWarning] = []

        for finding in findings:
            if finding.kind in seen_kinds:
                continue
            seen_kinds.add(finding.kind)

            if finding.kind == ValidationWarningKind.INTRINSIC_IDENTITY_INSTRUCTION:
                message = f"Identity instruction detected: '{finding.matched_text}' in {finding.field_name}"
            elif finding.kind == ValidationWarningKind.INTRINSIC_IDENTITY_ROLEPLAY:
                message = f"Roleplay instruction detected: '{finding.matched_text}' in {finding.field_name}"
            else:
                message = f"Persona definition detected: '{finding.matched_text}' in {finding.field_name}"

            warnings.append(
                ValidationWarning(
                    kind=finding.kind,
                    message=message,
                    line_number=line_number,
                    field_name=finding.field_name,
                    sample_index=sample_index,
                )
            )

        return warnings


class DatasetIdentityScanner:
    """Scans entire dataset for identity issues."""

    def __init__(self):
        """Initialize scanner."""
        self._linter = IntrinsicIdentityLinter()

    def scan_samples(
        self,
        samples: list[tuple[int, dict[str, Any], DatasetContentFormat]],
    ) -> list[ValidationWarning]:
        """Scan multiple samples for identity issues.

        Args:
            samples: List of (line_number, sample, format) tuples.

        Returns:
            Combined warnings from all samples.
        """
        all_warnings: list[ValidationWarning] = []

        for line_number, sample, fmt in samples:
            warnings = self._linter.lint(
                sample,
                detected_format=fmt,
                line_number=line_number,
                sample_index=len(all_warnings),
            )
            all_warnings.extend(warnings)

        return all_warnings

    def summarize_findings(
        self, warnings: list[ValidationWarning]
    ) -> dict[ValidationWarningKind, int]:
        """Summarize findings by kind.

        Args:
            warnings: List of warnings.

        Returns:
            Dict mapping warning kind to count.
        """
        counts: dict[ValidationWarningKind, int] = {}
        for w in warnings:
            counts[w.kind] = counts.get(w.kind, 0) + 1
        return counts
