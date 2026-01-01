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

"""Safety validation exceptions.

These exceptions are raised when safety operations cannot complete.
Fallbacks that mask failures are not acceptable - the caller must
handle the exception explicitly.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SafetyValidationError",
    "ModerationUnavailableError",
    "LayerAlphaMissingError",
]


class SafetyValidationError(Exception):
    """Base exception for safety validation failures.

    Provides structured error information including:
    - stage: Which validation stage failed
    - context: Dictionary with diagnostic details
    """

    def __init__(
        self,
        stage: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.context = context or {}
        full_message = f"[{stage}] {message}"
        super().__init__(full_message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"stage={self.stage!r}, "
            f"context={self.context!r})"
        )


class ModerationUnavailableError(SafetyValidationError):
    """External moderation API is unavailable.

    The caller must decide how to handle this - not the validator.
    Options: retry with backoff, use local-only validation, or fail.

    Context typically includes:
    - error_type: The type of error that occurred
    - error_message: The error message
    - suggestion: How to handle the error
    """

    pass


class LayerAlphaMissingError(SafetyValidationError):
    """Required layer alpha value not found.

    The alpha_by_layer dict must contain all layers being processed.
    Silent defaults mask configuration errors.

    Context typically includes:
    - layer: The missing layer identifier
    - available_layers: List of available layer keys
    """

    pass
