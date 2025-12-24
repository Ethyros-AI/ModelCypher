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

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ErrorDetail(Exception):
    code: str
    title: str
    detail: str
    hint: str | None = None
    docs_url: str | None = None
    trace_id: str | None = None

    @property
    def message(self) -> str:
        """Return human-readable error message."""
        return f"{self.title}: {self.detail}"

    @classmethod
    def from_exception(cls, exc: Exception, code: str = "INTERNAL_ERROR") -> "ErrorDetail":
        """Create ErrorDetail from a generic exception."""
        exc_type = type(exc).__name__
        return cls(
            code=code,
            title=exc_type,
            detail=str(exc),
            hint=None,
        )

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "hint": self.hint,
            "docsUrl": self.docs_url,
            "traceId": self.trace_id,
        }
