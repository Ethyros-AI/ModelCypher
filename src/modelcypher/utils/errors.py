from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorDetail(Exception):
    code: str
    title: str
    detail: str
    hint: Optional[str] = None
    docs_url: Optional[str] = None
    trace_id: Optional[str] = None

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
