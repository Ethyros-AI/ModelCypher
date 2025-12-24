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
