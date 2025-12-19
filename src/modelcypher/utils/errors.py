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

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "hint": self.hint,
            "docsUrl": self.docs_url,
            "traceId": self.trace_id,
        }
