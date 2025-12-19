from __future__ import annotations

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.models import CompareSession


class CompareService:
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()

    def list_sessions(self, limit: int = 50, status: str | None = None) -> dict:
        sessions = self.store.list_sessions(limit, status)
        return {"sessions": sessions}

    def get_session(self, session_id: str) -> CompareSession:
        session = self.store.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        return session
