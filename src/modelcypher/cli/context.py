from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass
class CLIContext:
    ai_mode: bool
    output_format: str
    quiet: bool
    very_quiet: bool
    yes: bool
    no_prompt: bool
    pretty: bool
    log_level: str
    trace_id: str | None


def resolve_ai_mode(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("MC_NO_AI") == "1":
        return False
    if os.environ.get("MC_AI_MODE") == "1":
        return True
    return not sys.stdout.isatty()


def resolve_output_format(ai_mode: bool, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("MC_OUTPUT")
    if env:
        return env
    return "json" if ai_mode else "text"
