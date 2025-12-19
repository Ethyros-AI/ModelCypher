from __future__ import annotations


def truncate(text: str, max_characters: int) -> str:
    if max_characters <= 0:
        return text
    if len(text) <= max_characters:
        return text
    return text[:max_characters]
