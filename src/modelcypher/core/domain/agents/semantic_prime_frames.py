from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.geometry.traversal_coherence import Path
from modelcypher.data import load_json


@dataclass(frozen=True)
class EnrichedPrime:
    id: str
    word: str
    frames: list[str]
    contrast: str | None
    exemplars: list[str]
    category: str

    @property
    def all_directional_texts(self) -> list[str]:
        texts: list[str] = []
        texts.extend(self.frames)
        if self.contrast:
            texts.append(self.contrast)
        texts.extend(self.exemplars)
        return texts


class SemanticPrimeFrames:
    _enriched: list[EnrichedPrime] | None = None

    @classmethod
    def enriched(cls) -> list[EnrichedPrime]:
        if cls._enriched is None:
            data = load_json("semantic_prime_frames.json")
            primes: list[EnrichedPrime] = []
            for item in data.get("enriched", []):
                primes.append(
                    EnrichedPrime(
                        id=str(item["id"]),
                        word=str(item["word"]),
                        frames=[str(text) for text in item.get("frames", [])],
                        contrast=str(item.get("contrast")) if item.get("contrast") is not None else None,
                        exemplars=[str(text) for text in item.get("exemplars", [])],
                        category=str(item.get("category", "")),
                    )
                )
            cls._enriched = primes
        return list(cls._enriched)

    @classmethod
    def directional_texts_grouped(cls) -> list[tuple[str, list[str]]]:
        return [(prime.id, prime.all_directional_texts) for prime in cls.enriched()]

    @classmethod
    def all_directional_texts(cls) -> list[str]:
        texts: list[str] = []
        for prime in cls.enriched():
            texts.extend(prime.all_directional_texts)
        return texts

    @staticmethod
    def standard_paths() -> list[Path]:
        return [
            Path(anchor_ids=["I", "WANT", "SOMETHING", "DO"]),
            Path(anchor_ids=["KNOW", "THINK", "TRUE"]),
            Path(anchor_ids=["BEFORE", "NOW", "AFTER"]),
            Path(anchor_ids=["HERE", "MOVE", "FAR"]),
            Path(anchor_ids=["GOOD", "NOT", "BAD"]),
            Path(anchor_ids=["ONE", "SOME", "ALL"]),
            Path(anchor_ids=["IF", "BECAUSE", "HAPPEN"]),
            Path(anchor_ids=["I", "SAY", "WORDS", "YOU"]),
        ]
