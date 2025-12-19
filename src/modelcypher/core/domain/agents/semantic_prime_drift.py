from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain.agents.semantic_prime_atlas import SemanticPrimeAtlas
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeSignature


class DriftMethod(str, Enum):
    prime_signature = "primeSignature"
    skipped = "skipped"


class DriftVerdict(str, Enum):
    stable = "stable"
    drifted = "drifted"
    unknown = "unknown"


@dataclass(frozen=True)
class SemanticPrimeDriftAssessment:
    method: DriftMethod
    verdict: DriftVerdict
    cosine_similarity: float | None = None
    threshold: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class SemanticPrimeDriftConfig:
    enabled: bool = True
    minimum_cosine_similarity: float = 0.65
    fail_closed: bool = False


class SemanticPrimeDriftDetector:
    def __init__(
        self,
        configuration: SemanticPrimeDriftConfig | None = None,
        atlas: SemanticPrimeAtlas | None = None,
    ) -> None:
        self._config = configuration or SemanticPrimeDriftConfig()
        self._atlas = atlas or SemanticPrimeAtlas()

    def assess(self, baseline: SemanticPrimeSignature, observed_text: str) -> SemanticPrimeDriftAssessment:
        if not self._config.enabled:
            return SemanticPrimeDriftAssessment(method=DriftMethod.skipped, verdict=DriftVerdict.unknown, note="disabled")

        observed = self._atlas.signature(observed_text)
        if observed is None:
            return SemanticPrimeDriftAssessment(
                method=DriftMethod.skipped,
                verdict=DriftVerdict.drifted if self._config.fail_closed else DriftVerdict.unknown,
                note="no_signature",
            )

        similarity = baseline.cosine_similarity(observed)
        if similarity is None:
            return SemanticPrimeDriftAssessment(
                method=DriftMethod.skipped,
                verdict=DriftVerdict.drifted if self._config.fail_closed else DriftVerdict.unknown,
                note="incompatible_signature",
            )

        verdict = DriftVerdict.stable if similarity >= self._config.minimum_cosine_similarity else DriftVerdict.drifted
        return SemanticPrimeDriftAssessment(
            method=DriftMethod.prime_signature,
            verdict=verdict,
            cosine_similarity=float(similarity),
            threshold=self._config.minimum_cosine_similarity,
            note="cosine_below_threshold" if verdict == DriftVerdict.drifted else None,
        )
