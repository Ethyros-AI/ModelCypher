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
from enum import Enum

from modelcypher.core.domain.agents.semantic_prime_atlas import SemanticPrimeAtlas
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeSignature


class DriftMethod(str, Enum):
    prime_signature = "primeSignature"
    skipped = "skipped"


class DriftVerdict(str, Enum):
    """Verdict on semantic drift detection.

    Note: 'drifted' means the model's semantic signature has changed from baseline.
    This is informational, not a compatibility judgment. The cosine_similarity
    value is the actual measurement - the verdict is just a convenience threshold.
    """

    stable = "stable"  # Similarity above configured threshold
    drifted = "drifted"  # Similarity below configured threshold
    unknown = "unknown"  # Could not compute (missing data)


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

    def assess(
        self, baseline: SemanticPrimeSignature, observed_text: str
    ) -> SemanticPrimeDriftAssessment:
        if not self._config.enabled:
            return SemanticPrimeDriftAssessment(
                method=DriftMethod.skipped, verdict=DriftVerdict.unknown, note="disabled"
            )

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
                note="signature_computation_failed",
            )

        verdict = (
            DriftVerdict.stable
            if similarity >= self._config.minimum_cosine_similarity
            else DriftVerdict.drifted
        )
        return SemanticPrimeDriftAssessment(
            method=DriftMethod.prime_signature,
            verdict=verdict,
            cosine_similarity=float(similarity),
            threshold=self._config.minimum_cosine_similarity,
            note="cosine_below_threshold" if verdict == DriftVerdict.drifted else None,
        )
