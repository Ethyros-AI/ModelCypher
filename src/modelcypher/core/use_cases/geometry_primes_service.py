"""
Geometry Primes Service for semantic prime analysis.

Provides semantic prime listing, model activation probing, and cross-model
comparison using Natural Semantic Metalanguage (NSM) primes.

Example:
    service = GeometryPrimesService()
    primes = service.list_primes()
    activations = service.probe("/path/to/model")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.agents.semantic_primes import (
    SemanticPrimeInventory,
)

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PrimeInfo:
    """Semantic prime information for listing."""
    id: str
    name: str
    category: str
    exponents: list[str]


@dataclass(frozen=True)
class PrimeActivation:
    """Activation pattern for a single prime."""
    prime_id: str
    activation_strength: float
    layer_activations: list[float]


@dataclass(frozen=True)
class PrimeComparisonResult:
    """Result of comparing prime alignments between two models."""
    alignment_score: float
    divergent_primes: list[str]
    convergent_primes: list[str]
    interpretation: str


class GeometryPrimesService:
    """Service for semantic prime analysis.

    Wraps the NSM semantic primes domain logic for CLI and MCP consumption.
    """

    def __init__(self) -> None:
        self._inventory = SemanticPrimeInventory.english2014()

    def list_primes(self) -> list[PrimeInfo]:
        """List all semantic primes in the inventory.

        Returns:
            List of PrimeInfo with id, name, category, and exponents.
        """
        return [
            PrimeInfo(
                id=prime.id,
                name=prime.canonical_english,
                category=prime.category.value,
                exponents=prime.english_exponents,
            )
            for prime in self._inventory
        ]

    def probe(self, model_path: str) -> list[PrimeActivation]:
        """Probe a model for semantic prime activation patterns.

        This is a placeholder that requires model inference infrastructure
        to actually measure activations. Currently returns uniform activations
        as a baseline.

        Args:
            model_path: Path to the model directory.

        Returns:
            List of PrimeActivation for each prime.

        Raises:
            ValueError: If the model path is invalid.
        """
        from pathlib import Path

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not path.is_dir():
            raise ValueError(f"Model path is not a directory: {model_path}")

        # Return baseline activations - real implementation would use
        # the inference engine to measure actual activation patterns
        return [
            PrimeActivation(
                prime_id=prime.id,
                activation_strength=0.5,  # Baseline neutral activation
                layer_activations=[0.5] * 32,  # Placeholder for layer-wise activations
            )
            for prime in self._inventory
        ]

    def compare(self, model_a: str, model_b: str) -> PrimeComparisonResult:
        """Compare semantic prime alignment between two models.

        This is a placeholder that requires model inference infrastructure
        to actually measure and compare activations. Currently returns
        a neutral comparison result.

        Args:
            model_a: Path to the first model.
            model_b: Path to the second model.

        Returns:
            PrimeComparisonResult with alignment metrics.

        Raises:
            ValueError: If either model path is invalid.
        """
        from pathlib import Path

        path_a = Path(model_a).expanduser().resolve()
        path_b = Path(model_b).expanduser().resolve()

        if not path_a.exists():
            raise ValueError(f"Model A path does not exist: {model_a}")
        if not path_a.is_dir():
            raise ValueError(f"Model A path is not a directory: {model_a}")
        if not path_b.exists():
            raise ValueError(f"Model B path does not exist: {model_b}")
        if not path_b.is_dir():
            raise ValueError(f"Model B path is not a directory: {model_b}")

        # Placeholder comparison - real implementation would:
        # 1. Probe both models for prime activations
        # 2. Compare activation patterns using cosine similarity
        # 3. Identify divergent and convergent primes
        return PrimeComparisonResult(
            alignment_score=0.85,  # Placeholder
            divergent_primes=[],
            convergent_primes=[p.id for p in self._inventory[:5]],
            interpretation=(
                "Comparison requires inference engine integration. "
                "This is a placeholder result showing the comparison structure."
            ),
        )
