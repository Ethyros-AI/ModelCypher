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

"""
Number Probe - Collects LLM representations of numbers for geometric analysis.

This module extracts how LLMs represent numbers in their hidden states,
enabling geometric analysis of number-theoretic properties (primes, composites, etc.).

Usage:
    from modelcypher.core.domain.geometry.number_probe import NumberProbe

    probe = NumberProbe(backend)
    result = probe.collect_number_representations(
        model, tokenizer, numbers=[2, 3, 4, 5, 6, 7],
        prompt_format="bare", layers="middle"
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class PromptFormat(Enum):
    """Prompt formats for probing number representations."""

    BARE = "bare"  # Just the number: "17"
    MATH = "math"  # Mathematical context: "In mathematics, 17 is"
    NEUTRAL = "neutral"  # Neutral: "Consider the integer 17:"
    WORD = "word"  # Spelled out: "seventeen"


class LayerSelection(Enum):
    """Layer selection strategies."""

    EARLY = "early"  # First quarter of layers
    MIDDLE = "middle"  # Middle half of layers
    LATE = "late"  # Last quarter of layers
    ALL = "all"  # All layers
    KEY = "key"  # Key positions: 0%, 25%, 50%, 75%, 100%


@dataclass(frozen=True)
class NumberRepresentation:
    """Representation of a single number across layers."""

    number: int
    prompt: str
    activations: dict[int, "Array"]  # layer_idx -> hidden state


@dataclass
class NumberProbeResult:
    """Result of probing a set of numbers."""

    numbers: list[int]
    prompt_format: PromptFormat
    layer_selection: LayerSelection
    representations: list[NumberRepresentation]
    model_name: str
    hidden_dim: int
    n_layers: int

    # Convenience methods
    def get_layer_matrix(self, layer: int, backend: "Backend | None" = None) -> "Array":
        """Get activation matrix [n_numbers, hidden_dim] at a specific layer."""
        backend = backend or get_default_backend()
        activations = []
        for rep in self.representations:
            if layer in rep.activations:
                activations.append(rep.activations[layer])
        if not activations:
            raise ValueError(f"No activations found for layer {layer}")
        return backend.stack(activations, axis=0)

    def get_middle_layer(self) -> int:
        """Get the middle layer index."""
        return self.n_layers // 2


# Number words for English (0-100)
_NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
    30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
    70: "seventy", 80: "eighty", 90: "ninety", 100: "hundred",
}


def _number_to_word(n: int) -> str:
    """Convert number to English word (supports 0-100)."""
    if n in _NUMBER_WORDS:
        return _NUMBER_WORDS[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return f"{_NUMBER_WORDS[tens]}-{_NUMBER_WORDS[ones]}"
    # For numbers > 100, just use the numeral
    return str(n)


def _format_prompt(n: int, fmt: PromptFormat) -> str:
    """Format a number into a prompt string."""
    if fmt == PromptFormat.BARE:
        return str(n)
    elif fmt == PromptFormat.MATH:
        return f"In mathematics, {n} is"
    elif fmt == PromptFormat.NEUTRAL:
        return f"Consider the integer {n}:"
    elif fmt == PromptFormat.WORD:
        return f"the number {_number_to_word(n)}"
    else:
        return str(n)


def _get_layer_indices(
    n_layers: int,
    selection: LayerSelection,
) -> list[int]:
    """Get layer indices based on selection strategy."""
    if selection == LayerSelection.ALL:
        return list(range(n_layers))
    elif selection == LayerSelection.EARLY:
        return list(range(0, n_layers // 4))
    elif selection == LayerSelection.MIDDLE:
        return list(range(n_layers // 4, 3 * n_layers // 4))
    elif selection == LayerSelection.LATE:
        return list(range(3 * n_layers // 4, n_layers))
    elif selection == LayerSelection.KEY:
        positions = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        return sorted(set(positions))
    else:
        return list(range(n_layers))


class NumberProbe:
    """
    Probes LLMs to extract number representations for geometric analysis.

    This is the core data collection tool for the prime geometry experiment.
    It extracts hidden state representations of numbers, enabling comparison
    of how models encode primes vs composites.
    """

    def __init__(self, backend: "Backend | None" = None):
        """Initialize the number probe."""
        self.backend = backend or get_default_backend()

    def collect_number_representations(
        self,
        model: Any,
        tokenizer: Any,
        numbers: list[int],
        prompt_format: PromptFormat | str = PromptFormat.BARE,
        layer_selection: LayerSelection | str = LayerSelection.MIDDLE,
        model_name: str = "unknown",
    ) -> NumberProbeResult:
        """
        Collect LLM representations for a list of numbers.

        Args:
            model: The LLM model (mlx_lm format).
            tokenizer: The tokenizer.
            numbers: List of integers to probe.
            prompt_format: How to format numbers as prompts.
            layer_selection: Which layers to extract.
            model_name: Name for logging/results.

        Returns:
            NumberProbeResult with representations for all numbers.
        """
        import mlx.core as mx

        # Handle string enum values
        if isinstance(prompt_format, str):
            prompt_format = PromptFormat(prompt_format)
        if isinstance(layer_selection, str):
            layer_selection = LayerSelection(layer_selection)

        # Get model info
        n_layers = self._get_n_layers(model)
        hidden_dim = self._get_hidden_dim(model)
        target_layers = set(_get_layer_indices(n_layers, layer_selection))

        logger.info(
            f"Probing {len(numbers)} numbers with format={prompt_format.value}, "
            f"layers={layer_selection.value} ({len(target_layers)} layers)"
        )

        representations = []

        for i, n in enumerate(numbers):
            prompt = _format_prompt(n, prompt_format)

            try:
                activations = self._collect_single(
                    model, tokenizer, prompt, target_layers
                )

                rep = NumberRepresentation(
                    number=n,
                    prompt=prompt,
                    activations=activations,
                )
                representations.append(rep)

                if (i + 1) % 50 == 0:
                    logger.info(f"  Collected {i + 1}/{len(numbers)} numbers...")

            except Exception as e:
                logger.warning(f"Failed to collect representation for {n}: {e}")
                continue

        logger.info(f"Collected {len(representations)}/{len(numbers)} representations")

        return NumberProbeResult(
            numbers=numbers,
            prompt_format=prompt_format,
            layer_selection=layer_selection,
            representations=representations,
            model_name=model_name,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    def _collect_single(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        target_layers: set[int],
    ) -> dict[int, "Array"]:
        """Collect activations for a single prompt."""
        import mlx.core as mx

        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids)
        input_ids = mx.array([token_ids])

        activations: dict[int, "Array"] = {}

        # Forward pass with layer extraction
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Standard mlx_lm model structure
            if hasattr(model.model, "embed_tokens"):
                h = model.model.embed_tokens(input_ids)
            elif hasattr(model.model, "wte"):
                h = model.model.wte(input_ids)
            else:
                h = model.embed(input_ids) if hasattr(model, "embed") else None

            if h is not None:
                for layer_idx, layer in enumerate(model.model.layers):
                    result = layer(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result

                    if layer_idx in target_layers:
                        # Mean-pool over sequence length to get [hidden_dim]
                        pooled = mx.mean(h, axis=(0, 1))
                        # Convert to float32 for numerical stability
                        pooled = pooled.astype(mx.float32)
                        mx.eval(pooled)
                        activations[layer_idx] = pooled

        return activations

    def _get_n_layers(self, model: Any) -> int:
        """Get number of layers in the model."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        return 1

    def _get_hidden_dim(self, model: Any) -> int:
        """Get hidden dimension of the model."""
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "embed_tokens"):
                return inner.embed_tokens.weight.shape[1]
            elif hasattr(inner, "wte"):
                return inner.wte.weight.shape[1]
        return 0


# =============================================================================
# Number Set Generators
# =============================================================================


def generate_primes(n: int, backend: "Backend | None" = None) -> list[int]:
    """
    Generate the first n prime numbers.

    Uses Sieve of Eratosthenes for efficiency.
    """
    from modelcypher.core.domain.geometry.prime_geometry import generate_primes as _gen_primes

    backend = backend or get_default_backend()
    prime_seq = _gen_primes(n, backend)

    # Convert to Python list
    primes_list = backend.tolist(prime_seq.primes)
    return [int(p) for p in primes_list]


def generate_composites_matched(primes: list[int]) -> list[int]:
    """
    Generate composite numbers matched by magnitude to primes.

    For each prime p, finds a composite in the range [p-2, p+2] if possible,
    otherwise the nearest composite.
    """
    prime_set = set(primes)
    composites = []

    for p in primes:
        # Look for nearest composite
        for offset in [1, -1, 2, -2, 3, -3]:
            candidate = p + offset
            if candidate > 1 and candidate not in prime_set:
                composites.append(candidate)
                break
        else:
            # Fallback: p+1 is always composite for p > 2
            composites.append(p + 1)

    return composites


def generate_composites_adjacent(primes: list[int]) -> list[int]:
    """Generate composites that are p+1 for each prime p (except p=2)."""
    return [p + 1 for p in primes if p > 2]


def generate_twin_primes(limit: int) -> tuple[list[int], list[int]]:
    """
    Generate twin prime pairs up to limit.

    Returns (smaller_primes, larger_primes) where each pair (smaller, larger)
    satisfies larger = smaller + 2 and both are prime.
    """
    # Generate enough primes
    primes = generate_primes(limit // 2)  # Rough estimate
    primes = [p for p in primes if p <= limit]

    prime_set = set(primes)
    smaller = []
    larger = []

    for p in primes:
        if p + 2 in prime_set:
            smaller.append(p)
            larger.append(p + 2)

    return smaller, larger


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


@dataclass
class NumberSetConfig:
    """Configuration for generating number sets for experiments."""

    n_primes: int = 100
    include_composites_matched: bool = True
    include_composites_adjacent: bool = False
    include_twin_primes: bool = False
    seed: int = 42


@dataclass
class NumberSets:
    """Collection of number sets for experiments."""

    primes: list[int] = field(default_factory=list)
    composites_matched: list[int] = field(default_factory=list)
    composites_adjacent: list[int] = field(default_factory=list)
    twin_primes_smaller: list[int] = field(default_factory=list)
    twin_primes_larger: list[int] = field(default_factory=list)

    @classmethod
    def generate(cls, config: NumberSetConfig) -> "NumberSets":
        """Generate all number sets based on config."""
        primes = generate_primes(config.n_primes)

        result = cls(primes=primes)

        if config.include_composites_matched:
            result.composites_matched = generate_composites_matched(primes)

        if config.include_composites_adjacent:
            result.composites_adjacent = generate_composites_adjacent(primes)

        if config.include_twin_primes:
            max_prime = max(primes) if primes else 100
            smaller, larger = generate_twin_primes(max_prime)
            result.twin_primes_smaller = smaller
            result.twin_primes_larger = larger

        return result
