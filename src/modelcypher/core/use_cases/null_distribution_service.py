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

"""Null distribution generation for statistical hypothesis testing.

Generates frequency-matched random word sets for comparison against semantic primes.
Used for Paper 1: Invariant Semantic Structure Across Language Model Families.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass
class NullSample:
    """A single null distribution sample."""

    sample_id: int
    words: list[str]
    cka_values: dict[str, float] = field(default_factory=dict)  # "modelA_modelB" -> CKA


@dataclass
class NullDistributionResult:
    """Complete null distribution results."""

    n_samples: int
    n_words_per_sample: int
    samples: list[NullSample]
    aggregate_cka_mean: float
    aggregate_cka_std: float
    model_pairs: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_samples": self.n_samples,
            "n_words_per_sample": self.n_words_per_sample,
            "aggregate_cka_mean": self.aggregate_cka_mean,
            "aggregate_cka_std": self.aggregate_cka_std,
            "model_pairs": self.model_pairs,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "words": s.words,
                    "cka_values": s.cka_values,
                }
                for s in self.samples
            ],
        }


class NullDistributionService:
    """Service for generating null distributions for hypothesis testing.

    Generates random word sets matched to the size of the semantic prime inventory,
    then computes CKA across model pairs for statistical comparison.
    """

    def __init__(self, backend: Backend | None = None):
        """Initialize the service.

        Args:
            backend: Optional backend for array operations.
        """
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend

            backend = get_default_backend()
        self._backend = backend

    def get_vocabulary_intersection(
        self,
        tokenizers: list[Any],
        min_word_length: int = 2,
        max_word_length: int = 15,
    ) -> list[str]:
        """Get vocabulary intersection across multiple tokenizers.

        Args:
            tokenizers: List of tokenizers from different models.
            min_word_length: Minimum word length to include.
            max_word_length: Maximum word length to include.

        Returns:
            List of words present in all vocabularies.
        """
        if not tokenizers:
            return []

        # Get vocabulary from first tokenizer
        vocab_sets = []
        for tok in tokenizers:
            vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
            # Filter to clean single words (no special tokens, subwords, etc.)
            words = {
                word
                for word in vocab.keys()
                if (
                    word.isalpha()
                    and min_word_length <= len(word) <= max_word_length
                    and not word.startswith("##")
                    and not word.startswith("▁")
                    and not word.startswith("<")
                    and word.islower()
                )
            }
            vocab_sets.append(words)

        # Intersection across all
        if not vocab_sets:
            return []

        intersection = vocab_sets[0]
        for vocab in vocab_sets[1:]:
            intersection = intersection & vocab

        return sorted(intersection)

    def sample_random_words(
        self,
        vocabulary: list[str],
        n_words: int,
        n_samples: int,
        seed: int | None = None,
        exclude_words: set[str] | None = None,
    ) -> list[list[str]]:
        """Sample multiple random word sets from vocabulary.

        Args:
            vocabulary: Available words to sample from.
            n_words: Number of words per sample (should match prime count).
            n_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
            exclude_words: Words to exclude (e.g., semantic primes).

        Returns:
            List of word lists, each containing n_words.
        """
        if seed is not None:
            random.seed(seed)

        # Filter excluded words
        available = vocabulary
        if exclude_words:
            available = [w for w in vocabulary if w not in exclude_words]

        if len(available) < n_words:
            raise ValueError(
                f"Vocabulary ({len(available)}) too small for {n_words} words per sample"
            )

        samples = []
        for _ in range(n_samples):
            sample = random.sample(available, n_words)
            samples.append(sorted(sample))

        return samples

    def generate_null_samples(
        self,
        vocabulary: list[str],
        n_words: int,
        n_samples: int,
        seed: int = 42,
        exclude_words: set[str] | None = None,
    ) -> list[NullSample]:
        """Generate null samples without CKA computation.

        Args:
            vocabulary: Available words to sample from.
            n_words: Number of words per sample.
            n_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
            exclude_words: Words to exclude from sampling.

        Returns:
            List of NullSample objects (without CKA values filled in).
        """
        word_samples = self.sample_random_words(
            vocabulary=vocabulary,
            n_words=n_words,
            n_samples=n_samples,
            seed=seed,
            exclude_words=exclude_words,
        )

        return [NullSample(sample_id=i, words=words) for i, words in enumerate(word_samples)]

    def save_null_samples(
        self,
        samples: list[NullSample],
        output_dir: Path,
    ) -> None:
        """Save null samples to disk.

        Args:
            samples: List of null samples to save.
            output_dir: Directory to save samples in.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each sample as individual file (for parallel CKA computation)
        for sample in samples:
            sample_file = output_dir / f"sample_{sample.sample_id:04d}.json"
            with open(sample_file, "w") as f:
                json.dump(
                    {
                        "sample_id": sample.sample_id,
                        "words": sample.words,
                        "cka_values": sample.cka_values,
                    },
                    f,
                    indent=2,
                )

        # Save index file
        index_file = output_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump(
                {
                    "n_samples": len(samples),
                    "n_words_per_sample": len(samples[0].words) if samples else 0,
                    "sample_ids": [s.sample_id for s in samples],
                },
                f,
                indent=2,
            )

    def load_null_samples(self, samples_dir: Path) -> list[NullSample]:
        """Load null samples from disk.

        Args:
            samples_dir: Directory containing sample files.

        Returns:
            List of NullSample objects.
        """
        samples_dir = Path(samples_dir)
        index_file = samples_dir / "index.json"

        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(index_file) as f:
            index = json.load(f)

        samples = []
        for sample_id in index["sample_ids"]:
            sample_file = samples_dir / f"sample_{sample_id:04d}.json"
            with open(sample_file) as f:
                data = json.load(f)
                samples.append(
                    NullSample(
                        sample_id=data["sample_id"],
                        words=data["words"],
                        cka_values=data.get("cka_values", {}),
                    )
                )

        return samples

    def aggregate_results(
        self,
        samples: list[NullSample],
        model_pairs: list[str],
    ) -> NullDistributionResult:
        """Aggregate null distribution results.

        Args:
            samples: List of null samples with CKA values computed.
            model_pairs: List of model pair names (e.g., ["qwen05_qwen15", ...]).

        Returns:
            NullDistributionResult with aggregate statistics.
        """
        from modelcypher.core.support.statistics import mean, standard_deviation

        # Collect all CKA values
        all_cka = []
        for sample in samples:
            for pair in model_pairs:
                if pair in sample.cka_values:
                    all_cka.append(sample.cka_values[pair])

        if not all_cka:
            return NullDistributionResult(
                n_samples=len(samples),
                n_words_per_sample=len(samples[0].words) if samples else 0,
                samples=samples,
                aggregate_cka_mean=0.0,
                aggregate_cka_std=0.0,
                model_pairs=model_pairs,
            )

        cka_mean = mean(all_cka)
        cka_std = standard_deviation(all_cka, cka_mean)

        return NullDistributionResult(
            n_samples=len(samples),
            n_words_per_sample=len(samples[0].words) if samples else 0,
            samples=samples,
            aggregate_cka_mean=cka_mean,
            aggregate_cka_std=cka_std,
            model_pairs=model_pairs,
        )
