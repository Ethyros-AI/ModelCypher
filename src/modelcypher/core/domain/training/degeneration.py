"""Degeneration measurement for behavioral coherence (G4).

Measures n-gram repetition rate as a proxy for degenerate text generation.
Degenerate models produce highly repetitive output — elevated n-gram
repetition rate indicates the model is stuck in a loop.

The n-gram order is derived from the birthday paradox on the readout layer's
Shannon effective rank.  For T words of generated text and r_eff effective
output modes: n = ceil(2 * log(T) / log(r_eff)).  This is the smallest n
where expected random n-gram collisions in non-degenerate text are negligible
(T^2 / (2 * r_eff^n) < 1).  The derivation treats greedy-decoding positions
as approximately conditionally independent given the hidden-state trajectory.

Validation against known models:
    r_eff ~ 50  (small model bottleneck):  n = 4
    r_eff ~ 100 (mid-size):               n = 3
    r_eff ~ 200 (large):                  n = 3
    r_eff ~ 10  (severely constrained):   n = 6
"""

from __future__ import annotations

import math


def derive_ngram_order(
    readout_effective_rank: float,
    generation_length_words: int,
) -> int:
    """Derive n-gram order from birthday paradox on readout effective rank.

    Given r_eff effective output modes and T words of generated text,
    selects smallest n where expected random n-gram collisions are negligible:
    T^2 / (2 * r_eff^n) < 1  =>  n = ceil(2 * log(T) / log(r_eff)).

    Args:
        readout_effective_rank: Shannon effective rank of readout weight matrix.
        generation_length_words: Approximate word count per generated response.
    """
    if readout_effective_rank < 2.0:
        return 2
    n = math.ceil(
        2.0 * math.log(generation_length_words) / math.log(readout_effective_rank)
    )
    return max(2, n)


def ngram_repetition_rate(text: str, n: int) -> float:
    """Fraction of n-grams in *text* that are repeated.

    Returns 0.0 for texts shorter than *n* words.
    """
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - unique / len(ngrams)
