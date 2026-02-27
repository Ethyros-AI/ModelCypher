"""Degeneration measurement for behavioral coherence (G4).

Measures n-gram repetition rate as a proxy for degenerate text generation.
Degenerate models produce highly repetitive output — elevated n-gram
repetition rate indicates the model is stuck in a loop.

IMPORTANT: The n-gram window n=4 is NOT derived from geometry. This module
is a DIAGNOSTIC, not a decision boundary. Do not use these measurements as
gates until the window is derived.

TODO(jk): derive n-gram window from measured trajectory geometry instead
of fixing n=4.  Candidate derivation: the effective rank of the activation
covariance at the readout layer determines the number of distinct output
patterns — this may constrain the n-gram order at which repetition becomes
detectable.
"""

from __future__ import annotations


def fourgram_repetition_rate(text: str) -> float:
    """Fraction of 4-grams in text that are repeated.

    Returns 0.0 for texts shorter than 4 words.

    The n=4 window is diagnostic only — not a decision boundary.
    TODO(jk): derive n-gram window from measured trajectory geometry.
    """
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - unique / len(ngrams)
