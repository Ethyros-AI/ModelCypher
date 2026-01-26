# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Self-Consistency Probing - the questions a model asks itself.

When humans think deeply, they:
1. Ask "What does this imply?"
2. Ask "What would make this false?"
3. Ask "How does this connect to other things I know?"

This module provides the same probing for models. The model generates
its own questions and answers, building coherent context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    pass


@dataclass
class ProbeResult:
    """Result of a self-consistency probe."""

    original_statement: str
    implications: List[str]
    contradictions: List[str]
    connections: Dict[str, str]  # concept -> relationship

    # Representations for measuring consistency
    original_representation: Optional["Array"] = None
    implication_representations: List["Array"] = field(default_factory=list)


class SelfConsistencyProber:
    """Probe a model's understanding through self-questioning.

    This is NOT about measuring consistency yet - that comes later.
    This is about generating the questions and answers that a model
    needs to "think" about a topic.

    Usage:
        prober = SelfConsistencyProber(model, tokenizer, get_activations)

        result = prober.probe("2 + 2 = 4")
        # result.implications = ["4 - 2 = 2", "2 + 2 is not 5", ...]
        # result.contradictions = ["2 + 2 = 5", "4 = 3", ...]
    """

    def __init__(
        self,
        model,
        tokenizer,
        get_activations: Callable[[str], "Array"],
        max_tokens: int = 50,
    ):
        """Initialize the prober.

        Args:
            model: The language model
            tokenizer: Tokenizer for the model
            get_activations: Function to get activations for a string
            max_tokens: Max tokens for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.get_activations = get_activations
        self.max_tokens = max_tokens

    def generate(self, prompt: str, n_completions: int = 1) -> List[str]:
        """Generate completions for a prompt."""
        import mlx.core as mx

        completions = []

        for _ in range(n_completions):
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            generated = []
            current = input_ids

            for _ in range(self.max_tokens):
                logits = self.model(current)
                mx.eval(logits)

                # Sample with temperature for diversity
                probs = mx.softmax(logits[0, -1, :] / 0.7, axis=-1)
                next_token = int(mx.argmax(probs).item())

                if next_token == self.tokenizer.eos_token_id:
                    break

                # Stop at period or newline
                generated.append(next_token)
                text_so_far = self.tokenizer.decode(generated)
                if '.' in text_so_far or '\n' in text_so_far:
                    break

                current = mx.concatenate([current, mx.array([[next_token]])], axis=1)

            completion = self.tokenizer.decode(generated).strip()
            if completion:
                completions.append(completion.split('.')[0].strip())

        return completions

    def probe_implications(self, statement: str, n: int = 3) -> List[str]:
        """What must be true if this statement is true?

        Args:
            statement: The statement to probe
            n: Number of implications to generate

        Returns:
            List of implied statements
        """
        prompts = [
            f"If '{statement}' is true, then it must also be true that",
            f"Given that '{statement}', we can conclude that",
            f"'{statement}' implies that",
        ]

        implications = []
        for prompt in prompts[:n]:
            completions = self.generate(prompt, n_completions=1)
            implications.extend(completions)

        return implications

    def probe_contradictions(self, statement: str, n: int = 3) -> List[str]:
        """What would contradict or falsify this statement?

        Args:
            statement: The statement to probe
            n: Number of contradictions to generate

        Returns:
            List of contradicting statements
        """
        prompts = [
            f"'{statement}' would be false if",
            f"The opposite of '{statement}' is",
            f"'{statement}' is contradicted by",
        ]

        contradictions = []
        for prompt in prompts[:n]:
            completions = self.generate(prompt, n_completions=1)
            contradictions.extend(completions)

        return contradictions

    def probe_connections(
        self,
        statement: str,
        concepts: List[str],
    ) -> Dict[str, str]:
        """How does this statement relate to other concepts?

        Args:
            statement: The statement to probe
            concepts: Concepts to connect to

        Returns:
            Dict mapping concept to relationship
        """
        connections = {}

        for concept in concepts:
            prompt = f"The relationship between '{statement}' and '{concept}' is that"
            completions = self.generate(prompt, n_completions=1)
            if completions:
                connections[concept] = completions[0]

        return connections

    def full_probe(
        self,
        statement: str,
        related_concepts: Optional[List[str]] = None,
        capture_representations: bool = True,
    ) -> ProbeResult:
        """Run a complete probe on a statement.

        Args:
            statement: The statement to probe
            related_concepts: Optional concepts to connect to
            capture_representations: Whether to capture activations

        Returns:
            ProbeResult with all probe results
        """
        implications = self.probe_implications(statement)
        contradictions = self.probe_contradictions(statement)

        connections = {}
        if related_concepts:
            connections = self.probe_connections(statement, related_concepts)

        result = ProbeResult(
            original_statement=statement,
            implications=implications,
            contradictions=contradictions,
            connections=connections,
        )

        if capture_representations:
            result.original_representation = self.get_activations(statement)
            result.implication_representations = [
                self.get_activations(impl) for impl in implications
            ]

        return result


__all__ = ["SelfConsistencyProber", "ProbeResult"]
