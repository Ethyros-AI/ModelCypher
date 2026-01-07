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

"""Token-based atlas for 100% dimension coverage.

Instead of 963 curated conceptual probes, this atlas generates probes from
every token in a model's vocabulary. This guarantees full-rank alignment
for any hidden dimension size:

- SmolLM vocab: 49,152 tokens -> 49K probes
- Qwen vocab: 151,936 tokens -> 151K probes

With 49K+ probes, we can achieve full-rank F for any model up to ~49K dims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.domains import AtlasDomain
from modelcypher.core.domain.agents.unified_atlas import AtlasProbe, AtlasSource

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenProbe:
    """A probe generated from a vocabulary token.
    
    Each token in the vocabulary becomes a probe with:
    - Single-token input (just the token itself)
    - The token's text representation as the probe name
    """
    token_id: int
    token_text: str
    
    @property
    def probe_text(self) -> str:
        """The text to feed to the model (just the token)."""
        return self.token_text
    
    def to_atlas_probe(self) -> AtlasProbe:
        """Convert to unified AtlasProbe format."""
        # Use TOKEN source (we'll add this to AtlasSource)
        # For now, use DOMAIN_SPECIFIC as placeholder
        return AtlasProbe(
            id=f"token_{self.token_id}",
            source=AtlasSource.DOMAIN_SPECIFIC,  # TODO: Add TOKEN source
            domain=AtlasDomain.FACTUAL,  # Tokens map to factual knowledge
            name=f"token:{self.token_text[:20]}",
            description=f"Vocabulary token {self.token_id}",
            cross_domain_weight=1.0,
            category_name="vocabulary",
            support_texts=(self.token_text,),
        )


def generate_token_probes(
    tokenizer,
    max_probes: int | None = None,
    min_token_id: int = 0,
) -> list[TokenProbe]:
    """Generate probes from all vocabulary tokens.
    
    Args:
        tokenizer: HuggingFace-compatible tokenizer with vocab
        max_probes: Optional limit on number of probes (for testing)
        min_token_id: Skip tokens below this ID (often special tokens)
    
    Returns:
        List of TokenProbe objects
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    logger.info("TOKEN ATLAS: Generating probes from %d vocabulary tokens", vocab_size)
    
    # Sort by token ID for deterministic ordering
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    
    probes = []
    for token_text, token_id in sorted_tokens:
        if token_id < min_token_id:
            continue
        if max_probes and len(probes) >= max_probes:
            break
            
        # Skip empty or whitespace-only tokens
        if not token_text or token_text.isspace():
            continue
            
        probes.append(TokenProbe(token_id=token_id, token_text=token_text))
    
    logger.info("TOKEN ATLAS: Generated %d probes", len(probes))
    return probes


def get_probe_texts(probes: list[TokenProbe]) -> list[str]:
    """Get probe texts for batched model inference.
    
    Returns list of single-token strings to feed to model.
    """
    return [p.probe_text for p in probes]


def get_probe_ids(probes: list[TokenProbe]) -> list[str]:
    """Get probe IDs for activation storage."""
    return [f"token_{p.token_id}" for p in probes]
