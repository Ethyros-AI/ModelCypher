"""Reference embeddings for cross-modal CKA comparison.

Extracts embeddings from known information-encoding modalities (CLIP, Whisper)
to compare geometry with FRB features.

Uses ModelCypher's existing MultiModalEmbeddingExtractor infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

    Array = np.ndarray


@dataclass
class ReferenceEmbeddings:
    """Container for reference modality embeddings."""

    embeddings: "Array"  # [N, D] embedding matrix
    modality: str  # "clip", "whisper", or "llm"
    model_name: str  # Model identifier
    concepts: tuple[str, ...]  # Concept strings used
    hidden_dim: int  # Embedding dimension


# Diverse concept set covering different semantic domains
# These are used to generate embeddings from CLIP/Whisper for comparison
DEFAULT_CONCEPTS = [
    # Natural phenomena
    "a bright flash of light",
    "radio waves from space",
    "electromagnetic burst",
    "cosmic signal pattern",
    "interstellar transmission",
    # Physical concepts
    "energy dispersing through plasma",
    "frequency spectrum shifting",
    "temporal pulse structure",
    "polarized light wave",
    "magnetic field distortion",
    # Abstract/technical
    "encoded information pattern",
    "structured data sequence",
    "noise versus signal",
    "coherent emission source",
    "random interference",
    # Astronomy
    "distant galaxy",
    "neutron star pulse",
    "magnetar burst",
    "cosmic microwave background",
    "gravitational wave ripple",
    # Additional variety
    "compressed digital signal",
    "acoustic waveform",
    "visual pattern recognition",
    "spectral fingerprint",
    "quantum fluctuation",
    # More natural
    "lightning strike",
    "aurora borealis",
    "solar flare eruption",
    "meteor shower",
    "tectonic shift",
    # Technology
    "radar ping echo",
    "sonar reflection",
    "satellite transmission",
    "fiber optic pulse",
    "wireless communication",
    # Abstract patterns
    "fractal structure",
    "self-similar repetition",
    "harmonic resonance",
    "phase transition",
    "entropy gradient",
    # Additional astronomy
    "pulsar rotation",
    "black hole accretion",
    "stellar wind",
    "supernova remnant",
    "dark matter halo",
    # More variety
    "earthquake tremor",
    "ocean wave pattern",
    "wind turbulence",
    "cloud formation",
    "crystal lattice",
]


def extract_clip_embeddings(
    n_samples: int,
    backend: "Backend",
    model_name: str = "openai/clip-vit-base-patch32",
) -> ReferenceEmbeddings:
    """Extract CLIP text embeddings for comparison with FRB features.

    Args:
        n_samples: Number of embedding samples to generate
        backend: Backend instance
        model_name: CLIP model identifier

    Returns:
        ReferenceEmbeddings with shape [n_samples, 512]
    """
    from modelcypher.adapters.multimodal_embedding_extractor import (
        MultiModalEmbeddingExtractor,
    )

    # Select concepts to match sample count
    concepts = list(DEFAULT_CONCEPTS[:n_samples])

    # Pad with variations if we need more samples
    while len(concepts) < n_samples:
        idx = len(concepts) % len(DEFAULT_CONCEPTS)
        base = DEFAULT_CONCEPTS[idx]
        concepts.append(f"another {base}")

    extractor = MultiModalEmbeddingExtractor(backend)
    result = extractor.extract_clip(concepts, model_name=model_name)

    return ReferenceEmbeddings(
        embeddings=result.embeddings,
        modality="clip",
        model_name=model_name,
        concepts=tuple(concepts),
        hidden_dim=result.hidden_dim,
    )


def extract_whisper_embeddings(
    n_samples: int,
    backend: "Backend",
    model_name: str = "openai/whisper-base",
) -> ReferenceEmbeddings:
    """Extract Whisper decoder embeddings for comparison with FRB features.

    Note: Uses text as proxy for audio concepts. This tests whether the
    geometric structure of Whisper's embedding space aligns with FRB features.

    Args:
        n_samples: Number of embedding samples to generate
        backend: Backend instance
        model_name: Whisper model identifier

    Returns:
        ReferenceEmbeddings with shape [n_samples, 512]
    """
    from modelcypher.adapters.multimodal_embedding_extractor import (
        MultiModalEmbeddingExtractor,
    )

    # Select concepts to match sample count
    concepts = list(DEFAULT_CONCEPTS[:n_samples])

    # Pad with variations if we need more samples
    while len(concepts) < n_samples:
        idx = len(concepts) % len(DEFAULT_CONCEPTS)
        base = DEFAULT_CONCEPTS[idx]
        concepts.append(f"the sound of {base}")

    extractor = MultiModalEmbeddingExtractor(backend)
    result = extractor.extract_whisper(concepts, model_name=model_name)

    return ReferenceEmbeddings(
        embeddings=result.embeddings,
        modality="whisper",
        model_name=model_name,
        concepts=tuple(concepts),
        hidden_dim=result.hidden_dim,
    )


def generate_synthetic_embeddings(
    n_samples: int,
    dim: int,
    embedding_type: str,
    backend: "Backend",
    seed: int = 42,
) -> ReferenceEmbeddings:
    """Generate synthetic embeddings for baseline comparison.

    Args:
        n_samples: Number of samples
        dim: Embedding dimension
        embedding_type: Type of embedding:
            - "random": IID Gaussian (baseline, CKA ~ 0 expected)
            - "structured": Low-rank structure (simulates semantic manifold)
        backend: Backend instance
        seed: Random seed

    Returns:
        ReferenceEmbeddings with synthetic data
    """
    rng = np.random.default_rng(seed)

    if embedding_type == "random":
        embeddings = rng.standard_normal((n_samples, dim)).astype(np.float32)

    elif embedding_type == "structured":
        # Low-rank structure simulating semantic manifold
        intrinsic_dim = min(10, dim // 2)
        latent = rng.standard_normal((n_samples, intrinsic_dim)).astype(np.float32)
        projection = rng.standard_normal((intrinsic_dim, dim)).astype(np.float32)
        embeddings = latent @ projection
        # Add small noise
        embeddings += 0.1 * rng.standard_normal((n_samples, dim)).astype(np.float32)

    else:
        msg = f"Unknown embedding type: {embedding_type}"
        raise ValueError(msg)

    return ReferenceEmbeddings(
        embeddings=backend.array(embeddings),
        modality=f"synthetic_{embedding_type}",
        model_name=f"synthetic_{embedding_type}_seed{seed}",
        concepts=tuple(f"synthetic_{i}" for i in range(n_samples)),
        hidden_dim=dim,
    )
