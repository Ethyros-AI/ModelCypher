#!/usr/bin/env python3
"""Cross-modal CKA scale analysis.

Tests the Anna Karenina hypothesis: does cross-modal CKA increase with model capability?

References:
- Huh et al. (2024) "The Platonic Representation Hypothesis" arXiv:2405.07987
- ModelCypher validation: docs/research/multi_modal_cka_validation.md
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations: (name, suffix, params)
DEFAULT_MODELS = [
    ("LFM2-350M", "LFM2-350M-MLX-bf16", 350_000_000),
    ("LFM2-700M", "LFM2-700M-bf16", 700_000_000),
    ("LFM2-1.2B", "LFM2-1.2B-bf16", 1_200_000_000),
    ("Qwen2.5-3B", "Qwen2.5-3B-Instruct-bf16", 3_000_000_000),
    ("Qwen3-8B", "Qwen3-8B-bf16", 8_000_000_000),
]

# Standard probe concepts for cross-modal comparison
# These should cover diverse semantic domains
PROBE_CONCEPTS = [
    # Colors
    "red", "blue", "green", "yellow", "purple", "orange", "black", "white",
    # Animals
    "dog", "cat", "bird", "fish", "elephant", "lion", "snake", "butterfly",
    # Objects
    "car", "house", "tree", "book", "phone", "chair", "table", "computer",
    # Actions
    "running", "jumping", "eating", "sleeping", "dancing", "swimming", "flying", "singing",
    # Emotions
    "happy", "sad", "angry", "surprised", "scared", "love", "peace", "joy",
    # Nature
    "mountain", "ocean", "forest", "desert", "river", "sky", "sun", "moon",
    # Abstract
    "time", "space", "music", "art", "science", "knowledge", "truth", "beauty",
]

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "cross_modal"


@dataclass
class ScaleResult:
    """Result for a single model."""

    model_name: str
    params: int
    cka_vision: float
    cka_audio: float
    hidden_dim: int


@dataclass
class ScaleAnalysisResults:
    """Complete scale analysis results."""

    models: list[ScaleResult]
    probe_concepts: list[str]
    vision_model: str
    audio_model: str


def extract_llm_embeddings(
    model_path: Path,
    concepts: list[str],
    highway_layers: tuple[int, ...] = (7, 8, 9),
) -> tuple:
    """Extract embeddings from LLM semantic highway.

    Returns:
        Tuple of (embeddings, hidden_dim) where embeddings is backend array.
    """
    from mlx_lm import load as mlx_load

    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    backend = get_default_backend()

    logger.info(f"Loading LLM from {model_path}")
    model, tokenizer = mlx_load(str(model_path))

    all_embeds = []
    for concept in concepts:
        tokens = mx.array([tokenizer.encode(concept)])

        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            hidden = model.model.embed_tokens(tokens)
            highway_states = []

            if hasattr(model.model, "layers"):
                # Adjust highway layers based on model depth
                n_layers = len(model.model.layers)
                adjusted_layers = tuple(
                    min(l, n_layers - 1) for l in highway_layers
                )

                for i, layer in enumerate(model.model.layers):
                    hidden = layer(hidden)
                    if i in adjusted_layers:
                        highway_states.append(hidden)

            if highway_states:
                highway_avg = mx.mean(mx.stack(highway_states, axis=0), axis=0)
            else:
                highway_avg = hidden

            mx.eval(highway_avg)
            pooled = mx.mean(highway_avg, axis=1)
            all_embeds.append(pooled)

    embeddings = mx.concatenate(all_embeds, axis=0)
    mx.eval(embeddings)

    hidden_dim = int(embeddings.shape[1])
    return backend.array(embeddings.tolist()), hidden_dim


def extract_clip_embeddings(concepts: list[str]) -> tuple:
    """Extract embeddings from CLIP text encoder.

    Returns:
        Tuple of (embeddings, hidden_dim) where embeddings is backend array.
    """
    from transformers import CLIPModel, CLIPProcessor
    import torch

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    backend = get_default_backend()

    model_name = "openai/clip-vit-base-patch32"
    logger.info(f"Loading CLIP from {model_name}")

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    inputs = processor(text=concepts, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    outputs_np = outputs.detach().cpu().numpy()
    hidden_dim = int(outputs.shape[1])

    return backend.array(outputs_np.tolist()), hidden_dim


def extract_whisper_embeddings(concepts: list[str]) -> tuple:
    """Extract embeddings from Whisper decoder.

    Returns:
        Tuple of (embeddings, hidden_dim) where embeddings is backend array.
    """
    from transformers import WhisperModel, WhisperProcessor
    import torch

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    backend = get_default_backend()

    model_name = "openai/whisper-base"
    logger.info(f"Loading Whisper from {model_name}")

    model = WhisperModel.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    tokenizer = processor.tokenizer
    all_embeds = []

    for concept in concepts:
        tokens = tokenizer(concept, return_tensors="pt").input_ids

        with torch.no_grad():
            embed_layer = model.decoder.embed_tokens
            embeds = embed_layer(tokens)
            pooled = embeds.mean(dim=1)
            all_embeds.append(pooled)

    embeddings = torch.cat(all_embeds, dim=0)
    embeddings_np = embeddings.detach().cpu().numpy()
    hidden_dim = int(embeddings.shape[1])

    return backend.array(embeddings_np.tolist()), hidden_dim


def compute_raw_cka(embeds_a, embeds_b) -> float:
    """Compute raw (unaligned) linear CKA between two embedding sets.

    Args:
        embeds_a: Embeddings from model A, shape [n_concepts, dim_a]
        embeds_b: Embeddings from model B, shape [n_concepts, dim_b]

    Returns:
        CKA value between 0 and 1.
    """
    from modelcypher.core.domain.geometry.cka import compute_cka
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    backend.eval(embeds_a, embeds_b)

    result = compute_cka(embeds_a, embeds_b, backend)
    return result.cka


def run_scale_analysis(
    models_dir: Path,
    models: list[tuple[str, str, int]] | None = None,
    concepts: list[str] | None = None,
) -> ScaleAnalysisResults:
    """Run scale analysis across multiple LLM sizes.

    Args:
        models_dir: Directory containing MLX models.
        models: List of (name, suffix, params) tuples. Uses defaults if None.
        concepts: List of probe concepts. Uses defaults if None.

    Returns:
        ScaleAnalysisResults with CKA values for each model.
    """
    if models is None:
        models = DEFAULT_MODELS
    if concepts is None:
        concepts = PROBE_CONCEPTS

    logger.info(f"Running scale analysis with {len(models)} models")
    logger.info(f"Using {len(concepts)} probe concepts")

    # Extract vision and audio embeddings once (they don't change)
    logger.info("Extracting CLIP (vision) embeddings...")
    clip_embeds, clip_dim = extract_clip_embeddings(concepts)
    logger.info(f"CLIP embeddings: {clip_dim}D")

    logger.info("Extracting Whisper (audio) embeddings...")
    whisper_embeds, whisper_dim = extract_whisper_embeddings(concepts)
    logger.info(f"Whisper embeddings: {whisper_dim}D")

    results = []

    for name, suffix, params in models:
        model_path = models_dir / suffix
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}, skipping")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {name} ({params/1e9:.1f}B params)")
        logger.info(f"{'='*50}")

        # Extract LLM embeddings
        llm_embeds, llm_dim = extract_llm_embeddings(model_path, concepts)
        logger.info(f"LLM embeddings: {llm_dim}D")

        # Compute cross-modal CKA
        cka_vision = compute_raw_cka(llm_embeds, clip_embeds)
        cka_audio = compute_raw_cka(llm_embeds, whisper_embeds)

        logger.info(f"CKA (text ↔ vision): {cka_vision:.4f}")
        logger.info(f"CKA (text ↔ audio): {cka_audio:.4f}")

        results.append(
            ScaleResult(
                model_name=name,
                params=params,
                cka_vision=cka_vision,
                cka_audio=cka_audio,
                hidden_dim=llm_dim,
            )
        )

    return ScaleAnalysisResults(
        models=results,
        probe_concepts=concepts,
        vision_model="openai/clip-vit-base-patch32",
        audio_model="openai/whisper-base",
    )


def save_results(results: ScaleAnalysisResults, output_file: Path) -> None:
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "models": [asdict(r) for r in results.models],
        "probe_concepts": results.probe_concepts,
        "vision_model": results.vision_model,
        "audio_model": results.audio_model,
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def print_summary(results: ScaleAnalysisResults) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("SCALE ANALYSIS RESULTS")
    print("=" * 70)
    print(f"{'Model':<15} {'Params':<12} {'CKA Vision':<12} {'CKA Audio':<12}")
    print("-" * 70)

    for r in results.models:
        params_str = f"{r.params/1e9:.2f}B"
        print(f"{r.model_name:<15} {params_str:<12} {r.cka_vision:<12.4f} {r.cka_audio:<12.4f}")

    print("=" * 70)

    # Check for Anna Karenina pattern
    if len(results.models) >= 2:
        vision_increasing = all(
            results.models[i].cka_vision <= results.models[i + 1].cka_vision
            for i in range(len(results.models) - 1)
        )
        audio_increasing = all(
            results.models[i].cka_audio <= results.models[i + 1].cka_audio
            for i in range(len(results.models) - 1)
        )

        if vision_increasing and audio_increasing:
            print("\n✓ Anna Karenina pattern CONFIRMED: CKA increases with model scale")
        elif vision_increasing or audio_increasing:
            print("\n~ Anna Karenina pattern PARTIAL: One modality shows increasing trend")
        else:
            print("\n✗ Anna Karenina pattern NOT OBSERVED: No clear scaling trend")


def main():
    parser = argparse.ArgumentParser(description="Cross-modal CKA scale analysis")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/Volumes/CodeCypher/models/mlx-community"),
        help="Directory containing MLX models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "scale_analysis_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Verify volume is mounted
    if not args.models_dir.exists():
        logger.error(f"Models directory not found: {args.models_dir}")
        logger.error("Is the volume mounted?")
        return 1

    results = run_scale_analysis(args.models_dir)
    save_results(results, args.output)
    print_summary(results)

    return 0


if __name__ == "__main__":
    exit(main())
