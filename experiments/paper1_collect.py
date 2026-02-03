#!/usr/bin/env python3
"""Paper 1: Invariant Semantic Structure Across Language Model Families.

Data collection script for computing cross-model CKA on semantic primes
and generating null distributions for statistical testing.

Usage:
    # Extract Gram matrices for all models
    python experiments/paper1_collect.py extract --models-dir /path/to/models

    # Compute pairwise CKA
    python experiments/paper1_collect.py cka

    # Generate null distribution
    python experiments/paper1_collect.py null --n-samples 200

    # Compute p-values
    python experiments/paper1_collect.py pvalues

    # Run full pipeline
    python experiments/paper1_collect.py all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
import logging
from itertools import combinations

# Initialize backend before any domain code
from modelcypher.backends import initialize_default_backend
initialize_default_backend()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "paper1"
GRAM_DIR = DATA_DIR / "gram_matrices"
NULL_DIR = DATA_DIR / "null_distribution"

# Target models (adjust paths as needed)
# Using available models on the volume
DEFAULT_MODELS = [
    # Format: (name, path_suffix)
    ("LFM2-350M", "LFM2-350M-MLX-bf16"),
    ("LFM2-700M", "LFM2-700M-bf16"),
    ("LFM2-1.2B", "LFM2-1.2B-bf16"),
    ("Qwen2.5-3B", "Qwen2.5-3B-Instruct-bf16"),
    ("Qwen2.5-Coder-3B", "Qwen2.5-Coder-3B-Instruct-bf16"),
    ("Qwen3-8B", "Qwen3-8B-bf16"),
]


def extract_gram_matrix(model_path: Path, output_file: Path) -> dict:
    """Extract Gram matrix for semantic primes from a model.

    Uses direct embedding lookup from the embedding weight matrix.
    """
    from mlx_lm import load as mlx_load

    from modelcypher.cli.commands.geometry.helpers import resolve_model_backbone
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
    from modelcypher.core.support.array_utils import array_to_list

    backend = get_default_backend()

    logger.info(f"Loading model: {model_path}")
    model_obj, tokenizer = mlx_load(str(model_path))

    backbone = resolve_model_backbone(model_obj)
    if backbone is None:
        raise ValueError(f"Could not resolve backbone for {model_path}")

    embed_tokens, layers, norm = backbone

    # Get embedding weight matrix
    embed_weight = embed_tokens.weight
    backend.eval(embed_weight)

    # Get primes
    primes = SemanticPrimeInventory.english2014()
    logger.info(f"Extracting embeddings for {len(primes)} primes")

    # Extract embedding for each prime's canonical token
    embeddings = {}
    for prime in primes:
        word = prime.canonical_english
        # Tokenize - handle different tokenizer APIs
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(word)
            # Remove special tokens if present (BOS/EOS)
            if hasattr(tokenizer, 'bos_token_id') and tokens and tokens[0] == tokenizer.bos_token_id:
                tokens = tokens[1:]
            if hasattr(tokenizer, 'eos_token_id') and tokens and tokens[-1] == tokenizer.eos_token_id:
                tokens = tokens[:-1]
        else:
            tokens = tokenizer(word)

        if not tokens:
            logger.warning(f"No tokens for prime '{word}', skipping")
            continue

        # Get embedding for first token
        token_id = tokens[0]
        embedding = embed_weight[token_id]
        embeddings[prime.id] = embedding

    # Stack into matrix
    prime_ids = sorted(embeddings.keys())
    vectors = [embeddings[pid] for pid in prime_ids]
    matrix = backend.stack(vectors, axis=0)
    backend.eval(matrix)

    # Compute Gram matrix: G = X @ X.T
    gram = backend.matmul(matrix, backend.transpose(matrix))
    backend.eval(gram)

    # Convert to lists
    gram_list = array_to_list(backend, gram)
    embeddings_list = [array_to_list(backend, v) for v in vectors]

    result = {
        "model_path": str(model_path),
        "model_name": model_path.name,
        "prime_ids": prime_ids,
        "n_primes": len(prime_ids),
        "hidden_dim": int(backend.shape(matrix)[1]),
        "gram_matrix": gram_list,
        "embeddings": embeddings_list,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved Gram matrix to {output_file}")
    return result


def compute_pairwise_cka(gram_dir: Path, output_file: Path) -> list[dict]:
    """Compute CKA between all model pairs."""
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.cka import compute_cka

    backend = get_default_backend()

    # Load all gram files
    gram_files = sorted(gram_dir.glob("*.json"))
    logger.info(f"Found {len(gram_files)} Gram matrix files")

    grams = {}
    for gf in gram_files:
        with open(gf) as f:
            data = json.load(f)
            model_name = data.get("model_name", gf.stem)
            matrix = backend.array(data["embeddings"])
            grams[model_name] = {
                "matrix": matrix,
                "prime_ids": data["prime_ids"],
                "hidden_dim": data["hidden_dim"],
            }

    # Compute all pairwise CKA
    model_names = sorted(grams.keys())
    results = []

    for model_a, model_b in combinations(model_names, 2):
        logger.info(f"Computing CKA: {model_a} <-> {model_b}")

        mat_a = grams[model_a]["matrix"]
        mat_b = grams[model_b]["matrix"]

        backend.eval(mat_a, mat_b)
        cka_result = compute_cka(mat_a, mat_b, backend)

        results.append({
            "model_a": model_a,
            "model_b": model_b,
            "cka": cka_result.cka,
            "hsic_xy": cka_result.hsic_xy,
            "hsic_xx": cka_result.hsic_xx,
            "hsic_yy": cka_result.hsic_yy,
            "n_primes": len(grams[model_a]["prime_ids"]),
        })

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_a", "model_b", "cka", "hsic_xy", "hsic_xx", "hsic_yy", "n_primes"],
        )
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved {len(results)} CKA pairs to {output_file}")
    return results


def generate_null_distribution(
    model_path: Path,
    n_samples: int,
    n_words: int,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Generate null distribution samples."""
    from mlx_lm import load as mlx_load

    from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
    from modelcypher.core.use_cases.null_distribution_service import NullDistributionService

    logger.info(f"Loading tokenizer from {model_path}")
    _, tokenizer = mlx_load(str(model_path))

    service = NullDistributionService()
    vocabulary = service.get_vocabulary_intersection([tokenizer])
    logger.info(f"Vocabulary size: {len(vocabulary)}")

    # Exclude semantic primes
    primes = SemanticPrimeInventory.english2014()
    prime_words = set()
    for p in primes:
        prime_words.add(p.canonical_english.lower())
        for exp in p.english_exponents:
            prime_words.add(exp.lower())

    # Generate samples
    samples = service.generate_null_samples(
        vocabulary=vocabulary,
        n_words=n_words,
        n_samples=n_samples,
        seed=seed,
        exclude_words=prime_words,
    )

    service.save_null_samples(samples, output_dir)
    logger.info(f"Saved {n_samples} null samples to {output_dir}")


def compute_null_cka(
    null_dir: Path,
    models_dir: Path,
    n_samples: int | None = None,
) -> None:
    """Compute CKA for null distribution samples.

    For each null sample (random word set), extracts embeddings from all models
    and computes pairwise CKA, storing results back in the sample files.
    """
    from itertools import combinations

    from mlx_lm import load as mlx_load

    from modelcypher.cli.commands.geometry.helpers import resolve_model_backbone
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.cka import compute_cka
    from modelcypher.core.use_cases.null_distribution_service import NullDistributionService

    backend = get_default_backend()
    service = NullDistributionService()

    # Load null samples
    samples = service.load_null_samples(null_dir)
    if n_samples is not None:
        samples = samples[:n_samples]
    logger.info(f"Processing {len(samples)} null samples")

    # Load all models
    models = {}
    for name, suffix in DEFAULT_MODELS:
        model_path = models_dir / suffix
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        logger.info(f"Loading model: {name}")
        model_obj, tokenizer = mlx_load(str(model_path))
        backbone = resolve_model_backbone(model_obj)
        if backbone is None:
            logger.warning(f"Could not resolve backbone for {name}")
            continue
        embed_tokens, layers, norm = backbone
        models[name] = {
            "tokenizer": tokenizer,
            "embed_weight": embed_tokens.weight,
        }

    model_names = sorted(models.keys())
    logger.info(f"Loaded {len(models)} models")

    # Process each sample
    for sample in samples:
        logger.info(f"Processing sample {sample.sample_id}")

        # Extract embeddings for this sample's words from all models
        model_embeddings = {}
        for model_name, model_data in models.items():
            tokenizer = model_data["tokenizer"]
            embed_weight = model_data["embed_weight"]

            vectors = []
            for word in sample.words:
                # Tokenize
                if hasattr(tokenizer, "encode"):
                    tokens = tokenizer.encode(word)
                    if hasattr(tokenizer, "bos_token_id") and tokens and tokens[0] == tokenizer.bos_token_id:
                        tokens = tokens[1:]
                    if hasattr(tokenizer, "eos_token_id") and tokens and tokens[-1] == tokenizer.eos_token_id:
                        tokens = tokens[:-1]
                else:
                    tokens = tokenizer(word)

                if not tokens:
                    continue

                token_id = tokens[0]
                embedding = embed_weight[token_id]
                vectors.append(embedding)

            if vectors:
                matrix = backend.stack(vectors, axis=0)
                backend.eval(matrix)
                model_embeddings[model_name] = matrix

        # Compute pairwise CKA
        for model_a, model_b in combinations(model_names, 2):
            if model_a not in model_embeddings or model_b not in model_embeddings:
                continue

            mat_a = model_embeddings[model_a]
            mat_b = model_embeddings[model_b]

            backend.eval(mat_a, mat_b)
            cka_result = compute_cka(mat_a, mat_b, backend)

            pair_key = f"{model_a}_{model_b}"
            sample.cka_values[pair_key] = cka_result.cka

        # Save updated sample
        sample_file = null_dir / f"sample_{sample.sample_id:04d}.json"
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

    logger.info(f"Completed CKA computation for {len(samples)} null samples")


def compute_pvalues(
    primes_cka_file: Path,
    null_dir: Path,
    output_file: Path,
) -> dict:
    """Compute p-values comparing primes to null distribution."""
    from modelcypher.core.support.statistics import (
        bootstrap_ci,
        cohens_d,
        mean,
        permutation_pvalue,
        standard_deviation,
    )
    from modelcypher.core.use_cases.null_distribution_service import NullDistributionService

    # Load primes CKA
    with open(primes_cka_file) as f:
        reader = csv.DictReader(f)
        primes_results = [row for row in reader if row.get("cka")]

    primes_cka_values = [float(r["cka"]) for r in primes_results]
    primes_mean = mean(primes_cka_values)
    primes_std = standard_deviation(primes_cka_values, primes_mean)

    # Load null distribution
    service = NullDistributionService()
    null_samples = service.load_null_samples(null_dir)

    # Collect null CKA values
    null_cka_values = []
    for sample in null_samples:
        null_cka_values.extend(sample.cka_values.values())

    if not null_cka_values:
        logger.warning("No CKA values in null samples - need to compute CKA for null samples")
        results = {
            "primes": {
                "mean_cka": primes_mean,
                "std_cka": primes_std,
                "n_pairs": len(primes_cka_values),
            },
            "null_distribution": {
                "status": "CKA not computed for null samples",
            },
            "note": "Run CKA computation on null samples first",
        }
    else:
        null_mean = mean(null_cka_values)
        null_std = standard_deviation(null_cka_values, null_mean)

        p_value = permutation_pvalue(primes_mean, null_cka_values)
        effect_size = cohens_d(primes_mean, null_mean, null_std)
        ci_lower, ci_upper = bootstrap_ci(primes_cka_values, confidence=0.95, seed=42)

        results = {
            "primes": {
                "mean_cka": primes_mean,
                "std_cka": primes_std,
                "n_pairs": len(primes_cka_values),
                "ci_95": [ci_lower, ci_upper],
            },
            "null_distribution": {
                "mean_cka": null_mean,
                "std_cka": null_std,
                "n_samples": len(null_samples),
                "n_cka_values": len(null_cka_values),
            },
            "statistical_test": {
                "p_value": p_value,
                "effect_size_cohens_d": effect_size,
                "significant_at_005": p_value < 0.05,
            },
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Paper 1 data collection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract Gram matrices")
    extract_parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/Volumes/CodeCypher/models/mlx-community"),
        help="Directory containing models",
    )

    # CKA command
    subparsers.add_parser("cka", help="Compute pairwise CKA")

    # Null command
    null_parser = subparsers.add_parser("null", help="Generate null distribution")
    null_parser.add_argument("--n-samples", type=int, default=200, help="Number of samples")
    null_parser.add_argument("--n-words", type=int, default=65, help="Words per sample")
    null_parser.add_argument(
        "--model",
        type=Path,
        default=Path("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"),
        help="Model for vocabulary",
    )

    # Null CKA command
    null_cka_parser = subparsers.add_parser("null-cka", help="Compute CKA for null samples")
    null_cka_parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/Volumes/CodeCypher/models/mlx-community"),
        help="Directory containing models",
    )
    null_cka_parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )

    # P-values command
    subparsers.add_parser("pvalues", help="Compute p-values")

    # All command
    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/Volumes/CodeCypher/models/mlx-community"),
    )
    all_parser.add_argument("--n-samples", type=int, default=200)

    args = parser.parse_args()

    if args.command == "extract":
        GRAM_DIR.mkdir(parents=True, exist_ok=True)
        for name, suffix in DEFAULT_MODELS:
            model_path = args.models_dir / suffix
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue
            output_file = GRAM_DIR / f"{name}.json"
            extract_gram_matrix(model_path, output_file)

    elif args.command == "cka":
        output_file = DATA_DIR / "cka_pairwise.csv"
        compute_pairwise_cka(GRAM_DIR, output_file)

    elif args.command == "null":
        generate_null_distribution(
            model_path=args.model,
            n_samples=args.n_samples,
            n_words=args.n_words,
            output_dir=NULL_DIR,
        )

    elif args.command == "null-cka":
        compute_null_cka(
            null_dir=NULL_DIR,
            models_dir=args.models_dir,
            n_samples=args.n_samples,
        )

    elif args.command == "pvalues":
        primes_cka = DATA_DIR / "cka_pairwise.csv"
        output = DATA_DIR / "results.json"
        results = compute_pvalues(primes_cka, NULL_DIR, output)
        print(json.dumps(results, indent=2))

    elif args.command == "all":
        # Full pipeline
        logger.info("=== STEP 1: Extract Gram matrices ===")
        GRAM_DIR.mkdir(parents=True, exist_ok=True)
        for name, suffix in DEFAULT_MODELS:
            model_path = args.models_dir / suffix
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue
            output_file = GRAM_DIR / f"{name}.json"
            if not output_file.exists():
                extract_gram_matrix(model_path, output_file)
            else:
                logger.info(f"Skipping {name} (already exists)")

        logger.info("=== STEP 2: Compute pairwise CKA ===")
        cka_file = DATA_DIR / "cka_pairwise.csv"
        compute_pairwise_cka(GRAM_DIR, cka_file)

        logger.info("=== STEP 3: Generate null distribution ===")
        # Use first available model for vocabulary
        model_for_null = None
        for name, suffix in DEFAULT_MODELS:
            model_path = args.models_dir / suffix
            if model_path.exists():
                model_for_null = model_path
                break

        if model_for_null:
            generate_null_distribution(
                model_path=model_for_null,
                n_samples=args.n_samples,
                n_words=65,
                output_dir=NULL_DIR,
            )
        else:
            logger.warning("No models found for null distribution")

        logger.info("=== STEP 4: Compute p-values ===")
        results = compute_pvalues(cka_file, NULL_DIR, DATA_DIR / "results.json")
        print("\n=== RESULTS ===")
        print(json.dumps(results, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
