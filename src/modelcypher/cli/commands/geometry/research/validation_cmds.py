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

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.output import write_error, write_output

from .common import get_context


def register(app: typer.Typer) -> None:
    @app.command("validate-transplant")
    def validate_transplant(
        ctx: typer.Context,
        original_profile: Path = typer.Option(
            ...,
            "--original",
            "-o",
            help="Path to pre-transplant density profile JSON",
        ),
        transplanted_model: Path = typer.Option(
            ...,
            "--model",
            "-m",
            help="Path to transplanted model",
        ),
        output: Path | None = typer.Option(
            None,
            "--output-file",
            help="Path to save comparison JSON",
        ),
    ) -> None:
        """Validate transplant by comparing density before/after.

        Profiles the transplanted model and compares to the original profile
        to verify the transplant succeeded.

        Checks:
        1. Domains became denser after transplant
        2. Overall density improved

        Example:
            mc geometry research validate-transplant \\
                --original ./profiles/target-before.json \\
                --model /path/to/transplanted-model
        """
        context = get_context(ctx)

        try:
            import json as json_module

            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
            from modelcypher.core.domain.geometry.knowledge_density import (
                KnowledgeDensityAnalyzer,
            )
            from modelcypher.core.domain.geometry.probe_calibration import (
                MLXActivationProvider,
            )

            # Load original profile
            if not original_profile.exists():
                write_error(
                    f"Original profile not found: {original_profile}",
                    context.output_format,
                )
                raise typer.Exit(1)

            with original_profile.open() as f:
                original_data = json_module.load(f)

            # Extract original domain densities
            original_densities: dict[str, float] = {}
            for ds in original_data.get("domainSummaries", []):
                original_densities[ds["domain"]] = ds.get("overallMeanDensity", 0.0)

            # If no layers specified, use original profile's layers
            layer_list = original_data.get("completedLayers", [])
            if not layer_list:
                layer_list = list(range(original_data.get("totalLayers", 16)))
            target_domains = sorted(original_densities.keys())

            # Load transplanted model
            model, tokenizer = load_model_for_training(str(transplanted_model))
            backend = get_default_backend()

            # Get probes
            probes = UnifiedAtlasInventory.all_probes()

            # Create activation provider
            provider = MLXActivationProvider(model=model, tokenizer=tokenizer, backend=backend)

            # Analyze transplanted model density
            analyzer = KnowledgeDensityAnalyzer(backend=backend)

            typer.echo(f"Profiling transplanted model: {transplanted_model}")
            typer.echo(f"Layers to analyze: {len(layer_list)}")

            profile = analyzer.analyze_model(
                probes=probes,
                activation_provider=provider,
                layers=layer_list,
            )

            # Compare densities
            transplanted_densities: dict[str, float] = profile.domain_densities

            comparisons = []
            domain_deltas: list[float] = []
            domain_abs_changes: list[float] = []

            for domain, original_density in original_densities.items():
                transplanted_density = transplanted_densities.get(domain, 0.0)
                delta = transplanted_density - original_density
                is_target = True
                comparison = {
                    "domain": domain,
                    "isTargetDomain": is_target,
                    "originalDensity": original_density,
                    "transplantedDensity": transplanted_density,
                    "delta": delta,
                }
                comparisons.append(comparison)

            # Compute summary statistics (raw measurements only, no interpretation)
            domain_deltas = [c["delta"] for c in comparisons]
            domain_abs_changes = [abs(c["delta"]) for c in comparisons]
            mean_delta = sum(domain_deltas) / len(domain_deltas) if domain_deltas else 0.0
            mean_abs_change = (
                sum(domain_abs_changes) / len(domain_abs_changes) if domain_abs_changes else 0.0
            )

            # Return raw measurements - let users interpret based on their context
            # Positive target improvement = target domains improved
            # Low non-target change = minimal interference (user decides threshold)
            result = {
                "_schema": "mc.geometry.research.validate_transplant.v1",
                "originalProfile": str(original_profile),
                "transplantedModel": str(transplanted_model),
                "targetDomains": target_domains,
                "layersAnalyzed": layer_list,
                "comparisons": comparisons,
                "summary": {
                    "meanDelta": mean_delta,
                    "meanAbsoluteChange": mean_abs_change,
                    "domainCount": len(comparisons),
                },
            }

            # Save to file if requested
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("w") as f:
                    json_module.dump(result, f, indent=2)
                typer.echo(f"Comparison saved to {output}")

            # Output result
            if context.output_format == "text":
                lines = [
                    "",
                    "=" * 60,
                    "TRANSPLANT VALIDATION RESULTS",
                    "=" * 60,
                    f"Domains analyzed: {', '.join(target_domains)}",
                    "",
                    "Domain Comparisons:",
                ]
                for c in comparisons:
                    marker = "[TARGET]" if c["isTargetDomain"] else "[other]"
                    direction = "+" if c["delta"] > 0 else ""
                    lines.append(
                        f"  {marker} {c['domain']}: "
                        f"{c['originalDensity']:.3f} -> {c['transplantedDensity']:.3f} "
                        f"({direction}{c['delta']:.3f})"
                    )
                lines.append("")
                lines.append("Summary:")
                lines.append(f"  Mean delta: {mean_delta:+.4f}")
                lines.append(f"  Mean absolute change: {mean_abs_change:.4f}")
                lines.append("")
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(result, context.output_format, context.pretty)

        except Exception as e:
            write_error(f"Validation failed: {e}", context.output_format)
            raise typer.Exit(1) from e

    @app.command("strong-test")
    def strong_test(
        ctx: typer.Context,
        model_a: Path = typer.Option(
            ...,
            "--model-a",
            help="Path to source model",
        ),
        model_b: Path = typer.Option(
            ...,
            "--model-b",
            help="Path to target model",
        ),
        train_size: int = typer.Option(
            50,
            "--train-size",
            help="Number of semantic primes for alignment training",
        ),
        test_size: int = typer.Option(
            500,
            "--test-size",
            help="Number of random words for held-out evaluation",
        ),
        wordlist: Path = typer.Option(
            Path("/usr/share/dict/words"),
            "--wordlist",
            help="Wordlist file for random-word sampling",
        ),
        seed: int = typer.Option(
            0,
            "--seed",
            help="Random seed for word sampling",
        ),
        include_words: bool = typer.Option(
            False,
            "--include-words",
            help="Include word lists in output payload",
        ),
        output: Path | None = typer.Option(
            None,
            "--output-file",
            help="Path to save results JSON",
        ),
    ) -> None:
        """Run the strong geometric generalization test (train primes, test random words).

        Example:
            mc geometry research strong-test \\
                --model-a /path/to/model-a \\
                --model-b /path/to/model-b
        """
        context = get_context(ctx)

        try:
            import json as json_module
            import random

            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.cli.commands.geometry.helpers import (
                extract_anchor_activations,
                resolve_model_backbone,
            )
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.agents.semantic_primes import (
                SemanticPrimeInventory,
            )
            from modelcypher.core.domain.geometry.cka import compute_linear_cka
            from modelcypher.core.domain.geometry.gram_aligner import find_alignment

            if train_size < 4:
                raise ValueError("train-size must be at least 4 for CKA")
            if test_size < 4:
                raise ValueError("test-size must be at least 4 for CKA")
            if not wordlist.exists():
                raise ValueError(f"Wordlist not found: {wordlist}")

            model_a_obj, tokenizer_a = load_model_for_training(str(model_a))
            model_b_obj, tokenizer_b = load_model_for_training(str(model_b))
            backend = get_default_backend()

            def is_single_token(word: str, tokenizer) -> bool:
                try:
                    tokens = tokenizer.encode(word, add_special_tokens=False)
                except TypeError:
                    tokens = tokenizer.encode(word)
                if isinstance(tokens, list):
                    return len(tokens) == 1
                return len(list(tokens.ids)) == 1

            def resolve_backbone(model, label: str):
                resolved = resolve_model_backbone(
                    model,
                    getattr(model, "model_type", None),
                )
                if resolved is None:
                    raise ValueError(f"Failed to resolve model backbone for {label}")
                return resolved

            embed_a, layers_a, norm_a = resolve_backbone(model_a_obj, "model-a")
            embed_b, layers_b, norm_b = resolve_backbone(model_b_obj, "model-b")
            layer_a = len(layers_a) // 2
            layer_b = len(layers_b) // 2

            primes = SemanticPrimeInventory.english2014()
            prime_words = [p.canonical_english for p in primes]
            prime_words = [
                word
                for word in prime_words
                if is_single_token(word, tokenizer_a) and is_single_token(word, tokenizer_b)
            ]

            if not prime_words:
                raise ValueError("No shared single-token semantic primes found")

            train_words = prime_words[:train_size]

            def sample_random_words() -> tuple[list[str], int]:
                rng = random.Random(seed)
                reservoir: list[str] = []
                eligible_count = 0
                seen: set[str] = set()
                exclude = set(train_words)

                with wordlist.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        word = line.strip().lower()
                        if not word or word in seen or word in exclude:
                            continue
                        if not word.isascii() or not word.isalpha():
                            continue
                        if not is_single_token(word, tokenizer_a):
                            continue
                        if not is_single_token(word, tokenizer_b):
                            continue
                        seen.add(word)
                        eligible_count += 1
                        if len(reservoir) < test_size:
                            reservoir.append(word)
                        else:
                            j = rng.randint(0, eligible_count - 1)
                            if j < test_size:
                                reservoir[j] = word

                return reservoir, eligible_count

            test_words, eligible_count = sample_random_words()
            if len(test_words) < 4:
                raise ValueError(
                    f"Only {len(test_words)} eligible random words found (need >= 4)."
                )

            def collect_activations(words, tokenizer, embed_tokens, layers, norm, target_layer):
                return extract_anchor_activations(
                    words,
                    tokenizer,
                    embed_tokens,
                    layers,
                    norm,
                    target_layer,
                    backend,
                )

            train_map_a = collect_activations(
                train_words, tokenizer_a, embed_a, layers_a, norm_a, layer_a
            )
            train_map_b = collect_activations(
                train_words, tokenizer_b, embed_b, layers_b, norm_b, layer_b
            )
            train_shared = [w for w in train_words if w in train_map_a and w in train_map_b]
            if len(train_shared) < 4:
                raise ValueError("Insufficient shared prime activations for training")

            source_train = backend.stack([train_map_a[w] for w in train_shared], axis=0)
            target_train = backend.stack([train_map_b[w] for w in train_shared], axis=0)
            source_train = backend.astype(source_train, "float32")
            target_train = backend.astype(target_train, "float32")
            backend.eval(source_train, target_train)

            test_map_a = collect_activations(
                test_words, tokenizer_a, embed_a, layers_a, norm_a, layer_a
            )
            test_map_b = collect_activations(
                test_words, tokenizer_b, embed_b, layers_b, norm_b, layer_b
            )
            test_shared = [w for w in test_words if w in test_map_a and w in test_map_b]
            if len(test_shared) < 4:
                raise ValueError("Insufficient shared random-word activations for testing")

            source_test = backend.stack([test_map_a[w] for w in test_shared], axis=0)
            target_test = backend.stack([test_map_b[w] for w in test_shared], axis=0)
            source_test = backend.astype(source_test, "float32")
            target_test = backend.astype(target_test, "float32")
            backend.eval(source_test, target_test)

            alignment = find_alignment(source_train, target_train, backend)
            aligned_train = backend.matmul(source_train, alignment.feature_transform)
            aligned_test = backend.matmul(source_test, alignment.feature_transform)
            backend.eval(aligned_train, aligned_test)

            train_cka_raw = compute_linear_cka(source_train, target_train, backend)
            train_cka_aligned = compute_linear_cka(aligned_train, target_train, backend)
            test_cka_raw = compute_linear_cka(source_test, target_test, backend)
            test_cka_aligned = compute_linear_cka(aligned_test, target_test, backend)

            n_train = int(source_train.shape[0])
            n_test = int(source_test.shape[0])
            source_dim = int(source_train.shape[1])
            target_dim = int(target_train.shape[1])
            rank_bound = min(n_train, source_dim, target_dim)

            result = {
                "_schema": "mc.geometry.research.strong_test.v1",
                "models": {
                    "source": str(model_a),
                    "target": str(model_b),
                },
                "layers": {
                    "sourceLayer": layer_a,
                    "targetLayer": layer_b,
                    "sourceLayerCount": len(layers_a),
                    "targetLayerCount": len(layers_b),
                },
                "train": {
                    "requested": train_size,
                    "used": len(train_shared),
                    "eligiblePrimes": len(prime_words),
                },
                "test": {
                    "requested": test_size,
                    "used": len(test_shared),
                    "eligibleRandomWords": eligible_count,
                },
                "dims": {
                    "sourceDim": source_dim,
                    "targetDim": target_dim,
                    "rankBound": rank_bound,
                },
                "cka": {
                    "trainRaw": train_cka_raw,
                    "trainAligned": train_cka_aligned,
                    "testRaw": test_cka_raw,
                    "testAligned": test_cka_aligned,
                },
            }

            if include_words:
                result["words"] = {
                    "trainPrimes": train_shared,
                    "testRandom": test_shared,
                }

            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json_module.dumps(result, indent=2), encoding="utf-8")

            if context.output_format == "text":
                lines = [
                    "STRONG GEOMETRIC TEST",
                    f"Model A: {model_a}",
                    f"Model B: {model_b}",
                    "",
                    f"Layer A: {layer_a} / {len(layers_a)}",
                    f"Layer B: {layer_b} / {len(layers_b)}",
                    "",
                    f"Train primes: {len(train_shared)} (requested {train_size}, eligible {len(prime_words)})",
                    f"Test random: {len(test_shared)} (requested {test_size}, eligible {eligible_count})",
                    "",
                    f"Train CKA (raw): {train_cka_raw:.6f}",
                    f"Train CKA (aligned): {train_cka_aligned:.6f}",
                    f"Test CKA (raw): {test_cka_raw:.6f}",
                    f"Test CKA (aligned): {test_cka_aligned:.6f}",
                    "",
                    f"Rank bound: {rank_bound} (n_train={n_train}, d_src={source_dim}, d_tgt={target_dim})",
                ]
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(result, context.output_format, context.pretty)

        except Exception as exc:
            write_error(f"Strong test failed: {exc}", context.output_format)
            raise typer.Exit(1) from exc
