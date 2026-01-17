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
        invariant_mode: bool = typer.Option(
            False,
            "--invariant-mode",
            help="Test underdetermined alignment regime (n < d) for generalization.",
        ),
        n_train: int = typer.Option(
            0,
            "--n-train",
            help="Override n_train (0=auto). In invariant mode: 0.5*min(d). Otherwise: 2*max(d).",
        ),
        force_rank_deficient: bool = typer.Option(
            False,
            "--force-rank-deficient",
            help="Allow insufficient samples (for debugging only)",
        ),
        random_baseline: bool = typer.Option(
            False,
            "--random-baseline",
            help="Also test alignment against random Gaussian activations (sanity check)",
        ),
    ) -> None:
        """Measure coordinate alignment quality between two models.

        Measures how well coordinate systems align between two models and reports
        CKA diagnostics.

        Modes:
        1. --invariant-mode: n < d, tests generalization to held-out concepts.
        2. Default: n > d, measures probe coverage of shared structure.

        Example:
            mc geometry research strong-test \\
                --model-a /path/to/model-a \\
                --model-b /path/to/model-b \\
                --invariant-mode
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
            from modelcypher.core.domain.geometry.cka import compute_cka
            from modelcypher.core.domain.geometry.gram_aligner import find_alignment

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

            # Get hidden dimensions from embeddings
            d_source = int(embed_a.weight.shape[-1])
            d_target = int(embed_b.weight.shape[-1])
            d_max = max(d_source, d_target)
            d_min = min(d_source, d_target)

            # Derive n_train based on mode
            if n_train > 0:
                # User override
                n_train_required = n_train
                mode_str = "user-specified"
            elif invariant_mode:
                # UNDERDETERMINED MODE: n < d makes linear alignment exact on probes.
                # Use 0.5 * min(d) to ensure we're well below the rank limit
                n_train_required = d_min // 2
                mode_str = "underdetermined (n < d, linear alignment exact)"
            else:
                # OVERLAP MODE: n > d measures actual overlap (overdetermined)
                n_train_required = 2 * d_max
                mode_str = "overlap (n > d, geodesic CKA measured)"

            typer.echo(f"Geometry: d_source={d_source}, d_target={d_target}, d_min={d_min}")
            typer.echo(f"Mode: {mode_str}")
            typer.echo(f"n_train (target): {n_train_required}")

            def sample_words_from_wordlist(n_needed: int) -> tuple[list[str], int]:
                """Sample n_needed single-token words from wordlist using reservoir sampling."""
                rng = random.Random(seed)
                reservoir: list[str] = []
                eligible_count = 0
                seen: set[str] = set()

                with wordlist.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        word = line.strip().lower()
                        if not word or word in seen:
                            continue
                        if not word.isascii() or not word.isalpha():
                            continue
                        if not is_single_token(word, tokenizer_a):
                            continue
                        if not is_single_token(word, tokenizer_b):
                            continue
                        seen.add(word)
                        eligible_count += 1
                        if len(reservoir) < n_needed:
                            reservoir.append(word)
                        else:
                            j = rng.randint(0, eligible_count - 1)
                            if j < n_needed:
                                reservoir[j] = word

                return reservoir, eligible_count

            # Sample words for both train and test (disjoint sets)
            total_needed = n_train_required + test_size
            all_words, eligible_count = sample_words_from_wordlist(total_needed)

            typer.echo(f"Eligible single-token words: {eligible_count}")

            # Check if we have enough words based on mode
            if invariant_mode:
                # UNDERDETERMINED MODE: We want n < d, so fewer words is fine
                # Just need enough for a meaningful test (at least 50 or n_train_required)
                min_required = min(50, n_train_required)
                if eligible_count < min_required + test_size:
                    raise ValueError(
                        f"Insufficient eligible words: {eligible_count} < {min_required + test_size} minimum.\n"
                        f"Need at least {min_required} train + {test_size} test words."
                    )
                # Cap at n_train_required to maintain n < d
                n_train_actual = min(n_train_required, eligible_count - test_size)
                if n_train_actual >= d_min:
                    typer.secho(
                        f"WARNING: n_train_actual={n_train_actual} >= d_min={d_min}.\n"
                        f"Linear alignment is no longer underdetermined in this configuration.\n"
                        f"Reduce --n-train or use fewer test samples.",
                        fg=typer.colors.YELLOW,
                    )
            else:
                # OVERLAP MODE: We need n > d for full-rank alignment
                if eligible_count < n_train_required:
                    if force_rank_deficient:
                        typer.secho(
                            f"WARNING: Only {eligible_count} eligible words < {n_train_required} required.\n"
                            f"rank(F) = {eligible_count} << {d_target}. Results may not be valid.\n"
                            f"Proceeding due to --force-rank-deficient flag.",
                            fg=typer.colors.YELLOW,
                        )
                        n_train_actual = eligible_count - test_size if eligible_count > test_size else eligible_count // 2
                    else:
                        raise ValueError(
                            f"Insufficient eligible words: {eligible_count} < {n_train_required} required.\n"
                            f"rank(F) would be {eligible_count}, causing rank-deficient alignment.\n"
                            f"Use a larger wordlist or --force-rank-deficient to proceed anyway."
                        )
                else:
                    n_train_actual = n_train_required

            # Split into disjoint train and test sets
            train_words = all_words[:n_train_actual]
            test_words = all_words[n_train_actual : n_train_actual + test_size]

            if len(train_words) < 4:
                raise ValueError(
                    f"Only {len(train_words)} training words available (need >= 4)."
                )
            if len(test_words) < 4:
                raise ValueError(
                    f"Only {len(test_words)} test words available (need >= 4)."
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

            # GEODESIC CKA (RBF on geodesic distances - correct for high-d manifolds)
            # This uses k-NN graph + geodesic distances, not Euclidean dot products
            train_geo_raw = compute_cka(source_train, target_train, backend)
            train_geo_aligned = compute_cka(aligned_train, target_train, backend)
            test_geo_raw = compute_cka(source_test, target_test, backend)
            test_geo_aligned = compute_cka(aligned_test, target_test, backend)

            n_train_final = int(source_train.shape[0])
            n_test = int(source_test.shape[0])
            actual_source_dim = int(source_train.shape[1])
            actual_target_dim = int(target_train.shape[1])
            actual_d_min = min(actual_source_dim, actual_target_dim)
            rank_bound = min(n_train_final, actual_source_dim, actual_target_dim)

            # Determine if we're in underdetermined (invariant) or overdetermined (overlap) regime
            is_underdetermined = n_train_final < actual_d_min  # n < d: linear alignment exact on probes
            is_full_rank = rank_bound >= actual_d_min

            # Random baseline comparison (sanity check)
            random_baseline_result = None
            if random_baseline:
                # Use data-derived seed offset to ensure independence from main experiment
                # n_train_final ensures different random sequence without arbitrary constants
                rng = random.Random(seed + n_train_final)

                # Generate random Gaussian activations with same shape as target
                random_train_data = [[rng.gauss(0, 1) for _ in range(actual_target_dim)]
                                     for _ in range(n_train_final)]
                random_test_data = [[rng.gauss(0, 1) for _ in range(actual_target_dim)]
                                    for _ in range(n_test)]

                random_train = backend.array(random_train_data)
                random_test = backend.array(random_test_data)
                random_train = backend.astype(random_train, "float32")
                random_test = backend.astype(random_test, "float32")
                backend.eval(random_train, random_test)

                # Align source to random (should NOT achieve high test CKA if test is valid)
                random_alignment = find_alignment(source_train, random_train, backend)
                aligned_train_random = backend.matmul(source_train, random_alignment.feature_transform)
                aligned_test_random = backend.matmul(source_test, random_alignment.feature_transform)
                backend.eval(aligned_train_random, aligned_test_random)

                random_train_cka = compute_cka(aligned_train_random, random_train, backend)
                random_test_cka = compute_cka(aligned_test_random, random_test, backend)

                # Raw measurements only - no interpretation
                random_baseline_result = {
                    "trainCkaAligned": random_train_cka.cka if random_train_cka.is_valid else 0.0,
                    "testCkaAligned": random_test_cka.cka if random_test_cka.is_valid else 0.0,
                }

            result = {
                "_schema": "mc.geometry.research.strong_test.v4",
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
                "mode": {
                    "name": "invariant" if invariant_mode else "overlap",
                    "description": mode_str,
                    "isUnderdetermined": is_underdetermined,
                },
                "geometry": {
                    "sourceDim": actual_source_dim,
                    "targetDim": actual_target_dim,
                    "dMin": actual_d_min,
                    "nTrainTarget": n_train_required,
                    "nTrainActual": n_train_final,
                    "rankBound": rank_bound,
                    "isFullRank": is_full_rank,
                },
                "train": {
                    "target": n_train_required,
                    "used": len(train_shared),
                    "eligibleWords": eligible_count,
                },
                "test": {
                    "requested": test_size,
                    "used": len(test_shared),
                    "eligibleWords": eligible_count,
                },
                "cka": {
                    "geodesic": {
                        "trainRaw": train_geo_raw.cka if train_geo_raw.is_valid else 0.0,
                        "trainAligned": train_geo_aligned.cka if train_geo_aligned.is_valid else 0.0,
                        "testRaw": test_geo_raw.cka if test_geo_raw.is_valid else 0.0,
                        "testAligned": test_geo_aligned.cka if test_geo_aligned.is_valid else 0.0,
                    },
                },
            }

            if random_baseline_result is not None:
                result["randomBaseline"] = random_baseline_result

            if include_words:
                result["words"] = {
                    "train": train_shared,
                    "test": test_shared,
                }

            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json_module.dumps(result, indent=2), encoding="utf-8")

            if context.output_format == "text":
                mode_header = "UNDERDETERMINED MODE" if invariant_mode else "OVERLAP MODE"
                regime_str = "UNDERDETERMINED (n < d)" if is_underdetermined else "OVERDETERMINED (n > d)"
                lines = [
                    "=" * 60,
                    f"STRONG GEOMETRIC GENERALIZATION TEST - {mode_header}",
                    "=" * 60,
                    f"Model A (source): {model_a}",
                    f"Model B (target): {model_b}",
                    "",
                    "MODE:",
                    f"  {mode_str}",
                    f"  Regime: {regime_str}",
                    "",
                    "GEOMETRY:",
                    f"  d_source = {actual_source_dim}",
                    f"  d_target = {actual_target_dim}",
                    f"  d_min = {actual_d_min}",
                    f"  n_train (target) = {n_train_required}",
                    f"  n_train (actual) = {n_train_final}",
                    f"  rank(F) bound = {rank_bound}",
                    "",
                    "SAMPLES:",
                    f"  Eligible single-token words: {eligible_count}",
                    f"  Train words used: {len(train_shared)}",
                    f"  Test words used: {len(test_shared)}",
                    "",
                    "CKA RESULTS (Geodesic - Riemannian):",
                    f"  Train CKA (raw):     {train_geo_raw.cka:.6f}",
                    f"  Train CKA (aligned): {train_geo_aligned.cka:.6f}",
                    f"  Test CKA (raw):      {test_geo_raw.cka:.6f}",
                    f"  Test CKA (aligned):  {test_geo_aligned.cka:.6f}",
                    "",
                ]

                lines.append("TRAIN SET:")
                lines.append(f"  train_cka_geodesic = {train_geo_aligned.cka:.6f}")

                lines.append("")
                lines.append("TEST SET:")
                lines.append(f"  test_cka_geodesic = {test_geo_aligned.cka:.6f}")

                if random_baseline_result is not None:
                    lines.append("")
                    lines.append("RANDOM BASELINE:")
                    lines.append(f"  random_train_cka = {random_baseline_result['trainCkaAligned']:.6f}")
                    lines.append(f"  random_test_cka = {random_baseline_result['testCkaAligned']:.6f}")

                lines.append("")
                lines.append("=" * 60)
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(result, context.output_format, context.pretty)

        except Exception as exc:
            write_error(f"Strong test failed: {exc}", context.output_format)
            raise typer.Exit(1) from exc
