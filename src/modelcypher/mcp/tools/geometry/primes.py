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

"""Geometry primes MCP tools.

Contains tools for:
- Semantic prime listing
- Prime probing
- Prime comparison
"""

from __future__ import annotations

from pathlib import Path

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
    require_existing_path,
)
from .safety import _forward_text_backbone, _resolve_text_backbone


def register_geometry_primes_tools(ctx: ServiceContext) -> None:
    """Register geometry primes tools with real implementations."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_primes_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_list() -> dict:
            """List all NSM semantic primes (Goddard & Wierzbicka 2014)."""
            from modelcypher.core.domain.agents.semantic_prime_atlas import (
                SemanticPrimeInventory,
            )

            primes = SemanticPrimeInventory.english_2014()
            categories = sorted(set(p.category.value for p in primes))
            return {
                "_schema": "mc.geometry.primes.list.v1",
                "primes": [
                    {"id": p.id, "category": p.category.value, "exponents": p.english_exponents}
                    for p in primes
                ],
                "count": len(primes),
                "categories": categories,
            }

    if "mc_geometry_primes_probe" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_probe(
            modelPath: str,
            outputFile: str | None = None,
        ) -> dict:
            """Probe model for semantic prime representations using CKA."""
            import json

            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.agents.semantic_prime_atlas import SemanticPrimeInventory
            from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka

            model_path = require_existing_directory(modelPath)
            model, tokenizer = load_model_for_training(model_path)
            backend = MLXBackend()

            # Resolve architecture
            resolved = _resolve_text_backbone(model)
            if not resolved:
                raise ValueError("Could not resolve model architecture")
            embed_tokens, layers, norm = resolved
            num_layers = len(layers)
            target_layer = num_layers - 1

            # Probe primes
            primes = SemanticPrimeInventory.english_2014()
            activations = {}
            for prime in primes:
                try:
                    probe_text = prime.english_exponents[0] if prime.english_exponents else prime.id
                    tokens = tokenizer.encode(probe_text)
                    input_ids = backend.array([tokens])
                    hidden = _forward_text_backbone(
                        input_ids, embed_tokens, layers, norm, target_layer, backend
                    )
                    activation = backend.mean(hidden[0], axis=0)
                    backend.eval(activation)
                    activations[prime.id] = activation
                except Exception:
                    pass  # Skip failed primes

            if not activations:
                raise ValueError("No activations extracted")

            # Optionally save activations
            if outputFile:
                from modelcypher.core.support.array_utils import array_to_list

                activations_json = {
                    name: array_to_list(backend, act) for name, act in activations.items()
                }
                Path(outputFile).write_text(json.dumps(activations_json, indent=2))

            # Compute coherence with CKA
            all_acts = [a for a in activations.values()]
            X_all = backend.stack(all_acts)
            backend.eval(X_all)
            result = compute_cka(
                X_all,
                X_all,
                estimator=HSICEstimator.AUTO,
                feature_bias_correction=True,
            )
            overall_cka = (
                result.cka_corrected if result.cka_corrected is not None else result.cka
            )

            # Compute category coherence
            category_primes: dict[str, list] = {}
            for prime in primes:
                cat = prime.category.value
                if cat not in category_primes:
                    category_primes[cat] = []
                if prime.id in activations:
                    category_primes[cat].append(activations[prime.id])

            category_coherence = {}
            for cat, acts in category_primes.items():
                if len(acts) >= 2:
                    X = backend.stack(acts)
                    backend.eval(X)
                    cat_result = compute_cka(
                        X,
                        X,
                        estimator=HSICEstimator.AUTO,
                        feature_bias_correction=True,
                    )
                    category_coherence[cat] = (
                        cat_result.cka_corrected
                        if cat_result.cka_corrected is not None
                        else cat_result.cka
                    )

            return {
                "_schema": "mc.geometry.primes.probe.v1",
                "modelPath": model_path,
                "layer": target_layer,
                "primesProbed": len(activations),
                "totalPrimes": len(primes),
                "overallCoherence": overall_cka,
                "overallCoherenceRaw": result.cka,
                "categoryCoherence": category_coherence,
            }

    if "mc_geometry_primes_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_compare(activationsA: str, activationsB: str) -> dict:
            """Compare prime representations between two saved activation files."""
            import json

            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
            from modelcypher.core.domain.geometry.vector_math import VectorMath

            backend = get_default_backend()
            path_a = require_existing_path(activationsA)
            path_b = require_existing_path(activationsB)

            acts_a = json.loads(Path(path_a).read_text())
            acts_b = json.loads(Path(path_b).read_text())
            common = sorted(set(acts_a.keys()) & set(acts_b.keys()))

            if len(common) < 2:
                raise ValueError("Need at least 2 common primes to compare")

            X = backend.stack([backend.array(acts_a[p]) for p in common])
            Y = backend.stack([backend.array(acts_b[p]) for p in common])
            backend.eval(X)
            backend.eval(Y)
            result = compute_cka(
                X,
                Y,
                estimator=HSICEstimator.AUTO,
                feature_bias_correction=True,
            )
            cka_val = result.cka_corrected if result.cka_corrected is not None else result.cka

            # Find most similar and divergent
            sims = []
            for p in common:
                sim = VectorMath.cosine_similarity(acts_a[p], acts_b[p])
                sims.append((p, sim))
            sims.sort(key=lambda x: x[1], reverse=True)

            return {
                "_schema": "mc.geometry.primes.compare.v1",
                "modelA": path_a,
                "modelB": path_b,
                "commonPrimes": len(common),
                "ckaSimilarity": cka_val,
                "ckaRaw": result.cka,
                "mostSimilarPrimes": [p for p, _ in sims[:5]],
                "mostDivergentPrimes": [p for p, _ in sims[-5:]],
            }
