# ModelCypher Paper Drafts

This directory contains drafts for the ModelCypher paper series (Paper 0–III), formerly hosted in the internal research repository.

The drafts are a narrative layer on top of the implementation. The canonical, code-aligned documentation lives in `../docs/`.

## Paper ↔ Repo Crosswalk

0. **Paper 0: The Shape of Knowledge** (Foundational)
   - Draft: [`paper-0-the-shape-of-knowledge.md`](paper-0-the-shape-of-knowledge.md)
   - Premise: Knowledge is high-dimensional geometry; inference is navigation (a synthesis of the “13 Pillars”).
   - Docs: [`../docs/START-HERE.md`](../docs/START-HERE.md), [`../docs/GLOSSARY.md`](../docs/GLOSSARY.md), [`../docs/geometry/mental_model.md`](../docs/geometry/mental_model.md)
   - Code touchpoints: `../src/modelcypher/core/domain/geometry/`, `../src/modelcypher/core/domain/semantics/`
   - Bibliography: [`../KnowledgeasHighDimensionalGeometryInLLMs.md`](../KnowledgeasHighDimensionalGeometryInLLMs.md)

1. **Paper 1: The Manifold Hypothesis of Agency** (Geometry)
   - Draft: [`paper-1-manifold-hypothesis-of-agency.md`](paper-1-manifold-hypothesis-of-agency.md)
   - Premise: Agentic behavior is a geometric property (a vector), not a prompt.
   - Docs: [`../docs/research/semantic_primes.md`](../docs/research/semantic_primes.md), [`../docs/research/falsification_experiments.md`](../docs/research/falsification_experiments.md)
   - Code touchpoints: `../src/modelcypher/core/domain/agents/semantic_primes.py`, `../src/modelcypher/core/domain/geometry/gate_detector.py`, `../src/modelcypher/core/domain/geometry/manifold_fidelity_sweep.py`, `../src/modelcypher/core/domain/geometry/topological_fingerprint.py`, `../src/modelcypher/core/domain/geometry/intrinsic_dimension.py`
   - CLI touchpoints: `mc geometry primes …`, `mc geometry path …`

2. **Paper 2: Linguistic Thermodynamics** (Physics)
   - Draft: [`paper-2-linguistic-thermodynamics.md`](paper-2-linguistic-thermodynamics.md)
   - Premise: Language models behave as physical systems obeying thermodynamic laws (Entropy, Energy).
   - Key concept: “Entropy Differential” (ΔH) as a safety/stability signal.
   - Docs: [`../docs/research/linguistic_thermodynamics.md`](../docs/research/linguistic_thermodynamics.md), [`../docs/research/entropy_differential_safety.md`](../docs/research/entropy_differential_safety.md)
   - Code touchpoints: `../src/modelcypher/core/domain/inference/entropy_dynamics.py`, `../src/modelcypher/core/domain/dynamics/`, `../src/modelcypher/core/domain/entropy/`, `../src/modelcypher/core/domain/safety/circuit_breaker_integration.py`
   - CLI touchpoints: `mc thermo …`, `mc geometry safety …`

3. **Paper 3: Unified Manifold Alignment** (Engineering)
   - Draft: [`paper-3-unified-manifold-alignment.md`](paper-3-unified-manifold-alignment.md)
   - Premise: Disparate models can be stitched together into a single coherent manifold using invariant anchors.
   - Key concepts: “Manifold Stitching” and “Frankenstein Models”.
   - Docs: [`../docs/geometry/manifold_stitching.md`](../docs/geometry/manifold_stitching.md), [`../docs/geometry/intersection_maps.md`](../docs/geometry/intersection_maps.md), [`../docs/research/cross_lora_transfer.md`](../docs/research/cross_lora_transfer.md), [`../docs/research/manifold_swapping.md`](../docs/research/manifold_swapping.md)
   - Code touchpoints: `../src/modelcypher/core/domain/geometry/manifold_stitcher.py`, `../src/modelcypher/core/domain/geometry/generalized_procrustes.py`, `../src/modelcypher/core/domain/geometry/gromov_wasserstein.py`, `../src/modelcypher/core/domain/geometry/transport_guided_merger.py`
   - CLI touchpoints: `mc geometry stitch …`, `mc model merge …`, `mc model analyze-alignment …`

## Notes

- These are drafts; some internal experiment artifacts referenced in the manuscripts may not be fully ported into `../docs/` yet.
- For current command names and code pointers, treat `../docs/` and `../src/modelcypher/` as the source of truth.
