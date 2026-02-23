# ModelCypher Geometry Guide (for AI + humans)

This guide explains what the geometry tooling measures and how to report the outputs accurately.
It is written for AI agents that call the CLI tools and then summarize results for humans.

Notes:
- In this repo, run commands as `poetry run mc ...`.
- Global CLI options can appear anywhere on the command line (example: `mc model info ./model --output text`).

Related docs:
- [MATH-PRIMER.md](MATH-PRIMER.md) - Intuition for the underlying geometry (distance/angle/alignment)
- [GLOSSARY.md](GLOSSARY.md) - Shared vocabulary for geometry concepts

Deep dives:
- [research/math/gromov_wasserstein.md](research/math/gromov_wasserstein.md) - Gromov-Wasserstein distance theory
- [research/manifold_alignment.md](research/manifold_alignment.md) - Cross-model alignment and intersection maps
- [research/topological_fingerprints.md](research/topological_fingerprints.md) - Persistent homology for model signatures
- [research/mental_model.md](research/mental_model.md) - Visual intuition for geometry concepts
- [research/dimensional_hierarchy.md](research/dimensional_hierarchy.md) - Alignment order (binary -> vocab -> activations)

---

## Why Geometry Matters [PROVEN]

When you merge two models by averaging their weights, you're assuming knowledge is stored in the same coordinates. Often it isn't: even when models learn similar features, they can be stored in rotated/permuted bases.

```mermaid
graph LR
    subgraph Naive["Naive Merge: Average Weights"]
        A1[Model A Layer 12] -->|0.5| M1[Merged]
        B1[Model B Layer 12] -->|0.5| M1
        M1 -->|?| X1[Collision]
    end

    subgraph Geometric["Geometric Merge: Align First"]
        A2[Model A Layer 12] --> P[Procrustes Align]
        B2[Model B Layer 12] --> P
        P --> M2[Merged]
        M2 -->|preserved| Y[Both Skills Intact]
    end

    style X1 fill:#f99,stroke:#933
    style Y fill:#9f9,stroke:#393
```

**Procrustes alignment** estimates an orthogonal transform that best aligns one representation space to another. This preserves geometric relationships while putting both models in a comparable coordinate system.

### The Rotation Problem

Two models trained on the same data can learn identical knowledge in rotated coordinate systems:

```
Model A: "cat" → [0.8, 0.2, 0.1]
Model B: "cat" → [0.2, 0.8, 0.1]  ← Same concept, rotated representation
```

Averaging these gives `[0.5, 0.5, 0.1]`—which is neither cat. Procrustes finds the rotation matrix that aligns them first.

### The Interference Problem

When concepts overlap in merged weight space, they interfere. ModelCypher predicts this *before* you merge:

```bash
mc --output text geometry interference predict /path/to/source_model /path/to/target_model
```

If high interference is predicted, you can use **null-space projection** to merge only in directions that don't collide.

### The Mathematical Foundation

ModelCypher applies established theory:

| Concept | Source | Application |
|---------|--------|-------------|
| Procrustes analysis | Gower (1975) | Aligning representation spaces |
| CKA similarity | Kornblith et al. (2019) | Comparing representations across models |
| Persistent homology | Naitzat et al. (2020) | Topological fingerprints |
| Information geometry | Amari & Nagaoka (2000) | Curvature in learning dynamics |

See [docs/references/BIBLIOGRAPHY.md](references/BIBLIOGRAPHY.md) for citations.

### The Bottom Line

> **Benchmarks measure outputs. Geometry measures structure.**
>
> You can game outputs. You can't fake topology.

## Inference Is Geometric Composition (Not Additive Layering) [PROVEN]

For a fixed prefix, inference is a composition of transformations on the same
state, not a process where each layer "adds new information":

```text
h_0 = Embed(prefix)
h_{l+1} = T_l(h_l)
h_L = (T_{L-1} ∘ ... ∘ T_1 ∘ T_0)(h_0)
```

In transformer blocks, each `T_l` typically has residual structure:

```text
h'_l    = h_l + Attn_l(LN_1(h_l))
h_{l+1} = h'_l + MLP_l(LN_2(h'_l))
```

The state is transformed repeatedly; it is not a layer-by-layer semantic
"construction" in the additive sense.

Key implications:
- **Order matters**: composition is non-commutative (`T_1(T_2(h)) != T_2(T_1(h))`).
- **Mechanism is geometric**: trajectory through representation space carries the signal.
- **Probability is readout**: distribution is computed from the terminal state.

Readout step:

```text
logits_t = W_out h_{L,t} + b
p(token_t | prefix) = softmax(logits_t)
```

`softmax` is a projection of terminal geometry into token likelihoods. It is
the observable shadow of where the trajectory landed, not the internal
mechanism that produced the trajectory.

Across autoregressive generation, decoding uses this readout to choose the next
token, which seeds the next forward pass. That loop boundary uses probability;
the in-pass mechanism remains geometric composition.

```mermaid
graph LR
    A["Prefix state h0"] --> B["T0"]
    B --> C["T1"]
    C --> D["..."]
    D --> E["TL-1"]
    E --> F["Terminal state hL (object)"]
    F --> G["Readout: logits = W_out hL + b"]
    G --> H["Softmax distribution (shadow)"]
    H --> I["Decode next token"]
    I --> J["Next pass with updated prefix"]
```

## Mental model (plain language)

- We treat weights, activations, and response trajectories as points in a very high-dimensional space.
- Geometry metrics summarize shape: curvature (flat vs sharp), distance (similar vs different),
  and direction (is training moving toward a known risk direction).
- Distances: smaller means closer. Similarities: larger means more similar.
- Many outputs are normalized (often, but not necessarily, to 0–1). Treat them as measurements, not grades.

## The "No Vibes" Principle

**Report raw measurements. Let the geometry speak for itself.**

ModelCypher deliberately avoids:
- Hardcoded thresholds ("0.7 is good")
- Qualitative labels ("excellent", "poor", "concerning")
- Interpretation strings that encode subjective judgment

Instead, we provide:
- **Raw measurements** - the actual geometric quantities
- **Baseline comparisons** - how this model compares to reference distributions
- **Z-scores and percentiles** - where this measurement falls relative to baselines

**Why?** Thresholds are model-specific, task-specific, and evolve over time. A researcher knows their domain; we provide measurements, they decide meaning.

## Numerical thresholds (dtype-derived)

When a threshold or epsilon is needed, derive it from the array dtype using
`numerical_stability` utilities (e.g., `division_epsilon`, `machine_epsilon`).
Avoid fixed constants like `1e-8` unless justified by data or machine precision.

## Spectral Capacity (Weight Space) [PROVEN]

`mc model capacity` analyzes 2D weight tensors directly in weight space and reports
per-layer spectral capacity measurements for LoRA planning.

Core formulas (for singular values `sigma_1 >= sigma_2 >= ...`):
- `effective_rank = (sum_i sigma_i)^2 / sum_i (sigma_i^2)`
- `stable_rank = ||W||_F^2 / ||W||_2^2`
- `numerical_rank_f32 = count(sigma_i > sigma_1 * sqrt(2^-23))`
- `numerical_rank_f16 = count(sigma_i > sigma_1 * sqrt(2^-10))`
- `null_space_dim_f32 = min(m, n) - numerical_rank_f32`
- `capacity_utilization = effective_rank / min(m, n)`
- `recommended_rank = argmax_r (sigma_r / sigma_{r+1})` (largest relative spectral gap)

CLI examples:
```bash
poetry run mc model capacity /path/to/model --output text
poetry run mc model capacity /path/to/model --target-modules q_proj --target-modules v_proj --min-dim 512
poetry run mc model capacity /path/to/model --sort-by recommended-rank --emit-lora-config ./configs/lora_capacity.yaml
```

Operational notes:
- The analysis is measure-first; it reports raw per-layer values.
- If backend SVD fails on an ill-conditioned layer, the analyzer falls back to
  Gram-eigenvalue decomposition, then iterative power-deflation estimation.
- Use `--target-modules` and `--min-dim/--max-dim` to constrain candidate
  layers before exporting LoRA rank configs.

## Intrinsic Dimension vs Support Manifold [PROVEN]

Intrinsic dimension (ID) is a local diagnostic of how many degrees of freedom
the data occupies around each point. It does not specify how large the
functional subspace must be to route and compose those concepts through layers.

ModelCypher treats "support manifold size" as the effective rank of centered
activation covariance, which captures how many directions carry variance.
This is the measurement that bounds the routing and composition overhead.

Formulas (implementation references in `src/modelcypher/core/domain/geometry/intrinsic_dimension.py`
and `src/modelcypher/core/domain/geometry/effective_rank.py`):

- TwoNN ID (Facco et al.): regress log(mu) vs log(1 - F(mu)), where mu = d2 / d1.
- Renyi effective rank: r_R = (sum_i lambda_i)^2 / sum_i (lambda_i^2)
- Shannon effective rank: r_S = exp(-sum_i p_i * log(p_i)), p_i = lambda_i / sum_i lambda_i

Here lambda_i are eigenvalues of the activation covariance (equivalently, the
singular values squared from centered activations). ID and effective rank are
complementary: ID estimates local manifold complexity, while effective rank
estimates how much support the model uses to carry that complexity.

Derived diagnostics reported by `mc analyze dimension-profile`:
- support ratio = effective_rank / ambient_dim (Renyi + Shannon)
- null ratio = 1 - support ratio (Renyi + Shannon)
- ID gap = effective_rank - intrinsic_dimension (Renyi + Shannon, when ID is available)

## Verification-Depth Profiling (Analyze-Only)

`mc analyze verification-depth-profile` measures how manifold observability changes
as probe verification depth increases (level ladder 0..4 from probe metadata).

First pass scope: diagnostics only. No merge gating and no merge-stage behavior changes.

Grouping modes:
- `cumulative`: use probes with `verification_depth <= level`
- `exact`: use probes with `verification_depth == level`

Per-layer raw metrics:
- `activation_rank`: numeric rank from positions-only spectrum (`X_l`)
- `trajectory_rank`: numeric rank from trajectory spectrum (`T_l = concat(X_l, V_l)` when velocities exist, else `X_l`)
- `intrinsic_dimension`: TwoNN ID from trajectory samples
- `condition_number`: Gram condition number from trajectory samples
- `hidden_dim`
- `probe_sample_count`
- `trajectory_sample_count`
- `d_plus_1_minimum = hidden_dim + 1`
- `d_plus_1_gap = (hidden_dim + 1) - probe_sample_count`
- `coverage_ratio_probe = probe_sample_count / hidden_dim`
- `coverage_ratio_trajectory = trajectory_sample_count / hidden_dim`
- `null_rank = hidden_dim - trajectory_rank`

Plateau diagnostics (across levels):
- `rank_plateau_level`: first level where canonical `trajectory_rank` equals final-level canonical rank
- `id_plateau_level`: first level where canonical ID is within dtype-derived tolerance of final-level canonical ID  
  `abs(ID_l - ID_final) <= sqrt(eps) * max(1, abs(ID_final))`
- `plateau_disagreement = id_plateau_level - rank_plateau_level`

## Manifold Evidence (Toward Theorem) [EMPIRICAL]

To move from thesis to theorem, we need evidence that:
1) representations occupy a low-dimensional manifold,
2) the manifold is curved (non-zero curvature),
3) local tangent dimension is stable (constant-rank behavior).

The `mc analyze dimension-profile` command reports:
- intrinsic dimension (TwoNN),
- effective rank (support manifold),
- tangent-space effective rank (log-map at Fréchet mean),
- sectional curvature summary.

These are raw measurements; they do not claim certainty by themselves, but they
are the empirical checks required to justify the assumptions behind a
constant-rank manifold theorem.

## Prompt-Manifold Jacobian Evidence [EMPIRICAL]

The `mc analyze jacobian-trace` command measures how many prompt
manifold directions *functionally* influence a layer's activation.

It does this by:
- deriving a prompt-manifold basis from pooled atlas prompt embeddings,
- sampling coefficient points from those same prompts,
- estimating the effective rank of the Jacobian (via random projections) of
  layer activations with respect to prompt-manifold coordinates.

If the intrinsic dimension is the semantic seed, this Jacobian rank estimates
the *support manifold* needed to route that seed through the network. It is a
data-derived diagnostic, not a thresholded interpretation.

## Geodesic Distance (Core Principle) [PROVEN]

**When ModelCypher reports a distance in representation space, it is usually geodesic (k-NN graph shortest path), not raw Euclidean.**

LLM representations live in high-dimensional spaces (768D to 8192D+). Euclidean distance—the straight-line "as the crow flies" distance—can become less informative in high dimensions due to the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). In extreme regimes, many points appear similarly distant under Euclidean measure.

**Geodesic distance** measures distance along the data manifold itself—like measuring road distance between cities rather than straight-line distance through the earth. This captures the true structure of the representation space.

### How it works

1. **Build a k-NN graph**: Connect each point to its k nearest neighbors (using Euclidean for initial edges—the one unavoidable bootstrap step)
2. **Compute shortest paths**: Use a shortest-path algorithm (e.g., Dijkstra) to find distances through the graph
3. **Report geodesic distances**: All subsequent distance computations use these manifold-aware distances

### What this means for you

- **Distance values are not directly comparable to Euclidean benchmarks**. A geodesic distance of 5.0 doesn't mean "5 units apart in space"—it means "5 hops through the nearest-neighbor graph."
- **Distances scale with manifold complexity**. A convoluted manifold will have larger geodesic distances between the same Euclidean-close points.
- **Comparisons are meaningful**. "Model A is 2x farther from B than from C" is geometrically meaningful, even if the raw numbers aren't.

### Why not Euclidean?

Euclidean distance is a special-case approximation that can work well in low-dimensional or near-linear settings. Geodesic distances better match curved manifold structure. Rather than maintain multiple mental models (“this command is Euclidean, that one is geodesic”), ModelCypher aims to report manifold-aware distances consistently, unless a subcommand explicitly says otherwise.

## Quick translation rules

- Report the raw metric values, not interpretations
- If baseline data is available, report the z-score or percentile
- If a metric is missing or null, say "not enough signal" rather than guessing
- Use 1 to 2 sentences in human summaries. Focus on measurements and comparisons.

## Analogy discipline (how to stay non-sci-fi)

Analogies make high-dimensional geometry intuitive, but they are not mechanisms.

- Pair an analogy with the **exact metric/artifact name** (e.g., "Venn diagram" → `IntersectionMap` overlap on a probe corpus).
- State the **measurement context** (probe corpus, decoding settings, layer) that makes the analogy applicable.
- Explicitly state what it **does not imply** (e.g., overlap on probes ≠ identical "knowledge", and a low-entropy regime ≠ "reasoning").

## What these metrics can and cannot tell you

They *can* help you:
- detect that something changed (drift, instability, unusual updates),
- localize where it changed (layers/components, when captured),
- decide when to investigate (circuit breaker style signals).

They *cannot*:
- certify a model is "safe",
- replace eval suites, policy review, or red teaming,
- establish causality ("metric went up, therefore X happened").

## Evidence Suite (Reproducible Measurements)

Use the evidence suite to generate raw measurements for alignment generalization,
geodesic/curvature convergence, and causal intervention effects.

```bash
# Cross-model reasoning geometry validation
poetry run mc analyze reasoning-geometry-validation \
  --model LFM2-350M \
  --benchmark arithmetic \
  --samples 20

# Per-layer geodesic deviation profile
poetry run mc analyze geodesic-profile --model /path/to/model --prompt "test"
```

**Synthetic evidence (docs/research/evidence.json):**
- Alignment generalization: train CKA=1.0, holdout CKA=0.9428425351, raw holdout CKA=0.8271114849, gain=0.1157310502, coverage=0.5249410713 (train=16, holdout=16).
- Geodesic convergence (circle): mean abs error small=0.0025231966, large=0.0006330672, ratio=0.2508988842; mean rel error small=0.0152261965, large=0.0074759978, ratio=0.4909957512.
- Curvature convergence (sphere, analytic curvature=1.0): mean abs error small=0.9999991556, large=0.9999983711, ratio=0.9999992156.
- Causal intervention: core mean shift=5.1242599487, core residual mean=5.8014249802, boundary max diff=0.0, boundary tolerance=0.0003452670.

**Real-model domain alignment (docs/research/evidence_real_models.json):**
- Models: SmolLM-135M vs LFM2-350M (layerA=0, layerB=0, probes=24).
- Domain `factual`: train CKA=1.0, holdout CKA=0.6864164951, raw holdout CKA=0.9581221736, gain=-0.2717056786, coverage=0.0019518083.

## Atlas Probe System

The atlas system provides 4596 semantic concept probes organized by domain and source. These probes are used for alignment, bridge generation, and manifold analysis.

### Probe Sources (23 categories)

Probes are loaded from JSON files in `data/probes/` and organized by source:

| Source | Description |
|--------|-------------|
| `semantic_prime` | Universal semantic primitives (Wierzbicka) |
| `computational_gate` | Code patterns (LITERAL, VARIABLE, LOOP, etc.) |
| `emotion_concept` | Affective concepts (joy, fear, anger, etc.) |
| `moral_concept` | Moral foundations (care, fairness, sanctity, etc.) |
| `temporal_concept` | Time-related concepts (past, future, duration) |
| `spatial_concept` | Space-related concepts (location, direction) |
| `social_concept` | Social relationships and concepts |
| `philosophical_concept` | Abstract philosophical ideas |
| `safety_ethics` | Safety-related probes |
| `physical_existence` | Physical world concepts |
| `compositional` | Compositional reasoning patterns |
| `conceptual_genealogy` | Etymology and concept evolution |
| `metaphor_invariant` | Cross-domain metaphor mappings |
| `conceptual_metaphor` | Conceptual metaphor theory |
| `syntax_concept` | Syntactic structure concepts |
| `perceptual` | Perception-related concepts |
| `numeric` | Number and quantity concepts |
| `common_object` | Everyday objects |
| `action_verb` | Action and motion verbs |
| `abstract_relation` | Abstract relations |
| `pronoun_perspective` | Pronoun and perspective concepts |
| `sequence_invariant` | Sequence patterns (Fibonacci, etc.) |
| `domain_specific` | Domain-specific concepts |

### JSON Probe Format

Each probe file follows this structure:

```json
{
  "domain": "moral",
  "probe_count": 23,
  "probes": [
    {
      "id": "moral_concept:cruelty",
      "name": "Cruelty",
      "description": "Deliberate infliction of suffering.",
      "support_texts": [
        "Cruelty is the willful causing of pain.",
        "To be cruel is to delight in suffering."
      ]
    }
  ]
}
```

### Adding Custom Probes

To add custom probes:
1. Create a new JSON file in `data/probes/`
2. Follow the format above with unique probe IDs
3. The probe loader will automatically include them

### Usage in Bridge Generation

The `mc merge bridge` command uses atlas probes:

```bash
# Default - uses all 4596 probes, achieves CKA = 1.0
mc merge bridge ./model_a ./model_b -o bridge.safetensors

# Filter by specific sources
mc merge bridge ./model_a ./model_b -o bridge.safetensors --probe-sources semantic_prime,emotion_concept
```

Atlas probes systematically span the semantic manifold. Procrustes alignment achieves CKA = 1.0 on training probes by construction. [PROVEN]

**Validation workflow**:
- Run `mc analyze reasoning-geometry-validation` for cross-model probe/pivot/topology measurements.
- Review raw outputs in the `analysis/` subdirectory of your specified output path.
- Review the `VALIDATION_REPORT.md` generated in the output directory.

This keeps the analysis reproducible through the CLI while preserving raw geometric measurements.

---

## Alignment Solver (Closed Form) [PROVEN]

Linear alignment uses the closed-form invariant:

```text
F = pinv(source) @ target
```

We solve this with closed-form normal equations on the backend:

- If `n >= d`: `F = (A^T A + λI)^{-1} A^T B`
- If `n < d`: `F = A^T (A A^T + λI)^{-1} B`

`λ` is derived from dtype machine epsilon (scale = max diag of the Gram),
so the system is strictly positive definite and the solve is well-defined.
No iterations; residuals are reported as raw measurements.

---

## Tool-by-tool explanations

### mc train status

Key fields:
- Training step, loss, val_loss, learning rate
- MASS step size (ceiling, SPS, Weyl), adapter saturation ratio
- Stopping certificate status

How to report:
- "Loss is 1.27 at step 420. Budget ratio is 0.95."

### mc analyze circuit-breaker

Key fields:
- `severity`: 0 to 1 aggregate safety signal.
- `dominantSource`: The strongest contributing signal (when available).

How to report:
- "Circuit breaker severity is 0.82. Dominant source: persona drift."

### mc analyze persona

Key fields:
- `overallDriftMagnitude`: 0 to 1 measure of alignment drift.
- `driftingTraits`: Which persona traits are moving most.

How to report:
- "Persona drift magnitude is 0.32. Highest drift in: helpfulness, directness."

### mc adapter analyze

Key fields:
- `amplification_cv`: Spectral selectivity of adapter deltas.
- `weyl_utilization`: Fraction of theoretical singular value shift bound used.

How to report:
- "Amplification CV is 1.23. Weyl utilization is 0.45."

## Example translations (JSON -> human)

### Example: mc train status

**JSON output:**
```json
{
  "step": 420,
  "loss": 1.27,
  "val_loss": 1.31,
  "learning_rate": 0.0001,
  "budget_ratio": 0.95
}
```

**Human summary:**
Loss is 1.27 at step 420. Val loss is 1.31. Budget ratio is 0.95.

### Example: mc analyze circuit-breaker

**JSON output:**
```json
{
  "severity": 0.82,
  "dominantSource": "persona_drift"
}
```

**Human summary:**
Circuit breaker severity is 0.82. Dominant source: persona drift.

### Example: mc adapter analyze

**JSON output:**
```json
{
  "adapter_path": "./adapters/adapter.safetensors",
  "base_model": "./models/base",
  "amplification_cv": 1.23,
  "weyl_utilization": 0.45
}
```

**Human summary:**
Amplification CV is 1.23. Weyl utilization is 0.45.

## Glossary (short)

- Gromov-Wasserstein distance: Compares the shape of two point clouds without requiring exact alignment.
- Procrustes alignment: Best-fit rotation (and optional scaling) to align two spaces.
- DoRA decomposition: Splits weight changes into magnitude (scale) and direction (rotation).
- DARE sparsity: Measures how sparse adapter updates are and their distribution.
- Gate path: A sequence of detected computational motifs in a response.

## If you only remember one thing

Report raw measurements with baseline context when available. Let the user interpret meaning.
