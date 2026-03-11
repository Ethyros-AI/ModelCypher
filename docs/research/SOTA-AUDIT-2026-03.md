# SOTA Audit: ModelCypher vs. Industry & Academic State of the Art

**Date:** 2026-03-03 (narrative audit); updated 2026-03-04 (10-model entropy-curvature evidence + 6-model F2/B5 refinement sync)
**Method:** Three independent audits (2 Claude + 1 Codex/Firecrawl), 80+ papers, 11 sub-areas
**Canonical Source:** This file is the single maintained SOTA audit narrative.
**Artifacts:** `results/sota_audit_2026_03/` (claim IDs, crosswalk, scorecard, action map)
**Verdict:** We occupy an underdeveloped intersection — geometry-first theory + practical
training/merging controls. The community has observations; we have mechanisms. The community
has tooling; we have derivations. Neither side has both.

**Important (2026-03-11):** The causal chain (Section 7) is mostly [EXPLORATORY] — most
links lack formal causal derivation per `FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`. The novelty
claim is structural (linking observations end-to-end), not that each link is proven.
Zero-hyperparameter training has not yet been demonstrated to produce better inference
outcomes than standard LoRA. RESEARCH-ROADMAP R1 is the active blocker.

---

## Executive Summary

The community has independently converged on the same observations we made — ID phases,
geometric over probabilistic analysis, Cayley-Stiefel optimization. ModelCypher has the
**causal explanation** (not just observation), the **end-to-end first-principles derivation**
(no tunable constants), and the **injectivity proof** (cleanest resolution of the information
bottleneck problem for residual networks).

The information bottleneck problem is widely recognized as broken for deterministic networks.
The community has pragmatically abandoned it in favor of CKA/ID analysis, but without a clean
proof of WHY MI decay can't work. Our injectivity argument (h_l = h_0 + Σδ_k, injective,
so I(X; h_l) = H(X) for all l) is more rigorous than any resolution in the IEEE TPAMI 2024
IB survey.

The null-space projection merging approach and the hyperparameter-free training pipeline have
no published counterparts in their specific formulations, though both areas have active
neighbors (CL null-space methods, Stiefel LoRA variants).

**Best strategy:** Import commodity tooling (MergeKit, HF PEFT baselines). Spend research
budget only on bedrock-mechanism questions and proofs. "Beats baseline" is insufficient
without a mechanism test matching our claim contract.

---

## Quick Reference: STOP / ADAPT / PUSH

### STOP — Already Solved

| Area | External Status | Our Status | Action |
|------|----------------|------------|--------|
| Norm growth through layers | Documented since 2023 (Pre-LN exponential) | I1 VALIDATED 3/3 | Keep as sanity check |
| ID-based phase detection | Cagnetta et al. (ICLR 2025), Ansuini (NeurIPS 2019) | Phase detection works | Cite them; our contribution is the causal chain ABOVE |
| Init-time theories fail on trained nets | Becoming consensus | 5/5 refuted | Meta-lesson captured, not publishable |
| Weight space Euclidean (P ≈ I) | Karakida 2021 | VALIDATED cross-family | Known result, not novel |
| Merge infrastructure / eval plumbing | MergeKit mature | Custom implementation | Import commodity tooling |
| PEFT method zoo | HF PEFT ships PiSSA/EVA/DoRA/rsLoRA etc. | Cayley-Stiefel only | Don't rebuild; benchmark against |

### ADAPT — Import and Benchmark

| Area | Priority | What To Do |
|------|----------|-----------|
| MergeBench protocol | HIGH | Head-to-head: null-space vs TA/TIES/DARE/RegMean++ |
| PEFT baselines | HIGH | Add PiSSA/EVA/LoRA+/DoRA/rsLoRA as mandatory controls |
| Null-space CL literature | HIGH | Articulate difference from GPM/GPCNS/MINGLE |
| GRIDE over TwoNN | MEDIUM | Upgrade ID estimator (scikit-dimension) |
| Cayley-Stiefel competition | AWARENESS | Benchmark against StelLA (NeurIPS 2025) |
| CKA limitations | AWARENESS | Feature-sampling bias, manipulability documented |

### PUSH — Genuine Frontier

| Area | Novelty | Status | Blocker |
|------|---------|--------|---------|
| Full causal chain (GQA → Phases) | HIGHEST | EXPLORATORY upstream, VALIDATED downstream | Entropy → curvature formalization |
| DPI violation mechanism | HIGH | PROVEN + VALIDATED 3/3 | None — publishable now |
| MI decay impossibility (injectivity) | HIGH | PROVEN | None — publishable now |
| Zero-HP training pipeline | MEDIUM | VALIDATED (single family) | PEFT baseline comparison |
| Null-space merging formulation | HIGH | VALIDATED 1.2B | 8B validation + MergeBench |
| DPI-compatible information observable | FRONTIER | Open | Fundamental research |
| Architecture + scale terms | OPEN | Split outcomes across models | Needs formal predictions |
| Expansion ratio at inference time | NOVEL | EMPIRICAL | Systematic validation |
| CKA/PPL ≠ behavioral preservation | NOVEL | Single experiment | Replication |
| Sigma calibration (IEEE 754) | NOVEL | VALIDATED 3/3 | Documentation as method |

---

## Detailed Analysis by Area

### 1. Mutual Information in Neural Networks

**Status: Active research, not solved. Estimation is the core bottleneck.**

Key estimators: MINE (popular but high variance at large MI), SMI/mSMI (principled, ICML 2024
achieved non-vacuous generalization bounds), DSE (graph-spectral, noise-resistant, ICMLW 2023),
matrix-based Rényi (Giraldo 2014, our approach, kernel-based, no density estimation needed).

The IB framework is pragmatically abandoned for deterministic networks. Community uses ID/CKA
instead. Our contributions:

- **MI decay impossibility** via injectivity: h_l = h_0 + Σδ_k is injective → I(X; h_l) = H(X).
  Goldfeld et al. (2019) proved MI preservation for continuous maps generally; our
  residual-specific version is standalone and cleaner than the IEEE TPAMI 2024 survey.
  References: arXiv 1810.05728, 2003.09671, 2510.15511.
- **DPI violation mechanism**: L2 normalization breaks the Markov property. Not in the
  literature. Peri-LN (2025) describes the symptom without the info-theoretic mechanism.
- **Sigma calibration**: IEEE 754-derived Gram matrix non-degeneracy constraints. Not a named
  method. Standard approach is the median heuristic (Garreau & Jitkrittum 2018).

Matrix-based Rényi (Giraldo 2014) is known in the info-theoretic learning subcommunity but
not mainstream DL. Recent: Nystrom approximation (Neural Networks, Oct 2025) for scaling.

### 2. Intrinsic Dimension and Phase Detection

**Status: Active. TwoNN widely used but GRIDE superseding it. Phase detection is a 2024-2025
finding gaining traction.**

Key paper: **Cagnetta et al. "Emergence of a High-Dimensional Abstraction Phase in Language
Transformers"** (ICLR 2025, arXiv 2405.15471). Across 5 transformer families, distinct high-ID
phase in intermediate layers. First full linguistic abstraction. Cross-model universality.
Earlier onset predicts better language modeling. Uses GRIDE + Information Imbalance.

Also: Ansuini et al. (NeurIPS 2019) for the "hunchback" ID profile. "The Shape of Learning"
(EACL 2024) for training-time ID dynamics. NeurIPS 2025 paper on localized TwoNN for tokens.

**Our position:** ID phase detection itself is NOT novel — cite Cagnetta et al. Our
contribution is the causal chain explaining WHY phases emerge (GQA → K capacity → QK
alignment → entropy → curvature → ID → phases). Nobody else has this.

**Action:** Consider upgrading TwoNN to GRIDE (scikit-dimension). Low urgency for results,
but reviewers will ask.

### 3. Residual Stream Analysis / Mechanistic Interpretability

**Status: Exploding. Anthropic-led at micro level; phase detection at macro level.**

Norm growth: well-documented. Pre-LN admits exponential growth (~1.045× per layer in GPT-2-XL).
Peri-LN (2025, Gemma/OLMo) constrains it.

Phase detection under various names: Activation Transport Operators, DenseFormer (NeurIPS 2024),
"Stages of Inference?" (NeurIPS 2025, behavioral), Cagnetta et al. (geometric).

Anthropic: Scaling Monosemanticity (May 2024), Circuit Tracing (March 2025 — cross-layer
transcoders, causal graphs). This is micro-level; our work is macro-level. Complementary.

Tooling: TransformerLens v3, NNSight, nnterp (2025, 50+ model families). We built our own
for MLX — fine for our purposes but not competing with this ecosystem.

**"The Bayesian Geometry of Transformer Attention" (Agarwal, Dalal, Misra — arXiv:2512.22471v3,
Jan 2026)** — Highly relevant. Uses "Bayesian wind tunnels" (controlled tasks with known
analytic posteriors, memorization provably impossible) to show transformers implement Bayesian
inference geometrically. Key findings:
- Theorem 1: CE minimizer = Bayesian posterior predictive (architecture-agnostic).
- Layer 0: orthogonal key bases (37% more orthogonal than random; p<0.001). Single frame
  head is catastrophically important — ablating it disrupts calibration.
- Progressive QK sharpening: each layer provides non-interchangeable suppression step;
  ablating any single layer causes >10× error increase.
- Value manifold at final checkpoint: 1D, coordinate = posterior entropy. Frame-precision
  dissociation: attention routing stable through training, value manifold unfurls continuously.
- Mamba: 5-cluster geometry (one per HMM state); R²=0.40 for entropy prediction (vs LSTM
  0.004); outperforms transformer on belief transport (0.024 vs 0.049 bits MAE).
- Architecture comparison: Transformer 3/3 inference primitives, Mamba 2.5/3, LSTM 1/3, MLP 0/3.

Alignment to ModelCypher: directly formalizes entropy→curvature→ID chain. Their "value
manifold parameterized by posterior entropy" is the theoretical interpretation of our
empirically measured entropy→Δcurvature (r=0.507) and cumulative curvature→ID (r=0.821).
Their "progressive QK sharpening" = our QK alignment measurements. Their Mamba findings
explain LFM2 rank-1 attention (SSM handles transport, attention degenerates to routing-only).
Full mapping: `docs/research/bayesian_geometry_connection.md`.

"Belief State Geometry in Residual Streams" (NeurIPS 2024) — Earlier Bayesian framing.
Complementary. See 2512.22471 for the more complete 2026 version.

### 4. Spectral Analysis of Activations

**Status: CKA established but under scrutiny. Spectral entropy of activation Gram matrices
is niche.**

CKA (Kornblith 2019): 1600+ citations, default metric. Known limitations: manipulable (ICLR
2022), feature-sampling bias (Murphy 2024, Chun 2025), RBF collapses to linear at high
bandwidth (Alvarez-Melis, IEEE TPAMI 2022). Our finding that CKA ≠ behavioral preservation
is consistent with this literature.

DSE (Diffusion Spectral Entropy, ICMLW 2023): graph-spectral approach, noise-resistant.
Closest to our spectral entropy work but uses diffusion matrices, not activation Gram matrices.

WeightWatcher / HTSR theory: spectral analysis of weight matrices (not activations). Heavy-tail
universality.

### 5. Model Merging

**Status: Extremely active, rapidly evolving, no clear winner.**

Established baselines: Task Arithmetic, TIES (NeurIPS 2023), DARE, SLERP. Tool: MergeKit.
Newer: EMR-Merging (NeurIPS 2024, tuning-free), AIM (NeurIPS 2025, activation-informed),
RobustMerge (NeurIPS 2025, SVD-aware), Evolutionary (Sakana AI, Nature MI 2025).

Null-space in CL: GPM (ICLR 2021, gradients into orthogonal complement of past tasks),
GPCNS (ACM MM 2024, common null space), MINGLE (2025, null-space gated experts), null-space
filtering (arXiv 2509.21413, data-free continual merging).

**Our formulation is distinct:** We project weight deltas into the null space of target
activations (not gradients into past task activation null spaces). Tikhonov-weighted
pseudoinverse + Marchenko-Pastur noise edge. This specific combination is not published.

**Gap:** Validated only to 1.2B. TIES/DARE validated at 7B-70B. 8B validation + MergeBench
comparison are the critical scale gates.

### 6. Cayley-Stiefel / Riemannian LoRA Training

**Status: Active, rapidly crowding. 4+ independent groups in 2025.**

- StelLA (NeurIPS 2025 Spotlight, Sony) — USV^T with U,V on Stiefel. arXiv 2510.01938
- Riemannian LoRA (EMNLP 2025) — B matrix on Stiefel. arXiv 2508.17901
- Manifold-LoRA (ICLR 2025) — retraction-free. OpenReview GP30inajOt
- OrthoGeoLoRA (Jan 2026) — SVD-inspired. arXiv 2601.09185
- Foundational: Li et al. (ICLR 2020), Cayley transform for Stiefel. arXiv 2002.01113

**Our differentiator is not the Cayley transform** — it's the zero-hyperparameter integration.
MASS step size (all bounds SVD-derived), auto-regime (Clopper-Pearson CI), every constant from
SVD/IEEE 754/data. No other group claims zero tunable hyperparameters.

### 7. Full Causal Chain Discovery

**Status: NOVEL as complete chain. Individual links studied by 4-5 groups.**

| Link | External Work | Status |
|------|--------------|--------|
| GQA → K capacity | GQA papers (Ainslie 2023) | Partially studied |
| K capacity → QK alignment | Bayesian Geometry (arXiv 2512.22471, Jan 2026) | Studied (orthogonal key bases) |
| QK alignment → Attention selectivity | Multiple mech interp papers | Studied |
| Attention selectivity → Entropy | Bayesian Geometry (entropy-parameterized value manifold) | Studied |
| Entropy → Curvature | Ricci curvature papers (arXiv 2509.22362, ICLR 2025 submission) | Emerging |
| Curvature → ID | Ricci flow (class separability) | Partially |
| ID → Phases | Cagnetta et al. (ICLR 2025), Ansuini (NeurIPS 2019) | Established |
| **Full chain end-to-end** | **No one** | **ModelCypher [EXPLORATORY — most links not formally derived]** |

Closest competitor: Bayesian geometry group (3 papers, Jan 2026). They cover the middle of
the chain (QK alignment → entropy) with rigorous Bayesian formalization. **Read (2026-03-03).**
Key findings: Theorem 1 (CE minimizer = Bayesian posterior, architecture-agnostic); value manifold
1D parameterized by posterior entropy at final checkpoint; frame-precision dissociation (attention
routing stable, value manifold improves with training); Mamba R²=0.40 entropy prediction vs LSTM
0.004. Full mapping: `docs/research/bayesian_geometry_connection.md`.
**What they do NOT cover:** GQA cross-architecture variation (fixed small model), no scaling laws,
no signed curvature measure, no hybrid SSM/attention architectures. The `entropy → curvature`
derivation gap remains open; their value manifold result formalizes the direction but does not
provide the curvature calculation.

### 8. DPI and Normalization

**Status: DPI-as-Markov-chain framing established. Normalization-breaks-DPI connection is
NOT well-studied.**

DPI in DNNs: Shwartz-Ziv & Tishby (2017) modeled layers as Markov chain.
Normalization effects: Peri-LN (2025) shows entanglement, not the info-theoretic mechanism.
Our proof that L2 normalization breaks the Markov property (X̃_{l+1} ≠ f(X̃_l) because scale
is lost) appears to be original. Scoped to matrix-Rényi, not Shannon DPI.

---

## Papers to Read Immediately

| # | Paper | arXiv/Venue | Why |
|---|-------|-------------|-----|
| ~~1~~ | ~~Bayesian Geometry of Transformer Attention (3 papers)~~ | ~~2512.22471, 2512.22473, 2512.23752~~ | **Read 2026-03-03.** Formalizes QK→entropy middle chain. `entropy→curvature` gap remains open. See `bayesian_geometry_connection.md`. |
| 2 | Emergence of High-Dimensional Abstraction Phase | 2405.15471 (ICLR 2025) | ID phases — cite and position relative to |
| 3 | Neural Feature Geometry as Discrete Ricci Flow | 2509.22362 | Curvature across 20K networks |
| 4 | Null-Space Filtering for Continual Merging | 2509.21413 | Compare merge approach |
| 5 | StelLA: Stiefel LoRA | 2510.01938 (NeurIPS 2025) | Benchmark against |
| 6 | MergeBench | 2505.10833 | Evaluation protocol |
| 7 | Injectivity in deterministic nets | 2510.15511 | Validates our direction |
| 8 | AIM: Activation-Informed Merging | (NeurIPS 2025) | Closest merge competitor |
| 9 | Constrained Belief Updates in Transformers | 2502.01954 (ICML 2025) | Attention as Bayesian inference |
| 10 | Belief State Geometry in Residual Streams | (NeurIPS 2024) | Complementary framing |

---

## Publication Priority

| # | Finding | Novelty | Self-Contained? | Blocker |
|---|---------|---------|-----------------|---------|
| 1 | DPI violation mechanism (normalization breaks Markov) | HIGH | YES | None |
| 2 | MI decay impossibility (injectivity in residual nets) | HIGH | YES | None |
| 3 | Full causal chain (GQA → Phases) | HIGHEST | NO | Entropy→curvature formalization |
| 4 | Null-space merging formulation | HIGH | NO | 8B validation + MergeBench |
| 5 | Zero-hyperparameter training | MEDIUM | NO | PEFT baselines |

---

## Experiment Queue (Next Actions)

1. Read Bayesian geometry papers (arXiv 2512.22471 series)
2. MergeBench head-to-head: null-space vs TA/TIES/DARE/RegMean++
3. PEFT baselines: PiSSA/EVA/DoRA/rsLoRA as controls in training evals
4. StelLA benchmark comparison
5. 8B null-space merge validation
6. GRIDE upgrade for ID estimator (medium priority)
7. Formalize entropy → curvature link (highest value, hardest)

---

## What Mainstream ML Doesn't Do (Our Real Differentiator)

Three audits converged on the same structural observation about what makes this work different
from the mainstream:

1. **Claim protocol** — derivation + falsifier + commensurability check before any experiment.
   Mainstream ML rarely enforces this.
2. **Derivable controls** — training hyperparameters from SVD/IEEE 754 instead of grid search.
   No published paper removes ALL hyperparameters via geometry.
3. **Operational geometry** — metrics linked to runtime decisions, not just post-hoc analysis.
   Expansion ratio at inference time has no published counterpart.

This combination — rigorous protocol + first-principles derivation + operational application —
is the intersection nobody else occupies.
