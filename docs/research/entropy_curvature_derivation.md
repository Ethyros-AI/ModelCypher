# Entropy → Curvature: Sublayer Decomposition and Architecture Dependence

**Status:** `[EMPIRICAL]` — mechanism partially validated, architecture term identified
**Date:** 2026-03-03
**Prerequisite:** `docs/research/bayesian_geometry_connection.md` (Agarwal 2026 mapping)
**Companion:** `docs/research/entropy-curvature-derivation.md` — formal population-level
derivation (covariance pushforward, small-angle expansion, rank envelopes, falsifiers F1-F5)

---

## 1. Problem Statement

The causal chain has a weak link:
```
Entropy → Curvature   (r = 0.507, Spearman, 6 models)
```

This r=0.507 uses **logit entropy** (Entropy-Lens: project h_l through unembedding, measure
Shannon entropy of resulting distribution). It measures how certain the model is about the
next token at depth l. This is distinct from **attention weight entropy** (Shannon entropy of
softmax(QK^T/√d_k)), which measures how concentrated the attention pattern is.

The Bayesian geometry paper (Agarwal 2026) shows the value manifold is parameterized by
posterior entropy, suggesting a mechanism connecting entropy to geometric properties.
This document formalizes the sublayer decomposition and reports what the data shows.

---

## 2. Definitions

**Angular curvature** θ_l:
```
θ_l = arccos(cos_sim(h_in_l, h_out_l))
```
Geodesic angle between layer input and output on the unit sphere. Measured per-token,
averaged across probes.

**Sublayer decomposition:**
```
θ_attn_l = arccos(cos_sim(h_in_l, h_post_attn_l))     (attention contribution)
θ_mlp_l  = arccos(cos_sim(h_post_attn_l, h_out_l))     (MLP contribution)
```
where h_post_attn_l = h_in_l + Attn(norm(h_in_l)).

**Attention weight entropy** H_l:
```
H_l = mean_heads(-Σ_k α_k log α_k)
```
where α = softmax(QK^T/√d_k), evaluated at the last query position, averaged across heads.

**MLP angular gain** G_mlp_l:
```
G_mlp_l = θ_mlp_l / θ_attn_l
```

---

## 3. The Mixing-Misalignment Mechanism (Proposed)

### Attention sublayer

The attention output is:
```
o_l = Σ_j α_j V(x_j)
```

Decompose into components parallel and perpendicular to h_in:
```
o_∥ = (o_l · ĥ_in) ĥ_in
o_⊥ = o_l - o_∥
```

The angular change from attention is:
```
θ_attn_l = arctan(||o_⊥|| / ||h_in + o_∥||)
```

**Predicted entropy dependence:**
- Low H (concentrated): o_l ≈ V(x_k) for dominant token k. The perpendicular component
  depends on angle(V(x_k), h_in). If value vectors are misaligned with h_in (as suggested
  by Agarwal 2026's orthogonal key bases), θ_attn is large.
- High H (diffuse): o_l ≈ centroid of value vectors. Averaging reduces ||o_⊥|| via
  cancellation of perpendicular components across tokens. θ_attn is smaller.

**Prediction P1:** r(H, θ_attn) < 0 for standard transformers (higher entropy → smaller
attention angular contribution due to centroid averaging).

### MLP sublayer

The MLP transforms h_post_attn via a gated bilinear form:
```
MLP(x) = W_down(SiLU(W_gate @ x) ⊙ (W_up @ x))
```

The MLP is NOT a constant angular amplifier. It responds differently to:
- **Vertex-like inputs** (from concentrated attention): specific gate neurons activate
  maximally → large MLP angular contribution
- **Centroid-like inputs** (from diffuse attention): distributed activation → smaller
  MLP angular contribution

**Prediction:** r(H, θ_mlp) has the SAME sign as r(H, θ_attn), or the signs OPPOSE
(architecture-dependent).

---

## 4. Experimental Results

### 4.1 Models Tested

| Model | Architecture | GQA | Layers | Attention Layers |
|-------|-------------|-----|--------|-----------------|
| LFM2-700M | Hybrid (conv + attention) | 3 | 16 | 6 |
| Qwen3.5-0.8B | Hybrid (linear + full attention) | 4 | 24 | 6 |
| Qwen2.5-3B | Standard transformer | 2 | 36 | 36 |

30 probes across 5 categories (math, reasoning, factual, creative, code).
Script: `scripts/entropy_curvature_verification.py`.
Results: `results/entropy_curvature/entropy_curvature_results.json`.

### 4.2 Prediction Results

| # | Prediction | LFM2-700M | Qwen3.5-0.8B | Qwen2.5-3B | Verdict |
|---|-----------|-----------|-------------|-----------|---------|
| P1 | r(H, θ_attn) and r(H, θ_mlp) have opposite signs | **PASS** (+0.829, -0.600) | FAIL (-0.257, -0.314) | FAIL (-0.062, -0.086) | **REFUTED** as universal |
| P2 | Attention fraction decreases with entropy | FAIL (r=+0.829) | FAIL (r=+0.314) | PASS (r=-0.055) | **REFUTED** |
| P3 | GQA modulates correlation strength | — | — | — | **FAIL** (no monotonic GQA relationship) |
| P4 | MLP gain varies (CV > 0.1) | **PASS** (CV=0.401) | **PASS** (CV=0.177) | **PASS** (CV=0.739) | **VALIDATED** 3/3 |
| P5 | Value alignment predicts sign | NOT TESTED | NOT TESTED | NOT TESTED | — |
| P6 | MLP gain explains residual variance | **PASS** (r=-0.829) | **PASS** (r=-0.314) | FAIL (r=+0.055) | **EMPIRICAL** 2/3 |

### 4.3 Correlation Tables

**LFM2-700M (6 attention layers):**
| Pair | Spearman r | p-value |
|------|-----------|---------|
| H vs θ_total | +0.486 | 0.329 |
| H vs θ_attn | **+0.829** | **0.042** |
| H vs θ_mlp | -0.600 | 0.208 |
| H vs attn_fraction | +0.829 | 0.042 |
| H vs G_mlp | -0.829 | 0.042 |

**Qwen3.5-0.8B (6 full attention layers):**
| Pair | Spearman r | p-value |
|------|-----------|---------|
| H vs θ_total | -0.314 | 0.544 |
| H vs θ_attn | -0.257 | 0.623 |
| H vs θ_mlp | -0.314 | 0.544 |
| H vs attn_fraction | +0.314 | 0.544 |
| H vs G_mlp | -0.314 | 0.544 |

**Qwen2.5-3B (36 attention layers):**
| Pair | Spearman r | p-value |
|------|-----------|---------|
| H vs θ_total | -0.036 | 0.835 |
| H vs θ_attn | -0.062 | 0.718 |
| H vs θ_mlp | -0.086 | 0.617 |
| H vs attn_fraction | -0.055 | 0.749 |
| H vs G_mlp | +0.055 | 0.749 |

---

## 5. Key Findings

### 5.1 Attention weight entropy ≠ logit entropy

The causal chain's r=0.507 uses **logit entropy** (Entropy-Lens). The experiments here
measure **attention weight entropy** (Shannon entropy of softmax weights). These are
different quantities:

- **Logit entropy** measures posterior certainty about the next token at depth l.
  It reflects the cumulative effect of all processing up to layer l.
- **Attention weight entropy** measures how concentrated the attention pattern is
  at layer l. It reflects only the current layer's attention mechanism.

The attention weight entropy shows essentially NO significant correlation with angular
curvature on standard transformers (Qwen2.5-3B: r=-0.036, p=0.835). The correlation
appears only on LFM2's hybrid architecture (r=+0.829 for θ_attn, significant at p=0.042).

**Implication:** The Bayesian geometry paper's "value manifold parameterized by posterior
entropy" (Agarwal 2026) connects to logit entropy (which captures posterior state), not
to attention weight entropy (which captures the current layer's routing). The r=0.507
in the causal chain is measuring posterior entropy evolution, not attention concentration.

### 5.2 MLP gain is NOT constant — VALIDATED 3/3

MLP angular gain G_mlp = θ_mlp / θ_attn varies substantially within each model:

| Model | G_mlp mean | G_mlp std | CV |
|-------|-----------|----------|-----|
| LFM2-700M | 1.28 | 0.52 | 0.401 |
| Qwen3.5-0.8B | 1.50 | 0.27 | 0.177 |
| Qwen2.5-3B | 2.04 | 1.51 | 0.739 |

The MLP is not a constant amplifier of the attention signal. This confirms Section 5
of OPEN-MATHEMATICAL-QUESTIONS.md (MLP nonlinearity geometry): the gate × up bilinear
form creates architecture- and layer-dependent angular contributions.

For Qwen2.5-3B, G_mlp ranges from 0.23 (layer 1, early) to 10.0 (layer 35, final) — a
43× range. The systematic increase with depth means the MLP contribution dominates
increasingly as depth increases (attention fraction drops from 0.71 at layer 0 to 0.09
at layer 35).

### 5.3 Architecture dependence — sign opposition is LFM2-specific

The mixing-misalignment mechanism (P1: opposite signs of r(H, θ_attn) and r(H, θ_mlp))
holds only for LFM2-700M. On LFM2:
- Higher attention entropy → larger θ_attn (+0.829) but smaller θ_mlp (-0.600)
- The attention and MLP sublayers compete: more entropy helps attention but hurts MLP

On standard transformers (Qwen2.5-3B), both correlations are near zero and same-signed.
The mechanism is not universal — it depends on whether the architecture uses hybrid
conv/attention (LFM2) or pure attention (Qwen).

**Possible explanation:** LFM2's convolution layers (ShortConv) handle the transport
function (Agarwal 2026's taxonomy), leaving the attention layers to focus on binding.
This specialization may create a different relationship between attention concentration
and curvature than in standard transformers where attention handles both transport and
binding.

### 5.4 Attention fraction trajectory

For Qwen2.5-3B (36 layers), the attention fraction θ_attn/θ_total shows a clear
depth-dependent trajectory:

```
Layer 0:  0.71  (attention-dominated)
Layer 5:  0.39
Layer 10: 0.38
Layer 15: 0.37
Layer 20: 0.32
Layer 25: 0.30
Layer 30: 0.20
Layer 35: 0.09  (MLP-dominated)
```

This is consistent with the Bayesian geometry paper's frame-precision dissociation:
early layers set the frame (attention-dominated), later layers refine precision
(MLP-dominated). The transition is monotonic, not a sudden switch.

---

## 6. Revised Mechanism: Logit Entropy as the Correct Observable

Given that attention weight entropy shows no significant correlation with curvature
on standard transformers, the causal chain's entropy → curvature link should be
understood as:

```
Logit entropy (Entropy-Lens) → Curvature   (r = 0.507)
```

This is consistent with the Bayesian geometry interpretation:
1. Logit entropy measures posterior certainty at depth l
2. When posterior is certain (low logit entropy), the representation is near the 1D
   value manifold → low-dimensional → low curvature
3. When posterior is uncertain (high logit entropy), the representation spans more
   dimensions of the value manifold → higher curvature

The attention weight entropy is an UPSTREAM variable that influences logit entropy
but does not directly predict curvature. The pathway is:

```
Attention weight entropy → (through value mixing + MLP processing) → Logit entropy → Curvature
```

The indirect pathway explains why the correlation is moderate (r=0.507, ~25% variance):
the MLP nonlinearity and the accumulated effect of prior layers add noise between
attention weight entropy at any single layer and the resulting curvature.

---

## 7. Falsification Status

| Prediction | Status | Evidence |
|-----------|--------|----------|
| P1 (sign opposition) | **REFUTED** as universal | LFM2 only; 0/2 standard transformers |
| P2 (attn fraction decreases with H) | **REFUTED** | 1/3 pass (barely), direction inconsistent |
| P3 (GQA modulates correlation) | **FAIL** | No monotonic GQA relationship |
| P4 (MLP gain varies) | **VALIDATED** 3/3 | CV = 0.177–0.739 |
| P5 (value alignment predicts sign) | NOT TESTED | Requires V weight extraction |
| P6 (MLP gain explains residual) | **EMPIRICAL** 2/3 | Significant for LFM2, trend for Qwen3.5 |

**Mixing-misalignment mechanism verdict:** Architecture-dependent. Not a universal
explanation for the entropy → curvature link. The universal explanation comes from
the Bayesian manifold interpretation (logit entropy parameterizes value manifold
dimensionality), with the attention sublayer decomposition being a secondary,
architecture-specific effect.

---

## 8. Contrast with Codex Falsifier Tests

Codex added falsifier tests F1, F3, F4, F5 in `scripts/curvature_accumulation_analysis.py`.
Key conflicts with these experimental results:

| Codex Test | Prediction | My Data Shows |
|-----------|-----------|---------------|
| F1 (slope H→θ_attn² ≥ 0) | Higher H → higher θ_attn² | Near-zero on Qwen2.5-3B (r=-0.062) |
| F3 (\|corr(H, θ_attn)\| > \|corr(H, θ_mlp)\|) | Attention dominates | **MLP dominates 2/3 models** |
| F5 (same sign across families) | Universal sign | **LFM2 is opposite sign from Qwen** |

F3 is the most significant conflict: Codex predicts attention-entropy should correlate
more strongly with attention curvature than MLP curvature. Data shows the opposite for
standard transformers. The MLP response to its input (which is conditioned on attention)
is the dominant curvature contributor, and its relationship to attention entropy is
architecture-dependent.

**Codex `try_compute_attn_entropy` implementation issue:** Does not apply post-projection
norms (q_norm, k_norm) and uses module-level `n_heads` attribute, which gives wrong
values on Qwen3.5 (returns 8 from linear attention config instead of 16 for full
attention layers). The implementation in `scripts/entropy_curvature_verification.py`
handles this correctly via `_get_head_config()`.

---

## 9. Updated Causal Chain Assessment

The entropy → curvature link remains `[EXPLORATORY, r=0.507]` but with refined understanding:

**Clarifications:**
1. The "entropy" in the chain is **logit entropy** (Entropy-Lens), not attention weight entropy
2. The mechanism is Bayesian manifold dimensionality (Agarwal 2026), not attention mixing
3. The sublayer decomposition reveals architecture-dependent contribution ratios
4. MLP dominance increases monotonically with depth (attention fraction: 0.71→0.09 on Qwen2.5-3B)

**What would promote to `[VALIDATED]`:**
- Logit entropy → curvature correlation replicated with decomposition on ≥3 model families
- Bayesian manifold dimensionality measured directly (not just proxy through curvature)
- Architecture term derived: why MLP gain trajectory differs between hybrid and pure-attention

**What this rules out:**
- Attention weight concentration as the direct curvature mechanism (no significant correlation on standard transformers)
- Constant MLP gain assumption (refuted 3/3)
- Universal mixing-misalignment mechanism (LFM2-specific)

---

## 10. References

- Agarwal, Dalal & Misra (2026). "The Bayesian Geometry of Transformer Attention."
  arXiv:2512.22471v3. Full mapping: `docs/research/bayesian_geometry_connection.md`.
- `docs/research/entropy-curvature-derivation.md`: Formal population-level derivation
  (Σ_α → Σ_y → projected orthogonal energy → θ², with rank envelope bounds and
  E_mix proxy). Contains falsifiers F1-F5 and pre-registered predictions P-EC1 through
  P-EC4. This document provides the theoretical framework; the present document
  provides the empirical validation and architecture-dependence findings.
- Section 5 of OPEN-MATHEMATICAL-QUESTIONS.md: MLP nonlinearity geometry (gate × up
  bilinear form as curvature source).
- `results/entropy_curvature/entropy_curvature_results.json`: Raw experimental data.
- `scripts/entropy_curvature_verification.py`: Measurement script.
