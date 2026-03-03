# Bayesian Geometry Paper: Connection to ModelCypher Causal Chain

**Status:** Technical mapping document (2026-03-03)
**Purpose:** Record how Agarwal, Dalal & Misra (2026) formalizes the entropy→geometry step
in ModelCypher's empirically-validated causal chain, and what it adds/confirms.

---

## 1. Paper Reference

Agarwal, N., Dalal, S.R., Misra, V. (2026, Jan 27).
"The Bayesian Geometry of Transformer Attention."
arXiv:2512.22471v3.

**Method:** "Bayesian wind tunnels" — controlled tasks with analytically known posteriors
(bijection elimination, HMM state tracking, associative recall, Bayesian regression).
Memorization provably impossible. Small transformers reproduce Bayesian posteriors with
10⁻³–10⁻⁴ bit accuracy (MAE); capacity-matched MLPs fail by orders of magnitude.

---

## 2. What the Paper Shows

**Theorem 1 (CE minimizer = Bayesian posterior predictive).**
The population minimizer of cross-entropy loss is the Bayesian posterior predictive
distribution q*(y|x,c) = ∫ p(y|x,θ) p(θ|c) dθ. Architecture-agnostic — true for any model
that minimizes CE. Follows directly from Bayes' rule and factorization (y ⊥ c | x,θ).

**Layer 0: Hypothesis frame formation.**
Keys form near-orthogonal basis over input tokens: mean absolute off-diagonal cosine = 0.052
(vs 0.082 for random; 37% reduction, p < 0.001). Single "hypothesis-frame head" is
catastrophically important — ablating it severely disrupts calibration. Attention maps at
Layer 0 are stable across training checkpoints.

**Progressive QK sharpening (middle layers).**
Each layer provides a non-interchangeable refinement step: ablating any single middle layer
increases error >10×. Early layers: broad attention across all tokens. Deeper layers:
concentrated attention on feasible hypotheses only. Formally: mirrors multiplicative
suppression in analytic Bayesian updates (each layer eliminates one step of inconsistent
hypotheses).

**Value manifold at final checkpoint: 1D, parameterized by posterior entropy.**
At intermediate training checkpoints: value representations of low-entropy states collapse,
cannot encode distinctions among small remaining hypothesis sets. At final checkpoint:
states lie on a smooth 1D manifold with posterior entropy as the coordinate.

**Frame-precision dissociation.**
Attention routing (frame) is stable throughout training. Value manifold (precision) improves
continuously. These decouple: the WHERE (attention routing) is fixed early; the HOW WELL
(value encoding) improves with more training. Predicted by gradient analyses of differential
convergence.

**Mamba: 5-cluster geometry.**
Final-layer Mamba representations form exactly 5 discrete clusters (one per HMM hidden state).
Within-cluster variation encodes entropy. Entropy prediction R²=0.40 (Layer 9; vs LSTM 0.004
— near random). Principal component 1 explains only 21.9% variance (vs LSTM 92.3%), meaning
Mamba maintains a distributed multi-dimensional representation. SSM selective state transitions
perform belief transport; attention handles binding.

**Architecture comparison (3 inference primitives):**
| Architecture | Accumulation | Transport | Binding | Primitives |
|---|---|---|---|---|
| Transformer | ✓ (0.007 bits) | ✓ (0.049 bits) | ✓ (100%) | 3/3 |
| Mamba | ✓ (0.010 bits) | ✓ (0.024 bits) | ~✓ (97.8%, slow) | 2.5/3 |
| LSTM | ✓ (0.009 bits) | ✗ (0.411 bits) | ✗ (0.5%) | 1/3 |
| MLP | ✗ (1.85 bits) | ✗ | — | 0/3 |

Mamba outperforms transformer on belief transport (HMM: 0.024 vs 0.049 bits).

---

## 3. Precise Alignment with ModelCypher's Causal Chain

### ModelCypher's empirical chain:
```
GQA → K capacity → QK alignment → Attention selectivity → Entropy → Curvature → ID → Phases
```

### Paper's mechanism:
```
Orthogonal key bases (L0) → Progressive QK sharpening (middle) →
Value manifold parameterized by entropy (late) → Frame-precision dissociation
```

### Point-by-point alignment:

**GQA → K capacity ↔ Orthogonal key bases**
The paper shows key orthogonality is a learned geometric property at Layer 0. ModelCypher
provides the cross-architecture explanation: GQA determines how many independent K-head
directions exist (r(log(GQA), L0_align) = -0.88). Fewer K heads → less capacity for
orthogonal key coverage → earlier forced alignment. The paper studies a single small
architecture; ModelCypher has the GQA variation that explains WHY orthogonality varies.

**QK alignment → Entropy ↔ Progressive QK sharpening**
ModelCypher: entropy < 0.3 = selective attention (−0.044 curvature avg); entropy > 0.8 =
diffuse attention (+0.043 curvature avg). Paper formalizes WHY: each layer does one step of
multiplicative Bayesian suppression of inconsistent hypotheses. Selective attention = most
hypotheses eliminated = lowest posterior entropy.

**Entropy → Curvature → ID ↔ Value manifold parameterized by entropy**
This is the key bridge. Paper: posterior entropy IS the 1D coordinate of the value manifold.
Low entropy → manifold is 1D (minimum dimension) → minimum curvature → minimum ID (highway).
High entropy → manifold is higher-dimensional → higher curvature → higher ID (processing).
ModelCypher's empirical chain (r=0.507, r=0.821) is the measurement of this Bayesian
manifold structure. Curvature = how far the manifold has "unfurled" from its
maximum-compression (minimum-entropy) state.

**Highway layers ↔ Frame-precision dissociation + Layer 0**
Paper: Layer 0 forms the hypothesis frame; attention routing is stable thereafter; value
manifold improves with training. Highway = layer where the frame is set = entropy minimum =
most certain posterior = most compressed = lowest-ID. The paper's "frame" (Layer 0 orthogonal
keys) is ModelCypher's "highway" (minimum-ID layer), both corresponding to the point of
maximum Bayesian certainty.

**LFM2 rank-1 attention ↔ Mamba 5-cluster geometry**
Paper: In Mamba-family architectures, SSM selective state transitions perform belief transport
(outperforming transformers: 0.024 vs 0.049 bits on HMM). Attention in Mamba handles only
binding (content-addressable routing), not transport. ModelCypher: LFM2 attention is rank-1
(Q/K near-orthogonal → uniform softmax). SSM layers handle structure; attention gets no
gradient signal for selectivity. The paper's mechanism explains this: in hybrid SSM/attention
architectures, SSM subsumes the transport role, leaving attention to become uniform/routing-only.

---

## 4. What This Formalizes for ModelCypher

**The entropy → geometry bridge is now theoretically grounded.**
Previously: empirical correlations (r=0.507, r=0.821) without a theoretical explanation for
the direction or mechanism. Now: entropy IS the sufficient statistic for the posterior
(provably, in structured tasks), and the value manifold's dimensionality is parameterized by
entropy. The causal chain has a Bayesian interpretation at each step.

**Highway = layer of maximum Bayesian accuracy.**
Previously: highway defined operationally (ID minimum). Now: the ID minimum is the layer
where the posterior is most accurately tracked — the frame is set and the precision is
maximized. This redefines highway in mechanistic terms.

**The LFM2 architecture finding is explained.**
Previously: observed that LFM2 attention layers are rank-1 because SSM makes attention
redundant. Now: this follows from the inference primitive taxonomy — SSM handles belief
transport, making the attention component's transport role redundant. Attention degenerates
to routing-only (the binding primitive), which explains the uniform/rank-1 pattern.

**Frame-precision dissociation predicts training dynamics.**
Previously: observed that DeepSeek-R1 shows constant expansion_ratio across tasks (MEMORY.md:
"RL training may produce stable geometric attractors"). Now: this is the frame-precision
dissociation. RL training refines the value manifold (precision) without changing the
attention routing (frame). Constant expansion_ratio = stable frame.

---

## 5. What This Paper Does Not Cover (ModelCypher's Remaining Contribution)

- **GQA cross-architecture variation**: Paper uses one fixed small architecture. ModelCypher
  has r(log(GQA), L0_align) = -0.88 across Granite/Llama/Qwen families — the architecture
  term explaining WHY key orthogonality varies across production models.
- **Scale terms**: No scaling laws. ModelCypher has validated 350M–8B.
- **The specific curvature measure**: Paper uses entropy R² and manifold dimensionality.
  ModelCypher uses signed Riemannian curvature — a more fine-grained measure.
- **Hybrid architectures (LFM2)**: Paper tests standard transformers and Mamba separately;
  ModelCypher has the hybrid (10 Mamba + 6 attention) case.
- **Production training pipeline**: MASS, Cayley-Stiefel, zero-HP — entirely orthogonal.
- **Null-space model merging**: No connection.

---

## 6. Theoretical Completion of the MI → Linear CKA Replacement

The Bayesian Geometry paper + ModelCypher's injectivity argument together close the case:

```
1. Theorem 1 (Agarwal et al.):
   CE minimization → network tracks Bayesian posterior predictive (architecture-agnostic).
   Every trained residual LLM converges toward Bayesian posterior tracking.

2. Injectivity (ModelCypher):
   h_l = h_0 + Σ_{k<l} δ_k → injective map (fixed weights, deterministic).
   → Shannon MI(h_0; h_l) = H(h_0) for all l (constant, cannot decay).

3. Therefore:
   The residual stream does not compress information — it geometrically reorganizes it
   toward optimal Bayesian posterior tracking. Information is preserved (injectivity),
   representation geometry changes (manifold unfurls).

4. Correct observable:
   Linear CKA measures second-order relational geometry (dot-product Gram matrices).
   CKA_linear(H_i, H_j) captures how the manifold's relational structure changes across
   depth — which IS the Bayesian inference process, not an information-compression process.

5. P1-R (CKA depth-distance decay):
   Measures how the value manifold's geometric organization changes as more Bayesian
   suppression steps accumulate with depth. The 2/3 families confirmation under linear CKA
   (P1-R [EXPLORATORY]) measures this without sigma amplification.
```

---

## 7. Open Questions This Paper Raises for ModelCypher

**Q1: Can we run Bayesian wind tunnel evaluation on production models?**
The paper uses small models (857k params) with known posteriors. Could run the bijection
elimination task on LFM2-350M and Qwen3.5-0.8B to directly measure whether our models
track Bayesian posteriors and at what layer depth the entropy trajectory aligns with analytic.

**Q2: Does the frame-precision dissociation appear in our training checkpoints?**
If we save intermediate checkpoints during training: do attention maps stabilize early while
CKA (value geometry) continues to improve? This would directly confirm the dissociation.

**Q3: Does LFM2's SSM-dominated highway correspond to 5-cluster geometry?**
The paper shows Mamba forms 5 discrete clusters at the highway layer. Does LFM2's Layer 0-1
(SSM-dominant) show the same cluster structure? Testable with per-layer ID + clustering.

**Q4: Is the curvature measure capturing manifold unfurling?**
Our curvature measure (from Riemannian distances) should correlate with the manifold
dimensionality change the paper measures. If so, curvature is a proxy for the "unfurling"
progress and can be used at inference time (unlike the wind tunnel posteriors).

---

## 8. Reference Chain

```
CE minimization → Bayesian posterior (Theorem 1)
    ↓
Residual injectivity → MI constant (ModelCypher)
    ↓
Geometric reorganization (not compression)
    ↓
Value manifold parameterized by entropy (Agarwal et al.)
    ↓
Entropy → curvature → ID (ModelCypher empirical, r=0.507, r=0.821)
    ↓
Phases: highway (entropy min, 1D manifold) → processing (entropy max, high-D) → exit
    ↓
Linear CKA = correct observable (relational geometry, not MI)
```

---

## 9. Citation

```
@misc{agarwal2026bayesian,
    title={The Bayesian Geometry of Transformer Attention},
    author={Naman Agarwal and Siddhartha R. Dalal and Vishal Misra},
    year={2026},
    eprint={2512.22471},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    note={v3, 27 Jan 2026}
}
```
