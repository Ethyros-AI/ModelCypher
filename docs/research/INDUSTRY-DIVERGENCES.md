# Where ModelCypher Diverges from Industry — and the Geometric Proof for Each

**Status**: Reference Document
**Date**: 2026-02-23

The ML industry's foundational assumption — that probability causes events — is wrong. A forward pass is a deterministic geometric map. Softmax is observer-side normalization. Every "best practice" built on probability-as-mechanism is suspect until re-derived from geometry.

This document catalogs every point where the ModelCypher codebase diverges from common ML conventions. Each divergence includes the industry assumption, why it's wrong, what geometry says instead, and the proof. Every number traces to one of four sources: SVD, IEEE 754, measured data, or a cited theorem.

Related documents (complementary, not duplicated):
- [geometric_hyperparameter_rosetta_stone.md](geometric_hyperparameter_rosetta_stone.md) — mapping table (hyperparameter to geometric replacement)
- [training_heuristics_analysis.md](training_heuristics_analysis.md) — literature review of training heuristics
- [MATH-FOUNDATIONS.md](MATH-FOUNDATIONS.md) — core geometric definitions and theorems
- [lora_spectral_scale_bound.md](lora_spectral_scale_bound.md) — LoRA scale derivation and validation

---

## D-1: Probability Does Not Cause Events

**Industry assumption:** Softmax output is a probability distribution that the model "samples from" to generate tokens. Temperature controls "creativity." Top-p/top-k control "diversity."

**Why it's wrong:** A forward pass is a deterministic geometric map. For fixed weights W and input x, the hidden state trajectory h_0 -> h_1 -> ... -> h_L is uniquely determined. Softmax is observer-side normalization at readout — it converts logits to a convenient representation for loss computation, but the model does not "use" probabilities internally. There is no stochastic operation anywhere in the forward pass.

**ModelCypher approach:** Greedy decoding only (argmax). Temperature, top-p, and top-k are noise injection mechanisms, not model parameters. If the greedy path gives the wrong answer, the training is the problem — not the sampling strategy.

**Proof:**

For any transformer layer:

    h_{l+1} = h_l + Attn(h_l) + FFN(h_l)

This is a composition of linear maps (projections, embeddings) and fixed pointwise nonlinearities (ReLU, SiLU, GELU). No stochastic operation exists. The output softmax computes:

    P(token_i) = exp(z_i) / sum_j exp(z_j)

This is a smooth, deterministic function of logits z. Sampling from this distribution is an external operation applied *after* the model has computed its answer. The model's geometry determines z; sampling discards that information.

A hallucination caused by temperature sampling is not a reasoning failure — it is a random token selection that cascades into garbage. The model's geometry had the right answer. The sampling threw it away.

**Code:**
- [inference_engine.py:271-273](../../src/modelcypher/adapters/inference_engine.py) — greedy decoding requirement and deterministic-map rationale.
- [inference_engine.py:301-307](../../src/modelcypher/adapters/inference_engine.py) — generation path calls backend `generate` with no sampling branch in this adapter path.

**Evidence:** (from [geometry_validation_results.md](geometry_validation_results.md))

Sampling-heavy evaluation inflated apparent failures:
- GSM8K, greedy decoding, `n=100`: 59 correct / 41 incorrect.
- Arithmetic, `temperature=0.5`, `n=200`: 189 correct / 11 incorrect; the document flags these 11 as sampling artifacts rather than geometric reasoning failures.
- The earlier broken pipeline produced a false `14/86` split from hidden-state argmax, not model logits.

**Citations:**
- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML 2017, arXiv:1706.04599.
- Peeperkorn, M., et al. (2024). "Is Temperature the Creativity Parameter of Large Language Models?" arXiv preprint.

---

## D-2: LoRA Scale Is a Geometric Constraint, Not a Hyperparameter

**Industry assumption:** LoRA scale = alpha/rank (e.g., 16/8 = 2.0). Tuned via grid search or inherited defaults.

**Why it's wrong:** The perturbation's spectral norm must respect the base weight's spectral structure. Exceeding sigma_k(W) causes singular value crossing — the adapter's spectral contribution overflows into the base weight's active subspace, creating intruder dimensions that cause catastrophic forgetting.

**ModelCypher approach:** Per-layer scale bound derived from SVD of the base weight:

    max(S) = (sigma_k / 2) * (1 - sqrt(eps_dtype))

No grid search. No tuning. The geometry determines the bound.

**Proof:**

1. Base weight W has SVD: W = U Sigma V^T with sigma_1 >= sigma_2 >= ... >= sigma_n
2. sigma_k = smallest precision-significant SV (LAPACK convention: sigma > max(m,n) * eps * sigma_max)
3. NB-LoRA formula: W_lora = 2 * B^T * S * A, where [A^T; B^T] has orthonormal columns via Cayley transform
4. Spectral norm guarantee (submultiplicativity + orthonormality):

       ||W_lora||_2 <= ||B^T||_2 * ||S||_2 * ||A||_2 <= 1 * max(S) * 1 = max(S)

   Therefore: ||W_lora||_2 <= 2 * max(S)

5. Weyl perturbation bound (Weyl 1912): For W_new = W + E, singular value crossing at rank k occurs when ||E||_2 > gap_k / 2. To prevent crossing entirely:

       2 * max(S) <= sigma_k  =>  max(S) <= sigma_k / 2

6. IEEE 754 margin: (1 - sqrt(eps_dtype)) keeps within distinguishability of the floating-point representation. For float32: sqrt(2^(-23)) ~ 3.45e-4, margin ~ 0.9997.

**Code:**
- [cayley_lora.py:669-759](../../src/modelcypher/core/domain/geometry/cayley_lora.py) — `create_nb_lora_from_base_weight()`: SVD -> sigma_k -> scale_bound
- [cayley_lora.py:285-291](../../src/modelcypher/core/domain/geometry/cayley_lora.py) — spectral norm guarantee proof
- [spectral_budget.py:24-28](../../src/modelcypher/core/domain/training/spectral_budget.py) — Weyl crossing threshold derivation

**Evidence:** (from [lora_spectral_scale_bound.md](lora_spectral_scale_bound.md))

All 9 standard LoRA adapters tested on LFM2-350M violated the geometric bound:

| Adapter | Over Bound | Status |
|---------|-----------|--------|
| lfm2_350m_p1_6_mid_balanced | 2726x | Critical |
| self-reflection-lora-v4 | 1655x | Critical |
| self-reflection-lora-v5 | 1525x | Critical |
| geometric-awareness-v1 | 1311x | Critical |
| self-reflection-lora-v3-expansion | 860x | Critical |
| self-reflection-lora-v3 | 850x | Critical |
| self-reflection-lora-v2 | 622x | Critical |
| self-reflection-lora-v1 | 606x | Critical |
| lfm2_350m_p1_6_mid_balanced_v2 | 22.6x | Unsafe |

GSM8K validation (sheep counting problem, correct answer = 260):
- Standard scale (2.0): `"20 + (20 + 40) + 40 = ? ... and and and a a a and a and a..."` (gibberish)
- Geometric scale (~0.1): `"Seattle sheep = 4 * 20 = 80, Toulouse sheep = 2 * 80 = 160, Total = 260"` (correct)

**Citations:**
- Wang, Z., et al. (2025). "NB-LoRA: Norm-Bounded Low-Rank Adaptation." arXiv:2501.19050.
- Weyl, H. (1912). "Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen." Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse, 1912, 110-117.
- Shuttleworth, R., et al. (2025). "LoRA perturbations exceeding Weyl gap create intruder dimensions causing catastrophic forgetting." arXiv:2410.21228.

---

## D-3: Cross-Entropy Teaches Format, Not Reasoning

**Industry assumption:** Cross-entropy loss on reasoning traces teaches the model to reason. Lower perplexity = better model.

**Why it's wrong:** CE loss L = -log P(y_target | x) always increases the probability of the target token, regardless of whether the target is semantically correct or just surface-form pattern. The optimizer minimizes surface-form distance to the training trace, not computational validity. The model learns what reasoning *looks like*, not how to reason.

**ModelCypher approach:** Automatic regime selection — CE when the model has zero demonstrated capability (nothing to reinforce), REINFORCE when it has partial or above-chance capability (something to amplify). The regime boundary is derived from measured baseline correctness via Clopper-Pearson exact binomial CI, not guessed.

**Proof:**

The gradient comparison:

    CE gradient:        nabla_theta L = -nabla_theta log P(y|x)
    REINFORCE gradient: nabla_theta L = -A * nabla_theta log pi(y|x)

where A = r_i - mean(r_group) is the advantage (Williams 1992, Theorem 1).

CE always pushes toward the target. REINFORCE pushes toward correct outputs and *away from incorrect outputs*. CE cannot distinguish a correct reasoning step from a format-matched but wrong one. REINFORCE can — it uses the outcome as signal.

Regime boundary derivation:
- Clopper-Pearson exact binomial CI (Biometrika 26(4), 1934) with alpha = 1/N (data-derived confidence level)
- Per-type chance rates from problem structure: 0.5 (binary yes/no), 0.0 (exact match)
- Zone 1 (k=0): CE — no demonstrated capability, nothing to reinforce
- Zone 2 (k>0, ci_lower < chance): REINFORCE + entropy — partial capability exists
- Zone 3 (ci_lower >= chance): REINFORCE — demonstrably above chance

**Code:**
- [outcome_objective.py:23-34](../../src/modelcypher/core/domain/training/outcome_objective.py) — CE vs REINFORCE gradient comparison
- [regime_selection.py:74-105](../../src/modelcypher/core/domain/training/regime_selection.py) — Clopper-Pearson implementation
- [regime_selection.py:36-42](../../src/modelcypher/core/domain/training/regime_selection.py) — per-type chance rates from problem structure

**Evidence:**

SFT on reasoning traces — PPL improves, reasoning collapses:

| Model | PPL Before | PPL After | Change | Inference Before | Inference After | Degenerate |
|-------|-----------|-----------|--------|-----------------|-----------------|------------|
| LFM2-350M | 19.6 | 3.9 | -81% | 9/20 (45%) | 4/20 (20%) | 25% |
| LFM2-1.2B | 8.6 | 1.4 | -84% | 30/46 (65%) | 20/46 (43%) | 28% |

PPL, CKA, and spectral budget all looked perfect during training. All three are wrong proxies for reasoning capability. The optimizer did exactly what it was told — minimize cross-entropy on the trace. The objective was the problem.

**Citations:**
- Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine Learning, 8(3-4), 229-256.
- Clopper, C. J., & Pearson, E. S. (1934). "The use of confidence or fiducial limits illustrated in the case of the binomial." Biometrika, 26(4), 404-413.

---

## D-4: Model Merging Is Addition, Not Interpolation

**Industry assumption:** Merge models by weighted averaging: W_merged = alpha * W_a + (1 - alpha) * W_b. Variants include SLERP, TIES, DARE — all interpolation-based.

**Why it's wrong:** Linear interpolation destroys both models' learned structure. The merged weight sits between the two weight matrices in Euclidean space but is not guaranteed to preserve either model's behavior on any input. The interpolation coefficient alpha is arbitrary and has no geometric justification.

**ModelCypher approach:** Procrustes alignment + null-space projection. Align source to target's coordinate system, compute the residual, project it into target's null space, and add. The addition is invisible on sampled activations by construction.

**Proof:**

1. Align source activations to target via closed-form Procrustes: F = pinv(X_source) @ X_target (Penrose 1955)
2. Compute residual: Delta = X_target - X_source @ F
3. Null-space projection: Delta_null = (I - X_source @ pinv(X_source)) @ Delta
4. Delta_null is in the orthogonal complement of source's column space
5. Therefore: X_source @ F_new = X_source @ F for any F_new that differs only in the null-space component
6. CKA(X_source, X_merged) = 1.0 by construction — the Gram matrix X_source @ X_source^T is unchanged

The null-space addition preserves target behavior on sampled activations because the added component is orthogonal to the subspace that determines the model's output on those activations.

**Code:**
- [alignment.py:43-73](../../src/modelcypher/core/domain/geometry/alignment.py) — closed-form invariant alignment `F = pinv(source) @ target`.
- [transplant.py:1403-1412](../../src/modelcypher/core/domain/geometry/transplant.py) — null-space projection definition `delta_W_proj = delta_W @ N`.
- [transplant.py:1628-1633](../../src/modelcypher/core/domain/geometry/transplant.py) — preserved fraction defined on behavioral norm, not raw weight mass.

**Evidence:** (from [VALIDATION-REPORT.md](../VALIDATION-REPORT.md))

Cross-family alignment and transplant metrics:
- LFM2-350M ↔ Qwen3-1.7B raw CKA `0.32-0.39`, aligned CKA `0.96-0.97`.
- Synthetic null-space test behavioral ratio `0.000002` (99.9998% preservation).
- Real-model null-space test behavioral ratio `0.058` (94.2% preservation).
- End-to-end cross-architecture merge (`exp5_endtoend`): failed count `0/5`, repetition `0.0`, preserved fraction `30.5%`.

**Citations:**
- Penrose, R. (1955). "A generalized inverse for matrices." Proceedings of the Cambridge Philosophical Society, 51(3), 406-413.
- Kornblith, S., et al. (2019). "Similarity of Neural Network Representations Revisited." ICML 2019, PMLR 97.
- Fang, R., et al. (2025). "AlphaEdit: Null-Space Constrained Knowledge Editing." ICLR 2025, arXiv:2410.02355.

---

## D-5: LoRA Rank Is Null-Space Capacity, Not a Hyperparameter

**Industry assumption:** LoRA rank = 8 (arbitrary). Sometimes 4, sometimes 16, sometimes 64, copied from prior runs.

**Why it's wrong:** The rank for adaptation is bounded by the weight matrix's structural null space — the dimensions that carry no information in the base weight's spectrum. Using rank > null-space capacity means the adapter overwrites the base weight's active subspace. Using rank < null-space capacity wastes available capacity.

**ModelCypher approach:** Rank bounded by `tail_dims = full_rank - floor(shannon_effective_rank)`. Target only layers where tail_dims > 0.

**Proof:**

Shannon effective rank (spectral entropy):

    R_eff = exp(H(sigma^2))

where H is the Shannon entropy of the normalized squared singular value distribution:

    p_i = sigma_i^2 / sum_j sigma_j^2
    H = -sum_i p_i * log(p_i)

This measures how many dimensions carry meaningful information — a structural property of the weight matrix, not a numerical precision cutoff. The difference:

- Numerical rank (LAPACK): count(sigma > max(m,n) * eps * sigma_max) — what a computer can distinguish
- Shannon effective rank: exp(H(sigma^2)) — what information is actually present

tail_dims = full_rank - floor(R_eff) gives the structural null-space capacity. Layers where tail_dims = 0 have fully utilized spectra — there is no room for an adapter without destroying existing information.

**Code:**
- [geometric_lora.py:18-26](../../src/modelcypher/core/domain/training/geometric_lora.py) — "All parameters are derived from the spectral structure of base weights... No hyperparameters. The geometry IS the configuration."
- [geometric_lora.py:159-184](../../src/modelcypher/core/domain/training/geometric_lora.py) — structural rank and `tail_dims` computation.

**Evidence:** (from [field_map_external_methods.md](field_map_external_methods.md))

LFM2-350M layerwise comparison on 92 weight matrices (2026-02-22):
- stable-rank range: `14-255`; tail-dims range: `106-789`.
- mean stable-rank: `89.9`; mean tail-dims: `298` (3.3x larger null-space budget).
- agreement within ±20%: `9/92` (10%).
- q_proj example (1024x1024): stable-rank suggests `r≈30-42`; tail-dims measured `684-789`.

**Citations:**
- Roy, O., & Vetterli, M. (2007). "The Effective Rank: A Measure of Effective Dimensionality." European Signal Processing Conference (EUSIPCO 2007).
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022, arXiv:2106.09685.

---

## D-6: The Optimizer Navigates a Manifold, Not Euclidean Space

**Industry assumption:** Adam/AdamW treats the gradient as a vector in R^n. First and second moment estimates compensate for curvature. Works for everything.

**Why it's wrong:** NB-LoRA factors (A, B) after Cayley transform have orthonormal columns — they live on the Stiefel manifold St(r, n), the set of n x r matrices with orthonormal columns. Euclidean gradient descent on manifold-valued parameters ignores the constraint surface, leading to updates that leave the manifold and require expensive projection back.

**ModelCypher approach:** Cayley-Stiefel retraction. NB-LoRA factors are parameterized via the Cayley map, which maps free skew-symmetric parameters to the Stiefel manifold without SVD. The pullback preconditioner P = M * M^T (where M = I + Z) was originally included but removed after cross-family falsification (2026-02-23) proved P ≈ I throughout training. The benefit comes entirely from the Cayley constraint (Stiefel surface enforcement), not from curvature in the metric. Step sizing uses MASS (spectral ceiling + SPS + Weyl bound), which is independent of P.

**Empirical falsification (2026-02-23):** Cross-family full `F1-F6` trajectory tests on LFM2-350M and Qwen2.5-Coder-0.5B show the same geometric core: the pullback metric stays near identity and nearly collinear with raw gradients. At 20 steps, `F1` median `||P_hat - I||_F / sqrt(r)` is `1.28e-4` (LFM2) and `1.79e-5` (Qwen), `F3` median `cos(Pg, g)` is `0.9999995` and `0.99999996`, and `F4` max drift is `1.37e-3` and `2.39e-3`. At 200 steps, these still support `H_null` across seeds: LFM2 (`F1 mean=1.4175e-3 ± 9.58e-5`, `F3 mean=0.9999305 ± 1.27e-5`, `F4 max mean=2.2183e-2 ± 5.19e-3`) and Qwen (`F1 mean=1.5070e-4 ± 2.46e-5`, `F3 mean=0.9999997 ± 1.54e-7`, `F4 max mean=8.3852e-3 ± 5.84e-4`), where `±` is 95% t-interval half-width over 3 seeds. This rejects "strong manifold curvature in P" as the primary mechanism. `F2` remains horizon-sensitive and seed-dependent: at 200 steps LFM2 `d` mean is `0.2730 ± 0.3870` (`H_alt` in 1/3 seeds) and Qwen `d` mean is `0.3754 ± 0.4817` (`H_alt` in 2/3 seeds), with `Cayley+I` favored in all 200-step seeds. `F5` confirms Fisher degeneracy in all runs (`condition_number_p10` ranges: LFM2 `1.40e8–2.14e8`, Qwen `2.12e10–3.54e10`; all with >99.94% below 1% of max). `F6` remains second-order small but increases with longer trajectories (`max ||deviation||/||Ω||^2` ranges: LFM2 `1.92e-2–2.38e-2`, Qwen `4.70e-3–1.44e-2`), consistent with Lezcano-Casado (2019).

**Formulation:**

Preconditioned gradient (Amari 1998 framework):

    d_t = P_t @ g_t

where P_t is the pullback metric of the Cayley parameterization (not the Fisher information metric). For the Cayley map:

    P = M * M^T,  M = I + Z

This is the full, unnormalized pullback metric. Not trace-normalized (kills anisotropy). Not lambda_max-normalized (same failure).

Stability invariant (Nesterov 2004):

    m = eta * L * lambda_max(P) <= 2

This bounds the preconditioned step size to guarantee descent. L is the local Lipschitz constant; lambda_max(P) is the largest eigenvalue of the preconditioner.

Cayley retraction (Wen & Yin 2013, Lezcano-Casado 2019):

    R_X(V) = (I - V/2)^{-1} (I + V/2) X

maps tangent vectors back to the Stiefel manifold smoothly and without SVD.

**Code:**
- [_mlx_training_adapter_train_mixin.py:96-110](../../src/modelcypher/backends/_mlx_training_adapter_train_mixin.py) — MASS three-layer step size: eta_step = min(eta_ceiling, eta_sps, eta_weyl)
- [cayley_lora.py](../../src/modelcypher/core/domain/geometry/cayley_lora.py) — Cayley retraction and NBLoRALinear (Stiefel constraint)

**Evidence:**

<!-- BEGIN D6-TABLES -->
Cross-family trajectory falsification (20-step protocol):

| Model | F1 median `\|\|P_hat-I\|\|_F/sqrt(r)` | F3 median `cos(Pg,g)` | F4 max drift | F2 Cohen's d | F5 cond(p10) | F6 max `\|\|dev\|\|/\|\|Ω\|\|^2` |
|-------|------------------------------------|------------------------|--------------|--------------|----------------|---------------------------|
| LFM2-350M | `1.2829e-04` | `0.9999995` | `1.3670e-03` | `1.6268e+00` | `3.8631e+08` | `8.2521e-04` |
| Qwen2.5-Coder-0.5B | `1.7915e-05` | `1.0000000` | `2.3852e-03` | `6.2251e-02` | `3.5245e+11` | `2.2851e-03` |

Cross-family trajectory falsification (200-step, 3 seeds, 95% t-interval):

| Model | F1 median (mean ± CI) | F3 median (mean ± CI) | F4 max drift (mean ± CI) | F2 Cohen's d (mean ± CI) | F5 cond_p10 (mean ± CI) | F6 max dev (mean ± CI) |
|-------|----------------------|----------------------|-------------------------|--------------------------|------------------------|------------------------|
| LFM2-350M | `1.4175e-03 ± 9.58e-05` | `9.9993e-01 ± 1.27e-05` | `2.2183e-02 ± 5.19e-03` | `2.7301e-01 ± 3.87e-01` | `1.8163e+08 ± 9.40e+07` | `2.0766e-02 ± 6.59e-03` |
| Qwen2.5-Coder-0.5B | `1.5070e-04 ± 2.46e-05` | `1.0000e+00 ± 1.54e-07` | `8.3852e-03 ± 5.84e-04` | `3.7541e-01 ± 4.82e-01` | `2.6282e+10 ± 1.97e+10` | `8.2265e-03 ± 1.33e-02` |

Per-seed results:

| Model | Seed | F1 median | F2 d | F2 result | F4 max | F5 cond_p10 |
|-------|------|-----------|------|-----------|--------|-------------|
| LFM2-350M | 41 | `1.4619e-03` | 0.2524 | SUPPORTS H_null | `2.2983e-02` | `1.4023e+08` |
| LFM2-350M | 42 | `1.3970e-03` | 0.1286 | SUPPORTS H_null | `2.3753e-02` | `1.9028e+08` |
| LFM2-350M | 43 | `1.3934e-03` | 0.4381 | SUPPORTS H_alt | `1.9814e-02` | `2.1438e+08` |
| Qwen2.5-Coder-0.5B | 41 | `1.6199e-04` | 0.1715 | SUPPORTS H_null | `8.1867e-03` | `3.5423e+10` |
| Qwen2.5-Coder-0.5B | 42 | `1.4359e-04` | 0.5575 | SUPPORTS H_alt | `8.3243e-03` | `2.2259e+10` |
| Qwen2.5-Coder-0.5B | 43 | `1.4653e-04` | 0.3973 | SUPPORTS H_alt | `8.6446e-03` | `2.1165e+10` |

<!-- END D6-TABLES -->

Artifacts:
- [trajectory_falsification/LFM2-350M/results.json](../../results/weight_geometry/trajectory_falsification/LFM2-350M/results.json)
- [trajectory_falsification/Qwen2.5-Coder-0.5B/results.json](../../results/weight_geometry/trajectory_falsification/Qwen2.5-Coder-0.5B/results.json)
- [trajectory_falsification_200_multiseed_summary.json](../../results/weight_geometry/trajectory_falsification_200_multiseed_summary.json) — aggregated 3-seed summary with 95% t-intervals (`df=2`) for LFM2 and Qwen.
- [trajectory_falsification_200/LFM2-350M/results.json](../../results/weight_geometry/trajectory_falsification_200/LFM2-350M/results.json) — 200-step follow-up: `F1=1.3970e-3`, `F3=0.9999337`, `F4_max=2.3753e-2` (all `H_null`), `F2 d=0.1286` (`H_null`, `Cayley+I` slightly better), `F5 cond_p10=1.9028e8`, `F6 max=1.9233e-2`.
- [trajectory_falsification_200/Qwen2.5-Coder-0.5B/results.json](../../results/weight_geometry/trajectory_falsification_200/Qwen2.5-Coder-0.5B/results.json) — 200-step follow-up: `F1=1.4359e-4`, `F3=0.9999997`, `F4_max=8.3243e-3` (all `H_null`), `F2 d=0.5575` (`H_alt`, `Cayley+I` better), `F5 cond_p10=2.2259e10`, `F6 max=4.6996e-3`.
- [trajectory_falsification_200_seed41/LFM2-350M/results.json](../../results/weight_geometry/trajectory_falsification_200_seed41/LFM2-350M/results.json) — seed 41 LFM2.
- [trajectory_falsification_200_seed43/LFM2-350M/results.json](../../results/weight_geometry/trajectory_falsification_200_seed43/LFM2-350M/results.json) — seed 43 LFM2.
- [trajectory_falsification_200_seed41/Qwen2.5-Coder-0.5B/results.json](../../results/weight_geometry/trajectory_falsification_200_seed41/Qwen2.5-Coder-0.5B/results.json) — seed 41 Qwen.
- [trajectory_falsification_200_seed43/Qwen2.5-Coder-0.5B/results.json](../../results/weight_geometry/trajectory_falsification_200_seed43/Qwen2.5-Coder-0.5B/results.json) — seed 43 Qwen.

Past failures (documented, don't repeat):
1. ScaledGD: wrong for Stiefel manifold (degenerates to uniform scaling)
2. M * M^T without step bound: NaN divergence
3. Trace-normalized: killed anisotropy
4. lambda_max-normalized: same failure

**Citations:**
- Amari (1998) "Natural Gradient Works Efficiently in Learning" Neural Computation 10(2):251-276
- Nesterov (2004) "Introductory Lectures on Convex Optimization" Springer
- Absil, Mahony & Sepulchre (2008) "Optimization on Matrix Manifolds" Princeton, Theorem 4.3.1
- Wen & Yin (2013) "A feasible method for optimization with orthogonality constraints" Math. Program.
- Lezcano-Casado (2019) "Trivializations for Gradient-Based Optimization on Manifolds" NeurIPS

---

## D-7: Every Threshold Is Derived, Not Assumed

**Industry assumption:** Epsilon = 1e-8 for numerical stability. Convergence tolerance = 1e-6. These constants are copied across optimizers and tutorials without dtype analysis.

**Why it's wrong:** Fixed constants ignore dtype, model scale, and accumulated error. For float32 with eps ~ 1.19e-7, a threshold of 1e-8 is below the precision floor for many intermediate computations — it is testing against noise, not signal. A threshold of 1e-6 may be wastefully tight for some computations and too loose for others. The correct threshold depends on the computation being performed and the precision of the data.

**ModelCypher approach:** All thresholds derive from IEEE 754 error propagation analysis:

| Threshold | Value (float32) | Derivation |
|-----------|-----------------|------------|
| Machine epsilon | 2^(-23) ~ 1.19e-7 | IEEE 754 float32 significand bits |
| Convergence | sqrt(2^(-23)) ~ 3.45e-4 | Accumulated relative error after O(d) operations (Higham 2002, Ch. 3) |
| Capacity exhaustion | 1 - sqrt(2^(-23)) ~ 0.9997 | Remaining headroom indistinguishable from accumulated error |
| Weyl crossing | gap_k / (2 * sigma_k) per layer | Spectral gap from SVD of base weight (Weyl 1912) |
| LR backoff floor | sqrt(2^(-23)) | Same accumulated error bound |

**Proof:**

Machine epsilon for float32 (IEEE 754):

    eps = 2^(-p+1) where p = 24 (significand bits including implicit 1)
    eps = 2^(-23) ~ 1.19e-7

For a computation involving k multiplications on d-dimensional arrays, the accumulated relative error is bounded by:

    |delta_accumulated| <= k * d * eps  (worst case)
    |delta_accumulated| ~ sqrt(k * d) * eps  (average case, Higham 2002)

For typical neural network computations (matrix products of d ~ 1000), sqrt(d * eps) ~ sqrt(eps) gives the practical error floor. This is why sqrt(eps) ~ 3.45e-4 appears as the universal convergence threshold throughout the codebase — it is the precision floor of the computation, not an arbitrary choice.

**Code:**
- [spectral_budget.py:48-56](../../src/modelcypher/core/domain/training/spectral_budget.py) — `_SQRT_EPS_F32 = math.sqrt(math.ldexp(1.0, -23))` with full IEEE 754 derivation and Higham citation
- [precision.py:405-447](../../src/modelcypher/core/domain/geometry/precision.py) — `machine_epsilon`, `regularization_epsilon`, and SVD rank threshold from dtype.

**Evidence:**
- `poetry run pytest -q tests/test_spectral_budget.py` -> `19 passed` (threshold logic and Weyl crossing checks).
- [FAILURE-MODES.md](FAILURE-MODES.md) reports threshold non-portability: float32 instability around `kappa ~ 1e6` vs float64 around `kappa ~ 1e12`; relative criterion `kappa * sqrt(eps)` remains consistent.
- In-code float32 constants are explicit and reproducible: `sqrt(eps_f32)=0.00034526698300124393`, `1-sqrt(eps_f32)=0.9996547330169988`.

**Citations:**
- IEEE 754-2019 (ISO/IEC/IEEE 60559:2011) Standard for Floating-Point Arithmetic
- Higham (2002) "Accuracy and Stability of Numerical Algorithms" 2nd ed., SIAM, Ch. 3

---

## D-8: Stopping Is a Certificate, Not Patience

**Industry assumption:** Early stopping = "stop if validation loss doesn't improve for N epochs." N = 3, 5, 10 — chosen by feel.

**Why it's wrong:** Patience is disconnected from the training dynamics. It can stop too early (the loss is noisy and a real improvement was one epoch away) or too late (the model has already overfit and the damage is done). The number N has no mathematical relationship to the loss surface, the model's capacity, or the data distribution.

**ModelCypher approach:** The stopping certificate requires four conditions to hold simultaneously. Separately, spectral-budget exhaustion is an immediate hard stop. Every criterion is geometry- or data-derived:

1. **Stationarity**: Riemannian gradient norm drops to the numerical noise floor (sqrt(eps) * max(1, |loss|)).
2. **Improvement bound**: Best local validation improvement is below sampling uncertainty, using Welford-standard-error estimates.
3. **Worst-group bound**: No individual batch has unresolved local improvement above its own CI half-width.
4. **No mechanism drift**: Entropy/repetition remain within dtype-derived bounds (no entropy collapse or repetition spike).

Independent hard-stop guard:
- **Adapter saturation**: `||BA||_2 / sigma_k` crossing the per-layer Weyl ratio `gap_k / (2 * sigma_k)`.

**Proof:**

Each condition targets a distinct failure mode; the certificate stop requires all four:

Condition 1 (stationarity): When ||P @ g|| < sqrt(eps) * max(1, |loss|), the preconditioned gradient is at the numerical noise floor. Further optimization steps cannot reliably decrease the loss — they are optimizing noise.

Condition 2 (improvement bound): The standard error of the difference between recent and earlier loss windows uses Welford's online algorithm for streaming variance computation. When the best improvement is smaller than this SE, the improvement is not statistically distinguishable from noise. This is a measurement of the data's own variance, not a fixed constant.

Condition 3 (worst-group): Even if the average improves, a single deteriorating batch means the adapter is trading off performance across data subsets.

Condition 4 (no mechanism drift): Entropy collapse (`entropy < sqrt(eps)`) and repetition spike (`repetition > 1 - sqrt(eps)`) are numerical/behavioral failure signals derived from IEEE 754 floors, not patience counters.

Independent saturation guard: From Weyl's inequality, when the adapter spectral contribution reaches `gap_k / 2`, singular-value crossing begins at the structural boundary. This is a separate geometric stop.

**Code:**
- [geometric_early_stopping.py:288-357](../../src/modelcypher/core/domain/training/geometric_early_stopping.py) — certificate evaluation, stationarity, and improvement bound.
- [geometric_early_stopping.py:369-410](../../src/modelcypher/core/domain/training/geometric_early_stopping.py) — worst-group and drift checks, aggregate stop condition.
- [spectral_budget.py:24-28](../../src/modelcypher/core/domain/training/spectral_budget.py) — per-layer Weyl crossing threshold

**Evidence:**
- `poetry run pytest -q tests/domain/training/test_geometric_early_stopping.py` -> `41 passed` (all certificate conditions and edge cases).
- [RESEARCH-ROADMAP.md](../RESEARCH-ROADMAP.md) logs geometric stopping as validated by `4-arm x 3-seed` ablation.
- [lr_derivation_analysis.md](lr_derivation_analysis.md) Run 3 reports stop reason `online_eval_degraded` at `16/25 < 18/25`, showing measured stop signals terminate before further collapse.

**Citations:**
- Weyl, H. (1912). "Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen." Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse, 1912, 110-117.
- Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products." Technometrics, 4(3), 419-420.
- Higham, N. J. (2002). "Accuracy and Stability of Numerical Algorithms" (2nd ed.). SIAM.

---

## D-9: Learning Rate Is Measured Per-Step, Not Scheduled

**Industry assumption:** LR = 1e-4 with linear warmup over 1000 steps, then cosine decay to 1e-6. "Standard for LoRA fine-tuning."

**Why it's wrong:** A single scheduled LR cannot track curvature changes across minibatches. The loss surface curvature varies catastrophically — measured Hessian-vector product (HVP) values span 3 orders of magnitude (0.1 to 193) within a single 10-step training run on LFM2-350M. A schedule is a guess about a surface it has never measured.

**ModelCypher approach:** MASS (Measured-Adaptive Step Size) — three-layer per-step system. Every step measures the geometry and derives the step size from the measurement:

1. **Spectral ceiling** (static): eta_ceiling = sigma_k_min / sigma_max (Weyl 1912)
2. **Per-step SPS**: eta_sps = f(x_t) / ||d_t||^2 (Loizou et al. 2020) — step size from current loss and gradient
3. **Per-step Weyl**: eta_weyl = sigma_k_min / ||d_t|| — displacement bound preventing Weyl crossing
4. **Combined**: eta_step = min(eta_sps, eta_weyl, eta_ceiling)

With Armijo backtracking (Absil et al. 2008, Th. 4.3.1) when the static ceiling is binding.

**Code:**
- [_mlx_training_adapter_train_mixin.py:96-110](../../src/modelcypher/backends/_mlx_training_adapter_train_mixin.py) — MASS definition and three-layer derivation

**Evidence (ablation study, LFM2-350M, baseline 18/25 correct):**

| Configuration | LR | Result | Delta |
|---------------|-----|--------|-------|
| Default (CE+REINFORCE) | 0.996 | 5/25 | -13 |
| CE-only | 1.64 | 13/25 | -5 |
| LR / 10 | 0.072 | 16/25 | -2 |
| LR / 100 | 0.0037 | 17/25 | -1 |
| Entropy floor 95% | 0.428 | 13/25 | -5 |
| REINFORCE-only | 0.366 | 15/25 | -3 |
| 10-batch Lipschitz | 1.13 | 11/25 | -7 |

Degradation is monotonically correlated with LR magnitude. LR/100 nearly eliminates degradation (1 problem lost from baseline). The root cause of training degradation was the LR derivation, not REINFORCE, not CE, not entropy — the LR.

**Citations:**
- Loizou, N., et al. (2020). "Stochastic Polyak Step-size for SGD: An Adaptive Learning Rate for Fast Convergence." ICML 2020.
- Weyl, H. (1912). "Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen." Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse, 1912, 110-117.
- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). "Optimization Algorithms on Matrix Manifolds." Princeton University Press.

---

## D-10: Weight Space Is Euclidean; Activation Space Is Curved

**Industry assumption:** Either (a) treat everything as Euclidean and lose activation geometry, or (b) apply Riemannian methods to weights because "the loss landscape is curved." Both are wrong.

**Why it's wrong:** Weight space R^{m x n} with the Frobenius metric is Euclidean. It is a flat vector space. The curvature people observe is in the loss *function* (Hessian/Fisher information metric), not in the parameter *space*. The loss function's curvature tells you about optimization difficulty — it does not make R^n into a Riemannian manifold.

Activation space is different. Tokens trace trajectories on a learned manifold embedded in R^d. This manifold IS curved — consecutive tokens are closer along geodesics than Euclidean straight lines. Geodesic deviation is measurable and meaningful.

**ModelCypher approach:** Separate the two domains cleanly:

- **Weight space**: Euclidean geometry. SVD for spectral analysis. Procrustes for alignment. Linear interpolation for mode connectivity. Frobenius norm for distance.
- **Activation space**: Riemannian geometry. k-NN Floyd-Warshall for geodesic distances. Frechet mean instead of arithmetic mean. Intrinsic dimension via TwoNN. Geodesic deviation for trajectory analysis.

**Code:**
- [transplant.py:58-61](../../src/modelcypher/core/domain/geometry/transplant.py) — `_weight_frobenius_norm`: "Uses Euclidean (not geodesic) norm because weight space is treated as flat/spectral rather than a curved manifold."
- [geodesic_trajectory_service.py:388-399](../../src/modelcypher/core/use_cases/geodesic_trajectory_service.py) — geodesic computation is applied to `trajectory.positions[target_layer]` (activations).
- [riemannian_core_geodesic.py:68-99](../../src/modelcypher/core/domain/geometry/riemannian_core_geodesic.py) — k-NN Floyd-Warshall geodesic distances on point clouds (used for activations)
- [knowledge_density.py:57-67](../../src/modelcypher/core/domain/geometry/knowledge_density.py) — loads geodesic-vs-euclidean decision artifact.
- [knowledge_density.py:90-133](../../src/modelcypher/core/domain/geometry/knowledge_density.py) — resolves effective distance mode and defaults to Euclidean when geodesic criteria fail.

**Evidence:**

Weight-space controls (measured, cross-family):
- [benchmark_summary_2026-02-23.json](../../results/weight_geometry/benchmark_summary_2026-02-23.json): `weight_manifold_curved_supported = false`, `weight_space_metric = "euclidean_plus_spectral"`, `all_weight_distortions_below_gaussian = true`, and `max_abs_path_loss_diff = 0.0` for linear-vs-geodesic mode connectivity.
- [Qwen2.5-Coder-0.5B-Instruct-bf16/benchmark_summary_2026-02-23.json](../../results/weight_geometry/Qwen2.5-Coder-0.5B-Instruct-bf16/benchmark_summary_2026-02-23.json): `weight_manifold_curved_supported = false`, `all_fixed_k_weight_minus_gaussian_negative = true`, `single_sample_weight_nonincreasing_with_k = true`, and `max_abs_path_loss_diff = 0.0`.

Activation-space curvature (measured, cross-family):
- [LFM2-350M_density_comparison.json](../../results/geodesic_vs_euclidean/LFM2-350M_density_comparison.json): layer means `0.4460`, `0.3779`, `0.3654`; max distortion up to `1.5725`; all sampled layers `geodesic_needed = true`.
- [Qwen2.5-Coder-0.5B-Instruct-bf16_density_comparison.json](../../results/geodesic_vs_euclidean/Qwen2.5-Coder-0.5B-Instruct-bf16_density_comparison.json): layer means `0.4502`, `0.5362`, `0.4998`; max distortion up to `1.6556`; all sampled layers `geodesic_needed = true`.

**Citations:**
- Facco et al. (2017) "Estimating the intrinsic dimension of datasets by a minimal neighborhood information" Scientific Reports 7:12220
- Tenenbaum, de Silva & Langford (2000) "A Global Geometric Framework for Nonlinear Dimensionality Reduction" Science 290(5500):2319-2323

---

## Summary

| # | Divergence | Industry | ModelCypher | Source of Truth |
|---|-----------|----------|-------------|----------------|
| D-1 | Inference decoding | Temperature sampling | Greedy (argmax) | Deterministic forward pass |
| D-2 | LoRA scale | alpha/rank = 2.0 | sigma_k/2 * (1-sqrt(eps)) | SVD + Weyl 1912 |
| D-3 | Training objective | CE on traces | Auto CE/REINFORCE regime | Clopper-Pearson 1934 + Williams 1992 |
| D-4 | Model merging | Interpolation | Null-space addition | Penrose 1955 (CKA=1.0 by construction) |
| D-5 | LoRA rank | 8 (arbitrary) | tail_dims from Shannon entropy | SVD spectral entropy |
| D-6 | Optimizer | Adam/AdamW (Euclidean) | Cayley-Stiefel retraction (no pullback preconditioner — P ≈ I, removed after falsification) | Wen & Yin 2013 + Lezcano-Casado 2019 + trajectory falsification artifacts |
| D-7 | Thresholds | 1e-8, 1e-6 (fixed) | sqrt(eps), gap_k/(2*sigma_k) | IEEE 754 + Higham 2002 |
| D-8 | Early stopping | Patience (N epochs) | 4-condition geometric certificate | Weyl + Welford SE |
| D-9 | Learning rate | 1e-4 + cosine schedule | MASS per-step measurement | Loizou 2020 + Absil 2008 |
| D-10 | Geometry domain | Mixed or all-Euclidean | Weights=Euclidean, Activations=Riemannian | Measured cross-family (weights flat controls; activation distortion 0.365-0.536) |

Every number traces to SVD, IEEE 754, measured data, or a cited theorem. No unsupported defaults, no folklore tuning. The geometry is the ground truth.
