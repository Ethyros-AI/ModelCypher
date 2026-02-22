# Aristotle Prompts for ModelCypher Research

Curated prompts for [Aristotle](https://aristotle.science/) by Autopoiesis Sciences. Each prompt is designed for a specific Aristotle model and includes enough context for Aristotle's pre-query system to classify the research domain correctly.

**How to use**: Copy-paste the prompt text into the Aristotle chat interface. Aristotle will ask clarifying pre-query questions — answer them to narrow the research context. Follow up on whatever threads look promising.

**When Aristotle asks your research context**, tell it:
> I'm working on geometric diagnostics for LLM internals. My research treats neural network activations as points on a high-dimensional manifold and applies differential geometry, spectral analysis, and Procrustes alignment to produce quantitative measurements. I'm currently focused on LoRA adapter training with Riemannian optimization on the Stiefel manifold and on understanding why cross-entropy loss on reasoning traces teaches format rather than reasoning ability.

---

## X1 Verify — Pressure-Test Our Claims

Use X1 Verify when you need a skeptical second opinion on a mathematical claim or empirical finding. It provides confidence scores and explores alternative explanations.

---

### 1. Cayley-Riemannian Natural Gradient — Is Our Derivation Correct?

```
I'm training LoRA adapters using Cayley-parameterized updates on the Stiefel manifold with
a Riemannian natural gradient. I want you to check whether my derivation is mathematically
sound or whether I've made errors.

Setup:
- LoRA factors A_tilde [r, in_features] and B_tilde [r, out_features] are free parameters
- The Cayley retraction maps them to semi-orthogonal matrices: (I - Z)(I + Z)^{-1}
  where Z is constructed from A_tilde and B_tilde
- The natural gradient preconditioner is P = M @ M^T where M = I + Z
  (full, unnormalized pullback metric inverse)
- The update is: d_t = P_t @ g_t (preconditioned gradient)
- Historical step-size path (superseded by MASS): eta_t <= 2 / (L_t * lambda_max(P_t))
  where L_t is the Lipschitz constant of the loss gradient
- Historical stability invariant: m = eta * L * lambda_max(P) <= 2

Citations I'm building on:
- Amari (1998) for natural gradient
- Nesterov (2004) for L-smoothness step size bounds
- Lezcano-Casado (2019) for Cayley retraction on Stiefel manifolds
- Wang et al. (2025), NB-LoRA (arXiv:2501.19050) for norm-bounded LoRA via Cayley

My specific questions:
1. Is the preconditioner P = M @ M^T the correct pullback metric inverse for the Cayley
   parameterization, or should it be something else?
2. For the historical Lipschitz path, is the step size bound eta <= 2/(L * lambda_max(P)) tight, or is there a tighter bound
   for Riemannian optimization specifically on the Stiefel manifold?
3. Are there known failure modes of Cayley retraction that I should watch for?
   (I've already encountered: trace-normalization kills anisotropy, lambda_max-normalization
   does the same, M @ M^T without the step bound causes NaN)

Validated result: On a 350M-parameter hybrid attention-convolution model, this approach
gives validation loss 1.27 vs 1.38 for plain SGD on the same data.
```

---

### 2. Weyl Perturbation Budget — Is Our Bound Tight?

```
I'm using Weyl's inequality to derive a spectral perturbation budget for LoRA adapters.
I want you to verify whether the bound is correct and whether it can be tightened.

My framework:
- Base weight matrix W with SVD: W = U @ Sigma @ V^T
- LoRA perturbation: W' = W + scale * (B @ A)
- Weyl's inequality gives: |sigma_i(W') - sigma_i(W)| <= ||scale * B @ A||_2
- I define "safe" as: scale * ||B @ A||_2 <= sigma_k(W)
  where sigma_k is the structural boundary singular value (Shannon effective-rank anchor)
- This gives the bound: scale <= sigma_k(W) / ||B @ A||_spectral

Eigengap refinement (Theorem 2):
- When a spectral gap exists at position k, I tighten to:
  scale_bound = min(sigma_k / ||Delta||_2, gap_k / (2 * ||Delta||_2))
- This is the Weyl no-crossing condition at boundary k (||E||_2 < gap_k/2)

For training, I use NB-LoRA Cayley parameterization which guarantees ||B @ A||_spectral <= 1
by construction, so I monitor the ratio ||BA||_2 / sigma_k(W) and stop when it approaches 1.0.

Empirical finding: Standard LoRA scale of 2.0 (alpha=16, rank=8) violates this bound by
factors of 600-2700x across 9 adapters tested on a 350M model.

Questions:
1. Is Weyl's inequality the tightest tool here, or do Stewart's sharper SVD perturbation
   bounds give meaningfully different results for low-rank perturbations?
2. My sqrt(eps) threshold for "significant" singular values — is this standard in numerical
   linear algebra, or is there a better-justified cutoff?
3. The Weyl no-crossing eigengap refinement — am I applying it correctly to singular values?
4. Are there results specifically about low-rank perturbations that give tighter bounds
   than the general Weyl/Wedin framework?
```

---

### 3. SFT Format Memorization — Is Our Diagnosis Correct?

```
We trained LoRA adapters on reasoning traces (chain-of-thought data with <think>...</think>
tags) using standard cross-entropy loss. The training metrics looked perfect but inference
got worse. I want you to pressure-test our diagnosis.

Evidence:
- 350M model: perplexity 19.6 -> 3.9, but inference accuracy dropped 9/20 -> 4/20,
  with 25% of outputs degenerate (repetitive loops)
- 1.2B model: perplexity 8.6 -> 1.4, but inference accuracy dropped 30/46 -> 20/46,
  with 28% degenerate outputs
- Both models: CKA alignment between base and adapted model remained high
- Both models: Weyl spectral budget looked healthy
- Both models: Training loss converged smoothly

Our diagnosis: Cross-entropy loss on reasoning traces teaches format (how to produce
<think> tags and chain-of-thought structure) rather than reasoning (how to arrive at
correct answers). The model learns to mimic the surface pattern of reasoning without
learning the underlying computation. PPL, CKA, and spectral budget are all wrong proxies
for reasoning ability — they measure format fidelity, not computational capability.

We tried several interventions, all failed:
- Constrained training with paired correct/incorrect examples (constraints hurt)
- Cross-projection rank coupling (improved knowledge but amplified repetition)
- Answer-span masking (zero loss on reasoning tokens, only train on answer tokens):
  eliminated degenerate output but didn't improve accuracy

Our current hypothesis: CE is fundamentally the wrong objective for teaching reasoning
to small models. The optimizer is correct; the objective is the problem. We've implemented
a REINFORCE outcome-based objective (Williams 1992) where A_i = r_i - mean(r_group)
but haven't validated it yet.

Questions:
1. Is our diagnosis — that CE on reasoning traces is format memorization — consistent
   with the literature? Are there published results showing the same failure mode?
2. Is REINFORCE with outcome-based rewards a reasonable alternative, or are there known
   problems with applying it to small models (350M-1.2B parameters)?
3. Are there other training objectives we should consider that are specifically designed
   to teach reasoning rather than format?
4. The fact that PPL improves while accuracy drops — has this been documented elsewhere?
   What's the standard term for this phenomenon?
```

---

### 4. Geometric Stopping Certificate — Are 4 Conditions Sufficient?

```
I've designed a geometric stopping certificate for LoRA training that uses 4 conditions
instead of loss-based early stopping. I want you to check whether these conditions are
sufficient for a sound stopping criterion.

The 4 conditions:
1. Stationarity: ||P @ g|| at noise floor (preconditioned gradient norm is small)
   - P is the Riemannian natural gradient preconditioner (pullback metric inverse)
   - "Noise floor" defined by the stochastic gradient variance, not sqrt(eps)
   - Checked via moving window stability of gradient norms

2. Improvement bound: Delta_max < confidence interval
   - Maximum possible remaining improvement is bounded
   - Estimated from the learning rate, gradient norm, and curvature

3. Worst-group: per-batch gradient norms are uniformly small
   - No single batch has anomalously large gradients
   - Prevents stopping when some data subgroups are still learning

4. No mechanism drift: entropy and repetition metrics are stable
   - Logit entropy hasn't collapsed (no mode collapse)
   - Output repetition rate hasn't increased (no degenerate loops)

Context: Standard early stopping monitors validation loss with patience. This fails for
us because validation loss (PPL) is a wrong proxy — it can improve while actual reasoning
accuracy degrades (we've confirmed this empirically).

Questions:
1. Are these 4 conditions jointly sufficient for a mathematically sound stopping criterion
   on Riemannian manifolds (Stiefel manifold specifically)?
2. Are there standard results on convergence certificates for Riemannian SGD that I should
   be incorporating?
3. The worst-group condition — is this related to distributionally robust optimization (DRO)?
   Are there stronger formulations I should use?
4. Is there literature on stopping criteria that explicitly decouple format learning from
   capability learning?
```

---

### 5. Intrinsic Dimension Compression — Is This a Real Phenomenon?

```
I've measured intrinsic dimension (ID) profiles across layers of three transformer LLMs
using TwoNN (Facco et al. 2017) and found a consistent pattern. I want you to check
whether this is a known phenomenon, an artifact, or a genuine observation.

Observation: All three models show:
1. High ID in early layers (~15.8 on average across semantic probe categories)
2. Sharp early-layer drop ("dimensionality cliff")
3. Low-ID plateau in middle layers (~1.8)
4. Recovery in output layers (~9.6)

Models tested: Qwen2.5-0.5B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3
Measurement method: TwoNN with geodesic distances (k-NN graph, k=10, Dijkstra paths)
Probe corpus: 12 semantic domains (mathematical, logical, spatial, temporal, moral, etc.)

Working hypothesis: Early transformer layers rapidly project tokenized representations
onto a low-dimensional conceptual manifold. Middle layers process within that manifold.
Output layers expand back for vocabulary prediction.

The pattern is domain-dependent: domains with higher initial ID show stronger compression
(measured on Qwen only).

Questions:
1. Is this "dimensionality cliff then plateau then recovery" pattern documented in the
   literature? What's the standard term for it?
2. The Platonic Representation Hypothesis (Huh et al. 2024) predicts convergence of
   representations across architectures. Does my ID profile observation support or
   contradict it?
3. Are there known artifacts in TwoNN estimation that could produce this pattern
   spuriously? (e.g., from small sample sizes, from geodesic distance estimation,
   from the specific k=10 choice)
4. Has anyone measured ID profiles on non-transformer architectures (state-space models,
   convolution-based models) to see if the pattern persists?
```

---

## X1 Search — Find What We're Missing

Use X1 Search for literature exploration. It combines LLM reasoning with a scientific knowledge graph and explicitly flags contradictions.

---

### 1. Riemannian Optimization on Stiefel Manifolds for LoRA

```
Search for recent work (2024-2026) on Riemannian optimization methods applied to
LoRA (Low-Rank Adaptation) training for large language models.

I'm specifically interested in:
- Methods that parameterize LoRA factors on the Stiefel manifold (orthogonal/semi-orthogonal)
- Cayley retraction vs geodesic retraction vs QR retraction for training stability
- Natural gradient methods adapted for manifold-constrained optimization
- NB-LoRA (arXiv:2501.19050) and any follow-up work or critiques
- RoRA (arXiv:2601.06305) on spectral strength as a root cause of LoRA failures
- Connections between Riemannian LoRA training and spectral normalization

Key paper I'm building on: Wang et al. (2025), "NB-LoRA: Norm-Bounded Low-Rank
Adaptation" which uses Cayley parameterization for spectral norm bounds.

I want to know: Who else is doing this, what approaches are they using, and what are
the known limitations or failure modes of manifold-constrained LoRA training?
```

---

### 2. Outcome-Based Training for Small Language Models

```
Search for work on training small language models (350M to 3B parameters) using
outcome-based or reinforcement learning objectives rather than cross-entropy.

Context: I've confirmed empirically that supervised fine-tuning with cross-entropy
loss on reasoning traces teaches format rather than reasoning — perplexity drops
while inference accuracy degrades. I've implemented a REINFORCE objective with
advantage baseline A_i = r_i - mean(r_group) (Williams 1992) but haven't validated it.

I'm specifically interested in:
- GRPO (Group Relative Policy Optimization) and variants
- STaR (Self-Taught Reasoner) and whether it works for models with zero baseline skill
- Outcome-based rewards vs process-based rewards for reasoning tasks
- Whether REINFORCE / policy gradient methods are sample-efficient enough for small models
- Any work on the minimum model size needed for RL-based training to work
- Alternatives: direct preference optimization (DPO), rejection sampling fine-tuning

Key concern: Small models may not have enough representation capacity for exploration
to find correct solutions. If the model can never generate a correct answer, REINFORCE
has zero signal. What's the literature on bootstrapping capabilities that the model
has zero signal for?
```

---

### 3. CKA Limitations and Alternatives

```
Search for recent critical analysis of Centered Kernel Alignment (CKA) as a measure
of representational similarity, specifically in the context of neural network diagnostics
and training monitoring.

I currently use CKA (Kornblith et al. 2019) for:
- Measuring alignment between base and adapted models during LoRA training
- Verifying merge quality after model merging
- Cross-architecture comparison of internal representations

I've found a failure mode: CKA can show high alignment (near 1.0) between base and
adapted models even when the adapted model has substantially degraded reasoning ability.
CKA appears to measure geometric similarity of the representation manifold without
capturing whether the model has learned the right computation.

I want to know:
- Published critiques of CKA (false positives, failure modes, sensitivity to data)
- Alternative representational similarity measures (SVCCA, projection-weighted CKA,
  Riemannian metrics, others)
- Whether anyone has proposed similarity measures that correlate with downstream task
  performance rather than geometric alignment
- Work on distinguishing "format similarity" from "capability similarity" in
  representation space
```

---

### 4. Spectral Perturbation Bounds for Low-Rank Adapters

```
Search for spectral perturbation theory applied specifically to low-rank matrix
perturbations, with applications to LoRA adapters or neural network weight modifications.

My current approach uses general Weyl inequality and gap-based no-crossing bounds to
the effect of LoRA perturbations on base weight singular values. But these are
worst-case bounds for arbitrary perturbations. Since LoRA perturbations are specifically
rank-r (typically r=8 to 128), there may be tighter results.

I'm looking for:
- Perturbation bounds specialized to rank-r updates (not full-rank perturbations)
- Connections between LoRA rank, base weight numerical rank, and perturbation stability
- Multiplicative perturbation bounds vs additive (Weyl is additive)
- Results on how low-rank perturbations affect singular subspaces (not just singular values)
- Tran et al. (2025), arXiv:2510.25670 on spectral perturbation under eigengap conditions
  — are there newer results?
- Any work connecting matrix perturbation theory to catastrophic forgetting in fine-tuning

My scale bound formula: scale <= sigma_k(W) / ||B @ A||_spectral
where sigma_k is the structural boundary singular value (Shannon effective-rank anchor).
Are there tighter formulations?
```

---

### 5. Intrinsic Dimensionality of Neural Network Representations

```
Search for the current state of research on intrinsic dimensionality of neural network
intermediate representations, measured across layers.

I've observed a "dimensionality cliff then plateau then recovery" pattern in 3 transformer
LLMs using TwoNN estimation: high ID in early layers (~15.8), sharp drop, low plateau
(~1.8), recovery in output layers (~9.6). I want to know the state of the field.

Specifically:
- Has the layer-wise ID profile pattern been documented across many architectures and scales?
- What methods beyond TwoNN are used? (MLE, correlation dimension, PCA-based, etc.)
  How sensitive are the results to estimation method?
- The Platonic Representation Hypothesis (Huh et al. 2024) — how does it relate to
  ID profiles? Does convergence of representations imply convergence of ID profiles?
- Ansuini et al. (2019), "Intrinsic dimension of data representations in deep neural
  networks" — this is the foundational paper. What has happened since?
- ID profiles for non-transformer architectures (Mamba, RWKV, state-space models,
  hybrid attention-convolution architectures like LFM2)
- Connection between ID compression and information bottleneck theory
- Practical uses of ID profiles: can they predict training failure, guide architecture
  design, or diagnose fine-tuning problems?
```

---

## X1 Spark — Generate New Research Directions

Use X1 Spark for hypothesis generation. It generates bold ideas first, then grounds them retroactively. Let it imagine freely.

---

### 1. Beyond CE: What Training Objective Teaches Reasoning?

```
I need bold hypotheses for training objectives that teach reasoning to small language
models (350M-1.2B parameters), given that cross-entropy on reasoning traces provably
fails (teaches format, not reasoning).

What I've ruled out:
- CE on chain-of-thought traces: format memorization (confirmed on 350M and 1.2B)
- CE with answer-span masking: eliminates degenerate output but doesn't improve accuracy
- CE with constrained training (paired correct/incorrect): constraints hurt
- Cross-projection rank coupling: improves knowledge, amplifies repetition

What I'm currently trying:
- REINFORCE with outcome-based rewards (Williams 1992): A_i = r_i - mean(r_group),
  using NB-LoRA spectral budget instead of KL regularization

Constraints:
- Small models (350M-1.2B params) — can't rely on massive exploration capacity
- Running on Apple Silicon (MLX) — single-GPU training, not cluster
- All thresholds must be derivable (from SVD, IEEE 754, or cited theorem)
- No heuristics or "works well in practice" — derive or cite everything

The fundamental question: How do you teach a computational skill to a model that has
zero baseline capability in that skill? REINFORCE requires at least some correct
completions to generate signal. If the model never solves the problem, there's no gradient.

Generate hypotheses. The crazier the better. I'll evaluate feasibility.
```

---

### 2. Geometric Training Signals Beyond Loss

```
I'm looking for geometric properties of the training trajectory that could serve as
better training signals than loss values, specifically for LoRA adapter training on
the Stiefel manifold.

Current problem: Loss (CE/PPL) can improve while actual model capability degrades.
We've confirmed this empirically — format memorization produces beautiful loss curves
and terrible outputs.

What I already monitor:
- Weyl spectral budget: ||BA||_2 / sigma_k(W), stops near 1.0
- Preconditioned gradient norm: ||P @ g|| (natural gradient on Stiefel manifold)
- Logit entropy: prevents collapse to deterministic output
- Expansion ratio: intrinsic dimension ratio (input ID / output ID) per layer
- CKA between base and adapted model (but this is a wrong proxy for capability)

What I want: A geometric signal computable during training that predicts whether the
model is learning genuine computational capability vs memorizing surface format.

Key insight: A forward pass is a deterministic geometric map. The model's weights define
a fixed high-dimensional landscape. There should be measurable geometric properties of
the training trajectory that distinguish "learning computation" from "memorizing pattern."

Generate hypotheses for what those properties might be. Think about:
- Curvature of the loss landscape along the training trajectory
- Singular value dynamics of the adapted weight matrices
- Information-geometric quantities (Fisher information, etc.)
- Topological properties of the representation manifold during training
- Any cross-disciplinary connections (physics, dynamical systems, etc.)
```

---

### 3. Geometric Curriculum for Small Model Training

```
I want hypotheses for what a "geometric curriculum" would look like for training small
language models (350M-1.2B parameters) on reasoning tasks.

Background:
- Neural nets don't learn from scratch. Pure self-play (STaR) reinforces existing
  capabilities but cannot bootstrap skills the model has zero signal for.
- New capabilities require teaching (seeded examples/rationalization), then exploration
  extends them.
- My framework: "CE for training. Generation for exploration. Teaching for new
  capabilities. Geometry for navigation."

The concept: Instead of ordering training examples by difficulty (standard curriculum
learning), order them by geometric properties of the representation space. For example:
- Start with problems where the model's intrinsic dimension profile is "healthy"
  (matches the base model's pattern)
- Gradually introduce problems that require expanding the representation manifold
- Monitor the Weyl spectral budget to ensure perturbations stay bounded
- Use the geometric stopping certificate to know when each curriculum stage is complete

What I want: Hypotheses for how to design a curriculum based on measurable geometric
properties rather than task difficulty heuristics.

Think about:
- What geometric property of a training example predicts how "teachable" it is?
- Is there a natural ordering of the representation manifold that training should follow?
- How does intrinsic dimension change predict learning readiness?
- Can we use spectral properties of the weight matrices to identify "capacity available"
  for new knowledge?
- Connections to manifold learning, geodesic paths, optimal transport
```

---

## Usage Notes

- **X1 Verify** will ask you to narrow your research context. Tell it you're in ML/deep learning, specifically geometric analysis of neural network representations and Riemannian optimization for LoRA training.
- **X1 Search** may surface papers you haven't seen. Follow up on anything that contradicts your assumptions — that's where the value is.
- **X1 Spark** generates bold ideas. Not all will be feasible. The goal is to find 1-2 directions worth investigating further.
- **Aristotle Instant** is good for quick fact-checks: "What is the standard convergence rate for Riemannian SGD on the Stiefel manifold?" or "What is the sample complexity of REINFORCE with a constant baseline?"
