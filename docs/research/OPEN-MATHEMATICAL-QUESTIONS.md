# Open Mathematical Questions

**Goal:** Derive, don't just observe. Every pattern should have a mathematical explanation.

---

## 1. Layer Jacobian Structure — CORRECTED (2026-02-03)

**Previous claim (WRONG):** Jacobians are rank-1 in trained transformers.

**Corrected finding:** True layer Jacobians are **full-rank, near-identity**.

When measured correctly (float32, ε=1e-3 to 1e-4):
```
ε=1e-03: eff_rank=63.9, σ_max=1.08, σ_2=1.02
ε=1e-04: eff_rank=63.9, σ_max=1.10, σ_2=1.05
```

**What this means:**
- Each layer is approximately identity: y ≈ x + small_delta
- Residual connections dominate: the "semantic highway" is real
- All singular values ≈ 1.0 means equal contribution from all directions
- This is NOT rank-1, it's full-rank near-identity

**Cause of original error:**
- bf16 model precision (3-4 significant digits)
- Tiny epsilon (1e-5 to 1e-6) for finite differences
- Combined to make perturbations invisible → artificial rank collapse

**New question:** What determines the magnitude of the "small_delta"?
- Is it constant across layers?
- Does it vary by layer type (attention vs MLP)?
- Does it correlate with highway position?

**Experiments:**
- [x] Verify with float32 precision ✓
- [x] Test multiple epsilon values ✓
- [ ] Measure delta magnitude across layers
- [ ] Compare attention delta vs MLP delta

---

## 2. What Determines Highway Location? — MAJOR PROGRESS (2026-02-03)

**Observation:**
- LFM2: Entry compression (layers 0-1) at 0-6% of depth
- Qwen/DeepSeek: Mid compression (layers 17-28) at 44-47% of depth
- Granite: Early compression (layers 5-28) at 16% of depth

**FINDINGS:**

### Factor 1: Hybrid Architecture (LFM2)

LFM2's entry highway is caused by **Mamba/SSM layers**, not transformer attention:
- Layers 0, 1, 3, 4, 6, 7, 9, 11, 13, 15 = Mamba (10 total)
- Layers 2, 5, 8, 10, 12, 14 = Attention (6 total)
- The highway (layers 0-1) consists of PURE MAMBA layers

**Why Mamba creates low ID:**
- SSM is a linear recurrence: h_t = A·h_{t-1} + B·x_t
- State h_t lives in a fixed-dimensional space
- This naturally creates low-dimensional compressed representations
- Then attention layers (starting layer 2) expand for processing

### Factor 2: GQA Ratio (Pure Transformers)

For pure transformer architectures, **GQA ratio predicts highway position:**

| Model | GQA Ratio | Highway Start |
|-------|-----------|---------------|
| Granite-3B | 1.0 | 16% |
| Qwen3-8B | 4.0 | 44% |
| Qwen2.5-3B | 8.0 | 47% |

**Fitted model (R² = 0.941):**
```
highway_start% = 17.6 + 15.7 × log(GQA)
```

**Theoretical derivation:**

With GQA, K/V weights are shared across query heads:
```
∂L/∂W_k^g = Σ_{h in group} ∂L/∂K_h @ x^T
```

This averaging forces K/V to learn **consensus representations** that satisfy all query heads in the group. Consensus requires:
1. Diverse query representations to be developed first
2. Then compression into shared K/V space

Higher GQA = more heads sharing = stronger consensus requirement = later compression.

### Summary

| Architecture | Highway Position | Cause |
|-------------|------------------|-------|
| Hybrid (LFM2) | Entry (0-6%) | SSM layers create low-dimensional state |
| GQA=1 | Early (16%) | No K/V sharing constraint |
| GQA=4-8 | Mid (44-47%) | K/V consensus requires query diversity first |

**Remaining questions:**
- [ ] Verify prediction on new architectures (Llama with different GQA)
- [ ] Test if FFN expansion ratio has secondary effect
- [ ] Derive GQA-highway relationship from gradient flow analysis

---

## 3. Why Does RLHF Flatten Geometry?

**Observation:** Specialist models (instruct, code, reasoning) have expansion_ratio variance ≈ 0.

**Hypothesis:** RL training creates stable attractors regardless of input type.

**What we need to derive:**
- What is the loss landscape geometry under RLHF?
- Does PPO/DPO explicitly or implicitly penalize geometric variance?
- Is this a side effect of reward hacking or intentional?

**Experiments to constrain:**
- [ ] Compare base vs instruct checkpoints of same model
- [ ] Measure geometry during RLHF training (if we have checkpoints)
- [ ] Test if flat geometry is necessary or sufficient for instruction following

---

## 4. Recovery Ratio vs Model Size

**Observation:**
| Size | Recovery Ratio |
|------|----------------|
| 350M | 14.04× |
| 1.2B | 4.83× |
| 3B | 3.76-5.78× |
| 8B | 2.64-5.06× |

**Unknown:** What's the functional form? Is it:
- Power law: R ∝ N^α
- Logarithmic: R ∝ log(N)
- Something else?

**What we need:**
- More data points (need 7B, 13B, 70B models)
- Fit functional form
- Derive from first principles why this relationship exists

---

## 5. Attention Eigenvalue Distribution — INITIAL RESULTS

**Finding (2026-02-03):** Attention matrices have dramatically lower effective rank than random.

| Model | Attn Eff. Rank | Entropy | Random | Rank Reduction |
|-------|----------------|---------|--------|----------------|
| LFM2-350M | **1.02** | 2.40 | 6.95 | **85%** |
| Qwen2.5-3B-Instruct | 3.85 | 1.48 | 6.95 | 45% |
| Qwen3-8B | 2.76 | 1.55 | 6.95 | **60%** |
| DeepSeek-R1-8B | 2.74 | 1.58 | 6.95 | **61%** |

**Key observations:**

1. **LFM2 attention is essentially rank-1** (eff_rank ≈ 1.02, spectral gap ~10^7)
   - This directly explains rank-1 Jacobians in LFM2
   - But entropy is HIGH (2.40 > random 1.93) - all rows attend similarly but broadly

2. **Qwen3 base is sharper than Qwen2.5-Instruct** (2.76 vs 3.85)
   - Counter-intuitive: specialist training doesn't always sharpen attention
   - The Qwen3 architecture itself may be responsible

3. **DeepSeek-R1 ≈ Qwen3 base** (2.74 vs 2.76)
   - Reasoning training didn't change attention rank
   - The geometry changes must be elsewhere (MLP? layer interactions?)

4. **Exit layers have near-zero entropy** (Qwen3/DeepSeek layers 29-35: entropy 0.04-0.26)
   - Final layers concentrate attention on 1-2 tokens
   - This is where the prediction is formed
   - Different heads focus on different tokens → rank > 1

**SOLVED PUZZLES:**

1. **LFM2: rank-1 + high entropy — EXPLAINED**

   LFM2 attention is **perfectly uniform**:
   ```
   [0.091 0.091 0.091 0.091 0.091 ...]
   [0.091 0.091 0.091 0.091 0.091 ...]
   [0.091 0.091 0.091 0.091 0.091 ...]
   ```
   - Row similarity = 1.0 (all rows identical) → rank-1
   - Uniform distribution over tokens → maximum entropy
   - **LFM2 attention has degenerated to mean-pooling**

   This explains rank-1 Jacobians: the attention contributes a rank-1 term because
   every position receives the same weighted sum of all positions.

   **Why does this happen?** LFM2 is a hybrid Mamba/attention architecture.
   The Mamba (SSM) layers may handle sequence modeling, leaving attention
   to simply aggregate information uniformly. Needs verification.

2. **CORRECTION: Rank-1 Jacobian was a numerical artifact!** (2026-02-03)

   Previous finding (WRONG): Jacobians are rank-1 in trained transformers.

   **Actual finding:** When measured correctly (float32, ε=1e-3 to 1e-4):
   ```
   ε=1e-03: eff_rank=63.9, σ_max=1.08, σ_2=1.02
   ε=1e-04: eff_rank=63.9, σ_max=1.10, σ_2=1.05
   ```

   The true layer Jacobian is:
   - **Full rank** (~64 effective rank for 64-probe measurement)
   - **Near-identity** (all singular values ≈ 1.0)
   - Each layer makes small incremental changes

   The rank-1 result was caused by:
   - bf16 model precision (3-4 significant digits)
   - Tiny finite difference epsilon (1e-5 to 1e-6)
   - These combined to make small perturbations invisible

   **Correct interpretation:** Transformer layers are approximately identity
   transformations with small incremental modifications. The "semantic highway"
   is real (residual connections dominate), but it's not rank-1 - it's full-rank
   near-identity.

3. **Why does LFM2 learn uniform attention? — EXPLAINED** (2026-02-03)

   **Answer:** Q and K projection matrices converge to **nearly orthogonal subspaces**.

   **Evidence:**
   | Layer | ||Q@K^T|| | ||Q@K^T|| (Qwen) |
   |-------|----------|------------------|
   | LFM2 | 1.0-2.0 | - |
   | Qwen | - | **14.75** (7.4×) |

   When W_q^T @ W_k ≈ 0, attention scores are near-zero regardless of input.
   Softmax of near-zero uniform scores → uniform attention.

   **Why does training converge to this?**
   1. LFM2 is a hybrid: 10 Mamba layers + 6 Attention layers
   2. Mamba handles sequence modeling (token dependencies)
   3. Attention layers receive no gradient signal to be selective
   4. Q/K projections drift toward orthogonality (stable attractor)
   5. Result: attention = mean-pooling

   **This is emergent specialization, not a bug.** The model learned that
   global averaging is sufficient when Mamba handles sequence modeling.

**REMAINING PUZZLES:**

1. **Why does Qwen3 architecture have sharper attention than Qwen2.5?**
   - Qwen3: effective rank 2.76
   - Qwen2.5: effective rank 3.85
   - Same company, different architecture choices
   - What changed between versions?

2. **GQA and attention sharpness relationship**
   - Qwen2.5 (GQA=8) has higher attention rank than Qwen3 (GQA=4)
   - Counter-intuitive: more K/V sharing doesn't mean sharper attention
   - Need: analytical relationship between GQA and attention spectrum

**Remaining experiments:**
- [x] Test more architectures (Qwen3, DeepSeek)
- [ ] Test at different training checkpoints
- [ ] Analytically derive Jacobian rank from attention rank
- [ ] Test pre-trained vs random initialization
- [ ] Layer-wise entropy trajectory analysis

---

## 6. Manifold Topology

**Unknown:** What is the topology of the activation manifold?

**Questions:**
- Is it simply connected or are there holes?
- What is its Betti number?
- Does topology change across layers?
- Are there distinct connected components for different semantic categories?

**Tools needed:**
- Persistent homology (we have ripser)
- Sufficient samples to estimate topology
- Multiple models to check universality

**Experiments:**
- [ ] Compute persistent homology of activations at each layer
- [ ] Track Betti numbers across layers
- [ ] Compare topology across architectures
- [ ] Test if semantic categories occupy topologically distinct regions

---

## 7. Layer-wise Invariants

**Unknown:** What properties are preserved vs transformed across layers?

**Candidates:**
- Norm (preserved? scaled?)
- Angles between vectors (preserved?)
- Rank of activation matrix (preserved?)
- Intrinsic dimension (varies - but by how much?)

**Experiments:**
- [ ] Track multiple geometric properties layer-by-layer
- [ ] Identify which are approximately preserved
- [ ] Derive why certain properties must be preserved given architecture

---

## 8. Training Dynamics → Geometry

**Unknown:** How do training hyperparameters affect final geometry?

**Key hyperparameters:**
- Learning rate
- Batch size
- Weight decay
- Warmup schedule
- Dropout

**What we'd need:**
- Train same architecture with varied hyperparameters
- Measure final geometry
- Find functional relationships

**This is expensive but would give us predictive power.**

---

## 9. Information-Theoretic Characterization

**Unknown:** What is the mutual information between layers?

**Questions:**
- I(layer_i; layer_j) as function of |i-j|
- Does MI decay exponentially?
- Is there a "information bottleneck" at the highway?

**Connection to geometry:**
- Low ID at highway → compressed representation
- Does low ID = low MI with input?

---

## 10. The Fundamental Question — FIRST FORMULA (2026-02-03)

**Can we write down an equation that predicts geometry from architecture?**

**Answer: Partial yes!**

### What We Can Now Predict

**Highway position (pure transformers):**
```
highway_start% = 17.6 + 15.7 × log(GQA_ratio)
```
- R² = 0.941 on 3 data points
- Needs validation on more architectures

**Highway type (architecture families):**
| Architecture | Highway Type |
|-------------|--------------|
| Hybrid (SSM + attention) | Entry (0-10%) |
| Transformer (GQA=1) | Early (15-20%) |
| Transformer (GQA>1) | Mid (~45%) |

### What We Still Can't Predict

- **Recovery ratio**: Have data, no formula yet
- **Expansion ratio variance**: Know RLHF flattens it, don't know why
- **Attention rank**: Know architectures differ, don't know what determines it

### Minimal Architectural Parameters

Based on our findings, these parameters most affect geometry:
1. **Architecture type**: Hybrid vs pure transformer
2. **GQA ratio**: n_heads / n_kv_heads
3. **Model depth**: n_layers (affects where "mid" is)

These parameters seem less important:
- FFN expansion ratio (4.0 vs 5.38 - similar highway positions)
- Hidden dimension size
- Head dimension

### Next Steps

1. Validate highway formula on more architectures
2. Derive recovery ratio formula
3. Understand RLHF geometric flattening mechanism

---

## Priority Ranking

| Question | Tractability | Impact | Priority | Status |
|----------|--------------|--------|----------|--------|
| Highway location | ~~Medium~~ High | High | **1** | **MAJOR PROGRESS** - GQA formula found |
| Attention eigenvalues | High | High | **2** | PARTIAL - LFM2 explained |
| Jacobian structure | ~~Medium~~ High | High | **3** | CORRECTED - not rank-1, is near-identity |
| Manifold topology | Medium | Medium | 4 | NOT STARTED |
| RLHF flattening | Low | Medium | 5 | NOT STARTED |
| Recovery ratio function | High | Low | 6 | DATA COLLECTED |
| Layer invariants | High | Medium | 7 | NOT STARTED |
| Training dynamics | Low | High | 8 | BLOCKED (need training runs) |
| Information theory | Medium | Medium | 9 | NOT STARTED |

---

## Next Steps

1. **Validate GQA-highway formula** - Test on Llama models with different GQA configurations
2. **Derive recovery ratio formula** - Fit functional form to size vs recovery data
3. **Persistent homology** - Compute Betti numbers across layers to understand topology
4. **GQA-attention relationship** - Why does higher GQA not always mean sharper attention?

**Completed:**
- ✓ Attention eigenvalue analysis (LFM2 explained)
- ✓ Jacobian structure (corrected from rank-1 to near-identity)
- ✓ Highway location (GQA formula discovered, R²=0.941)

*The goal is to move from "we measured X" to "X must be true because Y".*
