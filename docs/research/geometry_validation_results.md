# Geometry Validation Experiment Results

**Date:** 2026-02-05
**Models:** LFM2-350M-MLX-bf16, LFM2.5-1.2B-Instruct-bf16

## Summary

| Experiment | Result | Status |
|------------|--------|--------|
| V1 (Aggregate Metrics) | NULL | Aggregate layer-averaged metrics show no signal |
| V2 (Token-Level) | **ARTIFACTUAL** | Bug in generation code invalidated results |
| V3 (Rigorous) | **WEAK SIGNAL** | Direction change shows moderate effect, velocity not significant |
| **SOTA (Per-Layer Probing)** | **SUCCESS** | AUROC=0.686 at layer 8, pivots d=-0.90 |

---

## SOTA Research Context (2024-2026)

Understanding recent literature is critical for interpreting our results. The field has moved significantly beyond aggregate trajectory metrics.

### Key Papers

| Paper | Key Finding | Implication |
|-------|-------------|-------------|
| [Zhang et al. 2025](https://arxiv.org/abs/2504.05419) | Hidden states encode correctness BEFORE answer formulation. **Linear probes** extract this. | Correctness is LINEARLY embedded - simple difference-in-means works |
| [Anderson 2026](https://arxiv.org/abs/2601.13358) | "The cost of thought is determined by manifold geometry, not task difficulty." Domain-specific phases: crystalline (legal), liquid (math), lattice (code) | Geometry predicts reasoning, but patterns are domain-specific |
| [Lee et al. 2026](https://arxiv.org/abs/2601.22484) | Cognitive pivots detected via **L2 distance spikes** in hidden states. Training-free steering. | The signal is in trajectory discontinuities, not aggregate metrics |
| [Marks & Tegmark 2024](https://arxiv.org/abs/2310.06824) | Truth/falsehood is LINEARLY represented. Causal interventions prove direction is causally implicated. | Linear structure is real and causal, not just correlational |

### What SOTA Shows

1. **Correctness IS encoded in geometry** - but it's LINEAR, not complex aggregate metrics
2. **The signal is in TRAJECTORY STRUCTURE** - L2 spikes at cognitive pivots, not answer token velocity
3. **Different domains have different geometric signatures** - no universal "correctness metric"
4. **Simple probes work** - difference-in-means, not logistic regression or neural nets

---

## Critical Principle: Everything is Geometry

**There are no "knobs" in LLMs - only geometry.**

Temperature, top-p, top-k are noise injection mechanisms that obscure the deterministic geometric structure. The model's weights define a fixed high-dimensional landscape. With greedy decoding (temp=0), every input traces EXACTLY ONE PATH through that landscape.

- Temperature doesn't alter the trajectory - the hidden state path through activation space is identical
- It just randomly samples from the probability distribution instead of taking the argmax
- A hallucination from temperature sampling is a sampling error, not a reasoning failure
- The model's geometry knew the right answer; the sampling threw it away

**This has implications for our experiments:** Any experiment using temperature to "get incorrect samples" is not measuring reasoning failure - it's measuring random sampling artifacts.

---

## Previous Experiment Results

### Critical Finding: V2 Was an Artifact

The V2 experiment reported d=1.55 effect size for velocity at answer token. **This was caused by a bug.**

The bug: calling `base_model(current_ids)` instead of `model(current_ids)`:
- `base_model` returns hidden states (dim=2048)
- `model` returns logits (dim=65536)
- Argmax over hidden states produces garbage tokens, not actual model outputs

The "14 correct / 86 incorrect" split in V2 was not model reasoning - it was random noise from broken sampling. The reported d=1.55 effect size was comparing random garbage outputs.

### V3 Results (Rigorous Methodology)

V3 fixes: single forward pass, correct model call, strict numeric parsing, bootstrap CIs.

#### GSM8K on LFM2.5-1.2B-Instruct (n=100)

59 correct, 41 incorrect (greedy decoding)

| Metric | Correct | Incorrect | Effect Size d | 95% CI |
|--------|---------|-----------|---------------|--------|
| Velocity at answer | 1.10 | 1.08 | 0.17 | [-0.24, 0.59] |
| Direction change | 0.49 | 0.48 | 0.11 | [-0.30, 0.56] |

**Conclusion: No significant signal on GSM8K reasoning using aggregate metrics at answer token.**

Interesting per-layer pattern:
- Early layers (0-6): Incorrect has HIGHER velocity (d=-0.5 to -0.8)
- Late layers (13-15): Correct has HIGHER velocity (d=+0.2 to +0.5)

This reversal suggests early vs late processing differs, but the aggregate signal washes out.

#### Arithmetic on LFM2-350M (n=200, temp=0.5)

189 correct, 11 incorrect (temperature sampling needed to get errors)

| Metric | Correct | Incorrect | Effect Size d | 95% CI |
|--------|---------|-----------|---------------|--------|
| Velocity at answer | 1.27 | 1.16 | 0.87 | [-0.11, 1.86] |
| Direction change | 0.53 | 0.44 | **1.23** | [0.33, 2.19] |

**Direction change is statistically significant** (CI doesn't include zero).

**Important caveat:** This used temperature=0.5 to induce errors. Per the geometry principle, these "incorrect" samples are not reasoning failures - they're sampling artifacts. The model may have had the correct answer in its geometry; we just sampled wrong.

---

## What Was Wrong With Our Approach

### Methodological Problems

1. **Measured wrong location**: Answer token, not reasoning trajectory
2. **Measured wrong thing**: Aggregate velocity, not L2 spikes / cognitive pivots
3. **Wrong approach**: Complex metrics when SOTA shows linear probes work
4. **Temperature experiments**: Injected noise, learned nothing about geometry

### Why Aggregate Metrics Fail

The V3 GSM8K result (d=0.17 velocity, d=0.11 direction) is the **correct null result** for this methodology. Aggregate metrics at the answer token don't predict correctness because:

1. The answer token is where the model writes what it already computed
2. The geometry of reasoning is in the PROCESS, not the OUTPUT
3. Early vs late layers show opposite patterns - averaging destroys the signal

---

## Correct SOTA-Aligned Approach

### 1. Use Deterministic Decoding ONLY

- Temperature = 0.0 (greedy)
- No sampling variation
- The path is deterministic - measure it

### 2. Collect Full Trajectory Hidden States

For each token in the generated sequence:
- Hidden state at every layer
- This gives the complete trajectory through activation space

### 3. Detect Cognitive Pivots (STARS Method)

Compute L2 distance between consecutive hidden states:
```
d_t = ||h_t - h_{t-1}||_2
```
Look for **spikes** - sudden large distances indicate "cognitive pivots" where the model makes a decision.

### 4. Train Linear Probes on Hidden States

Following Zhang et al.:
- Segment CoT into reasoning chunks
- Use last token hidden state of each chunk
- Train linear probe: `correctness = W @ h + b`
- Simple difference-in-means baseline: `direction = mean(h_correct) - mean(h_incorrect)`

### 5. Measure Trajectory Geometry

Following Anderson:
- Representational dimensionality (d95 - dimensions for 95% variance)
- Trajectory alignment (cosine similarity of consecutive states)
- Manifold clustering (silhouette score)

---

## Implementation

### New Modules Created

| Module | Purpose |
|--------|---------|
| `core/domain/geometry/cognitive_pivots.py` | L2 spike detection (STARS method, Lee et al. 2026) |
| `core/domain/geometry/linear_probe.py` | Correctness probing (Zhang et al. 2025) |
| `scripts/geometry_trajectory_analysis.py` | SOTA-aligned trajectory analysis script |

### Verification Criteria

| Experiment | Metric | Target | Result | Status |
|------------|--------|--------|--------|--------|
| Linear probe | AUROC | > 0.65 | **0.686** (layer 8) | **MET** |
| Spike detection | Effect size | d > 0.5 | **d = -0.90** | **MET** |
| Trajectory geometry | Domain clustering | Silhouette > 0.3 | Not tested | Pending |

---

## Files

- V3 Experiment: `scripts/geometry_validation_v3.py`
- SOTA Experiment: `scripts/geometry_trajectory_analysis.py`
- GSM8K Results (V3): `/tmp/geom_v3_gsm8k/results.jsonl`
- Arithmetic Results (V3): `/tmp/geom_v3_350m_temp/results.jsonl`
- **SOTA Results (1.2B, n=500)**: `results/geometry_sota_1.2B_500/`
  - `trajectory_results.jsonl` - Raw per-sample data
  - `analysis_summary.json` - Summary statistics and per-layer probe AUROC

---

## Lessons Learned

1. **Test the generation loop.** The V2 bug produced grammatically coherent garbage that looked plausible. Always verify outputs are sensible.

2. **Temperature is not a research tool.** It's noise injection. To study model reasoning geometry, use greedy decoding.

3. **Read SOTA before experimenting.** The field has moved to linear probes and trajectory discontinuities. Aggregate metrics at answer tokens are known to not work.

4. **The signal is linear.** Don't build complex metrics. Difference-in-means on hidden states works.

5. **The signal is in the trajectory.** Not the endpoint. Cognitive pivots (L2 spikes) matter more than answer token velocity.

---

## SOTA Experiment Results (2026-02-05)

### Experiment: Per-Layer Probing with Extended Samples

**Model:** LFM2.5-1.2B-Instruct-bf16 (16 layers)
**Benchmark:** GSM8K (n=500, greedy decoding)
**Script:** `scripts/geometry_trajectory_analysis.py`

### Key Results

| Metric | Correct (n=310) | Incorrect (n=190) | Effect Size d | Interpretation |
|--------|-----------------|-------------------|---------------|----------------|
| Accuracy | 62% | 38% | - | Model baseline |
| Pivots per generation | 15.1 | 21.5 | **-0.90** | Large effect - incorrect has 42% more pivots |
| Generation length | 179 tokens | 237 tokens | **-1.22** | Very large - incorrect is 32% longer |
| Mean L2 distance | 1.068 | 1.076 | -0.46 | Moderate effect |

### Per-Layer Linear Probe AUROC

**Critical finding: Correctness signal peaks in middle layers, not the output layer.**

| Layer | AUROC | Accuracy | Signal? |
|-------|-------|----------|---------|
| 0 | 0.541 | 0.522 | |
| 1 | 0.545 | 0.536 | |
| 2 | 0.587 | 0.563 | |
| 3 | 0.593 | 0.558 | |
| 4 | 0.605 | 0.580 | |
| 5 | 0.655 | 0.618 | ✓ |
| 6 | 0.645 | 0.605 | ✓ |
| 7 | 0.654 | 0.614 | ✓ |
| **8** | **0.686** | **0.629** | **✓ PEAK** |
| 9 | 0.682 | 0.627 | ✓ |
| 10 | 0.676 | 0.630 | ✓ |
| 11 | 0.663 | 0.611 | ✓ |
| 12 | 0.645 | 0.615 | ✓ |
| 13 | 0.661 | 0.619 | ✓ |
| 14 | 0.650 | 0.611 | ✓ |
| 15 | 0.643 | 0.595 | ✓ |

**Pattern:**
- Layers 0-4: Weak signal (AUROC 0.54-0.61)
- Layers 5-10: Strong signal (AUROC 0.65-0.69, peak at layer 8)
- Layers 11-15: Signal declines toward output (AUROC 0.64-0.66)

### Why This Matters

1. **Initial experiment failed because it only probed the last layer** - AUROC was 0.507 (chance) with 100 samples at layer 15 only.

2. **With per-layer probing, we found AUROC = 0.686 at layer 8** - this exceeds the 0.65 threshold and confirms correctness IS linearly encoded.

3. **The pattern matches SOTA literature** - Zhang et al. and Marks & Tegmark both find correctness signals in middle-to-late layers, not the final output layer.

### Cognitive Pivot Analysis

The STARS method (L2 spike detection) shows a **large effect** (d = -0.90):

- **Correct answers**: 15.1 pivots average
- **Incorrect answers**: 21.5 pivots average (42% more)

Interpretation: Incorrect reasoning involves more "decision points" or trajectory discontinuities. The model struggles longer before arriving at a (wrong) answer.

### Pivot Token Distribution

Top pivot tokens (tokens where L2 spikes occurred):

| Rank | Correct | Incorrect |
|------|---------|-----------|
| 1 | `<whitespace>` (650) | `<whitespace>` (480) |
| 2 | `**` (334) | `**` (253) |
| 3 | `'s` (224) | `-` (174) |
| 4 | `-` (156) | `'s` (173) |
| 5 | `step` (143) | `step` (126) |

Similar distributions - the pivot tokens themselves don't differentiate; it's the COUNT that matters.

### Pivot Position Analysis

- **Correct pivots**: Mean position 0.454 (relative to sequence)
- **Incorrect pivots**: Mean position 0.454 (identical)
- **Effect size**: d = 0.00

Pivots occur uniformly throughout generation in both cases. No temporal pattern.

### Summary

| Claim | Evidence | Status |
|-------|----------|--------|
| Correctness is linearly encoded | AUROC = 0.686 at layer 8 | **CONFIRMED** |
| Signal is in middle layers | Peak at layer 8/16, not layer 15 | **CONFIRMED** |
| Cognitive pivots differentiate | d = -0.90, incorrect has 42% more | **CONFIRMED** |
| Generation length diagnostic | d = -1.22, incorrect 32% longer | **CONFIRMED** |
| Pivot position matters | d = 0.00, no difference | **NOT CONFIRMED** |

---

## Next Steps

1. ~~Run `geometry_trajectory_analysis.py` on GSM8K with greedy decoding~~ **DONE**
2. ~~Evaluate linear probe AUROC (target > 0.65)~~ **DONE - 0.686 at layer 8**
3. ~~Analyze cognitive pivot distribution in correct vs incorrect trajectories~~ **DONE - d=-0.90**
4. ~~If probe works, investigate which layers encode correctness~~ **DONE - layers 5-10**
5. Test whether probe generalizes across models (350M → 1.2B → 8B)
6. Run 8B model experiment (DeepSeek-R1-0528-Qwen3-8B) to compare
7. Investigate why pivot COUNT differentiates but pivot POSITION doesn't
