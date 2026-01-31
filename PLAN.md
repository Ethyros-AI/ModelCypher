# Research Plan: 2026-01-31

## Yesterday's Key Finding

**Geometry = Capability**

We discovered that specialist models (Qwen-Coder, DeepSeek-R1) lack "dimension recovery" in final layers, which is why they show constant comp/φ = 0.618. Base models recover to 3-8 effective dimensions, enabling task differentiation.

---

## Today's Options

### Option A: Induce Dimension Recovery (LoRA Experiment)

**Goal**: Can we teach a specialist model to recover dimensions?

**Approach**:
1. Take Qwen-Coder-0.5B (specialist, no recovery)
2. Add LoRA adapters to L22-24 (final layers)
3. Train on diverse tasks with dimension recovery loss
4. Measure if comp/φ variance increases

**Why interesting**: If this works, we can "generalize" specialist models.

---

### Option B: Geometry-Aware Merging

**Goal**: Transfer capability by allowing partial geometry change.

**Approach**:
1. Instead of null-space projection, use interpolation
2. Merge Qwen-Coder-3B → Qwen-3B-Instruct (same architecture)
3. Allow geometry to shift toward source
4. Measure capability transfer vs geometry change tradeoff

**Why interesting**: Current null-space merge protects geometry too well.

---

### Option C: Benchmark Correlation Study

**Goal**: Does dimension recovery predict downstream performance?

**Approach**:
1. Run dimension analysis on 5-10 models of varying quality
2. Benchmark each on standard tasks (MMLU, HumanEval, etc.)
3. Correlate final EffDim with benchmark scores

**Why interesting**: Could establish dimension recovery as a quality metric.

---

### Option D: Cross-Architecture Survey

**Goal**: Is dimension collapse + recovery universal?

**Approach**:
1. Analyze different architectures (Llama, Mistral, Phi, etc.)
2. Check if all show the same pattern:
   - Early exploration (high dim)
   - Mid-layer collapse (low dim)
   - Final recovery (varies by model type)

**Why interesting**: Validates the theory across architectures.

---

### Option E: CLI Tool for Geometric Fingerprinting

**Goal**: Make our discoveries usable.

**Approach**:
1. Create `mc model fingerprint` command
2. Output: comp/φ variance, compression gate strength, final EffDim
3. Single command to classify model as base/instruct/specialist

**Why interesting**: Practical tool from research findings.

---

## Recommended Priority

1. **Option E** (CLI tool) - Low effort, high impact, makes research usable
2. **Option B** (geometry-aware merge) - Directly addresses capability transfer
3. **Option C** (benchmark correlation) - Validates theory with data

---

## Quick Wins Available

- [ ] Add trajectory analysis to existing `mc model probe` command
- [ ] Create geometric fingerprint summary in merge diagnostics
- [ ] Document the 4-level theory in AGENTS.md

---

## Notes

Models available for testing:
- LFM2-350M, LFM2-1.2B (base)
- LFM2.5-1.2B-Instruct (general)
- Qwen2.5-Coder-0.5B, Qwen2.5-Coder-3B (specialist)
- Qwen2.5-3B-Instruct (general)
- DeepSeek-R1-8B (specialist)
- Granite-3B-Code (specialist)
