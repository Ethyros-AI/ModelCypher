# ModelCypher TODO

**Updated:** 2026-02-01

---

## Now (Quick Wins)

- [ ] Add trajectory analysis to `mc model probe`
- [ ] Add geometric fingerprint summary to merge diagnostics
- [ ] Document 4-level model classification theory in AGENTS.md

---

## Research: Geometric Validation

**Validate comp/φ findings:**
- [ ] Test pattern on Llama, Mistral, Phi models
- [ ] Correlate comp/φ variance with benchmark performance
- [ ] Test dimension recovery vs comp/φ variance correlation

**Mathematical verification:**
- [ ] Check both ratio directions (verify inverse constants exist)
- [x] Null hypothesis: random matrices vs trained weights
  - **RESULT:** Random matrices have MORE constant matches than trained weights
  - Constants in SVD ratios are pareidolia, not real structure
  - Removed all constant-matching code from codebase (2026-02-01)
- [x] Null hypothesis: expansion_ratio (comp/φ replacement)
  - **RESULT:** Training CREATES expansion/compression structure (random = flat)
  - DeepSeek-R1: ratio ≈ 1.2 (near target). LFM2-350M: ratio ≈ 3.3
  - The metric is REAL - different models have different natural ratios
  - **FINDING:** Instruction tuning creates PROMPT-ADAPTIVE geometry
    - Base models: Fixed peak position (0.78% variance across prompt types)
    - Fine-tuned models: Adaptive peaks (3-5% variance across prompt types)
    - Factual/Instruction prompts → earlier peak (87-92%) → compression
    - Reasoning prompts → later peak (98-100%) → full expansion
    - The model dynamically adjusts its geometry based on input type
- [ ] Compare untrained vs trained model geometry
- [ ] Pre vs post nonlinearity geometry comparison
- [ ] Gram matrix eigenvalue analysis
- [ ] Residual stream geometry tracking

**Manipulation experiments:**
- [ ] Orthogonal rotation (verify geometry unchanged)
- [ ] Uniform singular value scaling effects
- [ ] Surgical SVD modification (force specific ratios)
- [ ] Jacobian analysis of information flow
- [ ] Rank-1 perturbation toward missing constants

---

## Research: Merge Capabilities

**DeepSeek-R1 → LFM2.5 experiment:**
- [ ] Probe both models
- [ ] Run geometric merge
- [ ] Benchmark merged model
- [ ] Compare to baseline
- [ ] Write RESULTS.md

**Geometry-aware merging:**
- [ ] Test interpolation instead of pure null-space projection
- [ ] Measure capability transfer vs geometry change tradeoff

**LoRA dimension recovery:**
- [ ] Add LoRA to specialist final layers
- [ ] Train with dimension recovery loss
- [ ] Measure if comp/φ variance increases

---

## Papers: Data Collection

### Paper 1 (Manifold Hypothesis)

- [ ] Extract semantic prime embeddings from 6 models (Qwen 0.5B/1.5B/3B, Llama 1B/3B, TinyLlama)
- [ ] Compute Gram matrices → `data/paper1/gram_matrices/`
- [ ] Compute pairwise CKA → `data/paper1/cka_pairwise.csv`
- [ ] Create frequency-matched control word list (n=200)
- [ ] Run 200 random subset CKA measurements
- [ ] Compute p-values for prime CKA vs null

### Paper 2 (Entropy Safety)

- [ ] Curate 20 refusal-prone + 20 neutral prompts
- [ ] Define 10 intensity modifiers
- [ ] Run entropy sweep across 4 models × 4 temperatures
- [ ] Generate `data/paper2/modifier_entropy.csv`
- [ ] Generate `data/paper2/temperature_sweep.csv`
- [ ] Curate harmful/benign prompt sets (100 each) — **human review required**
- [ ] Compute safety AUROC → `data/paper2/safety_auroc.csv`

### Paper 3 (Cross-Architecture)

- [ ] Run intersection maps: Qwen-3B ↔ Llama-3B, Qwen-7B ↔ Mistral-7B
- [ ] Generate layer coverage scores and Jaccard overlap
- [ ] Create coding eval suite (50 HumanEval problems)
- [ ] Create creative eval suite (50 prompts with rubrics)
- [ ] Run 5-way baseline comparison (source, target, naive avg, TIES, ours)

### Paper 4 (Toolkit)

- [ ] Feature comparison table vs TransformerLens, CircuitsVis, mergekit, LM-Eval

---

## Publication

**arXiv submission (per paper):**
- [ ] Convert markdown to LaTeX
- [ ] Format BibTeX citations
- [ ] Add author/affiliation metadata
- [ ] Verify figure/table references
- [ ] Choose categories (cs.LG, cs.CL, cs.AI)

**GitHub + Zenodo:**
- [ ] Finalize papers
- [ ] Update pyproject.toml version
- [ ] Create release v0.1.0-papers
- [ ] Connect Zenodo for DOI
- [ ] Add DOI badge to README

---

## Code Debt

- [ ] `profile_service.py:399` — Add embedding trajectory support
- [ ] `compression/__init__.py` — Implement GeodesicNullSpaceCompressor
- [ ] `compression/__init__.py` — Implement RankingPreservingOptimizer
- [ ] `compression/__init__.py` — Implement ComposableLayerCompressor

---

## Decision Queue

When ready to focus, pick one:

| Option | What | Why |
|--------|------|-----|
| CLI fingerprint tool | `mc model fingerprint` command | Makes research usable |
| Geometry-aware merge | Allow partial geometry change | Improves capability transfer |
| Benchmark correlation | comp/φ vs downstream scores | Validates theory |
| Cross-arch survey | Test on Llama/Mistral/Phi | Proves universality |

---

*Source files: PLAN.md, .claude/plans/geometric-research-plan.md, papers/*.md, docs/findings/*.md*
