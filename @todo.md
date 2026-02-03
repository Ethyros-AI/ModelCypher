# ModelCypher TODO

**Updated:** 2026-02-02 (Paper 1 data collection complete)

---

## Now (Quick Wins)

- [ ] Add trajectory analysis to `mc model probe`
- [ ] Add geometric fingerprint summary to probe output

---

## Research: Stacked LoRA Self-Improvement

**Core thesis**: Models improve through curated training runs that stack LoRA over time, reinforcing increasingly difficult reasoning and fact patterns. Geometry guides what to train next.

**Curriculum design:**
- [ ] Define "difficulty" geometrically (e.g., trajectory curvature, low-density regions)
- [ ] Identify patterns the model struggles with via activation analysis
- [ ] Design training sets that target geometric weaknesses

**LoRA stacking mechanics:**
- [ ] Test LoRA composition (multiple adapters sequentially applied)
- [ ] Measure geometric change per LoRA layer
- [ ] Determine when to merge vs when to stack
- [ ] Track cumulative capability vs cumulative geometry change

**Self-exploration loop:**
- [ ] Model generates candidate problems
- [ ] Geometry identifies uncertainty/low-coverage regions
- [ ] Training loop targets those regions
- [ ] Measure improvement cycle-over-cycle

---

## Research: Geometric Validation

**Validate expansion_ratio findings:**
- [ ] Test pattern on Llama, Mistral, Phi models
- [ ] Correlate expansion_ratio variance with benchmark performance
- [ ] Test dimension recovery vs expansion_ratio variance correlation

**Mathematical verification:**
- [ ] Check both ratio directions (verify inverse constants exist)
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

## Papers: Data Collection

### Paper 1 (Manifold Hypothesis) — **COMPLETE (Negative Result)**

- [x] Extract semantic prime embeddings from 6 models (LFM2 350M/700M/1.2B, Qwen 3B/Coder-3B/8B)
- [x] Compute Gram matrices → `data/paper1/gram_matrices/`
- [x] Compute pairwise CKA → `data/paper1/cka_pairwise.csv`
- [x] Create frequency-matched control word list (n=200) → `data/paper1/null_distribution/`
- [x] Run 200 random subset CKA measurements
- [x] Compute p-values for prime CKA vs null → `data/paper1/results.json`

**Result**: Primes CKA (0.466) ≤ Random CKA (0.612), p=0.628. Confirms Paper 0 thesis: all vocabulary shares invariant structure. See `papers/NEGATIVE-RESULTS.md`.

### Paper 2 (Entropy Safety)

- [ ] Curate 20 refusal-prone + 20 neutral prompts
- [ ] Define 10 intensity modifiers
- [ ] Run entropy sweep across 4 models × 4 temperatures
- [ ] Generate `data/paper2/modifier_entropy.csv`
- [ ] Generate `data/paper2/temperature_sweep.csv`
- [ ] Curate harmful/benign prompt sets (100 each) — **human review required**
- [ ] Compute safety AUROC → `data/paper2/safety_auroc.csv`

### Paper 3 (Cross-Architecture Geometry)

- [ ] Run geometric comparison: Qwen-3B ↔ Llama-3B, Qwen-7B ↔ Mistral-7B
- [ ] Generate layer coverage scores and Jaccard overlap
- [ ] Document shared vs divergent geometric properties
- [ ] Correlate structure differences with benchmark differences

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

---

## Decision Queue

When ready to focus, pick one:

| Option | What | Why |
|--------|------|-----|
| **Stacked LoRA prototype** | Build minimal self-improvement loop | Core vision |
| Benchmark correlation | expansion_ratio vs downstream scores | Validates geometric theory |
| Cross-arch survey | Test on Llama/Mistral/Phi | Proves universality |

---

*Source files: PLAN.md, .claude/plans/geometric-research-plan.md, papers/*.md, docs/findings/*.md*
