# ModelCypher TODO

**Updated:** 2026-02-03

---

## Stacked LoRA Self-Improvement — COMPLETE ✓

**Infrastructure built:**
- `LoRAStacker` - Tracks cumulative barrier/CKA drift, decides when to merge
- `CurriculumProfiler` - Defines difficulty geometrically (Fisher-dominant composite score)
- CLI: `mc stack init|status|train|merge|improve|profile|select`
- 5 trained adapter phases with cumulative merges

**Curriculum design:** ✓
- Difficulty defined geometrically via CKA, barrier, Fisher, curvature, density, ID
- Fisher mean identifies uncertainty (9% higher for incorrect answers)
- `mc stack select` selects curriculum by difficulty strategy

**LoRA stacking mechanics:** ✓
- Barrier threshold: 0.03, CKA drift threshold: 0.1, max 5 adapters before merge
- Fisher-weighted merging
- State persistence with cumulative tracking

**Self-exploration loop:** ✓
- `mc stack improve` runs iterative self-improvement
- `mc stack profile` measures geometric difficulty
- Density percentile identifies sparse/low-coverage regions

---

## Geometric Validation — UPDATED ✓

**Completed:**
- Cross-architecture survey: LFM2, Qwen, Granite, DeepSeek
- Sandglass vs traditional highway patterns documented
- Recovery ratio inverse correlation with model size
- Attention eigenvalue analysis (LFM2 = rank-1 uniform, Qwen = rank 3-4 selective)

**Corrected (2026-02-03):**
- ~~Jacobian effective rank = 1.0~~ → **Numerical artifact of bf16 + small epsilon**
- True layer Jacobians are **full-rank, near-identity** (σ ≈ 1 for all directions)
- Lesson: Always verify numerical methods across precision levels

---

## Open Research Questions

- [ ] Correlate expansion_ratio variance with benchmark performance
- [x] Test on Llama → Downloaded Llama-3.2-3B, validated GQA→alignment chain
- [ ] Compare pre/post nonlinearity geometry
- [x] What determines highway location? → **GQA→Q/K alignment→selectivity** (see docs/research/)
- [x] Why do different architectures have different attention selectivity? → **GQA constrains K capacity**
- [x] Derive recovery ratio formula from model size → **Relational: gap/decay from measurables** (no arbitrary constants)
- [x] Why do same-GQA models (Granite vs Qwen at GQA=4) have different alignment? → **Subspace overlap (r=0.93)**
- [ ] Derive decay coefficients (0.6, 0.8) from first principles — currently unexplained
- [ ] What determines highway gap when convergence < 1? (Different mechanism from exit)

---

*Source files: docs/findings/*.md, src/modelcypher/core/use_cases/self_improve/, src/modelcypher/core/use_cases/curriculum_profiler.py*
