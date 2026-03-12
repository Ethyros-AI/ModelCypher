# Curriculum Learning SOTA Research Notes

**Date:** 2026-03-12
**Purpose:** Reference document for curriculum generation protocol design.

---

## 1. Curriculum Ordering for LLMs

**Key finding:** Curriculum ordering matters most for small models (up to 160M).
Reduces gradient noise and spectral saturation. At larger scales, gains shrink.

- arXiv:2601.21698 (Jan 2026): Curriculum as warmup reduces training steps by
  18-45% to reach baseline. Best difficulty signals: compression ratio, lexical
  diversity (MTLD), readability (Flesch Reading Ease).
- arXiv:2511.18903: Standard decaying LR schedules waste curriculum benefits
  because hardest (most valuable) data arrives when LR is lowest. Fix: moderate
  LR decay + curriculum + weight averaging.
- arXiv:2510.19099: No single ordering universally dominates. Forward vs.
  reverse depends on model capability and task complexity. Practical: easy-to-hard
  for first 30-50%, then random.

**Design implication:** Target sub-3B models. Easy-to-hard for first half of
each skill's data, then random sampling. Coordinate with MASS step sizing (no
manual LR schedule in our pipeline).

---

## 2. Synthetic Data Generation

**Key finding:** Synthetic data's structured token dependencies make it easier
to learn from than organic text. Phi-4 surpasses its teacher GPT-4 on STEM.

- **Phi-4** (Microsoft, Dec 2024): 50 synthetic dataset types, 400B synthetic
  tokens. Multi-agent prompting + self-revision + instruction reversal. Seeds
  from filtered web content prioritizing reasoning depth.
- **BeyondWeb** (Datology AI, Aug 2025): Rephrasing-based synthetic data at 8B
  scale matches 180B tokens of RedPajama in 23.2B tokens (7.7x speedup).
  Multiple generation formats essential (Q&A, summaries, reasoning, instructions).
- **Mixing ratios:** ~33% synthetic with natural text gives 5-10x speedup.
  Pure synthetic risks model collapse. 50-30-20 split (textbook/filtered-web/
  educational) is a strong baseline for pedagogical pretraining.

**Design implication:** Protocol requests multiple formats per skill. Don't
over-generate. Quality curation over volume.

---

## 3. Skill Decomposition

**Key finding:** 4K high-entropy compositional samples beat much larger random
datasets. Training on k=2,3 compositions transfers to k=4,5.

- **STEPS** (arXiv:2601.03676, Jan 2026): Hierarchical skill taxonomy using
  structural information theory. Synthesizes data by maximizing marginal
  structural entropy.
- arXiv:2409.19808 (NeurIPS 2024): Models acquire a meta-skill for composition.
  You do not need to enumerate all possible skill combinations.

**Design implication:** Protocol instructs frontier model to generate
compositional examples at k=2,3. Don't request all combinations. The DAG
defines the dependency order; compositional transfer handles the rest.

---

## 4. Noise and Diversity in Training Data

**Key finding:** Diversity is as important as quality. Over-aggressive quality
filtering hurts generalization.

- arXiv:2410.15226: Cluster-based diversity metrics correlate positively with
  both pretraining and SFT performance at 350M-1.4B scale. Synthetic data
  diversity affects downstream SFT more than pretraining.
- Negative examples (wrong answers, common mistakes) improve alignment. LLM-
  generated negatives from zero-shot outputs are as effective as human-written.
- arXiv:2510.00866: Over-aggressive quality filtering creates distributional
  bias.

**Design implication:** 10-20% negative examples. 3+ templates per skill for
surface-form diversity. Don't filter too aggressively.

---

## 5. Small Model Training Strategies

**Key finding:** 10B high-quality tokens sufficient for sub-3B models. Small
models (1B-3B) show the largest gains from fine-tuning.

- **TinyStories:** Models below 10M produce fluent text with constrained-vocab
  synthetic data.
- **Phi family:** 1.3B models match 10-25x larger models with data curation.
- Pedagogical pretraining (2025): 50+ experiments converged on static 50-30-20
  mix. Repeating data shows sharply diminishing returns.

**Design implication:** Focus on sub-3B models where curriculum gives the most
leverage. Small, focused datasets per skill.

---

## 6. Data Quality: LIMA / Less Is More

**Key finding:** Almost all knowledge comes from pretraining. Minimal
instruction tuning teaches format. 1K-5K samples can match much larger datasets.

- **LIMA:** 1,000 samples match much larger instruction tuning sets.
- **LIMR** (2025): 1,389 samples match 8,523 on math benchmarks (extended to RL).
- **LIMIT** (Databricks): Training data must align with target evaluation
  paradigm. Quality = "matches what you're evaluating."
- **s1** (Jan 2025): 1,000 carefully curated examples exceed o1-preview by 27%
  on math. Selection by difficulty, diversity, quality.

**Design implication:** Protocol emphasizes curation criteria over sample count.
50-500 per skill, not 50K.

---

## 7. Frontier-as-Teacher Methods

**Key finding:** Data distillation from strong models beats RL on the student
at small scale. AgentInstruct's multi-agent pipeline with 100+ skill
subcategories achieves 40-54% improvements.

- **AgentInstruct** (Microsoft 2024): Multi-agent pipeline generates both
  prompts and responses from raw documents. 100+ skill subcategories.
- **Orca:** Explanation traces from GPT-4 teach reasoning.
- **WizardLM/Evol-Instruct:** Complexity evolution of prompts.
- **Cosmopedia** (HuggingFace): 25B tokens of pedagogical synthetic data using
  topic taxonomies + retrieval-augmented prompting.
- **Magpie** (ICLR 2025): Generate alignment data by prompting aligned LLMs
  with just the template prefix.

---

## 8. STaR and Self-Improvement

**Key finding:** Absolute Zero Reasoner achieves SOTA with zero external data
via self-play task proposal + solution with code execution as verifier.

- **Quiet-STaR:** Generalizes reasoning to all tokens, not just QA.
- **RL-STaR:** Theoretical framework connecting STaR to policy gradients.
- **Absolute Zero Reasoner** (NeurIPS 2025 Spotlight): Self-play loop, code
  execution as verifier. Zero external data.
- **Open-Reasoner-Zero:** Vanilla PPO with rule-based rewards achieves superior
  reasoning in 1/10th training steps.

**Design implication:** After curriculum mastery, STaR can extend capability
further via self-play on verified problems. The curriculum provides the
foundation; STaR provides the amplification.

---

## 9. Chain-of-Thought Distillation

**Key finding:** Corrupted reasoning traces still transfer reasoning ability.
Structure matters more than correctness of individual steps.

- **DeepSeek-R1 distillation** (Jan 2025): 800K reasoning traces. Distilling
  from strong teacher beats RL on student at small scale.
- **s1:** 1,000 curated examples exceed o1-preview by 27% on math.
- arXiv:2511.05184: Even corrupted traces transfer reasoning ability.
- Adaptive approaches let the model decide when and how much to reason.

**Design implication:** Include reasoning traces in training data even if
imperfect. The structural pattern of multi-step work is what transfers.

---

## 10. Verification and Curation

**Key finding:** Three-tier verification: (a) code execution for verifiable
domains, (b) rejection sampling as simple baseline, (c) LLM-as-judge for
non-verifiable domains (~80% human agreement).

- **RLVR:** Code execution for math, code. Extending to medical.
- **RAFT:** Generate N, keep best (rejection sampling).
- **LLM-as-judge:** Rubric-based, 500-5000x cost reduction vs. human. Unreliable
  on factual correctness.
- Model collapse prevention: verification at every iteration + mixing real data.
- **TarGEN:** Generates synthetic benchmark-task variants (seedless). 1-3%
  higher than training on originals.

**Design implication:** Protocol includes verification field per skill. Code
execution for math/logic. Rubric for non-verifiable domains. The frontier model
provides the verification function.

---

## Synthesis: Design Principles

1. **Student-aware generation:** Profile student -> generate for gaps -> train -> re-profile.
2. **Skill taxonomy as DAG:** Formal dependency proofs, not heuristic difficulty.
3. **Compositional transfer:** Train k=2,3; k=4,5 transfers.
4. **Quality over quantity:** 100-500 per skill, multiple formats, curated.
5. **Negative examples:** 10-20% contrastive signal.
6. **Easy-to-hard ordering:** First 30-50% of each skill's data, then random.
7. **Verification pipeline:** Code execution for verifiable, rubric for rest.
8. **Multiple formats:** Textbook, Q&A, exercises, reasoning chains per topic.
9. **Template diversity:** 3+ surface forms per concept.
10. **Iteration is the product:** The loop (profile -> generate -> train -> eval -> re-profile) matters more than any single curriculum.
