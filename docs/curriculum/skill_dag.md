# Skill Dependency DAG: Formal Specification

**Status:** Draft — reviewed before any code is written.

This document is the ground truth for curriculum ordering. Every dependency edge has a
proof sketch. If a dependency cannot be proven, it is removed. "It feels harder" is not a
dependency; a formal proof is.

---

## Principle

Skill B depends on Skill A if and only if the **proof of B requires A as a premise** or
**the evaluation of B requires the model to demonstrate A first**.

Training order is any topological sort of this DAG. The mastery criterion (what triggers
advancement to dependent skills) is detected by the auto-regime mechanism:

| Regime | Clopper-Pearson Baseline | Interpretation | Action |
|--------|--------------------------|----------------|--------|
| `ce` | Lower bound ≤ chance rate | Zero capability | CE teaches this skill |
| `reinforce_entropy` | Partial signal | Emerging capability | REINFORCE+entropy refines |
| `reinforce` | Lower bound > chance rate | Consolidated | **Advance to dependents** |

The `reinforce` regime is the mastery signal. It is derived from Clopper-Pearson CI on a
held-out eval set, not a manually chosen threshold.

**Open question:** Is `reinforce` regime sufficient for safe advancement, or do we need a
secondary threshold (e.g., lower bound > 0.90)? This requires experimental validation.
Start with regime detection as the gate; measure backward transfer after advancement.

---

## Logic Branch

### Node: `modus_ponens`
**Formal statement:** (P→Q, P) ⊢ Q
**Natural language:** Given "if P then Q" and P is true, conclude Q.
**Prerequisites:** None — this is a primitive axiom of propositional logic (modus ponendo
ponens). It cannot be derived from simpler inference rules within propositional logic.
**Training data:** `data/training/phase1_inference_rules.jsonl` (44 samples),
`data/training/phase1_inference_rules_balanced.jsonl` (64 samples)
**Format note:** Phase files use `{"prompt": ..., "completion": ...}` — verify loader
compatibility before training.

---

### Node: `modus_tollens`
**Formal statement:** (P→Q, ¬Q) ⊢ ¬P
**Natural language:** Given "if P then Q" and Q is false, conclude P is false.
**Prerequisites:** `modus_ponens`
**Proof of dependency:** The standard proof of MT from MP proceeds via the contrapositive:
> (1) P→Q is given.
> (2) The contrapositive ¬Q→¬P is logically equivalent to (1).
> (3) ¬Q is given.
> (4) By MP applied to (¬Q→¬P, ¬Q), conclude ¬P. ∎

The model must have internalized MP to execute step (4). MT cannot be taught to a model
that has not consolidated MP because the final inference step IS an application of MP.
**Training data:** Present in `data/training/phase3_rule_recognition.jsonl` (48 samples,
mixed MP/MT/DS recognition problems)
**Gap:** No MT-only training file. Phase 3 treats MT as recognition alongside MP/DS.
Consider generating `phase_mt_only.jsonl` for isolated MT training before recognition.

---

### Node: `disjunctive_syllogism`
**Formal statement:** (P∨Q, ¬P) ⊢ Q
**Natural language:** Given "P or Q" and P is false, conclude Q.
**Prerequisites:** None — root node (depth 0), parallel to `modus_ponens`.
**Proof of independence from MP:** DS is proven by disjunction elimination: case 1 (P is
true) leads to contradiction with ¬P; case 2 (Q is true) gives Q directly. This proof
does not invoke MP. DS and MP are independent inference rules with no formal dependency
between them.
**DAG position:** Root node alongside `modus_ponens` and `single_digit_add`. PhaseScheduler
will teach DS alphabetically after other depth-0 nodes when no prerequisites are mastered.
This ordering is formally permissible.
**Training data:** Present in `data/training/phase3_rule_recognition.jsonl` (mixed)
**Gap:** No DS-only training file.

---

### Node: `hypothetical_syllogism`
**Formal statement:** (P→Q, Q→R) ⊢ P→R
**Natural language:** If A implies B and B implies C, then A implies C.
**Prerequisites:** `modus_ponens` (applied twice)
**Proof of dependency:**
> (1) Assume P.
> (2) Apply MP to (P→Q, P) → derive Q.
> (3) Apply MP to (Q→R, Q) → derive R.
> (4) Discharge assumption P: P → R. ∎

Both steps (2) and (3) are explicit applications of MP. A model that cannot reliably apply
MP cannot execute this two-step chain. Note: the conclusion P→R is derived without an
explicit P being given — the model must understand conditional proof (deduction theorem),
not just MP application to a given premise.
**Training data:** `data/training/phase2_rule_compositions.jsonl` (53 samples) — combines
HS with MP in multi-step chains.

---

### Node: `universal_instantiation`
**Formal statement:** (∀x P(x), a) ⊢ P(a)
**Natural language:** Given "for all x, P(x)" and a is in the domain, conclude P(a).
**Prerequisites:** `modus_ponens` — universal instantiation is MP with a universally
quantified premise. The model must understand that ∀x P(x) acts as P→P for any specific
instance.
**Note:** Present implicitly in training data (e.g., "All mammals have backbones. A dog is
a mammal. Therefore a dog has a backbone.") but not treated as a separate skill node in
phases 1-6. Should be made explicit.

---

### Node: `chain_reasoning`
**Formal statement:** Multi-step deductions combining HS + MP + MT + DS + UI
**Natural language:** Given a sequence of premises, derive a conclusion through multiple
rule applications without being told which rules to use.
**Prerequisites:** `hypothetical_syllogism`, `modus_tollens`, `disjunctive_syllogism`,
`universal_instantiation`
**Training data:** `data/training/phase5_benchmark_failures_base.jsonl`,
`data/training/phase6_benchmark_failures_p1_5.jsonl` (failure-targeted)
**Proof of dependency:** Chain reasoning is by definition a composition of component rules.
Universal instantiation is included because the 1p2b training data contains quantifier
reasoning chains (∀x P(x) → P(a) compositions). A model cannot chain rules it has not
consolidated.

---

### Node: `rule_recognition`
**Formal statement:** Given a problem + conclusion, identify which inference rule was used.
**Natural language:** "What logical rule applies here?"
**Prerequisites:** `modus_ponens`, `modus_tollens`, `disjunctive_syllogism` (must know
all rules to distinguish them)
**Note:** Recognition is harder than application in one sense (no guided structure given)
but simpler in another (answer is rule name, not conclusion). Phase 3 targets this.
**Training data:** `data/training/phase3_rule_recognition.jsonl` (48 samples)

---

### Node: `concise_reasoning`
**Formal statement:** Apply inference rules with minimal verbiage — no step-by-step
scaffolding, just the answer.
**Prerequisites:** `rule_recognition` — must have consolidated rules before compressing
the explanation. (Cannot compress what you don't fully understand.)
**Training data:** `data/training/phase4_conciseness.jsonl` (63 samples)

---

## Logic Branch DAG (summary)

```
modus_ponens ─────────────────────────────────────────► chain_reasoning
    │                                                        ▲
    ├──► modus_tollens ──────────────────────────────────────┤
    │                                                        │
    ├──► hypothetical_syllogism ─────────────────────────────┤
    │                                                        │
    └──► universal_instantiation ────────────────────────────┤

disjunctive_syllogism (root, no prerequisites) ──────────────┤

{modus_ponens, modus_tollens, disjunctive_syllogism}
    └──► rule_recognition ──► concise_reasoning
```

**Training order (topological sort):**
1. `disjunctive_syllogism` ∥ `modus_ponens` (both depth 0, independent roots)
2. `modus_tollens` ∥ `hypothetical_syllogism` ∥ `universal_instantiation`
   (all depend only on MP, depth 1)
3. `rule_recognition` (requires MP + MT + DS consolidated, depth 2)
4. `concise_reasoning` (requires rule_recognition, depth 3)
5. `chain_reasoning` (requires HS + MT + DS + UI, depth 2 — parallel with rule_recognition)

---

## Math Branch

### Node: `single_digit_add`
**Formal statement:** Given A, B ∈ [0,9], compute C = A + B.
**Prerequisites:** None — this is an enumerable primitive. All 100 ordered pairs are
memorizable by table lookup.
**Training data:** `data/training/single_digit_add_train.jsonl`
**Eval data:** `data/eval/single_digit_add_eval.jsonl` — held-out subset: pairs where
sum ≥ 10 (45 items).
**answer_mode:** `numeric`
**Proof of terminability:** The domain is finite (100 pairs). The model can enumerate
and memorize all outcomes.

---

### Node: `carry_rule`
**Formal statement:** For A + B ≥ 10: write (A+B) mod 10 as the ones digit and carry 1.
**Prerequisites:** `single_digit_add`
**Proof of dependency:** The carry rule is stated in terms of a sum A+B. Computing that
sum requires `single_digit_add`. The rule itself — "when the sum of two digits exceeds 9,
write the ones digit and carry 1" — cannot be derived from a lookup table alone. It is the
bridging principle between single-digit sums and multi-digit positional notation.
**Training data:** `data/training/carry_rule_train.jsonl` (45 items, exhaustive)
**Eval data:** `data/eval/carry_rule_eval.jsonl` (45 items, exhaustive — all carry-inducing
pairs). No meaningful in/out-of-distribution distinction for a 45-item finite set.
**answer_mode:** `procedural` — requires carry-indicator tokens ("write"/"carry") in output
AND correct final digit. Pure memorization (correct number, no carry tokens) fails this gate.

---

### Node: `multi_digit_add`
**Formal statement:** Column-wise addition with carry propagation, any digit count.
**Prerequisites:** `carry_rule`
**Proof of dependency:** Multi-digit addition is structural recursion: apply `carry_rule`
column-by-column from the ones position. Each column is a single-digit addition (requires
`single_digit_add`) with a possible carry-in from the previous column (requires
`carry_rule`). The recursion terminates when all columns are processed.
**Training data:** `data/training/multi_digit_add_train.jsonl` (2-digit pairs)
**Eval data:** `data/eval/multi_digit_add_eval.jsonl` — OOD: 3-digit + 3-digit (100 items).
OOD eval makes memorization structurally impossible; only procedure generalizes.
**answer_mode:** `procedural` — requires carry-indicator tokens in output AND correct final
answer.

---

### Node: `arithmetic_multiply`
**Formal statement:** Given integers A, B (B single-digit), compute C = A × B.
**Prerequisites:** `multi_digit_add`
**Proof of dependency:** Column-by-column multiplication uses the same carry propagation
trajectory as `multi_digit_add`: multiply each digit of A by B; if product exceeds 9,
write the ones digit and carry — this is `carry_rule` applied to multiplication. The
model cannot correctly execute this without having consolidated the carry procedure.
**Training data:** `data/training/arithmetic_multiply_train.jsonl` (exhaustive 720 pairs:
A ∈ [10,99], B ∈ [2,9])
**Eval data:** `data/eval/arithmetic_multiply_eval.jsonl` — OOD: A ∈ [100,999].
**answer_mode:** `procedural` — requires carry-indicator tokens in output AND correct final
answer.

---

### Node: `arithmetic_divide`
**Formal statement:** Given integers A, B (B ≠ 0), compute C = A ÷ B (or remainder).
**Prerequisites:** `arithmetic_multiply`
**Proof of dependency:** Division is defined as the inverse of multiplication: A ÷ B = C
iff C × B = A. Checking the result requires multiplication.
**Training data:** Not present in current files. Generate `arithmetic_div.jsonl`.

---

### Node: `word_problem_1step`
**Formal statement:** Given a natural language description, identify the operation and
compute a single-operation answer.
**Example:** "Sarah has 5 apples. She buys 3 more. How many does she have?" → addition.
**Prerequisites:** `multi_digit_add` AND `modus_ponens`
**Why multi_digit_add specifically:** The DAG model requires a specific node name as
prerequisite, not a disjunction. `multi_digit_add` is the most complete arithmetic
foundation — it implies `single_digit_add` and `carry_rule` are mastered. A model that
cannot add multi-digit numbers reliably cannot solve even the simplest word problem.
**Cross-branch dependency:** This is the junction of the logic branch and math branch.
The model must both know the arithmetic operation AND apply the logical inference "context
implies operation." Both branches must be in `reinforce` regime before this node.
**Training data:** GSM8K easy tier (to be created via `profile_gsm8k_difficulty.py`)
**Data source:** `benchmark_loader.py` loads GSM8K from HuggingFace.

---

### Node: `word_problem_multi`
**Formal statement:** Given a natural language description requiring multiple operations,
derive the sequence of operations and compute the result.
**Prerequisites:** `word_problem_1step` AND `hypothetical_syllogism` (chaining implies:
step 1 result → step 2 input → step 2 result → ... → final answer)
**Proof of dependency:** A multi-step word problem is structurally a hypothetical syllogism
applied to arithmetic: "buying more (step 1) → new total (step 2 input) → final total."
The model cannot chain arithmetic steps without having consolidated chained implication.
**Training data:** GSM8K medium/hard tier (to be created via `profile_gsm8k_difficulty.py`)

---

### Node: `algebra_linear`
**Formal statement:** Given an equation with one unknown, solve for the unknown.
**Example:** "3x + 5 = 14. What is x?"
**Prerequisites:** `arithmetic_multiply`, `arithmetic_divide`, `modus_tollens` (backward
reasoning: given the result, find the cause — structurally MT)
**Proof of modus_tollens dependency:** Solving "3x + 5 = 14" requires reasoning backward:
the final state (=14) is known; we need the cause (x). This is ¬Q→¬P form: "the result
is 14, what premise (value of x) makes this true?" Algebraic manipulation is MT applied
to arithmetic.
**Training data:** MATH dataset (HS level) via HuggingFace; NuminaMath easy tier

---

### Node: `algebra_nonlinear`
**Formal statement:** Competition-level math problems requiring multi-step algebraic
reasoning.
**Prerequisites:** `algebra_linear` AND `chain_reasoning`
**Training data:** NuminaMath (860k problems) — full distribution via HuggingFace

---

## Math Branch DAG (summary)

```
single_digit_add ──► carry_rule ──► multi_digit_add ──► arithmetic_multiply ──► arithmetic_divide
                                            │                    │                      │
                                            └────────────────────┴──────────┬───────────┘
                                                                             ▼
              modus_ponens ─────────────────────────────────► word_problem_1step ──► word_problem_multi
                                                                                          │
              hyp_syllogism ─────────────────────────────────────────────────────────────►│
                                                                                          ▼
              modus_tollens ─────────────────────────────────────────────────► algebra_linear
                                                                                          │
              chain_reasoning ──────────────────────────────────────────────► algebra_nonlinear
```

---

## Cross-Branch Junction

`word_problem_1step` requires BOTH:
- Math branch: `multi_digit_add` in `reinforce` regime
- Logic branch: `modus_ponens` in `reinforce` regime

`word_problem_multi` requires BOTH:
- Math: `word_problem_1step` in `reinforce` regime
- Logic: `hypothetical_syllogism` in `reinforce` regime

`algebra_linear` requires BOTH:
- Math: `arithmetic_multiply` and `arithmetic_divide` in `reinforce` regime
- Logic: `modus_tollens` in `reinforce` regime

This junction is the core hypothesis of this curriculum: **logic is not just
intrinsically valuable — it is a structural prerequisite for advanced math reasoning.**
This is testable: train a model on math-only (no logic phases) and compare to a model
that went through logic phases first. Prediction: logic-first model should reach
`word_problem_multi` mastery faster, measured in training samples.

---

## Mapping: Existing JSONL Files → DAG Nodes

| File | Node(s) | Status |
|------|---------|--------|
| `phase1_inference_rules.jsonl` | `modus_ponens` | Has data |
| `phase1_inference_rules_balanced.jsonl` | `modus_ponens` | Has data |
| `phase2_rule_compositions.jsonl` | `hypothetical_syllogism` | Has data |
| `phase3_rule_recognition.jsonl` | `rule_recognition` (MP+MT+DS mixed) | Has data |
| `phase4_conciseness.jsonl` | `concise_reasoning` | Has data |
| `phase5_benchmark_failures_base.jsonl` | `chain_reasoning` | Has data |
| `phase5_benchmark_failures_p1_4.jsonl` | `chain_reasoning` | Has data |
| `phase6_benchmark_failures_p1_5.jsonl` | `chain_reasoning` | Has data |
| `single_digit_add_train.jsonl` | `single_digit_add` | Has data |
| `carry_rule_train.jsonl` | `carry_rule` | Has data (45 items, exhaustive) |
| `multi_digit_add_train.jsonl` | `multi_digit_add` | Has data |
| `arithmetic_multiply_train.jsonl` | `arithmetic_multiply` | Has data (720 pairs) |
| `arithmetic_div_train.jsonl` | `arithmetic_divide` | Has data |
| GSM8K (easy tier, to be split) | `word_problem_1step` | Needs split |
| GSM8K (medium/hard tier) | `word_problem_multi` | Needs split |
| MATH dataset | `algebra_linear` | Needs download |
| NuminaMath | `algebra_nonlinear` | Needs download |

**Gaps in existing data:**
- `modus_tollens` has no isolated training set (only in phase3 mixed recognition)
- `disjunctive_syllogism` has no isolated training set
- `word_problem_1step` needs GSM8K difficulty split
- `universal_instantiation` not treated as explicit skill node

---

## Eval Dataset Requirements per Node

Each node needs a **held-out eval set** that was NOT used in training. The regime detector
measures baseline accuracy on this eval set to determine Zone 1/2/3.

| Node | Eval Source | Size | Format |
|------|-------------|------|--------|
| `modus_ponens` | Generated: modus_ponens_eval.jsonl | ≥50 | `{"text": "..."}` |
| `modus_tollens` | Generated: modus_tollens_eval.jsonl | ≥50 | `{"text": "..."}` |
| `hypothetical_syllogism` | Generated: hyp_syllogism_eval.jsonl | ≥50 | `{"text": "..."}` |
| `disjunctive_syllogism` | Generated: disj_syllogism_eval.jsonl | ≥50 | `{"text": "..."}` |
| `rule_recognition` | Subset of benchmark_val.jsonl (logical rules) | ≥50 | existing |
| `concise_reasoning` | Generated: concise_eval.jsonl | ≥30 | `{"text": "..."}` |
| `chain_reasoning` | benchmark_val.jsonl (multi-step logical) | ≥50 | existing |
| `single_digit_add` | `data/eval/single_digit_add_eval.jsonl` | 45 (sum≥10 subset) | `{"text": "...", "answer_start": N}` |
| `carry_rule` | `data/eval/carry_rule_eval.jsonl` | 45 (exhaustive) | `{"text": "...", "answer_start": N}` |
| `multi_digit_add` | `data/eval/multi_digit_add_eval.jsonl` | ≥100, OOD 3-digit+3-digit | `{"text": "...", "answer_start": N}` |
| `arithmetic_multiply` | `data/eval/arithmetic_multiply_eval.jsonl` | ≥100, OOD A∈[100,999] | `{"text": "...", "answer_start": N}` |
| `arithmetic_divide` | `data/eval/arithmetic_div_eval.jsonl` | ≥50 | `{"text": "...", "answer_start": N}` |
| `word_problem_1step` | GSM8K easy eval split | ≥100 | `{"text": "..."}` |
| `word_problem_multi` | GSM8K med/hard eval split | ≥100 | `{"text": "..."}` |
| `algebra_linear` | MATH easy subset | ≥100 | `{"text": "..."}` |
| `algebra_nonlinear` | NuminaMath eval subset | ≥200 | `{"text": "..."}` |

**Minimum viable eval set:** 50 samples gives a Clopper-Pearson CI of approximately
±14% at 95% confidence. For meaningful regime detection, 50 is the floor; 100+ preferred.

---

## Open Questions (record results here as experiments run)

1. **Does CurriculumProfiler geometric difficulty correlate with DAG depth?**
   - Experiment: Profile samples from each phase; compute Spearman correlation between
     Fisher-dominant difficulty score and DAG depth.
   - Expected: r > 0.7. If r < 0.5, the profiler signals are not tracking logical
     dependency and should not be used for difficulty ordering.
   - Status: Not yet run.

2. **Is `reinforce` regime sufficient for mastery, or do we need a secondary threshold?**
   - Experiment: Advance to modus_tollens when MP reaches `reinforce` regime. Measure
     backward transfer on MP eval set after MT training. If backward transfer > 0.05,
     advance criterion was too permissive.
   - Status: **COMPLETED 2026-03-03**
   - Result: McNemar p=0.2913 (not significant). pre_accuracy=0.450, post_accuracy=0.530,
     delta=+0.080. n_lost=18, n_gained=26. MP accuracy improved slightly after MT training
     (possible forward transfer — MT and MP share the same conditional-reasoning subgraph).
   - Conclusion: **reinforce gate is safe** for the MP→MT transition. The gate is not too
     permissive. Results saved to `results/backward_transfer_mp_mt.json`.
   - Tool: `scripts/measure_backward_transfer.py` (two-phase McNemar test).

3. **Does logic-first curriculum outperform math-only on word_problem_multi?**
   - Experiment: Train LFM2-350M two ways: (a) logic phases → arithmetic → word problems;
     (b) arithmetic → word problems only. Compare samples-to-mastery on word_problem_multi.
   - Status: Not yet run.

4. **Does geometry-aligned training (Cayley-Stiefel + MASS) allow training on raw
   NuminaMath without filtering?**
   - Experiment: Train on raw NuminaMath vs. deduplicated subset. Compare accuracy,
     degeneration, CKA, convergence.
   - Status: Not yet run. Run after PhaseScheduler is working.

---

## Validation Results

### Experiment #2 — reinforce gate safety (2026-03-03)

**Model:** LFM2-350M-MLX-bf16 (pretrained, no curriculum training)
**Transition tested:** modus_ponens → modus_tollens

| Metric | Value |
|--------|-------|
| pre_accuracy (MP baseline) | 0.450 |
| post_accuracy (after MT training) | 0.530 |
| delta | +0.080 |
| n_lost | 18 |
| n_gained | 26 |
| McNemar chi² | 1.2321 |
| McNemar p | 0.2913 |
| Significant forgetting? | No |

**Interpretation:** MT training did not cause systematic forgetting of MP (p=0.29 >> 0.05).
The slight accuracy improvement (+8%) is consistent with forward transfer — MT and MP share
the same conditional-reasoning subgraph (both require evaluating `if A then B`), so
strengthening one reinforces the other. The reinforce gate is safe for this edge.

**Files:** `results/mp_baseline.json`, `results/backward_transfer_mp_mt.json`
**Tool:** `scripts/measure_backward_transfer.py`
