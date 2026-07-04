# R1/R2 Local 350M Handoff

Status: `R1 no-go on seed 42`, `R2 active blocker: inference-representation
collapse`

Use this file as the single re-entry point for all R1 and R2 work before
opening logs, result directories, or code. All other docs
(`RESEARCH-ROADMAP.md`, `OPEN-MATHEMATICAL-QUESTIONS.md`,
`r2_closed_loop_controller_log.md`, `PISSA-BUDGET-TRACKING-STATUS.md`) defer
here for the next falsifier.

## Restart Sentence

```text
The R2 falsifier chain has closed optimizer, loop mechanics, LR schedule,
early stopping, training duration, and PiSSA structural budget as explanations.
The sole remaining mechanism is inference-representation collapse: train CKA
stays healthy (0.95) while inference CKA collapses (min 0.01, mean 0.58) across
all configurations, including full cosine-decay training to val_stable at
epoch 8. The structural sigma_k budget is now diagnostic-only — it does not
predict or correlate with behavioral damage.
```

## What Closed Today

### Pipeline mechanics are no longer the blocker

The canonical path now does all of the following correctly:

- derives geometry,
- injects the geometry-derived LoRA surface,
- trains,
- saves adapters,
- saves failure artifacts on gate failure,
- emits a coherent identity surface,
- verifies spectral bounds and CKA.

Relevant retained code:

- [`src/modelcypher/core/use_cases/dataset_training_service.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/core/use_cases/dataset_training_service.py)
- [`src/modelcypher/core/use_cases/_dataset_training_service_helpers_mixin.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/core/use_cases/_dataset_training_service_helpers_mixin.py)
- [`src/modelcypher/backends/_mlx_training_adapter_train_mixin.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/backends/_mlx_training_adapter_train_mixin.py)
- [`src/modelcypher/backends/_mlx_training_adapter_adapter_io_mixin.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/backends/_mlx_training_adapter_adapter_io_mixin.py)
- [`src/modelcypher/core/domain/training/identity.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/core/domain/training/identity.py)

### The identity surface is coherent

Canonical doctrine and artifacts now identify the shipped path as
`geometric_lora`.

Current components:

- `method=geometric_lora`
- `init_method=pissa`
- `optimizer=adamw_cosine` on the default canonical controller
- `optimizer=fisher_mass` on MASS-controlled branches
- `controller=mass`
- `stopping=geometric_certificate`

### The active R1 substrate is the quick-aligned corpus

The old synthetic benchmark pair is retained as a mechanical proof substrate,
not the active R1 closure corpus.

Active data:

- [`data/training/r1_quick_aligned_train.jsonl`](/Users/jasonkempf/ModelCypher/data/training/r1_quick_aligned_train.jsonl)
- [`data/training/r1_quick_aligned_val.jsonl`](/Users/jasonkempf/ModelCypher/data/training/r1_quick_aligned_val.jsonl)
- [`data/training/r1_quick_aligned_manifest.json`](/Users/jasonkempf/ModelCypher/data/training/r1_quick_aligned_manifest.json)
- [`scripts/build_r1_quick_aligned_dataset.py`](/Users/jasonkempf/ModelCypher/scripts/build_r1_quick_aligned_dataset.py)

### Raw RMT signal rank is no longer used as adapter rank

The active rank derivation now uses:

- adaptation-budget rank from weight geometry,
- data-rank ceiling,
- activation effective-rank ceiling,
- RMT signal-rank ceiling.

That reduced the active surface from the catastrophic 7.7M-parameter regime to
about 341K trainable parameters on the local 350M tuple.

### PiSSA budget re-anchor is exact, not overcounted

The controller now re-anchors on exact cumulative spectral displacement for the
implicit PiSSA operator instead of subtracting per-step Frobenius bounds that
over-counted the true spectral norm.

Relevant retained code:

- [`src/modelcypher/core/domain/training/spectral_budget.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/core/domain/training/spectral_budget.py)
- [`src/modelcypher/backends/_mlx_training_adapter_core_mixin.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/backends/_mlx_training_adapter_core_mixin.py)
- [`src/modelcypher/backends/_mlx_training_adapter_train_mixin.py`](/Users/jasonkempf/ModelCypher/src/modelcypher/backends/_mlx_training_adapter_train_mixin.py)

### Matched-trace MASS now reports the right optimizer identity

The MASS-controlled matched-trace branch now reports as `fisher_mass` on the
canonical identity surface.

Relevant retained tests:

- [`tests/domain/training/test_identity.py`](/Users/jasonkempf/ModelCypher/tests/domain/training/test_identity.py)
- [`tests/test_dataset_training_service_strict.py`](/Users/jasonkempf/ModelCypher/tests/test_dataset_training_service_strict.py)

## What Closed on 2026-03-13

### PiSSA structural budget is a false stop surface

The 96-step matched-trace MASS diagnostic showed ALL 18 PiSSA modules exceed
the structural sigma_k budget (ratios 1.3×–6.5×), but no spectral property
(budget ratio, sigma_k, gap_r, rank) correlates with inference CKA collapse.
The collapse is a cascade/position effect through the residual stream — L4
collapses to CKA≈0.0 with NO adapters (pure cascade from L2).

The structural budget protects `sigma_k = S[structural_rank - 1]` where
`structural_rank ≈ 222–340` (Shannon effective rank), but adapter ranks are
1–72. By Weyl-Lidskii interlacing, a rank-r perturbation shifts at most r
singular values significantly. The budget boundary is ~300 positions away from
where the adapter operates.

Code changes (2026-03-13):

- `_structural_sigma_budget_is_enforceable()` returns False when adapter rank
  is far below Shannon structural rank
- PiSSA ratios remain as diagnostic logging but no longer act as behavioral
  stop surface or controller input
- `adapter_saturation_exhausted` early-stop gated by enforceability check
- Tests updated: 11/11 passed in `test_mlx_training_adapter_strict.py`

### Early stopping and training duration are not the cause

Full-epoch canonical run (2026-03-13) with PiSSA budget diagnostic-only:

```bash
poetry run mc train validate-derived \
  -m /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  -d data/training/r1_quick_aligned_train.jsonl \
  --eval-data data/training/r1_quick_aligned_val.jsonl \
  --trials 1 --benchmark quick \
  --report-path /tmp/r1_default_no_budget_gate.json
```

Result:

- stop reason: `val_stable` at iter 7712 (epoch 8) — natural convergence
- online eval trajectory: 11 → 9 → 11 → 11 → 11 → 12 → 12
  (baseline=11, stable or slightly above)
- val loss: 2.4286 → 2.4103 → 2.4052 → 2.4036 → 2.4038 → 2.4034 → 2.4034
- adapter_sat: 1.30 → 1.54 → 1.64 → 1.69 → 1.69 → 1.70 → 1.70
  (plateaued, structurally "violated" but behaviorally inert)
- train CKA: min=0.9491, mean=0.9846 — healthy
- **inference CKA: min=0.0116, mean=0.5843 — still collapses**
- post benchmark: `gsm8k=0%`, `arc_easy=80%`, `boolq=90%`, overall `56.7%`
  (identical to the old premature-stop result)

Interpretation:

- Removing the false stop surface changed nothing about behavioral outcome.
- The canonical path trains to natural cosine-decay convergence without
  behavioral collapse on the online-eval surface.
- Inference-representation collapse is independent of training duration,
  stop regime, and spectral budget enforcement.

### Batched MLX eval path (performance)

The training loop had 20+ `mx.eval` synchronization barriers per step
(including 18 per-layer calls in `_layer_measurements_from_gradient`). Each
`mx.eval` stalls the GPU pipeline. Refactored to batch all per-step scalar
realization into minimal `mx.eval` calls. Tests: 2 new regression tests for
batched per-layer measurement helper (PiSSA and NB-LoRA paths).

## What Was Actually Measured

### Mechanical proof on the old benchmark pair

Retained adapter:

- [`350m-geometric-lora-r1`](/Volumes/CodeCypher/models/adapters/350m-geometric-lora-r1)

Result:

- the pipeline trained and saved correctly,
- validation loss improved,
- spectral bounds passed,
- train-space CKA stayed high,
- benchmark behavior stayed mixed because the corpus was curriculum-mismatched.

Interpretation:

- this run proved the pipeline is real,
- it did not close R1.

### Quick-aligned default controller: safer, but still not good enough

Command family:

```bash
poetry run mc train validate-derived \
  -m /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  -d data/training/r1_quick_aligned_train.jsonl \
  --eval-data data/training/r1_quick_aligned_val.jsonl \
  --trials 1 --benchmark quick \
  --report-path /tmp/r1_default.json
```

Measured result:

- pre benchmark: `gsm8k=50%`, `arc_easy=90%`, `boolq=70%`
- online eval baseline: `11/20`
- epoch 1 online eval: `11/20`
- stop reason: `adapter_saturation_exhausted`
- epoch 1:
  `train_loss=2.5239`, `val_loss=2.4286`, `adapter_sat=1.2965`,
  `remaining=0.0000e+00`
- post benchmark:
  `gsm8k=0%`, `arc_easy=80%`, `boolq=90%`, overall `56.7%`

Interpretation:

- the default controller is not immediately destructive on the online-eval
  surface,
- but it still overruns the reduced-rank adapter budget by epoch 1,
- and it still fails the R1 competitiveness requirement.

### Quick-aligned MASS matched-trace controller: worse than default

Command family:

```bash
poetry run mc train validate-derived \
  -m /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  -d data/training/r1_quick_aligned_train.jsonl \
  --eval-data data/training/r1_quick_aligned_val.jsonl \
  --trials 1 --benchmark quick \
  --optimizer-research-mode adamw_matched_trace \
  --report-path /tmp/r1_mass.json
```

Measured result:

- same pre benchmark as the default run
- PiSSA budget re-anchor active from step 0, then every `96` steps
- remaining budget hit `0` by step `96` and stayed there
- epoch 1 online eval: `2/20`
- stop reason: `adapter_saturation_exhausted`
- epoch 1:
  `train_loss=2.8026`, `val_loss=2.6904`, `adapter_sat=7.6725`,
  `remaining=0.0000e+00`
- post benchmark:
  `gsm8k=0%`, `arc_easy=50%`, `boolq=50%`, overall `33.3%`

Interpretation:

- the MASS matched-trace branch is strictly worse than the default controller
  on the active substrate,
- the gate is not the problem,
- the controller is still consuming the PiSSA spectral budget inside the first
  re-anchor window.

## What Is No Longer A Live Explanation

- "The pipeline might still be mechanically broken."
- "The failure is just the old synthetic benchmark pair."
- "Raw RMT signal rank is a valid adapter-rank prescription."
- "The PiSSA budget controller fails because exact displacement is
  unavailable."
- "The MASS branch only looked bad because the runtime mislabeled the
  optimizer."
- "The structural sigma_k budget violation causes benchmark damage."
  (adapter_sat=1.70 coexisted with stable 12/20 online eval for 4 epochs)
- "Premature stopping (adapter_saturation_exhausted) causes benchmark damage."
  (full cosine-decay training to val_stable produces identical 56.7% benchmark)
- "Training duration is insufficient."
  (7 epochs to convergence, same outcome as 1-epoch premature stop)

## R2 Falsifier Chain Status

| Falsifier | Status | Evidence |
|-----------|--------|----------|
| Optimizer (Cayley vs AdamW) | closed | r2 behavioral probe / V3 structural |
| MASS matched-trace step sizing | closed | r2 behavioral probe adamw |
| Closed-loop layer freeze | closed | r2 closed loop cayley |
| Loop mechanics (seq_len, batch, iter cap, early stops) | closed | r2 loop parity |
| Cosine LR schedule drift | closed | r2 adamw cosine schedule audit |
| PiSSA structural sigma_k budget | closed | 2026-03-13 full-epoch run |
| Early stopping / training duration | closed | 2026-03-13 val_stable at epoch 8 |
| Prompt distribution | closed | 2026-03-16 Phase A: C1→C2 = 12%, not primary |
| Causal masking | closed | 2026-03-16 Phase A: C2→C3 = 0% |
| KV-cache prefill path | closed | 2026-03-16 Phase A: C3→C4 = 0% |
| **Inference-representation collapse** | **CLOSED** | 2026-03-16 Phase A+D: step-0 decode divergence, not geometry |

## What Closed on 2026-03-16

### R2 geometry thread is closed

The inference-representation collapse is a measurement artifact. Two experiments
established this:

**Phase A** (5-condition CKA matrix, `scripts/r2_masking_diagnosis.py`):
Isolated prompt distribution, causal masking, KV-cache prefill, and
autoregressive decode as independent variables. Results:

- C1 (canonical train probes + bidirectional): min=0.9211, mean=0.9825
- C2 (benchmark prompts + bidirectional): min=0.7761, mean=0.9326
- C3 (benchmark prompts + causal): min=0.7817, mean=0.9310
- C4 (benchmark prompts + KV-cache prefill): min=0.7761, mean=0.9326
- C5 (benchmark prompts + 5 greedy decode steps): min=0.4749, mean=0.5686

Attribution: prompt distribution 12%, masking 0%, KV-cache 0%, autoregressive
feedback 88%. Geometry is preserved on same inputs (C1-C4 all healthy).

**Phase D** (logit divergence, `scripts/r2_logit_divergence.py`):
100% of benchmark prompts (30/30) diverge at the very first generated token
(step 0). The adapted model produces different token distributions than the
base model immediately:

- gsm8k: base → reasoning words (' Each', ' The'), adapted → numbers ('18', '250')
- arc_easy: base → letter labels (' A', ' C'), adapted → content words ('bone', 'light')
- boolq: base → ' yes'/' Yes', adapted → 'yes'/'no' (no leading space)

The "inference CKA collapse" (min 0.01, mean 0.58) was CKA computed on
divergent token sequences — different inputs produce low CKA by construction.

### Backend improvement: `mask_mode` parameter

Added `mask_mode` parameter to `collect_hidden_activations()` in the `Backend`
protocol and MLX/JAX/CUDA implementations. `mask_mode="causal"` routes per
layer type: attention layers get `mask="causal"`, conv layers get `mask=None`.
Masking was not the R2 cause (0% attribution), but the fix is correct
telemetry — the old measurement was bidirectional regardless of model behavior.

## What Closed on 2026-03-16 (Phases E+F)

### Training data format is the strongest current mechanism

**Phase E** (corpus audit + format-contract ablation):

The quick-aligned corpus teaches GSM8K as bare digits (100% of 2,251 samples
are single-token answers). The full reasoning chain exists in
`metadata["full_answer"]` but was stripped by `benchmark_loader.py:183`
(splits on `####`). Format contracts broke the 350M model (empty output) —
this model cannot follow meta-instructions.

**Phase F** (chain-preserved retrain):

Rebuilt the corpus with full GSM8K reasoning chains. Retrained with the same
geometry-derived config (only the data format changed).

| Task | Base | Old adapter | Chain adapter |
|------|:----:|:-----------:|:-------------:|
| gsm8k | 5/10 | 1/10 | **4/10** |
| arc_easy | 9/10 | 8/10 | 8/10 |
| boolq | 7/10 | 9/10 | 8/10 |
| overall | 70% | 60% | **66.7%** |

Step-0 divergence analysis on the chain adapter confirmed the mechanism
changed:

| GSM8K metric | Old adapter | Chain adapter |
|--------------|:-----------:|:-------------:|
| Step-0 top-1 match | 0/10 | **5/10** |
| Top-5 overlap | 0.1/5 | **3.2/5** |
| First-token type | Digits ('18', '250') | Reasoning words (' Janet', ' The') |
| Avg divergence step | 0.0 | **1.6** |

The chain adapter attempts multi-step reasoning matching the base model's
pattern. On 3/10 prompts it runs 4-6 matched steps before the arithmetic
breaks down. The remaining gap is arithmetic-execution granularity: the
chains assume `180/5 = 36` is a primitive, but the 350M model doesn't
reliably execute that operation.

**Token-mass note:** Chain-preserved GSM8K answers average ~53 words vs 1 word
in the original corpus. With equal sample counts per task, GSM8K dominates the
gradient by ~10x. Loss/CKA numbers are not directly comparable to the old
quick-aligned runs; only the benchmark suite is apples-to-apples.

Adapter retained: `/Volumes/CodeCypher/models/adapters/350m-geometric-lora-r1-chain-preserved`

## Restart Sentence (updated 2026-03-16)

```text
The R2 falsifier chain is fully closed. The inference CKA collapse was a
measurement artifact (CKA on divergent token sequences, not broken geometry).
The mechanism is step-0 decode divergence caused by training data that stripped
the token-space work tape needed for multi-step reasoning. Chain-preserved
retraining partially repaired GSM8K (1/10 → 4/10) and shifted the generation
pattern toward the base model's reasoning-word frontier, but the intermediate
arithmetic steps still break down. The remaining blocker is arithmetic-
execution granularity: the model needs curriculum from primitives before
GSM8K-level chains are useful.
```

## Current Direction

Three pillars for the ship path:

1. **Teach the loop.** Data is a control surface. Chain granularity must match
   the model's current arithmetic fluency. Curriculum from single-digit
   combinations → place-value → multi-digit operations → word problems.

2. **Preserve the loop at decode.** Bare-number training actively suppressed
   chain-of-thought. The adapter must not collapse intermediate computation.

3. **Escalate to tools.** When internal looping exceeds reliable capacity, the
   model should call out to calculators/Python. General-human competence + tool
   awareness, not calculator replacement.

Geometry is the instrumentation layer: CKA verifies representation preservation,
logit divergence detects readout collapse, corpus audit reveals what the data
actually teaches. These are diagnostic instruments, not the explanation for
benchmark failure.

## Exact Next Falsifier

Arithmetic-primitives curriculum. The highest-value experiments are:

1. **Arithmetic curriculum dataset**: single-digit combinations (all addition/
   subtraction/multiplication pairs through 9), place-value decomposition,
   multi-digit column operations. Train an adapter on these primitives first,
   then evaluate whether GSM8K-level chains become reliable.

2. **Tool-escalation training**: teach the model to recognize when a computation
   exceeds its reliable internal looping and emit a tool-call token.

3. **Token-balanced corpus**: if arithmetic curriculum helps GSM8K but regresses
   other tasks, balance the corpus by token mass rather than sample count.

## Do Not Spend Tomorrow On These

- do not revisit the R2 geometry thread (fully closed)
- do not run format-contract ablation (closed: 350M can't follow contracts)
- do not run broad layerwise CKA (would re-measure token divergence)
- do not expand to more seeds before arithmetic curriculum is tested
- do not promote MASS as a candidate default controller

## Verification Snapshot

- PiSSA budget tests: `38/38` passed
- full suite after re-anchor work: `7595` passed, `5` pre-existing failures
- optimizer identity follow-up: focused regression slice `4 passed`

## Historical Stage A Note

The retained Stage A frozen-tuple no-go still stands:

- canonical `nb_lora` on the old benchmark pair is not competitive enough to
  justify seed expansion,
- `geometric_pissa_nb_surface` was the best matched-surface arm,
- R1 seed expansion remains blocked until the canonical local 350M path becomes
  benchmark-competitive on the active substrate.
