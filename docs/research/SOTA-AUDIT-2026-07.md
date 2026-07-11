# SOTA Audit: J-Space and the 2026 Measurement Frontier

**Date:** 2026-07-10  
**Scan window:** 2025-12-10 through 2026-07-10, plus one older benchmark needed
to interpret the recent feature papers  
**Status:** Evidence review and roadmap decision record, not model validation  
**Source policy:** Primary lab reports, paper pages, and author code only  
**Supersedes:** `docs/research/SOTA-AUDIT-2026-03.md`

This review asks a narrow question: does recent work change what ModelCypher
should measure, benchmark, or stop claiming? It does not promote any external
preprint to doctrine, and it does not report a ModelCypher replication that was
not run.

## Executive Verdict

The measurement-workbench direction remains current. The important correction
is that whole-activation observables are not enough. Recent work increasingly
combines a specific measurement operator with a causal intervention, and shows
that low-variance structure can dominate behavior while being nearly invisible
to global variance, CKA, perplexity, or global intrinsic dimension.

Anthropic's J-space result is the clearest instance. Its Jacobian lens measures
a context-averaged first-order route from an intermediate residual stream to
future output. The resulting J-space is a sparse union of nonnegative cones,
not a low-dimensional linear subspace. It carries little activation variance
but can have large selective behavioral effects. This is relevant to future
same-input CKA-blindness questions. It does not reopen the retained 350M R2
geometry thread: that "inference CKA collapse" was already shown to be a
divergent-token measurement artifact, and the retained failure was traced to
training-data format. A new J-space study requires a new same-input failure.

The training field also moved. Schedule-free and spectral optimization are now
active, benchmarked areas. ModelCypher cannot present "no schedule" or spectral
optimization as a unique contribution. The defensible product remains the
operator-explicit measurement and derivation layer; optimizer claims remain
blocked on head-to-head evidence.

## Decision Ledger

| Finding | Roadmap link | Decision | Promotion gate |
| --- | --- | --- | --- |
| Jacobian lens and J-space | `A1`; conditional `R2` | Park as a research-only measurement operator; activate only for a new same-input unexplained failure | Open-model replication, corpus-convergence trace, same-size random-direction null, and selective intervention |
| Contextual curvature controls entropy | `WS4.2` | Replicate the published operator; retire generic entropy-curvature wording | Trajectory-aligned intervention changes entropy while a norm-matched misaligned control does not |
| Local ID predicts alignment where global ID does not | `A1`, `WS4.2` | Report local and global estimators separately | Synthetic coverage plus published-profile replication; no cross-model claim without commensurability |
| In-context learning reorganizes geometry | `A1` | Treat prompt context and example state as part of every observation bundle | Same model and task under controlled context changes; frozen probe manifest |
| LoRA delta activations form partly distinct feature geometry | `R2` | Measure delta activations and principal angles; retain as a candidate mechanism | Base-versus-adapter intervention predicts held-out behavior |
| Quantization can preserve perplexity while damaging features | `R4` | Add a fixed-basis feature-survival observable to bit-depth sweeps | Identical tokens and frozen basis across precisions; held-out family |
| SAE explanations need behavioral fidelity certificates | `A1`, `R2`, `R4` | Require reconstruction, pool-mismatch, and intervention evidence | Non-vacuous certificate or direct held-out intervention |
| Schedule-free spectral optimizers now have large-scale results | `R1` | Add applicable schedule-free controls; remove novelty language | Same model, data, horizon, compute, and evaluator |
| HiP-LoRA and Spectral Surgery overlap the spectral-adapter thesis | `R1`, `R5` | Benchmark or explicitly exclude with a compatibility reason | Matched adapter budget and retained preservation metrics |

## Anthropic J-Space

### What the report establishes

Anthropic defines the layerwise average Jacobian

```text
J_l = E_prompt,t,t' >= t [ d h_final,t' / d h_l,t ]
```

and reads an activation through the model's unembedding after transport by
`J_l`. Rows of `W_U J_l` are token-indexed J-lens directions. The report frames
these directions as a first-order causal readout, in contrast to a trained lens
whose objective is output prediction. See
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
and the [reference implementation](https://github.com/anthropics/jacobian-lens).

Three details matter for this repository:

1. J-space is not a linear subspace. The token-indexed frame is overcomplete,
   and J-space is defined by sparse nonnegative combinations. For a fixed
   occupancy it is a union of cones. PCA rank, a single projector, or a null
   space is therefore not an equivalent operator.
2. Low variance is not low causal importance. Anthropic reports that the
   J-space component is below 10 percent of activation variance and that a
   concept-vector decomposition has a median J-space share of 6-7 percent,
   while matched interventions on that component are much more effective than
   interventions on the remainder.
3. Occupancy is measurable rather than a product default. Although the paper
   explores fixed sparse budgets, it also defines occupancy where marginal
   reconstruction improvement falls below a same-size random-direction
   control. ModelCypher should use that null-derived boundary and must not copy
   a paper-specific `k` into production.

The report is careful about scope. Its lens is limited by single-token names,
the averaged Jacobian trades prompt-specific exactness for a fixed map, and the
method is first order. Anthropic's code is a one-commit reference release that
is explicitly not maintained. This is a replication substrate, not a library
dependency.

### Relation to current ModelCypher work

ModelCypher already has the pieces around the operator: canonical activation
capture and comparison bundles, backend `jvp`/`vjp` transforms, Jacobian
spectrum diagnostics, prompt-family manifests, and base-versus-adapter
comparison. It does not yet have the J-lens estimator or a sparse-frame
intervention contract.

The strongest potential connection is `R2`, but only after matching inputs.
A causally privileged, low-variance frame could move while whole-activation
similarity remains high. The existing 350M trace cannot test that hypothesis:
its low inference CKA compared different generated token sequences, and the
same-input geometry remained healthy. J-space is therefore a directional
hypothesis for a future failure, not an alternative explanation of the closed
one.

```text
prediction:
  on identical prompts and token positions, an unexplained failing adapter
  changes J-frame coordinates or J-space transport before held-out behavioral
  degradation, while whole-state CKA can remain high

falsifier:
  the J-space measurements do not separate retained safe and failing traces,
  or norm-matched J-space interventions are no more selective than random or
  non-J-space controls
```

If a same-input failure activates this candidate, keep it inside the existing
`A1` measurement atlas or `R2` pipeline-validation family. Do not create a
free-standing result family. The minimum observation bundle should record:

- frozen corpus and prompt-family manifest identity
- layer, source-position, target-position, architecture, scale, and precision
- corpus-convergence residual for `J_l`
- J-lens frame spectrum and Gram geometry
- occupancy derived against same-size random directions
- reconstruction fraction and excess over the random null
- base-versus-adapter J-frame alignment and delta-activation coordinates
- J-space, non-J-space, and random-direction intervention effects

The existing five-term observable contract remains authoritative. Context
distribution belongs in `geometry_state`; corpus averaging and intervention
belong in `measurement_operator`. Changing the doctrine-level function
signature requires an owner decision.

### Claims that are not allowed

- J-space is not evidence that a model is conscious.
- A top token from a lens is not a faithful explanation by itself.
- A low-variance component is not negligible by default.
- Whole-state CKA agreement is not a certificate of J-space preservation.
- Anthropic's occupancy choices are not ModelCypher constants.
- A first-order averaged Jacobian is not an exact per-prompt circuit.

## Recent Papers That Change the Work

### 1. Curvature now has a direct replication target

[Representational Curvature Modulates Behavioral Uncertainty in Large Language
Models](https://arxiv.org/abs/2604.23985) reports contextual trajectory
curvature correlated with next-token entropy on GPT-2 XL and Pythia-2.8B.
Trajectory-aligned interventions modulate entropy while geometrically
misaligned controls do not. This is precisely the `WS4.2` anchor.

Action: implement their contextual operator faithfully in `mc analyze`, compare
it with the current layer-rotation and curvature observables, and report any
disagreement. Do not revive the broader entropy-to-curvature-to-ID chain from
the March audit unless the published intervention survives locally.

### 2. Context is part of geometry state

[Large language models reorganize representational geometry during in-context
learning](https://arxiv.org/abs/2605.28854) reports that successful ICL is
accompanied by geometry that becomes more separable online and is consistent
with prototype-like evidence integration.

Action: `A1` bundles comparing prompts must identify demonstration order,
label mapping, and context state. A layer metric without that conditioning is
not commensurable across ICL conditions.

### 3. Local ID has stronger current support than global ID

[Local Intrinsic Dimension of Representations Predicts Alignment and
Generalization](https://arxiv.org/abs/2601.22722) reports that local ID explains
alignment and generalization relationships that global dimension misses.

Action: retain global TwoNN as one view, but make the existing local-dimension
map a co-equal output in replication studies. TwoNN is a finite-sample
estimate, not a direct dimension measurement; confidence intervals and sample
convergence belong in the artifact.

### 4. Adapter analysis should isolate delta activations

[Feature Geometry of LoRA Adapters](https://arxiv.org/abs/2605.28896) studies
adapter-specific delta activations and reports weak alignment between
adapter-specific SAE dictionaries and pretrained dictionaries on Gemma-2-9B.
It is a single-author preprint and does not establish a general law.

Action: add delta-activation CKA, principal angles, and fixed-frame
reconstruction to the `R2` candidate set. Do not adopt SAE features as causal
units without intervention.

### 5. Quantization needs a fixed-basis survival audit

[Perplexity Can Miss SAE Feature Damage Under Quantization](https://arxiv.org/abs/2606.03002)
uses a frozen SAE basis on identical tokens and reports graded feature damage
that task-level metrics can miss.

Action: extend `R4` bit-depth sweeps with a fixed full-precision basis. The
basis may be an SAE dictionary, a validated J-lens frame, or another declared
operator, but it must remain frozen across precision states. Perplexity and
whole-state CKA alone cannot close the frontier claim.

### 6. Sparse features need certificates and causal baselines

[From Sparse Features to Trustworthy Proxies](https://arxiv.org/abs/2606.18383)
derives a post-hoc certificate from proxy risk, reconstruction gap, concept-pool
mismatch, and sparse complexity. The older but field-defining
[MIB benchmark](https://arxiv.org/abs/2504.13151) found SAE features no better
than neurons for causal-variable localization, while supervised distributed
alignment search performed best.

Action: a feature explanation is exploratory unless reconstruction, held-out
generalization, and causal intervention all survive. ModelCypher should report
raw proxy terms and must not equate sparsity or a readable token with mechanism.

### 7. Optimizer novelty is crowded

[SF-NorMuon](https://arxiv.org/abs/2605.23061) reports a schedule-free spectral
optimizer matching or exceeding tuned AdamW across multiple horizons at 125M
and 772M parameters. [ScheduleFree+](https://arxiv.org/abs/2605.19095) reports
learning-rate-free and schedule-free LLM training at larger scale. These are
pretraining results, so direct applicability to low-rank adaptation must be
tested rather than assumed.

Action: remove field-position claims based on "zero schedule" or spectral
updates. `R1` needs an applicable schedule-free baseline or a documented
operator-level incompatibility before any such comparison is omitted.

### 8. Spectral adapter competitors are now direct

[HiP-LoRA](https://arxiv.org/abs/2604.17751) separates principal and orthogonal
update channels with a singular-value-weighted stability budget.
[Spectral Surgery](https://arxiv.org/abs/2603.03995) reweights trained LoRA
singular values using calibration gradients.

Action: HiP-LoRA is a direct comparator for spectral preservation and merge
claims. Spectral Surgery is a comparator for any future advice emitted from
`mc analyze lora-svd`. Both remain external preprints until independently
replicated; neither should be copied into the canonical path on abstract-level
evidence.

### 9. Patching studies need raw and prompt-only controls

[Patch-Effect Graph Kernels](https://arxiv.org/abs/2605.06480) explicitly
compares graph summaries with raw patch-effect and prompt-only controls.

Action: if `R2` uses compressed patching signatures, the raw patch tensor and
surface-cue control remain mandatory. Compression accuracy is not a causal
circuit result.

## Corrections to the March Audit

The following March statements are withdrawn:

- "The community has observations; we have mechanisms." Recent work contains
  direct interventions, and several ModelCypher chains remain exploratory.
- "Zero-hyperparameter integration" as a field differentiator. Schedule-free,
  learning-rate-free, and spectral optimization are now crowded research areas.
- Global ID as the preferred phase observable. Current evidence favors local ID
  for several alignment and generalization questions.
- Any implication that CKA or total variance captures all behaviorally relevant
  structure. J-space and quantized-feature results are counterexamples to that
  measurement assumption.

Nothing in this scan disproves deterministic geometric analysis. It does make
the measurement operator, context distribution, null control, and intervention
non-optional parts of a promotable claim.

## Updated Closure Order

1. Finish release hygiene and make the CPU/JAX and static gates truthful.
2. Have the owner run `WS4.2` on real models using the published contextual
   curvature operator and the ID estimator comparison.
3. Add explicit context and measurement-operator identity to `A1` bundles.
4. Only if a new same-input behavioral failure survives the existing controls,
   test J-space transport as an `R2` candidate; keep it research-only until the
   intervention gate passes.
5. Add fixed-basis feature survival to `R4` quantization sweeps.
6. Expand `R1` controls to include applicable schedule-free and spectral
   competitors before making optimizer-position claims.

## Evidence Boundary

Anthropic's July report is a primary lab report with released reference code,
not an independent replication. Most January-June items above are arXiv
preprints. MIB is the peer-reviewed benchmark anchor, but it predates the scan
window. These sources change what ModelCypher must test; they do not make any
ModelCypher result validated.
