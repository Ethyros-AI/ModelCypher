# Tangent-Subspace ID Falsifier Protocol

Date: 2026-03-08
Status: Pre-registered rerun protocol
Scope: Repaired rerun of `scripts/tangent_subspace_id_mechanism.py`

## Claim Under Test

TwoNN intrinsic-dimension trajectory changes are caused by layer-to-layer changes
in tangent geometry, specifically:

1. rotation within the shared tangent subspace,
2. added off-span tangent directions, and/or
3. local tangent misalignment between consecutive stages.

This protocol replaces the 2026-03-07 hand-written 60-prompt run as the
promotable measurement path. The historical run remains an exploratory artifact
only.

## Prediction Contract

Per mission contract:

```text
observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)
```

Instantiated observables:

```text
g_shared(l) = Grassmann_geodesic(T_l^shared, T_{l+1}^shared)
a_added(l) = ||(I - P_small) V_large^T||_F^2
c_added(l) = count(residual_norm > sqrt(eps))
theta_local(l) = mean principal angle between local tangent bases
delta_id(l) = ID_{l+1} - ID_l
```

with

```text
g_shared = f1(
    geometry_state = {stage activations, PCA tangent bases},
    architecture_state = {residual topology, core operator family},
    scale_state = {d_model, n_layers, n_params, probe_count, tangent_rank},
    precision_state = {bf16 model weights, fp32 exported activations, numpy SVD},
    measurement_operator = matched-rank Grassmann distance on consecutive stages
)

a_added, c_added = f2(
    geometry_state = {larger tangent basis projected onto smaller span},
    architecture_state = {residual topology, core operator family},
    scale_state = {k_l, k_{l+1}, probe_count},
    precision_state = {fp32 basis vectors, sqrt(eps) residual floor},
    measurement_operator = asymmetric projection residual
)

theta_local = f3(
    geometry_state = {local tangent bases around shared anchors},
    architecture_state = {core operator family},
    scale_state = {probe_count, neighbor_count, tangent_rank, coverage},
    precision_state = {backend dtype, tangent SVD precision},
    measurement_operator = TangentSpaceAlignment on consecutive stages
)
```

## Directional Predictions

1. If shared subspace rotation is causal, `sign(corr(g_shared, |delta_id|)) >= 0`
   within each resolvable model family.
2. If added off-span directions matter when ID grows, `a_added` and/or `c_added`
   increase on positive-`delta_id` transitions.
3. If local tangent misalignment is causal, `sign(corr(theta_local, |delta_id|)) >= 0`
   within each resolvable model family.

These are directional predictions only. No literal correlation cutoff is promotable
unless derived separately from geometry or uncertainty analysis.

## Measurement Operator And Commensurability

1. Use one frozen atlas-backed probe manifest for every compared model in the run.
2. Use the same stage definition for every model:
   embedding output at stage 0, then post-layer last-token hidden states.
3. Save `anchor_count`, `neighbor_count`, `tangent_rank`, and `coverage` for every
   local-tangent layer pair.
4. Treat local-rank telemetry from Euclidean `KDTree` neighborhoods as
   `[MEASUREMENT_INVALID]` for TwoNN causal adjudication until a geodesic operator is derived.
5. Use machine-precision floors only:
   `sqrt(eps)` is the only allowed binary floor for added-direction counts.

## Probe Manifest And Model Set

Promotable reruns use atlas-backed probes, not the historical hand-written prompt list.

Default rerun set:
- `LFM2-350M`
- `Qwen3.5-0.8B`
- `Llama-3.2-3B`

Derived probe budget:
- `neighbor_count = floor(sqrt(N))`
- `tangent_rank = floor(neighbor_count / 2)`
- Historical Llama non-stage-0 TwoNN peak is `max ID = 8.992...`, so
  `ceil(max ID) = 9`
- Therefore first acceptable shared probe count under the current local operator is
  `N = (2 * 9)^2 = 324`

The same frozen 324-probe manifest is used for all 3 models in the rerun.

4-bit `Mistral-7B` is not a primary adjudication model for mechanism discovery in
this protocol. Promotion on standard transformers remains blocked until another
bf16 pure-attention family is available.

## Falsifiers

F1 (added-direction floor falsifier):
- If `c_added = 0` and `a_added <= sqrt(eps)` for every measured layer pair across
  every model, added off-span directions are falsified on the measured domain.

F2 (local-angle sign falsifier):
- If any resolvable model family has `corr(theta_local, |delta_id|) < 0`, the local
  tangent-misalignment sign law is falsified for the declared domain.

F3 (measurement-validity falsifier):
- If compared models do not share the same probe manifest, or if local-rank
  adjudication still uses Euclidean neighborhoods, classify the affected claim as
  `[MEASUREMENT_INVALID]`, not refuted.

F4 (promotion block):
- Even if repaired reruns show positive directional evidence, keep status
  `[EXPLORATORY]` until a second bf16 pure-attention family is available.

## Required Artifacts

Each rerun must write:

1. `results/tangent_subspace_id_mechanism/<run_id>/results.json`
2. `results/tangent_subspace_id_mechanism/<run_id>/falsifier_outcome.json`
3. `results/tangent_subspace_id_mechanism/<run_id>/probe_manifest.json`

The historical 2026-03-07 artifact at `results/tangent_subspace_id_mechanism/results.json`
is retained only as a baseline reference for the derived 324-probe budget.

## Promotion Rule

Promotion beyond `[EXPLORATORY]` requires all of:

1. repaired rerun completed with atlas-backed frozen manifest,
2. no triggered measurement-validity falsifier,
3. no sign falsifier triggered for the promoted observable,
4. explicit architecture/scale/precision terms preserved in the final report,
5. second bf16 pure-attention family added for cross-family adjudication.
