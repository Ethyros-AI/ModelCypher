# Dynamic Training Control Law Recovery

## Status

`[EXPLORATORY]`

This note is the canonical contract for controller-derivation work during the
MASS recovery phase. It does not promote a new controller. It defines the
object that must be derived before any such promotion is allowed.

## Missing Object

The missing object is not a better static default. It is a closed-loop control
law:

```text
u_{l,t} = G(s_{l,t}, architecture_state, scale_state, precision_state, optimizer_state_t)
```

Where:

- `u_{l,t}` is the dynamic control applied to layer `l` at step `t`
- `s_{l,t}` is the measured state vector proximal to behavioral failure
- `architecture_state` captures operator-path differences across model families
- `scale_state` captures parameter-count / width / depth regime
- `precision_state` captures dtype and quantization terms
- `optimizer_state_t` captures moment / preconditioner state when present

Promotable claims must still use:

```text
observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)
```

## Separation Of Roles

### Static geometry configuration

These are derived before training and are not themselves the dynamic control
law:

- target layers
- initial rank ceilings
- spectral scale bounds
- precision ceilings

### Dynamic control variables

These are the variables the controller may emit over training:

- per-layer learning-rate multipliers
- total-step budget multipliers
- weight-decay scaling
- stop / freeze decisions

Dynamic rank growth and mid-run target-module changes remain out of scope for
this recovery phase.

### Behavioral state variables

These are the measured quantities the controller is trying to regulate:

- total effective step norm
- per-layer parameter-update norm
- per-layer behavioral transport norm `||X_l ΔW_l^T||`
- spectral budget ratio and remaining budget
- online-eval correctness deltas
- decision-boundary margin transport
- null accessibility / observability when measured on retained probes
- CKA-blindness quantities when measured on retained probes
- optimizer-state summaries

## Overlooked Assumptions

The current controller work must not assume:

- structural safety implies behavioral safety
- one global scalar controller is sufficient across layers and model families
- CE-only step norms adequately represent total update magnitude
- optimizer state is incidental
- phase / regime is stationary over training
- Euclidean or representational proxies are automatically commensurable with behavior
- precision enters only as a floor rather than a dynamical term

If any analysis relies on one of those assumptions without measuring the
corresponding term, the claim remains exploratory.

## Claim States

Only these claim states are allowed for this program:

- `[EXPLORATORY]`
- `[MEASUREMENT_INVALID]`
- `[MECHANISM_UNDERSPECIFIED]`
- `[VALIDATED]`

`[VALIDATED]` is reserved for a controller law that:

1. predicts failure before online-eval degradation,
2. improves known failing seeds without regressing structural gates,
3. remains expressible in the full contract above, and
4. survives the first-principles review protocol.

## Artifact Contract

Every research run in this program must preserve:

- machine-readable per-epoch / per-step controller trace
- machine-readable offline replay of emitted controller decisions
- frozen probe source for any retained behavioral measurements
- explicit controller mode and optimizer research mode

No interpretation strings belong in the telemetry surface. Artifacts carry raw
measurements only.

## Current Implementation Scope

The current codebase now supports research-only modes:

- `mass_structural_observe`
- `mass_behavioral_probe`
- `mass_behavioral_closed_loop` (reserved; fail-closed)
- `cayley_stiefel_mass`
- `adamw_matched_trace`

The default CLI behavior remains unchanged. These modes exist to derive the
controller, not to claim a solved controller.
