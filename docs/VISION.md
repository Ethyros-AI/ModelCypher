# Vision: Geometry as the Identity Layer

## The Trajectory

ModelCypher's training engine is the first destination, not the final one.

The long-term vision remains:

**personal, portable, sovereign AI identity carried as geometry, not data.**

The important correction is scope. That identity layer is not something the
repository can currently claim as operational. It is the downstream consequence
of promotably true geometry, not a narrative shortcut around unfinished
mechanism work.

## Scope Cascade

- **Mission**: close the canonical geometric engine, centered on
  `mc train run`.
- **Vision**: describe what mission success may eventually enable.
- **Roadmap**: define the closure order from current evidence to promotable
  claims.
- **Open Questions**: carry only the mathematical blockers on that closure
  order.

This file must stay downstream of:

- [MISSION.md](/Users/jasonkempf/ModelCypher/docs/MISSION.md)
- [RESEARCH-ROADMAP.md](/Users/jasonkempf/ModelCypher/docs/RESEARCH-ROADMAP.md)
- [OPEN-MATHEMATICAL-QUESTIONS.md](/Users/jasonkempf/ModelCypher/docs/research/OPEN-MATHEMATICAL-QUESTIONS.md)
- [FIRST_PRINCIPLES_REVIEW_PROTOCOL.md](/Users/jasonkempf/ModelCypher/docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md)

## Quantized First

The vision is quantized-first by design.

- `bf16/fp16` is the derivation regime.
- Quantized models are the deployment regime.
- If quantized behavior diverges from full precision, the response is operator
  tracing, not tolerance of unexplained damage.

The deployment story is still not "compress and accept loss." The target is
smaller-and-smarter behavior under measured geometric control.

## Hard Gates Before Identity-Layer Promotion

The identity-layer story is downstream of four gates. Until these are closed,
the vision stays directional rather than operational.

### Gate 1: Portable Cross-Architecture Certificate

We need a commensurable certificate that a personal adapter or transferred delta
preserves behavior across model families, not just probe alignment on one merge
pipeline.

Blocked by:

- [OPEN-MATHEMATICAL-QUESTIONS.md](/Users/jasonkempf/ModelCypher/docs/research/OPEN-MATHEMATICAL-QUESTIONS.md) `Q8`
- [RESEARCH-ROADMAP.md](/Users/jasonkempf/ModelCypher/docs/RESEARCH-ROADMAP.md) `R5`

### Gate 2: Stacking Preservation Certificate

Stacking must prove that multiple adapters can compose without silent drift in
the target model's preserved behavior.

This is not yet a shipped workflow. Today it is experimental infrastructure
without a promotable preservation certificate.

### Gate 3: Consolidation-Without-Forgetting Operator

Nightly consolidation is only promotable if there is a derived update operator
that adds new user structure while preserving old structure better than
meaningful continual-learning baselines.

Blocked by:

- [OPEN-MATHEMATICAL-QUESTIONS.md](/Users/jasonkempf/ModelCypher/docs/research/OPEN-MATHEMATICAL-QUESTIONS.md) `Q9`
- [RESEARCH-ROADMAP.md](/Users/jasonkempf/ModelCypher/docs/RESEARCH-ROADMAP.md) `R6`

### Gate 4: Sovereignty Infrastructure

Even after the geometry closes, sovereignty still requires serialization,
access-control, revocation, and user-owned runtime wiring. That is
infrastructure work, not yet a completed research claim.

## Why The Vision Still Holds

The repo already supports the direction of travel:

- the canonical training path removes runtime guesswork from the control plane,
- the merge work shows that activation-space geometry matters more than naive
  weight blending,
- the quantization work shows that full-precision derivation can improve
  low-precision outcomes instead of merely explaining failure after the fact.

What the repo does **not** yet support is talking as if portable identity is a
deployed capability. The correct statement is:

**the geometry-first engine is the prerequisite; the identity layer is the
destination after the certificates close.**

## Capability Scorecard

| Capability | Current status | Evidence | Promotion block |
| --- | --- | --- | --- |
| Geometry-derived training | `SHIPPED` on the canonical path | `mc train run`, `pipeline_gate_v1`, doctrine cleanup in runtime code | Behavioral preservation still fails in retained `pipeline_validation` trials |
| Quantized-first control | `PARTIAL` | `results/quantization_ab_survey/`, `results/closedform_sequential_correction/` | Frontier law is still open |
| Cross-architecture portability | `PARTIAL / EXPERIMENTAL` | `src/modelcypher/experimental/merge/`, `results/geometry_sota/analysis_summary.json` | No portable behavior certificate or MergeBench-style baseline closure |
| Nightly consolidation | `EXPERIMENTAL` | `src/modelcypher/experimental/continual/`, `src/modelcypher/experimental/use_cases/consolidation_service.py`, `results/continual_learning/` | Consolidation operator still open |
| Adapter stacking | `EXPERIMENTAL` | `src/modelcypher/experimental/self_improve/lora_stacker.py` | No preservation certificate |
| Adapter sovereignty | `NOT BUILT` | no user-owned runtime flow | infrastructure not built |

## What This File Does Not License

This vision file does not authorize:

- mixed-model "partial validation" language,
- user-facing portability claims from probe-only alignment,
- sovereignty claims from unbuilt infrastructure,
- stacking or consolidation claims without preservation certificates,
- treating experimental code as canonical just because it has a CLI entry
  point.

If a claim cannot survive the first-principles review protocol, it does not
belong in mission, vision, roadmap, or agent doctrine.

## Closure Order

The order stays strict:

1. baseline suite against standard practice
2. operator for behavioral failure when structural safety passes
3. 8B non-ceiling efficacy closure
4. quantization frontier law
5. portable adapter certificate
6. consolidation operator
7. stacking preservation certificate
8. sovereignty infrastructure

The identity-layer language becomes stronger only as those gates close in that
order.

## Bottom Line

ModelCypher is still building toward a world where the user owns the geometric
identity layer and can carry it across substrates. That remains the right
destination.

What changed is discipline: the repository should now talk about that future as
**downstream of closed certificates**, not as if the current experimental merge,
continual-learning, and stacking surfaces already deliver it.
