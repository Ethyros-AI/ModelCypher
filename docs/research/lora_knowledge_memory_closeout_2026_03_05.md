# LoRA Knowledge-Memory Closeout (2026-03-05)

## Scope

This note captures the March 5, 2026 review of Back et al. (2026),
arXiv:2603.01097, "Understanding LoRA as Knowledge Memory: An Empirical
Analysis," and the resulting ModelCypher research decision.

No new model experiments were run today.
This was a claim-audit and protocol-registration pass.

## What changed today

1. Reviewed the paper against the current ModelCypher code and research docs.
2. Corrected five overclaims in the initial mapping from paper phenomenon to
   ModelCypher mechanism.
3. Reframed the paper as an **empirical phenomenology source**, not a
   mechanistic validation.
4. Elevated the capacity-validation experiment to the primary next research
   target.
5. Registered a dedicated falsification protocol:
   `docs/research/LORA-KNOWLEDGE-MEMORY-CAPACITY-VALIDATION-PROTOCOL.md`

## The five corrections

### 1. Saturation is consistent with our mechanism, not validated by it

Back et al. measure saturation curves.
ModelCypher proposes a causal mechanism in terms of `tail_dims`, `sigma_k`, and
spectral safety.
The paper does not test that mechanism directly.

Correct status:
- paper observations are **not inconsistent with** the ModelCypher mechanism
- this is weaker than confirmation

### 2. Multi-LoRA has two regimes

The paper reports two different behaviors that should not be flattened into one:

- **Oracle / pure merge regime**: increasing `N` degrades performance because
  interference dominates and routing error is absent
- **Practical routing / recall regime**: `Top-3` can beat `Top-1` on multi-hop
  tasks because recall gains can exceed the interference cost

These are geometrically distinct:
- merge-time subspace collision
- routing-time coverage vs accuracy tradeoff

### 3. Synthetic-format results are richer than a single ordering

Q4 supports:
- QA is the strongest **single** format

Q5 adds:
- mixed supervision (`QA + Summary + Rewrite`) can beat QA alone

Correct read:
- QA is the best atomic supervision format
- mixtures can provide complementary signal and win overall

### 4. The paper's parameterization gap is narrower than originally stated

The paper does not test:
- norm-bounded parameterization
- null-space-aware parameterization

But it does test:
- DoRA
- PiSSA

Also, ModelCypher's previously observed `100x-2700x` scale violations came from
our own measured adapters and must not be projected onto the paper's Qwen/Llama
setups without measurement.

### 5. The merger mapping is aspirational, not current

Current `LoRAAdapterMerger` behavior is:
- Procrustes alignment
- CKA measurement
- averaging

The repo does contain the primitives needed to build a null-space-aware
interference check, especially in:
- `src/modelcypher/core/domain/geometry/null_space_accessibility.py`
- `src/modelcypher/core/domain/geometry/channel_projector.py`

But that check is not yet present in the current adapter-merging path.

Correct statement:
- we have the machinery to build it
- we do not yet have the integrated feature

## Research decision

Use this paper as a **falsification target**.

The right next question is not:
- "does the paper validate us?"

The right next question is:
- "when we recreate the paper's conditions with geometry-derived training, do
  the observed effects collapse onto `tail_dims`, spectral scale bounds,
  subspace overlap, and Cayley parameterization?"

## Primary follow-up

Primary artifact created today:
- `docs/research/LORA-KNOWLEDGE-MEMORY-CAPACITY-VALIDATION-PROTOCOL.md`

That protocol pre-registers:
- saturation prediction from utilized tail capacity
- efficiency-curve shift under spectral safety
- merge-loss prediction from subspace collision
- capacity-per-parameter shift under NB-LoRA
- the two-regime `N` law for merge vs routing

## Next step when work resumes

Start Area 1-2 only:

1. reproduce one paper-style baseline run surface
2. emit geometry and scale tables for the same run
3. test spectral safety before adding full NB-LoRA and routing decomposition

This keeps Area 3 primary and prevents architecture changes from outrunning
measurement.
