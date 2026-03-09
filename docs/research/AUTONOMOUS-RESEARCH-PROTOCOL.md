# Autonomous Research Protocol

**Status:** Canonical operating contract for repeated agent-driven experiment loops
**Effective date:** 2026-03-09
**Applies to:** `scripts/`, experimental runs under `results/`, agent sessions, and any repeated research iteration where code changes are evaluated against fixed observables

---

## 1. Purpose

This protocol adapts the strongest operational idea from
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch): treat the
research loop itself as programmable, constrained, and auditable.

For ModelCypher, the transfer is not "let the agent optimize one scalar
overnight." The transfer is:

1. freeze the measurement surface,
2. constrain the mutable surface,
3. log every attempt,
4. advance only on measured mechanism-preserving progress.

This protocol exists so autonomous iteration does not collapse into heuristic
thrashing, mixed-model narrative, or undocumented branch drift.

The governing rule is:

**no linked blocker, no canonical experiment.**

Every repeated run family must be linked to exactly one active roadmap item in
`docs/RESEARCH-ROADMAP.md` or one active question in
`docs/research/OPEN-MATHEMATICAL-QUESTIONS.md`.

If it cannot declare that linkage, it is parked exploration only. It may not
create a new canonical script, result family, or promotable claim.

---

## 2. Run Charter (Required Before First Edit)

Before any repeated experiment loop begins, write a run charter in the run doc
or manifest. It must include:

1. linked blocker ID (`R#` or `Q#`)
2. `run_id` or branch tag
3. claim contract:

```text
observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)
```

4. primary observable to optimize or falsify
5. explicit falsifier
6. mutable surface
7. frozen surfaces
8. baseline command
9. comparison budget
10. artifact directory
11. ledger path

If any field is missing, the loop is exploratory only and may not promote a
claim.

---

## 3. Mutable vs Frozen Surface

Autonomous research only works when scope is explicit.

### Mutable Surface

Per run family, change exactly one of:

- one script
- one module cluster serving one mechanism
- one protocol document plus its paired script

If more than one surface must change, declare a new run family and re-baseline.

### Frozen Surfaces

The following must be fixed for the whole run family unless the charter says the
run is specifically about them:

- evaluation command
- probe inventory or dataset slice
- model family and precision regime
- pass/fail operator
- artifact schema
- time, token, or step budget used for comparison

`prepare.py` is immutable in `autoresearch`. The ModelCypher equivalent is:
the evaluator, probe manifest, and artifact validator are immutable unless they
are themselves the object under test.

---

## 4. Baseline First

The first run is always the untouched baseline for the declared charter.

Record:

- exact command
- commit SHA
- model identifiers
- precision state
- raw observables
- artifact path

No "improvement" claim is valid without a baseline row in the same ledger and
under the same frozen surfaces.

---

## 5. One Variable Per Loop

ModelCypher already requires "one variable per day." Autonomous loops must make
that operational:

1. one causal idea per run
2. one declared reason the observable should change
3. one next falsifier if it fails

Bundled changes are allowed only when the theorem requires the bundle as one
operator.

Wrong:

- "try a few cleanup tweaks"
- "increase rank, swap probes, and change stopping"

Right:

- "replace the stopping observable with the derived spectral certificate"
- "swap the shared-sigma calibration operator and keep all other surfaces fixed"

---

## 6. Budget Contract

`autoresearch` uses a fixed 5-minute wall-clock budget. ModelCypher must use the
same principle, but the budget is architecture-conditioned and resource-aware.

For each run family, freeze one comparison budget:

- wall-clock time
- optimizer steps
- tokens processed
- evaluation probe count

Comparisons are valid only within a shared budget regime.

If the budget changes, start a new baseline.

Before any model-loading run, obey the repository concurrency guard:

```bash
pgrep -af 'python|mlx' | grep -v grep
```

If GPU-using processes are active, stop and ask the user before proceeding.

---

## 7. Append-Only Ledger

Every run in the family must append one row to a single ledger file. Do not
retroactively rewrite failures away.

Use a TSV or JSONL ledger that records at minimum:

- `run_id`
- `timestamp_utc`
- `commit`
- `status`
- `claim`
- `mutable_surface`
- `frozen_surfaces`
- `command`
- `primary_observable`
- `artifact_dir`
- `next_falsifier`

Status must be one of:

- `advance`
- `discard`
- `crash`
- `measurement_invalid`

Use [AUTONOMOUS_RESEARCH_LEDGER_TEMPLATE.tsv](AUTONOMOUS_RESEARCH_LEDGER_TEMPLATE.tsv)
as the minimum TSV header.

The ledger is an index into the artifact bundle, not a replacement for it.

---

## 8. Decision Rule

`autoresearch` advances on "metric improved." ModelCypher is stricter.

Advance a branch or preserve a run only if all are true:

1. the primary observable moved in the predicted direction or the falsifier was
   the point of the run,
2. no guardrail in `AGENTS.md` was violated,
3. the measurement operator remained valid,
4. the result is simpler or more explanatory than the previous state,
5. the artifact bundle is complete enough to audit.

Discard when:

- the observable worsened,
- the effect was mixed and not pre-registered,
- the measurement operator saturated or degenerated,
- the change adds complexity without mechanism gain.

Classify as `measurement_invalid` when the operator, calibration, or scale
regime broke comparability. Do not call this partial success.

---

## 9. Artifact Bundle

Each run must emit or link:

1. run charter or manifest
2. exact command
3. raw results
4. summary metrics
5. decision record
6. next falsifier

Where a validator exists, run it on the emitted artifacts before promotion.

If the run family is being promoted or reactivated as `canonical`, the family
must also retain:

1. `REPORT.md`
2. a machine-readable summary JSON
3. the manifest or charter
4. the append-only ledger

Historical families do not require blanket backfill. This rule applies at the
point of canonical promotion or reactivation.

---

## 10. Human and Agent Roles

The strongest `autoresearch` insight is that instruction files are part of the
research apparatus. In ModelCypher:

- humans define the charter and doctrine,
- agents execute within the charter,
- artifacts decide advancement,
- doctrine updates only after protocol-complete review.

Agent autonomy is allowed inside the loop, but not across these boundaries:

1. no evaluator rewrites without a new charter
2. no mixed-model promotion without commensurability proof
3. no hidden threshold invention
4. no silent deletion of failed runs
5. no creation of canonical work that is not linked to an active blocker

Use `results/repo_research_inventory/` as the triage source of truth:

- `canonical` stays live only while tied to an active blocker
- `summary_only` stays dormant unless explicitly reactivated
- `delete` stays off-limits unless a human explicitly reopens the thread

---

## 11. Recommended Loop

1. Write the charter.
2. Run the baseline.
3. Append the baseline ledger row.
4. Change one mutable surface.
5. Run the fixed command under the frozen budget.
6. Validate artifacts.
7. Append a ledger row with `advance`, `discard`, `crash`, or
   `measurement_invalid`.
8. Write the next falsifier before the next edit.

The loop may continue autonomously only while the charter remains valid.
If the mechanism hypothesis changes, stop, rewrite the charter, and re-baseline.
