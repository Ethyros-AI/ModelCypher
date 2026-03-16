# Product-First Doctrine Rewrite — Design Spec

**Date:** 2026-03-16
**Status:** Implemented

---

## Context

ModelCypher has spent four months building a geometry-first training pipeline. The infrastructure is mechanically sound. The documentation reads like a research manifesto guarding theoretical claims. The benchmarks are not closed.

The results as of 2026-03-16:
- `mc train run` is real and functional — the training workbench exists
- All hyperparameters (rank, LR, batch size, stopping) are derived from model geometry — this part works
- Current shipped runtime uses PiSSA init; local single-seed 350M comparison favors PiSSA over the retained Cayley/NB-LoRA arm on 6 of 7 tasks (intra-project, not vs external baselines)
- Head-to-head against external standard LoRA/rsLoRA/DoRA baselines is still open — R1 seed expansion blocked
- The Cayley/NB-LoRA parameterization is not the current shipped path and performs worse on the one comparison we have
- `identity.py` already uses `method=geometric_lora` and `init_method=pissa` — the code is ahead of the docs
- The docs were rewritten from a research-manifesto framing to a product-first training-workbench framing

This rewrite reorients all doctrine and onboarding docs to reflect what the tool actually is: **a training workbench for open-source model builders** that derives hyperparameters automatically and makes `mc train run` useful without requiring backend knowledge or manual knob tuning.

---

## Product Mission (New)

> ModelCypher is a training workbench for open-source model builders. Point it at a model and dataset. Get a working adapter. Every training parameter — rank, learning rate, batch size, when to stop — is derived from your model's geometry. No guessing. No grid search. No MLX knowledge required.

Current honest state: `mc train run` works. Current shipped runtime: PiSSA init + geometry-derived params. Head-to-head benchmark comparison against external standard LoRA baselines still in progress — not yet demonstrably better.

The shipped workflow:
```
mc data prepare → mc model info → mc train run --plan-only → mc train run → mc train evaluate → mc train compare
```

---

## Files In Scope

### Primary: Doctrine Files (full rewrite)

| File | Change |
|------|--------|
| `docs/MISSION.md` | Rewrite to product mission ~100-150 lines. Keep derivation table. Remove prediction/documentation contracts and 7-guardrail bureaucracy as standalone gates. |
| `docs/VISION.md` | Rewrite lead to community-tool framing. Remove sovereignty/identity-layer lead. Retain portability/stacking as downstream possibilities without gate language. |
| `AGENTS.md` | Reorient from "research purity first" to "ship a working tool." Add: product friction = priority bug. Keep derivation guardrails in service of shipping a reliable CLI. Remove "link every experiment to an active blocker" as primary operating constraint. |
| `CLAUDE.md` | Shorten. Most of the existing content is already operational (model locations, volume check, GPU process check, gotchas) — keep all of that. Remove: the "project philosophy" section's research-manifesto tone. The operational sections are mostly correct. Add: "trust live CLI truth over session lore." |

### Secondary: Workflow Docs (targeted sweep)

| File | Change |
|------|--------|
| `docs/START-HERE.md` | Task-first flow for OSS trainers. Fix `mc model add <org>/<model-id>` example (live CLI uses local paths). Move paper-reading to "background reading" section. |
| `docs/TRAINING-GUIDE.md` | Rename shipped method consistently to "geometry-derived LoRA" (remove NB-LoRA). Expand guide to cover full workbench: plan, train, evaluate, compare, export. |
| `docs/CLI-REFERENCE.md` | Reorganize examples by user job. Surface `train evaluate`, `train compare`, `data prepare`, `model capacity` — currently underdocumented in onboarding. |

### Code Identity

This pass is **docs-only**. The one code-adjacent exception:

| Location | Change |
|----------|--------|
| `src/modelcypher/core/domain/training/identity.py` | Already correct (`geometric_lora`, `pissa`). Verify only — no changes needed. |
| User-facing log label strings | Completed in this pass for the scoped user-facing logger strings in `src/modelcypher/core/domain/lora_memory_store.py`. Limit remains: only CLI-visible string literals, no structural code changes. |
| Output directory naming templates | Out of scope. If `model-nblora-*` patterns appear in src/ templates, treat as a separate follow-up task to be explicitly approved. |

---

## Design Decisions

### 1. Mission Statement

**Before:** 528-line research manifesto with prediction contracts, documentation contracts, 7 guardrails with formal test conditions, bedrock mandate, and "what done looks like" in terms of research closure.

**After:** ~100-150 lines focused on:
1. What the tool does (zero-config training workbench)
2. What it derives (rank, LR, batch size, stopping — from geometry)
3. What the shipped workflow looks like (data → plan → train → evaluate → compare)
4. What works now (honest current state with workbench commands)
5. What still needs to close (benchmark comparison, 8B validation) — stated as present limitations, not blocking gates

The derivation table (15 hyperparameters → geometric replacements) stays. It's the clearest expression of what's different about this tool and is genuinely differentiating.

### 2. Vision Statement

**Before:** Leads with "geometry as the identity layer," sovereign AI, portable certificates, 4 hard gates before any identity-layer promotion.

**After:** Leads with the community value:
> ModelCypher becomes the easiest serious way for the open-source community to train, inspect, and validate adapters on open models.

Portability, stacking, consolidation remain as downstream future-work possibilities. They're no longer framed as "gates blocking the vision" — they're things that would extend the tool's usefulness once the core works.

### 3. Agent Operating Principle

**Before:** "Link every experiment to an active blocker. No linked blocker, no experiment."

**After:** "Build what makes users able to get working adapters faster. Product friction, stale docs, inaccurate examples, and broken workflows are priority bugs."

The derivation guardrails (no magic numbers, no heuristics, derive from geometry) stay — they're the mechanism that makes the tool reliable. They're reframed: the goal is a reliable CLI, not proving a research claim.

Replace the scope-control rule with: "Before starting any new research thread or experiment, state the user-facing improvement it enables and how it will be measured. Undirected exploration belongs in `scripts/`, not in canonical surfaces or result families."

### 4. Honesty Contract

One invariant preserved across every file:
> ModelCypher does not claim to beat standard LoRA until benchmarks close. The workbench exists and works. The quality proof is still being closed.

No softening of what's actually true. No adding claims that aren't yet earned. The pivot is in framing ("product in progress, measured honestly") not in inflating the current evidence.

### 5. Naming

Across all touched files:
- "NB-LoRA" as user-facing name for the shipped path → "geometry-derived LoRA"
- "canonical" where "current CLI" or "shipped workflow" is clearer → use the clearer term
- "research code, not product" → "product in progress, measured honestly"
- "promotable claim" → "shipped feature" or "validated capability"
- "gate closure" → "what still needs to close" or "next milestone"

NB-LoRA and Cayley remain valid as names for the specific parameterization, historical result family, or comparison arm — just not as the user-facing identity of the shipped tool.

---

## Workflow Doc Changes (Detail)

### START-HERE.md

Replace the "Reality Check" opening (which leads with defensive skepticism) with a task-first opener:

> You want to fine-tune a model. You have a dataset. You don't want to read papers or tune hyperparameters. This is the right tool.

Then: quick install → inspect model → prepare data → plan → train → evaluate.

Fix the broken example: live `mc model add` registers local paths only (e.g., `mc model add /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16`). Remove the `<org>/<model-id>` Hub format entirely — there is no automated Hub download workflow. If users need to download from Hub, they use `mlx_lm.convert` or `huggingface-cli download` first; document this as a prerequisite step, not a CLI feature.

Move Geometry Guide, Verification, and RESEARCH-ROADMAP links to a "how it works" or "background reading" section at the bottom.

### TRAINING-GUIDE.md

Expand from "how to run training" to "the full training workbench":
1. Prepare data (`mc data prepare`)
2. Inspect model capacity (`mc model info`, `mc model capacity`)
3. Derive the plan (`mc train run --plan-only`) — show example plan output
4. Run training (`mc train run`)
5. Evaluate the adapter (`mc train evaluate`)
6. Compare runs (`mc train compare`)
7. Export / merge (when applicable)

Consistent naming: "geometry-derived LoRA" throughout. Remove "NB-LoRA" from headings and descriptions.

### CLI-REFERENCE.md

**Add missing command sections first:**
- `mc data` — not currently in CLI-REFERENCE.md; implemented in `cli/commands/data.py`. Add a full section.
- `mc train evaluate` and `mc train compare` — not in CLI-REFERENCE.md; referenced in TRAINING-GUIDE.md and START-HERE.md only. Add sections.

Then reorganize. Current structure: command group by group. Target structure: user job by user job.

Jobs:
- "I want to fine-tune a model" → data prepare, train run
- "I want to understand my model" → analyze commands, model info
- "I want to evaluate my adapter" → train evaluate, train compare
- "I want to merge adapters" → merge run
- "I want to inspect what happened" → system probe, geometry analysis

---

## Test Plan

1. Verify every command example against live CLI help:
   - `poetry run mc --help`
   - `poetry run mc train run --help`
   - `poetry run mc train evaluate --help`
   - `poetry run mc train compare --help`
   - `poetry run mc data --help`
   - `poetry run mc model --help`

2. Grep touched docs for stale terms:
   - `NB-LoRA` as user-facing shipped method name
   - `identity layer` as current shipped feature
   - `research code, not product`
   - `promotable claim` (replace with "shipped feature" or "validated")
   - Inaccurate command signatures

3. Check all doc cross-references resolve to existing files.

4. Run token budget audit if rewrites stay above 20k tokens:
   ```bash
   poetry run python scripts/report_token_budget.py --threshold 20000
   ```

---

## Out of Scope

- CLI behavior changes (no new commands, no flag changes)
- Code refactoring beyond identity label cleanup in log output
- Rewriting non-touched docs (GEOMETRY-GUIDE, GLOSSARY, MATH-PRIMER, etc.)
- Removing experimental/ code (stacking, continual, merge) — still valuable research arms
- Making claims that haven't been earned (benchmark comparison still in progress)

---

## Assumptions

- Primary audience: open-source trainers working on local or modest hardware who do not want to learn backend internals
- Tone: plain English, product-first, first-principles rigor as implementation standard not customer identity
- Honesty rule: no superiority claims until benchmark comparison closes; current state stated accurately
- The geometry work is the implementation mechanism, not the customer-facing headline
