# ModelCypher Epistemic History

This file is the short map through the repo's experimental arc. Use the commit
anchors with `git show <hash>` when archaeology is necessary.

## Arc At A Glance

| Period | State | What changed | Commit and artifact anchors |
| --- | --- | --- | --- |
| 2025-12-19 | TrainingCypher port becomes ModelCypher. | The repo starts with the Python package skeleton, clean-architecture services, geometry commands, storage, merge, MCP, and dataset tooling ported into `src/modelcypher/`. The product identity is still broad: geometry, training, merging, safety, and agent tooling are all active surfaces. | `00827aab` (project skeleton), `ebcf3ecb` (service foundation), `024e1509` and `1c53746e` (geometry service and CLI), `949099f3` (training/safety/adapter tools) |
| 2025-12 to 2026-01 | SmolLM-360M and multi-donor compression era. | The repo runs many direct compression, bridge, transplant, and self-improvement experiments. The important product lesson is the ceiling: many scripts can improve a local metric, but the workflow does not yet produce a stable user-facing measurement system. The audit records this as the 284-script, roughly 70 percent ceiling period. | `b958cec4` (lossless compression attempt), `826d2638` through `9eea6ff8` (RMT, entropy, perturbation, adaptive-rank compression family), `docs/research/DIMENSIONAL_COMPRESSION.md`, `docs/research/FAILURE-MODES.md` |
| 2026-01-29 | Archival reset. | The repo stops treating the script pile as the product surface. It archives 284 research scripts with documented insights, consolidates speculative docs, and re-centers around retained evidence and smaller surfaces. | `c3874dc4` (archive 284 research scripts), `137adabc`, `95e89cf5`, `3656df31`, `d92eb382`, `362336db`, `f93375f9`, `55ede6b0`, `1415cad7`, `2b40c2f5` (doc consolidation rounds) |
| 2026-02-07 to 2026-02-23 | Hyperparameter-derivation thesis. | Training becomes the downstream demonstration of derived geometry. The work introduces geometric training code, stopping criteria, Lipschitz measurement, NB-LoRA/geometric LoRA, and early command coverage. The thesis is ambitious: rank, scale, stopping, and other controls should be measured from the object rather than hand-picked. | `4450f06a` (ScaledGD and measured Lipschitz), `1daebb84` (data-derived stopping), `472f6f59` (validated geometric training modules), `8c4a2015` (NB-LoRA), `655ec9ca` (Cayley pullback Lipschitz), `a50f935e` (LR derivation), `56830ce3` (geometric measurements) |
| 2026-02-25 to 2026-03-03 | Refutation wave. | Several attractive mechanisms fail quickly or are narrowed. K-FAC is added and removed on 2026-02-25. On 2026-02-26, five external-theory prediction families are refuted on trained networks. On 2026-03-03, Information Bridge keeps only the solid measurement pieces and kills or invalidates 5 of 8 pre-registered predictions. | `ff84d4ac`, `8d767d57`, `7f9b9dc6`, `ff5be1be`, then `49060514` (K-FAC same-day reversal); `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md`; `docs/research/REFUTATION-LEDGER.md`; `2c2ce94a`, `b6b6d9d4`, `be38f22b`, `ef2a6149`, `887c796a`, `59354049`, `3b1aba5a` (Information Bridge and review protocol work) |
| 2026-03-26 | Measurement-workbench pivot. | `mc analyze` becomes the clearest public entrypoint. Observation bundles, prompt-family studies, measurement atlas traces, and report read-side tooling become the product center. Training remains shipped but is downstream of trustworthy measurement. | `3a83b19e` (workflow-first observation bundles), `c1dcca12` (service DI), `26bffea8` (analyze report workflow), `389f7970` (measurement atlas tracing and artifacts), `docs/OBSERVATION-BUNDLES.md`, `docs/RESEARCH-ROADMAP.md` |
| 2026-04 | R1 no-go and pause. | The retained downstream training question does not clear its benchmark gate. `docs/RESEARCH-ROADMAP.md` records the current state: the Stage A frozen tuple is a no-go, the canonical path won 0/7 tasks against surface-matched baselines on the old benchmark pair, and seed expansion remains deferred. April work shifts to observation/read-side cleanup and compatibility fixes rather than declaring training victory. | `docs/RESEARCH-ROADMAP.md` R1 section; `ddabe008`, `304891e3`, `ee2cc649`, `6eeaf699`, `34c81ef7` (report and atlas read-side cleanup); `26cf8605`, `325a990c`, `8a2606d0` (runtime compatibility fixes) |

## Reading The Current Repo

The current product identity is not the early compression script pile and not a
claim that training has beaten standard practice. The stable center is:

1. Measure model behavior below token level with `mc analyze`.
2. Keep claims tied to mechanism, equation, measurement, and falsifier.
3. Treat `mc train run` as a shipped downstream surface whose canonical method
   is `geometric_lora`, with some components still evidence-gated.
4. Use negative results as first-class evidence, especially
   `docs/research/REFUTATION-LEDGER.md`.

The history matters because the repo learned its shape by deleting or narrowing
what did not survive measurement.
