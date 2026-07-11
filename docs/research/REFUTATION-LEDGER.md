# Unified Refutation Ledger

This is the single repo ledger for killed, narrowed, or invalidated research
claims. A row belongs here only when the repo has a mechanism-level reason the
claim failed, not just a worse-than-expected number.

Authoritative repo-retained count: **11 refuted or invalidated claim families**.
The external `/Volumes/CodeCypher/archive/modelcypher-scripts/refuted/` index is
not present in this checkout, so the 15 archived script-level refutations are
listed as an owner-copy action instead of being counted below.

| ID | Hypothesis | Date registered | Date killed | Kill mechanism | Artifact paths |
| --- | --- | --- | --- | --- | --- |
| RFL-001 | Semantic primes have higher cross-model embedding CKA than random words. | Paper 1 preprint claim, before 2025-12-25 | 2025-12-25; reproduced 2026-02-02 | Random-word controls matched or exceeded semantic-prime CKA; the observed cross-model similarity is general representation geometry, not semantic-primeness. | `papers/NEGATIVE-RESULTS.md`; `data/paper1/`; `experiments/paper1_collect.py` |
| RFL-002 | Information Bridge P2/P4/P6 can test Shannon-style information bottleneck predictions with the repo's Renyi alpha=2 estimator. | Before 2026-03-03 | 2026-03-03 | The estimator does not satisfy DPI, and residual-stream maps are injective enough that monotone Shannon MI decay is structurally unavailable. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md`; `results/information_bridge/` |
| RFL-003 | Information Bridge P5 is a cross-model law without an architecture or scale term. | Before 2026-03-03 | 2026-03-03 | Correct sigma calibration preserved a 350M/700M split; the wider model had narrower ID variation, so the equation was missing a scale-conditioned term. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md`; `results/information_bridge/` |
| RFL-004 | Mean-field alpha^2 chi classifies trained-model geometric phases. | Before 2026-02-26 | 2026-02-26 | Mean-field theory applies to random or initialized networks; after training, alpha and chi measure learned signal routing and no longer predict phase structure. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md` |
| RFL-005 | Marchenko-Pastur predicts trained attention spectral gaps from architecture parameters alone. | Before 2026-02-26 | 2026-02-26 | MP describes iid random matrices; post-softmax attention is row-stochastic, causal-masked, and learned, so the measured gap is driven by learned QK alignment and entropy. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md` |
| RFL-006 | L/d ratio governs trained-model ID trajectory similarity. | Before 2026-02-26 | 2026-02-26 | Width is not the bottleneck because measured ID stays far below d; depth and architecture family dominate trajectory shape. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md` |
| RFL-007 | Task type orders H1 loop persistence, with math prompts above narrative prompts. | Before 2026-02-26 | 2026-02-26 | Loop formation followed layer/ID geometry rather than task category; task-type ordering inverted in multiple tested models. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md` |
| RFL-008 | SPS f* derived from measured geometry makes SPS bind and improves alignment. | Before 2026-02-26 | 2026-02-26 | SPS was already binding near continuously at f*=0; positive f* only tightens an already-binding bound and did not meaningfully change CKA in the recorded rerun. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md` |
| RFL-009 | Attention/MLP curvature decomposition closes the entropy-to-curvature weak link. | Before 2026-02-26 | 2026-02-26 | The attention/MLP split was nearly constant across transformer families and did not explain the remaining curvature variance; family-dependent curvature-to-ID mapping remained. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md`; `scripts/curvature_accumulation_analysis.py` |
| RFL-010 | K-FAC projector and curvature-monitor path is promotable without surviving real-model go/no-go tests. | Before removal commit `49060514` | Removal commit `49060514` | The validation surface, projector, diagnostic, curvature monitor, and tests were removed together. The recovered script shows the required tests, but no retained quantitative output exists in this checkout. | `git show ff5be1be:scripts/kfac_validation.py`; `git show --stat 49060514`; owner-copy pending for `/Volumes/CodeCypher/experiments/kfac-validation` |
| RFL-011 | Geodesic RBF CKA depth-distance evidence is operator-independent. | Before 2026-03-03 | 2026-03-03 | Linear CKA removed the bandwidth degree of freedom and produced a 2/3 result; the geodesic result must be scoped to the geodesic RBF operator with calibrated sigma*. | `docs/research/OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md`; `docs/research/linear_accessible_information_derivation.md`; `results/information_bridge_linear_cka/` |

## Recovered K-FAC Refutation Note

Recovered source:

- `git show ff5be1be:scripts/kfac_validation.py`
- removal commit: `49060514 Refactor K-FAC related components and remove unused code`

The removed script defined three real-model go/no-go tests:

1. `Null(G_cap)` versus `Null(K_cap)` gain ratio.
2. K-FAC factor signal-space agreement versus a full behavior Jacobian.
3. Curvature alignment for Cayley+MASS training.

The script required MLX models under `/Volumes/CodeCypher/models/...` and wrote
outputs under `/Volumes/CodeCypher/experiments/kfac-validation`. The same
removal commit deleted `kfac_diagnostic.py`, `kfac_projector.py`,
`kfac_curvature_monitor.py`, K-FAC imports from merge/transplant code, and the
K-FAC unit tests. That operator-chain evidence is enough to record that K-FAC
was de-orbited from the product path; it is not enough to recover quantitative
pass/fail values.

Owner action: copy any retained `/Volumes/CodeCypher/experiments/kfac-validation`
reports into a tracked report path if the exact numerical negative result is
needed for citation. Until then, cite this entry as a recovered abandonment
record, not as a reproduced benchmark result.

## External Refuted Script Index

The audit references 15 archived refuted scripts under:

`/Volumes/CodeCypher/archive/modelcypher-scripts/refuted/`

That volume is intentionally unavailable in this Linux checkout. Do not invent
script names or outcomes. Owner action: copy or summarize those archived scripts
into this repository, then add one row per distinct killed hypothesis above.
