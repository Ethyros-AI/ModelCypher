# REINFORCE Unlock Cycle U1 Closeout (2026-03-06)

## Scope

This note captures the March 6, 2026 closeout status for the fresh U1 rerun on
the current 1.2B frontier:

- model: `LFM2.5-1.2B-Instruct-bf16`
- train set: `data/training/1p2b_reasoning_foundation_train.jsonl`
- eval set: `data/training/1p2b_reasoning_foundation_val.jsonl`
- retention set: `data/training/retention_replay.jsonl`
- output root: `results/reinforce_unlock_cycle_u1_lfm25`

The active run was launched on March 5, 2026 and remains in progress at this
closeout checkpoint.

## What changed today

1. Ran the targeted preflight checks before launching training:
   - `poetry run pytest tests/test_reinforce_revalidation.py` -> `2 passed`
   - `poetry run pytest tests/test_mlx_training_adapter_strict.py -k "post_outcome or lost_only"` -> `2 passed`
   - The planned dataset-service `-k` expression matched zero tests, so the
     corresponding research-control coverage was verified with the exact test
     names:
     `poetry run pytest tests/test_dataset_training_service_strict.py -k "passes_research_controls_to_train_loop or research_controls_validate_values"` -> `2 passed`
2. Launched the pre-registered U1 runner with the current frontier model and
   explicit dataset paths rather than relying on the stale default model path.
3. Reviewed the orchestrator after code-review findings and patched the U1
   script so the experiment logic matches the intended interpretation.
4. Stopped the first in-memory U1 process and restarted the cycle with
   `--skip-existing` so completed artifacts were preserved while H1/H2 reran
   under the corrected bootstrap logic.

## Script fixes applied

File changed:
- `scripts/reinforce_unlock_cycle_u1.py`

Fixes applied:
- Corrected `DEFAULT_MODEL` to
  `/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16`
- Replaced `n_bootstrap = n*n` with a documented floor:
  `n_bootstrap = max(1000, n*n)`
- Added explicit bootstrap constants:
  `BOOTSTRAP_MIN_REPLICATES = 1000`
  `BOOTSTRAP_SEED = 20260224`
- Documented that H1 and H2 are paired treatment-vs-treatment comparisons
- Documented that E2 omits `ce_control` by design and uses a direct paired
  comparison between `lost_only` and `all`

## Active run status

The current active process at closeout is:

```bash
poetry run python scripts/reinforce_unlock_cycle_u1.py \
  --model-path /Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16 \
  --train-data data/training/1p2b_reasoning_foundation_train.jsonl \
  --eval-data data/training/1p2b_reasoning_foundation_val.jsonl \
  --retention-data data/training/retention_replay.jsonl \
  --output-root results/reinforce_unlock_cycle_u1_lfm25 \
  --seeds 41,42,43,44,45 \
  --regime-n 100 \
  --online-eval-n 100 \
  --eval-interval 10 \
  --skip-existing
```

Observed process state:
- PID: `21774`
- status at capture: running

Completed or live seed state observed at closeout:

- `e1_gate_stage/ce_control/seed41`
  - complete
  - final canonical online eval: `86/100`
  - stop reason:
    `degeneration_exceeded (max_ngram(2)=0.434 > baseline=0.269+eps, epoch=0)`
  - gate confound events: `0`
- `e1_gate_stage/ce_control/seed42`
  - complete
  - final canonical online eval: `72/100`
  - stop reason:
    `degeneration_exceeded (max_ngram(2)=0.538 > baseline=0.528+eps, epoch=1)`
  - gate confound events: `0`
- `e1_gate_stage/ce_control/seed43`
  - in progress at last inspection
  - latest visible online eval in `train.log`: `74/100` at epoch 1
  - latest visible degeneration check remained below the recorded baseline
    threshold (`max_ngram(2)=0.333`, `baseline_max=0.492`)

No phase-decision closeout artifacts are available yet:

- `results/reinforce_unlock_cycle_u1_lfm25/e1_gate_stage/h1_decision.json`
- `results/reinforce_unlock_cycle_u1_lfm25/e2_credit_targeting/h2_decision.json`
- `docs/research/reports/reinforce_unlock_cycle_u1_lfm25/e3_unlock_confirmation/REPORT.md`
- `docs/research/reports/reinforce_unlock_cycle_u1_lfm25/REPORT.md`

Those will only be interpretable after the cycle completes.

## Guardrails respected

- No tests were run after the training job started.
- No alternate aggregation or simulated result path was used.
- The rerun remains the authoritative path for U1 because the prior frontier
  aggregate mixed seed vintages across February 23, 2026 and March 5, 2026.

## Next step when work resumes

1. Let the U1 cycle finish under the patched bootstrap logic.
2. Read the artifacts in order:
   - `results/reinforce_unlock_cycle_u1_lfm25/e1_gate_stage/h1_decision.json`
   - `results/reinforce_unlock_cycle_u1_lfm25/e2_credit_targeting/h2_decision.json`
   - `docs/research/reports/reinforce_unlock_cycle_u1_lfm25/e3_unlock_confirmation/REPORT.md`
   - `docs/research/reports/reinforce_unlock_cycle_u1_lfm25/REPORT.md`
3. Inspect the winning force-arm `train.log` files for:
   - `REINFORCE vs CE cosine`
   - `orth_frac`
   - `REINFORCE budget`
   - `budget_exhausted`
   - `REINFORCE ROLLBACK`
   - `Gate confound events`
4. If E3 closes as `CEILING` with zero gate confounds, move next to a
   same-model parameterization falsifier, not a DPO reformulation.
