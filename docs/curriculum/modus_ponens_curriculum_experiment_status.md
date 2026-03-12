# Modus Ponens Curriculum Experiment Status

> Purpose: single-source handoff for the March 12, 2026 curriculum experiment and measurement cleanup.
> Read this file first before continuing the `modus_ponens` curriculum experiment family.

## Linkage

This work is preparatory evidence for open experiment #3 in `docs/curriculum/skill_dag.md`:

- "Does logic-first curriculum outperform math-only on `word_problem_multi`?"

The current `modus_ponens` run is not that final comparison. It is a smaller end-to-end certificate that:

1. the curriculum protocol can generate and ingest curricula safely,
2. the experiment runner can train and evaluate without infrastructure artifacts,
3. the measurement operator is strict enough to support future promotable claims.

## Current State

### Landed code fixes

Curriculum protocol hardening:

- external mastered prerequisites are stripped before `SkillDAG` construction
- path traversal is blocked during curriculum ingest
- duplicate dataset filenames are rejected as a hard validation error
- empty in-curriculum prerequisite lists no longer crash `_depth()`

Experiment/evaluation pipeline hardening:

- `scripts/curriculum_experiment.py` now merges all eval shards for baseline eval, post-training eval, and `mc train run --eval-data`
- `mc train run --json` output is parsed as an `AgentEnvelope`
- adapter resolution is stable across `result.adapter_path`, `metadata.adapter_path`, and default output path
- sample inference uses the same prompt/expected extraction logic as the evaluator
- geometric adapter loading supports repo-used filenames:
  `adapters.safetensors`, `adapter_model.safetensors`, `adapter.safetensors`,
  `lora_weights.safetensors`, `adapters.bin`, `adapter_model.bin`, `adapter_model.pt`
- `.safetensors` files are loaded with `load_safetensors()`, not binary weight loaders
- mastery eval now fails fast on systematic inference infrastructure failures instead of silently turning them into fake wrong answers

Measurement operator cleanup:

- `exact` mode now scores only the first-line answer span
- explanation-only matches are tracked as diagnostics and are not counted as correct
- `scripts/analyze_curriculum_results.py` now mirrors that same contract

### Added coverage

The following targeted tests were added for this workstream:

- `tests/test_curriculum_protocol.py`
- `tests/scripts/test_curriculum_experiment.py`
- `tests/adapters/test_inference_engine.py`
- `tests/adapters/test_curriculum_eval_adapter.py`

Current reported verification status:

- curriculum protocol tests: `44/44` passing
- evaluator / adapter loader / experiment-script tests: `26` passing across the three new focused files

## Trusted Empirical Status

### What is trustworthy

The original end-to-end `modus_ponens` run on `LFM2-350M-MLX-bf16` with the saved adapter is real training, not a fake adapter-load failure.

Observed training diagnostics from the run:

- baseline exact accuracy under the old operator: `45/100`
- post-training exact accuracy under the old operator: `56/100`
- train loss: `4.197 -> 1.029`
- post-validation loss: `3.716`
- CKA: `0.899`
- pipeline gate: passed

These numbers proved the training path worked, but the exact-match measurement operator was too permissive because it credited expected strings that appeared later in explanation text.

### What is no longer promotable without rerun

The `45 -> 56` score should be treated as superseded by the stricter measurement contract.

Manual audit of all 19 discordant items found:

- `12` clean direct-answer gains
- `3` explanation-only gains
- `3` genuine regressions
- `1` measurement artifact previously counted as a regression

The key artifact was item `70`: the baseline first response was wrong, but the old substring metric credited a later explanation phrase as correct.

Conservative interpretation:

- the paired effect is still positive
- the old exact-match numbers are not the final certified numbers
- the corrected baseline/adapter accuracies require a real GPU-backed rerun under the strict first-line operator

## Artifacts

External run artifacts currently referenced by this work:

- adapter: `/Volumes/CodeCypher/experiments/curriculum_e2e_v3/modus_ponens_adapter`
- first flip analysis JSON: `/Volumes/CodeCypher/experiments/curriculum_e2e_v3/modus_ponens_flip_analysis.json`
- next strict-operator output target:
  `/Volumes/CodeCypher/experiments/curriculum_e2e_v3/modus_ponens_flip_analysis_v2.json`

## Exact Next Step

Do not start this while another GPU job is active.

GPU safety check:

```bash
pgrep -af 'python|mlx' | grep -v grep
```

If the GPU is free, rerun the item-level analysis with the strict operator:

```bash
poetry run python scripts/analyze_curriculum_results.py \
    --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
    --adapter /Volumes/CodeCypher/experiments/curriculum_e2e_v3/modus_ponens_adapter \
    --eval-data data/eval/modus_ponens_eval.jsonl \
    --output /Volumes/CodeCypher/experiments/curriculum_e2e_v3/modus_ponens_flip_analysis_v2.json
```

## What To Check After Rerun

Inspect these summary fields first:

- `baseline_accuracy`
- `adapter_accuracy`
- `accuracy_delta`
- `flips.wrong_to_right`
- `flips.right_to_wrong`
- `baseline_explanation_only`
- `adapter_explanation_only`
- `miss_classification.explanation_only`

Expected qualitative outcome:

- baseline should drop by at least the known explanation-artifact item
- adapter should also drop because several prior gains were explanation-only
- the net paired effect should remain positive if the manual audit is directionally correct

## Files To Open First Tomorrow

- `docs/curriculum/modus_ponens_curriculum_experiment_status.md`
- `scripts/analyze_curriculum_results.py`
- `src/modelcypher/adapters/curriculum_eval_adapter.py`
- `scripts/curriculum_experiment.py`
- `tests/adapters/test_curriculum_eval_adapter.py`
- `tests/adapters/test_inference_engine.py`
- `tests/scripts/test_curriculum_experiment.py`

## Stop Condition For This Thread

This thread is ready to advance once the strict-operator rerun completes and the corrected paired table is written to `modus_ponens_flip_analysis_v2.json`.

After that, the next decision is whether the corrected `modus_ponens` result is strong enough to justify the larger logic-first vs math-only comparison for `word_problem_multi`.
