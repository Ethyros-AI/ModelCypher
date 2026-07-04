# CKA Decomposition

Retained family status: `summary_only`

## What This Bundle Keeps

- Analysis narrative:
  `results/cka_decomposition/ANALYSIS.md`
- Structured decomposition output:
  `results/cka_decomposition/cka_decomposition_results.json`

This family keeps the retained decomposition summary only. The raw adapter
directory and runner log are deleted.

## Retained Measurements

The retained decomposition output reports:

- model: `LFM2-350M-MLX-bf16`
- seed: `4231027559`
- probes: `181`
- layers analyzed: `16`
- worst CKA layer: `15` with `CKA = 0.9392586180636148`
- largest gram perturbation ratio: layer `15` with
  `epsilon = 0.35611799101201935`
- corresponding theoretical lower bound at layer `15`:
  `0.4747979255901442`
- largest measured budget usage among LoRA layers: layer `14` with
  `budget_usage = 0.4747119673867723`
- null-energy fraction at that layer: `0.6293562331742952`

## Historical Provenance

The retained summary files record the original adapter path:

- `results/cka_decomposition/adapter`

That adapter directory is now deleted under the retained-summary cleanup rule.
The current script already supports `--adapter-path`, so future reruns do not
depend on this historical path.

## Deleted Raw Artifacts

- `results/cka_decomposition/adapter`
- `results/cka_decomposition/run.log`

The deleted adapter payload was one `.safetensors` file of about `38.56 MB`
plus its matching config and geometry manifest. The retained JSON and markdown
capture the decomposition result without keeping the adapter dump in the
worktree.
