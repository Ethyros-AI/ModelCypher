# Observation Bundles

`mc analyze` is the workflow-first surface for measuring what a model is doing
below token level. The public contract has two stable file interfaces:

- `PromptFamilyManifest`
- `ObservationBundle`

This file explains both with concrete examples.

## When To Use Each Workflow

Use `capture` when you want one prompt or a prompt file turned into a bundle:

```bash
poetry run mc analyze capture \
  --model /path/to/model \
  --prompt "Explain geodesics."
```

Use `family` when you want controlled perturbation studies inside one target:

```bash
poetry run mc analyze family \
  --model /path/to/model \
  --manifest data/probes/prompt_family_casing_tone.json
```

Use `compare` when you want the same prompt family run against two targets:

```bash
poetry run mc analyze compare \
  --left-model /path/to/base \
  --right-model /path/to/base \
  --right-adapter /path/to/adapter \
  --manifest data/probes/prompt_family_formatting.json
```

Default output goes to `results/analysis/<timestamp-slug>/`. Use `--output` to
override the bundle location.

Use `report` when you already have a bundle and want the shared high-signal
view:

```bash
poetry run mc analyze report --bundle /path/to/bundle
```

## `PromptFamilyManifest`

Phase 1 keeps this interface explicit rather than transform-driven. Each row is
a concrete prompt variant.

Top-level fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `schema` | no | Optional schema id. Accepted values: `mc.analyze.prompt_family.v1` and `mc.analyze.prompt_family.v2` |
| `name` | no | Human-readable study name |
| `metadata` | no | Extra descriptive context |
| `variants` | yes | Flat list of prompt rows |

Variant row fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `case_id` | yes | Groups variants that should be compared against each other |
| `variant_id` | yes | Stable name for one variant inside a case |
| `text` | yes | Exact prompt text to run |
| `tags` | no | Labels such as `caps`, `markdown`, `profanity`, `persona` |
| `comparison_to` | no | Explicit baseline variant id for this row |
| `annotations` | no | Research-only metadata such as `study_role`, `perturbation_type`, `expected_label`, `allowed_label_aliases`, `reference_answer`, and `notes` |

### Example: Casing And Tone

```json
{
  "schema": "mc.analyze.prompt_family.v1",
  "name": "casing_tone",
  "variants": [
    {
      "case_id": "reasoning_request",
      "variant_id": "control",
      "text": "Explain why the conclusion follows from the premises."
    },
    {
      "case_id": "reasoning_request",
      "variant_id": "all_caps",
      "text": "EXPLAIN WHY THE CONCLUSION FOLLOWS FROM THE PREMISES.",
      "comparison_to": "control",
      "tags": ["caps", "formatting"]
    },
    {
      "case_id": "reasoning_request",
      "variant_id": "cussing",
      "text": "Explain why the damn conclusion follows from the premises.",
      "comparison_to": "control",
      "tags": ["tone", "profanity"]
    }
  ]
}
```

### Example: Formatting

```json
{
  "name": "formatting_output",
  "variants": [
    {
      "case_id": "structured_answer",
      "variant_id": "control",
      "text": "Summarize the tradeoffs clearly."
    },
    {
      "case_id": "structured_answer",
      "variant_id": "markdown",
      "text": "## Task\nSummarize the tradeoffs clearly.\n- Use short bullets.",
      "comparison_to": "control",
      "tags": ["markdown", "formatting"]
    },
    {
      "case_id": "structured_answer",
      "variant_id": "json",
      "text": "Return a JSON object with keys summary, risks, and next_step.",
      "comparison_to": "control",
      "tags": ["json", "formatting"]
    }
  ]
}
```

### Measurement Atlas Studies

The research-only measurement atlas runner in
`scripts/run_measurement_atlas.py` consumes the same manifest surface. Its
starter study pack lives in:

- `data/probes/measurement_atlas_casing.json`
- `data/probes/measurement_atlas_profanity_tone.json`
- `data/probes/measurement_atlas_grounded_hallucination.json`

Those manifests use `mc.analyze.prompt_family.v2` so they can carry explicit
`annotations` for grounded-label and perturbation studies without creating a
second study-file format.

The current atlas contract is narrower than `SUPPORTED_ANALYSIS_SPACES`:

- observed replay spaces: `hidden`, `embedding`
- observed live spaces: `hidden`
- requested vs observed surfaces are recorded separately in
  `run_manifest.json` so the bundle does not imply that deeper replay spaces
  were captured when they were only requested or are still future work

The retained alignment-closure evidence and old-vs-fixed bundle comparison live
in `results/measurement_atlas/REPORT.md`. Use that family-level report before
starting new atlas work so the replay-token boundary fix does not get
re-litigated from memory.

`mc analyze report --bundle ...` is now also the read-side entrypoint for
those atlas artifact directories. Atlas generation stays in
`scripts/run_measurement_atlas.py`; only the read-side is shared.

`capture` builds a synthetic manifest under the hood. Each prompt becomes a
case with `variant_id="capture"`.

## `ObservationBundle`

Every `capture`, `family`, and `compare` run writes the same file set:

| File | What It Contains |
| --- | --- |
| `manifest.json` | Run metadata, targets, spaces, max tokens, and embedded prompt-family manifest |
| `summary.json` | High-level counts and mean metrics across the run |
| `REPORT.md` | Human-readable summary of what moved and where |
| `variants.jsonl` | One row per executed prompt variant |
| `layer_metrics.jsonl` | Per-layer measurements across the observed spaces |
| `comparisons.jsonl` | Pairwise deltas within one target or across two targets |

The same report command can also read measurement-atlas artifacts under
`results/measurement_atlas/<run_id>/`. Those runs use `run_manifest.json`
instead of `manifest.json` and include atlas-specific JSONL files such as
`sequence_metrics.jsonl`, `step_metrics.jsonl`, `space_step_metrics.jsonl`,
and `onset_events.jsonl`.

### What Shows Up In `REPORT.md`

The report is meant to answer the first-pass questions quickly:

- which spaces were observed
- which comparisons moved the most
- which layers moved the most
- which variants produced errors during measurement

Sections:

| Section | Purpose |
| --- | --- |
| `Means` | Overall prompt, response, entropy, deviation, and curvature averages |
| `Observed Spaces` | Which spaces were measured and how many rows they produced |
| `Largest Scalar Shifts` | Biggest pairwise deltas across metrics like entropy and curvature |
| `Most Shifted Layers` | Largest per-layer changes across the bundle |
| `Variants` | Quick per-variant metric snapshot |
| `Comparisons` | First pairwise delta rows in plain language |
| `Measurement Errors` | Variants that failed one or more collection steps |

## Recommended Starter Manifests

These are ready to run:

- `data/probes/prompt_family_minimal_pairs.json`
- `data/probes/prompt_family_casing_tone.json`
- `data/probes/prompt_family_formatting.json`
- `data/probes/prompt_family_persona_verbosity.json`

## Reading The JSONL Files

Use `variants.jsonl` when you want prompt-level responses and rollup metrics.

Use `layer_metrics.jsonl` when you want to ask questions like:

- which layer changed curvature the most?
- did entropy move in `hidden` but not `embedding`?
- did the adapter move `q` or `k` more than `v`?

Use `comparisons.jsonl` when you want direct pairwise deltas without parsing
the whole report.
