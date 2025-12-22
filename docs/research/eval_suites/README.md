# Evaluation Suites (Planned)

This directory is reserved for reusable evaluation suite definitions referenced by the paper drafts.

These suites are **not yet finalized**. The goal is to standardize a small set of JSON suite files that can be executed via `mc infer suite` and scored consistently across:
- base models,
- adapters (LoRA “sidecars”),
- merged/stitched models.

## Planned suite files

- `coding_suite.v1.json`: coding + tool-use prompts (syntax, refactors, bugfixes, small algorithms).
- `creativity_suite.v1.json`: creative writing + style retention prompts.

## Status

Suite schema + scoring harness are under active development (see `../../PARITY.md`).
