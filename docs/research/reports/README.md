# Tracked Research Reports

This directory contains markdown-only report and audit files rescued from the
gitignored `results/` tree. The data artifacts, model outputs, logs, JSONL
streams, adapters, and checkpoints remain outside git.

Use these files as durable handoffs in a fresh clone. For current reruns, scripts
and CLI commands still write new artifacts under `results/`.

## Report Families

- `nblora_vs_standard/REPORT.md`: canonical R1/R2 handoff for the
  geometry-derived LoRA benchmark and behavioral-preservation blocker.
- `measurement_atlas/REPORT.md`: retained alignment-closure report for the
  shipped 350M measurement-atlas pack.
- Other subdirectories are report-only snapshots of retained result families.

## Stubs

When a report was referenced by tracked docs but was absent from this checkout,
the corresponding path contains a stub that names the missing owner-side source.
Those stubs are not evidence and must not be used as validation results.
