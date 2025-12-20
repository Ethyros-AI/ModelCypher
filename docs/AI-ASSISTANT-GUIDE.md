# AI Assistant Guide (ModelCypher)

This guide is for AI agents that call ModelCypher via the CLI or MCP, then explain what happened to a human operator.

If you are a human contributor reading this: these conventions exist so the repo stays explainable and the outputs stay safe to summarize.

## Golden rules (don’t get the human hurt)

1. Prefer fields explicitly designed for explanation: `interpretation`, `*_assessment`, `recommendedAction`, `tripped`.
2. If a metric is missing/`null`, say “not enough signal captured” — do not invent a number or threshold.
3. Treat geometry as *early warning*, not proof. Use “suggests”, “indicates”, “consistent with”, not “proves”.
4. When you recommend an action (pause/cancel), quote the tool’s own recommendation if available.
5. If you suspect a command is stubbed or partial, check `docs/PARITY.md` and communicate uncertainty.

## CLI vs MCP (which should I use?)

- Use **CLI** when you’re running locally in a terminal and can pipe/parse JSON yourself.
- Use **MCP** when you want tool schemas, idempotency support, and better client-side safety controls.

Both are thin shells over the same use cases.

## Minimal safe workflow (most tasks)

1. **Discover**: `tc inventory --output json` (or MCP `tc_inventory`)
2. **Validate inputs**: `tc dataset validate <path> --output json`
3. **Run**: `tc train start ... --output json`
4. **Monitor**: `tc train status <jobId> --output json` + geometry snapshots
5. **Summarize**: 1–2 sentences + the recommended next action

## Summarization templates (copy/paste mentally)

### Training status

“Job `<jobId>` is `<status>` at `<progress%>`; loss is `<loss>` with ETA `<etaSeconds>` (if present).”

### Geometry snapshot

“Training geometry is `<flatnessAssessment>` with `<snrAssessment>` gradients; circuit breaker is `<state>` (severity `<severity>`), so recommendation is: `<recommendedAction>`.”

### Dataset validation

“Dataset is `<valid/invalid>` with `<totalExamples>` examples; biggest issue: `<top error/warning>`.”

## Common questions from humans (and how to answer)

### “Why does distance/angle matter?”

Use `docs/MATH-PRIMER.md` for the intuitive story:
- distance ≈ “how much changed”
- angle ≈ “what kind of change”
- shape/curvature ≈ “how fragile the current training region is”

### “Is this model safe now?”

Never answer with certainty from geometry alone. Say:
“These signals look nominal/warning, but they are not a safety proof. Use evals + policy review for a decision.”

### “Why did you recommend stopping?”

If a tool returned `tripped=true` or a `recommendedAction`, cite it directly:
“The circuit breaker flagged alignment drift and recommended ‘Stop and request human review’.”

## Output contracts (how to parse safely)

- CLI JSON is stable and key-sorted (`sort_keys=true`). Use exact field names.
- MCP results often include a `_schema` string like `tc.geometry.training.status.v1`. Use it to route parsing and avoid brittle heuristics.
- Many outputs include `nextActions` as suggested follow-up commands. Prefer those over inventing workflows.

