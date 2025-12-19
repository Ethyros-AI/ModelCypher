# ModelCypher Agent Notes

- Use the CLI reference in docs/CLI-REFERENCE.md for command shape and output fields.
- MCP tool definitions mirror TrainingCypher docs in docs/MCP.md.
- Core logic lives in src/modelcypher/core and should not import adapters directly.
- MLX backend enforces eval after operations; keep weight layout [out, in].
- Use Python logging for structured output; avoid print in core logic.
- If unexpected file changes appear (likely from another agent), ignore them and keep working in unrelated files; do not pause for confirmation.
- Default collaboration rule: ignore and continue without pausing for user arbitration; if overlap with another agent, research best practice and explain the choice in code comments so consensus is clear.
