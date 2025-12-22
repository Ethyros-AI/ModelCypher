# ModelCypher Agent Notes

- Use the CLI reference in docs/CLI-REFERENCE.md for command shape and output fields.
- MCP tool definitions mirror docs/MCP.md.
- Core logic lives in src/modelcypher/core and should not import adapters directly.
- MLX backend enforces eval after operations; keep weight layout [out, in].
- Use Python logging for structured output; avoid print in core logic.
- If unexpected file changes appear, stop and ask before proceeding to avoid clobbering another agent's work.
- If overlap is likely, research best practice and explain the choice in code comments so consensus is clear.
