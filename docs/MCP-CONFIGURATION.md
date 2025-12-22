# MCP Configuration Guide

Detailed configuration for MCP servers used in ModelCypher development.

## Firecrawl Configuration

### Server Setup

**Cloud endpoint:**
```
https://mcp.firecrawl.dev/{FIRECRAWL_API_KEY}/v2/mcp
```

**Local/CLI:**
```bash
env FIRECRAWL_API_KEY=fc-... npx -y firecrawl-mcp
# For HTTP transport: add HTTP_STREAMABLE_SERVER=true
```

### MCP Server Config (`.mcp.json`)

```json
{
  "mcpServers": {
    "firecrawl-mcp": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "timeout": 300000,
      "env": {
        "FIRECRAWL_API_KEY": "YOUR_API_KEY",
        "FIRECRAWL_RETRY_MAX_ATTEMPTS": "5",
        "FIRECRAWL_RETRY_INITIAL_DELAY": "2000",
        "FIRECRAWL_RETRY_MAX_DELAY": "30000",
        "FIRECRAWL_RETRY_BACKOFF_FACTOR": "3"
      }
    }
  }
}
```

### Timeout Configuration

Two timeout layers must be configured for slow sites:

1. **MCP Transport Timeout** — Protocol-level (TypeScript SDK has 60s hard limit)
2. **Firecrawl Scrape Timeout** — Per-request scraper timeout

**Per-request scrape config:**
```json
{
  "url": "https://slow-site.gov/page",
  "formats": ["markdown"],
  "timeout": 90000,
  "maxAge": 172800000,
  "onlyMainContent": true,
  "waitFor": 5000
}
```

**Timeout strategy by site type:**
- Fast sites (docs, blogs): scrape `timeout: 30000`, `maxAge: 172800000`
- Slow sites (government): MCP `timeout: 300000`, scrape `timeout: 60000-90000`
- Very slow/dynamic: MCP `timeout: 300000`, scrape `timeout: 90000`, actions with wait

### Token Impact

| Tool | Token Impact |
|------|--------------|
| `search` (no scrapeOptions) | LOW (~5KB) |
| `map` | LOW (~10KB) |
| `scrape` + `extract` | MEDIUM |
| `scrape` (markdown) | HIGH (~50KB/page) |
| `crawl` | VERY HIGH |

### Available Tools

1. `firecrawl_scrape` — Single URL with actions
2. `firecrawl_batch_scrape` — Multi-URL queue
3. `firecrawl_check_batch_status` — Poll batch jobs
4. `firecrawl_map` — Discover site URLs
5. `firecrawl_search` — Web/news/images search
6. `firecrawl_crawl` — Multi-level crawl
7. `firecrawl_check_crawl_status` — Track crawl progress
8. `firecrawl_extract` — LLM-powered structured extraction

## ModelCypher MCP

See `docs/MCP.md` for full documentation.

**Quick setup:**
```bash
poetry install
poetry run modelcypher-mcp
```

**Claude Desktop config:**
```json
{
  "mcpServers": {
    "modelcypher": {
      "command": "poetry",
      "args": ["run", "modelcypher-mcp"],
      "env": {
        "MC_MCP_PROFILE": "training"
      }
    }
  }
}
```

## Swift Navigation MCP

**Setup:**
```bash
cd tools/swift-nav-mcp
./install-mcp.sh
```

**Tools:**
- `find_symbol` — Find classes, structs, actors, functions
- `find_definition` — Locate symbol definitions
- `find_references` — Find all references
- `list_symbols_in_file` — List top-level declarations

Server auto-detects Xcode's DerivedData index.

## Context7

No configuration needed. Use for curated library documentation:
- `resolve-library-id` — Find library ID
- `get-library-docs` — Fetch documentation

## GitHub MCP

Use GitHub search syntax with `search_code`:
```
repo:owner/repo path:Domain "SafeGPU" language:swift
```

Best practices:
1. Always include `repo:` filter
2. Add `path:` or `language:` when possible
3. Iterate: broad search → refine with facets
4. Pair with `get_file_contents` after finding paths
