<p align="center">
  <img src="https://img.shields.io/badge/MCP-compatible-blue" alt="MCP Compatible">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="MIT License">
</p>

<h1 align="center">🐱 Catcot</h1>
<p align="center"><strong>Semantic code search MCP server for Claude Code</strong></p>
<p align="center">Index your codebase, search with natural language, save tokens.</p>

---

Catcot is a local MCP server that gives Claude Code semantic code search superpowers. It indexes your projects using multi-provider embeddings and ChromaDB, then serves relevant code chunks instead of whole files — saving tokens and money.

## Features

- **Semantic Search** — Find code with natural language queries like "database connection pooling"
- **Token Savings Tracking** — See how many tokens and dollars you save per search
- **Code Review** — Multi-model review with Gemini, Ollama, Anthropic, or OpenAI
- **Git Integration** — Search modified files, review diffs with semantic context
- **Tree-sitter Chunking** — AST-aware code splitting for accurate results
- **Project Topology** — Visualize how your codebase clusters into components
- **Context Expansion** — Expand search results to see surrounding code
- **Persistent Memory** — Store project knowledge (env paths, build commands, arch decisions) across sessions
- **Watch Mode** — Auto-reindex files as you save them
- **Web Dashboard** — Visual overview with savings stats, embedding map, and memory browser

## Quick Start

### Prerequisites

- Python 3.10+
- At least one embedding provider (auto-detected, local-first):
  - **Ollama** (recommended): `ollama pull nomic-embed-text`
  - **Local/ONNX**: `pip install fastembed` — no server needed
  - **API-based**: Set `GOOGLE_API_KEY`, `OPENAI_API_KEY`, or `VOYAGE_API_KEY`

### Install

```bash
git clone https://github.com/Birkankader/catcot-mcp.git
cd catcot-mcp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Register with Claude Code

```bash
claude mcp add -s user catcot -- /path/to/catcot-mcp/.venv/bin/python /path/to/catcot-mcp/server.py
```

### Use

In Claude Code, just ask naturally:

```
> search for authentication middleware in my project
> index /path/to/my/project
> review the file src/auth.py
> show me the project map
```

Claude will automatically use Catcot's tools.

## Tools (15)

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search across indexed code |
| `index_project` | Index a project directory |
| `reindex_project` | Re-index from scratch |
| `list_indexed_projects` | List all indexed projects |
| `get_embedding_status` | Check active embedding provider and model |
| `code_review` | AI-powered code review with semantic context |
| `search_modified_files` | Search only in recently changed files (git) |
| `review_diff` | Review git diff with related code context |
| `generate_project_map` | Semantic map of project components |
| `get_chunk_context` | Expand search results with surrounding lines |
| `watch_project` | Start/stop auto-indexing on file save |
| `store_memory` | Store persistent project knowledge across sessions |
| `recall_memory` | Recall memories by key, tags, or natural language |
| `list_project_memories` | List all stored memories for a project |
| `delete_project_memory` | Delete a specific memory by key |

## Dashboard

Run the server directly to launch the web dashboard:

```bash
python server.py
```

Opens at `http://localhost:9850` with:
- Token savings stats and 7-day trend
- Indexed projects overview
- 2D embedding space visualization
- Architecture and quickstart guide

## Embedding Providers

Catcot auto-detects the best available embedding provider with a **local-first** approach:

| Priority | Provider | Model | Dimensions | Requires |
|----------|----------|-------|------------|----------|
| 1 | Ollama | nomic-embed-text | 768 | `ollama serve` |
| 2 | Local/ONNX | BAAI/bge-small-en-v1.5 | 384 | `pip install fastembed` |
| 3 | Google | text-embedding-004 | 768 | `GOOGLE_API_KEY` |
| 4 | OpenAI | text-embedding-3-small | 1536 | `OPENAI_API_KEY` |
| 5 | Voyage | voyage-3-lite | 512 | `VOYAGE_API_KEY` |

Force a specific provider: `CATCOT_EMBEDDING_PROVIDER=ollama|local|google|openai|voyage`

## Supported Languages

Tree-sitter AST chunking (preferred): **Python, JavaScript, TypeScript, TSX, Java, Kotlin, SQL**

Other file types fall back to a generic sliding-window chunker.

## Architecture

```
server.py           → MCP server (FastMCP, stdio transport)
config.py           → Centralized configuration & shared helpers
searcher.py         → ChromaDB vector search
indexer.py          → File scanning & chunk indexing
embedder.py         → Multi-provider embedding client
memory.py           → Persistent memory system (JSON + ChromaDB)
savings.py          → Token savings tracker
reviewer.py         → Multi-model code review
git_tools.py        → Git integration (status, diff, log)
topology.py         → Project component clustering
context_expander.py → Chunk context expansion
watcher.py          → Watchdog file watcher
web.py              → Dashboard HTTP server
dashboard.html      → Web dashboard UI
chunkers/           → Code chunkers (tree-sitter + regex fallbacks)
```

## License

MIT
