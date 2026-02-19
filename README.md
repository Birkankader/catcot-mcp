<p align="center">
  <img src="https://img.shields.io/badge/MCP-compatible-blue" alt="MCP Compatible">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="MIT License">
</p>

<h1 align="center">🐱 Catcot</h1>
<p align="center"><strong>Semantic code search MCP server for Claude Code</strong></p>
<p align="center">Index your codebase, search with natural language, save tokens.</p>

---

Catcot is a local MCP server that gives Claude Code semantic code search superpowers. It indexes your projects using Ollama embeddings and ChromaDB, then serves relevant code chunks instead of whole files — saving tokens and money.

## Features

- **Semantic Search** — Find code with natural language queries like "database connection pooling"
- **Token Savings Tracking** — See how many tokens and dollars you save per search
- **Code Review** — Multi-model review with Gemini, Ollama, Anthropic, or OpenAI
- **Git Integration** — Search modified files, review diffs with semantic context
- **Tree-sitter Chunking** — AST-aware code splitting for accurate results
- **Project Topology** — Visualize how your codebase clusters into components
- **Context Expansion** — Expand search results to see surrounding code
- **Watch Mode** — Auto-reindex files as you save them
- **Web Dashboard** — Visual overview with savings stats and embedding map

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with an embedding model:
  ```bash
  ollama pull nomic-embed-text
  ```

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

## Tools (10)

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search across indexed code |
| `index_project` | Index a project directory |
| `reindex_project` | Re-index from scratch |
| `list_indexed_projects` | List all indexed projects |
| `code_review` | AI-powered code review with semantic context |
| `search_modified_files` | Search only in recently changed files (git) |
| `review_diff` | Review git diff with related code context |
| `generate_project_map` | Semantic map of project components |
| `get_chunk_context` | Expand search results with surrounding lines |
| `watch_project` | Start/stop auto-indexing on file save |

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

## Supported Languages

Tree-sitter AST chunking (preferred): **Python, JavaScript, TypeScript, TSX, Java, Kotlin, SQL**

Other file types fall back to a generic sliding-window chunker.

## Architecture

```
server.py          → MCP server (FastMCP, stdio transport)
searcher.py        → ChromaDB vector search
indexer.py         → File scanning & chunk indexing
embedder.py        → Ollama embedding client
savings.py         → Token savings tracker
reviewer.py        → Multi-model code review
git_tools.py       → Git integration (status, diff, log)
topology.py        → Project component clustering
context_expander.py → Chunk context expansion
watcher.py         → Watchdog file watcher
web.py             → Dashboard HTTP server
dashboard.html     → Web dashboard UI
chunkers/          → Code chunkers (tree-sitter + regex fallbacks)
```

## License

MIT
