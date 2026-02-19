# Catcot - Semantic Code Search MCP Server

## What is Catcot?
Catcot is a local semantic code search tool that indexes your codebase and provides fast, intelligent search via the MCP protocol. It uses multi-provider embeddings + ChromaDB for vector storage.

## Usage Instructions for Claude Code

### Search First
When you need to find code in a project, **always use Catcot's `search_code` tool first** before reading entire files. This saves tokens and is faster.

### Available Tools

#### Core Tools
- `search_code(query, project_path, top_k)` — Semantic search across indexed code. Use natural language queries.
- `index_project(path)` — Index a new project for search.
- `reindex_project(path)` — Re-index from scratch when files changed significantly.
- `list_indexed_projects()` — See all indexed projects.
- `code_review(file_path, context)` — Review a file with Gemini CLI + Catcot context.

#### Git Integration Tools
- `search_modified_files(query, project_path, commits, top_k)` — Search only in recently modified files (from git status + last N commits). Use this when you want to focus on recent changes.
- `review_diff(project_path, staged)` — Review git diff with semantic context. Gets the current diff, finds related code via semantic search, and builds a contextual review report. Set `staged=True` for staged changes only.

#### Context & Exploration Tools
- `get_chunk_context(file_path, start_line, end_line, context_before, context_after)` — After finding a chunk via search_code, use this to see surrounding lines (default: 15 before/after). Useful for understanding function signatures, class context, or nearby logic.
- `generate_project_map(project_path)` — Generate a semantic map of project components and their relationships. Shows how files cluster into logical components, key symbols per component, and inter-component relationships based on embedding similarity.

#### Watch Mode
- `watch_project(project_path, action)` — Start/stop auto-indexing. When watching, Catcot automatically re-indexes files as they are saved. Actions: "start", "stop", "status".

### Best Practices
- Use descriptive natural language queries: "authentication middleware" instead of "auth"
- Catcot returns ranked code chunks with file paths and line numbers
- After finding relevant chunks, read only those specific files/sections
- When working on a new project, index it first: `index_project("/path/to/project")`
- Use `search_modified_files` when reviewing recent work or PRs
- Use `review_diff` before committing to get semantic context for your changes
- Use `generate_project_map` to understand unfamiliar codebases
- Use `get_chunk_context` to expand search results when you need more surrounding code
- Use `watch_project` to enable auto-indexing during active development
- `code_review` supports multiple backends: "auto", "gemini", "ollama:model", "anthropic:model", "openai:model"

### Chunking
Catcot uses two chunking strategies:
1. **Tree-sitter AST chunking** (preferred) — Accurate, AST-aware code splitting that correctly handles decorators, nested classes, multi-line signatures, and export statements. Requires `tree-sitter` and `tree-sitter-language-pack` packages.
2. **Regex-based chunking** (fallback) — Pattern-matching based splitting by function/class boundaries. Used when tree-sitter is not installed.

Supported languages: Python, JavaScript, TypeScript, TSX, Java, Kotlin, SQL. Other file types use a generic sliding-window chunker.

### Embedding Providers
Catcot auto-detects the embedding provider with a **local-first** approach — free local options are preferred over paid API calls.

**Auto-detection priority** (first match wins):
1. Ollama running locally → Ollama (nomic-embed-text, 768 dims)
2. `fastembed` installed → Local/ONNX (BAAI/bge-small-en-v1.5, 384 dims) — no server needed
3. `GOOGLE_API_KEY` or `GEMINI_API_KEY` → Google (text-embedding-004, 768 dims)
4. `OPENAI_API_KEY` → OpenAI (text-embedding-3-small, 1536 dims)
5. `VOYAGE_API_KEY` → Voyage (voyage-3-lite, 512 dims)

**Zero-cost options:**
- **Ollama**: `ollama serve` — needs Ollama installed, runs as a local server
- **Local (fastembed)**: `pip install fastembed` — pure Python, no server, no API keys, runs on CPU

**Force a specific provider:** `CATCOT_EMBEDDING_PROVIDER=ollama|local|google|openai|voyage`

**Override default models:** `CATCOT_OLLAMA_MODEL`, `CATCOT_LOCAL_MODEL`, `CATCOT_GOOGLE_MODEL`, `CATCOT_OPENAI_MODEL`, `CATCOT_VOYAGE_MODEL`

**Other env vars:** `OLLAMA_HOST` (default: `http://localhost:11434`) — custom Ollama server URL

**Check active provider:** Use the `get_embedding_status` tool to see which provider, model, and dimensions are active.

**Provider switching:** If you change providers on an already-indexed project, you must re-index (`reindex_project`) because embeddings from different providers are incompatible. Catcot will raise an error if it detects a mismatch.

### Step Indicator
When using Catcot tools, mention "Catcot kullaniliyor" (Catcot is being used) in your response so the user knows semantic search is active.
