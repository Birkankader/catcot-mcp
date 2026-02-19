"""Catcot -- Semantic Code Search MCP Server.

Provides semantic code search using multi-provider embeddings and ChromaDB.
Designed for Claude Code integration via MCP protocol.
Includes token savings tracking, code review, git integration, and
project topology analysis capabilities.
"""

import json
import os
import sys
import time

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server import FastMCP

from embedder import get_provider_info
from indexer import index_project as do_index, list_indexed_projects as do_list
from searcher import search_code as do_search
from savings import record_search, get_savings_summary
from reviewer import code_review as do_review
from git_tools import (
    get_modified_files as git_modified_files,
    get_diff as git_diff,
    get_recent_changes as git_recent_changes,
    get_file_diff as git_file_diff,
    is_git_repo,
)
from topology import generate_project_map as do_project_map
from context_expander import get_chunk_context as do_get_context
from watcher import start_watching, stop_watching, list_watched

mcp = FastMCP("catcot")


@mcp.tool()
async def index_project(path: str) -> str:
    """Index a project directory for Catcot semantic code search.

    Scans files, splits into meaningful chunks (by class/function/etc.),
    embeds them using the active embedding provider, and stores in ChromaDB.
    Skips unchanged files on subsequent runs.

    Args:
        path: Absolute path to the project directory to index.
    """
    provider = get_provider_info()
    sys.stderr.write(f"[Catcot] Indexing {path} using {provider['name']} ({provider['model']})\n")
    t0 = time.time()
    stats = await do_index(path)
    elapsed = time.time() - t0
    stats["embedding_provider"] = provider["name"]
    stats["embedding_model"] = provider["model"]
    stats["elapsed_seconds"] = round(elapsed, 1)
    sys.stderr.write(
        f"[Catcot] Indexed {stats['files_indexed']} files, "
        f"{stats['chunks_created']} chunks in {elapsed:.1f}s\n"
    )
    return json.dumps(stats, indent=2)


@mcp.tool()
async def search_code(query: str, project_path: str = "", top_k: int = 5) -> str:
    """Catcot semantic code search -- find relevant code using natural language.

    Returns the most relevant code chunks with file paths, line numbers,
    and source code. Use this to find relevant code without reading entire files.
    Catcot saves tokens by returning only the relevant chunks instead of whole files.

    Args:
        query: Natural language query or code pattern to search for.
        project_path: Optional: limit search to a specific project path.
        top_k: Number of results to return (default: 5).
    """
    provider = get_provider_info()
    sys.stderr.write(f"[Catcot] Searching: \"{query}\" via {provider['name']}\n")
    t0 = time.time()
    results = await do_search(
        query=query,
        project_path=project_path or None,
        top_k=top_k,
    )
    elapsed = time.time() - t0
    sys.stderr.write(f"[Catcot] Found {len(results)} results in {elapsed:.2f}s\n")

    # Record savings
    savings = record_search(
        query=query,
        results=results,
        project_path=project_path,
    )

    formatted = []
    for r in results:
        formatted.append(
            f"### {r['file_path']}:{r['start_line']}-{r['end_line']}"
            f" ({r['symbol_name'] or 'chunk'})"
            f" [similarity: {r['similarity']}]\n"
            f"```{r['language']}\n{r['content']}\n```"
        )
    if not formatted:
        return "[Catcot] No results found."

    header = (
        f"[Catcot] Found {len(results)} results "
        f"(saved ~{savings['tokens_saved']:,} tokens, ~${savings['dollars_saved']:.4f})\n\n"
    )
    return header + "\n\n".join(formatted)


@mcp.tool()
async def reindex_project(path: str) -> str:
    """Re-index a project from scratch with Catcot. Deletes existing index and rebuilds.

    Use when files have changed significantly or index seems stale.

    Args:
        path: Absolute path to the project directory to re-index.
    """
    provider = get_provider_info()
    sys.stderr.write(f"[Catcot] Re-indexing {path} from scratch using {provider['name']} ({provider['model']})\n")
    t0 = time.time()
    stats = await do_index(path, reindex=True)
    elapsed = time.time() - t0
    stats["embedding_provider"] = provider["name"]
    stats["embedding_model"] = provider["model"]
    stats["elapsed_seconds"] = round(elapsed, 1)
    sys.stderr.write(
        f"[Catcot] Re-indexed {stats['files_indexed']} files, "
        f"{stats['chunks_created']} chunks in {elapsed:.1f}s\n"
    )
    return f"[Catcot] Re-indexed from scratch:\n{json.dumps(stats, indent=2)}"


@mcp.tool()
async def list_indexed_projects() -> str:
    """List all projects indexed by Catcot for semantic code search."""
    projects = do_list()
    if not projects:
        return "[Catcot] No projects indexed yet."
    provider = get_provider_info()
    header = f"[Catcot] Active provider: {provider['name']} ({provider['model']})\n\n"
    return header + json.dumps(projects, indent=2)


@mcp.tool()
async def get_embedding_status() -> str:
    """Show the active embedding provider, model, and dimensions.

    Use this to check which embedding provider Catcot is currently using.
    """
    try:
        provider = get_provider_info()
        return json.dumps({
            "provider": provider["name"],
            "model": provider["model"],
            "dimensions": provider["dimensions"],
            "status": "active",
        }, indent=2)
    except RuntimeError as e:
        return json.dumps({
            "provider": None,
            "status": "unavailable",
            "error": str(e),
        }, indent=2)


@mcp.tool()
async def code_review(file_path: str, context: str = "", model: str = "auto") -> str:
    """Catcot code review -- review a file using semantic context + AI backend.

    Reads the file, finds related code via Catcot semantic search for context,
    then sends to an AI backend for review. Returns a combined review report.

    Args:
        file_path: Absolute path to the file to review.
        context: Optional additional context about what to focus on in the review.
        model: AI backend to use. Options: "auto" (try all), "gemini",
               "ollama", "ollama:deepseek-coder", "anthropic",
               "anthropic:claude-sonnet-4-20250514", "openai", "openai:gpt-4o".
    """
    result = await do_review(file_path=file_path, context=context, model=model)
    return f"[Catcot Review]\n\n{result}"


# ---------------------------------------------------------------------------
# Feature: Git Integration
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_modified_files(
    query: str, project_path: str, commits: int = 5, top_k: int = 10
) -> str:
    """Search only in recently modified files (git status + last N commits).

    Combines git status (working directory changes) with recent commit history
    to identify modified files, then performs semantic search filtered to those files.

    Args:
        query: Natural language query or code pattern to search for.
        project_path: Absolute path to the git project directory.
        commits: Number of recent commits to consider (default: 5).
        top_k: Maximum number of results to return (default: 10).
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    if not await is_git_repo(project_path):
        return f"[Catcot] Error: {project_path} is not a git repository."

    # Get list of modified files
    try:
        modified_files = await git_modified_files(project_path, commits=commits)
    except ValueError as e:
        return f"[Catcot] Error: {e}"

    if not modified_files:
        return "[Catcot] No modified files found in working directory or recent commits."

    # Search across the entire project
    all_results = await do_search(
        query=query,
        project_path=project_path,
        top_k=top_k * 3,  # Fetch more to filter down
    )

    # Filter results to only modified files
    modified_set = set(modified_files)
    filtered = [
        r for r in all_results
        if r.get("file_path", "") in modified_set
    ][:top_k]

    if not filtered:
        # Return the list of modified files even if no semantic matches
        files_list = "\n".join(f"  - {f}" for f in modified_files[:20])
        return (
            f"[Catcot] No semantic matches in modified files for: '{query}'\n\n"
            f"Modified files ({len(modified_files)}):\n{files_list}"
        )

    # Record savings
    savings = record_search(query=query, results=filtered, project_path=project_path)

    formatted = []
    for r in filtered:
        formatted.append(
            f"### {r['file_path']}:{r['start_line']}-{r['end_line']}"
            f" ({r['symbol_name'] or 'chunk'})"
            f" [similarity: {r['similarity']}]\n"
            f"```{r['language']}\n{r['content']}\n```"
        )

    header = (
        f"[Catcot] Found {len(filtered)} results in modified files "
        f"(from {len(modified_files)} changed files, last {commits} commits)\n"
        f"(saved ~{savings['tokens_saved']:,} tokens, ~${savings['dollars_saved']:.4f})\n\n"
    )
    return header + "\n\n".join(formatted)


@mcp.tool()
async def review_diff(project_path: str, staged: bool = False) -> str:
    """Review git diff with semantic context from the project.

    Gets the current git diff, identifies changed files, finds related code
    via semantic search for context, and builds a contextual review report.

    Args:
        project_path: Absolute path to the git project directory.
        staged: If True, review only staged changes. Otherwise, unstaged changes.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    if not await is_git_repo(project_path):
        return f"[Catcot] Error: {project_path} is not a git repository."

    # Get the diff
    try:
        diff_text = await git_diff(project_path, staged=staged)
    except ValueError as e:
        return f"[Catcot] Error: {e}"

    if diff_text == "(No changes found)":
        diff_type = "staged" if staged else "unstaged"
        return f"[Catcot] No {diff_type} changes found in {project_path}."

    # Parse changed files from diff
    changed_files = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            # Extract b/path from "diff --git a/path b/path"
            parts = line.split()
            if len(parts) >= 4:
                b_path = parts[3]
                if b_path.startswith("b/"):
                    changed_files.append(b_path[2:])

    # For each changed file, find related code via semantic search
    context_sections = []
    seen_context_files = set()

    for changed_file in changed_files[:5]:  # Limit to 5 files for performance
        try:
            # Search for code related to the changed file
            search_query = f"{changed_file} related code dependencies"
            related = await do_search(
                query=search_query,
                project_path=project_path,
                top_k=3,
            )
            for r in related:
                ctx_file = r.get("file_path", "")
                if ctx_file and ctx_file not in seen_context_files and ctx_file != changed_file:
                    seen_context_files.add(ctx_file)
                    context_sections.append(
                        f"**{ctx_file}:{r['start_line']}-{r['end_line']}** "
                        f"({r['symbol_name'] or 'chunk'}, similarity: {r['similarity']})\n"
                        f"```{r['language']}\n{r['content'][:500]}\n```"
                    )
        except Exception:
            continue

    # Build the review report
    diff_type = "Staged" if staged else "Unstaged"
    report_parts = [
        f"# [Catcot] {diff_type} Diff Review",
        f"**Project:** `{project_path}`",
        f"**Changed files:** {len(changed_files)}",
        "",
        "## Changed Files",
    ]

    for f in changed_files:
        report_parts.append(f"  - `{f}`")

    report_parts.extend(["", "## Diff", "```diff"])

    # Truncate diff if too large
    if len(diff_text) > 10000:
        report_parts.append(diff_text[:10000])
        report_parts.append(f"\n... (truncated, {len(diff_text)} total characters)")
    else:
        report_parts.append(diff_text)

    report_parts.append("```")

    if context_sections:
        report_parts.extend([
            "",
            "## Related Code Context (via Catcot semantic search)",
            "The following code is semantically related to the changed files "
            "and may be affected by or relevant to these changes:",
            "",
        ])
        report_parts.extend(context_sections[:10])

    return "\n".join(report_parts)


# ---------------------------------------------------------------------------
# Feature: Project Topology
# ---------------------------------------------------------------------------


@mcp.tool()
async def generate_project_map(project_path: str) -> str:
    """Generate a semantic map of project components and their relationships.

    Analyzes the indexed project to identify logical components (clusters of
    related code), their files, symbols, and inter-component relationships.
    Requires the project to be indexed first.

    Args:
        project_path: Absolute path to the indexed project directory.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    try:
        project_map = await do_project_map(project_path)
    except ValueError as e:
        return f"[Catcot] Error: {e}"

    # Format the output as a readable report with JSON data
    components = project_map.get("components", [])
    relationships = project_map.get("relationships", [])

    report_parts = [
        f"# [Catcot] Project Map: {os.path.basename(project_path)}",
        f"**Path:** `{project_path}`",
        f"**Files:** {project_map['total_files']} | "
        f"**Chunks:** {project_map['total_chunks']} | "
        f"**Components:** {len(components)}",
        "",
    ]

    # Components section
    report_parts.append("## Components")
    for comp in components:
        langs = ", ".join(comp["languages"]) if comp["languages"] else "unknown"
        report_parts.append(
            f"\n### {comp['label']} ({comp['file_count']} files, {langs})"
        )
        report_parts.append(f"**Directory:** `{comp['directory']}`")
        if comp["symbols"]:
            symbols_str = ", ".join(f"`{s}`" for s in comp["symbols"][:10])
            report_parts.append(f"**Key symbols:** {symbols_str}")
        for f in comp["files"]:
            report_parts.append(f"  - `{f}`")

    # Relationships section
    if relationships:
        report_parts.extend(["", "## Inter-Component Relationships"])
        for rel in relationships:
            report_parts.append(
                f"  - **{rel['source']}** <-> **{rel['target']}** "
                f"(similarity: {rel['similarity']})"
            )

    # Append raw JSON for programmatic use
    report_parts.extend([
        "",
        "## Raw Data (JSON)",
        "```json",
        json.dumps(project_map, indent=2, default=str),
        "```",
    ])

    return "\n".join(report_parts)


# ---------------------------------------------------------------------------
# Feature: Context Expansion
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_chunk_context(
    file_path: str,
    start_line: int,
    end_line: int,
    context_before: int = 15,
    context_after: int = 15,
) -> str:
    """Get surrounding context for a code chunk found by Catcot search.

    After using search_code to find relevant chunks, use this tool to see
    more lines around a specific result. Useful for understanding function
    signatures, class definitions, or nearby logic.

    Args:
        file_path: Absolute path to the source file (from search results).
        start_line: Start line of the chunk (from search results).
        end_line: End line of the chunk (from search results).
        context_before: Lines to show before the chunk (default: 15).
        context_after: Lines to show after the chunk (default: 15).
    """
    try:
        result = do_get_context(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            context_before=context_before,
            context_after=context_after,
        )
        return (
            f"[Catcot] Context for {file_path}:"
            f"{result['actual_start_line']}-{result['actual_end_line']}"
            f" (chunk at lines {start_line}-{end_line})\n\n"
            f"{result['content']}"
        )
    except (FileNotFoundError, ValueError) as e:
        return f"[Catcot] Error: {e}"


# ---------------------------------------------------------------------------
# Feature: Watch Mode (Auto-indexing)
# ---------------------------------------------------------------------------


@mcp.tool()
async def watch_project(project_path: str, action: str = "start") -> str:
    """Start or stop auto-indexing for a project directory.

    When watching is active, Catcot automatically re-indexes files
    as they are saved. No need to manually re-index.

    Args:
        project_path: Absolute path to the project directory.
        action: "start" to begin watching, "stop" to stop, "status" to check.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    if action == "start":
        try:
            result = start_watching(project_path)
            return f"[Catcot] {result}"
        except Exception as e:
            return f"[Catcot] Error starting watcher: {e}"
    elif action == "stop":
        try:
            result = stop_watching(project_path)
            return f"[Catcot] {result}"
        except Exception as e:
            return f"[Catcot] Error stopping watcher: {e}"
    elif action == "status":
        watched = list_watched()
        if not watched:
            return "[Catcot] No projects are currently being watched."
        lines = ["[Catcot] Currently watched projects:"]
        for w in watched:
            lines.append(f"  - {w}")
        return "\n".join(lines)
    else:
        return f"[Catcot] Unknown action: '{action}'. Use 'start', 'stop', or 'status'."


if __name__ == "__main__":
    from web import start_dashboard
    try:
        provider = get_provider_info()
        sys.stderr.write(
            f"[Catcot] Embedding provider: {provider['name']} "
            f"(model: {provider['model']}, dims: {provider['dimensions']})\n"
        )
    except RuntimeError as e:
        sys.stderr.write(f"[Catcot] Warning: {e}\n")
    port = start_dashboard(open_browser=True)
    sys.stderr.write(f"[Catcot] Dashboard running at http://localhost:{port}\n")
    mcp.run(transport="stdio")
