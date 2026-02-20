"""Catcot Code Reviewer — combines semantic search context with multi-model review."""

import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import httpx

from catcot.core.searcher import search_code


class ReviewBackend(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AUTO = "auto"


def _parse_model_param(model: str) -> Tuple[ReviewBackend, Optional[str]]:
    """Parse 'backend:model_name' into (backend, model_name).

    Examples:
        "auto"                              -> (AUTO, None)
        "gemini"                            -> (GEMINI, None)
        "ollama"                            -> (OLLAMA, "deepseek-coder")
        "ollama:deepseek-coder"             -> (OLLAMA, "deepseek-coder")
        "anthropic:claude-sonnet-4-20250514" -> (ANTHROPIC, "claude-sonnet-4-20250514")
        "openai:gpt-4o"                     -> (OPENAI, "gpt-4o")
    """
    if ":" in model:
        backend_str, model_name = model.split(":", 1)
    else:
        backend_str = model
        model_name = None

    try:
        backend = ReviewBackend(backend_str.lower())
    except ValueError:
        backend = ReviewBackend.AUTO

    # Defaults per backend
    if model_name is None:
        defaults = {
            ReviewBackend.OLLAMA: "deepseek-coder",
            ReviewBackend.ANTHROPIC: "claude-sonnet-4-20250514",
            ReviewBackend.OPENAI: "gpt-4o",
        }
        model_name = defaults.get(backend)

    return backend, model_name


async def code_review(file_path: str, context: str = "", model: str = "auto") -> str:
    """Review a file using semantic context from Catcot + configurable AI backend.

    Args:
        file_path: Absolute path to the file to review.
        context: Optional additional context about what to focus on in the review.
        model: Backend to use. One of:
               "auto" (try each in order), "gemini", "ollama", "ollama:model-name",
               "anthropic", "anthropic:model-name", "openai", "openai:model-name"
    """
    file_path = os.path.abspath(os.path.expanduser(file_path))

    if not os.path.isfile(file_path):
        return f"Error: File not found: {file_path}"

    # Read the file
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            file_content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    if len(file_content) > 50000:
        file_content = file_content[:50000] + "\n... (truncated)"

    # Build search query from file name + context
    file_name = Path(file_path).name
    search_query = f"{file_name} {context}" if context else file_name

    # Get related code chunks via Catcot
    try:
        related_chunks = await search_code(query=search_query, top_k=3)
        context_section = "\n\n".join(
            f"--- {r['file_path']}:{r['start_line']}-{r['end_line']} "
            f"({r['symbol_name'] or 'chunk'}) ---\n{r['content']}"
            for r in related_chunks
            if r["file_path"] != file_path
        )
    except Exception:
        context_section = "(No related context found)"
        related_chunks = []

    # Build the review prompt
    review_prompt = f"""Review the following code file. Provide:
1. A brief summary of what the code does
2. Potential bugs or issues
3. Code quality observations
4. Security concerns (if any)
5. Suggestions for improvement

File: {file_path}

```
{file_content}
```

Related code context from the project:
{context_section}

{f"Additional context: {context}" if context else ""}"""

    # Dispatch to backend(s)
    backend, model_name = _parse_model_param(model)
    review_text, used_backend = await _dispatch_review(review_prompt, backend, model_name)

    # Build combined report
    report_parts = [
        f"# Catcot Code Review: {file_name}",
        f"**File:** `{file_path}`",
        "",
    ]

    if review_text:
        backend_label = _backend_label(used_backend, model_name)
        report_parts.extend([
            f"## Review (via {backend_label})",
            review_text,
            "",
        ])
    else:
        report_parts.extend([
            "## Review",
            "(No AI backend available. Configure one of: Gemini CLI, Ollama, "
            "ANTHROPIC_API_KEY, or OPENAI_API_KEY)",
            "",
        ])

    if context_section and context_section != "(No related context found)":
        report_parts.extend([
            "## Related Code Context (via Catcot)",
            f"Found {len(related_chunks)} related code chunks for additional context.",
            "",
        ])

    return "\n".join(report_parts)


def _backend_label(backend: Optional[ReviewBackend], model_name: Optional[str]) -> str:
    """Return a human-readable backend label for the report header."""
    if backend is None:
        return "Unknown"
    labels = {
        ReviewBackend.GEMINI: "Gemini CLI",
        ReviewBackend.OLLAMA: f"Ollama ({model_name or 'deepseek-coder'})",
        ReviewBackend.ANTHROPIC: f"Anthropic ({model_name or 'claude-sonnet-4-20250514'})",
        ReviewBackend.OPENAI: f"OpenAI ({model_name or 'gpt-4o'})",
    }
    return labels.get(backend, str(backend))


async def _dispatch_review(
    prompt: str,
    backend: ReviewBackend,
    model_name: Optional[str],
) -> Tuple[Optional[str], Optional[ReviewBackend]]:
    """Route the review to the requested backend or auto-detect.

    Returns (review_text, backend_used).
    """
    if backend == ReviewBackend.AUTO:
        # Try each backend in priority order
        for try_backend, try_model in [
            (ReviewBackend.GEMINI, None),
            (ReviewBackend.ANTHROPIC, "claude-sonnet-4-20250514"),
            (ReviewBackend.OPENAI, "gpt-4o"),
            (ReviewBackend.OLLAMA, "deepseek-coder"),
        ]:
            result = await _call_backend(prompt, try_backend, try_model)
            if result is not None:
                return result, try_backend
        return None, None

    result = await _call_backend(prompt, backend, model_name)
    return result, backend if result is not None else None


async def _call_backend(
    prompt: str,
    backend: ReviewBackend,
    model_name: Optional[str],
) -> Optional[str]:
    """Call a single backend and return the review text, or None on failure."""
    if backend == ReviewBackend.GEMINI:
        return await _gemini_review(prompt)
    elif backend == ReviewBackend.OLLAMA:
        return await _ollama_review(prompt, model=model_name or "deepseek-coder")
    elif backend == ReviewBackend.ANTHROPIC:
        return await _anthropic_review(prompt, model=model_name or "claude-sonnet-4-20250514")
    elif backend == ReviewBackend.OPENAI:
        return await _openai_review(prompt, model=model_name or "gpt-4o")
    return None


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

async def _gemini_review(prompt: str) -> Optional[str]:
    """Run review through Gemini CLI (subprocess)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "gemini", "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode == 0 and stdout:
            return stdout.decode("utf-8", errors="replace").strip()
        return None
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        return None


async def _ollama_review(prompt: str, model: str = "deepseek-coder") -> Optional[str]:
    """Run review through local Ollama API (http://localhost:11434)."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "").strip()
                return text if text else None
            return None
    except (httpx.ConnectError, httpx.TimeoutException, Exception):
        return None


async def _anthropic_review(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
) -> Optional[str]:
    """Run review through Anthropic API using ANTHROPIC_API_KEY env var."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            if response.status_code == 200:
                data = response.json()
                content_blocks = data.get("content", [])
                text = "".join(
                    block.get("text", "")
                    for block in content_blocks
                    if block.get("type") == "text"
                ).strip()
                return text if text else None
            return None
    except (httpx.ConnectError, httpx.TimeoutException, Exception):
        return None


async def _openai_review(prompt: str, model: str = "gpt-4o") -> Optional[str]:
    """Run review through OpenAI API using OPENAI_API_KEY env var."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    return text if text else None
            return None
    except (httpx.ConnectError, httpx.TimeoutException, Exception):
        return None
