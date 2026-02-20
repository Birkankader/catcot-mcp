"""Multi-provider embedding wrapper.

Supports local (fastembed), Ollama, Google, OpenAI, and Voyage.
Auto-detects available provider, or set CATCOT_EMBEDDING_PROVIDER explicitly.

Priority (local-first): Ollama → local (fastembed) → Google → OpenAI → Voyage
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Callable, Awaitable

import httpx

# ── Truncation settings ──────────────────────────────────────────────
MAX_CHARS = 6000
MIN_CHARS = 500
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3

# ── Provider dataclass ───────────────────────────────────────────────

@dataclass
class _EmbeddingProvider:
    name: str
    dimensions: int
    model: str
    embed: Callable[[httpx.AsyncClient, list[str]], Awaitable[list[list[float]]]]


# ── Shared retry helper ──────────────────────────────────────────────

async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    """HTTP request with exponential backoff retry for transient errors."""
    for attempt in range(_MAX_RETRIES):
        resp = await client.request(method, url, **kwargs)
        if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES - 1:
            wait = 2 ** attempt
            sys.stderr.write(
                f"[Catcot] HTTP {resp.status_code}, retrying in {wait}s "
                f"(attempt {attempt + 1}/{_MAX_RETRIES})\n"
            )
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    return resp  # unreachable, but satisfies type checker


def _sanitize_texts(texts: list[str]) -> list[str]:
    """Ensure no empty strings (some providers reject them)."""
    return [(t.strip() or " ")[:MAX_CHARS] for t in texts]


# ── Local (fastembed) provider ───────────────────────────────────────

_FASTEMBED_MODEL_CACHE = None

def _get_fastembed_model():
    """Lazy-load fastembed model (cached per process)."""
    global _FASTEMBED_MODEL_CACHE
    if _FASTEMBED_MODEL_CACHE is None:
        from fastembed import TextEmbedding
        model_name = os.environ.get("CATCOT_LOCAL_MODEL", "BAAI/bge-small-en-v1.5")
        _FASTEMBED_MODEL_CACHE = TextEmbedding(model_name=model_name)
    return _FASTEMBED_MODEL_CACHE


def _check_fastembed_available() -> bool:
    """Check if fastembed is installed."""
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


async def _embed_local(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Embed using fastembed (runs locally, no server needed).

    Runs in a thread to avoid blocking the event loop.
    """
    model = _get_fastembed_model()
    sanitized = _sanitize_texts(texts)
    embeddings = await asyncio.to_thread(
        lambda: [e.tolist() for e in model.embed(sanitized)]
    )
    return embeddings


# ── Ollama provider ─────────────────────────────────────────────────

async def _embed_ollama(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    model = os.environ.get("CATCOT_OLLAMA_MODEL", "nomic-embed-text")
    ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    results: list[list[float]] = []

    for text in texts:
        content = text.strip() or " "
        limit = MAX_CHARS

        while True:
            truncated = content[:limit] if len(content) > limit else content
            resp = await client.post(
                f"{ollama_url}/api/embed",
                json={"model": model, "input": truncated},
            )

            if resp.status_code == 400 and "context length" in resp.text:
                limit = limit // 2
                if limit < MIN_CHARS:
                    limit = MIN_CHARS
                    truncated = content[:limit]
                    resp = await client.post(
                        f"{ollama_url}/api/embed",
                        json={"model": model, "input": truncated},
                    )
                    resp.raise_for_status()
                    break
                continue

            resp.raise_for_status()
            break

        data = resp.json()
        emb_list = data.get("embeddings") or []
        if emb_list:
            results.append(emb_list[0])
        else:
            raise RuntimeError(
                f"Ollama returned empty embeddings for model '{model}'. "
                "Ensure the model is pulled: ollama pull " + model
            )

    return results


# ── API providers ────────────────────────────────────────────────────

async def _embed_google(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("CATCOT_GOOGLE_MODEL", "text-embedding-004")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={api_key}"

    sanitized = _sanitize_texts(texts)
    requests_body = [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in sanitized]
    resp = await _request_with_retry(client, "POST", url, json={"requests": requests_body})
    data = resp.json()
    return [e["values"] for e in data["embeddings"]]


async def _embed_openai(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("CATCOT_OPENAI_MODEL", "text-embedding-3-small")
    resp = await _request_with_retry(
        client, "POST",
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": _sanitize_texts(texts)},
    )
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


async def _embed_voyage(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    model = os.environ.get("CATCOT_VOYAGE_MODEL", "voyage-3-lite")
    resp = await _request_with_retry(
        client, "POST",
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": _sanitize_texts(texts)},
    )
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


# ── Provider defaults ────────────────────────────────────────────────

_PROVIDER_DEFAULTS: dict[str, tuple[int, str, Callable]] = {
    "ollama":  (768,  "nomic-embed-text",       _embed_ollama),
    "local":   (384,  "BAAI/bge-small-en-v1.5", _embed_local),
    "google":  (768,  "text-embedding-004",      _embed_google),
    "openai":  (1536, "text-embedding-3-small",  _embed_openai),
    "voyage":  (512,  "voyage-3-lite",           _embed_voyage),
}

# ── Shared HTTP client ───────────────────────────────────────────────

_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared HTTP client for connection reuse."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=120.0)
    return _HTTP_CLIENT


# ── Provider resolution ──────────────────────────────────────────────

_PROVIDER_CACHE: _EmbeddingProvider | None = None


def _check_ollama_reachable() -> bool:
    """Check if Ollama is reachable (2s timeout).

    Uses a short-lived sync client to avoid event loop issues at startup.
    """
    ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = httpx.get(f"{ollama_url}/", timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _resolve_provider() -> _EmbeddingProvider:
    """Resolve which embedding provider to use.

    Priority (local-first, no token waste):
      1. Ollama running locally
      2. fastembed installed (pure local, no server)
      3. GOOGLE_API_KEY / GEMINI_API_KEY
      4. OPENAI_API_KEY
      5. VOYAGE_API_KEY
      6. None → error with instructions
    """
    global _PROVIDER_CACHE
    if _PROVIDER_CACHE is not None:
        return _PROVIDER_CACHE

    explicit = os.environ.get("CATCOT_EMBEDDING_PROVIDER", "").lower().strip()

    if explicit:
        if explicit not in _PROVIDER_DEFAULTS:
            raise RuntimeError(
                f"Unknown embedding provider: '{explicit}'. "
                f"Valid options: {', '.join(_PROVIDER_DEFAULTS.keys())}"
            )
        name = explicit
        _verify_provider_env(name)
    else:
        # Auto-detect: local-first priority
        if _check_ollama_reachable():
            name = "ollama"
        elif _check_fastembed_available():
            name = "local"
        elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            name = "google"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        elif os.environ.get("VOYAGE_API_KEY"):
            name = "voyage"
        else:
            raise RuntimeError(
                "No embedding provider available. Options:\n"
                "  1. Start Ollama locally: ollama serve\n"
                "  2. Install fastembed for pure-local embeddings: pip install fastembed\n"
                "  3. Set GOOGLE_API_KEY or GEMINI_API_KEY for Google embeddings\n"
                "  4. Set OPENAI_API_KEY for OpenAI embeddings\n"
                "  5. Set VOYAGE_API_KEY for Voyage embeddings\n"
                "  Or set CATCOT_EMBEDDING_PROVIDER explicitly."
            )

    dims, default_model, embed_fn = _PROVIDER_DEFAULTS[name]
    model_env = f"CATCOT_{name.upper()}_MODEL"
    model = os.environ.get(model_env, default_model)

    _PROVIDER_CACHE = _EmbeddingProvider(
        name=name,
        dimensions=dims,
        model=model,
        embed=embed_fn,
    )
    return _PROVIDER_CACHE


def _verify_provider_env(name: str) -> None:
    """Verify that required env vars exist for the chosen provider."""
    env_checks = {
        "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
        "voyage": ("VOYAGE_API_KEY",),
        "ollama": (),
        "local":  (),
    }
    keys = env_checks.get(name, ())
    if keys and not any(os.environ.get(k) for k in keys):
        raise RuntimeError(
            f"Provider '{name}' requires one of: {', '.join(keys)}"
        )
    if name == "local" and not _check_fastembed_available():
        raise RuntimeError(
            "Provider 'local' requires fastembed: pip install fastembed"
        )


def reset_provider_cache() -> None:
    """Reset the cached provider. Useful for testing or provider switching."""
    global _PROVIDER_CACHE, _HTTP_CLIENT
    _PROVIDER_CACHE = None
    if _HTTP_CLIENT and not _HTTP_CLIENT.is_closed:
        # Don't close here — it may be in use. Let GC handle it.
        _HTTP_CLIENT = None


# ── Public API ───────────────────────────────────────────────────────

def get_provider_info() -> dict:
    """Return info about the current embedding provider.

    Returns dict with keys: name, model, dimensions.
    """
    provider = _resolve_provider()
    return {
        "name": provider.name,
        "model": provider.model,
        "dimensions": provider.dimensions,
    }


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the active provider.

    Returns list of embedding vectors.
    """
    provider = _resolve_provider()
    client = _get_http_client()

    try:
        return await provider.embed(client, texts)
    except httpx.ConnectError:
        if provider.name == "ollama":
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve"
            )
        raise RuntimeError(
            f"Cannot connect to {provider.name} API. Check your network and API key."
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"{provider.name} API error: {e.response.status_code} - {e.response.text}"
        )


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([query])
    return results[0]
