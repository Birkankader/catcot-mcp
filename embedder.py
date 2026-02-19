"""Ollama embedding wrapper using nomic-embed-text model."""

import httpx

OLLAMA_URL = "http://localhost:11434"
MODEL = "nomic-embed-text"
# nomic-embed-text context: 2048 tokens. Token/char ratio varies by content,
# so we start generous and shrink on failure.
MAX_CHARS = 6000
MIN_CHARS = 500


async def _embed_one(client: httpx.AsyncClient, text: str) -> list[float]:
    """Embed a single text, automatically shrinking if it exceeds context length."""
    limit = MAX_CHARS
    content = text.strip() or " "

    while True:
        truncated = content[:limit] if len(content) > limit else content
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": MODEL, "input": truncated},
        )

        if resp.status_code == 400 and "context length" in resp.text:
            # Shrink by half and retry
            limit = limit // 2
            if limit < MIN_CHARS:
                limit = MIN_CHARS
                # Final attempt with minimum size
                truncated = content[:limit]
                resp = await client.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": MODEL, "input": truncated},
                )
                resp.raise_for_status()
                break
            continue

        resp.raise_for_status()
        break

    data = resp.json()
    emb_list = data.get("embeddings") or []
    return emb_list[0] if emb_list else [0.0] * 768


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Ollama nomic-embed-text.

    Returns list of embedding vectors (768 dimensions each).
    """
    embeddings = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for text in texts:
            try:
                emb = await _embed_one(client, text)
                embeddings.append(emb)
            except httpx.ConnectError:
                raise RuntimeError(
                    "Ollama is not running. Start it with: ollama serve"
                )
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Ollama API error: {e.response.status_code} - {e.response.text}"
                )
    return embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([query])
    return results[0]
