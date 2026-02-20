"""Catcot persistent memory system.

Stores key-value memories per project with semantic search support.
Memories persist across sessions as JSON files and are also embedded
in ChromaDB for natural-language recall.
"""

import json
import os
import time
import uuid
from pathlib import Path

from catcot.config import MEMORY_DIR, get_chroma_client, memory_collection_name
from catcot.core.embedder import embed_texts, embed_query


def _memory_file(project_path: str) -> str:
    """Return the JSON file path for a project's memories."""
    safe_name = project_path.replace("/", "_").replace("\\", "_").strip("_")
    # Truncate to avoid filesystem issues
    if len(safe_name) > 100:
        import hashlib
        h = hashlib.md5(project_path.encode()).hexdigest()[:12]
        safe_name = safe_name[:80] + "_" + h
    return os.path.join(MEMORY_DIR, f"{safe_name}.json")


def _load_memories(project_path: str) -> list[dict]:
    """Load all memories for a project from disk."""
    fpath = _memory_file(project_path)
    if not os.path.exists(fpath):
        return []
    try:
        with open(fpath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_memories(project_path: str, memories: list[dict]) -> None:
    """Persist memories to disk."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    fpath = _memory_file(project_path)
    with open(fpath, "w") as f:
        json.dump(memories, f, indent=2)


def _embed_text_for_memory(mem: dict) -> str:
    """Build the text to embed for a memory entry."""
    parts = [mem.get("key", ""), mem.get("value", "")]
    tags = mem.get("tags", [])
    if tags:
        parts.append(" ".join(tags))
    return " | ".join(parts)


async def _sync_to_chroma(project_path: str, memories: list[dict]) -> None:
    """Sync all memories to a ChromaDB collection for semantic search."""
    if not memories:
        return

    client = get_chroma_client()
    col_name = memory_collection_name(project_path)

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"project_path": project_path, "type": "memory", "hnsw:space": "cosine"},
    )

    # Upsert all memories
    ids = [m["id"] for m in memories]
    texts = [_embed_text_for_memory(m) for m in memories]
    metadatas = [{"key": m["key"], "project_path": project_path} for m in memories]

    embeddings = await embed_texts(texts)
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


async def store_memory(
    project_path: str,
    key: str,
    value: str,
    tags: list[str] | None = None,
    source: str = "ai",
) -> dict:
    """Store or update a memory (key-based upsert).

    Args:
        project_path: Absolute path to the project.
        key: Unique key for this memory (e.g. "jdk_path", "build_command").
        value: The value to remember.
        tags: Optional tags for categorization.
        source: Who stored it ("ai", "user", "system").

    Returns:
        The stored memory entry.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))
    memories = _load_memories(project_path)

    now = time.time()
    existing = next((m for m in memories if m["key"] == key), None)

    if existing:
        existing["value"] = value
        existing["tags"] = tags or existing.get("tags", [])
        existing["updated_at"] = now
        existing["source"] = source
        entry = existing
    else:
        entry = {
            "id": str(uuid.uuid4()),
            "key": key,
            "value": value,
            "tags": tags or [],
            "project_path": project_path,
            "created_at": now,
            "updated_at": now,
            "access_count": 0,
            "last_accessed": now,
            "source": source,
        }
        memories.append(entry)

    _save_memories(project_path, memories)
    await _sync_to_chroma(project_path, memories)
    return entry


async def recall_memory(
    project_path: str,
    query: str | None = None,
    key: str | None = None,
    tags: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Recall memories by exact key, tag filter, or semantic query.

    Args:
        project_path: Absolute path to the project.
        query: Natural language query for semantic search.
        key: Exact key to look up.
        tags: Filter by tags (any match).
        top_k: Max results for semantic search.

    Returns:
        List of matching memory entries.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))
    memories = _load_memories(project_path)
    now = time.time()

    # Exact key lookup
    if key:
        results = [m for m in memories if m["key"] == key]
        for m in results:
            m["access_count"] = m.get("access_count", 0) + 1
            m["last_accessed"] = now
        if results:
            _save_memories(project_path, memories)
        return results

    # Tag filter
    if tags:
        tag_set = set(tags)
        results = [m for m in memories if tag_set & set(m.get("tags", []))]
        for m in results:
            m["access_count"] = m.get("access_count", 0) + 1
            m["last_accessed"] = now
        if results:
            _save_memories(project_path, memories)
        return results

    # Semantic search
    if query and memories:
        client = get_chroma_client()
        col_name = memory_collection_name(project_path)
        try:
            collection = client.get_collection(col_name)
        except Exception:
            # Collection doesn't exist yet — rebuild
            await _sync_to_chroma(project_path, memories)
            collection = client.get_collection(col_name)

        query_embedding = await embed_query(query)
        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["metadatas", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        # Map back to full memory entries
        matched_ids = set(results["ids"][0])
        matched = []
        for m in memories:
            if m["id"] in matched_ids:
                m["access_count"] = m.get("access_count", 0) + 1
                m["last_accessed"] = now
                matched.append(m)

        _save_memories(project_path, memories)
        # Return in order of ChromaDB results
        id_order = {mid: i for i, mid in enumerate(results["ids"][0])}
        matched.sort(key=lambda m: id_order.get(m["id"], 999))
        return matched

    # No filter — return all
    return memories


def list_memories(project_path: str) -> list[dict]:
    """List all memories for a project.

    Args:
        project_path: Absolute path to the project.

    Returns:
        List of all memory entries, sorted by last updated.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))
    memories = _load_memories(project_path)
    memories.sort(key=lambda m: m.get("updated_at", 0), reverse=True)
    return memories


async def delete_memory(project_path: str, key: str) -> bool:
    """Delete a memory by key.

    Args:
        project_path: Absolute path to the project.
        key: The key of the memory to delete.

    Returns:
        True if deleted, False if not found.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))
    memories = _load_memories(project_path)

    original_len = len(memories)
    deleted_ids = [m["id"] for m in memories if m["key"] == key]
    memories = [m for m in memories if m["key"] != key]

    if len(memories) == original_len:
        return False

    _save_memories(project_path, memories)

    # Remove from ChromaDB
    try:
        client = get_chroma_client()
        col_name = memory_collection_name(project_path)
        collection = client.get_collection(col_name)
        collection.delete(ids=deleted_ids)
    except Exception:
        pass

    return True


def get_memory_stats(project_path: str) -> dict:
    """Get memory statistics for a project.

    Args:
        project_path: Absolute path to the project.

    Returns:
        Dict with total_memories, tags, most_accessed, recent entries.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))
    memories = _load_memories(project_path)

    if not memories:
        return {
            "total_memories": 0,
            "tags": [],
            "most_accessed": [],
            "recent": [],
        }

    # Collect all tags
    all_tags: set[str] = set()
    for m in memories:
        all_tags.update(m.get("tags", []))

    # Most accessed
    by_access = sorted(memories, key=lambda m: m.get("access_count", 0), reverse=True)
    most_accessed = [
        {"key": m["key"], "access_count": m.get("access_count", 0)}
        for m in by_access[:5]
    ]

    # Recent
    by_time = sorted(memories, key=lambda m: m.get("updated_at", 0), reverse=True)
    recent = [
        {"key": m["key"], "value": m["value"][:100], "updated_at": m.get("updated_at", 0)}
        for m in by_time[:5]
    ]

    return {
        "total_memories": len(memories),
        "tags": sorted(all_tags),
        "most_accessed": most_accessed,
        "recent": recent,
    }
