"""Searches indexed code using vector similarity."""

import hashlib
import os
from pathlib import Path

import chromadb

from embedder import embed_query, get_provider_info

CHROMA_DIR = os.path.expanduser("~/.code-rag-mcp/chroma_db")


def _collection_name(project_path: str) -> str:
    h = hashlib.md5(project_path.encode()).hexdigest()[:12]
    base = Path(project_path).name.replace(" ", "_")[:30]
    return f"{base}_{h}"


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_DIR)


async def search_code(
    query: str,
    project_path: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Search indexed code for relevant chunks.

    If project_path is None, searches all indexed projects.
    Returns list of results with file_path, start_line, end_line, content, score.
    """
    client = _get_client()
    query_embedding = await embed_query(query)

    collections_to_search = []
    if project_path:
        project_path = os.path.abspath(os.path.expanduser(project_path))
        col_name = _collection_name(project_path)
        try:
            col = client.get_collection(col_name)
            collections_to_search.append(col)
        except Exception:
            raise ValueError(
                f"Project not indexed: {project_path}. Run index_project first."
            )
    else:
        collections_to_search = client.list_collections()

    # Filter out collections indexed with a different provider
    current_provider = get_provider_info()["name"]
    all_results = []
    for col in collections_to_search:
        if col.count() == 0:
            continue
        col_meta = col.metadata or {}
        col_provider = col_meta.get("embedding_provider")
        if col_provider and col_provider != current_provider:
            import sys
            proj = col_meta.get("project_path", col.name)
            sys.stderr.write(
                f"[Catcot] Skipping '{proj}': indexed with '{col_provider}', "
                f"current provider is '{current_provider}'. Reindex to fix.\n"
            )
            continue
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                similarity = 1 - (distance / 2)
                all_results.append({
                    "file_path": meta.get("file_path", ""),
                    "start_line": meta.get("start_line", 0),
                    "end_line": meta.get("end_line", 0),
                    "symbol_name": meta.get("symbol_name", ""),
                    "language": meta.get("language", ""),
                    "content": doc,
                    "similarity": round(similarity, 4),
                    "project_path": meta.get("project_path", ""),
                })

    # Sort by similarity descending, take top_k
    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    return all_results[:top_k]
