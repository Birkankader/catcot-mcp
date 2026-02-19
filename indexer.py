"""Indexes project files: reads, chunks, embeds, stores in ChromaDB."""

import hashlib
import os
from pathlib import Path

from chunkers import Chunk, get_chunker
from config import CHROMA_DIR, collection_name, get_chroma_client
from embedder import embed_texts, get_provider_info

# Default ignore patterns
IGNORE_DIRS = {
    ".git", ".idea", ".vscode", "node_modules", "__pycache__",
    ".gradle", "build", "dist", "target", ".next", "venv", ".venv",
    ".mypy_cache", ".pytest_cache", ".tox", "vendor",
}

IGNORE_EXTENSIONS = {
    ".pyc", ".class", ".jar", ".war", ".o", ".so", ".dylib",
    ".exe", ".dll", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4",
    ".zip", ".tar", ".gz", ".lock", ".min.js", ".min.css",
}

MAX_FILE_SIZE = 500_000  # 500KB


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _should_ignore(path: Path) -> bool:
    for part in path.parts:
        if part in IGNORE_DIRS:
            return True
    if path.suffix in IGNORE_EXTENSIONS:
        return True
    return False


def _load_gitignore(project_path: Path) -> list[str]:
    """Load .gitignore patterns (basic support)."""
    gitignore = project_path / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    for line in gitignore.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _matches_gitignore(rel_path: str, patterns: list[str]) -> bool:
    """Basic gitignore matching."""
    for pattern in patterns:
        pattern = pattern.rstrip("/")
        if pattern in rel_path or rel_path.endswith(pattern):
            return True
    return False


def _collect_files(project_path: Path, gitignore_patterns: list[str]) -> list[Path]:
    """Collect all indexable files from project."""
    files = []
    for root, dirs, filenames in os.walk(project_path):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in filenames:
            fpath = Path(root) / fname
            if _should_ignore(fpath):
                continue
            rel = str(fpath.relative_to(project_path))
            if _matches_gitignore(rel, gitignore_patterns):
                continue
            if fpath.stat().st_size > MAX_FILE_SIZE:
                continue
            files.append(fpath)
    return files


async def index_project(project_path: str, reindex: bool = False) -> dict:
    """Index a project directory.

    Returns stats about the indexing operation.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))
    path = Path(project_path)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {project_path}")

    client = get_chroma_client()
    col_name = collection_name(project_path)

    if reindex:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    # Get current provider info
    provider = get_provider_info()

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={
            "project_path": project_path,
            "hnsw:space": "cosine",
            "embedding_provider": provider["name"],
            "embedding_model": provider["model"],
            "embedding_dimensions": provider["dimensions"],
        },
    )

    # Check for provider mismatch on existing collections
    col_meta = collection.metadata or {}
    stored_provider = col_meta.get("embedding_provider")

    if not reindex and collection.count() > 0:
        if stored_provider is None:
            # Legacy collection (no stored provider) — assume ollama
            stored_provider = "ollama"

        if stored_provider != provider["name"]:
            raise RuntimeError(
                f"Embedding provider mismatch: collection was indexed with '{stored_provider}' "
                f"but current provider is '{provider['name']}'. "
                f"Re-index the project to switch providers: reindex_project(\"{project_path}\")"
            )

    # Ensure metadata is written (get_or_create_collection ignores metadata for existing collections)
    # Note: hnsw:space cannot be changed after creation, so exclude it from modify
    collection.modify(metadata={
        "project_path": project_path,
        "embedding_provider": provider["name"],
        "embedding_model": provider["model"],
        "embedding_dimensions": provider["dimensions"],
    })

    # Load existing file hashes to skip unchanged files
    existing_hashes: dict[str, str] = {}
    if not reindex and collection.count() > 0:
        existing = collection.get(include=["metadatas"])
        if existing and existing["metadatas"]:
            for meta in existing["metadatas"]:
                if meta and "file_hash" in meta and "file_path" in meta:
                    existing_hashes[meta["file_path"]] = meta["file_hash"]

    gitignore_patterns = _load_gitignore(path)
    files = _collect_files(path, gitignore_patterns)

    stats = {"files_scanned": len(files), "files_indexed": 0, "files_skipped": 0, "chunks_created": 0}

    # Process files in batches
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_metas: list[dict] = []

    BATCH_SIZE = 20  # Embed 20 chunks at a time

    for fpath in files:
        try:
            content = fpath.read_text(errors="ignore")
        except Exception:
            continue

        fhash = _file_hash(content)
        rel_path = str(fpath.relative_to(path))

        # Skip if unchanged
        if rel_path in existing_hashes and existing_hashes[rel_path] == fhash:
            stats["files_skipped"] += 1
            continue

        # Remove old chunks for this file if re-indexing a changed file
        if rel_path in existing_hashes:
            try:
                collection.delete(where={"file_path": rel_path})
            except Exception:
                pass

        chunker = get_chunker(fpath.suffix)
        chunks = chunker.chunk(content, rel_path)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{hashlib.md5(rel_path.encode()).hexdigest()[:8]}_{i}"
            meta = {
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language or "",
                "symbol_name": chunk.symbol_name or "",
                "file_hash": fhash,
                "project_path": project_path,
            }
            batch_ids.append(chunk_id)
            batch_docs.append(chunk.content)
            batch_metas.append(meta)

        stats["files_indexed"] += 1
        stats["chunks_created"] += len(chunks)

        # Flush batch when large enough
        if len(batch_docs) >= BATCH_SIZE:
            embeddings = await embed_texts(batch_docs)
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=embeddings,
            )
            batch_ids, batch_docs, batch_metas = [], [], []

    # Flush remaining
    if batch_docs:
        embeddings = await embed_texts(batch_docs)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )

    return stats


def list_indexed_projects() -> list[dict]:
    """List all indexed projects."""
    client = get_chroma_client()
    collections = client.list_collections()
    projects = []
    for col in collections:
        meta = col.metadata or {}
        projects.append({
            "name": col.name,
            "project_path": meta.get("project_path", "unknown"),
            "chunks": col.count(),
            "embedding_provider": meta.get("embedding_provider", "ollama"),
            "embedding_model": meta.get("embedding_model", "unknown"),
        })
    return projects
