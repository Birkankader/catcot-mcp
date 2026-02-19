"""Catcot centralized configuration.

Single source of truth for paths, constants, and shared helpers
used across the codebase (indexer, searcher, topology, web, watcher, savings, memory).
"""

import hashlib
import os
from pathlib import Path

import chromadb

# ── Base directories ─────────────────────────────────────────────────
BASE_DIR = os.path.expanduser("~/.code-rag-mcp")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
SAVINGS_FILE = os.path.join(BASE_DIR, "savings.json")


# ── ChromaDB helpers ─────────────────────────────────────────────────

def collection_name(project_path: str) -> str:
    """Generate a stable collection name from project path."""
    h = hashlib.md5(project_path.encode()).hexdigest()[:12]
    # ChromaDB collection names: 3-63 chars, alphanumeric + underscores/hyphens
    base = Path(project_path).name.replace(" ", "_")[:30]
    return f"{base}_{h}"


def get_chroma_client() -> chromadb.ClientAPI:
    """Get a ChromaDB persistent client (single entry point)."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def memory_collection_name(project_path: str) -> str:
    """Generate a collection name for a project's memory store."""
    h = hashlib.md5(project_path.encode()).hexdigest()[:12]
    base = Path(project_path).name.replace(" ", "_")[:20]
    return f"memory_{base}_{h}"
