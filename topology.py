"""Project topology analysis for Catcot MCP server.

Generates a semantic map of project components by analyzing embeddings
stored in ChromaDB, grouping related files into logical components,
and identifying inter-component relationships.
"""

import math
import os
from collections import defaultdict

from config import collection_name, get_chroma_client


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _average_embedding(embeddings: list[list[float]]) -> list[float]:
    """Compute the element-wise average of a list of embeddings."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    avg = [0.0] * dim
    for emb in embeddings:
        for i in range(dim):
            avg[i] += emb[i]
    n = len(embeddings)
    return [x / n for x in avg]


def _find_components(
    file_embeddings: dict[str, list[float]],
    similarity_threshold: float = 0.7,
) -> list[set[str]]:
    """Group files into components using single-linkage clustering.

    Files with cosine similarity > threshold are grouped together.
    Uses union-find for efficient clustering.
    """
    files = list(file_embeddings.keys())
    n = len(files)

    if n == 0:
        return []
    if n == 1:
        return [{files[0]}]

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare all pairs and union if similar enough
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(
                file_embeddings[files[i]], file_embeddings[files[j]]
            )
            if sim >= similarity_threshold:
                union(i, j)

    # Group by root
    groups: dict[int, set[str]] = defaultdict(set)
    for i in range(n):
        root = find(i)
        groups[root].add(files[i])

    return list(groups.values())


def _label_component(files: set[str], file_symbols: dict[str, list[str]]) -> str:
    """Auto-generate a label for a component based on common path elements and symbols."""
    if not files:
        return "unknown"

    # Try common directory prefix
    paths = [f.split("/") for f in files]
    if all(len(p) > 1 for p in paths):
        common_dirs = set(paths[0][:-1])
        for p in paths[1:]:
            common_dirs &= set(p[:-1])
        if common_dirs:
            deepest = max(
                common_dirs,
                key=lambda d: max(
                    i for p in paths for i, part in enumerate(p) if part == d
                ),
            )
            if deepest and deepest != ".":
                return deepest

    # Single file: use its basename
    basenames = [f.rsplit("/", 1)[-1].rsplit(".", 1)[0] for f in files]
    if len(basenames) == 1:
        return basenames[0]

    # Use the most common symbol names
    all_symbols = []
    for f in files:
        all_symbols.extend(file_symbols.get(f, []))
    if all_symbols:
        symbol_counts: dict[str, int] = defaultdict(int)
        for s in all_symbols:
            if s and s not in ("(imports)", "(trailing)", "chunk"):
                symbol_counts[s] += 1
        if symbol_counts:
            top_symbol = max(symbol_counts, key=symbol_counts.get)  # type: ignore[arg-type]
            return top_symbol

    # Fallback: first filename
    return sorted(files)[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]


async def generate_project_map(project_path: str) -> dict:
    """Generate a semantic map of project components and relationships.

    Steps:
    1. Retrieve all chunks and embeddings from ChromaDB
    2. Compute file-level embeddings (average of chunk embeddings)
    3. Cluster files by semantic similarity
    4. Label components and find relationships
    5. Return structured project map

    Returns dict with:
        - project_path: str
        - total_files: int
        - total_chunks: int
        - components: list of {label, files, symbols, directory}
        - relationships: list of {source, target, similarity}
        - directory_structure: dict tree of directories
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    client = get_chroma_client()
    col_name = collection_name(project_path)

    try:
        collection = client.get_collection(col_name)
    except Exception:
        raise ValueError(
            f"Project not indexed: {project_path}. Run index_project first."
        )

    count = collection.count()
    if count == 0:
        raise ValueError(
            f"Project has no indexed chunks: {project_path}. Run index_project first."
        )

    # Fetch all data from the collection
    all_data = collection.get(
        include=["documents", "metadatas", "embeddings"],
        limit=count,
    )

    if not all_data or not all_data["ids"]:
        raise ValueError("Failed to retrieve data from ChromaDB.")

    # Organize by file: collect embeddings and symbols per file
    file_chunk_embeddings: dict[str, list[list[float]]] = defaultdict(list)
    file_symbols: dict[str, list[str]] = defaultdict(list)
    file_languages: dict[str, str] = {}
    total_chunks = len(all_data["ids"])

    for i in range(total_chunks):
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        embedding = all_data["embeddings"][i] if all_data["embeddings"] else None

        file_path = meta.get("file_path", "")
        symbol_name = meta.get("symbol_name", "")
        language = meta.get("language", "")

        if not file_path or embedding is None:
            continue

        file_chunk_embeddings[file_path].append(embedding)
        if symbol_name:
            file_symbols[file_path].append(symbol_name)
        if language:
            file_languages[file_path] = language

    # Compute file-level embeddings (average of chunk embeddings)
    file_embeddings: dict[str, list[float]] = {}
    for fp, embs in file_chunk_embeddings.items():
        file_embeddings[fp] = _average_embedding(embs)

    total_files = len(file_embeddings)

    # Cluster files into components
    components_sets = _find_components(file_embeddings, similarity_threshold=0.7)

    # Build component details
    components = []
    component_embeddings: dict[int, list[float]] = {}

    for idx, file_set in enumerate(
        sorted(components_sets, key=len, reverse=True)
    ):
        label = _label_component(file_set, file_symbols)
        files_sorted = sorted(file_set)

        # Collect all symbols in this component
        comp_symbols = set()
        for f in file_set:
            for s in file_symbols.get(f, []):
                if s not in ("(imports)", "(trailing)"):
                    comp_symbols.add(s)

        # Determine the primary directory
        dirs: dict[str, int] = defaultdict(int)
        for f in file_set:
            parts = f.split("/")
            if len(parts) > 1:
                dirs[parts[0]] += 1
            else:
                dirs["."] += 1
        primary_dir = max(dirs, key=dirs.get) if dirs else "."  # type: ignore[arg-type]

        # Component-level embedding
        comp_embs = [
            file_embeddings[f] for f in file_set if f in file_embeddings
        ]
        if comp_embs:
            component_embeddings[idx] = _average_embedding(comp_embs)

        languages = set()
        for f in file_set:
            if f in file_languages:
                languages.add(file_languages[f])

        components.append({
            "id": idx,
            "label": label,
            "files": files_sorted,
            "file_count": len(files_sorted),
            "symbols": sorted(comp_symbols)[:20],
            "directory": primary_dir,
            "languages": sorted(languages),
        })

    # Find inter-component relationships
    relationships = []
    comp_indices = list(component_embeddings.keys())
    for i_idx in range(len(comp_indices)):
        for j_idx in range(i_idx + 1, len(comp_indices)):
            ci = comp_indices[i_idx]
            cj = comp_indices[j_idx]
            sim = _cosine_similarity(
                component_embeddings[ci], component_embeddings[cj]
            )
            if sim >= 0.5:
                ci_label = next(c["label"] for c in components if c["id"] == ci)
                cj_label = next(c["label"] for c in components if c["id"] == cj)
                relationships.append({
                    "source": ci_label,
                    "target": cj_label,
                    "similarity": round(sim, 4),
                })

    relationships.sort(key=lambda r: r["similarity"], reverse=True)

    # Build directory structure tree
    dir_tree: dict = {}
    for fp in file_embeddings.keys():
        parts = fp.split("/")
        current = dir_tree
        for part in parts[:-1]:
            if part not in current or current[part] is None:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = None  # leaf file

    return {
        "project_path": project_path,
        "total_files": total_files,
        "total_chunks": total_chunks,
        "components": components,
        "relationships": relationships[:20],
        "directory_structure": dir_tree,
    }
