"""Catcot file watcher — auto-indexes files on save using watchdog."""

import asyncio
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    # Define dummy base class if watchdog is not available
    class FileSystemEventHandler:  # type: ignore
        pass

from catcot.chunkers import get_chunker, Chunk
from catcot.config import collection_name, get_chroma_client
from catcot.core.embedder import embed_texts, get_provider_info
from catcot.core.indexer import (
    IGNORE_DIRS,
    IGNORE_EXTENSIONS,
    _file_hash,
    _should_ignore,
)


class _WatcherState:
    """Thread-safe state for the watcher."""
    
    def __init__(self):
        self.watched_projects: dict[str, "Observer"] = {}  # type: ignore
        self.pending_files: dict[str, dict] = {}  # file_path -> {project_path, last_modified}
        self.lock = threading.Lock()
        self.debounce_timer: Optional[threading.Timer] = None
        self.debounce_delay = 2.0  # seconds


_watcher_state = _WatcherState()


async def _index_single_file(project_path: str, file_path: str) -> dict:
    """Re-index a single file. Returns stats dict.
    
    Handles chunking, embedding, and upserting to ChromaDB.
    Removes old chunks for the file first.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))
    file_path = os.path.abspath(os.path.expanduser(file_path))
    
    path_obj = Path(file_path)
    if not path_obj.exists():
        # File was deleted
        return {"status": "file_deleted", "file_path": file_path}
    
    if _should_ignore(path_obj):
        return {"status": "ignored", "file_path": file_path}
    
    if not path_obj.is_file():
        return {"status": "not_a_file", "file_path": file_path}
    
    if path_obj.stat().st_size > 500_000:  # MAX_FILE_SIZE
        return {"status": "too_large", "file_path": file_path}
    
    try:
        content = path_obj.read_text(errors="ignore")
    except Exception as e:
        return {"status": "read_error", "file_path": file_path, "error": str(e)}
    
    client = get_chroma_client()
    col_name = collection_name(project_path)
    
    try:
        collection = client.get_collection(col_name)
    except Exception:
        return {
            "status": "project_not_indexed",
            "file_path": file_path,
            "message": "Project not indexed. Run index_project first.",
        }

    # Check provider mismatch
    col_meta = collection.metadata or {}
    stored_provider = col_meta.get("embedding_provider")
    current_provider = get_provider_info()["name"]
    if stored_provider and stored_provider != current_provider:
        import sys
        sys.stderr.write(
            f"[Catcot Watcher] Skipping {file_path}: collection uses '{stored_provider}' "
            f"but current provider is '{current_provider}'. Reindex to fix.\n"
        )
        return {
            "status": "provider_mismatch",
            "file_path": file_path,
            "message": f"Provider mismatch: {stored_provider} vs {current_provider}",
        }

    # Calculate relative path
    try:
        rel_path = str(Path(file_path).relative_to(project_path))
    except ValueError:
        return {
            "status": "not_in_project",
            "file_path": file_path,
            "message": f"File is not in project directory {project_path}",
        }
    
    # Remove old chunks for this file
    try:
        collection.delete(where={"file_path": rel_path})
    except Exception:
        pass
    
    # Chunk the file
    chunker = get_chunker(path_obj.suffix)
    try:
        chunks = chunker.chunk(content, rel_path)
    except Exception as e:
        return {
            "status": "chunk_error",
            "file_path": file_path,
            "error": str(e),
        }
    
    if not chunks:
        return {
            "status": "no_chunks",
            "file_path": file_path,
        }
    
    # Embed chunks
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_metas: list[dict] = []
    
    file_hash = _file_hash(content)
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{hashlib.md5(rel_path.encode()).hexdigest()[:8]}_{i}"
        meta = {
            "file_path": chunk.file_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "language": chunk.language or "",
            "symbol_name": chunk.symbol_name or "",
            "file_hash": file_hash,
            "project_path": project_path,
        }
        batch_ids.append(chunk_id)
        batch_docs.append(chunk.content)
        batch_metas.append(meta)
    
    try:
        embeddings = await embed_texts(batch_docs)
    except Exception as e:
        return {
            "status": "embed_error",
            "file_path": file_path,
            "error": str(e),
        }
    
    try:
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )
    except Exception as e:
        return {
            "status": "upsert_error",
            "file_path": file_path,
            "error": str(e),
        }
    
    return {
        "status": "success",
        "file_path": file_path,
        "chunks_indexed": len(chunks),
    }


def _process_pending_files():
    """Process all pending files (debounce callback)."""
    with _watcher_state.lock:
        pending = dict(_watcher_state.pending_files)
        _watcher_state.pending_files.clear()
    
    if not pending:
        return
    
    # Run async operations in an isolated loop (don't pollute global state)
    try:
        for file_path, info in pending.items():
            project_path = info["project_path"]
            asyncio.run(_index_single_file(project_path, file_path))
    except Exception as e:
        import sys
        sys.stderr.write(f"[Catcot Watcher] Error processing {file_path}: {e}\n")


class _FileWatcherHandler(FileSystemEventHandler):
    """Handles file system events for a watched project."""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
    
    def on_modified(self, event):  # type: ignore
        if event.is_directory:
            return
        
        file_path = event.src_path
        path_obj = Path(file_path)
        
        # Skip ignored files
        if _should_ignore(path_obj):
            return
        
        with _watcher_state.lock:
            _watcher_state.pending_files[file_path] = {
                "project_path": self.project_path,
                "timestamp": time.time(),
            }
            
            # Restart debounce timer
            if _watcher_state.debounce_timer is not None:
                _watcher_state.debounce_timer.cancel()
            
            _watcher_state.debounce_timer = threading.Timer(
                _watcher_state.debounce_delay,
                _process_pending_files,
            )
            _watcher_state.debounce_timer.daemon = True
            _watcher_state.debounce_timer.start()
    
    def on_created(self, event):  # type: ignore
        # Treat as modified
        self.on_modified(event)
    
    def on_deleted(self, event):  # type: ignore
        """Remove chunks for deleted file."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        project_path = self.project_path
        
        try:
            rel_path = str(Path(file_path).relative_to(project_path))
        except ValueError:
            return
        
        client = get_chroma_client()
        col_name = collection_name(project_path)
        
        try:
            collection = client.get_collection(col_name)
            collection.delete(where={"file_path": rel_path})
        except Exception:
            pass


def start_watching(project_path: str) -> str:
    """Start watching a project directory for file changes.
    
    When files are modified or created, Catcot automatically re-indexes them.
    Uses debouncing (2 second delay) to avoid excessive re-indexing.
    
    Args:
        project_path: Absolute path to the project directory to watch.
    
    Returns:
        Status message.
    
    Raises:
        ValueError: If watchdog is not available or path is invalid.
    """
    if not WATCHDOG_AVAILABLE:
        raise ValueError(
            "watchdog library is required for watch mode. "
            "Install with: pip install watchdog>=3.0.0"
        )
    
    project_path = os.path.abspath(os.path.expanduser(project_path))
    
    if not os.path.isdir(project_path):
        raise ValueError(f"Not a directory: {project_path}")
    
    with _watcher_state.lock:
        if project_path in _watcher_state.watched_projects:
            return f"[Catcot Watcher] Already watching: {project_path}"
        
        # Create observer
        observer = Observer()  # type: ignore
        handler = _FileWatcherHandler(project_path)
        observer.schedule(handler, project_path, recursive=True)
        observer.start()
        
        _watcher_state.watched_projects[project_path] = observer
    
    return f"[Catcot Watcher] Now watching: {project_path}"


def stop_watching(project_path: str) -> str:
    """Stop watching a project directory.
    
    Args:
        project_path: Absolute path to the project directory to stop watching.
    
    Returns:
        Status message.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))
    
    with _watcher_state.lock:
        if project_path not in _watcher_state.watched_projects:
            return f"[Catcot Watcher] Not currently watching: {project_path}"
        
        observer = _watcher_state.watched_projects.pop(project_path)
        observer.stop()
        observer.join(timeout=5)
    
    return f"[Catcot Watcher] Stopped watching: {project_path}"


def list_watched() -> list[str]:
    """List all currently watched projects.
    
    Returns:
        List of project paths being watched.
    """
    with _watcher_state.lock:
        return list(_watcher_state.watched_projects.keys())


def stop_all():
    """Stop watching all projects. Used for cleanup."""
    with _watcher_state.lock:
        for project_path, observer in _watcher_state.watched_projects.items():
            observer.stop()
        for observer in _watcher_state.watched_projects.values():
            observer.join(timeout=5)
        _watcher_state.watched_projects.clear()
        
        if _watcher_state.debounce_timer is not None:
            _watcher_state.debounce_timer.cancel()
            _watcher_state.debounce_timer = None
