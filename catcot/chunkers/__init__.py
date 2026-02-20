"""Code chunkers package.

Provides language-specific code chunkers for splitting files into meaningful
chunks. Prefers tree-sitter AST-based chunking when available, falling back
to regex-based chunkers.
"""

from .base import BaseChunker, Chunk
from .kotlin_chunker import KotlinChunker
from .python_chunker import PythonChunker
from .java_chunker import JavaChunker
from .sql_chunker import SqlChunker
from .js_ts_chunker import JsTsChunker
from .generic_chunker import GenericChunker

# Toggle tree-sitter usage. Set to False to always use regex chunkers.
USE_TREESITTER: bool = True

# Regex-based fallback map
REGEX_EXTENSION_MAP: dict[str, type[BaseChunker]] = {
    ".kt": KotlinChunker, ".kts": KotlinChunker,
    ".py": PythonChunker, ".java": JavaChunker,
    ".sql": SqlChunker, ".js": JsTsChunker,
    ".jsx": JsTsChunker, ".ts": JsTsChunker, ".tsx": JsTsChunker,
}

# Full extension map (same keys, used for compatibility)
EXTENSION_MAP: dict[str, type[BaseChunker]] = dict(REGEX_EXTENSION_MAP)

# Check tree-sitter availability once at import time
_treesitter_available: bool = False
if USE_TREESITTER:
    try:
        from .treesitter_chunker import (
            TreeSitterChunker,
            is_tree_sitter_available,
            supports_extension,
        )
        _treesitter_available = is_tree_sitter_available()
    except ImportError:
        _treesitter_available = False


def get_chunker(extension: str) -> BaseChunker:
    """Get the best available chunker for the given file extension.

    Prefers tree-sitter AST chunking when available, falls back to
    regex-based chunkers, and finally to the generic sliding-window chunker.
    """
    # Try tree-sitter first
    if USE_TREESITTER and _treesitter_available:
        try:
            from .treesitter_chunker import TreeSitterChunker, supports_extension
            if supports_extension(extension):
                return TreeSitterChunker(extension)
        except (ImportError, ValueError, Exception):
            pass  # Fall through to regex chunkers

    # Fall back to regex-based chunkers
    cls = REGEX_EXTENSION_MAP.get(extension, GenericChunker)
    return cls()


__all__ = [
    "BaseChunker",
    "Chunk",
    "get_chunker",
    "EXTENSION_MAP",
    "USE_TREESITTER",
]
