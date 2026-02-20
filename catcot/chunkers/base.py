"""Base chunker interface."""

from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    language: str | None = None


class BaseChunker:
    """Base class for language-specific code chunkers."""

    language: str = "unknown"

    def chunk(self, content: str, file_path: str) -> list[Chunk]:
        """Split file content into meaningful chunks."""
        raise NotImplementedError
