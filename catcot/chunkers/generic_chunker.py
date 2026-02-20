"""Generic fallback chunker - sliding window based."""

from .base import BaseChunker, Chunk


class GenericChunker(BaseChunker):
    language = "unknown"

    def chunk(self, content: str, file_path: str) -> list[Chunk]:
        lines = content.split("\n")
        if len(lines) <= 50:
            return [Chunk(content=content, file_path=file_path, start_line=1,
                          end_line=len(lines), language=self.language)]

        chunks = []
        step = 40
        window = 50
        for i in range(0, len(lines), step):
            end = min(i + window, len(lines))
            chunks.append(Chunk(
                content="\n".join(lines[i:end]),
                file_path=file_path,
                start_line=i + 1,
                end_line=end,
                language=self.language,
            ))
        return chunks
