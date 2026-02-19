"""Kotlin code chunker - splits by class, fun, object, interface boundaries."""

import re
from .base import BaseChunker, Chunk

# Matches top-level and nested declarations
KOTLIN_DECL = re.compile(
    r"^(?:(?:public|private|protected|internal|abstract|open|data|sealed|inline|value|annotation|override|suspend)\s+)*"
    r"(?:class|interface|object|fun|val|var)\s+",
    re.MULTILINE,
)


class KotlinChunker(BaseChunker):
    language = "kotlin"

    def chunk(self, content: str, file_path: str) -> list[Chunk]:
        lines = content.split("\n")
        if len(lines) <= 30:
            return [Chunk(content=content, file_path=file_path, start_line=1,
                          end_line=len(lines), language=self.language)]

        declaration_starts: list[int] = []
        for i, line in enumerate(lines):
            # Only match top-level declarations (no leading whitespace)
            if line and not line[0].isspace() and KOTLIN_DECL.match(line):
                declaration_starts.append(i)

        if not declaration_starts:
            return self._sliding_window(lines, file_path)

        chunks = []
        # Package/import header
        if declaration_starts[0] > 0:
            header = "\n".join(lines[: declaration_starts[0]])
            if header.strip():
                chunks.append(Chunk(
                    content=header, file_path=file_path,
                    start_line=1, end_line=declaration_starts[0],
                    symbol_name="(imports)", language=self.language,
                ))

        for idx, start in enumerate(declaration_starts):
            end = declaration_starts[idx + 1] - 1 if idx + 1 < len(declaration_starts) else len(lines) - 1
            # Find the end by brace matching from start
            end = self._find_block_end(lines, start, end)
            chunk_lines = lines[start: end + 1]
            symbol = self._extract_name(lines[start])
            chunks.append(Chunk(
                content="\n".join(chunk_lines), file_path=file_path,
                start_line=start + 1, end_line=end + 1,
                symbol_name=symbol, language=self.language,
            ))
        return chunks if chunks else self._sliding_window(lines, file_path)

    def _find_block_end(self, lines: list[str], start: int, max_end: int) -> int:
        depth = 0
        for i in range(start, max_end + 1):
            depth += lines[i].count("{") - lines[i].count("}")
            if depth <= 0 and i > start:
                return i
        return max_end

    def _extract_name(self, line: str) -> str | None:
        m = re.search(r"(?:class|interface|object|fun)\s+(\w+)", line)
        return m.group(1) if m else None

    def _sliding_window(self, lines: list[str], file_path: str) -> list[Chunk]:
        chunks = []
        step = 40
        window = 50
        for i in range(0, len(lines), step):
            end = min(i + window, len(lines))
            chunks.append(Chunk(
                content="\n".join(lines[i:end]), file_path=file_path,
                start_line=i + 1, end_line=end, language=self.language,
            ))
        return chunks
