"""Python code chunker - splits by def/class boundaries using indentation."""

import re
from .base import BaseChunker, Chunk

PY_DECL = re.compile(r"^(class|def|async\s+def)\s+(\w+)", re.MULTILINE)


class PythonChunker(BaseChunker):
    language = "python"

    def chunk(self, content: str, file_path: str) -> list[Chunk]:
        lines = content.split("\n")
        if len(lines) <= 30:
            return [Chunk(content=content, file_path=file_path, start_line=1,
                          end_line=len(lines), language=self.language)]

        # Find top-level declarations (no indentation)
        decl_starts: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            m = PY_DECL.match(line)
            if m and not line[0].isspace():
                decl_starts.append((i, m.group(2)))

        if not decl_starts:
            return self._sliding_window(lines, file_path)

        chunks = []
        # Header (imports etc.)
        if decl_starts[0][0] > 0:
            header = "\n".join(lines[: decl_starts[0][0]])
            if header.strip():
                chunks.append(Chunk(
                    content=header, file_path=file_path,
                    start_line=1, end_line=decl_starts[0][0],
                    symbol_name="(imports)", language=self.language,
                ))

        for idx, (start, name) in enumerate(decl_starts):
            if idx + 1 < len(decl_starts):
                end = decl_starts[idx + 1][0] - 1
            else:
                end = len(lines) - 1
            chunk_content = "\n".join(lines[start: end + 1])
            chunks.append(Chunk(
                content=chunk_content, file_path=file_path,
                start_line=start + 1, end_line=end + 1,
                symbol_name=name, language=self.language,
            ))
        return chunks if chunks else self._sliding_window(lines, file_path)

    def _sliding_window(self, lines: list[str], file_path: str) -> list[Chunk]:
        chunks = []
        step, window = 40, 50
        for i in range(0, len(lines), step):
            end = min(i + window, len(lines))
            chunks.append(Chunk(
                content="\n".join(lines[i:end]), file_path=file_path,
                start_line=i + 1, end_line=end, language=self.language,
            ))
        return chunks
