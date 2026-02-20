"""JavaScript/TypeScript code chunker - splits by function/class/export boundaries."""

import re
from .base import BaseChunker, Chunk

JS_DECL = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?\s+(\w+)|class\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=)",
    re.MULTILINE,
)


class JsTsChunker(BaseChunker):
    language = "javascript"

    def chunk(self, content: str, file_path: str) -> list[Chunk]:
        if file_path.endswith((".ts", ".tsx")):
            self.language = "typescript"

        lines = content.split("\n")
        if len(lines) <= 30:
            return [Chunk(content=content, file_path=file_path, start_line=1,
                          end_line=len(lines), language=self.language)]

        decl_starts: list[tuple[int, str | None]] = []
        for i, line in enumerate(lines):
            m = JS_DECL.match(line)
            if m:
                name = m.group(1) or m.group(2) or m.group(3)
                decl_starts.append((i, name))

        if not decl_starts:
            return self._sliding_window(lines, file_path)

        chunks = []
        if decl_starts[0][0] > 0:
            header = "\n".join(lines[: decl_starts[0][0]])
            if header.strip():
                chunks.append(Chunk(
                    content=header, file_path=file_path,
                    start_line=1, end_line=decl_starts[0][0],
                    symbol_name="(imports)", language=self.language,
                ))

        for idx, (start, name) in enumerate(decl_starts):
            max_end = decl_starts[idx + 1][0] - 1 if idx + 1 < len(decl_starts) else len(lines) - 1
            end = self._find_block_end(lines, start, max_end)
            chunks.append(Chunk(
                content="\n".join(lines[start: end + 1]), file_path=file_path,
                start_line=start + 1, end_line=end + 1,
                symbol_name=name, language=self.language,
            ))
        return chunks if chunks else self._sliding_window(lines, file_path)

    def _find_block_end(self, lines: list[str], start: int, max_end: int) -> int:
        depth = 0
        for i in range(start, max_end + 1):
            depth += lines[i].count("{") - lines[i].count("}")
            if depth <= 0 and i > start:
                return i
        return max_end

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
