"""SQL code chunker - splits by statement boundaries."""

import re
from .base import BaseChunker, Chunk

SQL_STMT = re.compile(
    r"^\s*(?:CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT|WITH|GRANT|REVOKE|BEGIN|COMMIT|MERGE)\b",
    re.IGNORECASE | re.MULTILINE,
)


class SqlChunker(BaseChunker):
    language = "sql"

    def chunk(self, content: str, file_path: str) -> list[Chunk]:
        lines = content.split("\n")
        if len(lines) <= 30:
            return [Chunk(content=content, file_path=file_path, start_line=1,
                          end_line=len(lines), language=self.language)]

        stmt_starts: list[int] = []
        for i, line in enumerate(lines):
            if SQL_STMT.match(line):
                stmt_starts.append(i)

        if not stmt_starts:
            return [Chunk(content=content, file_path=file_path, start_line=1,
                          end_line=len(lines), language=self.language)]

        chunks = []
        # Header (comments etc.)
        if stmt_starts[0] > 0:
            header = "\n".join(lines[: stmt_starts[0]])
            if header.strip():
                chunks.append(Chunk(
                    content=header, file_path=file_path,
                    start_line=1, end_line=stmt_starts[0],
                    symbol_name="(header)", language=self.language,
                ))

        for idx, start in enumerate(stmt_starts):
            end = stmt_starts[idx + 1] - 1 if idx + 1 < len(stmt_starts) else len(lines) - 1
            chunk_content = "\n".join(lines[start: end + 1])
            # Extract statement type + object name
            name_match = re.match(
                r"\s*(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT)\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|FUNCTION|PROCEDURE|INDEX|TYPE|TRIGGER)?\s*(\w+)?",
                lines[start], re.IGNORECASE,
            )
            name = None
            if name_match:
                name = f"{name_match.group(1).upper()}"
                if name_match.group(2):
                    name += f" {name_match.group(2)}"
            chunks.append(Chunk(
                content=chunk_content, file_path=file_path,
                start_line=start + 1, end_line=end + 1,
                symbol_name=name, language=self.language,
            ))
        return chunks
