"""Tree-sitter AST-based code chunker.

Provides accurate, AST-aware code splitting that handles edge cases like
decorators, nested classes, multi-line signatures, and export statements.
Falls back to regex-based chunkers if tree-sitter is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChunker, Chunk

logger = logging.getLogger(__name__)

# Language config: maps language name to (tree-sitter language key, top-level node types)
LANGUAGE_CONFIG: dict[str, dict[str, Any]] = {
    "python": {
        "ts_lang": "python",
        "top_level_types": {
            "function_definition",
            "class_definition",
            "decorated_definition",
        },
        "wrapper_types": {"decorated_definition"},
        "extension": ".py",
    },
    "javascript": {
        "ts_lang": "javascript",
        "top_level_types": {
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
            "variable_declaration",
        },
        "wrapper_types": {"export_statement"},
        "extension": ".js",
    },
    "typescript": {
        "ts_lang": "typescript",
        "top_level_types": {
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
            "variable_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        },
        "wrapper_types": {"export_statement"},
        "extension": ".ts",
    },
    "tsx": {
        "ts_lang": "tsx",
        "top_level_types": {
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
            "variable_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        },
        "wrapper_types": {"export_statement"},
        "extension": ".tsx",
    },
    "java": {
        "ts_lang": "java",
        "top_level_types": {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "method_declaration",
            "constructor_declaration",
            "annotation_type_declaration",
        },
        "wrapper_types": set(),
        "extension": ".java",
    },
    "kotlin": {
        "ts_lang": "kotlin",
        "top_level_types": {
            "class_declaration",
            "object_declaration",
            "function_declaration",
            "property_declaration",
        },
        "wrapper_types": set(),
        "extension": ".kt",
    },
    "sql": {
        "ts_lang": "sql",
        "top_level_types": {
            "create_table_statement",
            "create_view_statement",
            "create_function_statement",
            "select_statement",
            "insert_statement",
            "update_statement",
            "delete_statement",
        },
        "wrapper_types": set(),
        "extension": ".sql",
    },
}

# Map file extensions to language config keys
EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".sql": "sql",
}


def _try_import_tree_sitter():
    """Try to import tree-sitter and language pack.

    Returns (tree_sitter_module, get_language_func) or raises ImportError.
    """
    import tree_sitter
    from tree_sitter_language_pack import get_language
    return tree_sitter, get_language


class TreeSitterChunker(BaseChunker):
    """Universal tree-sitter based chunker for multiple languages.

    Uses AST parsing to accurately identify code boundaries including
    decorators, annotations, export wrappers, and multi-line signatures.
    """

    def __init__(self, extension: str):
        self._extension = extension
        lang_key = EXTENSION_TO_LANG.get(extension)
        if not lang_key or lang_key not in LANGUAGE_CONFIG:
            raise ValueError(f"No tree-sitter config for extension: {extension}")
        self._config = LANGUAGE_CONFIG[lang_key]
        self.language = lang_key

    def chunk(self, content: str, file_path: str) -> list[Chunk]:
        """Parse content with tree-sitter and extract top-level declarations."""
        tree_sitter, get_language = _try_import_tree_sitter()

        lines = content.split("\n")

        # Small files: return as single chunk
        if len(lines) <= 30:
            return [Chunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                language=self.language,
            )]

        # Parse with tree-sitter
        ts_lang = get_language(self._config["ts_lang"])
        parser = tree_sitter.Parser(ts_lang)
        tree = parser.parse(content.encode("utf-8"))
        root = tree.root_node

        if root.has_error:
            logger.debug(
                "Tree-sitter parse had errors for %s, continuing with partial AST",
                file_path,
            )

        top_level_types = self._config["top_level_types"]
        chunks: list[Chunk] = []

        # Collect top-level declarations
        declarations = self._collect_declarations(root, top_level_types, lines)

        if not declarations:
            return self._sliding_window(lines, file_path)

        # Header/imports: lines before first declaration
        first_decl_start = declarations[0]["start_line"]
        if first_decl_start > 0:
            header_lines = lines[:first_decl_start]
            header_content = "\n".join(header_lines)
            if header_content.strip():
                chunks.append(Chunk(
                    content=header_content,
                    file_path=file_path,
                    start_line=1,
                    end_line=first_decl_start,
                    symbol_name="(imports)",
                    language=self.language,
                ))

        for decl in declarations:
            start = decl["start_line"]
            end = decl["end_line"]
            name = decl["name"]

            chunk_content = "\n".join(lines[start:end + 1])
            chunks.append(Chunk(
                content=chunk_content,
                file_path=file_path,
                start_line=start + 1,  # 1-indexed
                end_line=end + 1,      # 1-indexed
                symbol_name=name,
                language=self.language,
            ))

        # Trailing lines not covered by declarations
        last_decl_end = declarations[-1]["end_line"]
        if last_decl_end < len(lines) - 1:
            trailing_lines = lines[last_decl_end + 1:]
            trailing_content = "\n".join(trailing_lines)
            if trailing_content.strip():
                chunks.append(Chunk(
                    content=trailing_content,
                    file_path=file_path,
                    start_line=last_decl_end + 2,
                    end_line=len(lines),
                    symbol_name="(trailing)",
                    language=self.language,
                ))

        return chunks if chunks else self._sliding_window(lines, file_path)

    def _collect_declarations(
        self,
        root,
        top_level_types: set[str],
        lines: list[str],
    ) -> list[dict]:
        """Walk the AST root children and collect top-level declaration info."""
        declarations = []
        wrapper_types = self._config.get("wrapper_types", set())

        for node in root.children:
            if node.type not in top_level_types:
                continue

            start_line = node.start_point[0]
            end_line = node.end_point[0]
            name = self._extract_name(node, wrapper_types)

            declarations.append({
                "start_line": start_line,
                "end_line": end_line,
                "name": name,
                "type": node.type,
            })

        # Sort by start line
        declarations.sort(key=lambda d: d["start_line"])

        # Merge overlapping declarations
        merged = []
        for decl in declarations:
            if merged and decl["start_line"] <= merged[-1]["end_line"] + 1:
                prev = merged[-1]
                prev["end_line"] = max(prev["end_line"], decl["end_line"])
                if decl["name"] and not prev["name"]:
                    prev["name"] = decl["name"]
            else:
                merged.append(decl)

        return merged

    def _extract_name(self, node, wrapper_types: set[str]) -> str:
        """Extract the symbol name from a declaration node."""
        target = node
        if node.type in wrapper_types:
            for child in node.children:
                if child.type not in wrapper_types and child.is_named:
                    target = child
                    break

        # Look for a name/identifier child
        for child in target.children:
            if child.type in (
                "identifier", "name", "property_identifier", "type_identifier",
            ):
                text = child.text
                return text.decode("utf-8") if isinstance(text, bytes) else text

        # Check one level deeper
        for child in target.children:
            if child.is_named:
                for grandchild in child.children:
                    if grandchild.type in ("identifier", "name"):
                        text = grandchild.text
                        return text.decode("utf-8") if isinstance(text, bytes) else text

        return node.type

    def _sliding_window(self, lines: list[str], file_path: str) -> list[Chunk]:
        """Fallback sliding window chunking."""
        chunks = []
        step, window = 40, 50
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


def is_tree_sitter_available() -> bool:
    """Check if tree-sitter and language pack are installed."""
    try:
        _try_import_tree_sitter()
        return True
    except ImportError:
        return False


def supports_extension(extension: str) -> bool:
    """Check if tree-sitter chunking is available for the given file extension."""
    return extension in EXTENSION_TO_LANG
