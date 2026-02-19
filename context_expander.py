"""Catcot context expander — get surrounding lines for a code chunk."""

import os
from pathlib import Path


def get_chunk_context(
    file_path: str,
    start_line: int,
    end_line: int,
    context_before: int = 15,
    context_after: int = 15,
) -> dict:
    """Read a file and return the chunk with surrounding context lines.
    
    Args:
        file_path: Absolute path to the source file
        start_line: 1-based start line of the chunk
        end_line: 1-based end line of the chunk
        context_before: Number of lines to include before the chunk
        context_after: Number of lines to include after the chunk
    
    Returns:
        dict with:
            - content: The full text with context and chunk markers
            - actual_start_line: First line number in the output (1-based)
            - actual_end_line: Last line number in the output (1-based)
            - chunk_start_line: Line number where chunk actually starts (in output)
            - chunk_end_line: Line number where chunk actually ends (in output)
            - file_path: The file path
    
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If line numbers are invalid
    """
    file_path = os.path.abspath(os.path.expanduser(file_path))
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Validate line numbers
    if start_line < 1 or end_line < 1:
        raise ValueError("Line numbers must be >= 1")
    if start_line > end_line:
        raise ValueError(f"start_line ({start_line}) cannot be > end_line ({end_line})")
    
    # Read the file
    try:
        content = Path(file_path).read_text(errors="ignore")
    except Exception as e:
        raise ValueError(f"Cannot read file: {e}")
    
    lines = content.splitlines(keepends=False)
    total_lines = len(lines)
    
    # Validate against file length
    if start_line > total_lines:
        raise ValueError(
            f"start_line ({start_line}) exceeds file length ({total_lines})"
        )
    
    # Calculate the expanded range (clamped to file boundaries)
    expanded_start = max(1, start_line - context_before)
    expanded_end = min(total_lines, end_line + context_after)
    
    # Extract the lines (convert to 0-based indexing)
    expanded_lines = lines[expanded_start - 1 : expanded_end]
    
    # Build output with markers
    output_lines = []
    for i, line in enumerate(expanded_lines):
        line_num = expanded_start + i
        if line_num == start_line and line_num != end_line:
            output_lines.append(f">>> CHUNK START >>> {line}")
        elif line_num == end_line and line_num != start_line:
            output_lines.append(f"<<< CHUNK END <<<   {line}")
        elif line_num == start_line and line_num == end_line:
            output_lines.append(f">>> CHUNK START >>> {line} <<< CHUNK END <<<")
        elif start_line < line_num < end_line:
            output_lines.append(f"                   {line}")
        else:
            output_lines.append(f"                   {line}")
    
    # Calculate which lines in the output are the chunk
    chunk_start_in_output = start_line - expanded_start + 1
    chunk_end_in_output = end_line - expanded_start + 1
    
    return {
        "content": "\n".join(output_lines),
        "actual_start_line": expanded_start,
        "actual_end_line": expanded_end,
        "chunk_start_line": chunk_start_in_output,
        "chunk_end_line": chunk_end_in_output,
        "file_path": file_path,
    }
