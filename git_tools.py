"""Git integration utilities for Catcot MCP server.

Provides non-blocking git operations for searching modified files,
reviewing diffs, and tracking recent changes.
"""

import asyncio
import os
from dataclasses import dataclass, field


@dataclass
class GitStatus:
    """Parsed git status output."""
    staged: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def all_changed(self) -> list[str]:
        """All files that have any kind of change."""
        seen = set()
        result = []
        for f in self.staged + self.modified + self.untracked:
            if f not in seen:
                seen.add(f)
                result.append(f)
        return result


async def _run_git(project_path: str, *args: str) -> tuple[int, str, str]:
    """Run a git command asynchronously in the given project directory.

    Returns (returncode, stdout, stderr).
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=project_path,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace").strip(),
        stderr.decode("utf-8", errors="replace").strip(),
    )


async def is_git_repo(project_path: str) -> bool:
    """Check if the given path is inside a git repository."""
    try:
        rc, _, _ = await _run_git(project_path, "rev-parse", "--is-inside-work-tree")
        return rc == 0
    except Exception:
        return False


async def get_git_root(project_path: str) -> str | None:
    """Get the root directory of the git repository."""
    try:
        rc, stdout, _ = await _run_git(project_path, "rev-parse", "--show-toplevel")
        return stdout if rc == 0 else None
    except Exception:
        return None


async def get_status(project_path: str) -> GitStatus:
    """Parse git status --porcelain to get file change categories."""
    rc, stdout, _ = await _run_git(project_path, "status", "--porcelain")
    if rc != 0:
        return GitStatus()

    status = GitStatus()
    for line in stdout.splitlines():
        if len(line) < 3:
            continue
        index_status = line[0]
        worktree_status = line[1]
        filepath = line[3:].strip()

        # Handle renames: "R  old -> new"
        if " -> " in filepath:
            filepath = filepath.split(" -> ")[1]

        if index_status in ("A", "M", "R", "C"):
            status.staged.append(filepath)
        if worktree_status == "M":
            status.modified.append(filepath)
        elif worktree_status == "D" or index_status == "D":
            status.deleted.append(filepath)
        elif index_status == "?" and worktree_status == "?":
            status.untracked.append(filepath)

    return status


async def get_modified_files(project_path: str, commits: int = 5) -> list[str]:
    """Get a combined list of modified files from git status and recent commits.

    Combines:
    1. Working directory changes (git status)
    2. Files changed in the last N commits
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    if not await is_git_repo(project_path):
        raise ValueError(f"Not a git repository: {project_path}")

    # Get current working directory changes
    status = await get_status(project_path)
    all_files = set(status.all_changed)

    # Get files changed in recent commits
    try:
        rc, stdout, _ = await _run_git(
            project_path, "diff", "--name-only", f"HEAD~{commits}..HEAD"
        )
        if rc == 0 and stdout:
            for f in stdout.splitlines():
                f = f.strip()
                if f:
                    all_files.add(f)
    except Exception:
        # HEAD~N might not exist (fewer than N commits); try log instead
        try:
            rc, stdout, _ = await _run_git(
                project_path, "log", "--oneline", f"-{commits}",
                "--name-only", "--format=",
            )
            if rc == 0 and stdout:
                for f in stdout.splitlines():
                    f = f.strip()
                    if f:
                        all_files.add(f)
        except Exception:
            pass

    return sorted(all_files)


async def get_diff(project_path: str, staged: bool = False) -> str:
    """Get the git diff output.

    Args:
        project_path: Path to the project directory.
        staged: If True, get staged diff (--staged). Otherwise, unstaged changes.
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    if not await is_git_repo(project_path):
        raise ValueError(f"Not a git repository: {project_path}")

    args = ["diff"]
    if staged:
        args.append("--staged")

    rc, stdout, stderr = await _run_git(project_path, *args)
    if rc != 0:
        return f"Error getting diff: {stderr}"

    if not stdout:
        # Try combined diff (staged + unstaged) if unstaged was empty
        if not staged:
            rc2, stdout2, _ = await _run_git(project_path, "diff", "HEAD")
            if rc2 == 0 and stdout2:
                return stdout2
        return "(No changes found)"

    return stdout


async def get_recent_changes(project_path: str, commits: int = 5) -> dict:
    """Get recent commit history with changed files.

    Returns dict with:
        - commits: list of {hash, message, files}
        - all_files: combined list of all changed files
    """
    project_path = os.path.abspath(os.path.expanduser(project_path))

    if not await is_git_repo(project_path):
        raise ValueError(f"Not a git repository: {project_path}")

    rc, stdout, _ = await _run_git(
        project_path,
        "log", f"-{commits}", "--format=%H|%s", "--name-only",
    )
    if rc != 0 or not stdout:
        return {"commits": [], "all_files": []}

    commits_list = []
    all_files = set()
    current_commit = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        if "|" in line and len(line.split("|")[0]) == 40:
            parts = line.split("|", 1)
            current_commit = {
                "hash": parts[0][:8],
                "message": parts[1] if len(parts) > 1 else "",
                "files": [],
            }
            commits_list.append(current_commit)
        elif current_commit is not None:
            current_commit["files"].append(line)
            all_files.add(line)

    return {
        "commits": commits_list,
        "all_files": sorted(all_files),
    }


async def get_file_diff(project_path: str, file_path: str) -> str:
    """Get the diff for a specific file."""
    project_path = os.path.abspath(os.path.expanduser(project_path))

    # Try unstaged, then staged, then HEAD
    for args in [
        ["diff", "--", file_path],
        ["diff", "--staged", "--", file_path],
        ["diff", "HEAD", "--", file_path],
    ]:
        rc, stdout, _ = await _run_git(project_path, *args)
        if rc == 0 and stdout:
            return stdout

    return "(No diff available for this file)"
