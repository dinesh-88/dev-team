"""Small git helpers. This file has nothing to do with A2A.

The team never touches your checked-out files. For each repo it adds a git
WORKTREE: a second working folder on a new branch, backed by the same repo.
The agents edit and commit there; your own folder stays exactly as it was,
and the new branch simply appears in your repo.
"""

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def git(cwd: str | Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {cwd}:\n{result.stderr.strip()}")
    return result.stdout.strip()


def add_worktree(repo: str | Path, worktree: str | Path, branch: str, base: str = "HEAD") -> str:
    """Create `branch` from `base` in a new worktree folder. Returns the base commit."""
    base_commit = git(repo, "rev-parse", "--verify", f"{base}^{{commit}}")
    git(repo, "worktree", "add", "-b", branch, str(worktree), base_commit)
    return base_commit


def commit(worktree: str | Path, paths: list[str], message: str) -> str | None:
    """Commit exactly the given paths. Returns the new commit, or None if nothing changed."""
    if paths:
        git(worktree, "add", "-A", "--", *paths)
    if not git(worktree, "diff", "--cached", "--name-only"):
        return None
    git(worktree, "commit", "-m", message)
    return git(worktree, "rev-parse", "HEAD")


def diff(worktree: str | Path, base: str) -> str:
    """Everything committed on the branch since `base`."""
    return git(worktree, "diff", f"{base}..HEAD")


def remove_worktree(repo: str | Path, worktree: str | Path, branch: str, base: str) -> bool:
    """Remove the worktree folder. Drops the branch too if it has no commits.

    Returns True if the branch was kept.
    """
    git(repo, "worktree", "remove", "--force", str(worktree))
    if git(repo, "rev-parse", branch) == base:
        git(repo, "branch", "-D", branch)
        return False
    return True
