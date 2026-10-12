"""`devteam clean`: remove old runs, leftover worktrees and merged team branches.

No model is involved. What to delete follows plain rules (age, how many runs
to keep, whether a branch is merged), and deleting is exactly what a model
should not be left to decide.

Left behind by a run:
- its folder in .devteam/runs/ (plan, diffs, logs, summary)
- worktrees in that folder, if it ran with --keep or crashed. They are
  registered in the repo, so they are removed with git, not just deleted.
- its branch team/<request>-<time> in each repo it changed. That is your
  result to review, so only branches already merged are ever deleted.
"""

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from devteam import gitops, monitor

BUSY = timedelta(minutes=10)  # runs without status.json: busy if they wrote something this recently


@dataclass
class Run:
    path: Path
    started: datetime
    busy: bool
    size: int
    worktrees: list[Path]


def _started(path: Path) -> datetime:
    try:
        return datetime.strptime(path.name[:15], "%Y%m%d-%H%M%S")  # folder names start with it
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime)


def _size(path: Path) -> int:
    total = 0
    for folder, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(folder) / name).lstat().st_size
            except OSError:
                pass
    return total


def find_runs(folder: Path) -> list[Run]:
    """All runs in a .devteam/runs folder, oldest first."""
    if not folder.is_dir():
        return []
    runs = []
    for path in sorted(p for p in folder.iterdir() if p.is_dir()):
        status = monitor.read_status(path)
        if status:  # its process is alive and it has no summary yet
            busy = monitor.state(path, status) == "running"
        else:
            files = [path, *path.iterdir(), *(path / "logs").glob("*")]
            busy = datetime.now() - max(datetime.fromtimestamp(f.stat().st_mtime) for f in files) < BUSY
        worktrees = sorted(p for p in (path / "worktrees").glob("*") if p.is_dir())
        runs.append(Run(path, _started(path), busy, _size(path), worktrees))
    return runs


def pick(runs: list[Run], keep: int, older_than: int | None, everything: bool) -> list[Run]:
    """The runs to delete: all but the newest `keep`, optionally only old ones. Never busy ones."""
    idle = [run for run in runs if not run.busy]
    if not everything:
        idle = idle[:max(len(idle) - keep, 0)]
        if older_than is not None:
            idle = [run for run in idle if datetime.now() - run.started > timedelta(days=older_than)]
    return idle


def remove_worktree(path: Path) -> str:
    """Unregister and delete one worktree. Returns a warning, or ''."""
    warning = ""
    try:
        common = gitops.git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if gitops.git(path, "status", "--porcelain"):
            warning = f"{path.name}: uncommitted changes were discarded"
        gitops.git(Path(common).parent, "worktree", "remove", "--force", str(path))
    except (gitops.GitError, OSError):
        pass  # no longer a working worktree: deleting the folder is enough
    shutil.rmtree(path, ignore_errors=True)
    return warning


def remove_run(run: Run) -> list[str]:
    warnings = [w for w in (remove_worktree(path) for path in run.worktrees) if w]
    shutil.rmtree(run.path)
    return warnings


def team_branches(repo: dict) -> tuple[list[str], list[str]]:
    """(merged, unmerged) team/* branches of one repo, compared with its base branch."""
    def names(*args: str) -> list[str]:
        output = gitops.git(repo["path"], "branch", "--format=%(refname:short)", *args, "--list", "team/*")
        return output.splitlines()

    merged = names("--merged", repo["base"])
    return merged, [name for name in names() if name not in merged]


def delete_branches(repo: dict, branches: list[str]) -> tuple[int, list[str]]:
    """Delete merged branches with `git branch -d`, which refuses anything unmerged.

    Returns (how many were deleted, problems)."""
    deleted, problems = 0, []
    for branch in branches:
        try:
            gitops.git(repo["path"], "branch", "-d", branch)
            deleted += 1
        except gitops.GitError as error:
            problems.append(f"{repo['name']}: {branch} kept ({str(error).splitlines()[-1]})")
    return deleted, problems


def prune(repo: dict) -> None:
    """Forget worktrees whose folders were deleted by hand."""
    gitops.git(repo["path"], "worktree", "prune")


def human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""
