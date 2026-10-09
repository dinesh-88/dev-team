"""The file tools an agent's model may call. This file has nothing to do with A2A.

Safety rules, enforced here and not left to the model:
- every path must stay inside one of the repos the agent was given
- writes are only allowed in repos marked writable (the coder's own worktree)
- there is no "run a command" tool: models never choose what gets executed
"""

import subprocess
from pathlib import Path

MAX_LINES = 400


class ToolError(Exception):
    pass


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


class Toolbox:
    def __init__(self, roots: dict[str, str], writable: tuple[str, ...] | list[str] = ()):
        self.roots = {name: Path(path).resolve() for name, path in roots.items()}
        self.writable = set(writable)
        self.changed: dict[str, set[str]] = {name: set() for name in self.writable}

    # ---- what the model is told it can do ---------------------------------

    def specs(self) -> list[dict]:
        repo = {"type": "string", "enum": sorted(self.roots), "description": "Repository name"}
        path = {"type": "string", "description": "Path relative to the repository root"}
        specs = [
            _tool("list_files", "List files in a repo (respects .gitignore).",
                  {"repo": repo, "path": path}, ["repo"]),
            _tool("read_file", f"Read a text file, up to {MAX_LINES} lines per call.",
                  {"repo": repo, "path": path,
                   "start_line": {"type": "integer", "description": "First line to read, from 1"}},
                  ["repo", "path"]),
            _tool("search", "Search file contents with a regular expression (git grep -E).",
                  {"repo": repo, "pattern": {"type": "string"}, "path": path}, ["repo", "pattern"]),
        ]
        if self.writable:
            specs += [
                _tool("write_file", "Create a file or replace its whole content.",
                      {"repo": repo, "path": path, "content": {"type": "string"}},
                      ["repo", "path", "content"]),
                _tool("edit_file", "Replace one exact piece of text in a file. "
                      "`old` must appear exactly once.",
                      {"repo": repo, "path": path, "old": {"type": "string"},
                       "new": {"type": "string"}}, ["repo", "path", "old", "new"]),
                _tool("delete_file", "Delete a file.", {"repo": repo, "path": path},
                      ["repo", "path"]),
            ]
        return specs

    def call(self, name: str, args: dict) -> str:
        """Run one tool. Problems come back as text so the model can react."""
        if name not in {spec["name"] for spec in self.specs()}:
            return f"Error: unknown tool '{name}'"
        try:
            return getattr(self, f"_{name}")(**args)
        except UnicodeDecodeError:
            return "Error: not a UTF-8 text file"
        except (ToolError, OSError, TypeError) as error:
            return f"Error: {error}"

    # ---- helpers ----------------------------------------------------------

    def _resolve(self, repo: str, path: str, write: bool = False) -> Path:
        if repo not in self.roots:
            raise ToolError(f"unknown repo '{repo}'. Available: {', '.join(sorted(self.roots))}")
        if write and repo not in self.writable:
            raise ToolError(f"repo '{repo}' is read-only for you")
        root = self.roots[repo]
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            raise ToolError("path is outside the repository")
        if ".git" in target.relative_to(root).parts:
            raise ToolError("the .git folder is off limits")
        return target

    def _git(self, repo: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.roots[repo], capture_output=True,
                              text=True, encoding="utf-8", errors="replace")

    @staticmethod
    def _capped(lines: list[str], hint: str) -> str:
        if len(lines) <= MAX_LINES:
            return "\n".join(lines)
        return "\n".join(lines[:MAX_LINES]) + f"\n... {len(lines) - MAX_LINES} more lines. {hint}"

    @staticmethod
    def _read(target: Path) -> tuple[str, bool]:
        """Return (text with \\n line endings, whether the file used \\r\\n)."""
        raw = target.read_bytes().decode("utf-8")
        return raw.replace("\r\n", "\n"), "\r\n" in raw

    def _save(self, repo: str, target: Path, text: str, crlf: bool) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
        self.changed[repo].add(target.relative_to(self.roots[repo]).as_posix())

    # ---- the tools --------------------------------------------------------

    def _list_files(self, repo: str, path: str = ".") -> str:
        self._resolve(repo, path)
        result = self._git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "--", path)
        lines = sorted(set(result.stdout.splitlines()))
        return self._capped(lines, "Pass a narrower path.") or "(no files)"

    def _read_file(self, repo: str, path: str, start_line: int = 1) -> str:
        text, _ = self._read(self._resolve(repo, path))
        lines = text.split("\n")[max(start_line, 1) - 1:]
        return self._capped(lines, f"Call again with start_line={max(start_line, 1) + MAX_LINES}.")

    def _search(self, repo: str, pattern: str, path: str = ".") -> str:
        self._resolve(repo, path)
        result = self._git(repo, "grep", "-n", "-I", "-E", "--untracked", "-e", pattern, "--", path)
        if result.returncode > 1:
            raise ToolError(result.stderr.strip())
        return self._capped(result.stdout.splitlines(), "Narrow the pattern or path.") or "(no matches)"

    def _write_file(self, repo: str, path: str, content: str) -> str:
        target = self._resolve(repo, path, write=True)
        crlf = target.is_file() and self._read(target)[1]
        self._save(repo, target, content, crlf)
        return f"Wrote {path}"

    def _edit_file(self, repo: str, path: str, old: str, new: str) -> str:
        target = self._resolve(repo, path, write=True)
        text, crlf = self._read(target)
        count = text.count(old)
        if count != 1:
            raise ToolError(f"`old` appears {count} times in {path}; it must appear exactly once")
        self._save(repo, target, text.replace(old, new), crlf)
        return f"Edited {path}"

    def _delete_file(self, repo: str, path: str) -> str:
        target = self._resolve(repo, path, write=True)
        target.unlink()
        self.changed[repo].add(target.relative_to(self.roots[repo]).as_posix())
        return f"Deleted {path}"
