"""`devteam status`: how a run is doing. No model involved.

It reads what the run leaves in its folder, and asks the operating system and
the agents themselves:
- status.json  phase, each repo's state, the agents' URLs, the run's process id
- logs/*.log   each conversation's latest activity and how long ago it was
- the process  is the run still alive, or did it crash or get stopped?
- the agents   does each one still answer at /.well-known/agent-card.json?
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import httpx

QUIET = timedelta(minutes=5)  # a working conversation silent this long gets a warning
CARD_PATH = "/.well-known/agent-card.json"


def _ago(moment: datetime) -> str:
    seconds = int((datetime.now() - moment).total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s ago"
    if seconds < 86400:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m ago"
    return f"{seconds // 86400}d ago"


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)  # signal 0 only checks that the process exists
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _answers(url: str) -> bool:
    try:
        return httpx.get(url + CARD_PATH, timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


def read_status(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}  # runs from before status.json existed, or a half-created run


def state(run_dir: Path, status: dict) -> str:
    """running, finished or stopped (the process is gone without a summary)."""
    if (run_dir / "summary.md").exists():
        return "finished"
    return "running" if _alive(status.get("pid")) else "stopped"


def conversations(run_dir: Path) -> list[tuple[str, datetime, str]]:
    """(name, time of last activity, last activity) per log file."""
    found = []
    for path in sorted((run_dir / "logs").glob("*.log")):
        with path.open("rb") as file:
            file.seek(max(path.stat().st_size - 16384, 0))
            lines = file.read().decode("utf-8", "replace").splitlines()
        # Entries start with a time; indented lines continue the entry above.
        entries = [line[10:] for line in lines if line[:2].isdigit()]
        last = entries[-1] if entries else ""
        found.append((path.stem, datetime.fromtimestamp(path.stat().st_mtime), last))
    return found


def overview(runs: list[Path]) -> str:
    lines = []
    for run_dir in reversed(runs):
        status = read_status(run_dir)
        now = state(run_dir, status)
        phase = status.get("phase", "")
        lines.append(f"{now:<9} {phase[:28]:<28} {run_dir.name}")
    return "\n".join(lines) or "No runs yet."


def report(run_dir: Path) -> str:
    status = read_status(run_dir)
    now = state(run_dir, status)
    lines = [f"Run      {run_dir.name}"]
    if status.get("request"):
        lines.append(f"Request  {status['request']}")
    if status.get("branch"):
        lines.append(f"Branch   {status['branch']}")
    if status.get("started"):
        lines.append(f"Started  {_ago(datetime.fromisoformat(status['started']))}")
    detail = status.get("phase", "")
    if now == "stopped" and status.get("phase") != "stopped":
        detail += " (the process ended without finishing: crashed or killed)"
    lines.append(f"State    {now}" + (f": {detail}" if detail else ""))
    if status.get("error"):
        lines.append(f"Error    {status['error']}")

    if now == "running" and status.get("agents"):
        agents = [f"{name} {'up' if _answers(url) else 'NOT ANSWERING'}"
                  for name, url in status["agents"].items()]
        lines.append(f"Agents   {' · '.join(agents)}")

    if status.get("repos"):
        lines.append("\nRepos")
        width = max(map(len, status["repos"]))
        lines += [f"  {name:<{width}}  {what}" for name, what in status["repos"].items()]

    talks = conversations(run_dir)
    if talks:
        lines.append("\nLatest activity")
        width = max(len(name) for name, _, _ in talks)
        warnings = []
        for name, moment, last in talks:
            lines.append(f"  {name:<{width}}  {_ago(moment):>10}  {last[:90]}")
            done = last.startswith(("completed", "FAILED", "failed"))
            if now == "running" and not done and datetime.now() - moment > QUIET:
                warnings.append(f"  ! {name} has been quiet for {_ago(moment)[:-4]}. Check it "
                                f"with `devteam watch {name.replace('-', ':', 1)}`.")
        lines += [""] + warnings if warnings else []

    if now == "finished":
        lines.append(f"\nSummary: {run_dir / 'summary.md'}")
    return "\n".join(lines)
