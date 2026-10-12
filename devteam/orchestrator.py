"""The orchestrator: an A2A CLIENT that runs planner -> coder(s) -> QA.

It knows the agents' URLs and your repos.toml. Everything else about the agents it
learns from their Agent Cards. For one request it:

  1. asks the Planner which repos must change, in which order, and what they
     must agree on (the "contract")
  2. opens one Coder conversation per repo, each in a fresh git worktree on a
     shared branch name. Repos that do not wait for each other are coded at
     the same time (up to `parallel` conversations at once).
  3. asks QA to run every repo's tests and review the change as a whole
  4. sends failures back to the right repo's Coder conversation, up to
     MAX_ROUNDS times

Every event an agent streams is also written to <run>/logs/<agent>.log, which
`devteam watch` follows in another terminal.

`devteam run "..."` starts the agents and calls run_team() for you.
"""

import asyncio
import json
import os
import re
import sys
import tomllib
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import (
    get_data_parts,
    get_message_text,
    get_text_parts,
    new_data_part,
    new_message,
    new_text_part,
)
from a2a.types import GetTaskRequest, Part, Role, SendMessageRequest, Task, TaskState

from devteam import gitops

MAX_ROUNDS = 3
STATE_DIR = ".devteam"  # created next to repos.toml: runs/<time>-<request>/ with plans and diffs
DIFF_LIMIT = 20000  # characters of another repo's diff passed to the next coder


class AgentError(RuntimeError):
    """An agent rejected the work or its task ended in a failed state."""


# ---- configuration -----------------------------------------------------------

def load_repos(config_path: str | Path) -> dict[str, dict]:
    """Read repos.toml into {name: settings}, with absolute paths."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        sys.exit(f"Config file not found: {config_path}\nCreate one with `devteam setup <folder>`, "
                 "or try `devteam demo`.")
    entries = tomllib.loads(config_path.read_text(encoding="utf-8")).get("repo", [])
    repos = {}
    for entry in entries:
        path = Path(entry["path"]).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        try:
            gitops.git(path, "rev-parse", "--git-dir")
        except (gitops.GitError, OSError):
            sys.exit(f"Repo '{entry['name']}': {path} is not a git repository.")
        repos[entry["name"]] = {
            "name": entry["name"],
            "path": str(path.resolve()),
            "description": entry.get("description", ""),
            "base": entry.get("base", "HEAD"),
            "setup": entry.get("setup", ""),
            "test": entry.get("test", ""),
            "timeout": entry.get("timeout", 900),
        }
    if len(repos) != len(entries) or not repos:
        sys.exit(f"{config_path} needs at least one [[repo]] and every name must be unique.")
    return repos


# ---- talking A2A ------------------------------------------------------------

class RunLog:
    """One log file per agent conversation, e.g. logs/coder-web.log."""

    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def write(self, who: str, text: str) -> None:
        body = text.rstrip().replace("\n", "\n" + " " * 10)
        path = self.folder / (who.replace(":", "-") + ".log")
        with path.open("a", encoding="utf-8") as file:
            file.write(f"{datetime.now():%H:%M:%S}  {body}\n")


class RunStatus:
    """<run>/status.json: where the run stands, read by `devteam status`."""

    def __init__(self, path: Path, **fields):
        self.path = path
        self.data = {"pid": os.getpid(), "started": datetime.now().isoformat(timespec="seconds"),
                     "phase": "starting", "repos": {}, **fields}
        self.save()

    def update(self, repos: dict[str, str] | None = None, **fields) -> None:
        self.data["repos"].update(repos or {})
        self.data.update(fields, updated=datetime.now().isoformat(timespec="seconds"))
        self.save()

    def save(self) -> None:
        temporary = self.path.with_suffix(".tmp")  # write, then rename: readers never see half a file
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def describe(parts, limit: int = 4000) -> str:
    """Text and data parts as readable text, for the logs."""
    lines = ["\n".join(get_text_parts([part])) for part in parts if part.HasField("text")]
    lines += [json.dumps(data, indent=2, ensure_ascii=False) for data in get_data_parts(parts)]
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters)"


async def send(client, who: str, parts: list[Part], context_id: str | None = None,
               log: RunLog | None = None) -> Task:
    """Send one message to an agent, print its live updates, return the finished Task."""
    message = new_message(parts, context_id=context_id, role=Role.ROLE_USER)
    task_id = None

    def note(text: str, show: bool = True) -> None:
        if show:
            print(f"  [{who}] {text}")
        if log:
            log.write(who, text)

    note(f"<< message from the orchestrator >>\n{describe(parts)}", show=False)
    # send_message() streams events back while the agent works.
    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("task"):  # the agent accepted the work and created a Task
            task_id = event.task.id
            note(f"task {task_id[:8]} created")
        elif event.HasField("status_update"):  # state changes and progress notes
            status = event.status_update.status
            state = TaskState.Name(status.state).removeprefix("TASK_STATE_").lower()
            text, kind = "", None
            if status.HasField("message"):
                text = get_message_text(status.message)
                kind = dict(status.message.metadata).get("kind")
            if kind == "text":  # the model talking: only in the log, it is long
                note(f"says: {text}", show=False)
            else:
                note(state + (f": {text}" if text else ""))
        elif event.HasField("artifact_update"):  # a piece of the result arrived
            artifact = event.artifact_update.artifact
            note(f"artifact: {artifact.name}")
            note(describe(artifact.parts), show=False)

    if task_id is None:
        raise AgentError(f"{who} did not create a task")

    # Fetch the final Task: its state plus all artifacts in one object.
    task = await client.get_task(GetTaskRequest(id=task_id))
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        reason = get_message_text(task.status.message) if task.status.HasField("message") else ""
        note(f"FAILED: {reason}", show=False)
        raise AgentError(f"{who} did not complete the task. {reason}")
    return task


async def all_of(jobs: list[Awaitable]) -> None:
    """Run jobs at the same time; when all are done, raise the first error if any."""
    for result in await asyncio.gather(*jobs, return_exceptions=True):
        if isinstance(result, BaseException):
            raise result


def data_artifact(task: Task, name: str) -> dict:
    """The JSON content of a named data artifact."""
    artifact = next(a for a in task.artifacts if a.name == name)
    return get_data_parts(artifact.parts)[0]


# ---- the workflow ------------------------------------------------------------

def slugify(text: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:6])[:40].strip("-") or "change"


def next_wave(remaining: list[dict]) -> list[dict]:
    """The planned repos whose `after` repos are all finished (not still remaining)."""
    waiting = {step["name"] for step in remaining}
    ready = [step for step in remaining
             if not (set(step.get("after", [])) - {step["name"]}) & waiting]
    return ready or remaining[:1]  # a dependency cycle: just take them in plan order


def coder_brief(request: str, plan: dict, step: dict, done: dict[str, dict]) -> str:
    sections = [
        f"Request:\n{request}",
        f"Plan summary:\n{plan['summary']}",
        f"Contract shared by all repositories:\n{plan['contract']}",
        f"Your steps in `{step['name']}`:\n{step['steps']}",
    ]
    for name, work in done.items():
        diff = gitops.diff(work["worktree"], work["base"])[:DIFF_LIMIT]
        sections.append(f"Already changed in `{name}`:\n```diff\n{diff}\n```")
    return "\n\n".join(sections)


def feedback(name: str, result: dict) -> str:
    lines = [f"QA checked the whole change. Fix the following in `{name}`."]
    if result["tests"] == "failed":
        lines.append(f"The tests failed:\n```\n{result['output']}\n```")
    if result["issues"]:
        lines.append("Review issues:\n" + "\n".join(f"- {issue}" for issue in result["issues"]))
    return "\n\n".join(lines)


async def run_team(request: str, config_path: str | Path, urls: dict[str, str],
                   keep_worktrees: bool = False, parallel: int = 3,
                   on_start: Callable[[Path], None] | None = None) -> bool:
    """`urls` has the address of the "planner", "coder" and "qa" agents.

    `parallel` is how many coder conversations may run at once. `on_start` is
    called with the run folder as soon as it exists.
    """
    repos = load_repos(config_path)
    stamp = datetime.now()
    branch = f"team/{slugify(request)}-{stamp:%H%M%S}"
    run_dir = (Path(config_path).resolve().parent / STATE_DIR / "runs"
               / f"{stamp:%Y%m%d-%H%M%S}-{slugify(request)}")
    run_dir.mkdir(parents=True)
    log = RunLog(run_dir / "logs")
    status = RunStatus(run_dir / "status.json", request=request, branch=branch, agents=urls)
    if on_start:
        on_start(run_dir)

    # Agents can take minutes, so no read timeout (httpx defaults to 5 seconds).
    http = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None))
    config = ClientConfig(httpx_client=http)
    # create_client() downloads the Agent Card from
    # <url>/.well-known/agent-card.json and picks a transport from it.
    planner = await create_client(urls["planner"], client_config=config)
    coder = await create_client(urls["coder"], client_config=config)
    qa = await create_client(urls["qa"], client_config=config)

    work: dict[str, dict] = {}  # repo name -> worktree, base, coder context, last change
    verdict = None
    try:
        print("\n1. Planner")
        status.update(phase="planning")
        task = await send(planner, "planner", [
            new_text_part(request),
            new_data_part({"repos": [
                {"name": r["name"], "path": r["path"], "description": r["description"]}
                for r in repos.values()
            ]}),
        ], log=log)
        plan = data_artifact(task, "plan")
        unknown = [step["name"] for step in plan["repos"] if step["name"] not in repos]
        if unknown:
            raise AgentError(f"The plan names repos that are not in the config: {unknown}")
        plan_text = (
            f"# Plan\n\n{plan['summary']}\n\n## Contract\n\n{plan['contract']}\n\n"
            + "\n\n".join(f"## {step['name']}\n\n"
                          + (f"After: {', '.join(step['after'])}\n\n" if step.get("after") else "")
                          + step["steps"] for step in plan["repos"])
        )
        (run_dir / "plan.md").write_text(plan_text + "\n", encoding="utf-8")
        status.update(repos={step["name"]: "waiting" for step in plan["repos"]})
        print(f"  -> {len(plan['repos'])} repo(s): {', '.join(s['name'] for s in plan['repos']) or 'none'}")
        if not plan["repos"]:
            print("\nThe planner found nothing to change.")
            return True

        print("\n2. Coder")
        limit = asyncio.Semaphore(max(1, parallel))

        async def code(name: str, parts: list[Part], context_id: str | None = None) -> None:
            doing = "fixing" if context_id else "coding"
            status.update(repos={name: f"queued for {doing}"})
            async with limit:
                status.update(repos={name: doing})
                try:
                    task = await send(coder, f"coder:{name}", parts, context_id, log)
                except BaseException:
                    status.update(repos={name: f"{doing} failed"})
                    raise
            status.update(repos={name: "fixed" if context_id else "coded"})
            work[name]["context_id"] = task.context_id  # one conversation per repo

        status.update(phase="coding")
        remaining = list(plan["repos"])
        while remaining:
            # A wave: every repo whose dependencies are done. They run together.
            wave = next_wave(remaining)
            remaining = [step for step in remaining if step not in wave]
            finished = dict(work)
            print(f"  -> now: {', '.join(step['name'] for step in wave)}")
            jobs = []
            for step in wave:
                name, repo = step["name"], repos[step["name"]]
                worktree = run_dir / "worktrees" / name
                base = gitops.add_worktree(repo["path"], worktree, branch, repo["base"])
                # The coder may READ every other repo: finished ones through their
                # worktree (so it sees the new code), the rest through your checkout.
                read_only = {other: finished[other]["worktree"] if other in finished
                             else repos[other]["path"] for other in repos if other != name}
                brief = coder_brief(request, plan, step, finished)
                work[name] = {"worktree": str(worktree), "base": base}
                jobs.append(code(name, [
                    new_text_part(brief),
                    new_data_part({"repo": name, "worktree": str(worktree), "base": base,
                                   "read_only": read_only}),
                ]))
            await all_of(jobs)

        qa_brief = (f"Request:\n{request}\n\nPlan summary:\n{plan['summary']}\n\n"
                    f"Contract shared by all repositories:\n{plan['contract']}")
        qa_repos = [
            {"name": name, "worktree": w["worktree"], "base": w["base"],
             "setup": repos[name]["setup"], "test": repos[name]["test"],
             "timeout": repos[name]["timeout"]}
            for name, w in work.items()
        ]
        qa_context = None
        for round_number in range(1, MAX_ROUNDS + 1):
            print(f"\n3. QA (round {round_number})")
            status.update(phase=f"testing (round {round_number} of {MAX_ROUNDS})",
                          repos={name: "testing" for name in work})
            # Reusing QA's context_id tells it the worktrees are already set up.
            task = await send(qa, "qa", [new_text_part(qa_brief),
                                         new_data_part({"repos": qa_repos})], qa_context, log)
            qa_context = task.context_id
            verdict = data_artifact(task, "verdict")
            for name, result in verdict["repos"].items():
                print(f"  -> {name}: tests {result['tests']}, {len(result['issues'])} review issue(s)")
                status.update(repos={name: f"tests {result['tests']}, "
                                           f"{len(result['issues'])} review issue(s)"})
            if verdict["passed"] or round_number == MAX_ROUNDS:
                break

            print("\n2. Coder (fix)")
            status.update(phase=f"fixing (after round {round_number})")
            # Same context_id: each coder remembers the plan and its own work.
            await all_of([
                code(name, [new_text_part(feedback(name, result))], work[name]["context_id"])
                for name, result in verdict["repos"].items()
                if result["tests"] == "failed" or result["issues"]
            ])
    except BaseException as error:  # also Ctrl+C
        status.update(phase="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        await asyncio.gather(planner.close(), coder.close(), qa.close(), return_exceptions=True)
        report = finish(run_dir, repos, work, branch, verdict, keep_worktrees)
        (run_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
        if status.data["phase"] != "stopped":
            status.update(phase="QA passed" if verdict and verdict["passed"]
                          else "QA did not pass" if verdict else "nothing to change")
        print("\n" + report)

    return bool(verdict and verdict["passed"])


async def survey(url: str, folder: str | Path, out: str | Path, hints: str = "",
                 only: list[str] | None = None) -> Path:
    """Ask the Surveyor at `url` to describe the repos in `folder` and save its repos.toml to `out`.

    An existing file at `out` is kept as `<out>.bak`. Nothing is executed:
    the suggested setup/test commands only run later, in QA, after you review them.
    """
    http = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None))
    surveyor = await create_client(url, client_config=ClientConfig(httpx_client=http))
    try:
        print(f"\nSurveyor: {Path(folder).expanduser().resolve()}")
        data = {"folder": str(folder)} | ({"only": only} if only else {})
        text = "Describe these repositories for repos.toml."
        if hints:
            text += f"\n\nHints from the user:\n{hints}"
        task = await send(surveyor, "surveyor", [new_text_part(text), new_data_part(data)])
    finally:
        await surveyor.close()

    repos = data_artifact(task, "repos")["repos"]
    artifact = next(a for a in task.artifacts if a.name == "repos.toml")
    out = Path(out)
    if out.exists():
        backup = out.with_name(out.name + ".bak")
        out.replace(backup)
        print(f"\nThe previous {out.name} is now {backup.name}.")
    out.write_text(artifact.parts[0].text, encoding="utf-8")

    print(f"\nWrote {len(repos)} repo(s) to {out}:")
    for repo in repos:
        print(f"  {repo['name']:<20} {repo['path']}")
        print(f"  {'':<20} setup: {repo['setup'] or '-'}   test: {repo['test'] or '-'}")
        if repo["notes"]:
            print(f"  {'':<20} check: {repo['notes']}")
    print("\nReview it before `devteam run`: QA executes the setup and test commands.")
    return out


def finish(run_dir: Path, repos: dict, work: dict, branch: str, verdict: dict | None,
           keep_worktrees: bool) -> str:
    """Save the diffs, remove the worktrees, and describe what is left in each repo."""
    passed = bool(verdict and verdict["passed"])
    lines = ["RESULT: " + ("QA passed" if passed else "QA did NOT pass" if verdict
                           else "stopped before QA finished")]
    for name, w in work.items():
        path, base = repos[name]["path"], w["base"]
        diff = gitops.diff(w["worktree"], base)
        (run_dir / f"{name}.diff").write_text(diff + "\n", encoding="utf-8")
        kept = True
        if not keep_worktrees:
            kept = gitops.remove_worktree(path, w["worktree"], branch, base)
        result = verdict["repos"].get(name) if verdict else None
        status = f"tests {result['tests']}" if result else "not checked"
        if not kept:
            lines.append(f"\n{name}: no changes were committed, branch removed")
            continue
        lines.append(f"\n{name}: branch {branch} ({status})")
        lines.append(f"  review:  git -C \"{path}\" diff {base[:10]}..{branch}")
        if keep_worktrees:
            lines.append(f"  worktree kept at {w['worktree']}")
        for issue in (result or {}).get("issues", []):
            lines.append(f"  open issue: {issue}")
        if result and result["tests"] == "failed":
            lines.append("  last test output:\n    " + result["output"][-1500:].replace("\n", "\n    "))
    lines.append(f"\nPlan, diffs and this summary: {run_dir}")
    return "\n".join(lines)

