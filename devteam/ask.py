"""`devteam ask`: the agents YOU choose, on the repos and branches you choose.

    devteam ask qa web@feature/login                 test and check one branch
    devteam ask review api#42                        review GitHub pull request 42 of `api`
    devteam ask review,qa,bench api@perf-cache       review and QA together, then bench
    devteam ask coder web@feature/x -m "Fix the failing date test"
    devteam ask planner api web -m "How would we add SSO?"

`devteam run` sends one request through planner -> coder -> QA, always in that
order. Here you pick the agents, and the ones you name together run at the
same time, each in worktrees of its own. Only `bench` waits for the others and
runs alone, so that their tests do not slow down its measurements.

A target is a repo name from repos.toml, plus what to look at:
    web              the commit checked out in your repo (HEAD)
    web@feature/x    a branch, tag or commit
    web#42           GitHub pull request 42 (uses the `gh` command, fetches from origin)
It is compared with its base: --base, else `base` from repos.toml, else origin's
default branch (or main/master), from the point where the branch left it.
No target means every repo in repos.toml, at HEAD.

Your branches are never moved. Agents that only read and test get DETACHED
worktrees at the commit. The coder, the one that writes, commits to a new
branch team/<message>-<time> that starts at the target.
"""

import asyncio
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_data_part, new_text_part

from devteam import gitops
from devteam.orchestrator import (STATE_DIR, AgentError, RunLog, RunStatus, data_artifact,
                                  plan_markdown, send, slugify, text_artifact)

AGENTS = ["planner", "coder", "qa", "review", "bench"]


@dataclass
class Target:
    name: str
    repo: dict  # its entry from repos.toml
    label: str  # as you typed it, e.g. web@feature/x
    head: str  # the commit to look at
    base: str  # the commit it is compared with
    base_label: str  # the base branch, for people
    pr: dict | None = None  # title, body, url of a pull request


# ---- what to look at ---------------------------------------------------------

def pull_request(repo: dict, number: str) -> tuple[str, str, dict]:
    """Fetch pull request `number` from origin. Returns (head commit, base commit, PR info)."""
    try:
        result = subprocess.run(
            ["gh", "pr", "view", number, "--json", "title,body,url,baseRefName"],
            cwd=repo["path"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise AgentError("Pull requests need the GitHub CLI: install `gh`, then `gh auth login`.")
    if result.returncode != 0:
        raise AgentError(f"gh pr view {number} failed in {repo['name']}: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    gitops.git(repo["path"], "fetch", "-q", "origin", f"pull/{number}/head")
    head = gitops.commit_of(repo["path"], "FETCH_HEAD")
    gitops.git(repo["path"], "fetch", "-q", "origin", info["baseRefName"])
    return head, gitops.commit_of(repo["path"], "FETCH_HEAD"), info


def resolve_targets(specs: list[str], repos: dict[str, dict], base: str | None) -> list[Target]:
    """Turn `web`, `web@branch` and `web#42` into commits. Fails before any agent starts."""
    targets = []
    for spec in specs or list(repos):
        name, number, ref = spec, None, "HEAD"
        if "#" in spec:
            name, number = spec.split("#", 1)
        elif "@" in spec:
            name, ref = spec.split("@", 1)
        if name not in repos:
            raise AgentError(f"'{name}' is not in repos.toml. Repos: {', '.join(repos)}")
        if any(t.name == name for t in targets):
            raise AgentError(f"'{name}' is named twice: one branch per repo at a time")
        repo, path = repos[name], repos[name]["path"]
        pr = None
        if number:
            head, base_tip, pr = pull_request(repo, number)
            base_label = base or pr["baseRefName"]
            if base:
                base_tip = gitops.commit_of(path, base)
        else:
            head = gitops.commit_of(path, ref)
            base_label = base or (repo["base"] if repo["base"] != "HEAD" else gitops.default_branch(path))
            base_tip = gitops.commit_of(path, base_label)
        targets.append(Target(name, repo, spec, head, gitops.merge_base(path, base_tip, head),
                              base_label, pr))
    return targets


def brief(message: str, targets: list[Target]) -> str:
    lines = [f"- `{t.name}`: {t.label} ({t.head[:10]}), compared with {t.base_label} "
             f"from {t.base[:10]}" for t in targets]
    sections = [message] if message else []
    sections.append("What to look at:\n" + "\n".join(lines))
    for t in targets:
        if t.pr:
            sections.append(f"Pull request in `{t.name}`: {t.pr['title']}\n{t.pr['url']}\n\n"
                            f"{t.pr['body'] or '(no description)'}")
    return "\n\n".join(sections)


# ---- reports -----------------------------------------------------------------

def qa_markdown(verdict: dict) -> str:
    lines = ["## QA: " + ("passed" if verdict["passed"] else "did NOT pass")]
    for name, result in verdict["repos"].items():
        lines += ["", f"### {name}: tests {result['tests']}"]
        lines += [f"- {issue}" for issue in result["issues"]]
        if result["tests"] == "failed":
            lines += ["", "```", result["output"][-3000:], "```"]
    return "\n".join(lines)


# ---- running the agents ------------------------------------------------------

async def ask(agents: list[str], targets: list[Target], message: str, config_path: str | Path,
              urls: dict[str, str], keep_worktrees: bool = False, threshold: float = 5.0,
              on_start: Callable[[Path], None] | None = None) -> bool:
    """Run `agents` at the same time on `targets`. True if every agent is satisfied."""
    stamp = datetime.now()
    what = " ".join(t.label for t in targets)
    run_dir = (Path(config_path).resolve().parent / STATE_DIR / "runs"
               / f"{stamp:%Y%m%d-%H%M%S}-{'-'.join(agents)}-{slugify(what)}")
    run_dir.mkdir(parents=True)
    log = RunLog(run_dir / "logs")
    status = RunStatus(run_dir / "status.json", agents=urls,
                       request=f"{', '.join(agents)} on {what}" + (f": {message}" if message else ""),
                       repos={f"{agent}:{t.name}": "waiting" for agent in agents for t in targets})
    if on_start:
        on_start(run_dir)
    for t in targets:
        diff = gitops.git(t.repo["path"], "diff", f"{t.base}..{t.head}")
        (run_dir / f"{t.name}.diff").write_text(diff + "\n", encoding="utf-8")
        if t.head == t.base:
            print(f"  note: {t.label} has no commits that {t.base_label} does not have. "
                  "Name a branch (web@feature-x) or a different --base.")

    detached: list[tuple[str, Path]] = []  # (repo, worktree) to remove at the end
    branches: list[tuple[str, Path, str, str]] = []  # the coder's: (repo, worktree, branch, start)

    def checkout(t: Target, folder: str, commit: str) -> str:
        path = run_dir / "worktrees" / folder
        gitops.add_detached(t.repo["path"], path, commit)
        detached.append((t.repo["path"], path))
        return str(path)

    def mark(agent: str, text: str, names: list[str] | None = None) -> None:
        status.update(repos={f"{agent}:{n}": text for n in names or [t.name for t in targets]})

    def save(name: str, text: str) -> Path:
        (run_dir / name).write_text(text + "\n", encoding="utf-8")
        return run_dir / name

    # One coroutine per agent. Each returns (satisfied?, lines for the summary).

    async def qa(client) -> tuple[bool, list[str]]:
        repos = [{"name": t.name, "worktree": checkout(t, f"qa-{t.name}", t.head), "base": t.base,
                  "setup": t.repo["setup"], "test": t.repo["test"], "timeout": t.repo["timeout"]}
                 for t in targets]
        mark("qa", "testing")
        task = await send(client, "qa", [new_text_part(brief(message, targets)),
                                         new_data_part({"repos": repos})], log=log)
        verdict = data_artifact(task, "verdict")
        lines = [f"qa: {'passed' if verdict['passed'] else 'did NOT pass'}  -> "
                 f"{save('qa.md', qa_markdown(verdict))}"]
        for name in (t.name for t in targets):  # a data part does not keep the key order
            result = verdict["repos"][name]
            mark("qa", f"tests {result['tests']}, {len(result['issues'])} issue(s)", [name])
            lines.append(f"  {name}: tests {result['tests']}, {len(result['issues'])} issue(s)")
            lines += [f"    - {issue}" for issue in result["issues"]]
        return verdict["passed"], lines

    async def review(client) -> tuple[bool, list[str]]:
        repos = [{"name": t.name, "worktree": checkout(t, f"review-{t.name}", t.head),
                  "base": t.base} for t in targets]
        mark("review", "reviewing")
        task = await send(client, "review", [new_text_part(brief(message, targets)),
                                             new_data_part({"repos": repos})], log=log)
        result = data_artifact(task, "review")
        path = save("review.md", text_artifact(task, "review.md"))
        counts = {}
        for finding in result["findings"]:
            counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
        mark("review", result["verdict"])
        lines = [f"review: {result['verdict']}, {len(result['findings'])} finding(s)"
                 + (f" ({', '.join(f'{n} {s}' for s, n in counts.items())})" if counts else "")
                 + f"  -> {path}"]
        for t in targets:
            if t.pr:
                number = t.label.split("#", 1)[1]
                lines.append(f"  post it on the PR:  (cd \"{t.repo['path']}\" && "
                             f"gh pr review {number} --comment -F \"{path}\")")
        return result["verdict"] != "request_changes", lines

    async def bench(client) -> tuple[bool, list[str]]:
        repos = [{"name": t.name, "base_worktree": checkout(t, f"bench-{t.name}-base", t.base),
                  "head_worktree": checkout(t, f"bench-{t.name}", t.head),
                  "setup": t.repo["setup"], "bench": t.repo["bench"],
                  "runs": t.repo["bench_runs"], "timeout": t.repo["timeout"]} for t in targets]
        mark("bench", "benchmarking")
        task = await send(client, "bench", [new_text_part(brief(message, targets)),
                                            new_data_part({"threshold": threshold, "repos": repos})],
                          log=log)
        report = data_artifact(task, "benchmark")
        path = save("benchmark.md", text_artifact(task, "benchmark.md"))
        lines = [f"bench: {'no regressions' if report['passed'] else 'REGRESSION or failure'} "
                 f"(threshold {threshold:g}%)  -> {path}"]
        for name in (t.name for t in targets):
            repo = report["repos"][name]
            worse = [m for m in repo["metrics"] if m["verdict"] == "worse"]
            mark("bench", f"{repo['status']}, {len(worse)} worse", [name])
            lines.append(f"  {name}: {repo['status']}"
                         + (f": {repo['note'].splitlines()[0]}" if repo["note"] else ""))
            lines += [f"    {m['name']}: {m['base']:g} -> {m['head']:g} {m['unit']} "
                      f"({m['change']:+.1f}%, {m['verdict']})" for m in repo["metrics"]]
        return report["passed"], lines

    async def planner(client) -> tuple[bool, list[str]]:
        repos = [{"name": t.name, "path": checkout(t, f"planner-{t.name}", t.head),
                  "description": t.repo["description"]} for t in targets]
        mark("planner", "planning")
        task = await send(client, "planner", [new_text_part(brief(message, targets)),
                                              new_data_part({"repos": repos})], log=log)
        plan = data_artifact(task, "plan")
        mark("planner", "planned")
        return True, [f"planner: {len(plan['repos'])} repo(s) to change  -> "
                      f"{save('plan.md', plan_markdown(plan))}"]

    async def coder(client) -> tuple[bool, list[str]]:
        branch = f"team/{slugify(message)}-{stamp:%H%M%S}"
        folders = {}
        for t in targets:  # first all worktrees, so each coder can read the others
            folders[t.name] = run_dir / "worktrees" / f"coder-{t.name}"
            gitops.add_worktree(t.repo["path"], folders[t.name], branch, t.head)
            branches.append((t.repo["path"], folders[t.name], branch, t.head))
        mark("coder", "coding")
        tasks = await asyncio.gather(*(send(client, f"coder:{t.name}", [
            new_text_part(brief(message, targets) + f"\n\nYou work in `{t.name}`, on top of {t.label}."),
            new_data_part({"repo": t.name, "worktree": str(folders[t.name]), "base": t.head,
                           "read_only": {o.name: str(folders[o.name]) for o in targets if o is not t}}),
        ], log=log) for t in targets))
        lines = [f"coder: branch {branch}"]
        for t, task in zip(targets, tasks):
            change = data_artifact(task, "change")
            mark("coder", "committed" if change["commit"] else "no changes", [t.name])
            lines.append(f"  {t.name}: " + (f"{len(change['files'])} file(s) changed. Review: "
                                            f"git -C \"{t.repo['path']}\" diff {t.head[:10]}..{branch}"
                                            if change["commit"] else "nothing changed"))
        return True, lines

    jobs = {"qa": qa, "review": review, "bench": bench, "planner": planner, "coder": coder}
    http = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None))
    config = ClientConfig(httpx_client=http)
    clients = {agent: await create_client(urls[agent], client_config=config) for agent in agents}
    results: dict[str, tuple[bool, list[str]]] = {}
    try:
        # Everything at once, except the benchmark: it waits and runs alone, since
        # tests running next to it would make the timings meaningless.
        together = [agent for agent in agents if agent != "bench"]
        rounds = [together] * bool(together) + [["bench"]] * ("bench" in agents)
        outcomes = []
        for group in rounds:
            status.update(phase="working: " + ", ".join(group))
            print(f"\n{'At the same time' if len(group) > 1 else 'Now'}: {', '.join(group)}")
            outcomes += await asyncio.gather(*(jobs[agent](clients[agent]) for agent in group),
                                             return_exceptions=True)
        for agent, outcome in zip(together + ["bench"] * ("bench" in agents), outcomes):
            if isinstance(outcome, (AgentError, gitops.GitError)):
                mark(agent, "failed")
                outcome = (False, [f"{agent}: FAILED: {outcome}"])
            elif isinstance(outcome, BaseException):
                raise outcome
            results[agent] = outcome
    except BaseException as error:  # also Ctrl+C
        status.update(phase="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        await asyncio.gather(*(c.close() for c in clients.values()), return_exceptions=True)
        if not keep_worktrees:
            for repo, path in detached:
                gitops.drop_worktree(repo, path)
        lines = []
        for repo, path, branch, start in branches:
            if keep_worktrees:
                lines.append(f"worktree kept at {path}")
            elif not gitops.remove_worktree(repo, path, branch, start):
                lines.append(f"{Path(repo).name}: no commits, branch {branch} removed")
        ok = bool(results) and all(passed for passed, _ in results.values())
        report = "\n".join(
            ["RESULT: " + ("all good" if ok else "see below" if results else "stopped")]
            + [line for _, agent_lines in results.values() for line in ["", *agent_lines]]
            + ([""] + lines if lines else [])
            + ([f"\nworktrees kept in {run_dir / 'worktrees'}"] if keep_worktrees and detached else [])
            + [f"\nReports, diffs and logs: {run_dir}"])
        save("summary.md", report)
        if status.data["phase"] != "stopped":
            status.update(phase="done: all good" if ok else "done: needs attention")
        print("\n" + report)
    return ok
