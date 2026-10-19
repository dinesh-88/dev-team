"""`devteam ask`: the agents YOU choose, on the repos and branches you choose.

    devteam ask qa web@feature/login                 test and check one branch
    devteam ask review api#42                        review GitHub pull request 42 of `api`
    devteam ask review,qa,bench api@perf-cache       review and QA together, then bench
    devteam ask coder web@feature/x -m "Fix the failing date test"
    devteam ask planner api web -m "How would we add SSO?"
    devteam ask qa,review web@feature/x --fix        check, let the coder fix what is found, check again

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

--fix N (qa and/or review) sends what they find to the coder: failing tests,
QA issues, and review findings that block the merge (blocker, major). It fixes
them on a new branch team/fix-<...> that starts at the target, then qa and
review check again, now on that branch. Up to N fix rounds (default 2).
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
from devteam.orchestrator import (STATE_DIR, AgentError, RunLog, RunStatus, all_of,
                                  data_artifact, plan_markdown, send, slugify, text_artifact)

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
              on_start: Callable[[Path], None] | None = None, fix: int = 0) -> bool:
    """Run `agents` at the same time on `targets`. True if every agent is satisfied.

    With `fix` (qa and/or review only), what they find goes to the coder, which
    fixes it on a new branch, and they check again: up to `fix` times.
    """
    stamp = datetime.now()
    what = " ".join(t.label for t in targets)
    run_dir = (Path(config_path).resolve().parent / STATE_DIR / "runs"
               / f"{stamp:%Y%m%d-%H%M%S}-{'-'.join(agents)}-{slugify(what)}")
    run_dir.mkdir(parents=True)
    log = RunLog(run_dir / "logs")
    status = RunStatus(run_dir / "status.json", kind="ask", agents=urls,
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

    # With --fix, qa and review check the same worktree per repo: first one at the
    # target, then, once the coder has fixed a repo, the coder's worktree.
    folders: dict[str, str] = {}
    problems: dict[str, list[str]] = {}  # repo -> what the checks found, for the coder
    tag = {"round": "", "qa_context": None}  # report name suffix; QA conversation to reuse

    def where(t: Target, agent: str) -> str:
        return folders.get(t.name) or checkout(t, f"{agent}-{t.name}", t.head)

    def mark(agent: str, text: str, names: list[str] | None = None) -> None:
        status.update(repos={f"{agent}:{n}": text for n in names or [t.name for t in targets]})

    def save(name: str, text: str) -> Path:
        (run_dir / name).write_text(text + "\n", encoding="utf-8")
        return run_dir / name

    # One coroutine per agent. Each returns (satisfied?, lines for the summary).

    async def qa(client) -> tuple[bool, list[str]]:
        repos = [{"name": t.name, "worktree": where(t, "qa"), "base": t.base,
                  "setup": t.repo["setup"], "test": t.repo["test"], "timeout": t.repo["timeout"]}
                 for t in targets]
        mark("qa", "testing")
        # The same QA conversation, as long as the worktrees stay: then setup is not rerun.
        task = await send(client, "qa", [new_text_part(brief(message, targets)),
                                         new_data_part({"repos": repos})], tag["qa_context"], log)
        tag["qa_context"] = task.context_id
        verdict = data_artifact(task, "verdict")
        lines = [f"qa: {'passed' if verdict['passed'] else 'did NOT pass'}  -> "
                 f"{save('qa' + tag['round'] + '.md', qa_markdown(verdict))}"]
        for name in (t.name for t in targets):  # a data part does not keep the key order
            result = verdict["repos"][name]
            mark("qa", f"tests {result['tests']}, {len(result['issues'])} issue(s)", [name])
            lines.append(f"  {name}: tests {result['tests']}, {len(result['issues'])} issue(s)")
            lines += [f"    - {issue}" for issue in result["issues"]]
            if result["tests"] == "failed":
                problems.setdefault(name, []).append(
                    f"QA: the tests failed:\n```\n{result['output']}\n```")
            if result["issues"]:
                problems.setdefault(name, []).extend(f"QA: {issue}" for issue in result["issues"])
        return verdict["passed"], lines

    async def review(client) -> tuple[bool, list[str]]:
        repos = [{"name": t.name, "worktree": where(t, "review"), "base": t.base} for t in targets]
        mark("review", "reviewing")
        task = await send(client, "review", [new_text_part(brief(message, targets)),
                                             new_data_part({"repos": repos})], log=log)
        result = data_artifact(task, "review")
        path = save(f"review{tag['round']}.md", text_artifact(task, "review.md"))
        counts = {}
        for finding in result["findings"]:
            counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
            if finding["severity"] in ("blocker", "major"):  # what blocks the merge
                where_ = finding["file"] + (f":{finding['line']}" if finding.get("line") else "")
                problems.setdefault(finding["repo"], []).append(
                    f"Review ({finding['severity']}) {where_}: {finding['problem']}"
                    + (f" Suggestion: {finding['suggestion']}" if finding.get("suggestion") else ""))
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
    clients = {agent: await create_client(urls[agent], client_config=config)
               for agent in agents + ["coder"] * bool(fix)}
    results: dict[str, tuple[bool, list[str]]] = {}

    async def run_group(group: list[str], title: str) -> None:
        status.update(phase="working: " + ", ".join(group))
        print(f"\n{title}: {', '.join(group)}")
        outcomes = await asyncio.gather(*(jobs[agent](clients[agent]) for agent in group),
                                        return_exceptions=True)
        for agent, outcome in zip(group, outcomes):
            if isinstance(outcome, (AgentError, gitops.GitError)):
                mark(agent, "failed")
                outcome = (False, [f"{agent}: FAILED: {outcome}"])
            elif isinstance(outcome, BaseException):
                raise outcome
            results[agent] = outcome

    fix_branch = f"team/fix-{slugify(message or what)}-{stamp:%H%M%S}"
    contexts: dict[str, str] = {}  # repo -> the coder's conversation, kept across rounds

    async def fix_repo(t: Target, after_round: int) -> None:
        found = "\n\n".join(problems[t.name])
        if t.name in contexts:
            parts = [new_text_part(f"The checks ran again (round {after_round}) and found:\n\n"
                                   f"{found}\n\nFix these in `{t.name}`.")]
        else:  # the first fix in this repo: a worktree on a new branch, and the whole story
            worktree = run_dir / "worktrees" / f"coder-{t.name}"
            gitops.add_worktree(t.repo["path"], worktree, fix_branch, t.head)
            branches.append((t.repo["path"], worktree, fix_branch, t.head))
            folders[t.name] = str(worktree)
            tag["qa_context"] = None  # QA must set up the new worktree
            parts = [new_text_part(
                brief(message, targets) + f"\n\nYou work in `{t.name}`, on top of {t.label}. "
                f"The checks found these problems. Fix them, and nothing else:\n\n{found}"),
                new_data_part({"repo": t.name, "worktree": str(worktree), "base": t.head,
                               "read_only": {o.name: folders[o.name] for o in targets if o is not t}})]
        mark("coder", f"fixing (after round {after_round})", [t.name])
        task = await send(clients["coder"], f"coder:{t.name}", parts, contexts.get(t.name), log)
        contexts[t.name] = task.context_id
        change = data_artifact(task, "change")
        mark("coder", f"fixed: {len(change['files'])} file(s)" if change["commit"]
             else "changed nothing", [t.name])

    try:
        if not fix:
            # Everything at once, except the benchmark: it waits and runs alone, since
            # tests running next to it would make the timings meaningless.
            together = [agent for agent in agents if agent != "bench"]
            if together:
                await run_group(together, "At the same time" if len(together) > 1 else "Now")
            if "bench" in agents:
                await run_group(["bench"], "Now")
        else:
            folders.update({t.name: checkout(t, f"check-{t.name}", t.head) for t in targets})
            for number in range(1, fix + 2):
                tag["round"] = f"-round{number}"
                problems.clear()
                await run_group(agents, f"Check (round {number} of at most {fix + 1})")
                todo = [t for t in targets if problems.get(t.name)]
                if all(ok for ok, _ in results.values()) or number == fix + 1 or not todo:
                    break
                print(f"\nCoder (fix after round {number}): {', '.join(t.name for t in todo)}")
                status.update(phase=f"fixing (after round {number})")
                await all_of([fix_repo(t, number) for t in todo])
            if branches:
                fixed = [f"  {Path(repo).name}: git -C \"{repo}\" diff {start[:10]}..{branch}"
                         for repo, _, branch, start in branches]
                results["coder"] = (True, [f"coder: fixes on branch {fix_branch} "
                                           f"(started at your branch). Review:", *fixed])
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
