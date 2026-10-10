"""QA agent: runs each repo's own tests, then reviews the change across repos.

Run on its own with:  devteam agent qa   (or python -m devteam.qa)

Incoming message:  text part = the request, plan summary and contract
                   data part = {"repos": [{"name", "worktree", "base",
                                           "setup", "test", "timeout"}]}
Result artifact:   "verdict" data part = {"passed": bool, "repos": {name:
                   {"tests": "passed"|"failed"|"skipped", "output", "issues": [..]}}}

Two checks, because tests alone cannot see across repositories:
1. TESTS   - the `test` command from your repos.toml, run in each worktree.
             These commands come from YOUR config, never from a model.
2. REVIEW  - Claude reads all the diffs together and reports where the repos
             disagree with each other or with the contract.
"""

import asyncio
import subprocess

from a2a.helpers import get_data_parts, new_data_part
from a2a.types import AgentSkill

from devteam import brain, gitops, mock
from devteam.a2a_server import TeamAgent, build_card, main
from devteam.tools import Toolbox

PORT = 9103  # when run on its own; `devteam run` picks free ports
OUTPUT_TAIL = 6000  # characters of test output to keep
DIFF_LIMIT = 60000  # characters of each diff shown to the reviewer

SYSTEM = """You are the QA reviewer on a small software team. One change was made
across several repositories. You get the request, the contract the repositories
must agree on, each repository's diff and its test result. You can read files
for more context.

Report only real defects:
- the change does not do what was requested, or
- repositories disagree with each other or with the contract (names, fields,
  types, endpoints, versions), or
- an obvious bug in the changed lines.
No style comments, no optional improvements. Failing tests are already reported
separately, so do not repeat them.

Call submit_review with the issues, each assigned to the repository that must
change to fix it. An empty list means the change is good."""


def finish_tool(repo_names: list[str]) -> dict:
    return {
        "name": "submit_review",
        "description": "Hand in the review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "enum": repo_names},
                            "problem": {"type": "string"},
                        },
                        "required": ["repo", "problem"],
                    },
                }
            },
            "required": ["issues"],
        },
    }


def run_command(command: str, cwd: str, timeout: float) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"`{command}` did not finish within {timeout:.0f} seconds."
    return result.returncode == 0, (result.stdout + result.stderr).strip()[-OUTPUT_TAIL:]


class QA(TeamAgent):
    def __init__(self):
        self.setup_done: dict[str, set[str]] = {}  # context_id -> repos already set up

    def run_tests(self, repo: dict, setup_done: set[str]) -> tuple[str, str]:
        timeout = repo.get("timeout") or 900
        if repo.get("setup") and repo["name"] not in setup_done:
            ok, output = run_command(repo["setup"], repo["worktree"], timeout)
            if not ok:
                return "failed", f"Setup command `{repo['setup']}` failed:\n{output}"
            setup_done.add(repo["name"])
        if not repo.get("test"):
            return "skipped", "No test command configured for this repo."
        ok, output = run_command(repo["test"], repo["worktree"], timeout)
        return ("passed" if ok else "failed"), output

    async def work(self, context, updater):
        brief = context.get_user_input()
        repos = get_data_parts(context.message.parts)[0]["repos"]
        setup_done = self.setup_done.setdefault(context.context_id, set())

        # 1. Tests, all repos at the same time (each in its own thread)
        async def test(repo: dict) -> None:
            await self.progress(context, updater, f"{repo['name']}: running tests")
            status, output = await asyncio.to_thread(self.run_tests, repo, setup_done)
            await self.progress(context, updater, f"{repo['name']}: tests {status}")
            results[repo["name"]] = {"tests": status, "output": output, "issues": []}

        results = {}
        await asyncio.gather(*(test(repo) for repo in repos))
        results = {repo["name"]: results[repo["name"]] for repo in repos}  # keep config order

        # 2. Cross-repo review
        await self.progress(context, updater, "Reviewing the change across repos")
        sections = [brief]
        for repo in repos:
            diff = gitops.diff(repo["worktree"], repo["base"])[:DIFF_LIMIT] or "(no changes)"
            sections.append(
                f"## {repo['name']} (tests: {results[repo['name']]['tests']})\n```diff\n{diff}\n```"
            )
        toolbox = Toolbox({repo["name"]: repo["worktree"] for repo in repos})  # read-only
        chat = brain.Conversation(SYSTEM, toolbox, finish_tool(sorted(results)))
        review = await chat.ask(
            "\n\n".join(sections),
            **self.reporters(context, updater), mock=mock.REVIEW,
        )
        for issue in review["issues"]:
            if issue["repo"] in results:
                results[issue["repo"]]["issues"].append(issue["problem"])

        passed = all(r["tests"] != "failed" and not r["issues"] for r in results.values())
        await updater.add_artifact(
            [new_data_part({"passed": passed, "repos": results})], name="verdict"
        )


CARD = build_card(
    name="QA",
    description="Runs each repository's tests and reviews a change across repositories.",
    skill=AgentSkill(
        id="verify",
        name="Verify a multi-repo change",
        description="Send the brief as text and {repos:[{name,worktree,base,setup,test,timeout}]} "
        "as data. Returns a `verdict` data artifact with per-repo test results and issues.",
        tags=["testing", "review", "multi-repo"],
    ),
)

if __name__ == "__main__":
    main(CARD, QA(), PORT)
