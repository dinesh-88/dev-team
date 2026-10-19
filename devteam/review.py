"""Review agent: reviews a branch (or a GitHub pull request) the way a senior engineer would.

Run on its own with:  devteam agent review   (or python -m devteam.review)

Incoming message:  text part = what to focus on, plus PR titles and descriptions
                   data part = {"repos": [{"name", "worktree", "base"}]}
Result artifacts:  "review" data part = {"verdict": "approve"|"comment"|"request_changes",
                   "summary", "findings": [{"repo", "file", "line", "severity",
                   "problem", "suggestion"}]}
                   "review.md" file part = the same review as Markdown, ready to paste into a PR

How it differs from QA's review: QA only checks that repos agree with each
other and with the plan, so the team can loop. This agent reviews the change
on its own merits (bugs, security, tests, performance) and grades each
finding. The verdict follows from the grades by a fixed rule, not from the model.
"""

from a2a.helpers import get_data_parts, new_data_part
from a2a.types import AgentSkill, Part

from devteam import brain, gitops, mock
from devteam.a2a_server import TeamAgent, build_card, main
from devteam.tools import Toolbox

PORT = 9105  # when run on its own; `devteam ask` picks free ports
DIFF_LIMIT = 60000  # characters of each diff shown to the model
SEVERITIES = ["blocker", "major", "minor", "nit"]

SYSTEM = """You are a senior engineer reviewing a pull request. You get each
repository's diff against its base branch, and you can read any file of the new
version with your tools. Read the surrounding code before reporting something:
only report what you have confirmed.

Look for, most important first:
- bugs: wrong logic, unhandled errors and edge cases, broken callers of changed code
- security: injection, secrets in code, unchecked input, missing authorization
- tests: changed behaviour without tests, or tests that do not test it
- breaking changes to APIs, schemas or contracts other code relies on
- performance problems in code that runs often
- readability only where it is likely to cause bugs later

Severity: blocker (must not merge: data loss, security hole, crash on a common
path), major (a real bug or missing test for new behaviour), minor (an edge
case or a small risk), nit (optional). Nothing a formatter or linter would catch.

Call submit_review with a short summary of what the change does and how good
it is, and the findings. `line` is a line number in the NEW version of the file."""


def finish_tool(repo_names: list[str]) -> dict:
    return {
        "name": "submit_review",
        "description": "Hand in the review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "enum": repo_names},
                            "file": {"type": "string"},
                            "line": {"type": "integer"},
                            "severity": {"type": "string", "enum": SEVERITIES},
                            "problem": {"type": "string"},
                            "suggestion": {"type": "string"},
                        },
                        "required": ["repo", "file", "severity", "problem"],
                    },
                },
            },
            "required": ["summary", "findings"],
        },
    }


def verdict(findings: list[dict]) -> str:
    """A fixed rule, so the same findings always give the same verdict."""
    if any(f["severity"] in ("blocker", "major") for f in findings):
        return "request_changes"
    return "comment" if findings else "approve"


def as_markdown(review: dict) -> str:
    title = {"approve": "Approve", "comment": "Comment",
             "request_changes": "Request changes"}[review["verdict"]]
    lines = [f"## Review: {title}", "", review["summary"]]
    for severity in SEVERITIES:
        found = [f for f in review["findings"] if f["severity"] == severity]
        if found:
            lines += ["", f"### {severity.capitalize()} ({len(found)})", ""]
        for f in found:
            where = f"{f['repo']}/{f['file']}" + (f":{f['line']}" if f.get("line") else "")
            lines.append(f"- **`{where}`** {f['problem']}")
            if f.get("suggestion"):
                lines.append(f"  - Suggestion: {f['suggestion']}")
    if not review["findings"]:
        lines += ["", "No findings."]
    return "\n".join(lines)


class Reviewer(TeamAgent):
    async def work(self, context, updater):
        brief = context.get_user_input()
        repos = get_data_parts(context.message.parts)[0]["repos"]

        sections = [brief] if brief else []
        for repo in repos:
            stat = gitops.git(repo["worktree"], "diff", "--stat", f"{repo['base']}..HEAD")
            diff = gitops.diff(repo["worktree"], repo["base"])
            if len(diff) > DIFF_LIMIT:
                diff = diff[:DIFF_LIMIT] + "\n... (diff cut here: read the files for the rest)"
            sections.append(f"## {repo['name']}\n{stat or '(no changes)'}\n```diff\n{diff}\n```")

        await self.progress(context, updater, "Reviewing " + ", ".join(r["name"] for r in repos))
        toolbox = Toolbox({repo["name"]: repo["worktree"] for repo in repos})  # read-only
        chat = brain.Conversation(SYSTEM, toolbox, finish_tool(sorted(toolbox.roots)))
        result = await chat.ask("\n\n".join(sections), **self.reporters(context, updater),
                                mock=mock.REVIEWER)

        findings = [f for f in result["findings"] if f.get("severity") in SEVERITIES]
        findings.sort(key=lambda f: SEVERITIES.index(f["severity"]))
        review = {"verdict": verdict(findings), "summary": result["summary"], "findings": findings}
        await updater.add_artifact([new_data_part(review)], name="review")
        await updater.add_artifact(
            [Part(text=as_markdown(review), filename="review.md", media_type="text/markdown")],
            name="review.md",
        )


CARD = build_card(
    name="Reviewer",
    description="Reviews a branch or pull request for bugs, security, tests and breaking changes.",
    skill=AgentSkill(
        id="review",
        name="Review a pull request",
        description="Send focus notes as text and {repos:[{name,worktree,base}]} as data. "
        "Returns a `review` data artifact {verdict, summary, findings} and `review.md`.",
        tags=["review", "pull-request", "multi-repo"],
    ),
)

if __name__ == "__main__":
    main(CARD, Reviewer(), PORT)
