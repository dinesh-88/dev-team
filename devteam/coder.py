"""Coder agent: implements the plan in ONE repo per conversation, then commits.

Run on its own with:  devteam agent coder   (or python -m devteam.coder)

Incoming message:  text part = what to do
                   data part = {"repo", "worktree", "base", "read_only": {name: path}}
Result artifacts:  "change" data part = {"repo", "commit", "files", "summary"}
                   "<repo>.diff" file part = everything on the branch so far

A2A idea to notice here: CONTEXT. The orchestrator opens one conversation
(context_id) per repo. When it later sends QA feedback with the same
context_id, this agent finds the stored conversation and worktree and carries
on, so the follow-up only needs to contain the feedback.
"""

from a2a.helpers import get_data_parts, new_data_part
from a2a.types import AgentSkill, Part

from devteam import brain, gitops, mock
from devteam.a2a_server import TeamAgent, build_card, main
from devteam.tools import Toolbox

PORT = 9102  # when run on its own; `devteam run` picks free ports

SYSTEM = """You are the coder on a small software team, working in the repository
`{repo}`. Other repositories may be visible to you read-only for reference.

- Read the relevant code before changing it, and follow its existing style.
- Do exactly what your instructions and the contract say. The contract is shared
  with coders working in other repositories, so names and shapes must match it.
- Add or update tests as instructed.
- You cannot run commands. QA runs the tests afterwards and you will get the
  results if something fails.

When the work is complete, call submit_change with a commit message: a short
subject line, a blank line, then what changed and why."""

FINISH = {
    "name": "submit_change",
    "description": "Declare the work done. Your file changes are committed with this message.",
    "input_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "Commit message"}},
        "required": ["summary"],
    },
}


class Coder(TeamAgent):
    def __init__(self):
        self.conversations: dict[str, dict] = {}  # context_id -> state

    async def work(self, context, updater):
        state = self.conversations.get(context.context_id)
        if state is None:  # first message of this conversation: set up
            info = get_data_parts(context.message.parts)[0]
            repo = info["repo"]
            toolbox = Toolbox({**info.get("read_only", {}), repo: info["worktree"]}, writable=[repo])
            # Edits an interrupted earlier conversation left behind are this one's to commit.
            toolbox.changed[repo].update(gitops.uncommitted(info["worktree"]))
            state = self.conversations[context.context_id] = {
                "repo": repo,
                "worktree": info["worktree"],
                "base": info["base"],
                "toolbox": toolbox,
                "chat": brain.Conversation(SYSTEM.format(repo=repo), toolbox, FINISH),
                "round": 0,
            }
        repo, toolbox = state["repo"], state["toolbox"]
        state["round"] += 1

        # Same Conversation object as last time, so the model remembers its earlier work.
        result = await state["chat"].ask(
            context.get_user_input(),
            **self.reporters(context, updater),
            mock=mock.coder(repo, state["worktree"]),
        )

        # Commit only the files the model touched through its tools.
        files = sorted(toolbox.changed[repo])
        toolbox.changed[repo].clear()
        commit = gitops.commit(state["worktree"], files, result["summary"])
        diff = gitops.diff(state["worktree"], state["base"])

        await updater.add_artifact(
            [new_data_part({"repo": repo, "commit": commit, "files": files,
                            "summary": result["summary"]})],
            name="change",
        )
        if diff:
            # A Part with a filename and media type is a file.
            await updater.add_artifact(
                [Part(text=diff, filename=f"{repo}.diff", media_type="text/x-diff")],
                name="diff",
            )


CARD = build_card(
    name="Coder",
    description="Implements a plan in one repository worktree and commits the result.",
    skill=AgentSkill(
        id="implement",
        name="Implement changes in a repository",
        description="Send instructions as text and {repo, worktree, base, read_only} as data. "
        "Returns a `change` data artifact and a diff. Send QA feedback in the same "
        "context to get a fix.",
        tags=["coding", "git", "multi-repo"],
    ),
)

if __name__ == "__main__":
    main(CARD, Coder(), PORT)
