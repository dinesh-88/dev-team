"""Planner agent: reads the repos and splits a request into per-repo work.

Run on its own with:  devteam agent planner   (or python -m devteam.planner)

Incoming message:  text part = the request
                   data part = {"repos": [{"name", "path", "description"}]}
Result artifact:   "plan" data part =
                   {"summary", "contract", "repos": [{"name", "steps"}]}
                   The order of "repos" is the order to implement them in.
"""

from a2a.helpers import get_data_parts, new_data_part
from a2a.types import AgentSkill

from devteam import brain, mock
from devteam.a2a_server import TeamAgent, build_card, main
from devteam.tools import Toolbox

PORT = 9101  # when run on its own; `devteam run` picks free ports

SYSTEM = """You are the planner on a small software team. One request may need
changes in several repositories. Read the code with your tools first; do not guess.

Then call submit_plan with:
- summary: what will change, in two or three sentences.
- contract: everything the repos must agree on, stated exactly: names, fields,
  types, endpoints, versions. The coders work one repo at a time and rely on this.
- repos: ONLY the repos that need changes, ordered so that providers (APIs,
  libraries, schemas) come before their consumers. For each, concrete steps
  naming real files and functions, including which tests to add or update.

You cannot change files. Keep the plan as small as the request allows."""


def finish_tool(repo_names: list[str]) -> dict:
    return {
        "name": "submit_plan",
        "description": "Hand in the finished plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "contract": {"type": "string"},
                "repos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": repo_names},
                            "steps": {"type": "string"},
                        },
                        "required": ["name", "steps"],
                    },
                },
            },
            "required": ["summary", "contract", "repos"],
        },
    }


class Planner(TeamAgent):
    async def work(self, context, updater):
        request = context.get_user_input()  # the text parts of the message
        repos = get_data_parts(context.message.parts)[0]["repos"]  # the data part

        toolbox = Toolbox({repo["name"]: repo["path"] for repo in repos})  # read-only
        listing = "\n".join(f"- {repo['name']}: {repo['description']}" for repo in repos)
        chat = brain.Conversation(SYSTEM, toolbox, finish_tool(sorted(toolbox.roots)))
        plan = await chat.ask(
            f"Request:\n{request}\n\nRepositories:\n{listing}",
            on_tool=self.tool_reporter(context, updater), mock=mock.PLANNER,
        )

        # A data part carries structured JSON, so the caller can use the plan
        # directly instead of parsing prose.
        await updater.add_artifact([new_data_part(plan)], name="plan")


CARD = build_card(
    name="Planner",
    description="Reads the repositories and splits a request into ordered per-repo work.",
    skill=AgentSkill(
        id="plan",
        name="Plan a multi-repo change",
        description="Send the request as text and {repos:[{name,path,description}]} as data. "
        "Returns a `plan` data artifact: {summary, contract, repos:[{name, steps}]}.",
        tags=["planning", "multi-repo"],
    ),
)

if __name__ == "__main__":
    main(CARD, Planner(), PORT)
