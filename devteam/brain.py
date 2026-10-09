"""The "brain" each agent uses: Claude plus a tool loop. Nothing here is A2A.

A2A standardises how agents talk to EACH OTHER. What happens inside an agent
is its own business. Here each agent lets Claude call the file tools from
tools.py until it calls a "finish" tool with a structured result.

Three interchangeable brains, chosen with the TEAM_BRAIN environment variable
(the devteam command sets it from --brain):

  subscription  Your Claude Code login (Pro/Max/Team plan), through the Claude
                Agent SDK. Claude Code's own tools are switched off; the model
                only gets the tools from tools.py.
  api           An Anthropic API key (ANTHROPIC_API_KEY), pay per use.
  mock          No model at all: replays a canned list of tool calls. Used by
                the demo.
"""

import asyncio
import os

from devteam.tools import Toolbox

BRAIN = os.environ.get("TEAM_BRAIN") or (
    "api" if os.environ.get("ANTHROPIC_API_KEY") else "subscription"
)
# api: defaults to Sonnet. subscription: defaults to whatever your Claude Code uses.
MODEL = os.environ.get("TEAM_MODEL") or ("claude-sonnet-5-5" if BRAIN == "api" else None)
MOCK_DELAY = float(os.environ.get("TEAM_MOCK_DELAY", "0"))
MAX_TURNS = 60
MAX_TOKENS = 16000

if BRAIN == "subscription":
    # If an API key is present, Claude Code bills that key instead of using
    # your subscription. Remove it from this process so that cannot happen.
    os.environ.pop("ANTHROPIC_API_KEY", None)

_api_client = None


class Conversation:
    """One ongoing chat with the model. Call ask() again to continue it."""

    def __init__(self, system: str, toolbox: Toolbox, finish: dict):
        self.system = system
        self.toolbox = toolbox
        self.finish = finish  # tool spec; the model calls it to hand in its result
        self.messages: list[dict] = []  # api brain: the full history
        self.session_id: str | None = None  # subscription brain: Claude Code session

    async def ask(self, text: str, on_tool=None, mock=None) -> dict:
        """Send `text`, let the model work with the tools, return what it hands in.

        `on_tool(name, args)` is awaited before each tool call (for progress).
        `mock` is the canned list of tool calls for the mock brain.
        """
        if BRAIN == "mock":
            return await self._ask_mock(mock, on_tool)
        if BRAIN == "api":
            return await self._ask_api(text, on_tool)
        return await self._ask_subscription(text, on_tool)

    # ---- brain 1: Claude Code subscription, via the Claude Agent SDK ---------

    async def _ask_subscription(self, text: str, on_tool) -> dict:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ResultMessage,
            create_sdk_mcp_server,
            query,
            tool,
        )

        handed_in = {}

        def as_sdk_tool(spec: dict):
            @tool(spec["name"], spec["description"], spec["input_schema"])
            async def handler(args):
                if spec["name"] == self.finish["name"]:
                    handed_in["result"] = args
                    return {"content": [{"type": "text", "text": "Received. You are done."}]}
                if on_tool:
                    await on_tool(spec["name"], args)
                output = await asyncio.to_thread(self.toolbox.call, spec["name"], args)
                return {"content": [{"type": "text", "text": output}],
                        "is_error": output.startswith("Error")}

            return handler

        specs = [*self.toolbox.specs(), self.finish]
        for attempt in range(3):
            options = ClaudeAgentOptions(
                system_prompt=self.system,
                model=MODEL,
                tools=[],  # none of Claude Code's built-in tools: no shell, no free file access
                mcp_servers={"team": create_sdk_mcp_server("team", tools=[as_sdk_tool(s) for s in specs])},
                allowed_tools=[f"mcp__team__{spec['name']}" for spec in specs],
                permission_mode="dontAsk",  # anything not allowed above is refused
                strict_mcp_config=True,  # ignore MCP servers from your own settings
                setting_sources=[],  # ignore your CLAUDE.md files, hooks and plugins
                max_turns=MAX_TURNS,
                resume=self.session_id,  # continue the earlier session, if any
            )
            result = None
            async for message in query(prompt=text, options=options):
                if isinstance(message, ResultMessage):
                    result = message
            if result is None:
                raise RuntimeError("Claude Code ended without a result.")
            self.session_id = result.session_id
            if result.is_error:
                detail = result.result or ", ".join(result.errors or []) or result.subtype
                raise RuntimeError(
                    f"Claude Code reported: {detail} (if this is a login problem, run "
                    "`claude` once in a terminal and sign in with your subscription)"
                )
            if "result" in handed_in:
                return handed_in["result"]
            text = f"You have not called {self.finish['name']} yet. Finish the work and call it."

        raise RuntimeError(f"The model never called {self.finish['name']}.")

    # ---- brain 2: Anthropic API key ---------------------------------------

    async def _ask_api(self, text: str, on_tool) -> dict:
        global _api_client
        if _api_client is None:
            from anthropic import AsyncAnthropic

            _api_client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

        self._add_user_text(text)
        for _ in range(MAX_TURNS):
            response = await _api_client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self.system,
                messages=self.messages,
                tools=[*self.toolbox.specs(), self.finish],
            )
            self.messages.append({"role": "assistant", "content": response.content})
            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                self._add_user_text(f"Keep going, and call {self.finish['name']} when you are done.")
                continue

            results, handed_in = [], None
            for call in calls:
                if response.stop_reason == "max_tokens":
                    output = "Error: your reply was cut off, so this call was NOT run. Use smaller steps."
                elif call.name == self.finish["name"]:
                    handed_in, output = call.input, "Received."
                else:
                    if on_tool:
                        await on_tool(call.name, call.input)
                    output = await asyncio.to_thread(self.toolbox.call, call.name, call.input)
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
            self.messages.append({"role": "user", "content": results})
            if handed_in is not None:
                return handed_in

        raise RuntimeError(f"The model did not finish within {MAX_TURNS} turns.")

    def _add_user_text(self, text: str) -> None:
        """Append user text, keeping user and assistant turns alternating."""
        block = {"type": "text", "text": text}
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"].append(block)
        else:
            self.messages.append({"role": "user", "content": [block]})

    # ---- brain 3: canned script, for the demo ------------------------------

    async def _ask_mock(self, script, on_tool) -> dict:
        if script is None:
            raise RuntimeError("The mock brain only knows the demo repos.")
        await asyncio.sleep(MOCK_DELAY)
        for name, args in script:
            if name == self.finish["name"]:
                return args
            if on_tool:
                await on_tool(name, args)
            output = self.toolbox.call(name, args)
            if output.startswith("Error"):
                raise RuntimeError(f"Mock step {name} failed: {output}")
        raise RuntimeError("Mock script ended without calling the finish tool.")
