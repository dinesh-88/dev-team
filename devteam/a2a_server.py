"""Everything needed to turn a Python function into an A2A agent server.

Read this file once and you have seen the whole server side of the protocol:

  1. AGENT CARD   - a JSON document at /.well-known/agent-card.json that says
                    who the agent is, what it can do, and where to reach it.
                    Clients read it first. This is "discovery".
  2. MESSAGE      - what a client sends: a role plus a list of Parts
                    (text, structured data, files). Our agents get their
                    instructions as a text part and settings as a data part.
  3. TASK         - the unit of work the server creates for a message. It has
                    an id, a state (submitted -> working -> completed/failed)
                    and a context_id that groups related tasks together.
  4. ARTIFACT     - the result the agent attaches to the task (also Parts).
  5. STREAMING    - while it works, the agent pushes status and artifact
                    events to the client over Server-Sent Events.
"""

import argparse
import os
import threading
import time
from abc import abstractmethod

import uvicorn
from a2a.helpers import new_task_from_user_message, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

HOST = "127.0.0.1"


def build_card(name: str, description: str, skill: AgentSkill) -> AgentCard:
    """(1) The Agent Card: the agent's public business card.

    Where to reach the agent is added by serve(), once the port is known.
    """
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=[skill],
    )


class TeamAgent(AgentExecutor):
    """Base class for our agents. The SDK calls execute() for every message.

    Subclasses only implement work(); the A2A bookkeeping lives here so you
    can see it in one place.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # (3) First message of a conversation: create the Task and announce it.
        if context.current_task is None:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))

        # TaskUpdater is a helper that publishes events for this task.
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        try:
            await self.work(context, updater)
            await updater.complete()  # state -> completed (terminal)
        except Exception as error:  # noqa: BLE001 - report any failure to the client
            await updater.failed(self.say(context, f"{type(error).__name__}: {error}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()

    @staticmethod
    def say(context: RequestContext, text: str, kind: str | None = None):
        """Build an agent message tied to the current task.

        `kind` goes into the message's metadata, which A2A leaves free for
        extensions. Ours: "tool" (a tool call) or "text" (the model talking).
        """
        message = new_text_message(text, context_id=context.context_id, task_id=context.task_id)
        if kind:
            message.metadata.update({"kind": kind})
        return message

    async def progress(self, context: RequestContext, updater: TaskUpdater, text: str,
                       kind: str | None = None) -> None:
        """(5) Stream a 'still working' update to the client."""
        await updater.start_work(self.say(context, text, kind))

    def reporters(self, context: RequestContext, updater: TaskUpdater) -> dict:
        """Callbacks for brain.Conversation.ask() that stream what the model does."""

        async def on_tool(tool: str, args: dict) -> None:
            target = "/".join(str(args[key]) for key in ("repo", "path") if key in args)
            await self.progress(context, updater, f"{tool} {target}".strip(), "tool")

        async def on_text(text: str) -> None:
            await self.progress(context, updater, text.strip(), "text")

        return {"on_tool": on_tool, "on_text": on_text}

    @abstractmethod
    async def work(self, context: RequestContext, updater: TaskUpdater) -> None:
        """Do the job and attach results with updater.add_artifact(...)."""


def serve(card: AgentCard, agent: TeamAgent, port: int) -> None:
    """Wire the card and the agent into an HTTP app and run it."""
    # Where and how to reach the agent. JSONRPC over HTTP is the most
    # common binding; A2A also defines gRPC and plain REST.
    card.supported_interfaces.append(
        AgentInterface(protocol_binding="JSONRPC", url=f"http://{HOST}:{port}/")
    )
    handler = DefaultRequestHandler(
        agent_executor=agent,
        task_store=InMemoryTaskStore(),  # where tasks are remembered
        agent_card=card,
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(card),  # GET  /.well-known/agent-card.json
            *create_jsonrpc_routes(handler, rpc_url="/"),  # POST /  (JSON-RPC)
        ]
    )
    print(f"[{card.name}] listening on http://{HOST}:{port}", flush=True)
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


def main(card: AgentCard, agent: TeamAgent, default_port: int) -> None:
    """`python -m devteam.<agent> [--port N]`"""
    parser = argparse.ArgumentParser(description=card.description)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--parent", type=int, help="exit when this process (our starter) is gone")
    args = parser.parse_args()
    if args.parent:
        threading.Thread(target=exit_with_parent, args=(args.parent,), daemon=True).start()
    serve(card, agent, args.port)


def exit_with_parent(parent: int) -> None:
    """If `devteam` is killed hard it cannot stop us, so we notice and stop ourselves."""
    while os.getppid() == parent:
        time.sleep(1)
    os._exit(0)
