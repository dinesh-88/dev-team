"""The `devteam` command: start the agents, run a request through them, stop them.

    devteam setup ~/code                 write ./repos.toml for the git repos in ~/code
    devteam run "Add an email field ..." change your repos (listed in repos.toml)
    devteam demo                         two tiny demo repos, no Claude needed
    devteam agent planner                run one agent on its own, e.g. to curl it

`devteam <command> --help` shows the options of each command.
"""

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("Python 3.11 or newer is required.")

import httpx

from devteam import gitops
from devteam.orchestrator import AgentError, load_repos, run_team, survey

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
CARD_PATH = "/.well-known/agent-card.json"
CONFIG_NAME = "repos.toml"
USER_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "devteam" / CONFIG_NAME
AGENTS = {
    "planner": "Reads the repos and splits a request into per-repo work (default port 9101)",
    "coder": "Implements the plan in one repo worktree (default port 9102)",
    "qa": "Runs the tests and reviews the change across repos (default port 9103)",
    "surveyor": "Describes the repos in a folder for repos.toml (default port 9104)",
}
BRAINS = {
    "subscription": "your Claude Code login (subscription)",
    "api": "Anthropic API key",
    "mock": "canned demo script, no model",
}


# ---- starting and stopping the agents ----------------------------------------

def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_up(url: str, process: subprocess.Popen, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            sys.exit(f"The agent for {url} stopped during startup.")
        try:
            if httpx.get(url + CARD_PATH, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    sys.exit(f"The agent at {url} did not start within {timeout:.0f} seconds.")


def agent_command(name: str, port: int) -> list[str]:
    return [sys.executable, "-m", f"devteam.{name}", "--port", str(port)]


def agent_env() -> dict[str, str]:
    """Lets `python -m devteam.<agent>` work even when devteam is not installed."""
    path = os.pathsep.join(filter(None, [str(PACKAGE_PARENT), os.environ.get("PYTHONPATH")]))
    return os.environ | {"PYTHONPATH": path}


@contextmanager
def agents_running(names: list[str]):
    """Start each agent on a free port as its own process. Yields {name: url}."""
    ports = {name: free_port() for name in names}
    processes = {name: subprocess.Popen(agent_command(name, port), env=agent_env())
                 for name, port in ports.items()}
    urls = {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}
    try:
        for name, process in processes.items():
            wait_until_up(urls[name], process)
        yield urls
    finally:
        for process in processes.values():
            process.terminate()


# ---- choosing files and brains -----------------------------------------------

def find_config(explicit: str | None) -> Path:
    """--config, else ./repos.toml, else ~/.config/devteam/repos.toml."""
    if explicit:
        return Path(explicit).expanduser()
    local = Path.cwd() / CONFIG_NAME
    if not local.exists() and USER_CONFIG.exists():
        return USER_CONFIG
    return local


def choose_brain(parser: argparse.ArgumentParser, wanted: str | None, demo: bool,
                 allow_mock: bool = False) -> str:
    brain = wanted or ("mock" if demo else
                       "api" if os.environ.get("ANTHROPIC_API_KEY") else "subscription")
    if brain == "mock" and not (demo or allow_mock):
        parser.error("the mock brain only knows the demo repos: use `devteam demo`")
    if brain == "api" and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("--brain api needs the ANTHROPIC_API_KEY environment variable")
    os.environ["TEAM_BRAIN"] = brain  # inherited by the agent processes
    print(f"Brain:   {BRAINS[brain]}")
    return brain


# ---- the commands ------------------------------------------------------------

def team(request: str, config: Path, keep: bool) -> int:
    print(f"Request: {request}\nConfig:  {config}")
    with agents_running(["planner", "coder", "qa"]) as urls:
        try:
            return 0 if asyncio.run(run_team(request, config, urls, keep)) else 1
        except (AgentError, gitops.GitError) as error:
            print(f"\nStopped: {error}")
            return 1


def setup(folder: str, out: Path, hints: str, only: list[str] | None) -> int:
    with agents_running(["surveyor"]) as urls:
        try:
            asyncio.run(survey(urls["surveyor"], folder, out, hints, only))
            return 0
        except AgentError as error:
            print(f"\nStopped: {error}")
            return 1


def cmd_run(parser, args) -> int:
    request = " ".join(args.request)
    if not request:
        parser.error("say what you want changed")
    config = find_config(args.config)
    load_repos(config)  # fail on a bad config before starting any agent
    choose_brain(parser, args.brain, demo=False)
    return team(request, config, args.keep)


def cmd_setup(parser, args) -> int:
    choose_brain(parser, args.brain, demo=False)
    out = Path(args.config).expanduser() if args.config else Path.cwd() / CONFIG_NAME
    return setup(args.folder, out, " ".join(args.hints), args.only)


def cmd_demo(parser, args) -> int:
    from devteam.demo import DEMO_REQUEST, WORKSPACE, build

    brain = choose_brain(parser, args.brain, demo=True)
    config = Path(build())
    if args.setup:
        return setup(str(WORKSPACE), WORKSPACE / "repos.surveyed.toml", "", None)
    request = " ".join(args.request)
    if brain == "mock" or not request:
        request = DEMO_REQUEST
    return team(request, config, args.keep)


def cmd_agent(parser, args) -> int:
    choose_brain(parser, args.brain, demo=False, allow_mock=True)  # mock: answers for the demo repos
    port = args.port or {"planner": 9101, "coder": 9102, "qa": 9103, "surveyor": 9104}[args.name]
    try:
        return subprocess.call(agent_command(args.name, port), env=agent_env())
    except KeyboardInterrupt:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="devteam", description="An A2A team of AI agents that changes several repos at once.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    brain = argparse.ArgumentParser(add_help=False)
    brain.add_argument("--brain", choices=BRAINS, help="how the agents think. Default: api if "
                       "ANTHROPIC_API_KEY is set, otherwise subscription (mock for demo)")
    keep = argparse.ArgumentParser(add_help=False)
    keep.add_argument("--keep", action="store_true", help="keep the worktree folders")

    run = commands.add_parser("run", parents=[brain, keep], help="change your repos",
                              description="Plan, implement and test a change across the repos "
                              "in repos.toml. Results go to branches team/<request>-<time>.")
    run.add_argument("request", nargs="+", help="what you want changed")
    run.add_argument("--config", help=f"default: ./{CONFIG_NAME}, else {USER_CONFIG}")
    run.set_defaults(handler=cmd_run)

    setup_cmd = commands.add_parser("setup", parents=[brain], help="write repos.toml for a folder",
                                    description="Find the git repos in FOLDER, let the Surveyor "
                                    "describe them, and write repos.toml. Review it before `run`.")
    setup_cmd.add_argument("folder", help="folder that holds your repos (or one repo)")
    setup_cmd.add_argument("hints", nargs="*", help="anything the Surveyor should know")
    setup_cmd.add_argument("--only", nargs="+", metavar="REPO", help="only these repo folder names")
    setup_cmd.add_argument("--config", metavar="OUT",
                           help=f"where to write it (default: ./{CONFIG_NAME}; an old one becomes .bak)")
    setup_cmd.set_defaults(handler=cmd_setup)

    demo = commands.add_parser("demo", parents=[brain, keep], help="try it on two tiny repos",
                               description="Create two demo repos and run the team on them. "
                               "Needs no Claude unless you pass --brain.")
    demo.add_argument("request", nargs="*", help="your own request (not with the mock brain)")
    demo.add_argument("--setup", action="store_true", help="run the Surveyor on the demo repos instead")
    demo.set_defaults(handler=cmd_demo)

    agent = commands.add_parser("agent", parents=[brain], help="run one agent on its own",
                                description="Serve one agent until Ctrl+C, e.g. to read its "
                                "Agent Card with curl.\n\n" + "\n".join(
                                    f"  {name:<9} {text}" for name, text in AGENTS.items()),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    agent.add_argument("name", choices=AGENTS)
    agent.add_argument("--port", type=int)
    agent.set_defaults(handler=cmd_agent)

    args = parser.parse_args()
    return args.handler(commands.choices[args.command], args)


if __name__ == "__main__":
    sys.exit(main())
