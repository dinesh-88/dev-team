"""The `devteam` command: start the agents, run a request through them, stop them.

    devteam setup ~/code                 write ./repos.toml for the git repos in ~/code
    devteam run "Add an email field ..." change your repos (listed in repos.toml)
    devteam ask qa web@feature/x         chosen agents on chosen branches (qa, review, bench, ...)
    devteam demo                         two tiny demo repos, no Claude needed
    devteam watch coder                  follow one agent of the latest run (another terminal)
    devteam status                       how the latest run is doing (--follow to keep watching)
    devteam resume                       carry on with a run that stopped (Ctrl+C, crash, error)
    devteam clean                        delete old runs and leftover worktrees
    devteam agent planner                run one agent on its own, e.g. to curl it

`devteam <command> --help` shows the options of each command.
"""

import argparse
import asyncio
import os
import shlex
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("Python 3.11 or newer is required.")

import httpx

from devteam import cleanup, gitops, monitor
from devteam import ask
from devteam.orchestrator import (STATE_DIR, AgentError, can_resume, load_repos, new_run,
                                  run_team, survey)

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
CARD_PATH = "/.well-known/agent-card.json"
CONFIG_NAME = "repos.toml"
TEAM = ["planner", "coder", "qa"]  # the agents of `devteam run`
COLORS = ["36", "33", "35", "32", "34", "31"]  # ANSI colors for the agents in `watch`
USER_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "devteam" / CONFIG_NAME
AGENTS = {
    "planner": "Reads the repos and splits a request into per-repo work (default port 9101)",
    "coder": "Implements the plan in one repo worktree (default port 9102)",
    "qa": "Runs the tests and reviews the change across repos (default port 9103)",
    "surveyor": "Describes the repos in a folder for repos.toml (default port 9104)",
    "review": "Reviews a branch or pull request like a senior engineer (default port 9105)",
    "bench": "Benchmarks a branch against its base and reports regressions (default port 9106)",
}
PORTS = {"planner": 9101, "coder": 9102, "qa": 9103, "surveyor": 9104, "review": 9105, "bench": 9106}
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
    return [sys.executable, "-m", f"devteam.{name}", "--port", str(port), "--parent", str(os.getpid())]


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


# ---- following a run ---------------------------------------------------------

def runs_folder(config: Path) -> Path:
    return config.resolve().parent / STATE_DIR / "runs"


def latest_run(config: Path) -> Path | None:
    folder = runs_folder(config)
    runs = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
    return runs[-1] if runs else None  # names start with the date and time


def watch_command(name: str, run_dir: Path, config: Path) -> str:
    """The shortest `devteam watch` command that finds this run from the current folder."""
    if find_config(None).resolve() == config.resolve():
        return f"devteam watch {name}"
    return f"devteam watch {name} --run {shlex.quote(str(run_dir))}"


def open_terminals(run_dir: Path, names: list[str]) -> None:
    """Open one macOS Terminal window per agent, each following its log."""
    if sys.platform != "darwin":
        print("--terminals opens macOS Terminal windows. Elsewhere, run the commands above yourself.")
        return
    for name in names:
        command = (f"cd {shlex.quote(str(Path.cwd()))} && "
                   f"PYTHONPATH={shlex.quote(str(PACKAGE_PARENT))} {shlex.quote(sys.executable)} "
                   f"-m devteam watch {name} --run {shlex.quote(str(run_dir))}")
        quoted = command.replace("\\", "\\\\").replace('"', '\\"')  # inside an AppleScript string
        subprocess.run(["osascript", "-e", f'tell application "Terminal" to do script "{quoted}"'],
                       capture_output=True)


def follow(run_dir: Path, agent: str | None) -> None:
    """Print the run's log files as they grow. Stops when the run has finished."""
    logs = run_dir / "logs"
    want = agent.replace(":", "-") if agent else None
    color = sys.stdout.isatty()
    offsets: dict[Path, int] = {}
    partial: dict[Path, bytes] = {}
    print(f"Following {agent or 'all agents'} in {run_dir}\n(Ctrl+C to stop)\n", flush=True)
    while True:
        finished = (run_dir / "summary.md").exists()  # checked first, so no last lines are missed
        printed = False
        for number, path in enumerate(sorted(logs.glob("*.log"))):
            if want and path.stem != want and not path.stem.startswith(want + "-"):
                continue
            with path.open("rb") as file:
                file.seek(offsets.get(path, 0))
                chunk = file.read()
            if not chunk:
                continue
            offsets[path] = offsets.get(path, 0) + len(chunk)
            *lines, partial[path] = (partial.get(path, b"") + chunk).split(b"\n")
            prefix = "" if path.stem == want else f"[{path.stem}] "
            if prefix and color:
                prefix = f"\033[{COLORS[number % len(COLORS)]}m{prefix}\033[0m"
            for line in lines:
                print(prefix + line.decode("utf-8", "replace"), flush=True)
            printed = True
        if finished and not printed:
            print(f"\nThe run has finished. Summary: {run_dir / 'summary.md'}")
            return
        time.sleep(0.3)


# ---- the commands ------------------------------------------------------------

def team(request: str, config: Path, keep: bool, parallel: int, terminals: bool) -> int:
    print(f"Request: {request}\nConfig:  {config}")
    return drive(new_run(request, config), config, keep, parallel, terminals)


def drive(run_dir: Path, config: Path, keep: bool, parallel: int, terminals: bool,
          extra_rounds: int = 0) -> int:
    """Start the team's agents and let them work on the run in `run_dir`, new or resumed."""

    def started(run_dir: Path) -> None:
        print("\nFollow each agent in its own terminal:")
        for name in TEAM:
            print(f"  {watch_command(name, run_dir, config)}")
        if terminals:
            open_terminals(run_dir, TEAM)

    with agents_running(TEAM) as urls:
        try:
            passed = asyncio.run(run_team(run_dir, urls, keep, parallel, started, extra_rounds))
            return 0 if passed else 1
        except (AgentError, gitops.GitError) as error:
            print(f"\nStopped: {error}")
            return 1
        except KeyboardInterrupt:
            return 130  # the summary above says how to resume


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
    return team(request, config, args.keep, args.parallel, args.terminals)


def cmd_ask(parser, args) -> int:
    agents = list(dict.fromkeys(a.strip() for a in args.agents.split(",") if a.strip()))
    unknown = [a for a in agents if a not in ask.AGENTS]
    if not agents or unknown:
        parser.error(f"choose from {', '.join(ask.AGENTS)} (several with commas, e.g. review,bench)")
    if "coder" in agents and len(agents) > 1:
        parser.error("the coder changes the code, so run it on its own, not next to agents that check it")
    message = " ".join(args.message or [])
    if agents in (["coder"], ["planner"]) and not message:
        parser.error(f"say what the {agents[0]} should do with -m \"...\"")
    config = find_config(args.config)
    repos = load_repos(config)
    choose_brain(parser, args.brain, demo=False, allow_mock=True)  # mock: answers for the demo repos
    try:
        targets = ask.resolve_targets(args.targets, repos, args.base)
    except (AgentError, gitops.GitError) as error:
        print(f"\nStopped: {error}")
        return 1
    if "bench" in agents and not any(t.repo["bench"] for t in targets):
        parser.error("no `bench` command in repos.toml for these repos: add one, e.g. "
                     "bench = \"pytest benchmarks --benchmark-only\"")
    print(f"Agents:  {', '.join(agents)}\nConfig:  {config}")
    for t in targets:
        print(f"  {t.name:<16} {t.label} ({t.head[:10]}) vs {t.base_label} ({t.base[:10]})")

    def started(run_dir: Path) -> None:
        names = [f"coder:{t.name}" for t in targets] if agents == ["coder"] else agents
        print("\nFollow each agent in its own terminal:")
        for name in names:
            print(f"  {watch_command(name, run_dir, config)}")
        if args.terminals:
            open_terminals(run_dir, names)

    with agents_running(agents) as urls:
        try:
            ok = asyncio.run(ask.ask(agents, targets, message, config, urls, args.keep,
                                     args.threshold, started))
            return 0 if ok else 1
        except (AgentError, gitops.GitError) as error:
            print(f"\nStopped: {error}")
            return 1


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
    return team(request, config, args.keep, args.parallel, args.terminals)


def cmd_watch(parser, args) -> int:
    config = find_config(args.config)
    if args.run:
        run_dir = Path(args.run).expanduser()
    elif args.next:  # wait for a run that has not started yet
        before = latest_run(config)
        print(f"Waiting for the next run in {runs_folder(config)} ...", flush=True)
        while (run_dir := latest_run(config)) in (before, None):
            time.sleep(0.5)
    else:
        run_dir = latest_run(config)
        if run_dir is None:
            parser.error(f"no runs yet in {runs_folder(config)} (use --next to wait for one)")
    try:
        follow(run_dir, args.agent)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_resume(parser, args) -> int:
    config = find_config(args.config)
    run_dir = Path(args.run).expanduser().resolve() if args.run else latest_run(config)
    if run_dir is None:
        parser.error(f"no runs yet in {runs_folder(config)}")
    status = monitor.read_status(run_dir)
    if monitor.state(run_dir, status) == "running":
        parser.error(f"{run_dir.name} is still running (`devteam status` shows it)")
    if not status.get("kind"):
        parser.error(f"{run_dir.name} is from an older devteam that did not save its progress")
    why = can_resume(run_dir)
    if why and not (args.rounds and status.get("verdict") and not status["verdict"]["passed"]):
        parser.error(f"cannot resume {run_dir.name}: {why}")
    stored = status.get("brain")
    choose_brain(parser, args.brain or stored, demo=False, allow_mock=stored == "mock")
    done = (f"{status['qa_round']} QA round(s) done" if status.get("qa_round")
            else "planned" if status.get("plan") else "not planned yet")
    print(f"Run:     {run_dir.name}\nRequest: {status['request']}\nSo far:  {done}, "
          f"stopped at: {status.get('error') or status.get('phase')}")
    return drive(run_dir, Path(status["config"]), args.keep, args.parallel, args.terminals,
                 args.rounds)


def cmd_status(parser, args) -> int:
    config = find_config(args.config)
    if args.all:
        folder = runs_folder(config)
        print(monitor.overview(sorted(p for p in folder.iterdir() if p.is_dir())
                               if folder.is_dir() else []))
        return 0
    run_dir = Path(args.run).expanduser() if args.run else latest_run(config)
    if run_dir is None:
        parser.error(f"no runs yet in {runs_folder(config)}")
    if not args.follow:
        print(monitor.report(run_dir))
        return 0
    try:
        while True:
            text = monitor.report(run_dir)
            print("\033[H\033[2J" + text + "\n\n(refreshing every 2s, Ctrl+C to stop)", flush=True)
            if monitor.state(run_dir, monitor.read_status(run_dir)) != "running":
                return 0
            time.sleep(2)
    except KeyboardInterrupt:
        return 0


def cmd_clean(parser, args) -> int:
    config = find_config(args.config)
    runs = cleanup.find_runs(runs_folder(config))
    doomed = cleanup.pick(runs, args.keep, args.older_than, args.all)
    repos = load_repos(config) if config.is_file() else {}

    merged: dict[str, list[str]] = {}
    for name, repo in repos.items():
        done, open_ = cleanup.team_branches(repo)
        if args.branches and done:
            merged[name] = done
        for branch in open_:
            print(f"Not merged, kept: {name} {branch}  (review it, or `git -C {repo['path']} "
                  f"branch -D {branch}`)")

    busy = [run for run in runs if run.busy]
    for run in busy:
        print(f"Skipped, may still be running: {run.path.name}")
    if not doomed and not merged:
        kept = "" if args.all else f", keeping the newest {args.keep}"
        print(f"Nothing to clean in {runs_folder(config)} ({len(runs)} run(s){kept}).")
        return 0

    print(f"\nTo delete from {runs_folder(config)}:")
    for run in doomed:
        extra = f", {len(run.worktrees)} worktree(s)" if run.worktrees else ""
        print(f"  {run.path.name}  ({cleanup.human_size(run.size)}{extra})")
    for name, branches in merged.items():
        print(f"  {name}: merged branches {', '.join(branches)}")
    if not args.yes:
        if not sys.stdin.isatty():
            print("\nNothing deleted. Add --yes to delete without asking.")
            return 1
        if input("\nDelete these? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing deleted.")
            return 0

    warnings, branches_deleted = [], 0
    for run in doomed:
        warnings += cleanup.remove_run(run)
    for repo in repos.values():
        cleanup.prune(repo)
    for name, branches in merged.items():
        deleted, problems = cleanup.delete_branches(repos[name], branches)
        branches_deleted += deleted
        warnings += problems
    total = sum(run.size for run in doomed)
    print(f"\nDeleted {len(doomed)} run(s), {cleanup.human_size(total)}"
          + (f", and {branches_deleted} merged branch(es)" if merged else "") + ".")
    for warning in warnings:
        print(f"  note: {warning}")
    return 0


def cmd_agent(parser, args) -> int:
    choose_brain(parser, args.brain, demo=False, allow_mock=True)  # mock: answers for the demo repos
    port = args.port or PORTS[args.name]
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
    team_options = argparse.ArgumentParser(add_help=False)
    team_options.add_argument("--keep", action="store_true", help="keep the worktree folders")
    team_options.add_argument("--parallel", type=int, default=3, metavar="N",
                      help="coder conversations at the same time (default 3, 1 = one by one)")
    team_options.add_argument("--terminals", action="store_true",
                      help="open a macOS Terminal window per agent that follows its log")

    run = commands.add_parser("run", parents=[brain, team_options], help="change your repos",
                              description="Plan, implement and test a change across the repos "
                              "in repos.toml. Results go to branches team/<request>-<time>.")
    run.add_argument("request", nargs="+", help="what you want changed")
    run.add_argument("--config", help=f"default: ./{CONFIG_NAME}, else {USER_CONFIG}")
    run.set_defaults(handler=cmd_run)

    ask_cmd = commands.add_parser(
        "ask", parents=[brain], help="chosen agents on chosen repos and branches",
        description=ask.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask_cmd.add_argument("agents", metavar="AGENT[,AGENT]",
                         help=f"one or more of {', '.join(ask.AGENTS)}, comma-separated")
    ask_cmd.add_argument("targets", nargs="*", metavar="REPO[@REF|#PR]",
                         help="repos from repos.toml (default: all, at HEAD)")
    ask_cmd.add_argument("-m", "--message", nargs="+", metavar="TEXT",
                         help="instructions or focus (required for coder and planner)")
    ask_cmd.add_argument("--base", metavar="REF", help="compare with this branch or commit")
    ask_cmd.add_argument("--threshold", type=float, default=5.0, metavar="PCT",
                         help="bench: a change bigger than this percent counts (default 5)")
    ask_cmd.add_argument("--keep", action="store_true", help="keep the worktree folders")
    ask_cmd.add_argument("--terminals", action="store_true",
                         help="open a macOS Terminal window per agent that follows its log")
    ask_cmd.add_argument("--config", help=f"default: ./{CONFIG_NAME}, else {USER_CONFIG}")
    ask_cmd.set_defaults(handler=cmd_ask)

    setup_cmd = commands.add_parser("setup", parents=[brain], help="write repos.toml for a folder",
                                    description="Find the git repos in FOLDER, let the Surveyor "
                                    "describe them, and write repos.toml. Review it before `run`.")
    setup_cmd.add_argument("folder", help="folder that holds your repos (or one repo)")
    setup_cmd.add_argument("hints", nargs="*", help="anything the Surveyor should know")
    setup_cmd.add_argument("--only", nargs="+", metavar="REPO", help="only these repo folder names")
    setup_cmd.add_argument("--config", metavar="OUT",
                           help=f"where to write it (default: ./{CONFIG_NAME}; an old one becomes .bak)")
    setup_cmd.set_defaults(handler=cmd_setup)

    demo = commands.add_parser("demo", parents=[brain, team_options], help="try it on two tiny repos",
                               description="Create two demo repos and run the team on them. "
                               "Needs no Claude unless you pass --brain.")
    demo.add_argument("request", nargs="*", help="your own request (not with the mock brain)")
    demo.add_argument("--setup", action="store_true", help="run the Surveyor on the demo repos instead")
    demo.set_defaults(handler=cmd_demo)

    watch = commands.add_parser("watch", help="follow the agents of a run",
                                description="Print an agent's log as it grows: what it was asked, "
                                "what the model says, each tool call and the result. Run it in "
                                "its own terminal per agent. Works on finished runs too.")
    watch.add_argument("agent", nargs="?", help="planner, coder, qa or e.g. coder:web "
                       "(default: all, interleaved)")
    watch.add_argument("--next", action="store_true", help="wait for the next run to start")
    watch.add_argument("--run", metavar="DIR", help="this run folder instead of the latest")
    watch.add_argument("--config", help=f"find runs next to this repos.toml (default as `run`)")
    watch.set_defaults(handler=cmd_watch)

    resume = commands.add_parser(
        "resume", parents=[brain, team_options], help="carry on with a run that stopped",
        description="Continue a `devteam run` that was stopped (Ctrl+C, a crash, an error) "
        "where it left off: the plan is reused, repos already coded are not coded again, "
        "an interrupted coder gets its unfinished edits back, and QA goes on with the next "
        "round. --rounds N also gives a run that did not pass QA N more rounds.")
    resume.add_argument("--run", metavar="DIR", help="this run folder instead of the latest")
    resume.add_argument("--rounds", type=int, default=0, metavar="N",
                        help="N more QA rounds (also for a run that finished without passing)")
    resume.add_argument("--config", help="runs next to this repos.toml (default as `run`)")
    resume.set_defaults(handler=cmd_resume)

    status = commands.add_parser("status", help="how a run is doing",
                                 description="Show a run's phase, each repo's state, whether the "
                                 "agents still answer, and each conversation's latest activity. "
                                 "Warns about conversations that have gone quiet.")
    status.add_argument("--follow", "-f", action="store_true", help="refresh until the run ends")
    status.add_argument("--all", action="store_true", help="one line per run instead")
    status.add_argument("--run", metavar="DIR", help="this run folder instead of the latest")
    status.add_argument("--config", help="runs next to this repos.toml (default as `run`)")
    status.set_defaults(handler=cmd_status)

    clean = commands.add_parser("clean", help="delete old runs and leftover worktrees",
                                description="Delete old run folders (with any worktrees left in "
                                "them) and forget deleted worktrees. Shows what it will delete "
                                "and asks first. Unmerged team/* branches are never deleted.")
    clean.add_argument("--keep", type=int, default=5, metavar="N",
                       help="keep the newest N runs (default 5)")
    clean.add_argument("--older-than", type=int, metavar="DAYS",
                       help="only delete runs older than this")
    clean.add_argument("--all", action="store_true", help="delete every run that is not running")
    clean.add_argument("--branches", action="store_true",
                       help="also delete team/* branches already merged into the base branch")
    clean.add_argument("--yes", "-y", action="store_true", help="do not ask")
    clean.add_argument("--config", help="runs next to this repos.toml (default as `run`)")
    clean.set_defaults(handler=cmd_clean)

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
