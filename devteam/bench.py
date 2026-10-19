"""Benchmark agent: runs a repo's benchmark on the base and on the branch, and compares them.

Run on its own with:  devteam agent bench   (or python -m devteam.bench)

Incoming message:  text part = anything to pay attention to
                   data part = {"threshold": percent, "repos": [{"name", "base_worktree",
                                "head_worktree", "setup", "bench", "runs", "timeout"}]}
Result artifacts:  "benchmark" data part = {"passed", "threshold", "summary", "repos": {name:
                   {"status": "ok"|"failed"|"skipped", "note", "metrics": [{"name", "unit",
                   "base", "head", "change", "lower_is_better", "verdict"}]}}}
                   "benchmark.md" file part = the same as a Markdown table

Who does what, as everywhere in this team:
- The COMMAND comes from `bench` in your repos.toml, never from the model.
- The CODE runs it `runs` times on each side, alternating base and branch so
  that a busy machine slows both equally, and times every run.
- The MODEL only reads the output and pulls out the numbers it reports
  (requests/s, ns/op, ...), because every benchmark tool prints differently.
- The CODE then compares the numbers with the threshold and decides. Wall
  time is always shown, but only decides when the output had no numbers.
"""

import asyncio
import statistics
import subprocess
import time

from a2a.helpers import get_data_parts, new_data_part
from a2a.types import AgentSkill, Part

from devteam import brain, mock
from devteam.a2a_server import TeamAgent, build_card, main
from devteam.tools import Toolbox

PORT = 9106  # when run on its own; `devteam ask` picks free ports
OUTPUT_TAIL = 3000  # characters of each run's output shown to the model

SYSTEM = """You read benchmark output. The same benchmark command was run several
times on the BASE version of a repository and on the HEAD version (the branch).

Call submit_metrics with every measurement the output reports, per repository:
its name, unit, the BASE value and the HEAD value, and whether lower is better
(true for time and memory, false for throughput). When a measurement appears in
several runs, report the median. Copy numbers exactly; convert units only so
that BASE and HEAD use the same one. Do not report wall time: it is measured
separately. Leave out anything that is not a measurement.

In the summary, say in two or three sentences what changed and, if the diff in
the repository explains it, why. You can read the HEAD code with your tools."""


def finish_tool(repo_names: list[str]) -> dict:
    return {
        "name": "submit_metrics",
        "description": "Hand in the measurements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "enum": repo_names},
                            "name": {"type": "string"},
                            "unit": {"type": "string"},
                            "base": {"type": "number"},
                            "head": {"type": "number"},
                            "lower_is_better": {"type": "boolean"},
                        },
                        "required": ["repo", "name", "unit", "base", "head", "lower_is_better"],
                    },
                },
            },
            "required": ["summary", "metrics"],
        },
    }


def timed(command: str, cwd: str, timeout: float) -> tuple[bool, float, str]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, timeout, f"`{command}` did not finish within {timeout:.0f} seconds."
    seconds = time.perf_counter() - started
    return result.returncode == 0, seconds, (result.stdout + result.stderr).strip()[-OUTPUT_TAIL:]


def compare(metric: dict, threshold: float) -> dict:
    """Add the change in percent and a verdict: better, same or worse."""
    base, head = metric["base"], metric["head"]
    change = (head - base) / abs(base) * 100 if base else 0.0
    gain = -change if metric["lower_is_better"] else change
    verdict = "worse" if gain < -threshold else "better" if gain > threshold else "same"
    return {**metric, "change": round(change, 1), "verdict": verdict}


def measure(repo: dict) -> dict:
    """Set up both sides, then run the benchmark alternately. Returns timings and outputs."""
    timeout = repo.get("timeout") or 900
    sides = {"base": repo["base_worktree"], "head": repo["head_worktree"]}
    if repo.get("setup"):
        for side, folder in sides.items():
            ok, _, output = timed(repo["setup"], folder, timeout)
            if not ok:
                return {"status": "failed", "note": f"setup failed on {side}:\n{output}"}
    seconds = {"base": [], "head": []}
    outputs = {"base": [], "head": []}
    for _ in range(max(1, int(repo.get("runs") or 3))):  # JSON numbers arrive as floats
        for side, folder in sides.items():
            ok, took, output = timed(repo["bench"], folder, timeout)
            if not ok:
                return {"status": "failed", "note": f"the benchmark failed on {side}:\n{output}"}
            seconds[side].append(took)
            outputs[side].append(output)
    return {"status": "ok", "note": "", "seconds": seconds, "outputs": outputs}


def as_markdown(report: dict) -> str:
    lines = ["## Benchmark: " + ("no regressions" if report["passed"] else "REGRESSION or failure"),
             "", f"Threshold: {report['threshold']:g}%", "", report["summary"]]
    for name, repo in report["repos"].items():
        lines += ["", f"### {name}: {repo['status']}"]
        if repo["note"]:
            lines += ["", "```", repo["note"], "```"]
        if repo["metrics"]:
            lines += ["", "| metric | base | branch | change | |", "|---|---:|---:|---:|---|"]
            for m in repo["metrics"]:
                lines.append(f"| {m['name']} ({m['unit']}) | {m['base']:g} | {m['head']:g} "
                             f"| {m['change']:+.1f}% | {m['verdict']} |")
    return "\n".join(lines)


class Benchmarker(TeamAgent):
    async def work(self, context, updater):
        notes = context.get_user_input()
        data = get_data_parts(context.message.parts)[0]
        threshold = float(data.get("threshold", 5))
        report = {"threshold": threshold, "summary": "", "repos": {}}

        # One repo at a time, never in parallel: benchmarks running together slow each other.
        measured = {}
        for repo in data["repos"]:
            name = repo["name"]
            if not repo.get("bench"):
                report["repos"][name] = {"status": "skipped", "metrics": [],
                                         "note": "No `bench` command for this repo in repos.toml."}
                continue
            await self.progress(context, updater, f"{name}: running `{repo['bench']}` "
                                f"{int(repo.get('runs') or 3)}x on base and branch")
            result = await asyncio.to_thread(measure, repo)
            await self.progress(context, updater, f"{name}: benchmark {result['status']}")
            if result["status"] != "ok":
                report["repos"][name] = {"status": "failed", "note": result["note"], "metrics": []}
                continue
            measured[name] = result
            wall = {side: round(statistics.median(times), 3)
                    for side, times in result["seconds"].items()}
            report["repos"][name] = {"status": "ok", "note": "", "metrics": [compare(
                {"name": "wall time", "unit": "s", "lower_is_better": True, **wall}, threshold)]}

        if measured:
            await self.progress(context, updater, "Reading the measurements")
            sections = [notes] if notes else []
            for name, result in measured.items():
                for side in ("base", "head"):
                    runs = "\n\n".join(f"run {number}:\n{output}" for number, output
                                       in enumerate(result["outputs"][side], 1))
                    sections.append(f"## {name}, {side.upper()}\n```\n{runs}\n```")
            toolbox = Toolbox({r["name"]: r["head_worktree"] for r in data["repos"]
                               if r["name"] in measured})  # read-only
            chat = brain.Conversation(SYSTEM, toolbox, finish_tool(sorted(measured)))
            found = await chat.ask("\n\n".join(sections), **self.reporters(context, updater),
                                   mock=mock.BENCH)
            report["summary"] = found["summary"]
            for metric in found["metrics"]:
                repo = report["repos"].get(metric.pop("repo"))
                if repo and repo["status"] == "ok":
                    repo["metrics"].append(compare(metric, threshold))

        # Wall time includes start-up and the tool's own calibration, so it only
        # decides when the output had no measurements of its own.
        for repo in report["repos"].values():
            if len(repo["metrics"]) > 1:
                repo["metrics"][0]["verdict"] = "info"
        report["passed"] = all(
            repo["status"] != "failed" and all(m["verdict"] != "worse" for m in repo["metrics"])
            for repo in report["repos"].values())
        await updater.add_artifact([new_data_part(report)], name="benchmark")
        await updater.add_artifact(
            [Part(text=as_markdown(report), filename="benchmark.md", media_type="text/markdown")],
            name="benchmark.md",
        )


CARD = build_card(
    name="Benchmarker",
    description="Runs each repository's benchmark on the base and on a branch and reports regressions.",
    skill=AgentSkill(
        id="benchmark",
        name="Benchmark a branch against its base",
        description="Send notes as text and {threshold, repos:[{name,base_worktree,head_worktree,"
        "setup,bench,runs,timeout}]} as data. Returns a `benchmark` data artifact and "
        "`benchmark.md`.",
        tags=["performance", "benchmark", "multi-repo"],
    ),
)

if __name__ == "__main__":
    main(CARD, Benchmarker(), PORT)
