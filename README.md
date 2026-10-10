# A2A dev team for multi-repo changes: planner → coders → QA

Three AI agents that talk to each other over the **A2A (Agent2Agent)
protocol**, version 1.0, using the official Python SDK (`a2a-sdk`). You give
one request; they change several of your repositories on a shared branch name
and leave the branches for you to review.

```
you ──> orchestrator (A2A client, reads repos.toml)
          │ 1. request + repo list         ┌───────────────┐
          ├──────────────────────────────> │ Planner       │ reads the repos, returns
          │                                │               │ {summary, contract, repos in order}
          │ 2. one conversation per repo   ├───────────────┤
          ├──────────────────────────────> │ Coder         │ edits a git worktree, commits
          │ 3. all changed repos at once   ├───────────────┤
          ├──────────────────────────────> │ QA            │ runs each repo's tests,
          │                                └───────────────┘ reviews the diffs together
          └── failures go back to that repo's coder conversation (max 3 rounds)
```

A fourth agent, the **Surveyor** (`devteam/surveyor.py`), is used once
before all this: it looks through a folder of repos and writes `repos.toml`.

Each agent is its own process and HTTP server. `devteam` starts them on free
ports for every command and stops them when it is done.

## Install

Requires Python 3.11+ and git. From this folder:

```bash
pipx install -e .        # or: uv tool install -e .   (or: pip install -e . in a venv)
devteam --help
```

`-e` means your edits to the code take effect without reinstalling. Without
installing, `python -m devteam ...` works from this folder too.

## Try it (no Claude needed)

```bash
devteam demo
```

This creates two tiny repos in `~/.cache/devteam/demo/` (`backend` and `client`) and
replays a canned script through the real agents, tools, git and pytest. The
client's first attempt uses the wrong field name on purpose: QA fails it, the
coder fixes it, QA passes it.

Then let real Claude do the same demo: `devteam demo --brain subscription`

## Use it on your repos

1. Choose how the agents think (`--brain`):

   | Brain | What it uses | Setup |
   |---|---|---|
   | `subscription` (default) | Your Claude Code login on a Pro, Max or Team plan. Usage counts against your plan's limits. | Be signed in to Claude Code on this machine (run `claude` once and log in). |
   | `api` | An Anthropic API key, billed per use. Default when `ANTHROPIC_API_KEY` is set. | `export ANTHROPIC_API_KEY=sk-ant-...` |

   The subscription brain is for your own use on your own machine. Anthropic
   does not allow offering a Claude login to other people inside a product.
2. Pick a folder to work from (a workspace) and let the Surveyor write its
   `repos.toml`, then review it:

   ```bash
   cd ~/code
   devteam setup .                                   # every git repo in here (3 levels deep)
   devteam setup . --only api web                    # only these folders
   devteam setup . "tests need: docker compose up -d db"   # hints
   ```

   It reads each repo (README, build files, CI config) and fills in `name`,
   `description`, `setup` and `test`; `base` is the repo's default branch. An
   existing `repos.toml` is kept as `repos.toml.bak`. The `setup`/`test`
   commands are a model's suggestions, and QA will run them: check them, and
   the `# check:` notes, before step 3. Try it offline with
   `devteam demo --setup`.

   Or write it by hand from `repos.example.toml`. A good `description` and a
   working `test` command per repo matter most.
3. From the same folder: `devteam run "Add an email field to the user API and show it in the web app"`

   `devteam run` uses `./repos.toml`, else `~/.config/devteam/repos.toml`, or
   the file you pass with `--config`.

Start with a small, low-risk request and read the diffs. When it finishes you get:

- in each changed repo, a new branch `team/<request>-<time>` with commits
- next to `repos.toml`, in `.devteam/runs/<time>-<request>/`: `plan.md`, one
  `.diff` per repo, `summary.md`

Nothing is pushed and your checked-out files are not touched. Review with the
`git diff` command printed at the end, then merge or delete the branch.

## Watch the agents work

`devteam run` prints one line per step. For the full story per agent (what it
was asked, what the model says, each tool call, the results), follow its log
in another terminal, from the same folder:

```bash
devteam watch planner        # or: coder, qa, coder:web (one repo), or nothing for all
devteam watch coder --next   # open it before `devteam run`; waits for the run to start
devteam run --terminals "…"  # macOS: opens a Terminal window per agent for you
```

In VS Code, split the terminal panel and run one `devteam watch` in each.
The logs stay in `.devteam/runs/<run>/logs/`, so `devteam watch --run <folder>`
replays a finished run.

## Parallel work

The planner says which repos must wait for which (`after`: a consumer waits
for the API it uses). Repos that do not wait for each other are coded at the
same time, and fixes after QA run in parallel too. QA runs every repo's tests
at once. `--parallel N` caps how many coder conversations run together
(default 3; `--parallel 1` is one by one). Each parallel conversation uses
the model at the same time, so on a subscription the limits are reached sooner.

Other settings: `TEAM_MODEL` to pick a model (default: `claude-sonnet-5-5` for
`api`, your Claude Code default for `subscription`), `--keep` to leave the
worktree folders in place.

## How multi-repo works

- **Contract.** The planner writes down what the repos must agree on (field
  names, endpoints, types). Every coder and QA gets it.
- **Order.** The planner lists providers before consumers and says which repo
  waits for which. A coder receives the diffs of the repos finished before it
  and can read their new code; independent repos are coded in parallel.
- **Worktrees.** Each repo gets a separate working folder on the new branch
  (`git worktree`), so your own checkout stays as it is.
- **QA across repos.** Tests run per repo with the command from `repos.toml`.
  Then one review reads all diffs together and reports mismatches, assigned
  to the repo that has to change.

## Where each A2A concept lives

| Concept | What it is | Where to read it |
|---|---|---|
| Agent Card | JSON at `/.well-known/agent-card.json` describing the agent, its skills and its URL | `build_card()` and `serve()` in `devteam/a2a_server.py` |
| Message / Part | What you send: a role plus text, data or file parts | `send()` and its callers in `devteam/orchestrator.py` |
| Data part | Structured JSON next to the text | the plan (`devteam/planner.py`), the verdict (`devteam/qa.py`) |
| Task | The server's record of one piece of work, with a state | `TeamAgent.execute()` in `devteam/a2a_server.py` |
| Artifact | The result attached to a task | `updater.add_artifact(...)` in each agent |
| Streaming | Live status events while the agent works | `reporters()` in `devteam/a2a_server.py`, the `async for` loop in `send()` in `devteam/orchestrator.py` |
| Metadata | Free-form fields for extensions; ours tags status messages as `tool` or `text` | `say()` in `devteam/a2a_server.py`, read in `send()` |
| Context | `context_id` links follow-up messages to earlier ones | `devteam/coder.py`: one conversation per repo, reused for fixes |

Reading order (all in `devteam/`): `a2a_server.py` → `planner.py` →
`orchestrator.py` → `coder.py` → `qa.py`. `cli.py` is the `devteam` command.
`brain.py`, `tools.py` and `gitops.py` are not A2A: they are what happens
inside an agent.

## See the protocol with your own eyes

```bash
devteam demo                           # once, to create the demo repos
devteam agent planner --brain mock     # leave running (port 9101), use a second terminal

curl http://127.0.0.1:9101/.well-known/agent-card.json

curl -X POST http://127.0.0.1:9101/ \
  -H "Content-Type: application/json" -H "A2A-Version: 1.0" \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{
        "messageId":"m1","role":"ROLE_USER","parts":[
          {"text":"Add an email field to users"},
          {"data":{"repos":[{"name":"backend","path":"'"$HOME"'/.cache/devteam/demo/backend","description":"user service"},
                            {"name":"client","path":"'"$HOME"'/.cache/devteam/demo/client","description":"formats users"}]}}]}}}'
```

The reply is a Task with `status.state = TASK_STATE_COMPLETED` and the plan as
a data part in `artifacts`.

## Safety and limits

- **Models never choose commands.** They can list, read, search and edit files
  inside the repos you configured, and nothing else. Only the `setup` and
  `test` commands in `repos.toml` are executed. If the Surveyor suggested
  them, they are only run after you have saved and reviewed the file. With the
  subscription brain, Claude Code's own tools (shell, web, file access) are
  switched off and it only gets these same tools.
- **The tests do run the changed code on your machine.** Use a container or VM
  if that worries you for a given repo.
- **Code is sent to Anthropic** (with either brain) when the agents read your files. Check
  that this is allowed for work repositories before pointing the team at them.
- The worktree starts from the last commit, so uncommitted changes in your
  checkout are not part of what the coder builds on.
- A fresh worktree has no installed dependencies (`node_modules`, virtual
  environments). Use `setup` in `repos.toml` to install them.
- All agents must run on the machine that holds the repos, because they share
  folders. Tasks and conversations live in memory only.
- There is no authentication. The agents listen on 127.0.0.1 only.
- Usage grows with repo size and rounds, since every file an agent reads is
  sent to the model. On a subscription that eats into your plan's limits.

## Exercises

1. Show the run in a web page: serve `.devteam/runs/` and stream the logs.
2. Let the planner ask you a question: call `updater.requires_input(...)` and
   handle the `input_required` state in the orchestrator.
3. Add another agent that opens pull requests for the finished branches.
4. Give one agent a different brain (another model or framework) by changing
   only its use of `devteam/brain.py`. Nothing else changes, which is the point
   of A2A.
