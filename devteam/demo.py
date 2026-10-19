"""Builds two tiny git repos to try the team on: `backend` and `client`.

    devteam demo

The repos are recreated in ~/.cache/devteam/demo/ on every run.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "devteam" / "demo"
DEMO_REQUEST = "Add an email field to users in the backend and show it in the client's output."

FILES = {
    "backend": {
        "users.py": '''"""User service."""

USERS = {1: {"id": 1, "name": "Ada Lovelace"}}


def get_user(user_id: int) -> dict:
    """Return the public fields of a user."""
    user = USERS[user_id]
    return {"id": user["id"], "name": user["name"]}
''',
        "test_users.py": '''from users import get_user


def test_get_user():
    assert get_user(1) == {"id": 1, "name": "Ada Lovelace"}
''',
    },
    "client": {
        "display.py": '''"""Formats the user dictionaries returned by the backend's get_user()."""


def format_user(user: dict) -> str:
    return f"{user['name']} (#{user['id']})"
''',
        "test_display.py": '''from display import format_user


def test_format_user():
    assert format_user({"id": 1, "name": "Ada Lovelace"}) == "Ada Lovelace (#1)"
''',
    },
}
DESCRIPTIONS = {
    "backend": "Python user service. Owns the user data and get_user().",
    "client": "Python client that formats users returned by the backend.",
}


def _make_writable(function, path, _):
    os.chmod(path, stat.S_IWRITE)
    function(path)


def build() -> str:
    """Create the repos and a matching config file. Returns the config path."""
    if WORKSPACE.exists():
        key = "onexc" if sys.version_info >= (3, 12) else "onerror"
        shutil.rmtree(WORKSPACE, **{key: _make_writable})

    test_command = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'
    bench_commands = {  # for `devteam ask bench`: timeit prints "N loops, best of 5: X usec per loop"
        "backend": f'"{sys.executable}" -m timeit -s "from users import get_user" "get_user(1)"',
        "client": f'"{sys.executable}" -m timeit -s "from display import format_user" '
                  '"format_user({\'id\': 1, \'name\': \'Ada\', \'email\': \'a@b.c\'})"',
    }
    config = []
    for name, files in FILES.items():
        repo = WORKSPACE / name
        repo.mkdir(parents=True)
        for filename, content in files.items():
            (repo / filename).write_text(content, encoding="utf-8")
        (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
        for command in (
            ["git", "init", "-q", "-b", "main"],
            ["git", "config", "user.name", "Demo"],
            ["git", "config", "user.email", "demo@example.com"],
            ["git", "add", "-A"],
            ["git", "commit", "-q", "-m", "Initial commit"],
        ):
            subprocess.run(command, cwd=repo, check=True)
        config.append(
            "[[repo]]\n"
            f"name = {json.dumps(name)}\n"
            f"path = {json.dumps(name)}\n"
            f"description = {json.dumps(DESCRIPTIONS[name])}\n"
            f"test = {json.dumps(test_command)}\n"
            f"bench = {json.dumps(bench_commands[name])}\n"
        )

    config_path = WORKSPACE / "repos.toml"
    config_path.write_text("\n".join(config), encoding="utf-8")
    return str(config_path)


if __name__ == "__main__":
    print(f"Demo repos created. Config: {build()}")
