"""Canned model behaviour for the demo (`devteam demo`).

Each script is the list of tool calls a model might have made. The demo
request is: add an email field to users in `backend` and show it in `client`.
The client's first attempt deliberately uses the wrong field name, so you can
watch QA fail it and the coder fix it.
"""

PLANNER = [
    ("list_files", {"repo": "backend"}),
    ("read_file", {"repo": "backend", "path": "users.py"}),
    ("list_files", {"repo": "client"}),
    ("read_file", {"repo": "client", "path": "display.py"}),
    ("submit_plan", {
        "summary": "Expose each user's email from the backend and show it in the client.",
        "contract": 'get_user() returns {"id": int, "name": str, "email": str}. '
                    "The new key is exactly `email`.",
        "repos": [
            {"name": "backend", "steps": "Store an email per user in users.py, return it "
                                         "from get_user(), and update test_users.py."},
            {"name": "client", "steps": "format_user() in display.py renders "
                                        "'Name <email> (#id)'. Update test_display.py."},
        ],
    }),
]

_BACKEND = [
    ("read_file", {"repo": "backend", "path": "users.py"}),
    ("edit_file", {"repo": "backend", "path": "users.py",
                   "old": '"name": "Ada Lovelace"}',
                   "new": '"name": "Ada Lovelace", "email": "ada@example.com"}'}),
    ("edit_file", {"repo": "backend", "path": "users.py",
                   "old": '"name": user["name"]}',
                   "new": '"name": user["name"], "email": user["email"]}'}),
    ("edit_file", {"repo": "backend", "path": "test_users.py",
                   "old": '"name": "Ada Lovelace"}',
                   "new": '"name": "Ada Lovelace", "email": "ada@example.com"}'}),
    ("submit_change", {"summary": "Add email to users\n\nget_user() now returns the email."}),
]

_CLIENT_FIRST = [
    ("read_file", {"repo": "backend", "path": "users.py"}),  # peek at the other repo
    ("edit_file", {"repo": "client", "path": "display.py",
                   "old": "{user['name']} (#",
                   "new": "{user['name']} <{user['mail']}> (#"}),  # bug: wrong key
    ("edit_file", {"repo": "client", "path": "test_display.py",
                   "old": '"name": "Ada Lovelace"}) == "Ada Lovelace (#1)"',
                   "new": '"name": "Ada Lovelace", "email": "ada@example.com"}) '
                          '== "Ada Lovelace <ada@example.com> (#1)"'}),
    ("submit_change", {"summary": "Show the user's email\n\nformat_user() includes the email."}),
]

_CLIENT_FIX = [
    ("edit_file", {"repo": "client", "path": "display.py",
                   "old": "user['mail']", "new": "user['email']"}),
    ("submit_change", {"summary": "Read the email from the `email` key"}),
]

REVIEW = [("submit_review", {"issues": []})]


def coder(repo: str, round_number: int):
    if repo == "backend":
        return _BACKEND
    if repo == "client":
        return _CLIENT_FIRST if round_number == 1 else _CLIENT_FIX
    return None

SURVEYOR = [
    ("list_files", {"repo": "backend"}),
    ("read_file", {"repo": "backend", "path": "users.py"}),
    ("list_files", {"repo": "client"}),
    ("read_file", {"repo": "client", "path": "display.py"}),
    ("submit_repos", {"repos": [
        {"repo": "backend", "name": "backend",
         "description": "Python user service. Owns the user data and get_user().",
         "setup": "", "test": "python -m pytest -q"},
        {"repo": "client", "name": "client",
         "description": "Python client that formats the users returned by the backend's get_user().",
         "setup": "", "test": "python -m pytest -q",
         "notes": "Tests assume `python` has pytest installed."},
    ]}),
]
