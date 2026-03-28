#!/usr/bin/env python3
"""Block shell commands that would read repo-local secret files or print key values."""

from __future__ import annotations

import json
import re
import sys


ENV_FILE_PATTERNS = [
    re.compile(r"(^|[\s'\"`=])(?:\./)?\.env(?:\.[A-Za-z0-9_.-]+)?(?=$|[\s'\"`])"),
    re.compile(r"(^|[\s'\"`])(?:[^\s'\"`]+/)+\.env(?:\.[A-Za-z0-9_.-]+)?(?=$|[\s'\"`])"),
]

GOOGLE_KEY_PATTERNS = [
    re.compile(r"\$\{?GOOGLE_API_KEY\b"),
    re.compile(r"\bprintenv\s+GOOGLE_API_KEY\b"),
    re.compile(r"\benv\b[^\n]*\bGOOGLE_API_KEY\b"),
]

SOURCE_ENV_PATTERN = re.compile(
    r"(^|[;&|]\s*|\b)(?:source|\.)\s+(?:\./)?\.env(?:\.[A-Za-z0-9_.-]+)?(?=$|[\s'\"`])"
)


def should_block(command: str) -> str | None:
    if SOURCE_ENV_PATTERN.search(command):
        return "Reading secrets via `source .env` is blocked for this repository."

    for pattern in ENV_FILE_PATTERNS:
        if pattern.search(command):
            return "Reading `.env` or `.env.*` is blocked for this repository."

    for pattern in GOOGLE_KEY_PATTERNS:
        if pattern.search(command):
            return "Printing or inspecting `GOOGLE_API_KEY` is blocked for this repository."

    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    command = payload.get("tool_input", {}).get("command", "")
    if not isinstance(command, str) or not command:
        return 0

    reason = should_block(command)
    if reason is None:
        return 0

    response = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
        "systemMessage": (
            f"{reason} Ask the user for a redacted example or the expected variable names instead."
        ),
    }
    json.dump(response, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
