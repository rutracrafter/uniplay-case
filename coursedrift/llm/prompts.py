"""The prompt wire format: delimiters, task markers, and block access.

Defined here rather than inside the templates because both sides need it —
the templates write these blocks (phase 6) and the fake client reads them
(phase 5). Source documents always travel inside a labelled block, never
inline, so there is a syntactic boundary between our instructions and
material an admin uploaded.
"""

from __future__ import annotations

import re

TASK_ADJUDICATE = "ADJUDICATE"
TASK_REMEDIATE = "REMEDIATE"

# Block names used across both tasks.
TASK = "TASK"
QUESTION = "QUESTION"
OPTIONS = "OPTIONS"
CORRECT = "CORRECT"
SOURCE_OLD = "SOURCE_OLD"
SOURCE_NEW = "SOURCE_NEW"
DIFF = "DIFF"
CONTEXT = "CONTEXT"
SECTION = "SECTION"
VOTE = "VOTE"


def block(name: str, body: str) -> str:
    """Wrap `body` in a delimited, labelled block."""
    return f"<<<{name}>>>\n{body}\n<<<END {name}>>>"


def read_block(prompt: str, name: str) -> str | None:
    """Read one block back out of a prompt. Used by the fake client."""
    match = re.search(
        rf"<<<{re.escape(name)}>>>\n(.*?)\n<<<END {re.escape(name)}>>>",
        prompt,
        re.DOTALL,
    )
    return match.group(1) if match else None


def read_task(prompt: str) -> str | None:
    return read_block(prompt, TASK)
