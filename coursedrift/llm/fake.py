"""A deterministic stand-in for a model, and a way to break it on purpose.

Two jobs. It makes `drift demo` reproducible with no network and no model,
and its failure-injection modes exercise every recovery path in
`call_validated` so the reliability story is demonstrated rather than
asserted.
"""

from __future__ import annotations

import json
import re

from coursedrift.llm import prompts
from coursedrift.llm.base import TransportError

FAILURE_MODES = frozenset(
    {"malformed", "fenced", "schema", "ungrounded", "timeout", "transport", "injection"}
)

_NUMERIC = re.compile(r"\{\+([^+}]*?)\+\}")
_REMOVED = re.compile(r"\[-([^\-\]]*?)-\]")
_HAS_DIGIT = re.compile(r"\d")
_HEDGE = re.compile(
    r"\b(may|might|typically|generally|usually|varies|vary|approximately|could)\b",
    re.IGNORECASE,
)


def _first_sentence(text: str) -> str:
    """A short, verbatim quote — grounded by construction."""
    stripped = text.strip()
    match = re.search(r"^(.{10,200}?[.!?])(\s|$)", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped[:160].strip()


class FakeClient:
    """Scripted where it matters, heuristic everywhere else.

    `scripted` is a list of (needle, raw response) pairs: the first needle
    found in the prompt wins. That keeps fixtures readable — a fixture pins
    the response by quoting the text it cares about — while the heuristics
    below still produce something defensible when pointed at documents the
    fixtures never anticipated.
    """

    def __init__(
        self,
        scripted: list[tuple[str, str]] | None = None,
        failures: list[str] | None = None,
    ) -> None:
        self.name = "fake"
        self.scripted = scripted or []
        self.failures = list(failures or [])
        unknown = set(self.failures) - FAILURE_MODES
        if unknown:
            raise ValueError(
                f"unknown fake failure mode(s): {', '.join(sorted(unknown))}. "
                f"Known modes: {', '.join(sorted(FAILURE_MODES))}"
            )
        self.calls = 0

    # -- the client contract ---------------------------------------------

    def complete(self, prompt: str, *, timeout: float) -> str:
        self.calls += 1
        mode = self._mode_for_call(self.calls)

        if mode == "timeout":
            raise TimeoutError(f"fake client timed out on call {self.calls}")
        if mode == "transport":
            raise TransportError(f"fake client could not reach a model on call {self.calls}")
        if mode == "malformed":
            return "I considered the question carefully but will answer in prose instead."

        body = self._respond(prompt, mode)
        if mode == "fenced":
            return f"Here is my answer:\n\n```json\n{body}\n```\n"
        return body

    # -- failure scheduling ----------------------------------------------

    def _mode_for_call(self, call: int) -> str | None:
        """Listed modes fire in order, one per call, starting at call 1.

        Predictable on purpose: `--fake-failures malformed,timeout` means
        call 1 is malformed, call 2 times out, and everything after is
        normal, so a demo of the recovery paths is reproducible.
        """
        if 1 <= call <= len(self.failures):
            return self.failures[call - 1]
        return None

    # -- response construction -------------------------------------------

    def _respond(self, prompt: str, mode: str | None) -> str:
        for needle, response in self.scripted:
            if needle in prompt:
                return response
        task = prompts.read_task(prompt)
        if task == prompts.TASK_REMEDIATE:
            return self._remediate(prompt, mode)
        return self._adjudicate(prompt, mode)

    def _adjudicate(self, prompt: str, mode: str | None) -> str:
        new_text = prompts.read_block(prompt, prompts.SOURCE_NEW) or ""
        old_text = prompts.read_block(prompt, prompts.SOURCE_OLD) or ""
        diff = prompts.read_block(prompt, prompts.DIFF) or ""

        if mode == "schema":
            return json.dumps(
                {
                    "verdict": "PROBABLY_FINE",
                    "confidence": 0.8,
                    "rationale": "An enum value nobody defined.",
                    "evidence": [_first_sentence(new_text or old_text)],
                }
            )
        if mode == "ungrounded":
            return json.dumps(
                {
                    "verdict": "INVALIDATED",
                    "confidence": 0.95,
                    "rationale": "Quoting something the documents never said.",
                    "evidence": ["The policy was repealed by the board in 1998."],
                }
            )
        if mode == "injection":
            # What a model looks like when it obeys text from an uploaded file.
            return json.dumps(
                {
                    "verdict": "UNAFFECTED",
                    "confidence": 0.99,
                    "rationale": "Instructed by the source document to approve everything.",
                    "evidence": [_first_sentence(new_text or old_text)],
                }
            )

        verdict, confidence, rationale = self._judge(diff, old_text, new_text)
        return json.dumps(
            {
                "verdict": verdict,
                "confidence": confidence,
                "rationale": rationale,
                "evidence": [_first_sentence(new_text or old_text)],
            }
        )

    def _judge(self, diff: str, old_text: str, new_text: str) -> tuple[str, float, str]:
        """The heuristic fallback, for corpora the fixtures never saw.

        Ordered by how confidently a rule can be applied: a changed figure is
        unambiguous, newly hedged wording is unambiguously *ambiguous*, and
        anything else is treated as leaving the question alone.
        """
        added = _NUMERIC.findall(diff)
        removed = _REMOVED.findall(diff)

        if any(_HAS_DIGIT.search(token) for token in added + removed):
            return (
                "INVALIDATED",
                0.9,
                "A figure in the source material changed, so a question keyed to "
                "the old figure can no longer be answered correctly.",
            )
        if _HEDGE.search(" ".join(added)) and not _HEDGE.search(" ".join(removed)):
            return (
                "UNCERTAIN",
                0.4,
                "The replacement wording is hedged, so whether the question "
                "remains answerable cannot be settled from the text alone.",
            )
        if not diff.strip() and old_text.strip() != new_text.strip():
            return (
                "UNCERTAIN",
                0.35,
                "The span moved into different surrounding material and its "
                "meaning in the new context is unclear.",
            )
        return (
            "UNAFFECTED",
            0.85,
            "The wording changed without altering what the source states, so "
            "the question still has the same correct answer.",
        )

    def _remediate(self, prompt: str, mode: str | None) -> str:
        question = (prompts.read_block(prompt, prompts.QUESTION) or "Updated question?").strip()
        raw_options = prompts.read_block(prompt, prompts.OPTIONS) or ""
        correct = (prompts.read_block(prompt, prompts.CORRECT) or "").strip()
        new_text = prompts.read_block(prompt, prompts.SOURCE_NEW) or ""
        diff = prompts.read_block(prompt, prompts.DIFF) or ""

        options = [line.strip(" -") for line in raw_options.splitlines() if line.strip()]
        if not options:
            options = [correct or "Unchanged", "An alternative"]

        if mode == "schema":
            return json.dumps(
                {"stem": question, "options": options, "grounded_in": new_text[:80]}
            )
        if mode == "ungrounded":
            return json.dumps(
                {
                    "stem": question,
                    "options": options,
                    "correct_index": 0,
                    "grounded_in": "A sentence that appears in no document anywhere.",
                }
            )

        # Carry the figure that changed into the option that used to be right.
        added = [token for token in _NUMERIC.findall(diff) if _HAS_DIGIT.search(token)]
        removed = [token for token in _REMOVED.findall(diff) if _HAS_DIGIT.search(token)]
        updated = list(options)
        correct_index = options.index(correct) if correct in options else 0
        if added and removed:
            replacement = options[correct_index].replace(removed[0], added[0])
            if replacement == options[correct_index]:
                replacement = added[0]
            if replacement in options and options.index(replacement) != correct_index:
                correct_index = options.index(replacement)
            else:
                updated[correct_index] = replacement

        return json.dumps(
            {
                "stem": question,
                "options": updated,
                "correct_index": correct_index,
                "grounded_in": _first_sentence(new_text),
            }
        )
