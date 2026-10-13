# Coursedrift

Detects when a generated course has gone stale because its source documents
changed, proposes grounded fixes for what it is sure about, and routes
everything else to a human with the evidence attached.

**It never mutates the course.** Every output is a proposal.

```bash
uv sync        # once
make demo      # seeded end-to-end run: no network, no model, no flags
make test      # 378 tests, offline
make check     # ruff + mypy
```

`make demo` prints a drift report over the bundled fixture corpus, which is
built to exercise every branch of the pipeline — a changed figure, a harmless
reword, a deleted section, a reordered document, a sentence moved between
sections, content folded from a deleted file into another, a hedged rewrite
the model declines to judge, and a question with no recorded provenance.

## Running it on your own documents

```bash
# 1. write a course from one snapshot, recording where each question came from
uv run drift seed --docs path/to/docs_v1 --out course.json

# 2. change the documents, then compare the two snapshots
uv run drift run --old path/to/docs_v1 --new path/to/docs_v2 \
                 --course course.json --out ./report
```

With a local model instead of the deterministic fake client:

```bash
ollama serve
ollama pull qwen2.5:3b
uv run drift run --old docs_v1 --new docs_v2 --course course.json \
                 --client ollama --model qwen2.5:3b
```

Useful flags: `--client none` stops after the deterministic stages and prints
what they decided; `--votes 3` runs the adjudicator three times and escalates
any disagreement; `--fake-failures timeout,malformed,schema,ungrounded`
injects model failures to show the recovery paths; `--fail-on-drift` exits
non-zero for CI; `--timeout 60` widens the per-call budget.

**Thinking models.** A model that advertises a `thinking` capability spends
most of its generation on a reasoning trace returned in a separate field,
which this client discards. On qwen3.5:9b one adjudication prompt took 1432
tokens and 36.6s — against the 30s default, so every call failed and every
item fell through to review. Preflight now detects the capability and sends
`think: false`, which brings the same prompt to 78 tokens and 1.7s for an
identical verdict. Preflight also loads the model before the run starts, so a
large model's load time is not charged to the first item's timeout, and
generation is capped at `max_output_tokens` so a model that starts repeating
itself cannot run to the context limit.

## How it works

Ten stages. Eight are deterministic; a model is involved in exactly two.

| | Stages |
| --- | --- |
| Deterministic | corpus, segment, match, sections, anchor, impact |
| Model, validated | adjudicate, remediate |
| Decision, pure rules | triage, report |

The core invariant is that **no stage uses positional order to establish
identity**. Spans are matched by content hash corpus-wide, then the residue by
similarity; position is consulted only at the end, to tell a move from a
reorder. Without that, shuffling a handbook's sections reports every span as
deleted and proposes retiring content that is perfectly correct.

Four routes come out: `OK`, `AUTO_FIX_PROPOSED`, `NEEDS_REVIEW`,
`RETIRE_SUGGESTED`. Everything the system could not decide lands in review —
a failed call, a low-confidence verdict, disagreeing votes, an ambiguous
anchor, inferred provenance, a patch that failed validation.

[OVERVIEW.md](OVERVIEW.md) has the design rationale and
[PLAN.md](PLAN.md) the phase-by-phase build.

## Testing

```bash
make test        # 378 tests, offline and deterministic
make test-live   # opt-in: talks to a real Ollama daemon
make check       # ruff + mypy
```

| kind | what it covers |
| --- | --- |
| Unit | The LIS primitive, the normalization contract, every patch validator, and triage's precedence as a truth table. |
| Property-based | The core invariant over generated documents and arbitrary permutations, plus span conservation — every span accounted for exactly once — and the LIS cross-checked against a reference implementation. |
| Integration | Real files on disk, exercising corpus → segment → diff → anchor → impact together. |
| Fault injection | Every recovery path driven through the fake client's seven failure modes. |
| Contract | The Ollama HTTP shape and failure mapping, through a mock transport. |
| End-to-end | The full pipeline against byte-exact golden reports, both markdown and JSON; the CLI; and the seed → run loop. |
| Live, opt-in | A real model, including the regression for the thinking-model timeout. Excluded by default so the suite stays offline. |

Both golden reports are refreshed deliberately, never automatically:

```bash
uv run drift demo        > tests/golden/demo_report.md
uv run drift demo --json > tests/golden/demo_report.json
```

The JSON one matters because `drift_report.json` is the machine interface:
without it, a change to the report's shape passes the suite while silently
breaking anything consuming it.

## Treating the model as a component

- Every response parses into a Pydantic model with closed enums and
  `extra="forbid"`. Extraction is tolerant of code fences and commentary;
  validation is not tolerant of anything.
- Evidence quotes and a patch's `grounded_in` are checked against the exact
  text supplied. A patch is grounded against the *new* document only.
- One repair attempt naming the error, then fail closed to `NEEDS_REVIEW`.
  Timeouts are not retried.
- `call_validated` never raises; callers receive an outcome they must route.
- Responses are cached by `sha256(client + prompt)`, and a cache hit replays
  through the same validation path, so a stored response that no longer
  validates is dropped rather than returned.
- Uploaded documents are untrusted. They travel inside labelled blocks and
  the model is told to judge them, not obey them — but the real defence is
  structural: a verdict must be one of four enum values and a fix must quote
  the real document, so even a successful injection cannot produce output
  that passes validation. One fixture carries a payload to prove it.

## What a production version would add

- **Real ingestion.** PDF, DOCX and YouTube transcripts behind the existing
  loader interface. Fixtures here are Markdown so extraction noise does not
  muddy the diff.
- **A coverage-gap signal.** Source material with no questions is a different
  problem: it needs a notion of what is teachable, which belongs to the
  course generator rather than the drift detector.
- **A triage UI.** The Markdown report is the human interface today; the JSON
  is shaped for a review queue with approve, reject and escalate.
- **Embeddings for anchoring.** Hash plus fuzzy matching is enough here and is
  deterministic, which a similarity index is not.
- **A performance guard on Phase B.** Nothing currently asserts that shingle
  blocking keeps candidate scoring near-linear. See the deferred section of
  [PLAN.md](PLAN.md) for why it waits for a larger corpus.
