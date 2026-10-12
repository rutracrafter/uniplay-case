# Coursedrift — Implementation Plan

Companion to [OVERVIEW.md](OVERVIEW.md), which holds the design rationale. This
document is the build order: nine phases, each one sequentially implementable,
each one ending in a state you can run and verify.

**Sequencing property worth knowing up front:** phases 1–4 are the entire
deterministic core and contain no model code at all. By the end of phase 4 the
pipeline can already load two snapshots, match them, anchor a course to them and
decide which items need judgment — all under unit test, with no LLM in the
picture. The model only enters in phase 6. If the schedule compresses, the
deterministic core is the part that must not be rushed.

| Phase | Deliverable | Rough size | Model code? |
|---|---|---|---|
| 0 | Scaffolding, config, one runnable command | ~120 LOC | no |
| 1 | Domain models, corpus loading, segmentation | ~350 LOC | no |
| 2 | Three-phase matcher + section diff | ~450 LOC | no |
| 3 | Anchoring + bounded BM25 inference | ~250 LOC | no |
| 4 | Impact mapping + deterministic rules | ~200 LOC | no |
| 5 | LLM wrapper: validation, repair, timeout, cache, budget | ~300 LOC | yes |
| 6 | Adjudicate, remediate, triage | ~350 LOC | yes |
| 7 | Report, CLI, fixtures, golden test | ~400 LOC | no |
| 8 | OllamaClient + `drift seed` | ~200 LOC | yes |

---

## Phase 0 — Scaffolding

**Goal:** one command runs, one command tests, and every threshold in the system
already has a home.

### Files

```text
pyproject.toml
Makefile
coursedrift/__init__.py
coursedrift/config.py
coursedrift/cli.py          # skeleton only
tests/test_config.py
```

### Dependencies

Runtime: `pydantic>=2.6`, `typer`, `rapidfuzz`, `httpx`.
Dev: `pytest`, `ruff`, `mypy`.

`rapidfuzz` earns its place in phase 2 (token-set similarity and fast scoring of
blocked candidate pairs); `httpx` is unused until phase 8 but declared now so the
lockfile settles once.

### Config — every tunable, one object

```python
class Config(BaseModel):
    # phase 2 — matching
    similarity_match: float = 0.60      # Phase B pair threshold
    cosmetic_similarity: float = 0.95   # COSMETIC shortcut floor
    shingle_n: int = 3                  # blocking n-gram width
    jaccard_match: float = 0.40         # section identity
    jaccard_part: float = 0.25          # split/merge component floor
    # phase 3 — anchoring
    anchor_fuzzy: float = 0.85          # relocation by snapshot_text
    infer_top_k: int = 3                # BM25 candidates kept
    # phase 6 — judgment
    confidence_floor: float = 0.75
    votes: int = 1
    # phase 5 — reliability
    timeout_s: float = 30.0
    max_attempts: int = 2               # initial + one repair
    max_calls: int = 500                # whole-run budget
    cache_dir: Path = Path(".drift_cache")
```

Nothing else in the codebase may define a magic number that affects a verdict.
Every later phase reads from here, and `--config path.json` overrides.

### Done when

- `uv run drift --help` prints the command list
- `uv run pytest` passes
- `make check` runs `ruff check`, `ruff format --check` and `mypy coursedrift`
  clean

---

## Phase 1 — Domain models, corpus, segmentation

**Goal:** a directory of Markdown becomes a `Snapshot` of hashed, addressable,
section-aware spans.

### Files

```text
coursedrift/models.py
coursedrift/corpus.py
coursedrift/segment.py
tests/test_segment.py
tests/test_corpus.py
```

### Core types — `models.py`

```python
class Span(BaseModel):
    span_id: str                   # "handbook.md#p3.s2"
    doc_id: str
    section_path: tuple[str, ...]  # ("Leave Policy", "Parental Leave")
    para_index: int
    sent_index: int
    order: int                     # document-wide sentence ordinal
    text: str                      # as authored
    norm_text: str                 # normalization contract below
    hash: str                      # sha256(norm_text)[:16]

class Section(BaseModel):
    section_id: str
    doc_id: str
    path: tuple[str, ...]
    heading: str
    level: int
    span_hashes: frozenset[str]    # children — the Jaccard input
    order: int

class Document(BaseModel):
    doc_id: str
    content: str
    hash: str
    spans: list[Span]
    sections: list[Section]

class Snapshot(BaseModel):
    root: Path
    docs: dict[str, Document]

    def all_spans(self) -> Iterator[Span]: ...
    def by_hash(self) -> dict[str, list[Span]]: ...   # corpus-wide, phase 2
```

Course side:

```python
class ProvenanceRef(BaseModel):
    doc_id: str
    span_id: str
    snapshot_text: str
    snapshot_hash: str
    inferred: bool = False
    infer_score: float | None = None

class CourseItem(BaseModel):
    item_id: str
    level_id: str
    stem: str
    options: list[str]
    correct_index: int
    provenance: list[ProvenanceRef] = []

class Course(BaseModel):
    course_id: str
    title: str
    items: list[CourseItem]
```

### The normalization contract — decide this once, here

`normalize()` applies **exactly** these, in order:

1. Unicode NFKC
2. Curly quotes, en/em dashes and non-breaking spaces folded to ASCII equivalents
3. Whitespace runs collapsed to one space; leading/trailing stripped

It deliberately does **not** lowercase and does **not** strip punctuation. That
split is what keeps two mechanisms from collapsing into one:

- whitespace and unicode-variant edits hash **identical**, so Phase A absorbs
  them for free and they never become findings
- casing and punctuation edits do **not** hash identical, so they fall through to
  Phase B and trip the `COSMETIC` shortcut

Fold punctuation into the normalizer instead and `COSMETIC` becomes dead code;
leave whitespace out of it and trivial reflows flood the report. Both behaviours
are asserted in `tests/test_segment.py`.

### Segmentation rules — `segment.py`

```python
def normalize(text: str) -> str: ...
def segment_document(doc_id: str, content: str) -> tuple[list[Span], list[Section]]: ...
```

- ATX headings (`#`..`######`) open sections and never become spans; nesting
  builds `section_path`
- Paragraphs split on blank lines; sentences split on `[.!?]` followed by
  whitespace and a capital, with a small abbreviation guard (`e.g.`, `i.e.`,
  `No.`, `Dr.`, `vs.`)
- A fenced code block is one span, verbatim, never sentence-split
- Each list item is its own span
- Tables are one span per row

Imperfect sentence splitting is acceptable and bounded: a mis-split is stable
across both snapshots, so it hashes identically on both sides and costs nothing.
That is worth stating in a comment — it is the reason this does not need a
sentence-boundary model.

### Corpus — `corpus.py`

```python
def load_snapshot(root: Path) -> Snapshot: ...        # *.md recursive, doc_id = relpath
def diff_manifests(old: Snapshot, new: Snapshot) -> ManifestDiff: ...
```

`ManifestDiff` carries four `set[str]`: `unchanged`, `modified`, `added`,
`deleted`. Documents in `unchanged` are excluded from phase 2 entirely.

### Tests

- `normalize` is idempotent; `normalize(normalize(x)) == normalize(x)`
- whitespace-only and curly-quote-only edits produce **equal hashes**
- casing-only and comma-only edits produce **unequal hashes** (the contract's
  other half)
- nested headings produce the expected `section_path` tuples
- a code fence survives as one span with newlines intact
- `span_id` is stable across two loads of identical content
- `diff_manifests` classifies all four states on a fixture pair

### Done when

`uv run python -c "from coursedrift.corpus import load_snapshot; print(len(list(load_snapshot(Path('fixtures/docs_v1')).all_spans())))"`
prints a sane count, and the test module passes.

---

## Phase 2 — The matcher and the section diff

**The riskiest phase.** Write the tests first; this is where the core invariant
either holds or silently does not.

### Files

```text
coursedrift/diff.py
coursedrift/lis.py
tests/test_lis.py
tests/test_diff.py
tests/test_diff_reorder.py      # the invariant's regression test
tests/test_sections.py
```

### Types

```python
class ContentOp(StrEnum):
    IDENTICAL = "IDENTICAL"; COSMETIC = "COSMETIC"; MODIFIED = "MODIFIED"
    REMOVED = "REMOVED"; ADDED = "ADDED"

class PositionOp(StrEnum):
    STABLE = "STABLE"; MOVED = "MOVED"; REPARENTED = "REPARENTED"; NONE = "NONE"

class SpanChange(BaseModel):
    content: ContentOp
    position: PositionOp
    old: Span | None
    new: Span | None
    similarity: float | None = None
    inline_diff: str | None = None
    across_docs: bool = False
```

### Entry point and phase functions

```python
def match_corpus(old: Snapshot, new: Snapshot, cfg: Config) -> MatchResult: ...

def _phase_a_identity(old, new) -> tuple[list[Pair], Residue]: ...
def _phase_b_similarity(residue, cfg) -> list[Pair]: ...
def _classify_positions(pairs, cfg) -> None: ...          # STABLE / MOVED
def match_sections(old, new, cfg) -> SectionMap: ...
def _apply_reparenting(pairs, section_map) -> None: ...   # upgrades to REPARENTED
```

**Phase A — identity.** Build `dict[hash, list[Span]]` over the *whole* corpus,
old and new. For each hash present on both sides, zip the two occurrence lists in
`order` and emit pairs; surplus on either side returns to the residue. Corpus-wide
is the point: a pair whose `old.doc_id != new.doc_id` sets `across_docs=True`.

**Phase B — similarity.** Build a shingle index `frozenset[3-gram] -> [spans]`
over the residue's new spans. For each residual old span, gather candidates
sharing at least one shingle, score with `rapidfuzz.fuzz.token_set_ratio / 100`,
and collect `(score, old, new)` triples. Sort descending and accept greedily
one-to-one above `cfg.similarity_match`. Then, for each accepted pair, if
`score >= cfg.cosmetic_similarity` **and** the word-level diff touches only
punctuation or casing, the content op is `COSMETIC`, else `MODIFIED`. Unmatched
old spans are `REMOVED`; unmatched new spans are `ADDED`.

**Phase C — position.** Per document, over pairs where `across_docs is False`,
sort by `old.order` and take the LIS of `new.order`. Indices inside it are
`STABLE`; outside, `MOVED`. Pairs with `across_docs is True` are `MOVED`
unconditionally and never enter the LIS computation — a cross-document move has
no meaningful position to be stable within.

```python
def lis_indices(seq: Sequence[int]) -> set[int]: ...   # patience/bisect, O(n log n)
```

**Sections.** Match by exact heading text first, then by Jaccard over
`span_hashes`. One old section whose overlap is above `cfg.jaccard_match` with
exactly one new section is `RENAMED` (or identical); above `cfg.jaccard_part`
with two or more is a `SPLIT`; two old above `jaccard_part` against one new is a
`MERGE`; no match is `REMOVED`.

**Position is finalized in two passes, on purpose.** Phase C decides
`STABLE`/`MOVED` with no section knowledge. `REPARENTED` can only be assigned
after the section map exists, because a span sitting under a *renamed* section
has not been reparented — the heading changed around it. So `_apply_reparenting`
runs last and upgrades a pair to `REPARENTED` when its old and new `section_path`
differ **after** resolving renames through the section map.

### Tests

- `lis_indices` on swap `[1,0]`, rotation `[1,2,0,3]`, interleave `[0,2,1,3]`,
  already-sorted, and reversed
- **`test_diff_reorder.py`: shuffling a document's sections with zero text edits
  yields no `MODIFIED`, no `REMOVED`, no `ADDED`** — only `IDENTICAL` spans, some
  `MOVED`. This is the invariant's regression test and the single most important
  assertion in the suite
- a reworded sentence yields one `MODIFIED` pair, not `REMOVED` + `ADDED`
- a comma-only edit yields `COSMETIC`, and a case-only edit yields `COSMETIC`
- a whitespace-only edit never reaches Phase B at all (asserted by spying on the
  residue, so the free path stays free)
- content moved from `a.md` into `b.md` matches with `across_docs=True` and
  position `MOVED`
- boilerplate appearing 5× old and 3× new produces 3 `IDENTICAL` pairs and 2
  `REMOVED`
- section split and merge detected via Jaccard; a renamed section containing
  unmoved spans produces **no** `REPARENTED`
- a span relocated from one section to another produces `REPARENTED` with
  `content == IDENTICAL`

### Done when

The reorder test passes and `make check` is clean. Do not start phase 3 before
the reorder test is green.

---

## Phase 3 — Anchoring

**Goal:** a course item's recorded provenance finds its span in the new corpus,
or honestly admits it could not.

### Files

```text
coursedrift/anchor.py
tests/test_anchor.py
tests/test_infer.py
```

### Types and functions

```python
class AnchorStatus(StrEnum):
    RESOLVED = "RESOLVED"; MOVED = "MOVED"
    ORPHANED = "ORPHANED"; AMBIGUOUS = "AMBIGUOUS"

class ResolvedAnchor(BaseModel):
    ref: ProvenanceRef
    status: AnchorStatus
    new_span: Span | None = None
    candidates: list[Span] = []      # populated when AMBIGUOUS
    similarity: float | None = None

def resolve_anchor(ref, new: Snapshot, cfg) -> ResolvedAnchor: ...
def resolve_course(course, new: Snapshot, cfg) -> dict[str, list[ResolvedAnchor]]: ...
def infer_provenance(item, old: Snapshot, cfg) -> list[ProvenanceRef]: ...
```

Resolution order, corpus-wide:

1. `snapshot_hash` hit → `RESOLVED`, or `MOVED` if `doc_id` or `section_path`
   changed
2. no hash hit → fuzzy `snapshot_text` against all new spans above
   `cfg.anchor_fuzzy`
3. several above threshold → prefer the one whose `section_path` matches; still
   tied → `AMBIGUOUS` with all candidates attached
4. nothing above threshold → `ORPHANED`

### Bounded inference

A compact BM25 over old-corpus spans (`k1=1.5`, `b=0.75`), query = stem plus the
correct option, top-`cfg.infer_top_k` kept with scores, section-path agreement as
tiebreak. Every ref it returns carries `inferred=True`. No embeddings, no learned
weighting, no tuning — roughly 40 lines, and that is the whole ambition.

The hard rule lives in phase 6's triage, not here, but it is asserted in this
phase's tests so the constraint is pinned to the component it constrains:
**an inferred ref can never reach `AUTO_FIX_PROPOSED`.**

### Tests

- hash hit after an unrelated edit elsewhere in the doc → `RESOLVED`
- span moved to a different section → `MOVED`
- span lightly reworded → fuzzy `RESOLVED` with similarity recorded
- span deleted → `ORPHANED`
- snapshot text appearing twice in the new corpus, different sections → section
  tiebreak picks the right one
- same, but both candidates in the same section path → `AMBIGUOUS`, both
  candidates attached
- `infer_provenance` ranks the span containing the keyed answer first on the
  fixture corpus
- every inferred ref has `inferred is True` (guards the phase-6 rule)

---

## Phase 4 — Impact mapping

**Goal:** decide, deterministically, which items need a model and which do not.
This is the cost control and the honesty of the report's "skipped" count.

### Files

```text
coursedrift/impact.py
tests/test_impact.py
```

```python
class ImpactCandidate(BaseModel):
    item: CourseItem
    anchors: list[ResolvedAnchor]
    changes: list[SpanChange]
    needs_model: bool
    preset_route: Route | None = None    # set when no judgment is needed
    reasons: list[str] = []              # every reason, for the report

class SkipStats(BaseModel):
    total_items: int
    untouched: int
    cosmetic_only: int
    moved_only: int
    sent_to_model: int

def build_candidates(course, matches, anchors, section_map, cfg)
        -> tuple[list[ImpactCandidate], SkipStats]: ...
```

The two-axis table from the overview becomes one pure function, tested as a
truth table:

```python
def requires_adjudication(change: SpanChange) -> bool: ...
```

| content | position | result |
|---|---|---|
| `IDENTICAL` | `STABLE` | `False` — not a change |
| `IDENTICAL` | `MOVED` | `False` — structural only |
| `IDENTICAL` | `REPARENTED` | `True` |
| `COSMETIC` | any | `False` |
| `MODIFIED` | any | `True` |
| `REMOVED` | — | `False` — deterministic retire |

Deterministic presets, set here and never overridden later:

- every anchor `ORPHANED` with no relocation → `preset_route = RETIRE_SUGGESTED`
- any anchor `AMBIGUOUS` → `preset_route = NEEDS_REVIEW`
- no change requires adjudication → `preset_route = OK`

### Tests

- `requires_adjudication` over all six rows of the table
- an item whose spans are all untouched never appears in the model queue
- an item with one `MODIFIED` and three `IDENTICAL` spans is queued once, with
  only the modified change attached
- `SkipStats` sums to `total_items`
- an item with an orphaned anchor is preset to `RETIRE_SUGGESTED` without any
  model involvement

### Done when

The deterministic core is complete. `uv run drift run --client none` (a flag that
stops after phase 4) prints the skip statistics for the fixture pair. Everything
up to here is covered by tests and no LLM code exists yet.

---

## Phase 5 — The model boundary

**Goal:** exactly one function in the codebase turns model output into trusted
data, and it is hostile by default.

### Files

```text
coursedrift/llm/__init__.py
coursedrift/llm/base.py        # protocol, extraction, validation, repair, timeout
coursedrift/llm/cache.py
coursedrift/llm/fake.py
tests/test_llm_validation.py
tests/test_llm_failures.py
tests/test_cache.py
```

### The protocol and the wrapper

```python
class LLMClient(Protocol):
    name: str
    def complete(self, prompt: str, *, timeout: float) -> str: ...

class FailureKind(StrEnum):
    TIMEOUT = "TIMEOUT"; TRANSPORT = "TRANSPORT"; MALFORMED = "MALFORMED"
    SCHEMA = "SCHEMA"; REJECTED = "REJECTED"; BUDGET = "BUDGET"

class CallOutcome(BaseModel, Generic[T]):
    ok: bool
    value: T | None = None
    failure: FailureKind | None = None
    detail: str | None = None
    attempts: int = 0
    raw: list[str] = []          # every raw response, for the report

Validator = Callable[[T], str | None]       # returns an error message or None

def call_validated(
    client: LLMClient, prompt: str, schema: type[T], *,
    validators: Sequence[Validator[T]] = (),
    cfg: Config, cache: Cache, budget: CallBudget,
) -> CallOutcome[T]: ...
```

Sequence inside `call_validated`:

1. budget check → `BUDGET` failure if exhausted
2. cache lookup on `sha256(client.name + prompt)` → return the hit
3. `client.complete(..., timeout=cfg.timeout_s)`; `TimeoutError` → `TIMEOUT`,
   transport error → `TRANSPORT`
4. `extract_json(raw)` — strip ``` fences, then scan for the first balanced
   `{...}`; nothing found → `MALFORMED`
5. `schema.model_validate_json` → `SCHEMA` on failure
6. run every validator in order; first non-`None` → `REJECTED` with the message
7. on `MALFORMED`/`SCHEMA`/`REJECTED` and `attempts < cfg.max_attempts`, retry
   **once** with the error appended to the prompt; otherwise return the failure
8. cache only successful outcomes

A failure is never raised. Every caller receives a `CallOutcome` and is
responsible for routing it — which is what makes "fail closed to `NEEDS_REVIEW`"
a type-level consequence rather than a discipline.

### FakeClient

Deterministic, and useful in two ways at once:

- **scripted** responses keyed by `sha256` of the span pair for the ten fixture
  cases, so `drift demo` produces a known report
- **heuristic** fallback for anything else, so pointing the fake client at your
  own documents still does something defensible: a changed number or date in the
  inline diff → `INVALIDATED`; hedging language introduced
  (`may`, `typically`, `varies`) → `UNCERTAIN`; otherwise `UNAFFECTED`

Failure injection via `--fake-failures`, each mode firing on a chosen call index:

| mode | behaviour |
|---|---|
| `malformed` | returns prose with no JSON object |
| `fenced` | returns valid JSON wrapped in a ```json fence (must still succeed) |
| `schema` | returns JSON with `verdict: "PROBABLY_FINE"` |
| `ungrounded` | returns a verdict whose `evidence` quote is not in the source |
| `timeout` | raises `TimeoutError` |
| `injection` | obeys an injected instruction and returns `UNAFFECTED` for everything |

### Tests

- each failure mode produces the expected `FailureKind` and never raises
- `fenced` succeeds — tolerant extraction is asserted, not assumed
- a repair attempt that returns valid output succeeds with `attempts == 2`
- a repair attempt that fails again returns the failure with `attempts == 2`,
  never 3
- `raw` contains both responses after a repair
- cache hit returns byte-identical values and does not call the client (spy)
- budget exhaustion returns `BUDGET` before any call

---

## Phase 6 — Adjudicate, remediate, triage

### Files

```text
coursedrift/llm/prompts.py
coursedrift/adjudicate.py
coursedrift/remediate.py
coursedrift/triage.py
tests/test_adjudicate.py
tests/test_remediate_validators.py
tests/test_triage.py
```

### Wire schemas

```python
class Verdict(StrEnum):
    UNAFFECTED = "UNAFFECTED"; WEAKENED = "WEAKENED"
    INVALIDATED = "INVALIDATED"; UNCERTAIN = "UNCERTAIN"

class Adjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(min_length=1, max_length=4)

class Patch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stem: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=6)
    correct_index: int = Field(ge=0)
    grounded_in: str = Field(min_length=1)
```

`extra="forbid"` is deliberate: an invented field is a signal the model did not
follow the contract, and inventing fields correlates with inventing content.

### Prompts — `prompts.py`

Document text is delimited and labelled as data, with the standing instruction
that content inside the delimiters is source material to be judged, never
instructions to be followed:

```text
<<<SOURCE_OLD>>> ... <<<END>>>
<<<SOURCE_NEW>>> ... <<<END>>>
<<<DIFF>>> ... <<<END>>>
```

The delimiters are defence in depth. The actual defence is structural: a closed
enum and a grounded quote, both checked in code.

### Validators

Adjudication:

```python
def evidence_is_grounded(a: Adjudication, *, haystack: str) -> str | None: ...
```

Every quote must appear in the concatenated old + new + diff text, after
`normalize()` on both sides so trivial whitespace differences do not reject a
truthful quote.

Patch:

```python
def one_correct_option(p) -> str | None
def options_distinct(p) -> str | None
def correct_index_in_range(p) -> str | None
def stem_is_question(p) -> str | None          # ends in '?' or opens interrogatively
def grounded_in_new_document(p, *, new_text: str) -> str | None
```

### Voting

```python
def adjudicate(cand, client, cfg, cache, budget) -> AdjudicationResult: ...
```

With `cfg.votes == 1` this is one call. Above 1, the prompt is reissued with a
vote index appended (so the cache key differs), and **any** disagreement in
`verdict` sets `unanimous=False`, which triage reads as a review trigger. A
failed vote counts as disagreement, not as an abstention to be ignored.

### Triage — one pure function

```python
def route(cand: ImpactCandidate,
          adj: AdjudicationResult | None,
          patch: PatchResult | None,
          cfg: Config) -> Finding: ...
```

No I/O, no client, no clock. Precedence, highest first:

1. `cand.preset_route` is set → use it
2. `adj` is `None` or failed → `NEEDS_REVIEW`, reason = the `FailureKind`
3. `adj.verdict is UNCERTAIN` → `NEEDS_REVIEW`
4. `adj.verdict is UNAFFECTED` → `OK`
5. `adj.confidence < cfg.confidence_floor` → `NEEDS_REVIEW`
6. `not adj.unanimous` → `NEEDS_REVIEW`
7. any anchor `inferred` → `NEEDS_REVIEW`, patch attached as a suggestion
8. any change `REPARENTED` → `NEEDS_REVIEW`, low severity
9. `patch` is `None` or failed validation → `NEEDS_REVIEW`, rejected attempt
   attached
10. otherwise → `AUTO_FIX_PROPOSED`

Every `Finding` carries `reasons: list[str]` naming which rule fired, so the
report can explain itself.

### Tests

- `test_triage.py` walks the precedence list as a truth table, one case per rule,
  including two cases where a higher rule must win over a lower one
- each patch validator rejects its own failure case and passes a clean patch
- an ungrounded evidence quote is rejected; the same quote with different
  whitespace is accepted
- the `injection` fake mode yields `UNAFFECTED` from the model and the run still
  produces a **valid, schema-conformant** finding — the structural defence is
  asserted, not described
- a timeout on adjudication routes to `NEEDS_REVIEW` with `reason` containing
  `TIMEOUT`
- `votes=3` with one dissenting scripted response routes to `NEEDS_REVIEW`
- an inferred-provenance item with a perfect high-confidence patch still routes
  to `NEEDS_REVIEW`

---

## Phase 7 — Report, CLI, fixtures, golden test

**Goal:** `uv run drift demo` tells a complete story with no flags and no network.

### Files

```text
coursedrift/report.py
coursedrift/cli.py                   # filled in
fixtures/docs_v1/*.md
fixtures/docs_v2/*.md
fixtures/course.json
tests/test_report.py
tests/golden/demo_report.md
tests/test_demo_golden.py
```

### Fixture corpus

Four documents, chosen so one reorganization exercises several cases at once:

| file | v1 → v2 | cases |
|---|---|---|
| `leave-policy.md` | notice period 30 → 45 days; one paragraph reworded; one hedged rewrite | 01, 02, 05 |
| `security-handbook.md` | sections shuffled, no text edits; one comma added | 06, 08 |
| `expenses.md` | "Per-diem" section deleted; one sentence reparented into "Deprecated Practices" | 03, 07 |
| `onboarding.md` | **deleted in v2**, two paragraphs folded into `leave-policy.md` | 04 |
| `vendor-notes.md` | added in v2, contains an injection payload | 09 |

`course.json` holds twelve items across three levels, eleven with recorded
provenance and one with none at all (case 10).

### Report

```python
def render_json(run: RunReport) -> str: ...
def render_markdown(run: RunReport) -> str: ...
```

The Markdown report opens with the summary a reviewer needs before any detail:
counts by route, items skipped and why, model calls made and cache hits,
structural changes from the section map. Then findings, ordered by severity, each
one naming the document and span, the inline diff, the verdict with rationale and
evidence, the proposed patch, and the reasons that produced its route.

Determinism matters here because of the golden test: findings sort by
`(route_severity, doc_id, span order, item_id)` and the report carries no
timestamp in its body. The run timestamp goes in the JSON only, under a key the
golden comparison ignores.

### CLI

```text
drift run    --old DIR --new DIR --course FILE
             [--client fake|ollama|none] [--model NAME] [--votes N]
             [--config FILE] [--out DIR] [--fake-failures LIST]
drift demo   [--client fake|ollama]
drift seed   --docs DIR --out FILE          # phase 8
```

`demo` is `run` with the fixture paths pre-filled and `--out` to a temp
directory, printing the Markdown report to stdout.

### Tests

- `test_demo_golden.py` runs `demo` end to end with the fake client and compares
  stdout to `tests/golden/demo_report.md` byte for byte
- every one of the ten fixture cases appears in the golden report with its
  expected route — the fixture table above becomes a parametrized test
- running `demo` twice produces identical output
- `render_json` output round-trips through `RunReport.model_validate_json`

### Done when

`make demo` prints a report covering all ten cases, and `make test` is green with
no network access.

---

## Phase 8 — OllamaClient and `drift seed`

**Goal:** the same pipeline, unchanged, driven by a real small model.

### Files

```text
coursedrift/llm/ollama.py
coursedrift/seed.py
tests/test_ollama_client.py      # transport-level, no live daemon
```

```python
class OllamaClient:
    name: str                     # f"ollama:{model}"
    def __init__(self, model: str, host: str = "http://localhost:11434"): ...
    def preflight(self) -> None: ...                      # GET /api/tags
    def complete(self, prompt: str, *, timeout: float) -> str: ...
```

`POST /api/generate` with `stream=False`, `format="json"`, and
`options={"temperature": 0, "seed": 0}`. `preflight` runs once at startup and
raises an actionable message when the daemon is unreachable or the model is not
pulled — naming the exact `ollama pull <model>` to run, not a stack trace.

Nothing else changes. If phase 5 did its job, swapping the client requires no
edit to adjudicate, remediate, triage or report, and the only new failure surface
is `TRANSPORT`.

### `drift seed`

Generates a course from a document directory with provenance recorded as it goes:
pick spans, ask for a question grounded in each one, validate with the phase-6
patch validators, and write `course.json` with real `ProvenanceRef`s. This closes
the loop — it makes the whole system demonstrable on the user's own documents
rather than only on fixtures — and it is last because nothing else depends on it.

### Tests

- a mocked 200 response parses; a mocked connection error becomes `TRANSPORT`
- a mocked slow response becomes `TIMEOUT`
- `preflight` against a mocked `/api/tags` missing the model raises a message
  containing `ollama pull`

### Done when

`uv run drift run --client ollama --model qwen2.5:3b` completes against the
fixtures and produces a report with the same *shape* as the golden one. Verdicts
will differ from the scripted fake — that is expected and is not a test failure.

---

## Definition of done, whole project

- `make demo` — one command, no network, no model, covers all ten cases
- `make test` — green, no network
- `make check` — ruff and mypy clean
- `README.md` — the two commands, the ollama path, and what a production version
  would add (PDF extraction, the coverage-gap signal, a triage UI)
- No magic number outside `config.py` affects any verdict
- No model output reaches a report without passing `call_validated`
