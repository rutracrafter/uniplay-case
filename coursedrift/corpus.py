"""Loading snapshots and classifying documents between them."""

from __future__ import annotations

import hashlib
from pathlib import Path

from coursedrift.models import Course, Document, ManifestDiff, Snapshot
from coursedrift.segment import segment_document


def _doc_hash(content: str) -> str:
    """Identity of the file as authored.

    Raw content, not normalized: a whitespace-only reflow is honestly a
    modified file. It still produces no findings, because every one of its
    spans hashes identically in Phase A — it just costs one pass through the
    matcher to prove it.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class UnreadableDocument(RuntimeError):
    """A source file could not be decoded as UTF-8."""


class SuspectCorpus(RuntimeError):
    """The snapshot pair looks like a mistake rather than an edit."""


def load_document(root: Path, path: Path) -> Document:
    doc_id = path.relative_to(root).as_posix()
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        # Decoding with errors="replace" would be worse than failing: a
        # mangled character changes the span hash, so the document would
        # report drift that no author introduced. Guessing an encoding is
        # worse still. Name the file and stop.
        raise UnreadableDocument(
            f"{doc_id} is not valid UTF-8 (byte {error.object[error.start]:#04x} "
            f"at position {error.start}).\n"
            "Re-save it as UTF-8, or remove it from the snapshot."
        ) from error
    spans, sections = segment_document(doc_id, content)
    return Document(
        doc_id=doc_id,
        content=content,
        hash=_doc_hash(content),
        spans=spans,
        sections=sections,
    )


def load_snapshot(root: Path) -> Snapshot:
    """Load every Markdown file under `root`, recursively.

    `doc_id` is the POSIX path relative to the root, so the same logical
    document keeps its id across snapshots stored in sibling directories.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"snapshot root is not a directory: {root}")
    docs: dict[str, Document] = {}
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        doc = load_document(root, path)
        docs[doc.doc_id] = doc
    return Snapshot(root=root, docs=docs)


def check_snapshot_pair(old: Snapshot, new: Snapshot, *, allow_empty_new: bool) -> None:
    """Refuse a snapshot pair that is almost certainly a mistake.

    If the new snapshot holds no documents while the old one does, every
    course item orphans and the system proposes retiring the entire course —
    confidently, with no model calls and nothing to suggest anything went
    wrong. A typo'd path or a bad checkout produces the most destructive
    recommendation this system can make, so it is refused rather than
    reported.

    Genuinely emptying a corpus is possible, so there is an escape hatch; it
    just has to be asked for.
    """
    if new.docs or not old.docs or allow_empty_new:
        return
    raise SuspectCorpus(
        f"The new snapshot at {new.root} contains no Markdown documents, "
        f"while the old one at {old.root} has {len(old.docs)}.\n"
        "Every course item would orphan and the whole course would be "
        "proposed for retirement.\n"
        "Check the --new path. If the corpus really was emptied, pass "
        "--allow-empty-new."
    )


def diff_manifests(old: Snapshot, new: Snapshot) -> ManifestDiff:
    """Classify every document as unchanged, modified, added or deleted."""
    old_ids = set(old.docs)
    new_ids = set(new.docs)
    shared = old_ids & new_ids
    return ManifestDiff(
        unchanged={d for d in shared if old.docs[d].hash == new.docs[d].hash},
        modified={d for d in shared if old.docs[d].hash != new.docs[d].hash},
        added=new_ids - old_ids,
        deleted=old_ids - new_ids,
    )


def load_course(path: Path) -> Course:
    """Load a course, validating shape and internal consistency.

    The trust boundary for course input, which may come from `drift seed`,
    from a real course generator, or from hand editing. Every stage after this
    may assume an item's `correct_index` addresses an option that exists —
    `CourseItem` enforces that, so a bad file fails here with the offending
    item named rather than raising IndexError mid-run.
    """
    return Course.model_validate_json(path.read_text(encoding="utf-8"))
