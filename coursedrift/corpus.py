"""Loading snapshots and classifying documents between them."""

from __future__ import annotations

import hashlib
import json
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


def load_document(root: Path, path: Path) -> Document:
    content = path.read_text(encoding="utf-8")
    doc_id = path.relative_to(root).as_posix()
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
    """Load a course and validate it against the domain model."""
    return Course.model_validate(json.loads(path.read_text(encoding="utf-8")))
