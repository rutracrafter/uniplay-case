# Drift report: Employee Handbook

Comparing `fixtures/docs_v1` against `fixtures/docs_v2` using the `fake` client.

Nothing in this report has been applied. Every proposed fix needs a human.

## Summary

| Route | Items |
| --- | --- |
| RETIRE_SUGGESTED | 1 |
| NEEDS_REVIEW | 3 |
| AUTO_FIX_PROPOSED | 1 |
| OK | 8 |

| Course items | Count |
| --- | --- |
| total | 13 |
| untouched by any change | 1 |
| changed in ways that cannot matter | 6 |
| decided without a model | 1 |
| sent to the model | 5 |

Model calls: 7 (cache hits: 0). Cosmetic spans skipped: 1. Moved-but-identical spans suppressed: 2.

## Levels

| Level | Items | Stale | Needs review | Auto-fix | Retire |
| --- | --- | --- | --- | --- | --- |
| L1 | 5 | 40% | 1 | 1 | 0 |
| L2 | 4 | 0% | 0 | 0 | 0 |
| L3 | 4 | 75% | 2 | 0 | 1 |

## Structural changes

Documents:

  - unchanged: `benefits.md`
  - modified: `expenses.md`, `leave-policy.md`, `security-handbook.md`
  - added: `vendor-notes.md`
  - deleted: `onboarding.md`

Spans:

  - ADDED/NONE: 3
  - COSMETIC/STABLE: 1
  - IDENTICAL/MOVED: 4
  - IDENTICAL/REPARENTED: 1
  - IDENTICAL/STABLE: 7
  - MODIFIED/STABLE: 3
  - REMOVED/NONE: 2

Sections:

  - ADDED: 1
  - REMOVED: 1
  - RENAMED: 1
  - SAME: 8

## Findings

## RETIRE_SUGGESTED (1)

### q09 &middot; RETIRE_SUGGESTED

*Level L3*

**Why:** all 1 source span(s) removed with no relocation found

**Source changes:**
  - REMOVED / NONE in `expenses.md` &rarr; Expenses > Per-diem
    - removed text: 'The per-diem rate is forty euros per overnight stay.'

**Provenance:**
  - `expenses.md#p1.s0` &rarr; ORPHANED

## NEEDS_REVIEW (3)

### q10 &middot; NEEDS_REVIEW

*Level L3*

**Why:** 1 source span(s) require judgment; a source span moved to a different section while its text stayed the same

**Verdict:** UNAFFECTED at confidence 0.85

**Model's reasoning:** The wording changed without altering what the source states, so the question still has the same correct answer.

**Evidence quoted from the source:**
  - 'Mileage is reimbursed at the published rate.'

**Source changes:**
  - IDENTICAL / REPARENTED in `expenses.md` &rarr; Expenses > Deprecated Practices
    - was under: Expenses

**Provenance:**
  - `expenses.md#p0.s1` &rarr; MOVED

### q12 &middot; NEEDS_REVIEW

*Level L3*

**Why:** 2 source span(s) require judgment; provenance was inferred, not recorded

**Verdict:** INVALIDATED at confidence 0.90

**Model's reasoning:** A figure in the source material changed, so a question keyed to the old figure can no longer be answered correctly.

**Evidence quoted from the source:**
  - 'Notice must be given within 45 days of the intended start date.'

**Source changes:**
  - MODIFIED / STABLE in `leave-policy.md` &rarr; Leave Policy
    - diff: `Notice must be given within [-30-]{+45+} days of the intended start date.`
    - similarity: 0.98
  - IDENTICAL / STABLE in `leave-policy.md` &rarr; Leave Policy > Carry-over
  - MODIFIED / STABLE in `leave-policy.md` &rarr; Leave Policy
    - diff: `Unused days [-are-]{+may be+} forfeited at the end of the leave [-year.-]{+year, depending on local policy.+}`
    - similarity: 0.89

**Provenance:**
  - `leave-policy.md#p0.s0` &rarr; RESOLVED, inferred (BM25 13.95)
  - `leave-policy.md#p1.s0` &rarr; RESOLVED, inferred (BM25 6.00)
  - `leave-policy.md#p0.s2` &rarr; RESOLVED, inferred (BM25 5.45)

### q03 &middot; NEEDS_REVIEW

*Level L1*

**Why:** 1 source span(s) require judgment; the model could not determine the impact and abstained

**Verdict:** UNCERTAIN at confidence 0.40

**Model's reasoning:** The replacement wording is hedged, so whether the question remains answerable cannot be settled from the text alone.

**Evidence quoted from the source:**
  - 'Unused days may be forfeited at the end of the leave year, depending on local policy.'

**Source changes:**
  - MODIFIED / STABLE in `leave-policy.md` &rarr; Leave Policy
    - diff: `Unused days [-are-]{+may be+} forfeited at the end of the leave [-year.-]{+year, depending on local policy.+}`
    - similarity: 0.89

**Provenance:**
  - `leave-policy.md#p0.s2` &rarr; RESOLVED

## AUTO_FIX_PROPOSED (1)

### q01 &middot; AUTO_FIX_PROPOSED

*Level L1*

**Why:** 1 source span(s) require judgment; verdict INVALIDATED at confidence 0.90, and the proposed fix passed every validator

**Verdict:** INVALIDATED at confidence 0.90

**Model's reasoning:** A figure in the source material changed, so a question keyed to the old figure can no longer be answered correctly.

**Evidence quoted from the source:**
  - 'Notice must be given within 45 days of the intended start date.'

**Source changes:**
  - MODIFIED / STABLE in `leave-policy.md` &rarr; Leave Policy
    - diff: `Notice must be given within [-30-]{+45+} days of the intended start date.`
    - similarity: 0.98

**Provenance:**
  - `leave-policy.md#p0.s0` &rarr; RESOLVED

**Proposed fix** (not applied):

  stem: Within how many days must notice of leave be given?
  - 30 days
  * 45 days
  - 14 days
  - 90 days
  grounded in: 'Notice must be given within 45 days of the intended start date.'

## OK (8)

### q13 &middot; OK

*Level L2*

**Why:** no source span changed

**Provenance:**
  - `benefits.md#p0.s0` &rarr; RESOLVED

### q11 &middot; OK

*Level L3*

**Why:** source changed only in ways that cannot affect the question (IDENTICAL)

**Source changes:**
  - IDENTICAL / STABLE in `expenses.md` &rarr; Expenses

**Provenance:**
  - `expenses.md#p0.s0` &rarr; RESOLVED

### q02 &middot; OK

*Level L1*

**Why:** 1 source span(s) require judgment; the change does not affect this question

**Verdict:** UNAFFECTED at confidence 0.85

**Model's reasoning:** The wording changed without altering what the source states, so the question still has the same correct answer.

**Evidence quoted from the source:**
  - 'Managers must approve every leave request.'

**Source changes:**
  - MODIFIED / STABLE in `leave-policy.md` &rarr; Leave Policy
    - diff: `Managers must approve [-all-]{+every+} leave [-requests.-]{+request.+}`
    - similarity: 0.89

**Provenance:**
  - `leave-policy.md#p0.s1` &rarr; RESOLVED

### q05 &middot; OK

*Level L1*

**Why:** source changed only in ways that cannot affect the question (IDENTICAL)

**Source changes:**
  - IDENTICAL / STABLE in `leave-policy.md` &rarr; Leave Policy > Carry-over

**Provenance:**
  - `leave-policy.md#p1.s0` &rarr; RESOLVED

### q04 &middot; OK

*Level L1*

**Why:** source changed only in ways that cannot affect the question (IDENTICAL)

**Source changes:**
  - IDENTICAL / MOVED in `leave-policy.md` &rarr; Leave Policy > Parental Leave (moved from `onboarding.md`)

**Provenance:**
  - `onboarding.md#p0.s0` &rarr; MOVED

### q06 &middot; OK

*Level L2*

**Why:** source changed only in ways that cannot affect the question (IDENTICAL)

**Source changes:**
  - IDENTICAL / MOVED in `security-handbook.md` &rarr; Data Retention

**Provenance:**
  - `security-handbook.md#p2.s0` &rarr; RESOLVED

### q08 &middot; OK

*Level L2*

**Why:** source changed only in ways that cannot affect the question (IDENTICAL)

**Source changes:**
  - IDENTICAL / STABLE in `security-handbook.md` &rarr; Access Control

**Provenance:**
  - `security-handbook.md#p0.s0` &rarr; RESOLVED

### q07 &middot; OK

*Level L2*

**Why:** source changed only in ways that cannot affect the question (COSMETIC)

**Source changes:**
  - COSMETIC / STABLE in `security-handbook.md` &rarr; Physical Security
    - diff: `Badges must be [-worn visibly-]{+worn, visibly,+} at all times.`

**Provenance:**
  - `security-handbook.md#p3.s0` &rarr; RESOLVED
