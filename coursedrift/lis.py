"""Longest increasing subsequence, used to tell movement from reordering.

Phase C of the matcher needs to know which matched spans form the stable core
of a document and which genuinely moved. The LIS of the new positions, taken
in old-position order, is that core: everything outside it is a move.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence


def lis_indices(seq: Sequence[int]) -> set[int]:
    """Indices of one longest strictly increasing subsequence of `seq`.

    O(n log n) patience algorithm with predecessor reconstruction. When
    several longest subsequences exist any one of them is a correct stable
    core, so the choice among them is arbitrary but deterministic.

    >>> sorted(lis_indices([1, 2, 0, 3]))
    [0, 1, 3]
    """
    n = len(seq)
    if n == 0:
        return set()

    tail_index: list[int] = []  # tail_index[k]: index of the smallest tail of length k+1
    tail_value: list[int] = []  # parallel values, kept sorted for bisect
    prev: list[int] = [-1] * n

    for i, value in enumerate(seq):
        k = bisect_left(tail_value, value)
        if k == len(tail_index):
            tail_index.append(i)
            tail_value.append(value)
        else:
            tail_index[k] = i
            tail_value[k] = value
        prev[i] = tail_index[k - 1] if k > 0 else -1

    out: set[int] = set()
    cursor = tail_index[-1]
    while cursor != -1:
        out.add(cursor)
        cursor = prev[cursor]
    return out
