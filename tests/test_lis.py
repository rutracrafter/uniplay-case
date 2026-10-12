"""Phase 2 — the LIS primitive behind move classification."""

from __future__ import annotations

from coursedrift.lis import lis_indices


def _is_increasing(seq: list[int], keep: set[int]) -> bool:
    picked = [seq[i] for i in sorted(keep)]
    return all(a < b for a, b in zip(picked, picked[1:], strict=False))


def test_empty() -> None:
    assert lis_indices([]) == set()


def test_single() -> None:
    assert lis_indices([7]) == {0}


def test_already_sorted_keeps_everything() -> None:
    assert lis_indices([0, 1, 2, 3, 4]) == {0, 1, 2, 3, 4}


def test_reversed_keeps_exactly_one() -> None:
    keep = lis_indices([4, 3, 2, 1, 0])
    assert len(keep) == 1


def test_swap_marks_one_of_the_pair() -> None:
    keep = lis_indices([1, 0])
    assert len(keep) == 1


def test_rotation_matches_the_worked_example() -> None:
    """The plan's A B C D -> C A B D' case: C alone is the move."""
    assert lis_indices([1, 2, 0, 3]) == {0, 1, 3}


def test_interleave() -> None:
    keep = lis_indices([0, 2, 1, 3])
    assert len(keep) == 3
    assert _is_increasing([0, 2, 1, 3], keep)


def test_result_is_always_an_increasing_subsequence() -> None:
    cases = [
        [3, 1, 4, 1, 5, 9, 2, 6],
        [10, 9, 2, 5, 3, 7, 101, 18],
        [0, 0, 0],
        [5, 1, 6, 2, 7, 3],
    ]
    for seq in cases:
        assert _is_increasing(seq, lis_indices(seq)), seq


def test_strictly_increasing_so_duplicates_cannot_both_survive() -> None:
    """Two spans cannot occupy one position; the subsequence must be strict."""
    assert len(lis_indices([0, 0, 0])) == 1


def test_length_is_optimal_on_known_cases() -> None:
    assert len(lis_indices([10, 9, 2, 5, 3, 7, 101, 18])) == 4
    assert len(lis_indices([3, 1, 4, 1, 5, 9, 2, 6])) == 4
