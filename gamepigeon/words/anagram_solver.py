"""High-performance anagram solver backed by an alphagram index."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from gamepigeon.words.alphagram_index import AlphagramIndex


def normalize_rack(letters: Iterable[str]) -> str:
    return "".join(ch.lower() for ch in letters if ch.isalpha())


def iter_rack_alphagrams(rack: str, min_length: int, max_length: int) -> Iterable[str]:
    """Unique alphagrams formable from rack letters (with duplicates)."""
    sorted_letters = sorted(rack)
    letter_count = len(sorted_letters)
    if letter_count == 0:
        return

    max_length = min(max_length, letter_count)
    seen: set[str] = set()
    for size in range(max_length, min_length - 1, -1):
        for indices in combinations(range(letter_count), size):
            alphagram = "".join(sorted_letters[index] for index in indices)
            if alphagram not in seen:
                seen.add(alphagram)
                yield alphagram


class AnagramSolver:
    def __init__(
        self,
        index: AlphagramIndex,
        *,
        min_length: int = 3,
        max_length: int = 7,
    ) -> None:
        self.index = index
        self.min_length = min_length
        self.max_length = max_length

    def solve(
        self,
        letters: Iterable[str],
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        rack = normalize_rack(letters)
        if len(rack) < self.min_length:
            return []

        exclude = {word.lower() for word in (exclude or set())}
        rack_len = min(len(rack), self.max_length)
        matches: set[str] = set()

        for alphagram in iter_rack_alphagrams(rack, self.min_length, rack_len):
            for word in self.index.lookup(alphagram):
                if word not in exclude:
                    matches.add(word)

        return sorted(matches, key=lambda word: (-len(word), word))
