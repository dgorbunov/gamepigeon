"""Alphagram (sorted-letter) index for fast anagram lookup."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator


def to_alphagram(word: str) -> str:
    return "".join(sorted(word.strip().lower()))


def normalize_word(word: str, *, min_length: int = 3, max_length: int = 7) -> str | None:
    cleaned = word.strip().lower()
    if not cleaned.isalpha():
        return None
    if not (min_length <= len(cleaned) <= max_length):
        return None
    return cleaned


@dataclass(frozen=True)
class AlphagramIndex:
    """Maps alphagrams to all dictionary words with those letters."""

    by_alphagram: dict[str, tuple[str, ...]]

    @classmethod
    def from_words(
        cls,
        words: Iterable[str],
        *,
        min_length: int = 3,
        max_length: int = 7,
    ) -> AlphagramIndex:
        buckets: dict[str, list[str]] = defaultdict(list)
        for raw in words:
            word = normalize_word(raw, min_length=min_length, max_length=max_length)
            if word is None:
                continue
            buckets[to_alphagram(word)].append(word)

        by_alphagram = {
            key: tuple(sorted(set(values))) for key, values in buckets.items()
        }
        return cls(by_alphagram=by_alphagram)

    def lookup(self, alphagram: str) -> tuple[str, ...]:
        return self.by_alphagram.get(to_alphagram(alphagram), ())

    def __len__(self) -> int:
        return len(self.by_alphagram)

    def word_count(self) -> int:
        return sum(len(words) for words in self.by_alphagram.values())
