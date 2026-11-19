"""Anagram helpers and Word Hunt grid solver."""

from __future__ import annotations

from typing import Iterable

from gamepigeon.words.alphagram_index import AlphagramIndex
from gamepigeon.words.anagram_solver import AnagramSolver
from gamepigeon.words.dictionary import load_alphagram_index, load_prefix_set, load_scrabble_words


def find_anagrams(
    letters: Iterable[str],
    dictionary: list[str] | None = None,
    *,
    min_length: int = 3,
    max_length: int | None = None,
    exclude: set[str] | None = None,
    solver: AnagramSolver | None = None,
    **_: object,
) -> list[str]:
    """Find valid anagrams for a rack using the alphagram solver."""
    if solver is None:
        index = (
            AlphagramIndex.from_words(dictionary)
            if dictionary is not None
            else load_alphagram_index()
        )
        rack_len = len([ch for ch in letters if ch.isalpha()])
        solver = AnagramSolver(
            index,
            min_length=min_length,
            max_length=max_length or min(7, rack_len),
        )
    return solver.solve(letters, exclude=exclude)


def find_words_in_grid(
    grid: list[list[str]],
    dictionary: list[str],
    *,
    min_length: int = 3,
    max_length: int = 7,
    exclude: set[str] | None = None,
    prefixes: frozenset[str] | None = None,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """
    Find valid words on a letter grid (Word Hunt style).
    Empty / non-letter cells are treated as holes (cross, donut, etc.).
    Returns (word, path) sorted longest-first.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    exclude = exclude or set()
    dictionary_set = set(dictionary)
    prefix_set = prefixes if prefixes is not None else load_prefix_set(max_length=max_length)
    found: dict[str, list[tuple[int, int]]] = {}
    live_tiles = sum(1 for r in range(rows) for c in range(cols) if grid[r][c].isalpha())

    def neighbors(r: int, c: int):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc].isalpha():
                    yield nr, nc

    def dfs(path: list[tuple[int, int]], current: str) -> None:
        if len(current) >= min_length and current in dictionary_set and current not in exclude:
            found.setdefault(current, path.copy())

        if len(path) >= live_tiles or len(current) >= max_length:
            return

        r, c = path[-1]
        for nr, nc in neighbors(r, c):
            if (nr, nc) in path:
                continue
            next_word = current + grid[nr][nc].lower()
            if next_word not in prefix_set:
                continue
            path.append((nr, nc))
            dfs(path, next_word)
            path.pop()

    for r in range(rows):
        for c in range(cols):
            letter = grid[r][c].lower()
            if not letter.isalpha():
                continue
            dfs([(r, c)], letter)

    return sorted(found.items(), key=lambda item: (-len(item[0]), item[0]))


__all__ = [
    "find_anagrams",
    "find_words_in_grid",
    "load_scrabble_words",
]
