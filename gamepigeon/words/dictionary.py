"""Load offline Scrabble word lists (not shipped with the repo)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from gamepigeon.words.alphagram_index import AlphagramIndex, normalize_word

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
# Prefer a full CSW21 export; filtered one-word-per-line files also work.
CSW21_PATH = DATA_DIR / "CSW21.txt"
FILTERED_WORDS_PATH = DATA_DIR / "scrabble_words_3_7.txt"
LEGACY_WORD_PATH = DATA_DIR / "words.txt"

WORD_LIST_CANDIDATES: tuple[Path, ...] = (
    CSW21_PATH,
    FILTERED_WORDS_PATH,
    LEGACY_WORD_PATH,
)


def _word_list_hint() -> str:
    expected = ", ".join(path.name for path in WORD_LIST_CANDIDATES)
    return (
        f"No word list found in {DATA_DIR} (looked for {expected}). "
        "See README for how to download Collins CSW21."
    )


def load_word_lines(path: Path) -> list[str]:
    """Read a word list. Supports plain words or CSW/Zyzzyva lines (`WORD definition…`)."""
    if not path.exists():
        raise FileNotFoundError(f"Word list not found: {path}")

    words: list[str] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # CSW exports are "WORD rest of definition…"; plain lists are just the word.
            token = line.split(None, 1)[0]
            words.append(token)
    return words


def resolve_word_list_path() -> Path:
    for path in WORD_LIST_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(_word_list_hint())


@lru_cache(maxsize=1)
def load_scrabble_words(
    *,
    min_length: int = 3,
    max_length: int = 7,
) -> tuple[str, ...]:
    """Load CSW21 (or a compatible list) and keep words in [min_length, max_length]."""
    path = resolve_word_list_path()
    words = [
        word
        for raw in load_word_lines(path)
        if (word := normalize_word(raw, min_length=min_length, max_length=max_length)) is not None
    ]
    logger.info(
        "Loaded %d words (%d–%d letters) from %s",
        len(words),
        min_length,
        max_length,
        path.name,
    )
    return tuple(words)


@lru_cache(maxsize=1)
def load_prefix_set(
    *,
    min_length: int = 3,
    max_length: int = 7,
) -> frozenset[str]:
    """All word prefixes for pruning grid DFS."""
    prefixes: set[str] = set()
    for word in load_scrabble_words(min_length=min_length, max_length=max_length):
        for end in range(1, len(word) + 1):
            prefixes.add(word[:end])
    return frozenset(prefixes)


@lru_cache(maxsize=1)
def load_alphagram_index(
    *,
    min_length: int = 3,
    max_length: int = 7,
) -> AlphagramIndex:
    words = load_scrabble_words(min_length=min_length, max_length=max_length)
    index = AlphagramIndex.from_words(words, min_length=min_length, max_length=max_length)
    logger.info("Built alphagram index: %d keys, %d words", len(index), index.word_count())
    return index


def load_words(
    *,
    min_length: int = 3,
    max_length: int = 7,
) -> list[str]:
    """Backward-compatible word list loader for Word Hunt and other games."""
    return list(load_scrabble_words(min_length=min_length, max_length=max_length))
