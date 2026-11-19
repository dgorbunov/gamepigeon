"""Detect Anagrams playfield state (tiles vs rack)."""

from __future__ import annotations

import cv2
import numpy as np

from gamepigeon.vision.buttons import _valid_play_rack
from gamepigeon.vision.letters import read_tile_letter, tile_looks_filled


def tile_has_letter(tile_bgr: np.ndarray) -> bool:
    letter = read_tile_letter(tile_bgr)
    return letter is not None and letter != "?"


def _rack_slot_rects(
    tile_rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """One slot per bottom tile, in the word-building row directly above them."""
    tile_top = min(rect[1] for rect in tile_rects)
    tile_height = max(rect[3] - rect[1] for rect in tile_rects)
    bottom = max(0, tile_top - 4)
    top = max(0, bottom - int(tile_height * 1.05))
    return [(x1, top, x2, bottom) for x1, _y1, x2, _y2 in tile_rects]


def _rack_slot_has_letter(slot_bgr: np.ndarray) -> bool:
    """Detect a letter glyph sitting in a rack slot (not empty wood or UI chrome)."""
    if slot_bgr.size == 0:
        return False

    height, width = slot_bgr.shape[:2]
    inner = slot_bgr[
        int(height * 0.22) : int(height * 0.78),
        int(width * 0.18) : int(width * 0.82),
    ]
    if inner.size == 0:
        return False

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    dark = gray < 85
    dark_ratio = float(np.mean(dark))
    if dark_ratio < 0.10 or dark_ratio > 0.50:
        return False

    letter = read_tile_letter(inner)
    return letter is not None and letter != "?"


def count_rack_letters(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]],
) -> int:
    if not tile_rects or not _valid_play_rack(tile_rects, game_image.shape[0]):
        return 0

    count = 0
    for x1, y1, x2, y2 in _rack_slot_rects(tile_rects):
        if _rack_slot_has_letter(game_image[y1:y2, x1:x2]):
            count += 1
    return count


def rack_has_pending_letters(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]] | None,
    *,
    min_letters: int = 3,
) -> bool:
    """True when selected letters are waiting in the rack to be submitted."""
    if not tile_rects:
        return False
    return count_rack_letters(game_image, tile_rects) >= min_letters


def count_filled_tiles(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]],
) -> int:
    count = 0
    for x1, y1, x2, y2 in tile_rects:
        if tile_looks_filled(game_image[y1:y2, x1:x2]):
            count += 1
    return count


def count_tiles_with_letters(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]],
) -> int:
    count = 0
    for x1, y1, x2, y2 in tile_rects:
        if tile_has_letter(game_image[y1:y2, x1:x2]):
            count += 1
    return count
