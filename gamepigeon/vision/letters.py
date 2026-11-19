"""Letter recognition for wooden GamePigeon tiles."""

from __future__ import annotations

import cv2
import numpy as np

from gamepigeon.vision.letter_match import crop_wood_tile, match_tile_letter


def tile_looks_filled(tile_bgr: np.ndarray) -> bool:
    if tile_bgr.size == 0:
        return False

    wood = crop_wood_tile(tile_bgr)
    gray = cv2.cvtColor(wood, cv2.COLOR_BGR2GRAY)
    background = float(np.percentile(gray, 82))
    ink = float(np.mean(np.clip(background - gray, 0, 255)))
    return ink >= 5.0


def read_tile_letter(tile_bgr: np.ndarray) -> str | None:
    if tile_bgr.size == 0 or not tile_looks_filled(tile_bgr):
        return None

    letter, _score = match_tile_letter(tile_bgr)
    return letter
