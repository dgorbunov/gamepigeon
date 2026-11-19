"""OCR helpers for reading letters from tile images."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

TESSERACT_CONFIG = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def preprocess_tile(tile_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def read_single_letter(tile_bgr: np.ndarray) -> str | None:
    if tile_bgr.size == 0:
        return None

    processed = preprocess_tile(tile_bgr)
    text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG).strip().upper()
    text = "".join(ch for ch in text if ch.isalpha())

    if len(text) == 1:
        return text

    if text:
        return text[0]

    return guess_letter_from_contours(tile_bgr)


def guess_letter_from_contours(tile_bgr: np.ndarray) -> str | None:
    """Fallback when OCR fails — detect if tile has content vs empty/question mark."""
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ratio = np.count_nonzero(binary) / binary.size
    if ratio < 0.04:
        return None
    return "?"


def read_button_label(region_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(
        binary,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    return "".join(ch for ch in text.upper() if ch.isalpha())
