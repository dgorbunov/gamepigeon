"""Read letters from GamePigeon wood tiles via Tesseract."""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

TESSERACT_CONFIG = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Tuned on tests/fixtures rack crops.
_INNER_MARGINS = (0.08, 0.05, 0.12, 0.0)
_OCR_SIZE = 96
_BORDER = 20

_WOOD_RGB = (188, 178, 168)
_INK_RGB = (20, 20, 20)
_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
_FONT_SIZE = 40


def crop_wood_tile(tile_bgr: np.ndarray) -> np.ndarray:
    """Isolate a tile square when the crop still includes rack gaps or green background."""
    return _crop_tile_if_needed(tile_bgr)


def _crop_tile_if_needed(tile_bgr: np.ndarray) -> np.ndarray:
    if tile_bgr.size == 0:
        return tile_bgr

    height, width = tile_bgr.shape[:2]
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 100).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return tile_bgr

    x, y, box_width, box_height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if box_width * box_height < 0.76 * width * height:
        return tile_bgr[y : y + box_height, x : x + box_width]
    return tile_bgr


def preprocess_for_ocr(tile_bgr: np.ndarray, *, margin: float = _INNER_MARGINS[0]) -> np.ndarray:
    tile = _crop_tile_if_needed(tile_bgr)
    height, width = tile.shape[:2]
    inner = tile[
        int(height * margin) : int(height * (1 - margin)),
        int(width * margin) : int(width * (1 - margin)),
    ]
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (_OCR_SIZE, _OCR_SIZE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(binary, _BORDER, _BORDER, _BORDER, _BORDER, cv2.BORDER_CONSTANT, value=0)


def match_tile_letter(tile_bgr: np.ndarray) -> tuple[str | None, float]:
    if tile_bgr.size == 0:
        return None, 0.0

    for margin in _INNER_MARGINS:
        image = preprocess_for_ocr(tile_bgr, margin=margin)
        text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG).strip().upper()
        letter = next((character for character in text if character.isalpha()), None)
        if letter is not None:
            return letter, 1.0
    return None, 0.0


def render_tile_image(letter: str, *, size: int = 96) -> np.ndarray:
    """Render a wood tile for tests and debugging."""
    font = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
    image = Image.new("RGB", (size, size), _WOOD_RGB)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), letter.upper(), font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = (
        (size - text_width) // 2 - bbox[0],
        (size - text_height) // 2 - bbox[1],
    )
    draw.text(position, letter.upper(), fill=_INK_RGB, font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
