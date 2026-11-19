"""Detect START/ENTER buttons dynamically from the playfield."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)

# Lavender Start button on the Anagrams pre-game screen (hue ~177 in OpenCV HSV).
START_BUTTON_HSV_LOWER = np.array([165, 35, 175], dtype=np.uint8)
START_BUTTON_HSV_UPPER = np.array([180, 85, 255], dtype=np.uint8)


@dataclass(frozen=True)
class ButtonHit:
    name: str
    label: str
    rect: tuple[int, int, int, int]
    center: tuple[int, int]


def _tile_anchor(
    tile_rects: list[tuple[int, int, int, int]] | None,
) -> tuple[int | None, int | None]:
    if not tile_rects:
        return None, None
    tops = [rect[1] for rect in tile_rects]
    bottoms = [rect[3] for rect in tile_rects]
    return min(tops), max(bottoms)


def _valid_play_rack(
    tile_rects: list[tuple[int, int, int, int]] | None,
    game_height: int,
) -> bool:
    """True when tile_rects look like the bottom letter rack during active play."""
    if not tile_rects or len(tile_rects) not in (6, 7):
        return False

    heights = [rect[3] - rect[1] for rect in tile_rects]
    widths = [rect[2] - rect[0] for rect in tile_rects]
    if max(heights) > 150 or min(heights) < 35:
        return False
    if max(widths) > 150 or min(widths) < 35:
        return False

    mean_y = sum((rect[1] + rect[3]) / 2 for rect in tile_rects) / len(tile_rects)
    return mean_y > game_height * 0.68


def _search_band(
    game_image: np.ndarray,
    *,
    top: int,
    bottom: int,
    x_margin: float = 0.08,
) -> tuple[np.ndarray, int, int]:
    height, width = game_image.shape[:2]
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    left = int(width * x_margin)
    right = int(width * (1.0 - x_margin))
    return game_image[top:bottom, left:right].copy(), left, top


def _ocr_label_boxes(region_bgr: np.ndarray, label: str) -> list[tuple[int, int, int, int, str]]:
    if region_bgr.size == 0:
        return []

    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(
        binary,
        output_type=Output.DICT,
        config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )

    hits: list[tuple[int, int, int, int, str]] = []
    target = label.upper()
    n = len(data["text"])
    for index in range(n):
        text = str(data["text"][index]).strip().upper()
        conf = int(float(data["conf"][index])) if data["conf"][index] != "-1" else -1
        if conf < 35 or not text:
            continue
        if target not in text and text not in target:
            continue
        x = int(data["left"][index] / 2)
        y = int(data["top"][index] / 2)
        w = int(data["width"][index] / 2)
        h = int(data["height"][index] / 2)
        hits.append((x, y, x + w, y + h, text))
    return hits


def _find_label_button(
    game_image: np.ndarray,
    *,
    label: str,
    top: int,
    bottom: int,
    x_margin: float = 0.08,
) -> ButtonHit | None:
    region, offset_x, offset_y = _search_band(
        game_image,
        top=top,
        bottom=bottom,
        x_margin=x_margin,
    )
    boxes = _ocr_label_boxes(region, label)
    if not boxes:
        return None

    region_h, region_w = region.shape[:2]
    center_x = region_w / 2

    def score(box: tuple[int, int, int, int, str]) -> float:
        x1, y1, x2, y2, _ = box
        width = x2 - x1
        height = y2 - y1
        box_center = (x1 + x2) / 2
        return width * height - abs(box_center - center_x) * 10 - abs(y1 - region_h / 2) * 5

    best = max(boxes, key=score)
    x1, y1, x2, y2, _text = best
    pad_x = max(8, int((x2 - x1) * 0.6))
    pad_y = max(6, int((y2 - y1) * 1.2))
    gx1 = max(0, offset_x + x1 - pad_x)
    gy1 = max(0, offset_y + y1 - pad_y)
    gx2 = min(game_image.shape[1], offset_x + x2 + pad_x)
    gy2 = min(game_image.shape[0], offset_y + y2 + pad_y)
    center = ((gx1 + gx2) // 2, (gy1 + gy2) // 2)
    return ButtonHit(name=label.lower(), label=label, rect=(gx1, gy1, gx2, gy2), center=center)


def _find_how_to_play_modal(game_image: np.ndarray) -> tuple[int, int, int, int] | None:
    """The white instruction card on the Anagrams / Word Hunt start screen."""
    height, width = game_image.shape[:2]
    gray = cv2.cvtColor(game_image, cv2.COLOR_BGR2GRAY)
    mask = (gray > 235).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int, int, int] | None = None
    best_area = 0

    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        if area < width * height * 0.08:
            continue
        if box_h < height * 0.20 or box_h > height * 0.50:
            continue
        if box_w > width * 0.92:
            continue
        if abs((x + box_w / 2) - width / 2) > width * 0.15:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, x + box_w, y + box_h)

    return best


def is_how_to_play_screen(game_image: np.ndarray) -> bool:
    return _find_how_to_play_modal(game_image) is not None


def _find_colored_button_in_rect(
    game_image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
    name: str,
    label: str,
    near: tuple[int, int] | None = None,
) -> ButtonHit | None:
    left = max(0, left)
    top = max(0, top)
    right = min(game_image.shape[1], right)
    bottom = min(game_image.shape[0], bottom)
    region = game_image[top:bottom, left:right]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    region_h, region_w = region.shape[:2]
    best = None
    best_score = -1.0
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area < 2500:
            continue
        if width < region_w * 0.18:
            continue
        aspect = width / max(height, 1)
        if aspect < 1.2 or aspect > 6.0:
            continue
        center_x = left + x + width / 2
        center_y = top + y + height / 2
        center_penalty = abs(center_x - (left + region_w / 2))
        near_penalty = 0.0
        if near is not None:
            near_penalty = abs(center_x - near[0]) + abs(center_y - near[1]) * 1.5
        score = area - center_penalty * 15 - near_penalty * 2
        if score > best_score:
            best_score = score
            best = (x, y, x + width, y + height)

    if best is None:
        return None

    x1, y1, x2, y2 = best
    rect = (left + x1, top + y1, left + x2, top + y2)
    center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
    return ButtonHit(name=name, label=label, rect=rect, center=center)


def _find_lavender_button_in_rect(
    game_image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> ButtonHit | None:
    return _find_colored_button_in_rect(
        game_image,
        left,
        top,
        right,
        bottom,
        hsv_lower=START_BUTTON_HSV_LOWER,
        hsv_upper=START_BUTTON_HSV_UPPER,
        name="start",
        label="START",
    )


def _find_lavender_start_button(
    game_image: np.ndarray,
    *,
    top: int,
    bottom: int,
) -> ButtonHit | None:
    """Find the lavender Start button on the pre-game screen."""
    region, offset_x, offset_y = _search_band(game_image, top=top, bottom=bottom, x_margin=0.18)
    if region.size == 0:
        return None

    hit = _find_lavender_button_in_rect(
        game_image,
        offset_x,
        offset_y,
        offset_x + region.shape[1],
        offset_y + region.shape[0],
    )
    return hit


def _is_start_label(text: str) -> bool:
    """GamePigeon labels the button 'Start' (capital S, lowercase rest)."""
    return text == "Start" or text.lower() == "start"


def _modal_start_rect(modal: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Bottom of the how-to-play card where the Start button lives."""
    x1, y1, x2, y2 = modal
    pad_x = int((x2 - x1) * 0.08)
    top = y1 + int((y2 - y1) * 0.72)
    bottom = y2 - int((y2 - y1) * 0.02)
    return x1 + pad_x, top, x2 - pad_x, bottom


def _ocr_start_button(
    game_image: np.ndarray,
    top: int,
    bottom: int,
    *,
    left: int = 0,
    right: int | None = None,
) -> ButtonHit | None:
    """Find the Start button by reading its label text."""
    if right is None:
        right = game_image.shape[1]
    left = max(0, left)
    right = min(game_image.shape[1], right)
    region_bgr = game_image[top:bottom, left:right]
    if region_bgr.size == 0:
        return None

    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    height, width = gray.shape

    binaries: list[np.ndarray] = []
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaries.append(otsu)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(otsu_inv)

    hits: list[tuple[float, tuple[int, int, int, int, str]]] = []
    center_x = width / 2
    for binary in binaries:
        for psm in (6, 7, 8):
            try:
                data = pytesseract.image_to_data(
                    binary,
                    output_type=Output.DICT,
                    config=f"--psm {psm} -c tessedit_char_whitelist=Start",
                )
            except pytesseract.TesseractError:
                continue
            for index in range(len(data["text"])):
                text = str(data["text"][index]).strip()
                conf = int(float(data["conf"][index])) if data["conf"][index] != "-1" else -1
                if conf < 50 or not _is_start_label(text):
                    continue
                x1 = int(data["left"][index] / 2)
                y1 = int(data["top"][index] / 2)
                x2 = x1 + int(data["width"][index] / 2)
                y2 = y1 + int(data["height"][index] / 2)
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 24 or box_h < 8:
                    continue
                if box_w > width * 0.8 or box_h > height * 0.8:
                    continue
                box_center = (x1 + x2) / 2
                score = conf + box_w - abs(box_center - center_x) * 0.1
                hits.append((score, (x1, y1, x2, y2, text)))

    if not hits:
        return None

    _score, (x1, y1, x2, y2, label) = max(hits, key=lambda item: item[0])
    pad_x = max(16, int((x2 - x1) * 1.2))
    pad_y = max(12, int((y2 - y1) * 1.8))
    gx1 = max(0, left + x1 - pad_x)
    gy1 = max(0, top + y1 - pad_y)
    gx2 = min(game_image.shape[1], left + x2 + pad_x)
    gy2 = min(game_image.shape[0], top + y2 + pad_y)
    center = ((gx1 + gx2) // 2, (gy1 + gy2) // 2)
    return ButtonHit(name="start", label=label, rect=(gx1, gy1, gx2, gy2), center=center)


def _start_search_band(game_image: np.ndarray) -> tuple[int, int]:
    height = game_image.shape[0]
    return int(height * 0.35), int(height * 0.88)


def _enter_search_region(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]] | None,
) -> tuple[int, int]:
    """Everything above the letter tiles — ENTER lives in that band."""
    height = game_image.shape[0]
    top = max(0, int(height * 0.03))
    if tile_rects:
        tile_top = min(rect[1] for rect in tile_rects)
        bottom = max(top + 80, tile_top - max(4, int(height * 0.005)))
    else:
        bottom = int(height * 0.82)
    return top, min(height, bottom)


def _ocr_enter_button(
    game_image: np.ndarray,
    top: int,
    bottom: int,
) -> ButtonHit | None:
    region_bgr = game_image[top:bottom, :]
    if region_bgr.size == 0:
        return None

    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    height, width = gray.shape

    binaries: list[np.ndarray] = []
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(otsu_inv)
    binaries.append(
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaries.append(otsu)

    hits: list[tuple[float, tuple[int, int, int, int]]] = []
    center_x = width / 2
    for binary in binaries:
        for psm in (6, 7, 11, 3):
            try:
                data = pytesseract.image_to_data(
                    binary,
                    output_type=Output.DICT,
                    config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                )
            except pytesseract.TesseractError:
                continue
            for index in range(len(data["text"])):
                text = str(data["text"][index]).strip().upper()
                conf = int(float(data["conf"][index])) if data["conf"][index] != "-1" else -1
                if conf < 30 or not text:
                    continue
                if "ENTER" not in text and text not in {"ENTER", "ENT", "NTER", "ENTE"}:
                    continue
                x1 = int(data["left"][index] / 2)
                y1 = int(data["top"][index] / 2)
                x2 = x1 + int(data["width"][index] / 2)
                y2 = y1 + int(data["height"][index] / 2)
                box_w = x2 - x1
                box_center = (x1 + x2) / 2
                score = conf + box_w - abs(box_center - center_x) * 0.05
                hits.append((score, (x1, y1, x2, y2)))

    if not hits:
        return None

    _score, (x1, y1, x2, y2) = max(hits, key=lambda item: item[0])
    pad_x = max(12, int((x2 - x1) * 0.8))
    pad_y = max(10, int((y2 - y1) * 1.5))
    gx1 = max(0, x1 - pad_x)
    gy1 = max(0, top + y1 - pad_y)
    gx2 = min(game_image.shape[1], x2 + pad_x)
    gy2 = min(game_image.shape[0], top + y2 + pad_y)
    center = ((gx1 + gx2) // 2, (gy1 + gy2) // 2)
    return ButtonHit(name="enter", label="ENTER", rect=(gx1, gy1, gx2, gy2), center=center)


def _find_enter_by_light_band(
    game_image: np.ndarray,
    top: int,
    bottom: int,
) -> ButtonHit | None:
    """Fallback: find a wide, shallow light-grey band (the ENTER button)."""
    height, width = game_image.shape[:2]
    left = int(width * 0.06)
    right = int(width * 0.94)
    region = game_image[top:bottom, left:right]
    if region.size == 0:
        return None

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape
    light = (gray >= 95) & (gray <= 200)
    row_fill = np.mean(light, axis=1)

    best_rect: tuple[int, int, int, int] | None = None
    best_score = -1.0
    min_h = max(24, int(rh * 0.06))
    max_h = max(min_h + 1, int(rh * 0.22))

    for y1 in range(rh):
        for y2 in range(y1 + min_h, min(y1 + max_h, rh) + 1):
            band = row_fill[y1:y2]
            if band.size == 0 or float(np.mean(band)) < 0.40:
                continue
            mid = y1 + (y2 - y1) // 2
            cols = np.where(light[mid])[0]
            if len(cols) < rw * 0.35:
                continue
            x1, x2 = int(cols[0]), int(cols[-1])
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < rw * 0.40:
                continue
            aspect = box_w / max(box_h, 1)
            if aspect < 2.0:
                continue
            center_penalty = abs((x1 + x2) / 2 - rw / 2)
            score = box_w * box_h - center_penalty * 10
            if score > best_score:
                best_score = score
                best_rect = (left + x1, top + y1, left + x2, top + y2)

    if best_rect is None:
        return None

    x1, y1, x2, y2 = best_rect
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return ButtonHit(name="enter", label="ENTER", rect=(x1, y1, x2, y2), center=center)


def detect_start_button(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]] | None = None,
) -> ButtonHit | None:
    """Find Start by reading the button label text."""
    modal = _find_how_to_play_modal(game_image)
    if modal:
        left, top, right, bottom = _modal_start_rect(modal)
        hit = _ocr_start_button(game_image, top, bottom, left=left, right=right)
        if hit:
            return hit
        return _find_lavender_button_in_rect(game_image, left, top, right, bottom)

    top, bottom = _start_search_band(game_image)
    width = game_image.shape[1]
    margin = int(width * 0.08)
    hit = _ocr_start_button(game_image, top, bottom, left=margin, right=width - margin)
    if hit:
        return hit

    height = game_image.shape[0]
    return _find_lavender_start_button(game_image, top=int(height * 0.58), bottom=int(height * 0.72))


def detect_enter_button(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]] | None = None,
) -> ButtonHit | None:
    top, bottom = _enter_search_region(game_image, tile_rects)
    hit = _ocr_enter_button(game_image, top, bottom)
    if hit:
        return hit
    return _find_enter_by_light_band(game_image, top, bottom)


def find_button(
    game_image: np.ndarray,
    name: str,
    tile_rects: list[tuple[int, int, int, int]] | None = None,
) -> ButtonHit | None:
    if name == "enter":
        return detect_enter_button(game_image, tile_rects)
    if name == "start":
        return detect_start_button(game_image, tile_rects)
    raise KeyError(f"Unknown button: {name}")


def _valid_play_grid(
    tile_rects: list[tuple[int, int, int, int]] | None,
    game_height: int,
    *,
    tile_counts: frozenset[int] | None = None,
) -> bool:
    valid_counts = tile_counts or frozenset({16, 20, 21, 25})
    if not tile_rects or len(tile_rects) not in valid_counts:
        return False

    heights = [rect[3] - rect[1] for rect in tile_rects]
    widths = [rect[2] - rect[0] for rect in tile_rects]
    if max(heights) > game_height * 0.22 or min(heights) < 20:
        return False
    if max(widths) > game_height * 0.22 or min(widths) < 20:
        return False

    mean_y = sum((rect[1] + rect[3]) / 2 for rect in tile_rects) / len(tile_rects)
    return mean_y > game_height * 0.42


def is_start_screen(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]] | None = None,
) -> bool:
    if _valid_play_rack(tile_rects, game_image.shape[0]):
        return False
    if is_how_to_play_screen(game_image):
        return True

    top, bottom = _start_search_band(game_image)
    width = game_image.shape[1]
    margin = int(width * 0.08)
    return _ocr_start_button(game_image, top, bottom, left=margin, right=width - margin) is not None


def is_play_screen(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]] | None = None,
) -> bool:
    if is_how_to_play_screen(game_image):
        return False
    return _valid_play_rack(tile_rects, game_image.shape[0])


def is_ready_to_play(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]] | None = None,
    *,
    has_enter_button: bool = True,
) -> bool:
    """True when the bot can play — includes mid-game screens strict play detection misses."""
    if is_start_screen(game_image, tile_rects):
        return False
    if is_play_screen(game_image, tile_rects):
        return True
    if _valid_play_grid(tile_rects, game_image.shape[0]):
        return True
    if not has_enter_button:
        return False
    return detect_enter_button(game_image, tile_rects) is not None


def detect_buttons(
    game_image: np.ndarray,
    tile_rects: list[tuple[int, int, int, int]] | None = None,
    *,
    has_enter_button: bool = True,
) -> dict[str, ButtonHit]:
    buttons: dict[str, ButtonHit] = {}
    if is_start_screen(game_image, tile_rects):
        start = detect_start_button(game_image, tile_rects)
        if start:
            buttons["start"] = start
        return buttons

    if has_enter_button:
        enter = detect_enter_button(game_image, tile_rects)
        if enter:
            buttons["enter"] = enter
    return buttons
