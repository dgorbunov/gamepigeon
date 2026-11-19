"""Detect letter rack tiles in the Anagrams playfield."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from gamepigeon.config import NormalizedRect

logger = logging.getLogger(__name__)

SUPPORTED_TILE_COUNTS = frozenset({6, 7})
GRID_TILE_COUNTS = frozenset({16, 20, 21, 25})

# Tutorial tiles on the start screen (light tan wood in the how-to-play modal).
TUTORIAL_TILE_HSV_LOWER = np.array([90, 20, 150], dtype=np.uint8)
TUTORIAL_TILE_HSV_UPPER = np.array([110, 90, 255], dtype=np.uint8)


def _tile_search_region(
    game_image: np.ndarray,
    tile_area: NormalizedRect,
) -> tuple[np.ndarray, int, int]:
    left, top, right, bottom = tile_area.pixel_rect(game_image.shape[1], game_image.shape[0])
    return game_image[top:bottom, left:right].copy(), left, top


def _boxes_from_mask(
    mask: np.ndarray,
    *,
    min_area: int,
    min_side: int,
) -> list[tuple[int, int, int, int]]:
    if mask.size == 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area < min_area or width < min_side or height < min_side:
            continue

        aspect = width / height
        if aspect > 1.8 or aspect < 0.55:
            continue

        boxes.append((x, y, x + width, y + height))

    return boxes


def _dedupe_boxes(
    boxes: list[tuple[int, int, int, int]],
    *,
    overlap_threshold: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]), reverse=True):
        x1, y1, x2, y2 = box
        area = max((x2 - x1) * (y2 - y1), 1)
        duplicate = False
        for kept_box in kept:
            kx1, ky1, kx2, ky2 = kept_box
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            overlap = (ix2 - ix1) * (iy2 - iy1) / area
            if overlap >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    return sorted(kept, key=lambda rect: rect[0])


def _find_play_tile_boxes(
    region: np.ndarray,
    *,
    min_area: int = 2000,
    min_side: int = 28,
) -> list[tuple[int, int, int, int]]:
    """
    Find square rack tiles using local contrast instead of fixed color ranges.

    Works for light wood tiles, warm post-submit tiles, and darker taupe tiles.
    """
    if region.size == 0:
        return []

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    diff = gray.astype(np.int16) - blur.astype(np.int16)

    masks = [
        (diff > 12).astype(np.uint8) * 255,
        ((-diff) > 10).astype(np.uint8) * 255,
    ]

    boxes: list[tuple[int, int, int, int]] = []
    for mask in masks:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        boxes.extend(_boxes_from_mask(cleaned, min_area=min_area, min_side=min_side))

    return _dedupe_boxes(boxes)


def _find_tutorial_tile_boxes(region: np.ndarray) -> list[tuple[int, int, int, int]]:
    if region.size == 0:
        return []

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TUTORIAL_TILE_HSV_LOWER, TUTORIAL_TILE_HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return _boxes_from_mask(mask, min_area=1200, min_side=20)


def _merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int]],
    *,
    x_gap: int = 12,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    merged: list[tuple[int, int, int, int]] = []
    current = boxes[0]

    for candidate in boxes[1:]:
        cx1, cy1, cx2, cy2 = current
        nx1, ny1, nx2, ny2 = candidate
        if nx1 - cx2 < x_gap:
            x1 = min(cx1, nx1)
            y1 = min(cy1, ny1)
            x2 = max(cx2, nx2)
            y2 = max(cy2, ny2)
            current = (x1, y1, x2, y2)
        else:
            merged.append(current)
            current = candidate

    merged.append(current)
    return merged


def _cluster_rows(
    boxes: list[tuple[int, int, int, int]],
    *,
    y_tolerance: int | None = None,
) -> list[list[tuple[int, int, int, int]]]:
    if not boxes:
        return []

    if y_tolerance is None:
        heights = [box[3] - box[1] for box in boxes]
        median_height = float(np.median(heights)) if heights else 24.0
        y_tolerance = max(8, int(median_height * 0.35))

    sorted_boxes = sorted(boxes, key=lambda item: ((item[1] + item[3]) / 2, item[0]))
    rows: list[list[tuple[int, int, int, int]]] = []
    current_row: list[tuple[int, int, int, int]] = [sorted_boxes[0]]
    current_y = (sorted_boxes[0][1] + sorted_boxes[0][3]) / 2

    for box in sorted_boxes[1:]:
        center_y = (box[1] + box[3]) / 2
        if abs(center_y - current_y) <= y_tolerance:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            current_y = center_y

    rows.append(current_row)
    return rows


def _score_row(region: np.ndarray, row: list[tuple[int, int, int, int]]) -> tuple[float, float, float]:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    contrasts: list[float] = []
    brightness: list[float] = []
    for x1, y1, x2, y2 in row:
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        contrasts.append(float(np.std(patch)))
        brightness.append(float(np.mean(patch)))

    mean_y = sum((rect[1] + rect[3]) / 2 for rect in row) / len(row)
    contrast = float(np.mean(contrasts)) if contrasts else 0.0
    bright = float(np.mean(brightness)) if brightness else 0.0
    return (contrast, bright, mean_y)


def _row_to_tiles(
    row: list[tuple[int, int, int, int]],
    offset_x: int,
    offset_y: int,
) -> list[tuple[int, int, int, int]]:
    local_boxes = sorted(row, key=lambda rect: rect[0])
    if len(local_boxes) not in SUPPORTED_TILE_COUNTS:
        local_boxes = _merge_nearby_boxes(local_boxes)
    tiles: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in local_boxes:
        tiles.append((offset_x + x1, offset_y + y1, offset_x + x2, offset_y + y2))
    return tiles


def _pick_best_row(
    region: np.ndarray,
    rows: list[list[tuple[int, int, int, int]]],
) -> list[tuple[int, int, int, int]] | None:
    valid_rows = [row for row in rows if len(row) in SUPPORTED_TILE_COUNTS]
    if not valid_rows:
        return None

    # Prefer the rack row with visible letters, then brighter tiles, then lower on screen.
    return max(valid_rows, key=lambda row: _score_row(region, row))


def detect_wood_tiles(
    game_image: np.ndarray,
    tile_area: NormalizedRect,
) -> list[tuple[int, int, int, int]]:
    """
    Find rack tile rectangles in game-image coordinates.
    Returns (x1, y1, x2, y2) sorted left-to-right.
    """
    region, offset_x, offset_y = _tile_search_region(game_image, tile_area)
    if region.size == 0:
        return []

    candidates = _find_play_tile_boxes(region)
    if not candidates:
        candidates = _find_tutorial_tile_boxes(region)

    if not candidates:
        return []

    rows = _cluster_rows(candidates)
    best_row = _pick_best_row(region, rows)
    if best_row is None:
        logger.debug(
            "Detected rack rows with counts %s; expected one of %s",
            [len(row) for row in rows],
            sorted(SUPPORTED_TILE_COUNTS),
        )
        return []

    tiles = _row_to_tiles(best_row, offset_x, offset_y)
    if len(tiles) not in SUPPORTED_TILE_COUNTS:
        logger.debug("Merged to %d tiles; expected one of %s", len(tiles), sorted(SUPPORTED_TILE_COUNTS))
        return []

    return tiles


def _inset_rect(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    fraction: float = 0.04,
) -> tuple[int, int, int, int]:
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    inset_x = int(width * fraction)
    inset_y = int(height * fraction)
    return x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y


def _grid_tile_boxes(
    region: np.ndarray,
    *,
    expected_tiles: int = 16,
) -> list[tuple[int, int, int, int]]:
    """Find square grid tiles using bright-blob and local-contrast cues."""
    if region.size == 0:
        return []

    region_height, region_width = region.shape[:2]
    # Estimate cell size from how many tiles we expect in this search region.
    grid_side = max(4.0, float(np.sqrt(max(expected_tiles, 1))))
    cell = min(region_width, region_height) / grid_side
    min_side = max(12, int(cell * 0.35))
    min_area = max(400, int((cell * 0.35) ** 2))
    max_area = min(region_width * region_height * 0.12, (cell * 1.35) ** 2)

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    diff = gray.astype(np.int16) - blur.astype(np.int16)

    masks = [
        (gray > 100).astype(np.uint8) * 255,
        (gray > 130).astype(np.uint8) * 255,
        (diff > 8).astype(np.uint8) * 255,
        ((-diff) > 8).astype(np.uint8) * 255,
    ]

    boxes: list[tuple[int, int, int, int]] = []
    open_kernel = np.ones((5, 5), np.uint8)
    close_kernel = np.ones((3, 3), np.uint8)
    for mask in masks:
        # OPEN separates tightly packed 5x5/cross tiles; CLOSE helps sparse 4x4 racks.
        separated = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=2)
        cleaned = cv2.morphologyEx(separated, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        for box in _boxes_from_mask(cleaned, min_area=min_area, min_side=min_side):
            x1, y1, x2, y2 = box
            if (x2 - x1) * (y2 - y1) <= max_area:
                boxes.append(box)

    deduped = _dedupe_boxes(boxes)
    if not deduped:
        return []

    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in deduped]
    median_area = float(np.median(areas))
    consistent = [
        box
        for box, area in zip(deduped, areas)
        if 0.55 * median_area <= area <= 1.55 * median_area
    ]
    if len(consistent) >= max(expected_tiles // 2, 4):
        deduped = consistent
        median_area = float(
            np.median([(box[2] - box[0]) * (box[3] - box[1]) for box in deduped])
        )

    if len(deduped) > expected_tiles:
        deduped = sorted(
            deduped,
            key=lambda box: abs((box[2] - box[0]) * (box[3] - box[1]) - median_area),
        )[:expected_tiles]

    return sorted(deduped, key=lambda rect: ((rect[1] + rect[3]) / 2, (rect[0] + rect[2]) / 2))


def _assign_boxes_to_lattice(
    boxes: list[tuple[int, int, int, int]],
    *,
    rows: int,
    cols: int,
    occupied: frozenset[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int, int, int, int]] | None:
    """Map detected boxes onto a rows×cols lattice; None if assignment is ambiguous."""
    if not boxes:
        return None

    expected = occupied if occupied is not None else frozenset(
        (row, col) for row in range(rows) for col in range(cols)
    )
    if len(boxes) != len(expected):
        return None

    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    med_w = float(np.median(widths)) if widths else 1.0
    med_h = float(np.median(heights)) if heights else 1.0
    centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box) for box in boxes]
    min_cx = min(cx for cx, _, _ in centers)
    min_cy = min(cy for _, cy, _ in centers)

    # Prefer spacing implied by the full board span when tiles are missing on edges.
    max_cx = max(cx for cx, _, _ in centers)
    max_cy = max(cy for _, cy, _ in centers)
    col_step = (max_cx - min_cx) / max(cols - 1, 1) if cols > 1 else med_w
    row_step = (max_cy - min_cy) / max(rows - 1, 1) if rows > 1 else med_h
    if col_step < med_w * 0.55:
        col_step = med_w
    if row_step < med_h * 0.55:
        row_step = med_h

    assigned: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for cx, cy, box in centers:
        col = int(round((cx - min_cx) / col_step)) if col_step else 0
        row = int(round((cy - min_cy) / row_step)) if row_step else 0
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))
        key = (row, col)
        if key not in expected:
            return None
        if key in assigned:
            return None
        assigned[key] = box

    if set(assigned) != expected:
        return None

    return [
        (row, col, x1, y1, x2, y2)
        for (row, col), (x1, y1, x2, y2) in sorted(assigned.items())
    ]


def count_grid_tile_boxes(
    game_image: np.ndarray,
    tile_area: NormalizedRect,
    *,
    expected_tiles: int = 16,
) -> int:
    """Count candidate square tiles in a search band (for shape auto-detect)."""
    region, _, _ = _tile_search_region(game_image, tile_area)
    if region.size == 0:
        return 0
    return len(_grid_tile_boxes(region, expected_tiles=expected_tiles))


def detect_grid_tiles(
    game_image: np.ndarray,
    tile_area: NormalizedRect,
    *,
    rows: int = 4,
    cols: int = 4,
    occupied: frozenset[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
    """
    Find square tiles in a Word Hunt grid.
    Returns (row, col, x1, y1, x2, y2) in game-image coordinates.
    """
    region, offset_x, offset_y = _tile_search_region(game_image, tile_area)
    if region.size == 0:
        return []

    expected_tiles = len(occupied) if occupied is not None else rows * cols
    boxes = _grid_tile_boxes(region, expected_tiles=expected_tiles)
    if len(boxes) < expected_tiles:
        return []

    # Prefer an exact rectangular clustering when every cell is present.
    if occupied is None:
        row_groups = _cluster_rows(boxes)
        if len(row_groups) == rows and all(len(row) == cols for row in row_groups):
            tiles: list[tuple[int, int, int, int, int, int]] = []
            for row_index, row in enumerate(row_groups):
                for col_index, (x1, y1, x2, y2) in enumerate(sorted(row, key=lambda rect: rect[0])):
                    tiles.append(
                        (
                            row_index,
                            col_index,
                            offset_x + x1,
                            offset_y + y1,
                            offset_x + x2,
                            offset_y + y2,
                        )
                    )
            return sorted(tiles, key=lambda item: (item[0], item[1]))

    # Irregular boards (cross/donut) or imperfect clustering: snap to lattice.
    # If we found extras, keep the expected_tiles largest boxes.
    if len(boxes) > expected_tiles:
        boxes = sorted(
            boxes,
            key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]),
            reverse=True,
        )[:expected_tiles]

    assigned = _assign_boxes_to_lattice(
        boxes,
        rows=rows,
        cols=cols,
        occupied=occupied,
    )
    if assigned is None:
        return []

    return [
        (
            row,
            col,
            offset_x + x1,
            offset_y + y1,
            offset_x + x2,
            offset_y + y2,
        )
        for row, col, x1, y1, x2, y2 in assigned
    ]
