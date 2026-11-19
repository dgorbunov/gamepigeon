"""Tile grid model and detection."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from gamepigeon.config import (
    WORD_HUNT_SHAPES,
    WORD_HUNT_SHAPES_BY_TILE_COUNT,
    GameLayout,
    GridLayout,
    NormalizedRect,
)
from gamepigeon.vision.buttons import is_how_to_play_screen
from gamepigeon.vision.letters import read_tile_letter
from gamepigeon.vision.tile_detect import count_grid_tile_boxes, detect_grid_tiles, detect_wood_tiles

logger = logging.getLogger(__name__)


def normalize_letter(letter: str) -> str:
    """Game tiles are always uppercase A-Z."""
    if len(letter) == 1 and letter.isalpha():
        return letter.upper()
    return letter


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    letter: str
    center_x: int
    center_y: int  # absolute coordinates within captured frame
    rect: tuple[int, int, int, int]  # x1, y1, x2, y2 in game-image coordinates


@dataclass
class TileGrid:
    layout: GridLayout
    tiles: list[Tile]
    frame_offset: tuple[int, int] = (0, 0)

    @property
    def letters(self) -> list[str]:
        return [tile.letter for tile in self.tiles]

    @property
    def letter_counts(self) -> Counter[str]:
        return Counter(self.letters)

    def tile_at(self, row: int, col: int) -> Tile | None:
        for tile in self.tiles:
            if tile.row == row and tile.col == col:
                return tile
        return None

    def indices_for_word(self, word: str) -> list[int] | None:
        """Map a word to tile indices, respecting duplicate letters."""
        available = list(enumerate(self.letters))
        indices: list[int] = []
        for char in word.upper():
            for idx, (tile_index, letter) in enumerate(available):
                if letter.upper() == char:
                    indices.append(tile_index)
                    available.pop(idx)
                    break
            else:
                return None
        return indices

    def centers_for_word(self, word: str) -> list[tuple[int, int]] | None:
        indices = self.indices_for_word(word)
        if indices is None:
            return None
        return [(self.tiles[i].center_x, self.tiles[i].center_y) for i in indices]


def _rects_from_detected_grid(
    detected: list[tuple[int, int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int, int]]:
    """Normalize OCR crops only when detected tile sizes vary a lot."""
    sorted_tiles = sorted(detected, key=lambda item: (item[0], item[1]))
    widths = [rect[4] - rect[2] for rect in sorted_tiles]
    heights = [rect[5] - rect[3] for rect in sorted_tiles]
    tile_width = int(np.median(widths))
    tile_height = int(np.median(heights))
    if not widths or tile_width <= 0 or tile_height <= 0:
        return sorted_tiles

    width_spread = (max(widths) - min(widths)) / tile_width
    height_spread = (max(heights) - min(heights)) / tile_height
    # Keep native boxes when detection already found consistent tiles — forced
    # median crops can clip letters (e.g. P) on dense boards.
    if width_spread <= 0.25 and height_spread <= 0.25:
        return sorted_tiles

    rects: list[tuple[int, int, int, int, int, int]] = []
    for row, col, x1, y1, x2, y2 in sorted_tiles:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        rects.append(
            (
                row,
                col,
                center_x - tile_width // 2,
                center_y - tile_height // 2,
                center_x + tile_width // 2,
                center_y + tile_height // 2,
            )
        )
    return rects


def _equal_split_in_bounds(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    rows: int,
    cols: int,
    occupied: frozenset[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
    tile_w = (x2 - x1) / cols
    tile_h = (y2 - y1) / rows
    cells = occupied if occupied is not None else frozenset(
        (row, col) for row in range(rows) for col in range(cols)
    )
    rects: list[tuple[int, int, int, int, int, int]] = []
    for row, col in sorted(cells):
        left = int(x1 + col * tile_w)
        top = int(y1 + row * tile_h)
        right = int(x1 + (col + 1) * tile_w)
        bottom = int(y1 + (row + 1) * tile_h)
        rects.append((row, col, left, top, right, bottom))
    return rects


def _tile_rects_in_area(
    area_width: int,
    area_height: int,
    grid: GridLayout,
) -> list[tuple[int, int, int, int, int, int]]:
    left, top, right, bottom = grid.tile_area.pixel_rect(area_width, area_height)
    return _equal_split_in_bounds(
        left,
        top,
        right,
        bottom,
        rows=grid.rows,
        cols=grid.cols,
        occupied=grid.occupied,
    )


def _detect_tile_rects_for_layout(
    game_image: np.ndarray,
    layout: GameLayout,
    *,
    allow_equal_split: bool = True,
) -> list[tuple[int, int, int, int, int, int]]:
    grid = layout.grid
    if grid.rows > 1:
        # The how-to-play screen shows a tiny demo grid inside a modal; the real
        # board only appears after START and fills the play tile area.
        if is_how_to_play_screen(game_image):
            return []

        search_areas = (
            grid.tile_area,
            NormalizedRect(x=0.0, y=0.0, width=1.0, height=1.0),
        )
        for tile_area in search_areas:
            detected = detect_grid_tiles(
                game_image,
                tile_area,
                rows=grid.rows,
                cols=grid.cols,
                occupied=grid.occupied,
            )
            if detected:
                return _rects_from_detected_grid(detected)

        if not allow_equal_split:
            return []

        # Last resort: equal-split inside the configured search band only.
        return _tile_rects_in_area(game_image.shape[1], game_image.shape[0], grid)

    tile_rects = detect_wood_tiles(game_image, grid.tile_area)
    return [(0, index, x1, y1, x2, y2) for index, (x1, y1, x2, y2) in enumerate(tile_rects)]


def resolve_word_hunt_layout(
    game_image: np.ndarray,
    preferred: GameLayout | None = None,
) -> GameLayout | None:
    """Pick the Word Hunt board shape that matches detected tiles."""
    candidates = [preferred] if preferred is not None else []
    candidates.extend(shape for shape in WORD_HUNT_SHAPES if shape not in candidates)

    for layout in candidates:
        if layout is None:
            continue
        detected = _detect_tile_rects_for_layout(
            game_image,
            layout,
            allow_equal_split=False,
        )
        if len(detected) == layout.grid.tile_count:
            logger.info("Detected Word Hunt shape: %s (%d tiles)", layout.grid.shape_name, len(detected))
            return layout

    # Fall back to tile-count matching using a loose box search on the play band.
    for tile_area in (
        (preferred or WORD_HUNT_SHAPES[0]).grid.tile_area,
        NormalizedRect(x=0.0, y=0.0, width=1.0, height=1.0),
    ):
        for expected in (16, 20, 21, 25):
            count = count_grid_tile_boxes(game_image, tile_area, expected_tiles=expected)
            matched = WORD_HUNT_SHAPES_BY_TILE_COUNT.get(count)
            if matched is not None:
                logger.info(
                    "Detected Word Hunt shape by tile count: %s (%d tiles)",
                    matched.grid.shape_name,
                    count,
                )
                return matched

    return preferred


def _detect_tile_rects(
    game_image: np.ndarray,
    layout: GameLayout,
) -> list[tuple[int, int, int, int, int, int]]:
    return _detect_tile_rects_for_layout(game_image, layout, allow_equal_split=True)


def read_tile_grid(
    game_image: np.ndarray,
    layout: GameLayout,
    frame_offset: tuple[int, int] = (0, 0),
    letter_override: str | None = None,
    tile_rects: list[tuple[int, int, int, int]] | None = None,
    letters: list[str] | None = None,
) -> TileGrid:
    if tile_rects is not None:
        indexed_rects = [(0, index, x1, y1, x2, y2) for index, (x1, y1, x2, y2) in enumerate(tile_rects)]
    else:
        indexed_rects = _detect_tile_rects(game_image, layout)

    tiles: list[Tile] = []
    override = list(letter_override.upper()) if letter_override else None
    provided = list(letters) if letters else None

    for index, (row, col, x1, y1, x2, y2) in enumerate(indexed_rects):
        tile_img = game_image[y1:y2, x1:x2]
        if provided and index < len(provided):
            letter = normalize_letter(provided[index])
        elif override and index < len(override):
            letter = normalize_letter(override[index])
        else:
            letter = normalize_letter(read_tile_letter(tile_img) or "?")
        center_x = frame_offset[0] + (x1 + x2) // 2
        center_y = frame_offset[1] + (y1 + y2) // 2
        tiles.append(
            Tile(
                row=row,
                col=col,
                letter=letter,
                center_x=center_x,
                center_y=center_y,
                rect=(x1, y1, x2, y2),
            )
        )

    return TileGrid(layout=layout.grid, tiles=tiles, frame_offset=frame_offset)


def annotate_grid_debug(game_image: np.ndarray, grid: TileGrid) -> np.ndarray:
    debug = game_image.copy()
    for tile in grid.tiles:
        x1, y1, x2, y2 = tile.rect
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_x = x1 + max(4, (x2 - x1) // 2 - 8)
        label_y = max(y1 + 20, (y1 + y2) // 2 + 6)
        cv2.putText(
            debug,
            tile.letter,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return debug
