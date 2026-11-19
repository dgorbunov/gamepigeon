"""Shared configuration and layout definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedRect:
    """Rectangle expressed as fractions of the game content area (0-1)."""

    x: float
    y: float
    width: float
    height: float

    def pixel_rect(self, area_width: int, area_height: int) -> tuple[int, int, int, int]:
        left = int(self.x * area_width)
        top = int(self.y * area_height)
        right = int((self.x + self.width) * area_width)
        bottom = int((self.y + self.height) * area_height)
        return left, top, right, bottom


def _all_cells(rows: int, cols: int) -> frozenset[tuple[int, int]]:
    return frozenset((row, col) for row in range(rows) for col in range(cols))


# 5x5 with the middle cell of each edge removed (GamePigeon "Cross").
WORD_HUNT_CROSS_OCCUPIED = frozenset(
    (row, col)
    for row in range(5)
    for col in range(5)
    if (row, col) not in {(0, 2), (2, 0), (2, 4), (4, 2)}
)

# 5x5 donut: corners missing and a one-cell hole in the center.
WORD_HUNT_DONUT_OCCUPIED = frozenset(
    {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 1),
        (2, 3),
        (2, 4),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 1),
        (4, 2),
        (4, 3),
    }
)


@dataclass(frozen=True)
class GridLayout:
    """Tile grid layout within the game content area."""

    rows: int
    cols: int
    tile_area: NormalizedRect
    reading_order: str = "row-major"
    # When set, only these (row, col) cells exist (irregular Word Hunt boards).
    occupied: frozenset[tuple[int, int]] | None = None
    shape_name: str = "square"

    @property
    def occupied_cells(self) -> frozenset[tuple[int, int]]:
        if self.occupied is not None:
            return self.occupied
        return _all_cells(self.rows, self.cols)

    @property
    def tile_count(self) -> int:
        return len(self.occupied_cells)

    def is_occupied(self, row: int, col: int) -> bool:
        return (row, col) in self.occupied_cells


@dataclass(frozen=True)
class GameLayout:
    name: str
    grid: GridLayout
    min_word_length: int = 3
    round_duration_seconds: float = 60.0
    supported_tile_counts: tuple[int, ...] = (6, 7)

    def with_grid(self, grid: GridLayout) -> GameLayout:
        return GameLayout(
            name=self.name,
            grid=grid,
            min_word_length=self.min_word_length,
            round_duration_seconds=self.round_duration_seconds,
            supported_tile_counts=self.supported_tile_counts,
        )

    def with_tile_count(self, tile_count: int) -> GameLayout:
        if tile_count not in self.supported_tile_counts:
            raise ValueError(f"Unsupported tile count: {tile_count}")
        return GameLayout(
            name=self.name,
            grid=GridLayout(
                rows=1,
                cols=tile_count,
                tile_area=self.grid.tile_area,
                reading_order=self.grid.reading_order,
            ),
            min_word_length=self.min_word_length,
            round_duration_seconds=self.round_duration_seconds,
            supported_tile_counts=self.supported_tile_counts,
        )


# Loose search band for rack tiles; exact bounds are detected at runtime.
ANAGRAMS_TILE_AREA = NormalizedRect(x=0.0, y=0.50, width=1.0, height=0.48)

ANAGRAMS_LAYOUT = GameLayout(
    name="anagrams",
    grid=GridLayout(rows=1, cols=6, tile_area=ANAGRAMS_TILE_AREA),
    min_word_length=3,
    round_duration_seconds=60.0,
    supported_tile_counts=(6, 7),
)

WORD_HUNT_TILE_AREA = NormalizedRect(x=0.12, y=0.49, width=0.76, height=0.39)

WORD_HUNT_LAYOUT_4X4 = GameLayout(
    name="word_hunt",
    grid=GridLayout(
        rows=4,
        cols=4,
        tile_area=WORD_HUNT_TILE_AREA,
        shape_name="4x4",
    ),
    min_word_length=3,
    round_duration_seconds=80.0,
    supported_tile_counts=(16, 20, 21, 25),
)

WORD_HUNT_LAYOUT_5X5 = WORD_HUNT_LAYOUT_4X4.with_grid(
    GridLayout(
        rows=5,
        cols=5,
        tile_area=WORD_HUNT_TILE_AREA,
        shape_name="5x5",
    )
)

WORD_HUNT_LAYOUT_CROSS = WORD_HUNT_LAYOUT_4X4.with_grid(
    GridLayout(
        rows=5,
        cols=5,
        tile_area=WORD_HUNT_TILE_AREA,
        occupied=WORD_HUNT_CROSS_OCCUPIED,
        shape_name="cross",
    )
)

WORD_HUNT_LAYOUT_DONUT = WORD_HUNT_LAYOUT_4X4.with_grid(
    GridLayout(
        rows=5,
        cols=5,
        tile_area=WORD_HUNT_TILE_AREA,
        occupied=WORD_HUNT_DONUT_OCCUPIED,
        shape_name="donut",
    )
)

# Default / fallback is classic 4x4; play auto-detects among these.
WORD_HUNT_LAYOUT = WORD_HUNT_LAYOUT_4X4

WORD_HUNT_SHAPES: tuple[GameLayout, ...] = (
    WORD_HUNT_LAYOUT_4X4,
    WORD_HUNT_LAYOUT_5X5,
    WORD_HUNT_LAYOUT_CROSS,
    WORD_HUNT_LAYOUT_DONUT,
)

WORD_HUNT_SHAPES_BY_NAME: dict[str, GameLayout] = {
    layout.grid.shape_name: layout for layout in WORD_HUNT_SHAPES
}

WORD_HUNT_SHAPES_BY_TILE_COUNT: dict[int, GameLayout] = {
    layout.grid.tile_count: layout for layout in WORD_HUNT_SHAPES
}

DEFAULT_WINDOW_TITLES: tuple[str, ...] = (
    "iPhone Mirroring",
    "iPhone",
    "Screen Mirroring",
)
