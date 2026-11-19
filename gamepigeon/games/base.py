"""Base class for GamePigeon game bots."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field

import numpy as np

from gamepigeon.capture import Frame, capture_window, crop_game_content
from gamepigeon.config import GameLayout
from gamepigeon.input_controller import click_frame_point
from gamepigeon.picker_capture import CaptureTarget, PickerCaptureSession
from gamepigeon.vision.buttons import find_button
from gamepigeon.vision.tile_detect import detect_wood_tiles
from gamepigeon.vision.tiles import TileGrid, read_tile_grid

logger = logging.getLogger(__name__)


@dataclass
class GameSession:
    layout: GameLayout
    target: CaptureTarget
    capture_session: PickerCaptureSession
    submitted_words: set[str] = field(default_factory=set)
    started_at: float | None = None


class GameBot(abc.ABC):
    def __init__(self, layout: GameLayout) -> None:
        self.layout = layout
        self.session: GameSession | None = None

    def capture(self) -> Frame | None:
        if not self.session:
            return None

        result = self.session.capture_session.capture_bgr()
        if result is not None:
            image, scale = result
            return Frame(image=image, window=self.session.target.window, scale=scale)

        return capture_window(self.session.target.window)

    def read_grid(self, frame: Frame) -> tuple[TileGrid, np.ndarray, tuple[int, int, int, int]]:
        game_image, offset_rect = crop_game_content(frame)
        grid = read_tile_grid(
            game_image,
            self.layout,
            frame_offset=(offset_rect[0], offset_rect[1]),
        )
        return grid, game_image, offset_rect

    def tile_rects(self, game_image: np.ndarray, grid: TileGrid | None = None) -> list[tuple[int, int, int, int]]:
        if grid is not None:
            return [tile.rect for tile in grid.tiles]
        return detect_wood_tiles(game_image, self.layout.grid.tile_area)

    def click_button(
        self,
        frame: Frame,
        button_name: str,
        game_image: np.ndarray,
        offset_rect: tuple[int, int, int, int],
        tile_rects: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        hit = find_button(game_image, button_name, tile_rects)
        if hit is None:
            raise RuntimeError(f"Could not find {button_name.upper()} button on screen")

        click_frame_point(
            frame,
            offset_rect[0] + hit.center[0],
            offset_rect[1] + hit.center[1],
        )

    @abc.abstractmethod
    def play(self, target: CaptureTarget, capture_session: PickerCaptureSession) -> None:
        raise NotImplementedError
