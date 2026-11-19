"""Word Hunt game bot."""

from __future__ import annotations

import logging
import time

from gamepigeon.capture import crop_game_content, save_debug_image
from gamepigeon.config import (
    WORD_HUNT_LAYOUT,
    WORD_HUNT_SHAPES_BY_NAME,
    WORD_HUNT_SHAPES_BY_TILE_COUNT,
)
from gamepigeon.games.base import GameBot, GameSession
from gamepigeon.input_controller import drag
from gamepigeon.picker_capture import CaptureTarget, PickerCaptureSession, refresh_window_bounds
from gamepigeon.shutdown import shutdown_requested
from gamepigeon.window import focus_window
from gamepigeon.vision.buttons import (
    detect_start_button,
    is_ready_to_play,
    is_start_screen,
)
from gamepigeon.vision.tiles import TileGrid, annotate_grid_debug, read_tile_grid, resolve_word_hunt_layout
from gamepigeon.words.anagrams import find_words_in_grid
from gamepigeon.words.dictionary import load_prefix_set, load_scrabble_words

logger = logging.getLogger(__name__)


class WordHuntBot(GameBot):
    def __init__(
        self,
        *,
        drag_delay: float = 0.01,
        word_delay: float = 0.01,
        debug: bool = False,
        manual_letters: str | None = None,
        board_shape: str | None = None,
    ) -> None:
        if board_shape:
            layout = WORD_HUNT_SHAPES_BY_NAME.get(board_shape.lower())
            if layout is None:
                raise ValueError(
                    f"Unknown Word Hunt board shape {board_shape!r}; "
                    f"expected one of {sorted(WORD_HUNT_SHAPES_BY_NAME)}"
                )
        else:
            layout = WORD_HUNT_LAYOUT
        super().__init__(layout)
        self.drag_delay = drag_delay
        self.word_delay = word_delay
        self.debug = debug
        self.manual_letters = manual_letters.upper() if manual_letters else None
        self.fixed_board_shape = board_shape.lower() if board_shape else None
        self.dictionary = list(load_scrabble_words())
        self._prefixes = load_prefix_set()
        self._focused = False
        self._grid_locked = False
        self._grid: TileGrid | None = None
        self._letter_matrix: list[list[str]] | None = None

    def _reset_grid_cache(self) -> None:
        self._grid_locked = False
        self._grid = None
        self._letter_matrix = None

    def _refresh_target(self) -> None:
        assert self.session is not None
        window = refresh_window_bounds(self.session.target.window)
        self.session.target = CaptureTarget(window=window, content_filter=self.session.target.content_filter)

    def _focus_mirror_once(self) -> None:
        if self._focused:
            return
        assert self.session is not None
        focus_window(self.session.target.window)
        time.sleep(0.15)
        self._refresh_target()
        self._focused = True

    def _capture_frame(self):
        assert self.session is not None
        result = self.session.capture_session.capture_bgr()
        if result is None:
            return None
        image, scale = result
        from gamepigeon.capture import Frame

        return Frame(image=image, window=self.session.target.window, scale=scale)

    def grid_to_letter_matrix(self, grid: TileGrid) -> list[list[str]]:
        rows, cols = self.layout.grid.rows, self.layout.grid.cols
        matrix = [["" for _ in range(cols)] for _ in range(rows)]
        for tile in grid.tiles:
            matrix[tile.row][tile.col] = tile.letter.upper()
        return matrix

    def _apply_layout(self, layout) -> None:
        self.layout = layout
        if self.session is not None:
            self.session.layout = layout

    def _lock_grid(self, frame) -> TileGrid | None:
        if self._grid_locked and self._grid is not None:
            return self._grid

        game_image, offset_rect = crop_game_content(frame)
        if self.fixed_board_shape is None:
            resolved = resolve_word_hunt_layout(game_image, preferred=self.layout)
            if resolved is not None:
                self._apply_layout(resolved)

        override = self.manual_letters
        if override and len(override) != self.layout.grid.tile_count:
            matched = WORD_HUNT_SHAPES_BY_TILE_COUNT.get(len(override))
            if matched is not None and self.fixed_board_shape is None:
                self._apply_layout(matched)
            elif matched is None:
                logger.warning(
                    "Letter override length %d does not match a known board (%s)",
                    len(override),
                    sorted(WORD_HUNT_SHAPES_BY_TILE_COUNT),
                )

        grid = read_tile_grid(
            game_image,
            self.layout,
            frame_offset=(offset_rect[0], offset_rect[1]),
            letter_override=override,
        )

        matrix = self.grid_to_letter_matrix(grid)
        readable = [letter for row in matrix for letter in row if letter.isalpha() and letter != "?"]
        expected_tiles = self.layout.grid.tile_count
        if len(readable) < self.layout.min_word_length or len(grid.tiles) < expected_tiles:
            return None

        self._grid = grid
        self._letter_matrix = matrix
        self._grid_locked = True
        shape = self.layout.grid.shape_name
        logger.info("Locked %s grid (%d tiles):\n%s", shape, expected_tiles, self._format_matrix(matrix))
        return grid

    def _format_matrix(self, matrix: list[list[str]]) -> str:
        lines: list[str] = []
        for row in matrix:
            cells = [(cell if cell else ".") for cell in row]
            lines.append("".join(cells))
        return "\n".join(lines)

    def prepare_to_play(self, timeout: float = 300.0) -> None:
        deadline = time.time() + timeout
        logger.info("Waiting for Word Hunt — start screen or active play")

        while time.time() < deadline and not shutdown_requested():
            frame = self.capture()
            if frame is None:
                time.sleep(0.1)
                continue

            game_image, _offset = crop_game_content(frame)
            grid = read_tile_grid(game_image, self.layout)
            tile_rects = [tile.rect for tile in grid.tiles]

            if is_ready_to_play(game_image, tile_rects, has_enter_button=False):
                logger.info("Play screen ready")
                return

            if is_start_screen(game_image, tile_rects):
                logger.info("START screen detected")
                self.press_start()
                return

            time.sleep(0.1)

        if shutdown_requested():
            return

        raise RuntimeError("Timed out waiting for Word Hunt start or play screen")

    def wait_for_play_screen(self, timeout: float = 15.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.capture()
            if frame is None:
                time.sleep(0.05)
                continue

            game_image, _offset = crop_game_content(frame)
            grid = read_tile_grid(game_image, self.layout)
            if is_ready_to_play(game_image, [tile.rect for tile in grid.tiles], has_enter_button=False):
                logger.info("Play screen ready")
                return True
            time.sleep(0.05)
        return False

    def press_start(self) -> None:
        self._focus_mirror_once()
        for attempt in range(3):
            frame = self._capture_frame()
            if frame is None:
                raise RuntimeError("Could not capture window to press START")

            game_image, offset_rect = crop_game_content(frame)
            if not is_start_screen(game_image, None):
                return

            hit = detect_start_button(game_image)
            if hit is None:
                raise RuntimeError("Could not find START button on screen")

            from gamepigeon.input_controller import click_frame_point

            click_frame_point(
                frame,
                offset_rect[0] + hit.center[0],
                offset_rect[1] + hit.center[1],
                delay=0.03,
            )
            logger.info("Clicked START at frame (%d, %d)", hit.center[0], hit.center[1])

            if self.wait_for_play_screen(timeout=4.0):
                return

            logger.warning("START click attempt %d did not reach play screen", attempt + 1)

        raise RuntimeError("Failed to start the game after clicking START")

    def submit_word_path(self, frame, grid: TileGrid, path: list[tuple[int, int]]) -> bool:
        points: list[tuple[float, float]] = []
        for row, col in path:
            tile = grid.tile_at(row, col)
            if tile is None:
                logger.warning("Missing tile at (%d, %d)", row, col)
                return False
            points.append((float(tile.center_x), float(tile.center_y)))

        drag(frame.window, points, scale=frame.scale, step_delay=self.drag_delay)
        if self.word_delay:
            time.sleep(self.word_delay)
        return True

    def play_round(self) -> int:
        assert self.session is not None
        self._reset_grid_cache()
        deadline = time.time() + self.layout.round_duration_seconds
        submitted = self.session.submitted_words
        words_played = 0
        no_grid_rounds = 0

        while time.time() < deadline and not shutdown_requested():
            frame = self._capture_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            grid = self._lock_grid(frame)
            if grid is None:
                no_grid_rounds += 1
                if no_grid_rounds % 20 == 1:
                    logger.info("Waiting for grid tiles")
                time.sleep(0.05)
                continue
            break
        else:
            return words_played

        if self.debug:
            game_image, offset_rect = crop_game_content(frame)
            debug_img = annotate_grid_debug(game_image, grid)
            save_debug_image(debug_img, f"/tmp/gamepigeon_word_hunt_{int(time.time())}.png")

        matrix = self._letter_matrix or self.grid_to_letter_matrix(grid)
        candidates = find_words_in_grid(
            matrix,
            self.dictionary,
            min_length=self.layout.min_word_length,
            exclude=submitted,
            prefixes=self._prefixes,
        )
        if not candidates:
            logger.info("No words found for grid")
            return words_played

        logger.info("Found %d words", len(candidates))

        for word, path in candidates:
            if time.time() >= deadline or shutdown_requested():
                break
            if word in submitted:
                continue

            if self.submit_word_path(frame, grid, path):
                submitted.add(word)
                words_played += 1
                logger.info("Submitted: %s", word.upper())
            else:
                submitted.add(word)

        return words_played

    def play(self, target: CaptureTarget, capture_session: PickerCaptureSession) -> None:
        self.session = GameSession(
            layout=self.layout,
            target=target,
            capture_session=capture_session,
        )
        capture_session.start()
        self._focus_mirror_once()
        logger.info("Focused iPhone Mirroring — press Ctrl+C to stop at any time")

        self.prepare_to_play()
        self.session.started_at = time.time()
        logger.info(
            "Playing Word Hunt on %s (%dx%d, %d tiles)",
            self.layout.grid.shape_name,
            self.layout.grid.rows,
            self.layout.grid.cols,
            self.layout.grid.tile_count,
        )
        count = self.play_round()
        capture_session.stop()
        if shutdown_requested():
            logger.info("Stopped by user. Submitted %d words: %s", count, sorted(self.session.submitted_words))
        else:
            logger.info("Done. Submitted %d words: %s", count, sorted(self.session.submitted_words))
