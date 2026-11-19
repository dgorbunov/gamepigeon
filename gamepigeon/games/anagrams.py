"""Anagrams game bot."""

from __future__ import annotations

import logging
import time

from gamepigeon.capture import crop_game_content, save_debug_image
from gamepigeon.config import ANAGRAMS_LAYOUT
from gamepigeon.games.base import GameBot, GameSession
from gamepigeon.input_controller import click_frame_point
from gamepigeon.picker_capture import CaptureTarget, PickerCaptureSession, refresh_window_bounds
from gamepigeon.shutdown import shutdown_requested
from gamepigeon.window import focus_window
from gamepigeon.vision.buttons import (
    detect_start_button,
    find_button,
    is_ready_to_play,
    is_start_screen,
)
from gamepigeon.vision.letters import read_tile_letter
from gamepigeon.vision.tile_detect import detect_wood_tiles
from gamepigeon.vision.tiles import TileGrid, annotate_grid_debug, read_tile_grid
from gamepigeon.words.anagram_solver import AnagramSolver
from gamepigeon.words.anagrams import find_anagrams
from gamepigeon.words.dictionary import load_alphagram_index

logger = logging.getLogger(__name__)


class AnagramsBot(GameBot):
    def __init__(
        self,
        *,
        click_delay: float = 0.08,
        word_delay: float = 0.05,
        debug: bool = False,
        manual_letters: str | None = None,
    ) -> None:
        super().__init__(ANAGRAMS_LAYOUT)
        self.click_delay = click_delay
        self.word_delay = word_delay
        self.debug = debug
        self.manual_letters = manual_letters.upper() if manual_letters else None
        self.solver = AnagramSolver(load_alphagram_index())
        self.active_layout = ANAGRAMS_LAYOUT
        self._cached_enter_center: tuple[int, int] | None = None
        self._focused = False
        self._rack_tile_rects: list[tuple[int, int, int, int]] | None = None
        self._rack_letters: list[str] | None = None
        self._rack_locked = False

    def _reset_rack_cache(self) -> None:
        self._rack_tile_rects = None
        self._rack_letters = None
        self._rack_locked = False

    def _detect_tiles(self, frame, *, refresh: bool = False) -> tuple:
        game_image, offset_rect = crop_game_content(frame)
        if refresh or self._rack_tile_rects is None:
            tile_rects = detect_wood_tiles(game_image, self.active_layout.grid.tile_area)
            if len(tile_rects) in (6, 7):
                self._rack_tile_rects = tile_rects
                self.active_layout = ANAGRAMS_LAYOUT.with_tile_count(len(tile_rects))
        tile_rects = self._rack_tile_rects or []
        return game_image, offset_rect, tile_rects

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

    def _ocr_letters(
        self,
        game_image,
        tile_rects: list[tuple[int, int, int, int]],
    ) -> list[str]:
        if self.manual_letters:
            letters = list(self.manual_letters)
            return [letters[i] if i < len(letters) else "?" for i in range(len(tile_rects))]

        letters: list[str] = []
        for x1, y1, x2, y2 in tile_rects:
            tile_img = game_image[y1:y2, x1:x2]
            letter = read_tile_letter(tile_img) or "?"
            letters.append(letter.upper())
        return letters

    def _lock_rack(
        self,
        game_image,
        tile_rects: list[tuple[int, int, int, int]],
    ) -> list[str] | None:
        """Read the rack once per round."""
        if self._rack_locked and self._rack_letters is not None:
            return self._rack_letters

        letters = self._ocr_letters(game_image, tile_rects)
        readable = [letter for letter in letters if letter.isalpha() and letter != "?"]
        if len(readable) < self.active_layout.min_word_length:
            return None

        self._rack_letters = letters
        self._rack_locked = True
        logger.info("Rack letters: %s", "".join(letters))
        return letters

    def _build_grid(
        self,
        game_image,
        offset_rect: tuple[int, int, int, int],
        tile_rects: list[tuple[int, int, int, int]],
        letters: list[str],
    ) -> TileGrid:
        return read_tile_grid(
            game_image,
            self.active_layout,
            frame_offset=(offset_rect[0], offset_rect[1]),
            tile_rects=tile_rects,
            letters=letters,
        )

    def prepare_to_play(self, timeout: float = 300.0) -> None:
        """Wait for the start or play screen, clicking START only when needed."""
        deadline = time.time() + timeout
        logger.info("Waiting for Anagrams — start screen or active play")

        while time.time() < deadline and not shutdown_requested():
            frame = self.capture()
            if frame is None:
                time.sleep(0.1)
                continue

            game_image, _offset, tile_rects = self._detect_tiles(frame)

            if is_ready_to_play(game_image, tile_rects):
                logger.info("Play screen ready")
                return

            if is_start_screen(game_image, tile_rects):
                logger.info("START screen detected")
                self.press_start()
                return

            time.sleep(0.1)

        if shutdown_requested():
            return

        raise RuntimeError("Timed out waiting for Anagrams start or play screen")

    def wait_for_play_screen(self, timeout: float = 15.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.capture()
            if frame is None:
                time.sleep(0.05)
                continue

            game_image, _offset, tile_rects = self._detect_tiles(frame)
            if is_ready_to_play(game_image, tile_rects):
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

            x = offset_rect[0] + hit.center[0]
            y = offset_rect[1] + hit.center[1]
            click_frame_point(frame, x, y, delay=0.03)
            logger.info("Clicked START at frame (%d, %d)", x, y)

            if self.wait_for_play_screen(timeout=4.0):
                return

            logger.warning("START click attempt %d did not reach play screen", attempt + 1)

        raise RuntimeError("Failed to start the game after clicking START")

    def press_enter(
        self,
        frame,
        offset_rect: tuple[int, int, int, int],
        game_image=None,
        tile_rects: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        if self._cached_enter_center is None and game_image is not None:
            self._cache_enter_button(frame, game_image, offset_rect, tile_rects)
        if self._cached_enter_center is None:
            raise RuntimeError("Could not find ENTER button on screen")

        x = offset_rect[0] + self._cached_enter_center[0]
        y = offset_rect[1] + self._cached_enter_center[1]
        click_frame_point(frame, x, y, delay=0.02)
        logger.debug("Clicked ENTER at frame (%d, %d)", x, y)
        if self.word_delay:
            time.sleep(self.word_delay)

    def _cache_enter_button(self, frame, game_image, offset_rect, tile_rects) -> None:
        if self._cached_enter_center is not None:
            return
        hit = find_button(game_image, "enter", tile_rects)
        if hit is None:
            return
        if tile_rects:
            tile_top = min(rect[1] for rect in tile_rects)
            if hit.center[1] < game_image.shape[0] * 0.30:
                return
            if hit.center[1] > tile_top - 8:
                return
        self._cached_enter_center = hit.center
        logger.info("ENTER button at game (%d, %d)", hit.center[0], hit.center[1])

    def submit_word(
        self,
        word: str,
        frame,
        game_image,
        offset_rect: tuple[int, int, int, int],
        tile_rects: list[tuple[int, int, int, int]],
        letters: list[str],
        grid: TileGrid,
    ) -> bool:
        indices = grid.indices_for_word(word)
        if indices is None:
            logger.warning("Cannot map word to tiles: %s (letters=%s)", word, grid.letters)
            return False

        for index in indices:
            tile = grid.tiles[index]
            click_frame_point(frame, tile.center_x, tile.center_y, delay=self.click_delay)
            logger.debug("Clicked tile %s at frame (%d, %d)", tile.letter, tile.center_x, tile.center_y)

        self.press_enter(frame, offset_rect, game_image, tile_rects)

        time.sleep(0.08)
        logger.info("Submitted: %s", word.upper())
        return True

    def play_round(self) -> int:
        assert self.session is not None
        self._reset_rack_cache()
        deadline = time.time() + self.layout.round_duration_seconds
        submitted = self.session.submitted_words
        words_played = 0
        min_letters = self.active_layout.min_word_length
        no_tile_rounds = 0

        while time.time() < deadline and not shutdown_requested():
            frame = self._capture_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            game_image, offset_rect, tile_rects = self._detect_tiles(frame)
            if not tile_rects:
                no_tile_rounds += 1
                if no_tile_rounds % 20 == 1:
                    logger.info("Waiting for rack tiles")
                time.sleep(0.05)
                continue
            no_tile_rounds = 0

            self._cache_enter_button(frame, game_image, offset_rect, tile_rects)
            letters = self._lock_rack(game_image, tile_rects)
            if letters is None:
                time.sleep(0.05)
                continue
            break
        else:
            return words_played

        available_letters = [letter for letter in letters if letter.isalpha() and letter != "?"]
        if len(available_letters) < min_letters:
            return words_played

        candidates = find_anagrams(
            available_letters,
            exclude=submitted,
            solver=self.solver,
        )
        rack = "".join(available_letters)
        if not candidates:
            logger.info("No words found for %s", rack)
            return words_played

        logger.info("Found %d words for rack %s", len(candidates), rack)

        for word in candidates:
            if time.time() >= deadline or shutdown_requested():
                break

            frame = self._capture_frame()
            if frame is None:
                continue

            game_image, offset_rect, tile_rects = self._detect_tiles(frame)
            if not tile_rects:
                continue

            grid = self._build_grid(game_image, offset_rect, tile_rects, letters)

            if self.debug:
                debug_img = annotate_grid_debug(game_image, grid)
                save_debug_image(debug_img, f"/tmp/gamepigeon_anagrams_{int(time.time())}.png")

            if self.submit_word(word, frame, game_image, offset_rect, tile_rects, letters, grid):
                submitted.add(word)
                words_played += 1
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

        self._cached_enter_center = None
        self.session.started_at = time.time()
        logger.info("Playing with %d-letter rack", self.active_layout.grid.cols)
        count = self.play_round()
        capture_session.stop()
        if shutdown_requested():
            logger.info("Stopped by user. Submitted %d words: %s", count, sorted(self.session.submitted_words))
        else:
            logger.info("Done. Submitted %d words: %s", count, sorted(self.session.submitted_words))
