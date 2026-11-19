"""Calibration helper to tune layout regions."""

from __future__ import annotations

import argparse
import logging

import cv2

from gamepigeon.capture import crop_game_content, save_debug_image
from gamepigeon.config import ANAGRAMS_LAYOUT, WORD_HUNT_LAYOUT
from gamepigeon.picker_capture import CaptureTarget, PickerCaptureSession, resolve_capture_target
from gamepigeon.vision.buttons import (
    _find_how_to_play_modal,
    detect_buttons,
    detect_start_button,
    is_how_to_play_screen,
)
from gamepigeon.vision.tiles import annotate_grid_debug, read_tile_grid

logger = logging.getLogger(__name__)


def _draw_tile_area_outline(debug: cv2.Mat, layout) -> None:
    height, width = debug.shape[:2]
    left, top, right, bottom = layout.grid.tile_area.pixel_rect(width, height)
    cv2.rectangle(debug, (left, top), (right, bottom), (0, 255, 255), 2)
    cv2.putText(
        debug,
        "play grid area",
        (left + 4, max(top + 24, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )


def calibrate(
    game: str,
    output: str,
    target: CaptureTarget,
    session: PickerCaptureSession,
    *,
    screen: str = "auto",
) -> int:
    from gamepigeon.capture import Frame

    captured = session.capture_bgr()
    if captured is None:
        logging.error("Failed to capture selected window")
        return 1

    image, scale = captured
    frame = Frame(image=image, window=target.window, scale=scale)
    layout = ANAGRAMS_LAYOUT if game == "anagrams" else WORD_HUNT_LAYOUT
    game_image, offset_rect = crop_game_content(frame)
    on_start_screen = screen == "start" or is_how_to_play_screen(game_image)

    debug = game_image.copy()
    modal = _find_how_to_play_modal(game_image)
    if modal:
        x1, y1, x2, y2 = modal
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            debug,
            "how-to-play modal",
            (x1 + 4, max(y1 + 24, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    if layout.grid.rows > 1:
        _draw_tile_area_outline(debug, layout)

    buttons: dict[str, object] = {}
    if on_start_screen:
        start = detect_start_button(game_image)
        if start:
            buttons["start"] = start
        grid = read_tile_grid(
            game_image,
            layout,
            frame_offset=(offset_rect[0], offset_rect[1]),
        )
    else:
        grid = read_tile_grid(
            game_image,
            layout,
            frame_offset=(offset_rect[0], offset_rect[1]),
        )
        tile_rects = [tile.rect for tile in grid.tiles]
        buttons = detect_buttons(
            game_image,
            tile_rects or None,
            has_enter_button=(game == "anagrams"),
        )
        debug = annotate_grid_debug(debug, grid)

    for name, hit in buttons.items():
        x1, y1, x2, y2 = hit.rect
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            debug,
            name,
            (x1 + 4, max(y1 + 24, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    save_debug_image(debug, output)
    print(f"Screen mode: {'start' if on_start_screen else screen}")
    if on_start_screen:
        print("Start screen: skipping play grid detection (demo board is not the live grid)")
    else:
        print(f"Detected letters: {''.join(grid.letters)}")
        print(f"Grid tiles: {len(grid.tiles)}")
    print(f"Detected buttons: {', '.join(buttons) or 'none'}")
    for name, hit in buttons.items():
        print(f"  {name}: center={hit.center} rect={hit.rect}")
    print(f"Saved calibration image: {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate tile/button regions")
    parser.add_argument("--game", choices=["anagrams", "word_hunt"], default="anagrams")
    parser.add_argument("--output", default="/tmp/gamepigeon_calibrate.png")
    parser.add_argument(
        "--screen",
        choices=["auto", "start", "play"],
        default="play",
        help="Which screen to expect (default: active play screen)",
    )
    parser.add_argument(
        "--picker",
        action="store_true",
        help="Use the macOS system window picker (default: auto-find iPhone Mirroring)",
    )
    parser.add_argument("--picker-timeout", type=float, default=120.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        target = resolve_capture_target(use_picker=args.picker, picker_timeout=args.picker_timeout)
    except Exception as exc:
        logging.error("%s", exc)
        return 1

    logging.info(
        "Capturing window: %s (%s)",
        target.window.title or "(untitled)",
        target.window.owner,
    )

    session = PickerCaptureSession(target)
    session.start()
    try:
        return calibrate(args.game, args.output, target, session, screen=args.screen)
    finally:
        session.stop()


if __name__ == "__main__":
    raise SystemExit(main())
