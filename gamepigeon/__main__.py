#!/usr/bin/env python3
"""CLI entry point for the GamePigeon bot."""

from __future__ import annotations

import argparse
import logging

from gamepigeon.calibrate import calibrate as run_calibrate
from gamepigeon.games.anagrams import AnagramsBot
from gamepigeon.games.word_hunt import WordHuntBot
from gamepigeon.picker_capture import PickerCaptureSession, resolve_capture_target
from gamepigeon.shutdown import install_handler
from gamepigeon.window import list_on_screen_windows


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_window_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--picker",
        action="store_true",
        help="Use the macOS system window picker (default: auto-find iPhone Mirroring)",
    )
    parser.add_argument("--picker-timeout", type=float, default=120.0, help="Seconds to wait in the picker")


def _use_picker(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "picker", False))


def cmd_list_windows(_: argparse.Namespace) -> int:
    for window in list_on_screen_windows():
        x, y, w, h = window.bounds
        print(f"[{window.window_id}] {window.owner!r} / {window.title!r} ({w:.0f}x{h:.0f} @ {x:.0f},{y:.0f})")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    install_handler()

    try:
        target = resolve_capture_target(use_picker=_use_picker(args), picker_timeout=args.picker_timeout)
    except Exception as exc:
        logging.error("%s", exc)
        return 1

    logging.info(
        "Capturing window: %s (%s)",
        target.window.title or "(untitled)",
        target.window.owner,
    )

    capture_session = PickerCaptureSession(target)

    if args.game == "anagrams":
        bot = AnagramsBot(
            click_delay=args.click_delay if args.click_delay is not None else 0.08,
            word_delay=args.word_delay if args.word_delay is not None else 0.05,
            debug=args.debug,
            manual_letters=args.letters,
        )
    elif args.game == "word_hunt":
        bot = WordHuntBot(
            drag_delay=args.click_delay if args.click_delay is not None else 0.01,
            word_delay=args.word_delay if args.word_delay is not None else 0.01,
            debug=args.debug,
            manual_letters=args.letters,
            board_shape=args.board,
        )
    else:
        logging.error("Unknown game: %s", args.game)
        return 1

    bot.play(target, capture_session)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    try:
        target = resolve_capture_target(use_picker=_use_picker(args), picker_timeout=args.picker_timeout)
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
        return run_calibrate(args.game, args.output, target, session, screen=args.screen)
    finally:
        session.stop()


def cmd_calibrate_start(args: argparse.Namespace) -> int:
    args.screen = "start"
    if args.output == "/tmp/gamepigeon_calibrate.png":
        args.output = "/tmp/gamepigeon_calibrate-start.png"
    return cmd_calibrate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play GamePigeon games via iPhone Mirroring on macOS")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list-windows", help="List on-screen windows")
    list_cmd.set_defaults(func=cmd_list_windows)

    play_cmd = sub.add_parser("play", help="Pick a window and play a game")
    play_cmd.add_argument("--game", choices=["anagrams", "word_hunt"], default="anagrams")
    play_cmd.add_argument("--click-delay", type=float, default=None, help="Delay between tile clicks/drag steps")
    play_cmd.add_argument("--word-delay", type=float, default=None, help="Delay after submitting a word")
    play_cmd.add_argument(
        "--board",
        choices=["4x4", "5x5", "cross", "donut"],
        default=None,
        help="Word Hunt board shape (default: auto-detect). cross = 5x5 with edge middles removed",
    )
    play_cmd.add_argument(
        "--letters",
        help="Override OCR with known letters (e.g. ECTOEN) for testing",
    )
    play_cmd.add_argument("--debug", action="store_true", help="Save debug screenshots to /tmp")
    play_cmd.add_argument("-v", "--verbose", action="store_true")
    _add_window_flags(play_cmd)
    play_cmd.set_defaults(func=cmd_play)

    calibrate_cmd = sub.add_parser("calibrate", help="Capture iPhone Mirroring and draw tile/button regions (play screen)")
    calibrate_cmd.add_argument("--game", choices=["anagrams", "word_hunt"], default="anagrams")
    calibrate_cmd.add_argument("--output", default="/tmp/gamepigeon_calibrate.png")
    calibrate_cmd.add_argument(
        "--screen",
        choices=["auto", "start", "play"],
        default="play",
        help="Which screen to expect (default: active play screen)",
    )
    calibrate_cmd.add_argument("-v", "--verbose", action="store_true")
    _add_window_flags(calibrate_cmd)
    calibrate_cmd.set_defaults(func=cmd_calibrate)

    calibrate_start_cmd = sub.add_parser(
        "calibrate-start",
        help="Capture iPhone Mirroring and calibrate on the how-to-play / START screen",
    )
    calibrate_start_cmd.add_argument("--game", choices=["anagrams", "word_hunt"], default="anagrams")
    calibrate_start_cmd.add_argument("--output", default="/tmp/gamepigeon_calibrate-start.png")
    calibrate_start_cmd.add_argument("-v", "--verbose", action="store_true")
    _add_window_flags(calibrate_start_cmd)
    calibrate_start_cmd.set_defaults(func=cmd_calibrate_start)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "verbose"):
        configure_logging(args.verbose)
    else:
        configure_logging(False)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
