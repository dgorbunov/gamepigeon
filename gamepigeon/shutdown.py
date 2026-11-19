"""Cooperative shutdown so Ctrl+C can stop the bot mid-round."""

from __future__ import annotations

import signal

_shutdown_requested = False


def _handle_signal(_signum: int, _frame: object) -> None:
    global _shutdown_requested
    _shutdown_requested = True


def install_handler() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def shutdown_requested() -> bool:
    return _shutdown_requested


def clear() -> None:
    global _shutdown_requested
    _shutdown_requested = False
