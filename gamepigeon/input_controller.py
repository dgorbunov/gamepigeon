"""Mouse click simulation via macOS Quartz events."""

from __future__ import annotations

import logging
import time

from Quartz import (
    CGEventCreateMouseEvent,
    CGEventPost,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseUp,
    kCGEventMouseMoved,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
)

from gamepigeon.capture import Frame
from gamepigeon.window import WindowInfo

logger = logging.getLogger(__name__)


def _screen_point(window: WindowInfo, local_x: float, local_y: float, scale: float) -> tuple[float, float]:
    wx, wy, _, _ = window.bounds
    return wx + local_x / scale, wy + local_y / scale


def _move_pointer(screen_x: float, screen_y: float) -> None:
    move = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (screen_x, screen_y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, move)


def click(
    window: WindowInfo,
    local_x: float,
    local_y: float,
    scale: float = 1.0,
    delay: float = 0.05,
) -> None:
    screen_x, screen_y = _screen_point(window, local_x, local_y, scale)
    _move_pointer(screen_x, screen_y)
    time.sleep(0.005)
    down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (screen_x, screen_y), kCGMouseButtonLeft)
    up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (screen_x, screen_y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, down)
    time.sleep(delay)
    CGEventPost(kCGHIDEventTap, up)
    logger.debug("Clicked frame=(%.0f, %.0f) screen=(%.0f, %.0f)", local_x, local_y, screen_x, screen_y)


def click_frame_point(frame: Frame, x: int, y: int, delay: float = 0.05) -> None:
    click(frame.window, x, y, scale=frame.scale, delay=delay)


def _interpolate_points(
    points: list[tuple[float, float]],
    *,
    step_px: float = 28.0,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points

    dense: list[tuple[float, float]] = [points[0]]
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= step_px:
            dense.append(end)
            continue
        steps = max(1, int(distance / step_px))
        for step in range(1, steps + 1):
            t = step / steps
            dense.append((start[0] + dx * t, start[1] + dy * t))
    return dense


def drag(
    window: WindowInfo,
    points: list[tuple[float, float]],
    scale: float = 1.0,
    step_delay: float = 0.01,
) -> None:
    """Press and hold the mouse button while moving through each point."""
    if not points:
        return

    path = _interpolate_points(points)
    first_x, first_y = _screen_point(window, path[0][0], path[0][1], scale)
    move = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (first_x, first_y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, move)

    down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (first_x, first_y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, down)

    for local_x, local_y in path[1:]:
        screen_x, screen_y = _screen_point(window, local_x, local_y, scale)
        dragged = CGEventCreateMouseEvent(
            None, kCGEventLeftMouseDragged, (screen_x, screen_y), kCGMouseButtonLeft
        )
        CGEventPost(kCGHIDEventTap, dragged)
        if step_delay:
            time.sleep(step_delay)

    last_x, last_y = _screen_point(window, path[-1][0], path[-1][1], scale)
    up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (last_x, last_y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, up)
    logger.debug(
        "Dragged through %d points (%d steps) ending screen=(%.0f, %.0f)",
        len(points),
        len(path),
        last_x,
        last_y,
    )
