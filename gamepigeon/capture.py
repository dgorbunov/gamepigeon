"""Capture screenshots from a specific macOS window."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import mss
import numpy as np
from AppKit import NSScreen
from PIL import Image
from Quartz import (
    CGDataProviderCopyData,
    CGImageGetBitsPerPixel,
    CGImageGetBytesPerRow,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CGWindowListCreateImage,
    CGRectNull,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowListOptionIncludingWindow,
)

from gamepigeon.window import WindowInfo

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray  # BGR, uint8
    window: WindowInfo
    scale: float = 1.0

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


def _screen_scale_for_window(window: WindowInfo) -> float:
    center_x = window.bounds[0] + window.bounds[2] / 2
    center_y = window.bounds[1] + window.bounds[3] / 2
    for screen in NSScreen.screens():
        frame = screen.frame()
        if (
            frame.origin.x <= center_x <= frame.origin.x + frame.size.width
            and frame.origin.y <= center_y <= frame.origin.y + frame.size.height
        ):
            return float(screen.backingScaleFactor())
    return float(NSScreen.mainScreen().backingScaleFactor())


def _cg_image_to_bgr(cg_image) -> np.ndarray:
    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)
    bytes_per_row = CGImageGetBytesPerRow(cg_image)
    provider = CGImageGetDataProvider(cg_image)
    data = CGDataProviderCopyData(provider)
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = buffer.reshape((height, bytes_per_row // 4, 4))[:, :width, :3]
    return np.ascontiguousarray(image[:, :, ::-1])


def _capture_with_quartz(window: WindowInfo) -> np.ndarray | None:
    try:
        cg_image = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            window.window_id,
            kCGWindowImageBoundsIgnoreFraming,
        )
        if cg_image is None:
            return None

        bpp = CGImageGetBitsPerPixel(cg_image)
        if bpp != 32:
            logger.warning("Unexpected bits per pixel: %s", bpp)
        return _cg_image_to_bgr(cg_image)
    except Exception as exc:
        logger.debug("Quartz capture failed: %s", exc)
        return None


def _capture_with_mss(window: WindowInfo, scale: float) -> np.ndarray:
    x, y, width, height = window.bounds
    monitor = {
        "left": int(round(x * scale)),
        "top": int(round(y * scale)),
        "width": int(round(width * scale)),
        "height": int(round(height * scale)),
    }
    with mss.mss() as sct:
        shot = sct.grab(monitor)
    rgb = np.array(Image.frombytes("RGB", shot.size, shot.rgb))
    return rgb[:, :, ::-1].copy()


def capture_window(window: WindowInfo) -> Frame | None:
    scale = _screen_scale_for_window(window)
    image = _capture_with_quartz(window)

    if image is None:
        logger.warning(
            "Quartz window capture failed for %s. Falling back to screen region capture. "
            "Grant Screen Recording permission to Terminal/Cursor in System Settings if clicks misalign.",
            window.window_id,
        )
        try:
            image = _capture_with_mss(window, scale)
        except Exception as exc:
            logger.error("Screen capture fallback failed: %s", exc)
            return None

    measured_scale = image.shape[1] / window.bounds[2] if window.bounds[2] else scale
    if abs(measured_scale - scale) > 0.25:
        logger.debug(
            "Adjusting capture scale from %.2f to %.2f based on image size",
            scale,
            measured_scale,
        )
        scale = measured_scale

    return Frame(image=image, window=window, scale=scale)


def crop_region(frame: Frame, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    left = max(0, left)
    top = max(0, top)
    right = min(frame.width, right)
    bottom = min(frame.height, bottom)
    return frame.image[top:bottom, left:right].copy()


def save_debug_image(image: np.ndarray, path: str) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, rgb)
    logger.info("Saved debug image: %s", path)


def _purple_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([120, 20, 120], dtype=np.uint8), np.array([165, 180, 255], dtype=np.uint8))


def _trim_status_bar(mask: np.ndarray, left: int, top: int, right: int, bottom: int) -> int:
    """Skip the mirrored iOS status bar above the purple playfield."""
    region = mask[top:bottom, left:right]
    if region.size == 0:
        return top

    min_purple = 0.45
    streak = 0
    for row in range(region.shape[0]):
        if float(np.mean(region[row] > 0)) >= min_purple:
            streak += 1
            if streak >= 4:
                return top + row - 3
        else:
            streak = 0
    return top


def detect_game_content_rect(frame: Frame) -> tuple[int, int, int, int]:
    """Find the purple GamePigeon playfield, excluding the iOS status bar."""
    mask = _purple_mask(frame.image)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, frame.width, frame.height

    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if width * height < frame.width * frame.height * 0.20:
        return 0, 0, frame.width, frame.height

    top = _trim_status_bar(mask, x, y, x + width, y + height)
    pad = int(min(width, height) * 0.01)
    left = max(0, x + pad)
    right = min(frame.width, x + width - pad)
    bottom = min(frame.height, y + height - pad)
    return left, top, right, bottom


def crop_game_content(frame: Frame) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rect = detect_game_content_rect(frame)
    return crop_region(frame, *rect), rect
