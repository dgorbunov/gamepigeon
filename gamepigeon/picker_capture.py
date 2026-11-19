"""System window picker and capture session via ScreenCaptureKit."""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from dataclasses import dataclass

import AppKit
import CoreMedia
import Quartz
import ScreenCaptureKit as SCK
import libdispatch
import numpy as np
import objc
from Foundation import NSDate, NSObject, NSRunLoop, NSSelectorFromString

from gamepigeon.capture import capture_window
from gamepigeon.window import WindowInfo, list_on_screen_windows

logger = logging.getLogger(__name__)

SCContentSharingPickerObserver = objc.protocolNamed("SCContentSharingPickerObserver")
SCStreamOutput = objc.protocolNamed("SCStreamOutput")

STREAM_FRAME_WAIT_SECONDS = 3.0


def _ensure_app() -> None:
    AppKit.NSApplication.sharedApplication()


def _shared_picker() -> object:
    return SCK.SCContentSharingPicker.performSelector_(NSSelectorFromString("sharedPicker"))


def refresh_window_bounds(window: WindowInfo) -> WindowInfo:
    """Sync window bounds from the window server after picker selection."""
    for candidate in list_on_screen_windows():
        if candidate.window_id == window.window_id:
            return candidate
    return window


@dataclass(frozen=True)
class CaptureTarget:
    """A user-selected window for capture and clicking."""

    window: WindowInfo
    content_filter: object | None = None

    @classmethod
    def from_window(cls, window: WindowInfo) -> CaptureTarget:
        window = refresh_window_bounds(window)
        if window.bounds[2] <= 0 or window.bounds[3] <= 0:
            raise RuntimeError("Selected window has no visible bounds")
        return cls(window=window, content_filter=None)

    @classmethod
    def from_filter(cls, content_filter: object) -> CaptureTarget:
        sc_window = content_filter.individualWindow()
        if sc_window is None:
            raise RuntimeError("Selected content is not a single window")

        frame = sc_window.frame()
        bounds = (
            float(frame.origin.x),
            float(frame.origin.y),
            float(frame.size.width),
            float(frame.size.height),
        )

        window = WindowInfo(
            window_id=int(sc_window.windowID()),
            title=str(sc_window.title() or ""),
            owner=str(sc_window.owningApplication().applicationName() if sc_window.owningApplication() else ""),
            bounds=bounds,
        )
        window = refresh_window_bounds(window)
        if window.bounds[2] <= 0 or window.bounds[3] <= 0:
            raise RuntimeError("Selected window has no visible bounds")

        return cls(window=window, content_filter=content_filter)


class _PickerObserver(NSObject, protocols=[SCContentSharingPickerObserver]):
    def init(self) -> _PickerObserver:
        self = objc.super(_PickerObserver, self).init()
        if self is None:
            return None
        self.content_filter = None
        self.cancelled = False
        self.error = None
        self._event = threading.Event()
        return self

    def contentSharingPicker_didUpdateWithFilter_forStream_(self, picker, content_filter, stream) -> None:
        logger.info("Window selected via system picker")
        self.content_filter = content_filter
        self._event.set()

    def contentSharingPicker_didCancelForStream_(self, picker, stream) -> None:
        logger.info("Window picker cancelled")
        self.cancelled = True
        self._event.set()

    def contentSharingPickerStartDidFailWithError_(self, error) -> None:
        logger.error("Window picker failed: %s", error)
        self.error = error
        self._event.set()


class _StreamOutput(NSObject, protocols=[SCStreamOutput]):
    def init(self) -> _StreamOutput:
        self = objc.super(_StreamOutput, self).init()
        if self is None:
            return None
        self.latest_bgr: np.ndarray | None = None
        self.scale = 2.0
        self.lock = threading.Lock()
        self.frames_received = 0
        self.last_error: str | None = None
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type) -> None:
        if output_type != SCK.SCStreamOutputTypeScreen:
            return
        try:
            image = _sample_buffer_to_bgr(sample_buffer)
            scale = _sample_buffer_scale(sample_buffer)
            with self.lock:
                self.latest_bgr = image
                self.frames_received += 1
                if scale:
                    self.scale = scale
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
            logger.warning("Frame conversion failed: %s", exc)


def _cg_image_to_bgr(cg_image) -> np.ndarray:
    width = Quartz.CGImageGetWidth(cg_image)
    height = Quartz.CGImageGetHeight(cg_image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(cg_image))
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = buffer.reshape((height, bytes_per_row // 4, 4))[:, :width, :3]
    return np.ascontiguousarray(image[:, :, ::-1])


def _pixel_buffer_to_bgr(image_buffer) -> np.ndarray:
    width = Quartz.CVPixelBufferGetWidth(image_buffer)
    height = Quartz.CVPixelBufferGetHeight(image_buffer)
    row_bytes = Quartz.CVPixelBufferGetBytesPerRow(image_buffer)

    Quartz.CVPixelBufferLockBaseAddress(image_buffer, Quartz.kCVPixelBufferLock_ReadOnly)
    try:
        base = Quartz.CVPixelBufferGetBaseAddress(image_buffer)
        if base is not None:
            address = int(base) if not isinstance(base, int) else base
            raw = (ctypes.c_uint8 * (row_bytes * height)).from_address(address)
            array = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_bytes // 4, 4)[:, :width, :3]
            return np.ascontiguousarray(array[:, :, ::-1])
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(image_buffer, Quartz.kCVPixelBufferLock_ReadOnly)

    ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(image_buffer)
    if ci_image is None:
        raise RuntimeError("Could not create CIImage from pixel buffer")

    context = Quartz.CIContext.contextWithOptions_(None)
    cg_image = context.createCGImage_fromRect_(ci_image, ci_image.extent())
    if cg_image is None:
        raise RuntimeError("Could not render pixel buffer to CGImage")
    return _cg_image_to_bgr(cg_image)


def _sample_buffer_to_bgr(sample_buffer) -> np.ndarray:
    image_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
    if image_buffer is None:
        raise RuntimeError("Sample buffer has no image")
    return _pixel_buffer_to_bgr(image_buffer)


def _sample_buffer_scale(sample_buffer) -> float | None:
    attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sample_buffer, True)
    if not attachments or not attachments.count():
        return None
    metadata = attachments.objectAtIndex_(0)
    if metadata is None:
        return None
    value = metadata.get(SCK.SCStreamFrameInfoScaleFactor)
    return float(value) if value else None


def pick_window(timeout: float = 120.0) -> CaptureTarget:
    """Show the macOS system picker and wait for the user to choose a window."""
    _ensure_app()
    picker = _shared_picker()
    observer = _PickerObserver.alloc().init().retain()

    config = SCK.SCContentSharingPickerConfiguration.alloc().init()
    config.setAllowedPickingModes_(SCK.SCContentSharingPickerModeSingleWindow)
    picker.setDefaultConfiguration_(config)
    picker.setActive_(True)
    picker.addObserver_(observer)

    logger.info("Select the iPhone Mirroring window in the system picker")
    picker.presentPickerUsingContentStyle_(SCK.SCShareableContentStyleWindow)

    deadline = time.time() + timeout
    while time.time() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
        if observer._event.is_set():
            break

    if observer.cancelled:
        raise RuntimeError("Window selection cancelled")
    if observer.error is not None:
        raise RuntimeError(f"Window picker failed: {observer.error}")
    if observer.content_filter is None:
        raise TimeoutError("Timed out waiting for window selection")

    return CaptureTarget.from_filter(observer.content_filter)


def auto_capture_target() -> CaptureTarget:
    """Find the iPhone Mirroring window without the system picker."""
    from gamepigeon.window import find_mirror_window

    window = find_mirror_window()
    if window is None:
        raise RuntimeError(
            "Could not find iPhone Mirroring window. Open it and try again, "
            "or use --picker to select manually."
        )
    logger.info("Using iPhone Mirroring window automatically")
    return CaptureTarget.from_window(window)


def resolve_capture_target(*, use_picker: bool, picker_timeout: float = 120.0) -> CaptureTarget:
    if use_picker:
        return pick_window(timeout=picker_timeout)
    return auto_capture_target()


class PickerCaptureSession:
    """Capture frames from a picker-selected window."""

    def __init__(self, target: CaptureTarget) -> None:
        self.target = target
        self._output = _StreamOutput.alloc().init().retain()
        self._stream = None
        self._started = False
        self._use_fallback = False

    def _capture_with_fallback(self) -> tuple[np.ndarray, float] | None:
        window = refresh_window_bounds(self.target.window)
        self.target = CaptureTarget(window=window, content_filter=self.target.content_filter)
        frame = capture_window(window)
        if frame is None:
            return None
        return frame.image, frame.scale

    def _try_stream_frame(self, wait_seconds: float) -> tuple[np.ndarray, float] | None:
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            with self._output.lock:
                if self._output.latest_bgr is not None:
                    return self._output.latest_bgr.copy(), self._output.scale
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.02))

        with self._output.lock:
            if self._output.last_error:
                logger.warning("Stream produced no frames (%s)", self._output.last_error)
            else:
                logger.warning("Stream produced no frames within %.1fs", wait_seconds)
        return None

    def start(self) -> None:
        if self._started:
            return

        _ensure_app()
        self.target = CaptureTarget(
            window=refresh_window_bounds(self.target.window),
            content_filter=self.target.content_filter,
        )

        # The system picker selects the window; direct window capture is more
        # reliable than SCStream in a CLI process on macOS.
        self._use_fallback = True
        self._started = True
        logger.info(
            "Started window capture for %s",
            self.target.window.title or self.target.window.owner,
        )

    def stop(self) -> None:
        if not self._stream or not self._started:
            return
        if not self._use_fallback:
            done = threading.Event()
            self._stream.stopCaptureWithCompletionHandler_(lambda _: done.set())
            done.wait(timeout=5.0)
        self._started = False

    def capture_bgr(self, wait_seconds: float = STREAM_FRAME_WAIT_SECONDS) -> tuple[np.ndarray, float] | None:
        if not self._started:
            self.start()
        return self._capture_with_fallback()
