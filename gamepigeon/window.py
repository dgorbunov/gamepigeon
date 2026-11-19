"""Locate and manage the iPhone Mirroring window."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListOptionOnScreenOnly,
)

logger = logging.getLogger(__name__)

MIRROR_BUNDLE_IDS = frozenset(
    {
        "com.apple.ScreenContinuity",
        "com.apple.AirPlayUIAgent",
    }
)


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    title: str
    owner: str
    bounds: tuple[float, float, float, float]  # x, y, width, height (screen points)

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.bounds
        return x + width / 2, y + height / 2


def list_on_screen_windows() -> list[WindowInfo]:
    raw = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []
    windows: list[WindowInfo] = []
    for entry in raw:
        bounds = entry.get("kCGWindowBounds", {})
        windows.append(
            WindowInfo(
                window_id=int(entry["kCGWindowNumber"]),
                title=str(entry.get("kCGWindowName") or ""),
                owner=str(entry.get("kCGWindowOwnerName") or ""),
                bounds=(
                    float(bounds.get("X", 0)),
                    float(bounds.get("Y", 0)),
                    float(bounds.get("Width", 0)),
                    float(bounds.get("Height", 0)),
                ),
            )
        )
    return windows


def find_mirror_window(
    title_patterns: tuple[str, ...] | None = None,
    owner_patterns: tuple[str, ...] | None = None,
) -> WindowInfo | None:
    title_patterns = title_patterns or (
        r"iPhone Mirroring",
        r"iPhone",
        r"Screen Mirroring",
    )
    owner_patterns = owner_patterns or (
        r"iPhone Mirroring",
        r"ControlCenter",
    )

    candidates: list[WindowInfo] = []
    for window in list_on_screen_windows():
        if window.bounds[2] < 200 or window.bounds[3] < 300:
            continue
        title_match = any(re.search(pattern, window.title, re.I) for pattern in title_patterns)
        owner_match = any(re.search(pattern, window.owner, re.I) for pattern in owner_patterns)
        if title_match or owner_match:
            candidates.append(window)

    if not candidates:
        return None

    return max(candidates, key=lambda w: w.bounds[2] * w.bounds[3])


def focus_window(window: WindowInfo) -> bool:
    """Bring the mirroring app and window to the foreground."""
    activated = False
    owner_lower = window.owner.lower()

    for app in NSWorkspace.sharedWorkspace().runningApplications():
        bundle_id = str(app.bundleIdentifier() or "")
        name = str(app.localizedName() or "")
        if (
            name == window.owner
            or name.lower() == owner_lower
            or bundle_id in MIRROR_BUNDLE_IDS
            or "mirror" in name.lower()
        ):
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            activated = True
            logger.info("Activated application: %s", name or bundle_id)
            break

    script = f'tell application "{window.owner}" to activate'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        activated = True
        logger.info("Raised %s via AppleScript", window.owner)
    except Exception as exc:
        logger.warning("AppleScript focus failed: %s", exc)

    if not activated:
        logger.warning("Could not focus owner application: %s", window.owner)
    return activated
