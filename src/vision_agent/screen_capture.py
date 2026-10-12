
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from threading import RLock
from typing import Protocol

import mss
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True, slots=True)
class ScreenFrame:
    """An immutable, validated screenshot and its capture metadata."""

    data: bytes
    width: int
    height: int
    captured_at: datetime


@dataclass(frozen=True, slots=True)
class ScreenCaptureStatus:
    """Public snapshot of the capture service's current state."""

    active: bool
    last_error: str | None


class CaptureBackend(Protocol):
    """Backend contract, intentionally independent of the capture library."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def capture(self) -> bytes: ...


class MSSCaptureBackend:
    """Capture the primary Windows display using mss."""

    def __init__(self) -> None:
        self._capture: mss.MSS | None = None
        self._lock = RLock()

    def start(self) -> None:
        with self._lock:
            if self._capture is None:
                self._capture = mss.MSS()

    def stop(self) -> None:
        with self._lock:
            capture, self._capture = self._capture, None
            if capture is not None:
                capture.close()

    def capture(self) -> bytes:
        with self._lock:
            if self._capture is None:
                raise RuntimeError("Capture backend is not running")

            # mss monitor 1 is the primary display; monitor 0 is the
            # virtual desktop spanning all monitors.
            shot = self._capture.grab(self._capture.monitors[1])
            image = Image.frombytes("RGB", shot.size, shot.rgb)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()


class ScreenCaptureService:
    """Explicitly controlled, synchronous screen capture.

    This service never starts a background capture loop. The caller must
    start it and explicitly request each frame.
    """

    def __init__(
        self,
        backend: CaptureBackend | None = None,
        *,
        max_frame_bytes: int = 20 * 1024 * 1024,
        max_width: int = 7680,
        max_height: int = 4320,
    ) -> None:
        if (
            isinstance(max_frame_bytes, bool)
            or not isinstance(max_frame_bytes, int)
            or max_frame_bytes <= 0
        ):
            raise ValueError("max_frame_bytes must be a positive integer")

        if (
            isinstance(max_width, bool)
            or not isinstance(max_width, int)
            or max_width <= 0
            or isinstance(max_height, bool)
            or not isinstance(max_height, int)
            or max_height <= 0
        ):
            raise ValueError("Maximum dimensions must be positive integers")

        self._backend = backend if backend is not None else MSSCaptureBackend()
        self._max_frame_bytes = max_frame_bytes
        self._max_width = max_width
        self._max_height = max_height
        self._active = False
        self._latest_frame: ScreenFrame | None = None
        self._last_error: str | None = None
        self._lock = RLock()

    @property
    def status(self) -> ScreenCaptureStatus:
        with self._lock:
            return ScreenCaptureStatus(
                active=self._active,
                last_error=self._last_error,
            )

    @property
    def latest_frame(self) -> ScreenFrame | None:
        with self._lock:
            return self._latest_frame

    def start(self) -> None:
        with self._lock:
            if self._active:
                return

            try:
                self._backend.start()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

            self._latest_frame = None
            self._last_error = None
            self._active = True

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                self._latest_frame = None
                return

            # Mark inactive and clear retained screenshot before releasing
            # the backend, so no new request can begin through this service.
            self._active = False
            self._latest_frame = None

            try:
                self._backend.stop()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

    def capture_once(self) -> ScreenFrame:
        with self._lock:
            if not self._active:
                raise RuntimeError("Screen capture is not active")

            try:
                data = self._backend.capture()
                frame = self._validate_frame(data)
            except Exception as exc:
                self._latest_frame = None
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

            self._latest_frame = frame
            self._last_error = None
            return frame

    def _validate_frame(self, data: bytes) -> ScreenFrame:
        if not isinstance(data, bytes) or not data:
            raise ValueError("Frame must contain non-empty PNG image bytes")

        if len(data) > self._max_frame_bytes:
            raise ValueError("Frame exceeds the configured byte size limit")

        try:
            with Image.open(BytesIO(data)) as image:
                if image.format != "PNG":
                    raise ValueError("Frame must be a PNG image")

                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("Frame dimensions must be positive")

                if width > self._max_width or height > self._max_height:
                    raise ValueError("Frame exceeds the configured dimension limit")

                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ValueError("Invalid or corrupted PNG frame") from exc

        return ScreenFrame(
            data=data,
            width=width,
            height=height,
            captured_at=datetime.now(timezone.utc),
        )
