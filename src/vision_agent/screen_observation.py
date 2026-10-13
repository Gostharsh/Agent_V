
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from threading import RLock
from typing import Callable
from datetime import datetime, timezone
from vision_agent.screen_capture import ScreenCaptureService, ScreenFrame


@dataclass(frozen=True, slots=True)
class ScreenObservation:
    """A captured frame with change-detection metadata."""

    frame: ScreenFrame
    fingerprint: str
    changed: bool


class ScreenObservationController:
    """Explicitly controlled, throttled screen observation.

    Polling is caller-driven. This controller never creates a background
    thread and retains only the latest successful observation.
    """

    def __init__(
        self,
        capture_service: ScreenCaptureService,
        *,
        min_interval_seconds: float = 1.0,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        if (
            isinstance(min_interval_seconds, bool)
            or not isinstance(min_interval_seconds, (int, float))
            or not math.isfinite(min_interval_seconds)
            or min_interval_seconds <= 0
        ):
            raise ValueError("Refresh interval must be finite and positive")

        self._capture_service = capture_service
        self._min_interval_seconds = float(min_interval_seconds)
        self._clock = clock
        self._wall_clock = (
                wall_clock
                if wall_clock is not None
                else lambda: datetime.now(timezone.utc)
            )
        self._active = False
        self._last_poll_at: float | None = None
        self._latest_observation: ScreenObservation | None = None
        self._last_error: str | None = None
        self._lock = RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def latest_observation(self) -> ScreenObservation | None:
        with self._lock:
            return self._latest_observation

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def start(self) -> None:
        with self._lock:
            if self._active:
                return

            self._capture_service.start()
            self._active = True
            self._last_poll_at = None
            self._latest_observation = None
            self._last_error = None

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                self._latest_observation = None
                self._last_poll_at = None
                return

            # Disable observation and clear retained screen data first.
            self._active = False
            self._latest_observation = None
            self._last_poll_at = None

            try:
                self._capture_service.stop()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise
            else:
                self._last_error = None

    def poll(self) -> ScreenObservation | None:
        with self._lock:
            if not self._active:
                raise RuntimeError("Screen observation is not active")

            now = self._now()
            if self._last_poll_at is not None:
                elapsed = now - self._last_poll_at
                if elapsed < 0:
                    # A monotonic clock should not move backwards. Fail
                    # closed rather than accidentally bypassing throttling.
                    raise RuntimeError("Observation clock moved backwards")
                if elapsed < self._min_interval_seconds:
                    return None

            # Throttle even after a capture failure, avoiding a tight
            # retry loop if the display is unavailable.
            self._last_poll_at = now

            try:
                frame = self._capture_service.capture_once()
                fingerprint = hashlib.sha256(frame.data).hexdigest()
                previous = self._latest_observation
                changed = (
                    previous is None
                    or previous.fingerprint != fingerprint
                )
                observation = ScreenObservation(
                    frame=frame,
                    fingerprint=fingerprint,
                    changed=changed,
                )
            except Exception as exc:
                self._latest_observation = None
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

            self._latest_observation = observation
            self._last_error = None
            return observation



    def verify_observation_fingerprint(
        self,
        expected_fingerprint: str,
    ) -> bool:
        """Capture the screen and verify it still matches a decision frame."""
        if (
            not isinstance(expected_fingerprint, str)
            or len(expected_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_fingerprint
            )
        ):
            raise ValueError("Expected observation fingerprint is invalid")

        with self._lock:
            if not self._active:
                raise RuntimeError("Screen observation is not active")

            # The decision observation must still be available and fresh.
            original = self.get_fresh_observation()

            # Capture directly because poll() is throttled.
            frame = self._capture_service.capture_once()

            captured_at = frame.captured_at
            if (
                not isinstance(captured_at, datetime)
                or captured_at.tzinfo is None
                or captured_at.utcoffset() is None
            ):
                raise RuntimeError("Verification timestamp is invalid")

            now = self._wall_clock()
            if (
                not isinstance(now, datetime)
                or now.tzinfo is None
                or now.utcoffset() is None
            ):
                raise RuntimeError(
                    "Observation clock must return a timezone-aware datetime"
                )

            age_seconds = (now - captured_at).total_seconds()
            if age_seconds < 0:
                raise RuntimeError("Verification timestamp is in the future")
            if age_seconds > 5.0:
                raise RuntimeError("Verification capture is stale")

            if original.fingerprint != expected_fingerprint:
                return False

            current_fingerprint = hashlib.sha256(frame.data).hexdigest()
            return current_fingerprint == expected_fingerprint



    def get_fresh_observation(
        self,
        *,
        max_age_seconds: float = 5.0,
    ) -> ScreenObservation:
        """Return the latest observation only if it is fresh and valid."""
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(max_age_seconds)
            or max_age_seconds <= 0
        ):
            raise ValueError("Maximum observation age must be finite and positive")

        with self._lock:
            if not self._active:
                raise RuntimeError("Screen observation is not active")

            observation = self._latest_observation
            if observation is None:
                raise RuntimeError("No screen observation is available")

            captured_at = observation.frame.captured_at
            if (
                not isinstance(captured_at, datetime)
                or captured_at.tzinfo is None
                or captured_at.utcoffset() is None
            ):
                raise RuntimeError("Observation timestamp is invalid")

            now = self._wall_clock()
            if (
                not isinstance(now, datetime)
                or now.tzinfo is None
                or now.utcoffset() is None
            ):
                raise RuntimeError("Observation clock must return a timezone-aware datetime")

            age_seconds = (now - captured_at).total_seconds()

            if age_seconds < 0:
                raise RuntimeError("Observation timestamp is in the future")

            if age_seconds > max_age_seconds:
                raise RuntimeError("Screen observation is stale")

            return observation

    def _now(self) -> float:
        if self._clock is not None:
            value = self._clock()
        else:
            import time
            value = time.monotonic()

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("Observation clock must return a number")
        if not math.isfinite(value):
            raise RuntimeError("Observation clock returned a non-finite value")
        return float(value)
