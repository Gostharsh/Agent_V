from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4
from vision_agent.graph import run_agent
from vision_agent.screen_capture import ScreenCaptureService
from vision_agent.screen_observation import ScreenObservationController

@dataclass(frozen=True, slots=True)
class ApplicationStatus:
    """A snapshot of the application's screen-observation status."""

    observation_active: bool
    last_error: str | None


class VisionAgentApplication:
    """Synchronous application boundary for Vision Agent."""

    def __init__(
        self,
        *,
        thread_id: str | None = None,
        owner_id: str = "default",
        observation_interval_seconds: float = 1.0,
    ) -> None:


        if thread_id is not None and (
            not isinstance(thread_id, str) or not thread_id.strip()
        ):
            raise ValueError("thread_id must be a non-empty string when provided")
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")

        self._thread_id = (
            thread_id if thread_id is not None else "vision-agent-default"
        )
        self._owner_id = owner_id

        capture_service = ScreenCaptureService()
        self._observation = ScreenObservationController(
            capture_service,
            min_interval_seconds=observation_interval_seconds,
        )
        self._closed = False

    @property
    def status(self) -> ApplicationStatus:
        """Return the current observation status."""
        return ApplicationStatus(
            observation_active=self._observation.active,
            last_error=self._observation.last_error,
        )

    def start_screen_observation(self) -> None:
        """Explicitly activate screen capture and observation."""
        self._ensure_open()
        self._observation.start()

    def poll_screen_observation(self):
        """Capture an observation if the polling interval allows it."""
        self._ensure_open()
        return self._observation.poll()

    def stop_screen_observation(self) -> None:
        """Stop observation and release screen-capture resources."""
        self._ensure_open()
        self._observation.stop()

    def ask(self, message: str) -> str:
        """Submit a message to the existing agent conversation."""
        self._ensure_open()

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        return run_agent(
            message,
            thread_id=self._thread_id,
            owner_id=self._owner_id,
            screen_observation_controller=self._observation,
        )

    def close(self) -> None:
        """Close the application and release observation resources."""
        if self._closed:
            return

        try:
            self._observation.stop()
        finally:
            self._closed = True


    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Vision Agent application is closed")

    def __enter__(self) -> VisionAgentApplication:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
