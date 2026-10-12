
from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image

from vision_agent.screen_capture import ScreenCaptureService


def png_bytes(width=16, height=12, color="red"):
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=color).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


class FakeBackend:
    def __init__(self, frames=None):
        self.frames = list(frames or [png_bytes()])
        self.started = 0
        self.stopped = 0
        self.captures = 0
        self.running = False

    def start(self):
        self.started += 1
        self.running = True

    def stop(self):
        self.stopped += 1
        self.running = False

    def capture(self):
        if not self.running:
            raise RuntimeError("Backend is not running")
        self.captures += 1
        frame = self.frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame


def test_capture_is_inactive_until_started():
    backend = FakeBackend()
    service = ScreenCaptureService(backend=backend)

    assert service.status.active is False
    assert service.latest_frame is None
    assert backend.captures == 0


def test_start_and_stop_are_idempotent():
    backend = FakeBackend()
    service = ScreenCaptureService(backend=backend)

    service.start()
    service.start()
    assert service.status.active is True
    assert backend.started == 1

    service.stop()
    service.stop()
    assert service.status.active is False
    assert backend.stopped == 1


def test_capture_once_returns_validated_frame():
    backend = FakeBackend([png_bytes(16, 12)])
    service = ScreenCaptureService(backend=backend)
    service.start()

    frame = service.capture_once()

    assert frame.width == 16
    assert frame.height == 12
    assert frame.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert frame.captured_at.tzinfo is not None
    assert frame.captured_at.utcoffset() is not None
    assert isinstance(frame.captured_at, datetime)


def test_capture_is_rejected_while_inactive():
    backend = FakeBackend()
    service = ScreenCaptureService(backend=backend)

    with pytest.raises(RuntimeError, match="not active"):
        service.capture_once()

    assert backend.captures == 0


def test_stop_clears_latest_frame_and_prevents_capture():
    backend = FakeBackend()
    service = ScreenCaptureService(backend=backend)
    service.start()
    service.capture_once()

    service.stop()

    assert service.latest_frame is None
    with pytest.raises(RuntimeError, match="not active"):
        service.capture_once()
    assert backend.captures == 1


@pytest.mark.parametrize(
    "invalid_frame",
    [
        b"",
        b"not an image",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
    ],
)
def test_invalid_frames_are_rejected(invalid_frame):
    backend = FakeBackend([invalid_frame])
    service = ScreenCaptureService(backend=backend)
    service.start()

    with pytest.raises(ValueError, match="frame|image|PNG"):
        service.capture_once()

    assert service.latest_frame is None


def test_oversized_frame_is_rejected():
    backend = FakeBackend([png_bytes(64, 64)])
    service = ScreenCaptureService(
        backend=backend,
        max_frame_bytes=100,
    )
    service.start()

    with pytest.raises(ValueError, match="size|large|limit"):
        service.capture_once()

    assert service.latest_frame is None


def test_dimensions_are_bounded():
    backend = FakeBackend([png_bytes(64, 64)])
    service = ScreenCaptureService(
        backend=backend,
        max_width=32,
        max_height=32,
    )
    service.start()

    with pytest.raises(ValueError, match="dimension|large|limit"):
        service.capture_once()

    assert service.latest_frame is None


def test_backend_failure_is_reported_and_does_not_publish_frame():
    backend = FakeBackend([RuntimeError("display unavailable")])
    service = ScreenCaptureService(backend=backend)
    service.start()

    with pytest.raises(RuntimeError, match="display unavailable"):
        service.capture_once()

    assert service.latest_frame is None




def test_real_mss_backend_captures_png():
    from vision_agent.screen_capture import MSSCaptureBackend

    backend = MSSCaptureBackend()
    backend.start()

    try:
        data = backend.capture()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")

        with Image.open(BytesIO(data)) as image:
            image.verify()
    finally:
        backend.stop()

