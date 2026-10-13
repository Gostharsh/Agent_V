
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from vision_agent.screen_observation import ScreenObservationController


@dataclass(frozen=True)
class FakeFrame:
    data: bytes
    width: int = 16
    height: int = 12
    captured_at: datetime = datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )


class FakeCaptureService:
    def __init__(self, frames=None):
        self.frames = list(frames or [FakeFrame(b"frame-1")])
        self.active = False
        self.starts = 0
        self.stops = 0
        self.captures = 0

    @property
    def status(self):
        return type("Status", (), {"active": self.active})()

    @property
    def latest_frame(self):
        return None

    def start(self):
        if not self.active:
            self.starts += 1
            self.active = True

    def stop(self):
        if self.active:
            self.stops += 1
            self.active = False

    def capture_once(self):
        if not self.active:
            raise RuntimeError("Screen capture is not active")
        self.captures += 1
        frame = self.frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def make_controller(frames=None, interval=1.0, wall_clock=None):
    clock = FakeClock()
    service = FakeCaptureService(frames)
    controller = ScreenObservationController(
        capture_service=service,
        min_interval_seconds=interval,
        clock=clock,
        wall_clock=wall_clock,
    )
    return controller, service, clock


def test_observation_requires_explicit_start():
    controller, service, _ = make_controller()

    with pytest.raises(RuntimeError, match="not active"):
        controller.poll()

    assert service.captures == 0


def test_start_and_stop_are_idempotent():
    controller, service, _ = make_controller()

    controller.start()
    controller.start()
    assert service.starts == 1

    controller.stop()
    controller.stop()
    assert service.stops == 1
    assert controller.latest_observation is None


def test_first_poll_captures_frame():
    controller, service, _ = make_controller()
    controller.start()

    result = controller.poll()

    assert result is not None
    assert result.frame.data == b"frame-1"
    assert result.changed is True
    assert result.fingerprint
    assert service.captures == 1



def test_poll_is_throttled_until_interval_elapses():
    frames = [
        FakeFrame(b"frame-1"),
        FakeFrame(b"frame-2"),
    ]
    controller, service, clock = make_controller(
        frames, interval=2.0
    )
    controller.start()

    first = controller.poll()
    assert first is not None

    clock.advance(1.99)
    assert controller.poll() is None
    assert service.captures == 1

    clock.advance(0.01)
    second = controller.poll()
    assert second is not None
    assert service.captures == 2



def test_unchanged_frame_is_detected():
    frames = [FakeFrame(b"same"), FakeFrame(b"same")]
    controller, _, clock = make_controller(frames, interval=1.0)
    controller.start()

    first = controller.poll()
    clock.advance(1.0)
    second = controller.poll()

    assert first.changed is True
    assert second.changed is False
    assert first.fingerprint == second.fingerprint


def test_changed_frame_has_different_fingerprint():
    frames = [FakeFrame(b"before"), FakeFrame(b"after")]
    controller, _, clock = make_controller(frames, interval=1.0)
    controller.start()

    first = controller.poll()
    clock.advance(1.0)
    second = controller.poll()

    assert first.fingerprint != second.fingerprint
    assert second.changed is True


def test_stop_clears_latest_observation():
    controller, service, _ = make_controller()
    controller.start()
    controller.poll()

    controller.stop()

    assert controller.latest_observation is None
    assert service.active is False

    with pytest.raises(RuntimeError, match="not active"):
        controller.poll()


def test_capture_failure_does_not_publish_observation():
    controller, _, _ = make_controller(
        [RuntimeError("display unavailable")]
    )
    controller.start()

    with pytest.raises(RuntimeError, match="display unavailable"):
        controller.poll()

    assert controller.latest_observation is None


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_refresh_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="interval"):
        ScreenObservationController(
            capture_service=FakeCaptureService(),
            min_interval_seconds=interval,
        )


def test_get_fresh_observation_accepts_recent_frame():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    frame = FakeFrame(b"fresh", captured_at=now - timedelta(seconds=2))
    controller, _, _ = make_controller(
        [frame],
        wall_clock=lambda: now,
    )
    controller.start()
    controller.poll()

    assert controller.get_fresh_observation() is not None


def test_get_fresh_observation_rejects_stale_frame():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    frame = FakeFrame(b"stale", captured_at=now - timedelta(seconds=6))
    controller, _, _ = make_controller(
        [frame],
        wall_clock=lambda: now,
    )
    controller.start()
    controller.poll()

    with pytest.raises(RuntimeError, match="stale"):
        controller.get_fresh_observation()


def test_get_fresh_observation_rejects_missing_observation():
    controller, _, _ = make_controller()
    controller.start()

    with pytest.raises(RuntimeError, match="observation"):
        controller.get_fresh_observation()


def test_get_fresh_observation_rejects_inactive_controller():
    controller, _, _ = make_controller()

    with pytest.raises(RuntimeError, match="active"):
        controller.get_fresh_observation()


def test_get_fresh_observation_rejects_future_timestamp():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    frame = FakeFrame(b"future", captured_at=now + timedelta(seconds=2))
    controller, _, _ = make_controller(
        [frame],
        wall_clock=lambda: now,
    )
    controller.start()
    controller.poll()

    with pytest.raises(RuntimeError, match="future"):
        controller.get_fresh_observation()


@pytest.mark.parametrize(
    "max_age_seconds",
    [0, -1, float("nan"), float("inf"), True, "5"],
)
def test_get_fresh_observation_rejects_invalid_age_limit(max_age_seconds):
    controller, _, _ = make_controller()

    with pytest.raises(ValueError, match="age"):
        controller.get_fresh_observation(max_age_seconds=max_age_seconds)



def test_get_fresh_observation_rejects_naive_capture_timestamp():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    frame = FakeFrame(b"naive", captured_at=now.replace(tzinfo=None))
    controller, _, _ = make_controller([frame], wall_clock=lambda: now)
    controller.start()
    controller.poll()

    with pytest.raises(RuntimeError, match="timestamp"):
        controller.get_fresh_observation()


@pytest.mark.parametrize(
    "wall_clock_result",
    [
        None,
        "not-a-datetime",
        datetime(2026, 1, 1, 12, 0),
    ],
)
def test_get_fresh_observation_rejects_invalid_wall_clock(wall_clock_result):
    frame = FakeFrame(
        b"frame",
        captured_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    controller, _, _ = make_controller(
        [frame],
        wall_clock=lambda: wall_clock_result,
    )
    controller.start()
    controller.poll()

    with pytest.raises(RuntimeError, match="clock"):
        controller.get_fresh_observation()
