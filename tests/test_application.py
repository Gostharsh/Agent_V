
from datetime import datetime, timezone

import pytest

import vision_agent.application as application_module
from vision_agent.application import VisionAgentApplication
from vision_agent.screen_capture import ScreenFrame


class FakeCaptureService:
    def __init__(self):
        self.active = False
        self.start_calls = 0
        self.stop_calls = 0
        self.fail_on_stop = False

    def start(self):
        self.start_calls += 1
        self.active = True

    def stop(self):
        self.stop_calls += 1
        self.active = False
        if self.fail_on_stop:
            raise RuntimeError("simulated cleanup failure")

    def capture_once(self):
        if not self.active:
            raise RuntimeError("fake capture is inactive")

        return ScreenFrame(
            data=b"fake screenshot bytes",
            width=1,
            height=1,
            captured_at=datetime.now(timezone.utc),
        )


@pytest.fixture
def app_setup(monkeypatch):
    capture = FakeCaptureService()
    calls = []

    monkeypatch.setattr(
        application_module,
        "ScreenCaptureService",
        lambda: capture,
    )

    def fake_run_agent(message, **kwargs):
        calls.append((message, kwargs))
        return "mock agent response"

    monkeypatch.setattr(application_module, "run_agent", fake_run_agent)

    app = VisionAgentApplication(
        thread_id="test-thread",
        owner_id="test-owner",
        observation_interval_seconds=3600,
    )
    return app, capture, calls


def test_observation_is_inactive_until_explicitly_started(app_setup):
    app, capture, _ = app_setup

    assert app.status.observation_active is False
    assert capture.start_calls == 0

    with pytest.raises(RuntimeError, match="not active"):
        app.poll_screen_observation()

    assert capture.start_calls == 0


def test_start_and_poll_observation(app_setup):
    app, capture, _ = app_setup

    app.start_screen_observation()
    observation = app.poll_screen_observation()

    assert capture.start_calls == 1
    assert app.status.observation_active is True
    assert observation is not None
    assert observation.frame.data == b"fake screenshot bytes"
    assert observation.changed is True


def test_poll_respects_observation_interval(app_setup):
    app, _, _ = app_setup

    app.start_screen_observation()

    first = app.poll_screen_observation()
    second = app.poll_screen_observation()

    assert first is not None
    assert second is None


def test_ask_passes_message_ids_and_controller(app_setup):
    app, _, calls = app_setup

    response = app.ask("Describe my screen")

    assert response == "mock agent response"
    assert len(calls) == 1

    message, kwargs = calls[0]
    assert message == "Describe my screen"
    assert kwargs["thread_id"] == "test-thread"
    assert kwargs["owner_id"] == "test-owner"
    assert kwargs["screen_observation_controller"] is app._observation


@pytest.mark.parametrize("message", ["", "   ", "\n"])
def test_ask_rejects_empty_messages(app_setup, message):
    app, _, calls = app_setup

    with pytest.raises(ValueError, match="non-empty"):
        app.ask(message)

    assert calls == []


def test_stop_clears_observation_and_releases_capture(app_setup):
    app, capture, _ = app_setup

    app.start_screen_observation()
    assert app.poll_screen_observation() is not None

    app.stop_screen_observation()

    assert capture.stop_calls == 1
    assert app.status.observation_active is False
    assert app._observation.latest_observation is None


def test_close_is_idempotent(app_setup):
    app, capture, _ = app_setup

    app.start_screen_observation()
    app.close()
    app.close()

    assert capture.stop_calls == 1
    assert app.status.observation_active is False


def test_close_marks_app_closed_even_if_cleanup_fails(app_setup):
    app, capture, _ = app_setup

    app.start_screen_observation()
    capture.fail_on_stop = True

    with pytest.raises(RuntimeError, match="simulated cleanup failure"):
        app.close()

    with pytest.raises(RuntimeError, match="application is closed"):
        app.ask("Hello")

    with pytest.raises(RuntimeError, match="application is closed"):
        app.start_screen_observation()


def test_context_manager_closes_application(app_setup):
    app, capture, _ = app_setup

    with app as session:
        assert session is app
        app.start_screen_observation()

    assert capture.stop_calls == 1
    assert app.status.observation_active is False


def test_operations_are_rejected_after_close(app_setup):
    app, _, calls = app_setup
    app.close()

    with pytest.raises(RuntimeError, match="application is closed"):
        app.ask("Hello")

    with pytest.raises(RuntimeError, match="application is closed"):
        app.poll_screen_observation()

    with pytest.raises(RuntimeError, match="application is closed"):
        app.stop_screen_observation()

    assert calls == []





def test_default_thread_ids_are_unique(monkeypatch):
    capture = FakeCaptureService()
    monkeypatch.setattr(
        application_module,
        "ScreenCaptureService",
        lambda: capture,
    )

    first = VisionAgentApplication(owner_id="test-owner")
    second = VisionAgentApplication(owner_id="test-owner")

    try:
        assert first._thread_id != second._thread_id
    finally:
        first.close()
        second.close()
