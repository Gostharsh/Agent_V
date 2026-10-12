from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from vision_agent.screen_observation import ScreenObservationController
from vision_agent.screen_capture import ScreenFrame


class FakeCaptureService:
    def __init__(self, frame):
        self.frame = frame
        self.active = False
        self.starts = 0
        self.captures = 0

    def start(self):
        self.starts += 1
        self.active = True

    def stop(self):
        self.active = False

    def capture_once(self):
        if not self.active:
            raise RuntimeError("Screen capture is not active")
        self.captures += 1
        return self.frame


def make_png():
    buffer = BytesIO()
    Image.new("RGB", (16, 12), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def make_controller():
    data = make_png()
    frame = ScreenFrame(
        data=data,
        width=16,
        height=12,
        captured_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
    )
    service = FakeCaptureService(frame)
    controller = ScreenObservationController(service)
    return controller, service, data


def test_screen_vision_rejects_inactive_controller():
    controller, service, _ = make_controller()

    # This import will be implemented alongside the handoff module.
    from vision_agent.screen_vision import analyze_current_screen

    with patch("vision_agent.screen_vision.describe_image_bytes") as analyze:
        with pytest.raises(RuntimeError, match="active"):
            analyze_current_screen(controller, "Describe the screen")

    assert service.starts == 0
    assert service.captures == 0
    analyze.assert_not_called()


def test_screen_vision_requires_existing_observation():
    controller, service, _ = make_controller()
    controller.start()

    from vision_agent.screen_vision import analyze_current_screen

    with patch("vision_agent.screen_vision.describe_image_bytes") as analyze:
        with pytest.raises(RuntimeError, match="observation"):
            analyze_current_screen(controller, "Describe the screen")

    assert service.captures == 0
    analyze.assert_not_called()


def test_screen_vision_passes_latest_frame_bytes_to_vision():
    controller, service, data = make_controller()
    controller.start()
    observation = controller.poll()
    assert observation is not None

    from vision_agent.screen_vision import analyze_current_screen

    with patch(
        "vision_agent.screen_vision.describe_image_bytes",
        return_value="A blue screen.",
    ) as analyze:
        result = analyze_current_screen(
            controller,
            "What is visible?",
        )

    assert result == "A blue screen."
    analyze.assert_called_once_with(data, "What is visible?")
    assert service.captures == 1


def test_screen_vision_rejects_controller_after_stop():
    controller, _, _ = make_controller()
    controller.start()
    controller.poll()
    controller.stop()

    from vision_agent.screen_vision import analyze_current_screen

    with patch("vision_agent.screen_vision.describe_image_bytes") as analyze:
        with pytest.raises(RuntimeError, match="active"):
            analyze_current_screen(controller, "Describe the screen")

    analyze.assert_not_called()
