
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from vision_agent.vision import describe_image_bytes


def make_png(width=16, height=12):
    buffer = BytesIO()
    Image.new("RGB", (width, height), "blue").save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def test_describe_image_bytes_calls_vision_model():
    fake_response = MagicMock()
    fake_response.content = "A blue desktop."

    with patch("vision_agent.vision.ChatOllama") as model_class:
        model_class.return_value.invoke.return_value = fake_response

        result = describe_image_bytes(
            make_png(),
            "What is visible on screen?",
        )

    assert result == "A blue desktop."
    model_class.assert_called_once_with(
        model="qwen3-vl:latest",
        temperature=0,
    )
    model_class.return_value.invoke.assert_called_once()


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_describe_image_bytes_rejects_invalid_data(data):
    with patch("vision_agent.vision.ChatOllama") as model_class:
        with pytest.raises(ValueError, match="image|PNG|non-empty"):
            describe_image_bytes(data, "Describe the screen")

        model_class.assert_not_called()


def test_describe_image_bytes_rejects_empty_question():
    with pytest.raises(ValueError, match="question"):
        describe_image_bytes(make_png(), " ")

def test_describe_image_bytes_rejects_non_png():
    buffer = BytesIO()
    Image.new("RGB", (16, 12), "blue").save(buffer, format="JPEG")

    with pytest.raises(ValueError, match="PNG"):
        describe_image_bytes(buffer.getvalue(), "Describe the screen")
