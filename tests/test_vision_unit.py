
from unittest.mock import MagicMock, patch

import pytest

from vision_agent.vision import describe_image


def test_describe_image_rejects_empty_path():
    with pytest.raises(ValueError, match="non-empty"):
        describe_image("", "Describe this image")


def test_describe_image_rejects_empty_question():
    with pytest.raises(ValueError, match="non-empty"):
        describe_image("example.png", " ")


def test_describe_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        describe_image(str(tmp_path / "missing.png"), "What is shown?")


def test_describe_image_rejects_unsupported_format(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported image format"):
        describe_image(str(path), "What is shown?")


def test_describe_image_calls_model(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"test image bytes")

    fake_response = MagicMock()
    fake_response.content = "A red rectangle."

    with patch("vision_agent.vision.ChatOllama") as model_class:
        model_class.return_value.invoke.return_value = fake_response

        result = describe_image(str(path), "What color is the rectangle?")

    assert result == "A red rectangle."
    model_class.assert_called_once_with(
        model="qwen3-vl:latest",
        temperature=0,
    )
    model_class.return_value.invoke.assert_called_once()
