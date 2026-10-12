
from unittest.mock import patch

from vision_agent.tools import TOOLS, analyze_image


def test_analyze_image_is_registered():
    assert analyze_image in TOOLS


def test_analyze_image_returns_description():
    with patch(
        "vision_agent.tools.describe_image",
        return_value="A red rectangle on a white background.",
    ) as mock_describe:
        result = analyze_image.invoke(
            {
                "image_path": "sample.png",
                "question": "What is in this image?",
            }
        )

    assert result == "A red rectangle on a white background."
    mock_describe.assert_called_once_with(
        "sample.png",
        "What is in this image?",
    )


def test_analyze_image_handles_missing_file():
    with patch(
        "vision_agent.tools.describe_image",
        side_effect=FileNotFoundError("Image file not found"),
    ):
        result = analyze_image.invoke(
            {
                "image_path": "missing.png",
                "question": "Describe it",
            }
        )

    assert result == "Image analysis error: Image file not found"


def test_analyze_image_hides_unexpected_errors():
    with patch(
        "vision_agent.tools.describe_image",
        side_effect=RuntimeError("internal details"),
    ):
        result = analyze_image.invoke(
            {
                "image_path": "sample.png",
                "question": "Describe it",
            }
        )

    assert result == "Image analysis failed due to an unexpected error."

