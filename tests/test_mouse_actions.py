
from unittest.mock import patch

from vision_agent.windows_actions import click_screen_position


@patch("vision_agent.windows_actions.pyautogui.click")
@patch("vision_agent.windows_actions.pyautogui.size", return_value=(1920, 1080))
def test_clicks_valid_coordinates(mock_size, mock_click):
    result = click_screen_position(500, 400)

    assert result == "Clicked screen position (500, 400)."
    mock_click.assert_called_once_with(x=500, y=400)


@patch("vision_agent.windows_actions.pyautogui.click")
@patch("vision_agent.windows_actions.pyautogui.size", return_value=(1920, 1080))
def test_rejects_coordinates_outside_screen(mock_size, mock_click):
    result = click_screen_position(1920, 400)

    assert "outside the screen" in result
    mock_click.assert_not_called()


@patch("vision_agent.windows_actions.pyautogui.click")
def test_rejects_non_integer_coordinates(mock_click):
    result = click_screen_position(100.5, 200)

    assert "must be integers" in result
    mock_click.assert_not_called()


@patch("vision_agent.windows_actions.pyautogui.click")
@patch("vision_agent.windows_actions.pyautogui.size", return_value=(1920, 1080))
def test_reports_click_failure(mock_size, mock_click):
    mock_click.side_effect = RuntimeError("simulated failure")

    result = click_screen_position(100, 200)

    assert result == "Windows action error: click failed."
