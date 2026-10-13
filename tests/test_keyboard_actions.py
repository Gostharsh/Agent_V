
from unittest.mock import Mock, patch

from vision_agent.windows_actions import press_keyboard_action


@patch("vision_agent.windows_actions.pyautogui.press")
@patch("vision_agent.windows_actions.pygetwindow.getActiveWindow")
def test_presses_enter_in_expected_window(mock_active, mock_press):
    mock_active.return_value = Mock(title="Untitled - Notepad")

    result = press_keyboard_action("ENTER", "Notepad")

    assert "Sent ENTER" in result
    mock_press.assert_called_once_with("enter")


@patch("vision_agent.windows_actions.pyautogui.hotkey")
@patch("vision_agent.windows_actions.pygetwindow.getActiveWindow")
def test_uses_approved_shortcut(mock_active, mock_hotkey):
    mock_active.return_value = Mock(title="Untitled - Notepad")

    result = press_keyboard_action("CTRL+A", "Notepad")

    assert "Sent CTRL+A" in result
    mock_hotkey.assert_called_once_with("ctrl", "a")


@patch("vision_agent.windows_actions.pyautogui.press")
@patch("vision_agent.windows_actions.pygetwindow.getActiveWindow")
def test_rejects_unapproved_key(mock_active, mock_press):
    mock_active.return_value = Mock(title="Untitled - Notepad")

    result = press_keyboard_action("ALT+F4", "Notepad")

    assert "not permitted" in result
    mock_press.assert_not_called()


@patch("vision_agent.windows_actions.pyautogui.press")
@patch("vision_agent.windows_actions.pygetwindow.getActiveWindow")
def test_rejects_wrong_active_window(mock_active, mock_press):
    mock_active.return_value = Mock(title="Browser")

    result = press_keyboard_action("ENTER", "Notepad")

    assert "expected window is not active" in result
    mock_press.assert_not_called()


@patch("vision_agent.windows_actions.pyautogui.press")
@patch("vision_agent.windows_actions.pygetwindow.getActiveWindow")
def test_rejects_empty_window_title(mock_active, mock_press):
    result = press_keyboard_action("ENTER", "  ")

    assert "cannot be empty" in result
    mock_active.assert_not_called()
    mock_press.assert_not_called()
