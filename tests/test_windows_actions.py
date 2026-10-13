
from unittest.mock import Mock, patch

from vision_agent.windows_actions import open_notepad_and_type


def test_rejects_empty_text():
    result = open_notepad_and_type("   ")
    assert "cannot be empty" in result


def test_rejects_text_over_limit():
    result = open_notepad_and_type("x" * 5001)
    assert "exceeds" in result


@patch("vision_agent.windows_actions.pyautogui.write")
@patch("vision_agent.windows_actions.pygetwindow.getActiveWindow")
@patch("vision_agent.windows_actions.pygetwindow.getAllWindows")
@patch("vision_agent.windows_actions.subprocess.Popen")
def test_types_only_after_notepad_is_active(
    mock_popen,
    mock_get_windows,
    mock_active_window,
    mock_write,
):
    notepad = Mock()
    notepad._hWnd = 123
    notepad.title = "Untitled - Notepad"
    notepad.activate.return_value = None

    mock_get_windows.side_effect = [
        [],
        [notepad],
    ]
    mock_active_window.return_value = notepad

    with patch("vision_agent.windows_actions.time.sleep"):
        result = open_notepad_and_type("Vision Agent test")

    assert "sent the requested text" in result
    mock_popen.assert_called_once()
    mock_write.assert_called_once_with(
        "Vision Agent test",
        interval=0.01,
    )



@patch("vision_agent.windows_actions.pyautogui.write")
@patch("vision_agent.windows_actions.pygetwindow.getActiveWindow")
@patch("vision_agent.windows_actions.pygetwindow.getAllWindows")
@patch("vision_agent.windows_actions.subprocess.Popen")
def test_does_not_type_if_notepad_window_is_not_found(
    mock_popen,
    mock_get_windows,
    mock_active_window,
    mock_write,
):
    mock_get_windows.return_value = []

    with (
        patch("vision_agent.windows_actions.time.monotonic",
              side_effect=[0, 11]),
        patch("vision_agent.windows_actions.time.sleep"),
    ):
        result = open_notepad_and_type("Do not type this")

    assert "could not find the new Notepad window" in result
    mock_write.assert_not_called()
    mock_active_window.assert_not_called()


@patch("vision_agent.windows_actions.pyautogui.write")
@patch("vision_agent.windows_actions.pygetwindow.getActiveWindow")
@patch("vision_agent.windows_actions.pygetwindow.getAllWindows")
@patch("vision_agent.windows_actions.subprocess.Popen")
def test_does_not_type_if_notepad_is_not_active(
    mock_popen,
    mock_get_windows,
    mock_active_window,
    mock_write,
):
    notepad = Mock()
    notepad._hWnd = 123
    notepad.title = "Untitled - Notepad"
    notepad.activate.return_value = None

    other_window = Mock()
    other_window._hWnd = 456

    mock_get_windows.side_effect = [[], [notepad]]
    mock_active_window.return_value = other_window

    with patch("vision_agent.windows_actions.time.sleep"):
        result = open_notepad_and_type("Do not type this")

    assert "could not verify Notepad is active" in result
    mock_write.assert_not_called()
