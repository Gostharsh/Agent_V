
from __future__ import annotations

import subprocess
import time

import pyautogui
import pygetwindow


NOTEPAD_PATH = r"C:\WINDOWS\system32\notepad.exe"
MAX_TEXT_LENGTH = 5_000


def open_notepad_and_type(text: str) -> str:
    """Open Notepad and type text only after verifying its window."""
    if not isinstance(text, str):
        return "Windows action error: text must be a string."

    if not text.strip():
        return "Windows action error: text cannot be empty."

    if len(text) > MAX_TEXT_LENGTH:
        return (
            "Windows action error: text exceeds "
            f"the {MAX_TEXT_LENGTH}-character limit."
        )

    try:
        existing_windows = {
            window._hWnd
            for window in pygetwindow.getAllWindows()
        }

        subprocess.Popen([NOTEPAD_PATH])

        target_window = None
        deadline = time.monotonic() + 10

        while time.monotonic() < deadline:
            for window in pygetwindow.getAllWindows():
                if (
                    window._hWnd not in existing_windows
                    and "notepad" in window.title.lower()
                ):
                    target_window = window
                    break

            if target_window is not None:
                break

            time.sleep(0.2)

        if target_window is None:
            return "Windows action error: could not find the new Notepad window."

        target_window.activate()
        time.sleep(0.3)

        active_window = pygetwindow.getActiveWindow()
        if (
            active_window is None
            or active_window._hWnd != target_window._hWnd
        ):
            return "Windows action error: could not verify Notepad is active."

        pyautogui.write(text, interval=0.01)
        return "Opened Notepad and sent the requested text."

    except Exception:
        return "Windows action error: could not complete the Notepad action."


def click_screen_position(x: int, y: int) -> str:
    """Click a validated screen coordinate."""
    if type(x) is not int or type(y) is not int:
        return "Windows action error: coordinates must be integers."

    try:
        screen_width, screen_height = pyautogui.size()
    except Exception:
        return "Windows action error: could not determine screen dimensions."

    if not (0 <= x < screen_width and 0 <= y < screen_height):
        return "Windows action error: coordinates are outside the screen."

    try:
        pyautogui.click(x=x, y=y)
        return f"Clicked screen position ({x}, {y})."
    except Exception:
        return "Windows action error: click failed."




ALLOWED_KEYS = {
    "ENTER",
    "ESC",
    "TAB",
    "SPACE",
    "BACKSPACE",
    "DELETE",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
}

ALLOWED_SHORTCUTS = {
    "CTRL+A": ["ctrl", "a"],
    "CTRL+C": ["ctrl", "c"],
    "CTRL+V": ["ctrl", "v"],
    "CTRL+S": ["ctrl", "s"],
    "CTRL+Z": ["ctrl", "z"],
    "CTRL+Y": ["ctrl", "y"],
}


def press_keyboard_action(key: str, expected_window_title: str) -> str:
    """Press an approved key only when the expected window is active."""
    if not isinstance(key, str) or not isinstance(expected_window_title, str):
        return "Windows action error: key and window title must be strings."

    key = key.strip().upper()
    expected_window_title = expected_window_title.strip().casefold()

    if not expected_window_title:
        return "Windows action error: expected window title cannot be empty."

    if key not in ALLOWED_KEYS and key not in ALLOWED_SHORTCUTS:
        return "Windows action error: key or shortcut is not permitted."

    try:
        active_window = pygetwindow.getActiveWindow()
        if (
            active_window is None
            or expected_window_title not in active_window.title.casefold()
        ):
            return "Windows action error: expected window is not active."

        if key in ALLOWED_SHORTCUTS:
            pyautogui.hotkey(*ALLOWED_SHORTCUTS[key])
        else:
            pyautogui.press(key.lower())

        return f"Sent {key} to the active {active_window.title} window."

    except Exception:
        return "Windows action error: keyboard action failed."
