
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
