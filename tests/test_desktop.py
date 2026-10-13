
# from __future__ import annotations

# import os
# import threading

# os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# import pytest
# from PySide6.QtCore import QThread
# from PySide6.QtWidgets import QApplication

# from vision_agent.desktop import AgentWorker, VisionAgentWindow


# @pytest.fixture(scope="session")
# def qt_app():
#     app = QApplication.instance() or QApplication([])
#     yield app


# class FakeApplication:
#     def __init__(self):
#         self.closed = False
#         self.observation_active = False
#         self.messages = []

#     @property
#     def status(self):
#         class Status:
#             def __init__(self, active):
#                 self.observation_active = active
#                 self.last_error = None

#         return Status(self.observation_active)

#     def ask(self, message):
#         self.messages.append(message)
#         return f"Reply to: {message}"

#     def start_screen_observation(self):
#         self.observation_active = True

#     def stop_screen_observation(self):
#         self.observation_active = False

#     def close(self):
#         self.closed = True


# def test_window_does_not_start_observation_automatically(qt_app):
#     app = FakeApplication()
#     window = VisionAgentWindow(app)

#     assert not app.observation_active
#     assert window.start_observation_button.isEnabled()
#     assert not window.stop_observation_button.isEnabled()

#     window.close()


# def test_window_does_not_close_injected_application(qt_app):
#     app = FakeApplication()
#     window = VisionAgentWindow(app)

#     window.close()

#     assert not app.closed


# def test_worker_returns_agent_response():
#     app = FakeApplication()
#     worker = AgentWorker(app, "ask", "Hello")

#     results = []
#     errors = []

#     worker.finished.connect(
#         lambda message, result: results.append((message, result))
#     )
#     worker.failed.connect(errors.append)

#     worker.run()

#     assert results == [("Hello", "Reply to: Hello")]
#     assert errors == []
#     assert app.messages == ["Hello"]


# def test_worker_reports_application_errors():
#     class BrokenApplication(FakeApplication):
#         def ask(self, message):
#             raise RuntimeError("test failure")

#     app = BrokenApplication()
#     worker = AgentWorker(app, "ask", "Hello")

#     errors = []
#     worker.failed.connect(errors.append)

#     worker.run()

#     assert errors == ["RuntimeError: test failure"]




# def test_worker_runs_outside_the_main_thread(qt_app):
#     app = FakeApplication()
#     main_thread = threading.current_thread()
#     worker_threads = []

#     class WorkerThread(QThread):
#         def run(self):
#             worker = AgentWorker(app, "ask", "thread test")
#             worker_threads.append(threading.current_thread())
#             worker.run()

#     thread = WorkerThread()
#     thread.start()

#     assert thread.wait(3000), "Worker thread did not stop in time"
#     assert worker_threads
#     assert worker_threads[0] is not main_thread
#     assert app.messages == ["thread test"]



from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from vision_agent.desktop import VisionAgentWindow


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


class FakeApplication:
    def __init__(self):
        self.closed = False
        self.observation_active = False
        self.messages = []
        self.ask_started = threading.Event()
        self.ask_release = threading.Event()
        self.block_ask = False
        self.ask_thread = None

    @property
    def status(self):
        class Status:
            def __init__(self, active):
                self.observation_active = active
                self.last_error = None

        return Status(self.observation_active)

    def ask(self, message):
        self.ask_thread = threading.current_thread()
        self.messages.append(message)
        self.ask_started.set()

        if self.block_ask:
            self.ask_release.wait(timeout=5)

        return f"Reply to: {message}"

    def start_screen_observation(self):
        self.observation_active = True

    def stop_screen_observation(self):
        self.observation_active = False

    def close(self):
        self.closed = True


def wait_until(qt_app, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)

    qt_app.processEvents()
    return predicate()


def test_window_does_not_start_observation_automatically(qt_app):
    app = FakeApplication()
    window = VisionAgentWindow(app)

    assert not app.observation_active
    assert window.start_observation_button.isEnabled()
    assert not window.stop_observation_button.isEnabled()

    window.close()


def test_window_does_not_close_injected_application(qt_app):
    app = FakeApplication()
    window = VisionAgentWindow(app)

    window.close()

    assert not app.closed


def test_send_runs_agent_in_background_and_displays_reply(qt_app):
    app = FakeApplication()
    window = VisionAgentWindow(app)
    main_thread = threading.current_thread()

    window.message_input.setText("Hello")
    window._send_message()

    assert wait_until(
        qt_app,
        lambda: (
            "Vision Agent: Reply to: Hello"
            in window.chat_history.toPlainText()
            and window.send_button.isEnabled()
        ),
    )

    assert app.messages == ["Hello"]
    assert app.ask_thread is not main_thread
    assert window.send_button.isEnabled()

    window.close()




def test_failed_request_displays_error_and_restores_controls(qt_app):
    class BrokenApplication(FakeApplication):
        def ask(self, message):
            raise RuntimeError("test failure")

    app = BrokenApplication()
    window = VisionAgentWindow(app)

    window.message_input.setText("Hello")
    window._send_message()

    assert wait_until(
        qt_app,
        lambda: (
            "Error: RuntimeError: test failure"
            in window.chat_history.toPlainText()
            and window.send_button.isEnabled()
            and window.message_input.isEnabled()
        ),
    )

    window.close()




def test_close_waits_for_running_operation(qt_app):
    app = FakeApplication()
    app.block_ask = True
    window = VisionAgentWindow(app)

    window.message_input.setText("Hello")
    window._send_message()

    assert app.ask_started.wait(timeout=2)

    window.close()

    assert not app.closed

    app.ask_release.set()

    assert wait_until(qt_app, lambda: app.closed or not window.isVisible())
