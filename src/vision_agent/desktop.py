
# from __future__ import annotations

# import sys

# from PySide6.QtCore import QObject, QThread, Signal, Slot
# from PySide6.QtWidgets import (
#     QApplication,
#     QLabel,
#     QLineEdit,
#     QMainWindow,
#     QPushButton,
#     QPlainTextEdit,
#     QVBoxLayout,
#     QWidget,
# )

# from vision_agent.application import VisionAgentApplication


# class AgentWorker(QObject):
#     """Run one application operation outside the GUI thread."""

#     finished = Signal(str, str)
#     failed = Signal(str)
#     done = Signal()

#     def __init__(
#         self,
#         app: VisionAgentApplication,
#         operation: str,
#         message: str = "",
#     ) -> None:
#         super().__init__()
#         self._app = app
#         self.operation = operation
#         self.message = message

#     @Slot()
#     def run(self) -> None:
#         print("[Vision UI] Worker started:", self.operation, flush=True)
#         try:
#             if self.operation == "ask":
#                 result = self._app.ask(self.message)
#                 print("[Vision UI] Agent returned", flush=True)
#                 self.finished.emit(self.message, result)
#             elif self.operation == "start_observation":
#                 self._app.start_screen_observation()
#                 self.finished.emit("", "Screen observation started.")
#             elif self.operation == "stop_observation":
#                 self._app.stop_screen_observation()
#                 self.finished.emit("", "Screen observation stopped.")
#             else:
#                 raise ValueError(f"Unknown operation: {self.operation}")
#         except Exception as exc:
#             print(f"[Vision UI] Worker failed: {type(exc).__name__}: {exc}", flush=True)
#             self.failed.emit(f"{type(exc).__name__}: {exc}")
#         finally:
#             print("[Vision UI] Worker finished; emitting done", flush=True)
#             self.done.emit()


# class VisionAgentWindow(QMainWindow):
#     """Minimal chat interface for Vision Agent."""

#     def __init__(
#         self,
#         app: VisionAgentApplication | None = None,
#     ) -> None:
#         super().__init__()
#         self._agent_app = (
#             app if app is not None else VisionAgentApplication()
#         )
#         self._owns_app = app is None
#         self._thread: QThread | None = None
#         self._closing = False
#         self._app_closed = False

#         self.setWindowTitle("Vision Agent")
#         self.resize(760, 600)

#         self.status_label = QLabel(
#             "Ready — screen observation is off."
#         )
#         self.chat_history = QPlainTextEdit()
#         self.chat_history.setReadOnly(True)
#         self.message_input = QLineEdit()
#         self.message_input.setPlaceholderText("Ask Vision Agent...")
#         self.send_button = QPushButton("Send")
#         self.start_observation_button = QPushButton(
#             "Start observation"
#         )
#         self.stop_observation_button = QPushButton(
#             "Stop observation"
#         )

#         layout = QVBoxLayout()
#         layout.addWidget(self.status_label)
#         layout.addWidget(self.chat_history)
#         layout.addWidget(self.message_input)
#         layout.addWidget(self.send_button)
#         layout.addWidget(self.start_observation_button)
#         layout.addWidget(self.stop_observation_button)

#         central = QWidget()
#         central.setLayout(layout)
#         self.setCentralWidget(central)

#         self.send_button.clicked.connect(self._send_message)
#         self.message_input.returnPressed.connect(self._send_message)
#         self.start_observation_button.clicked.connect(
#             self._start_observation
#         )
#         self.stop_observation_button.clicked.connect(
#             self._stop_observation
#         )

#         self._sync_observation_controls()


#     def _start_operation(
#         self,
#         operation: str,
#         message: str = "",
#     ) -> None:
#         if self._thread is not None or self._closing:
#             return

#         thread = QThread(self)
#         worker = AgentWorker(self._agent_app, operation, message)
#         worker.moveToThread(thread)

#         thread.started.connect(worker.run)
#         worker.finished.connect(self._operation_succeeded)
#         worker.failed.connect(self._operation_failed)
#         worker.done.connect(thread.quit)
#         thread.finished.connect(worker.deleteLater)
#         thread.finished.connect(self._operation_finished)
#         thread.finished.connect(thread.deleteLater)

#         self._thread = thread
#         self._set_busy(True)
#         self.status_label.setText("Working...")

#         print(
#             f"[Vision UI] Starting thread for: {operation}",
#             flush=True,
#         )

#         # Start the worker after Qt has returned control to its event loop.
#         from PySide6.QtCore import QTimer
#         QTimer.singleShot(0, thread.start)


#     @Slot()
#     def _send_message(self) -> None:
#         print("[Vision UI] Send handler triggered", flush=True)
#         message = self.message_input.text().strip()
#         if not message or self._thread is not None or self._closing:
#             return

#         self.message_input.clear()
#         self.chat_history.appendPlainText(f"You: {message}")
#         self._start_operation("ask", message)

#     @Slot()
#     def _start_observation(self) -> None:
#         self._start_operation("start_observation")

#     @Slot()
#     def _stop_observation(self) -> None:
#         self._start_operation("stop_observation")

#     @Slot(str, str)
#     def _operation_succeeded(self, message: str, result: str) -> None:
#         if message:
#             self.chat_history.appendPlainText(
#                 f"Vision Agent: {result}"
#             )
#         else:
#             self.status_label.setText(result)

#     @Slot(str)
#     def _operation_failed(self, error: str) -> None:
#         self.status_label.setText(f"Error: {error}")
#         self.chat_history.appendPlainText(f"Error: {error}")

#     @Slot()
#     def _operation_finished(self) -> None:
#         self._thread = None

#         if self._closing:
#             # Re-enter closeEvent now that the worker thread has ended.
#             self.close()
#             return

#         self._set_busy(False)
#         self._sync_observation_controls()

#         if self.status_label.text().startswith("Working"):
#             self.status_label.setText(
#                 "Ready — screen observation is "
#                 + (
#                     "on."
#                     if self._agent_app.status.observation_active
#                     else "off."
#                 )
#             )

#     def _set_busy(self, busy: bool) -> None:
#         self.message_input.setEnabled(not busy)
#         self.send_button.setEnabled(not busy)
#         self.start_observation_button.setEnabled(not busy)
#         self.stop_observation_button.setEnabled(not busy)

#     def _sync_observation_controls(self) -> None:
#         active = self._agent_app.status.observation_active
#         self.start_observation_button.setEnabled(
#             not active and self._thread is None and not self._closing
#         )
#         self.stop_observation_button.setEnabled(
#             active and self._thread is None and not self._closing
#         )

#     def closeEvent(self, event) -> None:
#         if self._thread is not None:
#             self._closing = True
#             self._set_busy(True)
#             self.status_label.setText(
#                 "Waiting for the current operation to finish..."
#             )
#             event.ignore()
#             return

#         self._close_owned_application()
#         event.accept()

#     def _close_owned_application(self) -> None:
#         if self._owns_app and not self._app_closed:
#             self._app_closed = True
#             self._agent_app.close()


# def main() -> int:
#     qt_app = QApplication.instance() or QApplication(sys.argv)
#     window = VisionAgentWindow()
#     window.show()
#     return qt_app.exec()


# if __name__ == "__main__":
#     raise SystemExit(main())




from __future__ import annotations

import sys
from concurrent.futures import Future, ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from vision_agent.application import VisionAgentApplication
from vision_agent.graph import close_agent

class OperationSignals(QObject):
    succeeded = Signal(str, str)
    failed = Signal(str)
    finished = Signal()


class VisionAgentWindow(QMainWindow):
    """Minimal chat interface for Vision Agent."""

    def __init__(
        self,
        app: VisionAgentApplication | None = None,
    ) -> None:
        super().__init__()
        self._agent_app = app if app is not None else VisionAgentApplication()
        self._owns_app = app is None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._signals: OperationSignals | None = None
        self._closing = False
        self._app_closed = False

        self.setWindowTitle("Vision Agent")
        self.resize(760, 600)

        self.status_label = QLabel(
            "Ready — screen observation is off."
        )
        self.chat_history = QPlainTextEdit()
        self.chat_history.setReadOnly(True)
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask Vision Agent...")
        self.send_button = QPushButton("Send")
        self.start_observation_button = QPushButton("Start observation")
        self.stop_observation_button = QPushButton("Stop observation")

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.chat_history)
        layout.addWidget(self.message_input)
        layout.addWidget(self.send_button)
        layout.addWidget(self.start_observation_button)
        layout.addWidget(self.stop_observation_button)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.send_button.clicked.connect(self._send_message)
        self.message_input.returnPressed.connect(self._send_message)
        self.start_observation_button.clicked.connect(self._start_observation)
        self.stop_observation_button.clicked.connect(self._stop_observation)

        self._sync_observation_controls()


    def _start_operation(self, operation: str, message: str = "") -> None:
        if self._future is not None or self._closing:
            return

        signals = OperationSignals(self)
        self._signals = signals
        signals.succeeded.connect(self._operation_succeeded)
        signals.failed.connect(self._operation_failed)
        signals.finished.connect(self._operation_finished)

        self._set_busy(True)
        self.status_label.setText("Working...")

        def run_operation() -> tuple[str, str]:
            if operation == "ask":
                if self._agent_app.status.observation_active:
                    self._agent_app.poll_screen_observation()
                return message, self._agent_app.ask(message)
            if operation == "start_observation":
                self._agent_app.start_screen_observation()
                return "", "Screen observation started."
            if operation == "stop_observation":
                self._agent_app.stop_screen_observation()
                return "", "Screen observation stopped."
            raise ValueError(f"Unknown operation: {operation}")

        def task() -> None:
            try:
                reply_message, result = run_operation()
            except Exception as exc:
                signals.failed.emit(f"{type(exc).__name__}: {exc}")
            else:
                signals.succeeded.emit(reply_message, result)
            finally:
                signals.finished.emit()

        self._future = self._executor.submit(task)



    @Slot()
    def _send_message(self) -> None:
        message = self.message_input.text().strip()
        if not message or self._future is not None or self._closing:
            return

        self.message_input.clear()
        self.chat_history.appendPlainText(f"You: {message}")
        self._start_operation("ask", message)

    @Slot()
    def _start_observation(self) -> None:
        self._start_operation("start_observation")

    @Slot()
    def _stop_observation(self) -> None:
        self._start_operation("stop_observation")

    @Slot(str, str)
    def _operation_succeeded(self, message: str, result: str) -> None:
        if message:
            self.chat_history.appendPlainText(f"Vision Agent: {result}")
        else:
            self.status_label.setText(result)

    @Slot(str)
    def _operation_failed(self, error: str) -> None:
        self.status_label.setText(f"Error: {error}")
        self.chat_history.appendPlainText(f"Error: {error}")

    @Slot()
    def _operation_finished(self) -> None:
        self._future = None
        self._signals = None

        if self._closing:
            self._close_owned_application()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self.close()
            return

        self._set_busy(False)
        self._sync_observation_controls()

        if self.status_label.text() == "Working...":
            active = self._agent_app.status.observation_active
            self.status_label.setText(
                "Ready — screen observation is "
                + ("on." if active else "off.")
            )

    def _set_busy(self, busy: bool) -> None:
        self.message_input.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.start_observation_button.setEnabled(not busy)
        self.stop_observation_button.setEnabled(not busy)

    def _sync_observation_controls(self) -> None:
        active = self._agent_app.status.observation_active
        idle = self._future is None and not self._closing
        self.start_observation_button.setEnabled(idle and not active)
        self.stop_observation_button.setEnabled(idle and active)

    def closeEvent(self, event) -> None:
        if self._future is not None:
            self._closing = True
            self._set_busy(True)
            self.status_label.setText(
                "Waiting for the current operation to finish..."
            )
            event.ignore()
            return

        self._close_owned_application()
        self._executor.shutdown(wait=False, cancel_futures=True)
        event.accept()

    def _close_owned_application(self) -> None:
        if self._owns_app and not self._app_closed:
            self._app_closed = True
            self._agent_app.close()



def main() -> int:
    qt_app = QApplication.instance() or QApplication(sys.argv)
    window = VisionAgentWindow()
    window.show()

    try:
        return qt_app.exec()
    finally:
        close_agent()


if __name__ == "__main__":
    raise SystemExit(main())
