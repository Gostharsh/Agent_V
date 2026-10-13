from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def default_database_path() -> Path:
    """Return the per-user database path for Vision Agent."""
    import os

    app_data = os.environ.get("LOCALAPPDATA")
    if not app_data:
        app_data = str(Path.home() / "AppData" / "Local")

    return Path(app_data) / "VisionAgent" / "checkpoints.sqlite3"


class PersistentCheckpointer:
    """Own a SQLite connection and its LangGraph checkpointer."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = (
            Path(database_path)
            if database_path is not None
            else default_database_path()
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
        )
        self._closed = False
        self.saver = SqliteSaver(self._connection)
        self.saver.setup()

    def close(self) -> None:
        """Close the database connection once."""
        if self._closed:
            return
        self._closed = True
        self._connection.close()

    def __enter__(self) -> PersistentCheckpointer:
        if self._closed:
            raise RuntimeError("Persistent checkpointer is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
