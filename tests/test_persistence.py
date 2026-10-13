from pathlib import Path

import pytest

from vision_agent.persistence import PersistentCheckpointer, default_database_path


def test_default_database_path_is_in_user_application_data(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    path = default_database_path()

    assert path == tmp_path / "VisionAgent" / "checkpoints.sqlite3"


def test_persistent_checkpointer_creates_database(tmp_path):
    database_path = tmp_path / "nested" / "checkpoints.sqlite3"

    checkpointer = PersistentCheckpointer(database_path)
    try:
        assert database_path.is_file()
        assert checkpointer.saver is not None
    finally:
        checkpointer.close()


def test_persistent_checkpointer_close_is_idempotent(tmp_path):
    checkpointer = PersistentCheckpointer(tmp_path / "checkpoints.sqlite3")

    checkpointer.close()
    checkpointer.close()

    with pytest.raises(RuntimeError, match="closed"):
        checkpointer.__enter__()
