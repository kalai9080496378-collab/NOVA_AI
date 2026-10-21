import os
import sys
from pathlib import Path
import pytest
from database.db import init_db

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(autouse=True)
def isolate_test_database(tmp_path: Path, monkeypatch):
    """
    Autouse fixture ensuring EVERY test run across the entire repository
    runs against a fresh, temporary SQLite database.
    Guarantees zero tests or test runners can ever touch or pollute data/nova.db.
    """
    test_db = tmp_path / "test_session_nova.db"
    init_db(test_db)
    monkeypatch.setenv("DATABASE_PATH", str(test_db))
    monkeypatch.setattr("database.db.DATABASE_PATH", test_db)
    monkeypatch.setattr("nova_core.config.DATABASE_PATH", test_db)
    yield test_db
