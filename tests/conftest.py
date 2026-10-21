"""
Global pytest configuration and test environment isolation.
Ensures tests run in isolated temporary databases and never pollute the production data/nova.db.
"""
import pytest
from pathlib import Path
from database.db import init_db


@pytest.fixture(autouse=True)
def isolate_test_database(tmp_path: Path, monkeypatch):
    """
    Autouse fixture that ensures every test runs against a fresh, temporary SQLite database.
    Prevents tests from mutating or polluting production data/nova.db.
    """
    test_db = tmp_path / "test_session_nova.db"
    init_db(test_db)
    monkeypatch.setenv("DATABASE_PATH", str(test_db))
    monkeypatch.setattr("database.db.DATABASE_PATH", test_db)
    monkeypatch.setattr("nova_core.config.DATABASE_PATH", test_db)
    yield test_db
