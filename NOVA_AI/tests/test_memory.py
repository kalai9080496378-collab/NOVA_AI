"""
Unit Tests for NOVA Memory Manager
Tests user-controlled memory creation, retrieval, updates, deletions, and safety.
Uses an isolated temporary database for each test run.
"""

import pytest
from pathlib import Path

from database.db import init_db
from memory.memory_manager import (
    save_memory,
    get_all_memories,
    get_memory_by_id,
    update_memory,
    delete_memory,
    clear_all_memories,
    search_memories,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Creates a temporary isolated SQLite database for testing."""
    db_file = tmp_path / "test_memory_nova.db"
    init_db(db_file)
    return db_file


def test_save_memory_success(temp_db: Path):
    """Verify explicit memory creation."""
    mem = save_memory(
        content="I prefer writing clean code with type hints.",
        category="Preferences",
        db_path=temp_db,
    )
    assert mem is not None
    assert mem["id"] == 1
    assert mem["category"] == "Preferences"
    assert "type hints" in mem["content"]


def test_save_memory_empty_content_raises_error(temp_db: Path):
    """Verify empty memory content raises ValueError."""
    with pytest.raises(ValueError, match="Memory content cannot be empty"):
        save_memory(content="   ", db_path=temp_db)


def test_get_memories_filtering(temp_db: Path):
    """Verify category filtering."""
    save_memory("Study Deep Learning on weekends", category="Academic", db_path=temp_db)
    save_memory("My thesis advisor is Dr. Smith", category="Academic", db_path=temp_db)
    save_memory("Favorite IDE theme is dark navy", category="Preferences", db_path=temp_db)

    academic = get_all_memories(category="Academic", db_path=temp_db)
    assert len(academic) == 2

    prefs = get_all_memories(category="Preferences", db_path=temp_db)
    assert len(prefs) == 1

    all_mem = get_all_memories(db_path=temp_db)
    assert len(all_mem) == 3


def test_search_memories(temp_db: Path):
    """Verify keyword search in memory contents."""
    save_memory("Remember to use PyTorch for CNN experiments", category="Academic", db_path=temp_db)
    save_memory("Buy coffee beans", category="Personal", db_path=temp_db)

    results = search_memories("PyTorch", db_path=temp_db)
    assert len(results) == 1
    assert "PyTorch" in results[0]["content"]

    none_results = search_memories("TensorFlow", db_path=temp_db)
    assert len(none_results) == 0


def test_update_memory(temp_db: Path):
    """Verify updating memory content."""
    mem = save_memory("Original thought", category="Drafts", db_path=temp_db)
    mem_id = mem["id"]

    success = update_memory(mem_id, content="Refined thought", category="Final", db_path=temp_db)
    assert success is True

    updated = get_memory_by_id(mem_id, db_path=temp_db)
    assert updated["content"] == "Refined thought"
    assert updated["category"] == "Final"


def test_delete_individual_memory(temp_db: Path):
    """Verify deleting a single memory."""
    mem = save_memory("Temporary note", db_path=temp_db)
    mem_id = mem["id"]

    success = delete_memory(mem_id, db_path=temp_db)
    assert success is True
    assert get_memory_by_id(mem_id, db_path=temp_db) is None


def test_clear_all_memories(temp_db: Path):
    """Verify deleting all memories."""
    save_memory("Note 1", db_path=temp_db)
    save_memory("Note 2", db_path=temp_db)
    save_memory("Note 3", db_path=temp_db)

    deleted_count = clear_all_memories(db_path=temp_db)
    assert deleted_count == 3
    assert len(get_all_memories(db_path=temp_db)) == 0
