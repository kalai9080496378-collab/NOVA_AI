"""
Unit Tests for NOVA Task Manager
Tests task CRUD operations, input validations, filtering, and statistical summaries.
Uses an isolated temporary database for each test run.
"""

import pytest
from pathlib import Path
from datetime import date, timedelta

from database.db import init_db, get_connection
from tools.task_manager import (
    create_task,
    get_tasks,
    get_task_by_id,
    update_task,
    complete_task,
    delete_task,
    get_task_statistics,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Creates a temporary isolated SQLite database for testing."""
    db_file = tmp_path / "test_nova.db"
    init_db(db_file)
    return db_file


def test_create_task_success(temp_db: Path):
    """Verify task creation with valid inputs."""
    task = create_task(
        title="Complete Assignment 1",
        description="Data Science classification project",
        priority="High",
        due_date="2026-10-30",
        db_path=temp_db,
    )
    assert task is not None
    assert task["id"] == 1
    assert task["title"] == "Complete Assignment 1"
    assert task["priority"] == "High"
    assert task["due_date"] == "2026-10-30"
    assert task["status"] == "pending"


def test_create_task_empty_title_raises_error(temp_db: Path):
    """Verify empty title raises ValueError."""
    with pytest.raises(ValueError, match="Task title cannot be empty"):
        create_task(title="   ", db_path=temp_db)


def test_create_task_invalid_due_date_format(temp_db: Path):
    """Verify invalid date string format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid due date format"):
        create_task(title="Study for exam", due_date="30-10-2026", db_path=temp_db)


def test_get_tasks_filtering(temp_db: Path):
    """Verify filtering tasks by status and priority."""
    t1 = create_task(title="Task A", priority="Low", db_path=temp_db)
    t2 = create_task(title="Task B", priority="High", db_path=temp_db)
    t3 = create_task(title="Task C", priority="High", db_path=temp_db)

    complete_task(t1["id"], db_path=temp_db)

    # Filter by pending
    pending = get_tasks(status="pending", db_path=temp_db)
    assert len(pending) == 2
    assert all(t["status"] == "pending" for t in pending)

    # Filter by completed
    completed = get_tasks(status="completed", db_path=temp_db)
    assert len(completed) == 1
    assert completed[0]["title"] == "Task A"

    # Filter by priority
    high_prio = get_tasks(priority="High", db_path=temp_db)
    assert len(high_prio) == 2


def test_update_task(temp_db: Path):
    """Verify updating title, priority, and description."""
    task = create_task(title="Original Title", priority="Low", db_path=temp_db)
    task_id = task["id"]

    success = update_task(
        task_id=task_id,
        title="Updated Title",
        priority="High",
        description="Added notes",
        db_path=temp_db,
    )
    assert success is True

    updated = get_task_by_id(task_id, db_path=temp_db)
    assert updated["title"] == "Updated Title"
    assert updated["priority"] == "High"
    assert updated["description"] == "Added notes"


def test_delete_task(temp_db: Path):
    """Verify deleting a task removes it permanently."""
    task = create_task(title="To be deleted", db_path=temp_db)
    task_id = task["id"]

    del_success = delete_task(task_id, db_path=temp_db)
    assert del_success is True

    assert get_task_by_id(task_id, db_path=temp_db) is None
    # Deleting non-existent task returns False
    assert delete_task(9999, db_path=temp_db) is False


def test_task_statistics_accuracy(temp_db: Path):
    """Verify real calculation of task metrics without fake data."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    # 1 overdue pending task
    create_task(title="Overdue Task", due_date=yesterday, priority="High", db_path=temp_db)
    # 1 upcoming pending task
    create_task(title="Future Task", due_date=tomorrow, priority="Medium", db_path=temp_db)
    # 1 completed task
    t3 = create_task(title="Done Task", priority="Low", db_path=temp_db)
    complete_task(t3["id"], db_path=temp_db)

    stats = get_task_statistics(db_path=temp_db)
    assert stats["total"] == 3
    assert stats["pending"] == 2
    assert stats["completed"] == 1
    assert stats["overdue"] == 1
    assert stats["high_priority_pending"] == 1
    assert stats["completion_rate"] == 33.3
