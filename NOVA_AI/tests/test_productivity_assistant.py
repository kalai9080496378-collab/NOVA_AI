"""
Automated Test Suite for NOVA Smart Productivity Assistant (Phase 5)
Tests:
- Task management (categories, due times, estimated duration, completed_at, reopen)
- Database schema migration & row preservation
- Smart reminders (creation, recurrence, pause/resume, trigger checks)
- Daily planner (time blocking, plan saving, retrieval)
- Smart suggestions (explainable rankings, AI attribution)
- Analytics & completion trends
"""

import os
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
import pytest

from database.db import init_db, get_connection
from tools.task_manager import (
    create_task,
    get_tasks,
    get_task_by_id,
    update_task,
    complete_task,
    reopen_task,
    delete_task,
    get_task_statistics,
    validate_due_time,
)
from tools.reminder_manager import (
    create_reminder,
    get_reminders,
    get_reminder_by_id,
    update_reminder,
    pause_reminder,
    resume_reminder,
    dismiss_reminder,
    delete_reminder,
    check_and_trigger_due_reminders,
    get_reminder_statistics,
)
from modules.planner_and_suggestions import (
    get_smart_task_suggestions,
    generate_daily_plan,
    save_daily_plan,
    get_daily_plan,
)
from analytics.dashboard import (
    create_completion_trend_chart,
    create_category_distribution_chart,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provides a fresh, isolated SQLite database initialized with schema and migrations."""
    db_file = tmp_path / "test_productivity.db"
    init_db(db_file)
    return db_file


# -----------------------------------------------------------------------------
# 1. Task Management & Validation Tests
# -----------------------------------------------------------------------------

def test_create_task_with_time_category_and_duration(temp_db):
    task = create_task(
        title="Complete Machine Learning Lab",
        description="Implement ResNet from scratch",
        priority="High",
        due_date="2026-10-25",
        due_time="14:30",
        category="Academic",
        estimated_duration=90,
        db_path=temp_db,
    )
    assert task is not None
    assert task["title"] == "Complete Machine Learning Lab"
    assert task["priority"] == "High"
    assert task["due_date"] == "2026-10-25"
    assert task["due_time"] == "14:30"
    assert task["category"] == "Academic"
    assert task["estimated_duration"] == 90
    assert task["status"] == "pending"
    assert task["completed_at"] is None


def test_validate_due_time_formats():
    assert validate_due_time("09:30") == "09:30"
    assert validate_due_time("9:05") == "09:05"
    assert validate_due_time("23:59") == "23:59"
    assert validate_due_time(None) is None
    assert validate_due_time("") is None

    with pytest.raises(ValueError):
        validate_due_time("25:00")
    with pytest.raises(ValueError):
        validate_due_time("12:65")
    with pytest.raises(ValueError):
        validate_due_time("invalid_time")


def test_complete_and_reopen_task(temp_db):
    task = create_task(title="Submit grant proposal", db_path=temp_db)
    assert task["status"] == "pending"

    # Complete
    ok = complete_task(task["id"], db_path=temp_db)
    assert ok is True
    updated = get_task_by_id(task["id"], db_path=temp_db)
    assert updated["status"] == "completed"
    assert updated["completed_at"] is not None

    # Reopen
    reopened = reopen_task(task["id"], db_path=temp_db)
    assert reopened is True
    after_reopen = get_task_by_id(task["id"], db_path=temp_db)
    assert after_reopen["status"] == "pending"
    assert after_reopen["completed_at"] is None


def test_get_tasks_filtering_by_category(temp_db):
    create_task(title="Read paper", category="Academic", db_path=temp_db)
    create_task(title="Write code", category="Work", db_path=temp_db)
    create_task(title="Buy groceries", category="Personal", db_path=temp_db)

    academic_tasks = get_tasks(category="Academic", db_path=temp_db)
    assert len(academic_tasks) == 1
    assert academic_tasks[0]["title"] == "Read paper"

    all_tasks = get_tasks(db_path=temp_db)
    assert len(all_tasks) == 3


def test_database_migration_preserves_legacy_tasks(tmp_path: Path):
    """
    Simulates a legacy Phase 1 database with the old tasks schema (no category, due_time, etc.),
    inserts a row, runs init_db (migration), and verifies data was preserved.
    """
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_file))
    with conn:
        conn.execute("""
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT NOT NULL DEFAULT 'Medium',
                due_date TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            INSERT INTO tasks (title, description, priority, due_date, status, created_at)
            VALUES ('Legacy Task 1', 'Important pre-existing work', 'High', '2026-10-20', 'pending', '2026-10-18T10:00:00Z')
        """)
    conn.close()

    # Run migration via init_db
    init_db(db_file)

    # Verify column existence and row preservation
    retrieved = get_task_by_id(1, db_path=db_file)
    assert retrieved is not None
    assert retrieved["title"] == "Legacy Task 1"
    assert retrieved["priority"] == "High"
    assert retrieved["category"] == "General"  # Default migration value
    assert retrieved["estimated_duration"] == 60


# -----------------------------------------------------------------------------
# 2. Smart Reminders Tests
# -----------------------------------------------------------------------------

def test_create_and_manage_reminders(temp_db):
    rem = create_reminder(
        title="Weekly research sync",
        reminder_date="2026-10-21",
        reminder_time="10:00",
        recurrence="weekly",
        db_path=temp_db,
    )
    assert rem is not None
    assert rem["title"] == "Weekly research sync"
    assert rem["reminder_date"] == "2026-10-21"
    assert rem["reminder_time"] == "10:00"
    assert rem["recurrence"] == "weekly"
    assert rem["status"] == "active"

    # Pause
    pause_reminder(rem["id"], db_path=temp_db)
    paused = get_reminder_by_id(rem["id"], db_path=temp_db)
    assert paused["status"] == "paused"

    # Resume
    resume_reminder(rem["id"], db_path=temp_db)
    resumed = get_reminder_by_id(rem["id"], db_path=temp_db)
    assert resumed["status"] == "active"

    # Delete
    del_ok = delete_reminder(rem["id"], db_path=temp_db)
    assert del_ok is True
    assert get_reminder_by_id(rem["id"], db_path=temp_db) is None


def test_check_and_trigger_due_reminders_recurrence(temp_db):
    # One-time reminder that is due
    r1 = create_reminder(
        title="Doctor appointment",
        reminder_date="2026-10-18",
        reminder_time="09:00",
        recurrence="none",
        db_path=temp_db,
    )
    # Recurring daily reminder that is due
    r2 = create_reminder(
        title="Daily Standup",
        reminder_date="2026-10-18",
        reminder_time="09:30",
        recurrence="daily",
        db_path=temp_db,
    )
    # Future reminder that is NOT due
    r3 = create_reminder(
        title="Future meeting",
        reminder_date="2026-10-25",
        reminder_time="15:00",
        recurrence="none",
        db_path=temp_db,
    )

    # Reference time is 2026-10-18 10:00
    ref_now = datetime(2026, 10, 18, 10, 0)
    triggered = check_and_trigger_due_reminders(reference_dt=ref_now, db_path=temp_db)

    assert len(triggered) == 2
    trig_ids = {t["id"] for t in triggered}
    assert r1["id"] in trig_ids
    assert r2["id"] in trig_ids

    # Verify r1 status is now triggered
    after_r1 = get_reminder_by_id(r1["id"], db_path=temp_db)
    assert after_r1["status"] == "triggered"
    assert after_r1["last_triggered_at"] is not None

    # Verify r2 status remains active but reminder_date advanced to tomorrow
    after_r2 = get_reminder_by_id(r2["id"], db_path=temp_db)
    assert after_r2["status"] == "active"
    assert after_r2["reminder_date"] == "2026-10-19"
    assert after_r2["last_triggered_at"] is not None


# -----------------------------------------------------------------------------
# 3. Daily Planner & Smart Suggestions Tests
# -----------------------------------------------------------------------------

def test_smart_task_suggestions(temp_db):
    today_str = date.today().isoformat()
    # High priority due today
    t1 = create_task(title="Urgent Thesis Review", priority="High", due_date=today_str, due_time="11:00", db_path=temp_db)
    # Low priority due in 2 weeks
    t2 = create_task(title="Organize bookshelf", priority="Low", due_date="2026-11-05", db_path=temp_db)

    suggestions = get_smart_task_suggestions(limit=2, db_path=temp_db)
    assert len(suggestions) == 2
    assert suggestions[0]["task"]["id"] == t1["id"]
    assert suggestions[0]["ai_generated"] is True
    assert "Due today" in suggestions[0]["recommendation_reason"] or "High Priority" in suggestions[0]["recommendation_reason"]


def test_generate_and_save_daily_plan(temp_db):
    create_task(title="Data Preprocessing", priority="High", estimated_duration=90, db_path=temp_db)
    create_task(title="Model Training", priority="Medium", estimated_duration=60, db_path=temp_db)

    plan = generate_daily_plan(available_hours=4.0, db_path=temp_db)
    assert plan["ai_generated"] is True
    assert plan["total_estimated_minutes"] == 150
    assert len(plan["schedule_blocks"]) == 2
    assert "schedule_blocks" in plan

    # Save to SQLite
    plan_id = save_daily_plan(
        plan_date=plan["plan_date"],
        focus_theme=plan["focus_theme"],
        schedule_blocks=plan["schedule_blocks"],
        total_estimated_minutes=plan["total_estimated_minutes"],
        db_path=temp_db,
    )
    assert plan_id is not None

    # Retrieve saved plan
    saved = get_daily_plan(plan["plan_date"], db_path=temp_db)
    assert saved is not None
    assert saved["total_estimated_minutes"] == 150
    assert len(saved["schedule_blocks"]) == 2


# -----------------------------------------------------------------------------
# 4. Analytics & Completion Trends Tests
# -----------------------------------------------------------------------------

def test_completion_trend_chart_creation(temp_db):
    t1 = create_task(title="T1", db_path=temp_db)
    t2 = create_task(title="T2", db_path=temp_db)
    complete_task(t1["id"], db_path=temp_db)
    complete_task(t2["id"], db_path=temp_db)

    tasks = get_tasks(db_path=temp_db)
    fig = create_completion_trend_chart(tasks)
    assert fig is not None


def test_category_distribution_chart_creation(temp_db):
    create_task(title="Paper", category="Academic", db_path=temp_db)
    create_task(title="Bugfix", category="Work", db_path=temp_db)

    stats = get_task_statistics(db_path=temp_db)
    fig = create_category_distribution_chart(stats["category_counts"])
    assert fig is not None
