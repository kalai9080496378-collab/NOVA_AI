"""
NOVA Smart Reminders Manager
Phase 5: Smart Productivity Assistant
Persistent SQLite reminder scheduling, recurring cycles, overdue checking,
and verifiable notification delivery tracking.
"""

from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from database.db import execute_insert, execute_query, fetch_all, fetch_one, log_action


VALID_RECURRENCES = {"none", "daily", "weekly", "monthly"}
VALID_STATUSES = {"active", "paused", "triggered", "dismissed"}


def validate_reminder_time(time_str: str) -> str:
    """Validates and formats time as HH:MM."""
    cleaned = (time_str or "").strip()
    match = re.match(r"^(\d{1,2}):(\d{2})$", cleaned)
    if not match:
        raise ValueError(f"Invalid reminder time: '{time_str}'. Expected HH:MM (24-hour).")
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time value: '{time_str}'. Hours must be 0-23, minutes 0-59.")
    return f"{hour:02d}:{minute:02d}"


def validate_reminder_date(date_str: str) -> str:
    """Validates date formatted as YYYY-MM-DD."""
    cleaned = (date_str or "").strip()
    try:
        datetime.strptime(cleaned, "%Y-%m-%d")
        return cleaned
    except ValueError:
        raise ValueError(f"Invalid reminder date: '{date_str}'. Expected YYYY-MM-DD.")


def create_reminder(
    title: str,
    reminder_date: Optional[str] = None,
    reminder_time: Optional[str] = None,
    recurrence: str = "none",
    task_id: Optional[int] = None,
    remind_at: Optional[str] = None,
    description: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Creates a new persistent scheduled reminder in SQLite.
    
    Args:
        title: Content or alert text of the reminder.
        reminder_date: 'YYYY-MM-DD'.
        reminder_time: 'HH:MM' (24-hour format).
        recurrence: 'none', 'daily', 'weekly', or 'monthly'.
        task_id: Optional associated task ID.
        remind_at: Optional combined 'YYYY-MM-DD HH:MM' string.
        description: Optional notes/description.
        db_path: Optional database override.
        
    Returns:
        Created reminder dictionary.
    """
    cleaned_title = (title or "").strip()
    if not cleaned_title:
        raise ValueError("Reminder title cannot be empty.")

    if remind_at:
        parts = remind_at.strip().split()
        if len(parts) >= 2:
            reminder_date = parts[0]
            reminder_time = parts[1]
        else:
            raise ValueError(f"Invalid reminder timestamp format: '{remind_at}'. Expected 'YYYY-MM-DD HH:MM'.")
    elif not reminder_date or not reminder_time:
        raise ValueError("Both reminder_date and reminder_time (or remind_at) are required.")

    valid_date = validate_reminder_date(reminder_date)
    valid_time = validate_reminder_time(reminder_time)

    rec = (recurrence or "none").strip().lower()
    if rec not in VALID_RECURRENCES:
        raise ValueError(f"Invalid recurrence: '{recurrence}'. Allowed values: {VALID_RECURRENCES}")

    now = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO reminders (
            task_id, title, reminder_date, reminder_time,
            recurrence, status, last_triggered_at, created_at
        )
        VALUES (?, ?, ?, ?, ?, 'active', NULL, ?)
    """
    rem_id = execute_insert(
        sql,
        (task_id, cleaned_title, valid_date, valid_time, rec, now),
        db_path=db_path,
    )

    log_action("create_reminder", f"Created reminder #{rem_id}: '{cleaned_title}' on {valid_date} at {valid_time}", status="success", db_path=db_path)
    return get_reminder_by_id(rem_id, db_path=db_path)


def get_reminder_by_id(reminder_id: int, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single reminder by its ID, with associated task details if linked."""
    sql = """
        SELECT r.*, t.title AS task_title, t.priority AS task_priority, t.status AS task_status
        FROM reminders r
        LEFT JOIN tasks t ON r.task_id = t.id
        WHERE r.id = ?
    """
    return fetch_one(sql, (reminder_id,), db_path=db_path)


def get_reminders(
    status: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all reminders, optionally filtered by status, ordered by date and time.
    """
    params: List[Any] = []
    sql = """
        SELECT r.*, t.title AS task_title, t.priority AS task_priority, t.status AS task_status
        FROM reminders r
        LEFT JOIN tasks t ON r.task_id = t.id
    """
    if status and status.lower() != "all":
        sql += " WHERE r.status = ?"
        params.append(status.lower().strip())

    sql += " ORDER BY r.reminder_date ASC, r.reminder_time ASC, r.id DESC"
    return fetch_all(sql, tuple(params), db_path=db_path)


def update_reminder(
    reminder_id: int,
    title: Optional[str] = None,
    reminder_date: Optional[str] = None,
    reminder_time: Optional[str] = None,
    recurrence: Optional[str] = None,
    status: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> bool:
    """Updates an existing reminder."""
    existing = get_reminder_by_id(reminder_id, db_path=db_path)
    if not existing:
        return False

    updates = []
    params: List[Any] = []

    if title is not None:
        c_title = title.strip()
        if not c_title:
            raise ValueError("Reminder title cannot be empty.")
        updates.append("title = ?")
        params.append(c_title)

    if reminder_date is not None:
        v_date = validate_reminder_date(reminder_date)
        updates.append("reminder_date = ?")
        params.append(v_date)

    if reminder_time is not None:
        v_time = validate_reminder_time(reminder_time)
        updates.append("reminder_time = ?")
        params.append(v_time)

    if recurrence is not None:
        r = recurrence.strip().lower()
        if r in VALID_RECURRENCES:
            updates.append("recurrence = ?")
            params.append(r)

    if status is not None:
        s = status.strip().lower()
        if s in VALID_STATUSES:
            updates.append("status = ?")
            params.append(s)

    if not updates:
        return True

    params.append(reminder_id)
    sql = f"UPDATE reminders SET {', '.join(updates)} WHERE id = ?"
    execute_query(sql, tuple(params), db_path=db_path)
    log_action("update_reminder", f"Updated reminder #{reminder_id}", status="success", db_path=db_path)
    return True


def pause_reminder(reminder_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Pauses an active reminder schedule."""
    return update_reminder(reminder_id, status="paused", db_path=db_path)


def resume_reminder(reminder_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Resumes a paused reminder back to active."""
    return update_reminder(reminder_id, status="active", db_path=db_path)


def dismiss_reminder(reminder_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Marks a triggered reminder as dismissed."""
    return update_reminder(reminder_id, status="dismissed", db_path=db_path)


def delete_reminder(reminder_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Permanently deletes a reminder from SQLite."""
    sql = "DELETE FROM reminders WHERE id = ?"
    affected = execute_query(sql, (reminder_id,), db_path=db_path)
    if affected > 0:
        log_action("delete_reminder", f"Deleted reminder #{reminder_id}", status="success", db_path=db_path)
        return True
    return False


def check_and_trigger_due_reminders(
    reference_dt: Optional[datetime] = None,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Inspects all active reminders against the reference timestamp (or current local time).
    Triggers due reminders and calculates the next cycle for recurring reminders.
    
    Returns:
        List of reminders triggered in this check cycle.
    """
    ref_now = reference_dt or datetime.now()
    today_str = ref_now.strftime("%Y-%m-%d")
    time_str = ref_now.strftime("%H:%M")
    now_iso = ref_now.isoformat()

    active_reminders = get_reminders(status="active", db_path=db_path)
    triggered_list: List[Dict[str, Any]] = []

    for r in active_reminders:
        r_date = r["reminder_date"]
        r_time = r["reminder_time"]

        is_due = (r_date < today_str) or (r_date == today_str and r_time <= time_str)

        if is_due:
            rec = r.get("recurrence", "none").lower()
            if rec == "daily":
                # Advance to next day
                curr_date = datetime.strptime(r_date, "%Y-%m-%d").date()
                next_date = (max(curr_date, ref_now.date()) + timedelta(days=1)).isoformat()
                execute_query(
                    "UPDATE reminders SET reminder_date = ?, last_triggered_at = ? WHERE id = ?",
                    (next_date, now_iso, r["id"]),
                    db_path=db_path,
                )
            elif rec == "weekly":
                curr_date = datetime.strptime(r_date, "%Y-%m-%d").date()
                next_date = (max(curr_date, ref_now.date()) + timedelta(days=7)).isoformat()
                execute_query(
                    "UPDATE reminders SET reminder_date = ?, last_triggered_at = ? WHERE id = ?",
                    (next_date, now_iso, r["id"]),
                    db_path=db_path,
                )
            elif rec == "monthly":
                curr_date = datetime.strptime(r_date, "%Y-%m-%d").date()
                next_date = (max(curr_date, ref_now.date()) + timedelta(days=30)).isoformat()
                execute_query(
                    "UPDATE reminders SET reminder_date = ?, last_triggered_at = ? WHERE id = ?",
                    (next_date, now_iso, r["id"]),
                    db_path=db_path,
                )
            else:
                # One-time reminder: mark status = 'triggered'
                execute_query(
                    "UPDATE reminders SET status = 'triggered', last_triggered_at = ? WHERE id = ?",
                    (now_iso, r["id"]),
                    db_path=db_path,
                )

            # Record activity log
            try:
                act_sql = """
                    INSERT INTO activity_records (activity_type, duration, metadata, created_at)
                    VALUES ('reminder_triggered', 0.0, ?, ?)
                """
                execute_insert(act_sql, (f"Triggered reminder #{r['id']}: {r['title']}", now_iso), db_path=db_path)
            except Exception as e:
                print(f"[Warning] Failed to record reminder activity: {e}")

            triggered_list.append(r)

    return triggered_list


def get_reminder_statistics(db_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Returns summary counts of active, paused, triggered, and overdue reminders."""
    all_rems = get_reminders(db_path=db_path)
    today_str = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M")

    active_count = sum(1 for r in all_rems if r["status"] == "active")
    paused_count = sum(1 for r in all_rems if r["status"] == "paused")
    triggered_count = sum(1 for r in all_rems if r["status"] == "triggered")
    dismissed_count = sum(1 for r in all_rems if r["status"] == "dismissed")

    overdue_count = 0
    for r in all_rems:
        if r["status"] == "active":
            if r["reminder_date"] < today_str:
                overdue_count += 1
            elif r["reminder_date"] == today_str and r["reminder_time"] < now_time:
                overdue_count += 1

    return {
        "total": len(all_rems),
        "active": active_count,
        "paused": paused_count,
        "triggered": triggered_count,
        "dismissed": dismissed_count,
        "overdue": overdue_count,
    }
