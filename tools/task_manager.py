"""
NOVA Task Manager
Phase 5: Smart Productivity Assistant
Modular task management functions with full SQLite persistence, categories,
due dates/times, duration estimation, completion tracking, and audit logging.
"""

from datetime import datetime, timezone, date, time
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from database.db import execute_insert, execute_query, fetch_all, fetch_one, log_action


def validate_due_time(due_time: Optional[str]) -> Optional[str]:
    """
    Validates and standardizes a due time string as 'HH:MM' (24-hour format).
    Returns cleaned 'HH:MM' or None if empty.
    Raises ValueError on invalid formats.
    """
    if not due_time or not due_time.strip():
        return None

    cleaned = due_time.strip()
    match = re.match(r"^(\d{1,2}):(\d{2})$", cleaned)
    if not match:
        raise ValueError(f"Invalid time format: '{due_time}'. Expected HH:MM (24-hour).")

    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time value: '{due_time}'. Hours must be 0-23, minutes 0-59.")

    return f"{hour:02d}:{minute:02d}"


def create_task(
    title: str,
    description: str = "",
    priority: str = "Medium",
    due_date: Optional[str] = None,
    due_time: Optional[str] = None,
    category: str = "General",
    estimated_duration: int = 60,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a new task in the database.
    
    Args:
        title: Short title of the task (required).
        description: Detailed instructions or notes (optional).
        priority: 'Low', 'Medium', or 'High'.
        due_date: Due date formatted as 'YYYY-MM-DD' or None.
        due_time: Due time formatted as 'HH:MM' (24-hour) or None.
        category: Task category (e.g. 'Academic', 'Work', 'Personal', 'General').
        estimated_duration: Estimated completion time in minutes (default 60).
        db_path: Optional database override (useful for testing).
        user_id: Account owner identifier.
        
    Returns:
        Dictionary representing the created task.
    """
    cleaned_title = (title or "").strip()
    if not cleaned_title:
        raise ValueError("Task title cannot be empty.")

    valid_priorities = {"Low", "Medium", "High"}
    cleaned_priority = priority.capitalize() if priority else "Medium"
    if cleaned_priority not in valid_priorities:
        cleaned_priority = "Medium"

    # Validate due_date format if provided
    if due_date and due_date.strip():
        due_date_str = due_date.strip()
        try:
            datetime.strptime(due_date_str, "%Y-%m-%d")
            due_date = due_date_str
        except ValueError:
            raise ValueError(f"Invalid due date format: '{due_date}'. Expected YYYY-MM-DD.")
    else:
        due_date = None

    # Validate due_time format if provided
    cleaned_due_time = validate_due_time(due_time)

    # Sanitize category
    cleaned_category = (category or "General").strip().title()
    if not cleaned_category:
        cleaned_category = "General"

    # Validate estimated_duration
    try:
        dur = int(estimated_duration)
        duration_mins = max(5, min(1440, dur)) # Between 5 mins and 24 hours
    except (ValueError, TypeError):
        duration_mins = 60

    uid = (user_id or "user_student_default").strip()
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO tasks (
            user_id, title, description, priority, category, due_date, due_time,
            estimated_duration, status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
    """
    task_id = execute_insert(
        sql,
        (
            uid,
            cleaned_title,
            description.strip(),
            cleaned_priority,
            cleaned_category,
            due_date,
            cleaned_due_time,
            duration_mins,
            now,
        ),
        db_path=db_path,
    )

    # Audit log
    log_action("create_task", f"Created task #{task_id}: '{cleaned_title}' [{cleaned_category}]", status="success", db_path=db_path)

    # Log activity record
    try:
        act_sql = """
            INSERT INTO activity_records (user_id, activity_type, duration, metadata, created_at)
            VALUES (?, 'task_created', ?, ?, ?)
        """
        execute_insert(act_sql, (uid, float(duration_mins), f"Task #{task_id}: {cleaned_title}", now), db_path=db_path)
    except Exception as e:
        print(f"[Warning] Failed to record task activity: {e}")

    return get_task_by_id(task_id, db_path=db_path)


def get_task_by_id(task_id: int, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single task by its integer ID."""
    sql = "SELECT * FROM tasks WHERE id = ?"
    return fetch_one(sql, (task_id,), db_path=db_path)


def get_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    search_term: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves tasks with optional filters and user isolation.
    
    Args:
        status: 'pending', 'completed', or None/'all'.
        priority: 'Low', 'Medium', 'High', or None/'all'.
        category: Filter by specific category or None/'all'.
        search_term: Search query against title and description.
        db_path: Optional database override.
        user_id: Filter by owner user account.
    """
    conditions = []
    params: List[Any] = []

    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id.strip())

    if status and status.lower() != "all":
        conditions.append("status = ?")
        params.append(status.lower().strip())

    if priority and priority.lower() != "all":
        conditions.append("priority = ?")
        params.append(priority.capitalize().strip())

    if category and category.lower() != "all":
        conditions.append("category = ?")
        params.append(category.strip().title())

    if search_term:
        conditions.append("(title LIKE ? OR description LIKE ?)")
        wildcard = f"%{search_term.strip()}%"
        params.extend([wildcard, wildcard])

    sql = "SELECT * FROM tasks"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    # Order: pending first, then by due_date ascending (nulls last), then due_time, then id descending
    sql += """
        ORDER BY 
            CASE WHEN status = 'pending' THEN 0 ELSE 1 END,
            CASE WHEN due_date IS NULL THEN 1 ELSE 0 END,
            due_date ASC,
            CASE WHEN due_time IS NULL THEN 1 ELSE 0 END,
            due_time ASC,
            id DESC
    """
    return fetch_all(sql, tuple(params), db_path=db_path)


def update_task(
    task_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    due_date: Optional[str] = None,
    due_time: Optional[str] = None,
    estimated_duration: Optional[int] = None,
    status: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> bool:
    """
    Updates an existing task. Only fields provided with non-None values are updated.
    Preserves all existing data.
    """
    existing = get_task_by_id(task_id, db_path=db_path)
    if not existing:
        return False

    updates = []
    params: List[Any] = []

    if title is not None:
        cleaned_title = title.strip()
        if not cleaned_title:
            raise ValueError("Task title cannot be empty.")
        updates.append("title = ?")
        params.append(cleaned_title)

    if description is not None:
        updates.append("description = ?")
        params.append(description.strip())

    if priority is not None:
        p = priority.capitalize()
        if p in {"Low", "Medium", "High"}:
            updates.append("priority = ?")
            params.append(p)

    if category is not None:
        cat_clean = category.strip().title()
        if cat_clean:
            updates.append("category = ?")
            params.append(cat_clean)

    if due_date is not None:
        if due_date == "" or due_date.lower() == "none":
            updates.append("due_date = NULL")
        else:
            try:
                datetime.strptime(due_date.strip(), "%Y-%m-%d")
                updates.append("due_date = ?")
                params.append(due_date.strip())
            except ValueError:
                raise ValueError(f"Invalid due date format: '{due_date}'. Expected YYYY-MM-DD.")

    if due_time is not None:
        if due_time == "" or due_time.lower() == "none":
            updates.append("due_time = NULL")
        else:
            cleaned_time = validate_due_time(due_time)
            updates.append("due_time = ?")
            params.append(cleaned_time)

    if estimated_duration is not None:
        try:
            dur = max(5, min(1440, int(estimated_duration)))
            updates.append("estimated_duration = ?")
            params.append(dur)
        except (ValueError, TypeError):
            pass

    if status is not None:
        s = status.lower().strip()
        if s in {"pending", "completed"}:
            updates.append("status = ?")
            params.append(s)
            if s == "completed" and not existing.get("completed_at"):
                now = datetime.now(timezone.utc).isoformat()
                updates.append("completed_at = ?")
                params.append(now)
            elif s == "pending":
                updates.append("completed_at = NULL")

    if not updates:
        return True

    params.append(task_id)
    sql = f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?"
    execute_query(sql, tuple(params), db_path=db_path)

    log_action("update_task", f"Updated task #{task_id}", status="success", db_path=db_path)
    return True


def complete_task(task_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Marks a task as completed with completed_at timestamp and logs activity."""
    task = get_task_by_id(task_id, db_path=db_path)
    if not task:
        return False

    now = datetime.now(timezone.utc).isoformat()
    sql = "UPDATE tasks SET status = 'completed', completed_at = ? WHERE id = ?"
    affected = execute_query(sql, (now, task_id), db_path=db_path)

    if affected > 0:
        log_action("complete_task", f"Completed task #{task_id}: '{task['title']}'", status="success", db_path=db_path)

        act_sql = """
            INSERT INTO activity_records (activity_type, duration, metadata, created_at)
            VALUES ('task_completed', ?, ?, ?)
        """
        try:
            dur = float(task.get("estimated_duration") or 60.0)
            execute_insert(act_sql, (dur, f"Task #{task_id}: {task['title']}", now), db_path=db_path)
        except Exception as e:
            print(f"[Warning] Failed to log activity: {e}")
        return True
    return False


def reopen_task(task_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Reopens a completed task back to pending."""
    task = get_task_by_id(task_id, db_path=db_path)
    if not task:
        return False

    sql = "UPDATE tasks SET status = 'pending', completed_at = NULL WHERE id = ?"
    affected = execute_query(sql, (task_id,), db_path=db_path)
    if affected > 0:
        log_action("reopen_task", f"Reopened task #{task_id}: '{task['title']}'", status="success", db_path=db_path)
        return True
    return False


def delete_task(task_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Permanently deletes a task from the database."""
    task = get_task_by_id(task_id, db_path=db_path)
    if not task:
        return False

    sql = "DELETE FROM tasks WHERE id = ?"
    affected = execute_query(sql, (task_id,), db_path=db_path)
    if affected > 0:
        log_action("delete_task", f"Deleted task #{task_id}: '{task['title']}'", status="success", db_path=db_path)
        return True
    return False


def get_task_statistics(db_path: Optional[Path | str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes real summary metrics for the user dashboard.
    Accurately accounts for due dates, due times, categories, and completion rates.
    Never invents fake data.
    """
    all_tasks = get_tasks(db_path=db_path, user_id=user_id)
    total = len(all_tasks)
    pending = sum(1 for t in all_tasks if t["status"] == "pending")
    completed = sum(1 for t in all_tasks if t["status"] == "completed")

    today_str = date.today().isoformat()
    now_time_str = datetime.now().strftime("%H:%M")

    overdue = 0
    due_today = 0
    upcoming = []
    category_counts: Dict[str, int] = {}

    for t in all_tasks:
        cat = t.get("category") or "General"
        category_counts[cat] = category_counts.get(cat, 0) + 1

        if t["status"] == "pending" and t.get("due_date"):
            t_due = t["due_date"]
            t_time = t.get("due_time")

            if t_due < today_str:
                overdue += 1
            elif t_due == today_str:
                due_today += 1
                if t_time and t_time < now_time_str:
                    overdue += 1
                else:
                    upcoming.append(t)
            else:
                upcoming.append(t)

    # Sort upcoming by nearest due date & time
    upcoming.sort(key=lambda x: (x["due_date"], x.get("due_time") or "23:59"))

    high_priority = sum(1 for t in all_tasks if t["priority"] == "High" and t["status"] == "pending")
    completion_rate = (completed / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "pending": pending,
        "completed": completed,
        "overdue": overdue,
        "due_today": due_today,
        "high_priority_pending": high_priority,
        "completion_rate": round(completion_rate, 1),
        "upcoming_tasks": upcoming[:8],
        "category_counts": category_counts,
    }
