"""
NOVA Productivity Tracking & Analytics Service
Logs real activity data and computes verified aggregates without inventing records.
"""

from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import pandas as pd

from database.db import execute_insert, fetch_all


def record_activity(
    activity_type: str,
    duration: float = 0.0,
    metadata: str = "",
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> int:
    """
    Records an activity event in SQLite.
    
    Args:
        activity_type: e.g., 'task_created', 'task_completed', 'study_session'
        duration: duration in minutes
        metadata: context description
        user_id: ID of the user owning this activity
    """
    now = datetime.now(timezone.utc).isoformat()
    uid = user_id or "user_student_default"
    sql = """
        INSERT INTO activity_records (activity_type, duration, metadata, created_at, user_id)
        VALUES (?, ?, ?, ?, ?)
    """
    return execute_insert(sql, (activity_type, float(duration), metadata, now, uid), db_path=db_path)


def get_activity_records(
    limit: int = 50,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves recent real activity records, optionally filtered by user_id."""
    if user_id:
        sql = """
            SELECT id, activity_type, duration, metadata, created_at, user_id
            FROM activity_records 
            WHERE user_id = ?
            ORDER BY id DESC 
            LIMIT ?
        """
        return fetch_all(sql, (user_id, limit), db_path=db_path)
    else:
        sql = """
            SELECT id, activity_type, duration, metadata, created_at, user_id
            FROM activity_records 
            ORDER BY id DESC 
            LIMIT ?
        """
        return fetch_all(sql, (limit,), db_path=db_path)


def get_daily_activity_summary(
    days: int = 7,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregates activity counts by day for the last N days.
    Returns real data from SQLite, filtered by user if specified.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if user_id:
        sql = """
            SELECT 
                substr(created_at, 1, 10) as day,
                activity_type,
                COUNT(*) as count,
                SUM(duration) as total_duration
            FROM activity_records
            WHERE created_at >= ? AND user_id = ?
            GROUP BY day, activity_type
            ORDER BY day ASC
        """
        return fetch_all(sql, (cutoff, user_id), db_path=db_path)
    else:
        sql = """
            SELECT 
                substr(created_at, 1, 10) as day,
                activity_type,
                COUNT(*) as count,
                SUM(duration) as total_duration
            FROM activity_records
            WHERE created_at >= ?
            GROUP BY day, activity_type
            ORDER BY day ASC
        """
        return fetch_all(sql, (cutoff,), db_path=db_path)


def get_study_hours_summary(
    days: int = 30,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Returns actual study hours logged by the user.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if user_id:
        sql = """
            SELECT 
                COUNT(*) as session_count,
                COALESCE(SUM(duration), 0.0) as total_minutes
            FROM activity_records
            WHERE activity_type = 'study_session' AND created_at >= ? AND user_id = ?
        """
        row = fetch_all(sql, (cutoff, user_id), db_path=db_path)
    else:
        sql = """
            SELECT 
                COUNT(*) as session_count,
                COALESCE(SUM(duration), 0.0) as total_minutes
            FROM activity_records
            WHERE activity_type = 'study_session' AND created_at >= ?
        """
        row = fetch_all(sql, (cutoff,), db_path=db_path)

    if row:
        total_mins = row[0]["total_minutes"]
        sessions = row[0]["session_count"]
    else:
        total_mins = 0.0
        sessions = 0

    return {
        "days_window": days,
        "session_count": sessions,
        "total_hours": round(total_mins / 60.0, 2),
        "total_minutes": total_mins,
    }


def export_table_to_csv(
    table_name: str,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> str:
    """
    Exports a user table ('tasks', 'memories', 'reminders', 'daily_plans', or 'activity_records') to CSV format.
    Ensures user data ownership and transparency.
    """
    valid_tables = {"tasks", "memories", "activity_records", "action_logs", "reminders", "daily_plans"}
    if table_name not in valid_tables:
        raise ValueError(f"Invalid table: {table_name}. Choose from {valid_tables}")

    user_tables = {"tasks", "memories", "activity_records", "reminders", "daily_plans"}
    if user_id and table_name in user_tables:
        rows = fetch_all(f"SELECT * FROM {table_name} WHERE user_id = ?", (user_id,), db_path=db_path)
    else:
        rows = fetch_all(f"SELECT * FROM {table_name}", db_path=db_path)

    if not rows:
        return ""

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
