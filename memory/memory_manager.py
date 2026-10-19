"""
NOVA Personal Memory Manager
Provides user-controlled memory storage, retrieval, categorization, and deletion.
No conversation is converted to memory without explicit user consent.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from database.db import execute_insert, execute_query, fetch_all, fetch_one, log_action


def save_memory(
    content: str,
    category: str = "General",
    allow_duplicate: bool = False,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Saves an explicitly requested memory snippet to SQLite.
    
    Args:
        content: The text content to remember (required).
        category: 'General', 'Preferences', 'Academic', 'Projects', etc.
        allow_duplicate: Whether to bypass duplicate rejection.
        db_path: Optional database override for testing.
        user_id: Account owner identifier.
        
    Returns:
        The newly created memory record dictionary.
    """
    cleaned_content = content.strip()
    if not cleaned_content:
        raise ValueError("Memory content cannot be empty.")

    cleaned_category = category.strip().capitalize() if category else "General"
    uid = (user_id or "user_student_default").strip()

    if not allow_duplicate:
        dup = check_duplicate_memory(cleaned_content, db_path=db_path, user_id=uid)
        if dup:
            raise ValueError(f"Duplicate knowledge rejected: Memory #{dup['id']} already stores this information.")

    now = datetime.now(timezone.utc).isoformat()

    sql = """
        INSERT INTO memories (user_id, category, content, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """
    memory_id = execute_insert(
        sql,
        (uid, cleaned_category, cleaned_content, now, now),
        db_path=db_path
    )

    log_action("save_memory", f"Saved memory #{memory_id} in category '{cleaned_category}'", status="success", db_path=db_path)
    return get_memory_by_id(memory_id, db_path=db_path)


def check_duplicate_memory(content: str, db_path: Optional[Path | str] = None, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Checks if an identical memory is already saved in SQLite for this user."""
    if user_id:
        sql = "SELECT * FROM memories WHERE LOWER(content) = LOWER(?) AND user_id = ?"
        return fetch_one(sql, (content.strip(), user_id.strip()), db_path=db_path)
    sql = "SELECT * FROM memories WHERE LOWER(content) = LOWER(?)"
    return fetch_one(sql, (content.strip(),), db_path=db_path)


def get_memory_by_id(memory_id: Any, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single memory record by its integer or string ID."""
    try:
        m_id = int(memory_id)
    except (ValueError, TypeError):
        return None
    sql = "SELECT * FROM memories WHERE id = ?"
    return fetch_one(sql, (m_id,), db_path=db_path)


def get_all_memories(
    category: Optional[str] = None,
    search_query: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all memories, with optional user filter, category filter, and keyword search.
    """
    conditions = []
    params: List[Any] = []

    if user_id:
        conditions.append("user_id = ?")
        params.append(str(user_id).strip())

    if category and str(category).lower() != "all":
        conditions.append("LOWER(category) = ?")
        params.append(str(category).strip().lower())

    if search_query:
        conditions.append("content LIKE ?")
        params.append(f"%{str(search_query).strip()}%")

    sql = "SELECT * FROM memories"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY updated_at DESC, id DESC"
    results = fetch_all(sql, tuple(params), db_path=db_path)
    return results if isinstance(results, list) else []


def update_memory(
    memory_id: Any,
    content: str,
    category: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> bool:
    """
    Updates an existing memory record with user isolation.
    """
    try:
        m_id = int(memory_id)
    except (ValueError, TypeError):
        return False

    existing = get_memory_by_id(m_id, db_path=db_path)
    if not existing:
        return False
    if user_id and existing.get("user_id") and existing.get("user_id") != user_id:
        return False

    cleaned_content = str(content or "").strip()
    if not cleaned_content:
        raise ValueError("Memory content cannot be empty.")

    now = datetime.now(timezone.utc).isoformat()
    if category is not None:
        cat = str(category).strip().capitalize() or "General"
        sql = "UPDATE memories SET content = ?, category = ?, updated_at = ? WHERE id = ?"
        params = (cleaned_content, cat, now, m_id)
    else:
        sql = "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?"
        params = (cleaned_content, now, m_id)

    affected = execute_query(sql, params, db_path=db_path)
    if affected > 0:
        log_action("update_memory", f"Updated memory #{m_id}", status="success", db_path=db_path)
        return True
    return False


def delete_memory(
    memory_id: Any,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Deletes a single memory record by ID with user isolation."""
    try:
        m_id = int(memory_id)
    except (ValueError, TypeError):
        return False

    existing = get_memory_by_id(m_id, db_path=db_path)
    if not existing:
        return False
    if user_id and existing.get("user_id") and existing.get("user_id") != user_id:
        return False

    sql = "DELETE FROM memories WHERE id = ?"
    affected = execute_query(sql, (m_id,), db_path=db_path)
    if affected > 0:
        log_action("delete_memory", f"Deleted memory #{m_id}", status="success", db_path=db_path)
        return True
    return False


def clear_all_memories(db_path: Optional[Path | str] = None, user_id: Optional[str] = None) -> int:
    """
    Deletes all saved memories from SQLite, optionally filtered by user_id.
    Returns the count of deleted memories.
    """
    if user_id:
        sql = "DELETE FROM memories WHERE user_id = ?"
        affected = execute_query(sql, (str(user_id).strip(),), db_path=db_path)
    else:
        sql = "DELETE FROM memories"
        affected = execute_query(sql, (), db_path=db_path)
    log_action("clear_all_memories", f"Cleared memories ({affected} removed)", status="success", db_path=db_path)
    return affected


def search_memories(query: str, db_path: Optional[Path | str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience search function across memory contents and categories."""
    if not query or not str(query).strip():
        return get_all_memories(db_path=db_path, user_id=user_id)
    
    cleaned = f"%{str(query).strip()}%"
    conditions = ["(content LIKE ? OR category LIKE ?)"]
    params: List[Any] = [cleaned, cleaned]
    if user_id:
        conditions.append("user_id = ?")
        params.append(str(user_id).strip())

    sql = f"SELECT * FROM memories WHERE {' AND '.join(conditions)} ORDER BY updated_at DESC"
    return fetch_all(sql, tuple(params), db_path=db_path)


def export_memories_csv(db_path: Optional[Path | str] = None, user_id: Optional[str] = None) -> str:
    """Exports saved memories as a clean CSV string."""
    import csv
    import io
    if user_id is None and isinstance(db_path, str) and not (db_path.endswith(".db") or "/" in db_path or "\\" in db_path or db_path == ":memory:"):
        user_id = db_path
        db_path = None

    memories = get_all_memories(db_path=db_path, user_id=user_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "category", "content", "created_at", "updated_at"])
    for m in (memories or []):
        if isinstance(m, dict):
            writer.writerow([
                m.get("id", ""),
                m.get("category", "") or "General",
                m.get("content", "") or "",
                m.get("created_at", "") or "",
                m.get("updated_at", "") or "",
            ])
    return buf.getvalue()


def export_memories_json(db_path: Optional[Path | str] = None, user_id: Optional[str] = None) -> str:
    """Exports saved memories as formatted JSON."""
    import json
    if user_id is None and isinstance(db_path, str) and not (db_path.endswith(".db") or "/" in db_path or "\\" in db_path or db_path == ":memory:"):
        user_id = db_path
        db_path = None

    memories = get_all_memories(db_path=db_path, user_id=user_id)
    clean_mems = [dict(m) for m in (memories or []) if isinstance(m, dict)]
    return json.dumps({"count": len(clean_mems), "memories": clean_mems}, indent=2)

