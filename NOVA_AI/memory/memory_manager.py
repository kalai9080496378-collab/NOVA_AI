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
) -> Dict[str, Any]:
    """
    Saves an explicitly requested memory snippet to SQLite.
    
    Args:
        content: The text content to remember (required).
        category: 'General', 'Preferences', 'Academic', 'Projects', etc.
        allow_duplicate: Whether to bypass duplicate rejection.
        db_path: Optional database override for testing.
        
    Returns:
        The newly created memory record dictionary.
    """
    cleaned_content = content.strip()
    if not cleaned_content:
        raise ValueError("Memory content cannot be empty.")

    cleaned_category = category.strip().capitalize() if category else "General"

    if not allow_duplicate:
        dup = check_duplicate_memory(cleaned_content, db_path=db_path)
        if dup:
            raise ValueError(f"Duplicate knowledge rejected: Memory #{dup['id']} already stores this information.")

    now = datetime.now(timezone.utc).isoformat()

    sql = """
        INSERT INTO memories (category, content, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    """
    memory_id = execute_insert(
        sql,
        (cleaned_category, cleaned_content, now, now),
        db_path=db_path
    )

    log_action("save_memory", f"Saved memory #{memory_id} in category '{cleaned_category}'", status="success", db_path=db_path)
    return get_memory_by_id(memory_id, db_path=db_path)


def check_duplicate_memory(content: str, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Checks if an identical memory is already saved in SQLite."""
    sql = "SELECT * FROM memories WHERE LOWER(content) = LOWER(?)"
    return fetch_one(sql, (content.strip(),), db_path=db_path)


def get_memory_by_id(memory_id: int, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single memory record by its integer ID."""
    sql = "SELECT * FROM memories WHERE id = ?"
    return fetch_one(sql, (memory_id,), db_path=db_path)


def get_all_memories(
    category: Optional[str] = None,
    search_query: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all memories, with optional category filter and keyword search.
    """
    conditions = []
    params: List[Any] = []

    if category and category.lower() != "all":
        conditions.append("LOWER(category) = ?")
        params.append(category.strip().lower())

    if search_query:
        conditions.append("content LIKE ?")
        params.append(f"%{search_query.strip()}%")

    sql = "SELECT * FROM memories"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY updated_at DESC, id DESC"
    return fetch_all(sql, tuple(params), db_path=db_path)


def update_memory(
    memory_id: int,
    content: str,
    category: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> bool:
    """
    Updates an existing memory record.
    """
    existing = get_memory_by_id(memory_id, db_path=db_path)
    if not existing:
        return False

    cleaned_content = content.strip()
    if not cleaned_content:
        raise ValueError("Memory content cannot be empty.")

    now = datetime.now(timezone.utc).isoformat()
    if category is not None:
        cat = category.strip().capitalize() or "General"
        sql = "UPDATE memories SET content = ?, category = ?, updated_at = ? WHERE id = ?"
        params = (cleaned_content, cat, now, memory_id)
    else:
        sql = "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?"
        params = (cleaned_content, now, memory_id)

    affected = execute_query(sql, params, db_path=db_path)
    if affected > 0:
        log_action("update_memory", f"Updated memory #{memory_id}", status="success", db_path=db_path)
        return True
    return False


def delete_memory(memory_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Deletes a single memory record by ID."""
    existing = get_memory_by_id(memory_id, db_path=db_path)
    if not existing:
        return False

    sql = "DELETE FROM memories WHERE id = ?"
    affected = execute_query(sql, (memory_id,), db_path=db_path)
    if affected > 0:
        log_action("delete_memory", f"Deleted memory #{memory_id}", status="success", db_path=db_path)
        return True
    return False


def clear_all_memories(db_path: Optional[Path | str] = None) -> int:
    """
    Deletes all saved memories from SQLite.
    Returns the count of deleted memories.
    """
    sql = "DELETE FROM memories"
    affected = execute_query(sql, (), db_path=db_path)
    log_action("clear_all_memories", f"Cleared all memories ({affected} removed)", status="success", db_path=db_path)
    return affected


def search_memories(query: str, db_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """Convenience search function across memory contents and categories."""
    if not query or not query.strip():
        return get_all_memories(db_path=db_path)
    
    cleaned = f"%{query.strip()}%"
    sql = """
        SELECT * FROM memories 
        WHERE content LIKE ? OR category LIKE ?
        ORDER BY updated_at DESC
    """
    return fetch_all(sql, (cleaned, cleaned), db_path=db_path)


def export_memories_csv(db_path: Optional[Path | str] = None) -> str:
    """Exports all saved memories as a clean CSV string."""
    import csv
    import io
    memories = get_all_memories(db_path=db_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "category", "content", "created_at", "updated_at"])
    for m in memories:
        writer.writerow([m["id"], m["category"], m["content"], m["created_at"], m["updated_at"]])
    return buf.getvalue()


def export_memories_json(db_path: Optional[Path | str] = None) -> str:
    """Exports all saved memories as formatted JSON."""
    import json
    memories = get_all_memories(db_path=db_path)
    return json.dumps({"count": len(memories), "memories": memories}, indent=2)

