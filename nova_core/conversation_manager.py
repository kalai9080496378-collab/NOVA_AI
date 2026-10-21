"""
NOVA Conversation Manager
Multi-User ChatGPT-Style Conversation Platform
Manages conversation threads, persistence, auto-titling, archiving, pinning,
date grouping, user isolation, and message export.
"""

import json
import re
import uuid
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from database.db import execute_insert, execute_query, fetch_all, fetch_one, log_action


def create_thread(
    title: str = "New Chat",
    model: str = "gemini-3.8-flash",
    thread_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_id: str = "user_student_default",
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Creates a new persistent conversation thread for a specific user.
    
    Args:
        title: Title of the conversation.
        model: Selected model identifier.
        thread_id: Optional explicit UUID (generates one if None).
        system_prompt: Optional custom instructions for this thread.
        user_id: Unique account owner ID.
        db_path: Optional database override.
        
    Returns:
        Dictionary representing the created thread record.
    """
    tid = thread_id or f"thread_{uuid.uuid4().hex[:12]}"
    clean_title = (title or "New Chat").strip()
    clean_user_id = (user_id or "user_student_default").strip()
    now = datetime.now(timezone.utc).isoformat()

    sql = """
        INSERT INTO chat_threads (id, user_id, title, model, system_prompt, is_pinned, is_archived, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)
    """
    execute_query(sql, (tid, clean_user_id, clean_title, model, system_prompt, now, now), db_path=db_path)
    log_action("create_thread", f"Created thread '{clean_title}' [{tid}] for user {clean_user_id}", status="success", db_path=db_path)

    return {
        "id": tid,
        "user_id": clean_user_id,
        "title": clean_title,
        "model": model,
        "system_prompt": system_prompt,
        "is_pinned": 0,
        "is_archived": 0,
        "created_at": now,
        "updated_at": now,
    }


def get_thread(
    thread_id: str,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieves a single conversation thread by ID, optionally validating user ownership."""
    if not thread_id:
        return None
    if user_id:
        sql = "SELECT * FROM chat_threads WHERE id = ? AND user_id = ?"
        return fetch_one(sql, (thread_id, user_id), db_path=db_path)
    else:
        sql = "SELECT * FROM chat_threads WHERE id = ?"
        return fetch_one(sql, (thread_id,), db_path=db_path)


def get_all_threads(
    include_archived: bool = False,
    search_query: Optional[str] = None,
    limit: int = 100,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all conversation threads for a specific user (or all if user_id is None),
    ordered by pinned first, then updated_at descending.
    """
    conditions = []
    params: List[Any] = []

    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id)

    if not include_archived:
        conditions.append("is_archived = 0")

    if search_query and search_query.strip():
        conditions.append("(title LIKE ? OR id LIKE ?)")
        term = f"%{search_query.strip()}%"
        params.extend([term, term])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"""
        SELECT * FROM chat_threads
        {where_clause}
        ORDER BY is_pinned DESC, updated_at DESC
        LIMIT ?
    """
    params.append(limit)
    return fetch_all(sql, tuple(params), db_path=db_path)


def update_thread(
    thread_id: str,
    title: Optional[str] = None,
    is_pinned: Optional[bool] = None,
    is_archived: Optional[bool] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> bool:
    """Updates metadata properties for a conversation thread with optional user access control."""
    thread = get_thread(thread_id, user_id=user_id, db_path=db_path)
    if not thread:
        return False

    updates = []
    params: List[Any] = []

    if title is not None:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Thread title cannot be empty.")
        updates.append("title = ?")
        params.append(clean_title)

    if is_pinned is not None:
        updates.append("is_pinned = ?")
        params.append(1 if is_pinned else 0)

    if is_archived is not None:
        updates.append("is_archived = ?")
        params.append(1 if is_archived else 0)

    if model is not None:
        updates.append("model = ?")
        params.append(model.strip())

    if system_prompt is not None:
        updates.append("system_prompt = ?")
        params.append(system_prompt.strip() if system_prompt else None)

    if not updates:
        return True

    now = datetime.now(timezone.utc).isoformat()
    updates.append("updated_at = ?")
    params.append(now)

    params.append(thread_id)
    where_extra = ""
    if user_id:
        where_extra = " AND user_id = ?"
        params.append(user_id)

    sql = f"UPDATE chat_threads SET {', '.join(updates)} WHERE id = ?{where_extra}"
    execute_query(sql, tuple(params), db_path=db_path)
    return True


def delete_thread(
    thread_id: str,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> bool:
    """
    Permanently deletes a conversation thread and all its dialogue turns,
    verifying user ownership if user_id is provided.
    """
    thread = get_thread(thread_id, user_id=user_id, db_path=db_path)
    if not thread:
        return False

    # Delete messages in thread
    execute_query("DELETE FROM conversations WHERE conversation_id = ?", (thread_id,), db_path=db_path)
    # Delete thread record
    if user_id:
        execute_query("DELETE FROM chat_threads WHERE id = ? AND user_id = ?", (thread_id, user_id), db_path=db_path)
    else:
        execute_query("DELETE FROM chat_threads WHERE id = ?", (thread_id,), db_path=db_path)

    log_action("delete_thread", f"Deleted thread #{thread_id} ('{thread['title']}')", status="success", db_path=db_path)
    return True


def add_message(
    thread_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: str = "user_student_default",
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Appends a new turn to a conversation thread and bumps the thread's updated_at timestamp.
    """
    if not content or not content.strip():
        raise ValueError("Message content cannot be empty.")

    clean_content = content.strip()
    clean_role = role.strip().lower()
    if clean_role not in {"user", "assistant", "system"}:
        clean_role = "user"

    clean_user_id = (user_id or "user_student_default").strip()
    now = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata) if metadata else None

    # Ensure thread exists or create it automatically with user_id
    thread = get_thread(thread_id, user_id=clean_user_id, db_path=db_path)
    if not thread:
        # Check without user_id in case it was created without user_id
        raw_th = get_thread(thread_id, db_path=db_path)
        if not raw_th:
            auto_title = auto_generate_title(clean_content) if clean_role == "user" else "New Conversation"
            create_thread(title=auto_title, thread_id=thread_id, user_id=clean_user_id, db_path=db_path)
    elif thread["title"] in ("New Chat", "New Conversation") and clean_role == "user":
        # Auto-update initial thread title based on the first user query
        new_title = auto_generate_title(clean_content)
        update_thread(thread_id, title=new_title, user_id=clean_user_id, db_path=db_path)

    sql = """
        INSERT INTO conversations (conversation_id, user_id, role, message, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    msg_id = execute_insert(sql, (thread_id, clean_user_id, clean_role, clean_content, meta_json, now), db_path=db_path)

    # Bump thread updated_at timestamp
    execute_query("UPDATE chat_threads SET updated_at = ? WHERE id = ?", (now, thread_id), db_path=db_path)

    return {
        "id": msg_id,
        "thread_id": thread_id,
        "user_id": clean_user_id,
        "role": clean_role,
        "content": clean_content,
        "metadata": metadata or {},
        "created_at": now,
    }


def get_thread_messages(
    thread_id: str,
    limit: int = 200,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all dialogue messages for a thread ordered by created_at ascending."""
    if user_id:
        # Verify user ownership of this thread before returning messages
        thread = get_thread(thread_id, user_id=user_id, db_path=db_path)
        if not thread:
            return []

    sql = """
        SELECT id, conversation_id, user_id, role, message as content, metadata, created_at
        FROM conversations
        WHERE conversation_id = ?
        ORDER BY id ASC
        LIMIT ?
    """
    rows = fetch_all(sql, (thread_id, limit), db_path=db_path)
    parsed = []
    for r in rows:
        meta = {}
        if r.get("metadata"):
            try:
                meta = json.loads(r["metadata"])
            except Exception:
                meta = {}
        parsed.append({
            "id": r["id"],
            "thread_id": r["conversation_id"],
            "user_id": r.get("user_id", "user_student_default"),
            "role": r["role"],
            "content": r["content"],
            "metadata": meta,
            "created_at": r["created_at"],
        })
    return parsed


def auto_generate_title(prompt: str) -> str:
    """
    Generates a clean, readable 3-6 word conversation title from the first user prompt.
    """
    if not prompt or not prompt.strip():
        return "New Chat"

    cleaned = prompt.strip()

    # Strip conversational noise prefixes iteratively
    noise_patterns = [
        r"^(?:please|kindly)\s+",
        r"^(?:can\s+you|could\s+you|would\s+you)\s+",
        r"^(?:help\s+me\s+to|help\s+me)\s+",
        r"^(?:tell\s+me\s+about|tell\s+me)\s+",
        r"^(?:what\s+is|what\s+are|how\s+to|how\s+do\s+i|explain)\s+",
        r"^(?:write\s+a|create\s+a|generate\s+a)\s+",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in noise_patterns:
            new_cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True

    # Remove code blocks or markdown
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"[`*#_>]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = cleaned.split()
    if not words:
        return "Conversation"

    short_title = " ".join(words[:6])
    if len(short_title) > 42:
        short_title = short_title[:40].rsplit(" ", 1)[0]

    return short_title.capitalize()


def group_threads_by_date(threads: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorizes conversation threads into ChatGPT-style time buckets:
    'Pinned', 'Today', 'Yesterday', 'Previous 7 Days', 'Older'.
    """
    today_dt = date.today()
    yesterday_dt = today_dt - timedelta(days=1)
    week_ago_dt = today_dt - timedelta(days=7)

    groups: Dict[str, List[Dict[str, Any]]] = {
        "📌 Pinned": [],
        "Today": [],
        "Yesterday": [],
        "Previous 7 Days": [],
        "Older": [],
    }

    for t in threads:
        if t.get("is_pinned"):
            groups["📌 Pinned"].append(t)
            continue

        raw_updated = t.get("updated_at") or t.get("created_at") or ""
        try:
            thread_date = datetime.fromisoformat(raw_updated[:19]).date()
        except Exception:
            thread_date = today_dt

        if thread_date == today_dt:
            groups["Today"].append(t)
        elif thread_date == yesterday_dt:
            groups["Yesterday"].append(t)
        elif thread_date >= week_ago_dt:
            groups["Previous 7 Days"].append(t)
        else:
            groups["Older"].append(t)

    # Filter out empty buckets
    return {k: v for k, v in groups.items() if v}


def export_thread_markdown(
    thread_id: str,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> str:
    """Exports an entire conversation thread as standard markdown."""
    thread = get_thread(thread_id, user_id=user_id, db_path=db_path)
    if not thread:
        return ""

    messages = get_thread_messages(thread_id, user_id=user_id, db_path=db_path)
    lines = [
        f"# {thread['title']}",
        f"**Model:** {thread['model']} | **Date:** {thread['created_at'][:10]}",
        "",
        "---",
        "",
    ]
    for m in messages:
        sender = "**You**" if m["role"] == "user" else "**NOVA**"
        lines.append(f"### {sender} ({m['created_at'][:19]})")
        lines.append(m["content"])
        lines.append("")

    return "\n".join(lines)


def export_thread_json(
    thread_id: str,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> str:
    """Exports thread metadata and messages as structured JSON."""
    thread = get_thread(thread_id, user_id=user_id, db_path=db_path)
    if not thread:
        return "{}"

    messages = get_thread_messages(thread_id, user_id=user_id, db_path=db_path)
    export_payload = {
        "thread": dict(thread),
        "messages": messages,
    }
    return json.dumps(export_payload, indent=2)

