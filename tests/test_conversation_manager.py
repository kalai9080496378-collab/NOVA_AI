"""
Automated Test Suite for NOVA Multi-Conversation Manager (Multi-User Isolation)
Tests:
- Thread creation, retrieval, and updates (rename, pin, archive)
- Message appending, retrieval with metadata, and thread timestamp bumping
- Auto-generation of conversation titles from initial user prompts
- ChatGPT-style date grouping (Pinned, Today, Yesterday, Previous 7 Days, Older)
- Markdown and JSON conversation export
- Thread deletion and cascading message cleanup
- Non-destructive legacy session migration
"""

import json
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
import pytest

from database.db import init_db, get_connection
from nova_core.conversation_manager import (
    create_thread,
    get_thread,
    get_all_threads,
    update_thread,
    delete_thread,
    add_message,
    get_thread_messages,
    auto_generate_title,
    group_threads_by_date,
    export_thread_markdown,
    export_thread_json,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provides an isolated, initialized database for conversation tests."""
    db_file = tmp_path / "test_conversations.db"
    init_db(db_file)
    return db_file


def test_create_and_get_thread(temp_db):
    """Verify thread creation with metadata and default properties."""
    thread = create_thread(
        title="Deep Learning Study Plan",
        model="gemini-1.5-flash",
        db_path=temp_db,
    )
    assert thread["id"].startswith("thread_")
    assert thread["title"] == "Deep Learning Study Plan"
    assert thread["model"] == "gemini-1.5-flash"
    assert thread["is_pinned"] == 0
    assert thread["is_archived"] == 0

    fetched = get_thread(thread["id"], db_path=temp_db)
    assert fetched is not None
    assert fetched["title"] == "Deep Learning Study Plan"


def test_update_thread_rename_pin_archive(temp_db):
    """Verify renaming, pinning, and archiving threads."""
    thread = create_thread(title="Draft Title", db_path=temp_db)
    tid = thread["id"]

    # Rename
    assert update_thread(tid, title="Refined AI Research Notes", db_path=temp_db) is True
    assert get_thread(tid, db_path=temp_db)["title"] == "Refined AI Research Notes"

    # Empty title raises ValueError
    with pytest.raises(ValueError, match="cannot be empty"):
        update_thread(tid, title="   ", db_path=temp_db)

    # Pin
    assert update_thread(tid, is_pinned=True, db_path=temp_db) is True
    assert get_thread(tid, db_path=temp_db)["is_pinned"] == 1

    # Archive
    assert update_thread(tid, is_archived=True, db_path=temp_db) is True
    assert get_thread(tid, db_path=temp_db)["is_archived"] == 1

    # Active thread list excludes archived threads by default
    active_threads = get_all_threads(include_archived=False, db_path=temp_db)
    assert not any(t["id"] == tid for t in active_threads)

    # Included when requested
    all_threads = get_all_threads(include_archived=True, db_path=temp_db)
    assert any(t["id"] == tid for t in all_threads)


def test_add_and_retrieve_messages(temp_db):
    """Verify appending turns, JSON metadata persistence, and thread updated_at bumping."""
    thread = create_thread(title="Coding Chat", db_path=temp_db)
    tid = thread["id"]

    m1 = add_message(
        tid,
        role="user",
        content="How do I implement binary search in Python?",
        metadata={"source": "text"},
        db_path=temp_db,
    )
    assert m1["id"] > 0
    assert m1["role"] == "user"

    m2 = add_message(
        tid,
        role="assistant",
        content="Here is a clean implementation of binary search...",
        metadata={"tokens": 42, "provider": "gemini"},
        db_path=temp_db,
    )
    assert m2["id"] > 0
    assert m2["role"] == "assistant"

    messages = get_thread_messages(tid, db_path=temp_db)
    assert len(messages) == 2
    assert messages[0]["content"] == "How do I implement binary search in Python?"
    assert messages[1]["metadata"]["tokens"] == 42


def test_auto_generate_title_from_prompt():
    """Verify clean 3-6 word title generation from various prompt types."""
    assert auto_generate_title("") == "New Chat"
    assert auto_generate_title("   ") == "New Chat"
    
    t1 = auto_generate_title("Please explain what is gradient descent and how it works")
    assert "Gradient descent" in t1
    assert len(t1.split()) <= 6

    t2 = auto_generate_title("Can you help me write a Python script for web scraping")
    assert "Python script" in t2 or "Web scraping" in t2


def test_group_threads_by_date():
    """Verify time-based categorization into ChatGPT-style sidebar groups."""
    now_iso = datetime.now(timezone.utc).isoformat()
    yesterday_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    four_days_ago_iso = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    ten_days_ago_iso = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

    sample_threads = [
        {"id": "t1", "title": "Pinned Thread", "is_pinned": 1, "updated_at": ten_days_ago_iso},
        {"id": "t2", "title": "Today Thread", "is_pinned": 0, "updated_at": now_iso},
        {"id": "t3", "title": "Yesterday Thread", "is_pinned": 0, "updated_at": yesterday_iso},
        {"id": "t4", "title": "Past Week Thread", "is_pinned": 0, "updated_at": four_days_ago_iso},
        {"id": "t5", "title": "Older Thread", "is_pinned": 0, "updated_at": ten_days_ago_iso},
    ]

    groups = group_threads_by_date(sample_threads)
    assert "📌 Pinned" in groups and len(groups["📌 Pinned"]) == 1
    assert "Today" in groups and len(groups["Today"]) == 1
    assert "Yesterday" in groups and len(groups["Yesterday"]) == 1
    assert "Previous 7 Days" in groups and len(groups["Previous 7 Days"]) == 1
    assert "Older" in groups and len(groups["Older"]) == 1


def test_delete_thread_cascades_messages(temp_db):
    """Verify that deleting a thread removes the thread and all associated messages."""
    thread = create_thread(title="Temporary Thread", db_path=temp_db)
    tid = thread["id"]

    add_message(tid, "user", "Message 1", db_path=temp_db)
    add_message(tid, "assistant", "Message 2", db_path=temp_db)

    assert len(get_thread_messages(tid, db_path=temp_db)) == 2
    assert delete_thread(tid, db_path=temp_db) is True

    # Both thread and messages should be removed
    assert get_thread(tid, db_path=temp_db) is None
    assert len(get_thread_messages(tid, db_path=temp_db)) == 0


def test_export_thread_markdown_and_json(temp_db):
    """Verify exporting thread to Markdown and JSON formats."""
    thread = create_thread(title="Export Test Chat", db_path=temp_db)
    tid = thread["id"]
    add_message(tid, "user", "Hello NOVA!", db_path=temp_db)
    add_message(tid, "assistant", "Hello! How can I help you today?", db_path=temp_db)

    md = export_thread_markdown(tid, db_path=temp_db)
    assert "# Export Test Chat" in md
    assert "Hello NOVA!" in md
    assert "Hello! How can I help you today?" in md

    js = export_thread_json(tid, db_path=temp_db)
    payload = json.loads(js)
    assert payload["thread"]["title"] == "Export Test Chat"
    assert len(payload["messages"]) == 2


def test_legacy_session_migration_preserves_history(temp_db):
    """Verify that historical conversation turns without a chat_threads record are auto-backfilled."""
    conn = get_connection(temp_db)
    with conn:
        # Insert raw legacy turns directly into conversations table without thread
        conn.execute(
            "INSERT INTO conversations (conversation_id, role, message, created_at) VALUES (?, ?, ?, ?)",
            ("legacy_session_999", "user", "What is transformer attention mechanism?", "2026-10-15T10:00:00"),
        )
        conn.execute(
            "INSERT INTO conversations (conversation_id, role, message, created_at) VALUES (?, ?, ?, ?)",
            ("legacy_session_999", "assistant", "Attention allows the model to dynamically weight tokens...", "2026-10-15T10:00:05"),
        )

    # Re-run init_db (which runs backfill migration)
    init_db(temp_db)

    # The legacy session should now be a first-class thread
    threads = get_all_threads(include_archived=True, db_path=temp_db)
    legacy_match = [t for t in threads if t["id"] == "legacy_session_999"]
    assert len(legacy_match) == 1
    assert "transformer attention" in legacy_match[0]["title"].lower()


def test_assistant_process_message_stream(temp_db, monkeypatch):
    """Verify assistant.process_message_stream yields token chunks and persists turns."""
    from database import db as db_module
    from nova_core.assistant import NovaAssistant

    monkeypatch.setattr(db_module, "DATABASE_PATH", temp_db)
    assistant = NovaAssistant()

    thread = create_thread(title="Streaming Chat", db_path=temp_db)
    chunks = list(assistant.process_message_stream("stats", conversation_id=thread["id"]))

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "Productivity" in full_text or "Tasks" in full_text or "Completed" in full_text

    # Verify persisted in SQLite
    msgs = get_thread_messages(thread["id"], db_path=temp_db)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "stats"
    assert msgs[1]["role"] == "assistant"


def test_fresh_session_starts_with_empty_conversation_no_default_qa(temp_db):
    """Verify that a fresh session creates an empty conversation with no default Q&A."""
    user_id = "user_clean_test"
    existing = get_all_threads(include_archived=False, user_id=user_id, db_path=temp_db)
    empty_th = None
    for eth in existing:
        if not get_thread_messages(eth["id"], limit=1, user_id=user_id, db_path=temp_db):
            empty_th = eth
            break

    if empty_th:
        active_th = empty_th
    else:
        active_th = create_thread(title="New Conversation", user_id=user_id, db_path=temp_db)

    assert active_th is not None
    messages = get_thread_messages(active_th["id"], user_id=user_id, db_path=temp_db)
    assert messages == []


def test_new_chat_preserves_previous_threads_without_loading_them_by_default(temp_db):
    """Verify that previous chats are preserved, but a new chat starts completely empty."""
    user_id = "user_clean_test_2"
    # Thread 1 with a conversation
    th1 = create_thread(title="Past Discussion", user_id=user_id, db_path=temp_db)
    add_message(th1["id"], "user", "What is quantum computing?", user_id=user_id, db_path=temp_db)
    add_message(th1["id"], "assistant", "Quantum computing uses qubits.", user_id=user_id, db_path=temp_db)

    # User initiates a new chat or starts a new session
    existing = get_all_threads(include_archived=False, user_id=user_id, db_path=temp_db)
    empty_th = None
    for eth in existing:
        if not get_thread_messages(eth["id"], limit=1, user_id=user_id, db_path=temp_db):
            empty_th = eth
            break

    if empty_th:
        active_th = empty_th
    else:
        active_th = create_thread(title="New Conversation", user_id=user_id, db_path=temp_db)

    # Active thread must be fresh and completely empty
    assert active_th["id"] != th1["id"]
    messages = get_thread_messages(active_th["id"], user_id=user_id, db_path=temp_db)
    assert messages == []

    # Thread 1's history is preserved
    th1_msgs = get_thread_messages(th1["id"], user_id=user_id, db_path=temp_db)
    assert len(th1_msgs) == 2
    assert th1_msgs[0]["content"] == "What is quantum computing?"


