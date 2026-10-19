"""
Automated Test Suite for NOVA User Authentication & Multi-User Isolation
Validates:
- PBKDF2-HMAC-SHA256 password hashing and verification with unique salts
- User account creation, duplicate username prevention, and retrieval
- Multi-user conversation thread isolation (User A cannot view, update, or delete User B's threads)
- Assistant orchestrator user-specific message persistence and history isolation
- Thread export privacy enforcement across users
"""

import pytest
from pathlib import Path

from database.db import init_db
from nova_core.auth import (
    create_user,
    authenticate_user,
    get_user_by_username,
    get_user_by_id,
    ensure_default_users,
    hash_password,
    verify_password,
)
from nova_core.conversation_manager import (
    create_thread,
    get_thread,
    get_all_threads,
    update_thread,
    delete_thread,
    add_message,
    get_thread_messages,
    export_thread_markdown,
    export_thread_json,
)
from nova_core.assistant import NovaAssistant


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch) -> Path:
    """Creates a temporary isolated SQLite database with users and migrated schema."""
    db_file = tmp_path / "test_auth_isolation.db"
    init_db(db_file)
    monkeypatch.setattr("database.db.DATABASE_PATH", db_file)
    monkeypatch.setattr("nova_core.config.DATABASE_PATH", db_file)
    return db_file


def test_password_hashing_and_salts():
    """Verify passwords produce different hashes with unique salts and correctly verify."""
    pwd = "MySecretPassword123!"
    hash1, salt1 = hash_password(pwd)
    hash2, salt2 = hash_password(pwd)

    # Salts must be random and unique
    assert salt1 != salt2
    # Hashes must differ because salts differ
    assert hash1 != hash2

    # Verification must succeed with the matching salt
    assert verify_password(pwd, hash1, salt1) is True
    assert verify_password(pwd, hash2, salt2) is True

    # Verification must fail with wrong password
    assert verify_password("WrongPassword", hash1, salt1) is False
    assert verify_password("", hash1, salt1) is False


def test_user_creation_and_authentication(isolated_db: Path):
    """Verify creating users, authenticating them, and retrieving profiles."""
    user = create_user("alice", "securepass123", full_name="Alice Smith", db_path=isolated_db)
    assert user["username"] == "alice"
    assert user["full_name"] == "Alice Smith"
    assert "password_hash" not in user
    assert "salt" not in user
    assert user["id"].startswith("user_")

    # Correct password succeeds
    auth_user = authenticate_user("alice", "securepass123", db_path=isolated_db)
    assert auth_user is not None
    assert auth_user["id"] == user["id"]
    assert auth_user["username"] == "alice"

    # Wrong password fails
    fail_user = authenticate_user("alice", "wrong_password", db_path=isolated_db)
    assert fail_user is None

    # Nonexistent user fails
    none_user = authenticate_user("nonexistent", "securepass123", db_path=isolated_db)
    assert none_user is None

    # Retrieve by ID and username
    by_id = get_user_by_id(user["id"], db_path=isolated_db)
    assert by_id["username"] == "alice"

    by_uname = get_user_by_username("alice", db_path=isolated_db)
    assert by_uname["id"] == user["id"]


def test_duplicate_username_rejected(isolated_db: Path):
    """Ensure duplicate usernames cannot be created."""
    create_user("bob", "password123", full_name="Bob Jones", db_path=isolated_db)
    with pytest.raises(ValueError, match="already taken"):
        create_user("bob", "anotherpassword", full_name="Bob Clone", db_path=isolated_db)


def test_ensure_default_users(isolated_db: Path):
    """Verify default demo user 'student' is created automatically."""
    ensure_default_users(isolated_db)
    student = get_user_by_username("student", db_path=isolated_db)
    assert student is not None
    assert student["username"] == "student"
    assert student["id"] == "user_student_default"

    # Verify student password is password123
    auth = authenticate_user("student", "password123", db_path=isolated_db)
    assert auth is not None


def test_multi_user_thread_isolation(isolated_db: Path):
    """Verify User A's threads are invisible to User B."""
    user_a = create_user("user_a", "passwordA123", full_name="User Alpha", db_path=isolated_db)
    user_b = create_user("user_b", "passwordB123", full_name="User Beta", db_path=isolated_db)

    # User A creates 2 threads
    th_a1 = create_thread(title="Alpha Thread 1", user_id=user_a["id"], db_path=isolated_db)
    th_a2 = create_thread(title="Alpha Thread 2", user_id=user_a["id"], db_path=isolated_db)

    # User B creates 1 thread
    th_b1 = create_thread(title="Beta Thread 1", user_id=user_b["id"], db_path=isolated_db)

    # User A views threads: should only see Alpha threads
    threads_a = get_all_threads(user_id=user_a["id"], db_path=isolated_db)
    ids_a = [t["id"] for t in threads_a]
    assert th_a1["id"] in ids_a
    assert th_a2["id"] in ids_a
    assert th_b1["id"] not in ids_a

    # User B views threads: should only see Beta thread
    threads_b = get_all_threads(user_id=user_b["id"], db_path=isolated_db)
    ids_b = [t["id"] for t in threads_b]
    assert th_b1["id"] in ids_b
    assert th_a1["id"] not in ids_b
    assert th_a2["id"] not in ids_b


def test_cross_user_thread_access_denied(isolated_db: Path):
    """Verify User B cannot get, update, or delete User A's thread."""
    user_a = create_user("alice2", "passA1234", db_path=isolated_db)
    user_b = create_user("bob2", "passB1234", db_path=isolated_db)

    th_a = create_thread(title="Alice Confidential", user_id=user_a["id"], db_path=isolated_db)

    # User B tries to get User A's thread
    assert get_thread(th_a["id"], user_id=user_b["id"], db_path=isolated_db) is None

    # User B tries to update User A's thread
    up_res = update_thread(th_a["id"], title="Hacked Title", user_id=user_b["id"], db_path=isolated_db)
    assert up_res is False
    # Verify title unchanged
    th_check = get_thread(th_a["id"], user_id=user_a["id"], db_path=isolated_db)
    assert th_check["title"] == "Alice Confidential"

    # User B tries to delete User A's thread
    del_res = delete_thread(th_a["id"], user_id=user_b["id"], db_path=isolated_db)
    assert del_res is False
    # Verify thread still exists for Alice
    assert get_thread(th_a["id"], user_id=user_a["id"], db_path=isolated_db) is not None

    # User B tries to export User A's thread
    export_md = export_thread_markdown(th_a["id"], user_id=user_b["id"], db_path=isolated_db)
    assert export_md == ""

    export_json = export_thread_json(th_a["id"], user_id=user_b["id"], db_path=isolated_db)
    assert export_json == "{}"


def test_assistant_process_message_user_isolation(isolated_db: Path):
    """Verify Assistant messages and history stay isolated by user_id."""
    user_x = create_user("user_x", "passX12345", db_path=isolated_db)
    user_y = create_user("user_y", "passY12345", db_path=isolated_db)

    assistant = NovaAssistant()

    # User X asks for stats
    out_x = assistant.process_message("stats", conversation_id="thread_user_x_1", user_id=user_x["id"])
    assert out_x["conversation_id"] == "thread_user_x_1"

    # History retrieved for User X contains the interaction
    hist_x = assistant.get_conversation_history("thread_user_x_1", user_id=user_x["id"])
    assert len(hist_x) >= 2
    assert any("stats" in m["message"] for m in hist_x)

    # History retrieved for User Y on that same conversation ID returns empty
    hist_y = assistant.get_conversation_history("thread_user_x_1", user_id=user_y["id"])
    assert len(hist_y) == 0


def test_complete_user_data_isolation(isolated_db: Path):
    """Verify that Tasks, Memories, Documents, Reminders, and Daily Plans are isolated per user."""
    from tools.task_manager import create_task, get_tasks
    from memory.memory_manager import save_memory, get_all_memories
    from modules.document_assistant import save_and_process_document, get_documents
    from tools.reminder_manager import create_reminder, get_reminders
    from modules.planner_and_suggestions import save_daily_plan, get_daily_plan

    user_a = create_user("alice_data", "passA123", db_path=isolated_db)
    user_b = create_user("bob_data", "passB123", db_path=isolated_db)

    # 1. Tasks Isolation
    task_a = create_task(title="Alice Private Task", user_id=user_a["id"], db_path=isolated_db)
    task_b = create_task(title="Bob Private Task", user_id=user_b["id"], db_path=isolated_db)

    tasks_for_a = get_tasks(user_id=user_a["id"], db_path=isolated_db)
    tasks_for_b = get_tasks(user_id=user_b["id"], db_path=isolated_db)

    assert any(t["id"] == task_a["id"] for t in tasks_for_a)
    assert not any(t["id"] == task_b["id"] for t in tasks_for_a)

    assert any(t["id"] == task_b["id"] for t in tasks_for_b)
    assert not any(t["id"] == task_a["id"] for t in tasks_for_b)

    # 2. Memories Isolation
    mem_a = save_memory("Alice favorite framework is PyTorch", user_id=user_a["id"], db_path=isolated_db)
    mem_b = save_memory("Bob favorite framework is TensorFlow", user_id=user_b["id"], db_path=isolated_db)

    mems_for_a = get_all_memories(user_id=user_a["id"], db_path=isolated_db)
    mems_for_b = get_all_memories(user_id=user_b["id"], db_path=isolated_db)

    assert any(m["id"] == mem_a["id"] for m in mems_for_a)
    assert not any(m["id"] == mem_b["id"] for m in mems_for_a)

    assert any(m["id"] == mem_b["id"] for m in mems_for_b)
    assert not any(m["id"] == mem_a["id"] for m in mems_for_b)

    # 3. Documents Isolation
    doc_a = save_and_process_document(b"Alice Private Research Document", "alice_doc.txt", user_id=user_a["id"], db_path=isolated_db)
    doc_b = save_and_process_document(b"Bob Private Financial Report", "bob_doc.txt", user_id=user_b["id"], db_path=isolated_db)

    docs_for_a = get_documents(user_id=user_a["id"], db_path=isolated_db)
    docs_for_b = get_documents(user_id=user_b["id"], db_path=isolated_db)

    assert any(d["id"] == doc_a["id"] for d in docs_for_a)
    assert not any(d["id"] == doc_b["id"] for d in docs_for_a)

    assert any(d["id"] == doc_b["id"] for d in docs_for_b)
    assert not any(d["id"] == doc_a["id"] for d in docs_for_b)

    # 4. Reminders Isolation
    rem_a = create_reminder(title="Alice Doctor Appointment", reminder_date="2026-10-25", reminder_time="14:00", user_id=user_a["id"], db_path=isolated_db)
    rem_b = create_reminder(title="Bob Team Standup", reminder_date="2026-10-25", reminder_time="10:00", user_id=user_b["id"], db_path=isolated_db)

    rems_for_a = get_reminders(user_id=user_a["id"], db_path=isolated_db)
    rems_for_b = get_reminders(user_id=user_b["id"], db_path=isolated_db)

    assert any(r["id"] == rem_a["id"] for r in rems_for_a)
    assert not any(r["id"] == rem_b["id"] for r in rems_for_a)

    assert any(r["id"] == rem_b["id"] for r in rems_for_b)
    assert not any(r["id"] == rem_a["id"] for r in rems_for_b)

    # 5. Daily Plans Isolation
    plan_a_id = save_daily_plan(plan_date="2026-10-26", focus_theme="Alice Deep Research", user_id=user_a["id"], db_path=isolated_db)
    plan_b_id = save_daily_plan(plan_date="2026-10-26", focus_theme="Bob Sprint Planning", user_id=user_b["id"], db_path=isolated_db)

    plan_for_a = get_daily_plan("2026-10-26", user_id=user_a["id"], db_path=isolated_db)
    plan_for_b = get_daily_plan("2026-10-26", user_id=user_b["id"], db_path=isolated_db)

    assert plan_for_a is not None and plan_for_a["focus_theme"] == "Alice Deep Research"
    assert plan_for_b is not None and plan_for_b["focus_theme"] == "Bob Sprint Planning"
