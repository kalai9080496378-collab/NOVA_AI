"""
NOVA Database Manager
Provides safe, parameterized SQLite operations and table initialization.
"""

import sqlite3
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# Determine paths relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "nova.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# Allow environment override if set
_env_db = os.getenv("DATABASE_PATH")
if _env_db:
    DATABASE_PATH = Path(_env_db)
    if not DATABASE_PATH.is_absolute():
        DATABASE_PATH = BASE_DIR / DATABASE_PATH
else:
    DATABASE_PATH = DEFAULT_DB_PATH


def get_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Ensures that the target directory exists before opening.
    Enables sqlite3.Row for dictionary-like column access.
    """
    target_path = Path(db_path) if db_path else DATABASE_PATH
    
    # Ensure the parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect with check_same_thread=False for Streamlit concurrency
    conn = sqlite3.connect(str(target_path), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Path | str] = None) -> bool:
    """
    Initializes the database schema if tables do not already exist.
    Reads DDL statements from database/schema.sql.
    """
    target_path = Path(db_path) if db_path else DATABASE_PATH
    try:
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Schema file not found at: {SCHEMA_PATH}")
        
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn = get_connection(target_path)
        try:
            with conn:
                cursor = conn.cursor()

                # Check if tasks table exists already to apply column migrations first
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(tasks)")
                    columns = [row["name"] for row in cursor.fetchall()]

                    if "category" not in columns:
                        cursor.execute("ALTER TABLE tasks ADD COLUMN category TEXT NOT NULL DEFAULT 'General'")
                    if "due_time" not in columns:
                        cursor.execute("ALTER TABLE tasks ADD COLUMN due_time TEXT")
                    if "estimated_duration" not in columns:
                        cursor.execute("ALTER TABLE tasks ADD COLUMN estimated_duration INTEGER DEFAULT 60")
                    if "completed_at" not in columns:
                        cursor.execute("ALTER TABLE tasks ADD COLUMN completed_at TEXT")
                    if "user_id" not in columns:
                        cursor.execute("ALTER TABLE tasks ADD COLUMN user_id TEXT NOT NULL DEFAULT 'user_student_default'")

                # Check if memories table exists to add user_id column if missing
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(memories)")
                    mem_cols = [row["name"] for row in cursor.fetchall()]
                    if "user_id" not in mem_cols:
                        cursor.execute("ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'user_student_default'")

                # Check if documents table exists to add user_id column if missing
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(documents)")
                    doc_cols = [row["name"] for row in cursor.fetchall()]
                    if "user_id" not in doc_cols:
                        cursor.execute("ALTER TABLE documents ADD COLUMN user_id TEXT NOT NULL DEFAULT 'user_student_default'")

                # Check if reminders table exists to add user_id column if missing
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reminders'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(reminders)")
                    rem_cols = [row["name"] for row in cursor.fetchall()]
                    if "user_id" not in rem_cols:
                        cursor.execute("ALTER TABLE reminders ADD COLUMN user_id TEXT NOT NULL DEFAULT 'user_student_default'")

                # Check if daily_plans table exists to add user_id column if missing
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_plans'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(daily_plans)")
                    plan_cols = [row["name"] for row in cursor.fetchall()]
                    if "user_id" not in plan_cols:
                        cursor.execute("ALTER TABLE daily_plans ADD COLUMN user_id TEXT NOT NULL DEFAULT 'user_student_default'")

                # Check if activity_records table exists to add user_id column if missing
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='activity_records'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(activity_records)")
                    act_cols = [row["name"] for row in cursor.fetchall()]
                    if "user_id" not in act_cols:
                        cursor.execute("ALTER TABLE activity_records ADD COLUMN user_id TEXT NOT NULL DEFAULT 'user_student_default'")

                # Check if chat_threads table exists to add user_id column if missing
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_threads'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(chat_threads)")
                    th_cols = [row["name"] for row in cursor.fetchall()]
                    if "user_id" not in th_cols:
                        cursor.execute("ALTER TABLE chat_threads ADD COLUMN user_id TEXT NOT NULL DEFAULT 'user_student_default'")

                # Check if conversations table exists to add metadata and user_id columns if missing
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(conversations)")
                    c_cols = [row["name"] for row in cursor.fetchall()]
                    if "metadata" not in c_cols:
                        cursor.execute("ALTER TABLE conversations ADD COLUMN metadata TEXT")
                    if "user_id" not in c_cols:
                        cursor.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT 'user_student_default'")

                # Execute schema SQL (creates tables and indexes if not exists)
                conn.executescript(schema_sql)

                # Backfill chat_threads from existing conversation history if empty
                cursor.execute("SELECT count(*) as c FROM chat_threads")
                if cursor.fetchone()["c"] == 0:
                    cursor.execute("""
                        INSERT OR IGNORE INTO chat_threads (id, title, created_at, updated_at)
                        SELECT 
                            conversation_id as id,
                            COALESCE(
                                SUBSTR(MIN(CASE WHEN role='user' THEN message END), 1, 40),
                                'Conversation ' || SUBSTR(conversation_id, 1, 8)
                            ) as title,
                            MIN(created_at) as created_at,
                            MAX(created_at) as updated_at
                        FROM conversations
                        GROUP BY conversation_id;
                    """)

                # Seed baseline model in model_registry if empty
                cursor.execute("SELECT count(*) as c FROM model_registry")
                if cursor.fetchone()["c"] == 0:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    cursor.execute("""
                        INSERT OR IGNORE INTO model_registry (
                            id, version_tag, provider, base_model, status, is_fine_tuned, eval_score, eval_details, training_cost_usd, created_at, updated_at
                        ) VALUES (
                            'model_v1_base', 'v1.0.0-base', 'gemini', 'gemini-1.5-flash', 'active', 0, 85.0,
                            '{"accuracy": 85.0, "instruction_following": 88.0, "format_compliance": 92.0, "safety": 99.0}',
                            0.0, ?, ?
                        )
                    """, (now_iso, now_iso))

            # Ensure default user account is created if users table is empty
            try:
                from nova_core.auth import ensure_default_users
                ensure_default_users(db_path=target_path)
            except Exception as auth_e:
                print(f"[Warning] Could not initialize default users: {auth_e}")

            return True
        finally:
            conn.close()
    except Exception as e:
        print(f"[NOVA Database Error] Failed to initialize database: {e}")
        raise


def execute_query(sql: str, params: tuple = (), db_path: Optional[Path | str] = None) -> int:
    """
    Executes an UPDATE, DELETE, or DDL query with parameterized values.
    Returns the number of affected rows.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.rowcount
    finally:
        conn.close()


def execute_insert(sql: str, params: tuple = (), db_path: Optional[Path | str] = None) -> int:
    """
    Executes an INSERT query with parameterized values.
    Returns the last inserted row ID.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.lastrowid
    finally:
        conn.close()


def fetch_all(sql: str, params: tuple = (), db_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """
    Executes a SELECT query and returns all matching rows as dictionaries.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def fetch_one(sql: str, params: tuple = (), db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """
    Executes a SELECT query and returns a single row as a dictionary, or None.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def log_action(
    action_type: str,
    details: str,
    status: str = "success",
    db_path: Optional[Path | str] = None
) -> int:
    """
    Auditable action logging for tools and state modifications.
    Records status ('proposed', 'confirmed', 'success', 'failed').
    """
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO action_logs (action_type, details, status, created_at)
        VALUES (?, ?, ?, ?)
    """
    try:
        return execute_insert(sql, (action_type, details, status, now), db_path=db_path)
    except Exception as e:
        print(f"[NOVA Logging Error] Could not record action log: {e}")
        return -1
