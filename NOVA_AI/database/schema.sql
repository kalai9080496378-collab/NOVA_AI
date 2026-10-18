-- ============================================================
-- NOVA AI ASSISTANT - DATABASE SCHEMA (SQLite)
-- Phase 1: Core Foundation & Persistent Data Models
-- ============================================================

-- 1. Tasks Table: Tracks user to-dos, priorities, deadlines, and completion status
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT NOT NULL DEFAULT 'Medium', -- 'Low', 'Medium', 'High'
    category TEXT NOT NULL DEFAULT 'General',-- 'Academic', 'Work', 'Personal', 'General'
    due_date TEXT,                           -- ISO string 'YYYY-MM-DD' or NULL
    due_time TEXT,                           -- 24-hr 'HH:MM' or NULL
    estimated_duration INTEGER DEFAULT 60,   -- Duration in minutes
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'completed'
    completed_at TEXT,                       -- ISO timestamp when completed
    created_at TEXT NOT NULL                 -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category);

-- 2. Memories Table: Explicit user-approved persistent knowledge
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL DEFAULT 'General', -- e.g., 'General', 'Preferences', 'Academic', 'Work'
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,                  -- ISO timestamp
    updated_at TEXT NOT NULL                   -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);

-- 3. Conversations Table: Dialogue history for multi-turn chat
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,             -- Unique session identifier
    role TEXT NOT NULL,                        -- 'user', 'assistant', 'system'
    message TEXT NOT NULL,
    created_at TEXT NOT NULL                   -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_conversations_id ON conversations(conversation_id);

-- 4. Activity Records Table: Tracks actual user-approved productivity & usage
CREATE TABLE IF NOT EXISTS activity_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_type TEXT NOT NULL,               -- e.g., 'task_created', 'task_completed', 'study_session', 'document_extracted'
    duration REAL DEFAULT 0.0,                 -- Duration in minutes or hours if applicable
    metadata TEXT,                             -- JSON string or extra details
    created_at TEXT NOT NULL                   -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_activity_type ON activity_records(activity_type);
CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_records(created_at);

-- 5. Action Logs Table: Auditable safety log for tool and destructive operations
CREATE TABLE IF NOT EXISTS action_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,                 -- e.g., 'create_task', 'delete_task', 'save_memory', 'clear_memory'
    details TEXT NOT NULL,                     -- Description or JSON payload of the action
    status TEXT NOT NULL,                      -- 'proposed', 'confirmed', 'success', 'failed'
    created_at TEXT NOT NULL                   -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_action_status ON action_logs(status);
CREATE INDEX IF NOT EXISTS idx_action_created ON action_logs(created_at);

-- 6. Documents Table: Tracks uploaded files, metadata, and local storage references
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,                  -- 'pdf', 'txt', 'csv'
    file_size_kb REAL NOT NULL,
    word_count INTEGER DEFAULT 0,
    page_count INTEGER DEFAULT 1,
    storage_path TEXT NOT NULL,               -- Relative path under data/documents/
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);

-- 7. Document Chunks Table: Text segments for chunked RAG retrieval
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER DEFAULT 1,
    content TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);

-- 8. Reminders Table: Tracks scheduled reminders, recurrences, and trigger state
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    reminder_date TEXT NOT NULL,              -- 'YYYY-MM-DD'
    reminder_time TEXT NOT NULL,              -- 'HH:MM'
    recurrence TEXT NOT NULL DEFAULT 'none',  -- 'none', 'daily', 'weekly', 'monthly'
    status TEXT NOT NULL DEFAULT 'active',    -- 'active', 'paused', 'triggered', 'dismissed'
    last_triggered_at TEXT,                   -- ISO timestamp when triggered
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status);
CREATE INDEX IF NOT EXISTS idx_reminders_datetime ON reminders(reminder_date, reminder_time);

-- 9. Daily Plans Table: Persistent daily schedule allocations
CREATE TABLE IF NOT EXISTS daily_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_date TEXT NOT NULL,                  -- 'YYYY-MM-DD'
    focus_theme TEXT,
    items_json TEXT NOT NULL,                 -- JSON array of planned tasks/blocks
    total_estimated_minutes INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',    -- 'active', 'completed', 'archived'
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_daily_plans_date ON daily_plans(plan_date);
