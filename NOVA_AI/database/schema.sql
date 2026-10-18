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

-- 3. Chat Threads Table: Persistent multi-conversation management (ChatGPT-style)
CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,                       -- Unique thread ID (UUID)
    title TEXT NOT NULL DEFAULT 'New Chat',    -- User-visible conversation title
    model TEXT DEFAULT 'gemini-1.5-flash',     -- Model associated with thread
    system_prompt TEXT,                        -- Optional thread custom instructions
    is_pinned INTEGER NOT NULL DEFAULT 0,      -- 1 = pinned to top
    is_archived INTEGER NOT NULL DEFAULT 0,    -- 1 = archived (hidden from active list)
    created_at TEXT NOT NULL,                  -- ISO timestamp
    updated_at TEXT NOT NULL                   -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_chat_threads_updated ON chat_threads(updated_at);
CREATE INDEX IF NOT EXISTS idx_chat_threads_archived ON chat_threads(is_archived);
CREATE INDEX IF NOT EXISTS idx_chat_threads_pinned ON chat_threads(is_pinned);

-- 3b. Conversations / Messages Table: Dialogue turns in a chat thread
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,             -- Matches chat_threads.id
    role TEXT NOT NULL,                        -- 'user', 'assistant', 'system'
    message TEXT NOT NULL,
    metadata TEXT,                             -- Optional JSON: tokens, tool_called, audio, model
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

-- 10. Message Feedback Table: Tracks user rating (thumbs up / down) and feedback comments
CREATE TABLE IF NOT EXISTS message_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER,
    conversation_id TEXT NOT NULL,
    rating TEXT NOT NULL,                     -- 'thumbs_up', 'thumbs_down'
    feedback_text TEXT,
    is_approved_for_training INTEGER NOT NULL DEFAULT 0, -- Explicit user consent flag
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_msg ON message_feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_feedback_conv ON message_feedback(conversation_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON message_feedback(rating);

-- 11. Learning Examples Table: Verified user corrections and approved demonstration pairs
CREATE TABLE IF NOT EXISTS learning_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL DEFAULT 'Correction', -- 'Correction', 'Demonstration', 'DomainKnowledge'
    user_prompt TEXT NOT NULL,
    rejected_response TEXT,
    corrected_response TEXT NOT NULL,
    explanation TEXT,
    is_approved INTEGER NOT NULL DEFAULT 1,     -- Only approved examples are retrieved
    is_used_in_training INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'user_correction', -- 'user_correction', 'direct_entry'
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learning_approved ON learning_examples(is_approved);
CREATE INDEX IF NOT EXISTS idx_learning_category ON learning_examples(category);

-- 12. Training Datasets Table: Tracks curated, validated, split dataset releases
CREATE TABLE IF NOT EXISTS training_datasets (
    id TEXT PRIMARY KEY,                       -- e.g., 'ds_20261018_xxx'
    name TEXT NOT NULL,
    total_samples INTEGER NOT NULL DEFAULT 0,
    train_samples INTEGER NOT NULL DEFAULT 0,
    val_samples INTEGER NOT NULL DEFAULT 0,
    format TEXT NOT NULL DEFAULT 'chat_jsonl',  -- 'chat_jsonl', 'instruction_json'
    storage_path TEXT NOT NULL,                -- Relative file path under data/datasets/
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_datasets_created ON training_datasets(created_at);

-- 13. Model Registry Table: Tracks model checkpoints, evaluation benchmarks, and active version
CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,                       -- e.g., 'model_v1_base', 'model_v2_tuned'
    version_tag TEXT NOT NULL,                 -- 'v1.0.0-base', 'v1.1.0-incontext', etc.
    provider TEXT NOT NULL,                    -- 'gemini', 'openai', 'in_context_hybrid'
    base_model TEXT NOT NULL,                  -- 'gemini-1.5-flash', 'gpt-4o-mini', 'rule_based'
    status TEXT NOT NULL DEFAULT 'active',     -- 'active', 'evaluating', 'archived', 'rolled_back'
    is_fine_tuned INTEGER NOT NULL DEFAULT 0,  -- 0 = base/in-context, 1 = fine-tuned weight checkpoint
    eval_score REAL DEFAULT 0.0,               -- Benchmark accuracy score (0.0 to 100.0)
    eval_details TEXT,                         -- JSON string of test case evaluations
    training_cost_usd REAL DEFAULT 0.0,        -- Estimated or actual cost in USD
    dataset_id TEXT REFERENCES training_datasets(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_status ON model_registry(status);

