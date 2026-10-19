# 🌌 NOVA — Next-generation Organized Virtual Assistant

> *"Your world. Your intelligence. Your NOVA."*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-101%20Passed-success.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

NOVA is a personal AI assistant and modular agentic platform designed from the ground up for privacy, reliability, and true multi-tool autonomy. Built with a modular, layered architecture, NOVA avoids fake "ChatGPT wrapper" patterns by grounding all assistant capabilities in validated tools, explicit user-approved memory, and verifiable SQLite persistence.

---

## 🎯 Project Overview & Vision

NOVA is engineered to assist university students, researchers, and knowledge workers by:
- **Organizing Tasks & Deadlines**: High-efficiency priority, category, and due time tracking with persistent SQLite storage.
- **Smart Reminders**: Persistent time-based alerts with daily/weekly/monthly recurrence and verifiable delivery.
- **Daily Planning & AI Suggestions**: Algorithmic time-blocking and explainable "What to work on next" task suggestions.
- **Respecting User Privacy**: Personal memory is saved *only* with explicit user permission—conversations are never converted into memory behind the scenes.
- **Continuous Self-Learning & Corrections**: Human-in-the-loop feedback (👍/👎) and verified corrections retrieved dynamically at runtime with $0.00 cloud cost.
- **Fine-Tuning & Dataset Pipeline**: PII-scrubbed Chat JSONL dataset builder, compute/cost estimator, standardized benchmark suite, and model registry with rollbacks.
- **Assisting Studies**: Balanced study-schedule generation with clear distinction between AI-suggested schedules and saved tasks.
- **Document Intelligence**: Upload PDF, TXT, and CSV documents for grounded RAG Q&A with page citations.
- **Voice Intelligence**: One-click Speech-to-Text and Text-to-Speech audio pipeline with user consent.
- **Visualizing Real Productivity**: Transparent analytics computed strictly from real database records (no simulated metrics).
- **Graceful Progression**: Starts as a reliable, zero-cost **Rule-Based Assistant (Phase 1)** and cleanly extends across **LLM Integration (Phase 2)**, **RAG (Phase 3)**, **Voice (Phase 4)**, **Productivity Assistant (Phase 5)**, and **Self-Learning & Fine-Tuning (Phase 6)**.

---

## 🏗️ Architecture & Component Flow

```
User (Browser / Desktop)
          │
          ▼
   Streamlit UI (Dark Navy / Violet / Cyan Theme)
          │
          ▼
   Input Sanitization & Validation
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│                 Assistant Orchestrator                   │
│                                                          │
│  ├── Conversation Context (conversations table)          │
│  ├── User-Approved Memory (memories table)               │
│  ├── Intent & Pattern Classifier (Rule-based / LLM)      │
│  └── Permission & Safety Verification                    │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                  Modular Tool Registry                   │
│                                                          │
│  ├── create_task           ├── save_memory               │
│  ├── list_tasks            ├── search_memory             │
│  ├── complete_task         ├── create_reminder           │
│  ├── analyze_productivity  └── check_due_reminders       │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│            Data Persistence & Module Layer               │
│                                                          │
│  ├── SQLite (data/nova.db with non-destructive schema)   │
│  ├── Task & Reminder Managers (Deadlines & Recurrence)   │
│  ├── Daily Planner & Suggestions (Advisory Time Blocking)│
│  ├── Student Assistant (Spaced study blocks)            │
│  ├── Document Intelligence (TF-IDF Chunked RAG)          │
│  ├── Voice Intelligence (STT & TTS Engine)               │
│  └── Productivity Analytics (Matplotlib & CSV Export)    │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
    Result Verification & Audit Log (action_logs table)
                          │
                          ▼
            Verified User Response in UI
```

---

## 📂 Project Structure

```
nova/
│
├── app.py                      # Main Streamlit user interface & navigation
├── requirements.txt            # Project dependencies
├── .env.example                # Configuration template
├── .gitignore                  # Git privacy & artifact ignores
├── README.md                   # Complete architectural guide
│
├── nova_core/                  # Core assistant logic
│   ├── __init__.py
│   ├── config.py               # Path resolutions & environment variables
│   ├── tool_registry.py        # Validated tool execution engine
│   ├── llm_service.py          # Dedicated LLM provider abstraction (Gemini & OpenAI)
│   ├── conversation_manager.py # Thread lifecycle, archiving, pinned chats, and JSON/MD export
│   ├── learning_system.py      # Self-learning loop, feedback (👍/👎), corrections & in-context retrieval
│   ├── fine_tuning_pipeline.py # PII sanitization, Chat JSONL dataset builder, cost estimation & benchmarks
│   └── assistant.py            # Main orchestrator, intent classifier & learning injector
│
├── database/                   # Persistent data layer
│   ├── __init__.py
│   ├── db.py                   # Parameterized SQLite connector & non-destructive migrations
│   └── schema.sql              # DDL for tasks, reminders, memories, documents, feedback, datasets, model registry
│
├── memory/                     # User-approved knowledge
│   ├── __init__.py
│   └── memory_manager.py       # CRUD, deduplication guardrails & JSON/CSV exports
│
├── tools/                      # Tool implementations
│   ├── __init__.py
│   ├── task_manager.py         # Task scheduling, categories, time, priorities & statistics
│   ├── reminder_manager.py     # Smart reminders, recurrence cycles & delivery verification
│   └── productivity.py         # Activity logging & CSV data export
│
├── modules/                    # Specialized application modules
│   ├── __init__.py
│   ├── student_assistant.py    # Study plan generator & session logger
│   ├── document_assistant.py   # Page-aware PDF/TXT extraction & RAG engine
│   ├── voice_assistant.py      # Speech-to-Text & Text-to-Speech audio pipeline
│   └── planner_and_suggestions.py # Time-blocked daily planner & smart task suggestions
│
├── analytics/                  # Data visualization
│   ├── __init__.py
│   └── dashboard.py            # Matplotlib charts for status, priority, category & trends
│
├── tests/                      # Automated test suite (101 Tests Passed)
│   ├── test_tasks.py           # Task CRUD & metrics unit tests
│   ├── test_memory.py          # Memory safety & search unit tests
│   ├── test_assistant_and_tools.py # Orchestrator & tool validation tests
│   ├── test_llm_service.py     # LLM provider, fallback & error handling tests
│   ├── test_document_assistant.py  # RAG retrieval, citations & refusal tests
│   ├── test_voice_assistant.py # STT, TTS, privacy & audio pipeline tests
│   ├── test_productivity_assistant.py # Phase 5 tasks, reminders, planner & analytics tests
│   ├── test_security_and_deployment.py # Security audit, SQL injection & input validation tests
│   └── test_learning_and_finetuning.py # Self-learning, PII scrubbing, datasets, and benchmark tests
│
└── data/                       # Local data storage (Ignored by Git)
    ├── .gitkeep
    ├── documents/              # Local storage for indexed documents
    ├── datasets/               # Local Chat JSONL fine-tuning releases (train.jsonl / val.jsonl)
    └── nova.db                 # Auto-created SQLite database file
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- `pip` package manager

### 2. Setup Workspace & Install Dependencies
Open a terminal in the project directory:
```bash
# Optional: create a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Automated Tests
Ensure all components are validated across the entire 106-test test suite:
```bash
python -m pytest -v
```
*(All 106 tests across tasks, reminders, documents, voice, LLM, security, and learning pipelines should pass with 100% success).*

### 4. Launch NOVA
Start the Streamlit application:
```bash
streamlit run app.py
```
NOVA will launch in your default web browser (typically at `http://localhost:8501`).

---

## 🧭 Using NOVA (Phase 1–6 Features)

### 💬 AI Chat Commands
In Phase 1, NOVA operates in a transparent **Rule-Based Mode**. You can execute real actions via natural commands:

| Command Pattern | Example | What NOVA Does |
|---|---|---|
| `add task: <title>` | `add task: Submit thesis chapter due 2026-10-31 priority High` | Validates and creates a persistent task in SQLite |
| `show tasks` | `show tasks` or `list tasks` | Retrieves pending tasks ordered by urgency |
| `complete task <id>` | `complete task 1` or `done 1` | Marks task completed and logs activity |
| `remember: <info>` | `remember: [Academic] Advisor meeting is on Tuesdays` | Saves knowledge with explicit user permission |
| `recall: <keyword>` | `recall: Advisor` | Searches and returns relevant memories |
| `stats` | `stats` or `productivity` | Computes live task completion rate |
| `time` | `what time is it` | Returns current local system time |
| `help` | `help` or `commands` | Displays the command cheatsheet |

### 🧠 My Memory
- Store explicit snippets categorized under `Academic`, `Preferences`, `Projects`, etc.
- Search, edit, or delete individual memories at any time.
- Clear all memories with double safety verification.

### 📋 Task Manager (Phase 5 Upgrades)
- **Comprehensive Task Attributes**: Due dates, precise 24h due times (`HH:MM`), priority weighting (`High`, `Medium`, `Low`), categories (`General`, `Academic`, `Work`, `Personal`, `Research`, `Project`), and estimated duration (minutes).
- **Exact Overdue Detection**: Identifies overdue tasks comparing both calendar dates and timestamps.
- **Safety Confirmations**: Explicit confirmation dialogues prevent accidental task deletion.
- **Full Lifecycle**: Complete, reopen, edit inline, and filter by status, priority, category, or search keywords.

### ⏰ Smart Reminders (Phase 5)
- **Persistent SQLite Scheduling**: Time-based alerts stored persistently in `reminders` table.
- **Recurrence Cycles**: Supports `none` (one-off), `daily`, `weekly`, and `monthly` cycles that automatically advance upon triggering.
- **Task Association**: Optionally tie reminders directly to specific pending tasks.
- **Full Management**: Pause, resume, dismiss, or delete with confirmation prompts.
- **Runtime Delivery Verification & Platform Limitations**: Explicit transparency on local vs. sleeping container execution; catches up and guarantees delivery on active sessions.

### 📅 Daily Planner & Smart Suggestions (Phase 5)
- **Algorithmic Time-Blocking**: Constructs balanced schedules incorporating priority, deadlines, estimated durations, and automated 15-minute rest breaks.
- **Explainable Smart Suggestions**: Recommends "What should I work on next?" with transparent rationale and urgency scores.
- **Strict Advisory Mode**: Guarantees that AI-generated suggestions and schedule plans **never** alter, reschedule, or delete underlying database tasks without explicit user commands.
- **Plan History**: Edit notes, customize schedules, and save plans permanently to the `daily_plans` SQLite table.

### 📊 Productivity Analytics (Phase 5 Upgrades)
- **Comprehensive Metrics**: Real-time counts of total, completed, pending, and overdue tasks, plus completion rates.
- **Matplotlib Visualizations**:
  - Task Status Donut Chart
  - Priority Distribution Bar Chart
  - Category Distribution Chart
  - Completion Trends Over Time Chart (using real completion timestamps)
  - Daily Activity Event Frequency Timeline
- **Data Ownership**: Export all SQLite tables to CSV (`tasks.csv`, `reminders.csv`, `memories.csv`, `activity_records.csv`).

### 🎓 Student Mode
- **Algorithmic Study Plan Generator**: Input your modules, available study hours, and exam date. NOVA generates balanced deep work blocks (60%), active recall practice (30%), and spaced revision (10%).
- **One-Click Add**: Easily convert any suggested study session into a real task.
- **Session Logger**: Record actual time spent studying to measure real academic investment.

### 📄 Document Intelligence (RAG & Semantic Retrieval)
- **Multi-Format Ingestion**: Upload PDF and TXT documents with automated page/section extraction and overlapping chunk segmentation.
- **TF-IDF Vector Space Model**: Fast, offline-capable lexical and semantic chunk retrieval using scikit-learn.
- **Grounded Q&A with Citations**: Answers are strictly grounded in retrieved document excerpts, explicitly citing supporting page numbers (e.g. `[Page 2]`).
- **Anti-Hallucination Guardrails**: If the uploaded document does not contain enough information, NOVA explicitly reports: *"The uploaded document does not contain sufficient information to answer this question."*
- **Follow-up Dialogue Support**: Maintains conversation history for multi-turn contextual follow-ups within the active document.
- **Executive Summarization**: Synthesizes document overviews, key findings, and takeaways (LLM-powered when connected, structured extractive summary offline).
- **Document Management & Privacy**: View indexed documents, inspect retrieval chunks, and delete documents with complete local filesystem and SQLite cleanup.

### 🎙️ Voice Intelligence & Speech Recognition Engine

NOVA features a production-grade, privacy-respecting voice intelligence system engineered for high accuracy across diverse accents and code-switched dialects:

1. **Multilingual Speech-to-Text (STT)**:
   - **English (India) [`en-IN`] (Default)**: Calibrated specifically for Indian English phonetics, intonations, stress patterns, and technical vocabulary.
   - **Tamil [`ta-IN`]**: Direct native Tamil speech recognition returning Tamil script Unicode.
   - **Tanglish (Tamil + English Mixed)**: Dual-mode support allowing seamless bilingual code-switching (e.g., *"Python project submit pannanum"*).
   - **English (US) [`en-US`]**: Standard North American acoustic dictionary.

2. **Dual-Engine Recognition Architecture**:
   - **Google Web Speech API ($0.00 / Free)**: Zero configuration, zero API keys, no credit card required. Operates with dynamic room noise floor adaptation (`adjust_for_ambient_noise`).
   - **Google Gemini 1.5 Flash Multimodal Audio**: High-precision recognition for complex Tanglish and heavy acoustic environments. Activated seamlessly when `GEMINI_API_KEY` is present.
   - **Auto Mode**: Intelligently routes to Gemini Multimodal Audio when configured, with transparent fallback to Google Web Speech.

3. **Staged Review & Edit Workflow (User Safety & Control)**:
   - Voice recordings are **never automatically executed**.
   - Spoken audio is transcribed into a staged review panel.
   - Users can listen back to their recorded audio clip, inspect the transcript, edit any misheard words or slang in an inline editor, and click **`🚀 Send to NOVA`**.
   - If audio quality was poor, users can click **`🔄 Discard & Retry`** to re-record without leaving orphaned queries.

4. **Acoustic Calibration & Empty Recording Detection**:
   - Computes 16-bit PCM Root-Mean-Square (RMS) signal energy.
   - Distinguishes between silent/mic-muted recordings (RMS < 120) and low-confidence audio, providing targeted, actionable troubleshooting steps.
   - Includes an inline **Microphone Permission & Audio Troubleshooting Guide** for Chrome, Edge, and Windows Sound Settings.

5. **Bilingual Text-to-Speech (TTS)**:
   - Converts NOVA's responses into natural speech via `gTTS`.
   - **Script Auto-Detection**: Automatically identifies Tamil Unicode characters (`\u0B80-\u0BFF`) and routes synthesis to native Tamil (`lang="ta"`).
   - **Indian English Voice**: Defaults English synthesis to regional Indian TLD (`tld="co.in"`).
   - **Text Normalization**: Strips Markdown asterisks, backticks, code blocks, URLs, and citation brackets (e.g. `[Page 1]`) before synthesis for natural, conversational audio.

6. **Voice Studio**:
   - **Hands-Free Voice Dialogue**: Voice-driven conversation with audio turn history.
   - **Speech-to-Text Laboratory (STT)**: Independent acoustic testing bench displaying audio size, RMS energy level, container format, and word count.
   - **Text-to-Speech Laboratory (TTS)**: Custom phrase synthesis with preset English, Tamil, and Tanglish phrases.

### 🧠 Self-Learning & Fine-Tuning Intelligence

NOVA features an end-to-end self-learning architecture designed specifically for researchers and students who want continuous improvement without risking privacy or incurring unnecessary cloud costs:

1. **Human-in-the-Loop Feedback Loop**:
   - Every assistant turn in AI Chat includes inline action controls: **Copy (📋)**, **Helpful (👍)**, **Needs Improvement (👎)**, and **Teach NOVA (✏️)**.
   - Thumbs-up/down ratings are stored persistently in the `message_feedback` SQLite table with consent opt-in flags.
   - Submitting a correction stores a verified question-answer demonstration in `learning_examples`.

2. **Zero-Cost Dynamic In-Context Learning**:
   - When a user asks a question, NOVA computes token similarity and substring overlap against verified corrections in `learning_examples`.
   - Relevant corrections are instantly injected as few-shot demonstrations into the system prompt (or served directly in rule-based mode).
   - **Advantage**: Instant adaptation with **$0.00 compute cost** and zero catastrophic forgetting.

3. **Strict Separation of Four Knowledge Stores**:
   - **Conversation History** (`conversations`): Session dialogue, never automatically promoted to memory.
   - **Personal Memory** (`memories`): Explicit user facts, with duplicate and conflict prevention guardrails.
   - **Document Knowledge Base** (`documents`, `document_chunks`): Page-attributed RAG knowledge.
   - **Training Datasets** (`learning_examples`, `training_datasets`): Verified ground-truth Q&A pairs for fine-tuning.

4. **Automated PII & Secret Scrubbing**:
   - Before any example is included in a training release, NOVA runs regex-based redaction:
     - Google API keys (`AIza...`) &rarr; `[API_KEY_REDACTED]`
     - OpenAI API keys (`sk-...`) &rarr; `[API_KEY_REDACTED]`
     - Bearer tokens &rarr; `Bearer [TOKEN_REDACTED]`
     - Email addresses &rarr; `[EMAIL_REDACTED]`
     - Phone numbers &rarr; `[PHONE_REDACTED]`
     - IPv4 addresses &rarr; `[IP_REDACTED]`
     - Credit card numbers &rarr; `[CARD_REDACTED]`

5. **Chat JSONL Dataset Builder & Train/Val Split**:
   - Exports datasets into standard Chat JSONL format ready for OpenAI, Google Vertex AI, or local LoRA training.
   - Configurable validation split (e.g. 80% train / 20% val).
   - Saved locally to `data/datasets/<dataset_id>/` with downloadable `train.jsonl` and `val.jsonl`.

6. **Provider Transparency & Cost Estimator**:
   - **Google Gemini**: Explains that consumer free-tier API keys do not support `/fine_tuning` endpoints, while Google Cloud Vertex AI requires cloud project billing.
   - **OpenAI**: Explains that `gpt-4o-mini` supports the fine-tuning REST API ($3.00/1M tokens) with prepaid credit.
   - **Cost Estimator**: Computes token usage, OpenAI/Vertex AI cost, and local GPU time.
   - **Safety Guarantee**: NOVA will **never** trigger external cloud fine-tuning jobs automatically.

7. **Standardized Benchmark Evaluation & Model Rollbacks**:
   - Evaluates models across NumPy vector mathematics, Machine Learning principles (gradient descent), and Assistant Identity.
   - Scores responses via keyword recall and instruction adherence.
   - Tracks all model versions in `model_registry` with a one-click rollback mechanism if a new version degrades performance.

---

## 🔒 Safety & Privacy Principles

1. **Local-First Storage**: All data (tasks, memories, documents, and chunks) resides in local SQLite database (`data/nova.db`) and local files (`data/documents/`).
2. **Transparent AI Disclosure**: Only specific retrieved document excerpts are passed to the LLM when asking a question. Full raw files are never transmitted to external third parties.
3. **Voice Privacy & Ephemeral Processing**: Spoken audio is recorded only upon user click and held transiently in memory for transcription. Audio is never covertly recorded or permanently stored on remote servers.
4. **Input Validation**: Strict file type validation (.pdf, .txt, .csv), size enforcement (<= 15MB), time format validation (`HH:MM`), and audio validation.
5. **No Arbitrary Code Execution**: The assistant can never execute arbitrary Python code or shell commands.
6. **Action Audit Trail**: All task operations, memory updates, document deletions, and voice interactions are logged to `action_logs` and `activity_records` for complete accountability.

---

## 🗺️ Incremental Development Roadmap

- [x] **Phase 1: Working MVP** — Streamlit UI, SQLite persistence, Task Manager, User-controlled Memory, Rule-Based Natural Language Assistant, Analytics & CSV Export.
- [x] **Phase 2: Real AI** — Dedicated LLM service module, pluggable Google Gemini & OpenAI REST integration, multi-turn conversation context, rate limit & auth error handling, transparent fallback mode, and key masking.
- [x] **Phase 3: Document Intelligence (RAG)** — Document upload (PDF, TXT), page-attributed chunking, TF-IDF RAG retrieval, grounded Q&A with page citations, anti-hallucination refusal, automated summarization, and user-controlled deletion.
- [x] **Phase 4: Voice Intelligence** — Speech-to-Text (STT) via Google Web Speech, natural Text-to-Speech (TTS) via gTTS, microphone recording, HTML5 playback controls, explicit privacy consent, and hands-free conversational studio.
- [x] **Phase 5: Smart Productivity Assistant** — Categories, due times, estimated duration, persistent reminders with recurrence, explainable smart suggestions, time-blocked daily planner, and completion trend charts.
- [x] **Phase 6: Self-Learning & Fine-Tuning Pipeline** (Completed) — Feedback loop (👍/👎), verified human corrections store, zero-cost in-context learning retrieval, 4 separated knowledge stores, PII/secret scrubbing, Chat JSONL dataset builder, compute/cost estimator, benchmark evaluation suite, and model registry with rollback.
- [ ] **Phase 7: Advanced Data Science & Forecasting** — Activity pattern analysis, study habit forecasting, regression experiments.
- [ ] **Phase 8: Security & Quality Audit** — End-to-end stress testing, access control, and containerized deployment.
