# 🌌 NOVA — Next-generation Organized Virtual Assistant

> *"Your world. Your intelligence. Your NOVA."*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-69%20Passed-success.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

NOVA is a personal AI assistant and modular agentic platform designed from the ground up for privacy, reliability, and true multi-tool autonomy. Built with a modular, layered architecture, NOVA avoids fake "ChatGPT wrapper" patterns by grounding all assistant capabilities in validated tools, explicit user-approved memory, and verifiable SQLite persistence.

---

## 🎯 Project Overview & Vision

NOVA is engineered to assist university students, researchers, and knowledge workers by:
- **Organizing Tasks & Deadlines**: High-efficiency priority, category, and due time tracking with persistent SQLite storage.
- **Smart Reminders**: Persistent time-based alerts with daily/weekly/monthly recurrence and verifiable delivery.
- **Daily Planning & AI Suggestions**: Algorithmic time-blocking and explainable "What to work on next" task suggestions.
- **Respecting User Privacy**: Personal memory is saved *only* with explicit user permission—conversations are never converted into memory behind the scenes.
- **Assisting Studies**: Balanced study-schedule generation with clear distinction between AI-suggested schedules and saved tasks.
- **Document Intelligence**: Upload PDF and TXT documents for grounded RAG Q&A with page citations.
- **Voice Intelligence**: One-click Speech-to-Text and Text-to-Speech audio pipeline with user consent.
- **Visualizing Real Productivity**: Transparent analytics computed strictly from real database records (no simulated metrics).
- **Graceful Progression**: Starts as a reliable, zero-cost **Rule-Based Assistant (Phase 1)** and cleanly extends across **LLM Integration (Phase 2)**, **RAG (Phase 3)**, **Voice (Phase 4)**, and **Productivity Assistant (Phase 5)**.

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
│   └── assistant.py            # Main orchestrator & rule-based parser
│
├── database/                   # Persistent data layer
│   ├── __init__.py
│   ├── db.py                   # Parameterized SQLite connector & non-destructive migrations
│   └── schema.sql              # DDL for tasks, reminders, daily_plans, memories, documents
│
├── memory/                     # User-approved knowledge
│   ├── __init__.py
│   └── memory_manager.py       # CRUD & keyword search for user memories
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
├── tests/                      # Automated test suite (69 Tests Passed)
│   ├── test_tasks.py           # Task CRUD & metrics unit tests
│   ├── test_memory.py          # Memory safety & search unit tests
│   ├── test_assistant_and_tools.py # Orchestrator & tool validation tests
│   ├── test_llm_service.py     # LLM provider, fallback & error handling tests
│   ├── test_document_assistant.py  # RAG retrieval, citations & refusal tests
│   ├── test_voice_assistant.py # STT, TTS, privacy & audio pipeline tests
│   └── test_productivity_assistant.py # Phase 5 tasks, reminders, planner & analytics tests
│
└── data/                       # Local data storage (Ignored by Git)
    ├── .gitkeep
    ├── documents/              # Local storage for indexed documents
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
Ensure all components are validated:
```bash
python -m pytest -v
```
*(All 20 tests should pass with 100% success).*

### 4. Launch NOVA
Start the Streamlit application:
```bash
streamlit run app.py
```
NOVA will launch in your default web browser (typically at `http://localhost:8501`).

---

## 🧭 Using NOVA (Phase 1 Features)

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

### 🎙️ Voice Intelligence (Speech-to-Text & Text-to-Speech)
- **Interactive Microphone Widget**: Embedded `st.audio_input` allows one-click user-initiated voice capture with live browser recording indicators.
- **Zero-Cost Speech-to-Text (STT)**: Transcribes spoken speech using Google Web Speech API via `SpeechRecognition` ($0.00 / Free, no account or API key required).
- **Natural Text-to-Speech (TTS)**: Converts NOVA's responses into natural spoken audio via `gTTS` ($0.00 / Free).
- **Text Normalization**: Strips raw markdown tags, code blocks, URLs, and document citations (e.g. `[Page 1]`) before synthesis so spoken responses sound fluent and human.
- **Full Audio Controls**: HTML5 audio player supporting Play, Pause, seek/replay, speed, volume, and MP3 download.
- **Strict Privacy & User Consent**: Explicit consent checkbox required before transmitting audio for transcription. Absolutely no continuous background recording or microphone activation without explicit user click.
- **Dedicated Voice Studio**: Features an independent STT phonetics lab, a custom TTS synthesis bench with regional accent selection, and a hands-free conversational loop.

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
- [x] **Phase 5: Smart Productivity Assistant** (Completed) — Categories, due times, estimated duration, persistent reminders with recurrence, explainable smart suggestions, time-blocked daily planner, and completion trend charts.
- [ ] **Phase 6: Data Science & ML** — Activity pattern analysis, study habit forecasting, regression experiments.
- [ ] **Phase 7: Security & Quality** — End-to-end integration tests, access control, and performance profiling.
- [ ] **Phase 8: Deployment & Portfolio** — Docker containerization, public portfolio documentation, and live demo.
