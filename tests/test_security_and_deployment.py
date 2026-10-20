"""
Automated Test Suite: Security Audit, Input Validation & Deployment Resilience
Tests:
- API key masking & credential privacy
- SQL injection resilience via parameterized queries
- Path traversal defense for document storage & deletion
- File type and size validation guardrails
- Message length validation
- Reminder input validation and failure modes
- Empty-state resilience across analytics and daily planner
- CSV table whitelist access control
- Voice consent privacy enforcement
"""

import os
from pathlib import Path
import pytest

from database.db import init_db, get_connection, fetch_all
from nova_core.assistant import NovaAssistant
from nova_core.llm_service import _mask_key
from tools.task_manager import create_task, get_tasks
from memory.memory_manager import save_memory, search_memories
from tools.reminder_manager import create_reminder, get_reminders
from modules.document_assistant import (
    save_and_process_document,
    delete_document,
    DOCUMENTS_DIR,
)
from modules.planner_and_suggestions import generate_daily_plan
from analytics.dashboard import (
    create_task_status_chart,
    create_priority_distribution_chart,
    create_completion_trend_chart,
    create_category_distribution_chart,
)
from tools.productivity import export_table_to_csv
from modules.voice_assistant import transcribe_audio


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provides an isolated SQLite database initialized with schema."""
    db_file = tmp_path / "test_security.db"
    init_db(db_file)
    return db_file


def test_api_key_masking_comprehensive():
    """Verify that credentials of all formats are safely masked for display."""
    assert _mask_key("") == "Not Set"
    assert _mask_key(None) == "Not Set"
    assert _mask_key("   ") == "Not Set"
    assert _mask_key("short") == "****"
    assert _mask_key("12345678") == "****"

    long_key = "AIzaSyD-SecretKeySampleForGoogleAI123456"
    masked = _mask_key(long_key)
    assert masked.startswith("AIza")
    assert masked.endswith("3456")
    assert "SecretKeySample" not in masked
    assert "..." in masked


def test_sql_injection_defense_parameterized_queries(temp_db):
    """Verify that SQL injection strings are treated purely as text data and never executed."""
    malicious_title = "Finish thesis'; DROP TABLE tasks; --"
    malicious_desc = "' OR '1'='1'; DELETE FROM memories; --"
    
    # 1. Create task with injection payload
    task = create_task(title=malicious_title, description=malicious_desc, db_path=temp_db)
    assert task["id"] > 0

    # Verify tasks table is intact and has 1 record
    tasks = get_tasks(status="all", db_path=temp_db)
    assert len(tasks) == 1
    assert tasks[0]["title"] == malicious_title

    # 2. Save memory with injection payload
    mem = save_memory(malicious_desc, category="Academic", db_path=temp_db)
    assert mem["id"] > 0
    results = search_memories("DELETE FROM", db_path=temp_db)
    assert len(results) == 1

    # 3. Create reminder with injection payload
    rem = create_reminder(
        title="Alert'; DROP TABLE reminders; --",
        remind_at="2026-10-31 10:00",
        db_path=temp_db,
    )
    assert rem["id"] > 0
    rems = get_reminders(status="all", db_path=temp_db)
    assert len(rems) == 1


def test_document_path_traversal_sanitization(temp_db):
    """Verify that attempted directory traversal in filenames is stripped before saving to disk."""
    traversal_name = "../../etc/passwd.txt"
    content = b"Mock document content for testing path traversal sanitization."

    doc = save_and_process_document(file_bytes=content, filename=traversal_name, db_path=temp_db)
    
    saved_path = Path(doc["storage_path"]).resolve()
    docs_dir_res = DOCUMENTS_DIR.resolve()

    # The file must reside strictly inside DOCUMENTS_DIR
    assert str(saved_path).startswith(str(docs_dir_res))
    assert saved_path.exists()
    assert ".." not in saved_path.name
    assert "passwd.txt" in saved_path.name

    # Clean up test file
    delete_document(doc["id"], db_path=temp_db)
    assert not saved_path.exists()


def test_document_executable_rejection(temp_db):
    """Verify that executables and script files are strictly rejected."""
    bad_files = [
        ("malware.exe", b"MZ binary payload"),
        ("exploit.sh", b"#!/bin/bash\nrm -rf /"),
        ("script.bat", b"@echo off\ndir"),
        ("backdoor.py", b"import os\nos.system('dir')"),
        ("danger.php", b"<?php phpinfo(); ?>"),
    ]

    for fname, data in bad_files:
        with pytest.raises(ValueError) as excinfo:
            save_and_process_document(file_bytes=data, filename=fname, db_path=temp_db)
        assert "Unsupported file format" in str(excinfo.value)


def test_document_oversized_file_rejection(temp_db):
    """Verify that documents exceeding the 15MB size limit are rejected immediately."""
    oversized_bytes = b"0" * (16 * 1024 * 1024)  # 16 MB
    with pytest.raises(ValueError) as excinfo:
        save_and_process_document(file_bytes=oversized_bytes, filename="huge.txt", db_path=temp_db)
    assert "exceeds maximum allowed limit" in str(excinfo.value)


def test_assistant_oversized_input_rejection():
    """Verify that message inputs exceeding 4000 characters are safely rejected."""
    assistant = NovaAssistant()
    long_msg = "x" * 4005
    res = assistant.process_message(long_msg)
    assert "Input too long" in res["response"]
    assert res["tool_called"] is None


def test_reminder_input_validation_failures(temp_db):
    """Verify that invalid reminder inputs fail gracefully with descriptive ValueErrors."""
    # 1. Empty title
    with pytest.raises(ValueError, match="Reminder title cannot be empty"):
        create_reminder(title="", remind_at="2026-10-31 10:00", db_path=temp_db)

    # 2. Corrupted datetime format
    with pytest.raises(ValueError, match="Invalid reminder"):
        create_reminder(title="Doctor", remind_at="next Tuesday afternoon", db_path=temp_db)

    # 3. Invalid recurrence rule
    with pytest.raises(ValueError, match="Invalid recurrence"):
        create_reminder(
            title="Meeting",
            remind_at="2026-10-31 10:00",
            recurrence="hourly_every_minute",
            db_path=temp_db,
        )


def test_daily_plan_empty_and_zero_hours(temp_db):
    """Verify that the daily planner handles 0 tasks or 0 hours gracefully without exceptions."""
    # No pending tasks in temp_db
    plan = generate_daily_plan(
        target_date="2026-10-20",
        available_hours=4.0,
        db_path=temp_db,
    )
    assert plan["total_tasks_scheduled"] == 0
    assert plan["total_allocated_minutes"] == 0
    assert plan["schedule"] == []

    # Edge case: 0 available hours
    plan_zero = generate_daily_plan(
        target_date="2026-10-20",
        available_hours=0.0,
        db_path=temp_db,
    )
    assert plan_zero["total_tasks_scheduled"] == 0


def test_analytics_charts_empty_data_graceful():
    """Verify that Matplotlib dashboard charts do not crash when given empty data."""
    empty_tasks = []
    assert create_task_status_chart(empty_tasks) is None
    assert create_priority_distribution_chart(empty_tasks) is None
    assert create_category_distribution_chart(empty_tasks) is None
    assert create_completion_trend_chart(empty_tasks) is None


def test_csv_export_whitelist_security(temp_db):
    """Verify that arbitrary table export is rejected to prevent schema/system exposure."""
    # Whitelisted tables succeed
    csv_out = export_table_to_csv("tasks", db_path=temp_db)
    assert isinstance(csv_out, str)

    # Non-whitelisted or forbidden tables raise ValueError
    with pytest.raises(ValueError, match="Invalid table"):
        export_table_to_csv("sqlite_master", db_path=temp_db)

    with pytest.raises(ValueError, match="Invalid table"):
        export_table_to_csv("passwords", db_path=temp_db)


def test_voice_consent_guardrail_enforcement():
    """Verify that external audio transcription is strictly refused without explicit consent."""
    audio_sample = b"\x00" * 256  # 256 dummy bytes
    result = transcribe_audio(audio_sample, consent_external=False)
    assert result["success"] is False
    assert result["error_type"] == "consent_required"
    assert "External audio processing is disabled" in result["error"]


def test_streamlit_cloud_deployment_configuration():
    """Verify essential Streamlit Cloud deployment assets exist and are properly structured."""
    project_root = Path(__file__).resolve().parent.parent

    # 1. requirements.txt
    req_file = project_root / "requirements.txt"
    assert req_file.exists(), "requirements.txt must exist in project root"
    req_content = req_file.read_text(encoding="utf-8")
    assert "google-genai" in req_content, "google-genai must be declared in requirements.txt"
    assert "streamlit" in req_content
    assert "SpeechRecognition" in req_content

    # 2. .python-version
    py_ver_file = project_root / ".python-version"
    assert py_ver_file.exists(), ".python-version must exist for Streamlit Cloud runtime pinning"
    assert py_ver_file.read_text(encoding="utf-8").strip().startswith("3.11")

    # 3. packages.txt
    pkg_file = project_root / "packages.txt"
    assert pkg_file.exists(), "packages.txt must exist for Linux container audio/codec binaries"
    pkg_content = pkg_file.read_text(encoding="utf-8")
    assert "ffmpeg" in pkg_content
    assert "flac" in pkg_content

    # 4. .streamlit/config.toml
    cfg_file = project_root / ".streamlit" / "config.toml"
    assert cfg_file.exists(), ".streamlit/config.toml must exist"
    cfg_content = cfg_file.read_text(encoding="utf-8")
    assert "headless = true" in cfg_content

    # 5. .gitignore protects secrets and databases
    gi_file = project_root / ".gitignore"
    assert gi_file.exists()
    gi_content = gi_file.read_text(encoding="utf-8")
    assert ".streamlit/secrets.toml" in gi_content
    assert ".env" in gi_content
    assert "*.db" in gi_content


def test_secrets_resolution_streamlit_cloud(monkeypatch):
    """Verify that Gemini and OpenAI keys resolve properly from Streamlit secrets mocks."""
    from nova_core.config import get_gemini_api_key, get_openai_api_key, is_llm_configured
    import streamlit as st

    # Mock st.secrets with standard flat structure
    mock_secrets = {
        "GEMINI_API_KEY": "AIzaSy_MockStreamlitCloudSecretKey12345",
        "OPENAI_API_KEY": "sk-proj-MockOpenAiKey12345",
    }
    monkeypatch.setattr(st, "secrets", mock_secrets, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    key = get_gemini_api_key()
    assert key == "AIzaSy_MockStreamlitCloudSecretKey12345"
    assert get_openai_api_key() == "sk-proj-MockOpenAiKey12345"
    assert is_llm_configured() is True

