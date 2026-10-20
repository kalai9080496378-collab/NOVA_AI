"""
Unit Tests for NOVA Assistant Orchestrator & Tool Registry
Verifies input validation, rule-based natural language parsing, tool execution,
and safe fallback messaging.
"""

import pytest
from pathlib import Path

from database.db import init_db
from nova_core.assistant import NovaAssistant
from nova_core.tool_registry import ToolRegistry, default_tool_registry
from tools.task_manager import get_tasks
from memory.memory_manager import get_all_memories


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch) -> Path:
    """Creates a temporary isolated SQLite database and redirects assistant to it."""
    db_file = tmp_path / "test_assistant_nova.db"
    init_db(db_file)
    monkeypatch.setattr("database.db.DATABASE_PATH", db_file)
    monkeypatch.setattr("nova_core.config.DATABASE_PATH", db_file)
    return db_file


def test_tool_registry_validation():
    """Verify tool parameter type enforcement and missing argument detection."""
    registry = ToolRegistry()

    # Missing required argument
    res = registry.execute("create_task", {})
    assert res.status == "failed"
    assert "required" in res.message.lower()

    # Invalid parameter type
    res2 = registry.execute("complete_task", {"task_id": "not_an_int"})
    assert res2.status == "failed"
    assert "must be an integer" in res2.message.lower()

    # Unknown tool
    res3 = registry.execute("hack_system", {})
    assert res3.status == "failed"
    assert "not recognized" in res3.message.lower()


def test_assistant_empty_input():
    assistant = NovaAssistant()
    res = assistant.process_message("")
    assert "Please enter a message" in res["response"]


def test_assistant_create_and_list_task(temp_db: Path):
    assistant = NovaAssistant()

    # Create task via command
    cmd = "add task: Finish AI Deep Learning report due 2026-10-31 priority High"
    out = assistant.process_message(cmd)
    assert out["tool_called"] == "create_task"
    assert "Task Created Successfully" in out["response"]

    # Verify task in DB
    tasks = get_tasks(db_path=temp_db)
    assert len(tasks) == 1
    assert tasks[0]["priority"] == "High"
    assert tasks[0]["due_date"] == "2026-10-31"

    # List tasks via command
    out_list = assistant.process_message("show tasks")
    assert out_list["tool_called"] == "list_tasks"
    assert "Finish AI Deep Learning report" in out_list["response"]


def test_assistant_save_and_recall_memory(temp_db: Path):
    assistant = NovaAssistant()

    # Save memory
    cmd = "remember: [Academic] My master thesis is on Graph Neural Networks"
    out = assistant.process_message(cmd)
    assert out["tool_called"] == "save_memory"
    assert "Memory Saved with your permission" in out["response"]

    # Recall memory
    recall_out = assistant.process_message("recall: Graph Neural Networks")
    assert recall_out["tool_called"] == "search_memory"
    assert "Graph Neural Networks" in recall_out["response"]


def test_assistant_time_and_stats(temp_db: Path):
    assistant = NovaAssistant()

    out_time = assistant.process_message("what time is it")
    assert out_time["tool_called"] == "get_current_time"
    assert "Current local time is" in out_time["response"]

    out_stats = assistant.process_message("stats")
    assert out_stats["tool_called"] == "analyze_productivity"
    assert "Current Productivity Snapshot" in out_stats["response"]


def test_assistant_unrecognized_fallback_disclaimer(temp_db: Path):
    assistant = NovaAssistant()
    out = assistant.process_message("Can you write a poem about artificial intelligence?")
    assert out["mode"] == "rule_based"
    assert "Rule-Based Mode" in out["response"]
    assert "Free-form conversational intelligence will be enabled once an LLM provider" in out["response"]
