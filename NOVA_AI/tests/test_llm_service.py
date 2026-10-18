"""
Unit & Integration Tests for NOVA LLM Service & Fallback Architecture
Mocks external HTTP requests so tests run completely offline and free.
Tests provider configuration, Gemini/OpenAI parsing, auth errors, rate limits,
network timeouts, key masking, and assistant fallbacks.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import requests

from database.db import init_db
from nova_core.config import is_llm_configured, get_llm_status
from nova_core.llm_service import (
    GeminiProvider,
    OpenAIProvider,
    LLMResponse,
    _mask_key,
    get_llm_service,
)
from nova_core.assistant import NovaAssistant
from tools.task_manager import get_tasks


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch) -> Path:
    """Creates a temporary isolated SQLite database and redirects assistant to it."""
    db_file = tmp_path / "test_llm_nova.db"
    init_db(db_file)
    monkeypatch.setattr("database.db.DATABASE_PATH", db_file)
    monkeypatch.setattr("nova_core.config.DATABASE_PATH", db_file)
    return db_file


# =============================================================================
# 1. Fallback Mode & Credential Handling Tests
# =============================================================================

def test_fallback_mode_when_no_keys_configured(monkeypatch):
    """Verify system operates in Rule-Based mode when no API keys are present."""
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    assert is_llm_configured() is False
    assert get_llm_service() is None

    status = get_llm_status()
    assert status["status"] == "rule_based"
    assert "Rule-Based Mode" in status["provider"]


def test_api_key_masking():
    """Verify API keys are never exposed in plaintext."""
    assert _mask_key("") == "Not Set"
    assert _mask_key("1234") == "****"
    assert _mask_key("AIzaSyB1234567890XYZ") == "AIza...0XYZ"


def test_gemini_missing_key_returns_error():
    """Verify provider returns a clean error object when initialized without a key."""
    provider = GeminiProvider(api_key="")
    res = provider.generate(messages=[{"role": "user", "content": "Hello"}])
    assert res.success is False
    assert res.error_type == "missing_credentials"
    assert "missing" in res.error.lower()


def test_openai_missing_key_returns_error():
    """Verify OpenAI provider returns a clean error object when initialized without a key."""
    provider = OpenAIProvider(api_key="")
    res = provider.generate(messages=[{"role": "user", "content": "Hello"}])
    assert res.success is False
    assert res.error_type == "missing_credentials"


# =============================================================================
# 2. Mocked Google Gemini API Tests
# =============================================================================

@patch("requests.post")
def test_gemini_generate_success(mock_post):
    """Verify successful parsing of Google Gemini response and header auth."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Machine Learning is a subset of artificial intelligence."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "totalTokenCount": 42,
        },
    }
    mock_post.return_value = mock_response

    provider = GeminiProvider(api_key="AIzaSy_fake_test_key", model="gemini-1.5-flash")
    res = provider.generate(
        messages=[{"role": "user", "content": "What is ML?"}],
        system_prompt="You are NOVA.",
    )

    assert res.success is True
    assert "Machine Learning is a subset" in res.content
    assert res.provider == "gemini"
    assert res.model == "gemini-1.5-flash"
    assert res.tokens_used == 42

    # Verify API key was passed in HTTP header (x-goog-api-key) for security
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["x-goog-api-key"] == "AIzaSy_fake_test_key"


@patch("requests.post")
def test_gemini_auth_error_handling(mock_post):
    """Verify HTTP 403 Forbidden is translated into auth_error."""
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
    mock_post.return_value = mock_response

    provider = GeminiProvider(api_key="invalid_key")
    res = provider.generate(messages=[{"role": "user", "content": "Hello"}])

    assert res.success is False
    assert res.error_type == "auth_error"
    assert "Authentication Failed" in res.error


@patch("requests.post")
def test_gemini_rate_limit_handling(mock_post):
    """Verify HTTP 429 is translated into rate_limit error."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_post.return_value = mock_response

    provider = GeminiProvider(api_key="valid_key")
    res = provider.generate(messages=[{"role": "user", "content": "Hello"}])

    assert res.success is False
    assert res.error_type == "rate_limit"
    assert "Rate Limit" in res.error


@patch("requests.post")
def test_gemini_network_timeout(mock_post):
    """Verify request timeouts are handled gracefully without raising unhandled exceptions."""
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

    provider = GeminiProvider(api_key="valid_key", timeout=5)
    res = provider.generate(messages=[{"role": "user", "content": "Hello"}])

    assert res.success is False
    assert res.error_type == "timeout"
    assert "timed out" in res.error.lower()


# =============================================================================
# 3. Mocked OpenAI API Tests
# =============================================================================

@patch("requests.post")
def test_openai_generate_success(mock_post):
    """Verify successful parsing of OpenAI completions response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Deep learning uses multi-layer neural networks.",
                }
            }
        ],
        "usage": {"total_tokens": 55},
    }
    mock_post.return_value = mock_response

    provider = OpenAIProvider(api_key="sk-fake-openai-key", model="gpt-4o-mini")
    res = provider.generate(messages=[{"role": "user", "content": "Explain deep learning"}])

    assert res.success is True
    assert "multi-layer neural networks" in res.content
    assert res.provider == "openai"
    assert res.tokens_used == 55


# =============================================================================
# 4. End-to-End Assistant Orchestrator with LLM & Deterministic Tools
# =============================================================================

@patch("requests.post")
def test_assistant_llm_conversational_response(mock_post, temp_db: Path, monkeypatch):
    """Verify assistant delegates general natural language queries to the active LLM."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "A Convolutional Neural Network uses kernel filters to extract spatial features."}],
                    "role": "model",
                }
            }
        ],
        "usageMetadata": {"totalTokenCount": 25},
    }
    mock_post.return_value = mock_response

    assistant = NovaAssistant()
    res = assistant.process_message("How does a CNN work?")

    assert res["mode"] == "llm"
    assert res["provider"] == "gemini"
    assert "Convolutional Neural Network" in res["response"]


@patch("requests.post")
def test_assistant_preserves_deterministic_tools_even_with_llm(mock_post, temp_db: Path, monkeypatch):
    """Verify explicit tool commands (add task) bypass LLM API to save latency and guarantee reliability."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_key")

    assistant = NovaAssistant()
    # Explicit task command
    res = assistant.process_message("add task: Build LSTM model due 2026-11-15 priority High")

    # Should execute locally via tool registry, NOT calling mock_post!
    assert res["tool_called"] == "create_task"
    assert "Task Created Successfully" in res["response"]
    assert mock_post.called is False

    # Check persistence in SQLite
    tasks = get_tasks(db_path=temp_db)
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Build LSTM model"


@patch("requests.post")
def test_assistant_graceful_fallback_on_api_error(mock_post, temp_db: Path, monkeypatch):
    """Verify assistant reports error and engages fallback without crashing when API fails."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_key")

    # Simulate network failure
    mock_post.side_effect = requests.exceptions.ConnectionError("DNS failure")

    assistant = NovaAssistant()
    res = assistant.process_message("Can you explain backpropagation?")

    assert res["mode"] == "fallback_rule_based"
    assert "API Error" in res["response"]
    assert "Rule-Based Fallback Mode" in res["response"]
