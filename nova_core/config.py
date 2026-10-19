"""
NOVA Core Configuration Module
Handles project paths, environment variable loading, and feature flags.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory is the root of the project (parent of nova_core)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    # Also attempt standard load_dotenv()
    load_dotenv()

# Data directory and default SQLite database path
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "nova.db"

# Schema path
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# Application Metadata
APP_NAME = "NOVA"
APP_FULL_NAME = "Next-generation Organized Virtual Assistant"
APP_TAGLINE = "Your world. Your intelligence. Your NOVA."
APP_VERSION = "2.5.0 - Professional AI Platform"

# Environment & Provider Settings
NOVA_ENV = os.getenv("NOVA_ENV", "development")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = BASE_DIR / DATABASE_PATH

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Voice Intelligence Settings
VOICE_STT_ENGINE = os.getenv("VOICE_STT_ENGINE", "google").strip().lower()
VOICE_TTS_ENGINE = os.getenv("VOICE_TTS_ENGINE", "gtts").strip().lower()
VOICE_AUTO_SPEAK = os.getenv("VOICE_AUTO_SPEAK", "false").strip().lower() in ("true", "1", "yes")


def is_llm_configured() -> bool:
    """
    Checks if an LLM provider has an active API key configured in environment.
    Supports explicit LLM_PROVIDER or automatic key detection.
    """
    prov = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    gemini_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY).strip()

    if prov == "gemini" and gemini_key:
        return True
    if prov == "openai" and openai_key:
        return True
    # If a key is present even if provider was left as 'none' or default
    if gemini_key or openai_key:
        return True
    return False


def get_llm_status() -> dict:
    """Returns human-readable configuration status for the UI and Settings."""
    prov = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    gemini_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY).strip()

    if (prov == "gemini" and gemini_key) or (not openai_key and gemini_key):
        model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
        return {
            "status": "connected",
            "provider": "GEMINI",
            "model": model,
            "message": f"Connected to Google Gemini ({model}). Free tier active.",
            "free_tier": "15 RPM / 1,500 RPD free with Google AI Studio key.",
        }

    if (prov == "openai" and openai_key) or (not gemini_key and openai_key):
        model = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
        return {
            "status": "connected",
            "provider": "OPENAI",
            "model": model,
            "message": f"Connected to OpenAI ({model}).",
            "free_tier": "Pay-as-you-go billing per token.",
        }

    return {
        "status": "rule_based",
        "provider": "None (Rule-Based Mode)",
        "model": "Deterministic Regex & Tool Parser",
        "message": (
            "Running in Rule-Based Mode. Pattern-matching commands are active. "
            "Add a GEMINI_API_KEY to your .env file to enable free-tier AI conversation."
        ),
        "free_tier": "N/A (Local execution only - $0 cost)",
    }


def get_voice_status() -> dict:
    """Returns human-readable status for Voice Intelligence engines and cost tiers."""
    stt_eng = os.getenv("VOICE_STT_ENGINE", VOICE_STT_ENGINE).strip().lower()
    tts_eng = os.getenv("VOICE_TTS_ENGINE", VOICE_TTS_ENGINE).strip().lower()

    return {
        "stt_engine": stt_eng,
        "stt_name": "Google Web Speech API" if stt_eng == "google" else stt_eng.upper(),
        "stt_cost": "$0.00 (Free, no account needed)",
        "tts_engine": tts_eng,
        "tts_name": "Google Text-to-Speech (gTTS)" if tts_eng == "gtts" else tts_eng.upper(),
        "tts_cost": "$0.00 (Free, no account needed)",
        "privacy": "Encrypted HTTPS transit for speech conversion; no permanent audio storage.",
    }
