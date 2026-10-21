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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Voice Intelligence Settings
VOICE_STT_ENGINE = os.getenv("VOICE_STT_ENGINE", "google").strip().lower()
VOICE_TTS_ENGINE = os.getenv("VOICE_TTS_ENGINE", "gtts").strip().lower()
VOICE_AUTO_SPEAK = os.getenv("VOICE_AUTO_SPEAK", "false").strip().lower() in ("true", "1", "yes")


def get_gemini_api_key() -> str:
    """
    Securely retrieves GEMINI_API_KEY from Streamlit Cloud secrets,
    active session state, environment variables, or local .env file.
    Priority: Streamlit Secrets -> Session State -> Environment Variables -> .env.
    Never exposes key in logs or UI.
    """
    # 1. Check Streamlit Cloud secrets (st.secrets)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            # Check direct keys
            for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"):
                try:
                    val = st.secrets.get(k)
                    if val and str(val).strip() and not str(val).strip().startswith("your_"):
                        return str(val).strip()
                except Exception:
                    pass
            # Check nested sections: [gemini], [google], [general]
            for sec_name in ("gemini", "google", "general"):
                try:
                    sec = st.secrets.get(sec_name)
                    if sec is not None and (isinstance(sec, dict) or hasattr(sec, "get")):
                        for sub_k in ("api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY", "key"):
                            if hasattr(sec, "get"):
                                sub_v = sec.get(sub_k)
                                if sub_v and str(sub_v).strip() and not str(sub_v).strip().startswith("your_"):
                                    return str(sub_v).strip()
                except Exception:
                    pass
    except Exception:
        pass

    # 2. Check Streamlit session_state (if user entered/saved it in UI during active session)
    try:
        import streamlit as st
        if hasattr(st, "session_state") and "user_gemini_api_key" in st.session_state:
            val = str(st.session_state["user_gemini_api_key"]).strip()
            if val and not val.startswith("your_") and val != "YOUR_GEMINI_API_KEY":
                return val
    except Exception:
        pass

    # 3. Check os.environ for GEMINI_API_KEY or GOOGLE_API_KEY
    if "GEMINI_API_KEY" in os.environ or "GOOGLE_API_KEY" in os.environ:
        for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.getenv(env_var, "").strip()
            if val and not val.startswith("your_") and val != "YOUR_GEMINI_API_KEY":
                return val
        return ""

    # 4. Check .env if neither was defined in os.environ
    try:
        from dotenv import dotenv_values
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            vals = dotenv_values(dotenv_path=env_file)
            for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                val = (vals.get(env_var) or "").strip()
                if val and not val.startswith("your_") and val != "YOUR_GEMINI_API_KEY":
                    return val
    except Exception:
        pass

    return ""


def get_openai_api_key() -> str:
    """Securely retrieves OPENAI_API_KEY from environment variables or Streamlit secrets."""
    k = os.getenv("OPENAI_API_KEY", "").strip()
    if k:
        return k
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return ""


def is_llm_configured() -> bool:
    """
    Checks if an LLM provider has an active API key configured in environment or Streamlit secrets.
    Supports explicit LLM_PROVIDER or automatic key detection.
    """
    prov = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    gemini_key = get_gemini_api_key()
    openai_key = get_openai_api_key()

    if prov == "gemini" and gemini_key:
        return True
    if prov == "openai" and openai_key:
        return True
    if gemini_key or openai_key:
        return True
    return False


def get_llm_status() -> dict:
    """Returns human-readable configuration status for the UI and Settings."""
    prov = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    gemini_key = get_gemini_api_key()
    openai_key = get_openai_api_key()

    if (prov == "gemini" and gemini_key) or (not openai_key and gemini_key):
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
        return {
            "status": "connected",
            "provider": "GEMINI",
            "model": model,
            "message": f"Connected to Google Gemini ({model}). Free tier active.",
            "free_tier": "15 RPM / 1,500 RPD free with Google AI Studio key.",
            "is_real_ai": True,
        }

    if (prov == "openai" and openai_key) or (not gemini_key and openai_key):
        model = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
        return {
            "status": "connected",
            "provider": "OPENAI",
            "model": model,
            "message": f"Connected to OpenAI ({model}).",
            "free_tier": "Pay-as-you-go billing per token.",
            "is_real_ai": True,
        }

    return {
        "status": "rule_based",
        "provider": "None (Rule-Based Mode)",
        "model": "Deterministic Regex & Tool Parser",
        "message": (
            "Running in Rule-Based Mode. Pattern-matching commands are active. "
            "Add a GEMINI_API_KEY to your Streamlit secrets or .env file to enable free-tier Google Gemini AI conversation."
        ),
        "free_tier": "N/A (Local execution only - $0 cost)",
        "is_real_ai": False,
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
