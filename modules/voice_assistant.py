"""
NOVA Voice Intelligence Module
Phase 4: Speech-to-Text (STT) and Text-to-Speech (TTS) Engine

Provides robust, modular audio processing for NOVA:
- Speech-to-Text via SpeechRecognition (Google Web Speech API / Gemini Multimodal)
- Text-to-Speech via gTTS with natural markdown stripping and text normalization
- Transparent user privacy controls and explicit consent verification
- Zero-cost default architecture with no mandatory paid API keys
"""

import io
import re
import os
import math
import struct
import base64
import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, Tuple

# Optional imports with graceful availability detection
try:
    import speech_recognition as sr
    HAVE_SPEECH_RECOGNITION = True
except ImportError:
    sr = None
    HAVE_SPEECH_RECOGNITION = False

try:
    from gtts import gTTS
    HAVE_GTTS = True
except ImportError:
    gTTS = None
    HAVE_GTTS = False


# Supported language definitions and BCP-47 mappings
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en-IN": "English (India) — Recommended for Indian English & Tanglish",
    "en-US": "English (US)",
    "ta-IN": "Tamil (தமிழ் - India)",
    "ta-LK": "Tamil (தமிழ் - Sri Lanka)",
    "tanglish": "Tanglish / Mixed (Tamil + English)",
}


def resolve_language_tag(language: Optional[str]) -> str:
    """
    Normalizes colloquial or ISO language tags into standard BCP-47 speech recognition codes.
    Defaults to 'en-IN' for Indian English and regional phrasing.
    """
    if not language:
        return "en-IN"
    clean = str(language).strip().lower()
    if clean in ("ta", "ta-in", "tamil", "தமிழ்", "tam"):
        return "ta-IN"
    elif clean in ("ta-lk", "sri lanka"):
        return "ta-LK"
    elif clean in ("en-us", "us", "american"):
        return "en-US"
    elif clean in ("en-gb", "uk", "british"):
        return "en-GB"
    elif clean in ("tanglish", "mixed", "ta-en", "en-ta"):
        return "en-IN"  # Best acoustic baseline for Google Web Speech
    elif clean in ("en", "en-in", "india", "indian"):
        return "en-IN"
    return "en-IN"


def calculate_pcm_rms(audio_data: Any) -> float:
    """
    Calculates Root-Mean-Square (RMS) audio energy to distinguish silence
    from spoken speech and detect muted microphones.
    """
    try:
        raw = audio_data.get_raw_data()
        if not raw:
            return 0.0
        count = len(raw) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"<{count}h", raw)
        sum_sq = sum(s * s for s in shorts)
        return math.sqrt(sum_sq / count)
    except Exception:
        return 100.0  # Fallback: assume non-silent if calculation cannot run


def _transcribe_with_gemini(
    audio_bytes: bytes,
    language: str = "en-IN",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribes audio using Google Gemini 1.5 Flash multimodal understanding.
    Exceptional for noisy recordings, Indian English, Tamil, and Tanglish mixed speech.
    """
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        return {
            "success": False,
            "text": "",
            "engine_used": "Google Gemini 1.5 Flash",
            "error": "GEMINI_API_KEY is not configured.",
            "error_type": "missing_api_key",
        }

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    clean_lang = (language or "").strip().lower()

    if "ta" in clean_lang:
        prompt = (
            "Transcribe the spoken speech in this audio recording verbatim into Tamil script (தமிழ்). "
            "If any English technical terms are spoken, you may write them in English or phonetically in Tamil. "
            "Output ONLY the exact transcribed text without quotes, commentary, or markdown formatting."
        )
    elif "tanglish" in clean_lang or "mixed" in clean_lang:
        prompt = (
            "The speaker in this audio is speaking Tanglish (a mix of Tamil and English) or Indian English. "
            "Accurately transcribe the spoken words verbatim. Keep English words in English and Tamil words in Tamil script or Romanized script as spoken. "
            "Output ONLY the exact transcribed text without quotes, commentary, or markdown formatting."
        )
    else:
        prompt = (
            "Transcribe the spoken speech in this audio recording verbatim into English text. "
            "The speaker may have an Indian accent. Transcribe the words exactly as spoken. "
            "Output ONLY the exact transcribed text without quotes, commentary, or markdown formatting."
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": b64_audio,
                    }
                },
                {
                    "text": prompt,
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 350,
        }
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text_out = parts[0].get("text", "").strip()
                    text_out = re.sub(r'^["\']|["\']$', '', text_out).strip()
                    if text_out:
                        return {
                            "success": True,
                            "text": text_out,
                            "engine_used": "Google Gemini 1.5 Flash (Multimodal Audio)",
                            "error": None,
                            "error_type": None,
                        }
            return {
                "success": False,
                "text": "",
                "engine_used": "Google Gemini 1.5 Flash",
                "error": "Gemini returned an empty transcription for this recording.",
                "error_type": "no_speech_detected",
            }
    except urllib.error.HTTPError as he:
        err_msg = he.read().decode("utf-8") if he.fp else str(he)
        return {
            "success": False,
            "text": "",
            "engine_used": "Google Gemini 1.5 Flash",
            "error": f"Gemini API error ({he.code}): {err_msg[:200]}",
            "error_type": "api_error",
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "engine_used": "Google Gemini 1.5 Flash",
            "error": f"Gemini audio processing error: {str(e)}",
            "error_type": "network_error",
        }


def clean_text_for_speech(text: str, max_chars: int = 800) -> str:
    """
    Normalizes and cleans assistant markdown text for natural, fluent spoken pronunciation.
    
    Removes:
    - Code blocks and backticks
    - Markdown headings, bold, italics, strikethroughs, blockquotes
    - Hyperlinks and file URLs
    - Document citations (e.g., [Page 1], [Page 2-3])
    - Bullet prefixes and excessive whitespace
    - Raw emoji sequences that disrupt TTS flow
    
    Truncates gracefully at a sentence boundary if text exceeds max_chars.
    """
    if not text:
        return ""

    cleaned = str(text)

    # 1. Remove markdown code blocks completely
    cleaned = re.sub(r"```[\s\S]*?```", " [Code block omitted for speech.] ", cleaned)

    # 2. Remove inline code backticks: `variable` -> variable
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    # 3. Remove document citations: [Page 1], [Page 2-3]
    cleaned = re.sub(r"\[Page\s*[\d\-,\s]+\]", "", cleaned, flags=re.IGNORECASE)

    # 4. Remove markdown hyperlinks: [Link Text](https://...) -> Link Text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)

    # 5. Remove raw URLs
    cleaned = re.sub(r"https?://\S+", " web link ", cleaned)

    # 6. Remove file links: [text](file:///...) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\(file://[^\)]+\)", r"\1", cleaned)

    # 7. Remove markdown headers, list markers, and quotes: #, ##, *, -, >, |
    cleaned = re.sub(r"^[ \t]*[#*>\-+][ \t]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[*_~]+", "", cleaned)  # Bold, italic, strikethrough

    # 8. Clean up Markdown tables: remove table separator lines |---|---|
    cleaned = re.sub(r"\|[-:\s|]+\|", "", cleaned)
    cleaned = cleaned.replace("|", ", ")

    # 9. Clean up technical symbols and emojis (keep standard punctuation)
    # Remove emojis using unicode range
    cleaned = re.sub(
        r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\u2300-\u23ff\u2b50-\u2b55]",
        "",
        cleaned,
    )

    # 10. Normalize common abbreviations for clearer pronunciation
    cleaned = re.sub(r"\be\.g\.,?\s*", "for example, ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi\.e\.,?\s*", "that is, ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bvs\.?\s*", "versus ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\betc\.?\b", "etcetera", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAPI\b", "A P I", cleaned)
    cleaned = re.sub(r"\bSQL\b", "sequel", cleaned)
    cleaned = re.sub(r"\bSQLite\b", "SQLite", cleaned)
    cleaned = re.sub(r"\bNLP\b", "N L P", cleaned)
    cleaned = re.sub(r"\bLLM\b", "L L M", cleaned)

    # 11. Normalize excessive whitespace and line breaks
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 12. Safe truncation at sentence boundary
    if len(cleaned) > max_chars:
        truncated = cleaned[:max_chars]
        last_period = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
        if last_period > int(max_chars * 0.5):
            cleaned = truncated[:last_period + 1]
        else:
            cleaned = truncated.rstrip() + "..."

    return cleaned


def transcribe_audio(
    audio_bytes: bytes,
    audio_format: str = "wav",
    engine: str = "google",
    consent_external: bool = True,
    language: str = "en-IN",
) -> Dict[str, Any]:
    """
    Transcribes spoken audio into text with multi-language and acoustic calibration.
    
    Supports:
    - English (India): 'en-IN' (Default / Recommended for Indian English & tech vocabulary)
    - English (US): 'en-US'
    - Tamil (India): 'ta-IN' (தமிழ்)
    - Tanglish / Mixed: 'tanglish' (Uses en-IN / Gemini multimodal code-mixing)
    
    Engines:
    - 'google': Google Web Speech API (Always Free, zero API key required)
    - 'gemini': Google Gemini 1.5 Flash Multimodal Audio (Uses GEMINI_API_KEY)
    - 'auto': Intelligently selects Gemini if key configured, otherwise Google Web Speech
    
    Args:
        audio_bytes: Binary audio data from microphone recording.
        audio_format: File audio container format ('wav', 'mp3', 'ogg').
        engine: STT engine ('google', 'gemini', 'auto').
        consent_external: Explicit user consent for external network transmission.
        language: Language code ('en-IN', 'en-US', 'ta-IN', 'tanglish').
        
    Returns:
        Dict containing:
            - 'success': bool
            - 'text': transcribed text string
            - 'engine_used': string name of provider/engine
            - 'language': resolved language tag used
            - 'error': error message if failed, else None
            - 'error_type': category of error if failed
    """
    # 1. Validate Audio Presence
    if not audio_bytes or len(audio_bytes) < 128:
        return {
            "success": False,
            "text": "",
            "engine_used": engine,
            "language": language,
            "error": "No audio captured or recording was empty. Please click Record, speak clearly, and click Stop.",
            "error_type": "empty_recording",
        }

    # 2. Privacy & User Consent Guardrail (must be before any network call)
    if not consent_external:
        return {
            "success": False,
            "text": "",
            "engine_used": engine,
            "language": language,
            "error": (
                "External audio processing is disabled. "
                "To transcribe speech using Google Web Speech, please enable external processing consent in Voice settings."
            ),
            "error_type": "consent_required",
        }

    # 3. Resolve BCP-47 Language Tag
    resolved_lang = resolve_language_tag(language)

    # 4. Engine: Google Gemini Multimodal Audio (if explicitly selected or auto with key)
    use_gemini = False
    clean_engine = (engine or "google").strip().lower()
    if clean_engine == "gemini":
        use_gemini = True
    elif clean_engine == "auto" and os.getenv("GEMINI_API_KEY", "").strip():
        use_gemini = True

    if use_gemini:
        gemini_res = _transcribe_with_gemini(audio_bytes, language=language)
        if gemini_res["success"]:
            gemini_res["language"] = resolved_lang
            return gemini_res
        # If gemini was explicitly requested and failed, return its error
        if clean_engine == "gemini":
            gemini_res["language"] = resolved_lang
            return gemini_res
        # Otherwise if auto, fall back gracefully to Google Web Speech API

    # 5. Engine: Google Web Speech API (Default, Free, Zero-Config)
    if not HAVE_SPEECH_RECOGNITION:
        return {
            "success": False,
            "text": "",
            "engine_used": engine,
            "language": resolved_lang,
            "error": "SpeechRecognition library is not installed. Please run: pip install SpeechRecognition",
            "error_type": "library_missing",
        }

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    audio_buffer = io.BytesIO(audio_bytes)
    try:
        with sr.AudioFile(audio_buffer) as source:
            try:
                # Dynamic noise calibration to adjust energy threshold to ambient room acoustics
                dur = min(0.2, source.DURATION * 0.2 if hasattr(source, "DURATION") and source.DURATION else 0.2)
                recognizer.adjust_for_ambient_noise(source, duration=dur)
            except Exception:
                pass
            audio_data = recognizer.record(source)
    except ValueError as ve:
        return {
            "success": False,
            "text": "",
            "engine_used": "Google Web Speech API",
            "language": resolved_lang,
            "error": f"Audio format could not be decoded as PCM WAV ({ve}). Please ensure your browser supports WAV recording.",
            "error_type": "unsupported_format",
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "engine_used": "Google Web Speech API",
            "language": resolved_lang,
            "error": f"Failed to read audio data: {str(e)}",
            "error_type": "audio_read_error",
        }

    try:
        transcribed = recognizer.recognize_google(audio_data, language=resolved_lang)
        return {
            "success": True,
            "text": transcribed.strip(),
            "engine_used": f"Google Web Speech API ({resolved_lang})",
            "language": resolved_lang,
            "error": None,
            "error_type": None,
        }
    except sr.UnknownValueError:
        lang_name = SUPPORTED_LANGUAGES.get(resolved_lang, resolved_lang)
        rms = calculate_pcm_rms(audio_data)
        if rms < 25.0:
            msg = f"Could not recognize any speech in {lang_name} because the recording was silent or too quiet. Please check your microphone volume and speak clearly."
        else:
            msg = f"Could not recognize any speech in {lang_name}. Please speak clearly and ensure background noise is minimized."
        return {
            "success": False,
            "text": "",
            "engine_used": f"Google Web Speech API ({resolved_lang})",
            "language": resolved_lang,
            "error": msg,
            "error_type": "no_speech_detected",
        }
    except sr.RequestError as re_err:
        return {
            "success": False,
            "text": "",
            "engine_used": f"Google Web Speech API ({resolved_lang})",
            "language": resolved_lang,
            "error": f"Speech recognition service unreachable ({re_err}). Please check your internet connection.",
            "error_type": "network_error",
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "engine_used": f"Google Web Speech API ({resolved_lang})",
            "language": resolved_lang,
            "error": f"Speech recognition encountered an unexpected error: {str(e)}",
            "error_type": "transcription_error",
        }


def synthesize_speech(
    text: str,
    engine: str = "gtts",
    lang: str = "auto",
    slow: bool = False,
    max_chars: int = 800,
    tld: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesizes natural spoken audio from text using gTTS with bilingual (Tamil & English) support.
    Automatically detects Tamil unicode and selects Tamil speech synthesis.
    
    Args:
        text: Raw text or markdown string from assistant.
        engine: TTS engine ('gtts').
        lang: Language code ('auto', 'en', 'ta', 'en-in', 'en-us').
        slow: Speak at slower speed if True.
        max_chars: Maximum characters to synthesize for safety.
        tld: Top-level domain for regional accent/cadence (e.g. 'co.in', 'com', 'co.uk').
        
    Returns:
        Dict containing success, audio_bytes, format, engine_used, cleaned_text, language_used, error.
    """
    if not text or not str(text).strip():
        return {
            "success": False,
            "audio_bytes": None,
            "format": None,
            "engine_used": engine,
            "cleaned_text": "",
            "language_used": "en",
            "error": "Text is empty. Nothing to synthesize.",
            "error_type": "empty_text",
        }

    cleaned = clean_text_for_speech(text, max_chars=max_chars)
    if not cleaned:
        return {
            "success": False,
            "audio_bytes": None,
            "format": None,
            "engine_used": engine,
            "cleaned_text": "",
            "language_used": "en",
            "error": "Text contained only code blocks or unpronounceable characters.",
            "error_type": "unpronounceable_text",
        }

    if not HAVE_GTTS:
        return {
            "success": False,
            "audio_bytes": None,
            "format": None,
            "engine_used": engine,
            "cleaned_text": cleaned,
            "language_used": "en",
            "error": "gTTS library is not installed. Please run: pip install gTTS",
            "error_type": "library_missing",
        }

    # Auto-detect language: if Tamil script is present, synthesize in Tamil ('ta')
    target_lang = "en"
    clean_lang = (lang or "auto").strip().lower()

    if clean_lang in ("ta", "ta-in", "tamil") or (clean_lang in ("auto", "en", "en-in", "en-us") and bool(re.search(r"[\u0B80-\u0BFF]", cleaned))):
        target_lang = "ta"
        chosen_tld = tld or "com"
    elif clean_lang in ("en-us", "us"):
        target_lang = "en"
        chosen_tld = tld or "com"
    elif clean_lang in ("en-gb", "uk"):
        target_lang = "en"
        chosen_tld = tld or "co.uk"
    else:
        target_lang = "en"
        chosen_tld = tld or "co.in"

    try:
        try:
            tts = gTTS(text=cleaned, lang=target_lang, tld=chosen_tld, slow=slow)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
        except Exception:
            # Fallback to standard tld="com" and lang="en" if chosen tld or dialect failed
            tts = gTTS(text=cleaned, lang="en", tld="com", slow=slow)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            target_lang = "en"

        buffer.seek(0)
        audio_bytes = buffer.read()

        lang_label = "Tamil" if target_lang == "ta" else "English"
        return {
            "success": True,
            "audio_bytes": audio_bytes,
            "format": "audio/mp3",
            "engine_used": f"Google Text-to-Speech (gTTS - {lang_label})",
            "cleaned_text": cleaned,
            "language_used": target_lang,
            "error": None,
            "error_type": None,
        }
    except Exception as e:
        return {
            "success": False,
            "audio_bytes": None,
            "format": None,
            "engine_used": "gTTS",
            "cleaned_text": cleaned,
            "language_used": target_lang,
            "error": f"Text-to-speech synthesis failed ({str(e)}). Text output remains fully available.",
            "error_type": "synthesis_error",
        }


def get_voice_capabilities() -> Dict[str, Any]:
    """
    Returns diagnostic and privacy metadata regarding voice hardware,
    libraries, supported STT/TTS engines, bilingual support, and consent requirements.
    """
    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
    return {
        "stt": {
            "available": HAVE_SPEECH_RECOGNITION,
            "default_engine": "Google Web Speech API",
            "supported_engines": ["Google Web Speech API", "Google Gemini 1.5 Flash (Multimodal Audio)"] if has_gemini else ["Google Web Speech API"],
            "supported_languages": SUPPORTED_LANGUAGES,
            "cost": "$0.00 (Free, no API key required for Google Web Speech)",
            "privacy_model": (
                "Transient network transmission via HTTPS to Google's speech recognition endpoint. "
                "Audio is converted to text in real-time and never retained on disk."
            ),
            "requires_consent": True,
        },
        "tts": {
            "available": HAVE_GTTS,
            "default_engine": "Google Text-to-Speech (gTTS)",
            "supported_languages": {
                "en": "English (Indian English default via .co.in)",
                "ta": "Tamil (தமிழ்)",
            },
            "cost": "$0.00 (Free, no API key required)",
            "privacy_model": (
                "Text snippets are sent via HTTPS to generate MP3 speech audio. "
                "No private user identifiers are attached."
            ),
            "output_format": "audio/mp3",
        },
        "controls": [
            "Start & Stop recording in browser",
            "Language Selector: English (India), English (US), Tamil, Tanglish",
            "Transcript Preview & Inline Editing before AI submission",
            "Re-record & Retry controls",
            "Play / Pause audio playback",
            "Seek / Replay scrubber",
            "Download audio response (.mp3)",
        ],
        "recording_policy": (
            "User-initiated recording only. Continuous background listening is strictly forbidden."
        ),
    }
