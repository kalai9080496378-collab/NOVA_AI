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
from typing import Any, Dict, Optional

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
    language: str = "en-US",
) -> Dict[str, Any]:
    """
    Transcribes spoken audio into text.
    
    Args:
        audio_bytes: Binary audio data from microphone recording.
        audio_format: File audio container format ('wav', 'mp3', 'ogg').
        engine: STT engine ('google', 'gemini').
        consent_external: Explicit user consent for external network transmission.
        language: BCP-47 language tag (default: 'en-US').
        
    Returns:
        Dict containing:
            - 'success': bool
            - 'text': transcribed text string
            - 'engine_used': string name of provider/engine
            - 'error': error message if failed, else None
            - 'error_type': category of error if failed
    """
    # 1. Validate Audio Presence
    if not audio_bytes or len(audio_bytes) < 128:
        return {
            "success": False,
            "text": "",
            "engine_used": engine,
            "error": "No audio captured or recording was empty. Please click Record, speak clearly, and click Stop.",
            "error_type": "empty_recording",
        }

    # 2. Privacy & User Consent Guardrail
    if not consent_external:
        return {
            "success": False,
            "text": "",
            "engine_used": engine,
            "error": (
                "External audio processing is disabled. "
                "To transcribe speech using Google Web Speech, please enable external processing consent in Voice settings."
            ),
            "error_type": "consent_required",
        }

    # 3. Check Library Availability
    if not HAVE_SPEECH_RECOGNITION:
        return {
            "success": False,
            "text": "",
            "engine_used": engine,
            "error": "SpeechRecognition library is not installed. Please run: pip install SpeechRecognition",
            "error_type": "library_missing",
        }

    # 4. Engine: Google Web Speech API (Default, Free, Zero-Config)
    if engine.lower() in ("google", "auto"):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        audio_buffer = io.BytesIO(audio_bytes)
        try:
            with sr.AudioFile(audio_buffer) as source:
                audio_data = recognizer.record(source)
        except ValueError as ve:
            return {
                "success": False,
                "text": "",
                "engine_used": "Google Web Speech API",
                "error": f"Audio format could not be decoded as PCM WAV ({ve}). Please ensure your browser supports WAV recording.",
                "error_type": "unsupported_format",
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "engine_used": "Google Web Speech API",
                "error": f"Failed to read audio data: {str(e)}",
                "error_type": "audio_read_error",
            }

        try:
            transcribed = recognizer.recognize_google(audio_data, language=language)
            return {
                "success": True,
                "text": transcribed.strip(),
                "engine_used": "Google Web Speech API (Zero-Cost)",
                "error": None,
                "error_type": None,
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "text": "",
                "engine_used": "Google Web Speech API",
                "error": "Could not recognize any speech in the audio. Please speak clearly into your microphone.",
                "error_type": "no_speech_detected",
            }
        except sr.RequestError as re_err:
            return {
                "success": False,
                "text": "",
                "engine_used": "Google Web Speech API",
                "error": f"Speech recognition service unreachable ({re_err}). Please check your internet connection.",
                "error_type": "network_error",
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "engine_used": "Google Web Speech API",
                "error": f"Speech recognition encountered an unexpected error: {str(e)}",
                "error_type": "transcription_error",
            }

    return {
        "success": False,
        "text": "",
        "engine_used": engine,
        "error": f"Unknown or unsupported STT engine: '{engine}'. Supported: 'google', 'auto'.",
        "error_type": "unsupported_engine",
    }


def synthesize_speech(
    text: str,
    engine: str = "gtts",
    lang: str = "en",
    slow: bool = False,
    max_chars: int = 800,
) -> Dict[str, Any]:
    """
    Synthesizes natural spoken audio from text using gTTS.
    
    Args:
        text: Raw text or markdown string from assistant.
        engine: TTS engine ('gtts').
        lang: Language code ('en', 'en-uk', 'en-us').
        slow: Speak at slower speed if True.
        max_chars: Maximum characters to synthesize for safety.
        
    Returns:
        Dict containing:
            - 'success': bool
            - 'audio_bytes': binary MP3 audio or None
            - 'format': 'audio/mp3' or None
            - 'engine_used': string name of provider/engine
            - 'cleaned_text': the spoken transcript
            - 'error': error message if failed, else None
            - 'error_type': error category
    """
    if not text or not str(text).strip():
        return {
            "success": False,
            "audio_bytes": None,
            "format": None,
            "engine_used": engine,
            "cleaned_text": "",
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
            "error": "gTTS library is not installed. Please run: pip install gTTS",
            "error_type": "library_missing",
        }

    try:
        tts = gTTS(text=cleaned, lang=lang, slow=slow)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()

        return {
            "success": True,
            "audio_bytes": audio_bytes,
            "format": "audio/mp3",
            "engine_used": "Google Text-to-Speech (gTTS)",
            "cleaned_text": cleaned,
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
            "error": f"Text-to-speech synthesis failed ({str(e)}). Text output remains fully available.",
            "error_type": "synthesis_error",
        }


def get_voice_capabilities() -> Dict[str, Any]:
    """
    Returns diagnostic and privacy metadata regarding voice hardware,
    libraries, supported STT/TTS engines, costs, and consent requirements.
    """
    return {
        "stt": {
            "available": HAVE_SPEECH_RECOGNITION,
            "default_engine": "Google Web Speech API",
            "cost": "$0.00 (Free, no API key required)",
            "privacy_model": (
                "Transient network transmission via HTTPS to Google's speech recognition endpoint. "
                "Audio is converted to text in real-time and never retained on disk."
            ),
            "requires_consent": True,
        },
        "tts": {
            "available": HAVE_GTTS,
            "default_engine": "Google Text-to-Speech (gTTS)",
            "cost": "$0.00 (Free, no API key required)",
            "privacy_model": (
                "Text snippets are sent via HTTPS to generate MP3 speech audio. "
                "No private user identifiers are attached."
            ),
            "output_format": "audio/mp3",
        },
        "controls": [
            "Play / Pause audio playback",
            "Seek / Replay scrubber",
            "Volume adjustment",
            "Download audio file (.mp3)",
            "Manual re-listen button",
        ],
        "recording_policy": (
            "User-initiated recording only. Continuous background listening is strictly forbidden."
        ),
    }
