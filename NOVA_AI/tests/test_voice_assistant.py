"""
Automated Test Suite for NOVA Voice Intelligence (Phase 4)
Tests Speech-to-Text (STT), Text-to-Speech (TTS), text normalization,
privacy consent guardrails, error handling, and end-to-end voice loop.
"""

import io
import struct
import wave
from unittest.mock import MagicMock, patch
import pytest

from modules.voice_assistant import (
    clean_text_for_speech,
    transcribe_audio,
    synthesize_speech,
    get_voice_capabilities,
)
from nova_core.assistant import NovaAssistant


def _generate_synthetic_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates valid in-memory PCM 16-bit mono WAV bytes for testing."""
    buf = io.BytesIO()
    num_samples = int(duration_sec * sample_rate)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))
    buf.seek(0)
    return buf.read()


# -----------------------------------------------------------------------------
# 1. Text Normalization and Speech Cleaning Tests
# -----------------------------------------------------------------------------

def test_clean_text_for_speech_strips_code_blocks():
    raw = (
        "Here is the solution to your algorithm:\n"
        "```python\n"
        "def hello():\n"
        "    return 'world'\n"
        "```\n"
        "This function runs in O(1) time complexity."
    )
    cleaned = clean_text_for_speech(raw)
    assert "def hello" not in cleaned
    assert "return 'world'" not in cleaned
    assert "Code block omitted for speech." in cleaned
    assert "This function runs in O(1) time complexity." in cleaned


def test_clean_text_for_speech_strips_citations_and_links():
    raw = (
        "According to the syllabus [Page 3], the midterm covers Unit 2 [Page 4-5]. "
        "Visit [Course Portal](https://university.edu/ai) for updates."
    )
    cleaned = clean_text_for_speech(raw)
    assert "[Page 3]" not in cleaned
    assert "[Page 4-5]" not in cleaned
    assert "Course Portal" in cleaned
    assert "https://university.edu/ai" not in cleaned


def test_clean_text_for_speech_expands_abbreviations():
    raw = "You can use e.g. SQLite for persistence, i.e. local storage, with an API for NLP."
    cleaned = clean_text_for_speech(raw)
    assert "for example," in cleaned
    assert "that is," in cleaned
    assert "A P I" in cleaned
    assert "N L P" in cleaned


def test_clean_text_for_speech_sentence_truncation():
    long_text = "Sentence one is clear. Sentence two is very informative. Sentence three is detailed. Sentence four is extra."
    cleaned = clean_text_for_speech(long_text, max_chars=55)
    assert len(cleaned) <= 60
    assert cleaned.endswith(".") or cleaned.endswith("...")
    assert "Sentence one is clear." in cleaned


# -----------------------------------------------------------------------------
# 2. Speech-to-Text (STT) Tests
# -----------------------------------------------------------------------------

def test_transcribe_audio_empty_recording_returns_error():
    res = transcribe_audio(b"", consent_external=True)
    assert res["success"] is False
    assert res["error_type"] == "empty_recording"
    assert "empty" in res["error"].lower()

    res_small = transcribe_audio(b"12345", consent_external=True)
    assert res_small["success"] is False
    assert res_small["error_type"] == "empty_recording"


def test_transcribe_audio_refuses_external_when_no_consent():
    wav_bytes = _generate_synthetic_wav_bytes(0.5)
    res = transcribe_audio(wav_bytes, consent_external=False)
    assert res["success"] is False
    assert res["error_type"] == "consent_required"
    assert "consent" in res["error"].lower()


def test_transcribe_audio_corrupted_format_returns_error():
    corrupted_data = b"RIFF" + b"Not a real wav file header content at all" * 10
    res = transcribe_audio(corrupted_data, consent_external=True)
    assert res["success"] is False
    assert res["error_type"] in ("unsupported_format", "audio_read_error")


def test_transcribe_audio_successful_mocked_recognition():
    wav_bytes = _generate_synthetic_wav_bytes(1.0)
    with patch("speech_recognition.Recognizer.recognize_google", return_value="show pending tasks"):
        res = transcribe_audio(wav_bytes, consent_external=True)
        assert res["success"] is True
        assert res["text"] == "show pending tasks"
        assert "Google Web Speech API" in res["engine_used"]
        assert res["error"] is None


def test_transcribe_audio_handles_unknown_value_error():
    import speech_recognition as sr
    wav_bytes = _generate_synthetic_wav_bytes(1.0)
    with patch("speech_recognition.Recognizer.recognize_google", side_effect=sr.UnknownValueError()):
        res = transcribe_audio(wav_bytes, consent_external=True)
        assert res["success"] is False
        assert res["error_type"] == "no_speech_detected"
        assert "recognize any speech" in res["error"]


def test_transcribe_audio_handles_network_request_error():
    import speech_recognition as sr
    wav_bytes = _generate_synthetic_wav_bytes(1.0)
    with patch("speech_recognition.Recognizer.recognize_google", side_effect=sr.RequestError("Connection timeout")):
        res = transcribe_audio(wav_bytes, consent_external=True)
        assert res["success"] is False
        assert res["error_type"] == "network_error"
        assert "unreachable" in res["error"].lower()


# -----------------------------------------------------------------------------
# 3. Text-to-Speech (TTS) Tests
# -----------------------------------------------------------------------------

def test_synthesize_speech_empty_text():
    res = synthesize_speech("")
    assert res["success"] is False
    assert res["error_type"] == "empty_text"
    assert res["audio_bytes"] is None


def test_synthesize_speech_success_with_gtts():
    # Real test with a short prompt to verify gTTS works end-to-end
    res = synthesize_speech("Hello, I am NOVA.")
    assert res["success"] is True
    assert res["format"] == "audio/mp3"
    assert res["audio_bytes"] is not None
    assert len(res["audio_bytes"]) > 1000
    assert "Google Text-to-Speech" in res["engine_used"]


def test_synthesize_speech_network_error_graceful_handling():
    with patch("gtts.gTTS.write_to_fp", side_effect=Exception("Simulated socket timeout")):
        res = synthesize_speech("Valid message text here.")
        assert res["success"] is False
        assert res["error_type"] == "synthesis_error"
        assert "failed" in res["error"].lower()
        assert res["audio_bytes"] is None


# -----------------------------------------------------------------------------
# 4. Capabilities & End-to-End Voice Loop Tests
# -----------------------------------------------------------------------------

def test_get_voice_capabilities():
    caps = get_voice_capabilities()
    assert "stt" in caps
    assert "tts" in caps
    assert caps["stt"]["available"] is True
    assert "$0.00" in caps["stt"]["cost"]
    assert caps["tts"]["available"] is True
    assert "$0.00" in caps["tts"]["cost"]
    assert caps["stt"]["requires_consent"] is True


def test_end_to_end_voice_query_through_nova_assistant():
    """
    Simulates speech transcribed as a command -> processed by NovaAssistant -> converted to speech.
    """
    wav_bytes = _generate_synthetic_wav_bytes(1.0)
    
    # 1. Transcribe
    with patch("speech_recognition.Recognizer.recognize_google", return_value="what time is it"):
        transcription = transcribe_audio(wav_bytes, consent_external=True)
        assert transcription["success"] is True
        spoken_question = transcription["text"]
        assert spoken_question == "what time is it"

    # 2. Assistant Orchestration
    assistant = NovaAssistant()
    assistant_result = assistant.process_message(spoken_question)
    assert assistant_result["tool_called"] == "get_current_time"
    assert "Current local time" in assistant_result["response"]

    # 3. Text-to-Speech Synthesis
    tts_result = synthesize_speech(assistant_result["response"])
    assert tts_result["success"] is True
    assert tts_result["audio_bytes"] is not None
    assert len(tts_result["audio_bytes"]) > 500
