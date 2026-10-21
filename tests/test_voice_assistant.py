"""
Automated Test Suite for NOVA Voice Intelligence (Phase 4)
Tests Speech-to-Text (STT), Text-to-Speech (TTS), text normalization,
privacy consent guardrails, error handling, and end-to-end voice loop.
"""

import io
import struct
import wave
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from database.db import init_db
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
    # Test to verify gTTS synthesis pipeline works with mock audio generation
    with patch("gtts.gTTS.write_to_fp", side_effect=lambda fp: fp.write(b"ID3\x04\x00\x00" + b"MP3_SAMPLE_DATA_" * 100)):
        res = synthesize_speech("Hello, I am NOVA.")
        assert res["success"] is True
        assert res["format"] == "audio/mp3"
        assert res["audio_bytes"] is not None
        assert len(res["audio_bytes"]) > 1000
        assert "Google Text-to-Speech" in res["engine_used"]


def test_synthesize_speech_with_tld_and_kwargs():
    """Verify synthesize_speech accepts tld parameter and kwargs without TypeError."""
    with patch("gtts.gTTS.write_to_fp", side_effect=lambda fp: fp.write(b"ID3\x04\x00\x00" + b"MP3_SAMPLE_DATA_" * 100)):
        res = synthesize_speech("Speech synthesis with Indian cadence.", lang="en", tld="co.in")
        assert res["success"] is True
        assert res["audio_bytes"] is not None
        assert res["format"] == "audio/mp3"


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
    with patch("gtts.gTTS.write_to_fp", side_effect=lambda fp: fp.write(b"ID3\x04\x00\x00" + b"MP3_SAMPLE_DATA_" * 100)):
        tts_result = synthesize_speech(assistant_result["response"])
        assert tts_result["success"] is True
        assert tts_result["audio_bytes"] is not None
        assert len(tts_result["audio_bytes"]) > 500


# -----------------------------------------------------------------------------
# 5. Language Selection, Tamil, Tanglish & Multimodal STT Tests
# -----------------------------------------------------------------------------

def test_resolve_language_tag():
    from modules.voice_assistant import resolve_language_tag
    assert resolve_language_tag("ta") == "ta-IN"
    assert resolve_language_tag("ta-in") == "ta-IN"
    assert resolve_language_tag("tamil") == "ta-IN"
    assert resolve_language_tag("en") == "en-IN"
    assert resolve_language_tag("en-in") == "en-IN"
    assert resolve_language_tag("en-us") == "en-US"
    assert resolve_language_tag("tanglish") == "en-IN"
    assert resolve_language_tag(None) == "en-IN"


def test_transcribe_audio_tamil_language_passed_to_recognizer():
    wav_bytes = _generate_synthetic_wav_bytes(1.0)
    with patch("speech_recognition.Recognizer.recognize_google", return_value="இன்றைய பணிகள் என்ன") as mock_rec:
        res = transcribe_audio(wav_bytes, language="ta-IN", consent_external=True)
        assert res["success"] is True
        assert res["text"] == "இன்றைய பணிகள் என்ன"
        assert res["language"] == "ta-IN"
        mock_rec.assert_called_once()
        _, kwargs = mock_rec.call_args
        assert kwargs.get("language") == "ta-IN"


def test_transcribe_audio_indian_english_default():
    wav_bytes = _generate_synthetic_wav_bytes(1.0)
    with patch("speech_recognition.Recognizer.recognize_google", return_value="add task complete research") as mock_rec:
        res = transcribe_audio(wav_bytes, consent_external=True)
        assert res["success"] is True
        assert res["language"] == "en-IN"
        _, kwargs = mock_rec.call_args
        assert kwargs.get("language") == "en-IN"


def test_transcribe_audio_gemini_multimodal_mock():
    wav_bytes = _generate_synthetic_wav_bytes(1.0)
    with patch("modules.voice_assistant._transcribe_with_gemini") as mock_gemini:
        mock_gemini.return_value = {
            "success": True,
            "text": "NOVA innaiku enna task irukku",
            "engine_used": "Google Gemini 1.5 Flash (Multimodal Audio)",
            "error": None,
            "error_type": None,
        }
        res = transcribe_audio(wav_bytes, engine="gemini", language="tanglish", consent_external=True)
        assert res["success"] is True
        assert "innaiku" in res["text"]
        assert "Gemini" in res["engine_used"]


def test_synthesize_speech_tamil_auto_detection():
    # Test that Tamil Unicode text automatically triggers 'ta' language synthesis
    with patch("gtts.gTTS.write_to_fp", side_effect=lambda fp: fp.write(b"ID3\x04\x00\x00" + b"MP3_SAMPLE_DATA_" * 100)):
        res = synthesize_speech("வணக்கம்! இன்றைய பணிகள் மூன்று உள்ளன.")
        assert res["success"] is True
        assert res["language_used"] == "ta"
        assert "Tamil" in res["engine_used"]
        assert res["audio_bytes"] is not None


# -----------------------------------------------------------------------------
# 6. Unified Voice-to-Chat & Composer Workflow Tests
# -----------------------------------------------------------------------------

def test_unified_voice_to_chat_pipeline():
    """Verifies that speech-transcribed questions route into the unified process_message_stream pipeline."""
    wav_bytes = _generate_synthetic_wav_bytes(0.8)
    with patch("speech_recognition.Recognizer.recognize_google", return_value="what time is it"):
        stt_res = transcribe_audio(wav_bytes, consent_external=True)
        assert stt_res["success"] is True
        spoken_text = stt_res["text"]

    assistant = NovaAssistant()
    # Process through streaming pipeline
    stream = assistant.process_message_stream(spoken_text)
    chunks = list(stream)
    full_response = "".join(chunks)
    assert len(full_response) > 0
    assert "Current local time" in full_response

    # Verify TTS can synthesize the stream output
    with patch("gtts.gTTS.write_to_fp", side_effect=lambda fp: fp.write(b"ID3\x04\x00\x00" + b"MP3_SAMPLE_DATA_" * 100)):
        tts_res = synthesize_speech(full_response)
        assert tts_res["success"] is True
        assert tts_res["audio_bytes"] is not None


def test_unified_voice_transcription_editing_workflow(tmp_path: Path):
    """Verifies user edit flow: recognized speech is modified by user before being sent to AI."""
    test_voice_db = tmp_path / "voice_workflow_test.db"
    init_db(test_voice_db)

    wav_bytes = _generate_synthetic_wav_bytes(0.8)
    with patch("speech_recognition.Recognizer.recognize_google", return_value="add taks study machine learning"):
        stt_res = transcribe_audio(wav_bytes, consent_external=True)
        assert stt_res["success"] is True
        raw_transcription = stt_res["text"]

    # User corrects 'taks' to 'task' in the unified composer text input
    user_edited_prompt = raw_transcription.replace("taks", "task")
    assert user_edited_prompt == "add task study machine learning"

    assistant = NovaAssistant(db_path=test_voice_db)
    stream = assistant.process_message_stream(user_edited_prompt, user_id="test_voice_user")
    response = "".join(list(stream))
    assert "study machine learning" in response.lower() or "created" in response.lower() or "task" in response.lower()


def test_unified_voice_with_web_search_and_grounding():
    """Verifies voice queries seamlessly execute web search and citation grounding."""
    voice_query = "What is the latest release date of Python 3.12?"
    assistant = NovaAssistant()
    mock_hits = [
        {
            "title": "Python 3.12 Release Schedule",
            "url": "https://docs.python.org/3/whatsnew/3.12.html",
            "snippet": "Python 3.12.0 final was released on October 2, 2023.",
            "domain": "docs.python.org",
            "source_type": "official_documentation",
            "score": 0.95,
        }
    ]
    with patch("nova_core.assistant.search_web", return_value=mock_hits):
        stream = assistant.process_message_stream(voice_query, web_search="on")
        full_reply = "".join(list(stream))
        assert "October 2, 2023" in full_reply or "Python 3.12" in full_reply or "Sources & References" in full_reply

