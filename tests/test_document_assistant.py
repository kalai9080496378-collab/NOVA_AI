"""
Unit & Integration Tests for NOVA Document Intelligence (Phase 3)
Tests document ingestion, page extraction, chunking, TF-IDF RAG retrieval,
grounded Q&A with citations, refusal on out-of-scope questions,
follow-up conversations, user-controlled deletion, and error handling.
"""

import io
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pypdf

from database.db import init_db
from modules.document_assistant import (
    save_and_process_document,
    get_documents,
    get_document_by_id,
    get_document_chunks,
    delete_document,
    delete_all_documents,
    retrieve_relevant_chunks,
    answer_document_question,
    summarize_document,
)


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch) -> Path:
    """Creates an isolated temporary SQLite database and storage directory."""
    db_file = tmp_path / "test_docs_nova.db"
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    init_db(db_file)
    monkeypatch.setattr("database.db.DATABASE_PATH", db_file)
    monkeypatch.setattr("nova_core.config.DATABASE_PATH", db_file)
    monkeypatch.setattr("modules.document_assistant.DOCUMENTS_DIR", docs_dir)
    return db_file


def _create_sample_txt() -> bytes:
    content = (
        "Introduction to Deep Learning\n\n"
        "Deep learning is a subset of machine learning based on artificial neural networks. "
        "Convolutional Neural Networks (CNNs) are primarily used for image processing and computer vision.\n\n"
        "Optimization and Backpropagation\n\n"
        "Stochastic Gradient Descent (SGD) and Adam are common optimization algorithms. "
        "The learning rate determines the step size during gradient updates. A high learning rate causes divergence.\n\n"
        "Recurrent Architectures\n\n"
        "Recurrent Neural Networks (RNNs) and LSTMs process sequential time-series data and natural language text."
    )
    return content.encode("utf-8")


# =============================================================================
# 1. Ingestion, Chunking & Extraction Tests
# =============================================================================

def test_ingest_txt_document_success(temp_db: Path):
    """Verify TXT ingestion, metadata calculation, and chunk creation."""
    txt_bytes = _create_sample_txt()
    doc = save_and_process_document(txt_bytes, "deep_learning_notes.txt", db_path=temp_db)

    assert doc is not None
    assert doc["filename"] == "deep_learning_notes.txt"
    assert doc["file_type"] == "txt"
    assert doc["word_count"] > 40
    assert doc["page_count"] >= 1

    chunks = get_document_chunks(doc["id"], db_path=temp_db)
    assert len(chunks) >= 1
    assert all("content" in c and c["page_number"] >= 1 for c in chunks)


def test_ingest_empty_file_raises_error(temp_db: Path):
    """Verify empty files (0 bytes) are rejected with clear error."""
    with pytest.raises(ValueError, match="Uploaded file is empty"):
        save_and_process_document(b"", "empty.txt", db_path=temp_db)


def test_ingest_unsupported_format_raises_error(temp_db: Path):
    """Verify unsupported extensions are rejected."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        save_and_process_document(b"fake binary", "program.exe", db_path=temp_db)


def test_ingest_oversized_file_raises_error(temp_db: Path, monkeypatch):
    """Verify files exceeding size limits are rejected."""
    monkeypatch.setattr("modules.document_assistant.MAX_FILE_SIZE_MB", 0.001)  # ~1 KB
    huge_bytes = b"A" * 5000
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        save_and_process_document(huge_bytes, "huge.txt", db_path=temp_db)


# =============================================================================
# 2. Document Management & Deletion Tests
# =============================================================================

def test_document_listing_and_deletion(temp_db: Path):
    """Verify document listing, retrieval, and cascading deletion."""
    txt_bytes = _create_sample_txt()
    d1 = save_and_process_document(txt_bytes, "doc1.txt", db_path=temp_db)
    d2 = save_and_process_document(txt_bytes, "doc2.txt", db_path=temp_db)

    docs = get_documents(db_path=temp_db)
    assert len(docs) == 2

    # Delete d1
    del_ok = delete_document(d1["id"], db_path=temp_db)
    assert del_ok is True
    assert get_document_by_id(d1["id"], db_path=temp_db) is None
    assert len(get_document_chunks(d1["id"], db_path=temp_db)) == 0

    # Delete all
    del_count = delete_all_documents(db_path=temp_db)
    assert del_count == 1
    assert len(get_documents(db_path=temp_db)) == 0


# =============================================================================
# 3. TF-IDF Semantic & Lexical Retrieval Tests
# =============================================================================

def test_tfidf_retrieval_ranking(temp_db: Path):
    """Verify TF-IDF RAG retrieval ranks chunks by query relevance."""
    txt_bytes = _create_sample_txt()
    doc = save_and_process_document(txt_bytes, "ml_guide.txt", db_path=temp_db)

    # Query about learning rate / gradient descent
    chunks = retrieve_relevant_chunks(doc["id"], "What is the role of learning rate in optimization?", top_k=2, db_path=temp_db)
    assert len(chunks) > 0
    top_chunk = chunks[0]

    assert "learning rate" in top_chunk["content"].lower() or "gradient" in top_chunk["content"].lower()
    assert top_chunk["score"] > 0.1
    assert top_chunk["page_number"] >= 1


# =============================================================================
# 4. Grounded Question Answering & Citations Tests
# =============================================================================

@patch("requests.post")
def test_grounded_answer_with_citations(mock_post, temp_db: Path, monkeypatch):
    """Verify grounded answer generation with page citations using mocked LLM."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_key")

    txt_bytes = _create_sample_txt()
    doc = save_and_process_document(txt_bytes, "deep_learning.txt", db_path=temp_db)

    # Mock LLM grounded response with citation
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Convolutional Neural Networks (CNNs) are primarily used for image processing [Page 1]."}],
                    "role": "model",
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    res = answer_document_question(
        document_id=doc["id"],
        question="What are CNNs used for?",
        db_path=temp_db,
    )

    assert res["grounded"] is True
    assert res["mode"] == "llm"
    assert "image processing" in res["answer"]
    assert 1 in res["citations"]
    assert len(res["retrieved_chunks"]) > 0


def test_answer_when_information_not_present_in_document(temp_db: Path):
    """Verify system explicitly states when document does not contain the answer (anti-hallucination)."""
    txt_bytes = _create_sample_txt()
    doc = save_and_process_document(txt_bytes, "deep_learning.txt", db_path=temp_db)

    # Completely unrelated question not present in the text
    res = answer_document_question(
        document_id=doc["id"],
        question="What are the ingredients in an Italian pizza Margherita?",
        db_path=temp_db,
    )

    assert res["grounded"] is True
    assert res["mode"] == "not_found"
    assert "does not contain sufficient information" in res["answer"]


def test_offline_extractive_answer_fallback(temp_db: Path, monkeypatch):
    """Verify offline rule-based extractive answering when no LLM key is configured."""
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    txt_bytes = _create_sample_txt()
    doc = save_and_process_document(txt_bytes, "deep_learning.txt", db_path=temp_db)

    res = answer_document_question(
        document_id=doc["id"],
        question="What optimization algorithms are discussed?",
        db_path=temp_db,
    )

    assert res["grounded"] is True
    assert res["mode"] == "rule_based_extractive"
    assert "Page" in res["answer"]
    assert len(res["citations"]) > 0


# =============================================================================
# 5. Document Summarization Tests
# =============================================================================

def test_document_summarization_offline(temp_db: Path, monkeypatch):
    """Verify structured extractive summary generated when offline."""
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    txt_bytes = _create_sample_txt()
    doc = save_and_process_document(txt_bytes, "deep_learning.txt", db_path=temp_db)

    res = summarize_document(doc["id"], db_path=temp_db)
    assert res["status"] == "success"
    assert res["mode"] == "extractive"
    assert "Document Summary" in res["summary"]
    assert str(doc["word_count"]) in res["summary"]


@patch("requests.post")
def test_document_summarization_with_llm(mock_post, temp_db: Path, monkeypatch):
    """Verify executive summary generated using active LLM."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "## Executive Summary\n\n1. Overview: Deep Learning foundations.\n2. Key Findings: CNNs and Optimization."}],
                    "role": "model",
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    txt_bytes = _create_sample_txt()
    doc = save_and_process_document(txt_bytes, "deep_learning.txt", db_path=temp_db)

    res = summarize_document(doc["id"], db_path=temp_db)
    assert res["status"] == "success"
    assert res["mode"] == "llm"
    assert "Executive Summary" in res["summary"]
