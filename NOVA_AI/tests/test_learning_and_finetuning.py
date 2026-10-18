"""
Automated Test Suite for NOVA Self-Learning & Fine-Tuning Pipeline
Tests:
- Thumbs-up / Thumbs-down feedback collection and stats
- User correction recording, deduplication, and updating
- Dynamic in-context few-shot retrieval of verified corrections
- Memory duplicate rejection guardrail
- Memory export to CSV and JSON
- PII and secret scrubbing (API keys, emails, phone numbers, IPs)
- Dataset creation in Chat JSONL format with train/val split
- Compute and cost estimation calculations
- Model evaluation benchmark and rollback mechanism
- End-to-end self-learning: Assistant immediately applies user corrections
"""

import json
from pathlib import Path
import pytest

from database.db import init_db
from memory.memory_manager import save_memory, get_all_memories, export_memories_csv, export_memories_json
from nova_core.learning_system import (
    record_message_feedback,
    get_feedback_summary,
    record_correction,
    get_learning_examples,
    retrieve_relevant_corrections,
    check_memory_duplicate_or_conflict,
    get_learning_system_metrics,
    delete_correction,
)
from nova_core.fine_tuning_pipeline import (
    inspect_fine_tuning_support,
    scrub_pii_and_secrets,
    build_training_dataset,
    estimate_training_compute_and_cost,
    evaluate_model_performance,
    rollback_model_version,
)
from nova_core.assistant import NovaAssistant


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Isolated database fixture for self-learning tests."""
    db_file = tmp_path / "test_learning_nova.db"
    init_db(db_file)
    return db_file


# =============================================================================
# 1. Feedback Tests
# =============================================================================

def test_record_message_feedback(temp_db: Path):
    """Verify thumbs-up and thumbs-down recording with validation."""
    fb1 = record_message_feedback(
        conversation_id="conv_test_1",
        rating="thumbs_up",
        message_id=10,
        feedback_text="Very helpful explanation of Adam optimizer.",
        is_approved_for_training=True,
        db_path=temp_db,
    )
    assert fb1["id"] > 0
    assert fb1["rating"] == "thumbs_up"
    assert fb1["is_approved_for_training"] == 1

    fb2 = record_message_feedback(
        conversation_id="conv_test_1",
        rating="thumbs_down",
        feedback_text="Inaccurate matrix multiplication example.",
        is_approved_for_training=False,
        db_path=temp_db,
    )
    assert fb2["rating"] == "thumbs_down"

    summary = get_feedback_summary(db_path=temp_db)
    assert summary["total_feedback"] == 2
    assert summary["thumbs_up"] == 1
    assert summary["thumbs_down"] == 1
    assert summary["approved_for_training"] == 1

    # Invalid rating raises ValueError
    with pytest.raises(ValueError, match="thumbs_up' or 'thumbs_down"):
        record_message_feedback("conv_test_1", rating="neutral", db_path=temp_db)


# =============================================================================
# 2. User Corrections & In-Context Retrieval Tests
# =============================================================================

def test_record_and_retrieve_corrections(temp_db: Path):
    """Verify storing approved corrections and dynamic in-context retrieval."""
    ex = record_correction(
        user_prompt="What is the default activation function in our neural network?",
        corrected_response="The default activation is GELU with LayerNorm.",
        rejected_response="The default activation is ReLU.",
        explanation="Updated to modern transformer architecture.",
        category="DomainKnowledge",
        is_approved=True,
        db_path=temp_db,
    )
    assert ex["id"] > 0
    assert ex["category"] == "DomainKnowledge"

    # Exact or near query retrieval
    matches = retrieve_relevant_corrections(
        "What is the default activation function in our neural network?",
        threshold=0.5,
        db_path=temp_db,
    )
    assert len(matches) == 1
    assert "GELU" in matches[0]["corrected_response"]

    # Partial / semantic match
    partial = retrieve_relevant_corrections(
        "tell me the activation function neural network default",
        threshold=0.25,
        db_path=temp_db,
    )
    assert len(partial) == 1
    assert "GELU" in partial[0]["corrected_response"]

    # Unrelated query returns nothing
    unrelated = retrieve_relevant_corrections("How do I bake sourdough bread?", db_path=temp_db)
    assert len(unrelated) == 0


def test_correction_deduplication_and_update(temp_db: Path):
    """Verify submitting correction for the same prompt updates the existing record."""
    ex1 = record_correction(
        user_prompt="Which loss function is used for multi-class classification?",
        corrected_response="Cross-Entropy Loss",
        db_path=temp_db,
    )
    ex2 = record_correction(
        user_prompt="Which loss function is used for multi-class classification?",
        corrected_response="Categorical Cross-Entropy Loss with Softmax",
        db_path=temp_db,
    )
    # Same ID, updated content
    assert ex1["id"] == ex2["id"]
    all_ex = get_learning_examples(db_path=temp_db)
    assert len(all_ex) == 1
    assert "Softmax" in all_ex[0]["corrected_response"]


# =============================================================================
# 3. Knowledge Integrity & Deduplication Guardrails
# =============================================================================

def test_memory_deduplication_guardrail(temp_db: Path):
    """Verify exact duplicate memory insertion is rejected to prevent bloat."""
    save_memory("My thesis topic is Graph Neural Networks", category="Academic", db_path=temp_db)

    # Duplicate insertion should raise ValueError
    with pytest.raises(ValueError, match="Duplicate knowledge rejected"):
        save_memory("My thesis topic is Graph Neural Networks", category="Academic", db_path=temp_db)

    # Allowed when allow_duplicate=True
    mem2 = save_memory("My thesis topic is Graph Neural Networks", allow_duplicate=True, db_path=temp_db)
    assert mem2["id"] > 0


def test_memory_export_csv_and_json(temp_db: Path):
    """Verify exporting memories to clean CSV and JSON."""
    save_memory("Use AdamW optimizer with cosine learning rate schedule", category="ML", db_path=temp_db)
    save_memory("Evaluation metric is ROC-AUC", category="ML", db_path=temp_db)

    csv_data = export_memories_csv(db_path=temp_db)
    assert "AdamW" in csv_data
    assert "ROC-AUC" in csv_data
    assert "category,content" in csv_data

    json_data = export_memories_json(db_path=temp_db)
    parsed = json.loads(json_data)
    assert parsed["count"] == 2


# =============================================================================
# 4. PII & Secret Scrubbing Tests
# =============================================================================

def test_scrub_pii_and_secrets():
    """Verify detection and scrubbing of API keys, emails, phone numbers, and IPs."""
    dirty_text = (
        "Here is my Gemini key AIzaSyD9xK8qL1zO4v2N7bY9wP3mR6tU0sX5jQ8 and OpenAI key "
        "sk-1234567890abcdef1234567890abcdef. Contact me at student@university.ac.uk "
        "or +1-555-234-5678. Server IP is 192.168.1.100."
    )
    clean_text, redactions = scrub_pii_and_secrets(dirty_text)

    assert "AIza" not in clean_text
    assert "sk-" not in clean_text
    assert "student@university" not in clean_text
    assert "+1-555" not in clean_text
    assert "192.168.1.100" not in clean_text

    assert "[API_KEY_REDACTED]" in clean_text
    assert "[EMAIL_REDACTED]" in clean_text
    assert "[PHONE_REDACTED]" in clean_text
    assert "[IP_REDACTED]" in clean_text
    assert len(redactions) >= 4


# =============================================================================
# 5. Dataset Building & Compute Estimation Tests
# =============================================================================

def test_build_training_dataset_and_split(temp_db: Path):
    """Verify Chat JSONL dataset creation with train/val split and PII scrubbing."""
    record_correction(
        user_prompt="How do I normalize features in scikit-learn? Contact me at dev@test.com",
        corrected_response="Use StandardScaler from sklearn.preprocessing.",
        is_approved=True,
        db_path=temp_db,
    )
    record_correction(
        user_prompt="What is dropout in deep learning?",
        corrected_response="Dropout randomly zeroes neurons during training to prevent overfitting.",
        is_approved=True,
        db_path=temp_db,
    )

    dataset_meta = build_training_dataset(val_ratio=0.5, db_path=temp_db)
    assert dataset_meta["total_samples"] == 2
    assert dataset_meta["train_samples"] >= 1
    assert dataset_meta["val_samples"] >= 1

    train_file = Path(dataset_meta["train_file"])
    assert train_file.exists()
    content = train_file.read_text(encoding="utf-8")
    assert "StandardScaler" in content or "Dropout" in content
    assert "dev@test.com" not in content  # PII scrubbed!


def test_estimate_training_compute_and_cost():
    """Verify compute time and pricing calculations."""
    est = estimate_training_compute_and_cost(total_samples=100, avg_tokens_per_sample=200, epochs=3)
    assert est["total_samples"] == 100
    assert est["training_tokens"] == 60_000
    assert est["cost_openai_usd"] > 0.0
    assert est["cost_in_context_usd"] == 0.00
    assert est["approval_required"] is True


# =============================================================================
# 6. Model Evaluation Benchmark & Rollback Tests
# =============================================================================

def test_model_evaluation_and_rollback(temp_db: Path, monkeypatch):
    """Verify benchmark evaluation suite and version rollback."""
    from database import db as db_module
    monkeypatch.setattr(db_module, "DATABASE_PATH", temp_db)

    eval_result = evaluate_model_performance(model_id="model_v1_base", db_path=temp_db)
    assert eval_result["model_id"] == "model_v1_base"
    assert eval_result["overall_score"] >= 0.0
    assert eval_result["total_benchmarks"] >= 3

    # Test rollback
    success = rollback_model_version("model_v1_base", db_path=temp_db)
    assert success is True


# =============================================================================
# 7. End-to-End Self-Learning Verification
# =============================================================================

def test_assistant_applies_learned_correction_immediately(temp_db: Path, monkeypatch):
    """Verify that when a user correction is recorded, the assistant uses it immediately."""
    from database import db as db_module
    monkeypatch.setattr(db_module, "DATABASE_PATH", temp_db)

    # 1. Record an approved user correction
    record_correction(
        user_prompt="what is our lab's primary deep learning library?",
        corrected_response="Our lab exclusively uses PyTorch 2.5 with FlashAttention.",
        rejected_response="I don't have that information.",
        is_approved=True,
        db_path=temp_db,
    )

    # 2. Ask the assistant the exact question in rule-based mode
    assistant = NovaAssistant()
    res = assistant.process_message("what is our lab's primary deep learning library?", conversation_id="test_self_learn")

    # 3. Verify assistant directly provides the learned correction!
    assert "PyTorch 2.5 with FlashAttention" in res["response"]
    assert "verified user correction" in res["response"]


def test_get_learning_system_metrics_and_deletion(temp_db: Path):
    """Verify aggregated learning metrics across separated knowledge stores and deletion."""
    save_memory("Test memory for metrics", category="Research", db_path=temp_db)
    ex = record_correction(
        user_prompt="sample question for metric test",
        corrected_response="sample correct answer",
        is_approved=True,
        db_path=temp_db,
    )
    record_message_feedback("conv_m", rating="thumbs_up", is_approved_for_training=True, db_path=temp_db)

    metrics = get_learning_system_metrics(db_path=temp_db)
    assert metrics["approved_memories"] >= 1
    assert metrics["approved_corrections"] >= 1
    assert metrics["feedback_summary"]["thumbs_up"] >= 1
    assert metrics["current_model"]["id"] == "model_v1_base"

    # Test deletion
    deleted = delete_correction(ex["id"], db_path=temp_db)
    assert deleted is True
    metrics_after = get_learning_system_metrics(db_path=temp_db)
    assert metrics_after["approved_corrections"] == metrics["approved_corrections"] - 1


def test_inspect_fine_tuning_support_all_providers():
    """Verify provider inspection returns accurate capabilities and cost estimates."""
    gemini_info = inspect_fine_tuning_support(provider_name="gemini")
    assert gemini_info["supports_api_fine_tuning"] is False
    assert "Vertex AI" in gemini_info["supported_mechanism"]

    openai_info = inspect_fine_tuning_support(provider_name="openai")
    assert openai_info["supports_api_fine_tuning"] is True
    assert "$3.00" in openai_info["estimated_cost_per_1m_tokens"]

    local_info = inspect_fine_tuning_support(provider_name="local")
    assert local_info["supports_api_fine_tuning"] is False
    assert local_info["estimated_cost_per_1m_tokens"] == "$0.00"
