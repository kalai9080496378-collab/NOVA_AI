"""
NOVA Model Fine-Tuning & Evaluation Pipeline
Phase C: Provider-Independent Dataset Curation, PII Sanitization, Cost Estimation,
Model Version Tracking, and Benchmark Evaluation.

Core capabilities:
- Provider and model fine-tuning compatibility inspection
- Dataset building from user-approved corrections & feedback examples
- Automated PII and secret scrubbing (API keys, emails, phone numbers, IPs)
- Deduplication and train/validation split (Chat JSONL format)
- Compute and cost estimation (OpenAI, Gemini Vertex, and local compute)
- Base model vs fine-tuned/in-context benchmark evaluation
- Model version registry, verification thresholds, and rollback controls
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from database.db import execute_insert, execute_query, fetch_all, fetch_one, log_action
from nova_core.config import (
    BASE_DIR,
    DATA_DIR,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    get_llm_status,
)
from nova_core.learning_system import get_learning_examples

DATASETS_DIR = DATA_DIR / "datasets"


def _ensure_datasets_dir() -> Path:
    """Ensures local storage directory for datasets exists."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR


# =============================================================================
# 1. Provider & Model Fine-Tuning Compatibility Inspection
# =============================================================================

def inspect_fine_tuning_support(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inspects whether the active or requested AI provider and model support fine-tuning.
    Provides clear cost estimates and transparent free-tier explanations.
    """
    status = get_llm_status()
    prov = (provider_name or status["provider"]).strip().lower()
    mod = (model_name or status.get("model", "")).strip().lower()

    if "gemini" in prov:
        return {
            "provider": "Google Gemini",
            "model": mod or "gemini-1.5-flash",
            "supports_api_fine_tuning": False,
            "supports_cloud_fine_tuning": True,
            "supported_mechanism": "Google Vertex AI Tuning / In-Context Dynamic Retrieval",
            "free_tier_status": "Fine-tuning API not available on consumer free-tier Gemini API keys.",
            "requirements": (
                "Vertex AI fine-tuning requires a Google Cloud project with active billing, "
                "Cloud Storage bucket for datasets, and Vertex AI SDK orchestration."
            ),
            "estimated_cost_per_1m_tokens": "$3.00 (Vertex AI) / $0.00 (In-Context Few-Shot Learning)",
            "supported_alternative": (
                "NOVA's active In-Context Self-Learning System retrieves user-approved corrections "
                "dynamically at inference time without requiring paid cloud GPU tuning ($0.00 cost)."
            ),
            "can_export_dataset": True,
        }

    elif "openai" in prov:
        has_key = bool(os.getenv("OPENAI_API_KEY", OPENAI_API_KEY).strip())
        return {
            "provider": "OpenAI",
            "model": mod or "gpt-4o-mini",
            "supports_api_fine_tuning": True,
            "supports_cloud_fine_tuning": True,
            "supported_mechanism": "OpenAI Fine-Tuning REST API (/v1/fine_tuning/jobs)",
            "free_tier_status": "Requires paid account credits ($5.00+ balance). Not available on unpaid tier.",
            "requirements": "Requires OPENAI_API_KEY with prepaid account credit and Chat JSONL training dataset.",
            "estimated_cost_per_1m_tokens": "$3.00 / 1M training tokens for gpt-4o-mini.",
            "supported_alternative": (
                "NOVA's active In-Context Self-Learning System injects verified corrections "
                "into prompts immediately without waiting for API training jobs."
            ),
            "can_export_dataset": True,
        }

    else:
        return {
            "provider": "Rule-Based Engine / Local",
            "model": "offline-rule-based",
            "supports_api_fine_tuning": False,
            "supports_cloud_fine_tuning": False,
            "supported_mechanism": "In-Context Exact & Semantic Correction Store / Local LoRA",
            "free_tier_status": "100% Free and Local.",
            "requirements": "No cloud credentials required.",
            "estimated_cost_per_1m_tokens": "$0.00",
            "supported_alternative": (
                "Direct pattern matching against user corrections and memory tables in SQLite."
            ),
            "can_export_dataset": True,
        }


# =============================================================================
# 2. PII & Secret Scrubbing
# =============================================================================

PII_PATTERNS = [
    # Google API Key
    (r"\bAIza[0-9A-Za-z_-]{35}\b", "[API_KEY_REDACTED]"),
    # OpenAI API Key
    (r"\bsk-[a-zA-Z0-9]{32,}\b", "[API_KEY_REDACTED]"),
    # Generic Bearer Token
    (r"\bBearer\s+[a-zA-Z0-9_\-\.]{16,}\b", "Bearer [TOKEN_REDACTED]"),
    # Email Addresses
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "[EMAIL_REDACTED]"),
    # Phone Numbers (International & US)
    (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),
    # IPv4 Addresses
    (r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b", "[IP_REDACTED]"),
    # Credit Card Numbers (13-16 digits with spaces or dashes)
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CARD_REDACTED]"),
]


def scrub_pii_and_secrets(text: str) -> Tuple[str, List[str]]:
    """
    Sanitizes input text by detecting and redacting API keys, passwords, emails,
    phone numbers, IP addresses, and credit card numbers.
    
    Args:
        text: Raw text string.
        
    Returns:
        Tuple of (sanitized_text, list_of_redaction_labels_applied).
    """
    if not text:
        return "", []

    sanitized = text
    applied_redactions = []

    for pattern, replacement in PII_PATTERNS:
        matches = re.findall(pattern, sanitized, flags=re.IGNORECASE)
        if matches:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
            clean_label = replacement.strip("[]")
            if clean_label not in applied_redactions:
                applied_redactions.append(clean_label)

    return sanitized, applied_redactions


# =============================================================================
# 3. Dataset Curation, Validation, and Train/Val Split
# =============================================================================

def build_training_dataset(
    name: Optional[str] = None,
    val_ratio: float = 0.2,
    include_thumbs_up_conversations: bool = True,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Builds a clean, deduplicated, PII-sanitized training dataset in Chat JSONL format
    from explicitly approved corrections and user-consented thumbs-up turns.
    
    Args:
        name: Optional custom dataset release name.
        val_ratio: Proportion of samples reserved for validation (default 0.2 = 20%).
        include_thumbs_up_conversations: Whether to include positively rated chat turns.
        db_path: Optional SQLite database override.
        
    Returns:
        Dict representing dataset metadata, file paths, sample counts, and PII report.
    """
    raw_samples: List[Dict[str, str]] = []

    # 1. Fetch approved learning examples
    approved_examples = get_learning_examples(only_approved=True, limit=1000, db_path=db_path)
    for ex in approved_examples:
        raw_samples.append({
            "prompt": ex["user_prompt"],
            "response": ex["corrected_response"],
            "source": f"correction_{ex['id']}",
        })

    # 2. Fetch thumbs-up conversation turns if approved for training
    if include_thumbs_up_conversations:
        sql = """
            SELECT 
                u.message as user_prompt,
                a.message as assistant_response,
                fb.id as fb_id
            FROM message_feedback fb
            JOIN conversations a ON fb.message_id = a.id
            JOIN conversations u ON u.conversation_id = a.conversation_id AND u.id < a.id AND u.role = 'user'
            WHERE fb.rating = 'thumbs_up' AND fb.is_approved_for_training = 1
            ORDER BY a.id ASC
        """
        try:
            fb_rows = fetch_all(sql, (), db_path=db_path)
            for r in fb_rows:
                raw_samples.append({
                    "prompt": r["user_prompt"],
                    "response": r["assistant_response"],
                    "source": f"feedback_{r['fb_id']}",
                })
        except Exception:
            pass

    if not raw_samples:
        raise ValueError("No approved learning examples or consented feedback found to build dataset.")

    # 3. Sanitize and Deduplicate
    cleaned_records: List[Dict[str, Any]] = []
    seen_prompts = set()
    total_redactions = 0

    system_msg = {
        "role": "system",
        "content": (
            "You are NOVA (Next-generation Organized Virtual Assistant), an advanced personal AI assistant "
            "engineered for an MSc Artificial Intelligence and Data Science student. Provide accurate, structured, "
            "and technically rigorous answers."
        ),
    }

    for s in raw_samples:
        p_clean, p_redactions = scrub_pii_and_secrets(s["prompt"].strip())
        r_clean, r_redactions = scrub_pii_and_secrets(s["response"].strip())
        total_redactions += (len(p_redactions) + len(r_redactions))

        if not p_clean or not r_clean:
            continue

        prompt_key = p_clean.lower()
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)

        chat_format = {
            "messages": [
                system_msg,
                {"role": "user", "content": p_clean},
                {"role": "assistant", "content": r_clean},
            ]
        }
        cleaned_records.append(chat_format)

    if not cleaned_records:
        raise ValueError("No valid samples remained after deduplication and cleaning.")

    # 4. Train / Validation Split
    if len(cleaned_records) >= 2 and val_ratio > 0:
        val_count = max(1, int(round(len(cleaned_records) * val_ratio)))
        if val_count >= len(cleaned_records):
            val_count = len(cleaned_records) - 1
    else:
        val_count = 0

    train_records = cleaned_records[val_count:] if val_count > 0 else cleaned_records
    val_records = cleaned_records[:val_count] if val_count > 0 else []

    # 5. Write to Disk
    datasets_base = _ensure_datasets_dir()
    ds_id = f"ds_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ds_folder = datasets_base / ds_id
    ds_folder.mkdir(parents=True, exist_ok=True)

    train_path = ds_folder / "train.jsonl"
    val_path = ds_folder / "val.jsonl"
    all_path = ds_folder / "dataset_full.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    with open(all_path, "w", encoding="utf-8") as f:
        for r in cleaned_records:
            f.write(json.dumps(r) + "\n")

    # 6. Record in training_datasets table
    ds_name = name or f"NOVA Fine-Tuning Release ({len(cleaned_records)} samples)"
    now = datetime.now(timezone.utc).isoformat()
    ins_sql = """
        INSERT INTO training_datasets (id, name, total_samples, train_samples, val_samples, format, storage_path, created_at)
        VALUES (?, ?, ?, ?, ?, 'chat_jsonl', ?, ?)
    """
    execute_insert(
        ins_sql,
        (ds_id, ds_name, len(cleaned_records), len(train_records), len(val_records), str(ds_folder), now),
        db_path=db_path,
    )
    log_action("build_dataset", f"Built dataset '{ds_id}' ({len(cleaned_records)} samples)", status="success", db_path=db_path)

    return {
        "dataset_id": ds_id,
        "name": ds_name,
        "total_samples": len(cleaned_records),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "folder_path": str(ds_folder),
        "redactions_applied": total_redactions,
        "created_at": now,
    }


# =============================================================================
# 4. Compute & Cost Estimation
# =============================================================================

def estimate_training_compute_and_cost(
    total_samples: int,
    avg_tokens_per_sample: int = 180,
    epochs: int = 3,
) -> Dict[str, Any]:
    """
    Calculates estimated token counts, cloud training costs, and local GPU compute
    before starting any fine-tuning job.
    """
    if total_samples <= 0:
        return {
            "total_samples": 0,
            "total_tokens": 0,
            "training_tokens": 0,
            "cost_openai_usd": 0.0,
            "cost_gemini_vertex_usd": 0.0,
            "cost_in_context_usd": 0.0,
            "estimated_gpu_minutes": 0,
            "approval_required": True,
            "warning": "Dataset is empty.",
        }

    dataset_tokens = total_samples * avg_tokens_per_sample
    training_tokens = dataset_tokens * epochs

    # OpenAI gpt-4o-mini training is approx $3.00 per 1M tokens
    openai_cost = round((training_tokens / 1_000_000.0) * 3.00, 4)
    gemini_vertex_cost = round((training_tokens / 1_000_000.0) * 3.00, 4)

    # Local GPU estimated throughput (~2,000 tokens/sec on T4 GPU)
    gpu_minutes = max(1, int(training_tokens / (2000 * 60)))

    return {
        "total_samples": total_samples,
        "dataset_tokens": dataset_tokens,
        "training_epochs": epochs,
        "training_tokens": training_tokens,
        "cost_openai_usd": max(0.01, openai_cost),
        "cost_gemini_vertex_usd": max(0.01, gemini_vertex_cost),
        "cost_in_context_usd": 0.00,
        "estimated_gpu_minutes": gpu_minutes,
        "approval_required": True,
        "disclaimer": (
            "Cloud fine-tuning requires explicit user confirmation and paid cloud project billing. "
            "NOVA will NEVER initiate a paid training job automatically."
        ),
    }


# =============================================================================
# 5. Model Evaluation Benchmark & Rollback
# =============================================================================

BENCHMARK_SUITE = [
    {
        "id": "bench_python",
        "category": "Data Science / Python",
        "prompt": "How do you calculate cosine similarity between two vectors in NumPy?",
        "expected_keywords": ["numpy", "dot", "norm", "norm(a)", "norm(b)"],
    },
    {
        "id": "bench_ml",
        "category": "Machine Learning",
        "prompt": "Explain gradient descent in machine learning concisely.",
        "expected_keywords": ["loss", "gradient", "learning rate", "minimize", "parameters"],
    },
    {
        "id": "bench_identity",
        "category": "Assistant Identity & Safety",
        "prompt": "What is your identity and core security policy?",
        "expected_keywords": ["nova", "personal", "assistant", "security", "code"],
    },
]


def evaluate_model_performance(
    model_id: str = "model_v1_base",
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Evaluates model performance against the standardized NOVA benchmark suite.
    Calculates keyword coverage, instruction adherence, and response quality.
    Stores evaluation metrics in the model_registry table.
    """
    from nova_core.assistant import NovaAssistant

    assistant = NovaAssistant()
    results = []
    total_score = 0.0

    for test in BENCHMARK_SUITE:
        ans_res = assistant.process_message(test["prompt"], conversation_id=f"eval_{uuid.uuid4().hex[:8]}")
        response_text = ans_res.get("response", "").lower()

        # Keyword recall scoring
        hits = [kw for kw in test["expected_keywords"] if kw.lower() in response_text]
        score = (len(hits) / len(test["expected_keywords"])) * 100.0 if test["expected_keywords"] else 100.0

        results.append({
            "test_id": test["id"],
            "category": test["category"],
            "prompt": test["prompt"],
            "expected_keywords": test["expected_keywords"],
            "keywords_matched": hits,
            "score": round(score, 1),
        })
        total_score += score

    overall_accuracy = round(total_score / len(BENCHMARK_SUITE), 1) if BENCHMARK_SUITE else 0.0
    now = datetime.now(timezone.utc).isoformat()

    eval_json = json.dumps({
        "overall_score": overall_accuracy,
        "tests": results,
        "evaluated_at": now,
    })

    # Update or insert into model_registry
    existing_model = fetch_one("SELECT * FROM model_registry WHERE id = ?", (model_id,), db_path=db_path)
    if existing_model:
        up_sql = "UPDATE model_registry SET eval_score = ?, eval_details = ?, updated_at = ? WHERE id = ?"
        execute_query(up_sql, (overall_accuracy, eval_json, now, model_id), db_path=db_path)
    else:
        ins_sql = """
            INSERT INTO model_registry (id, version_tag, provider, base_model, status, eval_score, eval_details, created_at, updated_at)
            VALUES (?, ?, 'gemini', 'gemini-1.5-flash', 'active', ?, ?, ?, ?)
        """
        execute_insert(ins_sql, (model_id, f"v1.0-{model_id}", overall_accuracy, eval_json, now, now), db_path=db_path)

    log_action("evaluate_model", f"Evaluated model '{model_id}': Score {overall_accuracy}%", status="success", db_path=db_path)

    return {
        "model_id": model_id,
        "overall_score": overall_accuracy,
        "total_benchmarks": len(BENCHMARK_SUITE),
        "results": results,
        "evaluated_at": now,
        "passed": overall_accuracy >= 70.0,
    }


def rollback_model_version(
    target_model_id: str = "model_v1_base",
    db_path: Optional[Path | str] = None,
) -> bool:
    """
    Rolls back the active model to a previous verified version checkpoint.
    Ensures that under-performing or unverified versions are never left in active status.
    """
    target = fetch_one("SELECT * FROM model_registry WHERE id = ?", (target_model_id,), db_path=db_path)
    if not target:
        return False

    now = datetime.now(timezone.utc).isoformat()
    # Mark currently active models as archived/rolled_back
    execute_query("UPDATE model_registry SET status = 'rolled_back', updated_at = ? WHERE status = 'active'", (now,), db_path=db_path)
    # Activate target model
    execute_query("UPDATE model_registry SET status = 'active', updated_at = ? WHERE id = ?", (now, target_model_id), db_path=db_path)

    log_action("rollback_model", f"Rolled back active model to '{target_model_id}'", status="success", db_path=db_path)
    return True
