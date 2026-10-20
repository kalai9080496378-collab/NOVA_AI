"""
NOVA Document Intelligence Module
Supports PDF, TXT, and CSV document ingestion, persistent SQLite storage,
intelligent chunking, TF-IDF RAG retrieval, grounded Q&A with page citations,
document summarization, follow-up conversation context, and strict user privacy.
"""

import io
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

from database.db import (
    execute_insert,
    execute_query,
    fetch_all,
    fetch_one,
    log_action,
)
from nova_core.config import (
    BASE_DIR,
    DATA_DIR,
    is_llm_configured,
)
from nova_core.llm_service import get_llm_service

# Configuration & Constants
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv"}
MAX_FILE_SIZE_MB = 15
DOCUMENTS_DIR = DATA_DIR / "documents"


def _ensure_docs_dir() -> Path:
    """Ensures local storage directory for uploaded documents exists."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return DOCUMENTS_DIR


# =============================================================================
# 1. Document Ingestion & Chunking
# =============================================================================

def save_and_process_document(
    file_bytes: bytes,
    filename: str,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingests an uploaded file, extracts text page-by-page or section-by-section,
    segments it into overlapping chunks, and persists metadata and chunks in SQLite.
    
    Args:
        file_bytes: Raw binary bytes of the file.
        filename: Name of the file.
        db_path: Optional SQLite database path override.
        
    Returns:
        Dict representing the created document record.
    """
    cleaned_filename = Path(filename).name.strip()
    ext = Path(cleaned_filename).suffix.lower()

    # 1. Validation
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Allowed formats: PDF (.pdf), Plain Text (.txt), CSV (.csv)."
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_MB} MB."
        )

    # 2. Save file to local secure storage
    docs_dir = _ensure_docs_dir()
    unique_name = f"{uuid.uuid4().hex[:8]}_{cleaned_filename}"
    local_path = docs_dir / unique_name
    with open(local_path, "wb") as f:
        f.write(file_bytes)

    # 3. Extraction per page / section
    pages_text: List[Tuple[int, str]] = []  # List of (page_num, text)

    if ext == ".pdf":
        if not PYPDF_AVAILABLE:
            raise RuntimeError("pypdf is required to extract PDF documents.")
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if len(reader.pages) == 0:
                raise ValueError("PDF file contains no readable pages.")
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                pages_text.append((idx + 1, txt.strip()))
        except Exception as e:
            if local_path.exists():
                local_path.unlink()
            raise ValueError(f"Could not parse PDF document: {str(e)}")

    elif ext == ".txt":
        try:
            full_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            full_text = file_bytes.decode("latin-1", errors="replace")

        # Split large text files into logical pages/sections (~3000 chars per section)
        sections = _split_txt_into_pages(full_text)
        for idx, sec in enumerate(sections):
            pages_text.append((idx + 1, sec.strip()))

    elif ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            summary_txt = (
                f"CSV Dataset Summary: {len(df)} rows, {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n\n"
                f"Sample Data:\n{df.head(20).to_string()}"
            )
            pages_text.append((1, summary_txt))
        except Exception as e:
            if local_path.exists():
                local_path.unlink()
            raise ValueError(f"Failed to parse CSV file: {str(e)}")

    # Calculate overall metrics
    total_pages = len(pages_text)
    all_text = " ".join(txt for _, txt in pages_text)
    words = all_text.split()
    total_words = len(words)
    file_size_kb = round(len(file_bytes) / 1024, 2)
    now = datetime.now(timezone.utc).isoformat()

    uid = (user_id or "user_student_default").strip()

    # 4. Insert Document into SQLite
    doc_sql = """
        INSERT INTO documents (user_id, filename, file_type, file_size_kb, word_count, page_count, storage_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    doc_id = execute_insert(
        doc_sql,
        (uid, cleaned_filename, ext.lstrip("."), file_size_kb, total_words, total_pages, str(local_path), now),
        db_path=db_path,
    )

    # 5. Chunking: generate overlapping chunks with page tracking
    chunks = _generate_chunks(pages_text, chunk_size=350, overlap=50)

    for chunk in chunks:
        chunk_sql = """
            INSERT INTO document_chunks (document_id, chunk_index, page_number, content, token_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        approx_tokens = int(len(chunk["content"].split()) * 1.33)
        execute_insert(
            chunk_sql,
            (doc_id, chunk["chunk_index"], chunk["page_number"], chunk["content"], approx_tokens, now),
            db_path=db_path,
        )

    log_action("save_document", f"Processed document #{doc_id}: '{cleaned_filename}' ({len(chunks)} chunks)", status="success", db_path=db_path)
    return get_document_by_id(doc_id, db_path=db_path)


def _split_txt_into_pages(text: str, chars_per_page: int = 2500) -> List[str]:
    """Splits plain text into readable logical pages by paragraphs."""
    paragraphs = text.split("\n\n")
    pages = []
    current_page = []
    current_len = 0

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if current_len + len(p_clean) > chars_per_page and current_page:
            pages.append("\n\n".join(current_page))
            current_page = [p_clean]
            current_len = len(p_clean)
        else:
            current_page.append(p_clean)
            current_len += len(p_clean)

    if current_page:
        pages.append("\n\n".join(current_page))

    return pages or [text]


def _generate_chunks(
    pages_text: List[Tuple[int, str]],
    chunk_size: int = 350,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Splits page texts into chunks of `chunk_size` words with `overlap` words,
    attributing each chunk to its corresponding page number.
    """
    chunks = []
    chunk_idx = 1

    for page_num, text in pages_text:
        words = text.split()
        if not words:
            continue

        if len(words) <= chunk_size:
            chunks.append({
                "chunk_index": chunk_idx,
                "page_number": page_num,
                "content": text.strip(),
            })
            chunk_idx += 1
        else:
            start = 0
            while start < len(words):
                end = min(len(words), start + chunk_size)
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)

                chunks.append({
                    "chunk_index": chunk_idx,
                    "page_number": page_num,
                    "content": chunk_text.strip(),
                })
                chunk_idx += 1

                if end == len(words):
                    break
                start += (chunk_size - overlap)

    return chunks


# =============================================================================
# 2. Document Management (Listing, Lookup, Deletion)
# =============================================================================

def get_documents(db_path: Optional[Path | str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all stored documents ordered by upload date descending, optionally filtered by user_id."""
    if user_id:
        sql = "SELECT * FROM documents WHERE user_id = ? ORDER BY id DESC"
        return fetch_all(sql, (user_id.strip(),), db_path=db_path)
    sql = "SELECT * FROM documents ORDER BY id DESC"
    return fetch_all(sql, (), db_path=db_path)


def get_document_by_id(document_id: int, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single document by its integer ID."""
    sql = "SELECT * FROM documents WHERE id = ?"
    return fetch_one(sql, (document_id,), db_path=db_path)


def get_document_chunks(document_id: int, db_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """Retrieves all chunks belonging to a document ordered by chunk_index."""
    sql = """
        SELECT id, chunk_index, page_number, content, token_count 
        FROM document_chunks 
        WHERE document_id = ? 
        ORDER BY chunk_index ASC
    """
    return fetch_all(sql, (document_id,), db_path=db_path)


def delete_document(document_id: int, db_path: Optional[Path | str] = None) -> bool:
    """
    Permanently deletes a document, its database records, and local file storage.
    """
    doc = get_document_by_id(document_id, db_path=db_path)
    if not doc:
        return False

    # Delete local file safely within DOCUMENTS_DIR
    try:
        f_path = Path(doc["storage_path"]).resolve()
        docs_dir_res = DOCUMENTS_DIR.resolve()
        if f_path.is_file() and str(f_path).startswith(str(docs_dir_res)):
            f_path.unlink()
    except Exception as e:
        print(f"[Warning] Failed to delete file on disk: {e}")

    # Delete chunks first (in case foreign key cascade isn't active on old connection)
    execute_query("DELETE FROM document_chunks WHERE document_id = ?", (document_id,), db_path=db_path)
    execute_query("DELETE FROM documents WHERE id = ?", (document_id,), db_path=db_path)

    log_action("delete_document", f"Deleted document #{document_id}: '{doc['filename']}'", status="success", db_path=db_path)
    return True


def delete_all_documents(db_path: Optional[Path | str] = None) -> int:
    """Permanently deletes all documents and chunk records."""
    docs = get_documents(db_path=db_path)
    count = 0
    for doc in docs:
        if delete_document(doc["id"], db_path=db_path):
            count += 1
    return count


# =============================================================================
# 3. RAG Retrieval Engine (TF-IDF Vector Space Model)
# =============================================================================

def retrieve_relevant_chunks(
    document_id: int,
    query: str,
    top_k: int = 4,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves the most semantically and lexically relevant chunks for a user query
    using TF-IDF n-gram vectorization and cosine similarity.
    
    Returns top_k chunks with their page number, content, and similarity score.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []

    chunks = get_document_chunks(document_id, db_path=db_path)
    if not chunks:
        return []

    chunk_texts = [c["content"] for c in chunks]

    # Vectorize chunks and query with unigrams & bigrams
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=1.0,
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(chunk_texts)
        query_vec = vectorizer.transform([cleaned_query])

        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    except Exception:
        # Fallback keyword match if TF-IDF fails, filtering out common stop words
        common_stop = {
            "what", "are", "the", "in", "an", "is", "a", "of", "and", "or", "for", "to",
            "at", "by", "from", "with", "about", "it", "this", "that", "how", "why",
            "who", "where", "when", "does", "do", "can", "will", "on"
        }
        content_words = [w for w in re.findall(r"\b\w+\b", cleaned_query.lower()) if w not in common_stop]
        if content_words:
            scores = [sum(1.0 for w in content_words if w in c["content"].lower()) / len(content_words) for c in chunks]
        else:
            scores = [0.0 for _ in chunks]

    # Rank chunks by score
    scored_chunks = []
    for idx, chunk in enumerate(chunks):
        score = float(scores[idx])
        scored_chunks.append({
            "chunk_index": chunk["chunk_index"],
            "page_number": chunk["page_number"],
            "content": chunk["content"],
            "score": round(score, 4),
        })

    # Sort descending by score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


# =============================================================================
# 4. Grounded Question Answering (RAG with Citations)
# =============================================================================

def answer_document_question(
    document_id: int,
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Generates a grounded answer to a user question using retrieved chunks.
    Maintains conversation context for follow-up questions.
    If the document does not contain the answer, explicitly reports it without hallucinating.
    """
    cleaned_q = (question or "").strip()
    if not cleaned_q:
        return {
            "answer": "Please enter a valid question.",
            "citations": [],
            "retrieved_chunks": [],
            "grounded": False,
            "mode": "validation",
        }

    doc = get_document_by_id(document_id, db_path=db_path)
    if not doc:
        return {
            "answer": f"Document #{document_id} was not found.",
            "citations": [],
            "retrieved_chunks": [],
            "grounded": False,
            "mode": "error",
        }

    # Retrieve top 4 relevant chunks
    top_chunks = retrieve_relevant_chunks(document_id, cleaned_q, top_k=4, db_path=db_path)

    # Check if retrieval found any meaningful relevance
    max_score = top_chunks[0]["score"] if top_chunks else 0.0

    # If completely unrelated (no term overlap or score <= 0.01)
    if max_score <= 0.01:
        return {
            "answer": (
                f"The uploaded document (**{doc['filename']}**) does not contain sufficient "
                f"information to answer: *'{cleaned_q}'*."
            ),
            "citations": [],
            "retrieved_chunks": top_chunks,
            "grounded": True,
            "mode": "not_found",
            "document_name": doc["filename"],
        }

    # Format context blocks with exact page numbers
    context_blocks = []
    unique_pages = []
    for c in top_chunks:
        if c["page_number"] not in unique_pages:
            unique_pages.append(c["page_number"])
        context_blocks.append(f"--- [Page {c['page_number']}] ---\n{c['content']}")

    context_str = "\n\n".join(context_blocks)

    # If LLM is configured, generate grounded conversational synthesis
    if is_llm_configured():
        llm = get_llm_service()
        if llm:
            system_prompt = (
                f"You are NOVA's Document Intelligence assistant. You are answering questions grounded "
                f"strictly in the uploaded document '{doc['filename']}'.\n\n"
                f"DOCUMENT CONTEXT:\n{context_str}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Answer using ONLY the provided document context above.\n"
                f"2. Cite the supporting page numbers (e.g. '[Page 2]') for every key point or claim.\n"
                f"3. If the provided context does not contain enough information to answer the question, state: "
                f"'The uploaded document does not contain sufficient information to answer this question.'\n"
                f"4. Do NOT speculate, hallucinate, or extrapolate facts not present in the excerpts.\n"
                f"5. Answer in a clear, structured, and helpful tone."
            )

            # Build messages including recent follow-up history
            messages = []
            if conversation_history:
                for turn in conversation_history[-4:]:  # last 4 turns for follow-up context
                    messages.append({
                        "role": turn.get("role", "user"),
                        "content": turn.get("content", ""),
                    })

            messages.append({"role": "user", "content": cleaned_q})

            llm_res = llm.generate(messages=messages, system_prompt=system_prompt)
            if llm_res.success:
                # Find page citations present in the answer
                cited_pages = [p for p in unique_pages if f"page {p}" in llm_res.content.lower() or f"[{p}]" in llm_res.content]
                if not cited_pages:
                    cited_pages = unique_pages[:2]

                return {
                    "answer": llm_res.content,
                    "citations": cited_pages,
                    "retrieved_chunks": top_chunks,
                    "grounded": True,
                    "mode": "llm",
                    "provider": llm_res.provider,
                    "model": llm_res.model,
                    "document_name": doc["filename"],
                }

    # Fallback: Offline Extractive RAG Mode (No API key needed)
    best_chunk = top_chunks[0]
    extractive_answer = (
        f"📄 **Extracted Excerpt from '{doc['filename']}' [Page {best_chunk['page_number']}]:**\n\n"
        f"> \"{best_chunk['content'][:450]}...\"\n\n"
        f"*Found {len(top_chunks)} relevant passage(s) across pages: "
        f"{', '.join(f'Page {p}' for p in unique_pages)}.*\n\n"
        f"*(To enable generative conversational answers and synthesis, configure a free GEMINI_API_KEY in `.env`)*"
    )

    return {
        "answer": extractive_answer,
        "citations": unique_pages,
        "retrieved_chunks": top_chunks,
        "grounded": True,
        "mode": "rule_based_extractive",
        "document_name": doc["filename"],
    }


# =============================================================================
# 5. Document Summarization
# =============================================================================

def summarize_document(
    document_id: int,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Generates an executive summary of the document.
    Uses LLM synthesis when configured, or structured extractive summarization offline.
    """
    doc = get_document_by_id(document_id, db_path=db_path)
    if not doc:
        return {"summary": "Document not found.", "status": "error"}

    chunks = get_document_chunks(document_id, db_path=db_path)
    if not chunks:
        return {"summary": "Document contains no text chunks to summarize.", "status": "error"}

    # Sample representative chunks (beginning, middle, end)
    sample_indices = [0]
    if len(chunks) > 2:
        sample_indices.append(len(chunks) // 2)
    if len(chunks) > 1:
        sample_indices.append(len(chunks) - 1)

    sampled_text = "\n\n".join(
        f"[Page {chunks[i]['page_number']} / Section {chunks[i]['chunk_index']}]:\n{chunks[i]['content']}"
        for i in sorted(list(set(sample_indices)))
    )

    if is_llm_configured():
        llm = get_llm_service()
        if llm:
            prompt = (
                f"You are summarizing the document '{doc['filename']}'.\n\n"
                f"Document Excerpts:\n{sampled_text}\n\n"
                f"Generate a clear, structured Executive Summary formatted in Markdown:\n"
                f"1. **Overview & Purpose**: Core topic and objective of the document.\n"
                f"2. **Key Themes / Findings**: 3-4 bullet points highlighting main sections or results.\n"
                f"3. **Takeaway Summary**: A concise wrap-up.\n"
                f"Include supporting page numbers where available."
            )
            res = llm.generate(messages=[{"role": "user", "content": prompt}])
            if res.success:
                return {
                    "summary": res.content,
                    "mode": "llm",
                    "provider": res.provider,
                    "model": res.model,
                    "document_name": doc["filename"],
                    "status": "success",
                }

    # Offline Extractive Summary
    summary_text = (
        f"### 📋 Document Summary: {doc['filename']}\n\n"
        f"- **Document Type:** {doc['file_type'].upper()}\n"
        f"- **File Size:** {doc['file_size_kb']} KB\n"
        f"- **Total Words:** {doc['word_count']:,} words\n"
        f"- **Page Count:** {doc['page_count']} page(s)\n"
        f"- **Chunk Segments:** {len(chunks)} retrieval chunks\n\n"
        f"**Opening Excerpt (Page 1):**\n"
        f"> \"{chunks[0]['content'][:300]}...\"\n\n"
        f"*(Note: Connect an LLM in `.env` to generate deep generative executive summaries)*"
    )

    return {
        "summary": summary_text,
        "mode": "extractive",
        "document_name": doc["filename"],
        "status": "success",
    }


# =============================================================================
# 6. Legacy / Simple Extraction Helper (Maintained for Backward Compatibility)
# =============================================================================

def extract_document_content(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Simple synchronous document content extractor maintained for backward compatibility."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'.")

    extracted_text = ""
    pages = 1

    if ext == ".txt":
        try:
            extracted_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = file_bytes.decode("latin-1", errors="replace")

    elif ext == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        extracted_text = f"CSV Rows: {len(df)}, Columns: {len(df.columns)}\nPreview:\n{df.head(10).to_string()}"

    elif ext == ".pdf":
        if not PYPDF_AVAILABLE:
            raise RuntimeError("pypdf is required to extract PDF files.")
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = len(reader.pages)
        pages_text = [p.extract_text() or "" for p in reader.pages]
        extracted_text = "\n\n".join(pages_text)

    words = extracted_text.split()
    return {
        "filename": filename,
        "extension": ext,
        "size_kb": round(len(file_bytes) / 1024, 2),
        "word_count": len(words),
        "char_count": len(extracted_text),
        "line_count": len(extracted_text.splitlines()),
        "extra_meta": {"pages": pages},
        "extracted_text": extracted_text,
    }


def search_document(extracted_text: str, query: str) -> List[Dict[str, Any]]:
    """Legacy keyword finder within raw text."""
    if not query or not query.strip():
        return []
    lines = extracted_text.splitlines()
    matches = []
    q_lower = query.lower().strip()
    for idx, line in enumerate(lines):
        if q_lower in line.lower():
            start = max(0, idx - 1)
            end = min(len(lines), idx + 2)
            matches.append({
                "line_number": idx + 1,
                "matched_line": line.strip(),
                "snippet": "\n".join(lines[start:end]),
            })
            if len(matches) >= 10:
                break
    return matches
