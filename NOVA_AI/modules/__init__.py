"""
Modules package for NOVA AI Assistant.
"""

from modules.student_assistant import generate_study_plan, record_study_session
from modules.document_assistant import (
    extract_document_content,
    search_document,
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

from modules.voice_assistant import (
    clean_text_for_speech,
    transcribe_audio,
    synthesize_speech,
    get_voice_capabilities,
)
from modules.planner_and_suggestions import (
    get_smart_task_suggestions,
    generate_daily_plan,
    save_daily_plan,
    get_daily_plan,
)

__all__ = [
    "generate_study_plan",
    "record_study_session",
    "extract_document_content",
    "search_document",
    "save_and_process_document",
    "get_documents",
    "get_document_by_id",
    "get_document_chunks",
    "delete_document",
    "delete_all_documents",
    "retrieve_relevant_chunks",
    "answer_document_question",
    "summarize_document",
    "clean_text_for_speech",
    "transcribe_audio",
    "synthesize_speech",
    "get_voice_capabilities",
    "get_smart_task_suggestions",
    "generate_daily_plan",
    "save_daily_plan",
    "get_daily_plan",
]
