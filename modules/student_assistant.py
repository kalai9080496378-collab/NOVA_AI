"""
NOVA Student Assistant Module
Generates structured study plans, manages revision schedules, and tracks study hours.
Clearly distinguishes generated suggestions from verified database tasks.
"""

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import math

from tools.productivity import record_activity
from tools.task_manager import create_task


def generate_study_plan(
    subjects: List[str],
    available_hours_per_day: float,
    exam_date: Optional[str] = None,
    focus_topics: str = "",
) -> Dict[str, Any]:
    """
    Generates a structured, evidence-based study schedule based on available time.
    
    Returns structured recommendations that the user can review and convert to tasks.
    """
    cleaned_subjects = [s.strip() for s in subjects if s.strip()]
    if not cleaned_subjects:
        raise ValueError("Please provide at least one subject.")

    hours = max(0.5, float(available_hours_per_day))
    
    # Calculate days to exam if date given
    days_to_exam = None
    if exam_date:
        try:
            target_dt = datetime.strptime(exam_date.strip(), "%Y-%m-%d").date()
            today = date.today()
            diff = (target_dt - today).days
            days_to_exam = max(0, diff)
        except ValueError:
            pass

    # Time allocation per subject
    time_per_subject = round(hours / len(cleaned_subjects), 1)

    plan_sessions = []
    current_day = date.today()

    for idx, subject in enumerate(cleaned_subjects):
        # 1. Core Learning Block (60%)
        core_time = round(time_per_subject * 0.6 * 60) # minutes
        plan_sessions.append({
            "subject": subject,
            "session_type": "Deep Work / Core Concepts",
            "suggested_duration_mins": core_time,
            "description": f"Focus on understanding key concepts in {subject}. {f'Focus topic: {focus_topics}' if focus_topics else ''}",
            "recommended_day": (current_day + timedelta(days=idx % 3)).isoformat(),
        })

        # 2. Active Recall & Problem Solving (30%)
        practice_time = round(time_per_subject * 0.3 * 60)
        plan_sessions.append({
            "subject": subject,
            "session_type": "Active Practice & Problem Solving",
            "suggested_duration_mins": practice_time,
            "description": f"Solve problem sets, sample code, or past exam questions for {subject}.",
            "recommended_day": (current_day + timedelta(days=(idx % 3) + 1)).isoformat(),
        })

        # 3. Spaced Revision (10%)
        review_time = max(15, round(time_per_subject * 0.1 * 60))
        plan_sessions.append({
            "subject": subject,
            "session_type": "Spaced Revision",
            "suggested_duration_mins": review_time,
            "description": f"Quick flashcard or summary review of {subject} formulas and definitions.",
            "recommended_day": (current_day + timedelta(days=(idx % 3) + 2)).isoformat(),
        })

    return {
        "status": "success",
        "generated_at": datetime.now().isoformat(),
        "disclaimer": "⚠️ AI-Generated Study Recommendation. These are algorithmic suggestions and not yet stored as verified tasks.",
        "parameters": {
            "subjects": cleaned_subjects,
            "daily_hours": hours,
            "days_to_exam": days_to_exam,
            "exam_date": exam_date,
        },
        "sessions": plan_sessions,
    }


def record_study_session(
    subject: str,
    topic: str,
    duration_minutes: float,
    notes: str = "",
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> int:
    """
    Logs an actual completed study session into SQLite activity records.
    """
    metadata = f"Subject: {subject.strip()} | Topic: {topic.strip()} | Notes: {notes.strip()}"
    return record_activity(
        activity_type="study_session",
        duration=duration_minutes,
        metadata=metadata,
        user_id=user_id,
        db_path=db_path,
    )
