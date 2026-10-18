"""
NOVA Daily Planner & Smart Suggestions
Phase 5: Smart Productivity Assistant
Algorithmic daily plan generation, multi-factor priority scoring,
and explainable task recommendations grounded in SQLite records.
"""

from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

from database.db import execute_insert, execute_query, fetch_all, fetch_one, log_action
from tools.task_manager import get_tasks


def get_smart_task_suggestions(
    limit: int = 3,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Analyzes pending tasks and recommends which tasks to work on next.
    Generates explainable rationale based on deadlines, priorities, and duration.
    Always flags suggestions as AI-generated to distinguish from raw records.
    """
    pending_tasks = get_tasks(status="pending", db_path=db_path)
    if not pending_tasks:
        return []

    today_str = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M")

    scored_tasks = []
    for t in pending_tasks:
        score = 0
        reasons = []

        # 1. Deadline analysis
        due_d = t.get("due_date")
        due_t = t.get("due_time")

        if due_d:
            if due_d < today_str:
                score += 100
                reasons.append("🚨 Task is overdue and requires immediate intervention.")
            elif due_d == today_str:
                score += 75
                if due_t and due_t < now_time:
                    score += 30
                    reasons.append(f"⏰ Due today was at {due_t} (past due time).")
                elif due_t:
                    reasons.append(f"⏰ Due today at {due_t}.")
                else:
                    reasons.append("⏰ Due today.")
            else:
                days_left = (datetime.strptime(due_d, "%Y-%m-%d").date() - date.today()).days
                if days_left == 1:
                    score += 50
                    reasons.append("📅 Deadline is tomorrow.")
                elif days_left <= 3:
                    score += 30
                    reasons.append(f"📅 Deadline approaching in {days_left} days.")
                elif days_left <= 7:
                    score += 15
                    reasons.append(f"📅 Due within 1 week ({due_d}).")

        # 2. Priority weight
        prio = (t.get("priority") or "Medium").capitalize()
        if prio == "High":
            score += 40
            reasons.append("⚡ High Priority task.")
        elif prio == "Medium":
            score += 20
        else:
            score += 5

        # 3. Estimated Duration consideration
        dur = t.get("estimated_duration") or 60
        if dur <= 30:
            score += 10
            reasons.append(f"⚡ Quick-win task (~{dur} mins).")

        # 4. Synthesize primary explanation
        cat = t.get("category") or "General"
        primary_reason = " ".join(reasons) if reasons else f"Standard {prio} priority {cat} task."

        scored_tasks.append({
            "task": t,
            "score": score,
            "recommendation_reason": primary_reason,
            "urgency_rank": 0,
            "ai_generated": True,
        })

    # Sort descending by score
    scored_tasks.sort(key=lambda x: x["score"], reverse=True)

    recommendations = []
    for rank, item in enumerate(scored_tasks[:limit], start=1):
        item["urgency_rank"] = rank
        recommendations.append(item)

    return recommendations


def generate_daily_plan(
    plan_date: Optional[str] = None,
    available_hours: float = 6.0,
    focus_theme: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    target_date: Optional[str] = None,
    start_time: str = "09:00",
) -> Dict[str, Any]:
    """
    Generates a structured, balanced daily plan for the user based on real pending tasks.
    Allocates time blocks for Deep Work, Focused Execution, and Quick Reviews.
    
    IMPORTANT: Suggestions are advisory; underlying tasks are never deleted or modified.
    """
    effective_date = plan_date or target_date or date.today().isoformat()
    if available_hours <= 0:
        total_mins_available = 0
    else:
        total_mins_available = int(min(14.0, available_hours) * 60)

    pending_tasks = get_tasks(status="pending", db_path=db_path)
    smart_picks = get_smart_task_suggestions(limit=len(pending_tasks), db_path=db_path)

    # Allocate tasks into blocks until available time is filled
    allocated_tasks = []
    used_minutes = 0

    for pick in smart_picks:
        t = pick["task"]
        t_dur = t.get("estimated_duration") or 60
        if used_minutes + t_dur <= total_mins_available:
            allocated_tasks.append({
                "task_id": t["id"],
                "title": t["title"],
                "priority": t["priority"],
                "category": t.get("category", "General"),
                "duration_minutes": t_dur,
                "reason": pick["recommendation_reason"],
            })
            used_minutes += t_dur

    # Create structured schedule blocks
    schedule_blocks = []
    current_time_slot = datetime.strptime("09:00", "%H:%M")

    for idx, item in enumerate(allocated_tasks, start=1):
        start_str = current_time_slot.strftime("%H:%M")
        end_slot = current_time_slot + timedelta(minutes=item["duration_minutes"])
        end_str = end_slot.strftime("%H:%M")

        block_type = "Deep Work" if item["priority"] == "High" else ("Focused Execution" if item["priority"] == "Medium" else "Review & Quick Win")

        schedule_blocks.append({
            "block_id": idx,
            "time_range": f"{start_str} - {end_str}",
            "block_type": block_type,
            "task_id": item["task_id"],
            "title": item["title"],
            "category": item["category"],
            "duration": item["duration_minutes"],
            "notes": item["reason"],
            "completed": False,
        })

        # Add 15-min buffer break after every 90+ minutes
        current_time_slot = end_slot + timedelta(minutes=15)

    default_theme = focus_theme or (
        f"Goal-oriented execution across {len(allocated_tasks)} priority items"
        if allocated_tasks else "Planning & Goal Setting"
    )

    return {
        "plan_date": effective_date,
        "date": effective_date,
        "focus_theme": default_theme,
        "available_hours": available_hours,
        "total_estimated_minutes": used_minutes,
        "total_allocated_minutes": used_minutes,
        "total_tasks_scheduled": len(allocated_tasks),
        "schedule": schedule_blocks,
        "schedule_blocks": schedule_blocks,
        "allocated_task_count": len(allocated_tasks),
        "remaining_minutes": max(0, total_mins_available - used_minutes),
        "ai_generated": True,
        "disclaimer": "AI suggested schedule. Review and adjust your blocks below. Your tasks are not modified.",
    }


def save_daily_plan(
    plan_date: str,
    focus_theme: str,
    schedule_blocks: List[Dict[str, Any]],
    total_estimated_minutes: int,
    db_path: Optional[Path | str] = None,
) -> int:
    """
    Saves an approved or edited daily plan persistently to SQLite.
    Overwrites if a plan already exists for the given plan_date.
    """
    now = datetime.now(timezone.utc).isoformat()
    blocks_json = json.dumps(schedule_blocks)

    existing = fetch_one("SELECT id FROM daily_plans WHERE plan_date = ?", (plan_date,), db_path=db_path)
    if existing:
        plan_id = existing["id"]
        sql = """
            UPDATE daily_plans 
            SET focus_theme = ?, items_json = ?, total_estimated_minutes = ?, updated_at = ?
            WHERE id = ?
        """
        execute_query(sql, (focus_theme, blocks_json, total_estimated_minutes, now, plan_id), db_path=db_path)
        log_action("update_daily_plan", f"Updated daily plan for {plan_date}", status="success", db_path=db_path)
        return plan_id
    else:
        sql = """
            INSERT INTO daily_plans (plan_date, focus_theme, items_json, total_estimated_minutes, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'active', ?, ?)
        """
        plan_id = execute_insert(sql, (plan_date, focus_theme, blocks_json, total_estimated_minutes, now, now), db_path=db_path)
        log_action("save_daily_plan", f"Saved daily plan for {plan_date}", status="success", db_path=db_path)
        return plan_id


def get_daily_plan(
    plan_date: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieves saved daily plan for a specific date from SQLite."""
    target_date = plan_date or date.today().isoformat()
    sql = "SELECT * FROM daily_plans WHERE plan_date = ?"
    row = fetch_one(sql, (target_date,), db_path=db_path)
    if not row:
        return None

    try:
        blocks = json.loads(row["items_json"])
    except Exception:
        blocks = []

    return {
        "id": row["id"],
        "plan_date": row["plan_date"],
        "focus_theme": row["focus_theme"],
        "total_estimated_minutes": row["total_estimated_minutes"],
        "status": row["status"],
        "schedule_blocks": blocks,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
