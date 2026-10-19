"""
NOVA Daily Planner & Smart Suggestions
Phase 5 & 8: Smart Productivity Assistant & Production Daily Planner
Algorithmic daily plan generation, multi-factor priority scoring,
explainable task recommendations, weekly schedule views, and
resilient attribute/dictionary access.
"""

from datetime import datetime, timezone, date, time, timedelta
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

from database.db import execute_insert, execute_query, fetch_all, fetch_one, log_action
from tools.task_manager import get_tasks


class DailyPlan(dict):
    """
    A specialized dictionary that allows accessing daily plan fields either as
    dictionary keys (plan["schedule"]) or object attributes (plan.schedule).
    Guarantees zero AttributeErrors and zero KeyErrors regardless of caller syntax.
    """
    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            if key == "id":
                return super().get("id", None)
            elif key in ("schedule_blocks", "schedule"):
                return super().get("schedule", super().get("schedule_blocks", []))
            elif key == "allocated_tasks":
                return super().get("allocated_tasks", [])
            elif key in ("date", "plan_date"):
                return super().get("plan_date", super().get("date", ""))
            elif key in ("total_allocated_minutes", "total_estimated_minutes"):
                return super().get("total_allocated_minutes", super().get("total_estimated_minutes", 0))
            elif key in ("notes", "focus_theme"):
                return super().get("focus_theme", super().get("notes", ""))
            raise

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            # Provide sensible fallback aliases
            if name == "id":
                return self.get("id", None)
            elif name == "schedule_blocks":
                return self.get("schedule", [])
            elif name == "schedule":
                return self.get("schedule_blocks", [])
            elif name == "date":
                return self.get("plan_date", "")
            elif name == "plan_date":
                return self.get("date", "")
            elif name == "total_allocated_minutes":
                return self.get("total_estimated_minutes", 0)
            elif name == "total_estimated_minutes":
                return self.get("total_allocated_minutes", 0)
            elif name == "notes":
                return self.get("focus_theme", "")
            elif name == "focus_theme":
                return self.get("notes", "")
            raise AttributeError(f"'DailyPlan' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DailyPlan' object has no attribute '{name}'")


def _normalize_date_str(d: Optional[Any]) -> str:
    """Safely converts any date, datetime, or string object to an ISO 'YYYY-MM-DD' string."""
    if d is None:
        return date.today().isoformat()
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    clean = str(d).strip()
    if len(clean) >= 10:
        candidate = clean[:10]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except (ValueError, TypeError):
            pass
    return date.today().isoformat()


def _normalize_time_str(t: Optional[Any]) -> str:
    """Safely converts any time, datetime, or string object to a 24-hr 'HH:MM' string."""
    if t is None:
        return "09:00"
    if hasattr(t, "strftime"):
        return t.strftime("%H:%M")
    clean = str(t).strip()
    return clean if ":" in clean else "09:00"


def get_smart_task_suggestions(
    limit: int = 3,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Analyzes pending tasks and recommends which tasks to work on next.
    Generates explainable rationale based on deadlines, priorities, and duration.
    Always flags suggestions as AI-generated to distinguish from raw records.
    """
    if isinstance(limit, str) and not limit.isdigit() and user_id is None:
        user_id = limit
        limit = 3
    try:
        limit_int = max(1, int(limit)) if limit is not None else 3
    except (ValueError, TypeError):
        limit_int = 3

    pending_tasks = get_tasks(status="pending", db_path=db_path, user_id=user_id)
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
            clean_due = _normalize_date_str(due_d)
            if clean_due < today_str:
                score += 100
                reasons.append("🚨 Task is overdue and requires immediate intervention.")
            elif clean_due == today_str:
                score += 75
                if due_t and str(due_t) < now_time:
                    score += 30
                    reasons.append(f"⏰ Due today was at {due_t} (past due time).")
                elif due_t:
                    reasons.append(f"⏰ Due today at {due_t}.")
                else:
                    reasons.append("⏰ Due today.")
            else:
                try:
                    days_left = (datetime.strptime(clean_due, "%Y-%m-%d").date() - date.today()).days
                    if days_left == 1:
                        score += 50
                        reasons.append("📅 Deadline is tomorrow.")
                    elif days_left <= 3:
                        score += 30
                        reasons.append(f"📅 Deadline approaching in {days_left} days.")
                    elif days_left <= 7:
                        score += 15
                        reasons.append(f"📅 Due within 1 week ({clean_due}).")
                except Exception:
                    pass

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

        task_id = t.get("id") or t.get("task_id")
        scored_tasks.append({
            "id": task_id,
            "task_id": task_id,
            "task": t,
            "title": t.get("title", ""),
            "priority": prio,
            "category": cat,
            "due_date": due_d,
            "due_time": due_t,
            "estimated_duration": dur,
            "rationale": primary_reason,
            "recommendation_reason": primary_reason,
            "urgency_score": score,
            "score": score,
            "urgency_rank": 0,
            "ai_generated": True,
        })

    # Sort descending by score
    scored_tasks.sort(key=lambda x: x["score"], reverse=True)

    recommendations = []
    for rank, item in enumerate(scored_tasks[:limit_int], start=1):
        item["urgency_rank"] = rank
        recommendations.append(item)

    return recommendations


def generate_daily_plan(
    plan_date: Optional[Any] = None,
    available_hours: float = 6.0,
    focus_theme: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    target_date: Optional[Any] = None,
    start_time: Any = "09:00",
    user_id: Optional[str] = None,
) -> DailyPlan:
    """
    Generates a structured, balanced daily plan for the user based on real pending tasks.
    Allocates time blocks for Deep Work, Focused Execution, and Quick Reviews.
    Returns a DailyPlan object supporting both dictionary and attribute syntax without AttributeError.
    """
    # Detect if user_id was passed positionally as first argument
    if isinstance(plan_date, str) and user_id is None:
        try:
            datetime.strptime(plan_date[:10], "%Y-%m-%d")
        except (ValueError, IndexError):
            user_id = plan_date
            plan_date = None

    effective_date = _normalize_date_str(plan_date or target_date)
    start_time_str = _normalize_time_str(start_time)

    if available_hours <= 0:
        total_mins_available = 0
    else:
        total_mins_available = int(min(14.0, float(available_hours)) * 60)

    pending_tasks = get_tasks(status="pending", db_path=db_path, user_id=user_id)
    smart_picks = get_smart_task_suggestions(limit=len(pending_tasks), db_path=db_path, user_id=user_id)

    # Allocate tasks into blocks until available time is filled
    allocated_tasks = []
    used_minutes = 0

    for pick in smart_picks:
        t = pick.get("task") or pick
        t_id = pick.get("id") or pick.get("task_id") or t.get("id") or t.get("task_id")
        t_dur = t.get("estimated_duration") or 60
        if used_minutes + t_dur <= total_mins_available:
            allocated_tasks.append({
                "id": t_id,
                "task_id": t_id,
                "title": t.get("title", pick.get("title", "")),
                "priority": t.get("priority", "Medium"),
                "category": t.get("category", "General"),
                "due_date": t.get("due_date"),
                "due_time": t.get("due_time"),
                "duration_minutes": t_dur,
                "reason": pick.get("recommendation_reason", "Recommended task"),
            })
            used_minutes += t_dur

    # Create structured schedule blocks
    schedule_blocks = []
    try:
        current_time_slot = datetime.strptime(start_time_str, "%H:%M")
    except Exception:
        current_time_slot = datetime.strptime("09:00", "%H:%M")

    for idx, item in enumerate(allocated_tasks, start=1):
        start_str = current_time_slot.strftime("%H:%M")
        end_slot = current_time_slot + timedelta(minutes=item["duration_minutes"])
        end_str = end_slot.strftime("%H:%M")

        block_type = "Deep Work" if item["priority"] == "High" else ("Focused Execution" if item["priority"] == "Medium" else "Review & Quick Win")

        schedule_blocks.append({
            "id": item.get("id") or item.get("task_id") or idx,
            "block_id": idx,
            "start_time": start_str,
            "end_time": end_str,
            "time_range": f"{start_str} - {end_str}",
            "type": "task",
            "block_type": block_type,
            "task_id": item.get("task_id") or item.get("id"),
            "title": item["title"],
            "category": item["category"],
            "priority": item["priority"],
            "due_date": item.get("due_date"),
            "due_time": item.get("due_time"),
            "duration": item["duration_minutes"],
            "duration_minutes": item["duration_minutes"],
            "notes": item["reason"],
            "completed": False,
        })

        # Add 15-min buffer break after tasks to advance slot time
        current_time_slot = end_slot + timedelta(minutes=15)

    default_theme = focus_theme or (
        f"Goal-oriented execution across {len(allocated_tasks)} priority items"
        if allocated_tasks else "Planning & Goal Setting"
    )

    plan_data = {
        "id": None,
        "plan_date": effective_date,
        "date": effective_date,
        "focus_theme": str(default_theme),
        "notes": str(default_theme),
        "available_hours": available_hours,
        "total_estimated_minutes": used_minutes,
        "total_allocated_minutes": used_minutes,
        "total_tasks_scheduled": len(allocated_tasks),
        "allocated_tasks": allocated_tasks,
        "schedule": schedule_blocks,
        "schedule_blocks": schedule_blocks,
        "allocated_task_count": len(allocated_tasks),
        "remaining_minutes": max(0, total_mins_available - used_minutes),
        "ai_generated": True,
        "disclaimer": "AI suggested schedule. Review and adjust your blocks below. Your tasks are not modified.",
    }
    return DailyPlan(plan_data)


def save_daily_plan(
    plan_date: Any,
    focus_theme: Optional[str] = None,
    schedule_blocks: Optional[List[Dict[str, Any]]] = None,
    total_estimated_minutes: Optional[int] = None,
    schedule_data: Optional[List[Dict[str, Any]]] = None,
    total_allocated_minutes: Optional[int] = None,
    notes: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> int:
    """
    Saves an approved or edited daily plan persistently to SQLite.
    Overwrites if a plan already exists for the given plan_date and user_id.
    """
    date_str = _normalize_date_str(plan_date)
    uid = (user_id or "user_student_default").strip()
    now = datetime.now(timezone.utc).isoformat()

    theme_candidate = focus_theme or notes or "Daily Plan"
    theme = str(theme_candidate).strip()
    blocks = schedule_blocks if schedule_blocks is not None else (schedule_data or [])
    minutes = total_estimated_minutes if total_estimated_minutes is not None else (total_allocated_minutes or 0)
    blocks_json = json.dumps(blocks)

    if user_id:
        existing = fetch_one(
            "SELECT id FROM daily_plans WHERE plan_date = ? AND user_id = ?",
            (date_str, uid),
            db_path=db_path,
        )
    else:
        existing = fetch_one("SELECT id FROM daily_plans WHERE plan_date = ?", (date_str,), db_path=db_path)

    if existing:
        plan_id = existing["id"]
        sql = """
            UPDATE daily_plans 
            SET focus_theme = ?, items_json = ?, total_estimated_minutes = ?, updated_at = ?, user_id = ?
            WHERE id = ?
        """
        execute_query(sql, (theme, blocks_json, int(minutes), now, uid, plan_id), db_path=db_path)
        log_action("update_daily_plan", f"Updated daily plan for {date_str}", status="success", db_path=db_path)
        return plan_id
    else:
        sql = """
            INSERT INTO daily_plans (user_id, plan_date, focus_theme, items_json, total_estimated_minutes, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
        """
        plan_id = execute_insert(sql, (uid, date_str, theme, blocks_json, int(minutes), now, now), db_path=db_path)
        log_action("save_daily_plan", f"Saved daily plan for {date_str}", status="success", db_path=db_path)
        return plan_id


def get_daily_plan(
    plan_date: Optional[Any] = None,
    db_path: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> Optional[DailyPlan]:
    """Retrieves saved daily plan for a specific date from SQLite as a DailyPlan object."""
    target_date = _normalize_date_str(plan_date)
    if user_id:
        sql = "SELECT * FROM daily_plans WHERE plan_date = ? AND user_id = ?"
        row = fetch_one(sql, (target_date, user_id.strip()), db_path=db_path)
    else:
        sql = "SELECT * FROM daily_plans WHERE plan_date = ?"
        row = fetch_one(sql, (target_date,), db_path=db_path)

    if not row:
        return None

    try:
        blocks = json.loads(row["items_json"])
    except Exception:
        blocks = []

    mins = row["total_estimated_minutes"]
    theme = row["focus_theme"]
    return DailyPlan({
        "id": row["id"],
        "plan_date": row["plan_date"],
        "date": row["plan_date"],
        "focus_theme": theme,
        "notes": theme,
        "total_estimated_minutes": mins,
        "total_allocated_minutes": mins,
        "status": row["status"],
        "schedule_blocks": blocks,
        "schedule": blocks,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    })


def get_weekly_plan_summary(
    start_date: Optional[Any] = None,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Computes a 7-day weekly schedule summary starting from Monday of the selected week.
    Aggregates real tasks, deadlines, completed items, and total workload.
    """
    base_date = datetime.strptime(_normalize_date_str(start_date), "%Y-%m-%d").date()
    # Find Monday of this week
    monday = base_date - timedelta(days=base_date.weekday())
    sunday = monday + timedelta(days=6)

    all_tasks = get_tasks(db_path=db_path, user_id=user_id)
    days_data = []

    total_week_tasks = 0
    total_week_completed = 0
    total_week_pending = 0
    total_week_high_prio = 0
    total_week_minutes = 0

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for i in range(7):
        current_day = monday + timedelta(days=i)
        current_iso = current_day.isoformat()

        raw_day_tasks = [t for t in all_tasks if t.get("due_date") == current_iso]
        day_tasks = []
        for t in raw_day_tasks:
            t_copy = dict(t)
            t_id = t_copy.get("id") or t_copy.get("task_id")
            t_copy["id"] = t_id
            t_copy["task_id"] = t_id
            day_tasks.append(t_copy)
        pending_count = sum(1 for t in day_tasks if t.get("status") == "pending")
        completed_count = sum(1 for t in day_tasks if t.get("status") == "completed")
        day_minutes = sum(t.get("estimated_duration", 60) for t in day_tasks)
        high_prio_count = sum(1 for t in day_tasks if t.get("priority") == "High" and t.get("status") == "pending")
        all_high_in_day = sum(1 for t in day_tasks if t.get("priority") == "High")

        total_week_tasks += len(day_tasks)
        total_week_completed += completed_count
        total_week_pending += pending_count
        total_week_high_prio += all_high_in_day
        total_week_minutes += day_minutes

        # Check if a saved daily plan exists for this day
        saved_plan = get_daily_plan(current_iso, db_path=db_path, user_id=user_id)

        days_data.append({
            "date": current_iso,
            "day": day_names[i],
            "day_name": day_names[i],
            "is_today": (current_day == date.today()),
            "tasks": day_tasks,
            "total_tasks": len(day_tasks),
            "pending_count": pending_count,
            "completed_count": completed_count,
            "total_minutes": day_minutes,
            "high_prio_count": high_prio_count,
            "has_saved_plan": (saved_plan is not None),
            "saved_plan_theme": saved_plan.get("focus_theme") if saved_plan else None,
        })

    rate = (total_week_completed / total_week_tasks * 100) if total_week_tasks > 0 else 0.0

    return {
        "start_date": monday.isoformat(),
        "end_date": sunday.isoformat(),
        "week_start": monday.isoformat(),
        "week_end": sunday.isoformat(),
        "days": days_data,
        "total_tasks": total_week_tasks,
        "total_week_tasks": total_week_tasks,
        "completed_tasks": total_week_completed,
        "total_week_completed": total_week_completed,
        "pending_tasks": total_week_pending,
        "high_priority_tasks": total_week_high_prio,
        "total_week_minutes": total_week_minutes,
        "completion_rate": round(rate, 1),
    }
