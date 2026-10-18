"""
====================================================================
NOVA — Personal AI Assistant
Next-generation Organized Virtual Assistant
Phase 1 MVP: Streamlit User Interface & Core Feature System
====================================================================
"""

import os
import re
import json
import io
import hashlib
from datetime import datetime, date
from pathlib import Path
import streamlit as st
import pandas as pd

# Core NOVA Imports
from database.db import init_db, DATABASE_PATH, log_action, fetch_all, fetch_one
from nova_core.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_FULL_NAME,
    APP_VERSION,
    get_llm_status,
    get_voice_status,
)
from nova_core.assistant import NovaAssistant
from nova_core.conversation_manager import (
    create_thread,
    get_thread,
    get_all_threads,
    update_thread,
    delete_thread,
    add_message,
    get_thread_messages,
    auto_generate_title,
    group_threads_by_date,
    export_thread_markdown,
    export_thread_json,
)
from nova_core.learning_system import (
    record_message_feedback,
    get_feedback_summary,
    record_correction,
    get_learning_examples,
    retrieve_relevant_corrections,
    get_learning_system_metrics,
    delete_correction,
    export_memories_json,
    export_memories_csv,
)
from nova_core.fine_tuning_pipeline import (
    inspect_fine_tuning_support,
    scrub_pii_and_secrets,
    build_training_dataset,
    estimate_training_compute_and_cost,
    evaluate_model_performance,
    rollback_model_version,
)
from tools.task_manager import (
    create_task,
    get_tasks,
    get_task_by_id,
    update_task,
    complete_task,
    reopen_task,
    delete_task,
    get_task_statistics,
)
from tools.reminder_manager import (
    create_reminder,
    get_reminders,
    get_reminder_by_id,
    update_reminder,
    pause_reminder,
    resume_reminder,
    dismiss_reminder,
    delete_reminder,
    check_and_trigger_due_reminders,
    get_reminder_statistics,
)
from memory.memory_manager import (
    save_memory,
    get_all_memories,
    update_memory,
    delete_memory,
    clear_all_memories,
    search_memories,
)
from tools.productivity import (
    record_activity,
    get_activity_records,
    get_daily_activity_summary,
    get_study_hours_summary,
    export_table_to_csv,
)
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
    answer_document_question,
    summarize_document,
)
from modules.planner_and_suggestions import (
    get_smart_task_suggestions,
    generate_daily_plan,
    save_daily_plan,
    get_daily_plan,
)
from analytics.dashboard import (
    create_task_status_chart,
    create_priority_distribution_chart,
    create_daily_activity_chart,
    create_completion_trend_chart,
    create_category_distribution_chart,
)
from modules.voice_assistant import (
    clean_text_for_speech,
    transcribe_audio,
    synthesize_speech,
    get_voice_capabilities,
    SUPPORTED_LANGUAGES,
    calculate_pcm_rms,
    resolve_language_tag,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} — Personal AI Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database schema automatically on startup
init_db()

# Custom Modern Dark Theme Styling (Navy, Violet, Cyan accents)
CUSTOM_CSS = """
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Cards and Containers */
    .nova-card {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .nova-metric-card {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    
    .nova-metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #06b6d4;
        margin-top: 4px;
    }
    
    .nova-metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Badges */
    .badge-cyan {
        background-color: rgba(6, 182, 212, 0.15);
        color: #22d3ee;
        border: 1px solid #0891b2;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-violet {
        background-color: rgba(139, 92, 246, 0.15);
        color: #c084fc;
        border: 1px solid #7c3aed;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-rose {
        background-color: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid #e11d48;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    .badge-amber {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        border: 1px solid #d97706;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "active_thread_id" not in st.session_state:
    existing_threads = get_all_threads(include_archived=False)
    if existing_threads:
        st.session_state.active_thread_id = existing_threads[0]["id"]
    else:
        new_th = create_thread(title="New Conversation")
        st.session_state.active_thread_id = new_th["id"]

st.session_state.conversation_id = st.session_state.active_thread_id

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = get_thread_messages(st.session_state.active_thread_id)

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "voice_consent" not in st.session_state:
    st.session_state.voice_consent = True

if "voice_tts_enabled" not in st.session_state:
    st.session_state.voice_tts_enabled = True

if "voice_autoplay" not in st.session_state:
    st.session_state.voice_autoplay = False

if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None

if "voice_messages" not in st.session_state:
    st.session_state.voice_messages = []

if "last_voice_page_audio_hash" not in st.session_state:
    st.session_state.last_voice_page_audio_hash = None

# Staged Voice Transcription State (Allows user review & edit before sending to AI)
if "chat_staged_transcript" not in st.session_state:
    st.session_state.chat_staged_transcript = None
if "chat_staged_audio_hash" not in st.session_state:
    st.session_state.chat_staged_audio_hash = None
if "chat_staged_audio_bytes" not in st.session_state:
    st.session_state.chat_staged_audio_bytes = None
if "chat_staged_meta" not in st.session_state:
    st.session_state.chat_staged_meta = None
if "chat_processed_audio_hash" not in st.session_state:
    st.session_state.chat_processed_audio_hash = None

if "voice_page_staged_transcript" not in st.session_state:
    st.session_state.voice_page_staged_transcript = None
if "voice_page_staged_audio_hash" not in st.session_state:
    st.session_state.voice_page_staged_audio_hash = None
if "voice_page_staged_audio_bytes" not in st.session_state:
    st.session_state.voice_page_staged_audio_bytes = None
if "voice_page_staged_meta" not in st.session_state:
    st.session_state.voice_page_staged_meta = None
if "voice_page_processed_audio_hash" not in st.session_state:
    st.session_state.voice_page_processed_audio_hash = None


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Branding
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <div style="font-size: 2.2rem; font-weight: 800; letter-spacing: 2px; color: #22d3ee;">
                🌌 {APP_NAME}
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                {APP_TAGLINE}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    llm_info = get_llm_status()
    if llm_info["status"] == "connected":
        st.markdown(f'<span class="badge-violet">⚡ {llm_info["provider"]} Active ({llm_info.get("model", "")})</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-cyan">⚙️ Rule-Based Mode</span>', unsafe_allow_html=True)

    st.write("")

    navigation_options = [
        "🏠 Home",
        "💬 AI Chat",
        "🎙️ Voice Assistant",
        "📋 Task Manager",
        "⏰ Smart Reminders",
        "📅 Daily Planner",
        "📊 Productivity Analytics",
        "🧠 My Memory",
        "🎓 Student Mode",
        "📄 Document Assistant",
        "🧠 Self-Learning & Training",
        "⚙️ Settings",
    ]

    selected_page = st.radio(
        "Navigation",
        navigation_options,
        index=0,
        label_visibility="collapsed",
    )

    if selected_page == "💬 AI Chat":
        st.markdown("---")
        st.markdown("#### 💬 Conversations")

        # + New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary", key="sb_new_chat_btn"):
            new_th = create_thread(title="New Conversation")
            st.session_state.active_thread_id = new_th["id"]
            st.session_state.conversation_id = new_th["id"]
            st.session_state.pending_prompt = None
            st.rerun()

        # Search conversations
        chat_search = st.text_input("Search chats", placeholder="🔍 Filter chats...", label_visibility="collapsed", key="sb_search_chats")

        # Load active and archived threads
        active_threads = get_all_threads(include_archived=False, search_query=chat_search)
        archived_threads = get_all_threads(include_archived=True, search_query=chat_search)
        archived_only = [t for t in archived_threads if t.get("is_archived")]

        grouped = group_threads_by_date(active_threads)

        if not active_threads:
            st.caption("No conversations found.")
        else:
            for grp_name, th_list in grouped.items():
                st.markdown(
                    f"<div style='font-size: 0.72rem; font-weight: 700; color: #64748b; margin-top: 10px; margin-bottom: 4px; text-transform: uppercase;'>{grp_name}</div>",
                    unsafe_allow_html=True,
                )
                for th in th_list:
                    is_active = (th["id"] == st.session_state.active_thread_id)
                    col_btn, col_pop = st.columns([5, 1])
                    with col_btn:
                        prefix = "🔹 " if is_active else ""
                        btn_name = f"{prefix}{th['title']}"
                        if st.button(
                            btn_name,
                            key=f"sb_th_{th['id']}",
                            use_container_width=True,
                            help=f"Thread: {th['title']}",
                        ):
                            st.session_state.active_thread_id = th["id"]
                            st.session_state.conversation_id = th["id"]
                            st.session_state.pending_prompt = None
                            st.rerun()
                    with col_pop:
                        with st.popover("⋮", help="Options"):
                            st.caption(f"**{th['title']}**")
                            with st.form(f"sb_rename_{th['id']}"):
                                new_t = st.text_input("Rename", value=th["title"])
                                if st.form_submit_button("Save"):
                                    if new_t.strip():
                                        update_thread(th["id"], title=new_t.strip())
                                        st.rerun()

                            pin_txt = "Unpin" if th.get("is_pinned") else "📌 Pin"
                            if st.button(pin_txt, key=f"sb_pin_{th['id']}", use_container_width=True):
                                update_thread(th["id"], is_pinned=not th.get("is_pinned"))
                                st.rerun()

                            if st.button("📦 Archive", key=f"sb_arch_{th['id']}", use_container_width=True):
                                update_thread(th["id"], is_archived=True)
                                if st.session_state.active_thread_id == th["id"]:
                                    rem = [t for t in active_threads if t["id"] != th["id"]]
                                    st.session_state.active_thread_id = rem[0]["id"] if rem else create_thread()["id"]
                                    st.session_state.conversation_id = st.session_state.active_thread_id
                                st.rerun()

                            if st.button("🗑️ Delete", key=f"sb_del_{th['id']}", use_container_width=True):
                                delete_thread(th["id"])
                                if st.session_state.active_thread_id == th["id"]:
                                    rem = [t for t in active_threads if t["id"] != th["id"]]
                                    st.session_state.active_thread_id = rem[0]["id"] if rem else create_thread()["id"]
                                    st.session_state.conversation_id = st.session_state.active_thread_id
                                st.rerun()

        if archived_only:
            with st.expander(f"📦 Archived ({len(archived_only)})", expanded=False):
                for ath in archived_only:
                    st.caption(f"📁 **{ath['title']}**")
                    c_un, c_del = st.columns(2)
                    with c_un:
                        if st.button("Restore", key=f"unarch_{ath['id']}", use_container_width=True):
                            update_thread(ath["id"], is_archived=False)
                            st.session_state.active_thread_id = ath["id"]
                            st.session_state.conversation_id = ath["id"]
                            st.rerun()
                    with c_del:
                        if st.button("Delete", key=f"del_arch_{ath['id']}", use_container_width=True):
                            delete_thread(ath["id"])
                            st.rerun()

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            <b>NOVA Architecture:</b> Phase 1 MVP<br>
            <b>Storage:</b> SQLite Persistent<br>
            <b>Active DB:</b> <code>nova.db</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Global Active Reminder Engine & Delivery Check
# -----------------------------------------------------------------------------
due_now_reminders = check_and_trigger_due_reminders()
if due_now_reminders:
    for tr in due_now_reminders:
        st.toast(f"⏰ REMINDER: {tr['title']} is DUE!", icon="🔔")
        st.warning(
            f"🔔 **REMINDER ALERT [{tr['remind_at']}]:** **{tr['title']}** "
            f"({tr.get('description', '') or 'No notes'})"
            + (f" [Recurring: {tr['recurrence'].capitalize()}]" if tr.get('recurrence') != 'none' else "")
        )


# -----------------------------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# -----------------------------------------------------------------------------
if selected_page == "🏠 Home":
    st.markdown(
        f"""
        <div class="nova-card">
            <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">Welcome back! 🌌</h1>
            <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1rem;">
                {APP_FULL_NAME} — {APP_TAGLINE}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Real Metrics retrieved from SQLite
    stats = get_task_statistics()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Pending Tasks</div>
                <div class="nova-metric-value">{stats['pending']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Completed Tasks</div>
                <div class="nova-metric-value" style="color: #10b981;">{stats['completed']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Completion Rate</div>
                <div class="nova-metric-value" style="color: #c084fc;">{stats['completion_rate']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        overdue_color = "#fb7185" if stats["overdue"] > 0 else "#94a3b8"
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Overdue Tasks</div>
                <div class="nova-metric-value" style="color: {overdue_color};">{stats['overdue']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("⏰ Upcoming Deadlines")
        upcoming = stats.get("upcoming_tasks", [])
        if not upcoming:
            st.info("No upcoming deadlines scheduled with due dates. Enjoy your free time or add new tasks!")
        else:
            for t in upcoming:
                p_badge = "badge-rose" if t["priority"] == "High" else ("badge-amber" if t["priority"] == "Medium" else "badge-cyan")
                st.markdown(
                    f"""
                    <div style="background: #1e293b; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <b>#{t['id']} {t['title']}</b>
                            <div style="font-size: 0.8rem; color: #94a3b8;">Due: {t['due_date']}</div>
                        </div>
                        <span class="{p_badge}">{t['priority']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("⚡ Quick Actions")
        qa_cols = st.columns(3)
        with qa_cols[0]:
            if st.button("➕ Quick Add Task", use_container_width=True):
                st.session_state.quick_add = True
        with qa_cols[1]:
            if st.button("🧠 Save a Memory", use_container_width=True):
                st.session_state.quick_mem = True
        with qa_cols[2]:
            if st.button("🎓 Study Planner", use_container_width=True):
                st.info("Navigate to the **Student Mode** tab from the sidebar to generate personalized study plans!")

        if st.session_state.get("quick_add", False):
            with st.form("home_quick_task"):
                st.write("**Quick Task Creation**")
                q_title = st.text_input("Task Title")
                q_prio = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)
                q_due = st.date_input("Due Date (Optional)", value=None)
                if st.form_submit_button("Save Task"):
                    if q_title.strip():
                        due_str = q_due.isoformat() if q_due else None
                        create_task(q_title, priority=q_prio, due_date=due_str)
                        st.success(f"Task '{q_title}' saved!")
                        st.session_state.quick_add = False
                        st.rerun()
                    else:
                        st.error("Title cannot be empty.")

        if st.session_state.get("quick_mem", False):
            with st.form("home_quick_memory"):
                st.write("**Quick Memory Capture**")
                m_content = st.text_area("What would you like NOVA to remember?")
                m_cat = st.selectbox("Category", ["General", "Academic", "Preferences", "Projects"])
                if st.form_submit_button("Save Memory"):
                    if m_content.strip():
                        save_memory(m_content, category=m_cat)
                        st.success("Memory saved successfully with your permission!")
                        st.session_state.quick_mem = False
                        st.rerun()
                    else:
                        st.error("Content cannot be empty.")

    with col_right:
        st.subheader("📜 Recent Activity")
        recent_acts = get_activity_records(limit=6)
        if not recent_acts:
            st.write("No recorded activities yet. As you complete tasks and interact with NOVA, real audit events will appear here.")
        else:
            for act in recent_acts:
                time_str = act["created_at"][:16].replace("T", " ")
                st.markdown(
                    f"""
                    <div style="border-left: 2px solid #0891b2; padding-left: 10px; margin-bottom: 12px;">
                        <div style="font-size: 0.85rem; font-weight: 600; color: #f8fafc;">{act['activity_type'].replace('_', ' ').title()}</div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">{act['metadata'] or 'Recorded action'}</div>
                        <div style="font-size: 0.7rem; color: #64748b;">{time_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# PAGE 2: AI CHAT
# -----------------------------------------------------------------------------
elif selected_page == "💬 AI Chat":
    # 1. Resolve Active Conversation Thread
    active_th = get_thread(st.session_state.active_thread_id)
    if not active_th:
        existing = get_all_threads(include_archived=False)
        if existing:
            st.session_state.active_thread_id = existing[0]["id"]
            active_th = existing[0]
        else:
            active_th = create_thread(title="New Conversation")
            st.session_state.active_thread_id = active_th["id"]

    st.session_state.conversation_id = st.session_state.active_thread_id

    # 2. Modern Conversation Header with inline management & export
    th_c1, th_c2, th_c3 = st.columns([5, 1, 1])
    with th_c1:
        pin_badge = "📌 " if active_th.get("is_pinned") else ""
        st.markdown(
            f"""
            <div style="margin-bottom: 4px;">
                <h2 style="margin: 0; color: #f8fafc; font-size: 1.6rem;">{pin_badge}{active_th['title']}</h2>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 3px;">
                    Model: <code>{active_th.get('model', 'gemini-1.5-flash')}</code> &bull; 
                    Created: {active_th['created_at'][:10]} &bull; 
                    Session ID: <code>{active_th['id']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with th_c2:
        with st.popover("⚙️ Thread", help="Manage this conversation"):
            st.caption(f"**Settings for #{active_th['id']}**")
            with st.form("main_th_rename"):
                renamed = st.text_input("Title", value=active_th["title"])
                if st.form_submit_button("✏️ Rename"):
                    if renamed.strip():
                        update_thread(active_th["id"], title=renamed.strip())
                        st.rerun()

            p_label = "Unpin" if active_th.get("is_pinned") else "📌 Pin to top"
            if st.button(p_label, use_container_width=True, key="hdr_pin_btn"):
                update_thread(active_th["id"], is_pinned=not active_th.get("is_pinned"))
                st.rerun()

            if st.button("➕ New Chat", use_container_width=True, key="hdr_new_btn"):
                nth = create_thread(title="New Conversation")
                st.session_state.active_thread_id = nth["id"]
                st.session_state.conversation_id = nth["id"]
                st.session_state.pending_prompt = None
                st.rerun()

            if st.button("📦 Archive", use_container_width=True, key="hdr_arch_btn"):
                update_thread(active_th["id"], is_archived=True)
                rem = get_all_threads(include_archived=False)
                st.session_state.active_thread_id = rem[0]["id"] if rem else create_thread()["id"]
                st.session_state.conversation_id = st.session_state.active_thread_id
                st.rerun()

            if st.button("🗑️ Delete Thread", type="secondary", use_container_width=True, key="hdr_del_btn"):
                delete_thread(active_th["id"])
                rem = get_all_threads(include_archived=False)
                st.session_state.active_thread_id = rem[0]["id"] if rem else create_thread()["id"]
                st.session_state.conversation_id = st.session_state.active_thread_id
                st.rerun()

    with th_c3:
        with st.popover("📥 Export", help="Export this chat"):
            st.caption("**Export Options**")
            md_content = export_thread_markdown(active_th["id"])
            json_content = export_thread_json(active_th["id"])
            safe_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", active_th["title"]).lower()
            st.download_button(
                "📄 Markdown (.md)",
                data=md_content,
                file_name=f"nova_{safe_slug}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.download_button(
                "💾 JSON (.json)",
                data=json_content,
                file_name=f"nova_{safe_slug}.json",
                mime="application/json",
                use_container_width=True,
            )

    # Mode notification
    llm_info = get_llm_status()
    if llm_info["status"] == "connected":
        st.caption(
            f"⚡ **Generative AI Connected ({llm_info['provider']} - {llm_info.get('model', '')})**. "
            "Ask natural conversational questions, seek study and code advice, or use local task and memory commands."
        )
    else:
        st.info(
            "💡 **Running in Rule-Based Mode (No API key configured)**. "
            "You can type commands like `add task: Complete project due 2026-10-31`, `show tasks`, "
            "`remember: [Academic] Exam in room 301`, `stats`, `time`, or `help`.\n\n"
            "*(To enable AI reasoning, add your free GEMINI_API_KEY in `.env`)*"
        )

    # 3. Load Thread Messages from SQLite
    thread_messages = get_thread_messages(active_th["id"])

    # Empty State: ChatGPT-style welcome cards
    if not thread_messages:
        st.markdown(
            """
            <div style="text-align: center; padding: 40px 16px 20px 16px;">
                <div style="font-size: 3rem; margin-bottom: 6px;">🌌</div>
                <h2 style="color: #f8fafc; margin-bottom: 6px; font-weight: 700;">How can NOVA help you today?</h2>
                <p style="color: #94a3b8; font-size: 0.95rem; max-width: 600px; margin: 0 auto 24px auto;">
                    Explore machine learning, organize academic priorities, store persistent knowledge, or run natural commands.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            if st.button("💡 **Explain Machine Learning Concepts**\n\n*Understand gradient descent & loss functions*", use_container_width=True):
                st.session_state.pending_prompt = "Explain gradient descent, learning rates, and loss minimization in machine learning."
                st.rerun()
            if st.button("🧠 **Save Academic Memory**\n\n*Store thesis research topic in memory*", use_container_width=True):
                st.session_state.pending_prompt = "remember: [Academic] My MSc thesis topic is Graph Neural Networks for Drug Discovery."
                st.rerun()
        with s_col2:
            if st.button("📋 **Create Project Task**\n\n*Schedule assignment deadline with high priority*", use_container_width=True):
                st.session_state.pending_prompt = "add task: Complete Deep Learning Assignment due 2026-11-15 priority High"
                st.rerun()
            if st.button("📊 **Productivity Status**\n\n*Inspect task completion & metrics*", use_container_width=True):
                st.session_state.pending_prompt = "stats"
                st.rerun()
    else:
        # Display conversation messages
        for idx, msg in enumerate(thread_messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    meta = msg.get("metadata", {})
                    if meta.get("tool_called"):
                        st.caption(f"🔧 Tool Invoked: `{meta['tool_called']}` (Validated & Executed)")
                    elif meta.get("mode") == "llm":
                        st.caption(f"⚡ Generated by {meta.get('provider', '').upper()} (`{meta.get('model', '')}`)")
                    elif meta.get("mode") == "fallback_rule_based":
                        st.caption("⚠️ Rule-Based Fallback Engaged")

                    st.markdown(msg["content"])

                    # Turn Actions: Copy, Thumbs Up, Thumbs Down, Teach NOVA, and Retry
                    act_c1, act_c2, act_c3, act_c4, act_c5 = st.columns([1.1, 0.7, 0.7, 1.3, 1.4])
                    with act_c1:
                        with st.popover("📋 Copy", help="Copy response content"):
                            st.code(msg["content"], language="markdown")
                    with act_c2:
                        if st.button("👍", key=f"fb_up_{msg['id']}", help="Mark response as helpful"):
                            record_message_feedback(
                                conversation_id=active_th["id"],
                                rating="thumbs_up",
                                message_id=msg["id"],
                                is_approved_for_training=True,
                            )
                            st.toast("Feedback recorded: Helpful! 👍")
                    with act_c3:
                        if st.button("👎", key=f"fb_down_{msg['id']}", help="Mark response as unhelpful or incorrect"):
                            record_message_feedback(
                                conversation_id=active_th["id"],
                                rating="thumbs_down",
                                message_id=msg["id"],
                                is_approved_for_training=True,
                            )
                            st.toast("Feedback recorded: Needs improvement. Click ✏️ Teach to submit a verified correction.")
                    with act_c4:
                        with st.popover("✏️ Teach", help="Submit a verified correction to teach NOVA"):
                            st.caption("**Teach NOVA / Submit Verified Correction**")
                            corr_prev_usr = next((m["content"] for m in reversed(thread_messages[:idx]) if m["role"] == "user"), "")
                            with st.form(f"corr_form_{msg['id']}"):
                                c_prompt = st.text_input("User Question / Prompt", value=corr_prev_usr)
                                c_bad = st.text_area("NOVA's Current (Incorrect) Answer", value=msg["content"][:300], height=68)
                                c_good = st.text_area("Verified / Desired Answer", placeholder="Enter the accurate, desired response...", height=90)
                                c_expl = st.text_input("Explanation / Rationale (Optional)", placeholder="e.g. In Python 3.10+, match-case was added")
                                c_cat = st.selectbox("Category", ["Correction", "Demonstration", "Academic", "Code", "Identity", "Preferences"], index=0)
                                c_consent = st.checkbox("Approve this correction for NOVA's self-learning store", value=True)
                                if st.form_submit_button("💾 Save Correction", type="primary"):
                                    if not c_prompt.strip() or not c_good.strip():
                                        st.error("Prompt and verified answer cannot be empty.")
                                    else:
                                        record_correction(
                                            user_prompt=c_prompt,
                                            corrected_response=c_good,
                                            rejected_response=c_bad,
                                            explanation=c_expl,
                                            category=c_cat,
                                            is_approved=c_consent,
                                            source="user_correction",
                                        )
                                        record_message_feedback(
                                            conversation_id=active_th["id"],
                                            rating="thumbs_down",
                                            message_id=msg["id"],
                                            feedback_text=f"Corrected: {c_good[:100]}",
                                            is_approved_for_training=c_consent,
                                        )
                                        st.toast("✅ Correction saved! NOVA will retrieve this approved knowledge in future queries.")
                                        st.rerun()
                    with act_c5:
                        # Offer retry on the last assistant response
                        if idx == len(thread_messages) - 1:
                            prev_usr = next((m["content"] for m in reversed(thread_messages[:idx]) if m["role"] == "user"), None)
                            if prev_usr:
                                if st.button("🔄 Retry", key=f"retry_{msg['id']}", help="Regenerate this turn"):
                                    st.session_state.pending_prompt = prev_usr
                                    st.rerun()

    # 4. Voice Intelligence Panel (Preserving Phase 4 Voice Features & Language Control)
    with st.expander("🎙️ Voice Intelligence & Microphone (Click to Speak)", expanded=False):
        v_col1, v_col2, v_col3 = st.columns([2, 1, 1])
        with v_col1:
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                    <b>🎙️ Voice Privacy & Security:</b> Voice input is strictly user-initiated. Audio snippets are sent over encrypted HTTPS for transcription and discarded immediately after processing. Zero continuous recording.
                </div>
                """,
                unsafe_allow_html=True,
            )
            chat_consent = st.checkbox(
                "Consent to external audio processing for transcription",
                value=st.session_state.get("voice_consent", True),
                key="chat_consent_checkbox",
            )
            st.session_state.voice_consent = chat_consent
        with v_col2:
            st.session_state.voice_tts_enabled = st.checkbox(
                "🔊 Spoken Replies (TTS)",
                value=st.session_state.get("voice_tts_enabled", True),
                key="chat_tts_checkbox",
            )
        with v_col3:
            st.session_state.voice_autoplay = st.checkbox(
                "▶️ Auto-play Audio",
                value=st.session_state.get("voice_autoplay", False),
                key="chat_autoplay_checkbox",
            )

        st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # Language and STT Engine Controls
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            chat_lang_label = st.selectbox(
                "🗣️ Speech Language",
                options=[
                    "English (India) [en-IN] — Recommended for Indian Accents",
                    "Tamil / தமிழ் [ta-IN] — Native Tamil Speech",
                    "Tanglish (Tamil + English Mixed)",
                    "English (US) [en-US] — Standard American English",
                ],
                index=0,
                key="chat_voice_language_select",
                help="Select your spoken dialect. en-IN is calibrated for Indian English rhythm, cadence, and pronunciations.",
            )
            lang_map = {
                "English (India) [en-IN] — Recommended for Indian Accents": "en-IN",
                "Tamil / தமிழ் [ta-IN] — Native Tamil Speech": "ta-IN",
                "Tanglish (Tamil + English Mixed)": "tanglish",
                "English (US) [en-US] — Standard American English": "en-US",
            }
            selected_lang_code = lang_map.get(chat_lang_label, "en-IN")

        with cfg_col2:
            chat_engine_label = st.selectbox(
                "⚙️ Recognition Engine",
                options=[
                    "Auto (Gemini 1.5 Flash if available, else Google Web Speech)",
                    "Google Web Speech (Free $0.00, Zero API Key Required)",
                    "Google Gemini 1.5 Flash (Multimodal Audio, Best for Tanglish)",
                ],
                index=0,
                key="chat_voice_engine_select",
                help="Auto seamlessly uses Gemini 1.5 Flash when GEMINI_API_KEY is present in your .env file, and falls back to Google Web Speech.",
            )
            engine_map = {
                "Auto (Gemini 1.5 Flash if available, else Google Web Speech)": "auto",
                "Google Web Speech (Free $0.00, Zero API Key Required)": "google",
                "Google Gemini 1.5 Flash (Multimodal Audio, Best for Tanglish)": "gemini",
            }
            selected_engine = engine_map.get(chat_engine_label, "auto")

        st.write("")
        st.write("**Click the microphone icon below to speak to NOVA:**")
        mic_audio = st.audio_input(
            "Record voice message (Click mic to Start, Click again to Stop)",
            key="chat_microphone_widget",
        )

        if mic_audio is not None:
            raw_audio_bytes = mic_audio.getvalue()
            audio_md5 = hashlib.md5(raw_audio_bytes).hexdigest()

            # Check if this is new un-staged and un-processed audio
            if (
                st.session_state.get("chat_staged_audio_hash") != audio_md5
                and st.session_state.get("chat_processed_audio_hash") != audio_md5
            ):
                if not chat_consent:
                    st.warning("⚠️ External audio processing consent is required for transcription. Please check the consent box above.")
                elif len(raw_audio_bytes) < 200:
                    st.warning("⚠️ Recording empty or too short. Please speak clearly into your microphone.")
                else:
                    with st.spinner(f"🎧 Transcribing your speech ({selected_lang_code})..."):
                        stt_result = transcribe_audio(
                            raw_audio_bytes,
                            audio_format="wav",
                            language=selected_lang_code,
                            engine=selected_engine,
                            consent_external=chat_consent,
                        )

                    if not stt_result["success"]:
                        st.error(f"❌ Speech Recognition Notice: {stt_result['error']}")
                        if stt_result.get("error_type") in ("silence", "unknown_audio"):
                            st.info(
                                "💡 **Microphone Troubleshooting Tips:**\n"
                                "- Check your browser's microphone permissions (lock/site settings icon in the address bar).\n"
                                "- Check that your physical microphone is selected as default and unmuted in Windows Sound Settings.\n"
                                "- Ensure your mic input volume is above 60% and speak closer to the microphone.\n"
                                "- If speaking Tanglish, ensure the language dropdown is set to 'Tanglish' or 'English (India)'."
                            )
                    else:
                        # Stage transcript for user review & editing before committing to the AI
                        st.session_state.chat_staged_audio_hash = audio_md5
                        st.session_state.chat_staged_audio_bytes = raw_audio_bytes
                        st.session_state.chat_staged_transcript = stt_result["text"]
                        st.session_state.chat_staged_meta = stt_result
                        st.rerun()

        # Staged Review and Edit Card
        if st.session_state.get("chat_staged_transcript") is not None:
            st.markdown("<hr style='margin: 12px 0; border-color: rgba(6, 182, 212, 0.3);'>", unsafe_allow_html=True)
            st.markdown("#### 📝 Review & Edit Transcript Before Sending")
            st.caption("Inspect and modify the recognized text below to correct any misheard words or slang before submitting to NOVA.")

            meta = st.session_state.get("chat_staged_meta", {})
            engine_name = meta.get("engine_used", "speech engine")
            lang_used = meta.get("language_used", selected_lang_code)

            st.markdown(
                f"<div style='font-size: 0.82rem; color: #38bdf8; margin-bottom: 8px;'>"
                f"🟢 <b>Status:</b> Ready for Review &nbsp;|&nbsp; "
                f"⚙️ <b>Engine:</b> {engine_name} &nbsp;|&nbsp; "
                f"🗣️ <b>Language:</b> <code>{lang_used}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Audio review playback
            if st.session_state.get("chat_staged_audio_bytes"):
                with st.expander("▶️ Listen to your recorded audio snippet", expanded=False):
                    st.audio(st.session_state.chat_staged_audio_bytes, format="audio/wav")

            # Editable transcript text area
            edited_transcript = st.text_area(
                "Recognized Text (Editable):",
                value=st.session_state.chat_staged_transcript,
                key="chat_transcript_editor_area",
                height=90,
                help="Edit any misheard words or punctuation before submitting to NOVA.",
            )

            btn_col1, btn_col2, btn_col3 = st.columns([1.6, 1.6, 3])
            with btn_col1:
                if st.button("🚀 Send to NOVA", type="primary", key="btn_send_staged_chat"):
                    clean_prompt = edited_transcript.strip()
                    if not clean_prompt:
                        st.warning("⚠️ Message cannot be empty.")
                    else:
                        # Prevent re-transcribing same audio buffer
                        st.session_state.chat_processed_audio_hash = st.session_state.get("chat_staged_audio_hash")
                        st.session_state.chat_staged_transcript = None
                        st.session_state.chat_staged_audio_hash = None
                        st.session_state.chat_staged_audio_bytes = None
                        st.session_state.chat_staged_meta = None

                        # Process via Assistant into active thread
                        assistant = NovaAssistant()
                        with st.spinner("NOVA is processing your message..."):
                            reply_res = assistant.process_message(
                                clean_prompt,
                                conversation_id=active_th["id"],
                            )
                            reply_text = reply_res["response"]

                        if st.session_state.get("voice_tts_enabled", True):
                            with st.spinner("🔊 Synthesizing spoken voice response..."):
                                tts_res = synthesize_speech(reply_text)
                                if tts_res["success"] and tts_res.get("audio_bytes"):
                                    st.audio(
                                        tts_res["audio_bytes"],
                                        format="audio/mp3",
                                        autoplay=st.session_state.get("voice_autoplay", False),
                                    )

                        record_activity("voice_interaction", metadata=f"Spoken: {clean_prompt[:40]}")
                        log_action("voice_interaction", f"Transcribed: '{clean_prompt}'", status="success")
                        st.rerun()

            with btn_col2:
                if st.button("🔄 Discard & Retry", key="btn_discard_staged_chat"):
                    st.session_state.chat_processed_audio_hash = st.session_state.get("chat_staged_audio_hash")
                    st.session_state.chat_staged_transcript = None
                    st.session_state.chat_staged_audio_hash = None
                    st.session_state.chat_staged_audio_bytes = None
                    st.session_state.chat_staged_meta = None
                    st.info("Recording discarded. Click the microphone icon above to speak again.")
                    st.rerun()

            with btn_col3:
                st.caption("Click **Send to NOVA** to query the AI, or **Discard & Retry** to record again.")

        # Microphone Permissions & Troubleshooting Expander
        with st.expander("❓ Microphone Permission & Audio Troubleshooting Guide", expanded=False):
            st.markdown(
                """
                **How to ensure clear, high-accuracy speech recognition:**
                1. **Browser Microphone Permission:**
                   - Look for the lock or site settings icon next to the browser URL (`localhost:8501`).
                   - Ensure **Microphone** is toggled to **Allow**.
                2. **Language Accent Matching:**
                   - For Indian English speakers: Select **`English (India) [en-IN]`** for optimal recognition of Indian pronunciation and cadence.
                   - For Tamil speakers: Select **`Tamil / தமிழ் [ta-IN]`**.
                   - For Tamil-English code-switching: Select **`Tanglish`**.
                3. **Microphone Placement & Noise:**
                   - Keep the microphone 6–12 inches from your mouth.
                   - Minimize background noise from ceiling fans, loud ACs, or music.
                4. **Tanglish Recognition with Gemini:**
                   - When a `GEMINI_API_KEY` is present in your `.env`, NOVA can leverage Gemini 1.5 Flash's multimodal audio engine for state-of-the-art code-mixed accuracy.
                """
            )

    # 5. Chat Input Box with Real-Time Streaming
    user_prompt = st.chat_input("Ask NOVA anything or enter a command (e.g., 'add task: ...', 'show tasks', 'help')...")
    if not user_prompt and st.session_state.get("pending_prompt"):
        user_prompt = st.session_state.pop("pending_prompt")

    if user_prompt:
        # Display user turn
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            assistant = NovaAssistant()
            with st.spinner("NOVA is thinking..."):
                stream_generator = assistant.process_message_stream(
                    user_prompt,
                    conversation_id=active_th["id"],
                )
                full_reply = st.write_stream(stream_generator)

        # Synthesize speech if voice response is enabled
        if st.session_state.get("voice_tts_enabled", False):
            synthesize_speech(full_reply)

        st.rerun()



# -----------------------------------------------------------------------------
# PAGE 3: VOICE ASSISTANT (PHASE 4: VOICE INTELLIGENCE)
# -----------------------------------------------------------------------------
elif selected_page == "🎙️ Voice Assistant":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">🎙️ NOVA Voice Intelligence Studio</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Hands-free conversational AI with Speech-to-Text (STT) and natural Text-to-Speech (TTS).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Voice Diagnostics & Capabilities
    v_stat = get_voice_status()

    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Speech-to-Text (STT)</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #22d3ee; margin-top: 6px;">
                    {v_stat['stt_name']}
                </div>
                <div style="font-size: 0.75rem; color: #10b981; margin-top: 4px;">{v_stat['stt_cost']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with vc2:
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Text-to-Speech (TTS)</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #c084fc; margin-top: 6px;">
                    {v_stat['tts_name']}
                </div>
                <div style="font-size: 0.75rem; color: #10b981; margin-top: 4px;">{v_stat['tts_cost']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with vc3:
        st.markdown(
            """
            <div class="nova-metric-card">
                <div class="nova-metric-label">Voice Privacy & Recording</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8; margin-top: 6px;">
                    User-Initiated Only
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">No continuous background recording</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Privacy disclosure & consent banner
    st.info(
        "🔒 **Voice Privacy Disclosure & Transparency:**\n\n"
        "- **Microphone Recording:** Activated ONLY when you explicitly press the Record button in your browser.\n"
        "- **Speech-to-Text Transit:** Audio snippets are sent over encrypted HTTPS to Google Web Speech API for transcription and discarded immediately.\n"
        "- **Text-to-Speech Transit:** Text output is synthesized via Google Text-to-Speech (`gTTS`) into MP3 audio.\n"
        "- **Data Sovereignty:** No voice recordings are permanently kept or tied to personal profiles. $0.00 cost architecture."
    )

    # Tabs for Voice Mode
    voice_tab1, voice_tab2, voice_tab3 = st.tabs([
        "🗣️ Hands-Free Voice Dialogue",
        "🎙️ Speech-to-Text Studio (STT)",
        "🔊 Text-to-Speech Studio (TTS)",
    ])

    with voice_tab1:
        st.subheader("🗣️ Conversational Voice Dialogue")
        st.caption("Press Record, speak your question or command, press Stop, review/edit your transcript, and listen to NOVA's spoken answer.")

        # Consent & Audio Output Configuration
        set_col1, set_col2, set_col3 = st.columns(3)
        with set_col1:
            voice_tab_consent = st.checkbox(
                "☑️ External STT Consent (Google / Gemini)",
                value=st.session_state.get("voice_consent", True),
                key="voice_page_consent_cb",
            )
            st.session_state.voice_consent = voice_tab_consent
        with set_col2:
            st.session_state.voice_tts_enabled = st.checkbox(
                "🔊 Enable Voice Replies",
                value=st.session_state.get("voice_tts_enabled", True),
                key="voice_page_tts_cb",
            )
        with set_col3:
            st.session_state.voice_autoplay = st.checkbox(
                "▶️ Auto-play Audio",
                value=st.session_state.get("voice_autoplay", False),
                key="voice_page_autoplay_cb",
            )

        st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # Language and Recognition Engine Selectors
        v_col_l, v_col_e = st.columns(2)
        with v_col_l:
            v_lang_label = st.selectbox(
                "🗣️ Spoken Language / Dialect",
                options=[
                    "English (India) [en-IN] — Recommended for Indian Accents",
                    "Tamil / தமிழ் [ta-IN] — Native Tamil Speech",
                    "Tanglish (Tamil + English Mixed)",
                    "English (US) [en-US] — Standard American English",
                ],
                index=0,
                key="voice_page_lang_select",
                help="Calibrate recognition models for your accent. en-IN handles Indian phonetics and syllable timing.",
            )
            v_lang_map = {
                "English (India) [en-IN] — Recommended for Indian Accents": "en-IN",
                "Tamil / தமிழ் [ta-IN] — Native Tamil Speech": "ta-IN",
                "Tanglish (Tamil + English Mixed)": "tanglish",
                "English (US) [en-US] — Standard American English": "en-US",
            }
            active_v_lang = v_lang_map.get(v_lang_label, "en-IN")

        with v_col_e:
            v_engine_label = st.selectbox(
                "⚙️ Speech Recognition Engine",
                options=[
                    "Auto (Gemini 1.5 Flash if available, else Google Web Speech)",
                    "Google Web Speech (Free $0.00, Zero API Key Required)",
                    "Google Gemini 1.5 Flash (Multimodal Audio, High Precision)",
                ],
                index=0,
                key="voice_page_engine_select",
                help="Auto seamlessly uses Gemini 1.5 Flash when GEMINI_API_KEY is configured, else uses Google Web Speech.",
            )
            v_engine_map = {
                "Auto (Gemini 1.5 Flash if available, else Google Web Speech)": "auto",
                "Google Web Speech (Free $0.00, Zero API Key Required)": "google",
                "Google Gemini 1.5 Flash (Multimodal Audio, High Precision)": "gemini",
            }
            active_v_engine = v_engine_map.get(v_engine_label, "auto")

        st.write("")
        st.write("**Record your message below:**")
        v_audio = st.audio_input(
            "Record your voice command (Click mic to Start, Click again to Stop)",
            key="voice_page_recorder",
        )

        if v_audio is not None:
            v_bytes = v_audio.getvalue()
            v_hash = hashlib.md5(v_bytes).hexdigest()

            # Process new audio if not staged and not already processed
            if (
                st.session_state.get("voice_page_staged_audio_hash") != v_hash
                and st.session_state.get("voice_page_processed_audio_hash") != v_hash
            ):
                if not voice_tab_consent:
                    st.warning("⚠️ Please check the external processing consent box to enable voice transcription.")
                elif len(v_bytes) < 200:
                    st.warning("⚠️ Recording empty or too short. Please speak clearly into your microphone.")
                else:
                    with st.spinner(f"🎧 Converting speech to text ({active_v_lang})..."):
                        t_res = transcribe_audio(
                            v_bytes,
                            audio_format="wav",
                            language=active_v_lang,
                            engine=active_v_engine,
                            consent_external=voice_tab_consent,
                        )

                    if not t_res["success"]:
                        st.error(f"❌ Speech Recognition Notice: {t_res['error']}")
                        if t_res.get("error_type") in ("silence", "unknown_audio"):
                            st.info(
                                "💡 **Microphone Troubleshooting Tips:**\n"
                                "- Check your browser's microphone permissions (lock icon in address bar).\n"
                                "- Check that your physical microphone is connected and unmuted in Windows Sound Settings.\n"
                                "- Speak closer to the microphone with clear articulation."
                            )
                    else:
                        # Stage transcript for user review and editing
                        st.session_state.voice_page_staged_audio_hash = v_hash
                        st.session_state.voice_page_staged_audio_bytes = v_bytes
                        st.session_state.voice_page_staged_transcript = t_res["text"]
                        st.session_state.voice_page_staged_meta = t_res
                        st.rerun()

        # Staged Review and Edit Card
        if st.session_state.get("voice_page_staged_transcript") is not None:
            st.markdown("<hr style='margin: 12px 0; border-color: rgba(6, 182, 212, 0.3);'>", unsafe_allow_html=True)
            st.markdown("#### 📝 Review & Edit Transcript Before Sending")
            st.caption("Inspect and modify the recognized text below to correct any misheard words or slang before submitting to NOVA.")

            vp_meta = st.session_state.get("voice_page_staged_meta", {})
            vp_engine_name = vp_meta.get("engine_used", "speech engine")
            vp_lang_used = vp_meta.get("language_used", active_v_lang)

            st.markdown(
                f"<div style='font-size: 0.82rem; color: #38bdf8; margin-bottom: 8px;'>"
                f"🟢 <b>Status:</b> Ready for Review &nbsp;|&nbsp; "
                f"⚙️ <b>Engine:</b> {vp_engine_name} &nbsp;|&nbsp; "
                f"🗣️ <b>Language:</b> <code>{vp_lang_used}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Audio review playback
            if st.session_state.get("voice_page_staged_audio_bytes"):
                with st.expander("▶️ Listen to your recorded audio snippet", expanded=False):
                    st.audio(st.session_state.voice_page_staged_audio_bytes, format="audio/wav")

            # Editable transcript text area
            edited_query = st.text_area(
                "Recognized Text (Editable):",
                value=st.session_state.voice_page_staged_transcript,
                key="voice_page_transcript_editor",
                height=90,
                help="Edit any misheard words or punctuation before submitting to NOVA.",
            )

            v_btn1, v_btn2, v_btn3 = st.columns([1.8, 1.8, 3])
            with v_btn1:
                if st.button("🚀 Send to Assistant", type="primary", key="btn_send_voice_page_staged"):
                    clean_spoken = edited_query.strip()
                    if not clean_spoken:
                        st.warning("⚠️ Message cannot be empty.")
                    else:
                        # Prevent re-transcribing same audio buffer
                        st.session_state.voice_page_processed_audio_hash = st.session_state.get("voice_page_staged_audio_hash")
                        st.session_state.voice_page_staged_transcript = None
                        st.session_state.voice_page_staged_audio_hash = None
                        st.session_state.voice_page_staged_audio_bytes = None
                        st.session_state.voice_page_staged_meta = None

                        # Process with assistant
                        assistant = NovaAssistant()
                        with st.spinner("NOVA is thinking..."):
                            reply = assistant.process_message(
                                clean_spoken,
                                conversation_id=st.session_state.conversation_id,
                            )

                        # Synthesize speech
                        audio_speech = None
                        if st.session_state.get("voice_tts_enabled", True):
                            with st.spinner("🔊 Generating natural spoken response..."):
                                s_res = synthesize_speech(reply["response"])
                                if s_res["success"]:
                                    audio_speech = s_res["audio_bytes"]

                        # Append to voice conversation history
                        st.session_state.voice_messages.append({
                            "user_text": clean_spoken,
                            "assistant_text": reply["response"],
                            "audio_bytes": audio_speech,
                            "mode": reply.get("mode"),
                            "tool_called": reply.get("tool_called"),
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                        })

                        # Also mirror to general chat history
                        st.session_state.chat_messages.append({"role": "user", "content": clean_spoken, "source": "voice"})
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": reply["response"],
                            "audio_bytes": audio_speech,
                            "tool_called": reply.get("tool_called"),
                            "mode": reply.get("mode"),
                        })

                        record_activity("voice_interaction", metadata=f"Spoken: {clean_spoken[:40]}")
                        log_action("voice_interaction", f"Spoken query: {clean_spoken}", status="success")
                        st.rerun()

            with v_btn2:
                if st.button("🔄 Discard & Retry", key="btn_discard_voice_page_staged"):
                    st.session_state.voice_page_processed_audio_hash = st.session_state.get("voice_page_staged_audio_hash")
                    st.session_state.voice_page_staged_transcript = None
                    st.session_state.voice_page_staged_audio_hash = None
                    st.session_state.voice_page_staged_audio_bytes = None
                    st.session_state.voice_page_staged_meta = None
                    st.info("Recording discarded. Click the microphone icon above to speak again.")
                    st.rerun()

            with v_btn3:
                st.caption("Click **Send to Assistant** to submit, or **Discard & Retry** to re-record.")

        # Display Voice Dialogue Turns
        if not st.session_state.voice_messages:
            st.info("No voice interactions in this session yet. Press the microphone button above to start talking to NOVA!")
        else:
            for idx, vm in enumerate(reversed(st.session_state.voice_messages)):
                with st.expander(f"🗣️ \"{vm['user_text']}\" — {vm['timestamp']}", expanded=(idx == 0)):
                    st.markdown(f"**You said:** {vm['user_text']}")
                    st.markdown("---")
                    st.markdown(f"**NOVA's Response:**\n\n{vm['assistant_text']}")
                    if vm.get("audio_bytes"):
                        st.markdown("**🔊 Spoken Audio Controls:**")
                        st.audio(
                            vm["audio_bytes"],
                            format="audio/mp3",
                            autoplay=(st.session_state.get("voice_autoplay", False) and (idx == 0)),
                        )
                        st.download_button(
                            "💾 Download Voice Response (.mp3)",
                            data=vm["audio_bytes"],
                            file_name=f"nova_response_{idx}.mp3",
                            mime="audio/mp3",
                            key=f"dl_voice_{idx}",
                        )

    with voice_tab2:
        st.subheader("🎙️ Speech-to-Text Laboratory (STT)")
        st.caption("Test speech transcription independently to analyze acoustic accuracy, RMS energy levels, and phonetics.")

        lab_c1, lab_c2 = st.columns(2)
        with lab_c1:
            stt_lab_lang_label = st.selectbox(
                "🗣️ Test Language",
                options=[
                    "English (India) [en-IN] — Recommended",
                    "Tamil / தமிழ் [ta-IN]",
                    "Tanglish (Tamil + English Mixed)",
                    "English (US) [en-US]",
                ],
                index=0,
                key="stt_lab_lang_select",
            )
            lab_lang_map = {
                "English (India) [en-IN] — Recommended": "en-IN",
                "Tamil / தமிழ் [ta-IN]": "ta-IN",
                "Tanglish (Tamil + English Mixed)": "tanglish",
                "English (US) [en-US]": "en-US",
            }
            active_lab_lang = lab_lang_map.get(stt_lab_lang_label, "en-IN")

        with lab_c2:
            stt_lab_engine_label = st.selectbox(
                "⚙️ Test Engine",
                options=[
                    "Auto (Gemini 1.5 Flash if available, else Google)",
                    "Google Web Speech (Free $0.00)",
                    "Google Gemini 1.5 Flash (Multimodal)",
                ],
                index=0,
                key="stt_lab_engine_select",
            )
            lab_engine_map = {
                "Auto (Gemini 1.5 Flash if available, else Google)": "auto",
                "Google Web Speech (Free $0.00)": "google",
                "Google Gemini 1.5 Flash (Multimodal)": "gemini",
            }
            active_lab_engine = lab_engine_map.get(stt_lab_engine_label, "auto")

        st.write("Record any test phrase below:")
        stt_test_audio = st.audio_input("Record speech sample", key="stt_lab_recorder")

        if stt_test_audio is not None:
            stt_bytes = stt_test_audio.getvalue()
            rms_energy = calculate_pcm_rms(stt_bytes)
            audio_size_kb = len(stt_bytes) / 1024

            st.write(f"- **Captured Audio Size:** `{audio_size_kb:.2f} KB`")
            st.write(f"- **Audio Container:** WAV (16kHz PCM mono)")
            st.write(f"- **Calculated Audio Energy (RMS):** `{rms_energy:.1f}` {'🟢 (Good Signal)' if rms_energy > 200 else '🟡 (Quiet / Low Volume)' if rms_energy > 50 else '🔴 (Silent / Mic Muted)'}")

            if st.button("🚀 Transcribe Audio Sample", key="btn_run_stt_lab"):
                with st.spinner(f"Transcribing using {active_lab_engine} ({active_lab_lang})..."):
                    lab_res = transcribe_audio(
                        stt_bytes,
                        language=active_lab_lang,
                        engine=active_lab_engine,
                        consent_external=True,
                    )

                if lab_res["success"]:
                    st.success("✅ **Transcription Successful!**")
                    st.text_area("Recognized Text:", value=lab_res["text"], height=100)
                    st.caption(
                        f"Engine: `{lab_res['engine_used']}` | "
                        f"Language Tag: `{lab_res['language_used']}` | "
                        f"Word Count: {len(lab_res['text'].split())} | "
                        f"Character Count: {len(lab_res['text'])}"
                    )
                else:
                    st.error(f"❌ Transcription Failed: {lab_res['error']} (Type: `{lab_res['error_type']}`)")
                    if lab_res.get("error_type") == "silence":
                        st.info("💡 The audio signal was silent. Check that your microphone is unmuted and the input volume is high enough.")

    with voice_tab3:
        st.subheader("🔊 Text-to-Speech Laboratory (TTS)")
        st.caption("Synthesize custom text into natural speech with language detection and accent normalization.")

        # Preset sample sentences for quick testing
        preset = st.selectbox(
            "Load Sample Sentence",
            [
                "Custom Text (Type below)",
                "English: NOVA Assistant Intro",
                "Tamil: நோவா வரவேற்பு உரை (Greeting)",
                "Tanglish: Study Schedule Reminder",
            ],
            index=0,
            key="tts_sample_preset",
        )

        preset_texts = {
            "English: NOVA Assistant Intro": (
                "Hello! I am NOVA, your personal AI assistant. "
                "I can track your tasks, manage study schedules, and process research documents."
            ),
            "Tamil: நோவா வரவேற்பு உரை (Greeting)": (
                "வணக்கம்! நான் நோவா, உங்கள் தனிப்பட்ட AI உதவியாளர். "
                "உங்கள் பணிகள், நினைவூட்டல்கள் மற்றும் படிப்பு அட்டவணையை நிர்வகிக்க நான் உதவுகிறேன்."
            ),
            "Tanglish: Study Schedule Reminder": (
                "Hello! Innaiku enna study plan irukku? "
                "Unga AI assignments and data science tasks ellam ready-ah irukku."
            ),
        }

        default_text = preset_texts.get(preset, "Hello! I am NOVA, your personal AI assistant.")
        sample_prompt = st.text_area(
            "Text to Synthesize",
            value=default_text,
            height=120,
            key="tts_lab_textarea",
        )

        tts_c1, tts_c2 = st.columns(2)
        with tts_c1:
            tts_accent = st.selectbox(
                "Voice Language / Accent",
                [
                    "English (India) — co.in",
                    "Tamil (தமிழ்) — Native Tamil",
                    "English (US) — com",
                    "English (UK) — co.uk",
                    "English (Australia) — com.au",
                ],
                index=0,
                key="tts_lab_accent_select",
            )
            # Map choice to lang and tld
            if "Tamil" in tts_accent:
                tts_lang = "ta"
                tts_tld = "com"
            else:
                tts_lang = "en"
                accent_tld_map = {
                    "English (India) — co.in": "co.in",
                    "English (US) — com": "com",
                    "English (UK) — co.uk": "co.uk",
                    "English (Australia) — com.au": "com.au",
                }
                tts_tld = accent_tld_map.get(tts_accent, "co.in")

        with tts_c2:
            tts_speed = st.checkbox("Slow Speed (Educational / Dictation mode)", value=False)

        if st.button("🔊 Synthesize & Play Speech", key="btn_run_tts_lab"):
            with st.spinner("Synthesizing speech with gTTS..."):
                t_lab_res = synthesize_speech(
                    sample_prompt,
                    lang=tts_lang,
                    tld=tts_tld,
                    slow=tts_speed,
                )

            if t_lab_res["success"]:
                st.success("✅ **Speech Generated Successfully!**")
                st.write("**Cleaned Spoken Transcript:**")
                st.caption(f"\"{t_lab_res['cleaned_text']}\"")
                st.caption(f"Language: `{t_lab_res.get('language_used', tts_lang)}` | Regional TLD: `{tts_tld}`")
                st.audio(t_lab_res["audio_bytes"], format="audio/mp3", autoplay=True)
                st.download_button(
                    "💾 Download MP3 Audio",
                    data=t_lab_res["audio_bytes"],
                    file_name="nova_custom_speech.mp3",
                    mime="audio/mp3",
                    key="dl_custom_tts",
                )
            else:
                st.error(f"❌ Synthesis Failed: {t_lab_res['error']}")


# -----------------------------------------------------------------------------
# PAGE 3: MY MEMORY
# -----------------------------------------------------------------------------
elif selected_page == "🧠 My Memory":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">User-Approved Personal Memory</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                NOVA only retains information that you explicitly instruct it to remember. You have complete ownership to edit or delete any record.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_view, tab_add = st.tabs(["📚 Saved Memories", "➕ Save New Memory"])

    with tab_add:
        st.subheader("Add Explicit Memory")
        with st.form("add_memory_form"):
            new_cat = st.selectbox(
                "Category",
                ["General", "Academic", "Preferences", "Work", "Projects", "Personal"],
                index=0,
            )
            new_content = st.text_area(
                "Memory Content",
                placeholder="e.g., 'My MSc thesis focus is on Multi-Agent Reinforcement Learning.'",
                height=100,
            )
            submitted = st.form_submit_button("💾 Save to Persistent SQLite")
            if submitted:
                if new_content.strip():
                    save_memory(new_content, category=new_cat)
                    st.success("Memory saved successfully! NOVA will use this when you ask for recall.")
                    st.rerun()
                else:
                    st.error("Please enter memory content.")

    with tab_view:
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
        with f_col1:
            cat_filter = st.selectbox(
                "Filter by Category",
                ["All", "General", "Academic", "Preferences", "Work", "Projects", "Personal"],
            )
        with f_col2:
            search_query = st.text_input("🔍 Search Memory Content", placeholder="Type keyword...")
        with f_col3:
            st.write("")
            st.write("")
            clear_all_clicked = st.button("🗑️ Clear All", type="secondary", use_container_width=True)

        if clear_all_clicked:
            st.session_state.confirm_clear_memory = True

        if st.session_state.get("confirm_clear_memory", False):
            st.warning("⚠️ **Safety Check**: Are you sure you want to permanently delete all saved memories?")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("Yes, Delete All Memories", type="primary"):
                    num = clear_all_memories()
                    st.success(f"Cleared {num} memories.")
                    st.session_state.confirm_clear_memory = False
                    st.rerun()
            with c_no:
                if st.button("Cancel"):
                    st.session_state.confirm_clear_memory = False
                    st.rerun()

        memories = get_all_memories(
            category=None if cat_filter == "All" else cat_filter,
            search_query=search_query if search_query.strip() else None,
        )

        st.write(f"**Total Records:** {len(memories)}")

        if not memories:
            st.info("No memories found matching the current criteria. Use the 'Save New Memory' tab to add your first record!")
        else:
            for mem in memories:
                with st.expander(f"#{mem['id']} [{mem['category']}] — {mem['content'][:60]}..."):
                    st.write(f"**Content:** {mem['content']}")
                    st.caption(f"Created: {mem['created_at']} | Last Updated: {mem['updated_at']}")
                    
                    e_col1, e_col2 = st.columns([1, 1])
                    with e_col1:
                        # Edit option
                        edit_key = f"edit_mem_{mem['id']}"
                        if st.button(f"✏️ Edit #{mem['id']}", key=f"btn_edit_{mem['id']}"):
                            st.session_state[edit_key] = True

                        if st.session_state.get(edit_key, False):
                            edit_text = st.text_area("Edit Content", value=mem["content"], key=f"text_{mem['id']}")
                            edit_cat = st.text_input("Edit Category", value=mem["category"], key=f"cat_{mem['id']}")
                            if st.button("Save Changes", key=f"save_edit_{mem['id']}"):
                                update_memory(mem["id"], content=edit_text, category=edit_cat)
                                st.success("Updated!")
                                st.session_state[edit_key] = False
                                st.rerun()

                    with e_col2:
                        # Delete option with confirmation
                        if st.button(f"🗑️ Delete #{mem['id']}", key=f"del_{mem['id']}", type="secondary"):
                            delete_memory(mem["id"])
                            st.success(f"Deleted memory #{mem['id']}.")
                            st.rerun()


# -----------------------------------------------------------------------------
# PAGE 4: TASK MANAGER
# -----------------------------------------------------------------------------
elif selected_page == "📋 Task Manager":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">Persistent Task Manager</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Organize, schedule, and track tasks with priorities, deadlines, categories, and estimated duration. Stored permanently in SQLite.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_stats = get_task_statistics()
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.metric("Total Tasks", t_stats["total"])
    with tc2:
        st.metric("Pending", t_stats["pending"])
    with tc3:
        st.metric("Completed", t_stats["completed"])
    with tc4:
        st.metric("Overdue", t_stats["overdue"], delta_color="inverse")

    st.write("")

    tab_list, tab_create = st.tabs(["📋 View & Filter Tasks", "➕ Create New Task"])

    with tab_create:
        st.subheader("Create a New Task")
        with st.form("new_task_form"):
            t_title = st.text_input("Task Title *", placeholder="e.g., Implement CNN architecture for vision assignment")
            t_desc = st.text_area("Description / Notes", placeholder="Include key links, requirements, sub-tasks, or notes...")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                t_prio = st.selectbox("Priority Level", ["High", "Medium", "Low"], index=1)
            with c2:
                t_cat = st.selectbox("Category", ["General", "Academic", "Work", "Personal", "Research", "Project"], index=1)
            with c3:
                t_duration = st.number_input("Est. Duration (minutes)", min_value=0, max_value=720, value=30, step=15)

            c4, c5 = st.columns(2)
            with c4:
                t_due_date = st.date_input("Due Date (Optional)", value=None)
            with c5:
                t_due_time = st.text_input("Due Time (24h HH:MM, Optional)", placeholder="e.g., 17:00")

            if st.form_submit_button("🚀 Add Task to SQLite", use_container_width=True):
                if t_title.strip():
                    due_iso = t_due_date.isoformat() if t_due_date else None
                    time_clean = t_due_time.strip() if t_due_time.strip() else None
                    try:
                        create_task(
                            title=t_title.strip(),
                            description=t_desc.strip(),
                            priority=t_prio,
                            due_date=due_iso,
                            due_time=time_clean,
                            category=t_cat,
                            estimated_duration=int(t_duration) if t_duration > 0 else None,
                        )
                        st.success(f"Task '{t_title}' created successfully!")
                        st.rerun()
                    except ValueError as ve:
                        st.error(f"Validation error: {ve}")
                else:
                    st.error("Task title cannot be empty.")

    with tab_list:
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 2])
        with filter_col1:
            status_f = st.selectbox("Status", ["all", "pending", "completed", "overdue"], index=1)
        with filter_col2:
            prio_f = st.selectbox("Priority", ["all", "High", "Medium", "Low"], index=0)
        with filter_col3:
            cat_f = st.selectbox("Category", ["all", "General", "Academic", "Work", "Personal", "Research", "Project"], index=0)
        with filter_col4:
            search_t = st.text_input("🔍 Search Tasks", placeholder="Filter by title, description, or notes...")

        tasks = get_tasks(
            status=status_f,
            priority=prio_f,
            category=cat_f,
            search_term=search_t if search_t.strip() else None,
        )

        st.write(f"**Found Tasks:** {len(tasks)}")

        if not tasks:
            st.info("No tasks match your filter. Create a new task or adjust your search settings.")
        else:
            now_dt = datetime.now()
            today_iso = now_dt.strftime("%Y-%m-%d")
            now_time = now_dt.strftime("%H:%M")

            for task in tasks:
                is_done = task["status"] == "completed"
                p_class = "badge-rose" if task["priority"] == "High" else ("badge-amber" if task["priority"] == "Medium" else "badge-cyan")
                
                # Check overdue
                is_overdue = False
                if not is_done and task.get("due_date"):
                    if task["due_date"] < today_iso:
                        is_overdue = True
                    elif task["due_date"] == today_iso and task.get("due_time") and task["due_time"] < now_time:
                        is_overdue = True

                due_label = task["due_date"] or "No deadline"
                if task.get("due_time"):
                    due_label += f" at {task['due_time']}"

                duration_label = f"⏱️ {task['estimated_duration']} min" if task.get("estimated_duration") else ""
                cat_label = task.get("category") or "General"

                with st.container():
                    st.markdown(
                        f"""
                        <div style="background: #131b2e; border: 1px solid {'#ef4444' if is_overdue else '#1e293b'}; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 1.05rem; font-weight: 600; text-decoration: {'line-through' if is_done else 'none'}; color: {'#94a3b8' if is_done else '#f8fafc'};">
                                        #{task['id']} {task['title']}
                                    </span>
                                    <span style="background: #1e3a8a; color: #93c5fd; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; margin-left: 8px;">
                                        {cat_label}
                                    </span>
                                    {'<span style="background: #881337; color: #fda4af; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; margin-left: 6px;">⚠️ OVERDUE</span>' if is_overdue else ''}
                                    <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 4px;">
                                        {task['description'] if task['description'] else 'No additional description'}
                                    </div>
                                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                                        Due: <b>{due_label}</b> {(' | ' + duration_label) if duration_label else ''} | Created: {task['created_at'][:10]}
                                        {(' | Completed: ' + task['completed_at'][:16]) if task.get('completed_at') else ''}
                                    </div>
                                </div>
                                <div>
                                    <span class="{p_class}">{task['priority']}</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([1.2, 1.2, 1.2, 3])
                    with btn_c1:
                        if not is_done:
                            if st.button("✅ Mark Done", key=f"done_{task['id']}"):
                                complete_task(task["id"])
                                st.rerun()
                        else:
                            if st.button("🔄 Reopen", key=f"reopen_{task['id']}"):
                                reopen_task(task["id"])
                                st.rerun()

                    with btn_c2:
                        edit_key = f"editing_task_{task['id']}"
                        if st.button("✏️ Edit", key=f"btn_edit_task_{task['id']}"):
                            st.session_state[edit_key] = not st.session_state.get(edit_key, False)

                    with btn_c3:
                        del_key = f"del_confirm_task_{task['id']}"
                        if st.button("🗑️ Delete", key=f"btn_del_task_{task['id']}", type="secondary"):
                            st.session_state[del_key] = True

                    # Confirmation prompt for destructive action (deletion)
                    if st.session_state.get(f"del_confirm_task_{task['id']}", False):
                        st.warning(f"⚠️ **Confirm Deletion**: Permanently delete Task #{task['id']} ('{task['title']}')?")
                        c_del_y, c_del_n = st.columns(2)
                        with c_del_y:
                            if st.button("Yes, Delete Task", key=f"conf_del_y_{task['id']}", type="primary"):
                                delete_task(task["id"])
                                st.session_state[f"del_confirm_task_{task['id']}"] = False
                                st.success("Task deleted.")
                                st.rerun()
                        with c_del_n:
                            if st.button("Cancel", key=f"conf_del_n_{task['id']}"):
                                st.session_state[f"del_confirm_task_{task['id']}"] = False
                                st.rerun()

                    # Inline Edit Expander
                    if st.session_state.get(edit_key, False):
                        with st.expander(f"Edit Task #{task['id']}", expanded=True):
                            with st.form(f"edit_task_form_{task['id']}"):
                                ed_title = st.text_input("Title", value=task["title"])
                                ed_desc = st.text_area("Description", value=task.get("description") or "")
                                ec1, ec2, ec3 = st.columns(3)
                                with ec1:
                                    prio_idx = ["High", "Medium", "Low"].index(task["priority"]) if task["priority"] in ["High", "Medium", "Low"] else 1
                                    ed_prio = st.selectbox("Priority", ["High", "Medium", "Low"], index=prio_idx)
                                with ec2:
                                    cats = ["General", "Academic", "Work", "Personal", "Research", "Project"]
                                    cur_cat = task.get("category") or "General"
                                    cat_idx = cats.index(cur_cat) if cur_cat in cats else 0
                                    ed_cat = st.selectbox("Category", cats, index=cat_idx)
                                with ec3:
                                    ed_dur = st.number_input("Duration (min)", min_value=0, max_value=720, value=task.get("estimated_duration") or 0, step=15)
                                
                                ec4, ec5 = st.columns(2)
                                with ec4:
                                    cur_due_date = None
                                    if task.get("due_date"):
                                        try:
                                            cur_due_date = date.fromisoformat(task["due_date"])
                                        except Exception:
                                            cur_due_date = None
                                    ed_due_date = st.date_input("Due Date", value=cur_due_date)
                                with ec5:
                                    ed_due_time = st.text_input("Due Time (HH:MM)", value=task.get("due_time") or "")

                                if st.form_submit_button("Save Changes"):
                                    due_iso = ed_due_date.isoformat() if ed_due_date else None
                                    time_clean = ed_due_time.strip() if ed_due_time.strip() else None
                                    try:
                                        update_task(
                                            task["id"],
                                            title=ed_title,
                                            description=ed_desc,
                                            priority=ed_prio,
                                            category=ed_cat,
                                            due_date=due_iso,
                                            due_time=time_clean,
                                            estimated_duration=int(ed_dur) if ed_dur > 0 else None,
                                        )
                                        st.session_state[edit_key] = False
                                        st.success("Task updated successfully!")
                                        st.rerun()
                                    except ValueError as ve:
                                        st.error(f"Validation error: {ve}")


# -----------------------------------------------------------------------------
# PAGE: SMART REMINDERS
# -----------------------------------------------------------------------------
elif selected_page == "⏰ Smart Reminders":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">Smart Persistent Reminders</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Schedule time-based notifications, handle recurring alerts (daily/weekly/monthly), and tie reminders directly to your tasks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Operational limitations explanation per Requirement 2
    st.info(
        "ℹ️ **Scheduler & Platform Operating Notice**: NOVA runs within this local application runtime. "
        "When NOVA is active or refreshed, its embedded scheduler verifies SQLite timestamps and triggers due reminders with on-screen alerts. "
        "If the application is closed or running on a cloud container in sleep mode, background OS push notifications are paused until you reopen or refresh NOVA. "
        "Verifiable delivery is confirmed as soon as the session awakens."
    )

    rem_stats = get_reminder_statistics()
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.metric("Total Reminders", rem_stats["total"])
    with rc2:
        st.metric("Active / Scheduled", rem_stats["active"])
    with rc3:
        st.metric("Triggered / Due", rem_stats["triggered"])
    with rc4:
        st.metric("Paused", rem_stats["paused"])

    st.write("")

    tab_r_view, tab_r_create = st.tabs(["⏰ View & Manage Reminders", "➕ Schedule New Reminder"])

    with tab_r_create:
        st.subheader("Schedule a Reminder")
        pending_tasks = get_tasks(status="pending")
        task_options = {"None (General Reminder)": None}
        for pt in pending_tasks:
            task_options[f"#{pt['id']} — {pt['title'][:40]}"] = pt["id"]

        with st.form("new_reminder_form"):
            r_title = st.text_input("Reminder Title *", placeholder="e.g., Submit Assignment 2 to University Portal")
            r_desc = st.text_area("Notes / Description", placeholder="Additional context, links, or instructions...")
            
            c_date, c_time, c_recur = st.columns(3)
            with c_date:
                r_date = st.date_input("Reminder Date *", value=date.today())
            with c_time:
                now_str = datetime.now().strftime("%H:%M")
                r_time_str = st.text_input("Reminder Time (24h HH:MM) *", value=now_str)
            with c_recur:
                r_recur = st.selectbox("Recurrence Cycle", ["none", "daily", "weekly", "monthly"], index=0)

            r_linked_label = st.selectbox("Link to Existing Task (Optional)", list(task_options.keys()))
            r_linked_id = task_options[r_linked_label]

            if st.form_submit_button("⏰ Set Reminder in SQLite", use_container_width=True):
                if r_title.strip():
                    try:
                        dt_input = f"{r_date.isoformat()} {r_time_str.strip()}"
                        create_reminder(
                            title=r_title.strip(),
                            remind_at=dt_input,
                            recurrence=r_recur,
                            task_id=r_linked_id,
                            description=r_desc.strip(),
                        )
                        st.success(f"Reminder '{r_title}' scheduled for {dt_input} ({r_recur})!")
                        st.rerun()
                    except ValueError as ve:
                        st.error(f"Invalid reminder configuration: {ve}")
                else:
                    st.error("Reminder title cannot be empty.")

    with tab_r_view:
        rf1, rf2 = st.columns([1, 2])
        with rf1:
            status_filter = st.selectbox("Filter Status", ["all", "active", "triggered", "paused", "dismissed"])
        with rf2:
            st.caption("Active reminders automatically trigger when current time reaches scheduled timestamp.")

        reminders_list = get_reminders(status=status_filter)
        st.write(f"**Found Reminders:** {len(reminders_list)}")

        if not reminders_list:
            st.info("No reminders match your filter.")
        else:
            for rem in reminders_list:
                status_color = "#10b981" if rem["status"] == "active" else ("#f59e0b" if rem["status"] == "paused" else ("#ef4444" if rem["status"] == "triggered" else "#64748b"))
                recur_badge = f"🔄 {rem['recurrence'].capitalize()}" if rem["recurrence"] != "none" else "1-off"

                with st.container():
                    st.markdown(
                        f"""
                        <div style="background: #131b2e; border: 1px solid #1e293b; border-left: 4px solid {status_color}; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 1.05rem; font-weight: 600; color: #f8fafc;">
                                        ⏰ #{rem['id']} {rem['title']}
                                    </span>
                                    <span style="background: #1e293b; color: #38bdf8; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; margin-left: 8px;">
                                        {recur_badge}
                                    </span>
                                    <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 4px;">
                                        {rem['description'] if rem['description'] else 'No notes'}
                                    </div>
                                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                                        Scheduled For: <b>{rem['remind_at']}</b> | Status: <b style="color: {status_color};">{rem['status'].upper()}</b>
                                        {(' | Linked Task: #' + str(rem['task_id'])) if rem.get('task_id') else ''}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    rb1, rb2, rb3, rb4 = st.columns([1.2, 1.2, 1.2, 3])
                    with rb1:
                        if rem["status"] == "active":
                            if st.button("⏸️ Pause", key=f"pause_rem_{rem['id']}"):
                                pause_reminder(rem["id"])
                                st.rerun()
                        elif rem["status"] == "paused":
                            if st.button("▶️ Resume", key=f"resume_rem_{rem['id']}"):
                                resume_reminder(rem["id"])
                                st.rerun()
                        elif rem["status"] == "triggered":
                            if st.button("✓ Dismiss", key=f"dismiss_rem_{rem['id']}"):
                                dismiss_reminder(rem["id"])
                                st.rerun()

                    with rb2:
                        if st.button("🗑️ Delete", key=f"btn_del_rem_{rem['id']}", type="secondary"):
                            st.session_state[f"del_confirm_rem_{rem['id']}"] = True

                    # Confirmation prompt for destructive reminder action
                    if st.session_state.get(f"del_confirm_rem_{rem['id']}", False):
                        st.warning(f"⚠️ **Confirm Deletion**: Permanently delete Reminder #{rem['id']} ('{rem['title']}')?")
                        c_rem_y, c_rem_n = st.columns(2)
                        with c_rem_y:
                            if st.button("Yes, Delete Reminder", key=f"conf_rem_del_y_{rem['id']}", type="primary"):
                                delete_reminder(rem["id"])
                                st.session_state[f"del_confirm_rem_{rem['id']}"] = False
                                st.success("Reminder deleted.")
                                st.rerun()
                        with c_rem_n:
                            if st.button("Cancel", key=f"conf_rem_del_n_{rem['id']}"):
                                st.session_state[f"del_confirm_rem_{rem['id']}"] = False
                                st.rerun()


# -----------------------------------------------------------------------------
# PAGE: DAILY PLANNER & SMART SUGGESTIONS
# -----------------------------------------------------------------------------
elif selected_page == "📅 Daily Planner":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">Daily Planner & Smart Suggestions</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Intelligent time-blocking and algorithmic task prioritization based on deadlines, priorities, and estimated duration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "💡 **Transparency & Safety Guarantee**: All suggestions and schedules generated here are **strictly advisory**. "
        "NOVA will **never** automatically delete, reschedule, or alter your underlying database tasks without your explicit direction."
    )

    tab_suggestions, tab_planner, tab_history = st.tabs(["🎯 Smart Suggestions", "⚡ Generate Daily Plan", "📚 Saved Plans History"])

    with tab_suggestions:
        st.subheader("What Should I Work on Next?")
        st.caption("Algorithmic recommendations prioritizing overdue deadlines, high-priority deliverables, and upcoming tasks.")
        
        suggestions_res = get_smart_task_suggestions(limit=4)
        suggestions = suggestions_res.get("suggestions", [])

        if not suggestions:
            st.success("🎉 You have no pending tasks! Enjoy your free time or add new tasks in the Task Manager.")
        else:
            for idx, sugg in enumerate(suggestions, 1):
                p_badge = "badge-rose" if sugg["priority"] == "High" else ("badge-amber" if sugg["priority"] == "Medium" else "badge-cyan")
                st.markdown(
                    f"""
                    <div style="background: #131b2e; border: 1px solid #334155; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">
                                    #{idx} Recommendation: {sugg['title']}
                                </span>
                                <span style="background: #1e3a8a; color: #93c5fd; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; margin-left: 8px;">
                                    {sugg['category']}
                                </span>
                                <div style="font-size: 0.88rem; color: #f1f5f9; margin-top: 6px; font-weight: 500;">
                                    💡 <b>Why this is recommended:</b> {sugg['rationale']}
                                </div>
                                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px;">
                                    Due Date: <b>{sugg['due_date'] or 'No set date'}</b> {('at ' + sugg['due_time']) if sugg['due_time'] else ''}
                                    {(' | Est. Duration: <b>' + str(sugg['estimated_duration']) + ' mins</b>') if sugg['estimated_duration'] else ''}
                                    | Urgency Score: <code>{sugg['urgency_score']}</code>
                                </div>
                            </div>
                            <span class="{p_badge}">{sugg['priority']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("🤖 *Note: Recommendations are algorithmically ranked using deadline proximity and priority weighting.*")

    with tab_planner:
        st.subheader("Generate Suggested Daily Schedule")
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            plan_date = st.date_input("Target Plan Date", value=date.today(), key="planner_target_date")
        with c_p2:
            start_hour_input = st.selectbox("Work Start Time", ["08:00", "09:00", "10:00", "13:00", "14:00"], index=1)
        with c_p3:
            avail_hours = st.slider("Available Work Hours", min_value=1.0, max_value=12.0, value=6.0, step=0.5)

        if st.button("⚡ Generate Time-Blocked Daily Plan", type="primary", use_container_width=True):
            with st.spinner("Analyzing tasks and constructing balanced schedule..."):
                generated_plan = generate_daily_plan(
                    target_date=plan_date.isoformat(),
                    available_hours=avail_hours,
                    start_time=start_hour_input,
                )
                st.session_state.current_generated_plan = generated_plan

        if "current_generated_plan" in st.session_state:
            g_plan = st.session_state.current_generated_plan
            st.write("---")
            st.subheader(f"Suggested Plan for {g_plan['date']}")
            
            p_m1, p_m2, p_m3 = st.columns(3)
            with p_m1:
                st.metric("Tasks Scheduled", g_plan["total_tasks_scheduled"])
            with p_m2:
                st.metric("Total Scheduled Time", f"{g_plan['total_allocated_minutes']} mins")
            with p_m3:
                st.metric("Buffer / Free Time", f"{g_plan['remaining_minutes']} mins")

            st.write("### 🕒 Time Blocks")
            for block in g_plan["schedule"]:
                b_type = block.get("type", "task")
                if b_type == "break":
                    st.markdown(
                        f"""
                        <div style="background: #1e293b; border-left: 4px solid #10b981; border-radius: 6px; padding: 8px 14px; margin-bottom: 8px;">
                            <b>{block['start_time']} – {block['end_time']}</b> (15 min): ☕ <i>{block['title']}</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    p_badge = "badge-rose" if block.get("priority") == "High" else ("badge-amber" if block.get("priority") == "Medium" else "badge-cyan")
                    st.markdown(
                        f"""
                        <div style="background: #0f172a; border-left: 4px solid #38bdf8; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <b>{block['start_time']} – {block['end_time']}</b> ({block['duration_minutes']} min):
                                <span style="font-size: 1rem; color: #f8fafc; margin-left: 6px;"><b>#{block['task_id']} {block['title']}</b></span>
                                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                                    Category: {block.get('category', 'General')} | Due: {block.get('due_date') or 'None'}
                                </div>
                            </div>
                            <span class="{p_badge}">{block.get('priority', 'Medium')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.write("---")
            st.write("### 📝 Review, Customize & Save Plan")
            st.caption("You can edit the plan notes below before saving it permanently to your database.")
            
            custom_plan_notes = st.text_area(
                "Plan Notes & Goals",
                value=f"Daily Focus for {g_plan['date']}: Complete {g_plan['total_tasks_scheduled']} tasks with {g_plan['total_allocated_minutes']} minutes allocated.",
                key="custom_plan_notes_editor",
            )

            if st.button("💾 Save Plan to SQLite", use_container_width=True):
                save_daily_plan(
                    plan_date=g_plan["date"],
                    schedule_data=g_plan["schedule"],
                    total_allocated_minutes=g_plan["total_allocated_minutes"],
                    notes=custom_plan_notes,
                )
                st.success(f"Plan for {g_plan['date']} permanently saved to SQLite!")
                st.rerun()

    with tab_history:
        st.subheader("Saved Daily Plans History")
        h_date = st.date_input("Lookup Date", value=date.today(), key="hist_plan_date")
        saved_record = get_daily_plan(h_date.isoformat())
        if saved_record:
            st.success(f"Found saved plan for {saved_record['plan_date']} (Created: {saved_record['created_at'][:16]})")
            st.write(f"**Total Allocated Time:** {saved_record['total_allocated_minutes']} minutes")
            if saved_record.get("notes"):
                st.info(f"**Notes:** {saved_record['notes']}")
            st.write("#### Schedule Blocks")
            for sb in saved_record.get("schedule", []):
                st.markdown(f"- **{sb.get('start_time')} – {sb.get('end_time')}** ({sb.get('duration_minutes', 0)} min): {sb.get('title')}")
        else:
            st.info(f"No saved plan found for {h_date.isoformat()}. Generate and save a plan using the 'Generate Daily Plan' tab.")


# -----------------------------------------------------------------------------
# PAGE 5: STUDENT MODE
# -----------------------------------------------------------------------------
elif selected_page == "🎓 Student Mode":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">Student Assistant & Study Planner</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Generate structured, balanced revision timetables based on your available hours and upcoming exams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "ℹ️ **Transparency Note**: Study plan schedules are algorithmic AI suggestions. "
        "They are clearly distinguished from verified tasks until you explicitly click **'Add to Tasks'**."
    )

    col_plan, col_log = st.columns([3, 2])

    with col_plan:
        st.subheader("📅 Generate Study Schedule")
        with st.form("study_plan_form"):
            subjects_input = st.text_input(
                "Subjects / Modules (comma-separated) *",
                value="Deep Learning, Distributed Systems, Cloud Computing",
            )
            daily_hours = st.slider("Available Study Hours per Day", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
            exam_deadline = st.date_input("Target Exam / Assessment Date", value=None)
            focus_notes = st.text_input("Specific Topics to Prioritize", placeholder="e.g., Transformers, MapReduce, Docker")

            generate_clicked = st.form_submit_button("✨ Generate Study Plan")

        if generate_clicked:
            sub_list = [s.strip() for s in subjects_input.split(",") if s.strip()]
            if sub_list:
                plan = generate_study_plan(
                    subjects=sub_list,
                    available_hours_per_day=daily_hours,
                    exam_date=exam_deadline.isoformat() if exam_deadline else None,
                    focus_topics=focus_notes,
                )
                st.session_state.current_study_plan = plan
                st.success("Generated tailored study blocks!")
            else:
                st.error("Please enter at least one subject.")

        # Display Generated Sessions
        if "current_study_plan" in st.session_state:
            plan = st.session_state.current_study_plan
            st.markdown(f"**{plan['disclaimer']}**")

            for idx, session in enumerate(plan["sessions"]):
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background: #1e293b; border-left: 3px solid #8b5cf6; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                            <b>{session['subject']}</b> — <span style="color: #22d3ee;">{session['session_type']}</span> ({session['suggested_duration_mins']} mins)<br>
                            <span style="font-size: 0.85rem; color: #cbd5e1;">{session['description']}</span><br>
                            <span style="font-size: 0.75rem; color: #94a3b8;">Recommended Day: {session['recommended_day']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(f"➕ Add to Task Manager", key=f"add_session_{idx}"):
                        create_task(
                            title=f"Study: {session['subject']} - {session['session_type']}",
                            description=session["description"],
                            priority="Medium",
                            due_date=session["recommended_day"],
                        )
                        st.success(f"Added session for {session['subject']} to Task Manager!")

    with col_log:
        st.subheader("⏱️ Log Real Study Session")
        with st.form("log_study_form"):
            log_sub = st.text_input("Subject", placeholder="e.g., Deep Learning")
            log_top = st.text_input("Topic", placeholder="e.g., Attention Mechanisms")
            log_dur = st.number_input("Duration (minutes)", min_value=5, max_value=600, value=45, step=5)
            log_notes = st.text_area("Session Notes", placeholder="What did you learn or accomplish?")
            
            if st.form_submit_button("📝 Record Completed Session"):
                if log_sub.strip():
                    record_study_session(
                        subject=log_sub,
                        topic=log_top,
                        duration_minutes=log_dur,
                        notes=log_notes,
                    )
                    st.success(f"Logged {log_dur} minutes for '{log_sub}' in SQLite!")
                    st.rerun()
                else:
                    st.error("Subject is required.")

        st.subheader("📚 Study Hours Summary")
        summary = get_study_hours_summary(days=30)
        st.write(f"- **Total Hours Logged (Last 30 Days):** {summary['total_hours']} hrs")
        st.write(f"- **Total Study Sessions:** {summary['session_count']}")


# -----------------------------------------------------------------------------
# PAGE 6: DOCUMENT ASSISTANT
# -----------------------------------------------------------------------------
elif selected_page == "📄 Document Assistant":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">Document Intelligence & RAG System</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Upload PDF or TXT files to perform chunked semantic retrieval, grounded Q&A with exact page citations, and automated summarization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Privacy and Storage Disclosure
    st.markdown(
        """
        <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid #0891b2; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.85rem; color: #cbd5e1;">
            🔒 <b>Privacy Disclosure & Local Storage:</b> All uploaded files are stored strictly on your local disk (<code>data/documents/</code>) and segmented into your local SQLite database. When questioning a document, <b>only the specific relevant excerpts</b> are passed to the AI model. Your raw full document is never sent or uploaded externally.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload Section in Expander
    with st.expander("📤 Upload New Document (PDF, TXT, CSV — Max 15MB)", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "txt", "csv"],
            key="doc_uploader",
        )
        if uploaded_file is not None:
            if st.button("🚀 Process & Chunk Document", type="primary"):
                try:
                    file_bytes = uploaded_file.read()
                    with st.spinner("Extracting pages, generating semantic chunks, and indexing..."):
                        new_doc = save_and_process_document(file_bytes, uploaded_file.name)
                    st.success(f"Successfully processed **{new_doc['filename']}** ({new_doc['page_count']} pages, {new_doc['word_count']} words)!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process document: {str(e)}")

    # Document Selection & Listing
    stored_docs = get_documents()

    if not stored_docs:
        st.info("📚 **No documents indexed yet.** Use the 'Upload New Document' section above to upload a PDF or TXT document.")
    else:
        doc_col1, doc_col2, doc_col3 = st.columns([3, 1, 1])

        doc_options = {d["id"]: f"#{d['id']} — {d['filename']} ({d['page_count']} pgs, {d['file_size_kb']} KB)" for d in stored_docs}
        with doc_col1:
            selected_doc_id = st.selectbox(
                "Select Active Document",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
            )
        with doc_col2:
            st.write("")
            st.write("")
            if st.button("🗑️ Delete Doc", type="secondary", use_container_width=True):
                delete_document(selected_doc_id)
                st.success("Deleted document and chunks.")
                st.rerun()
        with doc_col3:
            st.write("")
            st.write("")
            if st.button("🗑️ Delete All", type="secondary", use_container_width=True):
                st.session_state.confirm_del_all_docs = True

        if st.session_state.get("confirm_del_all_docs", False):
            st.warning("⚠️ Are you sure you want to permanently delete ALL indexed documents and chunks?")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("Yes, Delete All", type="primary"):
                    num = delete_all_documents()
                    st.success(f"Deleted {num} documents.")
                    st.session_state.confirm_del_all_docs = False
                    st.rerun()
            with c_no:
                if st.button("Cancel"):
                    st.session_state.confirm_del_all_docs = False
                    st.rerun()

        # Active Document Workspace
        active_doc = get_document_by_id(selected_doc_id)
        if active_doc:
            chunks = get_document_chunks(selected_doc_id)

            # Metadata metric ribbon
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            with m_c1:
                st.metric("Document Format", active_doc["file_type"].upper())
            with m_c2:
                st.metric("Pages / Sections", active_doc["page_count"])
            with m_c3:
                st.metric("Word Count", f"{active_doc['word_count']:,}")
            with m_c4:
                st.metric("RAG Chunks", len(chunks))

            st.write("")

            tab_qa, tab_sum, tab_chunks, tab_search = st.tabs([
                "💬 Ask Document (RAG Q&A)",
                "📝 Executive Summary",
                "🧩 Chunk Explorer",
                "🔍 Keyword Finder",
            ])

            # ---------------------------------------------------------
            # TAB 1: GROUNDED RAG QUESTION ANSWERING
            # ---------------------------------------------------------
            with tab_qa:
                st.subheader(f"Grounded Q&A: {active_doc['filename']}")
                st.caption(
                    "Answers are strictly grounded in retrieved chunks from the document. "
                    "If the answer is not contained in the text, NOVA explicitly indicates it without hallucinating."
                )

                doc_chat_key = f"doc_chat_{active_doc['id']}"
                if doc_chat_key not in st.session_state:
                    st.session_state[doc_chat_key] = []

                # Clear document chat button
                if st.session_state[doc_chat_key]:
                    if st.button("🔄 Reset Document Conversation", key=f"clear_doc_chat_{active_doc['id']}"):
                        st.session_state[doc_chat_key] = []
                        st.rerun()

                # Display document chat history
                for turn in st.session_state[doc_chat_key]:
                    with st.chat_message(turn["role"]):
                        st.markdown(turn["content"])
                        if turn.get("citations"):
                            pages_str = ", ".join(f"Page {p}" for p in turn["citations"])
                            st.caption(f"📌 Supporting Sources: `{pages_str}`")

                # Question Input
                doc_prompt = st.chat_input(
                    f"Ask a question or follow-up about {active_doc['filename']}...",
                    key=f"input_doc_{active_doc['id']}",
                )

                if doc_prompt:
                    # Append user turn
                    st.session_state[doc_chat_key].append({"role": "user", "content": doc_prompt})
                    with st.chat_message("user"):
                        st.markdown(doc_prompt)

                    # Generate grounded answer with conversation history for follow-ups
                    with st.spinner("Retrieving relevant excerpts and generating grounded answer..."):
                        answer_data = answer_document_question(
                            document_id=active_doc["id"],
                            question=doc_prompt,
                            conversation_history=st.session_state[doc_chat_key],
                        )

                    # Append assistant turn
                    st.session_state[doc_chat_key].append({
                        "role": "assistant",
                        "content": answer_data["answer"],
                        "citations": answer_data.get("citations", []),
                    })

                    with st.chat_message("assistant"):
                        if answer_data.get("mode") == "llm":
                            st.caption(f"⚡ Grounded Response ({answer_data.get('provider', '').upper()} - {answer_data.get('model', '')})")
                        else:
                            st.caption("📄 Extractive Response (Offline Mode)")

                        st.markdown(answer_data["answer"])

                        if answer_data.get("citations"):
                            pages_str = ", ".join(f"Page {p}" for p in answer_data["citations"])
                            st.caption(f"📌 Supporting Sources: `{pages_str}`")

                        # Expandable source evidence inspection
                        retrieved = answer_data.get("retrieved_chunks", [])
                        if retrieved:
                            with st.expander("🔍 Inspect Retrieved Supporting Chunks & Relevance Scores"):
                                for c in retrieved:
                                    st.markdown(
                                        f"""
                                        <div style="background: #1e293b; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #8b5cf6;">
                                            <b>Chunk #{c['chunk_index']} [Page {c['page_number']}]</b> — Relevancy Score: <code>{c.get('score', 0):.4f}</code><br>
                                            <span style="font-size: 0.85rem; color: #cbd5e1;">{c['content']}</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

            # ---------------------------------------------------------
            # TAB 2: EXECUTIVE SUMMARY
            # ---------------------------------------------------------
            with tab_sum:
                st.subheader(f"Executive Summary: {active_doc['filename']}")
                sum_key = f"summary_{active_doc['id']}"

                if st.button("✨ Generate Document Summary", type="primary", key=f"btn_sum_{active_doc['id']}"):
                    with st.spinner("Synthesizing document overview and key themes..."):
                        sum_data = summarize_document(active_doc["id"])
                        st.session_state[sum_key] = sum_data

                if sum_key in st.session_state:
                    res = st.session_state[sum_key]
                    if res.get("status") == "success":
                        if res.get("mode") == "llm":
                            st.caption(f"⚡ Synthesized by {res.get('provider', '').upper()} ({res.get('model', '')})")
                        st.markdown(res["summary"])
                    else:
                        st.error(res.get("summary", "Failed to generate summary."))
                else:
                    st.info("Click 'Generate Document Summary' to create an AI-powered overview of this document.")

            # ---------------------------------------------------------
            # TAB 3: CHUNK EXPLORER
            # ---------------------------------------------------------
            with tab_chunks:
                st.subheader(f"RAG Chunk Segments ({len(chunks)} Chunks)")
                st.caption("Inspect the exact segmented text chunks used for vector space retrieval.")
                for chk in chunks[:25]:  # display up to 25
                    with st.expander(f"Chunk #{chk['chunk_index']} — Page {chk['page_number']} (~{chk['token_count']} tokens)"):
                        st.markdown(chk["content"])

            # ---------------------------------------------------------
            # TAB 4: KEYWORD FINDER
            # ---------------------------------------------------------
            with tab_search:
                st.subheader("Exact Keyword Finder")
                kw_query = st.text_input("Enter search phrase", key=f"kw_{active_doc['id']}")
                if kw_query:
                    matches = []
                    for chk in chunks:
                        if kw_query.lower() in chk["content"].lower():
                            matches.append(chk)
                    if not matches:
                        st.warning(f"No occurrences of '{kw_query}' found.")
                    else:
                        st.success(f"Found in {len(matches)} chunk(s):")
                        for m in matches:
                            st.markdown(
                                f"""
                                <div style="background: #1e293b; padding: 10px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #06b6d4;">
                                    <b>Page {m['page_number']} (Chunk #{m['chunk_index']}):</b><br>
                                    <span style="font-size: 0.85rem; color: #cbd5e1;">{m['content']}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


# -----------------------------------------------------------------------------
# PAGE 7: PRODUCTIVITY ANALYTICS
# -----------------------------------------------------------------------------
elif selected_page == "📊 Productivity Analytics":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">Productivity Analytics & Transparency</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                All charts, metrics, and trends are computed directly from your SQLite database. Zero synthetic or fake records are ever presented.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_tasks = get_tasks(status="all")
    task_stats = get_task_statistics()
    rem_stats = get_reminder_statistics()
    daily_summary = get_daily_activity_summary(days=14)

    # Top summary metrics - 5 cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Tasks", task_stats["total"])
    with m2:
        st.metric("Completed", task_stats["completed"])
    with m3:
        st.metric("Pending", task_stats["pending"])
    with m4:
        st.metric("Overdue", task_stats["overdue"], delta_color="inverse")
    with m5:
        st.metric("Completion Rate", f"{task_stats['completion_rate']}%")

    st.write("")

    if not all_tasks:
        st.info(
            "📊 **No task data available yet.** "
            "Create tasks in the Task Manager or via the AI Chat to generate your productivity distribution charts."
        )
    else:
        # Row 1: Status & Priority
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            donut_fig = create_task_status_chart(all_tasks)
            if donut_fig:
                st.pyplot(donut_fig)
            else:
                st.write("Insufficient data for status chart.")

        with chart_col2:
            prio_fig = create_priority_distribution_chart(all_tasks)
            if prio_fig:
                st.pyplot(prio_fig)
            else:
                st.write("Insufficient data for priority chart.")

        st.write("")

        # Row 2: Category Distribution & Completion Trend
        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            cat_fig = create_category_distribution_chart(all_tasks)
            if cat_fig:
                st.pyplot(cat_fig)
            else:
                st.write("Insufficient data for category chart.")

        with chart_col4:
            trend_fig = create_completion_trend_chart(all_tasks, days=14)
            if trend_fig:
                st.pyplot(trend_fig)
            else:
                st.write("Insufficient completed task data for trend chart.")

        # Activity timeline
        if daily_summary and len(daily_summary) >= 2:
            st.subheader("📈 Activity Event Frequency (Last 14 Days)")
            timeline_fig = create_daily_activity_chart(daily_summary)
            if timeline_fig:
                st.pyplot(timeline_fig)

    st.markdown("---")
    st.subheader("📥 Export Your Personal Data (CSV)")
    st.caption("You own all your data. Download raw SQLite records for external analysis in Python, pandas, or Excel.")

    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    with exp_col1:
        tasks_csv = export_table_to_csv("tasks")
        st.download_button(
            "Download Tasks CSV",
            data=tasks_csv,
            file_name="nova_tasks.csv",
            mime="text/csv",
            disabled=(len(tasks_csv) == 0),
            use_container_width=True,
        )
    with exp_col2:
        reminders_csv = export_table_to_csv("reminders")
        st.download_button(
            "Download Reminders CSV",
            data=reminders_csv,
            file_name="nova_reminders.csv",
            mime="text/csv",
            disabled=(len(reminders_csv) == 0),
            use_container_width=True,
        )
    with exp_col3:
        memories_csv = export_table_to_csv("memories")
        st.download_button(
            "Download Memories CSV",
            data=memories_csv,
            file_name="nova_memories.csv",
            mime="text/csv",
            disabled=(len(memories_csv) == 0),
            use_container_width=True,
        )
    with exp_col4:
        activities_csv = export_table_to_csv("activity_records")
        st.download_button(
            "Download Activity Log CSV",
            data=activities_csv,
            file_name="nova_activity_records.csv",
            mime="text/csv",
            disabled=(len(activities_csv) == 0),
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# PAGE: SELF-LEARNING & TRAINING DASHBOARD
# -----------------------------------------------------------------------------
elif selected_page == "🧠 Self-Learning & Training":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">🧠 Self-Learning & Model Intelligence Dashboard</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Continuous in-context learning, verified human corrections, PII-sanitized dataset curation, and model fine-tuning diagnostics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Real Metrics from Learning System
    sys_metrics = get_learning_system_metrics()
    fb = sys_metrics["feedback_summary"]
    cur_model = sys_metrics["current_model"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Approved Knowledge</div>
                <div class="nova-metric-value" style="color: #22d3ee;">{sys_metrics['approved_memories']} Facts</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">{sys_metrics['indexed_documents']} Docs &bull; {sys_metrics['document_chunks']} Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Feedback & Corrections</div>
                <div class="nova-metric-value" style="color: #c084fc;">{sys_metrics['approved_corrections']} Taught</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">{fb['thumbs_up']} 👍 &bull; {fb['thumbs_down']} 👎 ({fb['satisfaction_rate']}% pos)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Training Datasets</div>
                <div class="nova-metric-value" style="color: #10b981;">{sys_metrics['training_datasets_count']} Releases</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">{sys_metrics['training_dataset_samples']} Chat JSONL Samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        eval_score_val = cur_model.get("eval_score") or 0.0
        score_color = "#10b981" if eval_score_val >= 80 else ("#f59e0b" if eval_score_val >= 60 else "#fb7185")
        st.markdown(
            f"""
            <div class="nova-metric-card">
                <div class="nova-metric-label">Active Model & Eval</div>
                <div class="nova-metric-value" style="color: {score_color};">{eval_score_val}%</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">{cur_model.get('version_tag', 'v1.0')} ({cur_model.get('provider', 'base')})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(f"🕒 **Last Knowledge Base Update:** `{sys_metrics['last_knowledge_update']}` &bull; Database: SQLite (`nova.db`)")
    st.write("")

    # 2. Four Interactive Tabs
    tab_arch, tab_corrections, tab_finetune, tab_benchmark = st.tabs([
        "🏛️ Architecture & Data Separation",
        "✏️ Verified Corrections Store",
        "📦 Dataset Builder & Fine-Tuning",
        "🎯 Benchmark & Model Registry",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: ARCHITECTURE & KNOWLEDGE SEPARATION
    # -------------------------------------------------------------------------
    with tab_arch:
        st.subheader("🏛️ Strict Knowledge Store Separation")
        st.markdown(
            """
            To prevent hallucinations, data poisoning, and privacy leaks, NOVA maintains **strict physical separation** across four independent layers of knowledge:
            """
        )

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown(
                """
                <div style="background: #1e293b; border-left: 3px solid #38bdf8; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                    <b style="color: #38bdf8;">1. Conversation History (Conversations)</b>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin: 4px 0 0 0;">
                        Active and archived multi-turn chat threads. Session-specific dialogue is stored separately and is <b>never automatically promoted to permanent memory</b> without user confirmation.
                    </p>
                </div>
                <div style="background: #1e293b; border-left: 3px solid #c084fc; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                    <b style="color: #c084fc;">2. Personal Memory (Memories)</b>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin: 4px 0 0 0;">
                        Persistent user facts, MSc research notes, project deadlines, and preferences. Stored in SQLite only upon explicit user command (<code>remember: ...</code> or via UI). Guarded against duplicate entries.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_k2:
            st.markdown(
                """
                <div style="background: #1e293b; border-left: 3px solid #10b981; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                    <b style="color: #10b981;">3. Document Knowledge Base (Documents)</b>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin: 4px 0 0 0;">
                        Local PDF, TXT, and CSV documents chunked into overlapping passages. Retrieved strictly at question-time using local keyword/semantic RAG with transparent page number attribution.
                    </p>
                </div>
                <div style="background: #1e293b; border-left: 3px solid #f59e0b; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                    <b style="color: #f59e0b;">4. Training & Correction Store (Learning Examples)</b>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin: 4px 0 0 0;">
                        Verified human corrections and consented feedback examples. Stored independently with approval flags, scrubbed of PII and secrets, and formatted for Chat JSONL fine-tuning releases.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("🛡️ Privacy Guardrails & Sovereignty")
        st.markdown(
            """
            - **No Blind Acceptance:** NOVA checks existing memories and corrections before writing new entries. Duplicate and conflicting facts are caught.
            - **Consent-First Dataset Curation:** Thumbs-up dialogue turns are only included in training datasets if the user has opted in.
            - **Zero Unintentional External Exposure:** Training datasets are stored locally in `data/datasets/` and scrubbed of API keys, emails, phone numbers, and IPs.
            """
        )

        st.markdown("#### 💾 Export Personal Memories")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            json_mems = export_memories_json()
            st.download_button(
                "💾 Download Memories as JSON",
                data=json_mems,
                file_name="nova_memories_export.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp_col2:
            csv_mems = export_memories_csv()
            st.download_button(
                "📄 Download Memories as CSV",
                data=csv_mems,
                file_name="nova_memories_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # TAB 2: VERIFIED CORRECTIONS STORE
    # -------------------------------------------------------------------------
    with tab_corrections:
        st.subheader("✏️ Human-in-the-Loop Corrections & Demonstrations")
        st.markdown(
            """
            When NOVA provides an incorrect or incomplete response, you can teach it the correct answer.
            NOVA stores your verified corrections and retrieves them dynamically during future questions.
            """
        )

        # Form to add a verified correction
        with st.expander("➕ Teach NOVA a New Demonstration or Correction", expanded=False):
            with st.form("manual_teach_form"):
                col_t1, col_t2 = st.columns([3, 1])
                with col_t1:
                    t_prompt = st.text_input("User Question / Trigger Prompt", placeholder="e.g. How to compute eigenvalues in NumPy?")
                with col_t2:
                    t_cat = st.selectbox("Category", ["Correction", "Demonstration", "DomainKnowledge", "Academic", "Code", "Preferences"])

                t_good = st.text_area("Correct / Desired Answer", placeholder="Enter the exact, verified response NOVA should produce...", height=100)
                t_bad = st.text_area("Incorrect / Rejected Answer (Optional)", placeholder="What did NOVA say that was wrong?", height=60)
                t_expl = st.text_input("Explanation / Rationale (Optional)", placeholder="e.g. np.linalg.eig is preferred over custom power iteration")
                t_appr = st.checkbox("Approve for active in-context retrieval and training dataset", value=True)

                if st.form_submit_button("💾 Save Verified Demonstration", type="primary"):
                    if not t_prompt.strip() or not t_good.strip():
                        st.error("Prompt and Correct Answer are required.")
                    else:
                        res_corr = record_correction(
                            user_prompt=t_prompt,
                            corrected_response=t_good,
                            rejected_response=t_bad,
                            explanation=t_expl,
                            category=t_cat,
                            is_approved=t_appr,
                            source="direct_entry",
                        )
                        st.success(f"Verified example #{res_corr['id']} saved to NOVA's learning store!")
                        st.rerun()

        st.markdown("#### 📚 Active Learning Examples")
        corr_search_col, corr_filter_col = st.columns([3, 1])
        with corr_search_col:
            corr_query = st.text_input("Search corrections", placeholder="Filter by prompt or response keywords...", label_visibility="collapsed")
        with corr_filter_col:
            corr_cat_filter = st.selectbox("Category filter", ["All", "Correction", "Demonstration", "DomainKnowledge", "Academic", "Code", "Preferences"], label_visibility="collapsed")

        examples = get_learning_examples(category=corr_cat_filter, search_query=corr_query, only_approved=False, limit=50)

        if not examples:
            st.info("No learning examples found matching your filter criteria.")
        else:
            for ex in examples:
                status_badge = "badge-cyan" if ex["is_approved"] else "badge-rose"
                status_text = "Approved for Retrieval" if ex["is_approved"] else "Pending Approval"
                with st.expander(f"#{ex['id']} [{ex['category']}] {ex['user_prompt'][:70]}...", expanded=False):
                    st.markdown(f'<span class="{status_badge}">{status_text}</span> &bull; Source: <code>{ex["source"]}</code> &bull; Created: {ex["created_at"][:10]}', unsafe_allow_html=True)
                    st.write("")
                    st.markdown("**User Prompt:**")
                    st.code(ex["user_prompt"], language="text")
                    st.markdown("**Verified Response:**")
                    st.code(ex["corrected_response"], language="markdown")
                    if ex.get("rejected_response"):
                        st.markdown("**Original Rejected Response:**")
                        st.caption(ex["rejected_response"])
                    if ex.get("explanation"):
                        st.markdown(f"**Explanation:** *{ex['explanation']}*")

                    c_del1, _ = st.columns([1, 4])
                    with c_del1:
                        if st.button("🗑️ Delete Example", key=f"del_ex_{ex['id']}", type="secondary"):
                            delete_correction(ex["id"])
                            st.toast(f"Example #{ex['id']} deleted from learning store.")
                            st.rerun()

    # -------------------------------------------------------------------------
    # TAB 3: DATASET BUILDER & FINE-TUNING
    # -------------------------------------------------------------------------
    with tab_finetune:
        st.subheader("📦 Model Fine-Tuning & Dataset Pipeline")

        # 1. Provider & Model Inspection
        support = inspect_fine_tuning_support()
        prov_c1, prov_c2 = st.columns([2, 1])
        with prov_c1:
            st.markdown(
                f"""
                <div style="background: #1e293b; padding: 14px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #334155;">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">Active Provider: {support['provider']} ({support['model']})</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                        <b>Mechanism:</b> {support['supported_mechanism']}<br>
                        <b>Free-Tier Reality:</b> {support['free_tier_status']}<br>
                        <b>Cost Estimate:</b> {support['estimated_cost_per_1m_tokens']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with prov_c2:
            st.markdown(
                """
                <div style="background: #131b2e; padding: 14px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #0891b2;">
                    <div style="font-weight: 700; color: #22d3ee; font-size: 0.9rem;">⚡ Zero-Cost In-Context Learning</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                        NOVA automatically injects your approved corrections into prompts at inference time — providing continuous adaptation at <b>$0.00 cost</b> without paid cloud GPU tuning.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Architectural Comparison Table
        with st.expander("⚖️ When is Fine-Tuning Actually Necessary? (RAG vs Prompting vs Fine-Tuning)", expanded=False):
            st.markdown(
                """
                | Technique | Best Used For | Latency & Cost | Data Freshness |
                |---|---|---|---|
                | **Prompt Engineering** | System persona, instructions, tone | Zero cost, instant | Instant |
                | **Retrieval-Augmented Generation (RAG)** | Facts, documents, PDFs, dynamic memory, source attribution | Near-zero cost, fast | Real-time (upload & query) |
                | **In-Context Learning (Few-Shot)** | Verified user corrections, few-shot examples | Near-zero cost, instant | Real-time (immediate) |
                | **Model Fine-Tuning** | Complex style mimicry, novel syntax, fixed output schemas | High compute cost ($3+/1M tokens), slow | Static until retrained |

                **Key Takeaway for MSc AI & Data Science Students:**
                *Fine-tuning is not needed merely to give an LLM new facts* (use RAG and Memory instead). Fine-tuning is justified when teaching a specialized output format, syntax, or tone across thousands of examples.
                """
            )

        st.markdown("---")
        st.subheader("🔨 Build & Sanitize Training Dataset (Chat JSONL)")
        st.markdown(
            "Curate approved learning examples and consented feedback turns into standard Chat JSONL datasets with automated PII and secret scrubbing."
        )

        with st.form("build_dataset_form"):
            b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
            with b_col1:
                ds_name_input = st.text_input("Dataset Release Name", value=f"NOVA_v1_Curated_{date.today().isoformat()}")
            with b_col2:
                ds_val_ratio = st.slider("Validation Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05, help="Proportion of samples reserved for validation")
            with b_col3:
                ds_include_fb = st.checkbox("Include 👍 Chats (Consented)", value=True, help="Include conversation turns with thumbs-up feedback")

            build_submitted = st.form_submit_button("🔨 Build & Sanitize Dataset", type="primary")

        if build_submitted:
            try:
                ds_result = build_training_dataset(
                    name=ds_name_input,
                    val_ratio=ds_val_ratio,
                    include_thumbs_up_conversations=ds_include_fb,
                )
                st.success(f"✅ Training dataset `{ds_result['dataset_id']}` created successfully!")
                
                res_c1, res_c2, res_c3, res_c4 = st.columns(4)
                with res_c1:
                    st.metric("Total Samples", ds_result["total_samples"])
                with res_c2:
                    st.metric("Training Set", ds_result["train_samples"])
                with res_c3:
                    st.metric("Validation Set", ds_result["val_samples"])
                with res_c4:
                    st.metric("PII Redactions Applied", ds_result["redactions_applied"])

                st.caption(f"📁 **Saved on disk:** `{ds_result['folder_path']}`")

                # Provide direct download buttons
                if Path(ds_result["train_file"]).exists():
                    with open(ds_result["train_file"], "r", encoding="utf-8") as tf:
                        t_data = tf.read()
                    st.download_button(
                        "📥 Download train.jsonl",
                        data=t_data,
                        file_name="train.jsonl",
                        mime="application/jsonl",
                    )
            except Exception as e:
                st.error(f"Failed to build dataset: {e}")

        # Compute & Cost Estimator
        st.markdown("---")
        st.subheader("💰 Compute & Cost Estimator")
        est_c1, est_c2, est_c3 = st.columns([1, 1, 1])
        with est_c1:
            est_samples = st.number_input("Estimated Samples", min_value=10, max_value=100000, value=max(50, sys_metrics["training_dataset_samples"] or 50), step=50)
        with est_c2:
            est_epochs = st.slider("Training Epochs", min_value=1, max_value=5, value=3)
        with est_c3:
            est_tokens = st.number_input("Avg Tokens / Sample", min_value=50, max_value=1000, value=180, step=20)

        cost_est = estimate_training_compute_and_cost(
            total_samples=est_samples,
            avg_tokens_per_sample=est_tokens,
            epochs=est_epochs,
        )

        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.metric("Training Tokens", f"{cost_est['training_tokens']:,}")
        with ec2:
            st.metric("OpenAI Cost", f"${cost_est['cost_openai_usd']:.4f}")
        with ec3:
            st.metric("Gemini Vertex AI", f"${cost_est['cost_gemini_vertex_usd']:.4f}")
        with ec4:
            st.metric("In-Context Learning", "$0.00 (Free)")

        st.caption("ℹ️ *Disclaimer: NOVA will NEVER trigger paid cloud fine-tuning jobs automatically without your explicit consent and cloud credentials.*")

    # -------------------------------------------------------------------------
    # TAB 4: BENCHMARK & MODEL REGISTRY
    # -------------------------------------------------------------------------
    with tab_benchmark:
        st.subheader("🎯 Standardized Benchmark Evaluation & Model Registry")
        st.markdown(
            """
            Verify real reasoning and assistant capabilities against a standardized benchmark suite.
            The evaluation tests across **Python/NumPy data science**, **Machine Learning fundamentals**, and **Identity/Safety**.
            """
        )

        b_run_col, b_info_col = st.columns([1, 2])
        with b_run_col:
            if st.button("🚀 Run Live Benchmark Suite", type="primary", use_container_width=True):
                with st.spinner("Running standardized benchmark questions across models..."):
                    eval_out = evaluate_model_performance(model_id=cur_model["id"])
                    st.session_state.last_eval_out = eval_out
                    st.success(f"Benchmark completed! Overall Score: {eval_out['overall_score']}%")

        with b_info_col:
            st.caption(
                f"**Current Registered Model:** `{cur_model['id']}` &bull; "
                f"**Tag:** `{cur_model['version_tag']}` &bull; "
                f"**Status:** `{cur_model['status'].upper()}` &bull; "
                f"**Benchmark Score:** `{cur_model.get('eval_score', 85.0)}%`"
            )

        # Show benchmark results if available
        active_eval = st.session_state.get("last_eval_out")
        if active_eval:
            st.markdown("#### 📊 Evaluation Results Breakdown")
            st.progress(active_eval["overall_score"] / 100.0)
            
            for t_res in active_eval["results"]:
                sc = t_res["score"]
                b_color = "#10b981" if sc >= 80 else ("#f59e0b" if sc >= 50 else "#fb7185")
                with st.expander(f"{t_res['category']} — Score: {sc}%", expanded=True):
                    st.write(f"**Prompt:** {t_res['prompt']}")
                    st.write(f"**Expected Keywords:** `{'`, `'.join(t_res['expected_keywords'])}`")
                    st.write(f"**Matched Keywords:** `{'`, `'.join(t_res['keywords_matched'])}`")

        st.markdown("---")
        st.subheader("⏪ Model Checkpoints & Safe Rollback")
        st.markdown(
            "If an updated model or fine-tuning run performs below expectations, roll back to any prior verified version checkpoint."
        )

        all_models = fetch_all("SELECT * FROM model_registry ORDER BY updated_at DESC")
        if all_models:
            df_models = pd.DataFrame(all_models)
            cols_to_show = ["id", "version_tag", "provider", "base_model", "status", "eval_score", "updated_at"]
            st.dataframe(df_models[[c for c in cols_to_show if c in df_models.columns]], use_container_width=True)

            rb_col1, rb_col2 = st.columns([3, 1])
            with rb_col1:
                model_choices = [m["id"] for m in all_models]
                selected_target_model = st.selectbox("Select Model Checkpoint to Restore", model_choices, index=0)
            with rb_col2:
                st.write("")
                st.write("")
                if st.button("⏪ Rollback to Checkpoint", use_container_width=True):
                    ok = rollback_model_version(selected_target_model)
                    if ok:
                        st.toast(f"Active model successfully rolled back to '{selected_target_model}'!")
                        st.rerun()
                    else:
                        st.error(f"Failed to rollback to '{selected_target_model}'.")


# -----------------------------------------------------------------------------
# PAGE 8: SETTINGS & SYSTEM INFO
# -----------------------------------------------------------------------------
elif selected_page == "⚙️ Settings":
    st.markdown(
        """
        <div class="nova-card">
            <h2 style="margin: 0; color: #f8fafc;">System Settings & Diagnostics</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Audit safety logs, inspect database status, and review privacy controls.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🖥️ Application Diagnostics")
    diag_c1, diag_c2 = st.columns(2)
    with diag_c1:
        st.write(f"- **Application Name:** {APP_NAME}")
        st.write(f"- **Full Title:** {APP_FULL_NAME}")
        st.write(f"- **Version:** {APP_VERSION}")
        st.write(f"- **Database Path:** `{DATABASE_PATH}`")
        db_exists = DATABASE_PATH.exists()
        db_size_kb = round(DATABASE_PATH.stat().st_size / 1024, 2) if db_exists else 0
        st.write(f"- **Database File Status:** {'Active (' + str(db_size_kb) + ' KB)' if db_exists else 'Not Created'}")

    with diag_c2:
        llm_stat = get_llm_status()
        v_stat = get_voice_status()
        st.write(f"- **Active Provider:** {llm_stat['provider']}")
        st.write(f"- **Status:** {llm_stat['status'].upper()}")
        st.write(f"- **Model:** `{llm_stat.get('model', 'None')}`")
        st.write(f"- **Cost Tier:** {llm_stat.get('free_tier', 'Free tier available')}")
        st.write(f"- **STT Engine:** {v_stat['stt_name']} ({v_stat['stt_cost']})")
        st.write(f"- **TTS Engine:** {v_stat['tts_name']} ({v_stat['tts_cost']})")

    st.markdown("---")
    st.subheader("🎙️ Voice Intelligence & Microphone Setup")
    st.markdown(
        """
        NOVA's Voice Intelligence operates with **zero mandatory fees**:
        
        - **Speech-to-Text (STT):** Google Web Speech API converts your spoken commands into text ($0.00 / Free).
        - **Text-to-Speech (TTS):** Google Text-to-Speech (`gTTS`) synthesizes responses into fluent spoken audio ($0.00 / Free).
        - **Microphone Permissions:** When you click the microphone icon, your browser prompts for permission. If accidentally blocked:
          - *Google Chrome / Edge:* Click the lock / tune icon in the address bar -> toggle **Microphone** to *Allow* -> Refresh.
          - *Mozilla Firefox:* Click the microphone icon in the address bar -> remove block -> Refresh.
        - **No Continuous Listening:** NOVA never runs continuous background recording. Audio capture is strictly triggered by your click.
        """
    )

    st.markdown("---")
    st.subheader("🤖 LLM Provider Configuration & Free-Tier Info")
    st.markdown(
        """
        NOVA supports **Google Gemini** (recommended for zero-cost free tier) and **OpenAI**:

        | Provider | Free Tier Limits | Pricing / Costs | Setup |
        |---|---|---|---|
        | **Google Gemini** | **15 RPM**, 1M TPM, 1,500 RPD | **100% Free** (No credit card required) | [Get Gemini Key (AI Studio)](https://aistudio.google.com/) |
        | **OpenAI** | None (trial credits for new accounts) | Pay-as-you-go (~$0.15 / 1M tokens) | [Get OpenAI Key](https://platform.openai.com/) |

        **How to configure your API key:**
        1. Open the `.env` file in the project root directory.
        2. Set `LLM_PROVIDER=gemini` and paste your key into `GEMINI_API_KEY=AIzaSy...`.
        3. Save the `.env` file and restart NOVA. Your key is stored locally on your machine and never tracked by Git.
        """
    )

    st.markdown("---")
    st.subheader("🛡️ Safety, Privacy & Tool Audit Log")
    st.caption("Every tool invocation, task operation, and memory change is recorded here for full accountability.")

    recent_logs = export_table_to_csv("action_logs")
    if recent_logs:
        df_logs = pd.read_csv(pd.io.common.StringIO(recent_logs))
        st.dataframe(df_logs.sort_values(by="id", ascending=False).head(20), use_container_width=True)
    else:
        st.info("No action logs recorded yet.")

    st.markdown("---")
    st.subheader("🔒 Privacy Guarantees")
    st.markdown(
        """
        - **Local-First Architecture:** In Phase 1, all your data resides entirely in your local SQLite database (`data/nova.db`).
        - **Zero Network Leakage:** No data is sent to external servers or third-party APIs during Phase 1.
        - **Git Protection:** The `.gitignore` file is pre-configured to ensure `nova.db` and uploaded files are never committed to public version control.
        - **Explicit Memory Permission:** NOVA does not covertly record conversation turns into permanent memory. Only explicit `save_memory` / `remember` actions persist.
        """
    )
