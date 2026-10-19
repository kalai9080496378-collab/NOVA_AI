"""
====================================================================
NOVA — Personal AI Assistant
Next-generation Organized Virtual Assistant
Professional AI Platform: Streamlit User Interface & Multi-User System
====================================================================
"""

import sys
import os
import re
import json
import io
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path dynamically regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core NOVA Imports
from database.db import init_db, DATABASE_PATH, log_action, fetch_all, fetch_one
from nova_core.auth import (
    ensure_default_users,
    authenticate_user,
    create_user,
    get_user_by_username,
    get_user_by_id,
)
from nova_core.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_FULL_NAME,
    APP_VERSION,
    get_llm_status,
    get_voice_status,
    is_llm_configured,
    get_gemini_api_key,
)
from nova_core.llm_service import validate_gemini_key, _mask_key
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
    export_memories_json,
    export_memories_csv,
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
    get_weekly_plan_summary,
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

# Initialize database schema and default users automatically on startup
init_db()
ensure_default_users()

# -----------------------------------------------------------------------------
# Theme Engine: Dynamic Dark & Light Modes
# -----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"


def get_theme_css(theme: str = "Dark") -> str:
    """Returns CSS corresponding to the active theme (Dark or Light mode)."""
    if theme == "Light":
        return """
        <style>
            /* Global Background and Fonts - Light Mode */
            .stApp {
                background-color: #f8fafc;
                color: #0f172a;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            
            /* Sidebar Styling */
            section[data-testid="stSidebar"] {
                background-color: #ffffff;
                border-right: 1px solid #e2e8f0;
            }
            
            /* Cards and Containers */
            .nova-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            
            .nova-metric-card {
                background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                padding: 18px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            
            .nova-metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: #0284c7;
                margin-top: 4px;
            }
            
            .nova-metric-label {
                font-size: 0.85rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 600;
            }
            
            /* Badges */
            .badge-cyan {
                background-color: rgba(6, 182, 212, 0.12);
                color: #0891b2;
                border: 1px solid #06b6d4;
                padding: 3px 10px;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 600;
                display: inline-block;
            }
            
            .badge-violet {
                background-color: rgba(139, 92, 246, 0.12);
                color: #7c3aed;
                border: 1px solid #8b5cf6;
                padding: 3px 10px;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 600;
                display: inline-block;
            }
            
            .badge-rose {
                background-color: rgba(244, 63, 94, 0.12);
                color: #e11d48;
                border: 1px solid #f43f5e;
                padding: 3px 10px;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 600;
                display: inline-block;
            }

            .badge-amber {
                background-color: rgba(245, 158, 11, 0.12);
                color: #d97706;
                border: 1px solid #f59e0b;
                padding: 3px 10px;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 600;
                display: inline-block;
            }

            /* Headings */
            h1, h2, h3, h4 {
                color: #0f172a !important;
                font-weight: 700;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Global Background and Fonts - Dark Mode */
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
            h1, h2, h3, h4 {
                color: #f8fafc !important;
                font-weight: 700;
            }
        </style>
        """


st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Authentication & User Session Management
# -----------------------------------------------------------------------------
if "current_user" not in st.session_state:
    # Seed session with default demo user if available, or allow manual login
    st.session_state.current_user = get_user_by_username("student")

if st.session_state.current_user is None:
    # Display ChatGPT-inspired Authentication Gate
    auth_col1, auth_col2, auth_col3 = st.columns([1, 2, 1])
    with auth_col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 40px 10px 20px 10px;">
                <div style="font-size: 3.5rem; margin-bottom: 4px;">🌌</div>
                <h1 style="margin: 0; font-size: 2.3rem; color: #22d3ee; font-weight: 800;">{APP_NAME}</h1>
                <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">{APP_TAGLINE}</p>
                <div style="display: inline-block; background: rgba(139, 92, 246, 0.15); border: 1px solid #7c3aed; padding: 4px 14px; border-radius: 9999px; font-size: 0.8rem; color: #c084fc; font-weight: 600; margin-top: 4px;">
                    Personal AI Platform &bull; Multi-User Isolated Environment
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["🔑 Sign In", "✨ Create Account"])

        with tab_login:
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            with st.form("auth_login_form"):
                login_username = st.text_input("Username", placeholder="e.g. student")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_submitted = st.form_submit_button("Sign In to NOVA", type="primary", use_container_width=True)

                if login_submitted:
                    if not login_username.strip() or not login_password:
                        st.error("Please provide both username and password.")
                    else:
                        user = authenticate_user(login_username.strip(), login_password)
                        if user:
                            st.session_state.current_user = user
                            st.session_state.active_thread_id = None
                            st.session_state.chat_messages = []
                            st.session_state.pending_prompt = None
                            st.rerun()
                        else:
                            st.error("Invalid username or password. Please try again.")

            st.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
            st.caption("Testing or evaluating NOVA? Access immediately with the default MSc student profile:")
            if st.button("🚀 One-Click Demo Login (`student`)", use_container_width=True, type="secondary"):
                demo_user = get_user_by_username("student")
                if demo_user:
                    st.session_state.current_user = demo_user
                    st.session_state.active_thread_id = None
                    st.session_state.chat_messages = []
                    st.session_state.pending_prompt = None
                    st.rerun()

        with tab_register:
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            with st.form("auth_register_form"):
                reg_name = st.text_input("Full Name", placeholder="e.g. Alex Morgan")
                reg_username = st.text_input("Choose Username", placeholder="e.g. alex")
                reg_password = st.text_input("Choose Password", type="password", placeholder="At least 6 characters")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                reg_submitted = st.form_submit_button("Create Account & Sign In", type="primary", use_container_width=True)

                if reg_submitted:
                    if not reg_name.strip() or not reg_username.strip() or not reg_password:
                        st.error("All fields are required.")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters long.")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            new_user = create_user(reg_username.strip(), reg_password, full_name=reg_name.strip())
                            st.session_state.current_user = new_user
                            st.session_state.active_thread_id = None
                            st.session_state.chat_messages = []
                            st.session_state.pending_prompt = None
                            st.rerun()
                        except ValueError as ve:
                            st.error(str(ve))

    st.stop()


# -----------------------------------------------------------------------------
# 2b. Authenticated Session Context & State
# -----------------------------------------------------------------------------
current_user = st.session_state.current_user
active_user_id = current_user["id"]

# Verify active_thread_id belongs to the current user
active_th = None
if st.session_state.get("active_thread_id"):
    active_th = get_thread(st.session_state.active_thread_id, user_id=active_user_id)

if not active_th:
    user_threads = get_all_threads(include_archived=False, user_id=active_user_id)
    empty_thread = None
    for th in user_threads:
        if not get_thread_messages(th["id"], limit=1, user_id=active_user_id):
            empty_thread = th
            break
    if empty_thread:
        st.session_state.active_thread_id = empty_thread["id"]
    else:
        new_th = create_thread(title="New Conversation", user_id=active_user_id)
        st.session_state.active_thread_id = new_th["id"]

st.session_state.conversation_id = st.session_state.active_thread_id

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

# Unified Message Composer & Voice Transcription State
if "chat_composer_input" not in st.session_state:
    st.session_state.chat_composer_input = ""
if "chat_stt_audio_hash" not in st.session_state:
    st.session_state.chat_stt_audio_hash = None
if "voice_just_transcribed" not in st.session_state:
    st.session_state.voice_just_transcribed = False
if "voice_engine_used" not in st.session_state:
    st.session_state.voice_engine_used = "auto"
if "voice_tts_enabled" not in st.session_state:
    st.session_state.voice_tts_enabled = True
if "voice_autoplay" not in st.session_state:
    st.session_state.voice_autoplay = False
if "voice_consent" not in st.session_state:
    st.session_state.voice_consent = True


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Branding (User-Specific Threads)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 6px 0 12px 0;">
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

    # User Profile Card & Sign Out
    st.markdown(
        f"""
        <div style="background: #131b2e; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px;">
            <div style="font-weight: 700; color: #f8fafc; font-size: 0.88rem;">👤 {current_user['full_name']}</div>
            <div style="color: #94a3b8; font-size: 0.75rem;">@{current_user['username']} &bull; <span style="color: #22d3ee;">Active Account</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_out, col_sw = st.columns([1, 1])
    with col_out:
        if st.button("🚪 Sign Out", use_container_width=True, key="sb_btn_signout"):
            st.session_state.current_user = None
            st.session_state.active_thread_id = None
            st.session_state.chat_messages = []
            st.session_state.pending_prompt = None
            st.rerun()
    with col_sw:
        if st.button("🔄 Switch", use_container_width=True, key="sb_btn_switch_acc", help="Switch account"):
            st.session_state.current_user = None
            st.session_state.active_thread_id = None
            st.session_state.chat_messages = []
            st.session_state.pending_prompt = None
            st.rerun()

    # Theme quick toggle in sidebar
    cur_th = st.session_state.get("theme", "Dark")
    theme_btn_text = "☀️ Switch to Light Mode" if cur_th == "Dark" else "🌙 Switch to Dark Mode"
    if st.button(theme_btn_text, use_container_width=True, key="sb_theme_toggle_btn"):
        st.session_state.theme = "Light" if cur_th == "Dark" else "Dark"
        st.rerun()

    llm_info = get_llm_status()
    if llm_info["status"] == "connected":
        st.markdown(f'<span class="badge-violet">⚡ {llm_info["provider"]} Active ({llm_info.get("model", "")})</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-cyan">⚙️ Rule-Based Mode</span>', unsafe_allow_html=True)

    st.write("")

    navigation_options = [
        "🏠 Home",
        "💬 AI Chat",
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

        # + New Chat Button (Isolated to active_user_id)
        if st.button("➕ New Chat", use_container_width=True, type="primary", key="sb_new_chat_btn"):
            empty_th = None
            for eth in active_threads:
                if not get_thread_messages(eth["id"], limit=1, user_id=active_user_id):
                    empty_th = eth
                    break
            if empty_th:
                new_th = empty_th
            else:
                new_th = create_thread(title="New Conversation", user_id=active_user_id)
            st.session_state.active_thread_id = new_th["id"]
            st.session_state.conversation_id = new_th["id"]
            st.session_state["chat_composer_input"] = ""
            st.session_state.pending_prompt = None
            st.rerun()

        # Search conversations
        chat_search = st.text_input("Search chats", placeholder="🔍 Filter chats...", label_visibility="collapsed", key="sb_search_chats")

        # Load active and archived threads strictly for active_user_id
        active_threads = get_all_threads(include_archived=False, search_query=chat_search, user_id=active_user_id)
        archived_threads = get_all_threads(include_archived=True, search_query=chat_search, user_id=active_user_id)
        archived_only = [t for t in archived_threads if t.get("is_archived")]

        grouped = group_threads_by_date(active_threads)

        if not active_threads:
            st.caption("No conversations yet. Click **➕ New Chat** to begin.")
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
                                        update_thread(th["id"], title=new_t.strip(), user_id=active_user_id)
                                        st.rerun()

                            pin_txt = "Unpin" if th.get("is_pinned") else "📌 Pin"
                            if st.button(pin_txt, key=f"sb_pin_{th['id']}", use_container_width=True):
                                update_thread(th["id"], is_pinned=not th.get("is_pinned"), user_id=active_user_id)
                                st.rerun()

                            if st.button("📦 Archive", key=f"sb_arch_{th['id']}", use_container_width=True):
                                update_thread(th["id"], is_archived=True, user_id=active_user_id)
                                if st.session_state.active_thread_id == th["id"]:
                                    new_th = create_thread(title="New Conversation", user_id=active_user_id)
                                    st.session_state.active_thread_id = new_th["id"]
                                    st.session_state.conversation_id = new_th["id"]
                                    st.session_state["chat_composer_input"] = ""
                                st.rerun()

                            if st.button("🗑️ Delete", key=f"sb_del_{th['id']}", use_container_width=True):
                                delete_thread(th["id"], user_id=active_user_id)
                                if st.session_state.active_thread_id == th["id"]:
                                    new_th = create_thread(title="New Conversation", user_id=active_user_id)
                                    st.session_state.active_thread_id = new_th["id"]
                                    st.session_state.conversation_id = new_th["id"]
                                    st.session_state["chat_composer_input"] = ""
                                st.rerun()

        if archived_only:
            with st.expander(f"📦 Archived ({len(archived_only)})", expanded=False):
                for ath in archived_only:
                    st.caption(f"📁 **{ath['title']}**")
                    c_un, c_del = st.columns(2)
                    with c_un:
                        if st.button("Restore", key=f"unarch_{ath['id']}", use_container_width=True):
                            update_thread(ath["id"], is_archived=False, user_id=active_user_id)
                            st.session_state.active_thread_id = ath["id"]
                            st.session_state.conversation_id = ath["id"]
                            st.rerun()
                    with c_del:
                        if st.button("Delete", key=f"del_arch_{ath['id']}", use_container_width=True):
                            delete_thread(ath["id"], user_id=active_user_id)
                            st.rerun()

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            <b>NOVA Architecture:</b> Multi-User AI Platform<br>
            <b>Storage:</b> SQLite Persistent<br>
            <b>Active DB:</b> <code>nova.db</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Global Active Reminder Engine & Delivery Check
# -----------------------------------------------------------------------------
due_now_reminders = check_and_trigger_due_reminders(user_id=active_user_id)
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
            <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">Welcome back, {current_user['full_name']}! 🌌</h1>
            <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1rem;">
                {APP_FULL_NAME} — {APP_TAGLINE}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Real Metrics retrieved from SQLite for active user
    stats = get_task_statistics(user_id=active_user_id)

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
                        create_task(q_title, priority=q_prio, due_date=due_str, user_id=active_user_id)
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
                        save_memory(m_content, category=m_cat, user_id=active_user_id)
                        st.success("Memory saved successfully with your permission!")
                        st.session_state.quick_mem = False
                        st.rerun()
                    else:
                        st.error("Content cannot be empty.")

    with col_right:
        st.subheader("📜 Recent Activity")
        recent_acts = get_activity_records(limit=6, user_id=active_user_id)
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
    # 1. Resolve Active Conversation Thread for current user
    active_th = get_thread(st.session_state.active_thread_id, user_id=active_user_id)
    if not active_th:
        existing = get_all_threads(include_archived=False, user_id=active_user_id)
        empty_th = None
        for eth in existing:
            if not get_thread_messages(eth["id"], limit=1, user_id=active_user_id):
                empty_th = eth
                break
        if empty_th:
            active_th = empty_th
        else:
            active_th = create_thread(title="New Conversation", user_id=active_user_id)
        st.session_state.active_thread_id = active_th["id"]

    st.session_state.conversation_id = st.session_state.active_thread_id

    # 2. Minimal Chat Header
    hdr_c1, hdr_c2, hdr_c3 = st.columns([5, 1.3, 1.1], vertical_alignment="center")
    with hdr_c1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <h3 style="margin: 0; color: #f8fafc; font-size: 1.35rem; font-weight: 600;">{active_th['title']}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hdr_c2:
        if st.button("➕ New Chat", key="chat_page_new_chat_btn", use_container_width=True):
            user_threads = get_all_threads(include_archived=False, user_id=active_user_id)
            empty_th = None
            for eth in user_threads:
                if not get_thread_messages(eth["id"], limit=1, user_id=active_user_id):
                    empty_th = eth
                    break
            if empty_th:
                nth = empty_th
            else:
                nth = create_thread(title="New Conversation", user_id=active_user_id)
            st.session_state.active_thread_id = nth["id"]
            st.session_state.conversation_id = nth["id"]
            st.session_state["chat_composer_input"] = ""
            st.session_state.pop("pending_chat_prompt", None)
            st.rerun()
    with hdr_c3:
        with st.popover("⚙️ Manage", help="Thread options"):
            renamed = st.text_input("Title", value=active_th["title"], key=f"th_ren_{active_th['id']}")
            if st.button("Rename", key="btn_th_rename_save", use_container_width=True):
                if renamed.strip():
                    update_thread(active_th["id"], title=renamed.strip(), user_id=active_user_id)
                    st.rerun()
            if st.button("🗑️ Delete", key="btn_th_del_chat", type="secondary", use_container_width=True):
                delete_thread(active_th["id"], user_id=active_user_id)
                nth = create_thread(title="New Conversation", user_id=active_user_id)
                st.session_state.active_thread_id = nth["id"]
                st.session_state.conversation_id = nth["id"]
                st.session_state["chat_composer_input"] = ""
                st.session_state.pop("pending_chat_prompt", None)
                st.rerun()

    # 3. Check Real AI status
    llm_info = get_llm_status()
    if llm_info.get("status") != "connected":
        st.warning(
            "🔑 **Gemini API Key Required:** Please configure `GEMINI_API_KEY` in your `.env` file or Streamlit Cloud Secrets to enable real AI conversational responses. "
            "(Get a free key at [aistudio.google.com](https://aistudio.google.com)). Local commands like `add task: ...` and `remember: ...` work without an API key."
        )

    # 4. Message History Rendering
    thread_messages = get_thread_messages(active_th["id"], user_id=active_user_id)

    if not thread_messages:
        st.markdown(
            """
            <div style="text-align: center; padding: 48px 16px 24px 16px;">
                <div style="font-size: 2.8rem; margin-bottom: 8px;">🌌</div>
                <h2 style="color: #f8fafc; margin-bottom: 6px; font-weight: 600; font-size: 1.5rem;">Hi! I'm NOVA. How can I help you today?</h2>
                <p style="color: #94a3b8; font-size: 0.95rem; max-width: 480px; margin: 0 auto;">
                    Chat casually, ask questions, explore ideas, or check latest news.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in thread_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 5. Fixed Bottom Input Bar: Text box + Mic button + Send button
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    if "chat_composer_input" not in st.session_state:
        st.session_state["chat_composer_input"] = ""

    def handle_submit():
        val = st.session_state.get("chat_composer_input", "").strip()
        if val:
            st.session_state["pending_chat_prompt"] = val
            st.session_state["chat_composer_input"] = ""

    comp_c1, comp_c2, comp_c3 = st.columns([5.5, 1.3, 1.0], vertical_alignment="bottom")

    with comp_c1:
        st.text_input(
            "Message NOVA...",
            key="chat_composer_input",
            on_change=handle_submit,
            placeholder="Message NOVA or speak into the mic...",
            label_visibility="collapsed",
        )

    with comp_c2:
        mic_audio = st.audio_input(
            "Speak",
            key="chat_mic_widget",
            label_visibility="collapsed",
            help="Click to speak. Converts your voice to text directly in the message box.",
        )

    with comp_c3:
        st.button(
            "🚀 Send",
            key="chat_send_btn",
            on_click=handle_submit,
            type="primary",
            use_container_width=True,
        )

    # Convert voice input into text in the message input box
    if mic_audio is not None:
        raw_audio_bytes = mic_audio.getvalue()
        audio_md5 = hashlib.md5(raw_audio_bytes).hexdigest()

        if st.session_state.get("chat_stt_audio_hash") != audio_md5:
            st.session_state["chat_stt_audio_hash"] = audio_md5
            if len(raw_audio_bytes) < 200:
                st.toast("⚠️ Audio was empty or too quiet. Please speak clearly into your microphone.", icon="🎙️")
            else:
                with st.spinner("Listening & transcribing..."):
                    stt_result = transcribe_audio(
                        raw_audio_bytes,
                        audio_format="wav",
                        language=st.session_state.get("voice_lang_code", "en-IN"),
                        engine="auto",
                        consent_external=True,
                    )
                if stt_result.get("success") and stt_result.get("text"):
                    st.session_state["chat_composer_input"] = stt_result["text"]
                    st.toast("Voice recognized! You can review or send your message.", icon="✨")
                    st.rerun()
                else:
                    err_txt = stt_result.get("error") or "Could not recognize speech."
                    st.toast(f"Speech recognition notice: {err_txt}", icon="🎙️")

    # 6. Stream Assistant Response in Unified Pipeline
    prompt_to_run = st.session_state.pop("pending_chat_prompt", None)
    if prompt_to_run:
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        assistant = NovaAssistant()
        is_tool = assistant._is_explicit_tool_command(prompt_to_run)

        with st.chat_message("assistant"):
            if not is_llm_configured() and not is_tool:
                config_err = (
                    "⚠️ **Gemini API Key Required**\n\n"
                    "To chat with NOVA and receive real AI answers, please configure your free Gemini API key:\n\n"
                    "1. Get a free key at [aistudio.google.com](https://aistudio.google.com) (no credit card required).\n"
                    "2. Add `GEMINI_API_KEY = \"AIza...\"` to your `.env` file or Streamlit Cloud Secrets.\n\n"
                    "*Local commands like `add task: ...`, `show tasks`, and `remember: ...` work without an API key.*"
                )
                st.markdown(config_err)
                add_message(active_th["id"], "user", prompt_to_run, user_id=active_user_id)
                add_message(active_th["id"], "assistant", config_err, metadata={"mode": "configuration_error"}, user_id=active_user_id)
            else:
                with st.spinner("NOVA is thinking..."):
                    try:
                        stream_gen = assistant.process_message_stream(
                            prompt_to_run,
                            conversation_id=active_th["id"],
                            user_id=active_user_id,
                            web_search="auto",
                            deep_research=False,
                        )
                        st.write_stream(stream_gen)
                    except Exception as e:
                        st.error(f"Sorry, I encountered an issue generating an answer: {e}")
        st.rerun()



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
                cleaned_new = str(new_content or "").strip()
                if cleaned_new:
                    try:
                        save_memory(cleaned_new, category=new_cat, user_id=active_user_id)
                        st.success("Memory saved successfully! NOVA will use this when you ask for recall.")
                        st.rerun()
                    except ValueError as ve:
                        st.warning(str(ve))
                else:
                    st.error("Please enter memory content.")

    with tab_view:
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 1, 1])
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
            with st.popover("📥 Export", help="Export your memories"):
                m_json = export_memories_json(user_id=active_user_id)
                m_csv = export_memories_csv(user_id=active_user_id)
                st.download_button("JSON (.json)", data=m_json, file_name="nova_memories.json", mime="application/json", use_container_width=True)
                st.download_button("CSV (.csv)", data=m_csv, file_name="nova_memories.csv", mime="text/csv", use_container_width=True)
        with f_col4:
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
                    num = clear_all_memories(user_id=active_user_id)
                    st.success(f"Cleared {num} memories.")
                    st.session_state.confirm_clear_memory = False
                    st.rerun()
            with c_no:
                if st.button("Cancel"):
                    st.session_state.confirm_clear_memory = False
                    st.rerun()

        memories = get_all_memories(
            category=None if cat_filter == "All" else cat_filter,
            search_query=search_query if (search_query and search_query.strip()) else None,
            user_id=active_user_id,
        )

        mem_count = len(memories) if isinstance(memories, list) else 0
        st.write(f"**Total Records:** {mem_count}")

        if not memories or not isinstance(memories, list):
            st.info("No memories found matching the current criteria. Use the 'Save New Memory' tab to add your first record!")
        else:
            for mem in memories:
                if not isinstance(mem, dict):
                    continue
                m_id = mem.get("id", "")
                m_cat = str(mem.get("category", "General") or "General")
                m_content = str(mem.get("content", "") or "")
                m_created = str(mem.get("created_at", "") or "")
                m_updated = str(mem.get("updated_at", "") or "")

                with st.expander(f"#{m_id} [{m_cat}] — {m_content[:60]}..."):
                    st.write(f"**Content:** {m_content}")
                    st.caption(f"Created: {m_created} | Last Updated: {m_updated}")
                    
                    e_col1, e_col2 = st.columns([1, 1])
                    with e_col1:
                        # Edit option
                        edit_key = f"edit_mem_{m_id}"
                        if st.button(f"✏️ Edit #{m_id}", key=f"btn_edit_{m_id}"):
                            st.session_state[edit_key] = True

                        if st.session_state.get(edit_key, False):
                            edit_text = st.text_area("Edit Content", value=m_content, key=f"text_{m_id}")
                            edit_cat = st.text_input("Edit Category", value=m_cat, key=f"cat_{m_id}")
                            if st.button("Save Changes", key=f"save_edit_{m_id}"):
                                try:
                                    update_memory(m_id, content=edit_text, category=edit_cat, user_id=active_user_id)
                                    st.success("Updated!")
                                    st.session_state[edit_key] = False
                                    st.rerun()
                                except ValueError as ve:
                                    st.warning(str(ve))

                    with e_col2:
                        # Delete option with confirmation
                        if st.button(f"🗑️ Delete #{m_id}", key=f"del_{m_id}", type="secondary"):
                            delete_memory(m_id, user_id=active_user_id)
                            st.success(f"Deleted memory #{m_id}.")
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

    t_stats = get_task_statistics(user_id=active_user_id)
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
                            user_id=active_user_id,
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
            user_id=active_user_id,
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

    rem_stats = get_reminder_statistics(user_id=active_user_id)
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
        pending_tasks = get_tasks(status="pending", user_id=active_user_id)
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
                            user_id=active_user_id,
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

        reminders_list = get_reminders(status=status_filter, user_id=active_user_id)
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

    tab_daily, tab_weekly, tab_suggestions, tab_history = st.tabs([
        "📅 Daily Time-Blocking",
        "📆 Weekly Planning View",
        "🎯 Smart Suggestions",
        "📚 Saved Plans History",
    ])

    with tab_daily:
        st.subheader("Generate Suggested Daily Schedule")
        st.caption("Algorithmic time-blocking that groups high-priority deliverables and injects rejuvenating breaks.")
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            plan_date = st.date_input("Target Plan Date", value=date.today(), key="planner_target_date")
        with c_p2:
            start_hour_input = st.selectbox("Work Start Time", ["08:00", "09:00", "10:00", "11:00", "13:00", "14:00"], index=1)
        with c_p3:
            avail_hours = st.slider("Available Work Hours", min_value=1.0, max_value=12.0, value=6.0, step=0.5)

        if st.button("⚡ Generate Time-Blocked Daily Plan", type="primary", use_container_width=True):
            with st.spinner("Analyzing tasks and constructing balanced schedule..."):
                generated_plan = generate_daily_plan(
                    target_date=plan_date.isoformat(),
                    available_hours=avail_hours,
                    start_time=start_hour_input,
                    user_id=active_user_id,
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
            for b_idx, block in enumerate(g_plan["schedule"]):
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
                    b_col_info, b_col_act = st.columns([5, 1])
                    b_task_id = block.get("task_id") or block.get("id", "")
                    b_title = block.get("title", "")
                    with b_col_info:
                        st.markdown(
                            f"""
                            <div style="background: #0f172a; border-left: 4px solid #38bdf8; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; margin-bottom: 4px;">
                                <div>
                                    <b>{block.get('start_time', '')} – {block.get('end_time', '')}</b> ({block.get('duration_minutes', 0)} min):
                                    <span style="font-size: 1rem; color: #f8fafc; margin-left: 6px;"><b>#{b_task_id} {b_title}</b></span>
                                    <span class="{p_badge}" style="float: right;">{block.get('priority', 'Medium')}</span>
                                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                                        Category: {block.get('category', 'General')} | Due: {block.get('due_date') or 'None'}
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with b_col_act:
                        if b_task_id:
                            if st.button("✅ Done", key=f"plan_done_btn_{b_idx}_{b_task_id}", help="Mark task as complete"):
                                complete_task(b_task_id)
                                st.success(f"Task #{b_task_id} completed!")
                                st.rerun()

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
                    user_id=active_user_id,
                )
                st.success(f"Plan for {g_plan['date']} permanently saved to SQLite!")
                st.rerun()

    with tab_weekly:
        st.subheader("📆 7-Day Weekly Workload & Schedule")
        st.caption("Inspect tasks scheduled or due across the week (Monday – Sunday) to balance your workload.")

        cur_today = date.today()
        cur_monday = cur_today - timedelta(days=cur_today.weekday())
        selected_week_start = st.date_input("Week Starting On (Monday)", value=cur_monday, key="weekly_view_picker")

        weekly_data = get_weekly_plan_summary(start_date=selected_week_start.isoformat(), user_id=active_user_id)

        wm1, wm2, wm3, wm4 = st.columns(4)
        with wm1:
            st.metric("Total Tasks This Week", weekly_data["total_tasks"])
        with wm2:
            st.metric("Pending Tasks", weekly_data["pending_tasks"])
        with wm3:
            st.metric("Completed Tasks", weekly_data["completed_tasks"])
        with wm4:
            st.metric("High-Priority Items", weekly_data["high_priority_tasks"])

        st.write("")
        st.write("#### 🗓️ Day-by-Day Breakdown")

        w_cols = st.columns(7)
        for d_idx, day_info in enumerate(weekly_data["days"]):
            with w_cols[d_idx]:
                is_today = (day_info["date"] == cur_today.isoformat())
                border_color = "#06b6d4" if is_today else "#334155"
                st.markdown(
                    f"""
                    <div style="background: #131b2e; border: 2px solid {border_color}; border-radius: 8px; padding: 10px; margin-bottom: 8px; text-align: center;">
                        <div style="font-weight: 700; color: {'#38bdf8' if is_today else '#f8fafc'}; font-size: 0.95rem;">{day_info['day'][:3]}</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">{day_info['date'][-5:]}</div>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #22d3ee; margin-top: 4px;">{day_info['total_tasks']} tasks</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if not day_info["tasks"]:
                    st.caption("<div style='text-align: center; color: #64748b;'>No tasks</div>", unsafe_allow_html=True)
                else:
                    for t_item in day_info["tasks"]:
                        is_t_done = (t_item.get("status") == "completed")
                        p_badge_cls = "badge-rose" if t_item.get("priority") == "High" else "badge-cyan"
                        t_id = t_item.get("id") or t_item.get("task_id", "")
                        t_title = t_item.get("title", "")
                        st.markdown(
                            f"""
                            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 6px 8px; margin-bottom: 6px; font-size: 0.78rem;">
                                <div style="font-weight: 600; color: {'#94a3b8' if is_t_done else '#f8fafc'}; text-decoration: {'line-through' if is_t_done else 'none'};">
                                    #{t_id} {t_title[:22]}
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-top: 3px;">
                                    <span class="{p_badge_cls}" style="font-size: 0.65rem; padding: 1px 6px;">{t_item.get('priority', 'Medium')}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if not is_t_done and t_id:
                            if st.button("✓ Done", key=f"wk_done_{t_id}_{d_idx}", use_container_width=True):
                                complete_task(t_id)
                                st.rerun()

    with tab_suggestions:
        st.subheader("What Should I Work on Next?")
        st.caption("Algorithmic recommendations prioritizing overdue deadlines, high-priority deliverables, and upcoming tasks.")
        
        suggestions_res = get_smart_task_suggestions(limit=5, user_id=active_user_id)
        if isinstance(suggestions_res, dict):
            suggestions = suggestions_res.get("suggestions", [])
        elif isinstance(suggestions_res, list):
            suggestions = suggestions_res
        else:
            suggestions = []

        if not suggestions:
            st.success("🎉 You have no pending tasks! Enjoy your free time or add new tasks in the Task Manager.")
        else:
            for idx, sugg in enumerate(suggestions, 1):
                sugg_id = sugg.get("id") or sugg.get("task_id") or (sugg.get("task", {}).get("id") if isinstance(sugg.get("task"), dict) else None)
                sugg_title = sugg.get("title") or (sugg.get("task", {}).get("title") if isinstance(sugg.get("task"), dict) else "") or "Task"
                p_badge = "badge-rose" if sugg.get("priority") == "High" else ("badge-amber" if sugg.get("priority") == "Medium" else "badge-cyan")
                st.markdown(
                    f"""
                    <div style="background: #131b2e; border: 1px solid #334155; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">
                                    #{idx} Recommendation: {sugg_title}
                                </span>
                                <span style="background: #1e3a8a; color: #93c5fd; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; margin-left: 8px;">
                                    {sugg.get('category', 'General')}
                                </span>
                                <div style="font-size: 0.88rem; color: #f1f5f9; margin-top: 6px; font-weight: 500;">
                                    💡 <b>Why this is recommended:</b> {sugg.get('rationale', sugg.get('recommendation_reason', ''))}
                                </div>
                                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px;">
                                    Due Date: <b>{sugg.get('due_date') or 'No set date'}</b> {('at ' + str(sugg.get('due_time'))) if sugg.get('due_time') else ''}
                                    {(' | Est. Duration: <b>' + str(sugg.get('estimated_duration')) + ' mins</b>') if sugg.get('estimated_duration') else ''}
                                    | Urgency Score: <code>{sugg.get('urgency_score', sugg.get('score', 0))}</code>
                                </div>
                            </div>
                            <span class="{p_badge}">{sugg.get('priority', 'Medium')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if sugg_id:
                    if st.button(f"✅ Mark #{sugg_id} Completed", key=f"sugg_done_{sugg_id}"):
                        complete_task(sugg_id)
                        st.success(f"Completed '{sugg_title}'!")
                        st.rerun()

            st.caption("🤖 *Note: Recommendations are algorithmically ranked using deadline proximity and priority weighting.*")

    with tab_history:
        st.subheader("Saved Daily Plans History")
        h_date = st.date_input("Lookup Date", value=date.today(), key="hist_plan_date")
        saved_record = get_daily_plan(h_date.isoformat(), user_id=active_user_id)
        if saved_record:
            st.success(f"Found saved plan for {saved_record['plan_date']} (Created: {saved_record['created_at'][:16]})")
            st.write(f"**Total Allocated Time:** {saved_record['total_allocated_minutes']} minutes")
            if saved_record.get("notes"):
                st.info(f"**Notes:** {saved_record['notes']}")
            st.write("#### Schedule Blocks")
            for sb in saved_record.get("schedule", []):
                st.markdown(f"- **{sb.get('start_time')} – {sb.get('end_time')}** ({sb.get('duration_minutes', 0)} min): {sb.get('title')}")
        else:
            st.info(f"No saved plan found for {h_date.isoformat()}. Generate and save a plan using the 'Daily Time-Blocking' tab.")


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
                            user_id=active_user_id,
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
                        user_id=active_user_id,
                    )
                    st.success(f"Logged {log_dur} minutes for '{log_sub}' in SQLite!")
                    st.rerun()
                else:
                    st.error("Subject is required.")

        st.subheader("📚 Study Hours Summary")
        summary = get_study_hours_summary(days=30, user_id=active_user_id)
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
                        new_doc = save_and_process_document(file_bytes, uploaded_file.name, user_id=active_user_id)
                    st.success(f"Successfully processed **{new_doc['filename']}** ({new_doc['page_count']} pages, {new_doc['word_count']} words)!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process document: {str(e)}")

    # Document Selection & Listing (Filtered for active_user_id)
    stored_docs = get_documents(user_id=active_user_id)

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
            st.warning("⚠️ Are you sure you want to permanently delete ALL your indexed documents and chunks?")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("Yes, Delete All", type="primary"):
                    num = delete_all_documents(user_id=active_user_id)
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

    all_tasks = get_tasks(status="all", user_id=active_user_id)
    task_stats = get_task_statistics(user_id=active_user_id)
    rem_stats = get_reminder_statistics(user_id=active_user_id)
    daily_summary = get_daily_activity_summary(days=14, user_id=active_user_id)

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
        tasks_csv = export_table_to_csv("tasks", user_id=active_user_id)
        st.download_button(
            "Download Tasks CSV",
            data=tasks_csv,
            file_name="nova_tasks.csv",
            mime="text/csv",
            disabled=(len(tasks_csv) == 0),
            use_container_width=True,
        )
    with exp_col2:
        reminders_csv = export_table_to_csv("reminders", user_id=active_user_id)
        st.download_button(
            "Download Reminders CSV",
            data=reminders_csv,
            file_name="nova_reminders.csv",
            mime="text/csv",
            disabled=(len(reminders_csv) == 0),
            use_container_width=True,
        )
    with exp_col3:
        memories_csv = export_table_to_csv("memories", user_id=active_user_id)
        st.download_button(
            "Download Memories CSV",
            data=memories_csv,
            file_name="nova_memories.csv",
            mime="text/csv",
            disabled=(len(memories_csv) == 0),
            use_container_width=True,
        )
    with exp_col4:
        activities_csv = export_table_to_csv("activity_records", user_id=active_user_id)
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
            <h2 style="margin: 0; color: #f8fafc;">System Settings & Preferences</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Customize appearance, configure voice & AI providers, manage personal memory, and inspect safety audit logs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    set_tab1, set_tab2, set_tab3, set_tab4, set_tab5 = st.tabs([
        "🎨 Appearance & Theme",
        "🎙️ Voice & Audio",
        "🤖 AI Models & Search",
        "🧠 Memory & Data Ownership",
        "🛡️ Account & Safety Audit",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: THEME & APPEARANCE
    # -------------------------------------------------------------------------
    with set_tab1:
        st.subheader("🎨 Appearance & Interface Customization")
        st.caption("Switch between Dark and Light mode themes across all NOVA screens.")

        cur_theme = st.session_state.get("theme", "Dark")
        theme_choice = st.radio(
            "Select Global Theme",
            options=["Dark Mode 🌙 (Navy / Cyan / Violet)", "Light Mode ☀️ (Clean Slate / White)"],
            index=0 if cur_theme == "Dark" else 1,
            horizontal=True,
            key="settings_theme_radio",
        )
        new_theme = "Dark" if "Dark" in theme_choice else "Light"
        if new_theme != cur_theme:
            st.session_state.theme = new_theme
            st.rerun()

        st.write("")
        st.write("#### 👁️ Interface Preview")
        preview_c1, preview_c2, preview_c3 = st.columns(3)
        with preview_c1:
            st.markdown(
                """
                <div class="nova-card">
                    <b>Primary Card Container</b>
                    <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">Standard card layout with rounded borders and responsive padding.</p>
                    <span class="badge-cyan">Active Pill</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with preview_c2:
            st.markdown(
                """
                <div class="nova-metric-card">
                    <div class="nova-metric-label">Sample Metric</div>
                    <div class="nova-metric-value">98.5%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with preview_c3:
            st.markdown(
                """
                <div class="nova-card">
                    <b>Color Accents</b><br>
                    <span class="badge-violet">Violet</span>
                    <span class="badge-rose">Rose</span>
                    <span class="badge-amber">Amber</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # TAB 2: VOICE & AUDIO PREFERENCES
    # -------------------------------------------------------------------------
    with set_tab2:
        st.subheader("🎙️ Voice Intelligence & Audio Settings")
        st.caption("Configure speech recognition dialect, recognition engine, and speech synthesis.")

        v_c1, v_c2 = st.columns(2)
        with v_c1:
            current_lang = st.session_state.get("voice_lang_code", "en-IN")
            lang_options = [
                "English (India) [en-IN]",
                "Tamil / தமிழ் [ta-IN]",
                "Tanglish (Tamil + English Mixed)",
                "English (US) [en-US]",
            ]
            lang_code_map = {
                "English (India) [en-IN]": "en-IN",
                "Tamil / தமிழ் [ta-IN]": "ta-IN",
                "Tanglish (Tamil + English Mixed)": "tanglish",
                "English (US) [en-US]": "en-US",
            }
            code_to_lang_idx = {"en-IN": 0, "ta-IN": 1, "tanglish": 2, "en-US": 3}
            sel_lang_label = st.selectbox(
                "🗣️ Default Speech Language",
                options=lang_options,
                index=code_to_lang_idx.get(current_lang, 0),
                key="settings_voice_lang",
            )
            st.session_state.voice_lang_code = lang_code_map[sel_lang_label]

            current_engine = st.session_state.get("voice_engine", "auto")
            eng_options = [
                "Auto (Gemini 1.5 Flash if available, else Google Web Speech)",
                "Google Web Speech ($0.00 / Free)",
                "Google Gemini 1.5 Flash (Multimodal Audio)",
            ]
            eng_map = {
                "Auto (Gemini 1.5 Flash if available, else Google Web Speech)": "auto",
                "Google Web Speech ($0.00 / Free)": "google",
                "Google Gemini 1.5 Flash (Multimodal Audio)": "gemini",
            }
            code_to_eng_idx = {"auto": 0, "google": 1, "gemini": 2}
            sel_eng_label = st.selectbox(
                "⚙️ Speech Recognition Engine",
                options=eng_options,
                index=code_to_eng_idx.get(current_engine, 0),
                key="settings_voice_engine",
            )
            st.session_state.voice_engine = eng_map[sel_eng_label]

        with v_c2:
            st.write("**Audio Playback Controls**")
            set_tts = st.checkbox(
                "🔊 Enable Text-to-Speech (TTS) Voice Responses",
                value=st.session_state.get("voice_tts_enabled", True),
                key="settings_tts_toggle",
            )
            st.session_state.voice_tts_enabled = set_tts

            set_autoplay = st.checkbox(
                "▶️ Automatically Auto-play Spoken Responses",
                value=st.session_state.get("voice_autoplay", False),
                key="settings_autoplay_toggle",
            )
            st.session_state.voice_autoplay = set_autoplay

            set_consent = st.checkbox(
                "🔒 External Audio Processing Consent",
                value=st.session_state.get("voice_consent", True),
                key="settings_consent_toggle",
            )
            st.session_state.voice_consent = set_consent

        st.markdown("---")
        st.markdown("#### 🎧 Microphone Permissions & Diagnostics")
        v_diag = get_voice_status()
        st.write(f"- **STT Engine Status:** {v_diag['stt_name']} ({v_diag['stt_cost']})")
        st.write(f"- **TTS Engine Status:** {v_diag['tts_name']} ({v_diag['tts_cost']})")
        st.info(
            "💡 **Microphone Permission Guide:**\n"
            "- When clicking the microphone in AI Chat, your browser asks for permission.\n"
            "- *Chrome / Edge:* Click the lock/settings icon in the address bar -> toggle **Microphone** to *Allow* -> Refresh.\n"
            "- *Firefox:* Click the microphone icon in the address bar -> clear blocked permission -> Refresh."
        )

    # -------------------------------------------------------------------------
    # TAB 3: AI MODELS & SEARCH
    # -------------------------------------------------------------------------
    with set_tab3:
        st.subheader("🤖 AI Provider Diagnostics & Search Intelligence")
        llm_stat = get_llm_status()

        llm_c1, llm_c2 = st.columns(2)
        with llm_c1:
            st.write(f"- **Active Provider:** `{llm_stat['provider'].upper()}`")
            st.write(f"- **Connection Status:** `{llm_stat['status'].upper()}`")
            st.write(f"- **Default Model:** `{llm_stat.get('model', 'None')}`")
            st.write(f"- **Pricing / Tier:** {llm_stat.get('free_tier', 'Free tier available')}")

        with llm_c2:
            st.write("**Search Preferences**")
            st.write("- **Live Web Search API:** Native Gemini Grounding & DuckDuckGo Fallback")
            st.write("- **Deep Research:** Multi-step query decomposition & source synthesis")
            st.write("- **Domain Extraction:** Verified primary source URLs with direct markdown citations")

        st.markdown("---")
        st.markdown("#### 🔑 Gemini API Key Configuration")
        active_key = get_gemini_api_key()
        if active_key:
            st.success(f"✅ **Gemini API Key Active**: `{_mask_key(active_key)}` — Connected to `{llm_stat.get('model', 'gemini-1.5-flash')}`")
        else:
            st.warning("🟠 **Gemini API Key Not Configured**: Enter your key below or add `GEMINI_API_KEY` to your `.env` file or Streamlit Cloud Secrets.")

        new_key = st.text_input(
            "Configure Gemini API Key:",
            value="",
            type="password",
            placeholder="Paste your AIzaSy... key here",
            key="settings_gemini_key_input",
            help="Free key from aistudio.google.com. Stored securely on your local machine and never committed to Git.",
        )

        key_col1, key_col2 = st.columns([1, 1])
        with key_col1:
            if st.button("💾 Save Key", key="btn_save_gemini_key_settings", type="primary", use_container_width=True):
                clean_k = new_key.strip()
                if clean_k:
                    st.session_state["user_gemini_api_key"] = clean_k
                    # Update local .env file safely
                    env_p = PROJECT_ROOT / ".env"
                    lines = []
                    if env_p.exists():
                        with open(env_p, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    found = False
                    new_lines = []
                    for line in lines:
                        if line.strip().startswith("GEMINI_API_KEY="):
                            new_lines.append(f"GEMINI_API_KEY={clean_k}\n")
                            found = True
                        else:
                            new_lines.append(line)
                    if not found:
                        new_lines.append(f"\nGEMINI_API_KEY={clean_k}\n")
                    with open(env_p, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    st.toast("✅ Gemini API Key saved to .env and active session!", icon="🔑")
                    st.rerun()
                else:
                    st.error("Please enter a valid key.")

        with key_col2:
            if st.button("🧪 Test Key Connection", key="btn_test_gemini_key_settings", use_container_width=True):
                target_key = new_key.strip() or active_key
                if not target_key:
                    st.error("No API key available to test. Please enter a key above.")
                else:
                    with st.spinner("Verifying key with Google Gemini API..."):
                        val_res = validate_gemini_key(target_key)
                    if val_res["valid"]:
                        st.success(f"✅ {val_res['message']}")
                    else:
                        st.error(f"❌ Verification failed: {val_res['message']}")

        st.markdown("---")
        st.markdown(
            """
            #### 📋 Deployment & Configuration Guide
            NOVA supports both local and cloud deployment:
            - **Local Development (`.env` file):**
              Create or edit `.env` in the project root:
              ```env
              GEMINI_API_KEY=AIzaSy...
              GEMINI_MODEL=gemini-1.5-flash
              LLM_PROVIDER=gemini
              ```
            - **Streamlit Community Cloud (`secrets.toml`):**
              In your Streamlit Cloud app settings under **Secrets**, add:
              ```toml
              GEMINI_API_KEY = "AIzaSy..."
              ```
            - **Google AI Studio:** Get a free API key (zero cost, 15 RPM, 1,500 RPD) at **[aistudio.google.com](https://aistudio.google.com/)**.
            """
        )

    # -------------------------------------------------------------------------
    # TAB 4: MEMORY & DATA OWNERSHIP
    # -------------------------------------------------------------------------
    with set_tab4:
        st.subheader("🧠 Personal Memory & Data Governance")
        st.caption("You maintain full ownership of all personal memories stored by NOVA.")

        user_memories = get_all_memories(user_id=active_user_id)
        if not isinstance(user_memories, list):
            user_memories = []
        u_name = str(current_user.get("username", "user") or "user")
        st.write(f"**Total Saved Memories for @{u_name}:** `{len(user_memories)}` records")

        m_exp_c1, m_exp_c2, m_exp_c3 = st.columns(3)
        with m_exp_c1:
            m_json = export_memories_json(user_id=active_user_id)
            st.download_button("📥 Export Memories (JSON)", data=m_json, file_name=f"nova_memories_{u_name}.json", mime="application/json", use_container_width=True)
        with m_exp_c2:
            m_csv = export_memories_csv(user_id=active_user_id)
            st.download_button("📥 Export Memories (CSV)", data=m_csv, file_name=f"nova_memories_{u_name}.csv", mime="text/csv", use_container_width=True)
        with m_exp_c3:
            if st.button("🗑️ Wipe All Memories", type="secondary", use_container_width=True):
                st.session_state.confirm_wipe_mem_settings = True

        if st.session_state.get("confirm_wipe_mem_settings", False):
            st.warning("⚠️ **Caution**: Are you sure you want to permanently delete all personal memories for your account?")
            cw1, cw2 = st.columns(2)
            with cw1:
                if st.button("Yes, Wipe All My Memories", type="primary"):
                    deleted_count = clear_all_memories(user_id=active_user_id)
                    st.success(f"Successfully deleted {deleted_count} memories.")
                    st.session_state.confirm_wipe_mem_settings = False
                    st.rerun()
            with cw2:
                if st.button("Cancel"):
                    st.session_state.confirm_wipe_mem_settings = False
                    st.rerun()

    # -------------------------------------------------------------------------
    # TAB 5: ACCOUNT & SAFETY AUDIT
    # -------------------------------------------------------------------------
    with set_tab5:
        st.subheader("👤 Active User Account Details")
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            st.write(f"- **Full Name:** {current_user.get('full_name', 'Student User')}")
            st.write(f"- **Username:** `@{current_user.get('username', 'student')}`")
            st.write(f"- **Account ID:** `#{current_user.get('id', 'user_student_default')}`")
            created_val = str(current_user.get("created_at", "") or "")
            st.write(f"- **Created At:** {created_val[:10] if len(created_val) >= 10 else (created_val or 'Unknown')}")
        with u_col2:
            st.write(f"- **Database Path:** `{DATABASE_PATH}`")
            db_size = round(DATABASE_PATH.stat().st_size / 1024, 2) if DATABASE_PATH.exists() else 0
            st.write(f"- **Local DB Size:** `{db_size} KB`")
            if st.button("🚪 Sign Out of Account", type="secondary"):
                st.session_state.current_user = None
                st.session_state.active_thread_id = None
                st.session_state.chat_messages = []
                st.rerun()

        st.markdown("---")
        st.subheader("🛡️ Safety, Privacy & Action Audit Log")
        st.caption("Every tool invocation, task modification, and schedule event is logged for transparency.")

        recent_logs = export_table_to_csv("action_logs")
        if recent_logs:
            try:
                df_logs = pd.read_csv(io.StringIO(recent_logs))
                if not df_logs.empty and "id" in df_logs.columns:
                    st.dataframe(df_logs.sort_values(by="id", ascending=False).head(15), use_container_width=True)
                else:
                    st.dataframe(df_logs, use_container_width=True)
                st.download_button(
                    "📥 Download Action Audit Log (.csv)",
                    data=recent_logs,
                    file_name="nova_action_audit_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"Could not format logs: {e}")
        else:
            st.info("No action logs recorded yet.")

        st.markdown("---")
        st.subheader("🔒 Architecture & Privacy Guarantees")
        st.markdown(
            """
            - **Multi-User Isolation:** All conversations, tasks, schedules, memories, and documents are strictly isolated by `user_id` on the database level.
            - **Local-First Privacy:** Data is stored locally in `data/nova.db` and files in `data/documents/`.
            - **Zero Data Leakage:** Prompts sent to LLM providers are encrypted via HTTPS. No training or fine-tuning occurs on personal user data without explicit consent.
            - **Version Control Protection:** Sensitive files (`.env`, `nova.db`, uploaded documents) are excluded by `.gitignore`.
            """
        )
