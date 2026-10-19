"""
NOVA Assistant Orchestrator
Core controller coordinating intent analysis, memory access, validated tool execution,
and conversational response generation.
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from database.db import execute_insert, fetch_all, log_action
from memory.memory_manager import get_all_memories, search_memories
from nova_core.config import APP_NAME, APP_TAGLINE, is_llm_configured, get_llm_status
from nova_core.tool_registry import ToolRegistry, ToolResult, default_tool_registry
from nova_core.conversation_manager import add_message, get_thread_messages
from nova_core.llm_service import get_llm_service
from nova_core.web_search import search_web, format_sources_markdown, fetch_page_content
from nova_core.research_router import (
    should_search_web,
    is_deep_research_mode,
    run_deep_research,
)

logger = logging.getLogger("nova.assistant")


class NovaAssistant:
    """
    Main assistant orchestrator implementing the pipeline:
    Input Validation -> Intent & Tool Selection -> Safety Checks -> Validated Execution -> Response.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.registry = tool_registry or default_tool_registry

    def process_message(
        self,
        user_input: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        web_search: str = "auto",
        deep_research: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes a single user message and returns the assistant's response.
        Supports intelligent question routing (web search vs reasoning),
        multi-engine live search with inline citations, and Deep Research mode.
        """
        # 1. Input Validation
        cleaned_input = (user_input or "").strip()
        if not cleaned_input:
            return {
                "response": "Please enter a message or command.",
                "tool_called": None,
                "tool_result": None,
                "mode": "rule_based",
                "conversation_id": conversation_id or str(uuid.uuid4()),
            }

        if len(cleaned_input) > 4000:
            return {
                "response": "Input too long. Please limit your message to 4,000 characters.",
                "tool_called": None,
                "tool_result": None,
                "mode": "rule_based",
                "conversation_id": conversation_id or str(uuid.uuid4()),
            }

        conv_id = conversation_id or str(uuid.uuid4())

        # Persist user turn to SQLite conversations table with user_id
        self._save_conversation_turn(conv_id, "user", cleaned_input, user_id=user_id)

        # 2. Check if input is a deterministic tool command
        if self._is_explicit_tool_command(cleaned_input):
            output = self._handle_rule_based_flow(cleaned_input, conv_id)
        # 3. Check Deep Research Mode
        elif deep_research or is_deep_research_mode(cleaned_input):
            def llm_gen(msgs, sys_p):
                if is_llm_configured():
                    llm = get_llm_service()
                    if llm:
                        return llm.generate(msgs, system_prompt=sys_p)
                return None

            deep_res = run_deep_research(cleaned_input, llm_generator=llm_gen)
            output = {
                "response": deep_res["report"],
                "tool_called": "deep_research",
                "tool_result": {
                    "subqueries": deep_res.get("subqueries", []),
                    "source_count": len(deep_res.get("sources", [])),
                },
                "mode": "deep_research",
                "sources": deep_res.get("sources", []),
                "conversation_id": conv_id,
            }
        else:
            # 4. Intelligent Question Routing (Web Search vs Direct Reasoning)
            needs_web, reason = should_search_web(cleaned_input, web_mode=web_search)
            if needs_web:
                try:
                    search_res = search_web(cleaned_input, max_results=5, fetch_full_content=True)
                except Exception as e:
                    logger.warning(f"Live web search execution failed: {e}")
                    search_res = []

                if search_res:
                    if is_llm_configured():
                        web_blocks = []
                        for idx, s in enumerate(search_res, 1):
                            content = s.get("full_content") or s.get("snippet") or ""
                            web_blocks.append(
                                f"Source [{idx}]:\nTitle: {s['title']}\nURL: {s['url']}\nDomain: {s['domain']}\nContent: {content[:1500]}"
                            )
                        web_context = "\n---\n".join(web_blocks)
                        output = self._handle_llm_flow(
                            cleaned_input,
                            conv_id,
                            user_id=user_id,
                            web_context=web_context,
                            sources=search_res,
                        )
                    else:
                        # Offline structured search summary
                        body = [
                            f"🌐 **NOVA Web Search Findings for:** *\"{cleaned_input}\"*\n",
                            f"*(Routing trigger: {reason})*\n",
                            "Here is the verified information retrieved from live sources:\n",
                        ]
                        for idx, s in enumerate(search_res, 1):
                            snip = s.get("full_content") or s.get("snippet") or ""
                            snip_short = snip[:280] + ("..." if len(snip) > 280 else "")
                            body.append(f"- **[{idx}] [{s['title']}]({s['url']})**: {snip_short}\n")
                        body.append(format_sources_markdown(search_res))
                        output = {
                            "response": "\n".join(body),
                            "tool_called": "web_search",
                            "tool_result": {"source_count": len(search_res)},
                            "mode": "web_grounded_offline",
                            "sources": search_res,
                            "conversation_id": conv_id,
                        }
                else:
                    # Search returned no results or failed; fallback to LLM or rule-based
                    if is_llm_configured():
                        output = self._handle_llm_flow(cleaned_input, conv_id, user_id=user_id)
                    else:
                        output = self._handle_rule_based_flow(cleaned_input, conv_id)
            elif is_llm_configured():
                # Direct LLM Generative Dialogue (No external search needed)
                output = self._handle_llm_flow(cleaned_input, conv_id, user_id=user_id)
            else:
                # Rule-Based / Setup Guidance
                output = self._handle_rule_based_flow(cleaned_input, conv_id)

        # Persist assistant turn with metadata and user_id
        meta = {
            "mode": output.get("mode"),
            "provider": output.get("provider"),
            "model": output.get("model"),
            "tool_called": output.get("tool_called"),
            "sources": output.get("sources"),
        }
        self._save_conversation_turn(conv_id, "assistant", output["response"], metadata=meta, user_id=user_id)

        return output

    def process_message_stream(
        self,
        user_input: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        web_search: str = "auto",
        deep_research: bool = False,
    ):
        """
        Processes a user message and yields string chunks for real-time streaming,
        supporting live web search grounding, inline citations, and Deep Research.
        """
        cleaned_input = (user_input or "").strip()
        if not cleaned_input:
            msg = "Please enter a message or command."
            yield msg
            return

        if len(cleaned_input) > 4000:
            msg = "Input too long. Please limit your message to 4,000 characters."
            yield msg
            return

        conv_id = conversation_id or str(uuid.uuid4())
        self._save_conversation_turn(conv_id, "user", cleaned_input, user_id=user_id)

        if self._is_explicit_tool_command(cleaned_input):
            output = self._handle_rule_based_flow(cleaned_input, conv_id)
            meta = {
                "mode": output.get("mode"),
                "provider": output.get("provider"),
                "model": output.get("model"),
                "tool_called": output.get("tool_called"),
            }
            text = output["response"]
            words = text.split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")
            self._save_conversation_turn(conv_id, "assistant", text, metadata=meta, user_id=user_id)
            return

        # Handle Deep Research streaming
        if deep_research or is_deep_research_mode(cleaned_input):
            def llm_gen(msgs, sys_p):
                if is_llm_configured():
                    llm = get_llm_service()
                    if llm:
                        return llm.generate(msgs, system_prompt=sys_p)
                return None

            deep_res = run_deep_research(cleaned_input, llm_generator=llm_gen)
            report_text = deep_res["report"]
            words = report_text.split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")
            self._save_conversation_turn(
                conv_id,
                "assistant",
                report_text,
                metadata={"mode": "deep_research", "sources": deep_res.get("sources", [])},
                user_id=user_id,
            )
            return

        # Check Routing
        needs_web, reason = should_search_web(cleaned_input, web_mode=web_search)
        search_res = []
        web_context = ""
        if needs_web:
            try:
                search_res = search_web(cleaned_input, max_results=5, fetch_full_content=True)
                if search_res:
                    web_blocks = []
                    for idx, s in enumerate(search_res, 1):
                        content = s.get("full_content") or s.get("snippet") or ""
                        web_blocks.append(
                            f"Source [{idx}]:\nTitle: {s['title']}\nURL: {s['url']}\nDomain: {s['domain']}\nContent: {content[:1500]}"
                        )
                    web_context = (
                        "\n\nLive Internet Grounding Data (Verified Search Results):\n"
                        + "\n---\n".join(web_blocks)
                        + "\nInstructions for live web search answering:\n"
                        "- Base your answer strictly on the facts and data in the provided search results.\n"
                        "- Place inline bracket citations like [1], [2] immediately following the facts they support.\n"
                        "- If the provided sources do not answer the question, clearly state so.\n"
                        "- Never make up non-existent links or fake citations."
                    )
            except Exception as e:
                logger.warning(f"Streaming web search failed: {e}")
                search_res = []

        if not is_llm_configured():
            if search_res:
                body = [
                    f"🌐 **NOVA Web Search Findings for:** *\"{cleaned_input}\"*\n",
                    f"*(Routing trigger: {reason})*\n",
                    "Here is the verified information retrieved from live sources:\n",
                ]
                for idx, s in enumerate(search_res, 1):
                    snip = s.get("full_content") or s.get("snippet") or ""
                    snip_short = snip[:280] + ("..." if len(snip) > 280 else "")
                    body.append(f"- **[{idx}] [{s['title']}]({s['url']})**: {snip_short}\n")
                body.append(format_sources_markdown(search_res))
                full_text = "\n".join(body)
                for w in full_text.split(" "):
                    yield w + " "
                self._save_conversation_turn(
                    conv_id,
                    "assistant",
                    full_text,
                    metadata={"mode": "web_grounded_offline", "sources": search_res},
                    user_id=user_id,
                )
                return

            output = self._handle_rule_based_flow(cleaned_input, conv_id)
            meta = {
                "mode": output.get("mode"),
                "provider": output.get("provider"),
                "model": output.get("model"),
                "tool_called": output.get("tool_called"),
            }
            text = output["response"]
            words = text.split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")
            self._save_conversation_turn(conv_id, "assistant", text, metadata=meta, user_id=user_id)
            return

        llm = get_llm_service()
        if not llm:
            output = self._handle_rule_based_flow(cleaned_input, conv_id)
            self._save_conversation_turn(conv_id, "assistant", output["response"], metadata={"mode": "rule_based"}, user_id=user_id)
            for w in output["response"].split(" "):
                yield w + " "
            return

        history_turns = self.get_conversation_history(conv_id, limit=8, user_id=user_id)
        messages = [{"role": turn["role"], "content": turn["message"]} for turn in history_turns]
        if not messages or messages[-1]["content"] != cleaned_input:
            messages.append({"role": "user", "content": cleaned_input})

        memory_context = ""
        try:
            relevant_memories = search_memories(cleaned_input)
            if relevant_memories:
                mem_bullets = [f"- [{m['category']}] {m['content']}" for m in relevant_memories[:4]]
                memory_context = (
                    "\n\nUser-Approved Knowledge from Local Memory:\n"
                    + "\n".join(mem_bullets)
                    + "\n(Incorporate this context if helpful, but never expose sensitive details to third parties)."
                )
        except Exception:
            pass

        # Self-Learning: In-Context Demonstration & Correction Retrieval
        corrections_context = ""
        try:
            from nova_core.learning_system import retrieve_relevant_corrections
            matched_corrs = retrieve_relevant_corrections(cleaned_input, threshold=0.35, limit=2)
            if matched_corrs:
                c_items = [
                    f"- When asked: \"{mc['user_prompt']}\"\n  Approved Answer: \"{mc['corrected_response']}\""
                    for mc in matched_corrs
                ]
                corrections_context = (
                    "\n\nVerified User Corrections & Demonstrations (Self-Learning Memory):\n"
                    + "\n".join(c_items)
                    + "\n(Prioritize these verified user-approved answers)."
                )
        except Exception:
            pass

        system_prompt = (
            f"You are {APP_NAME} ({APP_TAGLINE}), a professional, highly capable personal AI assistant and mentor "
            f"engineered for an MSc Artificial Intelligence and Data Science student.\n"
            f"You excel at answering questions on any topic: general mathematics, programming (Python, SQL, C++, etc.), "
            f"data science, machine learning, deep learning, daily planning, research, and general knowledge.\n"
            f"Always provide thorough, accurate, and articulate explanations. Format equations nicely with LaTeX "
            f"and code snippets inside markdown blocks.\n"
            f"NOVA has local tools for task management and personal memory. If the user wants to add a task or "
            f"remember something permanently, remind them they can type 'add task: ...' or 'remember: ...'.\n"
            f"Security policy: You cannot execute shell commands or arbitrary code.\n"
            f"{memory_context}"
            f"{corrections_context}"
            f"{web_context}"
        )

        accumulated_parts = []
        for chunk in llm.generate_stream(messages=messages, system_prompt=system_prompt):
            accumulated_parts.append(chunk)
            yield chunk

        if search_res:
            sources_md = format_sources_markdown(search_res)
            accumulated_parts.append(sources_md)
            yield sources_md

        full_response = "".join(accumulated_parts)
        provider_name = "gemini" if "gemini" in getattr(llm, "endpoint", "").lower() else "openai"
        meta = {
            "mode": "web_grounded" if search_res else "llm",
            "provider": provider_name,
            "model": getattr(llm, "model", "default"),
            "sources": search_res if search_res else None,
        }
        self._save_conversation_turn(conv_id, "assistant", full_response, metadata=meta, user_id=user_id)

    def _handle_rule_based_flow(self, user_input: str, conv_id: str) -> Dict[str, Any]:
        """
        Rule-based intent classifier and local tool invoker.
        Translates natural language command patterns into validated tool executions.
        """
        lower = user_input.lower().strip()

        # Check learned user corrections first
        try:
            from nova_core.learning_system import retrieve_relevant_corrections
            exact_corrections = retrieve_relevant_corrections(user_input, threshold=0.85, limit=1)
            if exact_corrections:
                matched_corr = exact_corrections[0]
                reply = f"💡 *(Applied verified user correction)*\n\n{matched_corr['corrected_response']}"
                return {
                    "response": reply,
                    "tool_called": None,
                    "tool_result": None,
                    "mode": "rule_based_learned",
                    "conversation_id": conv_id,
                }
        except Exception:
            pass

        # Intent: Create Task
        # Patterns: "add task ...", "create task ...", "task: ...", "new task ..."
        task_match = re.match(r"^(?:add|create|new)\s+task[:\s]+(.+)$", user_input, re.IGNORECASE)
        if not task_match and lower.startswith("task:"):
            task_match = re.match(r"^task[:\s]+(.+)$", user_input, re.IGNORECASE)

        if task_match:
            raw_task = task_match.group(1).strip()
            # Extract optional priority and due date from string
            # e.g., "submit thesis due 2026-11-01 priority high"
            title, priority, due_date = self._parse_task_extras(raw_task)
            
            tool_res = self.registry.execute(
                "create_task",
                {"title": title, "priority": priority, "due_date": due_date}
            )
            
            if tool_res.status == "success":
                t = tool_res.data
                due_info = f", Due: {t['due_date']}" if t.get("due_date") else ""
                reply = (
                    f"✅ **Task Created Successfully!**\n\n"
                    f"- **ID**: #{t['id']}\n"
                    f"- **Title**: {t['title']}\n"
                    f"- **Priority**: {t['priority']}{due_info}\n"
                    f"- **Status**: Pending\n\n"
                    f"*You can view and manage all tasks in the Task Manager tab.*"
                )
            else:
                reply = f"❌ **Failed to create task:** {tool_res.message}"

            return {
                "response": reply,
                "tool_called": "create_task",
                "tool_result": tool_res.to_dict(),
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Intent: List Tasks
        # Patterns: "list tasks", "show tasks", "my tasks", "pending tasks", "tasks"
        if lower in {"list tasks", "show tasks", "my tasks", "pending tasks", "tasks", "view tasks"}:
            tool_res = self.registry.execute("list_tasks", {"status": "pending"})
            if tool_res.status == "success":
                tasks = tool_res.data
                if not tasks:
                    reply = "📋 You currently have **no pending tasks**. Great job staying on top of your work!"
                else:
                    items = []
                    for t in tasks:
                        due = f" *(Due: {t['due_date']})*" if t.get("due_date") else ""
                        prio = f"`[{t['priority']}]`"
                        items.append(f"- **#{t['id']}** {prio} {t['title']}{due}")
                    reply = f"📋 **Pending Tasks ({len(tasks)}):**\n\n" + "\n".join(items)
            else:
                reply = f"❌ Could not retrieve tasks: {tool_res.message}"

            return {
                "response": reply,
                "tool_called": "list_tasks",
                "tool_result": tool_res.to_dict(),
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Intent: Complete Task
        # Patterns: "complete task 3", "finish task 2", "done 5"
        complete_match = re.match(r"^(?:complete(?:\s+task)?|finish(?:\s+task)?|done)\s+(\d+)$", lower)
        if complete_match:
            task_id = int(complete_match.group(1))
            tool_res = self.registry.execute("complete_task", {"task_id": task_id})
            if tool_res.status == "success":
                reply = f"🎉 **Task #{task_id} marked as completed!** Activity recorded in your productivity log."
            else:
                reply = f"❌ **Could not complete task:** {tool_res.message}"

            return {
                "response": reply,
                "tool_called": "complete_task",
                "tool_result": tool_res.to_dict(),
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Intent: Save Memory
        # Patterns: "remember that ...", "remember ...", "save memory ...", "store memory ..."
        memory_match = re.match(r"^(?:remember(?:\s+that)?|save\s+memory|remember)[:\s]+(.+)$", user_input, re.IGNORECASE)
        if memory_match:
            content = memory_match.group(1).strip()
            # Check for optional category e.g., "[Academic] I prefer PyTorch"
            category = "General"
            cat_match = re.match(r"^\[(.*?)\]\s*(.+)$", content)
            if cat_match:
                category = cat_match.group(1).strip()
                content = cat_match.group(2).strip()

            tool_res = self.registry.execute("save_memory", {"content": content, "category": category})
            if tool_res.status == "success":
                m = tool_res.data
                reply = (
                    f"🧠 **Memory Saved with your permission:**\n\n"
                    f"- **ID**: #{m['id']}\n"
                    f"- **Category**: {m['category']}\n"
                    f"- **Content**: \"{m['content']}\"\n\n"
                    f"*You can view, edit, or delete memories anytime under 'My Memory'.*"
                )
            else:
                reply = f"❌ **Failed to save memory:** {tool_res.message}"

            return {
                "response": reply,
                "tool_called": "save_memory",
                "tool_result": tool_res.to_dict(),
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Intent: Search / Recall Memory
        # Patterns: "recall ...", "search memory ...", "what do you remember about ..."
        recall_match = re.match(r"^(?:recall|search\s+memory(?:\s+for)?|what\s+do\s+you\s+remember(?:\s+about)?)[:\s]+(.+)$", user_input, re.IGNORECASE)
        if recall_match:
            query = recall_match.group(1).strip()
            tool_res = self.registry.execute("search_memory", {"query": query})
            if tool_res.status == "success":
                memories = tool_res.data
                if not memories:
                    reply = f"🔍 No saved memories found matching: *'{query}'*."
                else:
                    items = [f"- **#{m['id']}** `[{m['category']}]`: {m['content']}" for m in memories]
                    reply = f"🧠 **Found {len(memories)} relevant memor{'y' if len(memories) == 1 else 'ies'}:**\n\n" + "\n".join(items)
            else:
                reply = f"❌ Search error: {tool_res.message}"

            return {
                "response": reply,
                "tool_called": "search_memory",
                "tool_result": tool_res.to_dict(),
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        if lower in {"my memory", "show memories", "list memories", "memories"}:
            all_m = get_all_memories()
            if not all_m:
                reply = "🧠 You haven't saved any memories yet. Tell me `remember: [info]` to store something!"
            else:
                items = [f"- **#{m['id']}** `[{m['category']}]`: {m['content']}" for m in all_m[:10]]
                reply = f"🧠 **Saved Memories ({len(all_m)} total):**\n\n" + "\n".join(items)
            return {
                "response": reply,
                "tool_called": None,
                "tool_result": None,
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Intent: Current Time & Date
        if lower in {"what time is it", "time", "current time", "what is today's date", "date", "what is the date"}:
            tool_res = self.registry.execute("get_current_time", {})
            data = tool_res.data
            reply = f"🕒 Current local time is **{data['current_time']}** on **{data['current_date']}**."
            return {
                "response": reply,
                "tool_called": "get_current_time",
                "tool_result": tool_res.to_dict(),
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Intent: Productivity Stats
        if lower in {"stats", "productivity", "analytics", "status", "summary"}:
            tool_res = self.registry.execute("analyze_productivity", {})
            stats = tool_res.data
            reply = (
                f"📊 **Current Productivity Snapshot:**\n\n"
                f"- **Pending Tasks**: {stats['pending']}\n"
                f"- **Completed Tasks**: {stats['completed']}\n"
                f"- **Overdue Tasks**: {stats['overdue']}\n"
                f"- **High Priority Pending**: {stats['high_priority_pending']}\n"
                f"- **Completion Rate**: {stats['completion_rate']}%\n\n"
                f"*Check the 'Productivity Analytics' tab for Matplotlib charts and data exports!*"
            )
            return {
                "response": reply,
                "tool_called": "analyze_productivity",
                "tool_result": tool_res.to_dict(),
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Intent: Help
        if lower in {"help", "commands", "what can you do", "info"}:
            reply = self._build_help_message()
            return {
                "response": reply,
                "tool_called": None,
                "tool_result": None,
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Greeting intent
        if lower in {"hello", "hi", "hey", "greetings", "good morning", "good evening"}:
            reply = (
                f"👋 Hello! I am **{APP_NAME}** ({APP_TAGLINE}).\n\n"
                f"I am ready to assist you with AI & Data Science queries, coding, task management, research, and daily planning.\n\n"
                f"Try asking:\n"
                f"- `How does backpropagation work in PyTorch?`\n"
                f"- `add task: Complete AI assignment due 2026-10-25 priority High`\n"
                f"- `remember: [Academic] My master's focus is Graph Neural Networks`\n"
                f"- `show tasks` or `stats`\n"
                f"- `help` for command shortcuts."
            )
            return {
                "response": reply,
                "tool_called": None,
                "tool_result": None,
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Fallback response for unmapped inputs when LLM is not configured
        reply = (
            f"🤖 **NOVA AI Assistant**\n\n"
            f"I received your question: *\"{user_input}\"*\n\n"
            f"NOVA is currently operating in Rule-Based Mode. "
            f"Free-form conversational intelligence will be enabled once an LLM provider (such as Google Gemini or OpenAI) is configured in your `.env` file.\n\n"
            f"To unlock conversational intelligence (for deep reasoning, coding, math, and open-ended Q&A), "
            f"please add your free API key in `.env`:\n\n"
            f"```bash\n"
            f"GEMINI_API_KEY=your_actual_gemini_key_here\n"
            f"```\n"
            f"*(Get a free Gemini API key with zero credit card required at [aistudio.google.com](https://aistudio.google.com).)*\n\n"
            f"**Built-in tools available right now:**\n"
            f"- **Add Task**: `add task: <title> [due YYYY-MM-DD] [priority High|Medium|Low]`\n"
            f"- **List Tasks**: `show tasks` or `list tasks`\n"
            f"- **Complete Task**: `complete task <id>`\n"
            f"- **Memory**: `remember: <information>`\n"
            f"- **Search Memory**: `recall: <keyword>`\n"
            f"- **Productivity**: `stats` or `time`"
        )
        return {
            "response": reply,
            "tool_called": None,
            "tool_result": None,
            "mode": "rule_based",
            "conversation_id": conv_id,
        }

    def _is_explicit_tool_command(self, user_input: str) -> bool:
        """
        Determines if an input matches a deterministic tool command.
        These are processed directly without consuming LLM API calls.
        """
        lower = user_input.lower().strip()

        # Task commands
        if re.match(r"^(?:add|create|new)\s+task[:\s]+(.+)$", user_input, re.IGNORECASE) or lower.startswith("task:"):
            return True
        if lower in {"list tasks", "show tasks", "my tasks", "pending tasks", "tasks", "view tasks"}:
            return True
        if re.match(r"^(?:complete(?:\s+task)?|finish(?:\s+task)?|done)\s+(\d+)$", lower):
            return True

        # Memory commands
        if re.match(r"^(?:remember(?:\s+that)?|save\s+memory|remember)[:\s]+(.+)$", user_input, re.IGNORECASE):
            return True
        if re.match(r"^(?:recall|search\s+memory(?:\s+for)?|what\s+do\s+you\s+remember(?:\s+about)?)[:\s]+(.+)$", user_input, re.IGNORECASE):
            return True
        if lower in {"my memory", "show memories", "list memories", "memories"}:
            return True

        # System info commands
        if lower in {"what time is it", "time", "current time", "what is today's date", "date", "what is the date"}:
            return True
        if lower in {"stats", "productivity", "analytics", "status", "summary"}:
            return True
        if lower in {"help", "commands", "what can you do", "info"}:
            return True

        # Greetings when LLM is NOT configured
        if not is_llm_configured() and lower in {"hello", "hi", "hey", "greetings", "good morning", "good evening"}:
            return True

        return False

    def _handle_llm_flow(
        self,
        user_input: str,
        conv_id: str,
        user_id: Optional[str] = None,
        web_context: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Processes message using the configured LLM provider (Gemini or OpenAI).
        Maintains conversational history, injects relevant user-approved memory,
        applies verified web search grounding, and provides graceful fallback on errors.
        """
        llm = get_llm_service()
        if not llm:
            return self._handle_rule_based_flow(user_input, conv_id)

        # 1. Retrieve recent multi-turn context (last 8 turns)
        history_turns = self.get_conversation_history(conv_id, limit=8, user_id=user_id)
        messages = []
        for turn in history_turns:
            # Avoid repeating the current turn if already in history
            messages.append({
                "role": turn["role"],
                "content": turn["message"],
            })

        # Ensure current user message is at the end of messages
        if not messages or messages[-1]["content"] != user_input:
            messages.append({"role": "user", "content": user_input})

        # 2. Privacy-preserving memory injection:
        # Search if any explicitly approved memories relate to the query
        memory_context = ""
        try:
            relevant_memories = search_memories(user_input)
            if relevant_memories:
                mem_bullets = [f"- [{m['category']}] {m['content']}" for m in relevant_memories[:4]]
                memory_context = (
                    "\n\nUser-Approved Knowledge from Local Memory:\n"
                    + "\n".join(mem_bullets)
                    + "\n(Incorporate this context if helpful, but never expose sensitive details to third parties)."
                )
        except Exception:
            pass

        # 2b. Self-Learning: In-Context Demonstration & Correction Retrieval
        corrections_context = ""
        try:
            from nova_core.learning_system import retrieve_relevant_corrections
            matched_corrs = retrieve_relevant_corrections(user_input, threshold=0.35, limit=2)
            if matched_corrs:
                c_items = [
                    f"- When asked: \"{mc['user_prompt']}\"\n  Approved Answer: \"{mc['corrected_response']}\""
                    for mc in matched_corrs
                ]
                corrections_context = (
                    "\n\nVerified User Corrections & Demonstrations (Self-Learning Memory):\n"
                    + "\n".join(c_items)
                    + "\n(Prioritize these verified user-approved answers)."
                )
        except Exception as e:
            print(f"[Warning] Failed to query learned corrections: {e}")

        # 3. System prompt enforcing identity, security, student support, and web grounding
        web_prompt_instructions = ""
        if web_context:
            web_prompt_instructions = (
                f"\n\nLive Internet Grounding Data (Verified Search Results):\n"
                f"{web_context}\n\n"
                f"Instructions for live web search grounding:\n"
                f"1. Base your response strictly on the factual details provided above.\n"
                f"2. Cite facts using inline numbered brackets like [1], [2] matching the sources.\n"
                f"3. If the search results do not provide the answer, explicitly state that recent information "
                f"could not be confirmed.\n"
                f"4. Never invent fake links or non-existent URLs."
            )

        system_prompt = (
            f"You are {APP_NAME} ({APP_TAGLINE}), a professional, highly capable personal AI assistant and mentor "
            f"engineered for an MSc Artificial Intelligence and Data Science student.\n"
            f"You excel at answering questions on any topic: general mathematics, programming (Python, SQL, C++, etc.), "
            f"data science, machine learning, deep learning, daily planning, research, and general knowledge.\n"
            f"Always provide thorough, accurate, and articulate explanations. Format equations nicely with LaTeX "
            f"and code snippets inside markdown blocks.\n"
            f"NOVA has local tools for task management and personal memory. If the user wants to add a task or "
            f"remember something permanently, remind them they can type 'add task: ...' or 'remember: ...'.\n"
            f"Security policy: You cannot execute shell commands or arbitrary code.\n"
            f"{memory_context}"
            f"{corrections_context}"
            f"{web_prompt_instructions}"
        )

        # 4. Generate response from LLM
        llm_res = llm.generate(messages=messages, system_prompt=system_prompt)

        if llm_res.success:
            log_action("llm_completion", f"Provider: {llm_res.provider}, Model: {llm_res.model}", status="success")
            final_reply = llm_res.content
            if sources:
                src_md = format_sources_markdown(sources)
                if "Sources & References" not in final_reply:
                    final_reply = final_reply + src_md

            return {
                "response": final_reply,
                "tool_called": "web_search" if sources else None,
                "tool_result": {"sources": len(sources)} if sources else None,
                "mode": "web_grounded" if sources else "llm",
                "provider": llm_res.provider,
                "model": llm_res.model,
                "tokens_used": llm_res.tokens_used,
                "sources": sources if sources else None,
                "conversation_id": conv_id,
            }
        else:
            log_action("llm_completion", f"Failed: {llm_res.error}", status="failed")
            # Clear, helpful error notification with automatic fallback guidance
            fallback_msg = (
                f"⚠️ **{llm_res.provider.upper()} API Error ({llm_res.error_type or 'General'}):**\n\n"
                f"> {llm_res.error}\n\n"
                f"NOVA has automatically engaged **Rule-Based Fallback Mode** so your assistant remains fully functional. "
                f"You can continue using local commands like `add task: ...`, `show tasks`, `remember: ...`, or `stats`."
            )
            return {
                "response": fallback_msg,
                "tool_called": None,
                "tool_result": None,
                "mode": "fallback_rule_based",
                "provider": llm_res.provider,
                "model": llm_res.model,
                "error": llm_res.error,
                "conversation_id": conv_id,
            }

    def _parse_task_extras(self, raw_text: str) -> Tuple[str, str, Optional[str]]:
        """
        Extracts optional 'due YYYY-MM-DD' and 'priority Low|Medium|High' from a task title.
        """
        priority = "Medium"
        due_date = None
        cleaned = raw_text

        # Priority extraction
        prio_match = re.search(r"\bpriority\s+(low|medium|high)\b", cleaned, re.IGNORECASE)
        if prio_match:
            priority = prio_match.group(1).capitalize()
            cleaned = cleaned[:prio_match.start()] + cleaned[prio_match.end():]

        # Due date extraction (YYYY-MM-DD)
        due_match = re.search(r"\bdue\s+(\d{4}-\d{2}-\d{2})\b", cleaned, re.IGNORECASE)
        if due_match:
            due_date = due_match.group(1)
            cleaned = cleaned[:due_match.start()] + cleaned[due_match.end():]

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned, priority, due_date

    def _build_help_message(self) -> str:
        return (
            f"### 🤖 NOVA Assistant Commands & Shortcuts\n\n"
            f"| Command | Example | Description |\n"
            f"|---|---|---|\n"
            f"| **Add Task** | `add task: Finish NLP project due 2026-10-30 priority High` | Creates a persistent task |\n"
            f"| **List Tasks** | `show tasks` or `list tasks` | Shows pending tasks |\n"
            f"| **Complete Task** | `complete task 1` or `done 1` | Marks task as completed |\n"
            f"| **Remember** | `remember: [Study] Exam is in Room 402` | Saves information explicitly |\n"
            f"| **Recall** | `recall: Room` | Searches stored memories |\n"
            f"| **Show Memories** | `memories` or `my memory` | Lists saved memories |\n"
            f"| **Time/Date** | `time` or `date` | Shows current local timestamp |\n"
            f"| **Productivity** | `stats` or `summary` | Real metrics from SQLite |\n\n"
            f"*(You can also ask any open-ended question on AI, coding, data science, or mathematics!)*"
        )

    def _save_conversation_turn(
        self,
        conversation_id: str,
        role: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Stores a chat message into the conversations table and bumps the active thread."""
        try:
            add_message(conversation_id, role, message, metadata=metadata, user_id=user_id or "user_student_default")
        except Exception as e:
            print(f"[Warning] Could not persist conversation turn: {e}")

    def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves stored turns for a specific conversation session."""
        try:
            raw_msgs = get_thread_messages(conversation_id, limit=limit, user_id=user_id)
            return [{"role": m["role"], "message": m["content"], "created_at": m["created_at"]} for m in raw_msgs]
        except Exception as e:
            print(f"[Warning] Could not retrieve conversation turns: {e}")
            return []
