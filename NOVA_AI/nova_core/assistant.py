"""
NOVA Assistant Orchestrator
Core controller coordinating intent analysis, memory access, validated tool execution,
and conversational response generation.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from database.db import execute_insert, fetch_all, log_action
from memory.memory_manager import get_all_memories, search_memories
from nova_core.config import APP_NAME, APP_TAGLINE, is_llm_configured, get_llm_status
from nova_core.tool_registry import ToolRegistry, ToolResult, default_tool_registry


from nova_core.llm_service import get_llm_service


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
    ) -> Dict[str, Any]:
        """
        Processes a single user message and returns the assistant's response.
        
        Args:
            user_input: Raw text message from the user.
            conversation_id: Session ID for conversation history.
            
        Returns:
            Dict containing:
                - 'response': text output for the user
                - 'tool_called': name of tool if one was invoked, else None
                - 'tool_result': ToolResult dict or None
                - 'mode': 'rule_based' or 'llm'
                - 'conversation_id': active session id
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

        # Persist user turn to SQLite conversations table
        self._save_conversation_turn(conv_id, "user", cleaned_input)

        # 2. Check if input is a deterministic tool command
        if self._is_explicit_tool_command(cleaned_input):
            output = self._handle_rule_based_flow(cleaned_input, conv_id)
        elif is_llm_configured():
            # Real LLM Generative Dialogue
            output = self._handle_llm_flow(cleaned_input, conv_id)
        else:
            # Phase 1 Rule-Based Orchestration
            output = self._handle_rule_based_flow(cleaned_input, conv_id)

        # Persist assistant turn
        self._save_conversation_turn(conv_id, "assistant", output["response"])

        return output

    def _handle_rule_based_flow(self, user_input: str, conv_id: str) -> Dict[str, Any]:
        """
        Rule-based intent classifier and tool invoker for Phase 1.
        Translates natural language command patterns into validated tool executions.
        """
        lower = user_input.lower().strip()

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
                f"I am operating in **Phase 1 Rule-Based Mode** with persistent SQLite storage for your tasks and memories.\n\n"
                f"Try typing:\n"
                f"- `add task: Complete AI assignment due 2026-10-25 priority High`\n"
                f"- `remember: [Academic] My master's focus is Graph Neural Networks`\n"
                f"- `show tasks` or `stats`\n"
                f"- `help` for all available commands."
            )
            return {
                "response": reply,
                "tool_called": None,
                "tool_result": None,
                "mode": "rule_based",
                "conversation_id": conv_id,
            }

        # Fallback response for unmapped rule-based inputs
        # HONEST DISCLAIMER - Never simulate fake AI reasoning
        reply = (
            f"ℹ️ **NOVA is currently in Phase 1 Rule-Based Mode.**\n\n"
            f"I did not recognize that command pattern. In Phase 1, I only execute explicit validated commands and tools. "
            f"Free-form conversational intelligence will be enabled once an LLM provider (such as Gemini or OpenAI) is configured in Phase 2.\n\n"
            f"**Supported commands right now:**\n"
            f"- `add task: <title> [due YYYY-MM-DD] [priority High|Medium|Low]`\n"
            f"- `show tasks` or `list tasks`\n"
            f"- `complete task <id>`\n"
            f"- `remember: <information>`\n"
            f"- `recall: <keyword>`\n"
            f"- `time` or `date`\n"
            f"- `stats`\n"
            f"- `help`"
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

    def _handle_llm_flow(self, user_input: str, conv_id: str) -> Dict[str, Any]:
        """
        Processes message using the configured LLM provider (Gemini or OpenAI).
        Maintains conversational history, injects relevant user-approved memory,
        and provides graceful fallback on errors.
        """
        llm = get_llm_service()
        if not llm:
            return self._handle_rule_based_flow(user_input, conv_id)

        # 1. Retrieve recent multi-turn context (last 8 turns)
        history_turns = self.get_conversation_history(conv_id, limit=8)
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
        except Exception as e:
            print(f"[Warning] Failed to query memories for context: {e}")

        # 3. System prompt enforcing identity, security, and student support
        system_prompt = (
            f"You are {APP_NAME} ({APP_TAGLINE}), an advanced personal AI assistant engineered for an MSc "
            f"Artificial Intelligence and Data Science student.\n"
            f"You have deep knowledge in Python, NumPy, pandas, SQL, Machine Learning, Deep Learning, and Academic planning.\n"
            f"Be encouraging, concise, technically rigorous, and structured in your explanations.\n"
            f"NOVA has local tools for task management and personal memory. If the user wants to add a task or "
            f"remember something permanently, remind them they can type 'add task: ...' or 'remember: ...'.\n"
            f"Security policy: You cannot execute shell commands or arbitrary code.\n"
            f"{memory_context}"
        )

        # 4. Generate response from LLM
        llm_res = llm.generate(messages=messages, system_prompt=system_prompt)

        if llm_res.success:
            log_action("llm_completion", f"Provider: {llm_res.provider}, Model: {llm_res.model}", status="success")
            return {
                "response": llm_res.content,
                "tool_called": None,
                "tool_result": None,
                "mode": "llm",
                "provider": llm_res.provider,
                "model": llm_res.model,
                "tokens_used": llm_res.tokens_used,
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
            f"### 🤖 NOVA Assistant Commands (Phase 1 MVP)\n\n"
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
            f"*(Full free-form generative chat will be activated in Phase 2 with LLM integration)*"
        )

    def _save_conversation_turn(self, conversation_id: str, role: str, message: str) -> None:
        """Stores a chat message into the conversations table for persistence."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT INTO conversations (conversation_id, role, message, created_at)
            VALUES (?, ?, ?, ?)
        """
        try:
            execute_insert(sql, (conversation_id, role, message, now))
        except Exception as e:
            print(f"[Warning] Could not persist conversation turn: {e}")

    def get_conversation_history(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves stored turns for a specific conversation session."""
        sql = """
            SELECT role, message, created_at FROM conversations
            WHERE conversation_id = ?
            ORDER BY id ASC
            LIMIT ?
        """
        return fetch_all(sql, (conversation_id, limit))
