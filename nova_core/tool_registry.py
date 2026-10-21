"""
NOVA Modular Tool Registry
Enforces strict argument validation, permission checks, safety confirmations,
and auditable execution. Never executes arbitrary shell/code.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import inspect

from database.db import log_action
from tools.task_manager import (
    create_task,
    get_tasks,
    complete_task,
    get_task_statistics,
)
from memory.memory_manager import (
    save_memory,
    search_memories,
    get_all_memories,
)


class ToolResult:
    """Standardized result returned by all tool invocations."""
    def __init__(
        self,
        status: str,  # 'success', 'failed', 'awaiting_confirmation'
        message: str,
        data: Optional[Any] = None,
        tool_name: str = "",
    ):
        self.status = status
        self.message = message
        self.data = data
        self.tool_name = tool_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "tool_name": self.tool_name,
        }

    def __repr__(self) -> str:
        return f"<ToolResult {self.tool_name} status={self.status}: {self.message}>"


class ToolDefinition:
    """Schema and handler definition for a registered tool."""
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        handler: Callable[..., Any],
        requires_confirmation: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.requires_confirmation = requires_confirmation


class ToolRegistry:
    """Central registry and execution engine for NOVA tools."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = db_path
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        handler: Callable[..., Any],
        requires_confirmation: bool = False,
    ) -> None:
        """Registers a tool with validation specifications."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            requires_confirmation=requires_confirmation,
        )

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns schemas of all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "requires_confirmation": t.requires_confirmation,
            }
            for t in self._tools.values()
        ]

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_confirmed: bool = False,
        user_id: Optional[str] = None,
        db_path: Optional[Path | str] = None,
    ) -> ToolResult:
        """
        Validates arguments and executes the tool safely, enforcing user isolation
        and target database routing.
        """
        effective_db = db_path or self.db_path
        tool = self.get_tool(tool_name)
        if not tool:
            log_action("tool_call", f"Requested unknown tool: '{tool_name}'", status="failed", db_path=effective_db)
            return ToolResult(
                status="failed",
                message=f"Tool '{tool_name}' is not recognized by NOVA.",
                tool_name=tool_name,
            )

        # Destructive confirmation check
        if tool.requires_confirmation and not user_confirmed:
            log_action("tool_call", f"Action '{tool_name}' requires confirmation", status="proposed", db_path=effective_db)
            return ToolResult(
                status="awaiting_confirmation",
                message=f"Action '{tool_name}' requires explicit user confirmation before proceeding.",
                data={"arguments": arguments},
                tool_name=tool_name,
            )

        # Validate arguments against parameter schema
        validated_args: Dict[str, Any] = {}
        for param_name, param_meta in tool.parameters.items():
            is_required = param_meta.get("required", False)
            expected_type = param_meta.get("type", str)
            default = param_meta.get("default", None)

            if param_name not in arguments or arguments[param_name] is None:
                if is_required:
                    log_action(tool_name, f"Missing required parameter '{param_name}'", status="failed", db_path=effective_db)
                    return ToolResult(
                        status="failed",
                        message=f"Parameter '{param_name}' is required for tool '{tool_name}'.",
                        tool_name=tool_name,
                    )
                validated_args[param_name] = default
            else:
                raw_val = arguments[param_name]
                # Type validation
                if expected_type == int:
                    try:
                        validated_args[param_name] = int(raw_val)
                    except (ValueError, TypeError):
                        return ToolResult(
                            status="failed",
                            message=f"Parameter '{param_name}' must be an integer, got: {raw_val}",
                            tool_name=tool_name,
                        )
                elif expected_type == str:
                    validated_args[param_name] = str(raw_val)
                elif expected_type == bool:
                    validated_args[param_name] = bool(raw_val)
                else:
                    validated_args[param_name] = raw_val

        # Forward user_id and db_path if handler supports them
        sig = inspect.signature(tool.handler)
        if "user_id" in sig.parameters and user_id is not None:
            validated_args["user_id"] = user_id
        if "db_path" in sig.parameters:
            validated_args["db_path"] = effective_db

        # Execute handler with validated arguments
        try:
            result_data = tool.handler(**validated_args)
            log_action(tool_name, f"Successfully executed tool '{tool_name}'", status="success", db_path=effective_db)
            return ToolResult(
                status="success",
                message=f"Successfully executed '{tool_name}'.",
                data=result_data,
                tool_name=tool_name,
            )
        except Exception as e:
            log_action(tool_name, f"Execution failed: {str(e)}", status="failed", db_path=effective_db)
            return ToolResult(
                status="failed",
                message=f"Execution error in '{tool_name}': {str(e)}",
                tool_name=tool_name,
            )

    # ============================================================
    # Default Tool Handlers
    # ============================================================
    def _register_default_tools(self) -> None:
        # 1. create_task
        self.register(
            name="create_task",
            description="Creates a new task in the user's task manager.",
            parameters={
                "title": {"type": str, "required": True, "description": "The title of the task"},
                "description": {"type": str, "required": False, "default": "", "description": "Details or notes"},
                "priority": {"type": str, "required": False, "default": "Medium", "description": "Low, Medium, or High"},
                "due_date": {"type": str, "required": False, "default": None, "description": "Due date in YYYY-MM-DD"},
            },
            handler=self._tool_create_task,
        )

        # 2. list_tasks
        self.register(
            name="list_tasks",
            description="Retrieves tasks filtered by status or priority.",
            parameters={
                "status": {"type": str, "required": False, "default": "pending", "description": "'pending', 'completed', or 'all'"},
                "priority": {"type": str, "required": False, "default": None, "description": "Filter by priority"},
            },
            handler=self._tool_list_tasks,
        )

        # 3. complete_task
        self.register(
            name="complete_task",
            description="Marks a task as completed given its integer ID.",
            parameters={
                "task_id": {"type": int, "required": True, "description": "The integer ID of the task to complete"},
            },
            handler=self._tool_complete_task,
        )

        # 4. save_memory
        self.register(
            name="save_memory",
            description="Saves a personal memory that the user explicitly wants NOVA to retain.",
            parameters={
                "content": {"type": str, "required": True, "description": "Information to remember"},
                "category": {"type": str, "required": False, "default": "General", "description": "Category name"},
            },
            handler=self._tool_save_memory,
        )

        # 5. search_memory
        self.register(
            name="search_memory",
            description="Searches saved user memories by keyword or topic.",
            parameters={
                "query": {"type": str, "required": True, "description": "Search keyword or phrase"},
            },
            handler=self._tool_search_memory,
        )

        # 6. get_current_time
        self.register(
            name="get_current_time",
            description="Gets the current local date and time.",
            parameters={},
            handler=self._tool_get_current_time,
        )

        # 7. analyze_productivity
        self.register(
            name="analyze_productivity",
            description="Computes task completion rates, counts, and upcoming deadlines.",
            parameters={},
            handler=self._tool_analyze_productivity,
        )

    # Tool Handlers Implementation
    def _tool_create_task(
        self,
        title: str,
        description: str = "",
        priority: str = "Medium",
        due_date: Optional[str] = None,
        user_id: Optional[str] = None,
        db_path: Optional[Path | str] = None,
    ):
        effective_db = db_path or self.db_path
        return create_task(
            title=title,
            description=description,
            priority=priority,
            due_date=due_date,
            user_id=user_id,
            db_path=effective_db,
        )

    def _tool_list_tasks(
        self,
        status: Optional[str] = "pending",
        priority: Optional[str] = None,
        user_id: Optional[str] = None,
        db_path: Optional[Path | str] = None,
    ):
        effective_db = db_path or self.db_path
        return get_tasks(
            status=status,
            priority=priority,
            user_id=user_id,
            db_path=effective_db,
        )

    def _tool_complete_task(
        self,
        task_id: int,
        user_id: Optional[str] = None,
        db_path: Optional[Path | str] = None,
    ):
        effective_db = db_path or self.db_path
        ok = complete_task(task_id, user_id=user_id, db_path=effective_db)
        if not ok:
            raise ValueError(f"Task #{task_id} not found or could not be completed.")
        return {"task_id": task_id, "status": "completed"}

    def _tool_save_memory(
        self,
        content: str,
        category: str = "General",
        user_id: Optional[str] = None,
        db_path: Optional[Path | str] = None,
    ):
        effective_db = db_path or self.db_path
        return save_memory(
            content=content,
            category=category,
            user_id=user_id,
            db_path=effective_db,
        )

    def _tool_search_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
        db_path: Optional[Path | str] = None,
    ):
        effective_db = db_path or self.db_path
        return search_memories(
            query=query,
            user_id=user_id,
            db_path=effective_db,
        )

    def _tool_get_current_time(self):
        now = datetime.now()
        return {
            "current_time": now.strftime("%I:%M:%S %p"),
            "current_date": now.strftime("%A, %B %d, %Y"),
            "iso": now.isoformat(),
        }

    def _tool_analyze_productivity(
        self,
        user_id: Optional[str] = None,
        db_path: Optional[Path | str] = None,
    ):
        effective_db = db_path or self.db_path
        return get_task_statistics(user_id=user_id, db_path=effective_db)


# Singleton instance of registry for application use
default_tool_registry = ToolRegistry()
