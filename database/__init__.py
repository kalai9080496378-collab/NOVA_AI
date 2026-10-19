"""
Database package for NOVA AI Assistant.
"""

from database.db import (
    get_connection,
    init_db,
    execute_query,
    execute_insert,
    fetch_all,
    fetch_one,
    log_action,
)

__all__ = [
    "get_connection",
    "init_db",
    "execute_query",
    "execute_insert",
    "fetch_all",
    "fetch_one",
    "log_action",
]
