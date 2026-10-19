"""
NOVA Authentication and User Management Module
Provides secure password hashing (PBKDF2-HMAC-SHA256), user registration,
credential verification, and user isolation for chat threads and conversations.
"""

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from database.db import execute_insert, execute_query, fetch_one, fetch_all, log_action


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Returns (hex_hash, hex_salt).
    """
    if not password:
        raise ValueError("Password cannot be empty.")
    
    salt_hex = salt or secrets.token_hex(16)
    salt_bytes = bytes.fromhex(salt_hex)
    
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        100_000,
    )
    return key.hex(), salt_hex


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verifies a plain password against the stored hash and salt."""
    if not password or not password_hash or not salt:
        return False
    computed_hash, _ = hash_password(password, salt=salt)
    return secrets.compare_digest(computed_hash, password_hash)


def create_user(
    username: str,
    password: str,
    full_name: Optional[str] = None,
    user_id: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Registers a new user account with secure credentials.
    """
    clean_username = (username or "").strip().lower()
    if not clean_username or len(clean_username) < 3:
        raise ValueError("Username must be at least 3 characters long.")

    if not password or len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    # Check if username already exists
    existing = get_user_by_username(clean_username, db_path=db_path)
    if existing:
        raise ValueError(f"Username '{clean_username}' is already taken.")

    uid = user_id or f"user_{uuid.uuid4().hex[:12]}"
    pwd_hash, salt = hash_password(password)
    now = datetime.now(timezone.utc).isoformat()
    disp_name = (full_name or clean_username).strip()

    sql = """
        INSERT INTO users (id, username, password_hash, salt, full_name, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    execute_insert(sql, (uid, clean_username, pwd_hash, salt, disp_name, now, now), db_path=db_path)
    log_action("create_user", f"Registered user '{clean_username}' [{uid}]", status="success", db_path=db_path)

    return {
        "id": uid,
        "username": clean_username,
        "full_name": disp_name,
        "created_at": now,
    }


def authenticate_user(
    username: str,
    password: str,
    db_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Authenticates a user by username and password.
    Returns user dict on success, or None on failure.
    """
    clean_username = (username or "").strip().lower()
    if not clean_username or not password:
        return None

    user = get_user_by_username(clean_username, db_path=db_path)
    if not user:
        return None

    if verify_password(password, user["password_hash"], user["salt"]):
        log_action("user_login", f"Successful login for '{clean_username}'", status="success", db_path=db_path)
        return {
            "id": user["id"],
            "username": user["username"],
            "full_name": user["full_name"],
            "created_at": user["created_at"],
        }
    
    log_action("user_login_failed", f"Failed login attempt for '{clean_username}'", status="failed", db_path=db_path)
    return None


def get_user_by_username(username: str, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a user record by username."""
    if not username:
        return None
    sql = "SELECT * FROM users WHERE username = ?"
    return fetch_one(sql, (username.strip().lower(),), db_path=db_path)


def get_user_by_id(user_id: str, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a user record by ID."""
    if not user_id:
        return None
    sql = "SELECT id, username, full_name, created_at, updated_at FROM users WHERE id = ?"
    return fetch_one(sql, (user_id,), db_path=db_path)


def ensure_default_users(db_path: Optional[Path | str] = None) -> None:
    """Seeds a default demo account if no users exist in the database."""
    count_res = fetch_one("SELECT count(*) as c FROM users", db_path=db_path)
    if count_res and count_res["c"] == 0:
        create_user(
            username="student",
            password="password123",
            full_name="MSc AI Student",
            user_id="user_student_default",
            db_path=db_path,
        )
