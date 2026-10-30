import time
from collections import defaultdict
from fastapi import HTTPException, status
from src.config import settings

# In-memory sliding window fallback store for rate limiting and token blocklists
# Format: { key: [timestamp, ...] }
_attempts_store: dict[str, list[float]] = defaultdict(list)
_revoked_tokens: dict[str, float] = {}  # { jti: expiry_timestamp }


def check_login_rate_limit(client_ip: str, identifier: str) -> None:
    """
    Checks if login attempts exceed threshold (default 5 attempts in 60s).
    Raises HTTPException(429) if exceeded.
    """
    key = f"rate_limit:login:{client_ip}:{identifier.lower()}"
    now = time.time()
    window = settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS
    max_attempts = settings.LOGIN_RATE_LIMIT_MAX_ATTEMPTS

    # Clean expired timestamps
    _attempts_store[key] = [t for t in _attempts_store[key] if now - t < window]

    if len(_attempts_store[key]) >= max_attempts:
        oldest = _attempts_store[key][0]
        retry_after = int(window - (now - oldest)) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Please try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


def record_failed_login_attempt(client_ip: str, identifier: str) -> None:
    """Records a failed login attempt for the sliding window."""
    key = f"rate_limit:login:{client_ip}:{identifier.lower()}"
    _attempts_store[key].append(time.time())


def reset_login_rate_limit(client_ip: str, identifier: str) -> None:
    """Resets attempts on successful login."""
    key = f"rate_limit:login:{client_ip}:{identifier.lower()}"
    _attempts_store.pop(key, None)


def revoke_token(jti: str, ttl_seconds: int) -> None:
    """Revokes a JWT token by adding its jti to the blocklist with TTL."""
    _revoked_tokens[jti] = time.time() + ttl_seconds


def is_token_revoked(jti: str) -> bool:
    """Checks whether token jti has been added to the revocation blocklist."""
    now = time.time()
    # Clean up expired entries
    expired = [k for k, exp in _revoked_tokens.items() if now > exp]
    for k in expired:
        _revoked_tokens.pop(k, None)
    return jti in _revoked_tokens
