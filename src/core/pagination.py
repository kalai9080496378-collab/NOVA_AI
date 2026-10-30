import base64
import json
import uuid
from datetime import datetime


def encode_cursor(published_at: datetime, item_id: uuid.UUID) -> str:
    """Encodes (published_at, id) into a base64 string cursor."""
    payload = {
        "p": published_at.isoformat(),
        "id": str(item_id),
    }
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def decode_cursor(cursor_str: str) -> tuple[datetime, uuid.UUID] | None:
    """Decodes a base64 cursor string into (published_at, id). Returns None if invalid."""
    try:
        raw = base64.urlsafe_b64decode(cursor_str.encode("utf-8"))
        payload = json.loads(raw.decode("utf-8"))
        published_at = datetime.fromisoformat(payload["p"])
        item_id = uuid.UUID(payload["id"])
        return published_at, item_id
    except Exception:
        return None
