import re
from fastapi import HTTPException, status

# Prohibited keywords and harmful patterns across safety categories
PROHIBITED_KEYWORDS = [
    # Illegal & Weapons
    r"\bbomb\b",
    r"\bexplosive\b",
    r"\bterrorist\b",
    r"\bterrorism\b",
    r"\bassassinate\b",
    # Hate Speech & Harassment
    r"\bhate speech\b",
    r"\bgenocide\b",
    # Severe Violence & Exploitation
    r"\btorture\b",
    r"\bsuicide\b",
    r"\bself-harm\b",
    r"\bchild abuse\b",
    r"\bcp\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PROHIBITED_KEYWORDS]


def scan_story_content(title: str, content: str) -> None:
    """
    Scans story title and body for prohibited terms.
    Raises HTTPException(400) if prohibited terms are detected.
    """
    full_text = f"{title} {content}".lower()
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(full_text):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Story content violates platform safety guidelines and cannot be processed.",
            )
