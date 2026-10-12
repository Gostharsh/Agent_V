
import re


SUPPORTED_CATEGORIES = {"preference", "project_fact"}

SENSITIVE_PATTERNS = [
    re.compile(
    r"\bpassword\b(?:\s+is)?\s*[:=]",
    re.IGNORECASE,
            ),
    re.compile(r"\bapi[_ -]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"\btoken\s*[:=]", re.IGNORECASE),
    re.compile(r"\bsecret\s*[:=]", re.IGNORECASE),
]


def should_store_memory(text: str, category: str) -> bool:
    """Return whether a candidate qualifies for long-term memory."""
    if not isinstance(text, str) or not text.strip():
        return False

    if not isinstance(category, str) or category not in SUPPORTED_CATEGORIES:
        return False

    if any(pattern.search(text) for pattern in SENSITIVE_PATTERNS):
        return False

    return True




def filter_memory_candidate(
    text: str,
    category: str,
) -> str | None:
    """Return normalized candidate text if accepted, otherwise None."""
    if not should_store_memory(text, category):
        return None

    return text.strip()

