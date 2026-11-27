"""
Minimal input guardrail for the task a workflow receives.

The task is the one piece of text the user types, and every agent in the workflow reads it, so it
is checked once, before any model is called. Same idea as Module 2's chatbot: cheap checks that
block obvious misuse and redact secrets, not a complete policy.

For real production use, swap these for a moderation API, a PII detector (Presidio, Comprehend)
and a scope classifier tuned for your domain.
"""

import re

# Very small list — enough to demo the pattern, not an exhaustive policy
DENY_PATTERNS = [
    r"\bphishing\b",
    r"\bmalware\b",
    r"how (do|to) (i )?(make|build) (a |an )?(bomb|explosive|weapon)",
    r"ignore (all |any )?(previous|prior|above) instructions",
]

PII_PATTERNS = [
    (re.compile(r"\b\d{16}\b"), "[REDACTED-CARD]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]"),
    (re.compile(r"sk-(proj-)?[A-Za-z0-9-_]{20,}"), "[REDACTED-APIKEY]"),
]


def check_input(text: str) -> dict:
    """
    Returns { "ok": bool, "reason": str | None, "redacted": str }.

    A task that matches a deny pattern is blocked; otherwise secrets and card-like numbers are
    replaced before the task reaches any agent.
    """
    lower = text.lower()
    for pattern in DENY_PATTERNS:
        if re.search(pattern, lower):
            return {"ok": False, "reason": "unsafe", "redacted": text}

    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return {"ok": True, "reason": None, "redacted": redacted}
