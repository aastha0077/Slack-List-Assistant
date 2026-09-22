"""Redacted exception diagnostics for external service boundaries."""
from __future__ import annotations

import logging
import os
import re


_SECRET_NAME = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|app[_-]?token|bot[_-]?token|"
    r"oauth[_-]?token|client[_-]?secret|password)\b([\s:=\"']+)([^\s,;\"']+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_SLACK_TOKEN = re.compile(r"\bxox[a-z]-[A-Za-z0-9-]+\b", re.I)


def redact(value) -> str:
    """Remove credential-shaped values without logging configuration contents."""
    text = str(value or "")
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _SLACK_TOKEN.sub("[REDACTED_SLACK_TOKEN]", text)
    text = _SECRET_NAME.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text)
    # Replace configured secrets if a dependency copied one into its exception.
    for name, secret in os.environ.items():
        if not secret or len(secret) < 8:
            continue
        if any(marker in name.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            text = text.replace(secret, f"[REDACTED_{name.upper()}]")
    return text


def log_exception(logger: logging.Logger, event: str, exc: BaseException, **metadata) -> None:
    """Log a complete traceback while sanitizing the exception message.

    Passing a replacement exception with the original traceback gives standard
    ``exc_info`` formatting without allowing an HTTP client to expose headers or
    credentials embedded in its exception string.
    """
    details = " ".join(f"{key}={redact(value)}" for key, value in metadata.items())
    safe_message = redact(str(exc)) or "(no exception message)"
    safe_exception = RuntimeError(safe_message).with_traceback(exc.__traceback__)
    logger.error(
        "%s%s exception_type=%s exception_message=%s",
        event,
        f" {details}" if details else "",
        type(exc).__name__,
        safe_message,
        exc_info=(RuntimeError, safe_exception, exc.__traceback__),
    )
