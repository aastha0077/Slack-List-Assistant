"""Central failure classification and bounded recovery policy."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import logging
import sqlite3
import time
import uuid
from urllib.error import HTTPError, URLError

from slack_sdk.errors import SlackApiError

from safe_diagnostics import redact


logger = logging.getLogger(__name__)
_request_id = ContextVar("request_id", default=None)


@dataclass(frozen=True)
class Failure:
    category: str
    recoverable: bool
    safe_to_retry: bool
    retry_count: int
    user_safe_message: str


class VerificationError(RuntimeError):
    """A mutation was attempted but its resulting Slack state was not confirmed."""


def begin_request(request_id=None):
    value = str(request_id or uuid.uuid4().hex)
    return value, _request_id.set(value)


def end_request(token):
    _request_id.reset(token)


def current_request_id():
    return _request_id.get() or "unscoped"


def _slack_status(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    data = getattr(response, "data", None)
    if isinstance(response, dict):
        data = response
    error = str((data or {}).get("error") or "") if isinstance(data, dict) else ""
    retry_after = None
    headers = getattr(response, "headers", None) or {}
    try:
        retry_after = float(headers.get("Retry-After")) if headers.get("Retry-After") else None
    except (TypeError, ValueError):
        pass
    return status, error.casefold(), retry_after


def classify(exc, retry_count=0):
    message = str(exc or "").casefold()
    status, slack_error, _ = _slack_status(exc)
    if isinstance(exc, PermissionError):
        return Failure("permission_denied", False, False, retry_count,
                       "You don't have permission to update this task.")
    if isinstance(exc, VerificationError):
        return Failure("verification_error", False, False, retry_count,
                       "The update was sent, but I couldn't verify the final task state, so I won't report it as successful.")
    if isinstance(exc, sqlite3.Error):
        return Failure("persistence_error", False, False, retry_count,
                       "I couldn't safely save the operation state. No successful change is being reported.")
    if type(exc).__name__ == "TranscriptionError" or "transcription" in type(exc).__module__:
        return Failure("transcription_error", False, False, retry_count,
                       "I couldn't transcribe the recording reliably. No task changes were made.")
    if isinstance(exc, SlackApiError) and (status == 429 or slack_error in {"ratelimited", "rate_limited"}):
        return Failure("slack_rate_limit", True, True, retry_count,
                       "Slack is temporarily rate limiting requests. No changes were confirmed.")
    if status == 429 or "rate limit" in message or "ratelimited" in message:
        return Failure("slack_rate_limit", True, True, retry_count,
                       "Slack is temporarily rate limiting requests. No changes were confirmed.")
    http_transient = isinstance(exc, HTTPError) and exc.code in {429, 500, 502, 503, 504}
    if (status in {500, 502, 503, 504} or http_transient
            or isinstance(exc, (TimeoutError, ConnectionError))
            or (isinstance(exc, URLError) and not isinstance(exc, HTTPError))
            or any(value in message for value in ("service_unavailable", "temporarily unavailable", "timed out", "timeout"))):
        return Failure("temporary_service_error", True, True, retry_count,
                       "I couldn't complete that update right now. No changes were confirmed.")
    if "duplicate" in message or "already exists" in message:
        return Failure("duplicate", False, False, retry_count,
                       "That task already exists. No duplicate was created.")
    if any(value in message for value in ("multiple matching", "which one", "ambiguous", "please clarify")):
        return Failure("ambiguity", False, False, retry_count,
                       "I found multiple matching tasks. Please specify which one.")
    if isinstance(exc, (ValueError, TypeError)):
        return Failure("validation_error", False, False, retry_count,
                       "That request contains an invalid value. No changes were made.")
    if any(value in message for value in ("ollama", "llm", "language model")):
        return Failure("llm_unavailable", True, True, retry_count,
                       "I couldn't understand that request right now. No task changes were made.")
    return Failure("unknown_error", False, False, retry_count,
                   "I couldn't complete that request safely. No changes were confirmed.")


def run(operation, *, idempotent=False, max_retries=2, sleeper=time.sleep,
        base_delay=0.25, operation_name="external_operation"):
    """Run an operation with retries only when classification and idempotency allow."""
    retry_count = 0
    while True:
        try:
            result = operation()
            if retry_count:
                logger.info(
                    "recovery_completed request_id=%s operation=%s retry_count=%d success=true error_category=none",
                    current_request_id(), operation_name, retry_count)
            return result
        except Exception as exc:
            failure = classify(exc, retry_count)
            can_retry = bool(
                idempotent and failure.recoverable and failure.safe_to_retry
                and retry_count < max(0, int(max_retries)))
            logger.warning(
                "recovery_started request_id=%s operation=%s retry_count=%d success=false error_category=%s retrying=%s error_type=%s message=%s",
                current_request_id(), operation_name, retry_count,
                failure.category, str(can_retry).lower(), type(exc).__name__, redact(exc))
            if not can_retry:
                logger.warning(
                    "recovery_completed request_id=%s operation=%s retry_count=%d success=false error_category=%s",
                    current_request_id(), operation_name, retry_count, failure.category)
                raise
            _, _, retry_after = _slack_status(exc)
            delay = retry_after if retry_after is not None else base_delay * (2 ** retry_count)
            retry_count += 1
            sleeper(max(0.0, delay))
