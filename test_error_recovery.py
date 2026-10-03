import logging
from urllib.error import HTTPError

import pytest
from slack_sdk.errors import SlackApiError

import error_recovery


def test_429_recovery_is_bounded_and_succeeds():
    calls = []
    sleeps = []

    def operation():
        calls.append(1)
        if len(calls) == 1:
            raise SlackApiError("rate limited", {"error": "ratelimited"})
        return "ok"

    assert error_recovery.run(
        operation, idempotent=True, sleeper=sleeps.append,
        operation_name="test_429") == "ok"
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_503_recovery_uses_bounded_backoff():
    calls = []
    sleeps = []

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise HTTPError("https://slack.test", 503, "unavailable", {}, None)
        return "ok"

    assert error_recovery.run(
        operation, idempotent=True, max_retries=2, sleeper=sleeps.append,
        operation_name="test_503") == "ok"
    assert len(calls) == 3
    assert sleeps == [.25, .5]


def test_retry_limit_is_enforced():
    calls = []

    def operation():
        calls.append(1)
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        error_recovery.run(
            operation, idempotent=True, max_retries=2, sleeper=lambda _: None)
    assert len(calls) == 3


@pytest.mark.parametrize("exc,category", [
    (PermissionError("denied"), "permission_denied"),
    (ValueError("invalid date"), "validation_error"),
    (RuntimeError("duplicate task already exists"), "duplicate"),
])
def test_non_transient_failures_never_retry(exc, category):
    calls = []

    def operation():
        calls.append(1)
        raise exc

    with pytest.raises(type(exc)):
        error_recovery.run(operation, idempotent=True, sleeper=lambda _: None)
    assert len(calls) == 1
    failure = error_recovery.classify(exc)
    assert failure.category == category
    assert not failure.safe_to_retry


def test_non_idempotent_operation_never_retries_transient_failure():
    calls = []

    def operation():
        calls.append(1)
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        error_recovery.run(operation, idempotent=False, sleeper=lambda _: None)
    assert len(calls) == 1


def test_recovery_logs_request_id_and_redacts_secrets(caplog, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-super-secret-value")
    request_id, token = error_recovery.begin_request("REQ-123")
    try:
        with caplog.at_level(logging.WARNING, logger="error_recovery"):
            with pytest.raises(TimeoutError):
                error_recovery.run(
                    lambda: (_ for _ in ()).throw(
                        TimeoutError("Bearer xoxb-super-secret-value timed out")),
                    idempotent=False, operation_name="secret_test")
    finally:
        error_recovery.end_request(token)
    assert request_id == "REQ-123"
    assert "request_id=REQ-123" in caplog.text
    assert "xoxb-super-secret-value" not in caplog.text
    assert "[REDACTED" in caplog.text
