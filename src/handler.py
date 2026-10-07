"""AWS Lambda HTTP entry point for the shared Bolt application."""
import base64
import json
import logging

logger = logging.getLogger("slack_list.lambda")
try:
    from .app import get_app
    from .config import RuntimeConfig
except ImportError:
    from src.app import get_app
    from src.config import RuntimeConfig

_request_handler = None


def _get_request_handler():
    global _request_handler
    if _request_handler is None:
        from slack_bolt.adapter.aws_lambda import SlackRequestHandler
        RuntimeConfig.from_environment().validate(socket_mode=False)
        _request_handler = SlackRequestHandler(get_app())
    return _request_handler


def lambda_handler(event, context):
    # Function URLs use API Gateway payload v2. Bolt's AWS adapter reads its
    # requestContext.http.method, body, headers and isBase64Encoded directly.
    http = (event.get("requestContext") or {}).get("http") or {}
    method = http.get("method") or (event.get("requestContext") or {}).get("httpMethod")
    kind = "unknown"
    event_type = "unknown"
    if method == "POST":
        try:
            body = event.get("body") or ""
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body).decode("utf-8")
            if str((event.get("headers") or {}).get("content-type", "")).startswith("application/json"):
                payload = json.loads(body)
                if isinstance(payload, dict):
                    kind = str(payload.get("type") or "unknown")
                    event_type = str((payload.get("event") or {}).get("type") or "unknown")
            elif "command=" in body:
                kind = "slash_command"
        except (ValueError, TypeError, UnicodeError):
            kind = "invalid_payload"
    logger.info("lambda_request_received method=%s request_type=%s event_type=%s",
                method or "unknown", kind, event_type)
    # Do not log the body or headers: both can contain credentials or user text.
    logger.info("bolt_request_handling_started request_type=%s event_type=%s", kind, event_type)
    try:
        response = _get_request_handler().handle(event, context)
    except Exception as exc:
        logger.error("bolt_request_handling_failed error_type=%s", type(exc).__name__)
        raise
    logger.info("bolt_request_handling_completed request_type=%s event_type=%s status=%s",
                kind, event_type, response.get("statusCode", "unknown"))
    logger.info("lambda_response_returned status=%s", response.get("statusCode", "unknown"))
    return response
