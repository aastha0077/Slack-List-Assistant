"""Detect and obtain text/transcript/audio/video content from Slack events."""
from dataclasses import dataclass
import logging
import os
import re
import time
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from slack_sdk.errors import SlackApiError

import transcription
from safe_diagnostics import log_exception, redact


logger = logging.getLogger(__name__)


class ContentError(ValueError):
    pass


class ContentAuthorizationError(ContentError):
    pass


class TemporaryContentError(ContentError):
    pass


@dataclass(frozen=True)
class IngestedContent:
    text: str
    source_type: str
    source_reference: str
    chunks: int = 1


@dataclass(frozen=True)
class NormalizedRequest:
    source_type: str
    raw_input: str
    transcript: str | None
    normalized_text: str
    requester_id: str | None
    channel_id: str | None
    thread_context: str | None
    source_file: str | None = None

    @property
    def text(self):
        return self.normalized_text


def normalize_text_request(text, *, requester_id=None, channel_id=None, thread_context=None):
    # Typed commands may legitimately be a numeric clarification ("1"). Do
    # not apply caption/VTT cleanup rules intended only for transcripts.
    normalized = re.sub(r"[ \t]+", " ", str(text or "").replace("\x00", ""))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        raise ContentError("The request contains no usable text.")
    return NormalizedRequest("text", str(text or ""), None, normalized,
                             requester_id, channel_id, thread_context)


def normalized_request(content, *, requester_id=None, channel_id=None, thread_context=None):
    normalized = normalize_transcript(content.text)
    if not normalized:
        raise ContentError("The transcript is empty or contains no usable text.")
    logger.info("transcript_normalized source=%s file_id=%s transcript_chars=%d",
                content.source_type, redact(content.source_reference), len(normalized))
    return NormalizedRequest(content.source_type, content.source_reference, normalized,
                             normalized, requester_id, channel_id, thread_context,
                             content.source_reference)


@dataclass(frozen=True)
class RequestRoute:
    """Immutable source classification made before either processing pipeline runs."""
    route: str
    source: str
    source_types: tuple[str, ...] = ()

    @property
    def is_shared_content(self):
        return self.route in {"media", "transcript"}


_MEDIA_REQUEST = re.compile(
    r"\b(?:extract|identify|find|capture|derive|turn|convert|create|add)\b.*"
    r"\b(?:action\s+items?|tasks?|todos?)\b|"
    r"\b(?:action\s+items?|tasks?|todos?)\b.*\b(?:from|in)\b.*"
    r"\b(?:audio|video|recording|meeting|transcript|attachment|file)\b", re.I | re.S)
_PREVIEW = re.compile(
    r"\b(?:preview|review\s+(?:first|only)|draft\s+only|extract\s+only|"
    r"do\s+not|don't|without)\b.*\b(?:create|add|change|mutate)\b", re.I | re.S)


def extraction_requested(text):
    return bool(_MEDIA_REQUEST.search(str(text or "")))


def preview_requested(text):
    return bool(_PREVIEW.search(str(text or "")))


def explicit_transcript_payload(text):
    """Detect delimited transcript content, never a mere transcript keyword."""
    value = str(text or "")
    if re.match(r"^\s*transcript\s*:\s*\S", value, re.I | re.S):
        return True
    if "\n" not in value:
        return False
    header, body = value.split("\n", 1)
    return bool(
        re.search(r"\btranscript\b[^:\n]{0,80}:\s*$", header, re.I)
        and body.strip()
    )


def classify_request(text, files=(), attachments=()):
    """Choose the text, media, or transcript pipeline from Slack event data.

    Instruction keywords and parser outcomes deliberately have no influence on
    this boundary. A Slack file entry is shared content even if its event stub
    needs ``files.info`` before its exact type is known.
    """
    kinds = []
    has_file = False
    for value in files or []:
        if not isinstance(value, dict):
            continue
        has_file = has_file or bool(value.get("id") or value.get("filetype")
                                    or value.get("mimetype") or value.get("mode"))
        kind = file_kind(value)
        if kind != "unsupported":
            kinds.append(kind)
    for value in attachments or []:
        if not isinstance(value, dict):
            continue
        kind = file_kind(value)
        if kind != "unsupported":
            has_file = True
            kinds.append(kind)

    unique = tuple(dict.fromkeys(kinds))
    if has_file:
        source = unique[0] if len(unique) == 1 else ("mixed" if unique else "file")
        route = "transcript" if unique and set(unique) == {"transcript"} else "media"
        return RequestRoute(route, source, unique)
    if explicit_transcript_payload(text):
        return RequestRoute("transcript", "transcript", ("transcript",))
    return RequestRoute("text", "text")


def should_ingest(text, files=(), attachments=()):
    """Compatibility predicate backed by the source-first request router."""
    return classify_request(text, files, attachments).is_shared_content


def file_kind(file_info):
    mime = str(file_info.get("mimetype") or "").split(";", 1)[0].strip().casefold()
    filetype = str(file_info.get("filetype") or "").strip().casefold().lstrip(".")
    mode = str(file_info.get("mode") or "").casefold()
    name = str(file_info.get("name") or file_info.get("title") or "")
    extension = os.path.splitext(name)[1].casefold().lstrip(".")
    generic_mime = mime in {"", "application/octet-stream", "binary/octet-stream",
                            "application/binary"}
    if not filetype and generic_mime:
        filetype = extension
    if mime.startswith("audio/") or mode == "audio" or filetype in {
            "mp3", "m4a", "wav", "ogg", "opus", "aac", "flac"}:
        return "audio"
    if mime.startswith("video/") or mode == "video" or filetype in {
            "mp4", "mov", "webm", "mkv", "avi", "mpeg"}:
        return "video"
    if mime.startswith("text/") or filetype in {"txt", "text", "vtt", "srt", "transcript"}:
        return "transcript"
    return "unsupported"


def _allowed_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    allowed = {"slack.com", "files.slack.com"}
    allowed.update(host.strip().casefold() for host in os.getenv("MEDIA_ALLOWED_HOSTS", "").split(",") if host.strip())
    host = parsed.hostname.casefold()
    return any(host == value or host.endswith("." + value) for value in allowed)


def _is_slack_url(url):
    host = (urlparse(str(url or "")).hostname or "").casefold()
    return host == "slack.com" or host.endswith(".slack.com")


def _safe_file_metadata(file_info):
    return {
        "file_id": str(file_info.get("id") or "unknown"),
        "file_name": redact(file_info.get("name") or file_info.get("title") or "unknown"),
        "mimetype": str(file_info.get("mimetype") or "unknown"),
        "file_size": file_info.get("size") or "unknown",
        "file_access": str(file_info.get("file_access") or "unknown"),
        "has_private_url": bool(file_info.get("url_private_download") or file_info.get("url_private")),
    }


def resolve_file_metadata(file_info, slack_client=None, attempts=3, sleeper=time.sleep):
    """Refresh an event file stub through files.info without changing identity."""
    current = dict(file_info or {})
    file_id = str(current.get("id") or "").strip()
    if not file_id or slack_client is None:
        logger.info("file_metadata_resolved source=event %s", _safe_file_metadata(current))
        return current
    transient = {"internal_error", "service_unavailable", "request_timeout", "ratelimited"}
    for attempt in range(1, attempts + 1):
        try:
            response = slack_client.files_info(file=file_id)
            fresh = dict(response.get("file") or {})
            if not fresh:
                raise ContentError("Slack returned no metadata for the shared file.")
            current.update(fresh)
            logger.info("file_metadata_resolved source=files.info %s", _safe_file_metadata(current))
            return current
        except SlackApiError as exc:
            error = str(exc.response.get("error") or "slack_api_error")
            needed = str(exc.response.get("needed") or "")
            logger.warning(
                "file_metadata_failed file_id=%s error=%s needed_scope=%s attempt=%d/%d",
                file_id, error, needed or "none", attempt, attempts,
            )
            if error == "missing_scope":
                raise ContentAuthorizationError(
                    "The Slack app is missing the files:read OAuth scope required to read uploaded files. "
                    "Add the bot scope and reinstall the app to the workspace.") from exc
            if error in {"invalid_auth", "not_authed", "token_expired", "token_revoked"}:
                raise ContentAuthorizationError("Slack could not authenticate access to the shared file.") from exc
            if error in {"access_denied", "no_permission", "not_visible"}:
                raise ContentAuthorizationError(
                    "The Slack app is not allowed to access this file or conversation.") from exc
            if error in {"file_deleted", "file_not_found"}:
                raise ContentError("The shared Slack file no longer exists or is unavailable.") from exc
            if error in transient and attempt < attempts:
                sleeper(min(2 ** (attempt - 1), 4))
                continue
            if error in transient:
                raise TemporaryContentError("Slack temporarily could not return the shared file metadata.") from exc
            log_exception(logger, "Slack file metadata request failed", exc,
                          function="resolve_file_metadata", file_id=file_id, slack_error=error)
            raise ContentError("Slack could not return metadata for the shared file.") from exc
        except ContentError:
            raise
        except Exception as exc:
            log_exception(logger, "Slack file metadata request failed", exc,
                          function="resolve_file_metadata", file_id=file_id, attempt=attempt)
            if attempt < attempts:
                sleeper(min(2 ** (attempt - 1), 4))
                continue
            raise TemporaryContentError("Slack temporarily could not return the shared file metadata.") from exc
    raise TemporaryContentError("Slack temporarily could not return the shared file metadata.")


class _SlackRedirectHandler(HTTPRedirectHandler):
    """Allow authenticated redirects only between Slack-owned HTTPS hosts."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_slack_url(newurl):
            raise ContentAuthorizationError("Slack redirected the private file to an unapproved host.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(file_info, bot_token, max_bytes=None, attempts=3, sleeper=time.sleep, opener=None):
    inline = file_info.get("content")
    if isinstance(inline, str):
        return inline.encode("utf-8")
    if isinstance(inline, bytes):
        return inline
    url = file_info.get("url_private_download") or file_info.get("url_private") or file_info.get("url")
    if not url or not _allowed_url(url):
        raise ContentError("The shared content is not available from an approved accessible URL.")
    max_bytes = int(max_bytes or os.getenv("MEDIA_MAX_BYTES", str(250 * 1024 * 1024)))
    is_slack = _is_slack_url(url)
    headers = {"Authorization": "Bearer " + bot_token} if bot_token and is_slack else {}
    if is_slack and not bot_token:
        raise ContentAuthorizationError("Slack file download requires the configured bot token.")
    opener = opener or build_opener(_SlackRedirectHandler()).open
    metadata = _safe_file_metadata(file_info)
    for attempt in range(1, attempts + 1):
        logger.info("file_download_started file_id=%s attempt=%d/%d", metadata["file_id"], attempt, attempts)
        try:
            with opener(Request(url, headers=headers), timeout=60) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                final_url = getattr(response, "url", url)
                if is_slack and not _is_slack_url(final_url):
                    raise ContentAuthorizationError("Slack redirected the private file to an unapproved host.")
                declared = response.headers.get("Content-Length")
                content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0]
                if declared and int(declared) > max_bytes:
                    raise ContentError("The shared file exceeds the configured media size limit.")
                data = response.read(max_bytes + 1)
                if content_type.casefold() == "text/html" and file_kind(file_info) in {"audio", "video"}:
                    raise ContentAuthorizationError(
                        "Slack returned a sign-in page instead of the private media file. "
                        "Verify the files:read bot scope and reinstall the app.")
                logger.info(
                    "file_download_completed file_id=%s download_status=%s content_type=%s "
                    "downloaded_bytes=%d",
                    metadata["file_id"], status, content_type or "unknown", len(data),
                )
                break
        except ContentError:
            raise
        except HTTPError as exc:
            logger.warning("file_download_failed file_id=%s status=%s attempt=%d/%d",
                           metadata["file_id"], exc.code, attempt, attempts)
            if exc.code in {401, 403}:
                raise ContentAuthorizationError(
                    "Slack denied access to the private file. Verify the files:read bot scope, "
                    "app installation, and channel membership.") from exc
            if exc.code == 404:
                raise ContentError("The shared Slack file no longer exists or is unavailable.") from exc
            if (exc.code == 429 or 500 <= exc.code < 600) and attempt < attempts:
                sleeper(min(2 ** (attempt - 1), 4))
                continue
            if exc.code == 429 or 500 <= exc.code < 600:
                raise TemporaryContentError("Slack temporarily could not download the shared file.") from exc
            raise ContentError(f"Slack file download failed with HTTP status {exc.code}.") from exc
        except (URLError, TimeoutError) as exc:
            log_exception(logger, "Slack file download failed", exc,
                          function="download", file_id=metadata["file_id"], attempt=attempt)
            if attempt < attempts:
                sleeper(min(2 ** (attempt - 1), 4))
                continue
            raise TemporaryContentError("Slack temporarily could not download the shared file.") from exc
        except Exception as exc:
            log_exception(logger, "Slack file download failed", exc,
                          function="download", file_id=metadata["file_id"], attempt=attempt)
            raise ContentError("Slack did not provide accessible file content.") from exc
    else:
        raise TemporaryContentError("Slack temporarily could not download the shared file.")
    if len(data) > max_bytes:
        raise ContentError("The shared file exceeds the configured media size limit.")
    if not data:
        raise ContentError("The shared Slack file is empty.")
    return data


def normalize_transcript(value):
    text = str(value or "").replace("\x00", "")
    text = re.sub(r"^WEBVTT.*?$", "", text, flags=re.I | re.M)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\d{1,2}:\d{2}(?::\d{2})?[.,]\d+\s+-->.*$", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _pasted_transcript(text):
    lines = str(text or "").strip().splitlines()
    if len(lines) > 1 and extraction_requested(text):
        content_lines = [line for line in lines if not extraction_requested(line)]
        return "\n".join(content_lines).strip()
    marker = re.search(r"\btranscript\s*:\s*", str(text or ""), re.I)
    if marker:
        return str(text or "")[marker.end():].strip()
    inline = re.match(r"^.*?\b(?:action\s+items?|tasks?|todos?)\b[^:]*:\s*(.+)$",
                      str(text or ""), re.I | re.S)
    return inline.group(1).strip() if inline and extraction_requested(text) else ""


def ingest(text, files=(), attachments=(), bot_token="", downloader=download,
           transcriber=transcription.transcribe_bytes, slack_client=None,
           metadata_resolver=resolve_file_metadata):
    sources = list(files or [])
    for attachment in attachments or []:
        if isinstance(attachment, dict) and file_kind(attachment) != "unsupported":
            sources.append(attachment)
    logger.info("shared_content_received files=%d attachments=%d text_chars=%d",
                len(files or []), len(attachments or []), len(str(text or "")))
    logger.info("media_received files=%d attachments=%d",
                len(files or []), len(attachments or []))
    results, errors = [], []
    for source in sources:
        display_label = str(source.get("title") or source.get("name") or "shared file")
        try:
            source = metadata_resolver(source, slack_client)
        except ContentError as exc:
            errors.append(f"{display_label}: {exc}")
            continue
        kind = file_kind(source)
        file_id = str(source.get("id") or "unknown")
        logger.info("media_type_detected file_id=%s type=%s mimetype=%s filetype=%s",
                    file_id, kind, str(source.get("mimetype") or "unknown"),
                    str(source.get("filetype") or "unknown"))
        display_label = str(source.get("title") or source.get("name") or "shared file")
        if kind == "unsupported":
            errors.append(f"{display_label}: unsupported content type")
            continue
        try:
            logger.info("media_download_started file_id=%s source_type=%s", file_id, kind)
            raw = downloader(source, bot_token)
            logger.info("media_download_completed file_id=%s source_type=%s bytes=%d",
                        file_id, kind, len(raw))
            if kind == "transcript":
                transcript = normalize_transcript(raw.decode("utf-8-sig"))
                chunks = 1
            else:
                logger.info("transcription_started file_id=%s source_type=%s bytes=%d",
                            file_id, kind, len(raw))
                if transcriber is transcription.transcribe_bytes:
                    observed = transcriber(
                        raw, kind, str(source.get("mimetype") or ""), file_id=file_id)
                else:
                    observed = transcriber(raw, kind, str(source.get("mimetype") or ""))
                transcript, chunks = normalize_transcript(observed.text), observed.chunks
                logger.info("transcription_completed file_id=%s media_type=%s chunk_count=%d transcript_chars=%d",
                            file_id, kind, chunks, len(transcript))
            if not transcript:
                raise ContentError("The transcript is empty or contains no usable text.")
            logger.info("transcript_normalized source=%s file_id=%s transcript_chars=%d",
                        kind, file_id, len(transcript))
            results.append(IngestedContent(transcript, kind, file_id, chunks))
        except (ContentError, transcription.TranscriptionError, UnicodeDecodeError) as exc:
            if kind in {"audio", "video"}:
                logger.warning(
                    "transcription_failed file_id=%s media_type=%s stage=%s "
                    "error_type=%s message=%s",
                    file_id, kind, getattr(exc, "stage", "transcript_normalization"),
                    type(exc).__name__, redact(exc),
                )
            errors.append(f"{display_label}: {exc}")
    pasted = _pasted_transcript(text)
    if pasted:
        normalized = normalize_transcript(pasted)
        if normalized:
            results.append(IngestedContent(normalized, "transcript", "Slack message"))
    if not results:
        if errors:
            raise ContentError("; ".join(errors))
        raise ContentError("Attach accessible audio/video, or include transcript text to extract action items.")
    return results, errors
