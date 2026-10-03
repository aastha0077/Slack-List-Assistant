from datetime import date
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import action_item_extraction as extraction
import content_ingestion as ingestion
import intent_parser
import transcription
from slack_sdk.errors import SlackApiError


def test_content_detection_uses_slack_metadata_not_only_filename():
    assert ingestion.file_kind({"mimetype": "audio/ogg", "name": "no-extension"}) == "audio"
    assert ingestion.file_kind({"mimetype": "video/mp4", "name": "notes.txt"}) == "video"
    assert ingestion.file_kind({"mimetype": "application/octet-stream", "filetype": "vtt"}) == "transcript"
    assert ingestion.file_kind({"mimetype": "application/pdf", "name": "recording.mp3"}) == "unsupported"
    assert ingestion.file_kind({
        "mimetype": "application/octet-stream", "name": "Assign and Prioritize Task.mp3"
    }) == "audio"
    assert ingestion.file_kind({"name": "voice-note.MP3"}) == "audio"


@pytest.mark.parametrize("filetype", ["mp4", "mov", "mkv", "webm"])
def test_common_video_types_are_supported(filetype):
    assert ingestion.file_kind({"filetype": filetype}) == "video"


@pytest.mark.parametrize("filetype", ["mp3", "wav", "m4a", "aac", "ogg"])
def test_common_audio_types_are_supported(filetype):
    assert ingestion.file_kind({"filetype": filetype}) == "audio"


def test_slack_file_stub_is_refreshed_through_files_info(caplog):
    class Client:
        def files_info(self, file):
            assert file == "F123"
            return {"ok": True, "file": {
                "id": file, "name": "meeting.mp3", "mimetype": "audio/mpeg",
                "filetype": "mp3", "size": 1234,
                "url_private_download": "https://files.slack.com/private/file",
            }}
    with caplog.at_level(logging.INFO, logger="content_ingestion"):
        resolved = ingestion.resolve_file_metadata({"id": "F123"}, Client())
    assert resolved["mimetype"] == "audio/mpeg"
    assert resolved["url_private_download"].startswith("https://files.slack.com/")
    assert "file_metadata_resolved" in caplog.text
    assert "has_private_url': True" in caplog.text
    assert "https://files.slack.com" not in caplog.text


def test_missing_files_read_scope_is_reported_without_retry():
    class Client:
        calls = 0
        def files_info(self, file):
            self.calls += 1
            raise SlackApiError("missing scope", {
                "ok": False, "error": "missing_scope", "needed": "files:read"})
    client = Client()
    with pytest.raises(ingestion.ContentAuthorizationError, match="files:read"):
        ingestion.resolve_file_metadata({"id": "F123"}, client, sleeper=lambda _: None)
    assert client.calls == 1


def test_authenticated_private_download_records_status_and_size(caplog):
    observed = {}
    class Response:
        status = 200
        url = "https://files.slack.com/private/file"
        headers = {"Content-Type": "audio/mpeg", "Content-Length": "5"}
        def read(self, limit): return b"audio"
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def opener(request, timeout):
        observed["authorization"] = request.get_header("Authorization")
        return Response()
    with caplog.at_level(logging.INFO, logger="content_ingestion"):
        data = ingestion.download({
            "id": "F123", "mimetype": "audio/mpeg",
            "url_private_download": "https://files.slack.com/private/file",
        }, "test-token", opener=opener)
    assert data == b"audio"
    assert observed["authorization"] == "Bearer test-token"
    assert "download_status=200" in caplog.text and "downloaded_bytes=5" in caplog.text
    assert "test-token" not in caplog.text


def test_empty_private_file_is_rejected():
    class Response:
        status = 200
        url = "https://files.slack.com/private/file"
        headers = {"Content-Type": "audio/mpeg", "Content-Length": "0"}
        def read(self, limit): return b""
        def __enter__(self): return self
        def __exit__(self, *args): pass
    with pytest.raises(ingestion.ContentError, match="empty"):
        ingestion.download({
            "id": "F123", "mimetype": "audio/mpeg",
            "url_private_download": "https://files.slack.com/private/file",
        }, "test-token", opener=lambda request, timeout: Response())


def test_audio_and_video_converge_to_transcript_content():
    observed = []
    def downloader(info, token):
        return b"media"
    def transcriber(data, kind, mime):
        observed.append((data, kind, mime))
        return transcription.Transcript(f"spoken content from {kind}", 2, 42.0)
    contents, errors = ingestion.ingest("", [
        {"id": "A", "mimetype": "audio/ogg"},
        {"id": "V", "mimetype": "video/mp4"},
    ], bot_token="token", downloader=downloader, transcriber=transcriber)
    assert [content.source_type for content in contents] == ["audio", "video"]
    assert [content.chunks for content in contents] == [2, 2]
    assert [entry[1] for entry in observed] == ["audio", "video"]
    assert errors == []


def test_completed_transcription_logs_metadata_but_not_transcript(caplog):
    secret_transcript = "private spoken action item"
    with caplog.at_level(logging.INFO, logger="content_ingestion"):
        contents, _ = ingestion.ingest("Extract action items from this audio", [{
            "id": "FLOG", "mimetype": "audio/mpeg", "content": b"media",
        }], transcriber=lambda *args: transcription.Transcript(secret_transcript, 3, 42.0))
    assert contents[0].text == secret_transcript
    assert "transcription_completed file_id=FLOG media_type=audio chunk_count=3" in caplog.text
    assert f"transcript_chars={len(secret_transcript)}" in caplog.text
    assert secret_transcript not in caplog.text


def test_failed_transcription_logs_actual_stage_and_returns_specific_reason(caplog):
    def fail(*args):
        raise transcription.TranscriptionError(
            "The transcription provider timed out.", stage="provider")

    with caplog.at_level(logging.WARNING, logger="content_ingestion"):
        with pytest.raises(ingestion.ContentError, match="shared file:.*provider timed out"):
            ingestion.ingest("Extract action items from this audio", [{
                "id": "FERR", "mimetype": "audio/mpeg", "content": b"media",
            }], transcriber=fail)
    assert "transcription_failed file_id=FERR media_type=audio stage=provider" in caplog.text


def test_mp3_fixture_flows_through_transcription_and_action_extraction():
    media = Path("test_fixtures/sample.mp3").read_bytes()
    assert media.startswith(b"\xff\xfb")
    observed = []
    def transcriber(data, kind, mime):
        observed.append((len(data), kind, mime))
        return transcription.Transcript(
            "<@U1> will prepare the client report by September 25.", 1, 0.1)
    contents, warnings = ingestion.ingest("", [{
        "id": "FMP3", "mimetype": "audio/mpeg", "filetype": "mp3", "content": media,
    }], transcriber=transcriber)
    def model(prompt, text):
        return {"items": [{
            "title": "Prepare the client report", "assignee": "<@U1>",
            "due_date": "2026-09-25", "priority": None, "status": "pending",
            "confidence": .98, "evidence": text, "clarification": None,
        }]}
    items = extraction.extract(contents, date(2026, 9, 22), model)
    assert observed == [(len(media), "audio", "audio/mpeg")]
    assert warnings == []
    assert [(item.title, item.assignee, item.due_date) for item in items] == [
        ("Prepare the client report", "<@U1>", "2026-09-25")]


def test_video_bytes_flow_to_transcriber_before_action_extraction():
    # Minimal ISO-BMFF signature: the configured provider owns codec decoding.
    video = b"\x00\x00\x00\x18ftypmp42" + bytes(32)
    observed = []
    def transcriber(data, kind, mime):
        observed.append((data, kind, mime))
        return transcription.Transcript("<@U2> will review deployment.", 1, 1.0)
    contents, warnings = ingestion.ingest("", [{
        "id": "FVIDEO", "mimetype": "video/mp4", "filetype": "mp4", "content": video,
    }], transcriber=transcriber)
    assert observed == [(video, "video", "video/mp4")]
    assert contents[0].text == "<@U2> will review deployment."
    assert warnings == []


def test_pasted_and_caption_transcripts_are_normalized():
    content, _ = ingestion.ingest(
        "Turn this transcript into action items:\nWEBVTT\n\n00:00:01.000 --> 00:00:03.000\nAlex will review the release.")
    assert content[0].source_type == "transcript"
    assert content[0].text == "Alex will review the release."


def test_explicit_transcript_payload_routes_to_content_ingestion_without_command_wording():
    assert ingestion.should_ingest("Transcript: Alex owns the release review.")


def test_source_first_router_never_treats_plain_task_commands_as_media():
    commands = [
        "create a task to prepare the internship demo checklist for Praveen "
        "by October 8 with priority P2",
        "extract action items",
        "extract action items from this audio",
        "create a task called Transcript: review the notes",
        "this parser input is not understood",
    ]
    for command in commands:
        route = ingestion.classify_request(command)
        assert route.route == "text"
        assert route.source == "text"
        assert not route.is_shared_content


def test_source_first_router_uses_actual_slack_file_metadata():
    audio = ingestion.classify_request(
        "extract tasks", [{"id": "FA", "mimetype": "audio/mpeg"}])
    video = ingestion.classify_request(
        "extract tasks", [{"id": "FV", "mimetype": "video/mp4"}])
    transcript = ingestion.classify_request(
        "extract tasks", [{"id": "FT", "mimetype": "text/plain"}])
    stub = ingestion.classify_request("extract tasks", [{"id": "FSTUB"}])
    assert (audio.route, audio.source) == ("media", "audio")
    assert (video.route, video.source) == ("media", "video")
    assert (transcript.route, transcript.source) == ("transcript", "transcript")
    assert (stub.route, stub.source) == ("media", "file")


def test_inaccessible_and_unsupported_files_fail_without_claiming_success():
    with pytest.raises(ingestion.ContentError, match="unsupported content type"):
        ingestion.ingest("", [{"id": "P", "mimetype": "application/pdf"}])
    with pytest.raises(ingestion.ContentError, match="not available"):
        ingestion.ingest("", [{"id": "A", "mimetype": "audio/ogg"}],
                         downloader=lambda *_: (_ for _ in ()).throw(
                             ingestion.ContentError("The shared content is not available.")))


def test_unconfigured_transcription_fails_clearly(monkeypatch):
    monkeypatch.delenv("MEDIA_TRANSCRIPTION_COMMAND", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(transcription, "_local_whisper_command", lambda: None)
    with pytest.raises(transcription.TranscriptionError, match="not configured"):
        transcription.transcribe_bytes(b"audio", "audio")


def test_missing_ffmpeg_is_reported_before_provider_execution(monkeypatch):
    monkeypatch.setattr(transcription.shutil, "which", lambda name: None)
    with pytest.raises(transcription.TranscriptionError, match="ffprobe.*not installed"):
        transcription.transcribe_bytes(
            b"real-media-bytes", "audio", command="trusted-stt {input}")


def test_structured_extraction_handles_multiple_items_missing_fields_dates_and_people():
    content = ingestion.IngestedContent("meeting words", "transcript", "message")
    def model(prompt, text):
        assert "CURRENT_DATE: 2026-09-22" in prompt
        return {"items": [
            {"title": "Review API docs by tomorrow", "assignee": "Alex", "due_date": "tomorrow",
             "priority": "high", "status": "pending", "confidence": .96,
             "evidence": "Alex will review API docs by tomorrow.", "clarification": None},
            {"title": "Prepare deployment notes", "assignee": None, "due_date": None,
             "priority": None, "status": "pending", "confidence": .91,
             "evidence": "Prepare deployment notes.", "clarification": None},
        ]}
    items = extraction.extract([content], date(2026, 9, 22), model)
    assert [(item.title, item.assignee, item.due_date, item.priority) for item in items] == [
        ("Review API docs", "Alex", "2026-09-23", "P1"),
        ("Prepare deployment notes", None, None, None),
    ]


def test_structured_extraction_preserves_slack_mentions_for_multiple_actions():
    content = ingestion.IngestedContent("meeting words", "transcript", "message")
    def model(prompt, text):
        return {"items": [
            {"title": "Review API contract", "assignee": "<@U111>", "due_date": None,
             "priority": None, "status": "pending", "confidence": .96,
             "evidence": "<@U111> will review the API contract.", "clarification": None},
            {"title": "Publish test plan", "assignee": "<@U222>", "due_date": None,
             "priority": "P2", "status": "pending", "confidence": .94,
             "evidence": "<@U222> owns the test plan.", "clarification": None},
        ]}
    items = extraction.extract([content], date(2026, 9, 22), model)
    assert [(item.title, item.assignee) for item in items] == [
        ("Review API contract", "<@U111>"), ("Publish test plan", "<@U222>"),
    ]


def test_explicit_transcript_due_date_survives_structured_normalization():
    content = ingestion.IngestedContent(
        "<@UP> will review the API documentation by September 27.",
        "transcript", "message")
    def model(prompt, text):
        return {"items": [{
            "title": "Review the API documentation",
            "assignee": "<@UP>",
            "due_date": "2026-09-27",
            "priority": None,
            "status": "pending",
            "confidence": .97,
            "evidence": text,
            "clarification": None,
        }]}
    item = extraction.extract([content], date(2026, 9, 22), model)[0]
    assert (item.title, item.assignee, item.due_date, item.priority, item.status) == (
        "Review the API documentation", "<@UP>", "2026-09-27", None, "pending")


def test_two_transcript_actions_keep_independent_assignees_and_dates():
    content = ingestion.IngestedContent("two actions", "transcript", "message")
    def model(prompt, text):
        return {"items": [
            {"title": "Prepare release brief", "assignee": "<@U1>",
             "due_date": "2026-09-25", "priority": "P3", "status": "pending",
             "confidence": .96, "evidence": "first", "clarification": None},
            {"title": "Review integration notes", "assignee": "<@U2>",
             "due_date": "2026-09-27", "priority": None, "status": "pending",
             "confidence": .95, "evidence": "second", "clarification": None},
        ]}
    items = extraction.extract([content], date(2026, 9, 22), model)
    assert [(item.assignee, item.due_date, item.priority) for item in items] == [
        ("<@U1>", "2026-09-25", "P3"), ("<@U2>", "2026-09-27", None)]


def test_explicit_transcript_update_is_normalized_without_becoming_a_create():
    content = ingestion.IngestedContent("Set the API review task to P1.", "transcript", "message")
    def model(prompt, text):
        return {"items": [{
            "title": "Review API documentation", "assignee": None, "due_date": None,
            "priority": "P1", "status": "pending", "operation": "update",
            "confidence": .98, "evidence": text, "clarification": None,
        }]}
    item = extraction.extract([content], date(2026, 9, 22), model)[0]
    assert item.operation == "update"
    assert extraction.workflow_command(item) == {
        "intent": "update", "task_name": "Review API documentation",
        "target_scope": "single",
        "_source": {"type": "transcript", "reference": "message",
                    "confidence": .98, "evidence": "Set the API review task to P1."},
        "changes": [{"field": "priority", "value": "P1"}],
    }


def test_conflicting_duplicate_mentions_require_clarification_instead_of_merging_fields():
    content = ingestion.IngestedContent("repeated action", "transcript", "message")
    def model(prompt, text):
        return {"items": [
            {"title": "Review API documentation", "assignee": "<@UP>",
             "due_date": "2026-09-25", "priority": "P1", "status": "pending",
             "confidence": .95, "evidence": "first", "clarification": None},
            {"title": "Review API documentation", "assignee": "<@UP>",
             "due_date": "2026-09-27", "priority": "P3", "status": "pending",
             "confidence": .96, "evidence": "second", "clarification": None},
        ]}
    item = extraction.extract([content], date(2026, 9, 22), model)[0]
    assert item.confidence < .75
    assert "due date" in item.clarification and "priority" in item.clarification


def test_extraction_logs_redacted_exception_type_message_and_traceback(caplog):
    content = ingestion.IngestedContent("private transcript text", "transcript", "message")
    leaked = "secret-value-that-must-not-appear"
    def model(prompt, text):
        raise ConnectionError(f"HTTP 503 Authorization: Bearer {leaked}")
    with caplog.at_level(logging.ERROR, logger="action_item_extraction"):
        with pytest.raises(extraction.ExtractionError, match="extraction service failed"):
            extraction.extract([content], date(2026, 9, 22), model)
    logged = caplog.text
    assert "Action-item extraction failed" in logged
    assert "exception_type=ConnectionError" in logged
    assert "Traceback (most recent call last)" in logged
    assert leaked not in logged
    assert "private transcript text" not in logged


def test_shared_ollama_client_logs_response_parsing_failure_without_body(monkeypatch, caplog):
    import langchain_ollama
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-secret-key")
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: SimpleNamespace(
        invoke=lambda messages: SimpleNamespace(content="not-json-private-output")))
    with caplog.at_level(logging.ERROR, logger="intent_parser"):
        with pytest.raises(RuntimeError, match="invalid structured output"):
            intent_parser.structured_model_json("system", "private transcript")
    logged = caplog.text
    assert "stage=response_parse" in logged
    assert "exception_type=JSONDecodeError" in logged
    assert "Traceback (most recent call last)" in logged
    assert "not-json-private-output" not in logged
    assert "unit-test-secret-key" not in logged


def test_shared_ollama_client_logs_redacted_http_failure(monkeypatch, caplog):
    import langchain_ollama
    leaked = "runtime-secret-that-must-not-appear"
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-placeholder")
    def fail(messages):
        raise ConnectionError(f"HTTP 503 Bearer {leaked}")
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: SimpleNamespace(invoke=fail))
    with caplog.at_level(logging.ERROR, logger="intent_parser"):
        with pytest.raises(RuntimeError, match="service request failed"):
            intent_parser.structured_model_json(
                "system", "private transcript", sleeper=lambda seconds: None)
    logged = caplog.text
    assert "stage=request" in logged
    assert "exception_type=ConnectionError" in logged
    assert "HTTP 503" in logged
    assert "Traceback (most recent call last)" in logged
    assert leaked not in logged


def test_shared_ollama_client_retries_one_transient_failure(monkeypatch):
    import langchain_ollama
    calls = []
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-placeholder")
    def invoke(messages):
        calls.append(True)
        if len(calls) == 1:
            raise TimeoutError("service temporarily unavailable")
        return SimpleNamespace(content='{"items": []}')
    monkeypatch.setattr(
        langchain_ollama, "ChatOllama",
        lambda **kwargs: SimpleNamespace(invoke=invoke))
    result = intent_parser.structured_model_json(
        "system", "transcript", sleeper=lambda seconds: None)
    assert result == {"items": []}
    assert len(calls) == 2


def test_shared_ollama_client_does_not_retry_permanent_failure(monkeypatch):
    import langchain_ollama
    calls = []
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "unit-test-placeholder")
    def invoke(messages):
        calls.append(True)
        raise RuntimeError("HTTP 401 unauthorized")
    monkeypatch.setattr(
        langchain_ollama, "ChatOllama",
        lambda **kwargs: SimpleNamespace(invoke=invoke))
    with pytest.raises(RuntimeError, match="service request failed"):
        intent_parser.structured_model_json(
            "system", "transcript", sleeper=lambda seconds: None)
    assert len(calls) == 1


@pytest.mark.parametrize("expression,expected", [
    ("in two days", "2026-09-24"),
    ("end of this week", "2026-09-25"),
    ("next Monday", "2026-09-28"),
])
def test_relative_dates_use_request_date(expression, expected):
    content = ingestion.IngestedContent("dates", "transcript", "message")
    def model(prompt, text):
        return {"items": [{"title": "Prepare brief", "assignee": None, "due_date": expression,
                           "priority": None, "status": "pending", "confidence": .9,
                           "evidence": "Prepare the brief.", "clarification": None}]}
    assert extraction.extract([content], date(2026, 9, 22), model)[0].due_date == expected


def test_duplicate_mentions_merge_and_preserve_evidence():
    content = ingestion.IngestedContent("repeated", "audio", "F1")
    def model(prompt, text):
        return {"items": [
            {"title": "Publish release notes", "assignee": "Morgan", "due_date": None,
             "priority": None, "status": "pending", "confidence": .8,
             "evidence": "Morgan will publish the release notes.", "clarification": None},
            {"title": "Publish the release notes", "assignee": "Morgan", "due_date": None,
             "priority": None, "status": "pending", "confidence": .95,
             "evidence": "The release notes are Morgan's action.", "clarification": None},
        ]}
    items = extraction.extract([content], date(2026, 9, 22), model)
    assert len(items) == 1
    assert items[0].confidence == .95
    assert " / " in items[0].evidence


def test_ambiguous_and_past_dated_extractions_require_review():
    content = ingestion.IngestedContent("ambiguous", "video", "F2")
    def model(prompt, text):
        return {"items": [{"title": "Send it", "assignee": None, "due_date": "2026-09-01",
                           "priority": None, "status": "pending", "confidence": .9,
                           "evidence": "He should send it.",
                           "clarification": "Which person and document does this refer to?"}]}
    item = extraction.extract([content], date(2026, 9, 22), model)[0]
    assert item.confidence < .5
    assert item.clarification


def test_long_transcripts_are_bounded_into_model_sized_chunks():
    chunks = extraction.chunk_text("Sentence. " * 4000, max_chars=1000)
    assert len(chunks) > 20
    assert all(len(chunk) <= 1000 for chunk in chunks)
