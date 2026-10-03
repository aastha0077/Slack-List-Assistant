from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import subprocess
import json

import pytest

import action_item_extraction as extraction
import content_ingestion as ingestion
import transcription
import intent_parser


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(["provider"], returncode, stdout, stderr)


def test_stdout_based_transcription_uses_safe_argv(monkeypatch, tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")
    observed = {}

    def run(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _completed("Prepare the release notes.\n")

    monkeypatch.setattr(transcription.subprocess, "run", run)
    assert transcription._invoke_provider(media, "speech-tool --input {input}", 30) == \
        "Prepare the release notes."
    assert observed["args"][:2] == ["speech-tool", "--input"]
    assert observed["args"][2] == str(media)
    assert "shell" not in observed["kwargs"]


def test_provider_stage_logs_include_file_and_chunk_without_transcript(
        monkeypatch, tmp_path, caplog):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")
    secret_transcript = "private provider transcript"
    monkeypatch.setattr(
        transcription.subprocess, "run",
        lambda *args, **kwargs: _completed(secret_transcript),
    )
    with caplog.at_level(logging.INFO, logger="transcription"):
        assert transcription._invoke_provider(
            media, "speech-tool {input}", 30, file_id="F123",
            chunk_index=2, chunk_count=4) == secret_transcript
    assert "transcription_provider_started file_id=F123 chunk_index=2 chunk_count=4" in caplog.text
    assert "transcription_provider_completed file_id=F123 chunk_index=2 chunk_count=4" in caplog.text
    assert "output_mode=stdout" in caplog.text
    assert secret_transcript not in caplog.text


def test_file_based_whisper_transcription_discovers_output_and_cleans_it(monkeypatch, tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")
    generated_dir = None

    def run(args, **kwargs):
        nonlocal generated_dir
        output_index = args.index("--output_dir") + 1
        generated_dir = Path(args[output_index])
        assert generated_dir != Path("/tmp")
        (generated_dir / f"{media.stem}.txt").write_text(
            "Review the API documentation.", encoding="utf-8")
        return _completed(stderr="provider progress")

    monkeypatch.setattr(transcription.subprocess, "run", run)
    text = transcription._invoke_provider(
        media, "whisper {input} --model turbo --output_format txt --output_dir /tmp", 30)
    assert text == "Review the API documentation."
    assert generated_dir is not None and not generated_dir.exists()


def test_declared_file_output_requires_expected_transcript_file(monkeypatch, tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")
    monkeypatch.setattr(
        transcription.subprocess, "run",
        lambda *args, **kwargs: _completed(stdout="provider progress only"),
    )
    with pytest.raises(transcription.TranscriptionError, match="expected .txt transcript file"):
        transcription._invoke_provider(
            media, "speech-tool {input} --output_format txt --output_dir /tmp", 30)


def test_empty_provider_transcript_is_rejected(monkeypatch, tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")
    monkeypatch.setattr(transcription.subprocess, "run", lambda *args, **kwargs: _completed())
    with pytest.raises(transcription.TranscriptionError, match="empty transcript"):
        transcription._invoke_provider(media, "speech-tool {input}", 30)


def test_openai_backend_posts_audio_and_reads_transcript(tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"normalized-wav")
    observed = {}
    class Response:
        def read(self): return json.dumps({"text": "List my tasks."}).encode()
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def opener(request, timeout):
        observed.update(url=request.full_url, auth=request.get_header("Authorization"),
                        content_type=request.get_header("Content-type"), body=request.data)
        return Response()
    assert transcription._invoke_openai(
        media, "test-secret", "gpt-4o-mini-transcribe", 45,
        opener=opener) == "List my tasks."
    assert observed["url"].endswith("/v1/audio/transcriptions")
    assert observed["auth"] == "Bearer test-secret"
    assert "multipart/form-data" in observed["content_type"]
    assert b"normalized-wav" in observed["body"]


def test_auto_backend_uses_openai_key_without_command(monkeypatch):
    monkeypatch.delenv("MEDIA_TRANSCRIPTION_COMMAND", raising=False)
    monkeypatch.delenv("MEDIA_TRANSCRIPTION_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "configured")
    assert transcription._transcription_backend() == ("openai", None)


def test_auto_backend_falls_back_to_installed_local_whisper(monkeypatch):
    monkeypatch.delenv("MEDIA_TRANSCRIPTION_COMMAND", raising=False)
    monkeypatch.delenv("MEDIA_TRANSCRIPTION_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(transcription, "_local_whisper_command",
                        lambda: "whisper {input} --output_format txt --output_dir {output_dir}")
    backend, command = transcription._transcription_backend()
    assert backend == "whisper" and command.startswith("whisper ")


@pytest.mark.parametrize("fixture,kind,mimetype", [
    ("spoken-list-my-tasks.mp3", "audio", "audio/mpeg"),
    ("spoken-list-my-tasks.mp4", "video", "video/mp4"),
])
def test_real_speech_media_is_validated_normalized_and_reaches_existing_parser(
        monkeypatch, fixture, kind, mimetype):
    media = Path("test_fixtures", fixture).read_bytes()
    monkeypatch.setattr(
        transcription, "_invoke_provider",
        lambda path, command, timeout, **kwargs: "List my tasks.")
    result = transcription.transcribe_bytes(
        media, kind, mimetype, command="speech-tool {input}")
    assert result.duration_seconds and result.duration_seconds > 0
    assert result.text == "List my tasks."
    assert intent_parser.parse_intent(result.text)["intent"] == "list"


def test_provider_nonzero_exit_includes_stderr_without_temp_path(monkeypatch, tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")
    monkeypatch.setattr(
        transcription.subprocess, "run",
        lambda *args, **kwargs: _completed(stderr=f"decoder failed for {media}", returncode=2),
    )
    with pytest.raises(transcription.TranscriptionError) as raised:
        transcription._invoke_provider(media, "speech-tool {input}", 30)
    assert "decoder failed" in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


def test_provider_failure_cleans_generated_transcript_files(monkeypatch, tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")
    generated = tmp_path / "chunk.txt"

    def fail_after_output(*args, **kwargs):
        generated.write_text("partial transcript", encoding="utf-8")
        return _completed(stderr="provider failed", returncode=1)

    monkeypatch.setattr(transcription.subprocess, "run", fail_after_output)
    with pytest.raises(transcription.TranscriptionError, match="provider failed"):
        transcription._invoke_provider(media, "speech-tool {input}", 30)
    assert not generated.exists()


def test_provider_timeout_is_clear(monkeypatch, tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("speech-tool", 30)

    monkeypatch.setattr(transcription.subprocess, "run", timeout)
    with pytest.raises(transcription.TranscriptionError, match="provider timed out"):
        transcription._invoke_provider(media, "speech-tool {input}", 30)


def test_missing_provider_executable_is_clear(monkeypatch, tmp_path):
    media = tmp_path / "chunk.wav"
    media.write_bytes(b"wav")

    def missing(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(transcription.subprocess, "run", missing)
    with pytest.raises(transcription.TranscriptionError, match="speech-tool.*not installed"):
        transcription._invoke_provider(media, "speech-tool {input}", 30)


def test_bare_provider_command_resolves_from_active_virtualenv(monkeypatch, tmp_path):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("", encoding="utf-8")
    provider = bin_dir / "speech-tool"
    provider.write_text("#!/bin/sh\n", encoding="utf-8")
    provider.chmod(0o755)
    monkeypatch.setattr(transcription.sys, "executable", str(python))
    monkeypatch.setattr(transcription.shutil, "which", lambda executable: None)
    argv = transcription._resolve_provider_executable(["speech-tool", "--version"])
    assert argv == [str(provider), "--version"]


def test_provider_invocations_use_separate_temporary_directories(monkeypatch, tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    output_dirs = []

    def run(args, **kwargs):
        output_dir = Path(args[args.index("--output_dir") + 1])
        output_dirs.append(output_dir)
        input_path = Path(args[1])
        (output_dir / f"{input_path.stem}.txt").write_text(
            f"text for {input_path.stem}", encoding="utf-8")
        return _completed()

    monkeypatch.setattr(transcription.subprocess, "run", run)
    command = "speech-tool {input} --output_format txt --output_dir /tmp"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda path: transcription._invoke_provider(path, command, 30),
            (first, second),
        ))
    assert set(results) == {"text for first", "text for second"}
    assert len(set(output_dirs)) == 2
    assert all(not directory.exists() for directory in output_dirs)


def _mock_normalization(monkeypatch, *, has_audio=True, chunks=1):
    monkeypatch.setattr(transcription.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(transcription, "_has_audio", lambda path: has_audio)
    monkeypatch.setattr(transcription, "_duration", lambda path: 42.0)

    def run(args, **kwargs):
        if args[0] == "ffmpeg":
            pattern = Path(args[-1])
            for index in range(chunks):
                Path(str(pattern).replace("%04d", f"{index:04d}")).write_bytes(b"normalized wav")
        return _completed()

    monkeypatch.setattr(transcription, "_run", run)


@pytest.mark.parametrize("mimetype", ["audio/mpeg", "audio/wav", "audio/mp4"])
def test_mp3_wav_and_m4a_are_normalized_before_transcription(monkeypatch, mimetype):
    _mock_normalization(monkeypatch)
    observed = []
    monkeypatch.setattr(
        transcription, "_invoke_provider",
        lambda path, command, timeout, **kwargs:
        observed.append(path.read_bytes()) or "spoken task",
    )
    result = transcription.transcribe_bytes(
        b"source media", "audio", mimetype, command="speech-tool {input}")
    assert result.text == "spoken task"
    assert observed == [b"normalized wav"]


def test_video_with_audio_is_normalized_and_transcribed(monkeypatch):
    _mock_normalization(monkeypatch, has_audio=True)
    monkeypatch.setattr(transcription, "_invoke_provider", lambda *args, **kwargs: "video speech")
    result = transcription.transcribe_bytes(
        b"video", "video", "video/mp4", command="speech-tool {input}")
    assert result.text == "video speech" and result.chunks == 1


def test_video_without_audio_returns_clear_error(monkeypatch):
    _mock_normalization(monkeypatch, has_audio=False)
    with pytest.raises(transcription.TranscriptionError, match="no usable audio track"):
        transcription.transcribe_bytes(
            b"silent video", "video", "video/mp4", command="speech-tool {input}")


def test_ffmpeg_failure_identifies_normalization_stage(monkeypatch):
    monkeypatch.setattr(transcription.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(transcription, "_has_audio", lambda path: True)
    monkeypatch.setattr(transcription, "_duration", lambda path: 10.0)

    def fail_ffmpeg(args, **kwargs):
        raise transcription.TranscriptionError("FFmpeg audio normalization failed: bad codec", stage="ffmpeg")

    monkeypatch.setattr(transcription, "_run", fail_ffmpeg)
    with pytest.raises(transcription.TranscriptionError, match="bad codec") as raised:
        transcription.transcribe_bytes(
            b"audio", "audio", command="speech-tool {input}")
    assert raised.value.stage == "ffmpeg"


def test_missing_ffmpeg_is_distinct_from_missing_ffprobe(monkeypatch):
    monkeypatch.setattr(
        transcription.shutil, "which",
        lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None,
    )
    with pytest.raises(transcription.TranscriptionError, match="ffmpeg.*not installed") as raised:
        transcription.transcribe_bytes(
            b"audio", "audio", command="speech-tool {input}")
    assert raised.value.stage == "ffmpeg"


def test_multiple_chunks_are_transcribed_in_order(monkeypatch):
    _mock_normalization(monkeypatch, chunks=3)
    monkeypatch.setattr(
        transcription, "_invoke_provider",
        lambda path, command, timeout, **kwargs: f"text from {path.stem}",
    )
    result = transcription.transcribe_bytes(
        b"long media", "audio", "audio/mpeg", command="speech-tool {input}")
    assert result.chunks == 3
    assert result.text.splitlines() == [
        "text from chunk-0000", "text from chunk-0001", "text from chunk-0002"]


def test_media_transcript_reuses_existing_action_item_extractor():
    transcript = "Aastha will prepare the client report by September 25."
    contents, warnings = ingestion.ingest(
        "Extract action items from this audio",
        [{"id": "F1", "mimetype": "audio/mpeg", "content": b"media"}],
        transcriber=lambda *args: transcription.Transcript(transcript, 1, 5.0),
    )
    observed = []

    def model(prompt, text):
        observed.append(text)
        return {"items": [{
            "title": "Prepare the client report", "assignee": "Aastha",
            "due_date": "2026-09-25", "priority": None, "status": "pending",
            "confidence": .98, "evidence": text, "clarification": None,
        }]}

    items = extraction.extract(contents, model=model)
    assert warnings == []
    assert observed == [transcript]
    assert items[0].title == "Prepare the client report"


def test_transcript_file_ingestion_never_invokes_speech_to_text():
    calls = []
    contents, warnings = ingestion.ingest(
        "Extract action items from this transcript",
        [{
            "id": "FTEXT", "mimetype": "text/plain", "filetype": "txt",
            "content": b"Alex will publish the release notes.",
        }],
        transcriber=lambda *args: calls.append(args),
    )
    assert calls == []
    assert warnings == []
    assert contents[0].source_type == "transcript"
    assert contents[0].text == "Alex will publish the release notes."
