"""Safe speech transcription for Slack media with built-in provider selection."""
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from safe_diagnostics import redact


logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    def __init__(self, message, stage="transcription"):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class Transcript:
    text: str
    chunks: int
    duration_seconds: float | None = None


_TEXT_OUTPUT_SUFFIXES = {".txt", ".vtt", ".srt", ".tsv", ".json"}
_OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"


def _safe_process_detail(value, hidden_paths=()):
    detail = str(value or "").strip()
    for path in sorted({str(value) for value in hidden_paths if value}, key=len, reverse=True):
        detail = detail.replace(path, "<temporary path>")
    return detail[-500:]


def _run(args, timeout=120, *, cwd=None, label="Media processing", hidden_paths=(),
         stage="media_processing"):
    """Run one trusted argv without a shell and normalize process failures."""
    try:
        result = subprocess.run(
            list(args), check=False, capture_output=True, text=True,
            timeout=timeout, cwd=str(cwd) if cwd else None,
        )
    except FileNotFoundError as exc:
        executable = Path(str(args[0])).name
        if stage == "provider":
            message = f"Transcription provider executable {executable!r} is not installed."
        else:
            message = f"Required media tool {executable!r} is not installed."
        raise TranscriptionError(message, stage=stage) from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"{label} timed out.", stage=stage) from exc
    if result.returncode:
        detail = _safe_process_detail(result.stderr or result.stdout, hidden_paths)
        if detail:
            raise TranscriptionError(f"{label} failed: {detail}", stage=stage)
        raise TranscriptionError(
            f"{label} failed with exit code {result.returncode}.", stage=stage)
    return result


def _duration(path):
    result = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], hidden_paths=(path.parent, path), label="ffprobe", stage="ffprobe")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise TranscriptionError(
            "ffprobe did not return a valid media duration.", stage="ffprobe")


def _has_audio(path):
    result = _run([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=index", "-of", "csv=p=0", str(path),
    ], hidden_paths=(path.parent, path), label="ffprobe", stage="ffprobe")
    return bool(result.stdout.strip())


def _provider_argv(path, command, output_dir):
    try:
        template = shlex.split(command)
    except ValueError as exc:
        raise TranscriptionError(
            "MEDIA_TRANSCRIPTION_COMMAND contains invalid quoting.", stage="configuration") from exc
    if not template or not any("{input}" in part for part in template):
        raise TranscriptionError(
            "MEDIA_TRANSCRIPTION_COMMAND must contain {input}.", stage="configuration")

    argv = []
    index = 0
    while index < len(template):
        part = template[index]
        # Put declared provider output in an invocation-specific directory. This
        # supports Whisper CLI and avoids guessing or hard-coding its filename.
        if part in {"--output_dir", "--output-dir"}:
            argv.extend((part, str(output_dir)))
            index += 2
            continue
        if part.startswith("--output_dir=") or part.startswith("--output-dir="):
            argv.append(part.split("=", 1)[0] + "=" + str(output_dir))
            index += 1
            continue
        argv.append(part.replace("{input}", str(path)).replace("{output_dir}", str(output_dir)))
        index += 1
    return argv


def _provider_output_format(argv):
    """Return a declared textual output format without assuming a provider."""
    for index, part in enumerate(argv):
        if part in {"--output_format", "--output-format"} and index + 1 < len(argv):
            return str(argv[index + 1]).casefold().lstrip(".")
        if part.startswith("--output_format=") or part.startswith("--output-format="):
            return part.split("=", 1)[1].casefold().lstrip(".")
    return None


def _resolve_provider_executable(argv):
    """Find a bare provider command on PATH or beside the running Python."""
    if not argv:
        return argv
    executable = str(argv[0])
    if os.sep in executable or (os.altsep and os.altsep in executable):
        return argv
    if shutil.which(executable):
        return argv
    # Do not resolve the Python symlink: virtual environments commonly point
    # at a base interpreter, while console scripts live beside the symlink.
    virtualenv_executable = Path(sys.executable).parent / executable
    if virtualenv_executable.is_file() and os.access(virtualenv_executable, os.X_OK):
        return [str(virtualenv_executable), *argv[1:]]
    return argv


def _output_snapshot(directory):
    snapshot = {}
    if not directory.exists():
        return snapshot
    for candidate in directory.iterdir():
        if candidate.is_file() and candidate.suffix.casefold() in _TEXT_OUTPUT_SUFFIXES:
            stat = candidate.stat()
            snapshot[candidate] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _read_transcript_file(path):
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise TranscriptionError(
            "The transcription provider produced an unreadable transcript file.",
            stage="provider_output") from exc
    if path.suffix.casefold() == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TranscriptionError(
                "The transcription provider produced malformed transcript JSON.",
                stage="provider_output") from exc
        if isinstance(payload, dict):
            raw = payload.get("text") or payload.get("transcript") or ""
        elif isinstance(payload, str):
            raw = payload
        else:
            raw = ""
    return str(raw).strip()


def _invoke_provider(path, command, timeout, *, file_id="unknown", chunk_index=1,
                     chunk_count=1):
    """Return stdout or a newly generated transcript file, then clean it up."""
    with tempfile.TemporaryDirectory(prefix="provider-", dir=path.parent) as work_dir_value:
        work_dir = Path(work_dir_value)
        before_parent = _output_snapshot(path.parent)
        argv = _resolve_provider_executable(_provider_argv(path, command, work_dir))
        output_format = _provider_output_format(argv)
        candidates = []
        generated_candidates = []
        logger.info(
            "transcription_provider_started file_id=%s chunk_index=%d chunk_count=%d",
            redact(file_id), chunk_index, chunk_count,
        )
        try:
            result = _run(
                argv, timeout=timeout, cwd=work_dir, label="The transcription provider",
                hidden_paths=(path.parent, path, work_dir), stage="provider",
            )
            for directory in (work_dir, path.parent):
                for candidate in directory.iterdir():
                    if (not candidate.is_file()
                            or candidate.suffix.casefold() not in _TEXT_OUTPUT_SUFFIXES):
                        continue
                    stat = candidate.stat()
                    signature = (stat.st_mtime_ns, stat.st_size)
                    if (directory == work_dir
                            or (before_parent.get(candidate) != signature
                                and candidate.stem == path.stem)):
                        candidates.append(candidate)
            generated_candidates = list(candidates)

            # For commands declaring a textual output format, resolve the exact
            # provider output from the input stem. This is how Whisper names files,
            # and the per-invocation directory prevents stale/cross-request reads.
            declared_file_output = output_format in {
                suffix.lstrip(".") for suffix in _TEXT_OUTPUT_SUFFIXES}
            if declared_file_output:
                expected = [directory / f"{path.stem}.{output_format}"
                            for directory in (work_dir, path.parent)]
                candidates = [candidate for candidate in expected if candidate in candidates]
            else:
                same_stem = [candidate for candidate in candidates if candidate.stem == path.stem]
                if same_stem:
                    candidates = same_stem
            if declared_file_output and not candidates:
                raise TranscriptionError(
                    f"The transcription provider did not create the expected .{output_format} "
                    "transcript file.", stage="provider_output")

            # A file is authoritative when the provider created one; stdout often
            # contains progress/status lines for file-producing CLIs.
            texts = []
            for candidate in sorted(set(candidates), key=lambda value: (value.name, str(value))):
                value = _read_transcript_file(candidate)
                if value:
                    texts.append(value)
            text = "\n".join(texts).strip()
            if not text and not declared_file_output:
                text = result.stdout.strip()
            if not text:
                raise TranscriptionError(
                    "The transcription provider returned an empty transcript.",
                    stage="provider_output")
            logger.info(
                "transcription_provider_completed file_id=%s chunk_index=%d chunk_count=%d "
                "output_mode=%s transcript_chars=%d",
                redact(file_id), chunk_index, chunk_count,
                "file" if texts else "stdout", len(text),
            )
            return text
        except TranscriptionError as exc:
            logger.warning(
                "transcription_provider_failed file_id=%s chunk_index=%d chunk_count=%d "
                "stage=%s error_type=%s message=%s",
                redact(file_id), chunk_index, chunk_count, exc.stage,
                type(exc).__name__, redact(exc),
            )
            raise
        finally:
            # work_dir is removed by TemporaryDirectory. Only remove files newly
            # created beside the input; never touch pre-existing operator files.
            cleanup_candidates = set(generated_candidates)
            cleanup_candidates.update(
                candidate for candidate in _output_snapshot(path.parent)
                if candidate not in before_parent and candidate.stem == path.stem)
            for candidate in cleanup_candidates:
                if candidate.parent == path.parent and candidate not in before_parent:
                    try:
                        candidate.unlink()
                    except FileNotFoundError:
                        pass


def _multipart_body(path, model):
    boundary = "----slack-list-" + uuid.uuid4().hex
    body = []
    for name, value in (("model", model),):
        body.extend((f"--{boundary}\r\n".encode(),
                     f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                     str(value).encode(), b"\r\n"))
    body.extend((f"--{boundary}\r\n".encode(),
                 b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n',
                 b"Content-Type: audio/wav\r\n\r\n", path.read_bytes(), b"\r\n",
                 f"--{boundary}--\r\n".encode()))
    return b"".join(body), boundary


def _invoke_openai(path, api_key, model, timeout, *, file_id="unknown",
                   chunk_index=1, chunk_count=1, opener=urlopen, sleeper=None):
    body, boundary = _multipart_body(path, model)
    request = Request(_OPENAI_TRANSCRIPTION_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    sleeper = sleeper or (lambda seconds: __import__("time").sleep(seconds))
    logger.info("transcription_provider_started provider=openai file_id=%s chunk_index=%d chunk_count=%d model=%s",
                redact(file_id), chunk_index, chunk_count, model)
    for attempt in range(2):
        try:
            with opener(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            transient = exc.code == 429 or exc.code >= 500
            if transient and attempt == 0:
                logger.warning("transcription_provider_retry provider=openai status=%s attempt=1/2", exc.code)
                sleeper(1)
                continue
            message = ("The configured transcription credential was rejected." if exc.code in {401, 403}
                       else "The transcription provider is temporarily unavailable." if transient
                       else "The transcription provider rejected the media request.")
            raise TranscriptionError(message, stage="provider") from exc
        except (TimeoutError, URLError) as exc:
            if attempt == 0:
                logger.warning("transcription_provider_retry provider=openai status=network attempt=1/2")
                sleeper(1)
                continue
            raise TranscriptionError("The transcription provider could not be reached or timed out.",
                                     stage="provider") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TranscriptionError("The transcription provider returned an unreadable response.",
                                     stage="provider_output") from exc
    text = str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
    if not text:
        raise TranscriptionError("The transcription provider returned an empty transcript.",
                                 stage="provider_output")
    logger.info("transcription_provider_completed provider=openai file_id=%s chunk_index=%d chunk_count=%d transcript_chars=%d",
                redact(file_id), chunk_index, chunk_count, len(text))
    return text


def _local_whisper_command():
    argv = _resolve_provider_executable(["whisper"])
    executable = str(argv[0]) if argv else "whisper"
    if not (shutil.which(executable) or Path(executable).is_file()):
        return None
    model = os.getenv("MEDIA_WHISPER_MODEL", "turbo").strip() or "turbo"
    return (f"{shlex.quote(executable)} {{input}} --model {shlex.quote(model)} "
            "--output_format txt --output_dir {output_dir}")


def _transcription_backend(command=None, provider=None):
    if command:
        return "command", command
    configured = os.getenv("MEDIA_TRANSCRIPTION_COMMAND", "").strip()
    selected = str(provider or os.getenv("MEDIA_TRANSCRIPTION_PROVIDER", "auto")).strip().casefold()
    if selected in {"", "auto"}:
        if configured:
            return "command", configured
        if os.getenv("OPENAI_API_KEY", "").strip():
            return "openai", None
        discovered = _local_whisper_command()
        if discovered:
            return "whisper", discovered
        raise TranscriptionError(
            "Audio/video transcription is not configured. Configure OPENAI_API_KEY, select the Whisper backend, or set MEDIA_TRANSCRIPTION_COMMAND.",
            stage="configuration")
    if selected == "openai":
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise TranscriptionError("OpenAI transcription requires OPENAI_API_KEY.", stage="configuration")
        return "openai", None
    if selected in {"command", "custom"}:
        if not configured:
            raise TranscriptionError("The command transcription backend requires MEDIA_TRANSCRIPTION_COMMAND.", stage="configuration")
        return "command", configured
    if selected in {"whisper", "local", "local-whisper"}:
        discovered = _local_whisper_command()
        if not discovered:
            raise TranscriptionError("The local Whisper transcription backend is not installed.", stage="configuration")
        return "whisper", discovered
    raise TranscriptionError("MEDIA_TRANSCRIPTION_PROVIDER must be auto, openai, whisper, or command.", stage="configuration")


def transcribe_bytes(data: bytes, media_kind: str, mimetype: str = "", command=None,
                     chunk_seconds=None, max_seconds=None, timeout=None, file_id="unknown",
                     provider=None):
    if not data:
        raise TranscriptionError("The media file is empty.", stage="validation")
    backend, command = _transcription_backend(command, provider)
    logger.info("transcription_provider_selected provider=%s file_id=%s", backend, redact(file_id))
    try:
        chunk_seconds = int(chunk_seconds or os.getenv("MEDIA_TRANSCRIPTION_CHUNK_SECONDS", "600"))
        max_seconds = int(max_seconds or os.getenv("MEDIA_TRANSCRIPTION_MAX_SECONDS", "14400"))
        timeout = int(timeout or os.getenv("MEDIA_TRANSCRIPTION_TIMEOUT_SECONDS", "900"))
    except (TypeError, ValueError) as exc:
        raise TranscriptionError(
            "Media transcription limits must be valid whole numbers.",
            stage="configuration") from exc
    if chunk_seconds <= 0 or max_seconds <= 0 or timeout <= 0:
        raise TranscriptionError(
            "Media transcription limits must be greater than zero.", stage="configuration")
    if not shutil.which("ffprobe"):
        raise TranscriptionError(
            "Required media tool 'ffprobe' is not installed.", stage="ffprobe")
    if not shutil.which("ffmpeg"):
        raise TranscriptionError(
            "Required media tool 'ffmpeg' is not installed.", stage="ffmpeg")

    suffix = ".mp4" if media_kind == "video" else ".audio"
    with tempfile.TemporaryDirectory(prefix="slack-media-") as temp_dir:
        source = Path(temp_dir) / ("source" + suffix)
        source.write_bytes(data)
        if not _has_audio(source):
            raise TranscriptionError(
                "The media contains no usable audio track.", stage="audio_validation")
        duration = _duration(source)
        if duration and duration > max_seconds:
            raise TranscriptionError(
                f"The recording is longer than the configured {max_seconds // 60}-minute limit.",
                stage="duration_validation")
        logger.info("media_validation_completed file_id=%s media_type=%s duration_seconds=%.3f",
                    redact(file_id), media_kind, duration or 0.0)
        pattern = Path(temp_dir) / "chunk-%04d.wav"
        _run([
            "ffmpeg", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
            "-f", "segment", "-segment_time", str(chunk_seconds), str(pattern),
        ], timeout=max(timeout, 120), hidden_paths=(temp_dir, source),
            label="FFmpeg audio normalization", stage="ffmpeg")
        chunks = sorted(Path(temp_dir).glob("chunk-*.wav"))
        if not chunks:
            raise TranscriptionError(
                "No usable speech audio could be extracted from the media.", stage="ffmpeg")
        if backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            model = os.getenv("MEDIA_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe").strip()
            transcripts = [_invoke_openai(
                chunk, api_key, model, timeout, file_id=file_id,
                chunk_index=index, chunk_count=len(chunks))
                for index, chunk in enumerate(chunks, 1)]
        else:
            transcripts = [_invoke_provider(
                chunk, command, timeout, file_id=file_id,
                chunk_index=index, chunk_count=len(chunks))
                for index, chunk in enumerate(chunks, 1)]
        combined = "\n".join(part for part in transcripts if part.strip()).strip()
        if not combined:
            raise TranscriptionError(
                "The recording did not produce usable transcript text.",
                stage="provider_output")
        return Transcript(combined, len(chunks), duration)
