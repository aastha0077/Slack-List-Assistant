"""Safe, provider-driven speech transcription for Slack media.

``MEDIA_TRANSCRIPTION_COMMAND`` is an argv template containing ``{input}``.
Providers may return transcript text on stdout or write a textual transcript
file. ffmpeg/ffprobe provide format normalization, audio-track validation, and
bounded chunking before the provider is invoked.
"""
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


def transcribe_bytes(data: bytes, media_kind: str, mimetype: str = "", command=None,
                     chunk_seconds=None, max_seconds=None, timeout=None, file_id="unknown"):
    if not data:
        raise TranscriptionError("The media file is empty.", stage="validation")
    command = command or os.getenv("MEDIA_TRANSCRIPTION_COMMAND", "").strip()
    if not command:
        raise TranscriptionError(
            "Audio/video transcription is not configured. Set MEDIA_TRANSCRIPTION_COMMAND "
            "to a trusted speech-to-text command that accepts {input}.", stage="configuration")
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
        transcripts = [
            _invoke_provider(
                chunk, command, timeout, file_id=file_id,
                chunk_index=index, chunk_count=len(chunks),
            )
            for index, chunk in enumerate(chunks, 1)
        ]
        combined = "\n".join(part for part in transcripts if part.strip()).strip()
        if not combined:
            raise TranscriptionError(
                "The recording did not produce usable transcript text.",
                stage="provider_output")
        return Transcript(combined, len(chunks), duration)
