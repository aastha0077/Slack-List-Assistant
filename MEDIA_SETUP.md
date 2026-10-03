# Slack media command and ingestion setup

Audio and video commands are transcribed and enter the same normalized request, intent, authorization, execution, and verification path as typed commands. Explicit action-item extraction requests retain the existing extraction workflow.

## Slack file access

Add the `files:read` **Bot Token Scope** under **OAuth & Permissions**, then reinstall the Slack app to the workspace so the bot token receives the new scope. Slack documents `files:read` as the scope for `files.info` and for downloading files shared in conversations where the app is present.

The bot refreshes event file stubs through `files.info` and downloads `url_private_download` with the existing bot token in the HTTP `Authorization` header. Tokens and private URLs are never logged.

## Speech-to-text provider

The default `auto` backend uses `MEDIA_TRANSCRIPTION_COMMAND` when configured,
then the OpenAI transcription API when `OPENAI_API_KEY` is configured, and
finally an installed local Whisper CLI. This project environment includes the
Whisper CLI and cached model, so media transcription works without setting a
custom command.

```text
MEDIA_TRANSCRIPTION_PROVIDER=auto
OPENAI_API_KEY=...
MEDIA_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
```

A local Whisper CLI is also supported with `MEDIA_TRANSCRIPTION_PROVIDER=whisper` and optional `MEDIA_WHISPER_MODEL=turbo`. The existing trusted command backend remains supported; its command must contain `{input}` and may return stdout or a `.txt`, `.vtt`, `.srt`, `.tsv`, or JSON transcript file.

For stdout-based providers:

```text
MEDIA_TRANSCRIPTION_COMMAND=/absolute/path/to/transcriber --input {input}
```

For Whisper CLI, file output is supported directly:

```text
MEDIA_TRANSCRIPTION_COMMAND=/absolute/path/to/whisper {input} --model turbo --output_format txt --output_dir /tmp
```

The application replaces a declared `--output_dir`/`--output-dir` with an isolated temporary directory. When `--output_format`/`--output-format` declares a supported textual format, the expected result is resolved deterministically from the normalized chunk's filename stem (for example, `chunk-0000.wav` → `chunk-0000.txt`). Other file-based providers can use `{output_dir}` in their command or write into their process working directory. Generated transcript files are removed after they are read, including when processing fails.

Successful media requests emit stage logs without transcript content:

```text
transcription_started
transcription_provider_started
transcription_provider_completed
transcription_completed
```

Failures emit `transcription_provider_failed` when applicable and `transcription_failed` with the failing stage (`ffprobe`, `ffmpeg`, `audio_validation`, `duration_validation`, `provider`, or `provider_output`).

The exact command depends on the speech-to-text provider installed by the operator. No provider credential should be placed in source code.

Optional controls:

```text
MEDIA_MAX_BYTES=262144000
MEDIA_TRANSCRIPTION_PROVIDER=auto
MEDIA_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
MEDIA_WHISPER_MODEL=turbo
MEDIA_TRANSCRIPTION_CHUNK_SECONDS=600
MEDIA_TRANSCRIPTION_MAX_SECONDS=14400
MEDIA_TRANSCRIPTION_TIMEOUT_SECONDS=900
```

`ffmpeg` and `ffprobe` are required. The application validates the audio stream, enforces the duration limit, converts audio/video audio tracks to mono 16 kHz WAV, and transcribes bounded chunks. Startup/runtime environments must make both media tools available on `PATH`. A bare transcription command is resolved from `PATH` first and then beside the active Python executable, so `whisper` installed in the project's active virtual environment works without changing the command; an absolute provider path is also supported.
