import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import pytest


@pytest.fixture
def lambda_environment(monkeypatch):
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "slack-list-assistant-test")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing-test")
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    return os.environ


"""Offline test isolation: no real credentials, network requests or persistent state."""
import socket
import os

# Set before test-module imports so .env credentials are never loaded by tests.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import pytest


@pytest.fixture(autouse=True)
def offline_state(monkeypatch, tmp_path):
    from src import app as main
    from src import graph as intent_parser
    from src import slack_client as slack_tools
    from src import config

    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setattr(intent_parser, "OLLAMA_API_KEY", "")
    monkeypatch.setattr(main, "BOT_TOKEN", "")
    monkeypatch.setattr(main, "APP_TOKEN", "")
    monkeypatch.setattr(config, "DEFAULT_LIST_ID", "L_TEST")
    main._pending.clear()
    monkeypatch.setattr(slack_tools, "user_display_name", lambda uid: uid or None)

    def no_network(*args, **kwargs):
        raise AssertionError("Network calls are forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    yield
    main._pending.clear()
