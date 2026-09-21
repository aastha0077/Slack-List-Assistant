"""Offline test isolation: no real credentials, network requests or persistent state."""
import socket
import os

# Set before test-module imports so .env credentials are never loaded by tests.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import pytest


@pytest.fixture(autouse=True)
def offline_state(monkeypatch, tmp_path):
    import main
    import intent_parser
    import slack_tools
    import config

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
