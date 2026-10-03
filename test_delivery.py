from datetime import date, datetime, timezone
from enum import Enum
import sqlite3
from uuid import UUID

import delivery


class State(Enum):
    READY = "ready"


def test_checkpoint_serializes_date_and_datetime_at_json_boundary(tmp_path):
    database = tmp_path / "delivery.sqlite3"
    db = lambda: sqlite3.connect(database)
    parsed = {
        "intent": "simulation",
        "scenario": {
            "due_date": date(2026, 10, 9),
            "created_at": datetime(2026, 10, 3, 12, 30, tzinfo=timezone.utc),
            "state": State.READY,
            "operation_id": UUID("12345678-1234-5678-1234-567812345678"),
        },
    }

    with delivery.event(db, "event-with-dates"):
        key = delivery.checkpoint_key("request", {"due": date(2026, 10, 9)})
        delivery.checkpoint_write(key, "parsed", {"parsed": parsed})
        saved = delivery.checkpoint_read(key)

    assert saved["parsed"]["scenario"]["due_date"] == "2026-10-09"
    assert saved["parsed"]["scenario"]["created_at"] == "2026-10-03 12:30:00+00:00"
    assert saved["parsed"]["scenario"]["state"] == "ready"
    assert saved["parsed"]["scenario"]["operation_id"] == "12345678-1234-5678-1234-567812345678"
