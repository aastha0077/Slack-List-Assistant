from datetime import date, datetime, timezone
import sqlite3

import delivery


def test_checkpoint_serializes_date_and_datetime_at_json_boundary(tmp_path):
    database = tmp_path / "delivery.sqlite3"
    db = lambda: sqlite3.connect(database)
    parsed = {
        "intent": "simulation",
        "scenario": {
            "due_date": date(2026, 10, 9),
            "created_at": datetime(2026, 10, 3, 12, 30, tzinfo=timezone.utc),
        },
    }

    with delivery.event(db, "event-with-dates"):
        key = delivery.checkpoint_key("request", {"due": date(2026, 10, 9)})
        delivery.checkpoint_write(key, "parsed", {"parsed": parsed})
        saved = delivery.checkpoint_read(key)

    assert saved["parsed"]["scenario"]["due_date"] == "2026-10-09"
    assert saved["parsed"]["scenario"]["created_at"] == "2026-10-03 12:30:00+00:00"
