from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from deadline_reminders import (
    DeadlineReminderScheduler, ReminderSettings, ReminderStore, ReminderTask,
    WeeklySummaryDelivery, WeeklySummarySettings,
)


TODAY = date(2026, 9, 23)


def _task(task_id="I1", *, due=TODAY, completed=False):
    return ReminderTask(task_id, "Review deployment", "UP", "Praveen", due, "P1", completed)


def _scheduler(tmp_path, tasks):
    sent = []
    now = datetime(2026, 9, 23, 9, 0, tzinfo=ZoneInfo("Asia/Kathmandu"))
    scheduler = DeadlineReminderScheduler(
        ReminderSettings(enabled=True), ReminderStore(tmp_path / "state.sqlite3"),
        lambda: list(tasks), lambda recipient, message: sent.append((recipient, message)),
        now=lambda: now,
    )
    return scheduler, sent


def test_task_due_today_generates_reminder(tmp_path):
    scheduler, sent = _scheduler(tmp_path, [_task()])
    result = scheduler.scan(TODAY)
    assert result == {"scanned": 1, "sent": 1, "skipped": 0}
    assert sent[0][0] == "UP"
    assert "Action Item Follow-up" in sent[0][1]
    assert "*Task:* Review deployment" in sent[0][1]
    assert "*Owner:* Praveen" in sent[0][1]


def test_task_due_tomorrow_generates_reminder(tmp_path):
    scheduler, sent = _scheduler(tmp_path, [_task(due=TODAY + timedelta(days=1))])
    scheduler.scan(TODAY)
    assert len(sent) == 1 and "due tomorrow" in sent[0][1]


def test_today_and_tomorrow_are_combined_in_one_deadline_message(tmp_path):
    scheduler, sent = _scheduler(tmp_path, [
        _task("I1", due=TODAY),
        _task("I2", due=TODAY + timedelta(days=1)),
    ])
    result = scheduler.scan(TODAY)
    assert result["sent"] == 2 and len(sent) == 1
    assert "Due today" in sent[0][1] and "Due tomorrow" in sent[0][1]


def test_completed_task_never_generates_reminder(tmp_path):
    scheduler, sent = _scheduler(tmp_path, [_task(completed=True)])
    result = scheduler.scan(TODAY)
    assert result == {"scanned": 0, "sent": 0, "skipped": 0}
    assert sent == []


def test_already_reminded_task_is_not_duplicated(tmp_path):
    scheduler, sent = _scheduler(tmp_path, [_task()])
    scheduler.scan(TODAY)
    result = scheduler.scan(TODAY)
    assert len(sent) == 1
    assert result == {"scanned": 1, "sent": 0, "skipped": 1}


def test_overdue_task_uses_separate_overdue_reminder(tmp_path):
    task = ReminderTask("I1", "Review deployment", "UP", "Praveen",
                        TODAY - timedelta(days=2), "P2")
    scheduler, sent = _scheduler(tmp_path, [task])
    scheduler.scan(TODAY)
    assert len(sent) == 1
    assert "Action Item Follow-up" in sent[0][1] and "is overdue" in sent[0][1]


def test_significantly_overdue_task_is_escalated(tmp_path):
    task = ReminderTask("I1", "Review deployment", "UP", "Praveen",
                        TODAY - timedelta(days=5), "P2")
    scheduler, sent = _scheduler(tmp_path, [task])
    scheduler.scan(TODAY)
    assert len(sent) == 1
    assert "significantly overdue" in sent[0][1]


def test_p1_overdue_is_clearly_highlighted(tmp_path):
    scheduler, sent = _scheduler(tmp_path, [_task(due=TODAY - timedelta(days=2))])
    scheduler.scan(TODAY)
    assert "P1 action item is overdue" in sent[0][1]


def test_configured_unassigned_fallback_is_not_a_random_user(tmp_path):
    fallback = ReminderTask(
        "I1", "Unassigned deadline", "channel:C-ADMIN", "Unassigned", TODAY, "P2")
    scheduler, sent = _scheduler(tmp_path, [fallback])
    scheduler.scan(TODAY)
    assert sent[0][0] == "channel:C-ADMIN"
    assert "Unassigned" in sent[0][1]


def test_due_soon_window_is_configurable(tmp_path):
    scheduler, sent = _scheduler(tmp_path, [
        ReminderTask("I1", "Soon", "UP", "Praveen", TODAY + timedelta(days=2), "P2")])
    scheduler.settings = ReminderSettings(enabled=True, due_soon_hours=48)
    scheduler.scan(TODAY)
    assert "due soon" in sent[0][1]


def test_high_priority_approaching_deadline_is_eligible(tmp_path):
    scheduler, sent = _scheduler(tmp_path, [_task(due=TODAY + timedelta(days=3))])
    scheduler.scan(TODAY)
    assert len(sent) == 1
    assert "high-priority action item" in sent[0][1]


def test_cancelled_task_is_excluded(tmp_path):
    task = ReminderTask("I1", "Cancelled", "UP", "Praveen", TODAY, "P1", False, "cancelled")
    scheduler, sent = _scheduler(tmp_path, [task])
    result = scheduler.scan(TODAY)
    assert result["sent"] == 0 and sent == []


def test_notification_failure_does_not_stop_other_recipients(tmp_path):
    tasks = [_task("I1"), ReminderTask("I2", "Other", "UA", "AasthaA", TODAY, "P2")]
    scheduler, sent = _scheduler(tmp_path, tasks)

    def deliver(recipient, message):
        if recipient == "UP":
            raise RuntimeError("Slack unavailable")
        sent.append((recipient, message))

    scheduler.send = deliver
    result = scheduler.scan(TODAY)
    assert result["sent"] == 1
    assert [recipient for recipient, _ in sent] == ["UA"]
    scheduler.send = lambda recipient, message: sent.append((recipient, message))
    retry = scheduler.scan(TODAY)
    assert retry["sent"] == 1
    assert [recipient for recipient, _ in sent] == ["UA", "UP"]


def test_notification_failure_has_structured_follow_up_log(tmp_path, caplog):
    caplog.set_level("ERROR")
    scheduler, _ = _scheduler(tmp_path, [_task()])
    scheduler.send = lambda recipient, message: (_ for _ in ()).throw(
        RuntimeError("Slack unavailable"))
    assert scheduler.scan(TODAY)["sent"] == 0
    assert "follow_up_failed stage=notification" in caplog.text


def test_shared_store_prevents_duplicate_scheduler_instances(tmp_path):
    tasks = [_task()]
    first, first_sent = _scheduler(tmp_path, tasks)
    second, second_sent = _scheduler(tmp_path, tasks)
    first.scan(TODAY)
    result = second.scan(TODAY)
    assert len(first_sent) == 1 and second_sent == []
    assert result["skipped"] == 1


def test_production_default_polls_without_changing_owner_local_nine_am(tmp_path, monkeypatch):
    monkeypatch.delenv("FOLLOW_UP_INTERVAL_MINUTES", raising=False)
    monkeypatch.delenv("FOLLOWUP_SCAN_INTERVAL_SECONDS", raising=False)
    settings = ReminderSettings.from_env()
    now = datetime(2026, 9, 23, 19, 1, tzinfo=ZoneInfo("Asia/Kathmandu"))
    scheduler = DeadlineReminderScheduler(
        settings, ReminderStore(tmp_path / "state.sqlite3"), lambda: [],
        lambda recipient, message: None, now=lambda: now)
    assert settings.interval_enabled is False
    assert scheduler._next_delay(now) == 3600


def test_explicit_one_minute_interval_uses_same_scheduler_and_dedup_store(
        tmp_path, monkeypatch):
    monkeypatch.setenv("FOLLOW_UP_INTERVAL_MINUTES", "1")
    monkeypatch.delenv("FOLLOWUP_SCAN_INTERVAL_SECONDS", raising=False)
    settings = ReminderSettings.from_env()
    sent = []
    now = datetime(2026, 9, 23, 19, 1, tzinfo=ZoneInfo("Asia/Kathmandu"))
    scheduler = DeadlineReminderScheduler(
        settings, ReminderStore(tmp_path / "state.sqlite3"), lambda: [_task()],
        lambda recipient, message: sent.append((recipient, message)), now=lambda: now)
    assert settings.interval_enabled is True
    assert settings.scan_interval_seconds == 60
    assert scheduler._next_delay(now) == 60
    assert scheduler.scan(TODAY)["sent"] == 1
    assert scheduler.scan(TODAY) == {"scanned": 1, "sent": 0, "skipped": 1}
    assert sent[0][0] == "UP" and "Review deployment" in sent[0][1]


def test_owner_local_time_controls_production_reminder_selection(tmp_path):
    now = datetime(2026, 9, 23, 3, 30, tzinfo=ZoneInfo("UTC"))
    tasks = [
        ReminderTask("IK", "Kathmandu task", "UK", "AasthaA", TODAY,
                     "P2", False, None, "Asia/Kathmandu"),
        ReminderTask("IN", "New York task", "UN", "Morgan", TODAY,
                     "P2", False, None, "America/New_York"),
    ]
    sent = []
    scheduler = DeadlineReminderScheduler(
        ReminderSettings(enabled=True, hour=9, minute=0, due_tomorrow=False,
                         due_soon_hours=0),
        ReminderStore(tmp_path / "state.sqlite3"), lambda: tasks,
        lambda recipient, message: sent.append((recipient, message)), now=lambda: now)
    result = scheduler.scan()
    assert result["sent"] == 1
    assert [recipient for recipient, _ in sent] == ["UK"]


def test_two_owner_timezones_receive_at_different_local_times(tmp_path):
    current = [datetime(2026, 9, 23, 3, 30, tzinfo=ZoneInfo("UTC"))]
    tasks = [
        ReminderTask("IK", "Kathmandu task", "UK", "AasthaA", TODAY,
                     "P2", False, None, "Asia/Kathmandu"),
        ReminderTask("IN", "New York task", "UN", "Morgan", TODAY,
                     "P2", False, None, "America/New_York"),
    ]
    sent = []
    scheduler = DeadlineReminderScheduler(
        ReminderSettings(enabled=True, hour=9, minute=0, due_tomorrow=False,
                         due_soon_hours=0),
        ReminderStore(tmp_path / "state.sqlite3"), lambda: tasks,
        lambda recipient, message: sent.append((recipient, message)), now=lambda: current[0])
    scheduler.scan()
    current[0] = datetime(2026, 9, 23, 14, 0, tzinfo=ZoneInfo("UTC"))
    scheduler.scan()
    assert [recipient for recipient, _ in sent] == ["UK", "UN"]


def test_scheduler_disabled_does_not_start_thread(tmp_path):
    scheduler, _ = _scheduler(tmp_path, [])
    scheduler.settings = ReminderSettings(enabled=False)
    assert scheduler.start() is False
    assert scheduler._thread is None


def test_scheduler_shutdown_interrupts_waiting_thread(tmp_path):
    scheduler, _ = _scheduler(tmp_path, [])
    assert scheduler.start() is True
    scheduler.stop(timeout=1)
    assert scheduler._thread is not None and not scheduler._thread.is_alive()


def test_automatic_weekly_summary_is_persistently_deduplicated(tmp_path):
    sent = []
    now = datetime(2026, 9, 25, 17, 0, tzinfo=ZoneInfo("Asia/Kathmandu"))
    settings = WeeklySummarySettings(
        enabled=True, day=4, hour=17, channel="C", actor_id="UA")

    def build(start, end):
        return WeeklySummaryDelivery("L", "C", start, end, "Weekly report")

    first = DeadlineReminderScheduler(
        ReminderSettings(), ReminderStore(tmp_path / "state.sqlite3"), lambda: [],
        lambda recipient, message: None, now=lambda: now,
        weekly_settings=settings, build_weekly=build,
        send_weekly=lambda channel, message: sent.append((channel, message)))
    second = DeadlineReminderScheduler(
        ReminderSettings(), ReminderStore(tmp_path / "state.sqlite3"), lambda: [],
        lambda recipient, message: None, now=lambda: now,
        weekly_settings=settings, build_weekly=build,
        send_weekly=lambda channel, message: sent.append((channel, message)))
    assert first.scan_weekly() == {"sent": 1, "skipped": 0}
    assert second.scan_weekly() == {"sent": 0, "skipped": 1}
    assert sent == [("C", "Weekly report")]


def test_weekly_summary_failure_is_retriable(tmp_path):
    now = datetime(2026, 9, 25, 17, 0, tzinfo=ZoneInfo("Asia/Kathmandu"))
    settings = WeeklySummarySettings(
        enabled=True, day=4, hour=17, channel="C", actor_id="UA")
    delivery = WeeklySummaryDelivery(
        "L", "C", date(2026, 9, 21), date(2026, 9, 27), "Weekly report")
    attempts = []
    scheduler = DeadlineReminderScheduler(
        ReminderSettings(), ReminderStore(tmp_path / "state.sqlite3"), lambda: [],
        lambda recipient, message: None, now=lambda: now,
        weekly_settings=settings, build_weekly=lambda start, end: delivery,
        send_weekly=lambda channel, message: (_ for _ in ()).throw(RuntimeError("Slack down")))
    assert scheduler.scan_weekly()["sent"] == 0
    scheduler.send_weekly = lambda channel, message: attempts.append(channel)
    assert scheduler.scan_weekly()["sent"] == 1
    assert attempts == ["C"]
