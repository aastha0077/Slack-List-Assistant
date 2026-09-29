"""Read-only deadline reminders for Slack List action items."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Callable, Iterable
from zoneinfo import ZoneInfo


logger = logging.getLogger("slack_list.reminders")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


@dataclass(frozen=True)
class ReminderSettings:
    enabled: bool = True
    hour: int = 9
    minute: int = 0
    timezone: str = "Asia/Kathmandu"
    due_today: bool = True
    due_tomorrow: bool = True
    overdue: bool = True
    scan_interval_seconds: int = 3600
    interval_enabled: bool = False
    repeat_hours: int = 24
    escalation_days: int = 3
    high_priority_days: int = 3
    notification_behavior: str = "direct"
    due_soon_hours: int = 24
    fallback_channel: str = ""
    fallback_actor_id: str = ""
    workspace_timezone: str = ""
    application_timezone_configured: bool = False

    @classmethod
    def from_env(cls) -> "ReminderSettings":
        configured_timezone = os.getenv("DEADLINE_REMINDER_TIMEZONE")
        timezone = (configured_timezone or "Asia/Kathmandu").strip()
        try:
            ZoneInfo(timezone)
        except Exception:
            logger.warning("scheduler_error stage=config reason=invalid_timezone fallback=Asia/Kathmandu")
            timezone = "Asia/Kathmandu"
        workspace_timezone = os.getenv("SLACK_WORKSPACE_TIMEZONE", "").strip()
        if workspace_timezone:
            try:
                ZoneInfo(workspace_timezone)
            except Exception:
                logger.warning("scheduler_error stage=config reason=invalid_workspace_timezone")
                workspace_timezone = ""
        behavior = os.getenv("FOLLOWUP_NOTIFICATION_BEHAVIOR", "direct").strip().casefold() or "direct"
        if behavior not in {"direct", "disabled"}:
            logger.warning("scheduler_error stage=config reason=invalid_notification_behavior fallback=direct")
            behavior = "direct"
        enabled = (_env_bool("FOLLOW_UP_ENABLED", _env_bool("DEADLINE_REMINDERS_ENABLED", True))
                   and behavior != "disabled")
        interval_minutes = os.getenv("FOLLOW_UP_INTERVAL_MINUTES")
        legacy_interval = os.getenv("FOLLOWUP_SCAN_INTERVAL_SECONDS")
        interval_enabled = interval_minutes is not None or legacy_interval is not None
        interval = (_env_int("FOLLOW_UP_INTERVAL_MINUTES", 60, 1, 1440) * 60
                    if interval_minutes is not None
                    else _env_int("FOLLOWUP_SCAN_INTERVAL_SECONDS", 3600, 60, 86400))
        return cls(
            enabled=enabled,
            hour=_env_int("DEADLINE_REMINDER_HOUR", 9, 0, 23),
            minute=_env_int("DEADLINE_REMINDER_MINUTE", 0, 0, 59),
            timezone=timezone,
            due_today=_env_bool("DEADLINE_REMINDER_DUE_TODAY", True),
            due_tomorrow=_env_bool("DEADLINE_REMINDER_DUE_TOMORROW", True),
            overdue=_env_bool("FOLLOW_UP_OVERDUE_ENABLED",
                              _env_bool("DEADLINE_REMINDER_OVERDUE", True)),
            scan_interval_seconds=interval,
            interval_enabled=interval_enabled,
            repeat_hours=_env_int("FOLLOWUP_REPEAT_HOURS", 24, 1, 720),
            escalation_days=_env_int("FOLLOWUP_ESCALATION_DAYS", 3, 1, 365),
            high_priority_days=_env_int("FOLLOWUP_HIGH_PRIORITY_DAYS", 3, 2, 30),
            notification_behavior=behavior,
            due_soon_hours=_env_int("FOLLOW_UP_DUE_SOON_HOURS", 24, 0, 720),
            fallback_channel=os.getenv("FOLLOW_UP_CHANNEL", "").strip(),
            fallback_actor_id=os.getenv("FOLLOW_UP_FALLBACK_ACTOR_ID", "").strip(),
            workspace_timezone=workspace_timezone,
            application_timezone_configured=configured_timezone is not None,
        )


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


@dataclass(frozen=True)
class WeeklySummarySettings:
    enabled: bool = False
    day: int = 4
    hour: int = 17
    minute: int = 0
    channel: str = ""
    actor_id: str = ""

    @classmethod
    def from_env(cls) -> "WeeklySummarySettings":
        raw_day = os.getenv("WEEKLY_SUMMARY_DAY", "friday").strip().casefold()
        try:
            day = int(raw_day)
        except ValueError:
            day = _WEEKDAYS.get(raw_day, 4)
        day = min(6, max(0, day))
        return cls(
            enabled=_env_bool("WEEKLY_SUMMARY_ENABLED", False),
            day=day,
            hour=_env_int("WEEKLY_SUMMARY_HOUR", 17, 0, 23),
            minute=_env_int("WEEKLY_SUMMARY_MINUTE", 0, 0, 59),
            channel=os.getenv("WEEKLY_SUMMARY_CHANNEL", "").strip(),
            actor_id=os.getenv("WEEKLY_SUMMARY_ACTOR_ID", "").strip(),
        )


@dataclass(frozen=True)
class WeeklySummaryDelivery:
    list_id: str
    channel_id: str
    period_start: date
    period_end: date
    message: str


@dataclass(frozen=True)
class ReminderTask:
    task_id: str
    name: str
    owner_id: str
    owner_name: str
    due_date: date
    priority: str | None = None
    completed: bool = False
    status: str | None = None
    timezone: str = "UTC"


class ReminderStore:
    """Persistent once-per-task/recipient/day delivery ledger."""

    def __init__(self, path: str):
        self.path = str(Path(path))
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS deadline_reminders (
                    task_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    reminder_date TEXT NOT NULL,
                    category TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, recipient_id, reminder_date)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS weekly_summary_runs (
                    list_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    sent_at REAL,
                    lease_until REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (list_id, channel_id, period_start, period_end)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS followup_notifications (
                    task_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    last_sent REAL,
                    lease_until REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (task_id, recipient_id, condition)
                )
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def was_sent(self, task_id: str, recipient_id: str, reminder_date: date) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM deadline_reminders WHERE task_id=? AND recipient_id=? AND reminder_date=?",
                (task_id, recipient_id, reminder_date.isoformat()),
            ).fetchone()
        return bool(row)

    def mark_sent(self, task_id: str, recipient_id: str, reminder_date: date,
                  category: str, sent_at: datetime) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO deadline_reminders VALUES (?, ?, ?, ?, ?)",
                (task_id, recipient_id, reminder_date.isoformat(), category, sent_at.isoformat()),
            )

    def claim(self, task: ReminderTask, condition: str, now: datetime,
              repeat_hours: int, lease_seconds: int = 300) -> bool:
        """Atomically lease one notification across scheduler instances."""
        state_hash = hashlib.sha256(json.dumps({
            "due": task.due_date.isoformat(), "priority": task.priority,
            "status": task.status, "completed": task.completed,
        }, sort_keys=True).encode()).hexdigest()
        timestamp = now.timestamp()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state_hash, last_sent, lease_until FROM followup_notifications "
                "WHERE task_id=? AND recipient_id=? AND condition=?",
                (task.task_id, task.owner_id, condition),
            ).fetchone()
            if row and row[2] > timestamp:
                return False
            if (row and row[0] == state_hash and row[1] is not None
                    and timestamp - row[1] < repeat_hours * 3600):
                return False
            connection.execute(
                "INSERT INTO followup_notifications VALUES (?, ?, ?, ?, NULL, ?) "
                "ON CONFLICT(task_id, recipient_id, condition) DO UPDATE SET "
                "state_hash=excluded.state_hash, lease_until=excluded.lease_until",
                (task.task_id, task.owner_id, condition, state_hash, timestamp + lease_seconds),
            )
        return True

    def delivered(self, task: ReminderTask, condition: str, now: datetime) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE followup_notifications SET last_sent=?, lease_until=0 "
                "WHERE task_id=? AND recipient_id=? AND condition=?",
                (now.timestamp(), task.task_id, task.owner_id, condition),
            )

    def release(self, task: ReminderTask, condition: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE followup_notifications SET last_sent=NULL, lease_until=0 "
                "WHERE task_id=? AND recipient_id=? AND condition=?",
                (task.task_id, task.owner_id, condition),
            )

    def claim_weekly(self, delivery: WeeklySummaryDelivery, now: datetime,
                     lease_seconds: int = 300) -> bool:
        """Atomically lease one channel/period summary across app instances."""
        timestamp = now.timestamp()
        key = (delivery.list_id, delivery.channel_id,
               delivery.period_start.isoformat(), delivery.period_end.isoformat())
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT sent_at, lease_until FROM weekly_summary_runs "
                "WHERE list_id=? AND channel_id=? AND period_start=? AND period_end=?", key,
            ).fetchone()
            if row and (row[0] is not None or row[1] > timestamp):
                return False
            connection.execute(
                "INSERT INTO weekly_summary_runs VALUES (?, ?, ?, ?, NULL, ?) "
                "ON CONFLICT(list_id, channel_id, period_start, period_end) DO UPDATE SET "
                "lease_until=excluded.lease_until",
                (*key, timestamp + lease_seconds),
            )
        return True

    def weekly_delivered(self, delivery: WeeklySummaryDelivery, now: datetime) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE weekly_summary_runs SET sent_at=?, lease_until=0 "
                "WHERE list_id=? AND channel_id=? AND period_start=? AND period_end=?",
                (now.timestamp(), delivery.list_id, delivery.channel_id,
                 delivery.period_start.isoformat(), delivery.period_end.isoformat()),
            )

    def release_weekly(self, delivery: WeeklySummaryDelivery) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE weekly_summary_runs SET lease_until=0 "
                "WHERE list_id=? AND channel_id=? AND period_start=? AND period_end=?",
                (delivery.list_id, delivery.channel_id,
                 delivery.period_start.isoformat(), delivery.period_end.isoformat()),
            )


def _task_line(task: ReminderTask, category: str, today: date) -> str:
    values = [escape(task.name, quote=False)]
    if task.priority:
        values.append(escape(task.priority, quote=False))
    values.append(escape(task.owner_name, quote=False))
    if category in {"overdue", "escalated", "high_priority_overdue"}:
        days = (today - task.due_date).days
        values.append(f"{days} day{'s' if days != 1 else ''} overdue")
    elif category in {"high_priority", "due_soon"}:
        days = (task.due_date - today).days
        values.append(f"Due in {days} days")
    elif task.due_date == today:
        values.append("Due today")
    else:
        values.append("Due tomorrow")
    return "• " + " — ".join(values)


def format_reminder(tasks: Iterable[ReminderTask], category: str, today: date) -> str:
    tasks = list(tasks)
    if len(tasks) == 1:
        task = tasks[0]
        due = task.due_date.strftime("%b %d").replace(" 0", " ")
        prompt = {
            "overdue": "This action item is overdue. Please review and complete it.",
            "high_priority_overdue": "This P1 action item is overdue. Please review it promptly.",
            "escalated": "This action item is significantly overdue and needs attention.",
            "today": "This action item is due today. Please review and complete it.",
            "tomorrow": "This action item is due tomorrow. Please review it.",
            "due_soon": "This action item is due soon. Please review and complete it.",
            "high_priority": "This high-priority action item is approaching its deadline.",
            "deadline": "Please review and complete this action item.",
        }.get(category, "Please review and complete this action item.")
        fields = [f"*Task:* {escape(task.name, quote=False)}",
                  f"*Owner:* {escape(task.owner_name, quote=False)}"]
        if task.priority:
            fields.append(f"*Priority:* {escape(task.priority, quote=False)}")
        fields.append(f"*Due:* {due}")
        return "🔔 *Action Item Follow-up*\n\n" + "\n".join(fields) + f"\n\n{prompt}"
    title = {
        "overdue": "⚠️ Follow-up · Overdue",
        "high_priority_overdue": "🚨 Follow-up · P1 Overdue",
        "escalated": "🚨 Escalated Follow-up",
        "high_priority": "🔔 High-priority Deadline",
        "due_soon": "🔔 Follow-up · Due Soon",
    }.get(category, "🔔 Follow-up · Deadline")
    return f"*{title}*\n\n" + "\n".join(_task_line(task, category, today) for task in tasks)


class DeadlineReminderScheduler:
    """Daily, non-blocking scheduler with explicit graceful shutdown."""

    def __init__(self, settings: ReminderSettings, store: ReminderStore,
                 load_tasks: Callable[[], Iterable[ReminderTask]],
                 send: Callable[[str, str], None], now: Callable[[], datetime] | None = None,
                 weekly_settings: WeeklySummarySettings | None = None,
                 build_weekly: Callable[[date, date], WeeklySummaryDelivery | None] | None = None,
                 send_weekly: Callable[[str, str], None] | None = None):
        self.settings = settings
        self.store = store
        self.load_tasks = load_tasks
        self.send = send
        self.now = now or (lambda: datetime.now(ZoneInfo(settings.timezone)))
        self.weekly_settings = weekly_settings or WeeklySummarySettings()
        self.build_weekly = build_weekly
        self.send_weekly = send_weekly
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not self.settings.enabled and not self.weekly_settings.enabled:
            logger.info("reminder_scheduler_disabled")
            return False
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="slack-deadline-reminders", daemon=True)
        self._thread.start()
        if self.settings.interval_enabled:
            logger.info(
                "reminder_scheduler_started mode=interval interval_minutes=%d "
                "production_hour=%d production_minute=%d timezone=%s",
                self.settings.scan_interval_seconds // 60,
                self.settings.hour, self.settings.minute, self.settings.timezone)
        else:
            logger.info(
                "reminder_scheduler_started mode=owner_local hour=%d minute=%d "
                "poll_minutes=%d fallback_timezone=%s",
                self.settings.hour, self.settings.minute,
                self.settings.scan_interval_seconds // 60, self.settings.timezone)
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        logger.info("reminder_scheduler_stopped")

    def _next_run(self, now: datetime) -> datetime:
        target = now.replace(hour=self.settings.hour, minute=self.settings.minute,
                             second=0, microsecond=0)
        reminder_target = target if target > now else target + timedelta(days=1)
        weekly = self.weekly_settings
        if not weekly.enabled:
            return reminder_target
        days = (weekly.day - now.weekday()) % 7
        weekly_target = (now + timedelta(days=days)).replace(
            hour=weekly.hour, minute=weekly.minute, second=0, microsecond=0)
        if weekly_target <= now:
            weekly_target += timedelta(days=7)
        return min(reminder_target, weekly_target) if self.settings.enabled else weekly_target

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                now = self.now()
                delay = self._next_delay(now)
                if self._stop.wait(delay):
                    return
                self.scan()
                self.scan_weekly()
            except Exception:
                logger.exception("scheduler_error stage=loop")
                if self._stop.wait(60):
                    return

    def _next_delay(self, now: datetime) -> float:
        """Poll safely; task eligibility is evaluated in each owner's timezone."""
        scheduled_delay = max(0.0, (self._next_run(now) - now).total_seconds())
        if self.settings.enabled and self.settings.interval_enabled:
            return min(float(self.settings.scan_interval_seconds), scheduled_delay)
        if self.settings.enabled:
            return min(float(self.settings.scan_interval_seconds), scheduled_delay)
        return scheduled_delay

    def scan(self, reminder_date: date | None = None) -> dict[str, int]:
        if not self.settings.enabled:
            return {"scanned": 0, "sent": 0, "skipped": 0}
        now = self.now()
        scan_date = reminder_date.isoformat() if reminder_date else "owner_local"
        logger.info("reminder_scan_started reminder_date=%s", scan_date)
        logger.info("follow_up_scan_started reminder_date=%s", scan_date)
        try:
            inspected = list(self.load_tasks())
            pending = [task for task in inspected if not task.completed
                       and str(task.status or "").casefold() not in {"cancelled", "canceled", "closed"}]
            for task in inspected:
                if task.completed:
                    logger.info("follow_up_skipped_completed task_id=%s", task.task_id)
            logger.info("reminder_scan_tasks_inspected count=%d", len(inspected))
            batches: dict[tuple[str, str], list[tuple[ReminderTask, str, date]]] = {}
            skipped = 0
            for task in pending:
                try:
                    local_now = now.astimezone(ZoneInfo(task.timezone))
                except Exception:
                    local_now = now.astimezone(ZoneInfo(self.settings.timezone))
                local_today = reminder_date or local_now.date()
                if reminder_date is None and not self.settings.interval_enabled:
                    local_minutes = local_now.hour * 60 + local_now.minute
                    scheduled_minutes = self.settings.hour * 60 + self.settings.minute
                    if local_minutes < scheduled_minutes:
                        continue
                delta = (task.due_date - local_today).days
                category = None
                if delta < 0 and str(task.priority or "").casefold() == "p1" and self.settings.overdue:
                    category = "high_priority_overdue"
                elif delta <= -self.settings.escalation_days and self.settings.overdue:
                    category = "escalated"
                elif delta < 0 and self.settings.overdue:
                    category = "overdue"
                elif delta == 0 and self.settings.due_today:
                    category = "today"
                elif delta == 1 and self.settings.due_tomorrow:
                    category = "tomorrow"
                elif (delta > 0 and delta * 24 <= self.settings.due_soon_hours):
                    category = "due_soon"
                elif (2 <= delta <= self.settings.high_priority_days
                      and str(task.priority or "").casefold() == "p1"):
                    category = "high_priority"
                if not category:
                    continue
                if not self.store.claim(task, category, now, self.settings.repeat_hours):
                    skipped += 1
                    logger.info("follow_up_skipped_already_sent task_id=%s reminder_type=%s",
                                task.task_id, category)
                    continue
                logger.info("follow_up_task_selected task_id=%s reminder_type=%s recipient=%s",
                            task.task_id, category,
                            "configured_channel" if task.owner_id.startswith("channel:") else "task_owner")
                batch_category = category if category in {
                    "overdue", "high_priority_overdue", "escalated", "high_priority", "due_soon"
                } else "deadline"
                batches.setdefault((task.owner_id, batch_category), []).append(
                    (task, category, local_today))

            sent = 0
            eligible = sum(len(entries) for entries in batches.values())
            logger.info("reminder_scan_eligible_tasks count=%d", eligible)
            escalations = 0
            for (recipient_id, category), entries in batches.items():
                tasks = [task for task, _, _ in entries]
                display_today = entries[0][2]
                message_category = entries[0][1] if len(entries) == 1 else category
                try:
                    self.send(recipient_id, format_reminder(tasks, message_category, display_today))
                except Exception:
                    for task, condition, _ in entries:
                        self.store.release(task, condition)
                    logger.exception("follow_up_failed stage=notification category=%s", category)
                    continue
                for task, condition, _ in entries:
                    self.store.delivered(task, condition, now)
                    sent += 1
                    escalations += int(condition == "escalated")
                    logger.info("follow_up_sent task_id=%s reminder_type=%s recipient=%s",
                                task.task_id, condition,
                                "configured_channel" if recipient_id.startswith("channel:") else "task_owner")
            logger.info("reminders_sent count=%d", sent)
            logger.info("reminders_skipped_already_sent count=%d", skipped)
            logger.info("reminder_escalations count=%d", escalations)
            return {"scanned": len(pending), "sent": sent, "skipped": skipped}
        except Exception:
            logger.exception("scheduler_error stage=scan reminder_date=%s", scan_date)
            raise

    def scan_weekly(self, now: datetime | None = None) -> dict[str, int]:
        """Send the configured weekly report once for its Monday-Sunday period."""
        now = now or self.now()
        settings = self.weekly_settings
        if not settings.enabled:
            return {"sent": 0, "skipped": 0}
        if not settings.channel or not settings.actor_id or not self.build_weekly or not self.send_weekly:
            logger.error("scheduler_error stage=weekly_config reason=missing_channel_actor_or_callback")
            return {"sent": 0, "skipped": 0}
        week_start = now.date() - timedelta(days=now.weekday())
        scheduled = datetime.combine(
            week_start + timedelta(days=settings.day),
            datetime.min.time(), now.tzinfo).replace(
                hour=settings.hour, minute=settings.minute)
        if now < scheduled:
            return {"sent": 0, "skipped": 0}
        period_start = week_start
        period_end = period_start + timedelta(days=6)
        try:
            delivery = self.build_weekly(period_start, period_end)
            if delivery is None:
                return {"sent": 0, "skipped": 0}
            if not self.store.claim_weekly(delivery, now):
                logger.info("weekly_summary_skipped_duplicate period_start=%s channel_id=%s",
                            period_start.isoformat(), delivery.channel_id)
                return {"sent": 0, "skipped": 1}
            try:
                self.send_weekly(delivery.channel_id, delivery.message)
            except Exception:
                self.store.release_weekly(delivery)
                logger.exception("scheduler_error stage=weekly_notification channel_id=%s",
                                 delivery.channel_id)
                return {"sent": 0, "skipped": 0}
            self.store.weekly_delivered(delivery, now)
            logger.info("weekly_summary_sent period_start=%s period_end=%s channel_id=%s",
                        period_start.isoformat(), period_end.isoformat(), delivery.channel_id)
            return {"sent": 1, "skipped": 0}
        except Exception:
            logger.exception("scheduler_error stage=weekly_summary")
            return {"sent": 0, "skipped": 0}
