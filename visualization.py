"""Presentation-only renderers for structured project analytics.

This module contains no Slack retrieval, filtering, RBAC, or calculations.
Its text renderer can later sit beside Block Kit or web renderers without
changing the analytics engine.
"""
from typing import Callable, Iterable
import slack_presentation


def bar(value, maximum, width=8):
    return slack_presentation.bar(value, maximum, width)


def distribution(title, values):
    return slack_presentation.distribution(title, values)


def series(title, value):
    return slack_presentation.series(title, value)


def render_progress(report, format_items: Callable[[Iterable, str], str]):
    """Render a ProgressReport-like value as Slack mrkdwn."""
    requested = set(report.requested)
    if "summary" in requested:
        requested.update({"overview", "workload", "at_risk", "upcoming", "completed_over_time"})
    sections = []
    snapshot = report.snapshot
    if requested & {"overview", "completion"}:
        rate = snapshot.get("completion_rate")
        progress = "Unavailable" if rate is None else f"{bar(rate, 100)} {rate:g}%"
        sections.append(
            "*Progress* · "
            f"{snapshot['total']} total · {snapshot['completed']} completed · "
            f"{snapshot['pending']} pending · {snapshot['overdue']} overdue\n"
            f"• Completion {progress} · Due today {snapshot['due_today']} · "
            f"Due this week {snapshot['due_this_week']}")
    if "workload" in requested:
        pending = {name: values["pending"] for name, values in report.workload.items()}
        sections.append(distribution("Pending workload by assignee", pending))
    if "status_distribution" in requested:
        sections.append(distribution("Status distribution", report.status_distribution))
    if "priority_distribution" in requested:
        sections.append(distribution("Priority distribution", report.priority_distribution))
    if "completed_over_time" in requested:
        sections.append(series("Completed tasks over time", report.completed_series))
    if "created_over_time" in requested:
        sections.append(series("Created tasks over time", report.created_series))
    if "comparison" in requested:
        if report.comparison.get("available"):
            sections.append(distribution("Completed-task comparison", {
                "This period": report.comparison["current"],
                "Previous period": report.comparison["previous"],
            }))
        else:
            sections.append("*Completed-task comparison*\nNo reliable timestamped records are available.")

    item_metrics = (
        ("overdue", "Overdue tasks", report.overdue_items),
        ("due_today", "Tasks due today", report.due_today_items),
        ("due_this_week", "Tasks due this week", report.due_this_week_items),
        ("upcoming", "Upcoming deadlines", report.upcoming_items[:10]),
        ("at_risk", "At-risk tasks", report.at_risk_items[:10]),
    )
    for metric, title, items in item_metrics:
        if metric not in requested:
            continue
        sections.append(format_items(items, title))
        report.displayed_items.extend(items)
    if report.limitations:
        sections.append("*Data limitations*\n" + "\n".join(
            f"• {message}" for message in dict.fromkeys(report.limitations)))
    return "\n".join(sections) if sections else "No progress metric was requested."
