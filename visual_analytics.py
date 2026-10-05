"""Presentation-only visual analytics over authorized normalized tasks."""
from __future__ import annotations

from collections import Counter
import calendar as month_calendar
from dataclasses import dataclass
from datetime import date, timedelta
from html import escape
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


MODES = {"text", "chart", "dashboard", "table"}
KINDS = {"workload", "priority", "completion", "deadlines", "completed_trend",
         "created_trend", "all_tasks", "overdue_tasks", "upcoming_tasks", "dashboard"}


@dataclass(frozen=True)
class VisualDataset:
    kind: str
    title: str
    scope: str
    series: tuple[tuple[str, int], ...]
    record_count: int
    summary: str

    @property
    def meaningful(self):
        return bool(self.series) and sum(value for _, value in self.series) > 0


@dataclass(frozen=True)
class TableDataset:
    title: str
    scope: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    summary: str

    @property
    def meaningful(self):
        return bool(self.rows)


def choose_response_mode(*, explicit_visual=False, dashboard=False, task_level=False,
                         value_count=0):
    """Choose presentation mode without consulting an LLM."""
    if task_level:
        return "table"
    if dashboard:
        return "dashboard" if value_count else "text"
    if explicit_visual and value_count:
        return "chart"
    return "chart" if value_count >= 2 else "text"


def build_dataset(tasks, kind, *, today: date, name_for_user) -> VisualDataset:
    """Transform normalized tasks into factual chart data without business I/O."""
    tasks = list(tasks)
    pending = [task for task in tasks if not task.completed]
    if kind == "priority":
        counts = Counter(task.priority or "Unspecified" for task in pending)
        rows = tuple((key, counts[key]) for key in ("P1", "P2", "P3", "P4", "Unspecified")
                     if counts[key])
        summary = (f"{len(pending)} pending task{'s' if len(pending) != 1 else ''} are shown by priority."
                   if pending else "No pending priority data is available to visualize.")
        return VisualDataset(kind, "Pending Task Priority Distribution", "pending tasks", rows,
                             len(pending), summary)
    if kind == "completion":
        completed = len(tasks) - len(pending)
        rows = (("Pending", len(pending)), ("Completed", completed)) if tasks else ()
        rate = round(completed * 100 / len(tasks), 1) if tasks else 0
        summary = (f"{completed} of {len(tasks)} tasks are completed ({rate:g}%)." if tasks
                   else "No task status data is available to visualize.")
        return VisualDataset(kind, "Current Task Completion", "all authorized tasks", rows,
                             len(tasks), summary)
    if kind == "deadlines":
        counts = Counter()
        for task in pending:
            due = task.due_date
            if not due:
                counts["No due date"] += 1
            elif due < today:
                counts["Overdue"] += 1
            elif due == today:
                counts["Due today"] += 1
            elif due == today + timedelta(days=1):
                counts["Next 24h"] += 1
            elif due == today + timedelta(days=2):
                counts["Next 48h"] += 1
            elif due <= today + timedelta(days=6):
                counts["This week"] += 1
            else:
                counts["Later"] += 1
        order = ("Overdue", "Due today", "Next 24h", "Next 48h", "This week", "Later", "No due date")
        rows = tuple((key, counts[key]) for key in order if counts[key])
        summary = (f"{len(pending)} pending task{'s' if len(pending) != 1 else ''} are shown by deadline window."
                   if pending else "No pending deadline data is available to visualize.")
        return VisualDataset(kind, "Pending Task Deadline Distribution", "pending tasks", rows,
                             len(pending), summary)
    raise ValueError(f"Unsupported visual analytics kind: {kind}")


def workload_dataset(report) -> VisualDataset:
    """Adapt the existing workload intelligence report without recalculating it."""
    rows = tuple(sorted(
        ((row["name"], row["pending"]) for row in report.rows.values() if row["pending"]),
        key=lambda value: (-value[1], value[0].casefold())))
    total = sum(value for _, value in rows)
    leader = rows[0] if rows else None
    summary = (f"{leader[0]} currently has the largest pending workload with "
               f"{leader[1]} task{'s' if leader[1] != 1 else ''}." if leader
               else "No pending workload is available to visualize.")
    return VisualDataset("workload", "Pending Workload by Owner", "pending tasks",
                         rows, total, summary)


def render_deadline_heatmap_png(points, *, today) -> bytes:
    """Render a restrained visual heatmap from precomputed deadline facts."""
    values = list(points)
    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="#F4F7FB")
    ax.set_facecolor("#F4F7FB")
    if values:
        labels = [day.strftime("%d %b") for day, _, _ in values]
        totals = [count for _, count, _ in values]
        p1 = [count for _, _, count in values]
        colors = ["#A63D40" if high else "#365F91" for high in p1]
        bars = ax.bar(labels, totals, color=colors, width=.68)
        for bar, total, high in zip(bars, totals, p1):
            ax.text(bar.get_x() + bar.get_width() / 2, total + .08,
                    f"{total} task{'s' if total != 1 else ''} · {high} P1",
                    ha="center", va="bottom", fontsize=9, color="#24344D")
        ax.set_ylim(0, max(totals) + 1.4)
    else:
        ax.text(.5, .5, "No upcoming authorized deadlines", ha="center", va="center",
                transform=ax.transAxes, fontsize=16, color="#52637A")
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"DEADLINE HEATMAP  |  FROM {today.strftime('%d %b %Y').upper()}",
                 loc="left", fontsize=18, fontweight="bold", color="#17263C", pad=20)
    ax.set_ylabel("Deadline count", color="#52637A")
    ax.grid(axis="y", color="#DCE3EC", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(pad=2)
    return _finish_png(fig)


def _panel(title, rows, x, y, width, *, color="#4C78A8"):
    height = max(180, 90 + len(rows) * 48)
    maximum = max((value for _, value in rows), default=1)
    parts = [f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="#FFFFFF"/>',
             f'<text x="{x + 28}" y="{y + 42}" class="panel">{escape(title)}</text>']
    label_width = min(260, int(width * .38))
    bar_width = max(120, width - label_width - 105)
    for index, (label, value) in enumerate(rows):
        row_y = y + 82 + index * 48
        scaled = 0 if maximum <= 0 else max(3, int(bar_width * value / maximum))
        clipped = label if len(label) <= 26 else label[:25] + "…"
        parts.extend((
            f'<text x="{x + 28}" y="{row_y + 17}" class="label">{escape(clipped)}</text>',
            f'<rect x="{x + label_width}" y="{row_y}" width="{bar_width}" height="22" rx="6" fill="#E8EDF5"/>',
            f'<rect x="{x + label_width}" y="{row_y}" width="{scaled}" height="22" rx="6" fill="{color}"/>',
            f'<text x="{x + label_width + bar_width + 16}" y="{row_y + 17}" class="value">{value}</text>',
        ))
    return "".join(parts), height


def render_svg(dataset: VisualDataset) -> str:
    """Render an exact, accessible vector bar chart with no temporary file."""
    width = 1200
    panel, panel_height = _panel(dataset.title, dataset.series, 55, 105, 1090)
    height = panel_height + 190
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(dataset.title)}">'
            '<style>.title{font:700 34px Arial,sans-serif;fill:#172B4D}.scope{font:18px Arial,sans-serif;fill:#5E6C84}'
            '.panel{font:700 23px Arial,sans-serif;fill:#172B4D}.label{font:18px Arial,sans-serif;fill:#344563}'
            '.value{font:700 18px Arial,sans-serif;fill:#172B4D}</style>'
            '<rect width="100%" height="100%" fill="#F4F6F8"/>'
            f'<text x="55" y="52" class="title">{escape(dataset.title)}</text>'
            f'<text x="55" y="80" class="scope">Scope: {escape(dataset.scope)}</text>'
            f'{panel}</svg>')


def render_donut_svg(dataset: VisualDataset) -> str:
    """Render a small-category part-to-whole chart with exact values."""
    total = sum(value for _, value in dataset.series)
    if total <= 0:
        raise ValueError("A donut chart requires positive data.")
    colors = ("#E45756", "#4C78A8", "#72B7B2", "#F2CF5B", "#B279A2")
    radius, circumference = 145, 2 * 3.141592653589793 * 145
    offset, circles, legend = 0.0, [], []
    for index, (label, value) in enumerate(dataset.series):
        length = circumference * value / total
        color = colors[index % len(colors)]
        circles.append(
            f'<circle cx="300" cy="300" r="{radius}" fill="none" stroke="{color}" '
            f'stroke-width="70" stroke-dasharray="{length:.3f} {circumference - length:.3f}" '
            f'stroke-dashoffset="{-offset:.3f}" transform="rotate(-90 300 300)"/>')
        legend.append(f'<rect x="610" y="{185 + index * 62}" width="24" height="24" rx="5" fill="{color}"/>'
                      f'<text x="650" y="{204 + index * 62}" class="label">{escape(label)}: {value}</text>')
        offset += length
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" '
            'viewBox="0 0 1200 650" role="img">'
            '<style>.title{font:700 34px Arial,sans-serif;fill:#172B4D}.scope{font:18px Arial,sans-serif;fill:#5E6C84}'
            '.label{font:21px Arial,sans-serif;fill:#344563}.total{font:700 42px Arial,sans-serif;fill:#172B4D}'
            '.small{font:18px Arial,sans-serif;fill:#5E6C84}</style>'
            '<rect width="100%" height="100%" fill="#F4F6F8"/>'
            f'<text x="55" y="58" class="title">{escape(dataset.title)}</text>'
            f'<text x="55" y="88" class="scope">Scope: {escape(dataset.scope)}</text>'
            '<rect x="55" y="120" width="1090" height="475" rx="18" fill="#FFFFFF"/>'
            + ''.join(circles) + f'<text x="300" y="295" text-anchor="middle" class="total">{total}</text>'
            '<text x="300" y="328" text-anchor="middle" class="small">tasks</text>'
            + ''.join(legend) + '</svg>')


def render_line_svg(dataset: VisualDataset) -> str:
    """Render a line only from actual ordered time-series observations."""
    rows = list(dataset.series)
    if len(rows) < 2:
        raise ValueError("A line chart requires at least two observed periods.")
    width, height = 1200, 650
    x0, y0, plot_width, plot_height = 120, 520, 980, 350
    maximum = max(value for _, value in rows) or 1
    points = []
    labels = []
    for index, (label, value) in enumerate(rows):
        x = x0 + (plot_width * index / (len(rows) - 1))
        y = y0 - plot_height * value / maximum
        points.append(f"{x:.1f},{y:.1f}")
        labels.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#4C78A8"/>'
                      f'<text x="{x:.1f}" y="{y - 15:.1f}" text-anchor="middle" class="value">{value}</text>'
                      f'<text x="{x:.1f}" y="{y0 + 34}" text-anchor="middle" class="axis">{escape(label)}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
            '<style>.title{font:700 34px Arial,sans-serif;fill:#172B4D}.scope{font:18px Arial,sans-serif;fill:#5E6C84}'
            '.axis{font:15px Arial,sans-serif;fill:#5E6C84}.value{font:700 17px Arial,sans-serif;fill:#172B4D}</style>'
            '<rect width="100%" height="100%" fill="#F4F6F8"/>'
            f'<text x="55" y="58" class="title">{escape(dataset.title)}</text>'
            f'<text x="55" y="88" class="scope">Scope: {escape(dataset.scope)}</text>'
            '<rect x="55" y="120" width="1090" height="470" rx="18" fill="#FFFFFF"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_width}" y2="{y0}" stroke="#B3BAC5" stroke-width="2"/>'
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#4C78A8" stroke-width="7" '
            'stroke-linejoin="round" stroke-linecap="round"/>' + ''.join(labels) + '</svg>')


def render_chart(dataset: VisualDataset) -> str:
    if dataset.kind in {"priority", "completion"}:
        return render_donut_svg(dataset)
    if dataset.kind in {"completed_trend", "created_trend"}:
        return render_line_svg(dataset)
    return render_svg(dataset)


COLORS = ("#4C78A8", "#E45756", "#72B7B2", "#F2CF5B", "#B279A2", "#FF9DA6")


def _finish_png(fig):
    output = BytesIO()
    fig.savefig(output, format="png", dpi=180, bbox_inches="tight",
                facecolor="#F7F9FC")
    plt.close(fig)
    return output.getvalue()


def _style_axis(ax, title, scope):
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color="#172B4D", pad=22)
    ax.text(0, 1.02, f"Scope: {scope}", transform=ax.transAxes, fontsize=10,
            color="#5E6C84", va="bottom")
    ax.set_facecolor("#FFFFFF")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#C1C7D0")
    ax.spines["bottom"].set_color("#C1C7D0")
    ax.tick_params(colors="#344563", labelsize=10)


_CALENDAR_PRIORITY = {
    "P1": ("#8C2F39", "#F9E9EB"),
    "P2": ("#9A5B13", "#FFF3E3"),
    "P3": ("#315D75", "#EAF3F7"),
    "P4": ("#486B56", "#EDF5F0"),
}


def _short(value, limit):
    value = str(value or "")
    return value if len(value) <= limit else value[:limit - 1].rstrip() + "…"


def render_team_calendar_png(tasks, clocks, *, today, name_for_user,
                             show_owner=True, show_priority=True) -> bytes:
    """Render an enterprise month grid from an already-authorized task snapshot."""
    tasks = [task for task in tasks if not task.completed and task.due_date]
    month_tasks = [task for task in tasks if
                   (task.due_date.year, task.due_date.month) == (today.year, today.month)]
    overdue = [task for task in tasks if task.due_date < today]
    due_today = [task for task in tasks if task.due_date == today]
    due_week = [task for task in tasks if today <= task.due_date <= today + timedelta(days=6)]
    grouped = {}
    for task in month_tasks:
        grouped.setdefault(task.due_date.day, []).append(task)
    collisions = sum(len(values) >= 3 for values in grouped.values())
    pressure = Counter()
    for task in tasks:
        if show_owner:
            for owner in task.owner_ids or ("Unassigned",):
                pressure[owner] += (3 if show_priority and task.priority == "P1"
                                    else 2 if show_priority and task.priority == "P2" else 1)

    fig = plt.figure(figsize=(18, 11), facecolor="#F4F6F8")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.text(.55, 10.48, "TEAM CALENDAR", fontsize=25, fontweight="bold", color="#172B4D")
    ax.text(17.45, 10.48, today.strftime("%B %Y").upper(), fontsize=17,
            fontweight="bold", color="#344563", ha="right")
    ax.text(.55, 10.08, "Authorized deadlines and team availability", fontsize=10.5,
            color="#5E6C84")

    left, bottom, width, height = .55, .62, 12.7, 9.05
    header_h = .52
    weeks = month_calendar.Calendar(firstweekday=0).monthdayscalendar(today.year, today.month)
    cell_w, cell_h = width / 7, (height - header_h) / len(weeks)
    for index, label in enumerate(("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")):
        x = left + index * cell_w
        ax.add_patch(Rectangle((x, bottom + height - header_h), cell_w, header_h,
                               facecolor="#E9EDF3", edgecolor="#D3D9E2", linewidth=1))
        ax.text(x + .12, bottom + height - .34, label, fontsize=9, fontweight="bold",
                color="#44546A")
    for row_index, week in enumerate(weeks):
        y = bottom + height - header_h - (row_index + 1) * cell_h
        for column, day in enumerate(week):
            x = left + column * cell_w
            background = "#FBFCFE" if column < 5 else "#F7F8FA"
            ax.add_patch(Rectangle((x, y), cell_w, cell_h, facecolor=background,
                                   edgecolor="#D9DEE7", linewidth=1))
            if not day:
                continue
            is_today = day == today.day
            if is_today:
                ax.add_patch(FancyBboxPatch((x + .08, y + cell_h - .39), .38, .28,
                                            boxstyle="round,pad=.02,rounding_size=.05",
                                            facecolor="#274C77", edgecolor="none"))
            ax.text(x + .13, y + cell_h - .26, str(day), fontsize=9.5,
                    fontweight="bold", color="#FFFFFF" if is_today else "#344563")
            values = sorted(grouped.get(day, []), key=lambda task: (
                {"P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(task.priority, 9), task.name.casefold()))
            card_height = min(.42, max(.31, (cell_h - .5) / 3.2))
            for card_index, task in enumerate(values[:3]):
                card_y = y + cell_h - .52 - (card_index + 1) * card_height
                foreground, background = (_CALENDAR_PRIORITY.get(task.priority, ("#44546A", "#EEF1F5"))
                                          if show_priority else ("#44546A", "#EEF1F5"))
                if task.due_date < today:
                    foreground, background = "#8C2F39", "#F7E5E7"
                ax.add_patch(FancyBboxPatch(
                    (x + .09, card_y), cell_w - .18, card_height - .045,
                    boxstyle="round,pad=.025,rounding_size=.05",
                    facecolor=background, edgecolor=foreground, linewidth=.8))
                owner = ((", ".join(name_for_user(value) for value in task.owner_ids) or "Unassigned")
                         if show_owner else "Owner restricted")
                status = "Overdue" if task.due_date < today else "Due today" if task.due_date == today else "Pending"
                ax.text(x + .16, card_y + card_height - .16, _short(task.name, 23),
                        fontsize=7.3, fontweight="bold", color="#172B4D", va="top")
                priority = task.priority or "—" if show_priority else "Priority restricted"
                ax.text(x + .16, card_y + .08, _short(f"{owner}  {priority}  {status}", 29),
                        fontsize=6.3, color=foreground, va="bottom")
            if len(values) > 3:
                ax.text(x + .13, y + .08, f"+{len(values) - 3} more", fontsize=6.5,
                        color="#5E6C84")

    panel_x, panel_w = 13.6, 3.85
    ax.add_patch(FancyBboxPatch((panel_x, 6.85), panel_w, 2.82,
                               boxstyle="round,pad=.08,rounding_size=.12",
                               facecolor="#FFFFFF", edgecolor="#D9DEE7"))
    ax.text(panel_x + .25, 9.28, "CALENDAR SUMMARY", fontsize=11, fontweight="bold", color="#172B4D")
    metrics = (("Active deadlines", len(tasks)), ("Overdue", len(overdue)),
               ("Due today", len(due_today)), ("Due this week", len(due_week)),
               ("High priority", sum(task.priority == "P1" for task in tasks) if show_priority else "Restricted"),
               ("Date collisions", collisions))
    for index, (label, value) in enumerate(metrics):
        row_y = 8.85 - index * .37
        ax.text(panel_x + .25, row_y, label, fontsize=8.2, color="#5E6C84")
        ax.text(panel_x + panel_w - .25, row_y, str(value), fontsize=9, fontweight="bold",
                color="#172B4D", ha="right")

    ax.add_patch(FancyBboxPatch((panel_x, 4.72), panel_w, 1.8,
                               boxstyle="round,pad=.08,rounding_size=.12",
                               facecolor="#FFFFFF", edgecolor="#D9DEE7"))
    ax.text(panel_x + .25, 6.14, "DEADLINE PRESSURE", fontsize=11, fontweight="bold", color="#172B4D")
    for index, (owner, score) in enumerate(pressure.most_common(3)):
        label = "Unassigned" if owner == "Unassigned" else name_for_user(owner)
        row_y = 5.75 - index * .4
        ax.text(panel_x + .25, row_y, _short(label, 20), fontsize=8.2, color="#344563")
        ax.text(panel_x + panel_w - .25, row_y, str(score), fontsize=8.5,
                fontweight="bold", color="#172B4D", ha="right")
        ax.add_patch(Rectangle((panel_x + .25, row_y - .15),
                               (panel_w - .5) * score / max(pressure.values(), default=1), .055,
                               facecolor="#6B7C93", edgecolor="none"))

    ax.add_patch(FancyBboxPatch((panel_x, .62), panel_w, 3.78,
                               boxstyle="round,pad=.08,rounding_size=.12",
                               facecolor="#FFFFFF", edgecolor="#D9DEE7"))
    ax.text(panel_x + .25, 4.02, "TEAM TIME ZONES", fontsize=11, fontweight="bold", color="#172B4D")
    for index, clock in enumerate(list(clocks)[:6]):
        row_y = 3.6 - index * .48
        ax.text(panel_x + .25, row_y, _short(clock.name, 18), fontsize=8.2,
                fontweight="bold", color="#344563")
        if clock.local_time:
            detail = f"{clock.local_time.strftime('%H:%M')}  {clock.utc_offset}"
            status = clock.availability
        else:
            detail, status = "Time zone not configured", ""
        ax.text(panel_x + panel_w - .25, row_y, detail, fontsize=7.3, color="#5E6C84", ha="right")
        if status:
            ax.text(panel_x + .25, row_y - .18, _short(status, 34), fontsize=6.5, color="#6B778C")
    ax.text(17.45, .28, "Read-only view · No task changes were made", fontsize=7.3,
            color="#6B778C", ha="right")
    return _finish_png(fig)


def render_team_clock_png(clocks, *, requester_clock=None) -> bytes:
    """Render configured team clocks and availability as a compact visual panel."""
    clocks = list(clocks)
    height = max(4.5, 2.25 + .72 * len(clocks))
    fig, ax = plt.subplots(figsize=(13, height), facecolor="#F4F6F8")
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, height)
    ax.text(.45, height - .45, "TEAM TIME ZONES", fontsize=21, fontweight="bold", color="#172B4D")
    ax.text(.45, height - .8, "Configured global working context", fontsize=10, color="#5E6C84")
    headers = ((.55, "PERSON"), (3.3, "LOCATION / ZONE"), (7.35, "LOCAL TIME"), (9.65, "UTC OFFSET"), (11.1, "WORK STATUS"))
    header_y = height - 1.35
    ax.add_patch(Rectangle((.4, header_y - .22), 12.2, .48, facecolor="#E9EDF3", edgecolor="none"))
    for x, label in headers:
        ax.text(x, header_y, label, fontsize=8, fontweight="bold", color="#44546A", va="center")
    for index, clock in enumerate(clocks):
        y = header_y - .62 - index * .66
        ax.add_patch(Rectangle((.4, y - .25), 12.2, .58,
                               facecolor="#FFFFFF" if index % 2 == 0 else "#F8F9FB",
                               edgecolor="#E1E5EB", linewidth=.6))
        zone = clock.location or clock.timezone_name or "Time zone not configured"
        local = clock.local_time.strftime("%a %d %b  %H:%M") if clock.local_time else "Not configured"
        status = clock.availability if clock.local_time else "Unavailable"
        values = ((.55, clock.name, True), (3.3, zone, False), (7.35, local, False),
                  (9.65, clock.utc_offset or "—", False), (11.1, status, False))
        for x, value, bold in values:
            ax.text(x, y, _short(value, 28 if x == 3.3 else 22), fontsize=8.2,
                    fontweight="bold" if bold else "normal", color="#172B4D" if bold else "#44546A",
                    va="center")
    ax.text(12.55, .2, "Time-zone data is never inferred", fontsize=7.5,
            color="#6B778C", ha="right")
    return _finish_png(fig)


def render_chart_png(dataset: VisualDataset, chart_type="auto") -> bytes:
    """Render a professional PNG from exact structured values."""
    if not dataset.meaningful:
        raise ValueError("No meaningful chart data is available.")
    chart_type = chart_type if chart_type != "auto" else (
        "pie" if dataset.kind in {"priority", "completion"}
        else "line" if dataset.kind in {"completed_trend", "created_trend"}
        else "bar")
    labels = [label for label, _ in dataset.series]
    values = [value for _, value in dataset.series]
    if chart_type == "line" and len(values) < 2:
        raise ValueError("A line chart requires at least two real observations.")
    if chart_type == "pie":
        fig, ax = plt.subplots(figsize=(8.5, 6.2), layout="constrained")
        wedges, _, autotexts = ax.pie(
            values, startangle=90, colors=COLORS[:len(values)],
            autopct=lambda percent: f"{percent:.1f}%" if percent >= 3 else "",
            wedgeprops={"width": .55, "edgecolor": "white", "linewidth": 2},
            textprops={"color": "#172B4D", "fontsize": 10})
        ax.legend(wedges, [f"{label} · {value}" for label, value in dataset.series],
                  loc="center left", bbox_to_anchor=(.88, .5), frameon=False, fontsize=10)
        ax.set_title(dataset.title, loc="left", fontsize=18, fontweight="bold",
                     color="#172B4D", pad=20)
        ax.text(0, 1.01, f"Scope: {dataset.scope}", transform=ax.transAxes,
                fontsize=10, color="#5E6C84")
        for value in autotexts:
            value.set_fontweight("bold")
        return _finish_png(fig)
    if chart_type == "line":
        fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
        x = list(range(len(values)))
        ax.plot(x, values, color=COLORS[0], linewidth=3, marker="o", markersize=7)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel("Task count", color="#344563")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#DFE1E6", linewidth=.8)
        for index, value in enumerate(values):
            ax.annotate(str(value), (index, value), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontweight="bold")
        _style_axis(ax, dataset.title, dataset.scope)
        return _finish_png(fig)
    if chart_type == "table":
        table = TableDataset(dataset.title, dataset.scope, ("Category", "Count"),
                             tuple((label, str(value)) for label, value in dataset.series),
                             dataset.summary)
        return render_table_png(table)
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    bars = ax.bar(labels, values, color=COLORS[:len(values)], width=.65)
    ax.set_xlabel("Owner" if dataset.kind == "workload" else "Category", color="#344563")
    ax.set_ylabel("Pending task count" if dataset.kind == "workload" else "Task count",
                  color="#344563")
    ax.set_ylim(bottom=0, top=max(values) * 1.22 if max(values) else 1)
    ax.grid(axis="y", color="#DFE1E6", linewidth=.8)
    ax.bar_label(bars, labels=[str(value) for value in values], padding=4,
                 fontsize=10, fontweight="bold", color="#172B4D")
    ax.tick_params(axis="x", rotation=20)
    _style_axis(ax, dataset.title, dataset.scope)
    return _finish_png(fig)


def render_table_png(dataset: TableDataset) -> bytes:
    if not dataset.meaningful:
        raise ValueError("No table rows are available.")
    height = max(3.8, 1.8 + .48 * len(dataset.rows))
    fig, ax = plt.subplots(figsize=(12, height), layout="constrained")
    ax.axis("off")
    ax.set_title(dataset.title, loc="left", fontsize=18, fontweight="bold",
                 color="#172B4D", pad=20)
    ax.text(0, .98, f"Scope: {dataset.scope}", transform=ax.transAxes,
            fontsize=10, color="#5E6C84", va="top")
    table = ax.table(cellText=dataset.rows, colLabels=dataset.columns,
                     loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DFE1E6")
        cell.set_facecolor("#E9F2FF" if row == 0 else "#FFFFFF")
        if row == 0:
            cell.set_text_props(weight="bold", color="#172B4D")
    return _finish_png(fig)


def render_dashboard_png(datasets) -> bytes:
    datasets = [dataset for dataset in datasets if dataset.meaningful][:4]
    if not datasets:
        raise ValueError("No meaningful dashboard data is available.")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle("Smart Task Visual Dashboard", fontsize=22, fontweight="bold",
                 color="#172B4D")
    for ax, dataset in zip(axes.flat, datasets):
        labels = [label for label, _ in dataset.series]
        values = [value for _, value in dataset.series]
        bars = ax.bar(labels, values, color=COLORS[:len(values)], width=.62)
        ax.set_ylim(bottom=0, top=max(values) * 1.25 if max(values) else 1)
        ax.grid(axis="y", color="#DFE1E6", linewidth=.7)
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.tick_params(axis="x", rotation=22, labelsize=8)
        _style_axis(ax, dataset.title, dataset.scope)
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    return _finish_png(fig)


def render_dashboard(datasets) -> str:
    """Render a compact two-column dashboard from independent exact datasets."""
    datasets = [dataset for dataset in datasets if dataset.meaningful][:4]
    if not datasets:
        raise ValueError("No meaningful dashboard data is available.")
    width, column_width = 1400, 630
    panels, bottoms = [], [115, 115]
    colors = ("#4C78A8", "#E45756", "#72B7B2", "#F2CF5B")
    for index, dataset in enumerate(datasets):
        column = index % 2
        x, y = 55 + column * 685, bottoms[column]
        panel, height = _panel(dataset.title, dataset.series, x, y, column_width,
                               color=colors[index])
        panels.append(panel)
        bottoms[column] += height + 30
    height = max(bottoms) + 30
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="Task analytics dashboard">'
            '<style>.title{font:700 34px Arial,sans-serif;fill:#172B4D}.panel{font:700 21px Arial,sans-serif;fill:#172B4D}'
            '.label{font:17px Arial,sans-serif;fill:#344563}.value{font:700 17px Arial,sans-serif;fill:#172B4D}</style>'
            '<rect width="100%" height="100%" fill="#F4F6F8"/>'
            '<text x="55" y="58" class="title">Smart Task Visual Dashboard</text>'
            + "".join(panels) + '</svg>')
