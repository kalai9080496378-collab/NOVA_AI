"""
NOVA Analytics & Dashboard Visualizations
Generates honest Matplotlib charts using real user SQLite data.
Never synthesizes artificial data.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Styling constants matching the NOVA UI Design Direction
THEME_BG = "#0f172a"      # Dark navy
CARD_BG = "#1e293b"       # Slightly lighter navy
TEXT_COLOR = "#f8fafc"    # Bright slate
CYAN_ACCENT = "#06b6d4"   # Cyan
VIOLET_ACCENT = "#8b5cf6" # Violet
AMBER_ACCENT = "#f59e0b"  # Amber
EMERALD_ACCENT = "#10b981"# Emerald
ROSE_ACCENT = "#f43f5e"   # Rose


def create_task_status_chart(tasks: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    """
    Renders a donut chart of Completed vs Pending tasks.
    Returns None if there are no tasks in the database.
    """
    if not tasks:
        return None

    pending_count = sum(1 for t in tasks if t["status"] == "pending")
    completed_count = sum(1 for t in tasks if t["status"] == "completed")

    if pending_count == 0 and completed_count == 0:
        return None

    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=THEME_BG)
    ax.set_facecolor(THEME_BG)

    labels = []
    sizes = []
    colors = []

    if pending_count > 0:
        labels.append(f"Pending ({pending_count})")
        sizes.append(pending_count)
        colors.append(CYAN_ACCENT)

    if completed_count > 0:
        labels.append(f"Completed ({completed_count})")
        sizes.append(completed_count)
        colors.append(EMERALD_ACCENT)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops=dict(color=TEXT_COLOR, fontsize=10),
        wedgeprops=dict(width=0.4, edgecolor=THEME_BG, linewidth=2),
    )

    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_weight("bold")

    ax.set_title("Task Status Distribution", color=TEXT_COLOR, fontsize=12, pad=12, weight="bold")
    plt.tight_layout()
    return fig


def create_priority_distribution_chart(tasks: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    """
    Renders a bar chart showing task distribution by priority level (High, Medium, Low).
    """
    if not tasks:
        return None

    priorities = ["High", "Medium", "Low"]
    counts = [sum(1 for t in tasks if t["priority"] == p) for p in priorities]

    if sum(counts) == 0:
        return None

    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=THEME_BG)
    ax.set_facecolor(THEME_BG)

    bar_colors = [ROSE_ACCENT, AMBER_ACCENT, CYAN_ACCENT]
    bars = ax.bar(priorities, counts, color=bar_colors, width=0.5, edgecolor=CARD_BG, linewidth=1)

    ax.set_title("Tasks by Priority Level", color=TEXT_COLOR, fontsize=12, pad=12, weight="bold")
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Add numeric labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=10,
            weight="bold",
        )

    plt.tight_layout()
    return fig


def create_daily_activity_chart(daily_summary: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    """
    Renders an activity trend over the past days.
    """
    if not daily_summary:
        return None

    # Group by day
    days = sorted(list(set(row["day"] for row in daily_summary)))
    if len(days) < 2:
        # Insufficient data for a meaningful trend
        return None

    day_counts = {}
    for row in daily_summary:
        d = row["day"]
        day_counts[d] = day_counts.get(d, 0) + row["count"]

    x_vals = [d[-5:] for d in days]  # MM-DD
    y_vals = [day_counts[d] for d in days]

    fig, ax = plt.subplots(figsize=(6, 3), facecolor=THEME_BG)
    ax.set_facecolor(THEME_BG)

    ax.plot(x_vals, y_vals, marker="o", color=VIOLET_ACCENT, linewidth=2.5, markersize=6)
    ax.fill_between(x_vals, y_vals, color=VIOLET_ACCENT, alpha=0.15)

    ax.set_title("Daily Activity Events", color=TEXT_COLOR, fontsize=12, pad=12, weight="bold")
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    return fig


def create_completion_trend_chart(tasks: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    """
    Renders task completion trend over time using real completed_at timestamps.
    Returns None if no completed tasks exist.
    """
    completed_tasks = [t for t in tasks if t.get("status") == "completed"]
    if not completed_tasks:
        return None

    day_counts: Dict[str, int] = {}
    for t in completed_tasks:
        ts = t.get("completed_at") or t.get("created_at")
        if ts:
            day_str = ts[:10]
            day_counts[day_str] = day_counts.get(day_str, 0) + 1

    if not day_counts:
        return None

    sorted_days = sorted(day_counts.keys())
    # Display last 14 unique completion dates
    recent_days = sorted_days[-14:]
    x_labels = [d[-5:] for d in recent_days]  # MM-DD
    y_values = [day_counts[d] for d in recent_days]

    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=THEME_BG)
    ax.set_facecolor(THEME_BG)

    ax.plot(x_labels, y_values, marker="s", color=EMERALD_ACCENT, linewidth=2.5, markersize=6)
    ax.fill_between(x_labels, y_values, color=EMERALD_ACCENT, alpha=0.15)

    ax.set_title("Task Completions Over Time", color=TEXT_COLOR, fontsize=12, pad=12, weight="bold")
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    return fig


def create_category_distribution_chart(category_counts: Dict[str, int]) -> Optional[plt.Figure]:
    """
    Renders a bar chart of task distribution by category.
    """
    if not category_counts or sum(category_counts.values()) == 0:
        return None

    cats = list(category_counts.keys())
    counts = [category_counts[c] for c in cats]

    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=THEME_BG)
    ax.set_facecolor(THEME_BG)

    palette = [CYAN_ACCENT, VIOLET_ACCENT, AMBER_ACCENT, EMERALD_ACCENT, ROSE_ACCENT]
    colors = [palette[i % len(palette)] for i in range(len(cats))]

    bars = ax.barh(cats, counts, color=colors, height=0.55, edgecolor=CARD_BG)

    ax.set_title("Tasks by Category", color=TEXT_COLOR, fontsize=12, pad=12, weight="bold")
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for spine in ax.spines.values():
        spine.set_color("#334155")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{int(w)}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=TEXT_COLOR,
            fontsize=10,
            weight="bold",
        )

    plt.tight_layout()
    return fig
