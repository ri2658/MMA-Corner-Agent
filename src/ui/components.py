"""Reusable HTML component builders for the Streamlit dashboard."""

from __future__ import annotations
from typing import Optional


def header_banner() -> str:
    """Render the main page header."""
    return """
    <div class="header-banner animate-in">
        <div class="header-title" style="
            background: linear-gradient(135deg, #fff 0%, #e53e3e 60%, #805ad5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">MMA Corner Agent</div>
        <div class="header-subtitle">
            AI-powered corner coach &mdash; real-time counter pattern detection &amp; tactical adjustments
        </div>
    </div>
    """


def stat_pills(stats: list[tuple[str, str]]) -> str:
    """Render a row of stat pills.

    Args:
        stats: List of (value, label) tuples.
    """
    pills = ""
    for i, (value, label) in enumerate(stats):
        delay_class = f"animate-in-delay-{min(i + 1, 3)}"
        pills += f"""
        <div class="stat-pill animate-in {delay_class}">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """
    return f'<div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.5rem;">{pills}</div>'


def advice_card(
    severity: str,
    headline: str,
    stats_text: str,
    adjustment: str,
    icon: str = "",
    index: int = 0,
) -> str:
    """Render a single advice card with severity styling."""
    sev_class = f"advice-{severity}"
    badge_class = f"badge-{severity}"
    pulse = "pulse-critical" if severity == "critical" else ""
    delay_class = f"animate-in-delay-{min(index + 1, 3)}"

    return f"""
    <div class="advice-card {sev_class} {pulse} animate-in {delay_class}">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span class="severity-badge {badge_class}">{icon} {severity}</span>
            <span style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">{stats_text}</span>
        </div>
        <div style="font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 6px;">
            {headline}
        </div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; padding-left: 4px; border-left: 2px solid rgba(255,255,255,0.06); margin-left: 2px;">
            {adjustment}
        </div>
    </div>
    """


def safe_actions_chips(action_names: list[str]) -> str:
    """Render safe actions as green chips."""
    if not action_names:
        return '<div style="color: var(--text-muted); font-size: 0.85rem;">No safe actions identified yet.</div>'

    chips = ""
    for name in action_names:
        chips += f'<span class="safe-chip">✓ {name}</span>'

    return f'<div style="display: flex; flex-wrap: wrap; gap: 4px;">{chips}</div>'


def action_bar_chart(
    actions: dict[str, int],
    max_count: int,
    bar_class: str = "action-bar-a",
    label_resolver=None,
) -> str:
    """Render a horizontal bar chart of action counts.

    Args:
        actions: {action_id: count} dictionary.
        max_count: Maximum count (for scaling bars).
        bar_class: CSS class for bar color.
        label_resolver: Callable(action_id) -> display_name.
    """
    if not actions:
        return '<div style="color: var(--text-muted); font-size: 0.85rem;">No actions recorded.</div>'

    sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:8]
    html = ""

    for action_id, count in sorted_actions:
        name = label_resolver(action_id) if label_resolver else action_id
        width_pct = (count / max(max_count, 1)) * 100

        html += f"""
        <div class="action-bar-container">
            <span class="action-bar-label">{name}</span>
            <div class="action-bar {bar_class}" style="width: {width_pct}%;"></div>
            <span class="action-bar-count">{count}</span>
        </div>
        """

    return html


def threat_pattern_rows(
    patterns: list[dict],
    max_threat: float,
    label_resolver=None,
) -> str:
    """Render ranked threat pattern rows.

    Args:
        patterns: List of pattern dicts from PatternAggregator.
        max_threat: Maximum threat score (for scaling).
        label_resolver: Callable(action_id) -> display_name.
    """
    if not patterns:
        return '<div style="color: var(--text-muted); font-size: 0.85rem;">No recurring patterns detected yet.</div>'

    html = ""
    for i, p in enumerate(patterns):
        your_name = label_resolver(p["your_action"]) if label_resolver else p["your_action"]
        their_name = label_resolver(p["their_counter"]) if label_resolver else p["their_counter"]
        threat = p.get("threat_score", 0)
        occ = p.get("occurrences", 0)
        landed = p.get("landed_count", 0)
        bar_width = (threat / max(max_threat, 1)) * 100

        rounds_str = ", ".join(f"R{r}" for r in p.get("rounds_seen", []))

        html += f"""
        <div class="threat-row">
            <span class="threat-rank">{i + 1}</span>
            <div style="flex: 1;">
                <div style="font-size: 0.88rem; font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
                    {your_name} <span class="threat-arrow">→</span> {their_name}
                </div>
                <div class="threat-score-bar" style="width: {bar_width}%;"></div>
                <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 3px; font-family: var(--font-mono);">
                    {occ}x &middot; {landed} landed &middot; threat {threat:.0f} &middot; {rounds_str}
                </div>
            </div>
        </div>
        """

    return html


def section_header(title: str, subtitle: str = "", icon: str = "") -> str:
    """Render a styled section header."""
    sub = f'<div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 2px;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="margin-bottom: 1rem; margin-top: 0.5rem;">
        <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
            {icon} {title}
        </div>
        {sub}
    </div>
    """


def glass_card_wrapper(content: str, extra_class: str = "") -> str:
    """Wrap content in a glass card."""
    return f'<div class="glass-card {extra_class}">{content}</div>'
