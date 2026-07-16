"""Reusable HTML component builders for the Streamlit dashboard.

All components use INLINE styles instead of CSS class references
so they render correctly inside Streamlit's sandboxed HTML containers.
"""

from __future__ import annotations


# ── Color constants ────────────────────────────────────────────────────
_BG_CARD = "rgba(22, 22, 35, 0.85)"
_BG_CARD_HOVER = "rgba(30, 30, 50, 0.95)"
_BORDER = "rgba(255, 255, 255, 0.06)"

_TEXT = "#e8e8f0"
_TEXT_SEC = "#8888a0"
_TEXT_MUTED = "#555570"

_RED = "#e53e3e"
_ORANGE = "#ed8936"
_GREEN = "#38a169"
_BLUE = "#4299e1"

_SEVERITY_COLORS = {
    "critical": {"border": "rgba(229,62,62,0.25)", "bar": _RED, "badge_bg": "rgba(229,62,62,0.15)", "badge_text": "#fc8181", "badge_border": "rgba(229,62,62,0.3)"},
    "warning":  {"border": "rgba(237,137,54,0.25)", "bar": _ORANGE, "badge_bg": "rgba(237,137,54,0.15)", "badge_text": "#fbd38d", "badge_border": "rgba(237,137,54,0.3)"},
    "working":  {"border": "rgba(56,161,105,0.25)", "bar": _GREEN, "badge_bg": "rgba(56,161,105,0.15)", "badge_text": "#68d391", "badge_border": "rgba(56,161,105,0.3)"},
}


def header_banner() -> str:
    """Render the main page header."""
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(229,62,62,0.08) 0%, rgba(128,90,213,0.08) 100%);
        border: 1px solid rgba(229,62,62,0.15);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    ">
        <div style="
            font-size: 2.5rem; font-weight: 900; letter-spacing: -0.04em;
            line-height: 1.1; font-family: Inter, sans-serif;
            background: linear-gradient(135deg, #fff 0%, #e53e3e 60%, #805ad5 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        ">MMA Corner Agent</div>
        <div style="font-size: 0.95rem; color: {_TEXT_SEC}; margin-top: 0.5rem; font-family: Inter, sans-serif;">
            AI-powered corner coach &mdash; real-time counter pattern detection &amp; tactical adjustments
        </div>
    </div>
    """


def stat_pills(stats: list[tuple[str, str]]) -> str:
    """Render a row of stat pills."""
    pills = ""
    for value, label in stats:
        pills += f"""
        <div style="
            display: inline-flex; flex-direction: column; align-items: center;
            background: {_BG_CARD}; border: 1px solid {_BORDER};
            border-radius: 12px; padding: 1rem 1.5rem; min-width: 110px; text-align: center;
        ">
            <div style="
                font-size: 2rem; font-weight: 800; line-height: 1;
                background: linear-gradient(135deg, #fff 30%, {_BLUE} 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; font-family: Inter, sans-serif;
            ">{value}</div>
            <div style="
                font-size: 0.7rem; font-weight: 600; color: {_TEXT_SEC};
                text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.35rem;
                font-family: Inter, sans-serif;
            ">{label}</div>
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
    sc = _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS["working"])

    return f"""
    <div style="
        background: {_BG_CARD}; border-radius: 16px;
        padding: 1.25rem 1.25rem 1.25rem 1.5rem; margin-bottom: 0.75rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        border: 1px solid {sc['border']}; border-left: 4px solid {sc['bar']};
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="
                display: inline-flex; align-items: center; gap: 6px;
                padding: 3px 10px; border-radius: 20px; font-size: 0.7rem;
                font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
                background: {sc['badge_bg']}; color: {sc['badge_text']};
                border: 1px solid {sc['badge_border']}; font-family: Inter, sans-serif;
            ">{icon} {severity}</span>
            <span style="font-size: 0.75rem; color: {_TEXT_MUTED}; font-family: 'JetBrains Mono', monospace;">{stats_text}</span>
        </div>
        <div style="font-size: 1rem; font-weight: 600; color: {_TEXT}; margin-bottom: 6px; font-family: Inter, sans-serif;">
            {headline}
        </div>
        <div style="
            font-size: 0.85rem; color: {_TEXT_SEC}; line-height: 1.5;
            padding-left: 10px; border-left: 2px solid {_BORDER}; margin-left: 2px;
            font-family: Inter, sans-serif;
        ">
            {adjustment}
        </div>
    </div>
    """


def safe_actions_chips(action_names: list[str]) -> str:
    """Render safe actions as green chips."""
    if not action_names:
        return f'<div style="color: {_TEXT_MUTED}; font-size: 0.85rem;">No safe actions identified yet.</div>'

    chips = ""
    for name in action_names:
        chips += f"""<span style="
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(56,161,105,0.1); border: 1px solid rgba(56,161,105,0.25);
            border-radius: 20px; padding: 6px 14px; font-size: 0.82rem;
            color: #68d391; font-weight: 500; margin: 3px 4px;
            font-family: Inter, sans-serif;
        ">&#10003; {name}</span>"""

    return f'<div style="display: flex; flex-wrap: wrap; gap: 4px;">{chips}</div>'


def action_bar_chart(
    actions: dict[str, int],
    max_count: int,
    color_gradient: str = "linear-gradient(90deg, #4299e1, #63b3ed)",
    label_resolver=None,
) -> str:
    """Render a horizontal bar chart of action counts."""
    if not actions:
        return f'<div style="color: {_TEXT_MUTED}; font-size: 0.85rem;">No actions recorded.</div>'

    sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:8]
    html = ""

    for action_id, count in sorted_actions:
        name = label_resolver(action_id) if label_resolver else action_id
        width_pct = (count / max(max_count, 1)) * 100

        html += f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span style="font-size: 0.82rem; color: {_TEXT_SEC}; min-width: 140px; text-align: right; font-weight: 500; font-family: Inter, sans-serif;">{name}</span>
            <div style="height: 20px; border-radius: 4px; width: {width_pct}%; background: {color_gradient}; min-width: 4px;"></div>
            <span style="font-size: 0.78rem; color: {_TEXT_SEC}; font-family: 'JetBrains Mono', monospace; font-weight: 500; min-width: 28px;">{count}</span>
        </div>
        """

    return html


def threat_pattern_rows(
    patterns: list[dict],
    max_threat: float,
    label_resolver=None,
) -> str:
    """Render ranked threat pattern rows."""
    if not patterns:
        return f'<div style="color: {_TEXT_MUTED}; font-size: 0.85rem;">No recurring patterns detected yet.</div>'

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
        <div style="display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-radius: 8px; margin-bottom: 4px;">
            <span style="font-size: 1.1rem; font-weight: 800; color: {_TEXT_MUTED}; min-width: 24px; font-family: Inter, sans-serif;">{i + 1}</span>
            <div style="flex: 1;">
                <div style="font-size: 0.88rem; font-weight: 600; color: {_TEXT}; margin-bottom: 4px; font-family: Inter, sans-serif;">
                    {your_name} <span style="color: {_RED}; font-weight: 600;">&rarr;</span> {their_name}
                </div>
                <div style="height: 6px; border-radius: 3px; width: {bar_width}%; background: linear-gradient(90deg, {_RED}, {_ORANGE}); min-width: 4px;"></div>
                <div style="font-size: 0.72rem; color: {_TEXT_MUTED}; margin-top: 3px; font-family: 'JetBrains Mono', monospace;">
                    {occ}x &middot; {landed} landed &middot; threat {threat:.0f} &middot; {rounds_str}
                </div>
            </div>
        </div>
        """

    return html


def section_header(title: str, subtitle: str = "", icon: str = "") -> str:
    """Render a styled section header."""
    sub = f'<div style="font-size: 0.82rem; color: {_TEXT_MUTED}; margin-top: 2px; font-family: Inter, sans-serif;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="margin-bottom: 1rem; margin-top: 0.5rem;">
        <div style="font-size: 1.1rem; font-weight: 700; color: {_TEXT}; display: flex; align-items: center; gap: 8px; font-family: Inter, sans-serif;">
            {icon} {title}
        </div>
        {sub}
    </div>
    """


def glass_card_wrapper(content: str) -> str:
    """Wrap content in a glass card."""
    return f"""
    <div style="
        background: {_BG_CARD}; border: 1px solid {_BORDER};
        border-radius: 16px; padding: 1.25rem; margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    ">{content}</div>
    """


def divider() -> str:
    """Render a styled horizontal divider."""
    return f'<div style="border-top: 1px solid {_BORDER}; margin: 1.5rem 0;"></div>'
