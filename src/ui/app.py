"""MMA Corner Agent — Streamlit Dashboard.

Launch with:
    streamlit run src/ui/app.py

A premium dark-themed dashboard that shows:
  - Round-by-round corner advice with severity badges
  - Action breakdown charts per fighter
  - Counter pattern threat rankings
  - Safe action identification
  - Fight-level trend analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go

from src.corner_agent import CornerAgent, RoundReport
from src.demo import SyntheticFight
from src.strategy.knowledge_base import KnowledgeBase
from src.ui.styles import get_styles
from src.ui import components as C


# ======================================================================
# Page config
# ======================================================================
st.set_page_config(
    page_title="MMA Corner Agent",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(get_styles(), unsafe_allow_html=True)


# ======================================================================
# Session state initialization
# ======================================================================
if "reports" not in st.session_state:
    st.session_state.reports = []
if "agent" not in st.session_state:
    st.session_state.agent = CornerAgent()
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if "fight_run" not in st.session_state:
    st.session_state.fight_run = False


def resolve_name(action_id: str) -> str:
    """Resolve an action ID to its human-readable name."""
    kb: KnowledgeBase = st.session_state.kb
    entry = kb.get_action(action_id)
    return entry["name"] if entry else action_id


# ======================================================================
# Sidebar
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 1.4rem; font-weight: 800; letter-spacing: -0.03em;
                    background: linear-gradient(135deg, #fff 0%, #e53e3e 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">
            🥊 Corner Agent
        </div>
        <div style="font-size: 0.78rem; color: #8888a0; margin-top: 2px;">
            Tactical fight analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("#### Simulation Settings")

    num_rounds = st.slider(
        "Rounds to simulate",
        min_value=1, max_value=12, value=3,
        help="Number of rounds in the simulated fight",
    )

    events_per_sec = st.slider(
        "Action frequency",
        min_value=0.3, max_value=2.0, value=0.8, step=0.1,
        help="Average offensive actions per second",
    )

    seed = st.number_input(
        "Random seed",
        min_value=1, max_value=999, value=42,
        help="Change this to get a different fight scenario",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    run_button = st.button("⚡ Run Fight Simulation", use_container_width=True)

    if run_button:
        agent = CornerAgent(min_pattern_occurrences=2, max_advice_items=4)
        sim = SyntheticFight(seed=int(seed))
        reports: list[RoundReport] = []

        progress = st.progress(0, text="Simulating fight...")

        for rd in range(1, num_rounds + 1):
            a_states, b_states = sim.generate_round(
                round_number=rd,
                events_per_second=events_per_sec,
            )
            agent.start_round(rd)
            agent.ingest_states(a_states, b_states, round_number=rd)
            report = agent.end_round()
            reports.append(report)
            progress.progress(rd / num_rounds, text=f"Round {rd}/{num_rounds}")

        st.session_state.reports = reports
        st.session_state.agent = agent
        st.session_state.fight_run = True
        progress.empty()
        st.rerun()

    if st.session_state.fight_run:
        st.success(f"✓ {len(st.session_state.reports)} rounds analyzed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.72rem; color: #555570; line-height: 1.6;">
        <strong>How it works:</strong><br>
        1. Detect fighter actions via pose estimation<br>
        2. Link actions to opponent counters by timing<br>
        3. Rank counter patterns by frequency × damage<br>
        4. Look up tactical adjustments from the knowledge base
    </div>
    """, unsafe_allow_html=True)


# ======================================================================
# Main content
# ======================================================================
st.markdown(C.header_banner(), unsafe_allow_html=True)

if not st.session_state.fight_run:
    # Landing state
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🥊</div>
        <div style="font-size: 1.2rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">
            Ready to Analyze
        </div>
        <div style="font-size: 0.9rem; color: var(--text-secondary); max-width: 500px; margin: 0 auto; line-height: 1.6;">
            Click <strong>⚡ Run Fight Simulation</strong> in the sidebar to generate
            a synthetic fight and see real-time corner advice. No video or GPU required.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────────────────────────────
# Data loaded — render the dashboard
# ──────────────────────────────────────────────────────────────────────
reports: list[RoundReport] = st.session_state.reports
agent: CornerAgent = st.session_state.agent

# ── Fight overview stats ──────────────────────────────────────────────
total_pairs = sum(r.pairs_detected for r in reports)
total_advice = sum(len(r.advice) for r in reports)
total_a_actions = sum(sum(r.fighter_a_actions.values()) for r in reports)
total_b_actions = sum(sum(r.fighter_b_actions.values()) for r in reports)
critical_count = sum(
    1 for r in reports for a in r.advice if a.severity == "critical"
)

st.markdown(
    C.stat_pills([
        (str(len(reports)), "Rounds"),
        (str(total_pairs), "Pairs Detected"),
        (str(total_a_actions), "Your Actions"),
        (str(total_b_actions), "Opp. Counters"),
        (str(critical_count), "Critical Alerts"),
    ]),
    unsafe_allow_html=True,
)


# ── Round tabs ────────────────────────────────────────────────────────
tab_labels = [f"Round {r.round_number}" for r in reports] + ["Fight Summary"]
tabs = st.tabs(tab_labels)

for idx, tab in enumerate(tabs[:-1]):
    report = reports[idx]

    with tab:
        # ── Round stats row ───────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{report.duration_s:.0f}s")
        with col2:
            st.metric("Pairs Detected", report.pairs_detected)
        with col3:
            st.metric(
                "Your Actions",
                sum(report.fighter_a_actions.values()),
            )
        with col4:
            st.metric(
                "Counters Received",
                sum(report.fighter_b_actions.values()),
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Corner Advice ─────────────────────────────────────────
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown(
                C.section_header(
                    "Corner Advice",
                    "Prioritized adjustments based on detected counter patterns",
                    "🎯",
                ),
                unsafe_allow_html=True,
            )

            if report.advice:
                for i, adv in enumerate(report.advice):
                    d = adv.to_display_dict()
                    st.markdown(
                        C.advice_card(
                            severity=adv.severity,
                            headline=d["headline"],
                            stats_text=d["stats"],
                            adjustment=d["adjustment"],
                            icon=d["icon"],
                            index=i,
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    C.glass_card_wrapper(
                        '<div style="text-align: center; color: var(--text-muted); padding: 1rem;">'
                        '✓ No significant counter patterns — fighter is performing well</div>'
                    ),
                    unsafe_allow_html=True,
                )

            # Safe actions
            if report.safe_actions:
                st.markdown(
                    C.section_header("Keep Doing This", "Actions not being effectively countered", "✅"),
                    unsafe_allow_html=True,
                )
                safe_names = [resolve_name(a) for a in report.safe_actions]
                st.markdown(
                    C.glass_card_wrapper(C.safe_actions_chips(safe_names)),
                    unsafe_allow_html=True,
                )

        # ── Right column: Action breakdown ────────────────────────
        with right_col:
            st.markdown(
                C.section_header("Action Breakdown", "Strike output per fighter", "📊"),
                unsafe_allow_html=True,
            )

            all_counts = list(report.fighter_a_actions.values()) + list(report.fighter_b_actions.values())
            max_count = max(all_counts) if all_counts else 1

            st.markdown(
                '<div style="font-size: 0.78rem; font-weight: 600; color: var(--accent-blue); margin-bottom: 6px;">YOUR FIGHTER</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                C.glass_card_wrapper(
                    C.action_bar_chart(report.fighter_a_actions, max_count, "action-bar-a", resolve_name)
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="font-size: 0.78rem; font-weight: 600; color: var(--accent-red); margin-bottom: 6px;">OPPONENT</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                C.glass_card_wrapper(
                    C.action_bar_chart(report.fighter_b_actions, max_count, "action-bar-b", resolve_name)
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Counter Pattern Details ───────────────────────────────
        st.markdown(
            C.section_header("Counter Pattern Detail", "All detected action → counter pairs ranked by threat", "🔍"),
            unsafe_allow_html=True,
        )

        patterns = [p.to_dict() for p in report.top_patterns]
        if not patterns:
            # Show all from aggregator summary if top_patterns is empty
            from src.analysis.pattern_aggregator import PatternAggregator
            patterns = []

        max_threat = max((p.get("threat_score", 0) for p in patterns), default=1)

        col_pat, col_chart = st.columns([1, 1])

        with col_pat:
            st.markdown(
                C.glass_card_wrapper(
                    C.threat_pattern_rows(patterns, max_threat, resolve_name)
                ),
                unsafe_allow_html=True,
            )

        with col_chart:
            if patterns:
                fig = go.Figure()
                names = [
                    f"{resolve_name(p['your_action'])} → {resolve_name(p['their_counter'])}"
                    for p in patterns
                ]
                scores = [p["threat_score"] for p in patterns]
                occ = [p["occurrences"] for p in patterns]

                fig.add_trace(go.Bar(
                    y=names,
                    x=scores,
                    orientation="h",
                    marker=dict(
                        color=scores,
                        colorscale=[[0, "#ed8936"], [0.5, "#e53e3e"], [1, "#9b2c2c"]],
                    ),
                    text=[f"{s:.0f}" for s in scores],
                    textposition="outside",
                    textfont=dict(color="#8888a0", size=11),
                    hovertemplate="<b>%{y}</b><br>Threat: %{x:.0f}<br>Occurrences: %{customdata}<extra></extra>",
                    customdata=occ,
                ))

                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#8888a0"),
                    margin=dict(l=10, r=30, t=10, b=10),
                    height=max(180, len(patterns) * 50),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.04)",
                        title="Threat Score",
                        title_font=dict(size=11),
                    ),
                    yaxis=dict(
                        showgrid=False,
                        autorange="reversed",
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)


# ── Fight Summary tab ─────────────────────────────────────────────────
with tabs[-1]:
    st.markdown(
        C.section_header(
            "Fight Summary",
            f"Accumulated analysis across {len(reports)} rounds",
            "📈",
        ),
        unsafe_allow_html=True,
    )

    # ── Fight-level patterns ──────────────────────────────────────
    fight_patterns = agent.get_fight_patterns()
    fight_advice = agent.get_fight_advice()

    sum_col1, sum_col2 = st.columns([3, 2])

    with sum_col1:
        st.markdown(
            C.section_header("Top Threats Across Fight", "Patterns that persisted across multiple rounds", "⚠️"),
            unsafe_allow_html=True,
        )

        if fight_patterns:
            f_patterns = [p.to_dict() for p in fight_patterns]
            max_t = max(p["threat_score"] for p in f_patterns)
            st.markdown(
                C.glass_card_wrapper(
                    C.threat_pattern_rows(f_patterns, max_t, resolve_name)
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            C.section_header("Fight-Level Adjustments", "", "🎯"),
            unsafe_allow_html=True,
        )

        if fight_advice:
            for i, adv in enumerate(fight_advice):
                d = adv.to_display_dict()
                st.markdown(
                    C.advice_card(
                        severity=adv.severity,
                        headline=d["headline"],
                        stats_text=d["stats"],
                        adjustment=d["adjustment"],
                        icon=d["icon"],
                        index=i,
                    ),
                    unsafe_allow_html=True,
                )

    with sum_col2:
        # ── Round-over-round trend chart ──────────────────────────
        st.markdown(
            C.section_header("Round Trend", "How threats evolved across rounds", "📉"),
            unsafe_allow_html=True,
        )

        fig_trend = go.Figure()

        rounds_x = [f"R{r.round_number}" for r in reports]
        pairs_y = [r.pairs_detected for r in reports]
        critical_y = [
            sum(1 for a in r.advice if a.severity == "critical")
            for r in reports
        ]
        warnings_y = [
            sum(1 for a in r.advice if a.severity == "warning")
            for r in reports
        ]

        fig_trend.add_trace(go.Scatter(
            x=rounds_x, y=pairs_y,
            mode="lines+markers",
            name="Pairs Detected",
            line=dict(color="#4299e1", width=2.5),
            marker=dict(size=8),
        ))
        fig_trend.add_trace(go.Scatter(
            x=rounds_x, y=critical_y,
            mode="lines+markers",
            name="Critical Alerts",
            line=dict(color="#e53e3e", width=2.5),
            marker=dict(size=8, symbol="diamond"),
        ))
        fig_trend.add_trace(go.Scatter(
            x=rounds_x, y=warnings_y,
            mode="lines+markers",
            name="Warnings",
            line=dict(color="#ed8936", width=2, dash="dot"),
            marker=dict(size=7, symbol="square"),
        ))

        fig_trend.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#8888a0"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.04)",
                title="Count",
                title_font=dict(size=11),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=11),
            ),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Action distribution pie ───────────────────────────────
        st.markdown(
            C.section_header("Your Action Distribution", "Across entire fight", "🎯"),
            unsafe_allow_html=True,
        )

        all_a_actions: dict[str, int] = {}
        for r in reports:
            for aid, cnt in r.fighter_a_actions.items():
                all_a_actions[aid] = all_a_actions.get(aid, 0) + cnt

        if all_a_actions:
            sorted_a = sorted(all_a_actions.items(), key=lambda x: x[1], reverse=True)[:10]
            labels = [resolve_name(a) for a, _ in sorted_a]
            values = [c for _, c in sorted_a]

            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(
                    colors=[
                        "#4299e1", "#63b3ed", "#90cdf4",
                        "#805ad5", "#b794f4", "#d6bcfa",
                        "#38a169", "#68d391", "#9ae6b4",
                        "#ed8936",
                    ]
                ),
                textinfo="percent",
                textfont=dict(size=11, color="#e8e8f0"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            )])

            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#8888a0"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                showlegend=True,
                legend=dict(
                    font=dict(size=10, color="#8888a0"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
