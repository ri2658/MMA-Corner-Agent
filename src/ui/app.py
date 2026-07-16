"""MMA Corner Agent — Streamlit Dashboard.

Launch with:
    streamlit run src/ui/app.py

Supports two modes:
  1. Synthetic simulation — instant, no dependencies
  2. Video analysis — upload a fight clip for real analysis
"""

from __future__ import annotations

import sys
import tempfile
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
# Session state
# ======================================================================
if "reports" not in st.session_state:
    st.session_state.reports = []
if "agent" not in st.session_state:
    st.session_state.agent = CornerAgent()
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if "fight_run" not in st.session_state:
    st.session_state.fight_run = False
if "mode" not in st.session_state:
    st.session_state.mode = "synthetic"


def resolve_name(action_id: str) -> str:
    """Resolve an action ID to its human-readable name."""
    entry = st.session_state.kb.get_action(action_id)
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
                    background-clip: text; font-family: Inter, sans-serif;">
            🥊 Corner Agent
        </div>
        <div style="font-size: 0.78rem; color: #8888a0; margin-top: 2px; font-family: Inter, sans-serif;">
            Tactical fight analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    mode = st.radio(
        "Analysis Mode",
        ["Synthetic Simulation", "Video Analysis"],
        index=0,
        help="Synthetic works instantly. Video requires MediaPipe + YOLO.",
    )

    st.divider()

    if mode == "Synthetic Simulation":
        st.markdown("#### Simulation Settings")

        num_rounds = st.slider(
            "Rounds to simulate", 1, 12, 3,
            help="Number of rounds in the simulated fight",
        )
        events_per_sec = st.slider(
            "Action frequency", 0.3, 2.0, 0.8, 0.1,
            help="Average offensive actions per second",
        )
        seed = st.number_input(
            "Random seed", 1, 999, 42,
            help="Change for a different fight scenario",
        )

        st.divider()

        if st.button("⚡ Run Fight Simulation", width="stretch"):
            agent = CornerAgent(min_pattern_occurrences=2, max_advice_items=4)
            sim = SyntheticFight(seed=int(seed))
            reports = []

            progress = st.progress(0, text="Simulating fight...")
            for rd in range(1, num_rounds + 1):
                a_states, b_states = sim.generate_round(
                    round_number=rd, events_per_second=events_per_sec,
                )
                agent.start_round(rd)
                agent.ingest_states(a_states, b_states, round_number=rd)
                reports.append(agent.end_round())
                progress.progress(rd / num_rounds, text=f"Round {rd}/{num_rounds}")

            st.session_state.reports = reports
            st.session_state.agent = agent
            st.session_state.fight_run = True
            st.session_state.mode = "synthetic"
            progress.empty()
            st.rerun()

    else:
        st.markdown("#### Video Settings")

        target_fps = st.slider(
            "Processing FPS", 5.0, 30.0, 15.0, 1.0,
            help="Higher = more accurate but slower",
        )
        max_frames = st.number_input(
            "Max frames (0 = all)", 0, 10000, 0,
            help="Limit frames for faster testing",
        )

        st.divider()

        # File upload
        uploaded = st.file_uploader(
            "Upload fight clip",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video of a fight round",
        )

        # Or use the bundled clip
        use_bundled = st.checkbox(
            "Use bundled clip (Yan vs Dvalishvili)",
            value=True,
            help="Use the included sample clip",
        )

        if st.button("🎬 Analyze Video", width="stretch"):
            video_path = None

            if uploaded is not None:
                # Save uploaded file to temp
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded.read())
                tmp.close()
                video_path = tmp.name
            elif use_bundled:
                bundled = Path(__file__).resolve().parent.parent.parent / "data" / "yan_vs_dvalishvili_2_clips.mp4"
                if bundled.exists():
                    video_path = str(bundled)
                else:
                    st.error(f"Bundled clip not found at {bundled}")

            if video_path:
                agent = CornerAgent(min_pattern_occurrences=2, max_advice_items=4)
                mf = int(max_frames) if max_frames > 0 else None

                progress = st.progress(0, text="Processing video...")
                status_text = st.empty()

                def progress_cb(frames, ts):
                    if frames % 10 == 0:
                        # Update progress based on frame count
                        if mf:
                            progress.progress(min(frames / mf, 1.0), text=f"Frame {frames} | t={ts:.1f}s")
                        else:
                            status_text.text(f"Frame {frames} | t={ts:.1f}s")

                try:
                    report = agent.analyze_video(
                        video_path,
                        round_number=1,
                        target_fps=target_fps,
                        max_frames=mf,
                        progress_callback=progress_cb,
                    )
                    st.session_state.reports = [report]
                    st.session_state.agent = agent
                    st.session_state.fight_run = True
                    st.session_state.mode = "video"
                    progress.empty()
                    status_text.empty()
                    st.rerun()
                except Exception as e:
                    progress.empty()
                    status_text.empty()
                    st.error(f"Error processing video: {e}")
                    st.info(
                        "Video analysis requires `mediapipe` and `ultralytics` (YOLO). "
                        "Install with: `pip install mediapipe ultralytics`"
                    )
            else:
                st.warning("Please upload a video or enable the bundled clip.")

    if st.session_state.fight_run:
        st.divider()
        st.success(f"✓ {len(st.session_state.reports)} round(s) analyzed")

    st.divider()
    st.markdown("""
    <div style="font-size: 0.72rem; color: #555570; line-height: 1.6; font-family: Inter, sans-serif;">
        <strong>How it works:</strong><br>
        1. Detect fighter actions via pose estimation<br>
        2. Link actions to opponent counters by timing<br>
        3. Rank counter patterns by frequency &times; damage<br>
        4. Look up tactical adjustments from the knowledge base
    </div>
    """, unsafe_allow_html=True)


# ======================================================================
# Main content
# ======================================================================
st.markdown(C.header_banner(), unsafe_allow_html=True)

if not st.session_state.fight_run:
    st.markdown(f"""
    <div style="
        background: {C._BG_CARD}; border: 1px solid {C._BORDER};
        border-radius: 16px; text-align: center; padding: 3rem 2rem;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🥊</div>
        <div style="font-size: 1.2rem; font-weight: 600; color: {C._TEXT}; margin-bottom: 0.5rem; font-family: Inter, sans-serif;">
            Ready to Analyze
        </div>
        <div style="font-size: 0.9rem; color: {C._TEXT_SEC}; max-width: 500px; margin: 0 auto; line-height: 1.6; font-family: Inter, sans-serif;">
            Choose <strong>Synthetic Simulation</strong> for an instant demo, or
            <strong>Video Analysis</strong> to process a real fight clip.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────────────────────────────
# Data loaded — render dashboard
# ──────────────────────────────────────────────────────────────────────
reports: list[RoundReport] = st.session_state.reports
agent: CornerAgent = st.session_state.agent

# ── Overview stat pills ───────────────────────────────────────────────
total_pairs = sum(r.pairs_detected for r in reports)
total_a = sum(sum(r.fighter_a_actions.values()) for r in reports)
total_b = sum(sum(r.fighter_b_actions.values()) for r in reports)
criticals = sum(1 for r in reports for a in r.advice if a.severity == "critical")

st.markdown(
    C.stat_pills([
        (str(len(reports)), "Rounds"),
        (str(total_pairs), "Pairs Detected"),
        (str(total_a), "Your Actions"),
        (str(total_b), "Opp. Counters"),
        (str(criticals), "Critical Alerts"),
    ]),
    unsafe_allow_html=True,
)

# ── Round tabs ────────────────────────────────────────────────────────
tab_labels = [f"Round {r.round_number}" for r in reports]
if len(reports) > 1:
    tab_labels.append("Fight Summary")

tabs = st.tabs(tab_labels)

# ── Render each round tab ─────────────────────────────────────────────
for idx in range(len(reports)):
    report = reports[idx]

    with tabs[idx]:
        # ── Round stats ───────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{report.duration_s:.0f}s")
        c2.metric("Pairs Detected", report.pairs_detected)
        c3.metric("Your Actions", sum(report.fighter_a_actions.values()))
        c4.metric("Counters Received", sum(report.fighter_b_actions.values()))

        st.markdown(C.divider(), unsafe_allow_html=True)

        # ── Two-column layout ─────────────────────────────────────
        left, right = st.columns([3, 2])

        # ── Left: Corner Advice + Safe Actions ────────────────────
        with left:
            st.markdown(
                C.section_header("Corner Advice", "Prioritized adjustments based on detected counter patterns", "🎯"),
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
                        f'<div style="text-align: center; color: {C._TEXT_MUTED}; padding: 1rem; font-family: Inter, sans-serif;">'
                        '&#10003; No significant counter patterns &mdash; fighter is performing well</div>'
                    ),
                    unsafe_allow_html=True,
                )

            # Safe actions
            if report.safe_actions:
                st.markdown(
                    C.section_header("Keep Doing This", "Actions not being effectively countered", "✅"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    C.glass_card_wrapper(
                        C.safe_actions_chips([resolve_name(a) for a in report.safe_actions])
                    ),
                    unsafe_allow_html=True,
                )

        # ── Right: Action Breakdown ───────────────────────────────
        with right:
            st.markdown(
                C.section_header("Action Breakdown", "Strike output per fighter", "📊"),
                unsafe_allow_html=True,
            )

            all_counts = list(report.fighter_a_actions.values()) + list(report.fighter_b_actions.values())
            max_count = max(all_counts) if all_counts else 1

            # Your fighter
            st.markdown(
                f'<div style="font-size: 0.78rem; font-weight: 600; color: {C._BLUE}; margin-bottom: 6px; font-family: Inter, sans-serif;">YOUR FIGHTER</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                C.glass_card_wrapper(
                    C.action_bar_chart(
                        report.fighter_a_actions, max_count,
                        "linear-gradient(90deg, #4299e1, #63b3ed)",
                        resolve_name,
                    )
                ),
                unsafe_allow_html=True,
            )

            # Opponent
            st.markdown(
                f'<div style="font-size: 0.78rem; font-weight: 600; color: {C._RED}; margin-bottom: 6px; font-family: Inter, sans-serif;">OPPONENT</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                C.glass_card_wrapper(
                    C.action_bar_chart(
                        report.fighter_b_actions, max_count,
                        "linear-gradient(90deg, #e53e3e, #fc8181)",
                        resolve_name,
                    )
                ),
                unsafe_allow_html=True,
            )

        st.markdown(C.divider(), unsafe_allow_html=True)

        # ── Counter Pattern Detail ────────────────────────────────
        st.markdown(
            C.section_header("Counter Pattern Detail", "All detected action → counter pairs ranked by threat", "🔍"),
            unsafe_allow_html=True,
        )

        patterns = [p.to_dict() for p in report.top_patterns]
        max_threat = max((p.get("threat_score", 0) for p in patterns), default=1)

        col_table, col_chart = st.columns([1, 1])

        with col_table:
            st.markdown(
                C.glass_card_wrapper(
                    C.threat_pattern_rows(patterns, max_threat, resolve_name)
                ),
                unsafe_allow_html=True,
            )

        with col_chart:
            if patterns:
                names = [
                    f"{resolve_name(p['your_action'])} → {resolve_name(p['their_counter'])}"
                    for p in patterns
                ]
                scores = [p["threat_score"] for p in patterns]
                occ = [p["occurrences"] for p in patterns]

                fig = go.Figure(go.Bar(
                    y=names, x=scores, orientation="h",
                    marker=dict(
                        color=scores,
                        colorscale=[[0, "#ed8936"], [0.5, "#e53e3e"], [1, "#9b2c2c"]],
                    ),
                    text=[f"{s:.0f}" for s in scores],
                    textposition="outside",
                    textfont=dict(color="#8888a0", size=11),
                    hovertemplate="<b>%{y}</b><br>Threat: %{x:.0f}<br>Occ: %{customdata}<extra></extra>",
                    customdata=occ,
                ))
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#8888a0"),
                    margin=dict(l=10, r=40, t=10, b=10),
                    height=max(180, len(patterns) * 50),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", title="Threat Score"),
                    yaxis=dict(showgrid=False, autorange="reversed"),
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")


# ── Fight Summary tab (only if multi-round) ───────────────────────────
if len(reports) > 1:
    with tabs[-1]:
        st.markdown(
            C.section_header("Fight Summary", f"Accumulated analysis across {len(reports)} rounds", "📈"),
            unsafe_allow_html=True,
        )

        fight_patterns = agent.get_fight_patterns()
        fight_advice = agent.get_fight_advice()

        left_s, right_s = st.columns([3, 2])

        with left_s:
            st.markdown(
                C.section_header("Top Threats Across Fight", "Patterns that persisted across multiple rounds", "⚠️"),
                unsafe_allow_html=True,
            )
            if fight_patterns:
                fp = [p.to_dict() for p in fight_patterns]
                mt = max(p["threat_score"] for p in fp)
                st.markdown(C.glass_card_wrapper(C.threat_pattern_rows(fp, mt, resolve_name)), unsafe_allow_html=True)

            st.markdown(C.section_header("Fight-Level Adjustments", "", "🎯"), unsafe_allow_html=True)
            if fight_advice:
                for i, adv in enumerate(fight_advice):
                    d = adv.to_display_dict()
                    st.markdown(
                        C.advice_card(adv.severity, d["headline"], d["stats"], d["adjustment"], d["icon"], i),
                        unsafe_allow_html=True,
                    )

        with right_s:
            # Round trend chart
            st.markdown(C.section_header("Round Trend", "How threats evolved", "📉"), unsafe_allow_html=True)

            rounds_x = [f"R{r.round_number}" for r in reports]
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=rounds_x, y=[r.pairs_detected for r in reports],
                mode="lines+markers", name="Pairs",
                line=dict(color="#4299e1", width=2.5), marker=dict(size=8),
            ))
            fig_t.add_trace(go.Scatter(
                x=rounds_x,
                y=[sum(1 for a in r.advice if a.severity == "critical") for r in reports],
                mode="lines+markers", name="Critical",
                line=dict(color="#e53e3e", width=2.5), marker=dict(size=8, symbol="diamond"),
            ))
            fig_t.add_trace(go.Scatter(
                x=rounds_x,
                y=[sum(1 for a in r.advice if a.severity == "warning") for r in reports],
                mode="lines+markers", name="Warning",
                line=dict(color="#ed8936", width=2, dash="dot"), marker=dict(size=7, symbol="square"),
            ))
            fig_t.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#8888a0"),
                margin=dict(l=10, r=10, t=10, b=10), height=280,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", title="Count"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
            )
            st.plotly_chart(fig_t, width="stretch")

            # Action distribution donut
            st.markdown(C.section_header("Your Action Distribution", "Across entire fight", "🎯"), unsafe_allow_html=True)

            all_a: dict[str, int] = {}
            for r in reports:
                for aid, cnt in r.fighter_a_actions.items():
                    all_a[aid] = all_a.get(aid, 0) + cnt

            if all_a:
                top_a = sorted(all_a.items(), key=lambda x: x[1], reverse=True)[:10]
                fig_pie = go.Figure(go.Pie(
                    labels=[resolve_name(a) for a, _ in top_a],
                    values=[c for _, c in top_a],
                    hole=0.55,
                    marker=dict(colors=[
                        "#4299e1", "#63b3ed", "#90cdf4", "#805ad5", "#b794f4",
                        "#d6bcfa", "#38a169", "#68d391", "#9ae6b4", "#ed8936",
                    ]),
                    textinfo="percent",
                    textfont=dict(size=11, color="#e8e8f0"),
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                ))
                fig_pie.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#8888a0"),
                    margin=dict(l=10, r=10, t=10, b=10), height=300,
                    legend=dict(font=dict(size=10, color="#8888a0"), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_pie, width="stretch")
