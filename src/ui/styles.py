"""Custom CSS for the MMA Corner Agent Streamlit dashboard.

Dark, premium fight-night aesthetic with glassmorphism cards,
severity-colored accents, and smooth micro-animations.
"""


def get_styles() -> str:
    """Return the full CSS stylesheet as a string."""
    return """
    <style>
    /* ───────────────────────────────────────────────────────────
       Google Font
    ─────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ───────────────────────────────────────────────────────────
       Root Variables
    ─────────────────────────────────────────────────────────── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(22, 22, 35, 0.85);
        --bg-card-hover: rgba(30, 30, 50, 0.95);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-glow: rgba(220, 50, 50, 0.25);

        --text-primary: #e8e8f0;
        --text-secondary: #8888a0;
        --text-muted: #555570;

        --accent-red: #e53e3e;
        --accent-red-glow: rgba(229, 62, 62, 0.3);
        --accent-orange: #ed8936;
        --accent-orange-glow: rgba(237, 137, 54, 0.3);
        --accent-green: #38a169;
        --accent-green-glow: rgba(56, 161, 105, 0.3);
        --accent-blue: #4299e1;
        --accent-blue-glow: rgba(66, 153, 225, 0.2);
        --accent-purple: #805ad5;

        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;

        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4);
        --shadow-elevated: 0 8px 40px rgba(0, 0, 0, 0.6);
    }

    /* ───────────────────────────────────────────────────────────
       Global Overrides
    ─────────────────────────────────────────────────────────── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: var(--font-main) !important;
        color: var(--text-primary) !important;
    }

    .stApp > header { background: transparent !important; }

    .stMainBlockContainer {
        max-width: 1200px !important;
        padding-top: 2rem !important;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden !important;
    }

    /* ───────────────────────────────────────────────────────────
       Typography
    ─────────────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-main) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    h1 {
        font-size: 2.2rem !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(135deg, #fff 0%, #e53e3e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    p, li, span, div {
        font-family: var(--font-main) !important;
    }

    /* ───────────────────────────────────────────────────────────
       Glass Card
    ─────────────────────────────────────────────────────────── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-card);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
        box-shadow: var(--shadow-elevated);
    }

    /* ───────────────────────────────────────────────────────────
       Severity Cards
    ─────────────────────────────────────────────────────────── */
    .advice-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        box-shadow: var(--shadow-card);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .advice-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 4px;
        border-radius: 4px 0 0 4px;
    }

    .advice-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-elevated);
    }

    .advice-critical {
        border: 1px solid rgba(229, 62, 62, 0.25);
    }
    .advice-critical::before { background: var(--accent-red); }
    .advice-critical:hover { border-color: rgba(229, 62, 62, 0.5); box-shadow: 0 8px 40px rgba(229, 62, 62, 0.15); }

    .advice-warning {
        border: 1px solid rgba(237, 137, 54, 0.25);
    }
    .advice-warning::before { background: var(--accent-orange); }
    .advice-warning:hover { border-color: rgba(237, 137, 54, 0.5); box-shadow: 0 8px 40px rgba(237, 137, 54, 0.15); }

    .advice-working {
        border: 1px solid rgba(56, 161, 105, 0.25);
    }
    .advice-working::before { background: var(--accent-green); }
    .advice-working:hover { border-color: rgba(56, 161, 105, 0.5); box-shadow: 0 8px 40px rgba(56, 161, 105, 0.15); }

    /* ───────────────────────────────────────────────────────────
       Severity Badge
    ─────────────────────────────────────────────────────────── */
    .severity-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .badge-critical {
        background: rgba(229, 62, 62, 0.15);
        color: #fc8181;
        border: 1px solid rgba(229, 62, 62, 0.3);
    }

    .badge-warning {
        background: rgba(237, 137, 54, 0.15);
        color: #fbd38d;
        border: 1px solid rgba(237, 137, 54, 0.3);
    }

    .badge-working {
        background: rgba(56, 161, 105, 0.15);
        color: #68d391;
        border: 1px solid rgba(56, 161, 105, 0.3);
    }

    /* ───────────────────────────────────────────────────────────
       Stat Pill
    ─────────────────────────────────────────────────────────── */
    .stat-pill {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1rem 1.5rem;
        min-width: 120px;
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(135deg, #fff 30%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.35rem;
    }

    /* ───────────────────────────────────────────────────────────
       Action Bar
    ─────────────────────────────────────────────────────────── */
    .action-bar-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }

    .action-bar-label {
        font-size: 0.82rem;
        color: var(--text-secondary);
        min-width: 160px;
        text-align: right;
        font-weight: 500;
    }

    .action-bar {
        height: 22px;
        border-radius: 4px;
        transition: width 0.6s ease;
    }

    .action-bar-a {
        background: linear-gradient(90deg, var(--accent-blue), #63b3ed);
    }

    .action-bar-b {
        background: linear-gradient(90deg, var(--accent-red), #fc8181);
    }

    .action-bar-count {
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-family: var(--font-mono);
        font-weight: 500;
        min-width: 30px;
    }

    /* ───────────────────────────────────────────────────────────
       Safe Action Chip
    ─────────────────────────────────────────────────────────── */
    .safe-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 161, 105, 0.1);
        border: 1px solid rgba(56, 161, 105, 0.25);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #68d391;
        font-weight: 500;
        margin: 3px 4px;
        transition: all 0.2s ease;
    }

    .safe-chip:hover {
        background: rgba(56, 161, 105, 0.2);
        border-color: rgba(56, 161, 105, 0.5);
        transform: translateY(-1px);
    }

    /* ───────────────────────────────────────────────────────────
       Threat Pattern Row
    ─────────────────────────────────────────────────────────── */
    .threat-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        border-radius: var(--radius-sm);
        margin-bottom: 4px;
        transition: background 0.2s ease;
    }

    .threat-row:hover {
        background: rgba(255, 255, 255, 0.03);
    }

    .threat-rank {
        font-size: 1.1rem;
        font-weight: 800;
        color: var(--text-muted);
        min-width: 24px;
    }

    .threat-arrow {
        color: var(--accent-red);
        font-weight: 600;
        font-size: 0.9rem;
    }

    .threat-score-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, var(--accent-red), var(--accent-orange));
        transition: width 0.6s ease;
    }

    /* ───────────────────────────────────────────────────────────
       Header Banner
    ─────────────────────────────────────────────────────────── */
    .header-banner {
        background: linear-gradient(135deg, rgba(229, 62, 62, 0.08) 0%, rgba(128, 90, 213, 0.08) 100%);
        border: 1px solid rgba(229, 62, 62, 0.15);
        border-radius: var(--radius-xl);
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .header-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(229, 62, 62, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin: 0;
        line-height: 1.1;
    }

    .header-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ───────────────────────────────────────────────────────────
       Round Tab Selector
    ─────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
        border: 1px solid transparent !important;
        border-bottom: none !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        border-color: var(--border-subtle) !important;
        color: var(--accent-red) !important;
    }

    /* ───────────────────────────────────────────────────────────
       Streamlit Widget Overrides
    ─────────────────────────────────────────────────────────── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, var(--accent-red) 0%, #c53030 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-family: var(--font-main) !important;
        letter-spacing: 0.02em;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(229, 62, 62, 0.3) !important;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(229, 62, 62, 0.5) !important;
    }



    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* ───────────────────────────────────────────────────────────
       Plotly Chart Container
    ─────────────────────────────────────────────────────────── */
    .stPlotlyChart {
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    /* ───────────────────────────────────────────────────────────
       Section Divider
    ─────────────────────────────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 1px solid var(--border-subtle);
        margin: 1.5rem 0;
    }

    /* ───────────────────────────────────────────────────────────
       Animations
    ─────────────────────────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeInUp 0.5s ease forwards;
    }

    .animate-in-delay-1 { animation-delay: 0.1s; opacity: 0; }
    .animate-in-delay-2 { animation-delay: 0.2s; opacity: 0; }
    .animate-in-delay-3 { animation-delay: 0.3s; opacity: 0; }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 8px rgba(229, 62, 62, 0.2); }
        50%      { box-shadow: 0 0 20px rgba(229, 62, 62, 0.4); }
    }

    .pulse-critical {
        animation: pulse-glow 2s ease-in-out infinite;
    }

    /* ───────────────────────────────────────────────────────────
       Scrollbar
    ─────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

    </style>
    """
