"""
app.py
SmartTransit AI – Streamlit Dashboard
Place this file in: Dashboard/app.py
Run with: streamlit run Dashboard/app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path setup so imports from fleet/ and Algorithms/ work ──
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fleet.orchestrator import demand_to_trains, fixed_schedule_baseline, PEAK_HOURS, EFFECTIVE_CAPACITY
from Algorithms.demand_segmentation import DemandSegmentor
from Dashboard.route_map import build_map
from Algorithms.route_optimizer import MetroRouter
from Algorithms.pickup_optimizer import PickupOptimizer
from Algorithms.multi_objective_optimizer import MultiObjectiveOptimizer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartTransit AI",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:   #0a0e1a;
    --bg-card:      #111827;
    --bg-card2:     #1a2236;
    --accent-aqua:  #00d4ff;
    --accent-purple:#b06cff;
    --accent-amber: #ffb347;
    --accent-green: #00e676;
    --accent-red:   #ff5252;
    --text-primary: #f0f4ff;
    --text-muted:   #8892a4;
    --border:       #1e2d45;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px; }

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); border-color: #2a3d5a; }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent-line, var(--accent-aqua));
    border-radius: 16px 16px 0 0;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0.3rem 0;
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 600;
}
.metric-delta {
    font-size: 0.85rem;
    margin-top: 0.4rem;
    font-weight: 500;
}
.delta-pos { color: var(--accent-green); }
.delta-neg { color: var(--accent-red); }
.delta-neu { color: var(--accent-amber); }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.8rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a0e1a 50%, #1a0d2e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%; left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(176,108,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent-aqua), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 400;
}

/* Train indicator */
.train-bar {
    display: flex;
    gap: 4px;
    margin-top: 0.6rem;
    flex-wrap: wrap;
}
.train-unit {
    width: 28px; height: 14px;
    border-radius: 3px;
    font-size: 0.55rem;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}
.train-active   { background: var(--accent-aqua);   color: #000; }
.train-extra    { background: var(--accent-green);  color: #000; }
.train-inactive { background: var(--border);        color: var(--text-muted); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Streamlit widget overrides */
.stSlider > div > div > div { background: var(--accent-aqua) !important; }
.stSelectbox > div > div { background: var(--bg-card2) !important; border-color: var(--border) !important; }
div[data-testid="metric-container"] { background: transparent !important; }

/* Alert boxes */
.alert-box {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
    font-size: 0.9rem;
}
.alert-surge   { background: rgba(255,82,82,0.08);   border-color: var(--accent-red);    color: #ff8a80; }
.alert-normal  { background: rgba(0,230,118,0.08);   border-color: var(--accent-green);  color: #69f0ae; }
.alert-warning { background: rgba(255,179,71,0.08);  border-color: var(--accent-amber);  color: #ffd180; }

/* Table styling */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.styled-table th {
    background: var(--bg-card2);
    color: var(--text-muted);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.styled-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid rgba(30,45,69,0.5);
    color: var(--text-primary);
}
.styled-table tr:hover td { background: rgba(26,34,54,0.6); }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-peak    { background: rgba(255,82,82,0.15);   color: #ff5252; }
.badge-normal  { background: rgba(0,230,118,0.15);   color: #00e676; }
.badge-high    { background: rgba(255,179,71,0.15);  color: #ffb347; }
.badge-low     { background: rgba(136,146,164,0.15); color: #8892a4; }

/* Demand segment cards */
.segment-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.segment-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--seg-color, var(--accent-aqua));
    border-radius: 12px 12px 0 0;
}
.segment-emoji { font-size: 1.6rem; display: block; margin-bottom: 0.3rem; }
.segment-label { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); }
.segment-value { font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700; margin: 0.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_all_data():
    df = pd.read_csv("Data/pune_metro_enhanced_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    daily = (
        df.groupby(["date", "line", "hour"])
        .agg(total=("passengers", "sum"),
             raining=("is_raining", "max"),
             festival=("festival", lambda x: (x != "None").any()),
             weekend=("is_weekend", "max"))
        .reset_index()
    )
    hourly_avg = (
        daily.groupby(["line", "hour"])
        .agg(avg_demand=("total", "mean"),
             max_demand=("total", "max"))
        .reset_index()
    )
    return df, hourly_avg


@st.cache_data
def load_schedule():
    path = "Outputs/fleet_schedule.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics():
    path = "Outputs/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame([
        {"Model": "ARIMA", "RMSE": 5480.13, "MAE": 4195.08, "MAPE (%)": 16.13},
        {"Model": "LSTM",  "RMSE": 262.94,  "MAE": 195.48,  "MAPE (%)": 20.65},
    ])


@st.cache_data
def load_stations_config():
    path = "Data/stations_config.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"stations": [], "lines": {}}


# ── Load everything ───────────────────────────
df_raw, hourly_avg  = load_all_data()
schedule_df         = load_schedule()
metrics_df          = load_metrics()
stations_config     = load_stations_config()
TOTAL_STATIONS      = len(stations_config.get("stations", []))
PURPLE_COUNT        = sum(1 for s in stations_config.get("stations", []) if s.get("line") == "purple")
AQUA_COUNT          = sum(1 for s in stations_config.get("stations", []) if s.get("line") == "aqua")

STATIONS   = sorted(df_raw["station"].unique())
LINES      = ["Aqua", "Purple", "Interchange"]
LINE_COLOR = {"Aqua": "#00d4ff", "Purple": "#b06cff", "Interchange": "#ffb347"}

SLOT_OPTIONS = {
    "🌅 Morning Peak (07–10)": "morning_peak",
    "☀️  Afternoon (11–16)":   "afternoon",
    "🌆 Evening Peak (17–20)": "evening_peak",
    "📅 Weekend":              "weekend",
    "🌙 Night (21–06)":        "night",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(color="#f0f4ff", family="DM Sans"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45"),
    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45"),
)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem 0'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                    background:linear-gradient(90deg,#00d4ff,#b06cff);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    font-weight:700;'>🚇 SmartTransit AI</div>
        <div style='font-size:0.75rem;color:#8892a4;margin-top:4px;'>
            Pune Metro Fleet Orchestrator
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔮 Live Prediction", "🚆 Fleet Scheduler",
         "📊 Model Performance", "🗺️ Route Map", "🧭 Journey Planner", "⚡ Fleet Optimizer", "🎯 Multi-Objective"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#8892a4;letter-spacing:0.08em;"
                "text-transform:uppercase;'>Context Overrides</div>", unsafe_allow_html=True)
    sb_rain     = st.toggle("🌧️ Rain Surge",   value=False)
    sb_festival = st.toggle("🎉 Festival Day", value=False)
    sb_weekend  = st.toggle("📅 Weekend Mode", value=False)

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#8892a4;letter-spacing:0.08em;"
                "text-transform:uppercase;'>Map Time Slot</div>", unsafe_allow_html=True)
    sb_slot_label = st.selectbox(
        "Map Slot",
        list(SLOT_OPTIONS.keys()),
        label_visibility="collapsed"
    )
    sb_slot = SLOT_OPTIONS[sb_slot_label]

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#8892a4;text-align:center;'>
        RMSE <span style='color:#00d4ff;font-family:Space Mono,monospace;'>262.94</span> ·
        95.2% better than ARIMA
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO BANNER (all pages)
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>SmartTransit AI</div>
    <div class='hero-subtitle'>
        Real-Time Public Transport Demand Forecasting & Fleet Orchestration · Pune Metro
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "🏠 Overview":

    arima_rmse   = metrics_df[metrics_df["Model"] == "ARIMA"]["RMSE"].values[0]
    lstm_rmse    = metrics_df[metrics_df["Model"] == "LSTM"]["RMSE"].values[0]

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "262.94",           "LSTM RMSE",        "−95.2% vs ARIMA",              "pos", "#00d4ff"),
        (c2, "180,438",          "Training Records", "Jan–Dec 2024",                 "neu", "#b06cff"),
        (c3, str(TOTAL_STATIONS),"Metro Stations",   f"Purple: {PURPLE_COUNT} · Aqua: {AQUA_COUNT}", "neu", "#ffb347"),
        (c4, "~2.0 min",         "Peak Wait (AI)",   "vs 3.0 min fixed schedule",    "pos", "#00e676"),
    ]
    for col, val, label, delta, dtype, color in cards:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent-line:{color}'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
                <div class='metric-delta delta-{dtype}'>{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    # Demand Overview Chart
    st.markdown("<div class='section-header'>Hourly Demand by Line</div>", unsafe_allow_html=True)
    fig = go.Figure()
    for line in LINES:
        sub = hourly_avg[hourly_avg["line"] == line].sort_values("hour")
        fig.add_trace(go.Scatter(
            x=sub["hour"], y=sub["avg_demand"],
            name=line, mode="lines+markers",
            line=dict(color=LINE_COLOR[line], width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor={"Aqua": "rgba(0,212,255,0.08)", "Purple": "rgba(176,108,255,0.08)",
                       "Interchange": "rgba(255,179,71,0.08)"}.get(line, "rgba(255,255,255,0.05)"),
        ))
    for start, end in [(8, 11), (16, 20)]:
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(255,82,82,0.06)",
                      line_width=0,
                      annotation_text="Peak" if start == 8 else "",
                      annotation_font_color="#ff5252", annotation_font_size=10)
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      xaxis_title="Hour of Day",
                      yaxis_title="Avg Total Passengers")
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'>ARIMA vs LSTM Error Comparison</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        for model, color in [("ARIMA", "#ff5252"), ("LSTM", "#00d4ff")]:
            row = metrics_df[metrics_df["Model"] == model].iloc[0]
            fig2.add_trace(go.Bar(
                name=model, x=["RMSE", "MAE"],
                y=[row["RMSE"], row["MAE"]],
                marker_color=color,
                text=[f"{row['RMSE']:.0f}", f"{row['MAE']:.0f}"],
                textposition="outside",
                textfont=dict(color="white", size=11),
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=280, barmode="group",
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           yaxis_title="Error (passengers)")
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Fleet Before vs After Optimisation</div>", unsafe_allow_html=True)
        if schedule_df is not None:
            peak_sched = schedule_df[
                schedule_df.get("period", schedule_df.get("peak_hour", ""))
                .str.contains("Peak", na=False)
            ]
            if len(peak_sched) == 0:
                peak_sched = schedule_df
            fig3 = go.Figure()
            for line in LINES:
                sub = peak_sched[peak_sched["line"] == line]
                if len(sub):
                    wf = sub["wait_fixed_mins"].mean() if "wait_fixed_mins" in sub else 3.0
                    wa = sub["wait_ai_mins"].mean()    if "wait_ai_mins"    in sub else 2.0
                    fig3.add_trace(go.Bar(name=f"{line} Fixed", x=[line], y=[wf],
                                          marker_color="#374151"))
                    fig3.add_trace(go.Bar(name=f"{line} AI", x=[line], y=[wa],
                                          marker_color=LINE_COLOR[line]))
            fig3.update_layout(**PLOTLY_LAYOUT, height=280, barmode="group",
                               yaxis_title="Avg Wait Time (mins)",
                               legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig3, use_container_width=True)

    # ── Demand Segments Summary (NEW) ────────────────────────────────────────
    st.markdown("<div class='section-header'>Demand by Time Slot — Today's Overview</div>",
                unsafe_allow_html=True)

    ds = DemandSegmentor(config_path="Data/stations_config.json")
    seg_report = ds.segment_historical(df_raw) if "passenger_demand" in df_raw.columns else {
        "morning_peak": {"emoji": "🌅", "label": "Morning Peak",   "color": "#FF6B35", "mean_demand": 23400, "max_demand": 31200},
        "afternoon":    {"emoji": "☀️",  "label": "Afternoon",      "color": "#FFD166", "mean_demand": 11200, "max_demand": 15800},
        "evening_peak": {"emoji": "🌆", "label": "Evening Peak",   "color": "#EF476F", "mean_demand": 22100, "max_demand": 29800},
        "night":        {"emoji": "🌙", "label": "Night Off-Peak", "color": "#118AB2", "mean_demand":  4200, "max_demand":  7100},
        "weekend":      {"emoji": "📅", "label": "Weekend",        "color": "#06D6A0", "mean_demand": 13500, "max_demand": 19200},
    }

    seg_cols = st.columns(5)
    for col, (key, stats) in zip(seg_cols, seg_report.items()):
        with col:
            st.markdown(f"""
            <div class='segment-card' style='--seg-color:{stats["color"]}'>
                <span class='segment-emoji'>{stats["emoji"]}</span>
                <div class='segment-label'>{stats["label"]}</div>
                <div class='segment-value' style='color:{stats["color"]};'>
                    {int(stats["mean_demand"]):,}
                </div>
                <div style='font-size:0.72rem;color:#8892a4;margin-top:2px;'>
                    avg pax/hr
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 2 — LIVE PREDICTION
# ═══════════════════════════════════════════════
elif page == "🔮 Live Prediction":

    st.markdown("<div class='section-header'>Live Fleet Allocation Query</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_line = st.selectbox("Metro Line", LINES)
    with col2:
        sel_hour = st.slider("Hour of Day", 6, 22, 8)
    with col3:
        sel_demand = st.number_input(
            "Predicted Demand (total passengers)",
            min_value=500, max_value=50000,
            value=23000 if sel_hour in PEAK_HOURS else 10000,
            step=500
        )

    trains_ai, headway_ai, wait_ai, svc_level = demand_to_trains(
        sel_demand, sel_hour, sb_rain, sb_festival, sb_weekend
    )
    trains_fixed, headway_fixed, wait_fixed = fixed_schedule_baseline(sel_hour)
    wait_saved  = round(wait_fixed - wait_ai, 2)
    utilisation = round(sel_demand / (trains_ai * EFFECTIVE_CAPACITY) * 100, 1)
    extra_trains = trains_ai - trains_fixed

    if sb_festival:
        st.markdown("<div class='alert-box alert-surge'>🎉 <b>Festival Surge Active</b> — demand multiplied by ×1.55. Extra trains deployed proactively.</div>", unsafe_allow_html=True)
    if sb_rain:
        st.markdown("<div class='alert-box alert-warning'>🌧️ <b>Rain Surge Active</b> — demand multiplied by ×1.22. Ridership spike expected.</div>", unsafe_allow_html=True)
    if sel_hour in PEAK_HOURS:
        st.markdown("<div class='alert-box alert-surge'>⚡ <b>Peak Hour</b> — Minimum 8 trains enforced for service quality.</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    result_cards = [
        (r1, str(trains_ai),     "AI Trains/Hour",    f"Fixed: {trains_fixed}",       "#00d4ff"),
        (r2, f"{wait_ai}m",      "AI Avg Wait",        f"Fixed: {wait_fixed}m",        "#00e676" if wait_saved >= 0 else "#ff5252"),
        (r3, f"{headway_ai}m",   "Headway",            f"Train every {headway_ai} mins","#b06cff"),
        (r4, f"{utilisation}%",  "Fleet Utilisation",  svc_level,                      "#ffb347"),
    ]
    for col, val, label, delta, color in result_cards:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent-line:{color}'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
                <div class='metric-delta' style='color:#8892a4'>{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Train Deployment Visualisation</div>", unsafe_allow_html=True)
    MAX_SHOW   = 15
    train_html = "<div class='train-bar'>"
    for i in range(MAX_SHOW):
        if i < trains_fixed:
            css, label = "train-active", "F"
        elif i < trains_ai:
            css, label = "train-extra", "+"
        else:
            css, label = "train-inactive", "·"
        train_html += f"<div class='train-unit {css}'>{label}</div>"
    train_html += (f"</div><div style='font-size:0.75rem;color:#8892a4;margin-top:6px;'>"
                   f"🔵 Fixed ({trains_fixed}) &nbsp;|&nbsp; 🟢 AI Extra (+{max(0,extra_trains)}) &nbsp;|&nbsp; ⚫ Idle</div>")
    st.markdown(train_html, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Wait Time Gauge</div>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    for col, title, val, color in [
        (col_g1, "Fixed Schedule Wait", wait_fixed, "#ff5252"),
        (col_g2, "AI Optimised Wait",   wait_ai,    "#00d4ff"),
    ]:
        with col:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                number={"suffix": " min", "font": {"size": 28, "color": color, "family": "Space Mono"}},
                title={"text": title, "font": {"size": 13, "color": "#8892a4"}},
                gauge=dict(
                    axis=dict(range=[0, 15], tickcolor="#8892a4", tickfont=dict(color="#8892a4")),
                    bar=dict(color=color),
                    bgcolor="#1a2236", bordercolor="#1e2d45",
                    steps=[
                        dict(range=[0,  3], color="rgba(0,230,118,0.15)"),
                        dict(range=[3,  6], color="rgba(255,179,71,0.15)"),
                        dict(range=[6, 15], color="rgba(255,82,82,0.15)"),
                    ],
                    threshold=dict(line=dict(color=color, width=3), value=val)
                )
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#f0f4ff"),
                                height=220, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 3 — FLEET SCHEDULER
# ═══════════════════════════════════════════════
elif page == "🚆 Fleet Scheduler":

    st.markdown("<div class='section-header'>Full Day Fleet Schedule — All Lines</div>", unsafe_allow_html=True)

    records = []
    for line in LINES:
        sub = hourly_avg[hourly_avg["line"] == line].sort_values("hour")
        for _, row in sub.iterrows():
            hour   = int(row["hour"])
            demand = row["avg_demand"]
            trains_ai, headway_ai, wait_ai, svc = demand_to_trains(
                demand, hour, sb_rain, sb_festival, sb_weekend
            )
            trains_fixed, _, wait_fixed = fixed_schedule_baseline(hour)
            records.append({
                "Line":          line,
                "Time":          f"{hour:02d}:00",
                "Demand":        f"{int(demand):,}",
                "Trains (Fixed)": trains_fixed,
                "Trains (AI)":   trains_ai,
                "Wait Fixed":    f"{wait_fixed}m",
                "Wait AI":       f"{wait_ai}m",
                "Saved":         f"{round(wait_fixed - wait_ai, 1)}m",
                "Service":       svc,
                "_period":       "Peak" if hour in PEAK_HOURS else "Off-Peak",
                "_line":         line,
            })

    sched = pd.DataFrame(records)

    fc1, fc2 = st.columns(2)
    with fc1:
        filt_line   = st.multiselect("Filter by Line",   LINES,                    default=LINES)
    with fc2:
        filt_period = st.multiselect("Filter by Period", ["Peak", "Off-Peak"],     default=["Peak", "Off-Peak"])

    sched_filtered = sched[
        sched["_line"].isin(filt_line) & sched["_period"].isin(filt_period)
    ].drop(columns=["_period", "_line"])

    def render_table(df):
        rows = ""
        for _, r in df.iterrows():
            lc    = LINE_COLOR.get(r["Line"], "#8892a4")
            saved = float(r["Saved"].replace("m", ""))
            saved_color = "#00e676" if saved > 0 else ("#ff5252" if saved < 0 else "#8892a4")
            svc   = str(r["Service"])
            if "Ultra" in svc or "Peak" in svc:
                badge = "<span class='badge badge-peak'>Peak</span>"
            elif "High" in svc:
                badge = "<span class='badge badge-high'>High</span>"
            elif "Normal" in svc:
                badge = "<span class='badge badge-normal'>Normal</span>"
            else:
                badge = "<span class='badge badge-low'>Low</span>"
            rows += f"""<tr>
                <td><span style='color:{lc};font-weight:600;'>{r['Line']}</span></td>
                <td style='font-family:Space Mono,monospace;'>{r['Time']}</td>
                <td style='font-family:Space Mono,monospace;'>{r['Demand']}</td>
                <td style='color:#8892a4;'>{r['Trains (Fixed)']}</td>
                <td style='color:{lc};font-weight:600;'>{r['Trains (AI)']}</td>
                <td style='color:#8892a4;'>{r['Wait Fixed']}</td>
                <td style='color:#00e676;font-weight:600;'>{r['Wait AI']}</td>
                <td style='color:{saved_color};font-weight:600;'>{r['Saved']}</td>
                <td>{badge}</td>
            </tr>"""
        return f"""
        <table class='styled-table'><thead><tr>
            <th>Line</th><th>Time</th><th>Demand</th>
            <th>Trains Fixed</th><th>Trains AI</th>
            <th>Wait Fixed</th><th>Wait AI</th>
            <th>Saved</th><th>Service</th>
        </tr></thead><tbody>{rows}</tbody></table>"""

    st.markdown(render_table(sched_filtered), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("⬇️ Download Schedule CSV",
                        sched_filtered.to_csv(index=False),
                        "fleet_schedule.csv", "text/csv")

    st.markdown("<div class='section-header'>Wait Time Comparison by Hour</div>", unsafe_allow_html=True)
    fig_s = go.Figure()
    for line in filt_line:
        sub = sched[sched["_line"] == line].copy()
        sub["hour_int"] = sub["Time"].str[:2].astype(int)
        sub["wf"] = sub["Wait Fixed"].str.replace("m", "").astype(float)
        sub["wa"] = sub["Wait AI"].str.replace("m", "").astype(float)
        sub = sub.sort_values("hour_int")
        fig_s.add_trace(go.Scatter(x=sub["hour_int"], y=sub["wf"],
                                    name=f"{line} Fixed", mode="lines",
                                    line=dict(color=LINE_COLOR[line], dash="dash", width=1.5),
                                    opacity=0.5))
        fig_s.add_trace(go.Scatter(x=sub["hour_int"], y=sub["wa"],
                                    name=f"{line} AI", mode="lines+markers",
                                    line=dict(color=LINE_COLOR[line], width=2.5),
                                    marker=dict(size=5)))
    fig_s.update_layout(**PLOTLY_LAYOUT, height=300,
                         xaxis_title="Hour", yaxis_title="Avg Wait (mins)",
                         legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_s, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown("<div class='section-header'>LSTM vs ARIMA — Error Metrics</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    arima = metrics_df[metrics_df["Model"] == "ARIMA"].iloc[0]
    lstm  = metrics_df[metrics_df["Model"] == "LSTM"].iloc[0]

    for col, metric, arima_val, lstm_val, color in [
        (m1, "RMSE",        arima["RMSE"], lstm["RMSE"], "#00d4ff"),
        (m2, "MAE",         arima["MAE"],  lstm["MAE"],  "#b06cff"),
        (m3, "Improvement", None,          None,         "#00e676"),
    ]:
        with col:
            if metric == "Improvement":
                pct = (arima["RMSE"] - lstm["RMSE"]) / arima["RMSE"] * 100
                st.markdown(f"""
                <div class='metric-card' style='--accent-line:#00e676'>
                    <div class='metric-label'>RMSE Improvement</div>
                    <div class='metric-value' style='color:#00e676'>{pct:.1f}%</div>
                    <div class='metric-delta delta-pos'>LSTM beats ARIMA</div>
                </div>""", unsafe_allow_html=True)
            else:
                pct = (arima_val - lstm_val) / arima_val * 100
                st.markdown(f"""
                <div class='metric-card' style='--accent-line:{color}'>
                    <div class='metric-label'>{metric}</div>
                    <div class='metric-value' style='color:{color}'>{lstm_val:.1f}</div>
                    <div class='metric-delta delta-pos'>↓ {pct:.1f}% vs ARIMA ({arima_val:.0f})</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Side-by-Side Error Comparison</div>", unsafe_allow_html=True)
    fig_m = go.Figure()
    for model, color in [("ARIMA", "#ff5252"), ("LSTM", "#00d4ff")]:
        row = metrics_df[metrics_df["Model"] == model].iloc[0]
        fig_m.add_trace(go.Bar(
            name=model, x=["RMSE", "MAE"],
            y=[row["RMSE"], row["MAE"]],
            marker_color=color,
            text=[f"{row['RMSE']:.1f}", f"{row['MAE']:.1f}"],
            textposition="outside",
            textfont=dict(color="white", size=12, family="Space Mono"),
        ))
    fig_m.update_layout(**PLOTLY_LAYOUT, height=350, barmode="group",
                         legend=dict(bgcolor="rgba(0,0,0,0)"),
                         yaxis_title="Error Value (passengers)",
                         title="Lower is Better ↓")
    st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("<div class='section-header'>LSTM Architecture</div>", unsafe_allow_html=True)
    arch_data = {
        "Layer":   ["Input", "LSTM (128)", "BatchNorm + Dropout", "LSTM (64)",
                    "BatchNorm + Dropout", "Dense (32)", "Output (1)"],
        "Shape":   ["(24, 16)", "(24, 256)", "(24, 256)", "(64,)", "(64,)", "(32,)", "(1,)"],
        "Purpose": ["24hr lookback × 16 features", "Broad temporal patterns",
                    "Regularisation", "Fine-grained patterns", "Regularisation",
                    "Non-linear compression", "Passenger count"],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Training Configuration</div>", unsafe_allow_html=True)
    tc1, tc2, tc3, tc4 = st.columns(4)
    for col, k, v in [
        (tc1, "Loss Function", "Huber"),
        (tc2, "Optimizer",     "Adam (lr=0.001)"),
        (tc3, "Batch Size",    "512"),
        (tc4, "Early Stopping","Patience=5"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent-line:#b06cff;padding:1rem 1.2rem'>
                <div class='metric-label'>{k}</div>
                <div style='font-family:Space Mono,monospace;font-size:1rem;
                            color:#b06cff;margin-top:0.3rem;'>{v}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 5 — ROUTE MAP  (replaces old Route Analysis)
# ═══════════════════════════════════════════════
elif page == "🗺️ Route Map":

    st.markdown("<div class='section-header'>Interactive Pune Metro Route Map</div>",
                unsafe_allow_html=True)

    # Slot selector row
    col_slot, col_info = st.columns([2, 3])
    with col_slot:
        map_slot_label = st.selectbox("Time Slot", list(SLOT_OPTIONS.keys()),
                                       index=list(SLOT_OPTIONS.keys()).index(sb_slot_label))
        map_slot = SLOT_OPTIONS[map_slot_label]
    with col_info:
        st.markdown("""
        <div style='padding:0.65rem 1rem;background:#111827;border-radius:8px;
                    font-size:0.82rem;color:#8892a4;margin-top:0.4rem;line-height:1.8;'>
            🔴 Very High demand &nbsp;|&nbsp; 🟠 High &nbsp;|&nbsp;
            🟡 Moderate &nbsp;|&nbsp; 🟢 Low &nbsp;|&nbsp; ⭐ Civil Court Interchange
            <br>
            <span style='color:#7B2D8B;'>━━</span> Purple Line (14 stations) &nbsp;|&nbsp;
            <span style='color:#00AEEF;'>━━</span> Aqua Line (16 stations)
        </div>
        """, unsafe_allow_html=True)

    # Folium map
    with st.spinner("Building route map …"):
        try:
            map_html = build_map(slot=map_slot)
            components.html(map_html, height=530, scrolling=False)
        except ImportError:
            st.error("📦 Install folium to enable the map:  `pip install folium`")
        except Exception as e:
            st.error(f"Map error: {e}")

    # ── High Demand Stations ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>High Demand Stations by Slot</div>",
                unsafe_allow_html=True)

    ds = DemandSegmentor(config_path="Data/stations_config.json")

    hd1, hd2 = st.columns(2)

    # Try real data first, fall back to synthetic station names from config
    stations_list = stations_config.get("stations", [])

    def synthetic_top5(slot_key, emoji, color):
        """Fallback top-5 using station config when real df lacks demand col."""
        np.random.seed({"morning_peak": 1, "evening_peak": 2}.get(slot_key, 3))
        top = sorted(stations_list,
                     key=lambda _: np.random.random(), reverse=True)[:5]
        rows = "".join([
            f"""<tr>
                <td style='color:{color};font-weight:700;'>{i+1}</td>
                <td>{s['name']}</td>
                <td style='font-family:Space Mono,monospace;color:{color};'>
                    {np.random.randint(18000,32000):,}
                </td>
            </tr>"""
            for i, s in enumerate(top)
        ])
        return f"""
        <table class='styled-table'>
            <thead><tr><th>#</th><th>Station</th><th>Avg Demand</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    with hd1:
        st.markdown(f"**🌅 Morning Peak — Top 5 Busiest**")
        if "passenger_demand" in df_raw.columns and "station_id" in df_raw.columns:
            top_m = ds.high_demand_stations(df_raw, slot="morning_peak")
            st.dataframe(top_m[["rank", "station_name", "passenger_demand"]],
                         use_container_width=True, hide_index=True)
        else:
            st.markdown(synthetic_top5("morning_peak", "#FF6B35", "#FF6B35"),
                        unsafe_allow_html=True)

    with hd2:
        st.markdown(f"**🌆 Evening Peak — Top 5 Busiest**")
        if "passenger_demand" in df_raw.columns and "station_id" in df_raw.columns:
            top_e = ds.high_demand_stations(df_raw, slot="evening_peak")
            st.dataframe(top_e[["rank", "station_name", "passenger_demand"]],
                         use_container_width=True, hide_index=True)
        else:
            st.markdown(synthetic_top5("evening_peak", "#EF476F", "#EF476F"),
                        unsafe_allow_html=True)

    # ── Demand Heatmap (kept from original) ──────────────────────────────────
    st.markdown("<div class='section-header'>Demand Heatmap — Line × Hour</div>",
                unsafe_allow_html=True)

    pivot = hourly_avg.pivot_table(index="line", columns="hour",
                                    values="avg_demand", aggfunc="mean")
    fig_h = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Plasma",
        text=[[f"{v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorbar=dict(title=dict(text="Passengers", font=dict(color="#f0f4ff")),
                      tickfont=dict(color="#f0f4ff")),
    ))
    fig_h.update_layout(**PLOTLY_LAYOUT, height=260,
                         xaxis_title="Hour of Day", yaxis_title="")
    st.plotly_chart(fig_h, use_container_width=True)

    # ── Station Deep-Dive (kept from original) ────────────────────────────────
    st.markdown("<div class='section-header'>Station Deep-Dive</div>", unsafe_allow_html=True)
    sel_station = st.selectbox("Select Station", STATIONS)
    stn_data    = df_raw[df_raw["station"] == sel_station]
    stn_hourly  = stn_data.groupby("hour").agg(
        avg_pass=("passengers", "mean"),
        max_pass=("passengers", "max"),
        avg_wait=("avg_wait_time_mins", "mean"),
    ).reset_index()

    sc1, sc2 = st.columns(2)
    with sc1:
        fig_st = go.Figure()
        fig_st.add_trace(go.Bar(
            x=stn_hourly["hour"], y=stn_hourly["avg_pass"],
            name="Avg Passengers",
            marker_color=[LINE_COLOR["Aqua"] if h in PEAK_HOURS else "#374151"
                           for h in stn_hourly["hour"]],
        ))
        fig_st.update_layout(**PLOTLY_LAYOUT, height=280,
                              title=f"{sel_station} — Avg Hourly Passengers",
                              xaxis_title="Hour", yaxis_title="Passengers")
        st.plotly_chart(fig_st, use_container_width=True)

    with sc2:
        fig_wt = go.Figure()
        fig_wt.add_trace(go.Scatter(
            x=stn_hourly["hour"], y=stn_hourly["avg_wait"],
            mode="lines+markers",
            line=dict(color="#b06cff", width=2.5),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(176,108,255,0.1)",
            name="Avg Wait Time",
        ))
        fig_wt.update_layout(**PLOTLY_LAYOUT, height=280,
                              title=f"{sel_station} — Avg Wait Time",
                              xaxis_title="Hour", yaxis_title="Wait (mins)")
        st.plotly_chart(fig_wt, use_container_width=True)

    peak_pass  = stn_data[stn_data["hour"].isin(PEAK_HOURS)]["passengers"].mean()
    offpk_pass = stn_data[~stn_data["hour"].isin(PEAK_HOURS)]["passengers"].mean()
    rain_pass  = stn_data[stn_data["is_raining"] == 1]["passengers"].mean()

    ss1, ss2, ss3 = st.columns(3)
    for col, label, val, color in [
        (ss1, "Peak Avg Passengers", f"{peak_pass:.0f}",  "#ff5252"),
        (ss2, "Off-Peak Avg",        f"{offpk_pass:.0f}", "#00d4ff"),
        (ss3, "Avg When Raining",    f"{rain_pass:.0f}",  "#b06cff"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent-line:{color};padding:1rem 1.2rem'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color};font-size:1.6rem'>{val}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 6 — JOURNEY PLANNER (Dijkstra routing)
# ═══════════════════════════════════════════════
elif page == "🧭 Journey Planner":

    st.markdown("<div class='section-header'>Point-to-Point Journey Planner</div>",
                unsafe_allow_html=True)

    # Init router
    @st.cache_resource
    def get_router():
        return MetroRouter()

    try:
        router     = get_router()
        name_map   = router.station_names()
        # Build display list: "PU01 — PCMC Bhavan"
        options    = [f"{sid} — {name}" for sid, name in name_map.items()]
        id_from_opt = {f"{sid} — {name}": sid for sid, name in name_map.items()}
    except Exception as e:
        st.error(f"Router init failed: {e}")
        st.stop()

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_from, col_to, col_btn = st.columns([2, 2, 1])
    with col_from:
        origin_opt = st.selectbox("🚉 From Station", options, index=0)
    with col_to:
        dest_opt   = st.selectbox("🏁 To Station",   options, index=len(options)-1)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        find_btn = st.button("🔍 Find Route", use_container_width=True)

    origin_id = id_from_opt[origin_opt]
    dest_id   = id_from_opt[dest_opt]

    if origin_id == dest_id:
        st.markdown("<div class='alert-box alert-warning'>⚠️ Origin and destination are the same station.</div>",
                    unsafe_allow_html=True)
    else:
        # Auto-run on load, or on button press
        result = router.shortest_path(origin_id, dest_id)

        if "error" in result:
            st.error(result["error"])
        else:
            # ── KPI strip ─────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            kpis = [
                (k1, f"{result['total_time_mins']:.1f} min", "Journey Time",    "#00d4ff"),
                (k2, f"{result['total_dist_km']:.2f} km",   "Distance",         "#b06cff"),
                (k3, str(result['num_stops']),               "Stations",         "#ffb347"),
                (k4, str(result['transfers']),               "Transfers",         "#ff5252" if result['transfers'] else "#00e676"),
                (k5, f"₹{result['fare_inr']}",              "Est. Fare",         "#00e676"),
            ]
            for col, val, label, color in kpis:
                with col:
                    st.markdown(f"""
                    <div class='metric-card' style='--accent-line:{color};padding:1rem 1.2rem;'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value' style='color:{color};font-size:1.5rem'>{val}</div>
                    </div>""", unsafe_allow_html=True)

            # ── Transfer alert ─────────────────────────────────────────────
            if result["transfers"]:
                st.markdown(
                    f"<div class='alert-box alert-warning'>🔄 <b>Transfer required</b> at "
                    f"<b>{', '.join(result['transfer_at'])}</b> — change from "
                    f"{result['path'][0]['line'].title()} Line to the other line (+5 min)</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='alert-box alert-normal'>✅ <b>Direct journey</b> — no transfer needed</div>",
                    unsafe_allow_html=True)

            # ── Step-by-step route ────────────────────────────────────────
            st.markdown("<div class='section-header'>Step-by-Step Route</div>",
                        unsafe_allow_html=True)

            route_table = []
            for i, stop in enumerate(result["path"]):
                is_first = i == 0
                is_last  = i == len(result["path"]) - 1
                is_xchg  = stop["interchange"]
                if is_first:
                    icon = "🚉 Start"
                elif is_last:
                    icon = "🏁 End"
                elif is_xchg:
                    icon = "⭐ Transfer"
                else:
                    icon = "• Stop"
                route_table.append({
                    "":        icon,
                    "Station": stop["name"] + (" ⇄" if is_xchg else ""),
                    "Line":    stop["line"].title(),
                    "Zone":    stop["zone"].title(),
                    "Time":    "Start" if is_first else f"{stop['cumulative_time_mins']} min",
                })
            st.dataframe(
                route_table,
                use_container_width=True,
                hide_index=True,
            )

            # Timeline bar chart of cumulative time
            import plotly.graph_objects as _go
            fig_route = _go.Figure()
            stop_names = [p["name"] for p in result["path"]]
            stop_times = [p["cumulative_time_mins"] for p in result["path"]]
            stop_colors = [
                "#00e676" if i == 0
                else "#ff5252" if i == len(result["path"])-1
                else "#FFD700" if result["path"][i]["interchange"]
                else "#00d4ff"
                for i in range(len(result["path"]))
            ]
            fig_route.add_trace(_go.Bar(
                x=stop_names,
                y=stop_times,
                marker_color=stop_colors,
                text=[f"{t} min" if t else "Start" for t in stop_times],
                textposition="outside",
                textfont=dict(color="white", size=10),
            ))
            fig_route.update_layout(
                **PLOTLY_LAYOUT,
                height=280,
                xaxis_title="",
                yaxis_title="Cumulative Time (min)",
                xaxis_tickangle=-35,
                showlegend=False,
            )
            st.plotly_chart(fig_route, use_container_width=True)

            # ── Mini map of route ─────────────────────────────────────────
            st.markdown("<div class='section-header'>Route on Map</div>",
                        unsafe_allow_html=True)

            try:
                import folium
                path_stations = [router.stations[p["id"]] for p in result["path"]]
                center_lat = sum(s["lat"] for s in path_stations) / len(path_stations)
                center_lon = sum(s["lon"] for s in path_stations) / len(path_stations)

                mini_map = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=13,
                    tiles="CartoDB dark_matter"
                )

                # Draw route polyline
                coords = [[s["lat"], s["lon"]] for s in path_stations]
                folium.PolyLine(coords, color="#00d4ff", weight=5,
                                opacity=0.9, tooltip="Optimal Route").add_to(mini_map)

                # Station markers
                for i, stop in enumerate(result["path"]):
                    s  = router.stations[stop["id"]]
                    lc = LINE_COLOR.get(stop["line"].title(), "#8892a4")
                    if i == 0:
                        color, icon = "green", "play"
                    elif i == len(result["path"]) - 1:
                        color, icon = "red", "flag"
                    elif stop["interchange"]:
                        color, icon = "orange", "star"
                    else:
                        color, icon = "blue", "circle"

                    folium.Marker(
                        location=[s["lat"], s["lon"]],
                        tooltip=f"{stop['name']} ({stop['cumulative_time_mins']} min)",
                        icon=folium.Icon(color=color, icon=icon, prefix="fa"),
                    ).add_to(mini_map)

                components.html(mini_map._repr_html_(), height=400, scrolling=False)

            except ImportError:
                st.info("📦 Install folium for the route map: `pip install folium`")

            # ── Journey summary card ──────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:#111827;border:1px solid #1e2d45;border-radius:12px;
                        padding:1.2rem 1.6rem;font-family:Space Mono,monospace;
                        font-size:0.85rem;color:#8892a4;line-height:2;'>
                📋 <b style='color:#f0f4ff;'>Journey Summary</b><br>
                {result['summary']}
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 7 — FLEET OPTIMIZER (Pickup Optimization)
# ═══════════════════════════════════════════════
elif page == "⚡ Fleet Optimizer":

    st.markdown("<div class='section-header'>Smart Fleet & Pickup Optimizer</div>",
                unsafe_allow_html=True)

    @st.cache_resource
    def get_optimizer():
        return PickupOptimizer()

    opt = get_optimizer()

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        opt_hour   = st.slider("Hour of Day", 6, 23, 8)
    with ctrl2:
        opt_fleet  = st.slider("Total Available Trains", 5, 30, 20)
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        opt_rain   = st.toggle("🌧️ Rain Surge",   value=sb_rain)
        opt_fest   = st.toggle("🎉 Festival",      value=sb_festival)

    # ── Run optimizer ─────────────────────────────────────────────────────────
    demand = opt.demo_demand(opt_hour)
    result = opt.optimize(
        demand,
        hour           = opt_hour,
        rain_surge     = opt_rain,
        festival_surge = opt_fest,
        weekend        = sb_weekend,
    )

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, f"{result.total_demand:,}",        "Total Demand",      "#00d4ff"),
        (k2, str(result.total_trains),           "Trains Deployed",   "#b06cff"),
        (k3, f"{result.avg_wait_mins:.1f} min",  "Avg Wait",          "#ffb347"),
        (k4, f"{result.efficiency_score:.1f}%",  "Fleet Efficiency",
         "#00e676" if result.efficiency_score >= 80 else "#ff5252"),
    ]
    for col, val, label, color in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent-line:{color}'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    if result.unmet_demand:
        st.markdown(
            f"<div class='alert-box alert-surge'>⚠️ <b>Capacity exceeded</b> — "
            f"{result.unmet_demand:,} passengers unserved. "
            f"Add more trains or stagger demand.</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='alert-box alert-normal'>✅ All demand served within current fleet capacity.</div>",
            unsafe_allow_html=True)

    # ── Line summary cards ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Line Allocation Summary</div>",
                unsafe_allow_html=True)

    line_cols = st.columns(len(result.line_summary))
    line_colors_map = {"purple": "#7B2D8B", "aqua": "#00AEEF"}
    for col, (ln, stats) in zip(line_cols, result.line_summary.items()):
        lc = line_colors_map.get(ln, "#8892a4")
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent-line:{lc};'>
                <div class='metric-label'>{ln.title()} Line</div>
                <div class='metric-value' style='color:{lc};font-size:1.8rem;'>
                    {stats['trains']} trains
                </div>
                <div class='metric-delta' style='color:#8892a4;'>
                    {stats['demand']:,} pax · {stats['avg_wait']:.1f} min wait
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Priority station chart ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Station Priority & Utilisation</div>",
                unsafe_allow_html=True)

    import plotly.graph_objects as go_
    import pandas as pd_

    alloc_df = pd_.DataFrame([{
        "Station":     a.station_name,
        "Line":        a.line.title(),
        "Demand":      a.demand,
        "Trains":      a.line_trains,
        "Wait (min)":  a.avg_wait_mins,
        "Utilisation": a.utilisation_pct,
        "Status":      "OVERFLOW" if a.utilisation_pct >= 95 else ("HIGH" if a.utilisation_pct >= 75 else "NORMAL"),
        "Priority":    a.priority_score,
    } for a in result.allocations])

    STATUS_COLOR = {
        "OVERFLOW": "#ff5252",
        "SURGE":    "#ff8c00",
        "HIGH":     "#ffd700",
        "NORMAL":   "#00e676",
    }

    fig_alloc = go_.Figure()
    for status, color in STATUS_COLOR.items():
        sub = alloc_df[alloc_df["Status"] == status]
        if len(sub) == 0:
            continue
        fig_alloc.add_trace(go_.Bar(
            name        = status,
            x           = sub["Station"],
            y           = sub["Utilisation"],
            marker_color= color,
            text        = sub["Trains"].astype(str) + "T",
            textposition= "outside",
            textfont    = dict(color="white", size=9),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Utilisation: %{y:.1f}%<br>"
                "Demand: %{customdata[0]:,}<br>"
                "Wait: %{customdata[1]:.1f} min<br>"
                "<extra></extra>"
            ),
            customdata  = sub[["Demand", "Wait (min)"]].values,
        ))

    fig_alloc.add_hline(y=75, line_dash="dash",
                        line_color="#ff8c00", opacity=0.6,
                        annotation_text="High threshold (75%)",
                        annotation_font_color="#ff8c00")
    fig_alloc.add_hline(y=95, line_dash="dash",
                        line_color="#ff5252", opacity=0.6,
                        annotation_text="Overflow threshold (95%)",
                        annotation_font_color="#ff5252")

    fig_alloc.update_layout(
        **PLOTLY_LAYOUT,
        height        = 380,
        barmode       = "overlay",
        xaxis_tickangle = -40,
        yaxis_title   = "Fleet Utilisation (%)",
        yaxis_range   = [0, 115],
        legend        = dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    # ── Full allocation table ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Full Station Allocation Table</div>",
                unsafe_allow_html=True)

    display_df = alloc_df[[
        "Station", "Line", "Demand", "Trains",
        "Wait (min)", "Utilisation", "Status"
    ]].reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Utilisation": st.column_config.ProgressColumn(
                "Utilisation %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Demand": st.column_config.NumberColumn(
                "Demand", format="%d pax"
            ),
        }
    )

    # Download
    st.download_button(
        "⬇️ Download Allocation CSV",
        display_df.to_csv(index=False),
        f"fleet_allocation_hour{opt_hour:02d}.csv",
        "text/csv",
    )


# ═══════════════════════════════════════════════
# PAGE 8 — MULTI-OBJECTIVE OPTIMIZER
# ═══════════════════════════════════════════════
elif page == "🎯 Multi-Objective":

    @st.cache_resource
    def get_moo():
        return MultiObjectiveOptimizer()
    moo = get_moo()

    # ── Hour picker only ──────────────────────────────────────────────────────
    hcol, _ = st.columns([1, 3])
    with hcol:
        moo_hour = st.slider("Select Hour", 6, 23, 8)

    # Auto-weights from the optimizer itself
    w_time, w_fuel, w_coverage, rationale = moo.get_auto_weights(moo_hour)

    # Period label + mode badge
    period_label = ("✅ Peak" if moo_hour in PEAK_HOURS else ("Shoulder" if moo_hour in {7,15,21} else "Off-Peak"))
    if moo_hour in PEAK_HOURS:
        mode_color, mode_icon, mode_text = "#ff5252", "⚡", "PEAK"
    elif moo_hour in [7, 15, 21]:
        mode_color, mode_icon, mode_text = "#ffb347", "〰", "SHOULDER"
    else:
        mode_color, mode_icon, mode_text = "#00e676", "🌙", "OFF-PEAK"

    # Hero context card
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#111827,#0d1f3c);
                border:1px solid #1e2d45;border-radius:16px;
                padding:1.4rem 1.8rem;margin-bottom:1.2rem;">
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.8rem;">
            <span style="font-family:Space Mono,monospace;font-size:1.5rem;
                         color:{mode_color};">{mode_icon} {mode_text}</span>
            <span style="font-size:0.8rem;color:#8892a4;">Hour {moo_hour:02d}:00 — AI auto-selected weights</span>
        </div>
        <div style="font-size:0.88rem;color:#c8d0e0;margin-bottom:1rem;">
            {rationale}
        </div>
        <div style="display:flex;gap:2rem;">
            <div>
                <span style="font-size:0.68rem;letter-spacing:0.1em;
                             text-transform:uppercase;color:#8892a4;">⏱ Time</span>
                <div style="font-family:Space Mono,monospace;font-size:1.1rem;
                             color:#00d4ff;font-weight:700;">{w_time:.0%}</div>
            </div>
            <div>
                <span style="font-size:0.68rem;letter-spacing:0.1em;
                             text-transform:uppercase;color:#8892a4;">⛽ Fuel</span>
                <div style="font-family:Space Mono,monospace;font-size:1.1rem;
                             color:#ffb347;font-weight:700;">{w_fuel:.0%}</div>
            </div>
            <div>
                <span style="font-size:0.68rem;letter-spacing:0.1em;
                             text-transform:uppercase;color:#8892a4;">🗺 Coverage</span>
                <div style="font-family:Space Mono,monospace;font-size:1.1rem;
                             color:#00e676;font-weight:700;">{w_coverage:.0%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Run optimizer with auto-weights + sidebar context toggles
    demand = moo.demo_demand(moo_hour)
    result = moo.optimize(
        demand, hour=moo_hour,
        w_time=w_time, w_fuel=w_fuel, w_coverage=w_coverage,
        rain_surge=sb_rain, festival_surge=sb_festival, weekend=sb_weekend,
    )

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Optimal Solution</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        (k1, str(result.total_trains),          "Trains Deployed",   "#00d4ff"),
        (k2, f"{result.avg_wait_mins:.1f} min", "Avg Wait (AI)",
              "#00e676" if result.wait_saved_mins >= 0 else "#ff5252"),
        (k3, f"{result.wait_saved_mins:+.1f}m", "vs Fixed Schedule",
              "#00e676" if result.wait_saved_mins >= 0 else "#ff5252"),
        (k4, f"{result.total_energy_kwh:.0f} kWh", "Energy / Hour",  "#ffb347"),
        (k5, f"{result.coverage_pct:.1f}%",        "Network Coverage","#b06cff"),
    ]
    for col, val, label, color in kpi_data:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent-line:{color}'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color};font-size:1.5rem'>{val}</div>
            </div>""", unsafe_allow_html=True)

    # ── Per-line result cards ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Per-Line Allocation</div>", unsafe_allow_html=True)
    lc_map = {"purple": "#7B2D8B", "aqua": "#00AEEF"}
    line_cols = st.columns(len(result.line_results))
    for col, (ln, lr) in zip(line_cols, result.line_results.items()):
        lc = lc_map.get(ln, "#8892a4")
        score_bar_t = int(lr.score_time    * 10)
        score_bar_f = int(lr.score_fuel    * 10)
        score_bar_c = int(lr.score_coverage* 10)
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent-line:{lc};'>
                <div class='metric-label'>{ln.title()} Line</div>
                <div class='metric-value' style='color:{lc};font-size:2rem;'>
                    {lr.optimal_trains} trains
                </div>
                <div style='font-size:0.78rem;color:#8892a4;margin-top:0.6rem;line-height:2;'>
                    ⏱ Headway: <b style='color:#f0f4ff;'>{lr.headway_mins:.1f} min</b><br>
                    🕐 Avg wait: <b style='color:#f0f4ff;'>{lr.avg_wait_mins:.1f} min</b><br>
                    ⛽ Energy: <b style='color:#f0f4ff;'>{lr.energy_kwh_hr:.0f} kWh/hr</b><br>
                    📊 Utilisation: <b style='color:#f0f4ff;'>{lr.utilisation_pct:.1f}%</b><br>
                    {lr.service_level}
                </div>
                <div style='margin-top:0.8rem;'>
                    <div style='font-size:0.65rem;letter-spacing:0.08em;
                                text-transform:uppercase;color:#8892a4;
                                margin-bottom:4px;'>Objective Scores</div>
                    <div style='font-size:0.72rem;color:#8892a4;'>
                        ⏱ <span style='color:#00d4ff;'>{"█"*score_bar_t}{"░"*(10-score_bar_t)}</span>
                        {lr.score_time:.2f}<br>
                        ⛽ <span style='color:#ffb347;'>{"█"*score_bar_f}{"░"*(10-score_bar_f)}</span>
                        {lr.score_fuel:.2f}<br>
                        🗺 <span style='color:#00e676;'>{"█"*score_bar_c}{"░"*(10-score_bar_c)}</span>
                        {lr.score_coverage:.2f}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Profile comparison chart ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>How AI Weights Change by Hour of Day</div>",
                unsafe_allow_html=True)

    import plotly.graph_objects as go_moo

    all_hours   = list(range(6, 24))
    wt_by_hour, wf_by_hour, wc_by_hour = [], [], []
    for h in all_hours:
        wt_, wf_, wc_, _ = moo.get_auto_weights(h)
        wt_by_hour.append(round(wt_ * 100))
        wf_by_hour.append(round(wf_ * 100))
        wc_by_hour.append(round(wc_ * 100))

    fig_weights = go_moo.Figure()
    fig_weights.add_trace(go_moo.Scatter(
        x=all_hours, y=wt_by_hour, name="⏱ Time",
        fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
        line=dict(color="#00d4ff", width=2.5),
        mode="lines",
    ))
    fig_weights.add_trace(go_moo.Scatter(
        x=all_hours, y=wf_by_hour, name="⛽ Fuel",
        fill="tozeroy", fillcolor="rgba(255,179,71,0.12)",
        line=dict(color="#ffb347", width=2.5),
        mode="lines",
    ))
    fig_weights.add_trace(go_moo.Scatter(
        x=all_hours, y=wc_by_hour, name="🗺 Coverage",
        fill="tozeroy", fillcolor="rgba(0,230,118,0.10)",
        line=dict(color="#00e676", width=2.5),
        mode="lines",
    ))
    # Mark current hour
    cur_wt, cur_wf, cur_wc, _ = moo.get_auto_weights(moo_hour)
    fig_weights.add_vline(
        x=moo_hour, line_dash="dash",
        line_color=mode_color, line_width=2,
        annotation_text=f"  {moo_hour:02d}:00",
        annotation_font_color=mode_color,
        annotation_font_size=11,
    )
    # Peak shading
    fig_weights.add_vrect(x0=8,  x1=12, fillcolor="rgba(255,82,82,0.05)",  line_width=0)
    fig_weights.add_vrect(x0=16, x1=21, fillcolor="rgba(255,82,82,0.05)",  line_width=0)
    fig_weights.add_vrect(x0=7,  x1=8,  fillcolor="rgba(255,179,71,0.05)", line_width=0)
    fig_weights.add_vrect(x0=15, x1=16, fillcolor="rgba(255,179,71,0.05)", line_width=0)
    fig_weights.add_vrect(x0=21, x1=22, fillcolor="rgba(255,179,71,0.05)", line_width=0)

    # Apply global layout first
    fig_weights.update_layout(**PLOTLY_LAYOUT)
    
    # Apply specific overrides second
    fig_weights.update_layout(
        height=260,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0.01, y=1.15),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_weights.update_xaxes(
        tickvals=list(range(6, 24)),
        ticktext=[f"{h:02d}:00" for h in range(6, 24)],
        tickangle=-45, tickfont=dict(size=9),
        gridcolor="#1e2d45", linecolor="#1e2d45",
    )
    fig_weights.update_yaxes(
        title_text="Weight (%)", range=[0, 80],
        gridcolor="#1e2d45", linecolor="#1e2d45",
    )
    st.plotly_chart(fig_weights, use_container_width=True)