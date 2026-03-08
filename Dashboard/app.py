"""
app.py
SmartTransit AI – Pune Metro Operations Control Center
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
import math
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path setup ──
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
    page_title="SmartTransit AI · Pune Metro OCC",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME & CSS — Control Center Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

:root {
    --bg-primary:    #060b14;
    --bg-card:       #0d1420;
    --bg-card2:      #111e30;
    --bg-card3:      #0a1628;
    --accent-aqua:   #00d4ff;
    --accent-purple: #b06cff;
    --accent-amber:  #ffb347;
    --accent-green:  #00e676;
    --accent-red:    #ff4444;
    --accent-orange: #ff7c3a;
    --text-primary:  #e8f0ff;
    --text-muted:    #6b7a99;
    --border:        #1a2540;
    --border-bright: #243355;
    --glow-aqua:     0 0 20px rgba(0,212,255,0.3);
    --glow-purple:   0 0 20px rgba(176,108,255,0.3);
    --glow-green:    0 0 20px rgba(0,230,118,0.3);
    --glow-red:      0 0 20px rgba(255,68,68,0.3);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
}

/* Scanline overlay for CRT effect */
body::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0.8rem 1.5rem 2rem 1.5rem !important; max-width: 1500px; }

/* ── Glowing KPI Cards ── */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card3) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-bright);
    box-shadow: var(--glow-aqua);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-line, var(--accent-aqua));
    box-shadow: 0 0 8px var(--accent-line, var(--accent-aqua));
}
.metric-card::after {
    content: '';
    position: absolute;
    top: -60%; right: -20%;
    width: 140%; height: 140%;
    background: radial-gradient(ellipse at top right, rgba(0,212,255,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0.3rem 0;
    text-shadow: 0 0 12px currentColor;
}
.metric-label {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 600;
}
.metric-delta { font-size: 0.82rem; margin-top: 0.35rem; font-weight: 500; }
.delta-pos { color: var(--accent-green); }
.delta-neg { color: var(--accent-red); }
.delta-neu { color: var(--accent-amber); }

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.6rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-bright), transparent);
}

/* ── OCC Header Banner ── */
.occ-header {
    background: linear-gradient(135deg, #060f1e 0%, #0a0e1a 40%, #0d0820 100%);
    border: 1px solid var(--border-bright);
    border-radius: 16px;
    padding: 1.2rem 2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.occ-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-aqua), var(--accent-purple), var(--accent-aqua));
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
.occ-header::after {
    content: '';
    position: absolute;
    top: -80%; right: -5%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 65%);
    pointer-events: none;
}
.occ-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg, #00d4ff 0%, #b06cff 50%, #00d4ff 100%);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
}
.occ-subtitle {
    font-family: 'Space Mono', monospace;
    color: var(--text-muted);
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

/* ── Animated status dot ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 6px var(--accent-green);
    animation: pulse-dot 2s ease-in-out infinite;
    margin-right: 6px;
}
.status-dot.warn { background: var(--accent-amber); box-shadow: 0 0 6px var(--accent-amber); }
.status-dot.crit { background: var(--accent-red);   box-shadow: 0 0 6px var(--accent-red);   }

/* ── Alert boxes ── */
.alert-box {
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
    border-left: 3px solid;
    font-size: 0.88rem;
    position: relative;
    overflow: hidden;
}
.alert-box::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    animation: alert-pulse 2s ease-in-out infinite;
}
.alert-surge   { background: rgba(255,68,68,0.07);   border-color: var(--accent-red);    color: #ff8a80; }
.alert-normal  { background: rgba(0,230,118,0.07);   border-color: var(--accent-green);  color: #69f0ae; }
.alert-warning { background: rgba(255,179,71,0.07);  border-color: var(--accent-amber);  color: #ffd180; }
.alert-info    { background: rgba(0,212,255,0.07);   border-color: var(--accent-aqua);   color: #80d8ff; }

/* ── Smart Alert animated ── */
.smart-alert {
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
    border: 1px solid;
    font-size: 0.83rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    animation: slide-in 0.4s ease-out;
    position: relative;
    overflow: hidden;
}
.smart-alert::after {
    content: '';
    position: absolute;
    left: -100%;
    top: 0; bottom: 0;
    width: 60%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
    animation: sweep 3s ease-in-out infinite;
}
.sa-critical { background: rgba(255,68,68,0.1);   border-color: rgba(255,68,68,0.3);   color: #ff8080; }
.sa-warning  { background: rgba(255,179,71,0.1);  border-color: rgba(255,179,71,0.3);  color: #ffd080; }
.sa-info     { background: rgba(0,212,255,0.08);  border-color: rgba(0,212,255,0.25);  color: #80d4ff; }
.sa-success  { background: rgba(0,230,118,0.08);  border-color: rgba(0,230,118,0.25);  color: #80ffb4; }

/* ── Train bar ── */
.train-bar { display: flex; gap: 3px; margin-top: 0.5rem; flex-wrap: wrap; }
.train-unit {
    width: 26px; height: 13px;
    border-radius: 3px;
    font-size: 0.5rem;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    transition: box-shadow 0.3s;
}
.train-active   { background: var(--accent-aqua);   color: #000; box-shadow: 0 0 6px rgba(0,212,255,0.5); }
.train-extra    { background: var(--accent-green);  color: #000; box-shadow: 0 0 6px rgba(0,230,118,0.5); }
.train-inactive { background: var(--border);        color: var(--text-muted); }

/* ── Styled table ── */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.styled-table th {
    background: var(--bg-card2);
    color: var(--text-muted);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.65rem 0.9rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.styled-table td {
    padding: 0.6rem 0.9rem;
    border-bottom: 1px solid rgba(26,37,64,0.6);
    color: var(--text-primary);
}
.styled-table tr:hover td { background: rgba(17,30,48,0.8); }

.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-peak    { background: rgba(255,68,68,0.15);   color: #ff5252; }
.badge-normal  { background: rgba(0,230,118,0.15);   color: #00e676; }
.badge-high    { background: rgba(255,179,71,0.15);  color: #ffb347; }
.badge-low     { background: rgba(107,122,153,0.15); color: #8892a4; }

/* ── Segment cards ── */
.segment-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card3) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.segment-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.segment-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--seg-color, var(--accent-aqua));
    box-shadow: 0 0 8px var(--seg-color, var(--accent-aqua));
}
.segment-emoji { font-size: 1.5rem; display: block; margin-bottom: 0.25rem; }
.segment-label { font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); }
.segment-value { font-family: 'Orbitron', monospace; font-size: 1.3rem; font-weight: 700; margin: 0.15rem 0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d18 0%, #060b14 100%) !important;
    border-right: 1px solid var(--border-bright) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Widget overrides ── */
.stSlider > div > div > div { background: var(--accent-aqua) !important; }
.stSelectbox > div > div { background: var(--bg-card2) !important; border-color: var(--border) !important; }
div[data-testid="metric-container"] { background: transparent !important; }
.stToggle > label { color: var(--text-primary) !important; }

/* ── Animations ── */
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}
@keyframes alert-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
@keyframes slide-in {
    from { transform: translateX(-12px); opacity: 0; }
    to   { transform: translateX(0);     opacity: 1; }
}
@keyframes sweep {
    0%   { left: -100%; }
    100% { left: 200%; }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
@keyframes float-up {
    0%   { transform: translateY(0);   opacity: 1; }
    100% { transform: translateY(-8px); opacity: 0; }
}

/* ── Heatwave timeline ── */
.heatwave-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    position: relative;
}

/* ── Train animation map ── */
.train-map-wrap {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border-bright);
    box-shadow: 0 0 30px rgba(0,212,255,0.08);
}

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
    plot_bgcolor="rgba(10,16,32,0.7)",
    font=dict(color="#e8f0ff", family="DM Sans"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#1a2540", linecolor="#1a2540"),
    yaxis=dict(gridcolor="#1a2540", linecolor="#1a2540"),
)


# ─────────────────────────────────────────────
# LIVE CONTEXT — Weather + Time
# ─────────────────────────────────────────────
import urllib.request
from datetime import datetime
import pytz

OWM_API_KEY     = "239f95528a0f41b838601a7331841595"
PUNE_LAT, PUNE_LON = 18.5204, 73.8567

@st.cache_data(ttl=300)
def fetch_pune_weather(api_key):
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={PUNE_LAT}&lon={PUNE_LON}&appid={api_key}&units=metric")
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

pune_tz         = pytz.timezone("Asia/Kolkata")
now_pune        = datetime.now(pune_tz)
live_hour       = now_pune.hour
live_day        = now_pune.strftime("%A")
live_date       = now_pune.strftime("%d %b %Y")
live_time       = now_pune.strftime("%H:%M")
live_secs       = now_pune.strftime("%S")
is_weekend_live = now_pune.weekday() >= 5

weather_data    = fetch_pune_weather(OWM_API_KEY)
weather_ok      = "error" not in weather_data

if weather_ok:
    temp_c          = round(weather_data["main"]["temp"], 1)
    feels_like      = round(weather_data["main"]["feels_like"], 1)
    humidity        = weather_data["main"]["humidity"]
    wind_kph        = round(weather_data["wind"]["speed"] * 3.6, 1)
    weather_id      = weather_data["weather"][0]["id"]
    weather_desc    = weather_data["weather"][0]["description"].title()
    is_raining_live = (200 <= weather_id <= 531)
    if weather_id >= 800:
        wx_icon = "☀️" if weather_id == 800 else ("🌤️" if weather_id == 801 else "☁️")
    elif weather_id >= 700: wx_icon = "🌫️"
    elif weather_id >= 600: wx_icon = "❄️"
    elif weather_id >= 300: wx_icon = "🌧️"
    elif weather_id >= 200: wx_icon = "⛈️"
    else:                   wx_icon = "🌡️"
else:
    temp_c, feels_like, humidity, wind_kph = 28.0, 27.0, 65, 12.0
    is_raining_live  = False
    weather_desc, wx_icon = "Weather unavailable", "🌡️"

period_now   = "Peak" if live_hour in PEAK_HOURS else ("Shoulder" if live_hour in {7,15,21} else "Off-Peak")
period_color = "#ff4444" if period_now == "Peak" else ("#ffb347" if period_now == "Shoulder" else "#00e676")


# ─────────────────────────────────────────────
# HELPER — SMART ALERTS ENGINE
# ─────────────────────────────────────────────
def generate_smart_alerts(rain, festival, weekend, hour, utilisation=None):
    alerts = []
    if rain:
        alerts.append(("critical", "🌧️", "Rain Surge Active", "Demand ×1.22 — extra trains deploying on both lines"))
    if festival:
        alerts.append(("critical", "🎉", "Festival Demand Spike", "Demand ×1.55 — all reserve fleet activated"))
    if hour in PEAK_HOURS:
        alerts.append(("warning", "⚡", "Peak Hour Window", f"Hour {hour:02d}:00 — minimum 8 trains enforced"))
    if weekend:
        alerts.append(("info", "📅", "Weekend Pattern Active", "Demand ×1.10 — adjusted headway schedules in effect"))
    if utilisation and utilisation > 90:
        alerts.append(("critical", "⚠️", "Fleet Near Capacity", f"Utilisation at {utilisation:.0f}% — monitor for overflow"))
    elif utilisation and utilisation > 75:
        alerts.append(("warning", "📊", "High Utilisation", f"Utilisation at {utilisation:.0f}% — approaching peak threshold"))
    if not alerts:
        alerts.append(("success", "✅", "All Systems Normal", f"Network operating nominally at {hour:02d}:00 IST"))
    return alerts


def render_smart_alerts(alerts):
    cls_map = {"critical": "sa-critical", "warning": "sa-warning", "info": "sa-info", "success": "sa-success"}
    dot_map = {"critical": "crit", "warning": "warn", "info": "", "success": ""}
    html = ""
    for level, icon, title, desc in alerts:
        cls  = cls_map.get(level, "sa-info")
        dot_cls = "status-dot " + dot_map.get(level, "")
        html += (
            f"<div class='smart-alert {cls}'>"
            f"<span class='{dot_cls.strip()}'></span>"
            f"<span style='font-size:1.1rem;'>{icon}</span>"
            f"<div><b>{title}</b><br>"
            f"<span style='font-size:0.78rem;opacity:0.8;'>{desc}</span></div>"
            f"</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER — ANIMATED TRAIN MAP (Plotly Mapbox)
# ─────────────────────────────────────────────
def build_animated_train_map(stations_cfg, hour, rain_surge=False, weekend=False):
    """
    Builds a Plotly figure showing animated trains moving along metro lines.
    Uses station coordinates from stations_config.json.
    Simulates train positions based on headway and current time.
    """
    stations_list = stations_cfg.get("stations", [])
    if not stations_list:
        return None

    purple_stns = [s for s in stations_list if s.get("line") == "purple"]
    aqua_stns   = [s for s in stations_list if s.get("line") == "aqua"]
    # sort by sequence / order
    purple_stns = sorted(purple_stns, key=lambda s: s.get("sequence", s.get("order", 0)))
    aqua_stns   = sorted(aqua_stns,   key=lambda s: s.get("sequence", s.get("order", 0)))

    if not purple_stns or not aqua_stns:
        return None

    # ── Demand-based train count ──
    demand_base = 18000 if hour in PEAK_HOURS else 9000
    if rain_surge:  demand_base = int(demand_base * 1.22)
    if weekend:     demand_base = int(demand_base * 1.10)
    trains_purple, headway_purple, _, _ = demand_to_trains(demand_base, hour, rain_surge, False, weekend)
    trains_aqua,   headway_aqua,   _, _ = demand_to_trains(demand_base, hour, rain_surge, False, weekend)

    fig = go.Figure()

    def add_line_trace(stns, color, name, dash="solid", width=4):
        lats = [s["lat"] for s in stns]
        lons = [s["lon"] for s in stns]
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(width=width, color=color),
            name=name,
            hoverinfo="skip",
        ))

    # ── Draw metro lines ──
    add_line_trace(purple_stns, "#7B2D8B", "Purple Line", width=5)
    add_line_trace(aqua_stns,   "#00AEEF", "Aqua Line",   width=5)

    # ── Draw stations ──
    for stns, color, line_name in [(purple_stns, "#b06cff", "Purple"), (aqua_stns, "#00d4ff", "Aqua")]:
        for s in stns:
            is_interchange = s.get("interchange", False) or s.get("is_interchange", False)
            np.random.seed(abs(hash(s.get("name","x"))) % 9999)
            pax = np.random.randint(800, 3200) if hour in PEAK_HOURS else np.random.randint(200, 1200)
            fig.add_trace(go.Scattermapbox(
                lat=[s["lat"]], lon=[s["lon"]],
                mode="markers+text",
                marker=dict(
                    size=14 if is_interchange else 9,
                    color="#FFD700" if is_interchange else color,
                    opacity=0.95,
                    symbol="circle",
                ),
                text=[s.get("name", "")],
                textposition="top right",
                textfont=dict(size=9, color="#c8d4e8"),
                name=s.get("name", ""),
                hovertemplate=(
                    f"<b>{s.get('name','')}</b><br>"
                    f"Line: {line_name}<br>"
                    f"Pax onboard: ~{pax:,}<br>"
                    + ("<b>⭐ Interchange</b><br>" if is_interchange else "") +
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

    # ── Animate trains along lines ──
    def interpolate_position(stns, frac):
        """Get lat/lon at fractional distance along a polyline of stations."""
        if len(stns) < 2:
            return stns[0]["lat"], stns[0]["lon"]
        # total segments
        segs = len(stns) - 1
        pos  = frac * segs
        idx  = int(pos)
        idx  = min(idx, segs - 1)
        t    = pos - idx
        lat  = stns[idx]["lat"] + t * (stns[idx+1]["lat"] - stns[idx]["lat"])
        lon  = stns[idx]["lon"] + t * (stns[idx+1]["lon"] - stns[idx]["lon"])
        return lat, lon

    # Use current minute to determine train positions (so they appear to move on refresh)
    current_min = now_pune.minute + now_pune.second / 60.0

    for line_stns, n_trains, headway_min, lcolor, lname, occ_base in [
        (purple_stns, trains_purple, headway_purple, "#c084fc", "Purple", 0.72),
        (aqua_stns,   trains_aqua,   headway_aqua,   "#38bdf8", "Aqua",   0.68),
    ]:
        for i in range(min(n_trains, 8)):   # cap display at 8 per line
            # Each train is offset by headway_min minutes apart in the cycle
            cycle_mins = headway_min * len(line_stns)
            offset_min = i * headway_min
            train_min  = (current_min + offset_min) % (cycle_mins if cycle_mins > 0 else 60)
            frac       = (train_min / cycle_mins) if cycle_mins > 0 else (i / max(n_trains, 1))
            frac       = frac % 1.0

            # Direction: alternate trains go forward/backward
            if i % 2 == 1:
                frac = 1.0 - frac

            t_lat, t_lon = interpolate_position(line_stns, frac)

            # Synthetic passengers/occupancy
            np.random.seed(i * 31 + live_hour * 7)
            capacity   = EFFECTIVE_CAPACITY
            onboard    = int(capacity * occ_base * (0.8 + 0.4 * np.random.random()))
            if rain_surge:  onboard = min(int(onboard * 1.2), capacity)
            if weekend:     onboard = min(int(onboard * 1.1), capacity)
            occupancy  = round(onboard / capacity * 100, 1)
            occ_color  = "#ff4444" if occupancy > 85 else ("#ffb347" if occupancy > 65 else "#00e676")

            train_label = f"T{i+1:02d}"
            fig.add_trace(go.Scattermapbox(
                lat=[t_lat], lon=[t_lon],
                mode="markers+text",
                marker=dict(
                    size=18,
                    color=occ_color,
                    opacity=1.0,
                    symbol="circle",
                ),
                text=[f"🚇"],
                textposition="middle center",
                textfont=dict(size=11),
                name=f"{lname} Train {i+1}",
                hovertemplate=(
                    f"<b>🚇 {lname} Line — Train {train_label}</b><br>"
                    f"Passengers: {onboard:,} / {capacity:,}<br>"
                    f"Occupancy: <b>{occupancy}%</b><br>"
                    f"Direction: {'→ Northbound' if i%2==0 else '← Southbound'}<br>"
                    f"<extra></extra>"
                ),
                showlegend=True,
            ))

    center_lat = np.mean([s["lat"] for s in stations_list])
    center_lon = np.mean([s["lon"] for s in stations_list])

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        legend=dict(
            bgcolor="rgba(10,16,32,0.85)",
            bordercolor="#1a2540",
            borderwidth=1,
            font=dict(color="#e8f0ff", size=10),
            x=0.01, y=0.99,
        ),
        font=dict(color="#e8f0ff"),
    )
    return fig


# ─────────────────────────────────────────────
# HELPER — AI DEMAND HEATWAVE TIMELINE
# ─────────────────────────────────────────────
def build_demand_heatwave(hourly_avg, live_hour, rain=False, weekend=False):
    """6-hour forward demand prediction heatmap with surge detection."""
    lines_plot = ["Aqua", "Purple", "Interchange"]
    hours_fwd  = [(live_hour + i) % 24 for i in range(7)]
    hour_labels = [f"{h:02d}:00" for h in hours_fwd]

    z_vals = []
    surge_mask = []
    for line in lines_plot:
        row_z   = []
        row_s   = []
        sub = hourly_avg[hourly_avg["line"] == line]
        for h in hours_fwd:
            match = sub[sub["hour"] == h]
            base  = float(match["avg_demand"].values[0]) if len(match) else 8000.0
            # apply modifiers
            if rain:    base *= 1.22
            if weekend: base *= 1.10
            # add small noise for realism
            np.random.seed(h * 13 + abs(hash(line)) % 100)
            base *= (0.95 + 0.1 * np.random.random())
            row_z.append(round(base))
            # surge = >20% above historical mean for that hour
            hist_mean = float(sub["avg_demand"].mean()) if len(sub) else 8000
            row_s.append(base > hist_mean * 1.20)
        z_vals.append(row_z)
        surge_mask.append(row_s)

    # Surge annotations
    annotations = []
    for r, line in enumerate(lines_plot):
        for c, h in enumerate(hours_fwd):
            if surge_mask[r][c]:
                annotations.append(dict(
                    x=hour_labels[c], y=line,
                    text="▲SURGE",
                    showarrow=False,
                    font=dict(size=8, color="#ff4444", family="Space Mono"),
                    xanchor="center", yanchor="middle",
                ))

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=hour_labels,
        y=lines_plot,
        colorscale=[
            [0.0,  "#060b14"],
            [0.25, "#0d2040"],
            [0.5,  "#1a4080"],
            [0.7,  "#7B2D8B"],
            [0.85, "#ff6b35"],
            [1.0,  "#ff1744"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Pax / hr", font=dict(color="#e8f0ff", size=10)),
            tickfont=dict(color="#e8f0ff", size=9),
            thickness=12,
        ),
        text=[[f"{v:,.0f}" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        hovertemplate="<b>%{y}</b> at %{x}<br>Demand: <b>%{z:,.0f} pax</b><extra></extra>",
    ))

    # Highlight current hour
   # Highlight current hour
    fig.add_vline(
        x=hour_labels[0],
        line_dash="dash", line_color="#00d4ff", line_width=2,
    )
    annotations.append(dict(
        x=hour_labels[0], y=1.02, xref="x", yref="paper",
        text=" NOW", showarrow=False,
        font=dict(color="#00d4ff", size=10, family="Space Mono"),
        xanchor="left", yanchor="bottom"
    ))

    # 1. Apply global base layout first
    fig.update_layout(**PLOTLY_LAYOUT)
    
    # 2. Apply specific chart overrides
    fig.update_layout(
        height=220,
        annotations=annotations,
        xaxis=dict(gridcolor="#1a2540", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1a2540", tickfont=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    
    # Make sure this return statement is here!
    return fig, surge_mask, z_vals

# ─────────────────────────────────────────────
# HELPER — STATION CONGESTION RADAR
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# HELPER — OCC CONTROL CENTER HEADER
# ─────────────────────────────────────────────
def render_occ_header(live_hour, live_time, live_day, live_date,
                      period_now, period_color, total_trains,
                      system_demand, utilisation,
                      is_raining_live, is_weekend_live, weather_ok,
                      temp_c, feels_like, humidity, wind_kph, wx_icon, weather_desc):

    status_cls = "crit" if period_now == "Peak" else ("warn" if period_now == "Shoulder" else "")
    sys_status = "PEAK OPS" if period_now == "Peak" else ("SHOULDER" if period_now == "Shoulder" else "NOMINAL")

    rain_pill = (
        "<span style='background:rgba(255,68,68,0.15);border:1px solid rgba(255,68,68,0.3);"
        "color:#ff8080;border-radius:20px;padding:2px 10px;font-size:0.68rem;"
        "font-family:Space Mono,monospace;font-weight:600;margin-left:8px;'>🌧 RAIN SURGE</span>"
        if is_raining_live else ""
    )
    wknd_pill = (
        "<span style='background:rgba(255,179,71,0.12);border:1px solid rgba(255,179,71,0.3);"
        "color:#ffd080;border-radius:20px;padding:2px 10px;font-size:0.68rem;"
        "font-family:Space Mono,monospace;font-weight:600;margin-left:6px;'>📅 WEEKEND</span>"
        if is_weekend_live else ""
    )

    html = (
        "<div class='occ-header'>"

        # Row 1 — title + status
        "<div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.8rem;'>"
        "<div>"
        "<div class='occ-title'>🚇 METRO OPERATIONS CONTROL CENTER</div>"
        "<div class='occ-subtitle'>PUNE METRO RAIL · REAL-TIME FLEET ORCHESTRATION SYSTEM</div>"
        "</div>"
        "<div style='display:flex;align-items:center;gap:0.8rem;'>"
        f"<span class='status-dot {status_cls}'></span>"
        f"<span style='font-family:Orbitron,monospace;font-size:0.9rem;color:{period_color};font-weight:700;'>{sys_status}</span>"
        f"<span style='font-family:Space Mono,monospace;font-size:0.78rem;color:#6b7a99;'>|</span>"
        f"<span style='font-family:Orbitron,monospace;font-size:1.1rem;color:#e8f0ff;font-weight:700;'>{live_time}</span>"
        f"<span style='font-family:Space Mono,monospace;font-size:0.72rem;color:#6b7a99;'>IST</span>"
        "</div>"
        "</div>"

        # Row 2 — KPI strip + weather
        "<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;margin-top:1rem;'>"

        # KPIs
        "<div style='display:flex;gap:2rem;flex-wrap:wrap;'>"

        f"<div style='text-align:center;'>"
        f"<div style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:700;color:#00d4ff;"
        f"text-shadow:0 0 12px rgba(0,212,255,0.6);'>{system_demand:,}</div>"
        f"<div style='font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:#6b7a99;'>System Demand</div>"
        f"</div>"

        f"<div style='text-align:center;'>"
        f"<div style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:700;color:#b06cff;"
        f"text-shadow:0 0 12px rgba(176,108,255,0.6);'>{total_trains}</div>"
        f"<div style='font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:#6b7a99;'>Active Trains</div>"
        f"</div>"

        f"<div style='text-align:center;'>"
        f"<div style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:700;"
        f"color:{'#ff4444' if utilisation>85 else '#ffb347' if utilisation>65 else '#00e676'};"
        f"text-shadow:0 0 12px rgba(0,230,118,0.5);'>{utilisation:.0f}%</div>"
        f"<div style='font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:#6b7a99;'>Utilisation</div>"
        f"</div>"

        f"<div style='text-align:center;'>"
        f"<div style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:700;color:{period_color};"
        f"text-shadow:0 0 12px {period_color}80;'>{live_day[:3].upper()}</div>"
        f"<div style='font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:#6b7a99;'>{live_date}</div>"
        f"</div>"
        "</div>"

        # Pills + weather
        "<div style='display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;'>"
        f"<div>{rain_pill}{wknd_pill}</div>"
        "<div style='display:flex;align-items:center;gap:0.8rem;'>"
        f"<span style='font-size:1.6rem;'>{wx_icon}</span>"
        "<div>"
        f"<div style='font-family:Orbitron,monospace;font-size:1rem;color:#00d4ff;font-weight:700;'>{temp_c}°C</div>"
        f"<div style='font-size:0.66rem;color:#6b7a99;'>{weather_desc} · {humidity}% RH</div>"
        "</div>"
        "</div>"
        "</div>"

        "</div>"  # end KPI strip row
        "</div>"  # end occ-header
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:0.8rem 0 0.4rem 0;'>"
        "<div style='font-family:Orbitron,monospace;font-size:0.9rem;"
        "background:linear-gradient(90deg,#00d4ff,#b06cff);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-weight:700;letter-spacing:0.05em;'>🚇 SMARTTRANSIT AI</div>"
        "<div style='font-size:0.68rem;color:#6b7a99;margin-top:3px;letter-spacing:0.08em;'>"
        "PUNE METRO OCC · v3.0</div>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔮 Live Prediction", "🚆 Fleet Scheduler",
         "📊 Model Performance", "🗺️ Route Map", "🧭 Journey Planner",
         "⚡ Fleet Optimizer", "🎯 Multi-Objective"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    _rain_dot    = "🟠" if is_raining_live  else "🟢"
    _wknd_dot    = "🟠" if is_weekend_live  else "🟢"
    _weather_src = "Live API" if weather_ok else "Fallback"
    st.markdown(
        f"<div style='font-size:0.65rem;color:#6b7a99;letter-spacing:0.1em;"
        f"text-transform:uppercase;margin-bottom:6px;'>Auto-Detected Context</div>"
        f"<div style='font-size:0.76rem;line-height:2;'>"
        f"{_rain_dot} Rain: <b style='color:#e8f0ff;'>{'Yes' if is_raining_live else 'No'}</b>"
        f"<span style='color:#404e6a;font-size:0.66rem;'> ({_weather_src})</span><br>"
        f"{_wknd_dot} Weekend: <b style='color:#e8f0ff;'>{'Yes' if is_weekend_live else 'No'}</b>"
        f"<span style='color:#404e6a;font-size:0.66rem;'> ({live_day})</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='font-size:0.65rem;color:#6b7a99;letter-spacing:0.1em;"
        "text-transform:uppercase;margin-top:10px;'>Manual Overrides</div>",
        unsafe_allow_html=True
    )
    sb_rain     = st.toggle("🌧️ Rain Surge",   value=is_raining_live)
    sb_festival = st.toggle("🎉 Festival Day", value=False)
    sb_weekend  = st.toggle("📅 Weekend Mode", value=is_weekend_live)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.65rem;color:#6b7a99;letter-spacing:0.1em;"
        "text-transform:uppercase;'>Map Time Slot</div>",
        unsafe_allow_html=True
    )
    sb_slot_label = st.selectbox("Map Slot", list(SLOT_OPTIONS.keys()), label_visibility="collapsed")
    sb_slot = SLOT_OPTIONS[sb_slot_label]

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#6b7a99;text-align:center;'>"
        "LSTM RMSE <span style='color:#00d4ff;font-family:Orbitron,monospace;'>262.94</span><br>"
        "<span style='font-size:0.62rem;'>95.2% better than ARIMA</span>"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# COMPUTE SYSTEM-WIDE LIVE STATS (for OCC header)
# ─────────────────────────────────────────────
_sys_demand  = 0
_sys_trains  = 0
for _line in ["Aqua", "Purple"]:
    _sub = hourly_avg[hourly_avg["line"] == _line]
    _h_match = _sub[_sub["hour"] == live_hour]
    _base = float(_h_match["avg_demand"].values[0]) if len(_h_match) else 9000.0
    if sb_rain:    _base *= 1.22
    if sb_festival:_base *= 1.55
    if sb_weekend: _base *= 1.10
    _t, _, _, _ = demand_to_trains(int(_base), live_hour, sb_rain, sb_festival, sb_weekend)
    _sys_demand += int(_base)
    _sys_trains += _t

_sys_util = round(_sys_demand / (_sys_trains * EFFECTIVE_CAPACITY * 2) * 100, 1) if _sys_trains > 0 else 0


# ─────────────────────────────────────────────
# OCC HEADER — rendered on every page
# ─────────────────────────────────────────────
render_occ_header(
    live_hour, live_time, live_day, live_date,
    period_now, period_color,
    _sys_trains, _sys_demand, _sys_util,
    is_raining_live, is_weekend_live, weather_ok,
    temp_c, feels_like, humidity, wind_kph, wx_icon, weather_desc
)


# ══════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════
if page == "🏠 Overview":

    arima_rmse = metrics_df[metrics_df["Model"] == "ARIMA"]["RMSE"].values[0]
    lstm_rmse  = metrics_df[metrics_df["Model"] == "LSTM"]["RMSE"].values[0]

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "262.94",           "LSTM RMSE",        "−95.2% vs ARIMA",                              "pos", "#00d4ff"),
        (c2, "180,438",          "Training Records", "Jan–Dec 2024",                                 "neu", "#b06cff"),
        (c3, str(TOTAL_STATIONS),"Metro Stations",   f"Purple {PURPLE_COUNT} · Aqua {AQUA_COUNT}",   "neu", "#ffb347"),
        (c4, "~2.0 min",         "Peak Wait (AI)",   "vs 3.0 min fixed schedule",                    "pos", "#00e676"),
    ]
    for col, val, label, delta, dtype, color in cards:
        with col:
            st.markdown(
                f"<div class='metric-card' style='--accent-line:{color}'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='color:{color}'>{val}</div>"
                f"<div class='metric-delta delta-{dtype}'>{delta}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

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
            fillcolor={"Aqua":"rgba(0,212,255,0.08)","Purple":"rgba(176,108,255,0.08)",
                       "Interchange":"rgba(255,179,71,0.08)"}.get(line,"rgba(255,255,255,0.05)"),
        ))
    for start, end in [(8, 11), (16, 20)]:
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(255,68,68,0.05)", line_width=0,
                      annotation_text="Peak" if start==8 else "",
                      annotation_font_color="#ff4444", annotation_font_size=10)
    fig.add_vline(x=live_hour, line_dash="dot", line_color="#00d4ff", line_width=2,
                  annotation_text=f" {live_time}", annotation_font_color="#00d4ff", annotation_font_size=10)
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      xaxis_title="Hour of Day", yaxis_title="Avg Total Passengers")
    st.plotly_chart(fig, use_container_width=True)

    # ── AI Demand Heatwave ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>AI Demand Heatwave — Next 6 Hours</div>", unsafe_allow_html=True)
    fig_hw, surge_mask, z_vals = build_demand_heatwave(hourly_avg, live_hour, sb_rain, sb_weekend)
    st.plotly_chart(fig_hw, use_container_width=True)

    # Surge detector pills
    surge_lines = []
    for r, line in enumerate(["Aqua", "Purple", "Interchange"]):
        hours_fwd = [(live_hour + i) % 24 for i in range(7)]
        for c, h in enumerate(hours_fwd):
            if surge_mask[r][c]:
                surge_lines.append(f"<span style='background:rgba(255,68,68,0.12);border:1px solid rgba(255,68,68,0.3);color:#ff8080;border-radius:20px;padding:2px 10px;font-size:0.7rem;font-family:Space Mono,monospace;margin:2px;display:inline-block;'>⚡ {line} @ {h:02d}:00 — {z_vals[r][c]:,.0f} pax</span>")
    if surge_lines:
        st.markdown(
            "<div style='margin-top:6px;'><span style='font-size:0.68rem;color:#6b7a99;letter-spacing:0.1em;text-transform:uppercase;'>Surge Detector: </span>"
            + " ".join(surge_lines) + "</div>",
            unsafe_allow_html=True
        )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("<div class='section-header'>ARIMA vs LSTM Error</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        for model, color in [("ARIMA", "#ff4444"), ("LSTM", "#00d4ff")]:
            row = metrics_df[metrics_df["Model"] == model].iloc[0]
            fig2.add_trace(go.Bar(
                name=model, x=["RMSE", "MAE"],
                y=[row["RMSE"], row["MAE"]],
                marker_color=color,
                marker=dict(opacity=0.85),
                text=[f"{row['RMSE']:.0f}", f"{row['MAE']:.0f}"],
                textposition="outside",
                textfont=dict(color="white", size=11),
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=260, barmode="group",
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           yaxis_title="Error (passengers)")
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Fleet Optimisation Savings</div>", unsafe_allow_html=True)
        if schedule_df is not None:
            if "period" in schedule_df.columns:
                _period_col = schedule_df["period"]
            elif "peak_hour" in schedule_df.columns:
                _period_col = schedule_df["peak_hour"].astype(str)
            else:
                _period_col = pd.Series([""] * len(schedule_df))
            peak_sched = schedule_df[_period_col.str.contains("Peak", na=False)]
            if len(peak_sched) == 0:
                peak_sched = schedule_df
            fig3 = go.Figure()
            for line in LINES:
                sub = peak_sched[peak_sched["line"] == line]
                if len(sub):
                    wf = sub["wait_fixed_mins"].mean() if "wait_fixed_mins" in sub else 3.0
                    wa = sub["wait_ai_mins"].mean()    if "wait_ai_mins"    in sub else 2.0
                    fig3.add_trace(go.Bar(name=f"{line} Fixed", x=[line], y=[wf], marker_color="#374151"))
                    fig3.add_trace(go.Bar(name=f"{line} AI",    x=[line], y=[wa], marker_color=LINE_COLOR[line]))
            fig3.update_layout(**PLOTLY_LAYOUT, height=260, barmode="group",
                               yaxis_title="Avg Wait (mins)",
                               legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig3, use_container_width=True)

    # ── Demand Segments ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Demand by Time Slot — Today's Overview</div>", unsafe_allow_html=True)
    ds = DemandSegmentor(config_path="Data/stations_config.json")
    seg_report = ds.segment_historical(df_raw) if "passenger_demand" in df_raw.columns else {
        "morning_peak": {"emoji":"🌅","label":"Morning Peak",  "color":"#FF6B35","mean_demand":23400,"max_demand":31200},
        "afternoon":    {"emoji":"☀️", "label":"Afternoon",     "color":"#FFD166","mean_demand":11200,"max_demand":15800},
        "evening_peak": {"emoji":"🌆","label":"Evening Peak",  "color":"#EF476F","mean_demand":22100,"max_demand":29800},
        "night":        {"emoji":"🌙","label":"Night Off-Peak","color":"#118AB2","mean_demand": 4200,"max_demand": 7100},
        "weekend":      {"emoji":"📅","label":"Weekend",       "color":"#06D6A0","mean_demand":13500,"max_demand":19200},
    }
    seg_cols = st.columns(5)
    for col, (key, stats) in zip(seg_cols, seg_report.items()):
        with col:
            _sc = stats["color"]
            _se = stats["emoji"]
            _sl = stats["label"]
            _sv = int(stats["mean_demand"])
            st.markdown(
                f"<div class='segment-card' style='--seg-color:{_sc}'>"
                f"<span class='segment-emoji'>{_se}</span>"
                f"<div class='segment-label'>{_sl}</div>"
                f"<div class='segment-value' style='color:{_sc};'>{_sv:,}</div>"
                f"<div style='font-size:0.68rem;color:#6b7a99;margin-top:2px;'>avg pax/hr</div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════
# PAGE 2 — LIVE PREDICTION
# ══════════════════════════════════════════════════
elif page == "🔮 Live Prediction":

    period_now_lp   = "Peak" if live_hour in PEAK_HOURS else ("Shoulder" if live_hour in {7,15,21} else "Off-Peak")
    period_color_lp = "#ff4444" if period_now_lp == "Peak" else ("#ffb347" if period_now_lp == "Shoulder" else "#00e676")

    rain_badge   = (
        "<span style='background:rgba(255,68,68,0.15);color:#ff8080;"
        "border-radius:20px;padding:2px 10px;font-size:0.68rem;"
        "font-family:Space Mono,monospace;font-weight:600;margin-left:8px;'>🌧️ RAIN SURGE</span>"
        if is_raining_live else ""
    )
    rain_multiplier_card = (
        "<div style='background:rgba(255,68,68,0.08);border:1px solid rgba(255,68,68,0.25);"
        "border-radius:10px;padding:0.5rem 0.9rem;text-align:center;'>"
        "<div style='font-size:0.62rem;color:#ff8080;letter-spacing:0.1em;text-transform:uppercase;'>Rain Multiplier</div>"
        "<div style='font-family:Orbitron,monospace;color:#ff8080;font-size:1rem;font-weight:700;'>×1.22</div></div>"
        if is_raining_live else ""
    )
    weekend_text = " · Weekend Mode" if is_weekend_live else " · Weekday"

    _banner = (
        "<div style='background:linear-gradient(135deg,#0a1428,#0d1420);"
        "border:1px solid #1a2540;border-radius:14px;"
        "padding:1.1rem 1.6rem;margin-bottom:0.8rem;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem;'>"
        "<div>"
        "<div style='font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:#6b7a99;'>Live Context · Pune Metro</div>"
        "<div style='font-family:Orbitron,monospace;font-size:1.25rem;"
        "color:#e8f0ff;font-weight:700;margin-top:4px;'>"
        + live_day + ", " + live_date
        + "<span style='color:" + period_color_lp + ";font-size:0.9rem;margin-left:12px;'>" + live_time + " IST</span>"
        + rain_badge
        + "</div>"
        "<div style='font-size:0.78rem;color:#6b7a99;margin-top:4px;'>"
        "<span style='color:" + period_color_lp + ";font-weight:600;'>" + period_now_lp + " Hour</span>"
        + weekend_text
        + "</div>"
        "</div>"
        "<div style='display:flex;gap:2rem;align-items:center;'>"
        "<div style='text-align:center;'>"
        "<div style='font-size:1.8rem;'>" + wx_icon + "</div>"
        "<div style='font-size:0.68rem;color:#6b7a99;'>" + weather_desc + "</div>"
        "</div>"
        "<div>"
        "<div style='font-size:1.4rem;font-family:Orbitron,monospace;color:#00d4ff;font-weight:700;'>" + str(temp_c) + "°C</div>"
        "<div style='font-size:0.68rem;color:#6b7a99;'>Feels " + str(feels_like) + "°C · " + str(humidity) + "% RH · " + str(wind_kph) + " km/h</div>"
        "</div>"
        + rain_multiplier_card
        + "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(_banner, unsafe_allow_html=True)

    if not weather_ok:
        st.markdown(
            f"<div class='alert-box alert-warning'>⚠️ Weather API unavailable "
            f"({weather_data.get('error','unknown')}) — add your OpenWeatherMap key.</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='section-header'>Prediction Controls</div>", unsafe_allow_html=True)
    sel_hour = live_hour
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        sel_line = st.selectbox("Metro Line", LINES)
    with ctrl2:
        sel_demand = st.number_input(
            "Predicted Demand (total passengers)",
            min_value=500, max_value=50000,
            value=23000 if sel_hour in PEAK_HOURS else 10000,
            step=500
        )

    st.markdown(
        f"<div style='font-size:0.74rem;color:#6b7a99;margin-bottom:0.5rem;'>"
        f"⏱ Live Pune time: <b style='color:#00d4ff;font-family:Orbitron,monospace;'>{live_time} IST</b> "
        f"(Hour {sel_hour:02d}:00)</div>",
        unsafe_allow_html=True
    )

    use_rain     = is_raining_live or sb_rain
    use_weekend  = is_weekend_live or sb_weekend
    use_festival = sb_festival

    trains_ai, headway_ai, wait_ai, svc_level = demand_to_trains(
        sel_demand, sel_hour, use_rain, use_festival, use_weekend
    )
    trains_fixed, headway_fixed, wait_fixed = fixed_schedule_baseline(sel_hour)
    wait_saved   = round(wait_fixed - wait_ai, 2)
    utilisation  = round(sel_demand / (trains_ai * EFFECTIVE_CAPACITY) * 100, 1)
    extra_trains = trains_ai - trains_fixed

    # ── Smart Alerts ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Smart Alert Feed</div>", unsafe_allow_html=True)
    alerts = generate_smart_alerts(use_rain, use_festival, use_weekend, sel_hour, utilisation)
    render_smart_alerts(alerts)

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    result_cards = [
        (r1, str(trains_ai),    "AI Trains/Hour",    f"Fixed: {trains_fixed}",         "#00d4ff"),
        (r2, f"{wait_ai}m",     "AI Avg Wait",        f"Fixed: {wait_fixed}m",          "#00e676" if wait_saved >= 0 else "#ff4444"),
        (r3, f"{headway_ai}m",  "Headway",            f"Every {headway_ai} mins",        "#b06cff"),
        (r4, f"{utilisation}%", "Fleet Utilisation",  svc_level,                        "#ffb347"),
    ]
    for col, val, label, delta, color in result_cards:
        with col:
            st.markdown(
                f"<div class='metric-card' style='--accent-line:{color}'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='color:{color}'>{val}</div>"
                f"<div class='metric-delta' style='color:#6b7a99'>{delta}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

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
    train_html += (
        f"</div><div style='font-size:0.72rem;color:#6b7a99;margin-top:5px;'>"
        f"🔵 Fixed ({trains_fixed}) &nbsp;|&nbsp; 🟢 AI Extra (+{max(0,extra_trains)}) &nbsp;|&nbsp; ⚫ Idle</div>"
    )
    st.markdown(train_html, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Wait Time Gauge</div>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    for col, title, val, color in [
        (col_g1, "Fixed Schedule Wait", wait_fixed, "#ff4444"),
        (col_g2, "AI Optimised Wait",   wait_ai,    "#00d4ff"),
    ]:
        with col:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                number={"suffix":" min","font":{"size":26,"color":color,"family":"Orbitron"}},
                title={"text":title,"font":{"size":12,"color":"#6b7a99"}},
                gauge=dict(
                    axis=dict(range=[0,15],tickcolor="#6b7a99",tickfont=dict(color="#6b7a99")),
                    bar=dict(color=color),
                    bgcolor="#0d1420",bordercolor="#1a2540",
                    steps=[
                        dict(range=[0, 3],  color="rgba(0,230,118,0.12)"),
                        dict(range=[3, 6],  color="rgba(255,179,71,0.12)"),
                        dict(range=[6, 15], color="rgba(255,68,68,0.12)"),
                    ],
                    threshold=dict(line=dict(color=color,width=3),value=val)
                )
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8f0ff"),
                                height=210, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

    # ── Demand Heatwave for Live Prediction ──────────────────────────────────
    st.markdown("<div class='section-header'>Demand Forecast — Next 6 Hours</div>", unsafe_allow_html=True)
    fig_hw2, _, _ = build_demand_heatwave(hourly_avg, live_hour, use_rain, use_weekend)
    st.plotly_chart(fig_hw2, use_container_width=True)


# ══════════════════════════════════════════════════
# PAGE 3 — FLEET SCHEDULER
# ══════════════════════════════════════════════════
elif page == "🚆 Fleet Scheduler":

    st.markdown("<div class='section-header'>Full Day Fleet Schedule — All Lines</div>", unsafe_allow_html=True)

    records = []
    for line in LINES:
        sub = hourly_avg[hourly_avg["line"] == line].sort_values("hour")
        for _, row in sub.iterrows():
            hour   = int(row["hour"])
            demand = row["avg_demand"]
            trains_ai, headway_ai, wait_ai, svc = demand_to_trains(demand, hour, sb_rain, sb_festival, sb_weekend)
            trains_fixed, _, wait_fixed = fixed_schedule_baseline(hour)
            records.append({
                "Line":           line,
                "Time":           f"{hour:02d}:00",
                "Demand":         f"{int(demand):,}",
                "Trains (Fixed)": trains_fixed,
                "Trains (AI)":    trains_ai,
                "Wait Fixed":     f"{wait_fixed}m",
                "Wait AI":        f"{wait_ai}m",
                "Saved":          f"{round(wait_fixed - wait_ai, 1)}m",
                "Service":        svc,
                "_period":        "Peak" if hour in PEAK_HOURS else "Off-Peak",
                "_line":          line,
            })

    sched = pd.DataFrame(records)

    fc1, fc2 = st.columns(2)
    with fc1:
        filt_line   = st.multiselect("Filter by Line",   LINES,               default=LINES)
    with fc2:
        filt_period = st.multiselect("Filter by Period", ["Peak","Off-Peak"], default=["Peak","Off-Peak"])

    sched_filtered = sched[
        sched["_line"].isin(filt_line) & sched["_period"].isin(filt_period)
    ].drop(columns=["_period","_line"])

    def render_table(df):
        rows = ""
        for _, r in df.iterrows():
            lc    = LINE_COLOR.get(r["Line"], "#6b7a99")
            saved = float(r["Saved"].replace("m",""))
            saved_color = "#00e676" if saved > 0 else ("#ff4444" if saved < 0 else "#6b7a99")
            svc   = str(r["Service"])
            if "Ultra" in svc or "Peak" in svc:
                badge = "<span class='badge badge-peak'>Peak</span>"
            elif "High" in svc:
                badge = "<span class='badge badge-high'>High</span>"
            elif "Normal" in svc:
                badge = "<span class='badge badge-normal'>Normal</span>"
            else:
                badge = "<span class='badge badge-low'>Low</span>"
            rows += (
                f"<tr>"
                f"<td><span style='color:{lc};font-weight:600;'>{r['Line']}</span></td>"
                f"<td style='font-family:Space Mono,monospace;'>{r['Time']}</td>"
                f"<td style='font-family:Space Mono,monospace;'>{r['Demand']}</td>"
                f"<td style='color:#6b7a99;'>{r['Trains (Fixed)']}</td>"
                f"<td style='color:{lc};font-weight:600;'>{r['Trains (AI)']}</td>"
                f"<td style='color:#6b7a99;'>{r['Wait Fixed']}</td>"
                f"<td style='color:#00e676;font-weight:600;'>{r['Wait AI']}</td>"
                f"<td style='color:{saved_color};font-weight:600;'>{r['Saved']}</td>"
                f"<td>{badge}</td>"
                f"</tr>"
            )
        return (
            "<table class='styled-table'><thead><tr>"
            "<th>Line</th><th>Time</th><th>Demand</th>"
            "<th>Trains Fixed</th><th>Trains AI</th>"
            "<th>Wait Fixed</th><th>Wait AI</th>"
            "<th>Saved</th><th>Service</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )

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
        sub["wf"] = sub["Wait Fixed"].str.replace("m","").astype(float)
        sub["wa"] = sub["Wait AI"].str.replace("m","").astype(float)
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


# ══════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════
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
                st.markdown(
                    "<div class='metric-card' style='--accent-line:#00e676'>"
                    "<div class='metric-label'>RMSE Improvement</div>"
                    f"<div class='metric-value' style='color:#00e676'>{pct:.1f}%</div>"
                    "<div class='metric-delta delta-pos'>LSTM beats ARIMA</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                pct = (arima_val - lstm_val) / arima_val * 100
                st.markdown(
                    f"<div class='metric-card' style='--accent-line:{color}'>"
                    f"<div class='metric-label'>{metric}</div>"
                    f"<div class='metric-value' style='color:{color}'>{lstm_val:.1f}</div>"
                    f"<div class='metric-delta delta-pos'>↓ {pct:.1f}% vs ARIMA ({arima_val:.0f})</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.markdown("<div class='section-header'>Side-by-Side Error Comparison</div>", unsafe_allow_html=True)
    fig_m = go.Figure()
    for model, color in [("ARIMA", "#ff4444"), ("LSTM", "#00d4ff")]:
        row = metrics_df[metrics_df["Model"] == model].iloc[0]
        fig_m.add_trace(go.Bar(
            name=model, x=["RMSE","MAE"],
            y=[row["RMSE"], row["MAE"]],
            marker_color=color,
            text=[f"{row['RMSE']:.1f}", f"{row['MAE']:.1f}"],
            textposition="outside",
            textfont=dict(color="white", size=12, family="Space Mono"),
        ))
    fig_m.update_layout(**PLOTLY_LAYOUT, height=340, barmode="group",
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        yaxis_title="Error Value (passengers)", title="Lower is Better ↓")
    st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("<div class='section-header'>LSTM Architecture</div>", unsafe_allow_html=True)
    arch_data = {
        "Layer":   ["Input","LSTM (128)","BatchNorm + Dropout","LSTM (64)","BatchNorm + Dropout","Dense (32)","Output (1)"],
        "Shape":   ["(24, 16)","(24, 256)","(24, 256)","(64,)","(64,)","(32,)","(1,)"],
        "Purpose": ["24hr lookback × 16 features","Broad temporal patterns","Regularisation",
                    "Fine-grained patterns","Regularisation","Non-linear compression","Passenger count"],
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
            st.markdown(
                f"<div class='metric-card' style='--accent-line:#b06cff;padding:1rem 1.2rem'>"
                f"<div class='metric-label'>{k}</div>"
                f"<div style='font-family:Orbitron,monospace;font-size:0.95rem;color:#b06cff;margin-top:0.3rem;'>{v}</div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════
# PAGE 5 — ROUTE MAP (with animated trains)
# ══════════════════════════════════════════════════
elif page == "🗺️ Route Map":

    # ── Animated Train Map ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Live Train Positions — Pune Metro Network</div>",
                unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.74rem;color:#6b7a99;margin-bottom:0.6rem;'>"
        "🚇 Trains simulated from live headway schedule · positions update on page refresh · "
        f"<b style='color:#00d4ff;'>{live_time} IST</b></div>",
        unsafe_allow_html=True
    )

    fig_trains = build_animated_train_map(stations_config, live_hour, sb_rain, sb_weekend)
    if fig_trains is not None:
        st.markdown("<div class='train-map-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(fig_trains, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Train legend strip
        trains_purple_n, hw_p, _, _ = demand_to_trains(18000, live_hour, sb_rain, False, sb_weekend)
        trains_aqua_n,   hw_a, _, _ = demand_to_trains(18000, live_hour, sb_rain, False, sb_weekend)
        st.markdown(
            "<div style='display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:6px;font-size:0.72rem;color:#6b7a99;'>"
            f"<span>🟣 Purple Line: <b style='color:#b06cff;'>{trains_purple_n} trains</b> · {hw_p:.1f}min headway</span>"
            f"<span>🔵 Aqua Line: <b style='color:#00d4ff;'>{trains_aqua_n} trains</b> · {hw_a:.1f}min headway</span>"
            "<span>🟡 Gold = Interchange</span>"
            "<span>🟢 Green occ &lt;65% · 🟠 Amber 65–85% · 🔴 Red &gt;85%</span>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Station coordinates not available — add lat/lon to stations_config.json to enable train animation.")

    # ── Folium static route map ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Static Route Map — Demand by Station</div>",
                unsafe_allow_html=True)

    col_slot, col_info = st.columns([2, 3])
    with col_slot:
        map_slot_label = st.selectbox("Time Slot", list(SLOT_OPTIONS.keys()),
                                      index=list(SLOT_OPTIONS.keys()).index(sb_slot_label))
        map_slot = SLOT_OPTIONS[map_slot_label]
    with col_info:
        st.markdown(
            "<div style='padding:0.6rem 0.9rem;background:#0d1420;border-radius:8px;"
            "font-size:0.78rem;color:#6b7a99;margin-top:0.4rem;line-height:1.8;'>"
            "🔴 Very High &nbsp;|&nbsp; 🟠 High &nbsp;|&nbsp; 🟡 Moderate &nbsp;|&nbsp; 🟢 Low &nbsp;|&nbsp; ⭐ Interchange<br>"
            "<span style='color:#7B2D8B;'>━━</span> Purple (14) &nbsp;|&nbsp; "
            "<span style='color:#00AEEF;'>━━</span> Aqua (16)"
            "</div>",
            unsafe_allow_html=True
        )

    with st.spinner("Building route map …"):
        try:
            map_html = build_map(slot=map_slot)
            components.html(map_html, height=500, scrolling=False)
        except ImportError:
            st.error("📦 Install folium: `pip install folium`")
        except Exception as e:
            st.error(f"Map error: {e}")

    # ── Demand Heatmap ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Demand Heatmap — Line × Hour</div>", unsafe_allow_html=True)
    pivot = hourly_avg.pivot_table(index="line", columns="hour", values="avg_demand", aggfunc="mean")
    fig_h = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Plasma",
        text=[[f"{v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorbar=dict(title=dict(text="Passengers", font=dict(color="#e8f0ff")),
                      tickfont=dict(color="#e8f0ff")),
    ))
    fig_h.update_layout(**PLOTLY_LAYOUT, height=240,
                        xaxis_title="Hour of Day", yaxis_title="")
    st.plotly_chart(fig_h, use_container_width=True)

    # ── High Demand Stations ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>High Demand Stations by Slot</div>", unsafe_allow_html=True)
    ds = DemandSegmentor(config_path="Data/stations_config.json")
    hd1, hd2 = st.columns(2)
    stations_list = stations_config.get("stations", [])

    def synthetic_top5(slot_key, color):
        np.random.seed({"morning_peak":1,"evening_peak":2}.get(slot_key,3))
        top = sorted(stations_list, key=lambda _: np.random.random(), reverse=True)[:5]
        rows = "".join([
            f"<tr><td style='color:{color};font-weight:700;'>{i+1}</td>"
            f"<td>{s['name']}</td>"
            f"<td style='font-family:Space Mono,monospace;color:{color};'>{np.random.randint(18000,32000):,}</td></tr>"
            for i, s in enumerate(top)
        ])
        return (
            f"<table class='styled-table'>"
            f"<thead><tr><th>#</th><th>Station</th><th>Avg Demand</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    with hd1:
        st.markdown("**🌅 Morning Peak — Top 5**")
        if "passenger_demand" in df_raw.columns and "station_id" in df_raw.columns:
            top_m = ds.high_demand_stations(df_raw, slot="morning_peak")
            st.dataframe(top_m[["rank","station_name","passenger_demand"]], use_container_width=True, hide_index=True)
        else:
            st.markdown(synthetic_top5("morning_peak","#FF6B35"), unsafe_allow_html=True)

    with hd2:
        st.markdown("**🌆 Evening Peak — Top 5**")
        if "passenger_demand" in df_raw.columns and "station_id" in df_raw.columns:
            top_e = ds.high_demand_stations(df_raw, slot="evening_peak")
            st.dataframe(top_e[["rank","station_name","passenger_demand"]], use_container_width=True, hide_index=True)
        else:
            st.markdown(synthetic_top5("evening_peak","#EF476F"), unsafe_allow_html=True)

    # ── Station Deep-Dive ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Station Deep-Dive</div>", unsafe_allow_html=True)
    sel_station = st.selectbox("Select Station", STATIONS)
    stn_data    = df_raw[df_raw["station"] == sel_station]
    stn_hourly  = stn_data.groupby("hour").agg(
        avg_pass=("passengers","mean"),
        max_pass=("passengers","max"),
        avg_wait=("avg_wait_time_mins","mean"),
    ).reset_index()

    sc1, sc2 = st.columns(2)
    with sc1:
        fig_st = go.Figure()
        fig_st.add_trace(go.Bar(
            x=stn_hourly["hour"], y=stn_hourly["avg_pass"],
            name="Avg Passengers",
            marker_color=[LINE_COLOR["Aqua"] if h in PEAK_HOURS else "#1e3050"
                          for h in stn_hourly["hour"]],
        ))
        fig_st.update_layout(**PLOTLY_LAYOUT, height=270,
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
        fig_wt.update_layout(**PLOTLY_LAYOUT, height=270,
                             title=f"{sel_station} — Avg Wait Time",
                             xaxis_title="Hour", yaxis_title="Wait (mins)")
        st.plotly_chart(fig_wt, use_container_width=True)

    peak_pass  = stn_data[stn_data["hour"].isin(PEAK_HOURS)]["passengers"].mean()
    offpk_pass = stn_data[~stn_data["hour"].isin(PEAK_HOURS)]["passengers"].mean()
    rain_pass  = stn_data[stn_data["is_raining"] == 1]["passengers"].mean()

    ss1, ss2, ss3 = st.columns(3)
    for col, label, val, color in [
        (ss1, "Peak Avg Passengers", f"{peak_pass:.0f}",  "#ff4444"),
        (ss2, "Off-Peak Avg",        f"{offpk_pass:.0f}", "#00d4ff"),
        (ss3, "Avg When Raining",    f"{rain_pass:.0f}",  "#b06cff"),
    ]:
        with col:
            st.markdown(
                f"<div class='metric-card' style='--accent-line:{color};padding:1rem 1.2rem'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='color:{color};font-size:1.5rem'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════
# PAGE 6 — JOURNEY PLANNER
# ══════════════════════════════════════════════════
elif page == "🧭 Journey Planner":

    st.markdown("<div class='section-header'>Point-to-Point Journey Planner</div>", unsafe_allow_html=True)

    @st.cache_resource
    def get_router():
        return MetroRouter()

    try:
        router      = get_router()
        name_map    = router.station_names()
        options     = [f"{sid} — {name}" for sid, name in name_map.items()]
        id_from_opt = {f"{sid} — {name}": sid for sid, name in name_map.items()}
    except Exception as e:
        st.error(f"Router init failed: {e}")
        st.stop()

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
        result = router.shortest_path(origin_id, dest_id)

        if "error" in result:
            st.error(result["error"])
        else:
            k1, k2, k3, k4, k5 = st.columns(5)
            kpis = [
                (k1, f"{result['total_time_mins']:.1f} min", "Journey Time", "#00d4ff"),
                (k2, f"{result['total_dist_km']:.2f} km",   "Distance",     "#b06cff"),
                (k3, str(result['num_stops']),               "Stations",     "#ffb347"),
                (k4, str(result['transfers']),               "Transfers",    "#ff4444" if result['transfers'] else "#00e676"),
                (k5, f"₹{result['fare_inr']}",              "Est. Fare",    "#00e676"),
            ]
            for col, val, label, color in kpis:
                with col:
                    st.markdown(
                        f"<div class='metric-card' style='--accent-line:{color};padding:1rem 1.2rem;'>"
                        f"<div class='metric-label'>{label}</div>"
                        f"<div class='metric-value' style='color:{color};font-size:1.4rem'>{val}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            if result["transfers"]:
                st.markdown(
                    f"<div class='alert-box alert-warning'>🔄 <b>Transfer required</b> at "
                    f"<b>{', '.join(result['transfer_at'])}</b> — change lines (+5 min)</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<div class='alert-box alert-normal'>✅ <b>Direct journey</b> — no transfer needed</div>",
                            unsafe_allow_html=True)

            st.markdown("<div class='section-header'>Step-by-Step Route</div>", unsafe_allow_html=True)
            route_table = []
            for i, stop in enumerate(result["path"]):
                is_first = i == 0
                is_last  = i == len(result["path"]) - 1
                is_xchg  = stop["interchange"]
                if is_first:      icon = "🚉 Start"
                elif is_last:     icon = "🏁 End"
                elif is_xchg:     icon = "⭐ Transfer"
                else:             icon = "• Stop"
                route_table.append({
                    "":        icon,
                    "Station": stop["name"] + (" ⇄" if is_xchg else ""),
                    "Line":    stop["line"].title(),
                    "Zone":    stop["zone"].title(),
                    "Time":    "Start" if is_first else f"{stop['cumulative_time_mins']} min",
                })
            st.dataframe(pd.DataFrame(route_table), use_container_width=True, hide_index=True)

            stop_names  = [p["name"] for p in result["path"]]
            stop_times  = [p["cumulative_time_mins"] for p in result["path"]]
            stop_colors = [
                "#00e676" if i == 0
                else "#ff4444" if i == len(result["path"])-1
                else "#FFD700" if result["path"][i]["interchange"]
                else "#00d4ff"
                for i in range(len(result["path"]))
            ]
            fig_route = go.Figure()
            fig_route.add_trace(go.Bar(
                x=stop_names, y=stop_times,
                marker_color=stop_colors,
                text=[f"{t} min" if t else "Start" for t in stop_times],
                textposition="outside",
                textfont=dict(color="white", size=10),
            ))
            fig_route.update_layout(**PLOTLY_LAYOUT, height=270,
                                    xaxis_title="", yaxis_title="Cumulative Time (min)",
                                    xaxis_tickangle=-35, showlegend=False)
            st.plotly_chart(fig_route, use_container_width=True)

            st.markdown("<div class='section-header'>Route on Map</div>", unsafe_allow_html=True)
            try:
                import folium
                path_stations = [router.stations[p["id"]] for p in result["path"]]
                center_lat = sum(s["lat"] for s in path_stations) / len(path_stations)
                center_lon = sum(s["lon"] for s in path_stations) / len(path_stations)
                mini_map = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB dark_matter")
                coords   = [[s["lat"], s["lon"]] for s in path_stations]
                folium.PolyLine(coords, color="#00d4ff", weight=5, opacity=0.9, tooltip="Optimal Route").add_to(mini_map)
                for i, stop in enumerate(result["path"]):
                    s = router.stations[stop["id"]]
                    color = "green" if i==0 else ("red" if i==len(result["path"])-1 else ("orange" if stop["interchange"] else "blue"))
                    icon  = "play"  if i==0 else ("flag" if i==len(result["path"])-1 else ("star" if stop["interchange"] else "circle"))
                    folium.Marker(
                        location=[s["lat"], s["lon"]],
                        tooltip=f"{stop['name']} ({stop['cumulative_time_mins']} min)",
                        icon=folium.Icon(color=color, icon=icon, prefix="fa"),
                    ).add_to(mini_map)
                components.html(mini_map._repr_html_(), height=380, scrolling=False)
            except ImportError:
                st.info("📦 Install folium for route map: `pip install folium`")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='background:#0d1420;border:1px solid #1a2540;border-radius:12px;"
                "padding:1.1rem 1.5rem;font-family:Space Mono,monospace;"
                "font-size:0.82rem;color:#6b7a99;line-height:2;'>"
                "📋 <b style='color:#e8f0ff;'>Journey Summary</b><br>"
                f"{result['summary']}"
                "</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════
# PAGE 7 — FLEET OPTIMIZER
# ══════════════════════════════════════════════════
elif page == "⚡ Fleet Optimizer":

    st.markdown("<div class='section-header'>Smart Fleet & Pickup Optimizer</div>", unsafe_allow_html=True)

    @st.cache_resource
    def get_optimizer():
        return PickupOptimizer()

    opt = get_optimizer()

    # Hour and fleet size taken live — no sliders
    opt_hour  = live_hour
    opt_fleet = _sys_trains  # use system-computed active trains

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        opt_rain = st.toggle("🌧️ Rain Surge", value=sb_rain)
    with ctrl2:
        opt_fest = st.toggle("🎉 Festival",    value=sb_festival)

    st.markdown(
        f"<div style='font-size:0.74rem;color:#6b7a99;margin-bottom:0.5rem;'>"
        f"⏱ Optimising for live time: <b style='color:#00d4ff;font-family:Orbitron,monospace;'>{live_time} IST</b> "
        f"(Hour {opt_hour:02d}:00) · Fleet: <b style='color:#b06cff;'>{opt_fleet} active trains</b></div>",
        unsafe_allow_html=True
    )

    demand = opt.demo_demand(opt_hour)
    result = opt.optimize(
        demand,
        hour           = opt_hour,
        rain_surge     = opt_rain,
        festival_surge = opt_fest,
        weekend        = sb_weekend,
    )

    # ── Smart Alerts ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Smart Alert Feed</div>", unsafe_allow_html=True)
    alerts = generate_smart_alerts(opt_rain, opt_fest, sb_weekend, opt_hour,
                                   getattr(result, "efficiency_score", None))
    render_smart_alerts(alerts)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, f"{result.total_demand:,}",        "Total Demand",     "#00d4ff"),
        (k2, str(result.total_trains),           "Trains Deployed",  "#b06cff"),
        (k3, f"{result.avg_wait_mins:.1f} min",  "Avg Wait",         "#ffb347"),
        (k4, f"{result.efficiency_score:.1f}%",  "Fleet Efficiency",
         "#00e676" if result.efficiency_score >= 80 else "#ff4444"),
    ]
    for col, val, label, color in kpis:
        with col:
            st.markdown(
                f"<div class='metric-card' style='--accent-line:{color}'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='color:{color}'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    if result.unmet_demand:
        st.markdown(
            f"<div class='alert-box alert-surge'>⚠️ <b>Capacity exceeded</b> — "
            f"{result.unmet_demand:,} passengers unserved.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='alert-box alert-normal'>✅ All demand served within current fleet capacity.</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='section-header'>Line Allocation Summary</div>", unsafe_allow_html=True)
    line_cols = st.columns(len(result.line_summary))
    line_colors_map = {"purple": "#7B2D8B", "aqua": "#00AEEF"}
    for col, (ln, stats) in zip(line_cols, result.line_summary.items()):
        lc = line_colors_map.get(ln, "#6b7a99")
        with col:
            st.markdown(
                f"<div class='metric-card' style='--accent-line:{lc};'>"
                f"<div class='metric-label'>{ln.title()} Line</div>"
                f"<div class='metric-value' style='color:{lc};font-size:1.7rem;'>{stats['trains']} trains</div>"
                f"<div class='metric-delta' style='color:#6b7a99;'>{stats['demand']:,} pax · {stats['avg_wait']:.1f} min wait</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<div class='section-header'>Station Priority & Utilisation</div>", unsafe_allow_html=True)
    alloc_df = pd.DataFrame([{
        "Station":     a.station_name,
        "Line":        a.line.title(),
        "Demand":      a.demand,
        "Trains":      a.line_trains,
        "Wait (min)":  a.avg_wait_mins,
        "Utilisation": a.utilisation_pct,
        "Status":      "OVERFLOW" if a.utilisation_pct >= 95 else ("HIGH" if a.utilisation_pct >= 75 else "NORMAL"),
        "Priority":    a.priority_score,
    } for a in result.allocations])

    STATUS_COLOR = {"OVERFLOW":"#ff4444","SURGE":"#ff7c3a","HIGH":"#ffd700","NORMAL":"#00e676"}

    fig_alloc = go.Figure()
    for status, color in STATUS_COLOR.items():
        sub = alloc_df[alloc_df["Status"] == status]
        if len(sub) == 0:
            continue
        fig_alloc.add_trace(go.Bar(
            name=status, x=sub["Station"], y=sub["Utilisation"],
            marker_color=color,
            text=sub["Trains"].astype(str) + "T",
            textposition="outside",
            textfont=dict(color="white", size=9),
            hovertemplate=(
                "<b>%{x}</b><br>Utilisation: %{y:.1f}%<br>"
                "Demand: %{customdata[0]:,}<br>Wait: %{customdata[1]:.1f} min<extra></extra>"
            ),
            customdata=sub[["Demand","Wait (min)"]].values,
        ))

    fig_alloc.add_hline(y=75, line_dash="dash", line_color="#ff7c3a", opacity=0.6,
                        annotation_text="High (75%)", annotation_font_color="#ff7c3a")
    fig_alloc.add_hline(y=95, line_dash="dash", line_color="#ff4444", opacity=0.6,
                        annotation_text="Overflow (95%)", annotation_font_color="#ff4444")
    fig_alloc.update_layout(
        **PLOTLY_LAYOUT, height=370, barmode="overlay",
        xaxis_tickangle=-40, yaxis_title="Fleet Utilisation (%)",
        yaxis_range=[0,115], legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.markdown("<div class='section-header'>Full Station Allocation Table</div>", unsafe_allow_html=True)
    display_df = alloc_df[["Station","Line","Demand","Trains","Wait (min)","Utilisation","Status"]].reset_index(drop=True)
    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        column_config={
            "Utilisation": st.column_config.ProgressColumn("Utilisation %", min_value=0, max_value=100, format="%.1f%%"),
            "Demand":      st.column_config.NumberColumn("Demand", format="%d pax"),
        }
    )
    st.download_button("⬇️ Download Allocation CSV",
                       display_df.to_csv(index=False),
                       f"fleet_allocation_hour{opt_hour:02d}.csv",
                       "text/csv")


# ══════════════════════════════════════════════════
# PAGE 8 — MULTI-OBJECTIVE OPTIMIZER
# ══════════════════════════════════════════════════
elif page == "🎯 Multi-Objective":

    @st.cache_resource
    def get_moo():
        return MultiObjectiveOptimizer()
    moo = get_moo()

    moo_hour = live_hour

    st.markdown(
        f"<div style='font-size:0.74rem;color:#6b7a99;margin-bottom:0.6rem;'>"
        f"⏱ Live Pune time: <b style='color:#00d4ff;font-family:Orbitron,monospace;'>{live_time} IST</b> "
        f"(Hour {moo_hour:02d}:00)</div>",
        unsafe_allow_html=True
    )

    w_time, w_fuel, w_coverage, rationale = moo.get_auto_weights(moo_hour)

    period_label = "✅ Peak" if moo_hour in PEAK_HOURS else ("Shoulder" if moo_hour in {7,15,21} else "Off-Peak")
    if moo_hour in PEAK_HOURS:
        mode_color, mode_icon, mode_text = "#ff4444", "⚡", "PEAK"
    elif moo_hour in [7, 15, 21]:
        mode_color, mode_icon, mode_text = "#ffb347", "〰", "SHOULDER"
    else:
        mode_color, mode_icon, mode_text = "#00e676", "🌙", "OFF-PEAK"

    _moo_card = (
        "<div style='background:linear-gradient(135deg,#0d1420,#0a1428);"
        "border:1px solid #1a2540;border-radius:14px;"
        "padding:1.3rem 1.7rem;margin-bottom:1rem;'>"
        "<div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.8rem;'>"
        "<span style='font-family:Orbitron,monospace;font-size:1.4rem;color:" + mode_color + ";font-weight:700;text-shadow:0 0 12px " + mode_color + "80;'>"
        + mode_icon + " " + mode_text + "</span>"
        "<span style='font-size:0.76rem;color:#6b7a99;font-family:Space Mono,monospace;'>Hour " + f"{moo_hour:02d}" + ":00 — AI auto-selected weights</span>"
        "</div>"
        "<div style='font-size:0.86rem;color:#b8c4d8;margin-bottom:1rem;'>" + rationale + "</div>"
        "<div style='display:flex;gap:2.5rem;'>"
        "<div><span style='font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#6b7a99;'>⏱ Time</span>"
        "<div style='font-family:Orbitron,monospace;font-size:1.05rem;color:#00d4ff;font-weight:700;text-shadow:0 0 8px rgba(0,212,255,0.5);'>" + f"{w_time:.0%}" + "</div></div>"
        "<div><span style='font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#6b7a99;'>⛽ Fuel</span>"
        "<div style='font-family:Orbitron,monospace;font-size:1.05rem;color:#ffb347;font-weight:700;text-shadow:0 0 8px rgba(255,179,71,0.5);'>" + f"{w_fuel:.0%}" + "</div></div>"
        "<div><span style='font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#6b7a99;'>🗺 Coverage</span>"
        "<div style='font-family:Orbitron,monospace;font-size:1.05rem;color:#00e676;font-weight:700;text-shadow:0 0 8px rgba(0,230,118,0.5);'>" + f"{w_coverage:.0%}" + "</div></div>"
        "</div></div>"
    )
    st.markdown(_moo_card, unsafe_allow_html=True)

    demand = moo.demo_demand(moo_hour)
    result = moo.optimize(
        demand, hour=moo_hour,
        w_time=w_time, w_fuel=w_fuel, w_coverage=w_coverage,
        rain_surge=sb_rain, festival_surge=sb_festival, weekend=sb_weekend,
    )

    # ── Smart Alerts ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Smart Alert Feed</div>", unsafe_allow_html=True)
    render_smart_alerts(generate_smart_alerts(sb_rain, sb_festival, sb_weekend, moo_hour))

    st.markdown("<div class='section-header'>Optimal Solution</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        (k1, str(result.total_trains),           "Trains Deployed",  "#00d4ff"),
        (k2, f"{result.avg_wait_mins:.1f} min",  "Avg Wait (AI)",
              "#00e676" if result.wait_saved_mins >= 0 else "#ff4444"),
        (k3, f"{result.wait_saved_mins:+.1f}m",  "vs Fixed Schedule",
              "#00e676" if result.wait_saved_mins >= 0 else "#ff4444"),
        (k4, f"{result.total_energy_kwh:.0f} kWh","Energy / Hour",   "#ffb347"),
        (k5, f"{result.coverage_pct:.1f}%",       "Network Coverage","#b06cff"),
    ]
    for col, val, label, color in kpi_data:
        with col:
            st.markdown(
                f"<div class='metric-card' style='--accent-line:{color}'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='color:{color};font-size:1.4rem'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<div class='section-header'>Per-Line Allocation</div>", unsafe_allow_html=True)
    lc_map = {"purple":"#7B2D8B","aqua":"#00AEEF"}
    line_cols = st.columns(len(result.line_results))
    for col, (ln, lr) in zip(line_cols, result.line_results.items()):
        lc = lc_map.get(ln, "#6b7a99")
        score_bar_t = int(lr.score_time    * 10)
        score_bar_f = int(lr.score_fuel    * 10)
        score_bar_c = int(lr.score_coverage* 10)
        with col:
            st.markdown(
                f"<div class='metric-card' style='--accent-line:{lc};'>"
                f"<div class='metric-label'>{ln.title()} Line</div>"
                f"<div class='metric-value' style='color:{lc};font-size:1.9rem;'>{lr.optimal_trains} trains</div>"
                f"<div style='font-size:0.76rem;color:#6b7a99;margin-top:0.6rem;line-height:2;'>"
                f"⏱ Headway: <b style='color:#e8f0ff;'>{lr.headway_mins:.1f} min</b><br>"
                f"🕐 Avg wait: <b style='color:#e8f0ff;'>{lr.avg_wait_mins:.1f} min</b><br>"
                f"⛽ Energy: <b style='color:#e8f0ff;'>{lr.energy_kwh_hr:.0f} kWh/hr</b><br>"
                f"📊 Utilisation: <b style='color:#e8f0ff;'>{lr.utilisation_pct:.1f}%</b><br>"
                f"{lr.service_level}"
                f"</div>"
                f"<div style='margin-top:0.8rem;'>"
                f"<div style='font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;color:#6b7a99;margin-bottom:4px;'>Objective Scores</div>"
                f"<div style='font-size:0.7rem;color:#6b7a99;font-family:Space Mono,monospace;'>"
                f"⏱ <span style='color:#00d4ff;'>{'█'*score_bar_t}{'░'*(10-score_bar_t)}</span> {lr.score_time:.2f}<br>"
                f"⛽ <span style='color:#ffb347;'>{'█'*score_bar_f}{'░'*(10-score_bar_f)}</span> {lr.score_fuel:.2f}<br>"
                f"🗺 <span style='color:#00e676;'>{'█'*score_bar_c}{'░'*(10-score_bar_c)}</span> {lr.score_coverage:.2f}"
                f"</div></div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<div class='section-header'>How AI Weights Change by Hour of Day</div>",
                unsafe_allow_html=True)

    all_hours   = list(range(6, 24))
    wt_by_hour, wf_by_hour, wc_by_hour = [], [], []
    for h in all_hours:
        wt_, wf_, wc_, _ = moo.get_auto_weights(h)
        wt_by_hour.append(round(wt_ * 100))
        wf_by_hour.append(round(wf_ * 100))
        wc_by_hour.append(round(wc_ * 100))

    fig_weights = go.Figure()
    fig_weights.add_trace(go.Scatter(x=all_hours, y=wt_by_hour, name="⏱ Time",
                                     fill="tozeroy", fillcolor="rgba(0,212,255,0.10)",
                                     line=dict(color="#00d4ff", width=2.5), mode="lines"))
    fig_weights.add_trace(go.Scatter(x=all_hours, y=wf_by_hour, name="⛽ Fuel",
                                     fill="tozeroy", fillcolor="rgba(255,179,71,0.10)",
                                     line=dict(color="#ffb347", width=2.5), mode="lines"))
    fig_weights.add_trace(go.Scatter(x=all_hours, y=wc_by_hour, name="🗺 Coverage",
                                     fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
                                     line=dict(color="#00e676", width=2.5), mode="lines"))

    fig_weights.add_vline(x=moo_hour, line_dash="dash", line_color=mode_color, line_width=2,
                          annotation_text=f"  {moo_hour:02d}:00",
                          annotation_font_color=mode_color, annotation_font_size=11)

    for x0, x1, c in [(8,12,"rgba(255,68,68,0.05)"),(16,21,"rgba(255,68,68,0.05)"),
                       (7,8,"rgba(255,179,71,0.05)"),(15,16,"rgba(255,179,71,0.05)"),
                       (21,22,"rgba(255,179,71,0.05)")]:
        fig_weights.add_vrect(x0=x0, x1=x1, fillcolor=c, line_width=0)

    # 1. Apply global base layout first
    fig_weights.update_layout(**PLOTLY_LAYOUT)
    
    # 2. Apply specific chart overrides
    fig_weights.update_layout(
        height=250,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0.01, y=1.15),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_weights.update_xaxes(tickvals=list(range(6,24)),
                              ticktext=[f"{h:02d}:00" for h in range(6,24)],
                              tickangle=-45, tickfont=dict(size=9),
                              gridcolor="#1a2540", linecolor="#1a2540")
    fig_weights.update_yaxes(title_text="Weight (%)", range=[0,80],
                              gridcolor="#1a2540", linecolor="#1a2540")
    st.plotly_chart(fig_weights, use_container_width=True)