"""
route_map.py
Generates an interactive Folium HTML map of the Pune Metro network.
Features:
  - Purple + Aqua line routes drawn on map
  - All 30 stations as clickable markers
  - Demand heatmap overlay (high-demand areas highlighted)
  - Interchange station callouts
  - Time-slot toggle (Morning / Evening / Weekend)

Usage:
    python Dashboard/route_map.py
    → saves to Outputs/pune_metro_map.html

Or import into app.py:
    from Dashboard.route_map import build_map
    map_html = build_map(df, slot="morning_peak")
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CONFIG_PATH = Path(__file__).parent.parent / "Data" / "stations_config.json"
OUTPUT_PATH = Path(__file__).parent.parent / "Outputs" / "pune_metro_map.html"

# ── Demand colours ────────────────────────────────────────────────────────────
def _demand_color(demand: float, max_demand: float) -> str:
    ratio = demand / max_demand if max_demand > 0 else 0
    if ratio > 0.75: return "#FF3B3B"   # red   — very high
    if ratio > 0.50: return "#FF8C00"   # orange — high
    if ratio > 0.25: return "#FFD700"   # yellow — moderate
    return "#4CAF50"                    # green  — low


def _demand_radius(demand: float, max_demand: float) -> int:
    ratio = demand / max_demand if max_demand > 0 else 0
    return int(8 + ratio * 18)          # 8px – 26px


# ── Synthetic demand fallback (if no real df passed) ─────────────────────────
def _synthetic_demand(stations: list, slot: str) -> dict:
    np.random.seed(42)
    slot_multipliers = {
        "morning_peak": lambda: np.random.randint(280, 520),
        "evening_peak": lambda: np.random.randint(260, 500),
        "afternoon":    lambda: np.random.randint(100, 220),
        "weekend":      lambda: np.random.randint(140, 300),
        "night":        lambda: np.random.randint(30,  100),
    }
    fn = slot_multipliers.get(slot, slot_multipliers["morning_peak"])
    return {s["id"]: fn() for s in stations}


# ── Purple line ordered sequence ──────────────────────────────────────────────
PURPLE_ORDER = ["PU01","PU02","PU03","PU04","PU05","PU06",
                "PU07","PU08","PU09","PU10","PU11","PU12","PU13","PU14"]

AQUA_ORDER   = ["AQ01","AQ02","AQ03","AQ04","AQ05","AQ06","AQ07","AQ08",
                "AQ09","AQ10","AQ11","AQ12","AQ13","AQ14","AQ15","AQ16"]


# ── Main map builder ──────────────────────────────────────────────────────────
def build_map(
    df: pd.DataFrame = None,
    slot: str = "morning_peak",
    save_path: Path = OUTPUT_PATH,
) -> str:
    """
    Build and return HTML string of the interactive metro map.
    df   : DataFrame with columns [station_id, passenger_demand] (optional)
    slot : one of morning_peak / evening_peak / afternoon / weekend / night
    """
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
    except ImportError:
        raise ImportError("Run:  pip install folium")

    # Load station config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    stations    = config["stations"]
    station_map = {s["id"]: s for s in stations}

    # Demand data
    if df is not None and "passenger_demand" in df.columns:
        demand_lookup = (
            df[df["slot"] == slot]
            .groupby("station_id")["passenger_demand"]
            .mean()
            .to_dict()
        ) if "slot" in df.columns else (
            df.groupby("station_id")["passenger_demand"].mean().to_dict()
        )
    else:
        demand_lookup = _synthetic_demand(stations, slot)

    max_demand = max(demand_lookup.values()) if demand_lookup else 1

    # ── Base map ──────────────────────────────────────────────────────────────
    m = folium.Map(
        location=[18.5531, 73.8673],
        zoom_start=13,
        tiles="CartoDB dark_matter",
    )

    # ── Draw lines ────────────────────────────────────────────────────────────
    def draw_line(order, color, name):
        coords = []
        for sid in order:
            s = station_map.get(sid)
            if s:
                coords.append([s["lat"], s["lon"]])
        if coords:
            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                opacity=0.9,
                tooltip=name,
            ).add_to(m)

    draw_line(PURPLE_ORDER, "#7B2D8B", "Purple Line — PCMC Bhavan ↔ Swargate")
    draw_line(AQUA_ORDER,   "#00AEEF", "Aqua Line — Vanaz ↔ Ramwadi")

    # ── Heatmap layer ─────────────────────────────────────────────────────────
    heat_data = []
    for s in stations:
        demand = demand_lookup.get(s["id"], 0)
        heat_data.append([s["lat"], s["lon"], demand / max_demand])

    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=15,
        radius=35,
        blur=25,
        gradient={0.2: "#4CAF50", 0.5: "#FFD700", 0.75: "#FF8C00", 1.0: "#FF3B3B"},
        name="Demand Heatmap",
    ).add_to(m)

    # ── Station markers ───────────────────────────────────────────────────────
    slot_labels = {
        "morning_peak": "🌅 Morning Peak (07–10)",
        "evening_peak": "🌆 Evening Peak (17–20)",
        "afternoon":    "☀️  Afternoon (11–16)",
        "weekend":      "📅 Weekend",
        "night":        "🌙 Night (21–06)",
    }

    for s in stations:
        demand  = demand_lookup.get(s["id"], 0)
        color   = _demand_color(demand, max_demand)
        radius  = _demand_radius(demand, max_demand)
        is_xchg = s.get("interchange", False)
        line_color = "#7B2D8B" if s["line"] == "purple" else "#00AEEF"

        popup_html = f"""
        <div style="font-family: monospace; min-width: 180px;">
            <b style="font-size:14px; color:{line_color};">{s['name']}</b><br>
            <span style="color:#888;">ID: {s['id']} | Zone: {s.get('zone','—')}</span><br>
            <hr style="margin:4px 0; border-color:#444;">
            <b>Avg Demand ({slot_labels.get(slot, slot)}):</b>
            <span style="color:{color}; font-size:15px;"> {int(demand):,} pax</span><br>
            {"<b style='color:#FFD700;'>⇄ INTERCHANGE STATION</b><br>" if is_xchg else ""}
            {"🔲 Underground" if s.get('underground') else "🏗 Elevated"}<br>
            <span style="color:#888;">Opened: {s.get('opened','—')}</span>
        </div>
        """

        # Interchange gets a star marker, others get circle
        if is_xchg:
            folium.Marker(
                location=[s["lat"], s["lon"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"⇄ {s['name']} — {int(demand):,} pax",
                icon=folium.Icon(color="orange", icon="star", prefix="fa"),
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[s["lat"], s["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{s['name']} — {int(demand):,} pax",
            ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: rgba(20,20,30,0.92); border: 1px solid #444;
        border-radius: 10px; padding: 14px 18px;
        font-family: monospace; font-size: 12px; color: #eee;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    ">
        <b style="font-size:13px;">🚇 Pune Metro Map</b><br>
        <span style="color:#888;">Slot: {slot_labels.get(slot, slot)}</span>
        <hr style="border-color:#444; margin:6px 0;">
        <b>Lines</b><br>
        <span style="color:#7B2D8B;">━━</span> Purple Line (14 stations)<br>
        <span style="color:#00AEEF;">━━</span> Aqua Line (16 stations)<br>
        <hr style="border-color:#444; margin:6px 0;">
        <b>Demand Level</b><br>
        <span style="color:#FF3B3B;">●</span> Very High (&gt;75%)<br>
        <span style="color:#FF8C00;">●</span> High (50–75%)<br>
        <span style="color:#FFD700;">●</span> Moderate (25–50%)<br>
        <span style="color:#4CAF50;">●</span> Low (&lt;25%)<br>
        <span style="color:#FFD700;">★</span> Interchange Station
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title bar ─────────────────────────────────────────────────────────────
    title_html = f"""
    <div style="
        position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: rgba(20,20,30,0.92);
        border: 1px solid #444; border-radius: 8px;
        padding: 8px 24px; font-family: monospace;
        font-size: 15px; color: #fff;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    ">
        🚇 <b>SmartTransit AI</b> — Pune Metro Demand Map &nbsp;|&nbsp;
        <span style="color:#FFD700;">{slot_labels.get(slot, slot)}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Layer control ─────────────────────────────────────────────────────────
    folium.LayerControl().add_to(m)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path.parent.mkdir(exist_ok=True)
    m.save(str(save_path))
    print(f"✅ Map saved → {save_path}")

    return m._repr_html_()


# ── CLI: generate all 5 slot maps ────────────────────────────────────────────
if __name__ == "__main__":
    slots = ["morning_peak", "afternoon", "evening_peak", "weekend", "night"]
    for slot in slots:
        out = OUTPUT_PATH.parent / f"pune_metro_map_{slot}.html"
        print(f"  Building {slot} map …")
        build_map(slot=slot, save_path=out)
    print("\n✅ All maps generated in Outputs/")