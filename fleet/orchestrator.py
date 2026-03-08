"""
orchestrator.py
SmartTransit AI – Fleet Orchestration Engine
Correctly calibrated to real Pune Metro daily demand
Place this file in: fleet/orchestrator.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PUNE METRO REAL-WORLD CONSTANTS
# ─────────────────────────────────────────────
# MahaMetro 6-car train: ~300 passengers/car × 6 = 1800/train
# 90% safe load factor  → 1620 effective capacity per train
# Signalling limit      → max 15 trains/hr (4-min headway)
# Minimum service       → 3 trains/hr (20-min headway)

TRAIN_CAPACITY     = 1800
MAX_OCCUPANCY      = 0.90
EFFECTIVE_CAPACITY = int(TRAIN_CAPACITY * MAX_OCCUPANCY)   # 1620

# Real Pune Metro peak windows
PEAK_HOURS     = list(range(8, 12)) + list(range(16, 21))  # 8-11 AM, 4-8 PM
SHOULDER_HOURS = [7, 15, 21]
OFF_PEAK_HOURS = [6, 12, 13, 14, 22]

# Context-aware demand multipliers (derived from dataset)
RAIN_MULTIPLIER     = 1.22
FESTIVAL_MULTIPLIER = 1.55
WEEKEND_MULTIPLIER  = 1.10

MODELS_DIR = "Models"
OUTPUT_DIR = "Outputs"
DATA_PATH  = "Data/pune_metro_enhanced_data.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────
def demand_to_trains(total_line_demand, hour,
                     is_raining=False, is_festival=False, is_weekend=False):
    """
    Converts TOTAL line demand (all stations) → optimal train frequency.

    Key fix: uses total_line_demand, not per-station demand.
    Pune Metro Aqua line peak = ~23,000 passengers/hour across all stations.
    At 1620/train → needs ceil(23000/1620) = 15 trains → headway 4 min → wait 2 min ✅
    """
    # Apply real-world context multipliers
    adjusted = total_line_demand
    if is_raining:   adjusted *= RAIN_MULTIPLIER
    if is_festival:  adjusted *= FESTIVAL_MULTIPLIER
    if is_weekend:   adjusted *= WEEKEND_MULTIPLIER

    # Core formula: trains needed to carry demand at 90% load
    trains_needed = int(np.ceil(adjusted / EFFECTIVE_CAPACITY))

    # Enforce minimum service floors
    if hour in PEAK_HOURS:
        trains_needed = max(trains_needed, 8)
    elif hour in SHOULDER_HOURS:
        trains_needed = max(trains_needed, 5)
    else:
        trains_needed = max(trains_needed, 3)

    # Physical signalling ceiling
    trains_per_hour = min(trains_needed, 15)

    headway_mins  = round(60 / trains_per_hour, 1)
    avg_wait_mins = round(headway_mins / 2, 1)   # uniform arrival model

    # Service level label
    if trains_per_hour >= 13:   svc = "🔴 Ultra Peak"
    elif trains_per_hour >= 10: svc = "🟠 Peak"
    elif trains_per_hour >= 7:  svc = "🟡 High"
    elif trains_per_hour >= 5:  svc = "🟢 Normal"
    else:                       svc = "⚪ Low"

    return trains_per_hour, headway_mins, avg_wait_mins, svc


def fixed_schedule_baseline(hour):
    """
    Current fixed timetable (no AI).
    Based on actual MahaMetro Pune published schedule:
      Peak    → 10 trains/hr → 6-min headway → 3-min avg wait
      Shoulder→  7 trains/hr → 8.6-min headway → 4.3-min avg wait
      Off-peak→  5 trains/hr → 12-min headway → 6-min avg wait
    """
    if hour in PEAK_HOURS:
        return 10, 6.0, 3.0
    elif hour in SHOULDER_HOURS:
        return 7, 8.6, 4.3
    else:
        return 5, 12.0, 6.0


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_inputs():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  SmartTransit AI – Fleet Orchestrator")
    print("  (Calibrated to Real Pune Metro Data)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    df = pd.read_csv(DATA_PATH)
    df["date"]     = pd.to_datetime(df["date"])
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

    with open(f"{MODELS_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load LSTM model for live predictions
    try:
        from tensorflow.keras.models import load_model
        model        = load_model(f"{MODELS_DIR}/lstm_best.keras")
        X_test       = np.load(f"{MODELS_DIR}/X_test.npy")
        preds_scaled = model.predict(X_test, verbose=0).flatten()
        n_feat       = scaler.n_features_in_
        dummy        = np.zeros((len(preds_scaled), n_feat))
        dummy[:, 0]  = preds_scaled
        lstm_preds   = np.clip(scaler.inverse_transform(dummy)[:, 0], 0, None)
        print(f"✅ LSTM model      | {len(lstm_preds):,} predictions loaded")
    except Exception as e:
        print(f"⚠️  LSTM load skipped ({e})")
        lstm_preds = None

    print(f"✅ Dataset         | {len(df):,} records across {df['station'].nunique()} stations")
    return df, scaler, lstm_preds


# ─────────────────────────────────────────────
# BUILD SCHEDULE
# ─────────────────────────────────────────────
def build_schedule(df):
    """
    Aggregate to DAILY AVERAGE total demand per line per hour,
    then compute AI-optimised vs fixed schedule allocation.
    """
    # Step 1: sum all stations per day per line per hour → daily total
    daily = (
        df.groupby(["date", "line", "hour"])
        .agg(
            total_passengers = ("passengers", "sum"),
            is_raining       = ("is_raining",  "max"),
            is_festival      = ("festival",    lambda x: (x != "None").any()),
            is_weekend       = ("is_weekend",  "max"),
        )
        .reset_index()
    )

    # Step 2: average across all days → representative hourly demand
    agg = (
        daily.groupby(["line", "hour"])
        .agg(
            avg_demand   = ("total_passengers", "mean"),
            max_demand   = ("total_passengers", "max"),
            is_raining   = ("is_raining",       lambda x: x.mean() > 0.3),
            is_festival  = ("is_festival",      lambda x: x.any()),
            is_weekend   = ("is_weekend",       lambda x: x.mean() > 0.4),
        )
        .reset_index()
    )

    records = []
    for _, row in agg.iterrows():
        line    = row["line"]
        hour    = int(row["hour"])
        demand  = row["avg_demand"]

        # AI-optimised allocation
        trains_ai, headway_ai, wait_ai, svc = demand_to_trains(
            demand, hour,
            bool(row["is_raining"]),
            bool(row["is_festival"]),
            bool(row["is_weekend"]),
        )

        # Fixed schedule baseline
        trains_fixed, headway_fixed, wait_fixed = fixed_schedule_baseline(hour)

        # Metrics
        wait_saved      = round(wait_fixed - wait_ai, 2)
        wait_pct        = round(wait_saved / wait_fixed * 100, 1)
        utilisation     = round(demand / (trains_ai * EFFECTIVE_CAPACITY) * 100, 1)
        overcrowded     = "⚠️ Yes" if demand > trains_fixed * EFFECTIVE_CAPACITY else "✅ No"

        records.append({
            "line"                  : line,
            "hour"                  : hour,
            "time_slot"             : f"{hour:02d}:00–{hour+1:02d}:00",
            "total_demand"          : int(demand),
            "max_demand"            : int(row["max_demand"]),
            "period"                : "✅ Peak" if hour in PEAK_HOURS else
                                      "Shoulder" if hour in SHOULDER_HOURS else "Off-Peak",
            "rain_surge"            : "🌧️ Yes" if row["is_raining"]  else "No",
            "festival_surge"        : "🎉 Yes" if row["is_festival"] else "No",
            # Fixed (before AI)
            "trains_fixed"          : trains_fixed,
            "headway_fixed_mins"    : headway_fixed,
            "wait_fixed_mins"       : wait_fixed,
            "overcrowding_risk"     : overcrowded,
            # AI optimised (after)
            "trains_ai"             : trains_ai,
            "headway_ai_mins"       : headway_ai,
            "wait_ai_mins"          : wait_ai,
            "service_level"         : svc,
            "fleet_utilisation_%"   : utilisation,
            # Delta
            "wait_saved_mins"       : wait_saved,
            "wait_improvement_%"    : wait_pct,
            "extra_trains"          : trains_ai - trains_fixed,
        })

    return pd.DataFrame(records).sort_values(["line", "hour"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────
def print_summary(df):
    print(f"\n{'═'*60}")
    print(f"  FLEET ORCHESTRATION RESULTS  (Pune Metro)")
    print(f"{'═'*60}")

    avg_fixed = df["wait_fixed_mins"].mean()
    avg_ai    = df["wait_ai_mins"].mean()
    pct       = (avg_fixed - avg_ai) / avg_fixed * 100

    peak  = df[df["period"] == "✅ Peak"]
    offpk = df[df["period"] == "Off-Peak"]

    print(f"""
  System-Wide (All Hours)
  ────────────────────────────────────────────────
  Fixed schedule avg wait  : {avg_fixed:.1f} mins
  AI-optimised avg wait    : {avg_ai:.1f} mins
  Overall improvement      : {pct:+.1f}%

  Peak Hours (8–11 AM, 4–8 PM)
  ────────────────────────────────────────────────
  Fixed avg wait (peak)    : {peak['wait_fixed_mins'].mean():.1f} mins
  AI avg wait   (peak)     : {peak['wait_ai_mins'].mean():.1f} mins
  Avg extra trains          : {peak['extra_trains'].mean():+.1f} per hour slot

  Off-Peak Hours
  ────────────────────────────────────────────────
  Fixed avg wait (off-peak): {offpk['wait_fixed_mins'].mean():.1f} mins
  AI avg wait   (off-peak) : {offpk['wait_ai_mins'].mean():.1f} mins
  Avg trains saved (idle)  : {offpk['extra_trains'].mean():+.1f} per hour slot
""")

    # Per line table
    print(f"  {'Line':<13} {'Avg Demand':>11} {'Fixed Wait':>11} "
          f"{'AI Wait':>9} {'Saved':>7} {'Trains AI':>11}")
    print(f"  {'─'*62}")
    for line in ["Aqua", "Purple", "Interchange"]:
        s = df[df["line"] == line]
        print(f"  {line:<13} {s['total_demand'].mean():>11,.0f} "
              f"{s['wait_fixed_mins'].mean():>11.1f} "
              f"{s['wait_ai_mins'].mean():>9.1f} "
              f"{s['wait_saved_mins'].mean():>7.1f} "
              f"{s['trains_ai'].mean():>11.1f}")

    # PPT-ready before vs after table
    print(f"\n  📋 Before vs After — Peak Hour (for PPT)")
    print(f"  {'─'*62}")
    print(f"  {'Route':<13} {'Trains B':>9} {'Wait B':>7} │ "
          f"{'Trains A':>9} {'Wait A':>7} {'Saved':>7}")
    print(f"  {'─'*62}")
    for line in ["Aqua", "Purple", "Interchange"]:
        s = df[(df["line"] == line) & (df["period"] == "✅ Peak")]
        if len(s):
            print(f"  {line:<13} "
                  f"{s['trains_fixed'].mean():>9.0f} "
                  f"{s['wait_fixed_mins'].mean():>6.1f}m │ "
                  f"{s['trains_ai'].mean():>9.0f} "
                  f"{s['wait_ai_mins'].mean():>6.1f}m "
                  f"{s['wait_saved_mins'].mean():>6.1f}m")
    print()


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def save_plots(df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes.flat:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    fig.suptitle("SmartTransit AI – Fleet Orchestration Dashboard",
                 fontsize=15, fontweight="bold", color="white")

    LC    = {"Aqua": "#00bcd4", "Purple": "#ce93d8", "Interchange": "#ffb74d"}
    hours = sorted(df["hour"].unique())

    # ── Plot 1: Wait time Before vs After ─────
    ax1 = axes[0, 0]
    for line in ["Aqua", "Purple", "Interchange"]:
        s = df[df["line"] == line].sort_values("hour")
        ax1.plot(s["hour"], s["wait_fixed_mins"],
                 color=LC[line], linewidth=1.5, linestyle="--", alpha=0.5)
        ax1.plot(s["hour"], s["wait_ai_mins"],
                 color=LC[line], linewidth=2.5, label=line, marker="o", markersize=4)
        ax1.fill_between(s["hour"], s["wait_fixed_mins"], s["wait_ai_mins"],
                         color=LC[line], alpha=0.1)
    ax1.axvspan(7.5,  11.5, alpha=0.07, color="red",    label="Peak AM")
    ax1.axvspan(15.5, 20.5, alpha=0.07, color="orange", label="Peak PM")
    ax1.set_title("⏱ Wait Time: Fixed (dashed) vs AI (solid)", color="white")
    ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("Avg Wait (mins)")
    ax1.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)
    ax1.grid(alpha=0.15, color="white"); ax1.set_xticks(hours)
    ax1.set_ylim(0, 10)

    # ── Plot 2: Trains deployed per hour ──────
    ax2 = axes[0, 1]
    w = 0.25
    x = np.arange(len(hours))
    for i, line in enumerate(["Aqua", "Purple", "Interchange"]):
        s = df[df["line"] == line].sort_values("hour")
        ax2.bar(x + i*w, s["trains_ai"].values,
                width=w, label=line, color=LC[line], alpha=0.85)
    ax2.axhline(10, color="#ef5350", ls="--", lw=1.5, label="Fixed Peak (10)", alpha=0.7)
    ax2.axhline(5,  color="#ff9800", ls=":",  lw=1.5, label="Fixed Off-Peak (5)", alpha=0.7)
    ax2.set_title("🚆 AI Trains Deployed vs Fixed Schedule", color="white")
    ax2.set_xlabel("Hour of Day"); ax2.set_ylabel("Trains / Hour")
    ax2.set_xticks(x + w); ax2.set_xticklabels(hours, fontsize=8)
    ax2.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=7)
    ax2.grid(axis="y", alpha=0.15, color="white")

    # ── Plot 3: Demand heatmap ─────────────────
    ax3 = axes[1, 0]
    pivot = df.pivot_table(index="line", columns="hour",
                           values="total_demand", aggfunc="mean")
    im = ax3.imshow(pivot.values, cmap="plasma", aspect="auto")
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels(pivot.columns, fontsize=8, color="white")
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index, fontsize=9, color="white")
    ax3.set_title("🔥 Total Demand Heatmap (Passengers/Line/Hour)", color="white")
    ax3.set_xlabel("Hour of Day")
    cbar = plt.colorbar(im, ax=ax3)
    cbar.ax.set_ylabel("Passengers", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # ── Plot 4: Before vs After per line ──────
    ax4 = axes[1, 1]
    lines = ["Aqua", "Purple", "Interchange"]
    x4 = np.arange(len(lines))
    w4 = 0.35
    fixed_w = [df[df["line"] == l]["wait_fixed_mins"].mean() for l in lines]
    ai_w    = [df[df["line"] == l]["wait_ai_mins"].mean()    for l in lines]
    b1 = ax4.bar(x4 - w4/2, fixed_w, w4, label="Fixed Schedule",
                 color="#ef5350", alpha=0.85)
    b2 = ax4.bar(x4 + w4/2, ai_w,   w4, label="AI Optimised",
                 color="#66bb6a", alpha=0.85)
    for b in list(b1) + list(b2):
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                 f"{b.get_height():.1f}m", ha="center",
                 fontsize=10, color="white", fontweight="bold")
    ax4.set_title("📊 Avg Wait: Before vs After per Line", color="white")
    ax4.set_ylabel("Avg Wait Time (mins)")
    ax4.set_xticks(x4); ax4.set_xticklabels(lines, color="white")
    ax4.legend(facecolor="#1a1d2e", labelcolor="white")
    ax4.grid(axis="y", alpha=0.15, color="white")

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/fleet_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"✅ Fleet dashboard  → {out}")


# ─────────────────────────────────────────────
# REAL-TIME FUNCTION (used by Streamlit dashboard)
# ─────────────────────────────────────────────
def get_realtime_allocation(line, hour, total_line_demand,
                             is_raining=False, is_festival=False, is_weekend=False):
    trains_ai, headway_ai, wait_ai, svc = demand_to_trains(
        total_line_demand, hour, is_raining, is_festival, is_weekend
    )
    trains_fixed, headway_fixed, wait_fixed = fixed_schedule_baseline(hour)
    return {
        "line"              : line,
        "hour"              : hour,
        "total_demand"      : int(total_line_demand),
        "trains_fixed"      : trains_fixed,
        "wait_fixed_mins"   : wait_fixed,
        "trains_ai"         : trains_ai,
        "headway_ai_mins"   : headway_ai,
        "wait_ai_mins"      : wait_ai,
        "service_level"     : svc,
        "wait_saved_mins"   : round(wait_fixed - wait_ai, 2),
        "improvement_%"     : round((wait_fixed - wait_ai) / wait_fixed * 100, 1),
        "fleet_utilisation" : round(total_line_demand / (trains_ai * EFFECTIVE_CAPACITY) * 100, 1),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_raw, scaler, lstm_preds = load_inputs()
    schedule_df = build_schedule(df_raw)

    out_csv = f"{OUTPUT_DIR}/fleet_schedule.csv"
    schedule_df.to_csv(out_csv, index=False)
    print(f"✅ Fleet schedule   → {out_csv}")
    print(f"   {len(schedule_df)} route × hour slots optimised\n")

    print_summary(schedule_df)
    save_plots(schedule_df)

    print("🏁 Fleet orchestration complete!")
    print("   Next step → python Dashboard/app.py\n")