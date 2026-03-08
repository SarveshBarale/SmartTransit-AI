import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Time-slot definitions ─────────────────────────────────────────────────────
SLOTS = {
    "morning_peak":    {"hours": range(7,  11), "label": "Morning Peak",    "emoji": "🌅", "color": "#FF6B35"},
    "afternoon":       {"hours": range(11, 17), "label": "Afternoon",       "emoji": "☀️",  "color": "#FFD166"},
    "evening_peak":    {"hours": range(17, 21), "label": "Evening Peak",    "emoji": "🌆", "color": "#EF476F"},
    "night":           {"hours": range(21, 24), "label": "Night Off-Peak",  "emoji": "🌙", "color": "#118AB2"},
}
WEEKEND_DAYS = {5, 6}   # Saturday=5, Sunday=6

SEQ_LEN      = 24       # must match training
FEATURE_COLS = [
    "passenger_demand", "hour", "day_of_week",
    "is_weekend", "is_morning_peak", "is_evening_peak", "fleet_required",
]


# ── Helper: feature flags ─────────────────────────────────────────────────────

def _add_feature_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["is_weekend"]      = df["day_of_week"].isin(WEEKEND_DAYS).astype(int)
    df["is_morning_peak"] = df["hour"].between(7, 10).astype(int)
    df["is_evening_peak"] = df["hour"].between(17, 20).astype(int)
    if "fleet_required" not in df.columns:
        df["fleet_required"] = (df["passenger_demand"] / 80).clip(lower=1).astype(int)
    return df


def _slot_for_hour(hour: int, is_weekend: bool) -> str:
    if is_weekend:
        return "weekend"
    for slot, meta in SLOTS.items():
        if hour in meta["hours"]:
            return slot
    return "night"


# ── Main class ────────────────────────────────────────────────────────────────

class DemandSegmentor:
    """
    Loads a trained LSTM + scaler, then predicts and segments demand
    across all defined time slots.
    """

    def __init__(
        self,
        model_path:  str = "Models/lstm_best.keras",
        scaler_path: str = "Models/scaler.pkl",
        config_path: str = "Data/stations_config.json",
    ):
        self.model_path  = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.config_path = Path(config_path)

        self.model  = None
        self.scaler = None
        self.config = None

        self._load_assets()

    # ── Asset loading ─────────────────────────────────────────────────────────

    def _load_assets(self):
        # Config
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = json.load(f)
        else:
            print(f"⚠  Config not found at {self.config_path} — station names unavailable.")

        # Scaler
        if self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            print(f"⚠  Scaler not found at {self.scaler_path} — will use raw predictions.")

        # LSTM model (optional — segmentor can work on raw data too)
        if self.model_path.exists():
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(self.model_path)
                print(f"✅ LSTM loaded from {self.model_path}")
            except Exception as e:
                print(f"⚠  Could not load LSTM model: {e}")
        else:
            print(f"⚠  Model not found at {self.model_path} — using historical data only.")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_next_24h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given the last SEQ_LEN hours of data, predict the next 24 hours
        and return a DataFrame with slot labels attached.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model and scaler must be loaded for prediction.")

        df = _add_feature_flags(df)
        data = df[FEATURE_COLS].values[-SEQ_LEN:].astype(float)

        if len(data) < SEQ_LEN:
            raise ValueError(f"Need at least {SEQ_LEN} rows; got {len(data)}.")

        data_scaled = self.scaler.transform(data)
        predictions = []
        window = data_scaled.copy()

        for step in range(24):
            x      = window[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLS))
            y_pred = self.model.predict(x, verbose=0)[0][0]

            # Build next-step feature row
            last_row   = window[-1].copy()
            next_hour  = (df["hour"].iloc[-1] + step + 1) % 24
            next_dow   = df["day_of_week"].iloc[-1] if next_hour > 0 else (df["day_of_week"].iloc[-1] + 1) % 7

            # Unscale to get absolute demand
            dummy          = last_row.copy()
            dummy[0]       = y_pred
            dummy_2d       = dummy.reshape(1, -1)
            unscaled       = self.scaler.inverse_transform(dummy_2d)[0]
            demand_abs     = max(0, unscaled[0])

            predictions.append({
                "step":            step + 1,
                "hour":            next_hour,
                "day_of_week":     next_dow,
                "is_weekend":      int(next_dow in WEEKEND_DAYS),
                "is_morning_peak": int(7 <= next_hour <= 10),
                "is_evening_peak": int(17 <= next_hour <= 20),
                "predicted_demand": round(demand_abs),
                "fleet_required":  max(1, int(demand_abs / 80)),
                "slot":            _slot_for_hour(next_hour, next_dow in WEEKEND_DAYS),
            })

            # Slide window
            next_row     = last_row.copy()
            next_row[0]  = y_pred
            window       = np.vstack([window, next_row])

        return pd.DataFrame(predictions)

    # ── Segmentation on historical / existing data ────────────────────────────

    def segment_historical(self, df: pd.DataFrame) -> dict:
        """
        Segments an existing demand DataFrame into time slots.
        Returns per-slot statistics.
        """
        df = _add_feature_flags(df)
        df["slot"] = df.apply(
            lambda r: _slot_for_hour(r["hour"], bool(r["is_weekend"])), axis=1
        )

        results = {}

        # Regular weekday slots
        for slot_key, meta in SLOTS.items():
            subset = df[df["slot"] == slot_key]["passenger_demand"]
            results[slot_key] = {
                "label":          meta["label"],
                "emoji":          meta["emoji"],
                "color":          meta["color"],
                "mean_demand":    round(subset.mean(), 1) if len(subset) else 0,
                "max_demand":     round(subset.max(),  1) if len(subset) else 0,
                "min_demand":     round(subset.min(),  1) if len(subset) else 0,
                "total_demand":   round(subset.sum(),  1) if len(subset) else 0,
                "sample_count":   len(subset),
            }

        # Weekend as its own segment
        wk = df[df["is_weekend"] == 1]["passenger_demand"]
        results["weekend"] = {
            "label":        "Weekend",
            "emoji":        "📅",
            "color":        "#06D6A0",
            "mean_demand":  round(wk.mean(), 1) if len(wk) else 0,
            "max_demand":   round(wk.max(),  1) if len(wk) else 0,
            "min_demand":   round(wk.min(),  1) if len(wk) else 0,
            "total_demand": round(wk.sum(),  1) if len(wk) else 0,
            "sample_count": len(wk),
        }

        return results

    # ── Per-station segmentation ──────────────────────────────────────────────

    def segment_by_station(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with mean demand per station per slot.
        Great for heatmap / bar chart in the dashboard.
        """
        df = _add_feature_flags(df)
        df["slot"] = df.apply(
            lambda r: _slot_for_hour(r["hour"], bool(r["is_weekend"])), axis=1
        )

        pivot = (
            df.groupby(["station_id", "station_name", "slot"])["passenger_demand"]
            .mean()
            .round(1)
            .reset_index()
            .rename(columns={"passenger_demand": "mean_demand"})
        )
        return pivot

    # ── High-demand alerts ────────────────────────────────────────────────────

    def high_demand_stations(
        self, df: pd.DataFrame, slot: str = "morning_peak", top_n: int = 5
    ) -> pd.DataFrame:
        """
        Returns the top-N busiest stations for a given time slot.
        Used for the dashboard's 'high demand areas' requirement.
        """
        df = _add_feature_flags(df)
        df["slot"] = df.apply(
            lambda r: _slot_for_hour(r["hour"], bool(r["is_weekend"])), axis=1
        )

        slot_df = df[df["slot"] == slot]
        top = (
            slot_df.groupby(["station_id", "station_name"])["passenger_demand"]
            .mean()
            .round(1)
            .reset_index()
            .sort_values("passenger_demand", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        top["rank"] = top.index + 1
        return top

    # ── Full report ───────────────────────────────────────────────────────────

    def full_report(self, df: pd.DataFrame) -> dict:
        """
        Returns a complete demand report dict:
        {
          'segments':         { slot -> stats },
          'by_station':       DataFrame,
          'high_demand_morning': DataFrame (top 5),
          'high_demand_evening': DataFrame (top 5),
        }
        """
        return {
            "segments":              self.segment_historical(df),
            "by_station":            self.segment_by_station(df),
            "high_demand_morning":   self.high_demand_stations(df, slot="morning_peak"),
            "high_demand_evening":   self.high_demand_stations(df, slot="evening_peak"),
        }


# ── Standalone demo / test ────────────────────────────────────────────────────

def _generate_demo_data(n_hours: int = 720) -> pd.DataFrame:
    """Generates synthetic Pune Metro demand data for testing."""
    np.random.seed(42)
    stations = [
        ("PU01", "PCMC Bhavan"),       ("PU10", "Shivaji Nagar"),
        ("PU11", "Civil Court"),       ("PU14", "Swargate"),
        ("AQ05", "Garware College"),   ("AQ08", "PMC"),
        ("AQ11", "Pune Railway Stn"),  ("AQ16", "Ramwadi"),
    ]
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    rows = []
    for ts in timestamps:
        hour  = ts.hour
        is_wk = ts.dayofweek >= 5
        mp    = 3.5 * np.exp(-0.5 * ((hour - 8.5) / 1.2) ** 2)
        ep    = 3.2 * np.exp(-0.5 * ((hour - 18.0) / 1.5) ** 2)
        mult  = max(0.2, mp + ep) if not is_wk else 0.55
        for sid, sname in stations:
            demand = max(0, int(120 * mult + np.random.normal(0, 15)))
            rows.append({
                "timestamp":        ts,
                "station_id":       sid,
                "station_name":     sname,
                "hour":             hour,
                "day_of_week":      ts.dayofweek,
                "is_weekend":       int(is_wk),
                "is_morning_peak":  int(7 <= hour <= 10),
                "is_evening_peak":  int(17 <= hour <= 20),
                "passenger_demand": demand,
                "fleet_required":   max(1, demand // 80),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 55)
    print("  Pune Metro — Demand Segmentation Demo")
    print("=" * 55)

    df = _generate_demo_data()
    ds = DemandSegmentor()   # model/scaler optional for historical segmentation

    report = ds.full_report(df)

    print("\n📊 Demand by Time Slot")
    print("-" * 55)
    for key, stats in report["segments"].items():
        print(f"  {stats['emoji']}  {stats['label']:<22}"
              f"  avg={stats['mean_demand']:>6}  max={stats['max_demand']:>6}")

    print("\n🔥 Top 5 Busiest Stations — Morning Peak")
    print("-" * 55)
    print(report["high_demand_morning"].to_string(index=False))

    print("\n🔥 Top 5 Busiest Stations — Evening Peak")
    print("-" * 55)
    print(report["high_demand_evening"].to_string(index=False))
