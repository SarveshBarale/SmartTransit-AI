import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "Data/pune_metro_enhanced_data.csv"
OUTPUT_DIR   = "Outputs"
MODELS_DIR   = "Models"
SEQ_LENGTH   = 24   # look-back window: 24 hours of history to predict next hour

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 – LOAD DATA
# ─────────────────────────────────────────────
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df):,} records | Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# STEP 2 – CLEAN & BASIC FIXES
# ─────────────────────────────────────────────
def clean_data(df):
    # Parse date and build a full datetime column
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(subset=["datetime", "station"], inplace=True)
    print(f"✅ Dropped {before - len(df)} duplicate rows")

    # Clip negative passenger values (data sanity)
    df["passengers"] = df["passengers"].clip(lower=0)

    # Fill any residual nulls (none expected but safe)
    num_cols = df.select_dtypes(include=[np.number]).columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    df[str_cols] = df[str_cols].fillna("None")

    return df


# ─────────────────────────────────────────────
# STEP 3 – FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    # Time features
    df["day_of_week"]  = df["datetime"].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"]        = df["datetime"].dt.month
    df["day_of_year"]  = df["datetime"].dt.dayofyear

    # Peak hour flag  (morning 8-10, evening 17-20)
    df["is_peak_hour"] = df["hour"].apply(
        lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 20) else 0
    )

    # Cyclical encoding for hour and day_of_week (so 23→0 wraps smoothly)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"]       / 12)

    # Festival → binary spike flag
    df["is_festival"]  = (df["festival"] != "None").astype(int)

    # Encode line as integer
    line_map = {"Aqua": 0, "Purple": 1, "Interchange": 2}
    df["line_enc"] = df["line"].map(line_map).fillna(0).astype(int)

    # Encode station as integer
    stations = sorted(df["station"].unique())
    station_map = {s: i for i, s in enumerate(stations)}
    df["station_enc"] = df["station"].map(station_map).astype(int)

    # Save maps for later use in dashboard / fleet logic
    with open(f"{MODELS_DIR}/station_map.pkl", "wb") as f:
        pickle.dump(station_map, f)
    with open(f"{MODELS_DIR}/line_map.pkl", "wb") as f:
        pickle.dump(line_map, f)

    print(f"✅ Feature engineering done | Shape: {df.shape}")
    return df, station_map


# ─────────────────────────────────────────────
# STEP 4 – SCALE FEATURES
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "passengers",          # target (also used as input in sequence)
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_peak_hour",
    "is_weekend",
    "is_festival",
    "is_raining",
    "temp_c",
    "precipitation_mm",
    "traffic_index",
    "line_enc",
    "station_enc",
]

def scale_features(df):
    scaler = MinMaxScaler()
    df_scaled = df[FEATURE_COLS].copy()
    df_scaled[FEATURE_COLS] = scaler.fit_transform(df_scaled[FEATURE_COLS])

    # Save scaler so we can inverse-transform predictions later
    with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Scaling done | Scaler saved to {MODELS_DIR}/scaler.pkl")
    return df_scaled, scaler


# ─────────────────────────────────────────────
# STEP 5 – BUILD LSTM SEQUENCES
# ─────────────────────────────────────────────
def build_sequences(df_scaled, station, station_map, seq_length=SEQ_LENGTH):
    """
    For a given station, build (X, y) pairs where:
      X = [seq_length hours of all features]
      y = passengers in the next hour
    """
    station_idx = station_map[station]
    station_data = df_scaled[df_scaled["station_enc"] == station_idx / max(station_map.values())]

    # Use all feature columns as model input
    data = station_data[FEATURE_COLS].values

    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])   # past seq_length hours
        y.append(data[i][0])               # index 0 = passengers (scaled)

    return np.array(X), np.array(y)


def build_all_sequences(df_scaled, station_map, seq_length=SEQ_LENGTH):
    """Build sequences for ALL stations combined (for a global model)."""
    data = df_scaled[FEATURE_COLS].values
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i][0])
    X, y = np.array(X), np.array(y)
    print(f"✅ Sequences built | X: {X.shape}, y: {y.shape}")
    return X, y


# ─────────────────────────────────────────────
# STEP 6 – TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
def split_data(X, y, train=0.7, val=0.15):
    n = len(X)
    t = int(n * train)
    v = int(n * (train + val))

    X_train, y_train = X[:t],  y[:t]
    X_val,   y_val   = X[t:v], y[t:v]
    X_test,  y_test  = X[v:],  y[v:]

    print(f"✅ Split → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────
# MAIN – run full pipeline
# ─────────────────────────────────────────────
def run_preprocessing():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  SmartTransit AI – Preprocessing")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    df                       = load_data()
    df                       = clean_data(df)
    df, station_map          = engineer_features(df)
    df_scaled, scaler        = scale_features(df)
    X, y                     = build_all_sequences(df_scaled, station_map)
    X_train, y_train, \
    X_val, y_val, \
    X_test, y_test           = split_data(X, y)

    # Save processed arrays for use in lstm_model.py
    np.save(f"{MODELS_DIR}/X_train.npy", X_train)
    np.save(f"{MODELS_DIR}/y_train.npy", y_train)
    np.save(f"{MODELS_DIR}/X_val.npy",   X_val)
    np.save(f"{MODELS_DIR}/y_val.npy",   y_val)
    np.save(f"{MODELS_DIR}/X_test.npy",  X_test)
    np.save(f"{MODELS_DIR}/y_test.npy",  y_test)

    print(f"\n✅ All arrays saved to {MODELS_DIR}/")
    print(f"   Input shape  : {X_train.shape}  → (samples, {SEQ_LENGTH} hours, {len(FEATURE_COLS)} features)")
    print(f"   Output shape : {y_train.shape}  → (samples, 1 passenger count)")
    print("\n🚀 Preprocessing complete! Run lstm_model.py next.\n")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, station_map


if __name__ == "__main__":
    run_preprocessing()