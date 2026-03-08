import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS_DIR   = "Models"
OUTPUT_DIR   = "Outputs"
EPOCHS       = 20
BATCH_SIZE   = 512
LEARNING_RATE = 0.001

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 – LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────
def load_data():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  SmartTransit AI – LSTM Training")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    X_train = np.load(f"{MODELS_DIR}/X_train.npy")
    y_train = np.load(f"{MODELS_DIR}/y_train.npy")
    X_val   = np.load(f"{MODELS_DIR}/X_val.npy")
    y_val   = np.load(f"{MODELS_DIR}/y_val.npy")
    X_test  = np.load(f"{MODELS_DIR}/X_test.npy")
    y_test  = np.load(f"{MODELS_DIR}/y_test.npy")

    with open(f"{MODELS_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"✅ Data loaded")
    print(f"   X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"   X_val   : {X_val.shape}  |  y_val   : {y_val.shape}")
    print(f"   X_test  : {X_test.shape}  |  y_test  : {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ─────────────────────────────────────────────
# STEP 2 – BUILD LSTM MODEL
# ─────────────────────────────────────────────
def build_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization, Input
    )
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=input_shape),

        # First LSTM layer – captures broad temporal patterns
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),

        # Second LSTM layer – captures fine-grained patterns
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        # Dense layers for final prediction
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1)   # output: scaled passenger count
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="huber",          # robust to outliers vs plain MSE
        metrics=["mae"]
    )

    print(f"\n✅ Model built")
    model.summary()
    return model


# ─────────────────────────────────────────────
# STEP 3 – TRAIN
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val):
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"{MODELS_DIR}/lstm_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print(f"\n🚀 Training started  (epochs={EPOCHS}, batch={BATCH_SIZE})\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Also save in legacy .h5 format for compatibility
    model.save(f"{MODELS_DIR}/lstm_model.h5")
    print(f"\n✅ Model saved → {MODELS_DIR}/lstm_model.h5")
    print(f"✅ Best model  → {MODELS_DIR}/lstm_best.keras")

    return history


# ─────────────────────────────────────────────
# STEP 4 – EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scaler):
    print("\n📊 Evaluating on test set...")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform: rebuild full-width array for scaler
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(y_test), n_features))
    dummy[:, 0] = y_test
    y_test_real = scaler.inverse_transform(dummy)[:, 0]

    dummy[:, 0] = y_pred_scaled
    y_pred_real = scaler.inverse_transform(dummy)[:, 0]

    # Clip negatives (passengers can't be negative)
    y_pred_real = np.clip(y_pred_real, 0, None)

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae  = mean_absolute_error(y_test_real, y_pred_real)
    mape = np.mean(np.abs((y_test_real - y_pred_real) /
                          np.clip(y_test_real, 1, None))) * 100

    print(f"\n{'─'*40}")
    print(f"  LSTM Model Performance")
    print(f"{'─'*40}")
    print(f"  RMSE : {rmse:>10.2f}  passengers")
    print(f"  MAE  : {mae:>10.2f}  passengers")
    print(f"  MAPE : {mape:>10.2f} %")
    print(f"{'─'*40}\n")

    return y_test_real, y_pred_real, rmse, mae, mape


# ─────────────────────────────────────────────
# STEP 5 – PLOT & SAVE CHARTS
# ─────────────────────────────────────────────
def save_plots(history, y_test_real, y_pred_real, rmse, mae):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("SmartTransit AI – LSTM Results", fontsize=15, fontweight="bold")

    # ── Plot 1: Training curves ──────────────────
    ax1 = axes[0]
    ax1.plot(history.history["loss"],     label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val Loss",   color="#ff7f0e",
             linewidth=2, linestyle="--")
    ax1.set_title("Training vs Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Huber Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Plot 2: Actual vs Predicted ───────────────
    ax2 = axes[1]
    sample = min(500, len(y_test_real))
    ax2.plot(y_test_real[:sample], label="Actual",    color="#2ca02c", linewidth=1.5)
    ax2.plot(y_pred_real[:sample], label="Predicted", color="#d62728",
             linewidth=1.5, linestyle="--", alpha=0.85)
    ax2.set_title(f"Actual vs Predicted Passengers\nRMSE={rmse:.1f} | MAE={mae:.1f}")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Passengers")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/lstm_results.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Charts saved → {out_path}")


# ─────────────────────────────────────────────
# STEP 6 – SAVE PREDICTIONS CSV
# ─────────────────────────────────────────────
def save_predictions(y_test_real, y_pred_real, rmse, mae, mape):
    import pandas as pd

    df = pd.DataFrame({
        "actual_passengers":    y_test_real.astype(int),
        "predicted_passengers": y_pred_real.astype(int),
        "error":                (y_test_real - y_pred_real).astype(int),
    })
    out_path = f"{OUTPUT_DIR}/lstm_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Predictions saved → {out_path}")

    # Also save metrics summary
    metrics = pd.DataFrame([{"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE": round(mape, 2)}])
    metrics.to_csv(f"{OUTPUT_DIR}/lstm_metrics.csv", index=False)
    print(f"✅ Metrics saved    → {OUTPUT_DIR}/lstm_metrics.csv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data()

    input_shape = (X_train.shape[1], X_train.shape[2])   # (24, 16)
    model       = build_model(input_shape)

    history     = train_model(model, X_train, y_train, X_val, y_val)

    y_test_real, y_pred_real, rmse, mae, mape = evaluate_model(model, X_test, y_test, scaler)

    save_plots(history, y_test_real, y_pred_real, rmse, mae)
    save_predictions(y_test_real, y_pred_real, rmse, mae, mape)

    print("\n🏁 LSTM pipeline complete!")
    print(f"   Next step → run: python Algorithms/arima_baseline.py")
    print(f"   Then      → run: python fleet/orchestrator.py\n")