# 🚇 SmartTransit AI — Pune Metro Fleet Orchestration

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://smarttransit-ai.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**SmartTransit AI** is a real-time public transport demand forecasting and fleet optimization system built for the Pune Metro network. It integrates deep learning with multi-objective optimization to reduce passenger wait times, minimize energy waste, and solve the gaps of underutilized transit capacity.

## 🔗 Live Deployment
### **[Click here to view the Live Dashboard](https://smarttransit-ai.streamlit.app/)**

---

## ## Key Results & Impact
* **LSTM Accuracy:** Achieved an **RMSE of 262.94** — a **95.2% improvement** over statistical baselines.
* **Wait Time Reduction:** Decreased average peak-hour wait times from **3.0 to ~2.0 minutes** (33% reduction).
* **Dynamic Scaling:** Automatically scales fleet deployment (up to 15 trains) based on:
    * 🌧️ **Rain Surges:** ×1.22
    * 🎉 **Festival Days:** ×1.55
    * 📅 **Weekend Patterns:** ×1.10

---

## ## System Features

### 1. **Hybrid Demand Forecasting**
Trained on **180,000+ hourly records** across 30 stations (Purple and Aqua lines). The system uses a two-layer LSTM architecture to predict commuter inflow with high precision.

### 2. **Multi-Objective Fleet Optimizer**
A sophisticated engine that balances three competing goals simultaneously:
* **Minimizing Wait Time:** Prioritized during peak windows.
* **Energy Efficiency:** Modeled at **8.5 kWh/km** for Alstom Metropolis sets; prioritized during off-peak to prevent idle waste.
* **Network Coverage:** Ensures maximum reach across both lines.

### 3. **Intelligent Routing**
* **Dijkstra-based Optimizer:** Computes shortest paths across the 30-station graph.
* **Transfer Logic:** Accounts for penalties at the **Civil Court interchange**.
* **Pickup Optimizer:** Identifies real-time congestion hotspots for station-level priority ranking.

---

## ## Tech Stack
* **Languages:** Python
* **Deep Learning:** TensorFlow / Keras (LSTM)
* **Data Science:** Pandas, NumPy, Scikit-learn, Statsmodels (ARIMA)
* **Visualization:** Streamlit, Plotly, Folium (Interactive Heatmaps)
* **Optimization:** Custom Dijkstra Implementation

---

## ## Project Structure
smarttransit-ai/
├── Algorithms/    # LSTM/ARIMA models & Route optimizers
├── Dashboard/     # 8-page Streamlit UI source code
├── Data/          # Metro records & station metadata
├── fleet/         # Fleet orchestrator & capacity logic
├── requirements.txt
└── .python-version

## Local Setup & Installation
Clone the repository

Bash
git clone [https://github.com/your-username/smarttransit-ai.git](https://github.com/SarveshBarale/smarttransit-ai.git)
cd smarttransit-ai
Install Dependencies

Bash
pip install -r requirements.txt
Run the Application

Bash
streamlit run Dashboard/app.py

## Hackathon Context
Problem Statement: PS4 — Coruscant Transit Command.

SDG Goals: 11 (Sustainable Cities), 7 (Clean Energy), 9 (Innovation), 13 (Climate Action).

License: MIT License.
