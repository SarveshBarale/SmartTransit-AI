# 🚇 SmartTransit AI — Pune Metro Fleet Orchestration

**Hackathon: Coruscant Transit Command · Problem Statement PS4**
&nbsp;·&nbsp; SDG 11 (Sustainable Cities) · SDG 7 (Clean Energy) · SDG 9 (Innovation) · SDG 13 (Climate Action)

---

SmartTransit AI is a real-time public transport demand forecasting and fleet optimization system built for the Pune Metro network. The system combines a two-layer LSTM neural network (trained on 180,000+ hourly records across 30 stations, Purple and Aqua lines) with an ARIMA baseline for benchmarking, achieving an RMSE of **262.94** — a **95.2% improvement** over the statistical baseline. Predictions feed directly into a live fleet orchestrator that computes the optimal number of trains per line per hour, accounting for real-world context: rain surges (×1.22), festival days (×1.55), and weekend patterns (×1.10). Against a fixed timetable of 10 trains during peak hours, the AI dynamically deploys up to 15 trains (signalling ceiling) when demand justifies it, reducing average passenger wait time from **3.0 minutes to ~2.0 minutes** during peak windows — a 33% reduction.

The optimization engine extends beyond single-objective scheduling into a full **multi-objective fleet optimizer** that simultaneously balances three competing goals: minimizing passenger wait time, minimizing energy consumption (modelled at 8.5 kWh/km for Pune Metro's 6-car Alstom Metropolis sets), and maximizing network coverage. Weights are auto-selected by time-of-day context — peak hours prioritize time (65%) and coverage (25%), while off-peak hours shift priority to fuel efficiency (60%) — eliminating any need for manual tuning. A **Dijkstra-based route optimizer** across the full 30-station graph computes shortest paths with transfer penalties at the Civil Court interchange, and a **per-station pickup optimizer** surfaces congestion hotspots and priority rankings across both lines in real time.

All components surface through an 8-page **Streamlit dashboard** with live controls for hour-of-day, rain/festival/weekend context, and an interactive Folium demand heatmap. The system directly addresses the problem statement's core gaps: 30–40% of transit running underutilized, wait times that could be cut by 35%, and operational costs 25% above optimal — while contributing to decarbonization by reducing idle train energy waste during off-peak periods. The codebase is fully modular (`Algorithms/`, `Dashboard/`, `fleet/`) with trained model artifacts checked in, making the system immediately runnable with `streamlit run Dashboard/app.py`.

---

**Tech Stack:** Python · TensorFlow/Keras · Streamlit · Plotly · Folium · Statsmodels · scikit-learn · Dijkstra (custom)
**Lines Covered:** Purple (14 stations, 16.59 km) · Aqua (16 stations, 14.66 km) · Civil Court interchange
**Key Results:** LSTM RMSE 262.94 · Peak wait ↓33% · Network coverage up to 100% at peak · Multi-objective Pareto-front optimization
