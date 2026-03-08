"""
pickup_optimizer.py
SmartTransit AI – Passenger Pickup & Fleet Allocation Optimizer
Calibrated to match orchestrator.py constants exactly.

KEY DESIGN:
  - Trains run the FULL LINE — wait time is determined by LINE-level headway
  - headway = 60 / line_trains  (same as orchestrator)
  - avg_wait = headway / 2      (same as orchestrator)
  - Per-station output shows demand + utilisation, but wait = line wait
  - This matches real metro operations: more trains = shorter headway for ALL stations

Usage:
    from Algorithms.pickup_optimizer import PickupOptimizer
    opt    = PickupOptimizer()
    result = opt.optimize(demand_dict, hour=8)
    print(result.summary())
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).parent.parent / "Data" / "stations_config.json"

# ── Constants — identical to orchestrator.py ─────────────────────────────────
TRAIN_CAPACITY      = 1800
MAX_OCCUPANCY       = 0.90
EFFECTIVE_CAPACITY  = int(TRAIN_CAPACITY * MAX_OCCUPANCY)   # 1620

MAX_TRAINS_PER_LINE = 15      # signalling ceiling
PEAK_HOURS          = set(range(8, 12)) | set(range(16, 21))
SHOULDER_HOURS      = {7, 15, 21}

MIN_PEAK            = 8
MIN_SHOULDER        = 5
MIN_OFFPEAK         = 3

RAIN_MULTIPLIER     = 1.22
FESTIVAL_MULTIPLIER = 1.55
WEEKEND_MULTIPLIER  = 1.10


# ── Helpers (mirror orchestrator logic exactly) ───────────────────────────────

def _service_label(trains: int) -> str:
    if trains >= 13: return "🔴 Ultra Peak"
    if trains >= 10: return "🟠 Peak"
    if trains >=  7: return "🟡 High"
    if trains >=  5: return "🟢 Normal"
    return "⚪ Low"

def _min_trains(hour: int) -> int:
    if hour in PEAK_HOURS:     return MIN_PEAK
    if hour in SHOULDER_HOURS: return MIN_SHOULDER
    return MIN_OFFPEAK

def _apply_multipliers(d: float, rain: bool, festival: bool, weekend: bool) -> float:
    if rain:     d *= RAIN_MULTIPLIER
    if festival: d *= FESTIVAL_MULTIPLIER
    if weekend:  d *= WEEKEND_MULTIPLIER
    return d

def _trains_for_line(line_demand: float, hour: int) -> int:
    """Exact replica of orchestrator.demand_to_trains()."""
    needed = int(math.ceil(line_demand / EFFECTIVE_CAPACITY))
    needed = max(needed, _min_trains(hour))
    return min(needed, MAX_TRAINS_PER_LINE)

def _headway(trains: int) -> float:
    return round(60 / max(trains, 1), 1)

def _avg_wait(trains: int) -> float:
    return round(_headway(trains) / 2, 1)

def _utilisation(station_demand: float, line_trains: int, n_stations: int) -> float:
    """
    Each train carries passengers across ALL stations on the line.
    Effective per-station capacity = train_capacity / n_stations (throughput model).
    """
    per_station_cap = (line_trains * EFFECTIVE_CAPACITY) / max(n_stations, 1)
    return round(min(100.0, station_demand / per_station_cap * 100), 1) if per_station_cap > 0 else 0.0

def _priority(demand: float, util: float, is_interchange: bool) -> float:
    score = (demand / EFFECTIVE_CAPACITY) * 0.5 + (util / 100) * 0.5
    if is_interchange:
        score *= 1.25
    return round(score, 4)

def _period(hour: int) -> str:
    if hour in PEAK_HOURS:     return "✅ Peak"
    if hour in SHOULDER_HOURS: return "Shoulder"
    return "Off-Peak"

def _fixed_baseline(hour: int) -> tuple:
    """Mirrors orchestrator.fixed_schedule_baseline()."""
    if hour in PEAK_HOURS:     return 10, 6.0, 3.0
    if hour in SHOULDER_HOURS: return  7, 8.6, 4.3
    return 5, 12.0, 6.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StationAllocation:
    station_id:       str
    station_name:     str
    line:             str
    zone:             str
    is_interchange:   bool
    demand:           int           # adjusted passengers this hour
    # Line-level metrics (what passenger actually experiences)
    line_trains:      int           # total trains on this line
    line_headway:     float         # minutes between trains at this station
    avg_wait_mins:    float         # = line_headway / 2
    line_service:     str           # 🔴/🟠/🟡/🟢/⚪
    # Station-level congestion metrics
    utilisation_pct:  float         # how loaded this station is vs capacity share
    unmet_demand:     int           # passengers who can't board
    priority_score:   float         # higher = needs more attention


@dataclass
class OptimizationResult:
    hour:             int
    period:           str
    total_demand:     int
    total_trains:     int
    allocations:      list          # List[StationAllocation], sorted by priority
    unmet_demand:     int
    efficiency_score: float
    avg_wait_mins:    float         # demand-weighted across all stations
    avg_wait_fixed:   float
    wait_saved_mins:  float
    line_summary:     dict

    def summary(self) -> str:
        out = [
            f"\nHour {self.hour:02d}:00  [{self.period}]",
            f"  Total demand    : {self.total_demand:,} pax",
            f"  Trains deployed : {self.total_trains}",
            f"  Avg wait (AI)   : {self.avg_wait_mins:.1f} min",
            f"  Avg wait (fixed): {self.avg_wait_fixed:.1f} min",
            f"  Wait saved      : {self.wait_saved_mins:+.1f} min",
            f"  Efficiency      : {self.efficiency_score:.1f}%",
        ]
        if self.unmet_demand:
            out.append(f"  ⚠  Unmet demand : {self.unmet_demand:,} pax")
        out.append("")
        for ln, s in self.line_summary.items():
            out.append(
                f"  {ln.title():<8} → {s['trains']:>2} trains | "
                f"headway {s['headway']:.1f} min | "
                f"wait {s['avg_wait']:.1f} min | "
                f"{s['demand']:>6,} pax | {s['service_level']}"
            )
        return "\n".join(out)


# ── Main optimizer ────────────────────────────────────────────────────────────

class PickupOptimizer:
    """
    Extends orchestrator.py's line-level logic to per-station outputs.

    The orchestrator decides trains per LINE (correct).
    This optimizer adds per-station demand analysis, congestion scoring,
    and pickup priority ranking on top of that foundation.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            cfg = json.load(f)
        self.stations = {s["id"]: s for s in cfg["stations"]}
        self.lines    = sorted({s["line"] for s in cfg["stations"]})

    def optimize(
        self,
        demand_dict:    dict,
        hour:           int  = 8,
        rain_surge:     bool = False,
        festival_surge: bool = False,
        weekend:        bool = False,
    ) -> OptimizationResult:

        # ── Step 1: adjust demand with multipliers ────────────────────────────
        adj_demand = {
            sid: int(_apply_multipliers(v, rain_surge, festival_surge, weekend))
            for sid, v in demand_dict.items()
        }

        # ── Step 2: line totals → trains per line (orchestrator formula) ──────
        line_stations = {ln: [s for s in self.stations.values() if s["line"] == ln]
                         for ln in self.lines}

        line_demand = {
            ln: sum(adj_demand.get(s["id"], 0) for s in line_stations[ln])
            for ln in self.lines
        }

        line_trains = {
            ln: _trains_for_line(line_demand[ln], hour)
            for ln in self.lines
        }

        # ── Step 3: per-station allocation ───────────────────────────────────
        allocations  = []
        total_unmet  = 0
        w_wait_sum   = 0.0
        w_dem_sum    = 0.0

        for ln in self.lines:
            n_sta    = len(line_stations[ln])
            n_trains = line_trains[ln]
            hw       = _headway(n_trains)
            wait     = _avg_wait(n_trains)   # ← LINE-level wait, same for all stations
            svc      = _service_label(n_trains)

            # Capacity per station = total line capacity / n_stations (throughput model)
            cap_per_station = (n_trains * EFFECTIVE_CAPACITY) / n_sta

            for s in line_stations[ln]:
                sid    = s["id"]
                demand = adj_demand.get(sid, 0)
                util   = round(min(100.0, demand / cap_per_station * 100), 1) if cap_per_station > 0 else 0.0
                unmet  = max(0, int(demand - cap_per_station))
                total_unmet += unmet

                priority = _priority(demand, util, s.get("interchange", False))

                if demand > 0:
                    w_wait_sum += wait * demand
                    w_dem_sum  += demand

                allocations.append(StationAllocation(
                    station_id      = sid,
                    station_name    = s["name"],
                    line            = ln,
                    zone            = s.get("zone", ""),
                    is_interchange  = bool(s.get("interchange", False)),
                    demand          = demand,
                    line_trains     = n_trains,
                    line_headway    = hw,
                    avg_wait_mins   = wait,
                    line_service    = svc,
                    utilisation_pct = util,
                    unmet_demand    = unmet,
                    priority_score  = priority,
                ))

        # ── Step 4: summary stats ─────────────────────────────────────────────
        avg_wait_ai   = round(w_wait_sum / w_dem_sum if w_dem_sum > 0 else 0, 1)
        _, _, avg_wait_fixed = _fixed_baseline(hour)
        total_dem     = sum(adj_demand.values())
        efficiency    = round((1 - total_unmet / max(total_dem, 1)) * 100, 1)

        line_summary = {
            ln: {
                "trains":        line_trains[ln],
                "headway":       _headway(line_trains[ln]),
                "avg_wait":      _avg_wait(line_trains[ln]),
                "demand":        line_demand[ln],
                "service_level": _service_label(line_trains[ln]),
            }
            for ln in self.lines
        }

        return OptimizationResult(
            hour             = hour,
            period           = _period(hour),
            total_demand     = total_dem,
            total_trains     = sum(line_trains.values()),
            allocations      = sorted(allocations,
                                      key=lambda a: a.priority_score,
                                      reverse=True),
            unmet_demand     = total_unmet,
            efficiency_score = efficiency,
            avg_wait_mins    = avg_wait_ai,
            avg_wait_fixed   = avg_wait_fixed,
            wait_saved_mins  = round(avg_wait_fixed - avg_wait_ai, 2),
            line_summary     = line_summary,
        )

    def full_day_schedule(
        self,
        hourly_demand:  dict,
        rain_surge:     bool = False,
        festival_surge: bool = False,
        weekend:        bool = False,
    ) -> list:
        return [
            self.optimize(hourly_demand.get(h, {}), hour=h,
                          rain_surge=rain_surge,
                          festival_surge=festival_surge,
                          weekend=weekend)
            for h in range(6, 24)
        ]

    def demo_demand(self, hour: int) -> dict:
        """Realistic per-station demand at real Pune Metro scale."""
        np.random.seed(hour * 7)
        is_peak     = hour in PEAK_HOURS
        is_shoulder = hour in SHOULDER_HOURS

        line_totals = {
            "aqua":   23000 if is_peak else (12000 if is_shoulder else 6000),
            "purple": 18000 if is_peak else (9000  if is_shoulder else 4500),
        }

        result = {}
        for ln in self.lines:
            ln_stations = [s for s in self.stations.values() if s["line"] == ln]
            n = len(ln_stations)
            weights = np.random.dirichlet(np.ones(n) * 2)
            for i, s in enumerate(ln_stations):
                if s.get("interchange"):
                    weights[i] *= 1.5
            weights /= weights.sum()
            total = line_totals.get(ln, 5000)
            for i, s in enumerate(ln_stations):
                noise = np.random.normal(0, total * 0.03)
                result[s["id"]] = max(0, int(total * weights[i] + noise))
        return result


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    opt = PickupOptimizer()
    print("=" * 62)
    print("  SmartTransit AI — Pickup & Fleet Optimizer")
    print("  (Calibrated to Real Pune Metro / orchestrator.py)")
    print("=" * 62)

    for hour in [8, 13, 18]:
        demand = opt.demo_demand(hour)
        result = opt.optimize(demand, hour=hour)
        print(result.summary())
        print(f"\n  Top 5 Priority Stations:")
        for a in result.allocations[:5]:
            bar = "█" * int(a.utilisation_pct / 10)
            print(f"    {a.station_name:<32} "
                  f"{a.line_trains:>2}T on line | "
                  f"util {a.utilisation_pct:>5.1f}% {bar:<10} | "
                  f"wait {a.avg_wait_mins:.1f}m | "
                  f"{a.line_service}")
        print()