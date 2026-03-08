"""
multi_objective_optimizer.py
SmartTransit AI – Multi-Objective Fleet Optimization
Bonus feature: optimizes across THREE competing objectives simultaneously:

  1. TIME      — minimize passenger wait time
  2. FUEL      — minimize energy/fuel consumption (fewer trains = less energy)
  3. COVERAGE  — maximize network coverage (all stations adequately served)

Uses a weighted scoring approach with Pareto-front analysis.
Fully calibrated to orchestrator.py constants.

Usage:
    from Algorithms.multi_objective_optimizer import MultiObjectiveOptimizer
    moo = MultiObjectiveOptimizer()
    result = moo.optimize(demand_dict, hour=8)
    print(result.summary())

    # Custom weights
    result = moo.optimize(demand_dict, hour=8,
                          w_time=0.6, w_fuel=0.1, w_coverage=0.3)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

CONFIG_PATH = Path(__file__).parent.parent / "Data" / "stations_config.json"

# ── Constants — identical to orchestrator.py ─────────────────────────────────
TRAIN_CAPACITY      = 1800
MAX_OCCUPANCY       = 0.90
EFFECTIVE_CAPACITY  = int(TRAIN_CAPACITY * MAX_OCCUPANCY)   # 1620
MAX_TRAINS_PER_LINE = 15
MIN_TRAINS_PER_LINE = 1

PEAK_HOURS     = set(range(8, 12)) | set(range(16, 21))
SHOULDER_HOURS = {7, 15, 21}

RAIN_MULTIPLIER     = 1.22
FESTIVAL_MULTIPLIER = 1.55
WEEKEND_MULTIPLIER  = 1.10

# Pune Metro energy model (kWh per train-km)
# 6-car Alstom Metropolis ≈ 8.5 kWh/km
# Aqua line 14.66 km, Purple line 16.59 km
ENERGY_PER_TRAIN_KWH = 8.5
LINE_LENGTH_KM = {"aqua": 14.66, "purple": 16.59}

# Default objective weights (must sum to 1.0)
DEFAULT_W_TIME     = 0.50
DEFAULT_W_FUEL     = 0.20
DEFAULT_W_COVERAGE = 0.30


# ── Helpers ───────────────────────────────────────────────────────────────────

def _service_label(trains: int) -> str:
    if trains >= 13: return "🔴 Ultra Peak"
    if trains >= 10: return "🟠 Peak"
    if trains >=  7: return "🟡 High"
    if trains >=  5: return "🟢 Normal"
    return "⚪ Low"

def _min_trains(hour: int) -> int:
    if hour in PEAK_HOURS:     return 8
    if hour in SHOULDER_HOURS: return 5
    return 3

def _apply_context(d: float, rain: bool, festival: bool, weekend: bool) -> float:
    if rain:     d *= RAIN_MULTIPLIER
    if festival: d *= FESTIVAL_MULTIPLIER
    if weekend:  d *= WEEKEND_MULTIPLIER
    return d

def _headway(trains: int) -> float:
    return round(60 / max(trains, 1), 1)

def _avg_wait(trains: int) -> float:
    return round(_headway(trains) / 2, 1)

def _fixed_baseline(hour: int) -> tuple:
    if hour in PEAK_HOURS:     return 10, 6.0, 3.0
    if hour in SHOULDER_HOURS: return  7, 8.6, 4.3
    return 5, 12.0, 6.0

def _period(hour: int) -> str:
    if hour in PEAK_HOURS:     return "✅ Peak"
    if hour in SHOULDER_HOURS: return "Shoulder"
    return "Off-Peak"


# ── Objective scoring functions ───────────────────────────────────────────────

def score_time(trains: int, max_trains: int = MAX_TRAINS_PER_LINE) -> float:
    """
    Time score: 0 (worst) → 1 (best).
    More trains = shorter wait = better time score.
    Normalized against max possible trains.
    """
    wait      = _avg_wait(trains)
    wait_max  = _avg_wait(1)           # worst case: 1 train = 30 min wait
    wait_min  = _avg_wait(max_trains)  # best case:  15 trains = 2 min wait
    return round(1 - (wait - wait_min) / (wait_max - wait_min), 4)


def score_fuel(trains: int, line: str, max_trains: int = MAX_TRAINS_PER_LINE) -> float:
    """
    Fuel score: 0 (worst) → 1 (best).
    Fewer trains = less energy = better fuel score.
    Based on kWh per hour for the line.
    """
    dist      = LINE_LENGTH_KM.get(line, 15.0)
    # Each train does ~2 round trips per hour at 34 km/h avg
    trips_hr  = (34 / dist)
    energy    = trains * trips_hr * dist * ENERGY_PER_TRAIN_KWH
    e_min     = 1     * trips_hr * dist * ENERGY_PER_TRAIN_KWH
    e_max     = max_trains * trips_hr * dist * ENERGY_PER_TRAIN_KWH
    return round(1 - (energy - e_min) / (e_max - e_min), 4)


def score_coverage(trains: int, line_demand: float, line: str) -> float:
    """
    Coverage score: 0 (worst) → 1 (best).
    Full coverage = all demand served within capacity.
    Partial coverage penalised proportionally.
    """
    capacity = trains * EFFECTIVE_CAPACITY
    if capacity <= 0:
        return 0.0
    covered  = min(1.0, capacity / max(line_demand, 1))
    # Bonus for interchange-heavy lines (more transfer options)
    return round(covered, 4)


# ── Candidate solution ────────────────────────────────────────────────────────

@dataclass
class Candidate:
    trains:          int
    line:            str
    line_demand:     float
    hour:            int
    score_time:      float = 0.0
    score_fuel:      float = 0.0
    score_coverage:  float = 0.0
    composite_score: float = 0.0
    headway:         float = 0.0
    avg_wait:        float = 0.0
    energy_kwh:      float = 0.0
    utilisation_pct: float = 0.0
    service_level:   str   = ""
    is_pareto:       bool  = False


@dataclass
class LineResult:
    line:             str
    optimal_trains:   int
    headway_mins:     float
    avg_wait_mins:    float
    energy_kwh_hr:    float
    utilisation_pct:  float
    service_level:    str
    score_time:       float
    score_fuel:       float
    score_coverage:   float
    composite_score:  float
    demand:           int
    wait_saved_mins:  float
    pareto_front:     List[Candidate] = field(default_factory=list)


@dataclass
class MOOResult:
    hour:             int
    period:           str
    w_time:           float
    w_fuel:           float
    w_coverage:       float
    total_demand:     int
    total_trains:     int
    total_energy_kwh: float
    avg_wait_mins:    float
    avg_wait_fixed:   float
    wait_saved_mins:  float
    coverage_pct:     float
    efficiency_score: float
    line_results:     dict    # { line: LineResult }

    def summary(self) -> str:
        out = [
            f"\n{'='*60}",
            f"  Multi-Objective Optimization — Hour {self.hour:02d}:00 [{self.period}]",
            f"  Weights: Time={self.w_time:.0%}  Fuel={self.w_fuel:.0%}  "
            f"Coverage={self.w_coverage:.0%}",
            f"{'='*60}",
            f"  Total demand    : {self.total_demand:,} pax",
            f"  Trains deployed : {self.total_trains}",
            f"  Avg wait (AI)   : {self.avg_wait_mins:.1f} min",
            f"  Avg wait (fixed): {self.avg_wait_fixed:.1f} min",
            f"  Wait saved      : {self.wait_saved_mins:+.1f} min",
            f"  Energy this hr  : {self.total_energy_kwh:.1f} kWh",
            f"  Network coverage: {self.coverage_pct:.1f}%",
            f"  Efficiency      : {self.efficiency_score:.1f}%",
            f"",
        ]
        for ln, r in self.line_results.items():
            out.append(
                f"  {ln.title():<8} → {r.optimal_trains:>2} trains | "
                f"wait {r.avg_wait_mins:.1f}m | "
                f"energy {r.energy_kwh_hr:.0f} kWh | "
                f"util {r.utilisation_pct:.1f}% | "
                f"composite {r.composite_score:.3f} | "
                f"{r.service_level}"
            )
        return "\n".join(out)


# ── Main optimizer ────────────────────────────────────────────────────────────

class MultiObjectiveOptimizer:
    """
    Pareto-front multi-objective optimizer for Pune Metro fleet.

    For each line and hour, evaluates ALL feasible train counts (min→max),
    scores each on time / fuel / coverage, then selects the solution
    with the highest weighted composite score.

    Also identifies the full Pareto front — solutions where you cannot
    improve one objective without worsening another.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            cfg = json.load(f)
        self.stations = {s["id"]: s for s in cfg["stations"]}
        self.lines    = sorted({s["line"] for s in cfg["stations"]})

    # ── Context-Aware AI Weighting ────────────────────────────────────────────

    def get_auto_weights(self, hour: int) -> tuple[float, float, float, str]:
        """
        Returns context-aware weights (Time, Fuel, Coverage) and a rationale string 
        based on the time of day.
        """
        if hour in PEAK_HOURS:
            # 8-11, 16-20: Prioritize moving massive crowds fast (Time & Coverage)
            return 0.65, 0.10, 0.25, "High passenger volume detected. Prioritizing reduced wait times and maximum network coverage."
        elif hour in SHOULDER_HOURS:
            # 7, 15, 21: Balanced approach as crowds build/dissipate
            return 0.40, 0.30, 0.30, "Transition period. Balancing passenger wait times with fuel efficiency."
        else:
            # Off-peak: Prioritize energy savings since trains are empty
            return 0.20, 0.60, 0.20, "Low demand period. Maximizing fuel and energy efficiency to reduce operational costs."

    # ── Pareto front detection ────────────────────────────────────────────────

    def _is_pareto(self, candidate: Candidate, all_candidates: list) -> bool:
        """
        A candidate is Pareto-optimal if no other candidate dominates it
        (i.e. no other solution is better or equal on ALL objectives
        and strictly better on at least one).
        """
        for other in all_candidates:
            if other.trains == candidate.trains:
                continue
            dominates = (
                other.score_time     >= candidate.score_time     and
                other.score_fuel     >= candidate.score_fuel     and
                other.score_coverage >= candidate.score_coverage and
                (other.score_time     > candidate.score_time     or
                 other.score_fuel     > candidate.score_fuel     or
                 other.score_coverage > candidate.score_coverage)
            )
            if dominates:
                return False
        return True

    # ── Energy calculation ────────────────────────────────────────────────────

    def _energy(self, trains: int, line: str) -> float:
        dist     = LINE_LENGTH_KM.get(line, 15.0)
        trips_hr = 34 / dist   # round trips per hour at avg 34 km/h
        return round(trains * trips_hr * dist * ENERGY_PER_TRAIN_KWH, 1)

    # ── Line optimizer ────────────────────────────────────────────────────────

    def _optimize_line(
        self,
        line:        str,
        line_demand: float,
        hour:        int,
        w_time:      float,
        w_fuel:      float,
        w_coverage:  float,
    ) -> LineResult:
        """Evaluate all feasible train counts and pick the best composite."""
        t_min = _min_trains(hour)
        t_max = MAX_TRAINS_PER_LINE

        # Demand-implied minimum (can't serve demand with too few trains)
        demand_min = int(math.ceil(line_demand / EFFECTIVE_CAPACITY))
        t_min      = max(t_min, min(demand_min, t_max))

        candidates = []
        for t in range(t_min, t_max + 1):
            st = score_time(t)
            sf = score_fuel(t, line)
            sc = score_coverage(t, line_demand, line)
            composite = w_time * st + w_fuel * sf + w_coverage * sc

            dist     = LINE_LENGTH_KM.get(line, 15.0)
            trips_hr = 34 / dist
            energy   = round(t * trips_hr * dist * ENERGY_PER_TRAIN_KWH, 1)
            cap      = t * EFFECTIVE_CAPACITY
            util     = round(min(100.0, line_demand / cap * 100), 1) if cap > 0 else 0.0

            candidates.append(Candidate(
                trains          = t,
                line            = line,
                line_demand     = line_demand,
                hour            = hour,
                score_time      = st,
                score_fuel      = sf,
                score_coverage  = sc,
                composite_score = round(composite, 4),
                headway         = _headway(t),
                avg_wait        = _avg_wait(t),
                energy_kwh      = energy,
                utilisation_pct = util,
                service_level   = _service_label(t),
            ))

        # Mark Pareto-optimal solutions
        for c in candidates:
            c.is_pareto = self._is_pareto(c, candidates)

        # Best = highest composite score
        best    = max(candidates, key=lambda c: c.composite_score)
        pareto  = [c for c in candidates if c.is_pareto]

        _, _, fixed_wait = _fixed_baseline(hour)

        return LineResult(
            line             = line,
            optimal_trains   = best.trains,
            headway_mins     = best.headway,
            avg_wait_mins    = best.avg_wait,
            energy_kwh_hr    = best.energy_kwh,
            utilisation_pct  = best.utilisation_pct,
            service_level    = best.service_level,
            score_time       = best.score_time,
            score_fuel       = best.score_fuel,
            score_coverage   = best.score_coverage,
            composite_score  = best.composite_score,
            demand           = int(line_demand),
            wait_saved_mins  = round(fixed_wait - best.avg_wait, 2),
            pareto_front     = pareto,
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def optimize(
        self,
        demand_dict:    dict,
        hour:           int   = 8,
        w_time:         float = DEFAULT_W_TIME,
        w_fuel:         float = DEFAULT_W_FUEL,
        w_coverage:     float = DEFAULT_W_COVERAGE,
        rain_surge:     bool  = False,
        festival_surge: bool  = False,
        weekend:        bool  = False,
    ) -> MOOResult:
        """
        Run multi-objective optimization for all lines.

        demand_dict : { station_id: passenger_count }
        w_time      : weight for minimising wait time  (0–1)
        w_fuel      : weight for minimising fuel/energy (0–1)
        w_coverage  : weight for maximising coverage    (0–1)
        Note: weights are auto-normalised to sum to 1.
        """
        # Normalise weights
        total_w = w_time + w_fuel + w_coverage
        w_time     = w_time     / total_w
        w_fuel     = w_fuel     / total_w
        w_coverage = w_coverage / total_w

        # Apply context multipliers
        adj = {
            sid: _apply_context(v, rain_surge, festival_surge, weekend)
            for sid, v in demand_dict.items()
        }

        # Line-level demand totals
        line_demand = {ln: 0.0 for ln in self.lines}
        for sid, s in self.stations.items():
            line_demand[s["line"]] += adj.get(sid, 0)

        # Optimise each line
        line_results = {}
        for ln in self.lines:
            line_results[ln] = self._optimize_line(
                ln, line_demand[ln], hour, w_time, w_fuel, w_coverage
            )

        # Aggregate stats
        total_trains  = sum(r.optimal_trains   for r in line_results.values())
        total_energy  = sum(r.energy_kwh_hr    for r in line_results.values())
        total_demand  = int(sum(adj.values()))
        total_cap     = sum(r.optimal_trains * EFFECTIVE_CAPACITY for r in line_results.values())
        coverage_pct  = round(min(100.0, total_cap / max(total_demand, 1) * 100), 1)

        # Demand-weighted avg wait
        w_wait = sum(
            r.avg_wait_mins * line_demand[ln]
            for ln, r in line_results.items()
        )
        avg_wait = round(w_wait / max(sum(line_demand.values()), 1), 1)
        _, _, fixed_wait = _fixed_baseline(hour)
        efficiency = round(
            sum(r.composite_score for r in line_results.values()) / len(line_results) * 100, 1
        )

        return MOOResult(
            hour             = hour,
            period           = _period(hour),
            w_time           = w_time,
            w_fuel           = w_fuel,
            w_coverage       = w_coverage,
            total_demand     = total_demand,
            total_trains     = total_trains,
            total_energy_kwh = total_energy,
            avg_wait_mins    = avg_wait,
            avg_wait_fixed   = fixed_wait,
            wait_saved_mins  = round(fixed_wait - avg_wait, 2),
            coverage_pct     = coverage_pct,
            efficiency_score = efficiency,
            line_results     = line_results,
        )

    # ── Sensitivity analysis — how does result change with different weights ──

    def weight_sensitivity(
        self,
        demand_dict: dict,
        hour:        int = 8,
    ) -> list:
        """
        Run optimization with 5 preset weight profiles.
        Returns list of (profile_name, MOOResult) — great for dashboard comparison.
        """
        profiles = [
            ("⏱  Time Priority",     0.70, 0.10, 0.20),
            ("⚡ Balanced",           0.50, 0.20, 0.30),
            ("⛽ Fuel Saver",         0.20, 0.60, 0.20),
            ("🗺  Max Coverage",       0.20, 0.10, 0.70),
            ("🏙  Off-Peak Economy",   0.30, 0.40, 0.30),
        ]
        results = []
        for name, wt, wf, wc in profiles:
            r = self.optimize(demand_dict, hour=hour,
                              w_time=wt, w_fuel=wf, w_coverage=wc)
            results.append((name, r))
        return results

    # ── Full day schedule ─────────────────────────────────────────────────────

    def full_day_schedule(
        self,
        hourly_demand:  dict,
        w_time:         float = DEFAULT_W_TIME,
        w_fuel:         float = DEFAULT_W_FUEL,
        w_coverage:     float = DEFAULT_W_COVERAGE,
        rain_surge:     bool  = False,
        festival_surge: bool  = False,
        weekend:        bool  = False,
    ) -> list:
        return [
            self.optimize(
                hourly_demand.get(h, {}), hour=h,
                w_time=w_time, w_fuel=w_fuel, w_coverage=w_coverage,
                rain_surge=rain_surge, festival_surge=festival_surge,
                weekend=weekend,
            )
            for h in range(6, 24)
        ]

    # ── Demo demand (same as pickup_optimizer scale) ──────────────────────────

    def demo_demand(self, hour: int) -> dict:
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
    moo = MultiObjectiveOptimizer()

    print("\n" + "="*62)
    print("  SmartTransit AI — Multi-Objective Fleet Optimizer")
    print("  Objectives: Time + Fuel + Coverage")
    print("="*62)

    # Single hour with default weights
    for hour in [8, 13, 18]:
        demand = moo.demo_demand(hour)
        result = moo.optimize(demand, hour=hour)
        print(result.summary())

    # Weight sensitivity at peak hour
    print("\n" + "="*62)
    print("  Sensitivity Analysis — Hour 08:00 (Peak)")
    print("  How does train count change under different priorities?")
    print("="*62)
    demand = moo.demo_demand(8)
    profiles = moo.weight_sensitivity(demand, hour=8)
    print(f"\n  {'Profile':<25} {'Aqua':>5} {'Purple':>7} {'Wait':>6} {'Energy':>8} {'Coverage':>9}")
    print(f"  {'-'*60}")
    for name, r in profiles:
        aq = r.line_results.get("aqua",   None)
        pu = r.line_results.get("purple", None)
        print(
            f"  {name:<25} "
            f"{aq.optimal_trains if aq else '—':>5} "
            f"{pu.optimal_trains if pu else '—':>7} "
            f"{r.avg_wait_mins:>5.1f}m "
            f"{r.total_energy_kwh:>7.0f}kWh "
            f"{r.coverage_pct:>8.1f}%"
        )