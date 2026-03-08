import json
import math
import heapq
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).parent.parent / "Data" / "stations_config.json"

# ── Haversine distance (km) between two lat/lon points ───────────────────────
def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Travel time model ─────────────────────────────────────────────────────────
AVG_SPEED_KMPH   = 34.0   # Pune Metro average
DWELL_SEC        = 30     # seconds per station stop
TRANSFER_PENALTY = 5.0    # extra minutes for line change at interchange

def _travel_time(dist_km: float, is_transfer: bool = False) -> float:
    """Returns travel time in minutes."""
    t = (dist_km / AVG_SPEED_KMPH) * 60 + (DWELL_SEC / 60)
    if is_transfer:
        t += TRANSFER_PENALTY
    return round(t, 2)


# ── Main Router class ─────────────────────────────────────────────────────────
class MetroRouter:
    """
    Builds a weighted graph of the Pune Metro network and exposes
    Dijkstra-based shortest path queries.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            self.config = json.load(f)

        self.stations   = {s["id"]: s for s in self.config["stations"]}
        self.graph      = {}   # { station_id: [(neighbor_id, weight_mins, dist_km)] }
        self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        """
        Connect consecutive stations on each line,
        plus interchange edges at Civil Court (PU11 ↔ AQ09).
        """
        # Line sequences (must match physical order)
        PURPLE = ["PU01","PU02","PU03","PU04","PU05","PU06",
                  "PU07","PU08","PU09","PU10","PU11","PU12","PU13","PU14"]
        AQUA   = ["AQ01","AQ02","AQ03","AQ04","AQ05","AQ06","AQ07","AQ08",
                  "AQ09","AQ10","AQ11","AQ12","AQ13","AQ14","AQ15","AQ16"]

        for node in self.stations:
            self.graph[node] = []

        def connect_sequence(seq):
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                sa, sb = self.stations[a], self.stations[b]
                dist = _haversine(sa["lat"], sa["lon"], sb["lat"], sb["lon"])
                time = _travel_time(dist, is_transfer=False)
                self.graph[a].append((b, time, dist))
                self.graph[b].append((a, time, dist))

        connect_sequence(PURPLE)
        connect_sequence(AQUA)

        # Interchange: Civil Court Purple (PU11) ↔ Civil Court Aqua (AQ09)
        # Same physical location — walking transfer, flat 5-min penalty
        self.graph["PU11"].append(("AQ09", TRANSFER_PENALTY, 0.0))
        self.graph["AQ09"].append(("PU11", TRANSFER_PENALTY, 0.0))

    # ── Dijkstra ──────────────────────────────────────────────────────────────

    def _dijkstra(self, source: str) -> tuple[dict, dict]:
        """
        Returns (dist_map, prev_map) from source to all reachable nodes.
        dist_map[node] = min travel time in minutes
        prev_map[node] = previous node in optimal path
        """
        dist = {n: float("inf") for n in self.graph}
        prev = {n: None         for n in self.graph}
        dist[source] = 0.0
        heap = [(0.0, source)]   # (cost, node)

        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist[u]:
                continue
            for v, weight, _ in self.graph[u]:
                new_cost = dist[u] + weight
                if new_cost < dist[v]:
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(heap, (new_cost, v))

        return dist, prev

    def _reconstruct_path(self, prev: dict, target: str) -> list:
        path = []
        node = target
        while node is not None:
            path.append(node)
            node = prev[node]
        return list(reversed(path))

    # ── Public API ────────────────────────────────────────────────────────────

    def shortest_path(self, origin_id: str, dest_id: str) -> dict:
        """
        Find the shortest (minimum time) path between two station IDs.

        Returns dict:
        {
          "origin":        { id, name, line },
          "destination":   { id, name, line },
          "path":          [ { id, name, line, cumulative_time_mins } ],
          "total_time_mins": float,
          "total_dist_km":   float,
          "num_stops":       int,
          "transfers":       int,
          "transfer_at":     [ station_name ],
          "fare_inr":        int,
          "summary":         str,
        }
        """
        if origin_id not in self.stations:
            raise ValueError(f"Unknown station ID: {origin_id}")
        if dest_id not in self.stations:
            raise ValueError(f"Unknown station ID: {dest_id}")

        dist_map, prev_map = self._dijkstra(origin_id)

        if dist_map[dest_id] == float("inf"):
            return {"error": f"No path found from {origin_id} to {dest_id}"}

        path_ids   = self._reconstruct_path(prev_map, dest_id)
        total_time = round(dist_map[dest_id], 2)

        # Build enriched path list + compute distance + detect transfers
        total_dist  = 0.0
        transfers   = 0
        transfer_at = []
        path_detail = []
        cumulative  = 0.0

        for i, sid in enumerate(path_ids):
            s = self.stations[sid]

            # Edge distance to next
            if i < len(path_ids) - 1:
                nxt = path_ids[i + 1]
                # Find edge weight
                for (nb, wt, dist_km) in self.graph[sid]:
                    if nb == nxt:
                        total_dist += dist_km
                        cumulative += wt
                        # Transfer detection: same physical station, different line
                        if dist_km == 0.0 and sid != nxt:
                            transfers += 1
                            transfer_at.append(s["name"])
                        break

            path_detail.append({
                "id":                   sid,
                "name":                 s["name"],
                "line":                 s["line"],
                "zone":                 s.get("zone", ""),
                "interchange":          s.get("interchange", False),
                "cumulative_time_mins": round(cumulative, 1),
            })

        # Simple fare model: ₹10 base + ₹2/km
        fare = max(10, int(10 + total_dist * 2))

        origin_s = self.stations[origin_id]
        dest_s   = self.stations[dest_id]

        summary = (
            f"{origin_s['name']} → {dest_s['name']} | "
            f"{len(path_ids)} stops | "
            f"{total_time:.1f} min | "
            f"{total_dist:.2f} km | "
            f"{'1 transfer at ' + transfer_at[0] if transfers else 'No transfer'}"
        )

        return {
            "origin":            {"id": origin_id,  "name": origin_s["name"],  "line": origin_s["line"]},
            "destination":       {"id": dest_id,    "name": dest_s["name"],    "line": dest_s["line"]},
            "path":              path_detail,
            "total_time_mins":   total_time,
            "total_dist_km":     round(total_dist, 2),
            "num_stops":         len(path_ids),
            "transfers":         transfers,
            "transfer_at":       transfer_at,
            "fare_inr":          fare,
            "summary":           summary,
        }

    def all_pairs_summary(self) -> list:
        """
        Returns a flat list of {origin, destination, time, dist, transfers}
        for every pair — useful for building a journey planner table.
        """
        results = []
        ids = list(self.stations.keys())
        for src in ids:
            dist_map, prev_map = self._dijkstra(src)
            for dst in ids:
                if src == dst:
                    continue
                if dist_map[dst] == float("inf"):
                    continue
                path = self._reconstruct_path(prev_map, dst)
                transfers = sum(
                    1 for i in range(len(path) - 1)
                    if self.stations[path[i]]["line"] != self.stations[path[i+1]]["line"]
                    and any(d == 0.0 for (nb, wt, d) in self.graph[path[i]] if nb == path[i+1])
                )
                results.append({
                    "origin":      self.stations[src]["name"],
                    "origin_id":   src,
                    "destination": self.stations[dst]["name"],
                    "dest_id":     dst,
                    "time_mins":   round(dist_map[dst], 1),
                    "transfers":   transfers,
                })
        return results

    def station_names(self) -> dict:
        """Returns {id: name} mapping — handy for dropdowns."""
        return {sid: s["name"] for sid, s in self.stations.items()}

    def stations_by_line(self, line: str) -> list:
        """Returns list of station dicts for a given line."""
        return [s for s in self.stations.values() if s["line"] == line]


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    router = MetroRouter()
    names  = router.station_names()

    test_pairs = [
        ("PU01", "PU14"),   # PCMC → Swargate (same line, end-to-end)
        ("AQ01", "AQ16"),   # Vanaz → Ramwadi (same line, end-to-end)
        ("PU01", "AQ16"),   # PCMC → Ramwadi  (cross-line, needs transfer)
        ("PU05", "AQ11"),   # Phugewadi → Pune Railway Station (cross-line)
        ("AQ01", "PU14"),   # Vanaz → Swargate (cross-line)
    ]

    print("=" * 65)
    print("  Pune Metro — Route Optimizer (Dijkstra)")
    print("=" * 65)

    for src, dst in test_pairs:
        result = router.shortest_path(src, dst)
        print(f"\n🚇 {result['summary']}")
        print(f"   Fare: ₹{result['fare_inr']}  |  Stops: {result['num_stops']}")
        stops = " → ".join(p["name"] for p in result["path"])
        # Wrap long lines
        if len(stops) > 80:
            stops = stops[:77] + "..."
        print(f"   Route: {stops}")