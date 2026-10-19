"""Match USGS water-level gauges to SWORD v17b nodes (rules D2 and D7 in DESIGN.md).

Output: data/gauge_match.csv (kept gauges) and output/gauge_exclusions.csv (counts by
reason, in the order the rules are applied).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from p39.paths import DATA, OUTPUT

SURVEYED = {"L", "D", "F", "W", "X", "Y", "Z"}
MAX_ACC_FT = 0.5
SQMI_TO_KM2 = 2.58998811
EARTH_R = 6371008.8


def to_xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * EARTH_R


def main():
    s = pd.read_csv(DATA / "usgs_sites.csv")
    log = [("stream sites with continuous water level 2023-09..2026-06", len(s))]
    s = s[s["vertical_datum"] == "NAVD88"]
    log.append(("datum NAVD88", len(s)))
    s = s[s["altitude_method_code"].isin(SURVEYED)]
    log.append(("datum from a survey method", len(s)))
    s = s[s["altitude_accuracy"].fillna(99) <= MAX_ACC_FT]
    log.append((f"stated datum accuracy <= {MAX_ACC_FT} ft", len(s)))
    s = s.dropna(subset=["lon", "lat"])

    nodes = pd.read_parquet(DATA / "sword_nodes.parquet")
    nodes = nodes[nodes["continent"] == "na"].reset_index(drop=True)
    reaches = pd.read_parquet(DATA / "sword_reaches.parquet")
    rtype = reaches.set_index("reach_id")["type"]
    tree = cKDTree(to_xyz(nodes["x"].values, nodes["y"].values))
    d, i = tree.query(to_xyz(s["lon"].values, s["lat"].values))
    nn = nodes.iloc[i].reset_index(drop=True)
    s = s.reset_index(drop=True)
    s["node_id"] = nn["node_id"].values
    s["reach_id"] = nn["reach_id"].values
    s["dist_to_node_m"] = d
    s["node_width_m"] = nn["width"].values
    s["node_dist_out_m"] = nn["dist_out"].values
    s["sword_facc_km2"] = nn["facc"].values
    s["reach_type"] = rtype.reindex(s["reach_id"]).values
    s["node_lakeflag"] = nn["lakeflag"].values

    tol = np.maximum(300.0, s["node_width_m"].fillna(0))
    s = s[s["dist_to_node_m"] <= tol]
    log.append(("within max(300 m, node width) of a SWORD node", len(s)))
    s = s[s["reach_type"] == 1]
    log.append(("reach type river (1)", len(s)))
    s = s[s["node_lakeflag"] == 0]
    log.append(("node not on a lake (lakeflag 0)", len(s)))
    da = s["drainage_area"] * SQMI_TO_KM2
    ratio = da / s["sword_facc_km2"]
    s["drainage_ratio"] = ratio
    ok = ratio.isna() | (s["sword_facc_km2"] <= 0) | ratio.between(0.8, 1.25)
    s = s[ok]
    log.append(("drainage area ratio 0.8-1.25 (or not available)", len(s)))
    # one gauge per node: keep the closest
    s = s.sort_values("dist_to_node_m").drop_duplicates("node_id")
    log.append(("one gauge per node", len(s)))

    s.to_csv(DATA / "gauge_match.csv", index=False)
    pd.DataFrame(log, columns=["rule", "n_remaining"]).to_csv(
        OUTPUT / "gauge_exclusions.csv", index=False)
    for r, n in log:
        print(f"{n:6d}  {r}")
    print("reaches with a gauge:", s["reach_id"].nunique())


if __name__ == "__main__":
    main()
