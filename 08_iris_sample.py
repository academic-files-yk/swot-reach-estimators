"""Select a stratified random sample of SWORD v17b reaches that have an IRIS slope.

Rules (fixed before any SWOT data are looked at):
  - IRIS combined slope available with at least 3 ICESat-2 crossings,
  - reach type river (1), lakeflag 0, no obstruction (obstr_type 0),
  - SWORD width >= 100 m (SWOT science requirement scale), reach length >= 5 km,
  - IRIS median slope > 0.
Sample: up to N_PER reaches per continent, numpy seed 39.
Output: data/iris_sample.csv and output/iris_sample_counts.csv.
"""
import numpy as np
import pandas as pd
import netCDF4 as nc
from p39.paths import IRIS_FILE, DATA, OUTPUT

N_PER = 250
SEED = 39


def main():
    with nc.Dataset(IRIS_FILE) as d:
        iris = pd.DataFrame({v: np.ma.filled(d[v][:].astype("float64"), np.nan) for v in
                             ["reach_id", "combined_flag", "avg_combined_slope",
                              "std_combined_slope", "n_combined_slope",
                              "min_date_combined_slope", "max_date_combined_slope"]})
    iris["reach_id"] = iris["reach_id"].astype("int64")
    r = pd.read_parquet(DATA / "sword_reaches.parquet")
    df = iris.merge(r, on="reach_id", how="inner")
    log = [("IRIS reaches matched to SWORD v17b", len(df))]
    for rule, m in [
        ("combined slope, >= 3 crossings",
         lambda d: (d["combined_flag"] == 1) & (d["n_combined_slope"] >= 3)),
        ("reach type 1, lakeflag 0, no obstruction",
         lambda d: (d["type"] == 1) & (d["lakeflag"] == 0) & (d["obstr_type"] == 0)),
        ("width >= 100 m, length >= 5 km",
         lambda d: (d["width"] >= 100) & (d["reach_length"] >= 5000)),
        ("IRIS median slope > 0", lambda d: d["avg_combined_slope"] > 0),
    ]:
        df = df[m(df)]
        log.append((rule, len(df)))
    rng = np.random.default_rng(SEED)
    parts = []
    for cont, g in df.groupby("continent"):
        k = min(N_PER, len(g))
        parts.append(g.iloc[rng.choice(len(g), k, replace=False)])
        log.append((f"sampled {cont}", k))
    s = pd.concat(parts)
    s.to_csv(DATA / "iris_sample.csv", index=False)
    pd.DataFrame(log, columns=["rule", "n"]).to_csv(OUTPUT / "iris_sample_counts.csv",
                                                     index=False)
    for a, b in log:
        print(f"{b:7d}  {a}")


if __name__ == "__main__":
    main()
