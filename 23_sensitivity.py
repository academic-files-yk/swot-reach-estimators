"""Sensitivity of the estimator ranking (DESIGN.md D5, D11 and reviewer-type checks).

Slope (68th percentile of |reach-median error| against IRIS, mm/km):
  base            clean node set, all sampled reaches
  all_nodes       outlier-flagged nodes kept ('all' node set)
  iris_stable     IRIS reaches with std/median <= 0.5 and >= 10 ICESat-2 slopes (D5)
  reach_q_good    reach medians from overpasses with reach_q <= 1 only
  wide_200        SWORD width >= 200 m
WSE (68th percentile of |relative error| at gauges, cm):
  base            clean node set, screened gauges removed
  no_screening    all gauges kept (D11)
  levelled        gauges whose datum was levelled (method L)
  reach_q_good    overpasses with reach_q <= 1
Output: output/sensitivity.csv
"""
import numpy as np
import pandas as pd
from p39.paths import RESULTS, DATA, OUTPUT
from p39.sets import WSE_REQUIRED

SL = ["official", "official_s2", "bayes_s2", "ols", "theilsen", "huber"]
WS = ["official", "official_s2", "official_sm", "bayes_s2", "ols", "wls", "theilsen", "huber"]
P68 = lambda x: np.quantile(np.abs(x[np.isfinite(x)]), .68)


def slope_rows():
    p = pd.read_parquet(RESULTS / "slope_pass.parquet").dropna(subset=["slope_mm_km"])
    iris = pd.read_csv(DATA / "iris_sample.csv").set_index("reach_id")

    def reach_median(d):
        g = d.groupby(["reach_id", "estimator"])
        r = g["slope_mm_km"].agg(["median", "count"]).reset_index()
        r = r[r["count"] >= 5]
        r["err"] = r["median"] - iris["avg_combined_slope"].reindex(r["reach_id"]).values
        w = r.pivot_table(index="reach_id", columns="estimator", values="err").dropna(subset=SL)
        return w

    base = p[p["node_set"] == "clean"]
    stable_ids = iris[(iris["std_combined_slope"] / iris["avg_combined_slope"] <= 0.5) &
                      (iris["n_combined_slope"] >= 10)].index
    wide_ids = iris[iris["width"] >= 200].index
    cases = {"base": base,
             "all_nodes": p[p["node_set"] == "all"],
             "iris_stable": base[base["reach_id"].isin(stable_ids)],
             "reach_q_good": base[base["off_reach_q"] <= 1],
             "wide_200": base[base["reach_id"].isin(wide_ids)]}
    rows = []
    for k, d in cases.items():
        w = reach_median(d)
        rows.append(dict(quantity="slope (mm/km)", case=k, n=len(w),
                         **{e: P68(w[e].values) for e in SL}))
    return rows


def wse_rows():
    d = pd.read_parquet(RESULTS / "wse_long.parquet")
    d = d[d["node_set"] == "clean"]
    gm = pd.read_csv(DATA / "gauge_match.csv").set_index("site")
    d = d.dropna(subset=[e + "_err" for e in WSE_REQUIRED])
    lev = gm.index[gm["altitude_method_code"] == "L"]
    cases = {"base": d[~d["screened"]], "no_screening": d,
             "levelled": d[~d["screened"] & d["site"].isin(lev)],
             "reach_q_good": d[~d["screened"] & (d["off_reach_q"] <= 1)]}
    rows = []
    for k, x in cases.items():
        r = {}
        for e in WS:
            rel = x[e + "_err"] - x.groupby("site")[e + "_err"].transform("median")
            r[e] = 100 * P68(rel.values)
        rows.append(dict(quantity="WSE relative (cm)", case=k, n=x["site"].nunique(), **r))
    return rows


def main():
    df = pd.DataFrame(slope_rows() + wse_rows())
    df.to_csv(OUTPUT / "sensitivity.csv", index=False)
    print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
