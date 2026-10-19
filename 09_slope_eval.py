"""Reach slope of every estimator against IRIS v3.4 (ICESat-2) median slope (D5).

Input: results/iris_estimates.parquet (06 on data/iris_sample.csv), data/iris_sample.csv.
Outputs:
  results/slope_pass.parquet   one row per reach overpass x node set x estimator
  results/slope_reach.parquet  per reach x node set x estimator: median SWOT slope over
                               passes, IRIS slope, error (mm/km) and log10 ratio
Slopes are in mm/km (1e-6 m/m). Only overpasses with >= 5 nodes in the node set are used.
"""
import numpy as np
import pandas as pd
from p39.paths import DATA, RESULTS
from p39 import estimators as E

SLOPE_EST = ["official", "official_s2", "bayes_s1", "bayes_s2"] + list(E.LINE)
MIN_PASSES = 5


def main():
    est = pd.read_parquet(RESULTS / "iris_estimates.parquet")
    iris = pd.read_csv(DATA / "iris_sample.csv")[
        ["reach_id", "continent", "avg_combined_slope", "std_combined_slope",
         "n_combined_slope", "width", "reach_length", "slope"]
    ].rename(columns={"avg_combined_slope": "iris_slope", "std_combined_slope": "iris_std",
                      "slope": "sword_slope"})
    est = est[est["n_nodes"] >= E.MIN_NODES].copy()
    est["official_slope"] = est["off_slope"]
    est["official_s2_slope"] = est["off_slope2"]
    fb = RESULTS / "iris_bayes.parquet"
    if fb.exists():
        b = pd.read_parquet(fb, columns=["reach_id", "cycle_id", "pass_id", "p1_slope", "p2_slope"])
        for c in ("cycle_id", "pass_id"):
            b[c] = b[c].astype(est[c].dtype)
        est = est.merge(b.rename(columns={"p1_slope": "bayes_s1_slope", "p2_slope": "bayes_s2_slope"}),
                        on=["reach_id", "cycle_id", "pass_id"], how="left")
    else:
        est["bayes_s1_slope"] = est["bayes_s2_slope"] = np.nan
    keep = ["reach_id", "cycle_id", "pass_id", "t", "node_set", "n_nodes", "scatter_nmad",
            "off_reach_q", "off_dark_frac", "off_xtrk_dist", "off_ice_clim_f",
            "off_obs_frac_n", "off_width", "node_layover"]
    long = est.melt(id_vars=keep, value_vars=[f"{k}_slope" for k in SLOPE_EST],
                    var_name="estimator", value_name="slope_mm_km")
    long["estimator"] = long["estimator"].str.replace("_slope", "", regex=False)
    long["slope_mm_km"] *= 1e6
    long = long.merge(iris, on="reach_id", how="inner")
    long["err_mm_km"] = long["slope_mm_km"] - long["iris_slope"]
    long.to_parquet(RESULTS / "slope_pass.parquet", index=False)

    g = long.groupby(["reach_id", "node_set", "estimator"])
    rch = g.agg(swot_slope=("slope_mm_km", "median"), n_pass=("slope_mm_km", "count"),
                pass_iqr=("slope_mm_km", lambda x: x.quantile(.75) - x.quantile(.25)),
                share_negative=("slope_mm_km", lambda x: (x <= 0).mean()),
                scatter_nmad=("scatter_nmad", "median"),
                iris_slope=("iris_slope", "first"), iris_std=("iris_std", "first"),
                continent=("continent", "first"), width=("width", "first"),
                reach_length=("reach_length", "first")).reset_index()
    rch = rch[rch["n_pass"] >= MIN_PASSES]
    rch["err_mm_km"] = rch["swot_slope"] - rch["iris_slope"]
    pos = rch["swot_slope"] > 0
    rch["log_ratio"] = np.where(pos, np.log10(rch["swot_slope"].where(pos) / rch["iris_slope"]),
                                np.nan)
    rch.to_parquet(RESULTS / "slope_reach.parquet", index=False)
    s = rch[rch["node_set"] == "clean"].groupby("estimator").agg(
        n=("err_mm_km", "size"),
        med_err=("err_mm_km", "median"),
        med_abs=("err_mm_km", lambda x: x.abs().median()),
        p68_abs=("err_mm_km", lambda x: x.abs().quantile(.68)),
        within_17=("err_mm_km", lambda x: (x.abs() <= 17).mean()),
        med_log=("log_ratio", "median"))
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
