"""WSE accuracy of all estimators against datum-converted gauge levels.

Inputs: results/wse_eval.parquet (07), optional results/gauge_bayes.parquet (11).
Outputs (output/): wse_summary_<set>_<abs|rel>.csv, wse_paired.csv, wse_screen.csv,
wse_uncertainty.csv, and results/wse_long.parquet for figures.
"""
import numpy as np
import pandas as pd
from p39.paths import RESULTS, OUTPUT
from p39 import stats as St
from p39.sets import WSE_REQUIRED

EST = ["official", "official_s2", "bayes_s2_reach", "official_sm", "bayes_s2", "nearest_node",
       "mean", "wmean", "median", "ols", "wls", "theilsen", "huber"]
SCREEN_M = 2.0


def add_bayes(df):
    """Node-level (bayes_s1/bayes_s2: reconstructed height at the gauge node) and
    reach-level (bayes_s1_reach/bayes_s2_reach: mean reconstructed height plus
    first-to-last slope moved to the gauge) values from 11_bayes_variants."""
    f = RESULTS / "gauge_bayes.parquet"
    cols = ["bayes_s1", "bayes_s2", "bayes_s1_reach", "bayes_s2_reach"]
    if not f.exists():
        for c in cols:
            df[c] = np.nan
        return df
    b = pd.read_parquet(f)
    key = ["reach_id", "cycle_id", "pass_id"]
    for c in ("cycle_id", "pass_id"):
        b[c] = b[c].astype(df[c].dtype)
    node_cols = [c for c in b.columns if c.startswith(("p1_node_", "p2_node_"))]
    long = b.melt(id_vars=key, value_vars=node_cols, var_name="k", value_name="h").dropna()
    long["node_id"] = long["k"].str.split("_").str[-1].astype("int64")
    long["p"] = long["k"].str[1].astype(int)
    nodes = long.pivot_table(index=key + ["node_id"], columns="p", values="h").reset_index()
    nodes = nodes.rename(columns={1: "bayes_s1", 2: "bayes_s2"})
    df = df.merge(nodes, on=key + ["node_id"], how="left")
    reach = b[key + ["p1_wse", "p1_slope", "p1_x_mean", "p2_wse", "p2_slope", "p2_x_mean"]]
    df = df.merge(reach, on=key, how="left")
    for p in (1, 2):
        df[f"bayes_s{p}_reach"] = df[f"p{p}_wse"] + df[f"p{p}_slope"] * (df["x_gauge"] - df[f"p{p}_x_mean"])
    return df.drop(columns=[c for c in reach.columns if c not in key])


def main():
    df = pd.read_parquet(RESULTS / "wse_eval.parquet")
    df = df[df["truth"].notna()].copy()
    df = add_bayes(df)
    for e in EST + ["bayes_s1", "bayes_s1_reach"]:
        df[e + "_err"] = df[e] - df["truth"]
    # D11 screening on the clean set
    c = df[df["node_set"] == "clean"]
    across = c[[e + "_err" for e in WSE_REQUIRED]].median(axis=1)
    g_off = across.groupby(c["site"]).median()
    bad = set(g_off[g_off.abs() > SCREEN_M].index)
    pd.DataFrame({"site": g_off.index, "median_offset_m": g_off.values,
                  "screened_out": g_off.index.isin(bad)}).to_csv(OUTPUT / "wse_screen.csv", index=False)
    print("gauges:", len(g_off), "screened out:", len(bad))
    df["screened"] = df["site"].isin(bad)
    df.to_parquet(RESULTS / "wse_long.parquet", index=False)

    for ns in ("clean", "all"):
        d = df[(df["node_set"] == ns) & ~df["screened"]].copy()
        # common overpasses: all single-reach and official estimators present
        d = d.dropna(subset=[e + "_err" for e in WSE_REQUIRED])
        for mode in ("abs", "rel"):
            dd = d.copy()
            if mode == "rel":
                for e in EST:
                    col = e + "_err"
                    dd[col] = dd[col] - dd.groupby("site")[col].transform("median")
            t = St.table(dd, EST, "site")
            t.to_csv(OUTPUT / f"wse_summary_{ns}_{mode}.csv", index=False)
            print(f"\n{ns} {mode}: {d['site'].nunique()} gauges, {len(d)} overpasses")
            print(t[["estimator", "n", "med", "mae", "p68", "p68_lo", "p68_hi"]].round(3).to_string(index=False))
    d = df[(df["node_set"] == "clean") & ~df["screened"]]
    pairs = [("official", e) for e in EST if e != "official"] + \
            [("bayes_s2", "official_sm"), ("bayes_s2_reach", "official"), ("bayes_s1", "official_sm"),
             ("theilsen", "ols"), ("huber", "ols"), ("wls", "ols")]
    pt = pd.DataFrame([dict(a=a, b=b, **St.paired(d, a, b)) for a, b in pairs])
    pt.to_csv(OUTPUT / "wse_paired.csv", index=False)
    print("\n", pt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
