"""Slope statistics against IRIS: bootstrap CIs, paired tests, continent breakdown,
pull of the multi-reach estimators toward the neighbour slope, per-overpass scatter,
stated-uncertainty calibration and first-order discharge error.

Inputs: results/slope_reach.parquet, results/slope_pass.parquet (09),
data/neighbours.csv, data/sword_reaches.parquet.
Outputs: output/slope_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from p39.paths import RESULTS, DATA, OUTPUT
from p39 import stats as St

EST = ["official", "official_s2", "bayes_s2", "ols", "wls", "theilsen", "huber"]


def main():
    r = pd.read_parquet(RESULTS / "slope_reach.parquet")
    r = r[r["node_set"] == "clean"]
    w = r.pivot_table(index="reach_id", columns="estimator", values="err_mm_km")
    have = [e for e in EST if e in w.columns and w[e].notna().any()]
    EST[:] = have
    w = w.dropna(subset=EST)
    common = w.index
    rr = r[r["reach_id"].isin(common)].copy()
    meta = rr.drop_duplicates("reach_id").set_index("reach_id")[
        ["continent", "iris_slope", "iris_std", "width", "reach_length"]]

    # summary with bootstrap (each reach is one unit here)
    rows = []
    for e in EST:
        d = rr[rr["estimator"] == e][["reach_id", "err_mm_km", "log_ratio"]]
        s = St.summary(d["err_mm_km"])
        lo, hi = St.cluster_bootstrap(d, "err_mm_km", "reach_id",
                                      lambda x: np.quantile(np.abs(x), .68))
        lo2, hi2 = St.cluster_bootstrap(d, "err_mm_km", "reach_id", lambda x: np.median(np.abs(x)))
        q = np.sqrt(10 ** d["log_ratio"].dropna())          # discharge ratio r^(1/2)
        rows.append(dict(estimator=e, **s, p68_lo=lo, p68_hi=hi, mae_lo=lo2, mae_hi=hi2,
                         within_17=(d["err_mm_km"].abs() <= 17).mean(),
                         med_log=d["log_ratio"].median(),
                         q_err_med=np.median(np.abs(q - 1)), q_err_p68=np.quantile(np.abs(q - 1), .68),
                         share_nonpositive=(10 ** d["log_ratio"]).isna().mean()))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUTPUT / "slope_summary.csv", index=False)
    print(f"reaches with all estimators: {len(common)}")
    print(summ.round(3).to_string(index=False))

    # paired Wilcoxon on |error|
    pr = []
    for a, b in [("ols", "official"), ("theilsen", "official"), ("huber", "official"),
                 ("official_s2", "official"), ("theilsen", "ols"), ("huber", "ols"),
                 ("wls", "ols"), ("ols", "official_s2"), ("bayes_s2", "official"),
                 ("theilsen", "bayes_s2")]:
        if a not in w.columns or b not in w.columns:
            continue
        x, y = w[a].abs(), w[b].abs()
        pr.append(dict(a=a, b=b, n=len(x), med_diff=np.median(x - y),
                       share_a_better=np.mean(x < y), p=stats.wilcoxon(x, y).pvalue))
    pr = pd.DataFrame(pr)
    pr.to_csv(OUTPUT / "slope_paired.csv", index=False)
    print(pr.round(4).to_string(index=False))

    # continent breakdown (p68 of |error|)
    cont = rr.merge(meta[["continent"]], left_on="reach_id", right_index=True,
                    suffixes=("_x", "")).groupby(["continent", "estimator"])["err_mm_km"] \
        .agg(lambda x: np.quantile(np.abs(x), .68)).unstack()
    cont.to_csv(OUTPUT / "slope_by_continent.csv")
    print(cont[EST].round(1).to_string())

    # pull toward neighbours: (official - OLS) vs (neighbour-window SWORD slope - reach SWORD slope)
    sw = pd.read_parquet(DATA / "sword_reaches.parquet").set_index("reach_id")
    nb = pd.read_csv(DATA / "neighbours.csv").set_index("reach_id").reindex(common)
    s0 = sw["slope"].reindex(common).values                 # m/km in SWORD
    su = sw["slope"].reindex(nb["up_id"]).values
    sd = sw["slope"].reindex(nb["dn_id"]).values
    L0 = sw["reach_length"].reindex(common).values
    Lu = sw["reach_length"].reindex(nb["up_id"]).values
    Ld = sw["reach_length"].reindex(nb["dn_id"]).values
    num = np.nansum(np.c_[s0 * L0, su * Lu, sd * Ld], axis=1)
    den = np.nansum(np.c_[L0, np.where(np.isfinite(su), Lu, np.nan),
                          np.where(np.isfinite(sd), Ld, np.nan)], axis=1)
    contrast = (num / den - s0) * 1000                       # mm/km
    pull = pd.DataFrame({"reach_id": common, "contrast_mm_km": contrast})
    for e in ["official", "official_s2", "theilsen"]:
        pull[f"{e}_minus_ols"] = (w[e] - w["ols"]).values
    pull = pull.dropna()
    res = []
    for e in ["official", "official_s2", "theilsen"]:
        y = pull[f"{e}_minus_ols"]
        x = pull["contrast_mm_km"]
        ts = stats.theilslopes(y, x)
        rho = stats.spearmanr(x, y)
        res.append(dict(estimator=e, n=len(x), theil_slope=ts.slope, lo=ts.low_slope,
                        hi=ts.high_slope, spearman=rho.statistic, p=rho.pvalue))
    res = pd.DataFrame(res)
    res.to_csv(OUTPUT / "slope_neighbour_pull.csv", index=False)
    pull.to_csv(OUTPUT / "slope_neighbour_pull_points.csv", index=False)
    print(res.round(3).to_string(index=False))

    # pull test with IRIS slopes of the neighbours (independent of SWORD DEM slopes):
    # (estimator - Theil-Sen) vs (length-weighted IRIS slope of both neighbours - IRIS slope
    # of the reach). The difference on the left does not contain the IRIS slope of the reach,
    # so IRIS noise can only attenuate, not create, an association.
    import netCDF4 as nc
    from p39.paths import IRIS_FILE
    with nc.Dataset(IRIS_FILE) as d:
        ir = pd.DataFrame({"reach_id": d["reach_id"][:].astype("int64"),
                           "s": np.ma.filled(d["avg_combined_slope"][:].astype(float), np.nan),
                           "n": np.ma.filled(d["n_combined_slope"][:].astype(float), np.nan)})
    ir = ir[ir["n"] >= 3].set_index("reach_id")["s"]
    sl = r.pivot_table(index="reach_id", columns="estimator", values="swot_slope").reindex(common)
    s0i, sui, sdi = (ir.reindex(common).values, ir.reindex(nb["up_id"]).values,
                     ir.reindex(nb["dn_id"]).values)
    ok = np.isfinite(sui) & np.isfinite(sdi)
    xi = ((sui * Lu + sdi * Ld) / (Lu + Ld) - s0i)[ok]
    res2 = []
    for e in [x for x in ["official", "official_s2", "bayes_s2", "ols", "wls", "huber"] if x in sl.columns]:
        y = (sl[e] - sl["theilsen"]).values[ok]
        g = np.isfinite(y) & np.isfinite(xi)
        ts = stats.theilslopes(y[g], xi[g])
        rho = stats.spearmanr(xi[g], y[g])
        res2.append(dict(estimator=e, n=int(g.sum()), theil_slope=ts.slope, lo=ts.low_slope,
                         hi=ts.high_slope, spearman=rho.statistic, p=rho.pvalue))
    res2 = pd.DataFrame(res2)
    res2.to_csv(OUTPUT / "slope_neighbour_pull_iris.csv", index=False)
    print(res2.round(3).to_string(index=False))

    # per-overpass scatter about the reach median and stated-uncertainty calibration
    p = pd.read_parquet(RESULTS / "slope_pass.parquet")
    p = p[(p["node_set"] == "clean") & p["reach_id"].isin(common)].dropna(subset=["slope_mm_km"])
    p["dev"] = p["slope_mm_km"] - p.groupby(["reach_id", "estimator"])["slope_mm_km"].transform("median")
    sc = p.groupby("estimator")["dev"].agg(lambda x: 1.4826 * np.median(np.abs(x))).rename("pass_nmad_mm_km")
    sc.to_csv(OUTPUT / "slope_pass_scatter.csv")
    print(sc.round(1).to_string())


if __name__ == "__main__":
    main()
