"""Stated uncertainty for the recommended single-reach slope estimators.

For every clean overpass of the IRIS sample: Theil-Sen slope with the Sen (1968) 95%
interval (sigma = interval width / 3.92), OLS and Huber slopes with their standard
errors (statsmodels). Calibration as for slope_u: z = (overpass slope - reach median of
that estimator) / sigma; a well-calibrated sigma gives NMAD(z) close to 1.
Output: output/estimator_uncertainty.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from p39.paths import NODE_DIR, DATA, OUTPUT

OUTLIER_BIT = 1 << 23


def main():
    ids = pd.read_csv(DATA / "iris_sample.csv")["reach_id"].values
    rows = []
    for rid in ids:
        f = NODE_DIR / f"{rid}.csv"
        if not f.exists() or f.stat().st_size < 10:
            continue
        n = pd.read_csv(f, usecols=["cycle_id", "pass_id", "wse", "node_q", "node_q_b", "p_dist_out"])
        ok = n["wse"].notna() & (n["node_q"] < 3) & \
            ((n["node_q_b"].fillna(0).astype("int64") & OUTLIER_BIT) == 0)
        for (c, p), g in n[ok].groupby(["cycle_id", "pass_id"]):
            if len(g) < 5:
                continue
            x = g["p_dist_out"].values - g["p_dist_out"].mean()
            y = g["wse"].values
            if np.ptp(x) <= 0:
                continue
            ts = stats.theilslopes(y, x, alpha=0.95)
            X = sm.add_constant(x)
            ols = sm.OLS(y, X).fit()
            hub = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
            rows.append(dict(reach_id=rid, ts=ts.slope, ts_u=(ts.high_slope - ts.low_slope) / 3.92,
                             ols=ols.params[1], ols_u=ols.bse[1], huber=hub.params[1], huber_u=hub.bse[1]))
    d = pd.DataFrame(rows)
    out = []
    for e in ("ts", "ols", "huber"):
        dev = d[e] - d.groupby("reach_id")[e].transform("median")
        z = (dev / d[e + "_u"]).replace([np.inf, -np.inf], np.nan).dropna()
        out.append(dict(estimator=e, n=len(z), nmad_z=1.4826 * np.median(np.abs(z - np.median(z))),
                        share_within_1=np.mean(np.abs(z) <= 1),
                        median_u_mm_km=1e6 * d[e + "_u"].median()))
    out = pd.DataFrame(out)
    out.to_csv(OUTPUT / "estimator_uncertainty.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
