"""Apply all node-to-reach estimators to every reach overpass.

Usage: python 06_reach_estimates.py <reach_list.csv> <out_name>
Writes results/<out_name>.parquet with one row per reach overpass and node set:
official reach values, observation descriptors and (x_ref, level, slope) for every
estimator. Node sets (DESIGN.md D6):
  'clean' - valid wse, node_q < 3, official outlier bit (node_q_b bit 23) not set
  'all'   - every node with a valid wse (outlier-flagged nodes kept; in Version D the
            bit-23 nodes are exactly the node_q = 3 nodes with a valid wse)
"""
import sys
import numpy as np
import pandas as pd
from p39.paths import NODE_DIR, REACH_DIR, RESULTS
from p39 import estimators as E

OUTLIER_BIT = 1 << 23
REACH_KEEP = ["wse", "wse_u", "slope", "slope_u", "slope2", "slope2_u", "width", "reach_q",
              "reach_q_b", "dark_frac", "ice_clim_f", "n_good_nod", "obs_frac_n",
              "partial_f", "xovr_cal_q", "p_length", "p_dist_out", "layovr_val",
              "xtrk_dist"]


def node_sets(nd):
    ok = nd["wse"].notna() & (nd["node_q"] < 3)
    flagged = (nd["node_q_b"].fillna(0).astype("int64") & OUTLIER_BIT) > 0
    return {"clean": nd[ok & ~flagged], "all": nd[nd["wse"].notna()]}


def one_reach(reach_id):
    fr, fn = REACH_DIR / f"{reach_id}.csv", NODE_DIR / f"{reach_id}.csv"
    if not fr.exists() or fr.stat().st_size < 10 or fn.stat().st_size < 10:
        return []
    rch = pd.read_csv(fr)
    nod = pd.read_csv(fn)
    rch["t"] = pd.to_datetime(rch["time_str"], utc=True)
    nod["t"] = pd.to_datetime(nod["time_str"], utc=True, errors="coerce")
    rows = []
    for (cyc, pas), r in rch.groupby(["cycle_id", "pass_id"]):
        r = r.iloc[0]
        nd = nod[(nod["cycle_id"] == cyc) & (nod["pass_id"] == pas)]
        nd_sm = nd[nd["wse_sm"].notna()]
        base = {"reach_id": reach_id, "cycle_id": cyc, "pass_id": pas, "t": r["t"],
                **{f"off_{k}": r[k] for k in REACH_KEEP if k in r},
                "n_nodes_reported": len(nd),
                "x_mean_sm": nd_sm["p_dist_out"].mean() if len(nd_sm) else np.nan,
                "wse_sm_mean": nd_sm["wse_sm"].mean() if len(nd_sm) else np.nan}
        for name, s in node_sets(nd).items():
            row = dict(base, node_set=name, n_nodes=len(s))
            if len(s) >= E.MIN_NODES:
                x, y, u = s["p_dist_out"].values, s["wse"].values, s["wse_u"].values
                row.update(span_m=np.ptp(x), scatter_nmad=E.detrended_scatter(x, y),
                           x_mean_obs=x.mean(),
                           node_dark_frac=s["dark_frac"].mean(),
                           node_xtrk_abs=np.abs(s["xtrk_dist"]).median(),
                           node_layover=np.abs(s["layovr_val"]).median(),
                           node_wse_u=np.median(u))
                for est, (xr, lev, slp) in E.run_all(x, y, u).items():
                    row.update({f"{est}_xref": xr, f"{est}_level": lev, f"{est}_slope": slp})
            rows.append(row)
    return rows


def main():
    lst = pd.read_csv(sys.argv[1])
    out = RESULTS / f"{sys.argv[2]}.parquet"
    rows = []
    ids = lst["reach_id"].astype("int64").unique()
    for i, rid in enumerate(ids):
        rows += one_reach(rid)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(ids)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(out, len(df), "rows")


if __name__ == "__main__":
    main()
