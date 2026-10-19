"""Bayesian reconstruction with the published (diag sigma) and the textbook (diag sigma^2)
noise covariance, on the multi-reach window of every target reach overpass.

Usage: python 11_bayes_variants.py <reach_list.csv> <out_name>
Window rule (RiverObs get_multi_reach_height): first valid upstream and downstream
neighbour (data/neighbours.csv), used when it has more than 5 nodes with node_q < 3 in
the same overpass; neighbour nodes enter only if observed; the current reach includes
its unobserved nodes (heights reconstructed there too). Node mask: valid wse and
node_q < 3.
Output: results/<out_name>.parquet, one row per reach overpass with, for p = 1 and 2,
reach wse (mean of reconstructed node heights), slope (first-to-last), and the
reconstructed height at every gauge node of the reach (gauge rows joined later),
plus the published values for checking.
"""
import sys
import numpy as np
import pandas as pd
from p39.paths import NODE_DIR, DATA, RESULTS
from p39.bayes import reconstruct


def load_nodes(rid, cache):
    if rid in cache:
        return cache[rid]
    f = NODE_DIR / f"{rid}.csv"
    df = pd.read_csv(f) if f.exists() and f.stat().st_size > 10 else pd.DataFrame()
    cache[rid] = df
    return df


def main():
    targets = pd.read_csv(sys.argv[1])["reach_id"].astype("int64").unique()
    out = RESULTS / f"{sys.argv[2]}.parquet"
    nb = pd.read_csv(DATA / "neighbours.csv").set_index("reach_id")
    gm = pd.read_csv(DATA / "gauge_match.csv")
    gauge_nodes = gm.groupby("reach_id")["node_id"].apply(list).to_dict()
    rows, cache = [], {}
    for i, rid in enumerate(targets):
        cur_all = load_nodes(rid, cache)
        if cur_all.empty:
            continue
        up, dn = (nb.loc[rid, "up_id"], nb.loc[rid, "dn_id"]) if rid in nb.index else (0, 0)
        side = {s: load_nodes(int(r), cache) if r else pd.DataFrame()
                for s, r in (("up", up), ("dn", dn))}
        for (cyc, pas), cur in cur_all.groupby(["cycle_id", "pass_id"]):
            if (cur["wse"].notna() & (cur["node_q"] < 3)).sum() < 2:
                continue
            blocks, used = [], {}
            for s, d in side.items():
                ok = False
                if not d.empty:
                    d = d[(d["cycle_id"] == cyc) & (d["pass_id"] == pas) &
                          (d["time_str"] != "no_data")]
                    ok = (d["wse"].notna() & (d["node_q"] < 3)).sum() > 5
                    if ok:
                        blocks.append(d.assign(is_cur=False))
                used[s] = ok
            blocks.append(cur.assign(is_cur=True))
            w = pd.concat(blocks, ignore_index=True).sort_values("p_dist_out")
            mask = (w["wse"].notna() & (w["node_q"] < 3)).values
            ic = w["is_cur"].values
            xs = w["p_dist_out"].values[ic]
            row = {"reach_id": rid, "cycle_id": cyc, "pass_id": pas,
                   "used_up": used["up"], "used_dn": used["dn"],
                   "n_window": len(w), "n_mask_cur": int(mask[ic].sum()),
                   "pub_wse_sm_mean": w["wse_sm"].values[ic].mean(),
                   "pub_slope_sm": (w["wse_sm"].values[ic][-1] - w["wse_sm"].values[ic][0])
                   / (xs[-1] - xs[0]) if len(xs) > 1 else np.nan}
            for p in (1, 2):
                rec = reconstruct(w["p_dist_out"].values, w["wse"].values,
                                  w["wse_r_u"].values, mask, noise_power=p)[ic]
                row[f"p{p}_wse"] = np.nanmean(rec)
                row[f"p{p}_slope"] = (rec[-1] - rec[0]) / (xs[-1] - xs[0]) if len(xs) > 1 else np.nan
                row[f"p{p}_x_mean"] = xs.mean()
                ids = w["node_id"].values[ic]
                for gn in gauge_nodes.get(rid, []):
                    j = np.where(ids == gn)[0]
                    if len(j):
                        row[f"p{p}_node_{gn}"] = rec[j[0]]
            rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(targets)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(out, len(df))


if __name__ == "__main__":
    main()
