"""Observation state of the end nodes of each reach overpass.

The official slope is the first-to-last difference of reconstructed heights (Eq. 18 of
the ATBD); unobserved or masked end nodes get heights from the prior and the
neighbours. For every reach overpass this script records whether the most upstream and
most downstream node of the reach has a valid height with node_q < 3, and the
fraction of such nodes.
Usage: python 16_end_nodes.py <reach_list.csv> <out_name>  -> results/<out_name>.parquet
"""
import sys
import pandas as pd
from p39.paths import NODE_DIR, RESULTS, DATA


def main():
    ids = pd.read_csv(sys.argv[1])["reach_id"].astype("int64").unique()
    sn = pd.read_parquet(DATA / "sword_nodes.parquet", columns=["node_id", "reach_id", "dist_out"])
    ends = sn.sort_values("dist_out").groupby("reach_id")["node_id"].agg(["first", "last"])
    rows = []
    for rid in ids:
        f = NODE_DIR / f"{rid}.csv"
        if not f.exists() or f.stat().st_size < 10 or rid not in ends.index:
            continue
        n = pd.read_csv(f, usecols=["node_id", "cycle_id", "pass_id", "wse", "node_q"])
        n["ok"] = n["wse"].notna() & (n["node_q"] < 3)
        dn_id, up_id = ends.loc[rid, "first"], ends.loc[rid, "last"]
        for (c, p), g in n.groupby(["cycle_id", "pass_id"]):
            ok = g.set_index("node_id")["ok"]
            ok = ok[~ok.index.duplicated()]
            rows.append({"reach_id": rid, "cycle_id": c, "pass_id": p,
                         "dn_end_obs": bool(ok.get(dn_id, False)),
                         "up_end_obs": bool(ok.get(up_id, False)),
                         "frac_ok": ok.mean()})
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"{sys.argv[2]}.parquet", index=False)
    print(len(df), "overpasses; both ends observed:", (df.dn_end_obs & df.up_end_obs).mean().round(3))


if __name__ == "__main__":
    main()
