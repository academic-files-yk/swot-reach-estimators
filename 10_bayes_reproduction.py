"""Test which noise covariance reproduces the published Version D wse_sm.

For a fixed random sample of gauge reaches and overpasses, the multi-reach window
(first valid upstream and downstream SWORD neighbour, types 3/4/5 skipped, used only if
it has more than 5 nodes with node_q < 3 in that overpass) is rebuilt from Hydrocron
node data and wse_sm is recomputed with noise covariance diag(sigma) and diag(sigma^2).
Output: output/bayes_reproduction.csv (per overpass) and a printed summary.
"""
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
import requests
from p39.paths import SWORD_DIR, NODE_DIR, REACH_DIR, DATA, OUTPUT, WORK
from p39.bayes import reconstruct
import importlib

dl = importlib.import_module("04_hydrocron_download")
NB_DIR = WORK / "swot" / "node_neighbour"
NB_DIR.mkdir(parents=True, exist_ok=True)
SEED, N_REACH, N_PASS = 39, 40, 5


def sword_topology():
    with nc.Dataset(glob.glob(str(SWORD_DIR / "**" / "na_sword_v17b.nc"), recursive=True)[0]) as d:
        g = d["reaches"]
        rid = g["reach_id"][:].astype("int64")
        up = np.ma.filled(g["rch_id_up"][:], 0).astype("int64")
        dn = np.ma.filled(g["rch_id_dn"][:], 0).astype("int64")
    return {r: (up[:, i], dn[:, i]) for i, r in enumerate(rid)}


def first_valid(ids):
    for x in ids:
        if x > 0 and x % 10 not in (3, 4, 5):
            return int(x)
    return None


def neighbour_nodes(reach_id, sword_nodes):
    f = NB_DIR / f"{reach_id}.csv"
    if f.exists():
        return pd.read_csv(f) if f.stat().st_size > 10 else pd.DataFrame()
    s = requests.Session()
    parts = [dl.query("Node", n, dl.NODE_FIELDS, s) for n in sword_nodes.get(reach_id, [])]
    parts = [p for p in parts if p is not None and not p.empty]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.to_csv(f, index=False)
    return df


def main():
    topo = sword_topology()
    sn = pd.read_parquet(DATA / "sword_nodes.parquet", columns=["node_id", "reach_id"])
    sword_nodes = sn.groupby("reach_id")["node_id"].apply(list).to_dict()
    have = sorted(int(p.split("/")[-1][:-4]) for p in glob.glob(str(NODE_DIR / "*.csv")))
    rng = np.random.default_rng(SEED)
    sample = rng.choice(have, min(N_REACH, len(have)), replace=False)
    rows = []
    for rid in sample:
        nod = pd.read_csv(NODE_DIR / f"{rid}.csv")
        if nod.empty or rid not in topo:
            continue
        up_id, dn_id = (first_valid(topo[rid][0]), first_valid(topo[rid][1]))
        nb = {k: (neighbour_nodes(v, sword_nodes) if v else pd.DataFrame())
              for k, v in (("up", up_id), ("dn", dn_id))}
        good = nod[nod["wse"].notna() & (nod["node_q"] < 3)]
        passes = good.groupby(["cycle_id", "pass_id"]).size()
        passes = passes[passes >= 10].index.tolist()
        if not passes:
            continue
        pick = [passes[i] for i in rng.choice(len(passes), min(N_PASS, len(passes)),
                                              replace=False)]
        for cyc, pas in pick:
            cur = nod[(nod["cycle_id"] == cyc) & (nod["pass_id"] == pas)]
            blocks = []
            used = {}
            for side in ("up", "dn"):
                d = nb[side]
                if d.empty:
                    used[side] = False
                    continue
                d = d[(d["cycle_id"] == cyc) & (d["pass_id"] == pas) & (d["time_str"] != "no_data")]
                ok = (d["wse"].notna() & (d["node_q"] < 3)).sum() > 5
                used[side] = bool(ok)
                if ok:
                    blocks.append(d.assign(is_cur=False))
            blocks.append(cur.assign(is_cur=True))
            w = pd.concat(blocks, ignore_index=True).sort_values("p_dist_out")
            mask = (w["wse"].notna() & (w["node_q"] < 3)).values
            res = {"reach_id": rid, "cycle_id": cyc, "pass_id": pas,
                   "n_window": len(w), "used_up": used["up"], "used_dn": used["dn"]}
            curm = w["is_cur"].values & w["wse_sm"].notna().values
            for p in (1, 2):
                rec = reconstruct(w["p_dist_out"].values, w["wse"].values,
                                  w["wse_r_u"].values, mask, noise_power=p)
                e = rec[curm] - w["wse_sm"].values[curm]
                res[f"p{p}_med_abs_mm"] = np.median(np.abs(e)) * 1000
                res[f"p{p}_max_abs_mm"] = np.max(np.abs(e)) * 1000
            rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "bayes_reproduction.csv", index=False)
    print(len(df), "overpasses from", df["reach_id"].nunique(), "reaches")
    print(df[["p1_med_abs_mm", "p2_med_abs_mm", "p1_max_abs_mm", "p2_max_abs_mm"]]
          .describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
    print("share with p1 median < 1 mm:", (df["p1_med_abs_mm"] < 1).mean().round(3),
          " p2:", (df["p2_med_abs_mm"] < 1).mean().round(3))


if __name__ == "__main__":
    main()
