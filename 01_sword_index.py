"""Flat node and reach tables from SWORD v17b (all continents).

Output: data/sword_nodes.parquet, data/sword_reaches.parquet
Reach type is the last digit of the reach ID (1 river, 3 lake on river, 4 dam, 5 unreliable,
6 ghost).
"""
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from p39.paths import SWORD_DIR, DATA

NODE_VARS = ["node_id", "reach_id", "x", "y", "node_length", "dist_out", "width",
             "wse", "facc", "lakeflag", "n_chan_max", "obstr_type", "trib_flag"]
REACH_VARS = ["reach_id", "x", "y", "reach_length", "n_nodes", "dist_out", "width",
              "wse", "facc", "slope", "lakeflag", "obstr_type", "n_rch_up",
              "n_rch_down", "low_slope_flag", "max_width"]


def read_group(ds, group, names):
    g = ds[group]
    out = {}
    for v in names:
        if v in g.variables:
            a = g[v][:]
            out[v] = np.ma.filled(a, np.nan) if np.ma.isMaskedArray(a) else a
    return pd.DataFrame(out)


def main():
    files = sorted(glob.glob(str(SWORD_DIR / "**" / "*_sword_v17b.nc"), recursive=True))
    if not files:
        raise SystemExit("SWORD v17b netCDF files not found")
    nodes, reaches = [], []
    for f in files:
        cont = f.split("/")[-1][:2]
        with nc.Dataset(f) as ds:
            n = read_group(ds, "nodes", NODE_VARS)
            r = read_group(ds, "reaches", REACH_VARS)
        n["continent"] = cont
        r["continent"] = cont
        nodes.append(n)
        reaches.append(r)
        print(cont, len(r), "reaches", len(n), "nodes")
    nodes = pd.concat(nodes, ignore_index=True)
    reaches = pd.concat(reaches, ignore_index=True)
    for df in (nodes, reaches):
        df["reach_id"] = df["reach_id"].astype("int64")
    nodes["node_id"] = nodes["node_id"].astype("int64")
    reaches["type"] = reaches["reach_id"] % 10
    nodes.to_parquet(DATA / "sword_nodes.parquet", index=False)
    reaches.to_parquet(DATA / "sword_reaches.parquet", index=False)
    print("total", len(reaches), "reaches", len(nodes), "nodes")


if __name__ == "__main__":
    main()
