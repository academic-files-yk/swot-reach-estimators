"""First valid upstream and downstream SWORD v17b neighbour of every target reach.

The rule follows RiverObs get_multi_reach_height: the first listed neighbour whose type is
not 3, 4 or 5. Output: data/neighbours.csv (reach_id, up_id, dn_id) and
data/neighbour_download.csv (neighbour reaches not already in a target list).
"""
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from p39.paths import SWORD_DIR, DATA


def first_valid(ids):
    for x in ids:
        if x > 0 and x % 10 not in (3, 4, 5):
            return int(x)
    return 0


def main():
    topo = {}
    for f in sorted(glob.glob(str(SWORD_DIR / "**" / "*_sword_v17b.nc"), recursive=True)):
        with nc.Dataset(f) as d:
            g = d["reaches"]
            rid = g["reach_id"][:].astype("int64")
            up = np.ma.filled(g["rch_id_up"][:], 0).astype("int64")
            dn = np.ma.filled(g["rch_id_dn"][:], 0).astype("int64")
        for i, r in enumerate(rid):
            topo[int(r)] = (first_valid(up[:, i]), first_valid(dn[:, i]))
    targets = set(pd.read_csv(DATA / "gauge_match.csv")["reach_id"]) | \
        set(pd.read_csv(DATA / "iris_sample.csv")["reach_id"])
    rows = [(r, *topo.get(int(r), (0, 0))) for r in sorted(targets)]
    nb = pd.DataFrame(rows, columns=["reach_id", "up_id", "dn_id"])
    nb.to_csv(DATA / "neighbours.csv", index=False)
    need = (set(nb["up_id"]) | set(nb["dn_id"])) - {0} - targets
    pd.DataFrame({"reach_id": sorted(need)}).to_csv(DATA / "neighbour_download.csv", index=False)
    print(len(nb), "targets;", len(need), "extra neighbour reaches to download")


if __name__ == "__main__":
    main()
