"""Download SWOT RiverSP Version D reach and node time series from Hydrocron.

Usage: python 04_hydrocron_download.py <reach_list.csv> [n_workers]
The list needs a column reach_id. Node IDs come from SWORD v17b. One CSV per reach is
written to swot/reach/<reach_id>.csv and swot/node/<reach_id>.csv; existing files are
skipped, so the script can be restarted.
"""
import io
import sys
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from p39.paths import HYDROCRON, COLLECTION, START, END, DATA, NODE_DIR, REACH_DIR

REACH_FIELDS = ("reach_id,time_str,wse,wse_u,wse_r_u,slope,slope_u,slope2,slope2_u,width,"
                "reach_q,reach_q_b,dark_frac,ice_clim_f,n_good_nod,obs_frac_n,partial_f,"
                "xovr_cal_q,p_length,p_dist_out,geoid_hght,layovr_val,xtrk_dist,crid,"
                "cycle_id,pass_id")
NODE_FIELDS = ("node_id,reach_id,time_str,wse,wse_u,wse_r_u,wse_sm,wse_sm_u,node_q,node_q_b,"
               "dark_frac,ice_clim_f,xtrk_dist,p_dist_out,lat,lon,layovr_val,width,"
               "geoid_hght,n_good_pix,crid,cycle_id,pass_id")
FILL = -999999999999.0


def query(feature, fid, fields, session, keep_no_data=False):
    params = {"feature": feature, "feature_id": str(fid), "start_time": START,
              "end_time": END, "output": "csv", "collection_name": COLLECTION,
              "fields": fields}
    for attempt in range(6):
        try:
            r = session.get(HYDROCRON, params=params, timeout=300)
            if r.status_code == 400 and "not found" in r.text.lower():
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * 2 ** attempt)
                continue
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.json()["results"]["csv"]))
            df = df[[c for c in df.columns if not c.endswith("_units")]]
            # Unobserved nodes come as time_str == "no_data" but still carry the
            # official smoothed height wse_sm, which enters the reach wse; keep them.
            if not keep_no_data:
                df = df[df["time_str"] != "no_data"]
            return df.replace(FILL, float("nan"))
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"{feature} {fid} failed")


def one_reach(reach_id, node_ids):
    fr, fn = REACH_DIR / f"{reach_id}.csv", NODE_DIR / f"{reach_id}.csv"
    if fr.exists() and fn.exists():
        return reach_id, "skip"
    s = requests.Session()
    r = query("Reach", reach_id, REACH_FIELDS, s)
    if r is None or r.empty:
        pd.DataFrame().to_csv(fr, index=False)
        pd.DataFrame().to_csv(fn, index=False)
        return reach_id, "no reach data"
    parts = [query("Node", n, NODE_FIELDS, s, keep_no_data=True) for n in node_ids]
    parts = [p for p in parts if p is not None and not p.empty]
    nodes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    r.to_csv(fr, index=False)
    nodes.to_csv(fn, index=False)
    return reach_id, f"{len(r)} reach rows, {len(nodes)} node rows"


def main():
    lst = pd.read_csv(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    nodes = pd.read_parquet(DATA / "sword_nodes.parquet", columns=["node_id", "reach_id"])
    want = set(lst["reach_id"].astype("int64"))
    groups = nodes[nodes["reach_id"].isin(want)].groupby("reach_id")["node_id"].apply(list)
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one_reach, rid, ids) for rid, ids in groups.items()]
        for f in as_completed(futs):
            rid, msg = f.result()
            done += 1
            if done % 25 == 0 or done == len(futs):
                print(f"{done}/{len(futs)} {rid} {msg}", flush=True)


if __name__ == "__main__":
    main()
