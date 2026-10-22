"""USGS instantaneous water level around each SWOT overpass of the matched reach.

Values come from the USGS Water Data API (continuous collection, api.waterdata.usgs.gov/ogcapi/v1),
which replaces the legacy NWIS instantaneous-values service (decommissioned in February 2027).
The data used in the paper were retrieved from the legacy service in October 2026 and are archived
with this repository (data/usgs_gauges_used.zip); this script reproduces them from the new API.
Parameter 63160 (water-surface elevation above NAVD88) is preferred over 00065 (gage height).
Only approved or provisional values without a qualifier (for example estimated or ice-affected) are
kept, and only values within +/-45 min of an overpass of the gauge reach.
Output: gauges/usgs/<site>.csv with time (UTC), value (ft), unit_of_measure,
parameter_code, approval_status.
"""
import sys
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from p39.paths import DATA, REACH_DIR, GAUGE_DIR, USGS_OGC, START, END

CONT = f"{USGS_OGC}/collections/continuous/items"
WINDOW = pd.Timedelta(minutes=45)
CHUNK = pd.Timedelta(days=1000)    # the API accepts at most 1100 days per request


def _pages(session, params):
    """All features of one query, following the 'next' links; retries on 429/5xx."""
    url, out = CONT, []
    while url:
        for attempt in range(5):
            try:
                r = session.get(url, params=params, timeout=240)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"service error: {url}")
        d = r.json()
        out += [f["properties"] for f in d.get("features", [])]
        url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
        params = None                  # the next link already carries the query
    return out


def fetch(site, session, pc):
    rows, a, b = [], pd.Timestamp(START), pd.Timestamp(END)
    while a < b:
        e = min(a + CHUNK, b)
        rows += _pages(session, {"monitoring_location_id": site, "parameter_code": pc, "limit": 50000,
                                 "time": f"{a:%Y-%m-%dT%H:%M:%SZ}/{e:%Y-%m-%dT%H:%M:%SZ}", "f": "json"})
        a = e
    return pd.DataFrame(rows)


def tidy(raw, pc):
    """Keep clean values; if the site has several sub-location series, keep the longest."""
    if raw.empty:
        return pd.DataFrame()
    ok = raw["approval_status"].isin(["Approved", "Provisional"]) & raw["value"].notna()
    ok &= raw["qualifier"].map(lambda q: q is None or (isinstance(q, float) and pd.isna(q)) or len(q) == 0)
    d = raw[ok]
    if d.empty:
        return pd.DataFrame()
    best = d["time_series_id"].value_counts().idxmax()
    d = d[d["time_series_id"] == best]
    return pd.DataFrame({"time": pd.to_datetime(d["time"], utc=True), "value": d["value"].astype(float),
                         "unit_of_measure": "ft", "parameter_code": pc,
                         "approval_status": d["approval_status"].str[0]}).sort_values("time")


def one_gauge(row):
    out = GAUGE_DIR / f"{row.site}.csv"
    if out.exists():
        return row.site, "skip"
    fr = REACH_DIR / f"{row.reach_id}.csv"
    if not fr.exists() or fr.stat().st_size < 10:
        return row.site, "no SWOT reach data"
    times = pd.to_datetime(pd.read_csv(fr)["time_str"], utc=True).dropna().drop_duplicates()
    try:
        session = requests.Session()
        df = pd.DataFrame()
        for pc in ("63160", "00065"):
            df = tidy(fetch(row.site, session, pc), pc)
            if not df.empty:
                break
    except RuntimeError:
        return row.site, "FAILED (service error; no file written, retried on next run)"
    if not df.empty:
        tv = df["time"].values.astype("datetime64[ns]")
        keep = pd.Series(False, index=df.index)
        for t in times:
            t0 = (t - WINDOW).tz_convert(None).to_datetime64()
            t1 = (t + WINDOW).tz_convert(None).to_datetime64()
            keep |= (tv >= t0) & (tv <= t1)
        df = df[keep.values]
    df.to_csv(out, index=False)
    return row.site, f"{len(df)} values"


def main():
    g = pd.read_csv(DATA / "gauge_match.csv")
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one_gauge, r) for r in g.itertuples()]
        for f in as_completed(futs):
            site, msg = f.result()
            done += 1
            if done % 20 == 0 or done == len(futs):
                print(f"{done}/{len(futs)} {site} {msg}", flush=True)


if __name__ == "__main__":
    main()
