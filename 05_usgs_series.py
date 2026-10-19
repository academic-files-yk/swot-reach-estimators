"""USGS instantaneous water level around each SWOT overpass of the matched reach.

The NWIS instantaneous-values service is queried once per gauge for the whole SWOT
period (parameters 63160 and 00065). Values are converted to UTC from the reported
time-zone code, and only values within +/-45 min of an overpass of the gauge reach are
kept. Values whose qualification code is not approved ('A') or provisional ('P'), for
example estimated ('e') or affected by ice, are dropped.
Output: gauges/usgs/<site>.csv with time (UTC), value (ft), unit_of_measure,
parameter_code, approval_status.
"""
import io
import sys
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from p39.paths import DATA, REACH_DIR, GAUGE_DIR

NWIS_IV = "https://nwis.waterservices.usgs.gov/nwis/iv/"
WINDOW = pd.Timedelta(minutes=45)
TZ = {"UTC": 0, "GMT": 0, "EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7,
      "MDT": -6, "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8, "HST": -10, "AST": -4}


def _get(session, site_no, a, b):
    """One request; returns a DataFrame, None (no data) or 'split' after repeated 503."""
    params = {"format": "rdb", "sites": site_no, "parameterCd": "00065,63160",
              "startDT": a, "endDT": b}
    for attempt in range(4):
        try:
            r = session.get(NWIS_IV, params=params, timeout=240)
            if r.status_code == 404:
                return None
            if r.status_code == 503:
                time.sleep(5)
                continue
            r.raise_for_status()
            lines = [l for l in r.text.splitlines() if not l.startswith("#")]
            if len(lines) < 3:
                return None
            return pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
    return "split"


def fetch(site_no, session):
    """Whole SWOT period in yearly pieces; a piece that keeps failing is split by month."""
    parts = []
    edges = ["2023-07-26", "2024-07-01", "2025-07-01", "2026-10-01"]
    for a, b in zip(edges[:-1], edges[1:]):
        res = _get(session, site_no, a, b)
        if isinstance(res, str):
            months = pd.date_range(a, b, freq="MS").strftime("%Y-%m-%d").tolist()
            months = sorted(set([a] + [m for m in months if a < m < b] + [b]))
            for m0, m1 in zip(months[:-1], months[1:]):
                sub = _get(session, site_no, m0, m1)
                if isinstance(sub, str):
                    # weekly pieces; a week that still fails is skipped and logged
                    days = pd.date_range(m0, m1, freq="7D").strftime("%Y-%m-%d").tolist()
                    days = sorted(set(days + [m1]))
                    for d0, d1 in zip(days[:-1], days[1:]):
                        w = _get(session, site_no, d0, d1)
                        if isinstance(w, str):
                            with open(GAUGE_DIR.parent / "usgs_skipped_periods.csv", "a") as fh:
                                fh.write(f"{site_no},{d0},{d1}\n")
                        elif w is not None:
                            parts.append(w)
                elif sub is not None:
                    parts.append(sub)
        elif res is not None:
            parts.append(res)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates(["datetime", "tz_cd"])


def tidy(raw):
    """Wide NWIS table -> long table in UTC, one parameter (63160 preferred)."""
    off = raw["tz_cd"].map(TZ)
    t = pd.to_datetime(raw["datetime"], format="%Y-%m-%d %H:%M") - pd.to_timedelta(off, unit="h")
    out = []
    for pc in ("63160", "00065"):
        cols = [c for c in raw.columns if c.endswith(f"_{pc}")]
        for c in cols:
            v = pd.to_numeric(raw[c], errors="coerce")
            q = raw.get(c + "_cd", pd.Series("", index=raw.index)).fillna("")
            tok = q.str.split()
            ok = v.notna() & off.notna() & tok.map(
                lambda x: len(x) > 0 and x[0] in ("A", "P") and set(x) <= {"A", "P", "R"})
            if ok.sum():
                out.append(pd.DataFrame({"time": t[ok].dt.tz_localize("UTC"), "value": v[ok],
                                         "unit_of_measure": "ft", "parameter_code": pc,
                                         "approval_status": q[ok].str[0]}))
        if out:
            # several sub-location series: keep the one with most values
            best = max(out, key=len)
            return best
    return pd.DataFrame()


def one_gauge(row):
    out = GAUGE_DIR / f"{row.site}.csv"
    if out.exists():
        return row.site, "skip"
    fr = REACH_DIR / f"{row.reach_id}.csv"
    if not fr.exists() or fr.stat().st_size < 10:
        return row.site, "no SWOT reach data"
    times = pd.to_datetime(pd.read_csv(fr)["time_str"], utc=True).dropna().drop_duplicates()
    try:
        agency, site_no = row.site.split("-", 1)
        raw = fetch(site_no if agency == "USGS" else f"{agency}:{site_no}", requests.Session())
    except RuntimeError:
        return row.site, "FAILED (service error; no file written, retried on next run)"
    df = tidy(raw) if raw is not None else pd.DataFrame()
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
