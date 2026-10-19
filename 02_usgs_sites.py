"""Inventory of USGS stream gauges with continuous stage or NAVD88 water level.

Output: data/usgs_sites.csv (one row per site) with location, datum, datum method,
drainage area and which continuous water-level series exist during the SWOT period.
"""
import time
import requests
import pandas as pd
from p39.paths import USGS_OGC, DATA

S = requests.Session()


def pages(url, params):
    """Yield features from an OGC API collection, following 'next' links."""
    while url:
        for attempt in range(6):
            try:
                r = S.get(url, params=params, timeout=180)
                if r.status_code == 429:
                    time.sleep(30 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"failed: {url}")
        d = r.json()
        yield from d["features"]
        url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
        params = None


def main():
    rows = []
    for f in pages(f"{USGS_OGC}/collections/monitoring-locations/items",
                   {"f": "json", "site_type_code": "ST", "limit": 10000,
                    "properties": "id,monitoring_location_name,state_name,altitude,"
                                  "altitude_accuracy,altitude_method_code,"
                                  "altitude_method_name,vertical_datum,drainage_area"}):
        p = f["properties"]
        g = f.get("geometry") or {}
        lon, lat = (g.get("coordinates") or [None, None])[:2]
        rows.append({**p, "site": f["id"], "lon": lon, "lat": lat})
    sites = pd.DataFrame(rows)
    print("stream sites:", len(sites))

    ts = []
    for pc in ("00065", "63160"):
        for f in pages(f"{USGS_OGC}/collections/time-series-metadata/items",
                       {"f": "json", "parameter_code": pc,
                        "computation_identifier": "Instantaneous", "limit": 10000,
                        "properties": "monitoring_location_id,parameter_code,begin,end,"
                                      "primary,sublocation_identifier"}):
            ts.append(f["properties"])
    ts = pd.DataFrame(ts)
    ts["begin"] = pd.to_datetime(ts["begin"], errors="coerce")
    ts["end"] = pd.to_datetime(ts["end"], errors="coerce")
    ts = ts[(ts["end"] >= "2023-09-01") & (ts["begin"] <= "2026-06-01")]
    ts.to_csv(DATA / "usgs_timeseries_meta.csv", index=False)
    has = ts.groupby(["monitoring_location_id", "parameter_code"]).size().unstack(fill_value=0)
    has = (has > 0).rename(columns={"00065": "has_00065", "63160": "has_63160"})
    sites = sites.merge(has, left_on="site", right_index=True, how="inner")
    sites.to_csv(DATA / "usgs_sites.csv", index=False)
    print("sites with continuous water level in SWOT period:", len(sites))
    print(sites["vertical_datum"].value_counts().head(8).to_string())
    print(sites["altitude_method_code"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
