"""Compare estimator WSE at the gauge location with datum-converted gauge water level.

Inputs: results/gauge_estimates.parquet (06), data/gauge_match.csv (03), gauges/usgs/*.csv
(05), swot/node/<reach_id>.csv (geoid height at the gauge node).
Output: results/wse_eval.parquet, one row per gauge x overpass x node set, with the
predicted WSE of every estimator and the gauge truth (both above EGM2008, m).

Predictions at the gauge flow distance x_g:
  official      reach wse + reach slope * (x_g - x_ref), x_ref = mean flow distance of
                nodes with wse_sm (checked in 08 against the alternatives)
  official_s2   as above with slope2
  official_sm   wse_sm of the gauge node (node-level output of the official smoother)
  nearest_node  raw wse of the gauge node (no aggregation)
  line fits     ols, wls, theilsen, huber: own level and slope
  level only    mean, wmean, median: own level, moved with the official slope
"""
import numpy as np
import pandas as pd
from p39.paths import DATA, RESULTS, GAUGE_DIR, NODE_DIR
from p39.geodesy import navd88_to_swot_ellipsoid, decimal_year, FT
from p39 import estimators as E

MAX_DT = pd.Timedelta(minutes=30)


def gauge_truth(site, times, alt_ft):
    """Gauge NAVD88 height (m) at SWOT times by linear interpolation within 30 min."""
    f = GAUGE_DIR / f"{site}.csv"
    if not f.exists() or f.stat().st_size < 10:
        return pd.Series(np.nan, index=times)
    g = pd.read_csv(f)
    if g.empty or "value" not in g:
        return pd.Series(np.nan, index=times)
    g["time"] = pd.to_datetime(g["time"], utc=True)
    g["value"] = pd.to_numeric(g["value"], errors="coerce")
    g = g.dropna(subset=["value"]).drop_duplicates("time").sort_values("time")
    if g.empty:
        return pd.Series(np.nan, index=times)
    pc = str(g["parameter_code"].iloc[0]).zfill(5)   # read as integer from CSV
    h = g["value"] * FT if g["unit_of_measure"].iloc[0] == "ft" else g["value"]
    if pc == "00065":
        h = h + alt_ft * FT
    tg = g["time"].values.astype("datetime64[ns]").astype("int64")
    out = []
    for t in times:
        ti = np.datetime64(t.tz_convert(None)).astype("datetime64[ns]").astype("int64")
        j = np.searchsorted(tg, ti)
        if 0 < j < len(tg) and (tg[j] - tg[j - 1]) <= 2 * MAX_DT.value \
                and min(ti - tg[j - 1], tg[j] - ti) <= MAX_DT.value:
            w = (ti - tg[j - 1]) / (tg[j] - tg[j - 1])
            out.append(h.iloc[j - 1] * (1 - w) + h.iloc[j] * w)
        elif j < len(tg) and tg[j] == ti:
            out.append(h.iloc[j])
        else:
            out.append(np.nan)
    return pd.Series(out, index=times), pc


def main():
    est = pd.read_parquet(RESULTS / "gauge_estimates.parquet")
    gm = pd.read_csv(DATA / "gauge_match.csv")
    rows = []
    for g in gm.itertuples():
        e = est[est["reach_id"] == g.reach_id]
        if e.empty:
            continue
        fn = NODE_DIR / f"{g.reach_id}.csv"
        nod = pd.read_csv(fn)
        gn = nod[nod["node_id"] == g.node_id]
        geoid_static = gn["geoid_hght"].median()     # EGM2008 at the node is static
        gn = gn.set_index(["cycle_id", "pass_id"])
        times = pd.DatetimeIndex(e["t"].drop_duplicates())
        res = gauge_truth(g.site, times, g.altitude)
        if isinstance(res, pd.Series):
            continue
        truth_navd, pc = res
        xg = g.node_dist_out_m
        for r in e.itertuples():
            key = (r.cycle_id, r.pass_id)
            node = gn.loc[key] if key in gn.index else None
            if isinstance(node, pd.DataFrame):
                node = node.iloc[0]
            geoid = geoid_static
            hn = truth_navd.get(r.t, np.nan)
            h_itrf = navd88_to_swot_ellipsoid(g.lon, g.lat, hn, decimal_year([r.t.tz_convert(None)])[0])
            row = {"site": g.site, "reach_id": g.reach_id, "node_id": g.node_id,
                   "cycle_id": r.cycle_id, "pass_id": r.pass_id, "t": r.t,
                   "node_set": r.node_set, "parameter_code": pc,
                   "truth_navd88": hn, "geoid_hght": geoid,
                   "truth": float(h_itrf) - geoid,
                   "x_gauge": xg, "x_ref_sm": r.x_mean_sm,
                   "dx_gauge_centre": xg - r.x_mean_sm,
                   "off_wse_u": r.off_wse_u, "off_slope_u": r.off_slope_u,
                   "off_slope": r.off_slope, "off_slope2": r.off_slope2,
                   "off_reach_q": r.off_reach_q, "off_obs_frac_n": r.off_obs_frac_n,
                   "off_dark_frac": r.off_dark_frac, "off_xtrk_dist": r.off_xtrk_dist,
                   "off_ice_clim_f": r.off_ice_clim_f, "off_width": r.off_width,
                   "off_partial_f": r.off_partial_f, "n_nodes": r.n_nodes,
                   "scatter_nmad": getattr(r, "scatter_nmad", np.nan),
                   "node_layover": getattr(r, "node_layover", np.nan),
                   "ols_slope": getattr(r, "ols_slope", np.nan)}
            off_s, off_s2 = r.off_slope, r.off_slope2
            row["official"] = E.level_at(r.x_mean_sm, r.off_wse, off_s, xg)
            row["official_s2"] = E.level_at(r.x_mean_sm, r.off_wse, off_s2, xg)
            row["official_sm"] = node["wse_sm"] if node is not None else np.nan
            row["nearest_node"] = node["wse"] if node is not None else np.nan
            row["nearest_node_q"] = node["node_q"] if node is not None else np.nan
            for k in E.LINE:
                row[k] = E.level_at(getattr(r, f"{k}_xref", np.nan),
                                    getattr(r, f"{k}_level", np.nan),
                                    getattr(r, f"{k}_slope", np.nan), xg)
            for k in E.LEVEL:
                row[k] = E.level_at(getattr(r, f"{k}_xref", np.nan),
                                    getattr(r, f"{k}_level", np.nan), off_s, xg)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "wse_eval.parquet", index=False)
    print(len(df), "rows;", df["truth"].notna().sum(), "with truth;",
          df["site"].nunique(), "gauges")


if __name__ == "__main__":
    main()
