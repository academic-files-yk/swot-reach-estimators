"""Error by observation condition (RQ3) and Fig. 5.

Slope: single-overpass error against the IRIS median (clean node set).
WSE: relative error at the gauge (gauge median removed), clean node set, screened gauges
excluded. For each condition, overpasses are split into quantile bins (fixed edges for
cross-track distance); the 68th percentile of |error| is computed per estimator and bin.
Outputs: output/drivers_slope.csv, output/drivers_wse.csv, figures/Fig5_drivers.*
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from p39.paths import RESULTS, OUTPUT, FIGURES
from p39 import figstyle as F

SL_EST = ["official", "official_s2", "ols", "theilsen", "huber"]
W_EST = ["official", "official_sm", "ols", "wls", "theilsen", "huber"]
LAB = {"official": "PUB", "official_s2": "PUB-S2", "official_sm": "SM-node", "ols": "OLS",
       "wls": "WLS", "theilsen": "Theil–Sen", "huber": "Huber"}
COL = {"official": F.PALETTE[1], "official_s2": F.PALETTE[3], "official_sm": F.PALETTE[4],
       "ols": F.PALETTE[5], "wls": "#b0aea8", "theilsen": F.PALETTE[0], "huber": F.PALETTE[2]}
P68 = lambda x: np.quantile(np.abs(x), 0.68) if len(x) >= 50 else np.nan


def binned(df, var, est, err_col_fn, edges):
    df = df.copy()
    df["bin"] = pd.cut(df[var], edges, include_lowest=True)
    out = []
    for b, g in df.groupby("bin", observed=True):
        row = {"var": var, "bin": str(b), "x": g[var].median(), "n": len(g)}
        for e in est:
            row[e] = P68(err_col_fn(g, e).dropna())
        out.append(row)
    return out


def main():
    F.style()
    p = pd.read_parquet(RESULTS / "slope_pass.parquet")
    p = p[p["node_set"] == "clean"].dropna(subset=["slope_mm_km"])
    w = p.pivot_table(index=["reach_id", "cycle_id", "pass_id"], columns="estimator",
                      values="err_mm_km").dropna(subset=SL_EST)
    meta = p.drop_duplicates(["reach_id", "cycle_id", "pass_id"]).set_index(
        ["reach_id", "cycle_id", "pass_id"])[["n_nodes", "scatter_nmad", "off_dark_frac",
                                              "off_xtrk_dist", "off_obs_frac_n", "width"]]
    s = w.join(meta, how="inner").reset_index()
    s["xtrk_km"] = s["off_xtrk_dist"].abs() / 1000
    s["scatter_cm"] = 100 * s["scatter_nmad"]
    q = lambda v: np.unique(np.nanquantile(s[v], np.linspace(0, 1, 6)))
    rows = []
    for v, edges in [("n_nodes", q("n_nodes")), ("scatter_cm", q("scatter_cm")),
                     ("off_obs_frac_n", q("off_obs_frac_n")), ("off_dark_frac", [0, .01, .05, .1, .2, 1]),
                     ("xtrk_km", [0, 15, 25, 35, 45, 70]), ("width", q("width"))]:
        rows += binned(s, v, SL_EST, lambda g, e: g[e], edges)
    ds = pd.DataFrame(rows)
    ds.to_csv(OUTPUT / "drivers_slope.csv", index=False)

    wl = pd.read_parquet(RESULTS / "wse_long.parquet")
    wl = wl[(wl["node_set"] == "clean") & ~wl["screened"]].dropna(subset=[e + "_err" for e in W_EST])
    for e in W_EST:
        wl[e + "_rel"] = wl[e + "_err"] - wl.groupby("site")[e + "_err"].transform("median")
    wl["xtrk_km"] = wl["off_xtrk_dist"].abs() / 1000
    wl["scatter_cm"] = 100 * wl["scatter_nmad"]
    qw = lambda v: np.unique(np.nanquantile(wl[v], np.linspace(0, 1, 6)))
    rows = []
    for v, edges in [("n_nodes", qw("n_nodes")), ("scatter_cm", qw("scatter_cm")),
                     ("off_obs_frac_n", qw("off_obs_frac_n")), ("off_dark_frac", [0, .01, .05, .1, .2, 1]),
                     ("xtrk_km", [0, 15, 25, 35, 45, 70]), ("off_width", qw("off_width"))]:
        rows += binned(wl, v, W_EST, lambda g, e: g[e + "_rel"], edges)
    dw = pd.DataFrame(rows)
    dw.to_csv(OUTPUT / "drivers_wse.csv", index=False)

    panels = [("scatter_cm", "Detrended node scatter (cm)", True),
              ("off_obs_frac_n", "Observed node fraction", False),
              ("off_dark_frac", "Dark water fraction", False),
              ("xtrk_km", "|Cross-track distance| (km)", False)]
    fig, axs = plt.subplots(2, 4, figsize=(F.COL2, 0.42 * F.COL2), sharey="row")
    for j, (v, xl, logx) in enumerate(panels):
        for i, (d, est, unit) in enumerate(((ds, SL_EST, "mm/km"), (dw, W_EST, "cm"))):
            ax = axs[i, j]
            vv = v if not (i == 1 and v == "width") else "off_width"
            g = d[d["var"] == vv]
            for e in est:
                yy = g[e] * (100 if unit == "cm" else 1)
                ax.plot(g["x"], yy, "-o", ms=2.2, lw=0.8, color=COL[e], label=LAB[e])
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xl if i == 1 else "")
            if j == 0:
                ax.set_ylabel(f"$|e|_{{68}}$ slope (mm/km)" if i == 0 else "$|e|_{68}$ WSE, relative (cm)")
            F.panel(ax, "abcdefgh"[i * 4 + j], ("Slope vs IRIS" if i == 0 else "WSE vs gauges") if j == 0 else "")
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 20, 50, 100, 200, 500]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0, 1].legend(fontsize=5.6, loc="upper right", handlelength=1.2)
    axs[1, 1].legend(fontsize=5.6, loc="lower left", handlelength=1.2)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.94, hspace=0.38, wspace=0.14)
    F.save(fig, "Fig5_drivers", FIGURES)
    print(ds.round(1).to_string(index=False))
    print(dw.assign(**{e: dw[e] * 100 for e in W_EST}).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
