"""Fig. 4: WSE at the gauges. (a) ECDF of |absolute error|, (b) ECDF of |relative error|
(gauge median removed), (c) per-gauge median error of SM-node against gauge latitude, by
datum survey method. Also writes output/wse_datum_audit.csv.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from p39.paths import RESULTS, DATA, OUTPUT, FIGURES
from p39 import figstyle as F
from p39.sets import WSE_REQUIRED

EST = ["official", "official_s2", "bayes_s2", "official_sm", "nearest_node", "ols", "wls",
       "theilsen", "huber"]
LAB = {"official": "PUB", "official_s2": "PUB-S2", "bayes_s2": "BAYES-$\\sigma^2$ node",
       "official_sm": "SM-node", "nearest_node": "Raw node", "ols": "OLS", "wls": "WLS",
       "theilsen": "Theil–Sen", "huber": "Huber"}
COL = {"official": F.PALETTE[1], "official_s2": F.PALETTE[3], "bayes_s2": F.PALETTE[6],
       "official_sm": F.PALETTE[4], "nearest_node": F.PALETTE[7], "ols": F.PALETTE[5],
       "wls": "#b0aea8", "theilsen": F.PALETTE[0], "huber": F.PALETTE[2]}
METHOD = {"L": "Levelling", "D": "GPS/GNSS", "F": "GPS/GNSS", "W": "GPS/GNSS", "X": "GPS/GNSS",
          "Y": "GPS/GNSS", "Z": "GPS/GNSS"}


def ecdf(ax, x, **kw):
    x = np.sort(np.abs(x[np.isfinite(x)]))
    ax.plot(x * 100, np.arange(1, len(x) + 1) / len(x), **kw)


def main():
    F.style()
    d = pd.read_parquet(RESULTS / "wse_long.parquet")
    d = d[(d["node_set"] == "clean") & ~d["screened"]]
    est = [e for e in EST if d[e + "_err"].notna().any()]
    d = d.dropna(subset=[e + "_err" for e in WSE_REQUIRED])
    fig, axs = plt.subplots(1, 3, figsize=(F.COL2, 0.33 * F.COL2), gridspec_kw={"wspace": 0.32})
    for e in est:
        ecdf(axs[0], d[e + "_err"].values, color=COL[e], lw=0.9, label=LAB[e])
        rel = d[e + "_err"] - d.groupby("site")[e + "_err"].transform("median")
        ecdf(axs[1], rel.values, color=COL[e], lw=0.9, label=LAB[e])
    for ax, letter, t in ((axs[0], "a", "Absolute error"), (axs[1], "b", "Relative error")):
        ax.set_xscale("log"); ax.set_xlim(1, 500); ax.set_ylim(0, 1)
        ax.axhline(0.68, color=F.INK2, lw=0.5, ls=":")
        ax.set_xlabel("|WSE error| (cm)")
        F.panel(ax, letter, t)
    axs[0].set_ylabel("Cumulative share of overpasses")
    handles, labels = axs[1].get_legend_handles_labels()   # shared by (a) and (b), placed above the panels
    gm = pd.read_csv(DATA / "gauge_match.csv").set_index("site")
    g = d.groupby("site").agg(off=("official_sm_err", "median"), pc=("parameter_code", "first"))
    g = g.join(gm[["lat", "altitude_method_code"]])
    g["method"] = g["altitude_method_code"].map(METHOD)
    ax = axs[2]
    for k, (m, gg) in enumerate(g.groupby("method")):
        ax.scatter(gg["lat"], 100 * gg["off"], s=2.5, lw=0, color=F.PALETTE[k * 2], alpha=0.8,
                   label=f"{m} ({len(gg)})")
    bins = np.arange(24, 51, 4)
    mid = g.groupby(pd.cut(g["lat"], bins), observed=True)["off"].median()
    ax.plot([b.mid for b in mid.index], 100 * mid.values, "-", color=F.INK, lw=1.0, label="Median (4° bins)")
    ax.axhline(0, color=F.INK2, lw=0.5)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("Gauge latitude (°N)"); ax.set_ylabel("Per-gauge median error (cm)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=3, fontsize=5.8, markerscale=2, handlelength=1.0, columnspacing=0.8,
              borderaxespad=0)
    out = ((100 * g["off"] < -60) | (100 * g["off"] > 80)).sum()
    ax.text(1.0, 1.01, f"outside axis: {out}", transform=ax.transAxes, fontsize=5.6, color=F.INK2,
            ha="right", va="bottom")
    F.panel(ax, "c", "SM-node datum audit")
    fig.legend(handles, labels, loc="upper left", ncol=5, fontsize=5.8, handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.05, 1.0))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.16, top=0.78)
    F.save(fig, "Fig4_wse", FIGURES)
    rows = [dict(group="all", n=len(g), median_cm=100 * g["off"].median(),
                 q25_cm=100 * g["off"].quantile(.25), q75_cm=100 * g["off"].quantile(.75))]
    for col in ("pc", "method"):
        for k, gg in g.groupby(col):
            rows.append(dict(group=f"{col}={k}", n=len(gg), median_cm=100 * gg["off"].median(),
                             q25_cm=100 * gg["off"].quantile(.25), q75_cm=100 * gg["off"].quantile(.75)))
    for k, gg in g.groupby(pd.cut(g["lat"], [24, 32, 38, 44, 50]), observed=True):
        rows.append(dict(group=f"lat={k}", n=len(gg), median_cm=100 * gg["off"].median(),
                         q25_cm=100 * gg["off"].quantile(.25), q75_cm=100 * gg["off"].quantile(.75)))
    pd.DataFrame(rows).to_csv(OUTPUT / "wse_datum_audit.csv", index=False)
    # does the reach-level offset depend on the gauge distance from the reach centre?
    gg = d.groupby("site").agg(pub=("official_err", "median"), sm=("official_sm_err", "median"),
                               dx=("dx_gauge_centre", "median"))
    gg["dx_bin"] = pd.cut(gg["dx"].abs() / 1000, [0, 1, 2, 3, 5, 10, 50])
    gg.groupby("dx_bin", observed=True)[["pub", "sm"]].agg(lambda x: 100 * x.abs().median()) \
        .assign(n=gg.groupby("dx_bin", observed=True).size()).to_csv(OUTPUT / "wse_offset_by_dx.csv")
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
