"""Fig. 3: reach slope against IRIS.
(a,b) reach-median SWOT slope vs IRIS slope (log-log density) for PUB and Theil-Sen;
(c) ECDF of |error| for all slope estimators; (d) 68th percentile of single-overpass
|error| by end-node observation state.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from p39.paths import RESULTS, FIGURES, OUTPUT
from p39 import figstyle as F

LAB = {"official": "PUB", "official_s2": "PUB-S2", "bayes_s2": "BAYES-$\\sigma^2$",
       "ols": "OLS", "wls": "WLS", "theilsen": "Theil–Sen", "huber": "Huber"}
COL = {"official": F.PALETTE[1], "official_s2": F.PALETTE[3], "bayes_s2": F.PALETTE[6],
       "ols": F.PALETTE[5], "wls": "#b0aea8", "theilsen": F.PALETTE[0], "huber": F.PALETTE[2]}


def main():
    F.style()
    r = pd.read_parquet(RESULTS / "slope_reach.parquet")
    r = r[r["node_set"] == "clean"]
    est = [e for e in LAB if e in r["estimator"].unique()]
    wide = r.pivot_table(index="reach_id", columns="estimator", values="err_mm_km").dropna(subset=est)
    r = r[r["reach_id"].isin(wide.index)]
    fig, axs = plt.subplots(1, 4, figsize=(F.COL2, 0.30 * F.COL2),
                            gridspec_kw={"wspace": 0.42})
    lim = (1, 3000)
    for ax, e, letter in ((axs[0], "official", "a"), (axs[1], "theilsen", "b")):
        d = r[(r["estimator"] == e) & (r["swot_slope"] > 0)]
        ax.hexbin(d["iris_slope"], d["swot_slope"], xscale="log", yscale="log", gridsize=28,
                  extent=(0, np.log10(lim[1]), 0, np.log10(lim[1])), mincnt=1, cmap="Blues",
                  linewidths=0.1)
        ax.plot(lim, lim, color=F.INK, lw=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("IRIS slope (mm/km)")
        ax.set_ylabel("SWOT reach-median slope (mm/km)" if letter == "a" else "")
        nneg = (r[r["estimator"] == e]["swot_slope"] <= 0).sum()
        ax.text(0.04, 0.96, f"$|e|_{{68}}$ = {np.quantile(np.abs(r[r.estimator == e].err_mm_km), .68):.0f} mm/km\n"
                f"$\\leq$0 not shown: {nneg}", transform=ax.transAxes, fontsize=6, va="top")
        F.panel(ax, letter, LAB[e])
    ax = axs[2]
    for e in est:
        x = np.sort(r[r["estimator"] == e]["err_mm_km"].abs().values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COL[e], lw=0.9, label=LAB[e])
    ax.axvline(17, color=F.INK2, lw=0.5, ls="--")
    ax.set_xscale("log"); ax.set_xlim(1, 1000); ax.set_ylim(0, 1)
    ax.set_xlabel("|Reach-median slope error| (mm/km)"); ax.set_ylabel("Cumulative share of reaches")
    handles, labels = ax.get_legend_handles_labels()   # one legend for (c) and (d), above the panels
    F.panel(ax, "c", "All estimators")
    ax = axs[3]
    p = pd.read_parquet(RESULTS / "slope_pass.parquet")
    p = p[(p["node_set"] == "clean") & p["reach_id"].isin(wide.index)].dropna(subset=["slope_mm_km"])
    en = pd.read_parquet(RESULTS / "iris_endnodes.parquet")
    for c in ("cycle_id", "pass_id"):
        en[c] = en[c].astype(p[c].dtype)
    m = p.merge(en, on=["reach_id", "cycle_id", "pass_id"])
    m["ends"] = np.select([m.dn_end_obs & m.up_end_obs, m.dn_end_obs | m.up_end_obs],
                          ["both", "one"], "none")
    order = ["both", "one", "none"]
    for k, e in enumerate(est):
        v = [np.quantile(np.abs(m[(m.ends == o) & (m.estimator == e)]["err_mm_km"]), .68) for o in order]
        ax.plot(np.arange(3) + (k - len(est) / 2) * 0.08, v, "o", ms=2.6, color=COL[e])
    ax.set_xticks(range(3), ["both", "one", "none"])
    ax.set_xlabel("Observed end nodes"); ax.set_ylabel("Single-overpass $|e|_{68}$ (mm/km)")
    ax.set_ylim(0, None)
    F.panel(ax, "d", "Overpass errors")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=5.8, handlelength=1.4,
               columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.18, top=0.80)
    F.save(fig, "Fig3_slope", FIGURES)
    m.groupby(["ends", "estimator"])["err_mm_km"].agg(lambda x: np.quantile(np.abs(x), .68)) \
        .unstack().to_csv(OUTPUT / "slope_by_endnodes.csv")


if __name__ == "__main__":
    main()
