"""Fig. 1: one reach overpass showing how the estimators use the nodes.

Selection rule (fixed, not by appearance): among IRIS-sample overpasses with both
neighbour reaches in the window, neither end node observed, at least 30 clean nodes and
both PUB and Theil-Sen slopes available, take the overpass whose difference
|PUB error| - |Theil-Sen error| (against IRIS) is closest to the median of that
difference in this class. Output: figure + output/example_overpass.csv (identifiers).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from p39.paths import RESULTS, DATA, NODE_DIR, OUTPUT, FIGURES
from p39.bayes import reconstruct
from p39 import estimators as E
from p39 import figstyle as F


def main():
    F.style()
    p = pd.read_parquet(RESULTS / "slope_pass.parquet")
    p = p[p["node_set"] == "clean"]
    w = p.pivot_table(index=["reach_id", "cycle_id", "pass_id"], columns="estimator",
                      values="err_mm_km").dropna(subset=["official", "theilsen"])
    nn = p.drop_duplicates(["reach_id", "cycle_id", "pass_id"]).set_index(
        ["reach_id", "cycle_id", "pass_id"])["n_nodes"]
    b = pd.read_parquet(RESULTS / "iris_bayes.parquet").set_index(["reach_id", "cycle_id", "pass_id"])
    en = pd.read_parquet(RESULTS / "iris_endnodes.parquet").set_index(["reach_id", "cycle_id", "pass_id"])
    for idx in (b, en):
        idx.index = idx.index.set_levels([lv.astype(w.index.levels[i].dtype)
                                          for i, lv in enumerate(idx.index.levels)])
    d = w.join(nn).join(b[["used_up", "used_dn"]]).join(en[["dn_end_obs", "up_end_obs"]]).dropna()
    d = d[d["used_up"].astype(bool) & d["used_dn"].astype(bool) & ~d["dn_end_obs"].astype(bool)
          & ~d["up_end_obs"].astype(bool) & (d["n_nodes"] >= 30)]
    diff = d["official"].abs() - d["theilsen"].abs()
    key = (diff - diff.median()).abs().idxmin()
    rid, cyc, pas = key
    pd.DataFrame([{"reach_id": rid, "cycle_id": cyc, "pass_id": pas, "class_size": len(d),
                   "median_diff_mm_km": diff.median(), "chosen_diff_mm_km": diff.loc[key]}]) \
        .to_csv(OUTPUT / "example_overpass.csv", index=False)

    nb = pd.read_csv(DATA / "neighbours.csv").set_index("reach_id")
    iris = pd.read_csv(DATA / "iris_sample.csv").set_index("reach_id")
    frames = []
    for r, role in ((nb.loc[rid, "up_id"], "nb"), (rid, "cur"), (nb.loc[rid, "dn_id"], "nb")):
        n = pd.read_csv(NODE_DIR / f"{int(r)}.csv")
        n = n[(n["cycle_id"] == cyc) & (n["pass_id"] == pas)]
        if role == "nb":
            n = n[n["time_str"] != "no_data"]
        frames.append(n.assign(role=role))
    win = pd.concat(frames).sort_values("p_dist_out")
    mask = (win["wse"].notna() & (win["node_q"] < 3)).values
    rec = {pw: reconstruct(win["p_dist_out"].values, win["wse"].values, win["wse_r_u"].values,
                           mask, noise_power=pw) for pw in (1, 2)}
    x0 = win.loc[win["role"] == "cur", "p_dist_out"].mean()
    km = (win["p_dist_out"].values - x0) / 1000
    cur = (win["role"] == "cur").values
    clean = mask & cur
    yref = np.nanmedian(win["wse"].values[clean])

    fig, ax = plt.subplots(figsize=(F.COL1, 0.88 * F.COL1))
    ax.axvspan(km[cur].min(), km[cur].max(), color="#f3f3f1", zorder=0)
    ax.scatter(km[mask & ~cur], win["wse"].values[mask & ~cur] - yref, s=3, color="#b9b9b4",
               lw=0, label="Neighbour nodes")
    ax.scatter(km[clean], win["wse"].values[clean] - yref, s=4, color=F.INK, lw=0,
               label="Reach nodes (clean)")
    bad = cur & win["wse"].notna().values & ~mask
    if bad.any():
        ax.scatter(km[bad], win["wse"].values[bad] - yref, s=7, marker="x", lw=0.6,
                   color=F.PALETTE[7], label="Flagged nodes")
    ax.plot(km[cur], win["wse_sm"].values[cur] - yref, color=F.PALETTE[1], lw=1.2,
            label="Published wse_sm (PUB)")
    ax.plot(km[cur], rec[2][cur] - yref, color=F.PALETTE[6], lw=1.0, ls="--",
            label="BAYES-$\\sigma^2$")
    xs, ys, us = (win["p_dist_out"].values[clean], win["wse"].values[clean],
                  win["wse_r_u"].values[clean])
    xr, lev, slp = E.est_theilsen(xs, ys, us)
    xx = np.array([km[cur].min(), km[cur].max()])
    ax.plot(xx, lev + slp * (xx * 1000 + x0 - xr) - yref, color=F.PALETTE[0], lw=1.0,
            label="Theil–Sen")
    s_iris = iris.loc[rid, "avg_combined_slope"] * 1e-6
    ax.plot(xx, lev + s_iris * (xx * 1000 + x0 - xr) - yref, color=F.PALETTE[2], lw=0.8, ls=":",
            label="IRIS slope (through Theil–Sen level)")
    s_pub = w.loc[key, "official"] + iris.loc[rid, "avg_combined_slope"]
    s_ts = w.loc[key, "theilsen"] + iris.loc[rid, "avg_combined_slope"]
    ax.text(0.02, 0.97, f"slope (mm/km): PUB {s_pub:.0f}, Theil–Sen {s_ts:.0f}, "
            f"IRIS {iris.loc[rid, 'avg_combined_slope']:.0f}", transform=ax.transAxes, fontsize=6,
            va="top")
    ax.set_xlabel("Flow distance from reach centre (km, upstream positive)")
    ax.set_ylabel("WSE relative to reach median (m)")
    good = win["wse"].values[mask] - yref
    lo, hi = np.nanpercentile(good, [1, 99])
    pad = 0.25 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    off = int(((win["wse"].values - yref > hi + pad) | (win["wse"].values - yref < lo - pad)).sum())
    if off:
        ax.text(0.98, 0.03, f"{off} nodes off scale", transform=ax.transAxes, fontsize=5.6,
                ha="right", color=F.INK2)
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              handlelength=1.6, columnspacing=0.8)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.31, top=0.97)
    F.save(fig, "Fig2_example", FIGURES)
    print("example:", key, "class size", len(d), "chosen diff", round(diff.loc[key], 1),
          "median", round(diff.median(), 1))


if __name__ == "__main__":
    main()
