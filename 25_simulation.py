"""Monte Carlo test of the reach slope estimators with known truth.

Each synthetic window has three 10-km reaches (50 nodes each, 200 m spacing) with a
continuous piecewise-linear water surface: slope S in the current reach (log-uniform,
20-500 mm/km) and k*S in both neighbours, k in {0.5, 1, 2}. Realism comes from the
data: each reach receives the complete pattern of one real overpass (IRIS sample):
which nodes are observed, their stated uncertainties and their residuals about a line
with the reach's independent IRIS slope (no estimator is favoured), so that
within-overpass error scale, spatial structure, outliers and real slope departures are
kept. A neighbour enters the window only
if more than five of its nodes are observed (processing rule). Estimators: published
form of the reconstruction (diag sigma), diag sigma^2, slope2, OLS, Theil-Sen, Huber.
Outputs: results/simulation.parquet, output/simulation_summary.csv,
output/simulation_by_ends.csv.
"""
import glob
import numpy as np
import pandas as pd
from p39.paths import NODE_DIR, DATA, RESULTS, OUTPUT
from p39.bayes import reconstruct
from p39.slope2 import slope2
from p39 import estimators as E

SEED = 39
N_PER_K = 6000
K = (0.5, 1.0, 2.0)
NODES, DX = 50, 200.0


def empirical_inputs(n_reach=400, seed=SEED):
    """Pool of real overpass patterns, each resampled to 50 node positions: observation
    mask, detrended residual (NaN where unobserved) and stated uncertainty. Residuals
    are taken about a line with the reach's IRIS slope."""
    rng = np.random.default_rng(seed)
    iris = pd.read_csv(DATA / "iris_sample.csv").set_index("reach_id")["avg_combined_slope"] * 1e-6
    ids = rng.choice(iris.index.values, n_reach, replace=False)
    M, R, U = [], [], []
    for rid in ids:
        f = NODE_DIR / f"{rid}.csv"
        if not f.exists() or f.stat().st_size < 10:
            continue
        n = pd.read_csv(f, usecols=["node_id", "cycle_id", "pass_id", "wse", "wse_r_u",
                                    "node_q", "node_q_b", "p_dist_out"]).sort_values("p_dist_out")
        for _, g in n.groupby(["cycle_id", "pass_id"]):
            if len(g) < 10:
                continue
            ok = (g["wse"].notna() & (g["node_q"] < 3) & g["wse_r_u"].gt(0) &
                  ((g["node_q_b"].fillna(0).astype("int64") & (1 << 23)) == 0)).values
            idx = np.round(np.linspace(0, len(g) - 1, NODES)).astype(int)
            m = ok[idx]
            r = np.full(NODES, np.nan)
            u = np.full(NODES, np.nan)
            if ok.sum() >= E.MIN_NODES:
                # residuals about a line with the independent IRIS slope, so that no
                # estimator is favoured; real slope changes and curvature stay in them
                x, y = g["p_dist_out"].values[ok], g["wse"].values[ok]
                xm = np.median(x)
                lev = np.median(y - iris[rid] * (x - xm))
                res = g["wse"].values - (lev + iris[rid] * (g["p_dist_out"].values - xm))
                r = np.where(m, res[idx], np.nan)
                u = np.where(m, g["wse_r_u"].values[idx], np.nan)
            else:
                m = np.zeros(NODES, bool) | (m & False)
            M.append(m); R.append(r); U.append(u)
    return np.array(R), np.array(U), np.array(M)


def one_window(rng, S, k, pool_r, pool_s, masks, iid=False):
    x = np.arange(3 * NODES) * DX                       # downstream -> upstream
    seg = np.repeat([0, 1, 2], NODES)                    # 0 dn, 1 current, 2 up
    slopes = np.array([k * S, S, k * S])
    dz = np.r_[0, np.diff(x)] * slopes[seg]
    truth = np.cumsum(dz)
    pick = rng.integers(0, len(masks), 3)
    m = np.concatenate([masks[i] for i in pick])
    r = np.concatenate([pool_r[i] for i in pick])
    s = np.concatenate([pool_s[i] for i in pick])
    if iid:
        # alternative noise model: same masks, node errors drawn independently from the
        # pooled residual/uncertainty pairs (no within-overpass structure)
        fin = np.isfinite(_POOL[0])
        j = rng.choice(np.flatnonzero(fin), r.size)
        r, s = _POOL[0][j], _POOL[1][j]
    y = np.where(m, truth + r, np.nan)
    cur = seg == 1
    use_nb = {sg: m[seg == sg].sum() > 5 for sg in (0, 2)}
    in_win = cur | ((seg == 0) & use_nb[0]) | ((seg == 2) & use_nb[2])
    keep = in_win & (cur | m)
    xw, yw, sw, mw, cw = x[keep], y[keep], s[keep], m[keep], cur[keep]
    out = {"S": S * 1e6, "k": k, "n_obs": int(m[cur].sum()),
           "dn_end": bool(m[cur][0]), "up_end": bool(m[cur][-1]),
           "nb_dn": use_nb[0], "nb_up": use_nb[2]}
    if m[cur].sum() < E.MIN_NODES:
        return out
    xc = xw[cw]
    for p, name in ((1, "pub"), (2, "bayes_s2")):
        rec = reconstruct(xw, yw, sw, mw, noise_power=p)[cw]
        out[name] = (rec[-1] - rec[0]) / (xc[-1] - xc[0]) * 1e6
    out["slope2"] = slope2(xw, yw, mw, cw) * 1e6
    xo, yo, so = x[cur & m], y[cur & m], s[cur & m]
    for name in ("ols", "theilsen", "huber"):
        out[name] = E.ESTIMATORS[name](xo, yo, so)[2] * 1e6
    return out


_POOL = (None, None)


def main():
    global _POOL
    rng = np.random.default_rng(SEED)
    pool_r, pool_s, masks = empirical_inputs()
    _POOL = (pool_r.ravel(), pool_s.ravel())
    rr = pool_r[np.isfinite(pool_r)]
    print("patterns:", len(masks), "; residual NMAD (cm):",
          round(100 * 1.4826 * np.median(np.abs(rr - np.median(rr))), 1))
    rows = []
    for model in ("pattern", "iid"):
        for k in K:
            for _ in range(N_PER_K):
                S = 10 ** rng.uniform(np.log10(20e-6), np.log10(500e-6))
                rows.append(dict(model=model, **one_window(rng, S, k, pool_r, pool_s, masks,
                                                           iid=(model == "iid"))))
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / "simulation.parquet", index=False)
    est = ["pub", "bayes_s2", "slope2", "ols", "theilsen", "huber"]
    for e in est:
        df[e + "_err"] = df[e] - df["S"]
    df = df.dropna(subset=[e + "_err" for e in est])
    summ = []
    brng = np.random.default_rng(SEED)
    for (model, k), g in df.groupby(["model", "k"]):
        for e in est:
            rel = (g[e + "_err"] / g["S"]).values
            boot = [np.median(rel[brng.integers(0, len(rel), len(rel))]) for _ in range(1000)]
            summ.append(dict(model=model, k=k, estimator=e, n=len(rel),
                             rel_bias_median=np.median(rel),
                             rel_bias_lo=np.quantile(boot, .025), rel_bias_hi=np.quantile(boot, .975),
                             p68=np.quantile(np.abs(g[e + "_err"]), .68)))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUTPUT / "simulation_summary.csv", index=False)
    df["ends"] = np.select([df.dn_end & df.up_end, df.dn_end | df.up_end], ["both", "one"], "none")
    be = df[(df["k"] == 1.0) & (df["model"] == "pattern")].groupby("ends")[[e + "_err" for e in est]] \
        .agg(lambda x: np.quantile(np.abs(x), .68))
    be.to_csv(OUTPUT / "simulation_by_ends.csv")
    print(summ.round(2).to_string(index=False))
    print(be.round(1).to_string())


if __name__ == "__main__":
    main()
