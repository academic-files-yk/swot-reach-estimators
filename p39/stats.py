"""Error statistics with cluster bootstrap and paired tests (DESIGN.md D8, D12)."""
import numpy as np
import pandas as pd
from scipy import stats

SEED = 39
B = 2000


def summary(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    a = np.abs(e)
    if len(e) == 0:
        return dict(n=0, med=np.nan, mae=np.nan, p68=np.nan)
    return dict(n=len(e), med=np.median(e), mae=np.median(a), p68=np.quantile(a, 0.68))


def cluster_bootstrap(df, col, cluster, fun, b=B, seed=SEED):
    """95% CI of fun(df[col]) resampling whole clusters."""
    rng = np.random.default_rng(seed)
    groups = [g[col].values for _, g in df.groupby(cluster)]
    k = len(groups)
    if k < 2:
        return np.nan, np.nan
    vals = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, k, k)
        vals[i] = fun(np.concatenate([groups[j] for j in idx]))
    return tuple(np.nanquantile(vals, [0.025, 0.975]))


def table(df, estimators, cluster, err_suffix="_err"):
    rows = []
    for est in estimators:
        col = est + err_suffix
        d = df[[cluster, col]].dropna()
        s = summary(d[col])
        lo, hi = cluster_bootstrap(d, col, cluster, lambda x: np.quantile(np.abs(x), 0.68))
        lo2, hi2 = cluster_bootstrap(d, col, cluster, np.median)
        rows.append(dict(estimator=est, **s, p68_lo=lo, p68_hi=hi, med_lo=lo2, med_hi=hi2))
    return pd.DataFrame(rows)


def paired(df, a, b, err_suffix="_err"):
    d = df[[a + err_suffix, b + err_suffix]].dropna()
    x, y = np.abs(d.iloc[:, 0]), np.abs(d.iloc[:, 1])
    if len(d) < 10 or np.allclose(x, y):
        return dict(n=len(d), med_diff=np.nan, p=np.nan, share_a_better=np.nan)
    p = stats.wilcoxon(x, y).pvalue
    return dict(n=len(d), med_diff=np.median(x - y), p=p, share_a_better=np.mean(x < y))
