"""Re-implementation of the RiverSP enhanced slope (slope2).

Follows SWOTRiverEstimator.compute_enhanced_slope and gaussian_averaging (RiverObs,
commit dc0c7fd): OLS detrending over the valid nodes of the multi-reach window,
Gaussian moving average of the residuals within a window w = min(10 km, L) with
sigma = max(1 km, w/5), where L is the distance between the first and last valid
node of the current reach; slope2 is the difference of the smoothed heights at those
two nodes divided by L.
"""
import numpy as np


def slope2(ss, wse, valid, is_cur, max_window=10000.0, min_sigma=1000.0, ratio=5.0):
    ss, wse = np.asarray(ss, float), np.asarray(wse, float)
    m = np.asarray(valid, bool) & np.isfinite(wse)
    cur = np.asarray(is_cur, bool) & m
    if cur.sum() < 3:
        return np.nan
    xs, ys = ss[m], wse[m]
    xc = ss[cur]
    L = xc.max() - xc.min()
    if L <= 0:
        return np.nan
    w = min(max_window, L)
    sig = max(min_sigma, w / ratio)
    b, a = np.polyfit(xs, ys, 1)
    r = ys - (a + b * xs)

    def smooth_at(x0):
        k = np.abs(xs - x0) <= w / 2
        wt = np.exp(-0.5 * ((xs[k] - x0) / sig) ** 2)
        return np.sum(wt * r[k]) / np.sum(wt) + a + b * x0

    lo, hi = xc.min(), xc.max()
    return (smooth_at(hi) - smooth_at(lo)) / L
