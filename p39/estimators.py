"""Node-to-reach estimators.

Coordinates: x is the SWORD flow distance from the outlet (m, increases upstream), y is
node WSE (m), s is the node WSE uncertainty (m). With this convention the fitted dy/dx is
the downstream water surface slope (positive for a normal river).

Every estimator returns (x_ref, level, slope): the WSE `level` at flow distance `x_ref`
and a slope (m/m). Level-only estimators return slope = NaN; the caller decides which
slope moves their level to an evaluation point.
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm

MIN_NODES = 5
LINE = ("ols", "wls", "theilsen", "huber")
LEVEL = ("mean", "wmean", "median")


def _weights(s):
    w = 1.0 / np.square(np.clip(s, 0.01, None))   # floor 1 cm to avoid huge weights
    return w / w.sum()


def est_mean(x, y, s):
    return x.mean(), y.mean(), np.nan


def est_wmean(x, y, s):
    w = _weights(s)
    return np.sum(w * x), np.sum(w * y), np.nan


def est_median(x, y, s):
    return np.median(x), np.median(y), np.nan


def est_ols(x, y, s):
    xm = x.mean()
    b, a = np.polyfit(x - xm, y, 1)
    return xm, a, b


def est_wls(x, y, s):
    w = _weights(s)
    xm = np.sum(w * x)
    b, a = np.polyfit(x - xm, y, 1, w=np.sqrt(w))
    return xm, a, b


def est_theilsen(x, y, s):
    xm = np.median(x)
    res = stats.theilslopes(y, x - xm)
    return xm, res.intercept, res.slope


def est_huber(x, y, s):
    xm = x.mean()
    X = sm.add_constant(x - xm)
    fit = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return xm, fit.params[0], fit.params[1]


ESTIMATORS = {
    "mean": est_mean, "wmean": est_wmean, "median": est_median,
    "ols": est_ols, "wls": est_wls, "theilsen": est_theilsen, "huber": est_huber,
}


def run_all(x, y, s):
    """Apply all estimators to one reach overpass. Returns dict name -> (x_ref, level, slope)."""
    x, y, s = map(np.asarray, (x, y, s))
    if len(x) < MIN_NODES or np.ptp(x) <= 0:
        return {}
    out = {}
    for name, f in ESTIMATORS.items():
        try:
            out[name] = tuple(float(v) for v in f(x, y, s))
        except Exception:
            out[name] = (np.nan, np.nan, np.nan)
    return out


def detrended_scatter(x, y):
    """Robust scatter (NMAD) of node residuals around an OLS line, m."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < MIN_NODES:
        return np.nan
    b, a = np.polyfit(x - x.mean(), y, 1)
    r = y - (a + b * (x - x.mean()))
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def level_at(x_ref, level, slope, x_eval):
    """Move a level along the flow axis with a given slope."""
    return level + slope * (x_eval - x_ref)
