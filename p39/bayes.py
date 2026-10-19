"""Re-implementation of the RiverSP Bayesian node-height reconstruction (wse_sm).

Follows SWOTAlgorithms/RiverObs (SWOTRiverEstimator.optimal_reconstruct, commit
dc0c7fd, 2026-07-29) with the operational parameters of swot_pixc2rivertile.py:
prior_wse_method = fit, prior_unc_alpha = 1.5 m, char_length_tau = 10 km,
exponential prior covariance, use_multiple_reaches = True, full_noise_cov = False.

The noise covariance is built as diag(r) where r is either the node random
uncertainty sigma (as written in RiverObs get_noise_autocov) or sigma**2 (the
textbook form). `noise_power` selects 1 or 2.
"""
import numpy as np

ALPHA = 1.5
TAU = 10000.0


def reconstruct(ss, wse, sig, mask, noise_power=1, alpha=ALPHA, tau=TAU):
    """Return reconstructed heights at all positions ss.

    ss    along-flow distance of every node in the multi-reach window (m)
    wse   node heights (NaN where not observed)
    sig   node random uncertainty wse_r_u (m)
    mask  True for nodes that enter the estimate
    """
    ss = np.asarray(ss, float) - np.mean(ss)
    wse, sig = np.asarray(wse, float), np.asarray(sig, float)
    msk = np.asarray(mask, bool) & np.isfinite(wse) & np.isfinite(sig) & (sig > 0)
    if msk.sum() < 2:
        return np.full(ss.shape, np.nan)
    # prior: weighted linear fit (weights 1/sigma^2, as in RiverObs)
    w = 1.0 / sig[msk] ** 2
    A = np.c_[ss[msk], np.ones(msk.sum())]
    coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], wse[msk] * np.sqrt(w), rcond=None)[0]
    prior = coef[0] * ss + coef[1]
    # prior covariance (exponential), scaled to alpha^2
    Ry = np.exp(-np.abs(ss[:, None] - ss[None, :]) / tau) * alpha ** 2
    Rv = np.diag(sig[msk] ** noise_power)
    H = np.eye(len(ss))[msk]
    try:
        Ry_inv = np.linalg.pinv(Ry)
        Rv_inv = np.linalg.pinv(Rv)
        post = np.linalg.pinv(Ry_inv + H.T @ Rv_inv @ H)
    except np.linalg.LinAlgError:
        # RiverObs flags these overpasses (BAYES_NO_RECONST) and writes fill values
        return np.full(ss.shape, np.nan)
    K = post @ H.T @ Rv_inv
    K_bar = post @ Ry_inv
    return K @ wse[msk] + K_bar @ prior
