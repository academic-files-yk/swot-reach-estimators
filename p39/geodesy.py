"""Vertical datum chain for USGS gauges (DESIGN.md, D3).

NAVD88 orthometric height -> NAD83(2011) ellipsoidal height with GEOID18 -> ITRF2014
ellipsoidal height at the observation epoch (time-dependent Helmert in PROJ; SWOT
products are realised in ITRF2014, PDD D-56413 p. 29). NAD83(2011) heights are
conventional tide-free, while SWOT heights include the permanent tide deformation, so
the permanent deformation of SWOT User Handbook (D-109532) Eq. 11.7 is added. The height
above EGM2008 is then h minus the geoid height that SWOT reports at the node, so gauge
and SWOT use the same geoid.
"""
import numpy as np
import pyproj
from pyproj import Transformer

pyproj.network.set_network_enabled(True)
FT = 0.3048
from pyproj.transformer import TransformerGroup
# Use the GEOID18 + 14-parameter Helmert (ITRF2014 <-> NAD83(2011)) pipeline explicitly.
_T = next(t for t in TransformerGroup("EPSG:6318+5703", "EPSG:7912", always_xy=True).transformers
          if "us_noaa_g2018u0" in t.definition)
H_PERM = -0.31460   # permanent tide-potential amplitude (m), IERS
H2 = 0.6078         # Love number h2


def permanent_deformation(lat):
    """Permanent tide deformation (m) to add to conventional tide-free heights."""
    s2 = np.sin(np.radians(np.asarray(lat, float))) ** 2
    return H2 * H_PERM * np.sqrt(5 / (4 * np.pi)) * (1.5 * s2 - 0.5)


def decimal_year(t):
    """pandas Timestamps (UTC) -> decimal year."""
    t = np.asarray(t, dtype="datetime64[s]")
    y = t.astype("datetime64[Y]")
    start = y.astype("datetime64[s]")
    end = (y + 1).astype("datetime64[s]")
    return y.astype(int) + 1970 + (t - start) / (end - start)


def navd88_to_swot_ellipsoid(lon, lat, h_navd88_m, epoch):
    """Ellipsoidal height (m) in the SWOT convention (ITRF2014, permanent tide
    deformation included) for NAVD88 heights at given epochs."""
    lon, lat, h, ep = np.broadcast_arrays(*(np.asarray(v, float) for v in
                                            (lon, lat, h_navd88_m, epoch)))
    _, _, hi, _ = _T.transform(lon, lat, h, ep)
    hi = np.asarray(hi, float)
    hi[~np.isfinite(hi) | (np.abs(hi) > 1e5)] = np.nan
    return hi + permanent_deformation(lat)


def offset_navd88_to_swot(lon, lat, epoch=2024.5):
    """h_SWOT - H_NAVD88 at a point (m); used for checks and reporting."""
    return float(navd88_to_swot_ellipsoid(lon, lat, 0.0, epoch))
