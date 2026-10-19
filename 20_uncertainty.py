"""Calibration of the stated slope uncertainties (slope_u, slope2_u) and, if the WSE
evaluation exists, of wse_u.

Slope: z = (single-overpass slope - reach median of that estimator) / stated u. The
deviation from the reach median isolates the overpass-to-overpass (random) part, which
the stated random uncertainty should describe; real temporal slope changes make this a
conservative (upper) check. WSE: z = error / wse_u, absolute and after removing each
gauge's median error.
Output: output/uncertainty_calibration.csv
"""
import numpy as np
import pandas as pd
from p39.paths import RESULTS, OUTPUT


def calib(z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    return dict(n=len(z), share_within_1=np.mean(np.abs(z) <= 1),
                nmad_z=1.4826 * np.median(np.abs(z - np.median(z))))


def main():
    rows = []
    e = pd.read_parquet(RESULTS / "iris_estimates.parquet",
                        columns=["reach_id", "node_set", "n_nodes", "off_slope", "off_slope_u",
                                 "off_slope2", "off_slope2_u"])
    e = e[(e["node_set"] == "clean") & (e["n_nodes"] >= 5)]
    for s, u, lab in (("off_slope", "off_slope_u", "slope"), ("off_slope2", "off_slope2_u", "slope2")):
        d = e.dropna(subset=[s, u])
        d = d[d[u] > 0]
        dev = d[s] - d.groupby("reach_id")[s].transform("median")
        rows.append(dict(quantity=lab, reference="reach median", **calib(dev / d[u]),
                         median_u_mm_km=1e6 * d[u].median()))
    f = RESULTS / "wse_long.parquet"
    if f.exists():
        w = pd.read_parquet(f)
        w = w[(w["node_set"] == "clean") & ~w["screened"]].dropna(subset=["official_err", "off_wse_u"])
        w = w[w["off_wse_u"] > 0]
        rows.append(dict(quantity="wse", reference="gauge, absolute",
                         **calib(w["official_err"] / w["off_wse_u"]),
                         median_u_mm_km=np.nan, median_u_cm=100 * w["off_wse_u"].median()))
        rel = w["official_err"] - w.groupby("site")["official_err"].transform("median")
        rows.append(dict(quantity="wse", reference="gauge, relative",
                         **calib(rel / w["off_wse_u"]), median_u_cm=100 * w["off_wse_u"].median()))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "uncertainty_calibration.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
