# swot-reach-estimators

Code for the manuscript

> Kaya, Y. *Diagnosing the operational reach estimator of SWOT river products with simulations, datum-audited gauges and ICESat-2 slopes.* Submitted to *Remote Sensing of Environment*.

The scripts download all public inputs, rebuild the operational SWOT RiverSP Version D reach values from the
node product, re-implement the Bayesian node-height reconstruction, apply all reach estimators to identical
node sets, and evaluate them against USGS gauges (NAVD88 datums converted to the SWOT reference) and ICESat-2
river slopes (IRIS v3.4). Every number, table and figure of the paper is written by a script; the LaTeX number
macros are produced by `18_numbers.py` and the tables by `19_tables.py`.

## Data (all public, no login needed except where noted)

| Data | Source | Script |
|---|---|---|
| SWOT L2 RiverSP Version D, reach and node time series | NASA PO.DAAC Hydrocron API, doi:10.5067/SWOT-RIVERSP-D | 04 |
| SWORD v17b | Zenodo, doi:10.5281/zenodo.15299138 | 01, 04b |
| IRIS v3.4 (ICESat-2 river surface slope) | Zenodo, doi:10.5281/zenodo.23233774 | 08 |
| USGS stream gauges: metadata | USGS Water Data API (OGC API) | 02 |
| USGS instantaneous water level (63160, 00065) | Retrieved from the legacy NWIS instantaneous-values service on 12–13 Oct 2026 and archived in `data/usgs_gauges_used.zip` (public domain, USGS); `05` downloads the same series from the USGS Water Data API | 05 |
| Boundaries for Fig. 2 | U.S. Census Bureau cartographic boundary file 2024 (states 1:20M); Natural Earth 1:50m countries v5.1.1 | 13 |

Large inputs and intermediate files go to `~/swot_p39_work` (override with `P39_WORK`):
`sword/netcdf/*_sword_v17b.nc`, `iris/IRIS_3.4.nc`, `map/`, and the folders written by the scripts
(`data/`, `swot/reach`, `swot/node`, `gauges/usgs`, `results/`).

### Note on the USGS gauge archive

The legacy NWIS services are decommissioned on 22 February 2027, so `05_usgs_series.py` uses the
USGS Water Data API (`continuous` collection). The numbers in the paper come from the archived values in
`data/usgs_gauges_used.zip`, which `run_all.sh` unpacks by default. A check on 10 randomly drawn gauges
(22 Oct 2026) found 7,117 of 7,219 archived values (98.6%) returned unchanged by the new API; the
differences come from later revision of provisional values by USGS (at most 0.16 ft at one gauge) and
from provisional values no longer served. Set `P39_REFRESH_GAUGES=1` to download the gauges again.

## Environment

Python 3.14, PROJ 9.8 (through pyproj). `pip install -r requirements.txt`. The geodetic chain needs the
PROJ grid `us_noaa_g2018u0.tif` (GEOID18); network access for PROJ is enabled in `p39/geodesy.py`.

## Order of execution (`run_all.sh`)

| Step | Script | Output |
|---|---|---|
| SWORD tables | `01_sword_index.py` | `data/sword_nodes.parquet`, `data/sword_reaches.parquet` |
| USGS inventory | `02_usgs_sites.py` | `data/usgs_sites.csv` |
| Gauge-node matching | `03_match_gauges.py` | `data/gauge_match.csv`, `output/gauge_exclusions.csv` |
| IRIS sample | `08_iris_sample.py` | `data/iris_sample.csv`, `output/iris_sample_counts.csv` |
| Neighbour reaches | `04b_neighbour_list.py` | `data/neighbours.csv` |
| SWOT download | `04_hydrocron_download.py <list> [workers]` | `swot/reach`, `swot/node` |
| Gauge download | `05_usgs_series.py` | `gauges/usgs` |
| Estimators | `06_reach_estimates.py <list> <name>` | `results/<name>.parquet` |
| WSE at gauges | `07_wse_eval.py` | `results/wse_eval.parquet` |
| Slope against IRIS | `09_slope_eval.py` | `results/slope_*.parquet` |
| Reconstruction test | `10_bayes_reproduction.py` | `output/bayes_reproduction.csv` |
| Reconstruction variants | `11_bayes_variants.py <list> <name>` | `results/<name>.parquet` |
| WSE statistics | `12_wse_stats.py` | `output/wse_*.csv` |
| Slope statistics | `15_slope_stats.py` | `output/slope_*.csv` |
| End-node state | `16_end_nodes.py` | `results/iris_endnodes.parquet` |
| Uncertainty calibration | `20_uncertainty.py` | `output/uncertainty_calibration.csv` |
| Conditions | `21_drivers.py` | `output/drivers_*.csv`, Fig. 5 |
| Sensitivity | `23_sensitivity.py` | `output/sensitivity.csv` |
| Example overpass | `24_fig_example.py` | Fig. 1, `output/example_overpass.csv` |
| Monte Carlo simulation | `25_simulation.py` | `output/simulation_*.csv` |
| Uncertainty of alternative estimators | `26_estimator_uncertainty.py` | `output/estimator_uncertainty.csv` |
| Figures | `13_fig1_map.py`, `17_fig_slope.py`, `22_fig_wse.py` | Figs. 2, 3, 4 |
| Tables and numbers | `19_tables.py`, `18_numbers.py` | LaTeX tables and `numbers.tex` |

`DESIGN.md` lists the decisions (matching rules, node sets, datum chain, screening, statistics) that were fixed
before the comparisons were run.

## Package `p39`

- `paths.py`: folders and service addresses.
- `estimators.py`: node mean, weighted mean, median, OLS, WLS, Theil-Sen, Huber.
- `bayes.py`: re-implementation of the RiverSP Bayesian reconstruction with a choice of noise covariance.
- `slope2.py`: re-implementation of the RiverSP enhanced slope.
- `sets.py`: shared definition of the WSE evaluation sample.
- `geodesy.py`: NAVD88 -> GEOID18 -> NAD83(2011) -> ITRF2014 + permanent tide.
- `stats.py`: error summaries, reach/gauge cluster bootstrap, paired Wilcoxon tests.
- `figstyle.py`: IEEE figure sizes, overlap check, PDF + 600 dpi PNG/TIFF output.

## License

MIT (see `LICENSE`).
