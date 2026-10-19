# swot-reach-estimators

Code for the manuscript

> Kaya, Y. *Diagnosing the Operational Reach Estimator of SWOT River Products With Simulations, Datum-Audited Gauges, and ICESat-2 Slopes.* Submitted to *IEEE Transactions on Geoscience and Remote Sensing*.

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
| USGS instantaneous water level (63160, 00065) | USGS NWIS instantaneous-values service | 05 |
| Boundaries for Fig. 2 | U.S. Census Bureau cartographic boundary file 2024 (states 1:20M); Natural Earth 1:50m countries v5.1.1 | 13 |

Large inputs and intermediate files go to `~/swot_p39_work` (override with `P39_WORK`):
`sword/netcdf/*_sword_v17b.nc`, `iris/IRIS_3.4.nc`, `map/`, and the folders written by the scripts
(`data/`, `swot/reach`, `swot/node`, `gauges/usgs`, `results/`).

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
