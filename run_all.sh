#!/bin/sh
set -e
W=${P39_WORK:-$HOME/swot_p39_work}
python 01_sword_index.py
python 02_usgs_sites.py
python 03_match_gauges.py
python 08_iris_sample.py
python 04b_neighbour_list.py
python 04_hydrocron_download.py $W/data/gauge_match.csv 12
python 04_hydrocron_download.py $W/data/iris_sample.csv 12
python 04_hydrocron_download.py $W/data/neighbour_download.csv 12
python 05_usgs_series.py 6
python 06_reach_estimates.py $W/data/gauge_match.csv gauge_estimates
python 06_reach_estimates.py $W/data/iris_sample.csv iris_estimates
python 07_wse_eval.py
python 10_bayes_reproduction.py
python 11_bayes_variants.py $W/data/gauge_match.csv gauge_bayes
python 11_bayes_variants.py $W/data/iris_sample.csv iris_bayes
python 16_end_nodes.py $W/data/iris_sample.csv iris_endnodes
python 09_slope_eval.py
python 12_wse_stats.py
python 15_slope_stats.py
python 20_uncertainty.py
python 21_drivers.py
python 13_fig1_map.py
python 17_fig_slope.py
python 22_fig_wse.py
python 23_sensitivity.py
python 24_fig_example.py
python 25_simulation.py
python 26_estimator_uncertainty.py
python 19_tables.py
python 18_numbers.py
