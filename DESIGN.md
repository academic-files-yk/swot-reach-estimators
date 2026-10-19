# Design notes: node-to-reach estimators for SWOT RiverSP

Working title: *Which reach elevation and slope to trust? Benchmarking SWOT RiverSP
reach estimators against datum-audited gauges and ICESat-2 river slopes*

## Research questions
- RQ1. How accurate are the official reach WSE and the two official slopes (`slope`,
  `slope2`) against independent truth?
- RQ2. On the same node set, do classical (mean, weighted mean, OLS, WLS) or robust
  (median, Theil-Sen, Huber) estimators do better or worse than the official products?
- RQ3. Under which observation conditions (detrended node scatter, node count, dark
  water fraction, cross-track distance, layover, ice, width, slope magnitude) does the
  ranking change?
- RQ4. How large is the discharge error implied by the slope differences (Manning,
  Q proportional to S^(1/2))?

## Decisions
- D1. Product: SWOT_L2_HR_RiverSP_D (reprocessed Version D, CRID PGD0/PID0, SWORD v17b)
  through Hydrocron. Version C is not mixed in (reach IDs and geometry differ).
- D2. WSE truth: USGS continuous stage at gauges whose datum is NAVD88 and whose datum
  method is surveyed (not read from a map). Parameter 63160 (stream water level elevation
  above NAVD88) is used where it exists, otherwise 00065 stage + gauge datum.
  Surveyed datum methods kept: L (level), D (differential GPS), F (survey-grade GPS),
  W/X/Y/Z (GNSS survey levels 1-4), with stated altitude accuracy <= 0.5 ft. Excluded:
  map, DEM, lidar, altimeter, reported, unknown and plain GPS.
- D3. Datum chain: NAVD88 height -> NAD83(2011) ellipsoidal height (GEOID18) ->
  ITRF2014 ellipsoidal height at the observation epoch (PROJ time-dependent Helmert;
  SWOT products are realised in ITRF2014) -> add the permanent tide deformation
  (SWOT User Handbook Eq. 11.7; +2.8 to -4.3 cm across CONUS), because NAD83 heights
  are conventional tide-free and SWOT heights are not -> height above EGM2008 using the
  `geoid_hght` reported by SWOT at the gauge node.
- D4. WSE evaluation point: the gauge location. Every estimator returns a level at the
  reach centre and a slope; the level is moved to the gauge along SWORD flow distance.
  Level-only estimators (mean, weighted mean, median) use the official slope for this
  move, so that only the level estimate differs.
- D5. Slope truth: IRIS v3.4 (ICESat-2 river surface slope, SWORD v17b) reach median
  slope. SWOT slope per reach is summarised as the median over passes for each
  estimator. Periods differ (IRIS 2018-2026, SWOT 2023-2026); this is stated, and a
  sensitivity uses only IRIS reaches with low slope variability.
- D6. Node set: nodes with valid wse and without the official outlier flag
  (node_q_b bit 23). A second set keeps flagged nodes, to test the outlier rejection.
  Every estimator in a comparison uses the same nodes.
- D7. Matching rules (fixed before looking at results): gauge within max(300 m, node
  width) of a SWORD node centre; drainage area ratio USGS/SWORD facc between 0.8 and 1.25
  where both exist; reach type river (type 1); time difference to the gauge record at
  most 30 min, gauge value linearly interpolated.
- D8. Metrics: signed error, median absolute error, 68th percentile of absolute error
  (SWOT requirement convention), NMAD; slope errors in mm/km and as log ratio.
  Confidence intervals by reach-level (cluster) bootstrap, paired Wilcoxon tests between
  estimators.
- D9. No reach or gauge is dropped because of its result. Exclusions are counted and
  reported with reasons.

- D10. The official reach values come from a Bayesian reconstruction that can use the
  first valid upstream and downstream neighbour reach (RiverObs configuration). Their
  node set therefore differs from the single-reach estimators; this is part of what is
  evaluated, not a defect of the comparison. Reproduction tests in 10_bayes_reproduction.

- D11. Gauge screening for gross mismatch (wrong channel, wrong datum record): a gauge is
  set aside if the median over its overpasses of the across-estimator median error exceeds
  2 m in absolute value. The rule acts on an offset common to all estimators, so it
  cannot change their ranking; counts are reported and a sensitivity keeps all gauges.
- D12. Bootstrap: 2000 resamples of reaches (WSE: gauges), seed 39. Paired Wilcoxon
  signed-rank tests on absolute errors over overpasses where both estimators exist.

## Changelog
- 2026-10-12: initial version.
- 2026-10-12: D3 moved to ITRF2014 + permanent tide (official documents); D10 added.
