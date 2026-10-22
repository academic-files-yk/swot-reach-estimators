"""LaTeX tables generated from output/*.csv (three-rule booktabs layout)."""
import numpy as np
import pandas as pd
from p39.sets import WSE_REQUIRED
from p39.paths import OUTPUT, CODE, TEX

LAB = {"official": "PUB", "official_s2": "PUB-S2", "bayes_s2_reach": "BAYES-$\\sigma^2$",
       "bayes_s2": "BAYES-$\\sigma^2$ node",
       "official_sm": "SM-node", "nearest_node": "Raw node", "mean": "Mean",
       "wmean": "W-mean", "median": "Median", "ols": "OLS", "wls": "WLS",
       "theilsen": "Theil--Sen", "huber": "Huber"}
CONT = ["af", "as", "eu", "na", "oc", "sa"]
CNAME = {"af": "AF", "as": "AS", "eu": "EU", "na": "NA", "oc": "OC", "sa": "SA"}


def minus(x, fmt):
    s = fmt.format(x)
    return s.replace("-", "$-$") if not s.startswith("$") else s


def slope_table():
    lab = {**LAB, "bayes_s2": "BAYES-$\\sigma^2$"}      # here bayes_s2 is the reach slope
    s = pd.read_csv(OUTPUT / "slope_summary.csv").set_index("estimator")
    c = pd.read_csv(OUTPUT / "slope_by_continent.csv").set_index("continent")
    order = [e for e in ["official", "official_s2", "bayes_s2", "ols", "wls", "theilsen", "huber"]
             if e in s.index]
    best = s.loc[order, "p68"].idxmin()
    rows = []
    for e in order:
        r = s.loc[e]
        p68 = f"{r.p68:.0f} ({r.p68_lo:.0f}--{r.p68_hi:.0f})"
        if e == best:
            p68 = "\\textbf{" + p68 + "}"
        cells = [lab[e], p68, f"{r.mae:.1f}", f"{100 * r.within_17:.0f}",
                 f"{100 * r.share_nonpositive:.1f}", f"{100 * r.q_err_p68:.0f}"]
        cells += [f"{c.loc[k, e]:.0f}" for k in CONT]
        rows.append(" & ".join(cells) + " \\\\")
    n = int(s.loc["official", "n"])
    tex = r"""\begin{table*}[!t]
\caption{Reach-median slope against IRIS ($n=%d$ reaches, clean node set).}\label{tab:slope}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lccccc*{6}{c}@{}}
\toprule
 & $|e|_{68}$ (95\%% CI) & Median $|e|$ & $\le$17 mm/km & $\hat S\le0$ & $|\Delta Q|_{68}$ & \multicolumn{6}{c}{$|e|_{68}$ by continent (mm/km)} \\
\cmidrule(l){7-12}
Estimator & (mm/km) & (mm/km) & (\%%) & (\%%) & (\%%) & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\\[2pt]\parbox{0.95\textwidth}{\footnotesize $|e|_{68}$: 68th percentile of the absolute difference between the median SWOT slope over at least five overpasses and the IRIS median slope; CI from 2000 bootstrap resamples of reaches. $\hat S\le0$: reaches whose median SWOT slope is zero or negative. $|\Delta Q|_{68}$: 68th percentile of $|(\hat S/S_{\mathrm{IRIS}})^{1/2}-1|$ over reaches with positive median slope. Best value in bold.}
\end{table*}
""" % (n, " & ".join(CNAME[k] for k in CONT), "\n".join(rows))
    (TEX / "tab_slope.tex").write_text(tex)


def sample_table():
    g = pd.read_csv(OUTPUT / "gauge_exclusions.csv")
    i = pd.read_csv(OUTPUT / "iris_sample_counts.csv")
    rows = ["\\multicolumn{2}{@{}l}{\\emph{USGS gauges}} \\\\"]
    rows += [f"{r.rule} & {r.n_remaining:,} \\\\" for r in g.itertuples()]
    f = OUTPUT / "wse_screen.csv"
    if f.exists():
        sc = pd.read_csv(f)
        rows.append(f"with $\\geq$1 matched overpass & {len(sc):,} \\\\")
        rows.append(f"after gross-mismatch screening & {(~sc.screened_out).sum():,} \\\\")
        from p39.paths import RESULTS
        w = pd.read_parquet(RESULTS / "wse_long.parquet")
        est = WSE_REQUIRED
        w = w[(w["node_set"] == "clean") & ~w["screened"]].dropna(subset=[e + "_err" for e in est])
        rows.append(f"with all estimators on $\\geq$1 overpass & {w['site'].nunique():,} \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{2}{@{}l}{\\emph{IRIS reaches}} \\\\")
    rows += [f"{r.rule} & {r.n:,} \\\\" for r in i.itertuples() if not r.rule.startswith("sampled")]
    rows.append(f"random sample, 250 per continent & {int(i[i.rule.str.startswith('sampled')]['n'].sum()):,} \\\\")
    f = OUTPUT / "slope_summary.csv"
    if f.exists():
        rows.append(f"with $\\geq$5 overpasses for all estimators & {int(pd.read_csv(f)['n'].iloc[0]):,} \\\\")
    body = "\n".join(rows).replace(">=", "$\\geq$").replace("<=", "$\\leq$")
    body = (body.replace("2023-09..2026-06", "Sep 2023--Jun 2026").replace("0.8-1.25", "0.8--1.25")
            .replace("slope > 0", "slope $>0$").replace("max(300 m, node width)", "$\\max$(300 m, width)")
            .replace("stream sites", "Stream gauges").replace("(1)", ""))
    tex = r"""\begin{table}[!t]
\caption{Sample selection.}\label{tab:samples}
\centering\footnotesize
\begin{tabular}{@{}lr@{}}
\toprule
Rule (applied in order) & Remaining \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % body
    (TEX / "tab_samples.tex").write_text(tex)


def wse_table():
    f1, f2 = OUTPUT / "wse_summary_clean_abs.csv", OUTPUT / "wse_summary_clean_rel.csv"
    if not (f1.exists() and f2.exists()):
        return
    a = pd.read_csv(f1).set_index("estimator")
    r = pd.read_csv(f2).set_index("estimator")
    order = [e for e in LAB if e in a.index and a.loc[e, "n"] > 0]
    rows = []
    best_a = a.loc[order, "p68"].idxmin()
    best_r = r.loc[order, "p68"].idxmin()
    for e in order:
        pa = f"{100 * a.loc[e, 'p68']:.1f} ({100 * a.loc[e, 'p68_lo']:.1f}--{100 * a.loc[e, 'p68_hi']:.1f})"
        pr = f"{100 * r.loc[e, 'p68']:.1f} ({100 * r.loc[e, 'p68_lo']:.1f}--{100 * r.loc[e, 'p68_hi']:.1f})"
        if e == best_a:
            pa = "\\textbf{" + pa + "}"
        if e == best_r:
            pr = "\\textbf{" + pr + "}"
        rows.append(f"{LAB[e]} & {minus(100 * a.loc[e, 'med'], '{:.1f}')} & {pa} & {pr} \\\\")
    n = int(a.loc["official", "n"])
    tex = r"""\begin{table}[!t]
\caption{WSE at the gauge location (clean node set, $n=%s$ overpasses).}\label{tab:wse}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccc@{}}
\toprule
 & Median $e$ & \multicolumn{2}{c}{$|e|_{68}$ (95\%% CI) (cm)} \\
\cmidrule(l){3-4}
Estimator & (cm) & absolute & relative \\
\midrule
%s
\bottomrule
\end{tabular}
\\[2pt]\parbox{\columnwidth}{\footnotesize Relative: after removing each gauge's median error. CI from 2000 bootstrap resamples of gauges. Raw node: only overpasses in which the gauge node itself was observed ($n=%s$). Best values in bold.}
\end{table}
""" % (f"{n:,}", "\n".join(rows), f"{int(a.loc['nearest_node', 'n']):,}")
    (TEX / "tab_wse.tex").write_text(tex)


def sensitivity_table():
    f = OUTPUT / "sensitivity.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    lab = {"base": "Main analysis", "all_nodes": "Outlier-flagged nodes kept",
           "iris_stable": "Stable IRIS reaches only", "reach_q_good": "\\texttt{reach\\_q} $\\leq$ 1",
           "wide_200": "Width $\\geq$ 200 m", "no_screening": "No gauge screening",
           "levelled": "Levelled datums only"}
    rows = ["\\multicolumn{5}{@{}l}{\\emph{Slope, $|e|_{68}$ (mm/km)}} & n \\\\"]
    for r in s[s.quantity.str.startswith("slope")].itertuples():
        rows.append(f"{lab[r.case]} & {r.official:.0f} & {r.bayes_s2:.0f} & {r.ols:.0f} & {r.theilsen:.0f} & {r.n:,} \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{@{}l}{\\emph{WSE, relative $|e|_{68}$ (cm)}} & \\\\")
    for r in s[s.quantity.str.startswith("WSE")].itertuples():
        rows.append(f"{lab[r.case]} & {r.official:.1f} & {r.bayes_s2:.1f} & {r.wls:.1f} & {r.huber:.1f} & {r.n:,} \\\\")
    tex = r"""\begin{table}[!t]
\caption{Sensitivity of the estimator ranking.}\label{tab:sens}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccr@{}}
\toprule
 & PUB & BAYES-$\sigma^2$ & OLS/WLS & TS/Huber & \\
\midrule
%s
\bottomrule
\end{tabular}
\\[2pt]\parbox{\columnwidth}{\footnotesize Slope: OLS and Theil--Sen (TS); WSE: WLS and Huber; BAYES-$\sigma^2$ node height for WSE. n: reaches (slope) or gauges (WSE). Stable IRIS reaches: at least ten ICESat-2 slopes and spread at most half of the median.}
\end{table}
""" % "\n".join(rows)
    (TEX / "tab_sens.tex").write_text(tex)


def simulation_table():
    f = OUTPUT / "simulation_summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    est = ["pub", "bayes_s2", "slope2", "ols", "theilsen", "huber"]
    head = ["PUB", "BAYES-$\\sigma^2$", "PUB-S2", "OLS", "TS", "Huber"]
    rows = []
    for k in sorted(s.k.unique()):
        cells = []
        for e in est:
            a = s[(s.model == "pattern") & (s.k == k) & (s.estimator == e)].rel_bias_median.iloc[0]
            b = s[(s.model == "iid") & (s.k == k) & (s.estimator == e)].rel_bias_median.iloc[0]
            fmt = lambda v: "0" if round(100 * v) == 0 else f"{100 * v:+.0f}".replace("-", "$-$")
            cells.append(f"{fmt(a)}/{fmt(b)}")
        rows.append(f"{k:g} & " + " & ".join(cells) + " \\\\")
    n = int(s.n.min())
    tex = r"""\begin{table}[!t]
\caption{Simulated median relative slope bias (\%%).}\label{tab:sim}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
$k$ & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\\[2pt]\parbox{\columnwidth}{\footnotesize $k$: neighbour slope divided by the reach slope. Values: real overpass error patterns / independent node errors; at least %s windows per cell; 95\%% bootstrap intervals are within $\pm$2 percentage points.}
\end{table}
""" % (" & ".join(head), "\n".join(rows), f"{n:,}")
    (TEX / "tab_sim.tex").write_text(tex)


if __name__ == "__main__":
    slope_table(); sample_table(); wse_table(); sensitivity_table(); simulation_table()
    print("tables written")
