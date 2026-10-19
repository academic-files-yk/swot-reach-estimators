"""Write every number quoted in the manuscript as a LaTeX macro (numbers.tex).

The text uses only these macros, so each number traces back to a table in output/.
Missing inputs leave the macro as '??' so that they are visible in the PDF.
"""
import numpy as np
import pandas as pd
from p39.sets import WSE_REQUIRED
from p39.paths import OUTPUT, RESULTS, DATA, CODE, TEX

M = {}


def put(name, value, fmt="{:.0f}"):
    try:
        M[name] = fmt.format(value) if not isinstance(value, str) else value
    except (TypeError, ValueError):
        M[name] = "??"


def pct(x):
    return 100 * x


def load(name):
    f = OUTPUT / name
    return pd.read_csv(f) if f.exists() else None


def samples():
    g = load("gauge_exclusions.csv")
    if g is not None:
        put("NusgsAll", g["n_remaining"].iloc[0], "{:,.0f}")
        put("NgaugeMatched", g["n_remaining"].iloc[-1], "{:,.0f}")
    i = load("iris_sample_counts.csv")
    if i is not None:
        put("Nper", 250)
        put("NirisElig", i[i["rule"] == "IRIS median slope > 0"]["n"].iloc[0], "{:,.0f}")


def slope():
    s = load("slope_summary.csv")
    if s is None:
        return
    s = s.set_index("estimator")
    put("NirisEval", s.loc["official", "n"], "{:,.0f}")
    for e, tag in [("official", "Pub"), ("official_s2", "Ptwo"), ("ols", "Ols"),
                   ("wls", "Wls"), ("theilsen", "Ts"), ("huber", "Hub"), ("bayes_s2", "Bay")]:
        if e not in s.index:
            continue
        put(f"Sl{tag}P", s.loc[e, "p68"]); put(f"Sl{tag}Plo", s.loc[e, "p68_lo"])
        put(f"Sl{tag}Phi", s.loc[e, "p68_hi"]); put(f"Sl{tag}Mae", s.loc[e, "mae"], "{:.1f}")
        put(f"Sl{tag}Win", pct(s.loc[e, "within_17"])); put(f"Sl{tag}Neg", pct(s.loc[e, "share_nonpositive"]), "{:.1f}")
        put(f"Sl{tag}Qmed", pct(s.loc[e, "q_err_med"]), "{:.1f}"); put(f"Sl{tag}Qp", pct(s.loc[e, "q_err_p68"]), "{:.0f}")
    p = load("slope_paired.csv")
    if p is not None:
        r = p[(p.a == "theilsen") & (p.b == "official")].iloc[0]
        put("SlTsBetter", pct(r["share_a_better"]))
        r = p[(p.a == "official_s2") & (p.b == "official")].iloc[0]
        put("SlTwoBetter", pct(r["share_a_better"])); put("SlTwoP", r["p"], "{:.3f}")
    n = load("slope_neighbour_pull.csv")
    if n is not None:
        n = n.set_index("estimator")
        put("PullPub", n.loc["official", "spearman"], "{:.2f}")
        put("PullTwo", n.loc["official_s2", "spearman"], "{:.2f}")
        put("PullTs", n.loc["theilsen", "spearman"], "{:.2f}")
    n2 = load("slope_neighbour_pull_iris.csv")
    if n2 is not None:
        n2 = n2.set_index("estimator")
        put("PullIrisN", n2.loc["official", "n"], "{:,.0f}")
        put("PullIrisPub", n2.loc["official", "spearman"], "{:.2f}")
        put("PullIrisTwo", n2.loc["official_s2", "spearman"], "{:.2f}")
        put("PullIrisOls", n2.loc["ols", "spearman"], "{:.2f}")
        put("PullIrisPubP", n2.loc["official", "p"], "{:.0e}")
    en = load("slope_by_endnodes.csv")
    if en is not None:
        en = en.set_index("ends")
        for o, tag in [("both", "Both"), ("one", "One"), ("none", "None")]:
            for e, t2 in [("official", "Pub"), ("theilsen", "Ts"), ("official_s2", "Ptwo")]:
                put(f"End{tag}{t2}", en.loc[o, e])
    f = RESULTS / "iris_endnodes.parquet"
    if f.exists():
        e = pd.read_parquet(f)
        put("EndBothShare", pct((e.dn_end_obs & e.up_end_obs).mean()))
    c = load("slope_by_continent.csv")
    if c is not None:
        c = c.set_index("continent")
        put("ContWorseAll", "six" if (c["official"] > c["theilsen"]).all() else "??")
    sc = OUTPUT / "slope_pass_scatter.csv"
    if sc.exists():
        sc = pd.read_csv(sc).set_index("estimator")["pass_nmad_mm_km"]
        for e, tag in [("official", "Pub"), ("theilsen", "Ts"), ("ols", "Ols"), ("official_s2", "Ptwo")]:
            if e in sc.index:
                put(f"Scat{tag}", sc[e])


def wse():
    for ns in ("clean",):
        for mode in ("abs", "rel"):
            t = load(f"wse_summary_{ns}_{mode}.csv")
            if t is None:
                continue
            t = t.set_index("estimator")
            for e, tag in [("official", "Pub"), ("official_s2", "Ptwo"), ("bayes_s2_reach", "Bay"),
                           ("bayes_s2", "Bayn"), ("official_sm", "Sm"), ("nearest_node", "Raw"), ("mean", "Mean"),
                           ("wmean", "Wmean"), ("median", "Med"), ("ols", "Ols"),
                           ("wls", "Wls"), ("theilsen", "Ts"), ("huber", "Hub")]:
                if e not in t.index:
                    continue
                k = "A" if mode == "abs" else "R"
                put(f"W{k}{tag}P", 100 * t.loc[e, "p68"], "{:.1f}")
                put(f"W{k}{tag}Plo", 100 * t.loc[e, "p68_lo"], "{:.1f}")
                put(f"W{k}{tag}Phi", 100 * t.loc[e, "p68_hi"], "{:.1f}")
                put(f"W{k}{tag}Med", 100 * t.loc[e, "med"], "{:.1f}")
                put(f"W{k}{tag}N", t.loc[e, "n"], "{:,.0f}")
    s = load("wse_screen.csv")
    if s is not None:
        put("NgaugeScreened", s["screened_out"].sum())
    f = RESULTS / "wse_long.parquet"
    if f.exists():
        w = pd.read_parquet(f)
        est = WSE_REQUIRED
        w = w[(w["node_set"] == "clean") & ~w["screened"]].dropna(subset=[e + "_err" for e in est])
        put("NgaugeEval", w["site"].nunique(), "{:,.0f}")


def repro():
    f = RESULTS / "gauge_estimates.parquet"
    if f.exists():
        d = pd.read_parquet(f, columns=["node_set", "off_wse", "wse_sm_mean"])
        d = d[(d["node_set"] == "clean") & d["off_wse"].notna()]
        r = (d["off_wse"] - d["wse_sm_mean"]).abs()
        put("NreproPasses", len(r), "{:,.0f}")
        put("ReproShare", pct((r < 1e-3).mean()), "{:.1f}")
    b = load("bayes_reproduction.csv")
    if b is not None:
        put("NbayesPasses", len(b)); put("NbayesReaches", b["reach_id"].nunique())
        put("BayesPoneMed", b["p1_med_abs_mm"].median(), "{:.2f}")
        put("BayesPoneShare", pct((b["p1_med_abs_mm"] < 1).mean()))
        put("BayesPtwoMed", b["p2_med_abs_mm"].median())
        put("BayesPtwoShare", pct((b["p2_med_abs_mm"] < 1).mean()))


def audit():
    a = load("wse_datum_audit.csv")
    if a is not None:
        a = a.set_index("group")
        put("AudMed", a.loc["all", "median_cm"], "{:.1f}")
        put("AudLo", a.loc["all", "q25_cm"], "{:.1f}"); put("AudHi", a.loc["all", "q75_cm"], "{:.1f}")
        put("AudLev", a.loc["method=Levelling", "median_cm"], "{:.1f}")
        put("AudGnss", a.loc["method=GPS/GNSS", "median_cm"], "{:.1f}")
        put("AudPcA", a.loc["pc=63160", "median_cm"], "{:.1f}")
        put("AudPcB", a.loc["pc=00065", "median_cm"], "{:.1f}")
        lat = a[a.index.str.startswith("lat=")]
        put("AudLatS", lat["median_cm"].iloc[0], "{:.1f}"); put("AudLatN", lat["median_cm"].iloc[-1], "{:.1f}")
        put("NlevGauges", a.loc["method=Levelling", "n"]); put("NgnssGauges", a.loc["method=GPS/GNSS", "n"])
    u = load("uncertainty_calibration.csv")
    if u is not None:
        u = u.set_index(["quantity", "reference"])
        put("UslNmad", u.loc[("slope", "reach median"), "nmad_z"], "{:.2f}")
        put("UslIn", pct(u.loc[("slope", "reach median"), "share_within_1"]))
        put("UslU", u.loc[("slope", "reach median"), "median_u_mm_km"])
        put("UsltNmad", u.loc[("slope2", "reach median"), "nmad_z"], "{:.1f}")
        put("UsltIn", pct(u.loc[("slope2", "reach median"), "share_within_1"]))
        put("UsltU", u.loc[("slope2", "reach median"), "median_u_mm_km"], "{:.1f}")
        if ("wse", "gauge, relative") in u.index:
            put("UwRelNmad", u.loc[("wse", "gauge, relative"), "nmad_z"], "{:.1f}")
            put("UwU", u.loc[("wse", "gauge, relative"), "median_u_cm"], "{:.1f}")
    p = load("wse_paired.csv")
    if p is not None:
        r = p[(p.a == "official") & (p.b == "huber")].iloc[0]
        put("WHubBetter", pct(1 - r["share_a_better"]))


def bayes_extra():
    p = load("slope_paired.csv")
    if p is not None and (p.a == "bayes_s2").any():
        r = p[(p.a == "bayes_s2") & (p.b == "official")].iloc[0]
        put("SlBayBetter", pct(r["share_a_better"])); put("SlBayPval", r["p"], "{:.3f}")
        r = p[(p.a == "theilsen") & (p.b == "bayes_s2")].iloc[0]
        put("SlTsBayBetter", pct(r["share_a_better"]))
    n2 = load("slope_neighbour_pull_iris.csv")
    if n2 is not None and (n2.estimator == "bayes_s2").any():
        put("PullIrisBay", n2.set_index("estimator").loc["bayes_s2", "spearman"], "{:.2f}")
    f = RESULTS / "slope_pass.parquet"
    if f.exists():
        r = pd.read_parquet(f, columns=["reach_id", "cycle_id", "pass_id", "node_set", "estimator", "slope_mm_km"])
        r = r[r.node_set == "clean"]
        w = r.pivot_table(index=["reach_id", "cycle_id", "pass_id"], columns="estimator", values="slope_mm_km")
        if "bayes_s1" in w:
            d = (w["bayes_s1"] - w["official"]).abs().dropna()
            put("CtrlSlopeShare", pct((d < 1).mean()))
    wp = load("wse_paired.csv")
    if wp is not None and (wp.a == "bayes_s2").any():
        r = wp[(wp.a == "bayes_s2") & (wp.b == "official_sm")].iloc[0]
        put("WBaynBetter", pct(r["share_a_better"]))


def example():
    e = load("example_overpass.csv")
    if e is not None:
        r = e.iloc[0]
        put("ExReach", int(r.reach_id), "{:d}"); put("ExCycle", int(r.cycle_id), "{:d}")
        put("ExPass", int(r.pass_id), "{:d}"); put("ExClass", int(r.class_size), "{:,}")
        put("ExDiff", r.median_diff_mm_km, "{:.1f}")


def sensitivity():
    s = load("sensitivity.csv")
    if s is None:
        return
    s = s.set_index(["quantity", "case"])
    sl = s.loc["slope (mm/km)"]
    put("SenStableN", sl.loc["iris_stable", "n"]); put("SenStablePub", sl.loc["iris_stable", "official"])
    put("SenStableTs", sl.loc["iris_stable", "theilsen"])
    put("SenAllOls", sl.loc["all_nodes", "ols"]); put("SenAllTs", sl.loc["all_nodes", "theilsen"])
    put("SenAllHub", sl.loc["all_nodes", "huber"])
    ws = s.loc["WSE relative (cm)"]
    put("SenNoScrN", ws.loc["no_screening", "n"], "{:,.0f}")
    put("SenNoScrPub", ws.loc["no_screening", "official"], "{:.1f}"); put("SenNoScrHub", ws.loc["no_screening", "huber"], "{:.1f}")
    put("SenLevN", ws.loc["levelled", "n"]); put("SenLevPub", ws.loc["levelled", "official"], "{:.1f}")
    put("SenLevHub", ws.loc["levelled", "huber"], "{:.1f}")


def simulation():
    s = load("simulation_summary.csv")
    if s is None:
        return
    def g(m, k, e):
        v = round(100 * s[(s.model == m) & (s.k == k) & (s.estimator == e)].rel_bias_median.iloc[0])
        return "0" if v == 0 else (f"{v:+d}".replace("-", "$-$"))
    put("SimTwoUp", g("pattern", 2.0, "slope2")); put("SimTwoDn", g("pattern", 0.5, "slope2"))
    put("SimPubUp", g("pattern", 2.0, "pub")); put("SimPubDn", g("pattern", 0.5, "pub"))
    put("SimTwoUpIid", g("iid", 2.0, "slope2"))
    put("SimN", int(s.n.min()), "{:,}")
    put("SimHalfWidth", 100 * (s.rel_bias_hi - s.rel_bias_lo).max() / 2, "{:.0f}")


def budget():
    u = load("estimator_uncertainty.csv")
    if u is not None:
        u = u.set_index("estimator")
        put("UtsNmad", u.loc["ts", "nmad_z"], "{:.1f}"); put("UhubNmad", u.loc["huber", "nmad_z"], "{:.1f}")
        put("UolsNmad", u.loc["ols", "nmad_z"], "{:.1f}")
    f = RESULTS / "wse_long.parquet"
    if f.exists():
        w = pd.read_parquet(f, columns=["site", "node_set", "screened"])
        sites = w[(w.node_set == "clean") & ~w.screened]["site"].unique()
        gm = pd.read_csv(DATA / "gauge_match.csv").set_index("site").reindex(sites)
        put("DatumAccMedCm", 30.48 * gm["altitude_accuracy"].median(), "{:.0f}")
        put("DatumAccPninetyCm", 30.48 * gm["altitude_accuracy"].quantile(.9), "{:.0f}")


def dxcheck():
    d = load("wse_offset_by_dx.csv")
    if d is not None:
        put("DxNear", d["pub"].iloc[0], "{:.0f}"); put("DxFar", d["pub"].iloc[4], "{:.0f}")
        put("DxNearSm", d["sm"].median(), "{:.0f}")
    try:
        import p39.geodesy as G
        put("ProjAccCm", 100 * G._T.accuracy, "{:.1f}")
    except Exception:
        pass


def drivers():
    d = load("drivers_slope.csv")
    if d is None:
        return
    x = d[d["var"] == "xtrk_km"].iloc[0]
    put("DrvXtrkPub", x["official"]); put("DrvXtrkTs", x["theilsen"])
    sc = d[d["var"] == "scatter_cm"]
    put("DrvScatPub", sc.iloc[-1]["official"]); put("DrvScatTs", sc.iloc[-1]["theilsen"])
    put("DrvScatLoEdge", float(sc.iloc[0]["bin"].split(",")[1].strip(" ]")), "{:.0f}")
    put("DrvLowPub", sc.iloc[0]["official"]); put("DrvLowTs", sc.iloc[0]["theilsen"])


def main():
    samples(); slope(); wse(); repro(); audit(); drivers(); dxcheck(); bayes_extra(); example(); sensitivity(); simulation(); budget()
    put("Tend", "30 September 2026")
    put("CodeURL", "\\url{https://github.com/academic-files-yk/swot-reach-estimators}")
    # macros used in the text but not computed yet are shown as ?? in the PDF
    import re
    tex_dir = TEX
    used = set()
    for f in tex_dir.glob("*.tex"):
        if f.name != "numbers.tex":
            used |= set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", f.read_text()))
    for k in used - set(M):
        M[k] = "??"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    out = TEX / "numbers.tex"
    out.write_text("% generated by 18_numbers.py - do not edit\n" + "\n".join(lines) + "\n")
    print(len(M), "macros ->", out)


if __name__ == "__main__":
    main()
