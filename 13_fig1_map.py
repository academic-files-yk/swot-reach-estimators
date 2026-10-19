"""Fig. 2: evaluated USGS gauges (CONUS, Albers EPSG:5070) and IRIS reaches (world,
Equal Earth EPSG:8857). Boundaries: U.S. Census Bureau cartographic boundary file 2024
(states, 1:20M) and Natural Earth 1:50m countries v5.1.1.
"""
import numpy as np
import pandas as pd
from p39.sets import WSE_REQUIRED
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from p39.paths import WORK, RESULTS, DATA, FIGURES
from p39 import figstyle as F

MAP = WORK / "map"
CONT = {"af": "Africa", "as": "Asia", "eu": "Europe", "na": "North America",
        "oc": "Oceania", "sa": "South America"}


def graticule(ax, crs, lons, lats, lon_rng, lat_rng, label_lon=True, label_lat=True):
    for lo in lons:
        g = gpd.GeoSeries([LineString([(lo, la) for la in np.linspace(*lat_rng, 60)])],
                          crs=4326).to_crs(crs)
        ax.plot(*g.iloc[0].xy, color=F.GRID, lw=0.4, zorder=0)
        if label_lon:
            x, y = g.iloc[0].coords[0]
            ax.text(x, y, f"{abs(lo):.0f}°{'W' if lo < 0 else 'E'}", fontsize=6, color=F.INK2,
                    ha="center", va="top")
    for la in lats:
        g = gpd.GeoSeries([LineString([(lo, la) for lo in np.linspace(*lon_rng, 120)])],
                          crs=4326).to_crs(crs)
        ax.plot(*g.iloc[0].xy, color=F.GRID, lw=0.4, zorder=0)
        if label_lat:
            x, y = g.iloc[0].coords[0]
            ax.text(x, y, f"{abs(la):.0f}°{'S' if la < 0 else ('N' if la > 0 else '')}", fontsize=6, color=F.INK2,
                    ha="right", va="center")


def scalebar(ax, x0, y0, km):
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=F.INK, lw=1.2)
    ax.text(x0 + km * 500, y0, f"{km} km", fontsize=6, ha="center", va="bottom", color=F.INK)


def main():
    F.style()
    wl = pd.read_parquet(RESULTS / "wse_long.parquet")
    wl = wl[(wl["node_set"] == "clean") & ~wl["screened"]]
    est = WSE_REQUIRED
    wl = wl.dropna(subset=[e + "_err" for e in est])     # same overpasses as Table V
    n = wl.groupby("site").size().rename("n")
    gm = pd.read_csv(DATA / "gauge_match.csv").set_index("site").join(n, how="inner")
    sr = pd.read_parquet(RESULTS / "slope_reach.parquet")
    used = sr[(sr["node_set"] == "clean") & (sr["estimator"] == "official")]["reach_id"]
    iris = pd.read_csv(DATA / "iris_sample.csv")
    iris = iris[iris["reach_id"].isin(used)]

    states = gpd.read_file(MAP / "cb_2024_us_state_20m.shp")
    states = states[~states["STUSPS"].isin(["AK", "HI", "PR"])].to_crs(5070)
    world = gpd.read_file(MAP / "ne_50m_admin_0_countries.shp")
    world = world[world["CONTINENT"] != "Antarctica"].to_crs(8857)

    fig, axs = plt.subplots(1, 2, figsize=(F.COL2, 0.34 * F.COL2),
                            gridspec_kw={"width_ratios": [1.05, 1.3], "wspace": 0.03})
    ax = axs[0]
    states.plot(ax=ax, color="#f4f4f2", edgecolor="#b8b8b8", lw=0.3)
    g = gpd.GeoSeries(gpd.points_from_xy(gm["lon"], gm["lat"]), crs=4326).to_crs(5070)
    sc = ax.scatter(g.x, g.y, c=gm["n"], s=3, cmap="Blues", vmin=0,
                    vmax=np.quantile(gm["n"], 0.95), edgecolor="#1f4e8c", lw=0.15, zorder=3)
    cax = ax.inset_axes([0.93, 0.1, 0.025, 0.55])
    cb = fig.colorbar(sc, cax=cax, orientation="vertical")
    cb.ax.set_title("Over-\npasses", fontsize=6, loc="left", pad=3)
    cb.ax.tick_params(labelsize=6)
    graticule(ax, 5070, [-120, -100, -80], [30, 40], (-126, -66), (24, 50), False, True)
    xmin, ymin, xmax, ymax = states.total_bounds
    ax.set_xlim(xmin - 1e5, xmax + 1e5)
    ax.set_ylim(ymin - 1.5e5, ymax + 1e5)
    scalebar(ax, xmin + 1.0e5, ymin + 1.0e5, 500)
    ax.set_axis_off()
    F.panel(ax, "a", f"USGS gauges (n = {len(gm)})")

    ax = axs[1]
    world.plot(ax=ax, color="#f4f4f2", edgecolor="#c8c8c8", lw=0.2)
    gi = gpd.GeoSeries(gpd.points_from_xy(iris["x"], iris["y"]), crs=4326).to_crs(8857)
    for k, (c, name) in enumerate(CONT.items()):
        m = (iris["continent"] == c).values
        ax.scatter(gi.x[m], gi.y[m], s=2.2, color=F.PALETTE[k], lw=0, zorder=3,
                   label=f"{name} ({m.sum()})")
    graticule(ax, 8857, [-120, -60, 0, 60, 120], [-30, 0, 30, 60], (-180, 180), (-58, 84),
              False, True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=5.8, markerscale=2.5,
              handletextpad=0.2, ncol=3, columnspacing=0.8)
    scalebar(ax, -1.62e7, -7.2e6, 2000)
    ax.set_axis_off()
    F.panel(ax, "b", f"IRIS reaches (n = {len(iris)})")
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.1, top=0.84)
    F.save(fig, "Fig1_map", FIGURES)


if __name__ == "__main__":
    main()
