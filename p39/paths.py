"""Central paths. Large data live outside the synced folder (WORK)."""
from pathlib import Path
import os

CODE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("P39_WORK", Path.home() / "swot_p39_work"))
SWORD_DIR = WORK / "sword"
IRIS_FILE = WORK / "iris" / "IRIS_3.4.nc"
NODE_DIR = WORK / "swot" / "node"
REACH_DIR = WORK / "swot" / "reach"
GAUGE_DIR = WORK / "gauges" / "usgs"
DATA = WORK / "data"
RESULTS = WORK / "results"
OUTPUT = CODE / "output"          # small tables that go with the code
FIGURES = Path(os.environ.get("P39_FIG", CODE / "figures"))
TEX = Path(os.environ.get("P39_TEX", CODE / "tex"))
FIGURES.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)

HYDROCRON = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
COLLECTION = "SWOT_L2_HR_RiverSP_D"
USGS_OGC = "https://api.waterdata.usgs.gov/ogcapi/v1"
START = "2023-07-26T00:00:00Z"     # start of the SWOT science orbit
END = "2026-10-01T00:00:00Z"

for p in (DATA, RESULTS, OUTPUT, NODE_DIR, REACH_DIR, GAUGE_DIR):
    p.mkdir(parents=True, exist_ok=True)
