from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_PJM = ROOT / "data" / "raw"
DATA_RAW_NOAA = ROOT / "data" / "raw" / "noaa"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

# PJM
PJM_API_URL = "https://api.pjm.com/api/v1/hrl_load_metered"
PJM_API_KEY = os.getenv("PJM_API_KEY", "")
PJM_LOAD_AREA = "RTO"

# Current test dataset: January 2025
START_YEAR = 2025
END_YEAR = 2025

# NOAA station(s)
NOAA_STATIONS = {
    "USW00093819": "Indianapolis International Airport",
}

NOAA_BASE_URL = (
    "https://www.ncei.noaa.gov/"
    "oa/global-historical-climatology-network/hourly/access/by-station"
)
