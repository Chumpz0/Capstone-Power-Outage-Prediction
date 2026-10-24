"""Project-wide settings: which states and years we work with, and where data lives."""
import os
from pipeline.paths import ROOT, DATA as DATA_DIR, MODEL_INPUT
from pipeline.settings import SOUTHEAST_STATES, SPLIT_YEARS

# Repo root = the folder that contains src/

RAW_DIR = DATA_DIR / "raw"              # big original downloads (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"  # filtered Parquet files (git-ignored)

# Shared copy of data/processed on Hugging Face (private dataset in the team org).
# Change the default here if you pick a different org/repo name.
HF_REPO_ID = os.getenv("HF_REPO_ID", "DSCI-Capstone/storm-outage-data")

# Study area: Southeast US. Keys are 2-digit state FIPS codes.
STATES = dict(SOUTHEAST_STATES.values())

# Train 2014-2023, validate 2024, test 2025
TRAIN_YEARS = list(range(2014, SPLIT_YEARS['train'][1] + 1))
VALID_YEARS = [SPLIT_YEARS['validation'][0]]
TEST_YEARS = [SPLIT_YEARS['test'][0]]
YEARS = TRAIN_YEARS + VALID_YEARS + TEST_YEARS

# Immutable release path; separate from legacy data/processed and its HF files.
MAJOR_INPUT_DIR = MODEL_INPUT / 'southeast_v1'
HF_EDA_RELEASE = 'southeast_v1/releases/2026-10-23-eda-v2'

# Data sources
FIGSHARE_ARTICLE_ID = 24237376  # EAGLE-I outages 2014-2025 (ORNL)
STORM_EVENTS_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
ZONE_COUNTY_PAGE = "https://www.weather.gov/gis/ZoneCounty"
CENSUS_ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"
# Free key from https://api.census.gov/data/key_signup.html (the API now rejects keyless calls)
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_counties_national.zip"
)
EIA861_URL = "https://www.eia.gov/electricity/data/eia861/zip/f861{year}.zip"
