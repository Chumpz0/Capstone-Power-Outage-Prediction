# Storm-to-Outage: Predicting County Power Outages from Severe Weather

DSCI 4350 Data Science Operations & Engineering capstone, Georgia State University, Fall 2026.

We predict how badly each county loses power when a severe storm hits: peak share of customers out, and whether the outage is "major". The study area is six Southeast states (AL, FL, GA, NC, SC, TN). We train on 2014–2023, validate on 2024 and test on 2025.

## Getting the data

The raw data is about 12 GB, so it is **not** stored in GitHub. One person builds the filtered Southeast data once and uploads it to our **private Hugging Face dataset** (`DSCI-Capstone/storm-outage-data`, set in `src/config.py`). Everyone else just reads it from there.

### Teammates: read the shared data (no download)

1. Make a free [Hugging Face](https://huggingface.co/join) account and accept the invite to the `DSCI-Capstone` organization.
2. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Set up and log in once:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   hf auth login                   # paste the Read token
   ```
4. Load data in any notebook or script:
   ```python
   from src.data import load_outages, load_storm_counties, load_census, load_customer_counts

   out = load_outages([2024], columns=["fips_code", "customers_out", "run_start_time"], states=["GA"])
   storms = load_storm_counties()
   ```
   The loaders read `data/processed/` if you have a local copy, and stream from Hugging Face otherwise. One year of outages is 10M+ rows, so start with one year and only the columns you need.

**In Google Colab:** add your token as a Colab secret named `HF_TOKEN` (key icon in the left sidebar), then:
```python
!git clone https://github.com/<your-org>/Capstone-Power-Outage-Prediction.git
%cd Capstone-Power-Outage-Prediction
!pip install -q -r requirements.txt
from google.colab import userdata; import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from src.data import load_storm_counties
```

### Data owner: build and upload (one person, once)

```bash
pip install -r requirements.txt
export CENSUS_API_KEY=...                              # free key: api.census.gov/data/key_signup.html (PowerShell: $env:CENSUS_API_KEY="...")
python scripts/download_data.py --only storms census   # quick test, a few minutes
python scripts/download_data.py                        # full run; EAGLE-I is ~1.2 GB/yr raw
python scripts/check_data.py                           # sanity summary
hf auth login                                          # paste a WRITE token
python scripts/upload_to_hf.py                         # creates the private dataset if needed
```
Re-run the last command whenever the processed data changes; only changed files upload.

Useful options:

| Option | What it does |
| --- | --- |
| `--years 2024 2025` | Only these years |
| `--only eaglei storms zones census eia` | Only these steps (EIA-861 runs only when asked) |
| `--keep-raw` | Keep the big raw EAGLE-I CSVs after filtering (default: deleted) |

Re-running is safe: finished files are skipped, so an interrupted download just picks up where it stopped. You need about 2 GB of free disk space while a year is processing.

## Data sources

| Source | What it gives us | Output |
| --- | --- | --- |
| [EAGLE-I outages 2014–2025](https://figshare.com/articles/dataset/24237376) (Oak Ridge National Lab) | Customers without power per county, every 15 minutes | `data/processed/eaglei/outages_YYYY.parquet` |
| EAGLE-I reference files | Modeled customer count per county (MCC), coverage history, data quality index | `data/processed/eaglei/*.csv` |
| [NOAA Storm Events](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) | Every recorded storm: type, time, magnitude, damage | `data/processed/storms/storm_events_YYYY.parquet` |
| [NWS zone–county lookup](https://www.weather.gov/gis/ZoneCounty) | Maps forecast-zone storms to counties | `data/processed/reference/zone_county.parquet` |
| Combined storm table | One row per (storm event, affected county) | `data/processed/storms/storm_event_counties.parquet` |
| [Census ACS 5-year](https://www.census.gov/data/developers/data-sets/acs-5year.html) + Gazetteer | County population, housing units, land area, densities | `data/processed/reference/county_census.parquet` |
| [EIA-861](https://www.eia.gov/electricity/data/eia861/) (optional) | Utility characteristics (raw zips only for now) | `data/raw/eia861/` |

See [`data/README.md`](data/README.md) for column details and known data issues.

## Repo layout

```
src/config.py              states, years, splits, source URLs, Hugging Face repo id
src/data.py                load_outages(), load_storm_counties(), ... (local or Hugging Face)
scripts/download_data.py   download + filter all data
scripts/check_data.py      quick data summary
scripts/upload_to_hf.py    push data/processed/ to the shared Hugging Face dataset
data/                      git-ignored except data/README.md
notebooks/                 EDA and modeling notebooks (add yours here)
```

## Project plan (CRISP-DM sprints)

1. **Business + data understanding:** problem statement, data download, tools set up
2. **Data preparation:** storm-to-outage join, data quality report, EDA → Checkpoint 1
3. **Modeling + evaluation:** baselines, LightGBM/XGBoost, year-based validation, SHAP → Checkpoint 2
4. **Tuning + deployment:** final 2025 test, Streamlit county risk map, final report
