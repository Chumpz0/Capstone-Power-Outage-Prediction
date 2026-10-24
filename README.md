# Storm-to-Outage: Southeast county outage prediction

**Current endpoint: completed EDA and prepared inputs. Baseline modeling belongs to the teammate.** Predict whether a county reaches **5,000 simultaneous customers out** in the **72 hours after its first storm-family NWS watch or warning**. Fit/evaluate on covered windows (`cov90`, not already major at prediction time). Peak customer counts are the supporting objective; percentages require validated denominators.

## Current EDA and baseline handoff

- [Finished HTML report](reports/eda/major_outage_eda.html): download/open locally in a browser; five charts and ten tables. GitHub displays HTML source rather than running it.
- [Training-only notebook](notebooks/06_major_outage_eda.ipynb), [dataset card](docs/dataset_card.md), [feature dictionary](docs/feature_dictionary.md).
- [Teammate handoff](docs/teammate_handoff.md), [approved decisions](docs/decision_log.md), [evaluation safeguards](docs/evaluation_protocol.md).

Training ends in 2023; 2024 is development validation; 2025 is scored once after freezing. Prior 2024 development and earlier 2025 exploration remain disclosed. Previous experimental models are not included. The earlier percentage/variable-window EDA is retained under `notebooks/legacy/` with outputs cleared and execution disabled.

### Setup and approved data

The EDA environment is tested on Python 3.14. From the repo root:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-eda.txt
.venv/Scripts/hf.exe auth login
.venv/Scripts/python.exe scripts/download_eda_data.py
.venv/Scripts/python.exe pipeline/14_export_eda_report.py
.venv/Scripts/python.exe -m pytest -q -rs
```

The downloader needs `docs/eda_release.json`, created after publication to the private team dataset. It pins a commit, verifies SHA-256 hashes and refuses to overwrite different local inputs. Until publication, obtain the approved inputs from the data owner or follow the [full pipeline instructions](docs/pipeline_setup.md). `POWER_DATA_ROOT` can point to an existing complete data directory; otherwise paths use this repo's `data/`.

```python
from src.major_outage import load_major_outage_xy
X_train, y_train, groups_train = load_major_outage_xy("train")
X_validation, y_validation, groups_validation = load_major_outage_xy("validation")
```

These loaders select covered windows and allowlisted features. They do not impute, fit or score. The separate final-test loader exposes features and IDs only. Declare baseline experiments before running them.

`requirements-eda.txt` records the tested EDA versions. `requirements.txt` preserves team modeling dependencies; add them as needed and validate version changes. Full-base output tests skip when only the small release is installed; inspect the skip report.

`pipeline/`, `tests/`, `docs/` and `reports/` contain the authoritative preparation and EDA. `src/major_outage.py` serves approved inputs. The existing `src/data.py` and `scripts/download_data.py` remain a separate legacy schema, not a substitute for stage 13. Shared paths/geography/splits are defined in the pipeline and imported by `src/config.py`.

## Legacy acquisition workflow (reference only)

The material below describes the original low-level data workflow. Its current-zone mapping and percentage guidance are superseded for the approved baseline by the dataset card and historical mapping pipeline above. Do not rerun its historical notebooks or target summaries on 2025.

DSCI 4350 Data Science Operations & Engineering capstone, Georgia State University, Fall 2026.

The original exploration considered peak shares; the approved objective now uses customer counts as described above. The study area and chronological splits remain AL, FL, GA, NC, SC, TN; train 2014–2023, validate 2024, test 2025.

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
src/plotting.py            shared chart colors and helpers for the notebooks
scripts/download_data.py   download + filter all data
scripts/check_data.py      quick data summary
scripts/upload_to_hf.py    push data/processed/ to the shared Hugging Face dataset
data/                      git-ignored except data/README.md
notebooks/                 EDA and modeling notebooks (add yours here)
```
