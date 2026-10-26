# Storm-to-Outage: Predicting County Power Outages from Severe Weather

DSCI 4350 Data Science Operations & Engineering capstone, Georgia State University, Fall 2026.

**Question:** given a storm, what is the probability that a US county loses power?

- **Unit:** one county hit by one NOAA storm episode.
- **Target:** at least **5% of the county's customers out** within **24 hours** of the episode starting there.
- **Scope:** lower 48 states + DC, 2017–2025.
- **Split by time:** train 2017–2023, validation 2024, test 2025. Test is used once, at the end.
- **Use case:** utility crew staging, i.e. pre-positioning crews in the counties most at risk.

Data: [EAGLE-I](https://figshare.com/articles/dataset/24237376) county outage snapshots (every 15 minutes) and [NOAA Storm Events](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/).

## Status

| Step | Notebook | Summary | Status |
|---|---|---|---|
| 01 Data audit | [01_data_audit](notebooks/01_data_audit.ipynb) | [summary](reports/01_data_audit_summary.md) | Done |
| 02 Data prep | [02_data_prep](notebooks/02_data_prep.ipynb) | [summary](reports/02_data_prep_summary.md) | Done |
| 03 EDA (training years only) | [03_eda](notebooks/03_eda.ipynb) | [summary](reports/03_eda_summary.md) | Done |
| 04 Modeling table | [04_modeling_table](notebooks/04_modeling_table.ipynb) | [summary](reports/04_modeling_table_summary.md) | Done |
| 05 Baseline models | | | Next |

**Success criteria** were fixed before any model was trained: [reports/success_criteria.md](reports/success_criteria.md). The bar to beat is a no-ML rule (storm family rate × county history) with a 2024 PR-AUC of 0.472.

Each step has a notebook with commented code, and a summary in `reports/` laid out as **[GOAL] – [FINDINGS] – [NEXT STEPS]**. Start with the summaries.

## Repo layout

```
notebooks/                 one notebook per step, run in order (01 -> 04)
reports/                   step summaries, success criteria
reports/figures/           EDA charts
reports/*_tables/          small CSV result and check tables written by the notebooks
requirements.txt           pinned packages (Python 3.14, see .python-version)
data/        (not in git)  input data, read-only
processed/   (not in git)  cleaned tables + modeling table, built by notebooks 02 and 04
work/        (not in git)  DuckDB cache of the raw CSVs, built by notebook 01
```

## Setup

```bash
uv venv .venv --python 3.14          # or: python -m venv .venv
uv pip install -r requirements.txt   # or: .venv\Scripts\activate && pip install -r requirements.txt
```

Select the `.venv` interpreter as the Jupyter kernel.

## Data

The data is about 20 GB, so it isn't stored in GitHub. **A Hugging Face dataset is coming next**, so teammates won't need to rebuild anything. Until then, notebooks 01–02 expect these files under `data/`:

```
data/02_combined/eaglei_outages_2014_2025.csv          EAGLE-I yearly files combined (13 GB)
data/02_combined/storm_events_details_all_years.csv    NOAA Storm Events details, 2014-2025
data/01_raw/census_counties/co-est2023-alldata.csv     Census county population, Vintage 2023
data/reference/eaglei_archive/MCC.csv                  EAGLE-I modeled customers per county
data/reference/eaglei_archive/DQI.csv                  EAGLE-I data quality index
data/reference/eaglei_archive/coverage_history.csv     EAGLE-I customer coverage by state
data/reference/nws_zone_county_versions.parquet        dated NWS forecast-zone -> county crosswalk
data/reference/ct_region_to_county_weights.csv         Connecticut planning region -> county weights
```

Running the notebooks in order rebuilds `work/` (~2 min) and `processed/` (~20 min), then the EDA and the modeling table (under a minute each).

Modeling table outputs (`processed/model_table/`):
- `train.parquet`, `validation.parquet`
- `test_features.parquet`, and test labels kept separately in `test_labels.parquet`

## Things to know about the data

- A missing EAGLE-I row means **0 outages or no data**, so we can't tell which. Rows are only labeled when the county was being observed (see the 02 summary).
- The denominator is EAGLE-I's modeled customer count (MCC), with a Census-based estimate where MCC is broken: all of North Carolina, plus 21 single counties.
- Columns starting with `retro_` are known only after the storm. Never use them as model inputs.
- Storm reports are *observed* events, standing in for forecasts. Expect real forecast-time accuracy to be somewhat lower.

## Roadmap

1. ~~Repo refactor to this structure~~ (this branch)
2. Host the data on Hugging Face
3. Weights & Biases experiment tracking
4. Align with the course checkpoint expectations
5. Baseline models: logistic regression, then gradient boosting, judged against [success criteria](reports/success_criteria.md)
