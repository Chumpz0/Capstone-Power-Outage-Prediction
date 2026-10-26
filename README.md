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
| 05 Baseline models | [05_baseline_models](notebooks/05_baseline_models.ipynb) | | In progress (naive rules logged to W&B) |
| 06 EDA of the ML-ready dataset | [06_ml_dataset_eda](notebooks/06_ml_dataset_eda.ipynb) | [summary](reports/06_ml_dataset_eda_summary.md) | Done: [EDA report (PDF)](reports/EDA_report.pdf), [dataset card](reports/dataset_card.md) |

**Success criteria** were fixed before any model was trained: [reports/success_criteria.md](reports/success_criteria.md). The bar to beat is a no-ML rule (storm family rate × county history) with a 2024 PR-AUC of 0.469. Every model is logged to the Weights & Biases team project **CAPSTONE MODEL** (`leslie-ortega-us-georgia-state-university`), with the shared checks in [src/evaluation.py](src/evaluation.py).

Each step has a notebook with commented code, and a summary in `reports/` laid out as **[GOAL] – [FINDINGS] – [NEXT STEPS]**. Start with the summaries.

## Repo layout

```
notebooks/                 one notebook per step (01 -> 04 build the data, 05 models, 06 profiles the ML-ready table)
src/data.py                data_path(): local file if present, otherwise streamed from Hugging Face; model-table loader
src/features.py            the feature list, excluded features (with reasons), preprocessing groups
src/models.py              logistic regression + gradient boosting pipelines, run_experiment()
src/evaluation.py          shared success-criteria checks + W&B logging (used from step 05)
scripts/train.py           train one model and log it:  python scripts/train.py --model hgb --learning-rate 0.05
scripts/build_eda_report.py  renders reports/eda_report.md + dataset_card.md to reports/EDA_report.pdf
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

The data isn't stored in GitHub. It lives in the private Hugging Face dataset **[DSCI-Capstone/county-outage-data](https://huggingface.co/datasets/DSCI-Capstone/county-outage-data)** (5.3 GB), laid out like this repo (`data/`, `processed/`, `holdout/`).

**You don't have to download anything.** Each notebook reads a file from the repo if it's there, and otherwise streams it from Hugging Face. You only need access to the `DSCI-Capstone` org and a one-time login:

```bash
hf auth login     # paste a Read token from huggingface.co/settings/tokens
```

To work offline or faster, download what you need into the repo root. The notebooks then read the local copy:

```bash
hf download DSCI-Capstone/county-outage-data --repo-type dataset --local-dir . --include "processed/*"   # EDA + modeling (4.2 GB)
hf download DSCI-Capstone/county-outage-data --repo-type dataset --local-dir . --include "data/*"        # raw inputs for 01-02 (0.7 GB)
```

`holdout/test_labels.parquet` holds the 2025 test labels. **Don't download or open it until the final evaluation.**

The two big CSVs are stored as Parquet on Hugging Face, with every column still text (a lossless copy). Notebooks 01–02 read the CSV when it's there, and otherwise the Parquet:

```
data/02_combined/eaglei_outages_2014_2025.csv          EAGLE-I yearly files combined (13 GB)  -> HF: data/02_combined/eaglei_outages/SOURCE_YEAR=*/
data/02_combined/storm_events_details_all_years.csv    NOAA Storm Events details, 2014-2025   -> HF: data/02_combined/storm_events_details_all_years.parquet
data/01_raw/census_counties/co-est2023-alldata.csv     Census county population, Vintage 2023
data/reference/eaglei_archive/MCC.csv                  EAGLE-I modeled customers per county
data/reference/eaglei_archive/DQI.csv                  EAGLE-I data quality index
data/reference/eaglei_archive/coverage_history.csv     EAGLE-I customer coverage by state
data/reference/nws_zone_county_versions.parquet        dated NWS forecast-zone -> county crosswalk
data/reference/ct_region_to_county_weights.csv         Connecticut planning region -> county weights
```

Running the notebooks in order rebuilds `work/` (~2 min) and `processed/` (~20 min), then the EDA and the modeling table (under a minute each). Outputs are always written locally.

Modeling table outputs (`processed/model_table/`):
- `train.parquet`, `validation.parquet`
- `test_features.parquet`. Notebook 04 writes the test labels to a separate `test_labels.parquet` (on Hugging Face it sits in `holdout/`)

## Things to know about the data

- A missing EAGLE-I row means **0 outages or no data**, so we can't tell which. Rows are only labeled when the county was being observed (see the 02 summary).
- The denominator is EAGLE-I's modeled customer count (MCC), with a Census-based estimate where MCC is broken: all of North Carolina, plus 21 single counties.
- Columns starting with `retro_` are known only after the storm. Never use them as model inputs.
- Storm reports are *observed* events, standing in for forecasts. Expect real forecast-time accuracy to be somewhat lower.

## Roadmap

1. ~~Repo refactor to this structure~~ (this branch)
2. ~~Host the data on Hugging Face~~ ([DSCI-Capstone/county-outage-data](https://huggingface.co/datasets/DSCI-Capstone/county-outage-data))
3. ~~Weights & Biases experiment tracking~~ (team project CAPSTONE MODEL)
4. Align with the course checkpoint expectations
5. Baseline models: logistic regression, then gradient boosting, judged against [success criteria](reports/success_criteria.md)
