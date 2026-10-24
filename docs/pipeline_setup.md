# Southeast storm outage prediction

**Read the finished [EDA report](../reports/eda/major_outage_eda.html)** in a browser. It contains five charts, ten tables, interpretations and handoff notes. The executable source is [the training-only notebook](../notebooks/06_major_outage_eda.ipynb).

Predict the probability that a storm causes a **major power outage** in a county in Alabama, Florida, Georgia, North Carolina, South Carolina or Tennessee, and estimate the **peak number of customers out** as a supporting task. Outages come from EAGLE-I; storms come from NOAA Storm Events and archived NWS alerts.

**Approved task:** for utility crew staging, predict whether a county will have at least 5,000 customers out at the same time in the 72 hours after the first storm-family NWS watch or warning issued for it. Models are fit and evaluated on covered windows, where both outcomes needed near-continuous reporting. Repository endpoint: completed data preparation and training-only EDA. The teammate owns the first baseline; previous modeling code and results have been removed. 2024 was previously used for model development; 2025 has not been scored. Rules for contributors and agents are in `AGENTS.md`.

| Document | Contents |
|---|---|
| [docs/project_specification.md](project_specification.md) | Approved and open decisions, information rules, checkpoints |
| [docs/decision_log.md](decision_log.md) | Every decision with date, status and evidence |
| [docs/forecast_feasibility.md](forecast_feasibility.md) | Step 2: can we predict before arrival? |
| [docs/label_feasibility_report.md](label_feasibility_report.md) | Steps 3-4: label trust, selection effects, candidate thresholds |
| [docs/southeast_dataset.md](southeast_dataset.md) | Step 5: six-state build, exclusions, checks |
| [docs/feature_dictionary.md](feature_dictionary.md), [docs/dataset_card.md](dataset_card.md) | Step 6: predictors and their timing; model-input dataset card |
| [docs/evaluation_protocol.md](evaluation_protocol.md) | Evaluation safeguards and decisions for the teammate before baseline work |
| [docs/data_sources.md](data_sources.md) | Sources, versions, redistribution notes |

This page documents the full preparation route. For the smaller versioned data download, see [teammate_handoff.md](teammate_handoff.md).

## Setup from a fresh clone

Python 3.14 and [uv](https://docs.astral.sh/uv/):

```
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-eda.txt
```

Put the NOAA Storm Events `.csv.gz` files and the EAGLE-I archive `24237376.zip` in `data/01_raw/` (sources in `docs/data_sources.md`), then check them:

```
.venv/Scripts/python.exe pipeline/00_verify_raw_sources.py
```

## Pipeline

Run from any working directory. Each stage reads earlier outputs.

**Shared preparation**

| Stage | Command | Output |
|---|---|---|
| 1 | `pipeline/01_combine_noaa.py` | `data/02_combined/storm_events_*_all_years.csv` |
| 2 | `pipeline/02_combine_eaglei.py` | `data/02_combined/eaglei_outages_2014_2025.csv`, `data/reference/eaglei_archive/` |
| 3 | `pipeline/03_build_ct_crosswalk.py` | `data/reference/ct_region_to_county_weights.csv` |
| profile | `pipeline/profile_full.py` | `reports/full_profile/` (required: stage 4 checks row counts against it) |
| 4 | `pipeline/04_clean_datasets.py` | `data/03_cleaned/*.parquet`, `reports/quality/` (national, cleaning v3) |

**Southeast major-outage pipeline (authoritative)**

| Stage | Command | Output |
|---|---|---|
| 8 | `pipeline/08_fetch_reference_sources.py` | NWS zone-county versions, IEM alert archive, Census county population in `data/01_raw/` |
| 9 | `pipeline/09_build_zone_crosswalk.py` | versioned zone-county crosswalk; zone and county event mapping; `reports/zone_crosswalk/` |
| 10 | `pipeline/10_build_southeast_base.py` | `data/03_cleaned/southeast/` base tables and onset opportunities |
| 11 | `pipeline/11_audit_major_labels.py` | `reports/label_audit/` |
| 12 | `pipeline/12_audit_alert_feasibility.py` | `reports/forecast_feasibility/`, county alert events, alert-triggered opportunities |
| 13 | `pipeline/13_build_major_outage_dataset.py` | `data/04_model_input/southeast_v1/` features and labels (train and validation; test features only) |
| EDA | `notebooks/06_major_outage_eda.ipynb` | training-only EDA |
| 14 | `pipeline/14_export_eda_report.py` | Executes the EDA and exports `reports/eda/major_outage_eda.html` with embedded charts and tables |

Prefix commands with `.venv/Scripts/python.exe`. DuckDB defaults to 2 threads and 4 GB; set `POWER_DUCKDB_THREADS` and `POWER_DUCKDB_MEMORY` for faster builds (stage 10 reads the 13 GB EAGLE-I CSV in about 9 minutes with 6 threads).

Stages 4 and 10 reuse a Parquet output while its `.meta.json` build hash still matches. The hash covers the rule version, shared SQL macros, query text, source-file contents (hashed once, then cached by size and modification time) and upstream outputs. `--check` lists stale outputs without building; `--force` rebuilds everything. Stage 2 refuses to overwrite existing outputs; stage 8 verifies existing downloads against their manifests.

Start the handoff with `notebooks/06_major_outage_eda.ipynb`, `docs/dataset_card.md`, and `docs/feature_dictionary.md`. Stage 13 produces candidate features and labels; it does not fit a model. Use `covered_window` for the approved modeling population. The teammate will declare the baseline and evaluation choices before running experiments.

Rebuild the browser-readable report with `.venv/Scripts/python.exe pipeline/14_export_eda_report.py`. Supporting full-file profiling and cleaning audits remain in notebooks 02 and 03. Superseded sample-only checks, national modeling splits and variable-window join work have been removed; raw sources are preserved.

## Tests

```
.venv/Scripts/python.exe -m pytest
```

The original 47 rule/output tests cover identifiers, UTC and window boundaries, label rules, pre-`t0` feature timing, duplicate joins, split leakage and cache invalidation; output tests check the built tables and skip when data are absent.

## Folder layout

- `data/`: raw, combined, cleaned and model-input data plus reference files. Not tracked in version control.
- `pipeline/`: numbered preparation stages, shared modules (`paths.py`, `common.py`, `labels.py`) and the EDA exporter.
- `notebooks/`: full-profile, cleaning and current Southeast EDA.
- `reports/`: finished EDA report and supporting quality, crosswalk, label and feasibility reports.
- `docs/`: specification, decision log and step reports.
- `tests/`: pytest suite.
