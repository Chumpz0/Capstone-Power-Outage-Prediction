# EDA handoff to the modeling teammate

Start with [the final EDA report](eda_report.md), then [modeling criteria](modeling_criteria.md) and [feature availability](feature_availability.md). The current task is hourly county preparation using future 6/12-hour customer counts and >5,000 classification. Notebook 03's storm-conditioned >=5% analysis is descriptive context, not the model label.

## How to reproduce

Use the existing processed data snapshot with the hashes in `docs/eda_metadata.json`. The scripts require local files under `data/processed/`; remote-streaming loaders alone do not satisfy the batch audit paths. Obtain the team's existing snapshot through its established access process. Do not refresh sources just to reproduce this handoff.

From the repo root, with the project environment and `requirements.txt` installed:

```powershell
# Rebuild training-only quality and target summaries (several minutes).
.venv\Scripts\python.exe scripts/check_observation_reliability.py
.venv\Scripts\python.exe scripts/check_customer_targets.py

# Rebuild geography, run tests and all notebooks, refresh audit and metadata.
.venv\Scripts\python.exe scripts/finalize_eda.py
```

If the quality/target summaries already exist and their hashes match, only the last command is needed. Notebook 04 checks source and calculation-code fingerprints. If they differ, or an older summary lacks code fingerprints, rerun the first two commands. The finalizer executes notebooks 01-04 in order with the current Python kernel, rebuilds the geography overlay, writes `docs/eda_validation.json`, and refreshes `docs/eda_metadata.json`. Descriptive notebooks 01-03 include 2024/2025; running them is not a blind-test procedure.

For interactive work, select this repo's `.venv` kernel, build the geography output first, and restart/run all cells. The finalizer is the reproducible route when executing notebooks and updating the audit. The environment versions used for this handoff are recorded in metadata; `requirements.txt` is a minimum-version list rather than a lockfile. No command above downloads new data, uploads, commits or trains a model.

## What to reuse

| Component | Use |
| --- | --- |
| `src/observation_quality.py` | Exact-grid future-window observation status; outcome audit only |
| `src/outage_targets.py` | Uncapped recorded peak, median/proxy sensitivity, threshold labels |
| `src/forecast_criteria.py` | UTC chronological assignment with horizon-specific boundary exclusions |
| `scripts/check_customer_targets.py` | Bounded county/year processing pattern and training cohort reconciliation |
| `data/processed/eda/geography_criteria/storm_event_counties.parquet` | Full corrected historical event table for descriptive attribution |
| `docs/eda_metadata.md` and `.json` | Field meanings, keys, null semantics, schemas, provenance and environment |

The primary per-origin cohort mask is `in_bounds & (usable == 4 * horizon) & ~flagged`. `labels()` alone also returns partial observed positives, so do not mistake its nonmissing values for the approved strict cohort. Keep audit keys and observation status alongside targets. The full county-hour feature/label training table is deliberately not materialized here; constructing it is part of the teammate's modeling work.

The commands generate local summary Parquets under `data/processed/eda/` and aggregate checkpoint JSON files under `docs/`. The geography JSON records each supported correction. Charts remain embedded in the notebooks; no separate figure files are exported.

## What remains undecided

Before building the baseline, agree on:

1. Which past outage features to use, how much reporting delay to assume, and how to handle missing inputs.
2. Which baseline models to compare using the chronological splits.
3. The minimum severe-outage recall and acceptable false-alert rate needed for county preparation.

Use the [modeling criteria](modeling_criteria.md) for evaluation rules, including
coverage by county/state, errors during severe outages and repeated alerts.
Freeze model and alert choices before the historical test evaluation, and disclose
the earlier EDA exposure to that data.

The next step is to build the county-hour modeling table. Advance weather data
and operational deployment validation remain future work.
