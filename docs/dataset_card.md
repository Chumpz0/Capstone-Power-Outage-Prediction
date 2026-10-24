# Dataset card: southeast_v1 major-outage model inputs

Built by `pipeline/13_build_major_outage_dataset.py` into `data/04_model_input/southeast_v1/`. Version 2, 2026-10-23 (72-hour alert windows, decision D-21).

## Purpose

Development data for estimating the probability that a county in AL, FL, GA, NC, SC or TN will have **at least 5,000 customers out at the same time** in the 72 hours after a storm watch or warning. The intended users are utility and mutual-aid planners deciding where to pre-position crews. This is a research dataset; it does not support operational use or causal claims.

## Rows

| Design | Row | Prediction time `t0` | Outcome window | Files |
|---|---|---|---|---|
| Alert (primary) | County window starting at the first storm-family NWS watch or warning for the county; later alerts within 72 hours join the same window | Product issuance time | `[t0, t0 + 72 h)` | `alert_train.parquet`, `alert_validation.parquet`, `alert_test_features.parquet` |
| Onset (comparison) | County window starting at a NOAA-recorded storm onset (retrospective) | Recorded begin time | `[t0, t0 + 24 h)` | `onset_*` |

Windows never overlap within a county, so a county outcome is counted once.

## Targets

- **`y_major`:** 1 when an observed value reaches 5,000; 0 when every observed value is below 5,000 and `cov90` coverage holds (at least 90% of scheduled runs observed and no gap over 1 hour: 260 of 288 for alert rows, 87 of 96 for onset rows); missing otherwise (`label = 'unknown'`). Windows already at 5,000 or more at `t0` have `label = 'ongoing'` and no `y_major`.
- **`covered_window`:** coverage holds and the window is not ongoing. This is the primary population for fitting and evaluation (decision D-22) and the provisional severity population.
- **`peak_customers`:** maximum observed simultaneous customers out in the window.
- Test files contain no outcome columns.

## Size

| Design | Split | Rows | Major | Non-major | Unknown | Ongoing | Covered windows (major) |
|---|---|---|---|---|---|---|---|
| Alert | Train 2014-11 to 2023 | 75,594 | 3,139 | 1,468 | 70,825 | 162 | 2,153 (685) |
| Alert | Validation 2024 | 8,093 | 553 | 318 | 7,157 | 65 | 442 (124) |
| Alert | Test 2025 | 8,669 | features only | | | | |
| Onset | Train | 33,462 | 2,444 | 5,452 | 25,383 | 183 | 7,295 (1,843) |
| Onset | Validation | 3,444 | 413 | 851 | 2,140 | 40 | 1,216 (365) |
| Onset | Test | 3,657 | features only | | | | |

Excluded before labelling: windows in five Tennessee counties EAGLE-I never reports, windows before EAGLE-I coverage (alerts from January to October 2014), and windows crossing a split boundary (342 alert windows from the last three days of 2024). Counts by reason are in `manifest.json`.

## Features

76 numeric alert-design and 52 numeric onset-design predictors, plus `state`, from NWS alerts, EAGLE-I conditions before `t0`, county outage history from earlier days, calendar fields, and Census and EAGLE-I county context. Definitions, timestamps and justifications: [feature_dictionary.md](feature_dictionary.md). Analysis columns (`size_bin`, `primary_family`, `outage_plausibility`, `trigger_partial_county`, `retro_noaa_event_in_window`) are for subgroup reporting only.

## Lineage

NOAA Storm Events and EAGLE-I (stages 01-04); NWS zone-county versions, IEM alerts and Census population (stage 08); zone crosswalk (09); six-state base tables (10); county alert events (12); model inputs (13). Input hashes are recorded in `manifest.json`.

## Known biases and limitations

- **Covered windows are few and selected.** Three days of near-continuous reporting occur mostly in large counties with persistent small outages: in validation, 158 of 442 covered windows are in counties over 500,000 people and 71 are under 25,000.
- **Labels outside covered windows are asymmetric.** One crossing makes a major, so 64% of labelled validation windows are major, and reporting features reveal the label. Use them only as diagnostics.
- **Customer counts are EAGLE-I counts.** Coverage is 71-99% of customers by state; 103 counties have recorded peaks above their customers (possible double counting); unique customers and restoration are not measured.
- **Geography.** Zone alerts map to every county they touch; partial-county zones are attributed to the whole county.
- **Dependence.** Windows from the same storm are correlated; use `cluster_date` (UTC date of `t0`) for grouped resampling.
- **2025 reporting convention differs** (nulls in place of zeros) and must be checked before the final evaluation.

## Access and sharing

Built from public U.S. government data, the IEM archive and the EAGLE-I figshare release. Confirm redistribution terms in [data_sources.md](data_sources.md) before publishing derived tables.
