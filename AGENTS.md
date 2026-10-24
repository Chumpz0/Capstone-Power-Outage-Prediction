# Project goal

Predict the probability that a storm causes a major power outage in a county in Alabama, Florida, Georgia, North Carolina, South Carolina or Tennessee, using EAGLE-I outage observations, NOAA Storm Events and archived NWS alerts. Supporting objective: peak simultaneous customers out. Approved and open decisions are in `docs/project_specification.md` and `docs/decision_log.md`; check them before changing labels, rows or features.

## Current endpoint

The EDA contribution ends at completed data preparation and training-only EDA (D-25). Previous local modeling code, artifacts, tracking and results were removed. The teammate owns and may develop the baseline in this shared repository, declaring experiments before running them. Agents working on the EDA contribution must not independently restart modeling. Preserve prior 2024 development and 2025 exploration disclosures. Historical notebooks in `notebooks/legacy/` are disabled and must not be rerun to explore 2025 outcomes.

## Evaluation plan

- Train through 2023; validate on 2024; final held-out test on 2025, scored once after the model, thresholds, features and protocol are frozen.
- Prefer chronological splits. Keep held-out years out of model fitting and preprocessing fitting, and do not use final test results to tune the model.
- Previous 2025 EDA is disclosed; freeze further target-driven exploration or tuning on 2025. Structural coverage audits of 2025 are allowed; outcome-window summaries and label counts are not.
- Overlapping storms are dependent observations: never split randomly by event; estimate uncertainty by resampling storm episodes or dates.

## Decision checkpoints

- **A (approved 2026-10-23):** audience = utility crew staging 12-48 h ahead; prediction time = first storm-family NWS watch or warning for the county (alert rows primary; NOAA-onset rows as comparison).
- **B (approved 2026-10-23):** major = at least 5,000 customers out simultaneously; non-major under `cov90`; windows already major at `t0` excluded and reported separately. Never change a threshold because it produces attractive model scores.
- **After the first baselines (approved 2026-10-23):** alert rows use 72-hour windows (non-overlapping per county); models are fit and evaluated on covered windows; alert threshold = recall 0.80; storms only.
- **Open before freezing the protocol:** teammate baseline and evaluation design (O-12), severity population (O-7), implausible-peak counties (O-8) (`docs/decision_log.md`). Declare any new experiment in `docs/evaluation_protocol.md` before running it. Do not score 2025 until the protocol is frozen.

## Modeling and cleaning guidance

- Rows are county prediction opportunities at a prediction time `t0`; the outcome window is `[t0, t0 + 72 h)` for alert rows (288 EAGLE-I runs) and `[t0, t0 + 24 h)` for onset rows. Windows never overlap within a county, so each county outcome is counted once (stages 10, 13).
- Labels: *major* when an observed value crosses the threshold; *non-major* only under the approved coverage rule; otherwise *unknown*. Never read absent EAGLE-I rows as zero (EAGLE-I: absence means zero or a collection gap). Rules live in `pipeline/labels.py`.
- Report new threshold crossings and outages already major at `t0` separately.
- Severity needs its own completeness policy: a peak is exact only when all scheduled runs are observed (288 for alert windows, 96 for onset windows).
- Use only information available at `t0`: EAGLE-I runs that started at or before `t0 - 15 min`, alerts issued by `t0`, history from earlier windows. Exclude realized duration, magnitude, damage, narratives, later storm records, outcome-window summaries, and IEM `utc_expire`, `status` and `utc_updated` (final alert state). Columns realized after `t0` carry a `retro_` prefix.
- Customer counts are the primary severity measure. Use percentages only with a validated denominator (`county_reference.mcc_status = 'valid'` and plausible outage peaks), reported separately from count labels.
- Results apply to the labelled population, which is selected toward large counties and recent years; never present them as all storms or as causal effects.
- Previous national and Southeast modeling has been removed under D-25. Superseded national preparation and variable-window join work were removed in the EDA cleanup.
- Preserve raw datasets; make cleaning and feature selection reproducible in scripts. Initial data-quality outputs used the first 10,000 rows per CSV; full-file profiles supersede them.

## Workspace

Stages are numbered and run in order; commands are in `README.md`. All paths are defined once in `pipeline/paths.py`; shared rules (study area, DuckDB session, SQL macros, cache hashing) in `pipeline/common.py`.

- `data/01_raw/`: NOAA Storm Events, the EAGLE-I archive, and stage-08 downloads (`nws_zone_county/`, `iem_vtec/`, `census_counties/`, each with a checksum manifest). Never modified.
- `data/02_combined/`, `data/03_cleaned/`: combined CSVs and national cleaning v3 Parquet (`.meta.json` sidecars). `data/03_cleaned/southeast/`: six-state base tables.
- `data/reference/`: EAGLE-I archive files (MCC, DQI, coverage history), Connecticut crosswalk, `nws_zone_county_versions.parquet`.
- `data/04_model_input/southeast_v1/`: current prepared Southeast inputs; test files hold features only.
- `pipeline/00_verify_raw_sources.py`: checks raw files against `pipeline/raw_sources.json`.
- `pipeline/01`-`04`, `profile_full.py`: shared preparation (stage 4 needs the full profile first).
- `pipeline/08`-`13`: Southeast pipeline: fetch references, zone crosswalk, base tables and onset opportunities, label audit, alert events and opportunities, model inputs (`data/04_model_input/southeast_v1/`; test files hold features only).
- `notebooks/06_major_outage_eda.ipynb`: current training-only EDA and repository endpoint.
- `pipeline/14_export_eda_report.py`: executes the training-only EDA notebook and exports the standalone `reports/eda/major_outage_eda.html` report.
- `reports/`: generated reports (`quality/`, `full_profile/`, `zone_crosswalk/`, `label_audit/`, `forecast_feasibility/`).
- `docs/`: specification, decision log, step reports, data sources. `tests/`: pytest suite.

## Data rules

- Five-character county FIPS strings and UTC timestamps for joins. The Census county list (534 counties) is authoritative.
- NOAA `CZ_TYPE=Z` codes are forecast zones. Map them with the NWS correlation version in effect at the event date and verify names (stage 09); NOAA zone codes can lag NWS changes. County-coded records must match their county's name; failures are reinterpreted as zones only on a verified zone name.
- NOAA time conversion follows the recorded numeric UTC offset, with daylight-time labels flagged. Retain original identifiers and times for auditing.
- EAGLE-I conventions change by source year: zeros omitted 2014-2021 (nulls from 2017), intermittent zeros 2022-2024, nulls and rare zeros in 2025. 2014-2021 files end at Dec 31 00:00 UTC; the Southeast pipeline keeps Dec 31 and lets coverage rules decide.
- Customer denominators: MCC is invalid for North Carolina; EAGLE-I 2024 `total_customers` is inflated in FL, NC and SC. See `reports/label_audit/denominator_validation*.csv`.
- Versioning: stages 4 and 10 rebuild an output when its build hash (rules, macros, query, source content, upstream sidecars) or version changes. Bump `VERSION` when a rule changes meaning.
- Cleaning v3 (national outputs only): `total_customers` dropped; Dec 31 UTC observations dropped; Drought and over-72-hour events excluded from join preparation; Connecticut regions allocated to legacy counties.
