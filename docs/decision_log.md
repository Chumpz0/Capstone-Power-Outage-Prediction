# Decision log

Newest first within each section. Status is **approved**, **provisional** (an engineering default that the team should review), **open** (waiting for a checkpoint) or **superseded**. Owners are unassigned until the team allocates them.

## 2026-10-23: EDA handoff (approved by the user)

| ID | Decision | Status |
|---|---|---|
| D-25 | End this repository at completed data preparation and EDA. Remove all previous model training code, fitted artifacts, tracking runs and model results, including the superseded national modeling/EDA package. The teammate owns the baseline. Preserve prepared data, current Southeast EDA and approved task/label/split rules. Retire old model grids and provisional performance criteria. | approved |

Prior 2024 model development and earlier 2025 exploration remain disclosed. Removing experiments does not make either period untouched. D-25 supersedes earlier instructions to retain preliminary models or use local MLflow; D-21 through D-24 remain approved.

EDA cleanup follow-up (2026-10-23, user approved): removed empty experiment folders, superseded national preparation/splits, the historical variable-window join and its outputs, and the superseded 10,000-row sample notebook. Retained raw sources, shared preparation, full-file quality/cleaning audits, current Southeast data and EDA. The browser-readable report is `reports/eda/major_outage_eda.html`, rebuilt by stage 14. Historical entries below describe prior states, not files that must remain present.

Integration follow-up (2026-10-23, user approved): import the EDA contribution into `Chumpz0/Capstone-Power-Outage-Prediction` on a review branch. The teammate remains free to implement the baseline under the approved safeguards. Preserve main's earlier notebooks as disabled historical source with outputs cleared. Use versioned, checksum-verified inputs beside legacy data; prepare a private Hugging Face upload without replacing existing files. Upload status is recorded by `eda_release.json` only after a successful publication.

## Open

| ID | Question | Options under consideration | Evidence | When |
|---|---|---|---|---|
| O-7 | Severity population | Provisional: covered windows (same population as classification). Alternative: conditional on major windows | label report section 6 | before freezing the protocol |
| O-8 | Counties with implausible outage peaks | Kept and reported as a subgroup; exclude or keep before freezing | `reports/label_audit/denominator_validation.csv` | before freezing the protocol |
| O-12 | Baseline and evaluation design, owned by the teammate | Declare baseline, metrics, calibration and success criteria before experiments; prior grids and criteria retired by D-25 | [evaluation_protocol.md](evaluation_protocol.md) | before new experiments |
| O-13 | Step 9 prerequisites | W&B and Hugging Face accounts, the existing GitHub repository, redistribution terms for EAGLE-I-derived tables | [data_sources.md](data_sources.md) | before step 9 |

## 2026-10-23: decisions after the first baselines (approved by the user)

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D-21 | Alert rows use a 72-hour outcome window `[t0, t0 + 72 h)` and non-overlapping 72-hour county windows; onset rows keep 24 hours. `cov90` becomes at least 90% of scheduled runs (260 of 288) with no gap over 1 hour. Supersedes D-10 for alert rows. | Of first major crossings within 96 h of a tropical or winter alert, 19-23% fall inside 24 h and 97% inside 72 h (`reports/forecast_feasibility/outcome_window_timing.csv`) | approved (closes O-10) |
| D-22 | The primary population for model selection and success criteria is covered windows: `cov90` coverage holds and the window is not ongoing, so majors and non-majors both needed coverage. Models are fit on the same population. | Prior development exposed asymmetric label selection; experimental outputs removed under D-25 | approved (closes O-11) |
| D-23 | Alert threshold: highest probability with validation recall at least 0.80. | Approved crew-staging recall requirement | approved (closes the threshold part of O-9; baseline protocol open under O-12) |
| D-24 | Storm-type scope stays storms only; heat and cold alerts and events are excluded. | `common.HAZARD_GROUPS`, `STORM_ALERT_FAMILIES` | approved (closes O-6) |

## 2026-10-23: checkpoints A and B (approved by the user)

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D-16 | Operational audience: utility operations and mutual-aid planners deciding where to pre-position crews 12-48 hours ahead. Missed major outages are the costly error. | [forecast_feasibility.md](forecast_feasibility.md) | approved (closes O-1) |
| D-17 | Prediction time: alert-triggered rows are primary. A row starts at the first storm-family NWS watch or warning issued for the county (greedy non-overlapping 24-hour county windows). NOAA-onset rows are kept as a labeled retrospective comparison. | forecast feasibility tables | approved (closes O-2) |
| D-18 | Major outage: at least 5,000 customers out at the same time in the 24-hour outcome window. Percentage views stay supporting analyses for counties with a validated denominator. | [label_feasibility_report.md](label_feasibility_report.md) section 7 | approved (closes O-3) |
| D-19 | Non-major requires the `cov90` coverage rule (at least 87 of 96 runs observed, no gap over 1 hour); everything else is unknown and excluded from training and evaluation. | label report sections 3-4 | approved (closes O-4) |
| D-20 | Windows already major at the prediction time are excluded from the primary target and reported separately. | label report section 5 | approved (closes O-5) |

## 2026-10-23: revised objective (approved by the project plan)

| ID | Decision | Status |
|---|---|---|
| D-08 | Predict the probability of a major outage; peak simultaneous customers out is the supporting objective. | approved |
| D-09 | Study area: AL, FL, GA, NC, SC, TN. County is the prediction unit; zone weather enters through historical zone-county mappings. | approved |
| D-10 | Keep the fixed 24-hour outcome window initially. | approved |
| D-11 | Customer counts are the primary severity measure. Retain customer totals where available and validate them before computing percentages; percentage analysis is optional until a denominator validates. | approved |
| D-12 | Train through 2023, validate on 2024, final evaluation on 2025. | approved (unchanged) |
| D-13 | Move tracking to W&B and data artifacts to Hugging Face after the dataset and task stabilize. | approved |
| D-14 | The national severity baseline is preliminary. The modeling dataset is rebuilt around the revised objective before results count as findings. | approved |
| D-15 | Three decisions stay open until checkpoints: lead time, definition of major, operational audience. | approved |

## 2026-10-23: engineering defaults made while implementing steps 1-5

| ID | Decision | Rationale | Status |
|---|---|---|---|
| E-01 | Build the six-state pipeline (stages 08-12) next to the national cleaning v3 outputs instead of rewriting them. | Keeps national preparation separate; raw data untouched. Preliminary models subsequently removed under D-25. | provisional |
| E-02 | Map zone events with the NWS correlation version in effect at onset, verified by zone and county names; fall back to the nearest verified version; reinterpret county-coded records as zones only on a verified zone name. | NWS reuses zone codes, NOAA lagged NWS code changes (Mobile, 2015 vs 2017), archived versions have gaps, and NOAA's 2025 Georgia winter-storm records use zone numbers as county codes. 99.5% of zone events verify at the effective version. | provisional |
| E-03 | No blanket Dec 31 exclusion in the Southeast pipeline; missing runs are unknown under the coverage rules. | 2014-2021 files simply end at Dec 31 00:00 UTC; 2022-2025 contain valid Dec 31 data (9.6k-23.6k Southeast rows per year), all of which cleaning v3 discarded. | provisional |
| E-04 | No exclusion by realized storm duration. | Duration is not known at prediction time; excluding on it selects on the outcome. Drought and non-storm types are excluded by type instead. | provisional |
| E-05 | Prediction rows are greedy, non-overlapping 24-hour county windows; later records in a window are `retro_` audit fields. | Prevents the same county outcome being counted once per overlapping NOAA record. | provisional |
| E-06 | Census county population (Vintage 2023) is the independent anchor for county lists, names, size groups and denominator checks. | MCC is invalid for North Carolina (state sum is 34% of EIA customers) and EAGLE-I's 2024 totals exceed EIA by 28-106% in FL, NC and SC. | provisional |
| E-07 | Build hashes cover source-file content, shared SQL macros, query text and upstream outputs. | Cleaning-v3 hashes ignored source content and macro definitions. | provisional |
| E-08 | The variable-window join moved to `pipeline/historical/`. | Its labels used realized duration and "no positive row" negatives. | provisional |
| E-09 | Provisional storm scope: convective wind, hail, lightning, rain/flood, coastal surge, tropical, synoptic wind, winter. | Matches the "given a storm" framing; heat and cold left for O-6. | provisional |
| E-10 | "Already major at `t0`" uses the last available run, or the latest observation in the prior hour when that run has no row. | One missing run should not hide an outage that is underway. | provisional |
| E-11 | Alert rows start at storm-family watches and warnings; advisories and other alerts enter only as features. | Watches and warnings are the products planners stage on; advisories would multiply low-risk rows. | provisional |
| E-12 | Test files hold features only; outcomes are computed once by the frozen final evaluation. | Makes accidental test-label use impossible during development. | provisional |
| E-13 | Previous baseline diagnostic requirements | Retired with previous experiments under D-25; covered-window population remains approved. | superseded |

## Earlier decisions

| ID | Date | Decision | Status |
|---|---|---|---|
| D-07 | before 2026-10-23 | Severity-only first baseline on the national county-coded onset cohort; occurrence modeling deferred (84 training / 41 validation confirmed negatives). | superseded by D-08; model code and results removed under D-25 |
| D-06 | before 2026-10-23 | Retrospective storm-onset prototype, county events, fixed 24 hours after onset. | retained as a comparison design pending O-2 |
| D-05 | before 2026-10-23 | Cleaning v3: drop `total_customers`; drop Dec 31 UTC observations; exclude Drought and events over 72 hours from join preparation. | retained for national v3 outputs; revised for the Southeast pipeline by E-03, E-04 and D-11 |
| D-04 | before 2026-10-23 | Missing outage observations are unknown; negatives need adequate observed coverage; severity is separate. | approved, unchanged |
| D-03 | before 2026-10-23 | Connecticut planning regions allocated to legacy counties by population. | not relevant to the study area |
