# Storm-to-Outage: County Power Outage Dataset

This dataset links EAGLE-I county power-outage observations with NOAA Storm Events to study the probability of a substantial county outage during a recorded storm episode. It contains 271,644 county-by-episode records for the lower 48 states and Washington, DC, spanning 2017-2025. A positive label means at least 5% of the county's estimated customers were without power during the defined 24-hour window.

The dataset supports the DSCI 4350 undergraduate capstone at Georgia State University, Fall 2026. It is intended for retrospective analysis and model comparison, with utility crew staging as a motivating application. Whole-episode storm summaries stand in for forecasts; this is not a validated operational forecast dataset. Labels indicate outages coinciding with storms, not proven storm causation.

**Dataset:** [DSCI-Capstone/county-outage-data](https://huggingface.co/datasets/DSCI-Capstone/county-outage-data)

**Code:** [Capstone-Power-Outage-Prediction](https://github.com/Chumpz0/Capstone-Power-Outage-Prediction)

**Card reviewed:** 2026-10-27

**Authors and contact:** To be supplied by the capstone team; see open items below.

Adapted from the [PAIR Data Cards Playbook extended template](https://github.com/PAIR-code/datacardsplaybook/blob/main/templates/DataCardsExtendedTemplate.md). Relevant sections are condensed for this capstone. Publishers, funding, retention, and wipeout/deletion procedures are omitted. This card documents the current implementation; it does not imply a maintenance or deployment commitment.

## Dataset Overview

### Content and snapshot

The primary release described here is `processed/model_table/`, not the much larger raw and intermediate tables. Records describe weather, places, and electricity-system behavior. The data modality is tabular, with temporal and geographic identifiers.

| Property | Current dataset |
|---|---|
| Observation unit | One county FIPS code paired with one NOAA `episode_id` |
| Geographic scope | Lower 48 states and DC; 49 state/DC codes in each split |
| Time coverage | 2017-2025, split by the UTC year of the first episode report in each county |
| Instances | 271,644 across train, validation, and test |
| Candidate model inputs | 36: 25 storm, 2 time, 4 county, 2 pre-storm, 3 county-history features |
| Training/validation schema | 46 columns: 6 identifiers, 36 features, 1 target, 3 retrospective fields |
| Test feature schema | 42 columns: identifiers and features only |
| Classes | Binary: 0 = below the threshold; 1 = threshold reached |
| Label origin | Computed from outage observations; no capstone human-labeling workforce |
| Local file sizes | Train 9,686,396 bytes; validation 1,601,759; test features 1,304,975 |

Sizes describe the local Parquet snapshot inspected on the review date. The separate test-label file was not opened to prepare this card.

### Split statistics

| Split | Years | Rows | Counties | Episodes | Positive labels | Positive rate |
|---|---|---:|---:|---:|---:|---:|
| Train | 2017-2023 | 208,482 | 3,073 | 54,580 | 29,454 | 14.13% |
| Validation | 2024 | 31,064 | 3,023 | 7,734 | 5,120 | 16.48% |
| Test features | 2025 | 32,098 | 3,014 | 7,866 | Withheld | Withheld |

Source: [dataset overview](ml_eda_tables/dataset_overview.csv), checked against the local training, validation, and test-feature files. Episode identifiers do not overlap between these three local splits. Counties recur across years, so this evaluates future-year performance rather than generalization to entirely unseen counties.

### Descriptive statistics

Statistics below use training rows only. Numeric summaries exclude missing values; missingness is a percentage of all training rows.

| Field | Mean | Median | 25th-75th percentile | Maximum | Missing |
|---|---:|---:|---:|---:|---:|
| Reports in county-episode (`n_events`) | 1.88 | 1 | 1-2 | 83 | 0% |
| Maximum wind (`max_wind_kts`, knots) | 54.03 | 52 | 50-56 | 173 | 59.05% |
| Maximum hail (`max_hail_in`, inches) | 1.29 | 1 | 1-1.5 | 6.42 | 87.19% |
| County episode duration (hours) | 19.08 | 4 | 0.13-15 | 743.98 | 0% |
| Historical outage rate (`hist_outage_rate`) | 0.1343 | 0.1151 | 0.0641-0.1847 | 0.5832 | 0% |
| Peak share out (`retro_peak_24h`, analysis only) | 0.0335 | 0.0019 | 0.00005-0.0182 | 1 | 0% |

See the [full feature dictionary and statistics](ml_eda_tables/feature_dictionary.csv) for all 46 fields, data types, units, and missingness by split. Shares are fractions from 0 to 1, not percentages. `fips` must remain text to preserve leading zeros; Connecticut is harmonized to its eight legacy counties rather than using a uniform Census 2023 geography.

### Version and maintenance

The existing summaries record the dataset build as 2026-10-25. The evaluation module records Hugging Face revision `5c01325258aa922bf395da1e39acda7c234fefcf`. That is the recorded data reference, not a claim that the current remote HEAD or every local file was independently verified against it. Some notebook fallback paths read the Hub without a revision pin.

No formal dataset release number, future refresh schedule, or post-capstone maintenance commitment is documented. Reproduction uses the notebooks and [pinned requirements](../requirements.txt); release naming and source-version confirmation remain open items.

## Example of Data Points

These are selected fields from two actual training rows, chosen as examples rather than a statistically representative sample. Timestamps are UTC. Rounded fractions are shown for readability.

| Field | One-report negative example | Already-out positive example |
|---|---|---|
| `fips` | `53055` | `28041` |
| `episode_id` | 112713 | 112815 |
| `t0` | 2017-01-02 01:15 | 2017-01-02 16:47 |
| `n_events` | 1 | 1 |
| `max_wind_kts` | 35 | 52 |
| `pre_peak_24h` | 0 | 0.054521 |
| `pct_out_at_start` | 0 | 0.054521 |
| `target` | 0 | 1 |
| `retro_peak_24h` | 0 | 0.051482 |

The second row remains positive even though the peak during the label window is below the pre-storm level: the target requires reaching 5%, not an increase caused by the storm. `target` and all `retro_*` columns must be excluded from model inputs.

### Field groups

| Group | Fields and meaning |
|---|---|
| Identifiers | `fips`, `episode_id`, `state_abbr`, `fema_region`, `year`, `t0`; keys, grouping, and temporal split metadata |
| Storm composition | `n_events`; counts for thunderstorm wind, hail, tornado, high wind, flood, winter, ice storm, tropical, cold, lightning, wildfire, and other reports |
| Storm location and intensity | County/zone report counts, zone-coded share, partial-county flag, maximum wind/hail/hurricane category |
| Storm extent | County episode duration; episode counties, log counties, states, and distinct events |
| Time and county | Month, approximate local standard-time hour, log10 customers, log10 population, DQI, customer coverage share |
| Pre-storm | `pre_peak_24h`: peak in the preceding 24 hourly bins; `pct_out_at_start`: share out in the preceding hour |
| History | Number of eligible county-episodes, smoothed outage rate, and smoothed mean peak share |
| Outcome and retrospective analysis | `target`, `retro_peak_24h`, `retro_peak_customers_out`, `retro_hour_of_peak` |

The dictionary's `model_input` designation identifies candidate features in the modeling table, not a guarantee that each is available at forecast time or used by every fitted model.

## Motivations & Intentions

The research question is: **given a storm episode, how well can we rank counties by their probability of experiencing a substantial outage?** The project combines outage time series with storm reports to support reproducible feature engineering, evaluation, and comparison with simple historical rules.

Suitable uses include undergraduate research, retrospective storm-outage analysis, temporal model evaluation, and comparisons of ranking, probability accuracy, and illustrative prioritization budgets.

Unsuitable uses without further validation include live crew dispatch, household or utility-circuit risk estimates, forecasts for excluded regions, claims that storms caused individual outages, and prediction of all outages regardless of weather. Actual forecast-time use requires timestamped forecast or warning inputs, a feature-availability review, and utility-specific evaluation.

## Access

The project README describes the Hugging Face dataset as private. Access requires permission for `DSCI-Capstone/county-outage-data` and a Hugging Face login. Code, reports, and small CSV evidence tables are kept in this repository; raw and processed datasets are kept outside Git.

For model development, download only the following files after `hf auth login`:

```bash
hf download DSCI-Capstone/county-outage-data --repo-type dataset --revision 5c01325258aa922bf395da1e39acda7c234fefcf --local-dir . --include "processed/model_table/train.parquet" "processed/model_table/validation.parquet" "processed/model_table/test_features.parquet"
```

The Hub stores test labels in `holdout/test_labels.parquet`; notebook 04 writes a local copy under `processed/model_table/`. Reserve those labels for the selected model's final evaluation. Separate files support the evaluation protocol but are not evidence of access-control enforcement.

Dataset licensing and redistribution terms have not been established in this card. Public source availability is not a substitute for documenting the applicable terms; see open items.

## Provenance

### Sources and collection

The capstone integrates existing datasets rather than collecting new observations from people.

| Source | Role | Repository evidence |
|---|---|---|
| [EAGLE-I outage archive](https://figshare.com/articles/dataset/24237376) | County outage readings on an approximately 15-minute grid; combined source files cover 2014-2025 | [Data audit](01_data_audit_summary.md) |
| [NOAA Storm Events](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) | Reported weather-event types, timing, locations, intensity, and event/episode identifiers | [Data audit](01_data_audit_summary.md) |
| Census Vintage 2023 county population | Population feature and fallback customer denominator | `data/01_raw/census_counties/co-est2023-alldata.csv` |
| EAGLE-I MCC, DQI, coverage-history references | Modeled customer denominator, regional quality score, state customer coverage | `data/reference/eaglei_archive/` |
| Dated NWS zone-to-county crosswalk | Expands forecast-zone-coded reports into county records | `data/reference/nws_zone_county_versions.parquet` |
| Connecticut region-to-county weights | Maps planning-region outage records back to legacy counties | `data/reference/ct_region_to_county_weights.csv` |

The exact upstream release identifiers, download dates, and provenance of the supporting crosswalk/weight files need a completed source manifest. The pipeline records local paths and a hosted dataset revision, but these do not fully establish each upstream source version.

### Selection and integration

The raw audit covers 2014-2025; processing and modeling use 2017-2025. Earlier EAGLE-I years have poorer county coverage, and 2014 contains only November-December. The modeling population consists of eligible reported storm episodes with sufficient inferred observation coverage, not a random sample of all county-days.

| Modeling-table step | Rows remaining |
|---|---:|
| Prepared event-by-county records | 643,757 |
| Lower 48 and DC | 629,256 |
| Exclude Drought, Heat, and Excessive Heat triggers | 558,402 |
| Aggregate to county-by-episode | 298,398 |
| Require observation coverage for label-window days | 271,644 |

Marine/waterspout events are removed during preparation. Alaska, Hawaii, and territories are outside the modeling scope. Source: [row waterfall](model_table_tables/row_waterfall.csv) and [modeling-table notebook](../notebooks/04_modeling_table.ipynb).

## Transformations

The preparation and modeling notebooks use DuckDB, pandas, NumPy, and Parquet storage. The main operations are:

1. **Clean outage readings.** Parse identifiers and timestamps, snap off-grid readings to the 15-minute grid, resolve collisions by retaining the larger reading, remove null/zero readings from the positive-outage table, and preserve observation information separately. Cap readings above the denominator and retain raw counts and an over-denominator flag in the cleaned table.
2. **Construct customer denominators.** Use MCC when suitable; otherwise use Census population times the median MCC-to-population ratio, approximately 0.508. The preparation summary reports 130 fallbacks: 9 missing-MCC counties, all 100 North Carolina counties, and 21 other implausible county estimates. The state rule uses half the national median; county cutoffs of 0.1 and 5 customers per person are provisional heuristics.
3. **Harmonize geography and time.** Normalize territory/state codes, convert NOAA local standard times to UTC, expand zone events with the dated crosswalk, and allocate Connecticut planning-region outages to legacy counties using population weights. The model's `local_hour` uses the state's main standard-time zone, an approximation for states spanning time zones.
4. **Build hourly and observation tables.** Hourly outage values use the maximum snapshot in each hour. A county-day is considered observed when the county appears somewhere in that year, at least 90% of the day's national snapshots exist, and that county-day does not consist entirely of null readings. This is a coverage heuristic, not proof of continuous county monitoring.
5. **Aggregate storms and join outages.** Collapse eligible reports to county-by-episode, set `t0` to the first report, and summarize the whole episode. Require observation on the UTC dates containing label hours 0 and 23. Missing hourly outage records are then interpreted as zero under that observation assumption.
6. **Build features.** Add log transforms, storm-family counts, quality references, pre-storm outage features, and county history. Quality references exist for 2018-2022: 2017 uses 2018, while 2023-2025 use 2022. Pre-storm features become missing when the preceding day is not observed.
7. **Separate outcomes.** Exclude damage, casualties, narratives, source descriptions, and tornado damage/path fields from model inputs. Keep the three `retro_*` fields for analysis only and place test outcomes in a separate file.

There is no documented PCA or other learned dimensionality reduction. Raw numeric feature missingness is retained in the modeling table; any model-specific imputation or scaling must be fit using training data only. See [preparation summary](02_data_prep_summary.md) and [notebook 02](../notebooks/02_data_prep.ipynb).

### County history: exact construction and caveat

For validation/test, history uses 2017-2023 labels. For a training row, county-specific counts and sums exclude that row's entire year. Outage rates are smoothed as `(positive_count + 20 * p0) / (episode_count + 20)`; mean peak uses the corresponding formula. The prior weight of 20 is provisional.

The global priors `p0` and `peak0` are computed from all training years, including the row's year. Other training years can also be later than the row's date. Thus, this is leave-one-year-out county aggregation with a full-training prior, not a strictly historical, as-of-time feature pipeline. The saved history check is marked true by construction rather than independently testing exclusion of every own-year label contribution.

## Annotations & Labeling

Labels are assigned algorithmically:

```text
start_hour = floor(t0 to the UTC hour)
peak_share = maximum hourly outage share in hours 0 through 23 from start_hour
target = 1 if peak_share >= 0.05, otherwise 0
```

The 24-hour interval is aligned to the start hour, not the exact minute of `t0`; it can include observations before the report within that first hour. The denominator is estimated customers, not necessarily a measured customer total at the event date.

The 5% threshold and 24-hour window are project choices informed by training EDA: the earlier analysis found stronger storm-versus-background separation at 5% and 83% of examined peaks within 24 hours. They are not utility-approved service standards. Already-out counties remain eligible, and a negative label means the threshold was not reached under the observation assumptions, not that no customer lost power. See [EDA summary](03_eda_summary.md).

## Validation Types

Saved automated checks cover unique keys, required targets, bounded outage shares, geographic scope, temporal splits, absence of retrospective inputs, and agreement between target and peak-share threshold. Notebook 06 also checks that the feature dictionary covers the columns and that the test-feature file contains no target. Results are in [preparation checks](prep_tables/output_checks.csv), [model-table checks](model_table_tables/checks.csv), and [ML EDA checks](ml_eda_tables/checks.csv).

For this card, local schema sizes, split counts, training/validation positives, examples, and disjoint episode identifiers were checked directly. No test labels were read. These checks establish structural consistency; they do not independently validate outage completeness, storm causality, or real-time feature availability. No utility/operator validation or independent labeling review is documented.

## Extended Use and Sampling Methods

Preserve the time-based split when comparing models. Rows from the same storm episode are dependent; use episode-level resampling for uncertainty estimates and avoid treating county rows as independent storm samples. Rebuild history features from the permitted training portion if changing splits or running cross-validation. Do not reuse the existing history columns unchanged in a different validation design.

Additional data should join through checked county geography and timestamps. Forecasts, warnings, infrastructure, or land-cover features would require their own provenance and availability checks. Do not join retrospective damage, future outage aggregates, or test labels into predictors.

The current dataset uses all retained eligible rows rather than a documented class-balancing sample. Any subset or resampling should record its selection rule and preserve the natural prevalence in evaluation data. The top-10% scoring budget is an illustrative prioritization scenario, not a measured crew-capacity constraint.

## Known Applications & Benchmarks

The documented application is this capstone. Saved 2024 validation results are:

| Predictor | Average precision (reported as PR-AUC) | Brier score | Brier skill | Recall at top 10% |
|---|---:|---:|---:|---:|
| Storm family rate x county history reference | 0.469056 | 0.111560 | 0.189567 | 32.58% |
| Initial four-feature logistic regression | 0.363576 | 0.125422 | 0.088865 | 24.24% |

Sources: [reference rules](baseline_tables/naive_rules_validation.csv) and [initial logistic regression](baseline_tables/logistic_regression_initial_validation.csv). The latter uses month, local hour, pre-storm peak, and preceding-hour outage share; it is not a result for all 36 candidate features.

The [success criteria](success_criteria.md), revised on 2026-10-27 after initial model results existed, require improved ranking over the reference and no worse Brier score or top-10% recall. Calibration, subgroup results, and episode-bootstrap uncertainty are reported separately. These are validation benchmarks, not final test performance or evidence of operational readiness.

## Reflections on Data

### Coverage, representativeness, and missingness

EAGLE-I coverage varies by county and year. A missing record can mean zero outages or missing observation; the coverage heuristic cannot fully resolve that ambiguity. Coverage improves over time, while the training positive rate rises from 11.15% in 2017 to 16.85% in 2023. These changes can reflect measurement differences as well as weather and grid conditions. A nationwide geographic scope does not imply uniform county representation.

Wind, hail, and hurricane-category missingness is largely conditional on event type. Hurricane category is 99.92% missing in training and entirely missing in the test features. Filling these values with zero would conflate unknown intensity with an observed zero.

### Forecast-time limitations and dependence

Whole-episode maxima, duration, counts, and extent can include information observed after `t0`. They stand in for forecast descriptions, so retrospective model performance may overstate advance-warning performance. Population and quality references are also not uniformly as-of-event measurements. Removing `retro_*` alone does not establish absence of temporal leakage.

### Measurement choices and uneven effects

Zone-coded reports can cover only part of a county; expansion can associate an entire county with localized weather. Connecticut population-weighted allocations and overlapping legacy/planning-region records can distort outage totals. MCC repairs, capping, and the 5% threshold affect labels, particularly in small counties. These assumptions warrant sensitivity analysis before stronger conclusions.

### Drift and subgroup performance

The saved drift table shows large DQI shifts (PSI about 3.69 for validation and 3.75 for test), consistent with its year mapping. Hurricane category has an additional sparse-data shift and should not be treated as stable. Tropical episodes are rare in the test features. Training outage rates differ substantially by storm family: about 62.5% for tropical episodes versus 7.7% for winter episodes. Families overlap, so these are not mutually exclusive classes. See [drift](ml_eda_tables/feature_drift_psi.csv) and [family rates](ml_eda_tables/rate_by_storm_family.csv).

### Sensitivity and human attributes

The modeling schema contains county aggregates and storm descriptors, with no individual names, customer addresses, or demographic attributes such as race or income. Population is an aggregate exposure feature. This is a schema description, not a formal privacy audit of every raw source. Geographic and coverage differences can still produce unequal model errors; no demographic fairness assessment or privacy guarantee is claimed.

## Terms of Art

| Term | Meaning in this project |
|---|---|
| FIPS | Five-digit county identifier; stored as text |
| Episode | NOAA grouping of related storm reports; may affect multiple counties |
| MCC | EAGLE-I modeled customer count used as the default denominator |
| DQI | EAGLE-I regional data-quality index |
| Observed | Project heuristic that a county-day has sufficient observation coverage |
| PR-AUC | Here, scikit-learn average precision, not trapezoidal PR integration |
| Brier skill | `1 - Brier / (p * (1-p))`, using observed prevalence `p` in the evaluation split |
| PSI | Population stability index, used here to compare feature distributions |

## Open Items for the Capstone Team

- [ ] Add data-card authors, contributor roles, and a preferred project contact.
- [ ] Confirm the intended release/version name and whether the recorded Hugging Face revision is the final deliverable; record file hashes or a release manifest.
- [ ] Record exact upstream release IDs, download dates, and origins of the NWS crosswalk and Connecticut allocation weights.
- [ ] Confirm source licenses/attribution requirements and the license and redistribution permissions for the derived dataset.
- [ ] Supply a preferred project citation and any instructor-required acknowledgments.
- [ ] State whether the dataset will remain private and whether any maintenance is planned after the course.

Publishers, funding, retention, deletion policies, and human-annotator demographics are intentionally not left as mandatory blanks. They can be added only if the project's requirements change.
