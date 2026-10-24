# EDA metadata and calculation contract

The machine-readable [metadata manifest](eda_metadata.json) records source file sizes and SHA-256 hashes, every derived Parquet schema and stored null count, artifact hashes, the Python/package versions used, and the base Git commit. The base commit identifies the starting revision; artifact hashes identify the files used in the run. [Validation results](eda_validation.json) identify executed notebooks. The manifest excludes itself to avoid a circular hash.

## Source datasets

[The source field dictionary](data_dictionary.md) gives every stored column, type and null count. [The source audit](data_audit.json) covers the original processed inputs only; generated `data/processed/eda/` outputs are inventoried separately in the manifest. Raw download inventory is retained in the source audit. No source files were replaced or refreshed.

| Dataset | Grain / intended key | Units and interpretation |
| --- | --- | --- |
| EAGLE-I annual outages | County and `run_start_time` | Customers currently reported out; naive timestamps represent UTC. FIPS is a five-character string. Missing rows and null counts are not zero. |
| NOAA annual events | `event_id` | Historical event reports; episode IDs can contain multiple reports. Event time is not publication time. Original local and converted UTC times are retained. |
| Original storm county mapping | `event_id, fips_code` | Event-to-county attribution, including source mapping defects; not a forecast observation frame. |
| Census | `fips_code, acs_year` | Population and housing counts, square-mile land area, densities per square mile. Estimate year does not prove release availability. |
| Zone lookup | `state, zone, fips_code` | One local lookup vintage, not a historical boundary series. |
| MCC | County FIPS | Modeled customer denominator; not observed outage count or verified capacity. |
| Coverage history | State/year | Broad customer-coverage summaries, not quarter-hour observation completeness. |
| DQI | FEMA region/year | Regional quality indicators, not county observation eligibility. |

## Derived table inventory and keys

All paths below are relative to ignored `data/processed/eda/`. Detailed schemas/types/null counts are generated, not hand-maintained, in `eda_metadata.json`.

| Path | Grain / key | Content |
| --- | --- | --- |
| `observation_reliability/county_year.parquet` | Year, horizon, county | Full-frame eligibility and quarter-hour quality counts; county name, housing and within-year housing quartile |
| `observation_reliability/state_month.parquet` | Year, state, horizon, origin month | Same quality measures aggregated across counties |
| `observation_reliability/state_hour.parquet` | Year, state, horizon, origin UTC hour | Same quality measures aggregated across counties |
| `observation_reliability/slot_distribution.parquet` | Year, state, horizon, usable-slot count | Number of in-bounds windows at each observed-slot count |
| `observation_reliability/raw_county_year.parquet` | Year, county | Source row/flag counts and first/last recorded timestamps; counts each original row once |
| `customer_targets/county_year.parquet` | Year, horizon, county, threshold | Target-label and sensitivity counts; descriptive state/name/housing quartile |
| `customer_targets/state_season.parquet` | Year, state, horizon, threshold, season | Target counts aggregated by UTC origin season |
| `geography_criteria/mapping_overlay.parquet` | Event ID, original FIPS | Supported correction and evidence; original report fields retained |
| `geography_criteria/unmatched_events.parquet` | Event ID | Unresolved historical zone report with explicit disposition |
| `geography_criteria/audited_event_counties.parquet` | Event ID, corrected FIPS | Corrected ID mapping with original FIPS/method; unmapped reports remain outside this table |
| `geography_criteria/storm_event_counties.parquet` | Event ID, corrected FIPS | Full original event attributes joined to audited IDs for notebook 03; includes original FIPS/method and correction disposition |

## Observation fields

`year` is the decision-origin year, `horizon` is 6 or 12 hours, and `origins` counts all scheduled county-hours. `boundary_excluded` counts origins before 2014-11-01 04:00 UTC or with endpoints at/after 2024-01-01 for training analysis. `in_bounds = origins - boundary_excluded`. Other gaps stay in the denominator.

`unknown` has no usable slots; `partial` has 1 through 23/47; `complete` has all 24/48. `complete_unflagged` is complete without duplicate, invalid, off-grid or above-proxy readings. `any_observed`, `at_least_half` and `at_least_90pct` use at least 1, 50% and ceiling(90% of expected) usable slots. `expected_slots` is in-bounds windows times 24/48. `flagged_windows` counts windows with any anomaly flag. These are counts, not rates; displayed rates specify their denominator.

For each quality family (`usable`, `present`, `null`, `invalid`, `duplicate`, `zero`, `proxy_flag`, `off_grid`), `*_window_slots` sums its counts over overlapping windows and `*_windows` counts windows containing at least one. `present` includes null/invalid stored exact-grid slots; `usable` requires a single finite nonnegative integer count. `duplicate` counts distinct duplicated grid slots, not duplicate excess rows. `invalid` excludes nulls but includes nonfinite, negative or fractional numeric values. `zero` requires a usable explicit zero. `proxy_flag` means a valid count above 1.2 times housing; it is not a verified error. `off_grid` counts timestamped records outside the exact quarter-hour grid, including seconds; despite the generic suffix, these are records, not usable slots.

Raw-table `rows`, `null`, `zero`, `invalid`, `off_grid`, `duplicate_rows` and `proxy_flags` describe original readings once. Raw `duplicate_rows` counts excess duplicate timestamps. `first` and `last` are stored string timestamp summaries, with `NaT` for never-observed county-years; they are not availability guarantees. Window sums deliberately repeat readings and must not be presented as unique observations.

## Target fields and null policy

Reusable per-origin arrays from `src/outage_targets.py` are not materialized as a full model dataset. `recorded_peak` is the uncapped maximum of unique valid exact-grid counts in `(t,t+h]`. `median_peak` is the maximum of medians in `(hour,hour+1h]`. `excluded_peak` removes above-proxy readings; `excluded_usable` counts remaining valid slots. The starting year uses `housing_units`. Readings switch to the next housing estimate at January 1, regardless of chunk length; when no next-year estimate is supplied, the current estimate is reused. Empty or out-of-envelope peaks are NaN. Partial peaks are observed lower bounds; complete recorded peaks are still not guaranteed continuous-time true maxima.

`labels()` returns 1 for an observed strict exceedance, 0 only for a complete nonexceeding window, and NaN otherwise. Flags remain separate; calling `labels()` does not apply the strict cohort by itself. `exclude_proxy=True` also requires retained completeness for a negative. Preserve `in_bounds`, `usable` and `flagged` alongside returned peaks/labels.

In target summaries:

- `complete` includes flagged windows; `complete_positive` counts their recorded exceedances.
- `strict` means complete and unflagged; `strict_positive + strict_negative = strict`.
- `partial_positive_unflagged` and `partial_positive_flagged` retain separate observed evidence; `partial_uncertain` is partial with no recorded exceedance. `unknown` has no usable slots.
- `median_positive_strict` and `median_lost_positive_strict` compare medians on the same strict cohort.
- `proxy_excluded_positive_complete`, `proxy_lost_positive_complete` and `proxy_uncertain_complete` compare removal of above-proxy readings on the original complete cohort. A lost positive is not automatically a reliable negative.

`threshold` is customers, always compared with strict `>`. Repeated threshold rows share the same cohort; do not sum cohorts across thresholds. Horizons also overlap and must not be summed as distinct decisions. Missing summary-group rates are undefined when denominators are zero. Housing quartiles are recalculated over all 534 counties within each year. Seasons DJF/MAM/JJA/SON use the origin month, not the endpoint month. Stored counts are integer; displayed percentages are rounded. Quantiles use NumPy's default linear interpolation.

## Geography provenance and availability

`original_cz_type`, `original_cz_fips`, `original_cz_name`, `original_fips` and `original_fips_valid` describe the source. `corrected_fips`, `disposition` and `evidence` describe a supported override. Unchanged rows have null correction-specific fields; this does not mean their geography is missing. `original_mapped_from` preserves county/zone origin; corrected rows use `mapped_from = zone_evidence_override`. A unique local lookup match is evidence for the documented correction, not proof of historical validity for all zone boundaries.

All outcome quality, same-year proxy estimates and geography corrections are retrospective audit information. Consult [feature availability](feature_availability.md) before using any field as a predictor. The [criteria document](modeling_criteria.md) defines exclusive split boundaries and evaluation denominators. Full descriptive notebook reruns include already-exposed holdout years; no holdout outcome analysis is used to change the training target definition.
