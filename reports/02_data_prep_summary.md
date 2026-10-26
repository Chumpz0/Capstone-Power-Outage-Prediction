# 02 – Data Preparation: Cleaned Outage + Storm Tables (2017–2025)

Notebook: [notebooks/02_data_prep.ipynb](../notebooks/02_data_prep.ipynb) · Check tables: [reports/prep_tables/](prep_tables/) · Outputs: `processed/` · Built 2026-10-25

---

## [GOAL]

Turn the raw EAGLE-I and NOAA Storm Events files into clean, consistent tables for EDA and modeling. The goal is still to predict the **probability that a US county loses power given a storm**.

Decisions applied (approved in step 01 follow-up):
- All US counties, years **2017–2025**: train 2017–2023, validation 2024, test 2025.
- Drop `total_customers`, the leakage columns and the four `TOR_OTHER_*` columns.
- Zero-outage rows: drop them and use an observation flag instead.
- Null readings: drop them and mark the gaps.
- Readings above the county's customer count: cap at MCC and flag.
- Also save an hourly outage table.

`data/` was only read. Every output is new.

---

## [FINDINGS]

### Outputs (`processed/`, 4.4 GB total)

| File | Grain | Rows | Size | Purpose |
|---|---|---|---|---|
| `outages_15min/year=*/` | county × 15-min snapshot (only when ≥1 customer out) | 208.1 M | 3.4 GB | Clean outage series |
| `outages_hourly/year=*/` | county × hour (worst snapshot in the hour) | 61.4 M | 0.9 GB | Fast table for EDA and labels |
| `county_observed_daily.parquet` | county × UTC day | 10.7 M | 26 MB | Was EAGLE-I watching this county that day? |
| `storm_events_clean.parquet` | storm event | 573 K | 21 MB | Storms without leakage columns, times in UTC |
| `storm_county_events.parquet` | storm event × affected county | 643 K | 16 MB | Starting table for EDA and labels |
| `counties.parquet` | county | 3,249 | 0.1 MB | County lookup: names, state, FEMA region, population, denominator |

### A. Outages, 15-minute
**Row waterfall (2017–2025):**
- 221.4 M raw rows
- minus 8.7 M null readings (2017–2021 and 2025)
- minus 4.8 M zero rows (2022–2025)
- = 207.8 M positive readings
- = **208.1 M clean rows** (splitting CT planning-region rows across counties adds ~8 K rows in 2025)

Full table: [outage_row_waterfall.csv](prep_tables/outage_row_waterfall.csv).

- **Timestamps:** 20 K off-grid readings were snapped to the 15-minute grid. Only 4 of them collided with an existing reading (kept the larger one).
- **Capped at MCC:** 5.6–27 K rows per year (~0.05%), with `over_mcc = True` and the original value kept in `customers_out_raw`. 2024 has the most capped rows (27 K).
- **Denominator:** 3,118 counties use MCC. 130 counties use an estimate of Census population × 0.508 customers per person (`denominator_source` says which). **Every outage row has a denominator.** The estimate covers three cases:
  - 9 counties that have no MCC value
  - **all 100 North Carolina counties.** NC's MCC total is 1.8 M customers, or 0.17 per person, while every other state falls between 0.35 and 0.60. That made NC's `pct_out` about 3× too high. *(Fix added during EDA: any state whose MCC customers per person is below half the national median now uses the estimate. Only NC triggers it.)*
  - **21 single counties whose MCC is far off** (below 0.1 or above 5 customers per person, provisional cut-offs). Examples: Buena Vista city, VA has 1 customer for 6,566 people; Waynesboro city, VA has 6 for 23 K; McDonald County, MO has 178 K for 24 K. These counties were showing an outage after 50–80% of storms. *(Fix added in step 04.)*
- **Lookup fix (step 04):** MCC's "Grand Total" row had leaked into `counties.parquet` as a fake county. It is now dropped.
- **Connecticut:** EAGLE-I switched CT from 8 counties to 9 planning regions on **2025-05-29**, with a ~4-day overlap (to Jun 2). Planning-region rows were converted back to the 8 counties with population weights. CT customer-hours out: 3.96 M (2023), 4.01 M (2024), **4.91 M (2025)**. The 4-day overlap may double count a little, and that may explain part of the 2025 increase.
- **US Virgin Islands:** the two state spellings are merged into one.

### B. Outages, hourly
- One row per county per hour with an outage: worst customers out and % out in the hour, plus the number of snapshots that had an outage (1–4).
- From 4.5 M rows (2017) to 8.0 M (2025). Rows grow every year because EAGLE-I covers more counties and tracks more customers. **Raw yearly counts are not directly comparable.**

### C. County observation flag (daily)
`observed = TRUE` when all of these hold:
- the county has any EAGLE-I row in that year
- EAGLE-I collected ≥90% of that day's snapshots nationally
- the county's readings that day are not all null

| Year | % county-days observed | County-days with all readings null | National gap days |
|---|---|---|---|
| 2017 | 86.6 | 6.7 K | 7 |
| 2018 | 82.5 | 63 K | 6 |
| 2019–2021 | 85.9–87.6 | 59–100 K | 1–3 |
| 2022–2024 | 94.0–94.1 | 0 | 0–1 |
| 2025 (test) | 90.7 | 41 K | 1 |

- **Dec 31 is a gap day in 2017–2021** (the raw data stops at Dec 31 00:00). Other gaps include 2017-04-01/02, 2017-07-07, 2018-05-25 and 2024-05-24.
- **Each year, 180–360 counties are never reported at all.** These are mostly Alaska, the Pacific territories and parts of the Plains and Mountain states (KS, NE, ND, MT, CO, MN). This list fits the low EAGLE-I coverage of those states seen in the audit.
- **Quality columns:** state coverage share (`pct_customers_tracked`) and FEMA-region DQI score (`dqi`) are attached to each row. They only exist for 2018–2022, so 2017 uses 2018 values and 2023–2025 use 2022 values (`quality_ref_year` shows which).
- **Train vs test:** the training years (2017–2021) have a lower observed rate (~86%) than 2022–2025 (~94%). Most of the difference comes from county-days where every reading is null.

### D. Storm Events, clean
- **573 K events** after dropping 25 K marine and waterspout events.
- **33 columns dropped:**
  - leakage: damage, injuries, deaths, narratives, `SOURCE`, tornado F-scale, length and width
  - mostly empty: the four `TOR_OTHER_*`
  - replaced by UTC timestamps: the separate date and time parts
  - location detail that isn't needed: range, azimuth and location text
  - constant: `DATA_SOURCE`
- **Kept:**
  - identifiers: event and episode id
  - storm: event type, UTC begin and end, duration, magnitude and its type, hurricane category, flood cause
  - location: begin/end lat/lon, WFO, original zone or county code
- **Territory codes fixed:** NOAA uses its own state codes for territories (96 VI, 97 AS, 98 GU, 99 PR). These were converted to Census codes (78, 60, 66, 72) so Puerto Rico and the Virgin Islands join correctly.

### E. Storm × county
- **643 K event-county rows** (61–81 K per year). 40–57% come from zone-coded events expanded with the dated crosswalk.
- **21–30% of rows are `partial_county`**: the zone covers only part of the county.
- **Only ~590 events across all years map to no county.** Most are winter storm, drought and high wind events in zones missing from the crosswalk.
- **0.1–0.2% of rows have a FIPS code not in the county lookup** (Pacific territories, and retired or renamed Alaska and Connecticut codes). They stay in the table with `fips_known = False`.
- **The county was observed by EAGLE-I on the storm's start day for 87–92% of rows in 2017–2021 and 95–97% in 2022–2025.** (2024 is the lowest of the later years at 94.6%.)

### Output checks: all passed
[output_checks.csv](prep_tables/output_checks.csv):
- **Uniqueness:** no duplicate county + time rows in the 15-minute, hourly or daily tables; no duplicate event ids; no duplicate event + county rows.
- **Value ranges:** no outage values ≤ 0, no `pct_out` above 1, and no timestamps off the 15-minute grid.
- **Codes and times:** every outage FIPS is in the county lookup, no CT planning-region codes remain, and no storm ends before it begins.

Two bugs were caught by these checks during the build and fixed before the final run: the Virgin Islands counties were duplicated, and NOAA's territory codes didn't match Census codes.

---

## [NEXT STEPS]

**EDA (step 03), built on these tables:**
1. **Outage behaviour:**
   - distribution of peak % out and outage duration
   - how outages vary by season and time of day
   - the biggest outage events, and whether each lines up with a storm
2. **Storms → outages:**
   - per event type and magnitude: the share of storm-county rows followed by an outage (several % thresholds and time windows)
   - the same comparison for zone-coded and partial-county rows vs county-coded rows
3. **Geography:** outage rate per storm by county and state on maps. Compare EAGLE-I coverage (observed rate, DQI) with outage rates to see where low coverage hides outages.
4. **Over time:** check whether outage rates drift between 2017–2023, 2024 and 2025 because coverage grew. That drift would affect validation and test results.
5. **Outages with no matching storm:** find large outages with no storm nearby, which hints at missing storm records or non-weather causes.

**Decisions EDA should inform (yours):**
1. **Target definition:** the % of customers out that counts as "losing power", and the time window after storm start (e.g. ≥1% within 24 h).
2. **Prediction unit:** county × storm event (the current table) or county × day.
3. **Overlapping storms:** whether to merge storm events that hit the same county close together into one episode.
4. **Unobserved and partial counties:** whether to drop or down-weight rows where the county wasn't observed, or where the zone covers only part of the county.

**Housekeeping:**
- `work/audit.duckdb` (2.7 GB) is still used as the raw-data cache.
- `processed/` can be fully rebuilt by running the notebook (~15 min).
