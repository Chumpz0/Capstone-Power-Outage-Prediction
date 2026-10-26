# Dataset Card: Storm-to-Outage County Episodes (ML-ready)

| | |
|---|---|
| **Name** | Storm-to-Outage County Episodes, 2017–2025 |
| **Version** | Hugging Face dataset commit `5c01325` · GitHub commit `02c4471` (notebook 04) · built 2026-10-25 |
| **Hosted at** | [huggingface.co/datasets/DSCI-Capstone/county-outage-data](https://huggingface.co/datasets/DSCI-Capstone/county-outage-data) (private; access through the `DSCI-Capstone` org) |
| **Code** | [github.com/Chumpz0/Capstone-Power-Outage-Prediction](https://github.com/Chumpz0/Capstone-Power-Outage-Prediction), notebooks 01–04 build it, 06 profiles it |
| **Maintainer** | Christopher Walker, DSCI 4350 capstone team (Georgia State University, Fall 2026) |
| **License** | CC BY 4.0 (inherits EAGLE-I's license; NOAA, NWS and Census inputs are public domain) |

---

## 1. Summary

One row per **US county hit by one NOAA storm episode**, 2017–2025, lower 48 states + DC. Each row describes the storm in that county, the county itself, outage conditions just before the storm, and the county's past outage history. The binary **target** says whether at least **5% of the county's electricity customers lost power within 24 hours** of the storm's first report there.

**Intended use:** train and evaluate models of *P(county outage | storm)* for **utility crew staging**, i.e. ranking which counties to pre-position crews in when a storm system arrives.

**Not intended for:** predicting outages without a NOAA storm record (most outages, see §7), counting customers affected, or real-time use without swapping the observed storm summary for a forecast.

## 2. Structure

| File (HF path) | Rows | Columns | Years | Positive rate |
|---|---|---|---|---|
| `processed/model_table/train.parquet` | 208,482 | 46 | 2017–2023 | 14.1% |
| `processed/model_table/validation.parquet` | 31,064 | 46 | 2024 | 16.5% |
| `processed/model_table/test_features.parquet` | 32,098 | 42 | 2025 | not shown |
| `holdout/test_labels.parquet` | 32,098 | 6 | 2025 | **held out: open once, for the final evaluation** |

- **Columns:** 6 identifiers, **36 model features**, `target`, and 3 `retro_*` columns that are only known after the storm (analysis only, **never model inputs**).
- **Keys:** `fips` (5-character text, keep leading zeros) + `episode_id` are unique across all splits.
- **Splits are by time** (year of the episode's first report in the county). The test year is touched once, at the end.
- **Coverage:** 3,073 counties, 54,580 storm episodes and 49 states/DC in train; 3,023 counties and 7,734 episodes in validation; 3,014 and 7,866 in test.

Upstream tables are in the same dataset: `data/` (raw inputs) and `processed/` (cleaned 15-minute and hourly outages, daily observation flags, cleaned storm events, storm × county, county lookup).

## 3. Origin

| Source | What | Version used | License |
|---|---|---|---|
| **EAGLE-I** (Oak Ridge National Laboratory), [figshare 24237376](https://figshare.com/articles/dataset/24237376) | County customers-out snapshots every 15 minutes, 2014–2025 (249.3 M rows) plus reference files: modeled customer counts (MCC), data quality index (DQI), coverage history | Yearly files combined into one CSV (13 GB) | CC BY 4.0 |
| **NOAA Storm Events** (NCEI), [csvfiles](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) | Storm reports with type, time, location, magnitude, episode id, 2014–2025 (771 K events) | `StormEvents_details-ftp_v1.0_dYYYY_c2026MMDD` (files created 2026-03-23 to 2026-08-19) | Public domain |
| **NWS zone–county correlation** ([weather.gov](https://www.weather.gov/gis/ZoneCounty)) | Which counties each forecast zone covers | 23 dated versions (Oct 2014 – Apr 2026), combined into a dated crosswalk | Public domain |
| **US Census** | County population estimates (Vintage 2023); Connecticut planning region ↔ county crosswalk | `co-est2023-alldata`, `sub-est2023_9` | Public domain |

## 4. Lineage

```
EAGLE-I CSV (249.3 M rows) --+
NOAA Storm Events (771 K) ---+-- 01 audit: types, nulls, coverage, leakage tags (no data changed)
NWS zone crosswalk ----------+
Census, MCC, DQI ------------+-- 02 prep
                                  -> outages_15min (208.1 M) -> outages_hourly (61.4 M)
                                  -> county_observed_daily (10.7 M county-days)
                                  -> storm_events_clean (573 K) -> storm_county_events (643,757)
                                  -> counties (3,248)
                                         |
04 modeling table <----------------------+
   643,757  storm x county rows
   629,256  lower 48 + DC
   558,402  drought / heat / excessive heat dropped
   298,398  collapsed to county x episode
   271,644  county observed on the start day and the window's end day (FINAL)
            train 208,482 | validation 31,064 | test 32,098
```

## 5. Cleaning and transformation steps

**Outages (notebook 02)**

1. Kept 2017–2025 (2014 starts on Nov 1, and county coverage only reaches ~92% from 2017). 221.4 M raw rows.
2. **Dropped 8.7 M null readings** (they are gaps, not zeros) and **4.8 M explicit zero rows** (zeros are only reported from 2022). A missing row means "0 or no data", so a separate **daily observation flag** records whether EAGLE-I was watching the county (county reported that year, ≥90% of the day's snapshots collected nationally, readings not all null).
3. Snapped 20 K off-grid timestamps to the 15-minute grid.
4. **Denominator:** EAGLE-I's modeled customer count (MCC). `total_customers` from the raw file was dropped (100% null except 2024, where it is unreliable). Where MCC is broken, an estimate of Census population × 0.508 customers per person is used: **all 100 North Carolina counties** (MCC gives 0.17 customers per person vs 0.35–0.60 elsewhere), 9 counties with no MCC, and 21 counties below 0.1 or above 5 customers per person.
5. **Capped customers out at the denominator** (~0.05% of rows), flagged `over_mcc`, raw value kept.
6. Connecticut: 2025 planning-region rows converted back to the 8 old counties with population weights.
7. Built an hourly table (worst snapshot per county-hour).

**Storms (notebook 02)**

1. Dropped marine and waterspout events, and **33 leakage or unneeded columns** (damage, injuries, deaths, narratives, tornado scale and path size, which are only known afterwards).
2. Converted local times to **UTC**; fixed NOAA territory state codes.
3. Expanded **forecast-zone reports to counties** with the zone crosswalk version in force on the event date; flagged zones that cover only part of a county (`partial_county`).

**Modeling table (notebook 04)**

1. Scope: lower 48 + DC; Drought, Heat and Excessive Heat are not storm triggers.
2. **Unit:** all reports of one NOAA episode in one county collapse into one row; `t0` = first report there.
3. **Kept only county-episodes EAGLE-I observed** on the start day and the 24 h window's end day (drops 26,754 rows, mostly in 2017–2021 when coverage was lower).
4. **Target:** peak share of customers out over hours 0–23 from the start hour ≥ 5%. Hours with no outage row count as 0 (the county is known to be observed).
5. **Storm features** summarise the whole episode in the county: report counts by family, how reports were located, max wind/hail, duration, and storm-system size. This **observed summary stands in for a forecast**.
6. **Pre-storm features** from the 24 h before the start hour (missing when the day before wasn't observed).
7. **County history** from training-year labels only, shrunk toward the national rate with 20 pseudo-episodes. Training rows use the *other* training years (leave-one-year-out) so no row sees its own label; validation and test rows use all training years.
8. Test labels are written to a separate file.

## 6. Dataset statistics

**Target**

- 14.1% positive in train, rising over time: 11.2% (2017), 12.2%, 12.4%, 16.7% (2020), 15.5%, 14.0%, 16.8% (2023); validation 2024 16.5%.
- The label-defining peak: 22% of county-episodes have no customers out at all in the window, 32% reach 1%, 14.1% reach 5%, 8.3% reach 10%, 1.2% reach 50%. The median peak among positives is 11.9%.

**Features** (training rows; full table in `reports/ml_eda_tables/feature_dictionary.csv`, missing values by split included)

- A typical county-episode has **1 report** (65% of rows; mean 1.9, max 83) and is active in the county for a median of 4 hours (75th percentile 15 h). The whole storm system touches a median of 7 counties (mean 9.7).
- **71% of rows have no thunderstorm wind report and 87% no hail**; tropical, ice storm and lightning are present in ≤1%.
- **Half of all rows are zone-coded only** (101,801), half county-coded only (104,315), ~1% mixed.
- **Missing values:** `max_hurricane_cat` 99.9%, `max_hail_in` 87.2%, `max_wind_kts` 59.0% (empty when no report of that kind; the empty value carries information), `pre_peak_24h` and `pct_out_at_start` 3.3% (day before unobserved), `log_pop` 0.4% (Connecticut old counties). All other features are complete.

## 7. Known limitations and biases

- **Only storm-linked outages.** In 2017–2023 only ~20% of significant outage county-days have a NOAA storm record within ±1 day. The data says nothing about outages without one.
- **Observed storms stand in for forecasts.** Real forecast-time accuracy will be lower.
- **Coverage grows over time:** EAGLE-I watched 84–88% of county-episodes in 2017–2021 vs 92–98% in 2022–2025, and tracks more customers each year. This is part of why the positive rate rises; models trained on all years may under-predict 2024–2025.
- **`dqi` and `pct_customers_tracked` are year-mapped quality scores** (values exist for 2018–2022 only; 2023–2025 all use 2022). `dqi` shifts strongly between train and validation/test (PSI 3.7) and partly acts as a year marker. Use with care.
- **`hist_episodes` scale differs by split:** training rows count 6 other training years, validation/test rows count all 7 (mean 87 vs 102).
- **Small-county effect:** 5% of a small county is few customers, so small counties cross the threshold more easily (the positive rate is ~18% for mid-sized counties and falls to ~5% for the largest).
- **Zone-coded storms are blurry:** a zone report is assigned to every county the zone touches, so their outage rate (9.9%) is half that of county-coded storms (17.9%).
- **Denominator estimates** for North Carolina and 30 other counties rely on a national customers-per-person ratio.
- **Overlapping storms make rows dependent** (one storm system, many counties). Evaluate with resampling by `episode_id`.
- 2025 (test) has almost no tropical activity in the features (9 rows), so tropical performance can't be checked on test.

## 8. Feature dictionary

Statistics are from the training rows. Shares are fractions (0.05 = 5%). `retro_*` columns are **leakage**: they describe the outcome and must never be model inputs.

| Column | Group | Unit | Description | Missing (train) | Mean | Median | Max |
|---|---|---|---|---|---|---|---|
| `fips` | id | text | 5-digit county FIPS code (Census 2023) | 0 |  |  |  |
| `episode_id` | id | integer | NOAA Storm Events EPISODE_ID (one storm system) | 0 |  |  |  |
| `state_abbr` | id | text | State postal code (lower 48 + DC) | 0 |  |  |  |
| `fema_region` | id | 1-10 | FEMA region of the county | 0 |  |  |  |
| `year` | id | year | UTC year of t0; sets the split (train 2017-2023, validation 2024, test 2025) | 0 |  |  |  |
| `t0` | id | UTC time | Begin time of the episode's first report in this county | 0 |  |  |  |
| `n_events` | storm | count | Storm reports in this county during the episode (drought/heat excluded) | 0 | 1.88 | 1.00 | 83.00 |
| `n_tstm_wind` | storm | count | Reports of type Thunderstorm Wind | 0 | 0.56 | 0.00 | 60.00 |
| `n_hail` | storm | count | Reports of type Hail | 0 | 0.26 | 0.00 | 57.00 |
| `n_tornado` | storm | count | Reports of type Tornado or Funnel Cloud | 0 | 0.0538 | 0.00 | 35.00 |
| `n_high_wind` | storm | count | Reports of type High Wind or Strong Wind | 0 | 0.20 | 0.00 | 83.00 |
| `n_flood` | storm | count | Flash Flood, Flood, Heavy Rain, Coastal/Lakeshore Flood, Storm Surge/Tide, Debris Flow | 0 | 0.27 | 0.00 | 68.00 |
| `n_winter` | storm | count | Winter Weather, Winter Storm, Heavy Snow, Blizzard, Lake-Effect Snow, Sleet | 0 | 0.38 | 0.00 | 35.00 |
| `n_ice_storm` | storm | count | Reports of type Ice Storm | 0 | 0.0084 | 0.00 | 12.00 |
| `n_tropical` | storm | count | Tropical Storm, Hurricane (Typhoon), Tropical Depression | 0 | 0.0123 | 0.00 | 10.00 |
| `n_cold` | storm | count | Extreme Cold/Wind Chill, Cold/Wind Chill, Frost/Freeze | 0 | 0.0626 | 0.00 | 12.00 |
| `n_lightning` | storm | count | Reports of type Lightning | 0 | 0.0097 | 0.00 | 8.00 |
| `n_wildfire` | storm | count | Reports of type Wildfire | 0 | 0.0196 | 0.00 | 7.00 |
| `n_other` | storm | count | Any other event type (e.g. Dense Fog, Rip Current, Dust Storm) | 0 | 0.0409 | 0.00 | 11.00 |
| `n_county_coded` | storm | count | Reports NOAA located by county (CZ_TYPE = C) | 0 | 1.14 | 1.00 | 68.00 |
| `n_zone_coded` | storm | count | Reports issued for an NWS forecast zone, mapped to counties with the dated crosswalk | 0 | 0.74 | 0.00 | 83.00 |
| `share_zone_coded` | storm | 0-1 | n_zone_coded / n_events | 0 | 0.49 | 0.00 | 1.00 |
| `any_partial_county` | storm | boolean | At least one zone report whose zone covers only part of the county | 0 | 0.22 | 0.00 | 1.00 |
| `max_wind_kts` | storm | knots | Highest wind magnitude among the reports (measured or estimated); empty if none reported | 59.0% | 54.03 | 52.00 | 173 |
| `max_hail_in` | storm | inches | Largest hail size among Hail reports; empty if no hail | 87.2% | 1.29 | 1.00 | 6.42 |
| `max_hurricane_cat` | storm | 0-5 | Highest Saffir-Simpson category reported; almost always empty | 99.9% | 1.76 | 1.00 | 5.00 |
| `county_episode_hours` | storm | hours | First report begin to last report end in this county | 0 | 19.08 | 4.00 | 744 |
| `episode_counties` | storm | count | Counties touched by the whole episode (storm-system size) | 0 | 9.70 | 7.00 | 96.00 |
| `log_episode_counties` | storm | log10 | log10(episode_counties) | 0 | 0.81 | 0.85 | 1.98 |
| `episode_states` | storm | count | States touched by the whole episode | 0 | 1.00 | 1.00 | 3.00 |
| `episode_events` | storm | count | Distinct reports in the whole episode, all counties | 0 | 15.46 | 10.00 | 251 |
| `month` | time | 1-12 | Month of t0 (UTC) | 0 | 5.71 | 5.00 | 12.00 |
| `local_hour` | time | 0-23 | Hour of t0 in local standard time (state's main time zone) | 0 | 12.30 | 14.00 | 23.00 |
| `log_customers` | county | log10 | log10 of customers served: EAGLE-I MCC, or Census population x 0.508 where MCC is broken (all NC + 30 counties) | 0 | 4.33 | 4.29 | 6.58 |
| `log_pop` | county | log10 | log10 of Census 2023 county population; empty for Connecticut's 8 old counties | 0.4% | 4.63 | 4.58 | 6.99 |
| `dqi` | county | 0-100 | EAGLE-I data quality index of the FEMA region (2018-2022 values; 2017 uses 2018, 2023+ use 2022) | 0 | 78.19 | 79.90 | 94.04 |
| `pct_customers_tracked` | county | 0-1 | Share of the state's customers EAGLE-I tracks (same year mapping as dqi) | 0 | 0.87 | 0.90 | 1.00 |
| `pre_peak_24h` | pre-storm | 0-1 | Worst share of customers out in the 24 h before the start hour; empty if the day before wasn't observed | 3.3% | 0.0085 | 0.0003 | 1.00 |
| `pct_out_at_start` | pre-storm | 0-1 | Share of customers out in the hour before the start hour; empty if the day before wasn't observed | 3.3% | 0.0032 | 0.00 | 1.00 |
| `hist_episodes` | history | count | County-episodes for this county in the training years (training rows: other training years only) | 0 | 86.63 | 68.00 | 463 |
| `hist_outage_rate` | history | 0-1 | County's past outage rate, shrunk toward the national training rate: (pos + 20 x p0) / (n + 20) | 0 | 0.13 | 0.12 | 0.58 |
| `hist_mean_peak` | history | 0-1 | County's past mean peak share out, shrunk the same way | 0 | 0.0315 | 0.0248 | 0.27 |
| `target` | target | 0/1 | 1 if peak share out in hours 0-23 from the start hour is >= 5% of customers | 0 | 0.14 | 0.00 | 1.00 |
| `retro_peak_24h` | retro (leakage) | 0-1 | Peak share of customers out in the 24 h window (defines the target) | 0 | 0.0335 | 0.0019 | 1.00 |
| `retro_peak_customers_out` | retro (leakage) | customers | Peak customers out in the 24 h window | 0 | 1,237 | 56.00 | 630,764 |
| `retro_hour_of_peak` | retro (leakage) | 0-23 | Hour of the peak after the start hour (earliest if tied) | 22.0% | 7.52 | 5.00 | 23.00 |

## 9. Responsible use and maintenance

- **No personal data:** all records are county-level aggregates or public storm reports.
- **Leakage guard:** never use `retro_*` columns as inputs; never fit anything on `holdout/`. County history must be rebuilt from training years only if the split changes.
- **Rebuilding:** run notebooks 01 → 02 → 04 (~25 minutes); 06 regenerates every statistic in this card.
- **Updates:** a new year needs the next EAGLE-I and NOAA releases, a crosswalk version check, and a re-run of notebooks 02 and 04. Then bump the version above.

## 10. Citation

- Oak Ridge National Laboratory. *EAGLE-I power outage data, 2014–2025*. figshare, dataset 24237376. https://figshare.com/articles/dataset/24237376
- NOAA National Centers for Environmental Information. *Storm Events Database*. https://www.ncdc.noaa.gov/stormevents/
