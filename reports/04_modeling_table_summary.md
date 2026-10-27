# 04 – Modeling Table: County × Storm Episode (2017–2025)

Notebook: [notebooks/04_modeling_table.ipynb](../notebooks/04_modeling_table.ipynb) · Check tables: [reports/model_table_tables/](model_table_tables/) · Outputs: `processed/model_table/` · Built 2026-10-25

---

## [GOAL]

Build the labeled table the models will train on. Each row is one county hit by one NOAA storm episode, with features known when the storm starts and a yes/no target.

Decisions applied (yours, from the 03 follow-up):
- **Target:** at least **5% of the county's customers out** at some point in the **24 h** after the episode starts in that county.
- **Unit:** county × NOAA storm episode (`EPISODE_ID`). Many reports from one storm system in the same county become one row.
- **Scope:** **lower 48 states + DC**. Puerto Rico, the Virgin Islands, Alaska, Hawaii and the Pacific territories are excluded.
- **Triggers:** Drought, Heat and Excessive Heat are dropped.
- **Unobserved counties:** dropped. If EAGLE-I wasn't watching the county, we can't tell "no outage" from "no data".
- **Storm features:** a summary of the whole episode (types, counts, max wind or hail, size). This stands in for a storm forecast.

Split: train 2017–2023, validation 2024, test 2025 (by the year of the episode start).

---

## [FINDINGS]

### 1. Rows

| Step | Rows |
|---|---|
| Storm × county rows from step 02 | 643,757 |
| Lower 48 + DC only | 629,256 |
| Drought / heat dropped | 558,402 |
| Collapsed to county × episode | 298,398 |
| **County observed on the start day and the end day of the window (final)** | **271,644** |

| Split | Rows | Counties | % positive (≥5% out) |
|---|---|---|---|
| Train 2017–2023 | 208,482 | 3,073 | **14.1%** |
| Validation 2024 | 31,064 | n/a | 16.5% |
| Test 2025 | 32,098 | 3,014 | *not looked at* |

- **One storm episode reaches a median of 7 counties**, and most county × episode rows (median) hold just 1 report.
- **Observed share:** 84–88% of county-episodes in 2017–2021, vs 92–98% in 2022–2025. This repeats the coverage pattern from step 02.
- **Positive rate rises over time: 11–12% in 2017–2019, 15–17% from 2020, and 16.5% in validation.** Part of this is real (2020 and 2023 were active years), and part is EAGLE-I tracking more customers each year. **A model trained on all years will likely underpredict slightly in 2024–2025.** We should check calibration on validation.

### 2. Features (36 columns, all known at prediction time)

| Group | Features |
|---|---|
| Storm (whole episode) | report counts per storm family (12 families), county- vs zone-coded counts, share zone-coded, any partial-county zone, max wind (kt), max hail (in), hurricane category, hours of activity in the county, episode size (counties, states, reports) |
| Time | month, local hour of start (from the state's main time zone) |
| County | log customers, log population, EAGLE-I quality (DQI, % customers tracked) |
| Before the storm | worst % out in the 24 h before the start, % out in the hour before the start |
| County history (training labels only) | number of past episodes, past outage rate, past mean peak, all shrunk toward the national rate |

- **Leakage guard:**
  - Validation and test rows get county history from all training years.
  - Training rows get history from the *other* training years, so a row never sees its own label.
  - History correlates 0.30 with the target in training and 0.34 in validation. If history were leaking labels, training would score higher, so this pattern is reassuring.
  - Columns known only after the window (`retro_peak_24h`, `retro_peak_customers_out`, `retro_hour_of_peak`) are kept for analysis only, never as inputs. When several hours tie for the peak (~19% of rows), `retro_hour_of_peak` is the earliest one.
- **Gaps:**
  - Max wind exists for 41% of rows and max hail for 13% (only those storm types report them).
  - Hurricane category is almost always empty.
  - Pre-storm features are empty for 3% of training rows (the day before wasn't observed).
  - Population is missing for Connecticut's 8 old counties (Census now uses planning regions).

### 3. First look at the signal (training rows)

| Storm family in the episode | Rows | % positive |
|---|---|---|
| Tropical | 2,029 | **62.5%** |
| Ice storm | 1,574 | 33.2% |
| Tornado | 8,110 | 29.0% |
| Thunderstorm wind | 59,883 | 24.8% |
| High wind | 26,049 | 16.2% |
| Hail / flood | 26–30 K | 12–13% |
| Winter / cold / wildfire / other | | 3–8% |

- **County-coded reports matter.** Episodes with at least one county-coded report are positive 18.2% of the time, vs 9.9% for zone-coded only.
- **The pre-storm outage level is a strong signal:**

| Worst % out in the 24 h before the start | % positive |
|---|---|
| under 0.1% | 9.9% |
| 1–5% | 24.8% |
| 5% or more | 51.9% |

### 4. Data fix found during this step
Some single counties had broken MCC customer counts. For example, Buena Vista city, VA showed **1 customer for 6,566 people**, and McDonald County, MO showed **178 K for 24 K**. Those counties were labeled positive after 50–80% of their storms.
- Step 02 now uses the Census estimate for any county below 0.1 or above 5 customers per person. **21 counties** are affected. These cut-offs are provisional.
- Step 02 also now drops a fake "Grand Total" county.
- `processed/` was rebuilt and steps 03 and 04 were re-run. The changes are small: the training positive rate went from 14.2% to 14.1%, and the 02 and 03 summaries are updated.

### 5. Checks: all passed
[checks.csv](model_table_tables/checks.csv):
- one row per county × episode, and no null targets
- only lower 48 + DC
- peak % out within 0–1
- no year in two splits
- no `retro_` column among the features
