# Predicting county power outages: EDA report

## 1. What are we trying to do?

**Help counties prepare for large power outages by predicting how many customers may be without power in the next 6 and 12 hours.**

We propose two outputs for each county, updated hourly:

- **Outage size:** the highest recorded number of customers out during the next 6 or 12 hours.
- **Severe-outage probability:** the chance that this number will exceed **5,000 customers**.

Catching severe outages is our priority, but alerts must also be useful enough to justify preparation. This target includes outages already in progress; it does not specifically predict when a new outage will begin.

**Current status:** data review and exploratory analysis are complete. The next stage is building and evaluating a baseline model; no model has been trained yet.

## 2. What data do we have?

Our study covers **534 counties** in Alabama, Florida, Georgia, North Carolina, South Carolina and Tennessee.

| Data | What we have | How we use it |
| --- | --- | --- |
| EAGLE-I outages | 51.3 million county readings, 2014-2025, generally every 15 minutes | Measure outage size and derive past outage patterns |
| NOAA Storm Events | 94,350 historical weather reports, 2014-2025 | Explore which reported weather events coincide with outages |
| Census county data | Population, housing and land area; 5,874 county/year records, 2014-2024 | Compare county size and check whether outage counts look plausible |
| Zone-to-county lookup | 644 geographic relationships | Connect zone-based storm reports to counties |
| Customer-count and coverage references | Modeled customer totals and broader collection-quality summaries | Understand coverage and investigate unusual counts |

Not every dataset is suitable as a model input. In particular, historical storm reports describe what happened; we do not have a weather-forecast archive showing what was known before each prediction.

## 3. How did we prepare the data?

1. **Aligned county IDs and time.** We use five-character county FIPS codes and UTC timestamps. We corrected 119 supported Georgia storm-mapping errors while preserving the original values. Another 1,498 zone reports remain unmatched.
2. **Checked observations.** We checked missing values, duplicate timestamps, invalid counts and readings off the expected 15-minute schedule. Missing readings stay unknown; they are not changed to zero.
3. **Created hourly decision windows.** Each county receives a decision time every hour, whether or not a storm was reported. We examine readings after that time through the next 6 or 12 hours.
4. **Calculated outcomes.** The target is the highest recorded customer count in that window, without capping it. A severe outcome means strictly more than 5,000 customers.
5. **Separated reliable labels from uncertain ones.** The primary modeling group requires all 24 or 48 expected readings and no flagged anomalies. Partial windows that exceed 5,000 provide positive evidence, but partial windows below 5,000 cannot reliably establish a negative.

We also built a historical storm/outage join for exploration. It links events to county outage observations around the reported storm time. That join explains associations; it is not the proposed forecasting table.

## 4. What did we learn?

| Finding | What it means for our project |
| --- | --- |
| Only **7.70% of 6-hour** and **4.74% of 12-hour** training decisions have complete, unflagged outcomes | Missing observations are the main limitation. Model scores will initially describe this selected subset. |
| Larger counties and Florida are better represented in that subset | We must report coverage and performance across locations, not just one overall score. |
| **2.16% / 4.05%** of the retained 6/12-hour windows exceed 5,000 customers | Severe outcomes are uncommon; high accuracy alone would not demonstrate a useful model. |
| Hourly medians remove **18.6% / 21.1%** of recorded severe-window positives | We retain recorded peaks so that brief severe readings are not averaged away. |
| Tropical systems are associated with large historical outage peaks | Weather matters for interpretation, but completed storm reports cannot stand in for advance weather forecasts. |

These counts describe overlapping county-hour windows, not separate outage incidents. A long outage can contribute many positive windows. We developed the forecasting definitions using data through 2023.

## 5. What features do we propose using?

The starting model should use information that could be known when the prediction is made.

| Proposed feature group | Examples | Status |
| --- | --- | --- |
| Location | County ID, state | Available for the baseline |
| Calendar | Hour, day of week, month, season | Available for the baseline |
| Recent outage level | Latest available customers out; earlier observed values | Proposed, subject to a stated reporting-delay assumption |
| Recent outage pattern | Maximum, median and change over trailing 1/6/12/24 hours | Proposed; use past readings only |
| Recent reporting quality | Number of past readings, missing-value indicators, time since last reading | Proposed; describes what was known before prediction |

We are **not proposing realized storm type, damage, narratives or future observation quality as predictors**. Census and customer-denominator features also need release-date checks before inclusion. The modeler will finalize the feature list and reporting-delay assumptions. See the [feature-availability matrix](feature_availability.md) for the technical rules.

## 6. What will the modeling table look like?

**Proposed layout: one row per county, decision hour and forecast horizon.** A county therefore has separate rows for its 6-hour and 12-hour predictions. This full table has not been built yet; the EDA provides target-building functions and summary tables.

| Column group | Example columns | Role |
| --- | --- | --- |
| Row identifiers | `fips_code`, `decision_time`, `horizon_hours` | Identify the county and prediction window |
| Model inputs | `state`, `hour`, `day_of_week`, `month`, `latest_customers_out`, `past_6h_max`, `past_24h_median`, `hours_since_last_reading` | Information available before the prediction |
| Outcomes | `future_peak_customers`, `future_over_5000` | Values the model learns to predict |
| Outcome checks | `usable_future_readings`, `outcome_flagged`, `label_eligible` | Decide whether the outcome can be scored; never model inputs |
| Evaluation assignment | `split` | Training, validation or test |

To assemble it, start with every county-hour, attach past outage summaries and calendar/location fields, then attach the future outcomes separately. Keep rows with missing outcomes identifiable. **A storm match is not required to create a prediction row.**

## 7. How will we judge success?

Use **2014-2023 for development, 2024 for validation and 2025 for historical test evaluation**. Exclude outcome windows that reach across split boundaries. Earlier EDA already examined the later years, so 2025 is not an untouched blind test.

- **Severe-outage recall:** how many severe windows did we catch? This is the primary measure.
- **Precision and false alerts per 1,000 decisions:** how much unnecessary preparation would our alerts cause?
- **Customer-count error:** how far are peak estimates from the observed counts, especially during severe outages?
- **Baseline and subgroup comparisons:** does the model improve on simply carrying forward the latest outage level, and where does it perform poorly?

Compare false-alert tradeoffs at 80%, 90% and 95% recall. These are proposed comparison points, not agreed pass/fail requirements. Final acceptance needs a minimum recall and an acceptable alert burden. Detailed evaluation rules are in [modeling criteria](modeling_criteria.md).

## 8. What happens next?

1. Agree on reporting-delay assumptions and finalize the baseline features.
2. Build the county-hour modeling table using the EDA's target and missingness rules.
3. Train simple baseline models, compare them chronologically and select alert thresholds using validation data.
4. Report severe-outage detection, false alerts, count errors and coverage by county/state.
5. Decide whether better observation coverage or advance weather data is needed before making operational claims.

The handoff includes **four executed notebooks and 23 passing calculation tests**. The main unresolved issue is whether performance on the small, reliably labeled subset will extend to other county decisions.

For reproduction and field definitions, use the [teammate handoff](teammate_handoff.md) and [metadata guide](eda_metadata.md).
