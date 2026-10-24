# Understanding the storm and power-outage data

**Repository review · October 24, 2026**

**Baseline snapshot:** This guide records the review before the notebook cleanup. The subsequent cleanup preserves unknown outcomes, uses observed-hour pre-event means, deduplicates grouped burden, and keeps NOAA episodes separate instead of merging narrative names. The notebooks now execute cleanly. Sections 8–11 describe the earlier behavior and rates; `data_audit.json` is refreshed from the revised shared code. See [notebook execution notes](../notebooks/README.md) for the current conventions.

This project combines records of severe weather, county power outages, and county characteristics to study how outage severity varies around storms in Alabama, Florida, Georgia, North Carolina, South Carolina, and Tennessee.

The data is substantial enough for useful analysis. The main challenge is defining trustworthy outcomes: a storm record does not establish that the storm caused an outage, and a missing outage reading does not establish that power stayed on. Before modeling, we need to decide which observations can support a label and what information would actually be available when making a prediction.

This guide explains every local dataset, the current join, the strongest findings, and the decisions required to finish EDA. It is a review, not a declaration that the data is ready for modeling.

## 1. What was reviewed

The review scanned all **30 processed files**, totaling about **174 MB**, including every stored row for field completeness and the aggregate checks described below. It also inspected the download/load code, all three notebooks and their saved findings, and the inventory of retained raw files. The storm–outage join was rerun independently from notebook 03's current code.

The local files are the authority for the counts in this document. Upstream documentation was consulted for interpretation; no datasets were refreshed or replaced. The retained national NOAA archives were inventoried, but their out-of-study-area records were not independently reprocessed. Raw EAGLE-I CSVs are absent, so this review cannot establish byte-for-byte equivalence with those original downloads.

Supporting files:

- [Full field inventory](data_dictionary.md): all columns, types, and stored-cell null counts.
- [Machine-readable audit](data_audit.json): detailed results and SHA-256 fingerprints of the processed files.
- [Audit script](../scripts/audit_data_review.py): rerun with `python scripts/audit_data_review.py` from the repository root.

## 2. The datasets at a glance

| Dataset | What one row represents | Local size | Purpose |
| --- | --- | ---: | --- |
| `eaglei/outages_YYYY.parquet` | A county at one collection timestamp | 51,252,656 rows, 12 files | Observed number of customers without power |
| `storms/storm_events_YYYY.parquet` | One NOAA weather event | 94,350 rows, 12 files | Weather type, timing, location, reported strength and impacts |
| `storms/storm_event_counties.parquet` | An event assigned to a county | 92,852 rows | Connect weather events to county outage records |
| `reference/zone_county.parquet` | One forecast-zone/county relationship | 644 rows | Translate zone-based weather reporting into counties |
| `reference/county_census.parquet` | A county and ACS estimate year | 5,874 rows | Population, housing, area and density |
| `eaglei/MCC.csv` | A modeled county customer count | 3,234 national rows | Candidate denominator for outage percentages |
| `eaglei/coverage_history.csv` | A state and reference year | 280 national rows | How much of a state's customer base EAGLE-I covers |
| `eaglei/DQI.csv` | A FEMA region and year | 50 national rows | Broad indicators of collection quality |

The county universe in Census is **534 counties**: AL 67, FL 67, GA 159, NC 100, SC 46, TN 95. Outage records contain **529 distinct counties at least once**, but fewer in individual years.

The reference CSVs include places outside the six-state study area. They must be filtered before computing study-area summaries. The optional EIA-861 utility dataset has downloader support, but no local EIA files or processed EIA features were found.

## 3. EAGLE-I: what the outage records tell us

### Reading a row

The five columns are `fips_code`, `county`, `state`, `customers_out`, and `run_start_time`. The intended key is county plus timestamp. A record saying 500 customers out describes the reported outage level at that collection time; it does not mean 500 new customers lost power since the previous record. Do not add consecutive readings to count unique affected customers. The file also does not identify individual customers or the cause of the outage.

Keep FIPS identifiers as five-character strings. For example, Alabama county identifiers begin with `01`; converting them carelessly to numbers loses the leading zero needed for joins.

The original EAGLE-I data descriptor confirms UTC timestamps and explains that absent entries can represent either zero outages or a collection gap. It does not distinguish those cases. Its original release description covers 2014–2022; the local files extend to 2025 and contain explicit zeros in newer years, so the original zero-omission description does not fully characterize this local release. [EAGLE-I data descriptor](https://www.nature.com/articles/s41597-024-03095-5).

### Coverage changes substantially over time

| Year | Stored outage rows | Counties seen | Null outage counts | Explicit zero counts |
| --- | ---: | ---: | ---: | ---: |
| 2014 | 358,659 | 410 | 0 | 0 |
| 2015 | 2,580,145 | 449 | 0 | 0 |
| 2016 | 2,960,707 | 449 | 0 | 0 |
| 2017 | 3,388,524 | 508 | 28,781 | 0 |
| 2018 | 4,553,450 | 508 | 315,048 | 0 |
| 2019 | 4,808,385 | 525 | 275,065 | 0 |
| 2020 | 5,194,354 | 523 | 429,447 | 0 |
| 2021 | 4,832,707 | 523 | 218,042 | 0 |
| 2022 | 5,289,340 | 527 | 0 | 342,871 |
| 2023 | 5,412,736 | 527 | 0 | 195,956 |
| 2024 | 5,837,620 | 527 | 0 | 214,507 |
| 2025 | 6,036,029 | 528 | 489,434 | 86,898 |

The first record is November 1, 2014 at 04:00 UTC. Therefore, 2014 is not a full observation year. Files for 2014–2021 end at December 31 at 00:00, leaving almost all of that final day outside the file's coverage. Later files reach December 31 at 23:45.

Five Tennessee counties never appear: Benton, Decatur, Hardin, Humphreys, and Perry. A storm in one of these counties cannot receive an observed outage target from these files. Their absence must not become evidence of zero storm impact.

**Interpretation:** increasing row counts over time can reflect changing collection coverage, changing zero/null representation, or more outages. The row counts alone cannot distinguish those explanations.

### What passes basic quality checks

Across all outage files, the audit found no duplicate county/timestamp keys, no null keys, and no negative or fractional non-null customer counts. All files use the same five-column structure. These are valuable structural checks, but do not establish that every reported outage count is accurate.

There are **1,755,817 null customer counts**, or **3.43% of stored rows**. Null rates vary by year: 2020 is about 8.27%, and 2025 about 8.11%. A null cell, an explicit zero, and an absent timestamp are three different situations and should remain distinguishable in the analysis.

There are **1,003 timestamps off the usual quarter-hour minute grid**, concentrated in 2019, 2020, 2021 and 2023. Any rounding policy needs a collision check: two records could land on the same county/timestamp after rounding.

### Customer counts are not automatically credible because they are numeric

The largest stored value is **1,777,800 customers** in 2017. Notebook 01 identifies suspicious county peaks, especially around Irma. These need investigation against the denominator and surrounding readings; the audit does not independently prove their cause.

The current EDA reduces isolated spikes by taking the median within each hour. This changes the target from the highest recorded 15-minute count to the highest hourly median. It can suppress real short-lived peaks as well as erroneous spikes. A median based on a single available reading offers no spike protection.

## 4. NOAA Storm Events: the weather side

### What an event means

There are **94,350 unique event IDs**, **35 event types**, and no missing or reversed UTC start/end times in the processed annual storm files. An event is a reporting unit, not necessarily one whole meteorological storm. `episode_id` groups related reports; a large system can involve many event records and multiple episodes. Naming or grouping a whole storm is an additional project decision.

NOAA's Storm Events database is a historical report collection. The National Weather Service describes reports as becoming available 90–120 days after an event. Consequently, the complete row should not be treated as a feature vector that existed before a storm arrived. [NWS explanation of storm reports](https://www.weather.gov/unr/storm_reports).

### Important groups of fields

| Fields | Meaning for this project | Caution |
| --- | --- | --- |
| `event_id`, `episode_id` | Record identity and grouping | IDs are grouping keys, not measures of storm strength |
| `event_type` | Thunderstorm wind, hail, tornado, flood, etc. | Final reported type may not equal the forecast available beforehand |
| `begin_*`, `end_*`, `cz_timezone` | Timing and local-to-UTC conversion inputs | Use UTC for joins; preserve local standard time for local-hour descriptions |
| `state_fips`, `cz_type`, `cz_fips`, `cz_name` | Reported county or forecast zone | A zone code is not a county FIPS code |
| `magnitude`, `magnitude_type` | Event-specific strength and reporting method | Wind and hail use different units; missing can mean not applicable |
| `tor_f_scale`, `tor_length`, `tor_width` | Tornado characteristics | Final assessment information requires an availability check |
| `damage_*`, `injuries_*`, `deaths_*` | Reported consequences | Suitable for impact description; exclude from advance-prediction inputs |
| `begin_lat/lon`, `end_lat/lon`, location fields | Reported spatial information | Not present for every event; points are not complete storm footprints |
| `source`, `wfo`, `data_source` | Reporting provenance | Can reflect differences in reporting practices |
| `event_narrative`, `episode_narrative` | Written accounts | Can explicitly reveal outages and damage; high leakage risk |

The preprocessing code adds parsed local and UTC timestamps and numeric damage amounts. For example, a damage string of `25.00K` becomes 25,000 dollars. It does not turn an unreported damage amount into an observed zero.

### The data is dominated by a few types

| Event type | Raw study-area reports |
| --- | ---: |
| Thunderstorm Wind | 45,237 |
| Hail | 8,958 |
| Flash Flood | 7,446 |
| Drought | 3,696 |
| Winter Weather | 3,680 |
| Tornado | 3,405 |
| Flood | 2,648 |
| Winter Storm | 2,498 |
| Tropical Storm | 2,248 |
| Hurricane (Typhoon) | 224 |

Nearly half of the reports are Thunderstorm Wind. A model that performs well on this frequent type may still perform poorly on rare, high-impact events. There are only six Blizzard reports; an apparently high outage rate for that category is not a stable basis for generalization.

### Missing fields often mean “not applicable”

`magnitude` is populated in every stored Thunderstorm Wind, High Wind, Strong Wind and Hail row, but absent for the other event types. Across all events, it is 40.12% null. Filling those missing values with zero would incorrectly equate “this field does not apply” with “zero strength.”

Tornado scale is present on the 3,405 tornado rows, making its overall null rate of 96.39% unsurprising. Property damage is missing in 21.86% of rows, and coordinates in 24.33%. These patterns call for event-specific treatment rather than a single blanket imputation rule. [Complete field profiles](data_dictionary.md).

## 5. Geography: connecting events to counties

The pipeline assigns county-reported events directly using state and county codes. For zone-reported events, it uses the NWS zone–county lookup. The retained lookup filename is `bp16ap26.dbx`; the code applies this one lookup across historical event years. It is not a dated history of zone boundaries.

Of **22,837 zone reports**, **1,498 fail to map**, or **6.56%**. The largest losses include 358 Coastal Flood reports, 227 Tropical Storm reports, 142 Strong Wind reports, 79 High Wind reports, and 22 Hurricane reports. These losses are concentrated enough to matter for the study's severe-weather focus.

The resulting mapped table has **92,852 rows and 92,852 distinct events**. Although the code supports expanding an event to multiple counties, this particular local output has one row per retained event. Do not assume an expansion occurred simply because the table is named “event counties.”

The table still contains **59 Georgia Winter Storm rows with county identifiers absent from the Census county universe**. Notebook 02 diagnoses them as zone codes entered as county codes; this review independently confirms the invalid matches, not a corrected mapping. Notebook 03 drops them through its Census inner join.

**What needs to change:** document historical mapping evidence, preserve original codes, record any corrected mapping and its reason, and publish exclusions by year, state and event type. A current code match alone does not establish historical correctness. Do not infer a county solely because a nearby outage looks plausible.

## 6. Census and customer denominators

### Census describes the county

The Census table contains **534 counties for each of 11 years, 2014–2024**. Its fields are population, housing units, land area in square miles, population density, housing density, county name, FIPS and ACS year. The audit found no missing or nonpositive population, housing or area values.

The code uses ACS five-year estimates and joins a fixed 2023 Gazetteer land-area file. Five-year ACS values summarize multiple years, rather than a point-in-time county census. Census generally releases them after the reference period; a forecasting join should use the latest release available on the prediction date. [Census explanation of ACS estimates and release timing](https://www.census.gov/help/topics/faq.what-acs-estimates-are-available-for-my-area.html).

The current notebook joins the same ACS year as the outage/event and reuses 2024 values for 2025. That is a retrospective convention, not a verified historical-availability rule. It also means population and density features would need a different, release-aware join for advance prediction.

### MCC describes an estimated customer base

MCC supplies `County_FIPS` and `Customers`. It matches **528 of 534 study counties**. All six missing matches are in North Carolina: Graham, Haywood, Jackson, Macon, Swain and Transylvania.

Comparing MCC against 2023 housing units gives the following local result:

| State | Median customers per housing unit | Counties below 0.5 |
| --- | ---: | ---: |
| AL | 1.09 | 0 |
| FL | 1.21 | 0 |
| GA | 1.23 | 0 |
| NC | 0.50 | 48 |
| SC | 1.22 | 0 |
| TN | 1.16 | 0 |

Notebook 01 reports an especially implausible example: Henderson County, NC has an MCC value of 24 against 57,338 housing units in 2023. These discrepancies make MCC unsafe as an unquestioned denominator. They do not, by themselves, prove the exact cause or validate a replacement formula.

### The notebook's replacement is a modeling assumption

Notebook 03 estimates customers as **1.2 × housing units for every county**, then caps hourly outages at that estimate. A county with 10,000 housing units gets an estimated customer base of 12,000; a 600-customer peak becomes 5% and meets the notebook's “major outage” threshold.

With a factor of 1.0, that same peak is 6%; with 1.3, it is about 4.6%. The binary label can therefore change without any change in the underlying outage reading.

The rerun found **2,865 capped county-hours out of 14,969,039 observed county-hours**. This is a small share overall, but capping acts on the extreme values that matter most for severe-outage prediction. Preserve both the original value and a cap flag, and assess sensitivity before adopting the proxy as final.

The ratio is best described as **reported customers out divided by estimated county customers**. A denominator adjustment alone does not correct incomplete reporting in the numerator. Also avoid using a county's maximum outage across all years as its denominator: that lets future outcomes define earlier labels and features.

## 7. Coverage history and DQI: context, not row-level truth

`coverage_history.csv` contains a state/year date, estimated total customers, minimum and maximum covered customers, and minimum and maximum coverage fractions. The local reference spans 2018–2022. Its `max_pct_covered` values are fractions: 0.75 means 75%.

Notebook 01's saved state summaries show 2022 maximum coverage of 81% in AL, 98% in FL, 91% in GA, 92% in NC, 99% in SC and 75% in TN. Those are broad coverage indicators, not evidence that every county-hour was successfully observed. A state's maximum coverage is particularly inappropriate as a guarantee for every date.

`DQI.csv` has ten FEMA regions over five years. Fields include collection success, enabled collection, spatial precision, customer coverage and an overall index. The notebook uses Region 4 for this study; its saved index varies from about 65 to 85 over 2018–2022. These regional values cannot identify which individual missing readings were true zeros. Neither reference supplies equivalent year-specific quality evidence for 2023–2025.

## 8. How the current EDA joins the datasets

```text
NOAA event records -- county/zone mapping --> event + county + time window
                                                   |
EAGLE-I readings -- hourly median ------------------+--> peak, share, major flag
                                                   |
Census housing units -- multiply by 1.2 ------------+--> estimated denominator
```

The implementation is in [notebook 03](../notebooks/03_eda_storm_outage.ipynb), section 2. The following describes its current behavior, not an endorsed final specification.

1. Build `cust`, the county/ACS-year table with estimated customers.
2. Build `hourly`, taking the median of available non-null readings in each county-hour and capping it at estimated customers.
3. Build `ev`, retaining mapped events whose starts fall within the overall outage date range and excluding selected event types.
4. Define an inclusive hourly window from one hour before the floored start to 12 hours after the floored end; event duration is capped at 48 hours before the extra 12 hours are added.
5. Build `imp`: peak hourly outage, peak time, summed hourly outage values, the preceding 24-hour average, peak share, and a major flag at 5%.

For an illustrative event starting 15:20 and ending 16:10, the window contains hourly keys from 14:00 through 04:00 the next day, inclusive. This is 15 hourly keys, and includes pre-event observations. The target can therefore capture outages that began before the event. It is not automatically the incremental outage caused by that storm.

The excluded categories include drought, heat/excessive heat, fog and several other types. Treat this as a scope restriction. The notebooks' phrasing that such types “don't affect the grid” is too broad to serve as an exclusion rationale.

### Results of reproducing the join

| Check | Result | Interpretation |
| --- | ---: | --- |
| Retained event-county rows | 78,509 | Current analysis population, before fixing label eligibility |
| Rows with no observed hourly peak | 11,190 (14.25%) | Current SQL converts these to zero and non-major |
| Windows with a usable value in every hourly key | 26,887 (34.25%) | Does not imply all four quarter-hour readings existed |
| Partially observed windows | 40,432 (51.50%) | At least one usable hour, but not every hour |
| Current major labels | 14,858 (18.92%) | Provisional prevalence under the present rules |
| Distinct county-hours covered by event windows | 767,330 | Window union, including hours with no usable outage value |
| County-hours belonging to multiple event windows | 235,681 (30.71%) | Overlapping exposure to the same outcome |
| Most event windows sharing one county-hour | 118 | Some observations are reused extensively |

The SQL uses `coalesce(..., 0)` when an entire window has no usable observations. It also divides the pre-window sum by 24 regardless of the number of observed hours. Both operations build a zero assumption into the result.

The missing-peak share changes sharply: about 32.6% of 2015 event rows, 6.0% of 2024 rows, and 12.4% of 2025 rows. A model could learn reporting differences between years instead of differences in storm impact.

### Customer-hours and storm totals need precise names

With complete hourly observations, summing the hourly outage level gives an approximate customer-hour burden. Here the levels are medians of available samples, and missing hours contribute nothing. Call the current quantity an **observed hourly burden estimate**, with its observation completeness alongside it.

Summing that value across overlapping events counts some county-hours multiple times. Notebook 03 deduplicates county-hours for one county-level analysis, but type-level totals still overlap. Its episode-level use of the maximum event burden is also not the same as integrating the union of the episode's windows.

Similarly, adding each county's peak does not give a simultaneous regional peak or the number of unique customers affected. “Sum of county peaks” is the defensible name for that quantity.

## 9. What the EDA currently suggests

The independently reproduced join shows large differences in the current major-label rate by reported event type:

| Type | Event-county rows | Current major-label rate |
| --- | ---: | ---: |
| Hurricane (Typhoon) | 202 | 98.5% |
| Tropical Storm | 2,007 | 57.9% |
| Storm Surge/Tide | 182 | 53.8% |
| Ice Storm | 210 | 35.7% |
| High Wind | 804 | 30.7% |
| Tornado | 3,237 | 30.2% |
| Thunderstorm Wind | 42,354 | 21.3% |
| Hail | 8,150 | 8.3% |

These are **descriptive associations under the current join**, not validated forecast probabilities or causal estimates. The rows are correlated within storms, their windows differ, and missing-peak frequencies differ by type. For example, no Hurricane rows lack an observed peak, but 53 of 210 Ice Storm rows do.

The useful direction is that tropical systems have much higher observed severity, while thunderstorm wind provides a large number of examples. More formal comparisons should use uncertainty estimates that respect storm grouping and should be repeated after label-quality rules are settled.

Notebook 03 also reports that only 36% of 15,521 major-outage onsets have a storm recorded in the county within its chosen matching window, while 47% have one only elsewhere in the state. These are saved notebook findings, not recomputed by this audit. They use a different definition requiring both 5% and 1,000 customers. They should not be mixed with the event-table rates above. An unmatched record is not proof of a non-weather cause, and a statewide match is not proof of a county mapping failure.

## 10. What must be decided before modeling

### First, decide what the model is allowed to know

The local data directly supports a retrospective question: **given a documented weather event, how severe were the reported outages around it?** Advance forecasting asks a different question: **using only information available at a particular time, what outage will occur over a future horizon?**

For the forecasting version, specify the decision time and lead time. Forecast weather, warnings, and their issuance times are not currently part of the processed dataset. Final event magnitude, EF assessment, observed duration, damage and narratives cannot simply substitute for those inputs. Even a pre-event outage feature needs a reporting-latency rule.

### A concrete EDA-to-modeling checklist

| Decision | Evidence needed to finish EDA |
| --- | --- |
| Outcome eligibility | Counts of observed zeros, unobserved windows, null readings and partial windows by year/state/type; a policy that preserves unknown outcomes |
| Historical geography | Documented correction or exclusion for unmatched zones and invalid county codes; retained original values |
| Analysis unit | Choose event-county, episode-county or another unit; define grouping and overlap behavior |
| Target | Fix aggregation, denominator, horizon and major threshold; distinguish total severity from change over baseline |
| Denominator | Compare plausible customer estimates and disclose how many labels change; retain uncapped values |
| Window sensitivity | Compare proposed horizons on training-period data; quantify incomplete and boundary-crossing windows |
| Inclusion period | Compare coverage and exclusions before choosing whether early years belong in training |
| Features | A per-feature availability table; units and missingness rules separated by event type |
| Evaluation split | Chronological split with a rule for storm groups and windows crossing boundaries; transformations fitted only on training data |
| Model success | Baselines, primary metrics, alert-cost assumptions and subgroup reporting defined before model comparison |

A proposed classification evaluation should include precision–recall performance, calibration, and recall at a stated alert budget. A proposed severity regression should include MAE in percentage points and errors on severe cases. Compare against simple prevalence/median baselines and, when available at prediction time, persistence based on pre-event outage levels. These are proposed criteria, not completed experiments or agreed numerical acceptance thresholds.

The configuration currently proposes 2014–2023 training, 2024 validation and 2025 testing. All years have already been examined in EDA. Preserve that disclosure, freeze the specification before further tuning, and do not describe 2025 as completely unseen. Within the training period, use chronological validation and keep related storm observations together where applicable.

## 11. Documentation and reproducibility work remaining

The repository's documentation currently disagrees with parts of the analysis:

- `data/README.md` recommends MCC normalization; notebook 03 uses housing units × 1.2.
- The same README leaves UTC unresolved; source documentation supports UTC.
- Notebook 01 suggests absent readings are mostly zero; the source confirms ambiguity between zero and collection gaps.
- The README and loader examples describe larger annual outage sizes than the local six-state subset actually contains.
- The notebook's final suggested modeling window differs from the implemented EDA window.
- Causal language, “customers affected,” and “customer-hours” need qualifications matching the actual computation.

The installed DuckDB version is **2.0.0**. Notebook 03's relation `.fetchone()` calls fail in this environment. For this review only, the audit replaced those calls in memory with `.fetchall()[0]` and successfully reproduced the join. The source notebook was left unchanged. The complete three-notebook workflow was not rerun end to end.

The audit fingerprints all processed files; its raw inventory also records the retained NOAA archive names, including creation-date suffixes. Those suffixes show that historical reports came from later releases. A historical year in a filename does not prove the record had its current contents in that year.

The audit script relies on notebook 03 cells 1 and 6 remaining the setup and join cells. It is a reproducible snapshot-review helper, not yet a standalone production target-building pipeline. Extracting that logic into a parameterized module is a useful next step once the definitions are settled.

## 12. Recommended next work

Start by adding observation-quality fields to the joined table: expected hours, usable hours, null-reading counts, pre-window usable hours, boundary flags, and whether any outage level was observed. Keep unknown outcomes separate from observed negatives. Because omitted zeros and collection gaps are confounded, do not impose an arbitrary “90% complete” filter without examining the resulting selection bias.

Then resolve mapping exclusions and compare target definitions using training years. Recompute the core EDA summaries after those decisions. Finally, update the README and notebook summaries from that same specification, and freeze the modeling task, feature availability and evaluation criteria.

The project already has useful weather and outage signal. Finishing EDA means making the observation process, label assumptions and forecasting limits explicit enough that a later model result has a clear meaning.
