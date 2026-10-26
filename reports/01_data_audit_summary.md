# 01 – Data Quality Audit: EAGLE-I Outages + NOAA Storm Events

Notebook: [notebooks/01_data_audit.ipynb](../notebooks/01_data_audit.ipynb) · Result tables: [reports/audit_tables/](audit_tables/) · Audited 2026-10-25

---

## [GOAL]

Build a model that answers: **given a storm (or storm forecast), what is the probability that people in a specific US county lose power?**

Data: EAGLE-I county outage snapshots and NOAA Storm Events, 2014–2025. Split by time: **train 2014–2023, validate 2024, test 2025.**

This audit checks:
- data types and null rates
- which columns to drop (more than 50% null where the column applies)
- data quality problems that affect the goal
- whether the Southeast is a better scope than the whole country
- which features are worth building

Sources audited: `data/02_combined/` raw combined CSVs, plus the NWS zone→county crosswalk, Census 2023 county population, and the EAGLE-I MCC, DQI and coverage-history references.

---

## [FINDINGS]

### 1. EAGLE-I outages (249.3 M rows, 7 columns, all text in the file)

| Check | Result |
|---|---|
| Type conversion | Clean. FIPS codes are always 5 digits, and counts and timestamps all convert to numbers and dates. **FIPS must stay text** (leading zeros). |
| Duplicates (county + timestamp) | **None** |
| Negative counts | None |
| `customers_out` nulls | 0% in most years, but **1–9% in 2017–2021 and 6.5% in 2025**. They are spread across ~3,000 counties and nearly every snapshot, not whole missing snapshots. They are different from zeros, so they should not be filled with 0. |
| Zeros (no outage) | **Never stored in 2014–2021.** From 2022 some zeros appear (~1.1–1.3 M rows a year, ~5% of rows), so zeros are still only partly reported. **A missing row means "0 outages or no data". We can't tell which.** |
| `total_customers` | **100% null in every year except 2024.** In 2024 it is unreliable: it is **3.8× the MCC count in NC, 3.2× in PR and about 2× in MA and SC**. → **Drop it. Use MCC as the denominator.** |
| Impossible values | 0.05–0.15% of rows per year have `customers_out` greater than the county's MCC customer count. The largest is Miami-Dade during Hurricane Irma: 1.78 M out vs 0.95 M customers. → Cap at MCC or flag these rows. |
| Distribution | Very skewed: median 4, p99 2,640, max 1.78 M customers out. |
| County coverage | It grows over time: **2,152 counties (68%) in 2014 → ~2,900 (92%) in 2017–18 → ~3,060 (97%) from 2019.** 2014 only starts on **Nov 1**. |
| Snapshot grid (15 min) | 97.5–99.7% of the expected snapshots are present in 2014–2021, and 99.9%+ in 2022–2025. **In 2014–2021, every year is missing Dec 31** (the data stops at Dec 31 00:00). There are a few multi-hour collection gaps, the longest 24 h on Jan 6–7 2015. Some rows sit off the 15-minute grid (up to 15 K in 2018). |
| Location codes | 19 FIPS codes are not in the Census 2023 list: **Connecticut's 8 old counties** (Census now uses planning regions 09110–09190. EAGLE-I uses those **only in 2025**, alongside the old county codes, so the test year mixes both systems), **Puerto Rico**, **US Virgin Islands** and Alaska's retired Valdez-Cordova. The US Virgin Islands also appear under two names ("US Virgin Islands" in 2019–20, "United States Virgin Islands" from 2021). |

### 2. NOAA Storm Events details (771 K events, 52 columns)

| Check | Result |
|---|---|
| Type conversion | Clean. Times parse; magnitude, lat/lon and damage values parse. No event ends before it starts. Damage is text like `10.00K`/`1.5M`/`2B` and needs parsing. |
| Duplicates | None (1 row per `EVENT_ID`) |
| Time zones | Times are **local standard time** with mixed zones (`CST-6`, `EST-5`, …). They must be converted to UTC before joining to EAGLE-I. Done in the notebook. |
| How events are located | **~57% are coded by county (`C`)** and **~43% by NWS forecast zone (`Z`)**. Zone codes are **not** county codes. With the dated crosswalk, **99.4–99.9% of zone events map to counties** (about 1.3 counties per event). The ones that don't map are marine events (expected) plus a few thousand winter, drought and high-wind events. |
| Event-type vs coding | **Thunderstorm wind, hail, tornado, flood, flash flood and lightning are always county-coded with lat/lon.** **High wind, winter storm, ice storm, tropical storm and hurricane are always zone-coded**, with no lat/lon and no county. Several of these are the biggest causes of outages, so **the zone crosswalk is required, not optional.** |
| County codes | 1.1% of county-coded events have a FIPS not in Census 2023 (mostly CT, PR and Alaska changes). |

**Null rates and the 50% drop rule.** Nulls were measured overall and also within the rows where each column applies.

| Column(s) | Null % all rows | Null % where it applies | Decision |
|---|---|---|---|
| `TOR_OTHER_WFO/CZ_STATE/CZ_FIPS/CZ_NAME` | 99.7 | 86.4 (tornadoes) | **Drop**, only filled when a tornado crosses a county line |
| `MAGNITUDE`, `MAGNITUDE_TYPE` | 48 / 62 | **0** (wind, hail) | Keep. Empty by design for other event types |
| `TOR_F_SCALE`, `TOR_LENGTH`, `TOR_WIDTH` | 97.7 | **0** (tornadoes) | Not a null problem, but **leakage** (see below) |
| `CATEGORY` (hurricane) | 99.95 | **0** | Keep, but only 385 events |
| `FLOOD_CAUSE` | 89 | **0** (floods) | Keep (low value) |
| `BEGIN/END_LAT/LON`, `*_RANGE/AZIMUTH/LOCATION` | 38.9 | **0.04** (county events) | Keep lat/lon. Zone events have none by design |
| `DAMAGE_PROPERTY/CROPS`, `EVENT_NARRATIVE` | ~20 | ~20 | Below the cutoff, but **leakage**, so don't use as model inputs |
| Everything else (ids, times, type, location codes) | 0 | 0 | Keep |

**Leakage (known only after the storm).** Damage, injuries, deaths, narratives, `SOURCE` and the tornado rating, path length and width are recorded after the event. They must not be model inputs. They are only useful for explaining results. Full tagging is in [column_tags.csv](audit_tables/column_tags.csv).

**Observed, not forecast.** Every Storm Events field is an *observed* report, including event type and magnitude. To make predictions "given a storm prediction", the model inputs will eventually have to come from forecasts or warnings. Observed storms can stand in for forecasts during training, but the model will look more accurate than it really is.

### 3. Supporting files
- **MCC (modeled customers per county)** covers 3,234 counties, including CT's old counties, PR and the Virgin Islands. It is the best denominator for "% of customers out". **24 Census counties have no MCC value**, mostly in Alaska.
- **Census population** correlates **0.97** with MCC, at about **0.51 customers per person**. It is a reasonable backup denominator, and a feature too.
- **DQI by FEMA region (2018–2022).** Region IV (Southeast) is **below average**: 74.7 / 65.0 / 77.5 / 85.2 / 79.2, vs ~87–94 for the best regions. Every region improves over time.
- **Coverage history** (share of customers EAGLE-I tracks): about 88% in the Southeast in 2022, vs 97–100% in the Northeast and Mid-Atlantic and only 55% in Region IX (West). Plains and mountain states are low: NE 48%, KS 59%, ND 66%. **TN is the lowest Southeast state at 71–75%.** Low coverage means outages are undercounted.

### 4. Southeast (FEMA IV) vs all counties
| Region | Counties covered 2023 | County-coded storms | DQI 2022 | Customers tracked 2022 |
|---|---|---|---|---|
| **IV Southeast** | **98.8%** | **73%** (highest) | **79.2** (below avg) | 88% |
| Best others (I, II, III, V) | 87–100% | 47–67% | 87.6–93.7 | 90–100% |
| Weakest (VII, VIII, IX, X) | 81–98% | 23–71% | 79.9–87.4 | 55–85% |

- The Southeast has **26% of all EAGLE-I rows** and the most storm events (131 K). It does not dominate the national data, but it is the largest single region.
- **The Southeast is not clearly cleaner.** It has the most county-coded storms, which makes joining easier. But its EAGLE-I quality index is among the lowest and its customer coverage is only average.
- **Recommendation: keep all counties.** Use DQI and coverage as quality weights or filters instead of cutting regions. If we need a smaller first version for speed, the Southeast is a reasonable choice for practical reasons, but not for data quality.

### 5. Signal check: do storms line up with outages? (2022–2025, county-coded events)
- For each storm, I took the worst hour of outages in the 24 h after the storm started. Baseline: the same county and hours one week earlier.
- **Share of events where at least 1% of the county's customers lost power:**

| Event type | After storm | Baseline (1 week earlier) | Median peak % out |
|---|---|---|---|
| Thunderstorm wind | **71%** | 12% | 3.1% |
| Tornado | **71%** | 9% | 4.0% |
| Flash flood | 49% | 15% | 0.9% |
| Hail | 44% | 10% | 0.7% |

- **Storms clearly line up with outages**, so the target is learnable. **9% of storm windows had no EAGLE-I data at all**, which shows the coverage gaps above.
- Rough check only: 1% is an arbitrary cutoff, and zone-coded events are not included.

### 6. Train / validation / test readiness
| Split | Years | Notes |
|---|---|---|
| Train | 2014–2023 | 2014 is only Nov–Dec with 68% of counties. 2015–16 cover ~77–80% of counties. 2014–2021 never store zeros and miss Dec 31. |
| Validation | 2024 | Good coverage (97% counties, 99.9% snapshots). Only year with `total_customers`, which is unreliable. |
| Test | 2025 | Good coverage, but **6.5% null `customers_out`**. Rules must be set before we look at test results. |

Storm Events looks consistent across all years: 56–76 K events a year, about 57% county-coded and 99%+ of zone events mapped.

---

## [NEXT STEPS]

**Decisions for you**
1. **Scope:** all counties (recommended) or Southeast only.
2. **Training years:** keep all of 2014–2023, or start in **2015** (2014 has only 2 months of data). A later start such as **2017–2018** would give the model the ~92%+ county coverage that matches what 2024–2025 will look like.
3. **Target definition:** what counts as "losing power"? For example, at least X% of the county's customers (MCC) out, or at least N customers, within H hours of storm start. The signal check used ≥1% within 24 h only as a test.
4. **Prediction unit:** one row per county per storm event, or per county per day.

**Data preparation (after the decisions above)**
1. Build a clean typed EAGLE-I table:
   - Drop `total_customers`; join MCC as the denominator, with Census population as backup.
   - Cap `customers_out` at MCC, or flag rows above it.
   - Keep nulls as nulls (not zero).
   - Snap off-grid timestamps to the 15-minute grid.
   - Merge the two US Virgin Islands names.
   - Handle the Connecticut county → planning region change.
2. Build a **"was this county being observed?" flag** per county and time window, so missing rows aren't silently treated as "no outage". It should use the county's first and last reporting dates, collection gaps, coverage history and DQI.
3. Expand Storm Events to **one row per county per event**: county-coded events directly, zone-coded events through the dated crosswalk. Convert all times to UTC and parse damage to dollars (damage only for analysis).
4. Drop the four `TOR_OTHER_*` columns. Keep the leakage columns out of model inputs.

**Feature ideas**
- **Storm:** event type, wind or hail magnitude, hurricane category, duration, number of storm reports in the county and its neighbours during the episode, and day/night and season.
- **County exposure:** MCC customers, population and population density, plus urban/rural (needs land area).
- **County history** (train years only, to avoid leakage): past outage rate per storm, typical peak % out, and typical restoration time.
- **Outage severity and duration targets:** peak % out, customer-hours out, and hours until back below 1% out. These can be computed from the 15-minute series for a second model.
- **Forecast-time data (later):** NWS warning and watch alerts (IEM VTEC files exist for the Southeast in `data/01_raw/iem_vtec/`) or gridded forecasts. These would replace the observed storm fields so the model matches the "given a storm prediction" use case.
- **Possible extra data:** land cover and tree canopy, underground vs overhead power lines, and which utility serves the county.

**Housekeeping:** `work/audit.duckdb` (2.7 GB) is a working copy of the raw CSVs for fast queries. Delete it any time, and the notebook will rebuild it (~2 min).
