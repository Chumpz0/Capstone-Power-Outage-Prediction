# Six-state dataset build (step 5)

How the Southeast tables are built, what is excluded and why, and which checks guard them. Commands are in the [README](../README.md); decisions are in the [decision log](decision_log.md).

## Stages

| Stage | Script | Output | Notes |
|---|---|---|---|
| 00 | `00_verify_raw_sources.py` | — | Checks NOAA and EAGLE-I raw files against `pipeline/raw_sources.json` |
| 01-04 | existing | `data/03_cleaned/*.parquet` (national, cleaning v3) | Unchanged rules; build hash now covers source content and SQL macros |
| 08 | `08_fetch_reference_sources.py` | `data/01_raw/nws_zone_county/`, `iem_vtec/`, `census_counties/` | Pinned URLs; SHA-256 manifests; existing files verified, never overwritten |
| 09 | `09_build_zone_crosswalk.py` | `data/reference/nws_zone_county_versions.parquet`, `data/03_cleaned/southeast/zone_event_counties.parquet`, `county_event_checks.parquet` | 23 correlation versions, October 2014 to April 2026 |
| 10 | `10_build_southeast_base.py` | `data/03_cleaned/southeast/` base tables | Cached by content hash (`--check` lists stale outputs) |
| 11 | `11_audit_major_labels.py` | `reports/label_audit/` | Training and validation windows only |
| 12 | `12_audit_alert_feasibility.py` | `reports/forecast_feasibility/` (24 h feasibility audit, crossing timing), `county_alert_events.parquet` | Training years only for analysis |
| 13 | `13_build_major_outage_dataset.py` | `data/04_model_input/southeast_v1/` | Builds 72 h alert rows and 24 h onset rows; features for all splits; outcomes for train and validation only ([dataset_card.md](dataset_card.md)) |

## Base tables (`data/03_cleaned/southeast/`)

| Table | Grain | Rows |
|---|---|---|
| `eaglei_southeast.parquet` | county x EAGLE-I run, as reported | 51,252,656 (2014-11 to 2025) |
| `county_reference.parquet` | Census county | 534 |
| `storm_events_southeast.parquet` | NOAA event x county | 94,363 |
| `onset_opportunities.parquet` | county prediction opportunity at recorded onset | 40,959 (train 33,784; validation 3,470; test 3,705) |
| `county_alert_events.parquet` | county x NWS alert event (first issuance reaching the county) | 583,785 |

Identifiers are five-character county FIPS strings; times are UTC. Columns realized after the prediction time carry a `retro_` prefix and are audit fields only.

## Exclusions and changes from the national v3 outputs

| Item | Count | Reason |
|---|---|---|
| Zone events without a verified zone | 1 (Rip Current) | Name conflicts with the only candidate zone |
| County-coded records reinterpreted as zones | 119 (Georgia, January 2025 winter storm) | Codes are zone numbers ("SOUTH FULTON"); taken as county codes they would land in the wrong counties |
| Event-county rows outside the provisional storm scope | 4,942 other (drought, rip current, fog and other non-storm types) and 4,695 temperature (heat, cold, frost) | Not storms under decision E-09; temperature hazards are open (O-6) |
| Event-county rows before EAGLE-I begins (2014-11-01 04:00 UTC) | 6,038 | No outcome data |
| Event-county rows with daylight-offset time labels | 49 | Time zone ambiguous; flagged for review |
| Opportunities in counties EAGLE-I never reports | 265 train, 23 validation, 48 test | Five Tennessee counties |
| Opportunities touching a split boundary | 3 validation, 3 test | Outcome or baseline window crosses into another split |
| Dec 31 observations | none excluded | Kept; 2014-2021 files end at Dec 31 00:00 UTC and coverage rules treat later runs as unknown |
| Long events (over 72 hours) | none excluded | Duration is realized after the prediction time |

## Validation checks

Assertions inside the stages stop the build when they fail:

- Unique `(county_fips, observation_utc)` in EAGLE-I; no unparsed customer values; every EAGLE-I and event county in the Census list.
- Unique `(event_id, county_fips)` in events; zone mappings only from accepted statuses.
- Unique opportunity ids and no overlapping 24-hour windows within a county.
- Malformed correlation lines are allowed only outside the six states (a few archived VA, NM, AZ and CA lines).
- The label audit never summarizes test-split windows.

`pytest` (47 tests in `tests/`) covers:

| File | What it checks |
|---|---|
| `test_identifiers.py` | FIPS padding, state codes, correlation dates, county and zone name verification |
| `test_timestamps.py` | NOAA offset to UTC, quarter-hour slots, 96-run windows, exclusive window end, t0 state uses the last complete run, year-end gaps stay unknown |
| `test_labels.py` | major / non-major / unknown rules, null runs, cov90 gaps, off-grid evidence, continuing vs new crossings, spike and gap diagnostics |
| `test_joins.py` | Overlapping records form one county window; windows never overlap; zone mappings are one row per event and county |
| `test_splits.py` | Split years; final alert state never read; realized fields prefixed `retro_` |
| `test_cache.py` | Hashes change with source content, rules and version but not project location; `--check` builds nothing |
| `test_features.py` | Pre-`t0` features unchanged by outcome-window data; "already major at `t0`" fallback; approved rule; no outcome or `retro_` columns among predictors or in test files |
| `test_outputs.py` | The same invariants on the built tables (skipped when data are absent) |

## Known limitations

- Three zone-correlation releases were never archived (Mobile 2015, Wilmington December 2017, Jacksonville May 2018); 72 events fall in those periods and rely on name-verified fallback. One archived capture (`bp08se15`) appears to contain pre-February-2015 content; name verification protects the mapping.
- A zone covering part of a county is attributed to the whole county (`partial_county` flags 5,057 of 22,969 zone-mapped rows).
- Customer denominators: MCC is invalid for all North Carolina counties; EAGLE-I's 2024 totals are inflated in FL, NC and SC (see the label report).
- NOAA event types and begin times are retrospective; their availability at onset is assumed in the onset design.
