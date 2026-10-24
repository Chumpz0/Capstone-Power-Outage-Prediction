# Data

Everything in this folder except this file is git-ignored. The shared copy lives in the team's private Hugging Face dataset; load it with the functions in `src/data.py`. To build it yourself, run `python scripts/download_data.py` (see the main README). The same paths below are used locally and on Hugging Face.

```
data/
  raw/          temporary downloads (EAGLE-I raw CSVs are deleted after filtering)
  processed/
    eaglei/     outages_2014.parquet ... outages_2025.parquet, MCC.csv, coverage_history.csv, DQI.csv
    storms/     storm_events_YYYY.parquet, storm_event_counties.parquet
    reference/  zone_county.parquet, county_census.parquet
```

## Key columns

**EAGLE-I outages** (`eaglei/outages_YYYY.parquet`), one row per county per 15-minute snapshot:

| Column | Meaning |
| --- | --- |
| `fips_code` | 5-digit county FIPS code (string, zero-padded). **Join key.** |
| `county`, `state` | Names |
| `customers_out` | Customers without power at that snapshot |
| `run_start_time` | Snapshot timestamp |

`MCC.csv` holds a modeled customer denominator for plausibility checks. It is not verified customer capacity. The approved forecast target uses absolute uncapped counts, not percentages.

**Storm event counties** (`storms/storm_event_counties.parquet`), one row per storm event per affected county. It has all NOAA columns (lower-case) plus:

| Column | Meaning |
| --- | --- |
| `fips_code` | County the event affected. **Join key.** |
| `event_id`, `episode_id` | NOAA ids (an episode groups related events, e.g. one hurricane) |
| `event_type` | e.g. Thunderstorm Wind, Tropical Storm, Hurricane, Winter Storm, Ice Storm |
| `begin_local` / `end_local` | Event start/end in local standard time |
| `begin_utc` / `end_utc` | Same, converted to UTC using `cz_timezone` |
| `magnitude` | Wind speed (knots) for wind events, hail size for hail |
| `damage_property_usd` | Property damage in dollars (parsed from "25.00K" style text) |
| `mapped_from` | `county` (reported by county) or `zone` (expanded from an NWS forecast zone) |

**County census** (`reference/county_census.parquet`): `fips_code`, `acs_year`, `population`, `housing_units`, `land_sqmi`, `pop_density`, `housing_density`.

## Current EDA conventions

Timestamps represent UTC. Missing rows and null counts stay distinct from explicit
zero. Exact-grid checks include seconds. Future targets use `(t, t+h]` for 6/12
hours, with strict >5,000 classification and a complete/unflagged primary cohort.

The original storm mapping contains 119 supported Georgia code/name conflicts.
Use the locally generated `processed/eda/geography_criteria/storm_event_counties.parquet`
for corrected retrospective attribution. Original IDs and methods are retained;
1,498 unmatched zone reports remain unresolved. Unmatched storms never remove
county-hours from the forecasting frame.

See [the report](../docs/eda_report.md), [field dictionary](../docs/data_dictionary.md),
[derived metadata](../docs/eda_metadata.md) and [handoff commands](../docs/teammate_handoff.md).
The `processed/eda/` directory contains reproducible local outputs and is not
assumed to be published to the shared dataset. No model feature matrix has been built.
